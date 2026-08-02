"""Trainable routed-expert LoRA engine across supported dispatch layouts."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from mirai.core.models.adapters.expert_condenser import ExpertCondenserComponent
from mirai.core.models.adapters.lora_allocation import lora_scale
from mirai.core.models.adapters.lora_initialization import initialize_from_quantization_error
from mirai.core.models.adapters.lora_initialization import initialize_lora_a
from mirai.core.models.adapters.lora_initialization import validate_lora_initializer
from mirai.core.models.adapters.lora_fa import configure_lora_fa_factors
from mirai.core.models.adapters.lora_parameter_dropout import (
    apply_lora_parameter_dropout,
)
from mirai.core.models.adapters.lora_parameter_dropout import (
    validate_lora_parameter_dropout,
)
from mirai.core.models.adapters.tc_lora import combine_gate_with_mask

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


class RoutedExpertActivationObserver(Protocol):
    """Semantic activation layouts exposed to calibration extensions."""

    def observe_single(self, activations: Any, *, expert_idx: int) -> None: ...

    def observe_batched(self, activations: Any, *, expert_indices: Any) -> None: ...

    def observe_segmented(
        self, activations: Any, *, expert_indices: Any, counts: Any
    ) -> None: ...


class ActiveExpertLoRA(nn.Module):
    """Low-rank factors evaluated for one routed expert at a time."""

    _mirai_expert_lora_adapter = True

    def __init__(
        self,
        *,
        adapter_name: str,
        num_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        init: str = "kaiming",
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("Expert LoRA rank must be > 0.")
        self.adapter_name = str(adapter_name)
        self.rank = int(rank)
        self.use_rslora = bool(use_rslora)
        self._lora_init = validate_lora_initializer(init)
        self.lora_a = nn.Parameter(
            torch.empty(int(num_experts), self.rank, int(in_features))
        )
        self.lora_b = nn.Parameter(
            torch.zeros(int(num_experts), int(out_features), self.rank)
        )
        self.register_buffer(
            "lora_alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
            persistent=True,
        )
        # Routing-guided expert selection (MoE-Sieve): when active, only the
        # experts flagged in ``active_expert_mask`` produce a non-zero LoRA
        # contribution; the masked experts' adapter output is multiplied by 0,
        # which makes their grads exactly zero (so a paged optimizer applies no
        # update, weight_decay=0). Default: selection disabled -> the mask is
        # not consulted while selection is disabled.
        self.register_buffer(
            "active_expert_mask",
            torch.ones(int(num_experts), dtype=torch.float32),
            persistent=True,
        )
        self._expert_selection_active = False
        self._lora_scale = 1.0
        self._rank_dropout = 0.0
        self._lora_parameter_dropout = 0.0
        self._rank_schedule_scale = 1.0
        self._alpha_value: float | None = None
        self._lora_fa_enabled = False
        self._lora_fa_hook: Any | None = None
        self._condenser_lora_fa_hook: Any | None = None
        # Timestep-axis rank mask (T-LoRA / bands). Routed-expert token layouts
        # have no recoverable sample axis inside the adapter, so only the
        # conservative batch-uniform mask applies here (exact per-sample when
        # batch_size == 1). Plain transient attribute -- default None keeps the
        # routed-token forwards unchanged.
        self._timestep_mask_per_sample: Any | None = None
        self._timestep_mask_uniform: Any | None = None
        # TC-LoRA gate provider (uniform: routed-token layouts have no sample
        # axis). Recomputed in the mask chokepoint so gradient checkpointing owns
        # a segment-local subgraph; hypernet held in a 1-tuple to avoid duplicate
        # submodule registration. None disables the gate.
        self._tc_gate_hypernet: tuple[Any] | None = None
        self._tc_gate_sigma: Any | None = None
        self._condenser = ExpertCondenserComponent(self)
        # Optional semantic observer seam for data-driven initializers. Kept in
        # a tuple so an nn.Module observer would not become checkpoint state.
        self._activation_calibration_observer: tuple[Any] | None = None
        self._condenser.initialize_disabled()
        initialize_lora_a(self.lora_a, self._lora_init)

    def enable_condenser(
        self, *, rank: int, alpha: float, init: str = "kaiming"
    ) -> None:
        self._condenser.enable(rank=rank, alpha=alpha, init=init)
        if self._lora_fa_enabled:
            self.set_lora_fa_enabled(True)

    def _initialize_condenser_a(self, tensor: Any, init: str) -> None:
        initialize_lora_a(tensor, validate_lora_initializer(init))

    def initialize_expert_from_quantized_base(
        self,
        *,
        expert_idx: int,
        reference_weight: Any,
        quantized_weight: Any,
    ) -> Any | None:
        if self._lora_init != "loftq":
            return None
        idx = int(expert_idx)
        return initialize_from_quantization_error(
            lora_a=self.lora_a[idx],
            lora_b=self.lora_b[idx],
            reference_weight=reference_weight,
            quantized_weight=quantized_weight,
            scale=lora_scale(
                self._alpha(), self.rank, use_rslora=self.use_rslora
            ),
        )

    def has_condenser(self) -> bool:
        return self._condenser.is_enabled()

    def set_lora_fa_enabled(self, enabled: bool) -> None:
        self._lora_fa_enabled = bool(enabled)
        self._lora_fa_hook = configure_lora_fa_factors(
            lora_a=self.lora_a,
            lora_b=self.lora_b,
            scale=lambda: (
                lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
                * self._lora_scale
                * self._rank_schedule_scale
            ),
            enabled=self._lora_fa_enabled,
            hook=self._lora_fa_hook,
        )
        if self.cond_a is not None and self.cond_b is not None:
            self._condenser_lora_fa_hook = configure_lora_fa_factors(
                lora_a=self.cond_a,
                lora_b=self.cond_b,
                scale=self._condenser_scale,
                enabled=self._lora_fa_enabled,
                hook=self._condenser_lora_fa_hook,
            )

    def _condenser_scale(self) -> float:
        return self._condenser._scale()

    def _condenser_delta(self, x: torch.Tensor) -> torch.Tensor | None:
        return self._condenser.delta(x)

    def set_activation_calibration_observer(
        self, observer: RoutedExpertActivationObserver | None
    ) -> None:
        self._activation_calibration_observer = (
            None if observer is None else (observer,)
        )

    def _observe_single(self, x: torch.Tensor, *, expert_idx: int) -> None:
        holder = self._activation_calibration_observer
        if holder is not None:
            holder[0].observe_single(x, expert_idx=int(expert_idx))

    def _observe_batched(
        self, x: torch.Tensor, *, expert_indices: torch.Tensor
    ) -> None:
        holder = self._activation_calibration_observer
        if holder is not None:
            holder[0].observe_batched(x, expert_indices=expert_indices)

    def _observe_segmented(
        self,
        x: torch.Tensor,
        *,
        expert_indices: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        holder = self._activation_calibration_observer
        if holder is not None:
            holder[0].observe_segmented(
                x, expert_indices=expert_indices, counts=counts
            )

    def set_active_experts(self, expert_ids: Iterable[int]) -> None:
        """Restrict the LoRA contribution to ``expert_ids`` (routing_topk).

        Idempotent and deterministic: builds a 0/1 mask over all experts and
        marks selection active. Passing every expert is equivalent to disabling
        selection numerically, but the active flag stays set for reporting."""
        mask = torch.zeros_like(self.active_expert_mask)
        ids = sorted({int(i) for i in expert_ids})
        for idx in ids:
            if idx < 0 or idx >= mask.numel():
                raise ValueError(
                    f"Expert id {idx} out of range for {mask.numel()} experts."
                )
            mask[idx] = 1.0
        self.active_expert_mask.copy_(mask)
        self._expert_selection_active = True

    def clear_active_experts(self) -> None:
        """Disable selection so the adapter contributes for all experts."""
        self.active_expert_mask.fill_(1.0)
        self._expert_selection_active = False

    def active_expert_ids(self) -> list[int]:
        return [
            int(i)
            for i in torch.nonzero(self.active_expert_mask > 0.0, as_tuple=False)
            .reshape(-1)
            .tolist()
        ]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:  # type: ignore[override]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._alpha_value = None
        self._condenser.reset_cache()
        # A loaded mask that is not all-ones implies selection was active in the
        # source checkpoint; restore the flag so batched_subset_forward honors it.
        if f"{prefix}active_expert_mask" in state_dict:
            self._expert_selection_active = bool(
                (self.active_expert_mask < 1.0).any().item()
            )

    def _alpha(self) -> float:
        if self._alpha_value is None:
            self._alpha_value = float(self.lora_alpha.detach().float().item())
        return self._alpha_value

    def set_lora_scale(self, scale: float) -> None:
        self._lora_scale = float(scale)

    def set_rank_dropout(self, dropout: float) -> None:
        value = float(dropout)
        if value < 0.0 or value > 1.0:
            raise ValueError("adapter.rank_dropout must be in [0, 1].")
        self._rank_dropout = value

    def set_lora_parameter_dropout(self, dropout: float) -> None:
        self._lora_parameter_dropout = validate_lora_parameter_dropout(dropout)

    def set_rank_schedule_scale(self, scale: float) -> None:
        self._rank_schedule_scale = float(scale)

    def set_timestep_rank_masks(
        self, per_sample: Any | None, batch_uniform: Any | None
    ) -> None:
        """Install (or clear) timestep-axis rank masks (T-LoRA / bands).

        Only ``batch_uniform`` ``[rank]`` is applied (token layouts carry no
        sample axis); ``per_sample`` is stored for introspection. Masked rank
        columns multiply the low-rank intermediate by zero: exactly-zero
        output and exactly-zero grads for the masked ranks / gated batches.
        """
        self._timestep_mask_per_sample = per_sample
        self._timestep_mask_uniform = batch_uniform

    def set_tc_gate(self, hypernet: Any | None, sigmas: Any | None) -> None:
        """Install (or clear) the TC-LoRA gate provider (uniform).

        The gate is recomputed inside ``_apply_timestep_rank_mask`` so gradient
        checkpointing gives each recomputed block a segment-local subgraph; the
        hypernet is held in a 1-tuple to avoid duplicate submodule registration.
        """
        self._tc_gate_hypernet = None if hypernet is None else (hypernet,)
        self._tc_gate_sigma = sigmas

    def _apply_timestep_rank_mask(self, low_rank: torch.Tensor) -> torch.Tensor:
        mask = self._timestep_mask_uniform
        if mask is not None:
            low_rank = low_rank * mask.to(device=low_rank.device, dtype=low_rank.dtype)
        holder = self._tc_gate_hypernet
        sigma = self._tc_gate_sigma
        if holder is not None and sigma is not None:
            # Recompute the per-rank gate here (segment-local under gradient
            # checkpointing); uniform-reduce over the sample axis, then multiply.
            gate = holder[0](sigma).amin(dim=0)
            low_rank = combine_gate_with_mask(low_rank, gate)
        return low_rank

    @staticmethod
    def _apply_route_gate(out: torch.Tensor, route_gate: Any | None) -> torch.Tensor:
        if route_gate is None:
            return out
        if tuple(route_gate.shape) != tuple(out.shape[:-1]):
            raise ValueError("route_gate must match every adapter-output row.")
        return out * route_gate.to(device=out.device, dtype=out.dtype).unsqueeze(-1)

    def forward(
        self, x: torch.Tensor, *, expert_idx: int, route_gate: Any | None = None
    ) -> torch.Tensor:
        if self._activation_calibration_observer is not None:
            self._observe_single(x, expert_idx=int(expert_idx))
        if self.training and self._rank_dropout > 0.0:
            x = F.dropout(x, p=self._rank_dropout, training=True)
        a = self.lora_a[int(expert_idx)].to(device=x.device, dtype=x.dtype)
        b = self.lora_b[int(expert_idx)].to(device=x.device, dtype=x.dtype)
        a, b = apply_lora_parameter_dropout(
            a,
            b,
            probability=self._lora_parameter_dropout,
            training=self.training,
        )
        scale = (
            lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
            * self._lora_scale
            * self._rank_schedule_scale
        )
        out = F.linear(self._apply_timestep_rank_mask(F.linear(x, a)), b) * scale
        if self._expert_selection_active:
            gate = self.active_expert_mask[int(expert_idx)].to(
                device=out.device, dtype=out.dtype
            )
            out = out * gate
        out = self._apply_route_gate(out, route_gate)
        cond = self._condenser_delta(x)
        if cond is not None:
            out = out + cond
        return out

    def grouped_forward(self, x: torch.Tensor, *, offsets: torch.Tensor) -> torch.Tensor:
        if not hasattr(torch, "_grouped_mm") or x.device.type != "cuda":
            raise RuntimeError("Grouped expert LoRA requires CUDA torch._grouped_mm.")
        if self._activation_calibration_observer is not None:
            starts = torch.cat((offsets.new_zeros(1), offsets[:-1]))
            self._observe_segmented(
                x,
                expert_indices=torch.arange(
                    int(offsets.numel()), device=offsets.device, dtype=torch.long
                ),
                counts=offsets - starts,
            )
        if self.training and self._rank_dropout > 0.0:
            x = F.dropout(x, p=self._rank_dropout, training=True)
        a = self.lora_a.to(device=x.device, dtype=x.dtype)
        b = self.lora_b.to(device=x.device, dtype=x.dtype)
        a, b = apply_lora_parameter_dropout(
            a,
            b,
            probability=self._lora_parameter_dropout,
            training=self.training,
        )
        alignment_elements = max(1, 16 // x.element_size())
        padded_rank = (
            (self.rank + alignment_elements - 1)
            // alignment_elements
            * alignment_elements
        )
        rank_padding = padded_rank - self.rank
        grouped_a = a.transpose(-2, -1)
        grouped_b = b.transpose(-2, -1)
        if rank_padding:
            grouped_a = F.pad(grouped_a, (0, rank_padding))
            grouped_b = F.pad(grouped_b, (0, 0, 0, rank_padding))
        low_rank = torch._grouped_mm(x, grouped_a, offs=offsets)
        if rank_padding:
            masked_low_rank = self._apply_timestep_rank_mask(
                low_rank[..., : self.rank]
            )
            low_rank = F.pad(masked_low_rank, (0, rank_padding))
        else:
            low_rank = self._apply_timestep_rank_mask(low_rank)
        scale = (
            lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
            * self._lora_scale
            * self._rank_schedule_scale
        )
        out = torch._grouped_mm(low_rank, grouped_b, offs=offsets) * scale
        cond = self._condenser_delta(x)
        if cond is not None:
            out = out + cond
        return out

    def batched_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._activation_calibration_observer is not None:
            self._observe_batched(
                x,
                expert_indices=torch.arange(
                    int(x.shape[0]), device=x.device, dtype=torch.long
                ),
            )
        if self.training and self._rank_dropout > 0.0:
            x = F.dropout(x, p=self._rank_dropout, training=True)
        a = self.lora_a.to(device=x.device, dtype=x.dtype)
        b = self.lora_b.to(device=x.device, dtype=x.dtype)
        a, b = apply_lora_parameter_dropout(
            a,
            b,
            probability=self._lora_parameter_dropout,
            training=self.training,
        )
        low_rank = self._apply_timestep_rank_mask(torch.bmm(x, a.transpose(-2, -1)))
        scale = (
            lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
            * self._lora_scale
            * self._rank_schedule_scale
        )
        out = torch.bmm(low_rank, b.transpose(-2, -1)) * scale
        if self._expert_selection_active:
            gate = self.active_expert_mask.to(device=out.device, dtype=out.dtype)
            out = out * gate.reshape(-1, *([1] * (out.dim() - 1)))
        cond = self._condenser_delta(x)
        if cond is not None:
            out = out + cond
        return out

    def batched_subset_forward(
        self,
        x: torch.Tensor,
        *,
        expert_indices: torch.Tensor,
        route_gate: Any | None = None,
    ) -> torch.Tensor:
        if self._activation_calibration_observer is not None:
            self._observe_batched(x, expert_indices=expert_indices)
        if self.training and self._rank_dropout > 0.0:
            x = F.dropout(x, p=self._rank_dropout, training=True)
        source_indices = expert_indices.to(device=self.lora_a.device, dtype=torch.long)
        a = self.lora_a.index_select(0, source_indices).to(
            device=x.device, dtype=x.dtype
        )
        b = self.lora_b.index_select(0, source_indices).to(
            device=x.device, dtype=x.dtype
        )
        a, b = apply_lora_parameter_dropout(
            a,
            b,
            probability=self._lora_parameter_dropout,
            training=self.training,
        )
        low_rank = self._apply_timestep_rank_mask(torch.bmm(x, a.transpose(-2, -1)))
        scale = (
            lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
            * self._lora_scale
            * self._rank_schedule_scale
        )
        out = torch.bmm(low_rank, b.transpose(-2, -1)) * scale
        if self._expert_selection_active:
            # Zero the adapter output for masked experts. Because the mask
            # multiplies the output, the gradient scattered back into lora_a /
            # lora_b for those experts is exactly zero -> no optimizer update.
            gate = self.active_expert_mask.index_select(0, source_indices).to(
                device=out.device, dtype=out.dtype
            )
            out = out * gate.reshape(-1, *([1] * (out.dim() - 1)))
        out = self._apply_route_gate(out, route_gate)
        cond = self._condenser_delta(x)
        if cond is not None:
            out = out + cond
        return out

    def sorted_subset_forward(
        self,
        x: torch.Tensor,
        *,
        expert_indices: torch.Tensor,
        m_offsets: torch.Tensor,
        counts: torch.Tensor,
        route_gate: Any | None = None,
    ) -> torch.Tensor:
        """LoRA on the SORTED-CONTIGUOUS layout for the ``triton_persistent`` path.

        ``x`` is ``[m_chunk, in]`` sorted by expert, segmented by ``m_offsets``
        (inclusive cumsum of ``counts`` over the active experts, ascending). Two
        segment-wise grouped GEMMs (the SAME vendored ``grouped_gemm`` autograd
        Function used for the frozen linears -- SM80+, unlike ``torch._grouped_mm``
        which is Hopper-only) apply the per-expert A then B factors, matching
        ``batched_subset_forward`` (padded bmm) within bf16 tolerance. ``lora_a``
        is ``[E, rank, in]`` and ``lora_b`` ``[E, out, rank]`` -- already the
        ``[E, N, K]`` weight layout the kernel wants, so no transpose is needed.
        Gradients flow to ``lora_a``/``lora_b`` through the Function's ``dw`` path.
        The ``active_expert_mask`` is applied as a per-row gate built by
        ``repeat_interleave`` so masked experts contribute exactly zero (and thus
        exactly-zero grads), identical to the padded path's per-expert gate.
        """
        if x.device.type != "cuda":
            raise RuntimeError("Persistent expert LoRA requires CUDA tensors.")
        if self._activation_calibration_observer is not None:
            self._observe_segmented(
                x, expert_indices=expert_indices, counts=counts
            )
        from mirai.core.moe.runtime.gemm import resolve_moe_gemm_backend
        from mirai.vendors.qwen3_moe_fused import grouped_gemm

        if self.training and self._rank_dropout > 0.0:
            x = F.dropout(x, p=self._rank_dropout, training=True)
        source = expert_indices.to(device=self.lora_a.device, dtype=torch.long)
        a = (
            self.lora_a.index_select(0, source)
            .to(device=x.device, dtype=x.dtype)
            .contiguous()
        )
        b = (
            self.lora_b.index_select(0, source)
            .to(device=x.device, dtype=x.dtype)
            .contiguous()
        )
        a, b = apply_lora_parameter_dropout(
            a,
            b,
            probability=self._lora_parameter_dropout,
            training=self.training,
        )
        offs = m_offsets.to(device=x.device, dtype=torch.int32).contiguous()
        dw_backend = resolve_moe_gemm_backend("dw", device=x.device)
        rank_padding = 0
        a_exec = a
        b_exec = b
        if dw_backend == "torch_grouped":
            alignment_elements = max(1, 16 // int(x.element_size()))
            padded_rank = (
                (self.rank + alignment_elements - 1) // alignment_elements
            ) * alignment_elements
            rank_padding = padded_rank - self.rank
            if rank_padding:
                a_exec = F.pad(a, (0, 0, 0, rank_padding))
                b_exec = F.pad(b, (0, rank_padding))

        def trainable_grouped(input_tensor, weight):
            if dw_backend in {"auto", "persistent"}:
                return grouped_gemm(input_tensor.contiguous(), weight, offs)
            if dw_backend == "torch_grouped":
                from .torch_grouped import _grouped_mm

                operand = weight.transpose(-2, -1)
                candidates = (operand, operand.contiguous())
                for candidate in candidates:
                    if all(
                        stride == 1
                        or int(stride) * int(candidate.element_size()) % 16 == 0
                        for stride in candidate.stride()
                    ):
                        break
                else:
                    raise RuntimeError(
                        "Framework grouped expert LoRA cannot represent this "
                        "factor with 16-byte-aligned matrix strides."
                    )
                return _grouped_mm(
                    input_tensor.contiguous(),
                    candidate,
                    offs,
                )
            boundaries = offs.detach().to(device="cpu", dtype=torch.int64).tolist()
            chunks = []
            start = 0
            for local_expert, stop in enumerate(boundaries):
                stop = int(stop)
                if stop > start:
                    chunks.append(
                        F.linear(input_tensor[start:stop], weight[local_expert])
                    )
                start = stop
            if not chunks:
                return input_tensor.new_empty((0, weight.shape[1]))
            return torch.cat(chunks, dim=0)

        low_rank = self._apply_timestep_rank_mask(
            trainable_grouped(x, a_exec)[..., : self.rank]
        )
        if rank_padding:
            low_rank = F.pad(low_rank, (0, rank_padding))
        scale = (
            lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
            * self._lora_scale
            * self._rank_schedule_scale
        )
        out = trainable_grouped(low_rank, b_exec) * scale
        if self._expert_selection_active:
            gate = self.active_expert_mask.index_select(0, source).to(
                device=out.device, dtype=out.dtype
            )
            per_row = torch.repeat_interleave(
                gate,
                counts.to(device=out.device, dtype=torch.long),
                output_size=x.shape[0],
            )
            out = out * per_row.unsqueeze(-1)
        out = self._apply_route_gate(out, route_gate)
        cond = self._condenser_delta(x)
        if cond is not None:
            out = out + cond
        return out

    def megablocks_forward(
        self,
        x: torch.Tensor,
        *,
        counts: torch.Tensor,
        grouped_gemm_ops: Any,
    ) -> torch.Tensor:
        if self._activation_calibration_observer is not None:
            self._observe_segmented(
                x,
                expert_indices=torch.arange(
                    int(counts.numel()), device=counts.device, dtype=torch.long
                ),
                counts=counts,
            )
        a = self.lora_a.to(device=x.device, dtype=x.dtype)
        b = self.lora_b.to(device=x.device, dtype=x.dtype)
        a, b = apply_lora_parameter_dropout(
            a,
            b,
            probability=self._lora_parameter_dropout,
            training=self.training,
        )
        if self.training and self._rank_dropout > 0.0:
            x = F.dropout(x, p=self._rank_dropout, training=True)
        batch_sizes = counts.detach().to(device="cpu", dtype=torch.int64)
        low_rank = self._apply_timestep_rank_mask(
            grouped_gemm_ops.gmm(x, a, batch_sizes, trans_b=True)
        )
        scale = (
            lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
            * self._lora_scale
            * self._rank_schedule_scale
        )
        out = grouped_gemm_ops.gmm(
            low_rank, b, batch_sizes, trans_b=True
        ) * scale
        cond = self._condenser_delta(x)
        if cond is not None:
            out = out + cond
        return out
