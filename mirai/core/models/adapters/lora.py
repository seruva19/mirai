"""Generic LoRA adapters for native Mirai model wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
import weakref

from mirai.core.moe.runtime.specs import ExpertTensorSpec
from mirai.core.models.compressed_weights import CompressedGroupedExperts
from mirai.core.models.compressed_weights import CompressedLinear
from mirai.core.models.adapters.expert_condenser import CONDENSER_A_SUFFIX
from mirai.core.models.adapters.expert_condenser import CONDENSER_ALPHA_SUFFIX
from mirai.core.models.adapters.expert_condenser import CONDENSER_B_SUFFIX
from mirai.core.models.adapters.expert_condenser import load_condenser_state
from mirai.core.models.adapters.expert_condenser import write_condenser_state
from mirai.core.models.adapters.dora import DORA_MAGNITUDE_SUFFIX
from mirai.core.models.adapters.dora import dora_effective_weight
from mirai.core.models.adapters.dora import dora_row_magnitude
from mirai.core.models.adapters.lora_allocation import PARA_RANK_STATE_SUFFIX
from mirai.core.models.adapters.lora_allocation import RSLORA_STATE_SUFFIX
from mirai.core.models.adapters.lora_allocation import lora_scale
from mirai.core.models.adapters.lora_initialization import initialize_from_quantization_error
from mirai.core.models.adapters.lora_initialization import initialize_from_principal_weight
from mirai.core.models.adapters.lora_initialization import (
    initialize_from_gradient_approximation,
)
from mirai.core.models.adapters.lora_initialization import initialize_lora_a
from mirai.core.models.adapters.lora_initialization import validate_lora_initializer
from mirai.core.models.adapters.lora_fa import configure_lora_fa_factors
from mirai.core.models.adapters.lora_gora import resize_lora_module
from mirai.core.models.adapters.lora_parameter_dropout import (
    apply_lora_parameter_dropout,
)
from mirai.core.models.adapters.lora_parameter_dropout import (
    validate_lora_parameter_dropout,
)
from mirai.core.models.adapters.tc_lora import combine_gate_with_mask
from mirai.core.models.adapters.sparse_expert_export import MASK_SUFFIX
from mirai.core.models.adapters.sparse_expert_export import SPARSE_A_SUFFIX
from mirai.core.models.adapters.sparse_expert_export import SPARSE_B_SUFFIX
from mirai.core.models.adapters.sparse_expert_export import SPARSE_IDS_SUFFIX
from mirai.core.models.adapters.sparse_expert_export import SparseExpertExportPolicy
from mirai.core.models.adapters.sparse_expert_export import expand_sparse_expert_state  # noqa: F401
from mirai.core.models.adapters.sparse_expert_export import load_compact_expert_adapter
from mirai.core.models.adapters.sparse_expert_export import selection_is_active

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LoRAApplicationReport:
    target_preset: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: float
    matched_modules: tuple[str, ...]
    skipped_modules: tuple[str, ...] = ()
    allocations: tuple[tuple[str, int, float, str], ...] = ()


class LoRALinear(nn.Module):
    """LoRA wrapper for a frozen linear-like module.

    The wrapped base may be a regular ``nn.Linear`` or a quantized Mirai linear.
    Regular CPU-resident base weights are copied to the activation device during
    forward, which lets large frozen bases stay in host memory while LoRA params
    and activations live on the accelerator.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        rank: int,
        alpha: float,
        init: str = "kaiming",
        use_rslora: bool = False,
        use_dora: bool = False,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("LoRALinear requires torch.")
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be > 0.")
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise ValueError(
                f"LoRALinear expected a linear-like base module, got {type(base).__name__}."
            )
        self.base = base
        self.rank = int(rank)
        self.in_features = int(getattr(base, "in_features"))
        self.out_features = int(getattr(base, "out_features"))
        self.use_rslora = bool(use_rslora)
        self.use_dora = bool(use_dora)
        if self.use_dora and not isinstance(base, nn.Linear):
            raise ValueError(
                "DoRA requires an unpacked nn.Linear base; compressed linear "
                "weights cannot provide the normalized direction."
            )
        self.lora_a = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, self.rank))
        if self.use_dora:
            self.dora_magnitude = nn.Parameter(
                torch.empty(
                    self.out_features,
                    device=getattr(base, "weight").device,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("dora_magnitude", None)
        self.register_buffer(
            "lora_alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
            persistent=True,
        )
        self._lora_scale = 1.0
        self._rank_dropout = 0.0
        self._lora_parameter_dropout = 0.0
        self._rank_schedule_scale = 1.0
        self._alpha_value: float | None = float(alpha)
        self._lora_init = validate_lora_initializer(init)
        self._lora_fa_hook: Any | None = None
        # Timestep-axis rank masks (T-LoRA schedule / timestep bands). Plain
        # attributes, not buffers: transient per-forward state set by the
        # pipeline; None disables timestep gating.
        self._timestep_mask_per_sample: Any | None = None
        self._timestep_mask_uniform: Any | None = None
        # TC-LoRA gate provider (hypernet + per-sample sigma). Stored so the gate
        # is RECOMPUTED inside this forward: under gradient checkpointing each
        # recomputed block then owns its own gate subgraph (no single
        # differentiable tensor shared across -- and backwarded once per --
        # checkpoint segment). The hypernet is held in a 1-tuple so nn.Module
        # does not register it as a duplicate submodule. Default None keeps the
        # the default forward unchanged.
        self._tc_gate_hypernet: tuple[Any] | None = None
        self._tc_gate_sigma: Any | None = None
        self.reset_parameters()
        if self._lora_init == "pissa":
            initialize_from_principal_weight(
                lora_a=self.lora_a,
                lora_b=self.lora_b,
                base_weight=getattr(self.base, "weight"),
                scale=lora_scale(
                    self._alpha(), self.rank, use_rslora=self.use_rslora
                ),
            )
        self.initialize_dora_magnitude()
        self.freeze_base()

    def _alpha(self) -> float:
        if self._alpha_value is None:
            self._alpha_value = float(self.lora_alpha.detach().float().item())
        return self._alpha_value

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._alpha_value = float(self.lora_alpha.detach().float().item())

    def reset_parameters(self) -> None:
        initialize_lora_a(self.lora_a, self._lora_init)
        nn.init.zeros_(self.lora_b)

    def initialize_from_quantized_base(
        self, *, reference_weight: Any, quantized_weight: Any
    ) -> Any | None:
        if self.use_dora:
            raise ValueError(
                "DoRA cannot migrate to a packed linear base; keep frozen-weight "
                "quantization disabled."
            )
        if self._lora_init != "loftq":
            return None
        return initialize_from_quantization_error(
            lora_a=self.lora_a,
            lora_b=self.lora_b,
            reference_weight=reference_weight,
            quantized_weight=quantized_weight,
            scale=lora_scale(
                self._alpha(), self.rank, use_rslora=self.use_rslora
            ),
        )

    def initialize_from_full_gradient(
        self, gradient: Any, *, stable_gamma: float
    ) -> None:
        if self._lora_init != "lora_ga":
            return
        initialize_from_gradient_approximation(
            lora_a=self.lora_a,
            lora_b=self.lora_b,
            base_weight=getattr(self.base, "weight"),
            gradient=gradient,
            scale=lora_scale(
                self._alpha(), self.rank, use_rslora=self.use_rslora
            ),
            stable_gamma=float(stable_gamma),
        )
        self.initialize_dora_magnitude()

    def initialize_dora_magnitude(self) -> None:
        """Match magnitude to the currently initialized direction."""

        if not self.use_dora:
            return
        weight = getattr(self.base, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise ValueError("DoRA requires a materialized dense base weight.")
        with torch.no_grad():
            direction = weight.float() + (
                torch.matmul(self.lora_b.float(), self.lora_a.float())
                * lora_scale(
                    self._alpha(), self.rank, use_rslora=self.use_rslora
                )
            )
            self.dora_magnitude.copy_(
                dora_row_magnitude(direction).to(
                    device=self.dora_magnitude.device,
                    dtype=self.dora_magnitude.dtype,
                )
            )

    def set_lora_fa_enabled(self, enabled: bool) -> None:
        self._lora_fa_hook = configure_lora_fa_factors(
            lora_a=self.lora_a,
            lora_b=self.lora_b,
            scale=lambda: (
                lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
                * self._lora_scale
                * self._rank_schedule_scale
            ),
            enabled=bool(enabled),
            hook=self._lora_fa_hook,
        )

    @property
    def weight(self) -> Any:
        return getattr(self.base, "weight")

    @property
    def bias(self) -> Any:
        return getattr(self.base, "bias", None)

    @property
    def lora_scale(self) -> float:
        return float(self._lora_scale)

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
        """Install (or clear, with None/None) the timestep-axis rank masks.

        ``per_sample`` is ``[B, rank]``; ``batch_uniform`` is ``[rank]`` and is
        the fallback when the activation's leading dim is not the sample axis.
        Masked rank columns multiply the low-rank intermediate by zero, so the
        masked ranks (and out-of-band samples) get exactly-zero output and
        exactly-zero grads -- the MoE-Sieve masking pattern.
        """
        self._timestep_mask_per_sample = per_sample
        self._timestep_mask_uniform = batch_uniform

    def set_tc_gate(self, hypernet: Any | None, sigmas: Any | None) -> None:
        """Install (or clear) the TC-LoRA gate provider.

        ``hypernet`` maps ``sigmas`` -> a per-rank gate ``[B, rank]`` and is
        called INSIDE ``forward`` so gradient checkpointing recomputes the gate
        per segment. ``sigmas`` is detached (a constant leaf); the gate stays
        differentiable w.r.t. the hypernet parameters.
        """
        self._tc_gate_hypernet = None if hypernet is None else (hypernet,)
        self._tc_gate_sigma = sigmas

    def _apply_timestep_mask(self, down: Any) -> Any:
        per_sample = self._timestep_mask_per_sample
        if per_sample is None:
            return down
        if down.dim() >= 2 and int(down.shape[0]) == int(per_sample.shape[0]):
            view = [int(per_sample.shape[0])] + [1] * (down.dim() - 2) + [self.rank]
            mask = per_sample.view(*view)
        else:
            mask = self._timestep_mask_uniform
            if mask is None:
                return down
        return down * mask.to(device=down.device, dtype=down.dtype)

    def _apply_tc_gate(self, down: Any) -> Any:
        holder = self._tc_gate_hypernet
        sigma = self._tc_gate_sigma
        if holder is None or sigma is None:
            return down
        gate = holder[0](sigma)  # [B, rank], recomputed => segment-local subgraph
        if down.dim() >= 2 and int(down.shape[0]) == int(gate.shape[0]):
            view = [int(gate.shape[0])] + [1] * (down.dim() - 2) + [self.rank]
            gate_view = gate.view(*view)
        else:
            gate_view = gate.amin(dim=0)  # uniform fallback (no sample axis)
        return combine_gate_with_mask(down, gate_view)

    def freeze_base(self) -> None:
        for param in self.base.parameters(recurse=True):
            param.requires_grad_(False)

    def forward(self, x: Any) -> Any:
        base_out = self._base_forward(x)
        if self.training and self._rank_dropout > 0.0:
            lora_in = F.dropout(x, p=float(self._rank_dropout), training=True)
        else:
            lora_in = x
        a = self.lora_a.to(device=x.device, dtype=x.dtype)
        b = self.lora_b.to(device=x.device, dtype=x.dtype)
        a, b = apply_lora_parameter_dropout(
            a,
            b,
            probability=self._lora_parameter_dropout,
            training=self.training,
        )
        down = F.linear(lora_in, a)
        if self._timestep_mask_per_sample is not None:
            down = self._apply_timestep_mask(down)
        if self._tc_gate_hypernet is not None:
            down = self._apply_tc_gate(down)
        delta = F.linear(down, b)
        scale = (
            lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
            * float(self._lora_scale)
            * float(self._rank_schedule_scale)
        )
        if self.use_dora:
            base_weight = getattr(self.base, "weight")
            if base_weight.device != x.device or base_weight.dtype != x.dtype:
                base_weight = base_weight.to(device=x.device, dtype=x.dtype)
            effective = dora_effective_weight(
                base_weight=base_weight,
                lora_a=a,
                lora_b=b,
                magnitude=self.dora_magnitude.to(
                    device=x.device, dtype=torch.float32
                ),
                direction_scale=lora_scale(
                    self._alpha(), self.rank, use_rslora=self.use_rslora
                ),
                adapter_scale=(
                    float(self._lora_scale) * float(self._rank_schedule_scale)
                ),
            )
            return base_out + F.linear(lora_in, effective - base_weight, None)
        return base_out + (delta * scale)

    def _base_forward(self, x: Any) -> Any:
        base = self.base
        if isinstance(base, nn.Linear):
            weight = base.weight
            bias = base.bias
            if weight.device != x.device or weight.dtype != x.dtype:
                weight = weight.to(device=x.device, dtype=x.dtype)
            if bias is not None and (bias.device != x.device or bias.dtype != x.dtype):
                bias = bias.to(device=x.device, dtype=x.dtype)
            return F.linear(x, weight, bias)
        return base(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={float(self.lora_alpha.detach().float().item())}, "
            f"use_dora={self.use_dora}"
        )


class LoRAExpertTensorParametrization(nn.Module):
    """LoRA parametrization for routed expert tensors declared by ExpertTensorSpec."""

    def __init__(
        self,
        *,
        adapter_name: str,
        shape: tuple[int, ...],
        layout: tuple[str, ...],
        rank: int,
        alpha: float,
        init: str = "kaiming",
        use_rslora: bool = False,
        use_dora: bool = False,
        base_weight: Any | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("LoRAExpertTensorParametrization requires torch.")
        super().__init__()
        self._lora_init = validate_lora_initializer(init)
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be > 0.")
        axes = tuple(str(axis) for axis in layout)
        dims = tuple(int(dim) for dim in shape)
        if axes not in {("expert", "out", "in"), ("out", "in")}:
            raise ValueError(
                "Tensor LoRA supports layouts ('expert', 'out', 'in') and "
                "('out', 'in') only."
            )
        self.adapter_name = str(adapter_name)
        self._base_weight_ref = (
            None if base_weight is None else weakref.ref(base_weight)
        )
        self.rank = int(rank)
        self.use_rslora = bool(use_rslora)
        self.use_dora = bool(use_dora)
        if axes == ("expert", "out", "in"):
            if len(dims) != 3 or any(dim <= 0 for dim in dims):
                raise ValueError("Expert tensor LoRA requires a positive rank-3 shape.")
            experts, out_features, in_features = dims
            self.num_experts = int(experts)
            self.out_features = int(out_features)
            self.in_features = int(in_features)
            self.lora_a = nn.Parameter(
                torch.empty(self.num_experts, self.rank, self.in_features)
            )
            self.lora_b = nn.Parameter(
                torch.zeros(self.num_experts, self.out_features, self.rank)
            )
        else:
            if len(dims) != 2 or any(dim <= 0 for dim in dims):
                raise ValueError("Matrix tensor LoRA requires a positive rank-2 shape.")
            out_features, in_features = dims
            self.num_experts = 0
            self.out_features = int(out_features)
            self.in_features = int(in_features)
            self.lora_a = nn.Parameter(torch.empty(self.rank, self.in_features))
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, self.rank))
        if self.use_dora:
            if base_weight is None:
                raise ValueError("DoRA expert adapters require a bound base weight.")
            self.dora_magnitude = nn.Parameter(
                torch.empty(
                    dims[:-1],
                    device=base_weight.device,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("dora_magnitude", None)
        self.register_buffer(
            "lora_alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
            persistent=True,
        )
        self._lora_scale = 1.0
        self._rank_dropout = 0.0
        self._lora_parameter_dropout = 0.0
        self._rank_schedule_scale = 1.0
        self._alpha_value: float | None = float(alpha)
        self._lora_fa_hook: Any | None = None
        # Timestep-axis rank mask (batch-uniform only: this parametrization
        # acts in weight space, which is shared across samples).
        self._timestep_mask_per_sample: Any | None = None
        self._timestep_mask_uniform: Any | None = None
        # TC-LoRA gate provider (uniform: weight space has no sample axis).
        # Recomputed in forward for gradient-checkpointing safety; hypernet held
        # in a 1-tuple to avoid duplicate submodule registration.
        self._tc_gate_hypernet: tuple[Any] | None = None
        self._tc_gate_sigma: Any | None = None
        self.reset_parameters()

    def gora_base_weight(self) -> Any:
        """Return the frozen tensor parameter represented by this adapter."""

        if self._base_weight_ref is None:
            raise ValueError(
                f"GoRA target {self.adapter_name!r} has no bound base weight."
            )
        base_weight = self._base_weight_ref()
        if base_weight is None:
            raise RuntimeError(
                f"GoRA base weight for {self.adapter_name!r} no longer exists."
            )
        return base_weight

    def reset_parameters(self) -> None:
        initialize_lora_a(self.lora_a, self._lora_init)
        nn.init.zeros_(self.lora_b)

    def initialize_from_principal_base(self, base_weight: Any) -> None:
        if self._lora_init != "pissa":
            return
        initialize_from_principal_weight(
            lora_a=self.lora_a,
            lora_b=self.lora_b,
            base_weight=base_weight,
            scale=lora_scale(
                self._alpha(), self.rank, use_rslora=self.use_rslora
            ),
        )

    def initialize_dora_magnitude(self, base_weight: Any | None = None) -> None:
        """Match magnitude to the currently initialized grouped direction."""

        if not self.use_dora:
            return
        base = self.gora_base_weight() if base_weight is None else base_weight
        with torch.no_grad():
            direction = base.float() + (
                torch.matmul(self.lora_b.float(), self.lora_a.float())
                * lora_scale(
                    self._alpha(), self.rank, use_rslora=self.use_rslora
                )
            )
            self.dora_magnitude.copy_(
                dora_row_magnitude(direction).to(
                    device=self.dora_magnitude.device,
                    dtype=self.dora_magnitude.dtype,
                )
            )

    def set_lora_fa_enabled(self, enabled: bool) -> None:
        self._lora_fa_hook = configure_lora_fa_factors(
            lora_a=self.lora_a,
            lora_b=self.lora_b,
            scale=lambda: (
                lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
                * self._lora_scale
                * self._rank_schedule_scale
            ),
            enabled=bool(enabled),
            hook=self._lora_fa_hook,
        )

    def set_timestep_rank_masks(
        self, per_sample: Any | None, batch_uniform: Any | None
    ) -> None:
        """Install (or clear) timestep-axis rank masks.

        Weight-space adapter: only the conservative ``batch_uniform`` mask
        ``[rank]`` can apply (the delta is added to a weight shared by every
        sample); ``per_sample`` is stored for introspection only.
        """
        self._timestep_mask_per_sample = per_sample
        self._timestep_mask_uniform = batch_uniform

    def set_tc_gate(self, hypernet: Any | None, sigmas: Any | None) -> None:
        """Install (or clear) the TC-LoRA gate provider (uniform, weight-space).

        The gate is recomputed in ``forward`` so gradient checkpointing owns a
        segment-local subgraph; the hypernet is held in a 1-tuple so it is not
        registered as a duplicate submodule.
        """
        self._tc_gate_hypernet = None if hypernet is None else (hypernet,)
        self._tc_gate_sigma = sigmas

    def _tc_gate_uniform(self) -> Any | None:
        holder = self._tc_gate_hypernet
        sigma = self._tc_gate_sigma
        if holder is None or sigma is None:
            return None
        return holder[0](sigma).amin(dim=0)  # [rank], weight-space uniform gate

    def _alpha(self) -> float:
        if self._alpha_value is None:
            self._alpha_value = float(self.lora_alpha.detach().float().item())
        return self._alpha_value

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._alpha_value = float(self.lora_alpha.detach().float().item())

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

    def activation_factors(self, like: Any) -> tuple[Any, Any, float]:
        """Return effective factors for associative activation-space execution."""
        if self.use_dora:
            raise RuntimeError(
                "DoRA is a weight-normalized parametrization and cannot execute "
                "through activation-space expert LoRA."
            )
        if self.num_experts <= 0:
            raise RuntimeError("Activation-space expert LoRA requires a rank-3 tensor.")
        a = self.lora_a.to(device=like.device, dtype=like.dtype)
        b = self.lora_b.to(device=like.device, dtype=like.dtype)
        if self.training and self._rank_dropout > 0.0:
            a = F.dropout(a, p=float(self._rank_dropout), training=True)
        a, b = apply_lora_parameter_dropout(
            a,
            b,
            probability=self._lora_parameter_dropout,
            training=self.training,
        )
        mask = self._timestep_mask_uniform
        if mask is not None:
            a = a * mask.to(device=a.device, dtype=a.dtype).view(1, -1, 1)
        gate = self._tc_gate_uniform()
        if gate is not None:
            a = a * gate.to(device=a.device, dtype=a.dtype).view(1, -1, 1)
        scale = (
            lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
            * float(self._lora_scale)
            * float(self._rank_schedule_scale)
        )
        return a, b, float(scale)

    def forward(self, base: Any) -> Any:
        a = self.lora_a.to(device=base.device, dtype=base.dtype)
        b = self.lora_b.to(device=base.device, dtype=base.dtype)
        if self.training and self._rank_dropout > 0.0:
            a = F.dropout(a, p=float(self._rank_dropout), training=True)
        a, b = apply_lora_parameter_dropout(
            a,
            b,
            probability=self._lora_parameter_dropout,
            training=self.training,
        )
        mask = self._timestep_mask_uniform
        if mask is not None:
            mask = mask.to(device=a.device, dtype=a.dtype)
            # Zero the masked ranks' A rows: delta = B @ (A * mask) has zero
            # contribution and exactly-zero grads for masked ranks on both A
            # (multiplied by 0) and B (zeroed matmul operand columns).
            if a.dim() == 3:
                a = a * mask.view(1, -1, 1)
            else:
                a = a * mask.view(-1, 1)
        gate = self._tc_gate_uniform()
        if gate is not None:
            # Multiplicative TC-LoRA gate on the rank axis of A (recomputed here
            # for gradient-checkpointing safety); preserves any zeroed rank.
            gate = gate.to(device=a.device, dtype=a.dtype)
            if a.dim() == 3:
                a = a * gate.view(1, -1, 1)
            else:
                a = a * gate.view(-1, 1)
        delta = torch.matmul(b, a)
        scale = (
            lora_scale(self._alpha(), self.rank, use_rslora=self.use_rslora)
            * float(self._lora_scale)
            * float(self._rank_schedule_scale)
        )
        if self.use_dora:
            return dora_effective_weight(
                base_weight=base,
                lora_a=a,
                lora_b=b,
                magnitude=self.dora_magnitude.to(
                    device=base.device, dtype=torch.float32
                ),
                direction_scale=lora_scale(
                    self._alpha(), self.rank, use_rslora=self.use_rslora
                ),
                adapter_scale=(
                    float(self._lora_scale) * float(self._rank_schedule_scale)
                ),
            )
        return base + (delta * scale)


def collect_lora_linear_target_names(
    root: nn.Module,
    *,
    target_modules: list[str],
    include: Callable[[str, nn.Module], bool] | None = None,
) -> tuple[str, ...]:
    """Return concrete linear targets before any adapter mutates the graph."""
    targets = tuple(str(item) for item in target_modules)
    matched: list[str] = []
    for module_name, module in root.named_modules():
        if not module_name or isinstance(module, LoRALinear):
            continue
        if not isinstance(module, (nn.Linear, CompressedLinear)):
            continue
        should_wrap = (
            bool(include(module_name, module))
            if include is not None
            else any(module_name.endswith(target) for target in targets)
        )
        if should_wrap:
            matched.append(module_name)
    return tuple(matched)


def collect_lora_expert_target_names(
    *,
    target_specs: list[str],
    expert_tensor_specs: list[ExpertTensorSpec],
) -> tuple[str, ...]:
    """Return provider-declared expert adapter targets before mutation."""
    targets = tuple(str(item) for item in target_specs)
    return tuple(
        spec.name
        for spec in expert_tensor_specs
        if bool(spec.adapter_targetable) and _matches_target(spec.name, targets)
    )


def apply_lora_to_linear_modules(
    root: nn.Module,
    *,
    target_preset: str,
    target_modules: list[str],
    rank: int,
    alpha: float,
    include: Callable[[str, nn.Module], bool] | None = None,
    require_match: bool = True,
    init: str = "kaiming",
    target_allocations: Mapping[str, Any] | None = None,
    use_dora: bool = False,
) -> LoRAApplicationReport:
    if torch is None:  # pragma: no cover
        raise RuntimeError("LoRA application requires torch.")
    targets = tuple(str(item) for item in target_modules)
    matched: list[str] = []
    skipped: list[str] = []

    if use_dora:
        for module_name, module in root.named_modules():
            if not module_name or not isinstance(module, CompressedLinear):
                continue
            should_wrap = (
                bool(include(module_name, module))
                if include is not None
                else any(module_name.endswith(target) for target in targets)
            )
            if should_wrap:
                raise ValueError(
                    f"DoRA target {module_name!r} is already compressed; "
                    "DoRA requires an unpacked dense base."
                )

    for module_name, module in list(root.named_modules()):
        if not module_name:
            continue
        if isinstance(module, LoRALinear):
            skipped.append(module_name)
            continue
        if not isinstance(module, (nn.Linear, CompressedLinear)):
            continue
        if include is not None:
            should_wrap = bool(include(module_name, module))
        else:
            should_wrap = any(module_name.endswith(target) for target in targets)
        if not should_wrap:
            continue
        allocation = (target_allocations or {}).get(module_name)
        resolved_rank = int(getattr(allocation, "rank", rank))
        resolved_alpha = float(getattr(allocation, "alpha", alpha))
        use_rslora = bool(getattr(allocation, "use_rslora", False))
        _replace_child_module(
            root,
            module_name,
            LoRALinear(
                module,
                rank=resolved_rank,
                alpha=resolved_alpha,
                init=init,
                use_rslora=use_rslora,
                use_dora=use_dora,
            ),
        )
        matched.append(module_name)

    if not matched and bool(require_match):
        raise ValueError(
            f"LoRA target_preset='{target_preset}' did not match any linear modules."
        )
    return LoRAApplicationReport(
        target_preset=str(target_preset),
        target_modules=targets,
        rank=int(rank),
        alpha=float(alpha),
        matched_modules=tuple(matched),
        skipped_modules=tuple(skipped),
        allocations=tuple(
            (
                name,
                int((target_allocations or {})[name].rank),
                float((target_allocations or {})[name].alpha),
                (
                    "alpha_over_sqrt_rank"
                    if bool((target_allocations or {})[name].use_rslora)
                    else "alpha_over_rank"
                ),
            )
            for name in matched
            if name in (target_allocations or {})
        ),
    )


def apply_lora_to_expert_tensors(
    root: nn.Module,
    *,
    target_preset: str,
    target_specs: list[str],
    expert_tensor_specs: list[ExpertTensorSpec],
    rank: int,
    alpha: float,
    require_match: bool = True,
    init: str = "kaiming",
    target_allocations: Mapping[str, Any] | None = None,
    use_dora: bool = False,
) -> LoRAApplicationReport:
    if torch is None:  # pragma: no cover
        raise RuntimeError("LoRA application requires torch.")
    from torch.nn.utils import parametrize

    targets = tuple(str(item) for item in target_specs)
    matched: list[str] = []
    skipped: list[str] = []
    for spec in expert_tensor_specs:
        if not bool(spec.adapter_targetable):
            continue
        if not _matches_target(spec.name, targets):
            continue
        try:
            owner = root.get_submodule(spec.owner_module)
        except AttributeError:
            skipped.append(spec.name)
            continue
        tensor_name = str(spec.tensor_name)
        allocation = (target_allocations or {}).get(spec.name)
        resolved_rank = int(getattr(allocation, "rank", rank))
        resolved_alpha = float(getattr(allocation, "alpha", alpha))
        use_rslora = bool(getattr(allocation, "use_rslora", False))
        attach_packed = getattr(owner, "attach_expert_lora", None)
        has_packed = getattr(owner, "has_packed_weight", None)
        if callable(attach_packed) and (
            hasattr(owner, f"{tensor_name}_int8")
            or (callable(has_packed) and bool(has_packed(tensor_name)))
        ):
            if use_dora:
                raise ValueError(
                    f"DoRA target {spec.name!r} is packed; DoRA requires "
                    "weight-space access to the complete expert tensor."
                )
            if str(init).strip().lower() == "pissa":
                raise ValueError(
                    "PiSSA requires a mutable dense residual base and cannot be "
                    "initialized from an already packed expert artifact."
                )
            attach_packed(
                tensor_name=tensor_name,
                adapter_name=spec.name,
                rank=resolved_rank,
                alpha=resolved_alpha,
                init=init,
                use_rslora=use_rslora,
            )
            matched.append(spec.name)
            continue
        if not hasattr(owner, tensor_name):
            skipped.append(spec.name)
            continue
        if parametrize.is_parametrized(owner, tensor_name):
            skipped.append(spec.name)
            continue
        param = getattr(owner, tensor_name)
        if not isinstance(param, nn.Parameter):
            skipped.append(spec.name)
            continue
        param.requires_grad_(False)
        parametrize.register_parametrization(
            owner,
            tensor_name,
            LoRAExpertTensorParametrization(
                adapter_name=spec.name,
                shape=tuple(spec.shape),
                layout=tuple(spec.layout),
                rank=resolved_rank,
                alpha=resolved_alpha,
                init=init,
                use_rslora=use_rslora,
                use_dora=use_dora,
                base_weight=param,
            ),
            unsafe=False,
        )
        owner.parametrizations[tensor_name].original.requires_grad_(False)
        parametrization = owner.parametrizations[tensor_name][-1]
        parametrization.initialize_from_principal_base(
            owner.parametrizations[tensor_name].original
        )
        parametrization.initialize_dora_magnitude(
            owner.parametrizations[tensor_name].original
        )
        matched.append(spec.name)
    if not matched and bool(require_match):
        raise ValueError(
            f"LoRA target_preset='{target_preset}' did not match any expert tensors."
        )
    return LoRAApplicationReport(
        target_preset=str(target_preset),
        target_modules=targets,
        rank=int(rank),
        alpha=float(alpha),
        matched_modules=tuple(matched),
        skipped_modules=tuple(skipped),
        allocations=tuple(
            (
                name,
                int((target_allocations or {})[name].rank),
                float((target_allocations or {})[name].alpha),
                (
                    "alpha_over_sqrt_rank"
                    if bool((target_allocations or {})[name].use_rslora)
                    else "alpha_over_rank"
                ),
            )
            for name in matched
            if name in (target_allocations or {})
        ),
    )


def iter_lora_modules(root: nn.Module):
    for name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def iter_lora_expert_tensor_modules(root: nn.Module):
    for _, module in root.named_modules():
        if isinstance(module, LoRAExpertTensorParametrization) or bool(
            getattr(module, "_mirai_expert_lora_adapter", False)
        ):
            yield module.adapter_name, module


def lora_state_dict(
    root: nn.Module, *, sparse_expert_export: bool = False
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    sparse_policy = SparseExpertExportPolicy(enabled=bool(sparse_expert_export))
    for name, module in iter_lora_modules(root):
        state[f"{name}.lora_a"] = module.lora_a.detach()
        state[f"{name}.lora_b"] = module.lora_b.detach()
        state[f"{name}.lora_alpha"] = module.lora_alpha.detach().reshape(1)
        if bool(module.use_rslora):
            state[f"{name}{RSLORA_STATE_SUFFIX}"] = torch.ones(
                1, dtype=torch.uint8, device=module.lora_a.device
            )
        if bool(module.use_dora):
            state[f"{name}{DORA_MAGNITUDE_SUFFIX}"] = (
                module.dora_magnitude.detach()
            )
    for name, module in iter_lora_expert_tensor_modules(root):
        selection_active = selection_is_active(module)
        if bool(getattr(module, "use_rslora", False)):
            state[f"{name}{RSLORA_STATE_SUFFIX}"] = torch.ones(
                1, dtype=torch.uint8, device=module.lora_a.device
            )
        if bool(getattr(module, "use_dora", False)):
            state[f"{name}{DORA_MAGNITUDE_SUFFIX}"] = (
                module.dora_magnitude.detach()
            )
        if sparse_policy.write_module_state(state, name=name, module=module):
            continue
        state[f"{name}.lora_a"] = module.lora_a.detach()
        state[f"{name}.lora_b"] = module.lora_b.detach()
        state[f"{name}.lora_alpha"] = module.lora_alpha.detach().reshape(1)
        # Persist routing-guided expert selection (MoE-Sieve) so the per-expert
        # mask survives checkpoint save/load. Only exported when selection is
        # active; "all" mode retains full expert tensors.
        if selection_active:
            state[f"{name}{MASK_SUFFIX}"] = module.active_expert_mask.detach()
        # Condenser LoRA (shared always-on guardrail): exported only when
        # attached; rank 0 disables the condenser.
        write_condenser_state(state, name=name, module=module)
    return state


def load_lora_state_dict(root: nn.Module, state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise ValueError("Adapter state must be a dictionary.")
    modules = dict(iter_lora_modules(root))
    modules.update(dict(iter_lora_expert_tensor_modules(root)))
    missing: list[str] = []
    _known_suffixes = (
        ".lora_a",
        ".lora_b",
        ".lora_alpha",
        RSLORA_STATE_SUFFIX,
        DORA_MAGNITUDE_SUFFIX,
        PARA_RANK_STATE_SUFFIX,
        MASK_SUFFIX,
        SPARSE_A_SUFFIX,
        SPARSE_B_SUFFIX,
        SPARSE_IDS_SUFFIX,
        CONDENSER_A_SUFFIX,
        CONDENSER_B_SUFFIX,
        CONDENSER_ALPHA_SUFFIX,
    )
    unexpected = [
        key
        for key in state
        if any(key.endswith(sfx) for sfx in _known_suffixes)
        and key.rsplit(".", 1)[0] not in modules
    ]
    with torch.no_grad():
        for name, module in modules.items():
            para_rank_key = f"{name}{PARA_RANK_STATE_SUFFIX}"
            if para_rank_key in state:
                raw_para_rank = state[para_rank_key]
                if (
                    not isinstance(raw_para_rank, torch.Tensor)
                    or raw_para_rank.numel() != 1
                    or int(raw_para_rank.detach().cpu().item()) <= 0
                ):
                    raise ValueError(
                        f"PARA runtime rank '{para_rank_key}' must be a "
                        "one-element positive integer tensor."
                    )
                para_rank = int(raw_para_rank.detach().cpu().item())
                if para_rank != int(module.rank):
                    resize_lora_module(module, para_rank)
            scaling_key = f"{name}{RSLORA_STATE_SUFFIX}"
            configured_rslora = bool(getattr(module, "use_rslora", False))
            stored_rslora = False
            if scaling_key in state:
                raw_rule = state[scaling_key]
                if not isinstance(raw_rule, torch.Tensor) or raw_rule.numel() != 1:
                    raise ValueError(
                        f"Adapter scaling rule '{scaling_key}' must be a one-element tensor."
                    )
                stored_rslora = bool(raw_rule.detach().cpu().item())
            if configured_rslora != stored_rslora:
                raise ValueError(
                    f"Adapter scaling rule mismatch for '{name}': checkpoint "
                    f"use_rslora={stored_rslora}, configured use_rslora={configured_rslora}."
                )
            magnitude_key = f"{name}{DORA_MAGNITUDE_SUFFIX}"
            configured_dora = bool(getattr(module, "use_dora", False))
            stored_dora = magnitude_key in state
            if configured_dora != stored_dora:
                raise ValueError(
                    f"DoRA state mismatch for '{name}': checkpoint "
                    f"use_dora={stored_dora}, configured use_dora={configured_dora}."
                )
            a_sel_key = f"{name}{SPARSE_A_SUFFIX}"
            if a_sel_key in state:
                load_compact_expert_adapter(module, name, state)
                if hasattr(module, "_alpha_value"):
                    module._alpha_value = float(
                        module.lora_alpha.detach().float().item()
                    )
                continue
            a_key = f"{name}.lora_a"
            b_key = f"{name}.lora_b"
            if a_key not in state or b_key not in state:
                missing.append(name)
                continue
            if str(getattr(module, "_lora_init", "")).strip().lower() == "gora":
                raw_a = state[a_key]
                raw_b = state[b_key]
                if not isinstance(raw_a, torch.Tensor) or not isinstance(
                    raw_b, torch.Tensor
                ):
                    raise ValueError(
                        f"GoRA adapter factors for '{name}' must be tensors."
                    )
                stored_rank = int(raw_a.shape[-2])
                if int(raw_b.shape[-1]) != stored_rank:
                    raise ValueError(
                        f"GoRA adapter factors for '{name}' disagree on rank."
                    )
                if stored_rank != int(module.rank):
                    resize_lora_module(module, stored_rank)
            module.lora_a.copy_(
                _adapter_tensor(state[a_key], like=module.lora_a, name=a_key)
            )
            module.lora_b.copy_(
                _adapter_tensor(state[b_key], like=module.lora_b, name=b_key)
            )
            if configured_dora:
                module.dora_magnitude.copy_(
                    _adapter_tensor(
                        state[magnitude_key],
                        like=module.dora_magnitude,
                        name=magnitude_key,
                    ).reshape_as(module.dora_magnitude)
                )
            alpha_key = f"{name}.lora_alpha"
            if alpha_key in state:
                module.lora_alpha.copy_(
                    _adapter_tensor(
                        state[alpha_key],
                        like=module.lora_alpha,
                        name=alpha_key,
                    ).reshape_as(module.lora_alpha)
                )
            mask_key = f"{name}.active_expert_mask"
            if mask_key in state and hasattr(module, "active_expert_mask"):
                module.active_expert_mask.copy_(
                    _adapter_tensor(
                        state[mask_key],
                        like=module.active_expert_mask,
                        name=mask_key,
                    ).reshape_as(module.active_expert_mask)
                )
                if hasattr(module, "_expert_selection_active"):
                    module._expert_selection_active = bool(
                        (module.active_expert_mask < 1.0).any().item()
                    )
            load_condenser_state(state, name=name, module=module)
            if hasattr(module, "_alpha_value"):
                module._alpha_value = float(
                    module.lora_alpha.detach().float().item()
                )
    if missing:
        raise ValueError(
            "Adapter state is missing LoRA tensors for modules: "
            + ", ".join(sorted(missing))
        )
    if unexpected:
        raise ValueError(
            "Adapter state contains LoRA tensors for unknown modules: "
            + ", ".join(sorted(unexpected))
        )


def set_lora_scale(root: nn.Module, scale: float) -> None:
    for _, module in iter_lora_modules(root):
        module.set_lora_scale(scale)
    for _, module in iter_lora_expert_tensor_modules(root):
        module.set_lora_scale(scale)


def set_lora_rank_dropout(root: nn.Module, dropout: float) -> None:
    for _, module in iter_lora_modules(root):
        module.set_rank_dropout(dropout)
    for _, module in iter_lora_expert_tensor_modules(root):
        module.set_rank_dropout(dropout)


def set_lora_parameter_dropout(root: nn.Module, dropout: float) -> None:
    for _, module in iter_lora_modules(root):
        module.set_lora_parameter_dropout(dropout)
    for _, module in iter_lora_expert_tensor_modules(root):
        module.set_lora_parameter_dropout(dropout)


def set_lora_rank_schedule_scale(root: nn.Module, scale: float) -> None:
    for _, module in iter_lora_modules(root):
        module.set_rank_schedule_scale(scale)
    for _, module in iter_lora_expert_tensor_modules(root):
        module.set_rank_schedule_scale(scale)


def enable_expert_lora_condensers(
    root: nn.Module, *, rank: int, alpha: float, init: str = "kaiming"
) -> int:
    """Attach the shared condenser LoRA to every routed-expert adapter.

    Returns the number of adapters that received a condenser. No-op at rank<=0
    or for adapter modules that do not support a condenser (e.g. the weight-space
    parametrization path); the count lets the caller assert coverage.
    """
    if int(rank) <= 0:
        return 0
    attached = 0
    for _, module in iter_lora_expert_tensor_modules(root):
        enable = getattr(module, "enable_condenser", None)
        if callable(enable):
            enable(rank=int(rank), alpha=float(alpha), init=str(init))
            if bool(getattr(module, "has_condenser", lambda: False)()):
                attached += 1
    return attached


def collect_lora_adapter_modules(root: nn.Module) -> list[nn.Module]:
    """All adapter modules (LoRALinear + expert-tensor adapters) under root.

    Caches the module list once at configuration time so per-step state
    distribution (timestep masks) avoids a full named_modules() walk.
    """
    modules: list[nn.Module] = [m for _, m in iter_lora_modules(root)]
    modules.extend(m for _, m in iter_lora_expert_tensor_modules(root))
    return modules


def set_lora_timestep_rank_masks(
    modules: Any,
    per_sample: Any | None,
    batch_uniform: Any | None,
) -> None:
    """Distribute timestep-axis rank masks to adapter modules.

    ``modules`` is either a root ``nn.Module`` or a pre-collected list from
    ``collect_lora_adapter_modules``. Pass ``None, None`` to clear.
    """
    if isinstance(modules, nn.Module):
        modules = collect_lora_adapter_modules(modules)
    for module in modules:
        module.set_timestep_rank_masks(per_sample, batch_uniform)


def set_lora_tc_gate(
    modules: Any,
    hypernet: Any | None,
    sigmas: Any | None,
) -> None:
    """Distribute the TC-LoRA gate provider to adapter modules.

    Each adapter RECOMPUTES the gate from ``hypernet(sigmas)`` inside its own
    forward, so gradient checkpointing gives every recomputed block a
    segment-local gate subgraph (the shared-differentiable-tensor double-backward
    is avoided) while gradient still flows to the hypernet. Pass ``None, None``
    to clear. ``modules`` is a root ``nn.Module`` or a pre-collected list.
    """
    if isinstance(modules, nn.Module):
        modules = collect_lora_adapter_modules(modules)
    for module in modules:
        setter = getattr(module, "set_tc_gate", None)
        if callable(setter):
            setter(hypernet, sigmas)


def move_lora_resident_tensors_to_device(root: nn.Module, *, device: Any) -> None:
    """Move LoRA/small resident tensors to device and keep frozen packed weights on CPU."""

    if torch is None:  # pragma: no cover
        return
    target = torch.device(device)
    _move_resident_module(root, device=target)


def _move_resident_module(module: nn.Module, *, device: Any) -> None:
    _move_own_tensors(module, device=device)
    for child in module.children():
        if isinstance(child, (CompressedLinear, CompressedGroupedExperts)):
            _move_module_tensors(child, device=torch.device("cpu"))
            for param in child.parameters(recurse=True):
                if bool(param.requires_grad) and param.device != device:
                    param.data = param.data.to(device=device)
            continue
        if isinstance(child, LoRALinear):
            _move_own_tensors(child, device=device)
            _move_module_tensors(child.base, device=torch.device("cpu"))
            continue
        _move_resident_module(child, device=device)


def _move_own_tensors(module: nn.Module, *, device: Any) -> None:
    for param in module.parameters(recurse=False):
        if param.device != device:
            param.data = param.data.to(device=device)
        if param.grad is not None and param.grad.device != device:
            param.grad.data = param.grad.data.to(device=device)
    for buffer in module.buffers(recurse=False):
        if buffer.device != device:
            buffer.data = buffer.data.to(device=device)


def _move_module_tensors(module: nn.Module, *, device: Any) -> None:
    for param in module.parameters(recurse=True):
        if param.device != device:
            param.data = param.data.to(device=device)
        if param.grad is not None and param.grad.device != device:
            param.grad.data = param.grad.data.to(device=device)
    for buffer in module.buffers(recurse=True):
        if buffer.device != device:
            buffer.data = buffer.data.to(device=device)


def _replace_child_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def _matches_target(name: str, targets: tuple[str, ...]) -> bool:
    if targets == ("*",):
        return True
    return any(str(name).endswith(str(target)) for target in targets)


def _adapter_tensor(value: Any, *, like: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=like.dtype)
    tensor = value.detach().to(device=like.device, dtype=like.dtype)
    if tuple(tensor.shape) != tuple(like.shape):
        if int(tensor.numel()) != int(like.numel()):
            raise ValueError(
                f"Adapter tensor '{name}' has shape {tuple(tensor.shape)}, "
                f"expected {tuple(like.shape)}."
            )
        tensor = tensor.reshape_as(like)
    return tensor
