"""Grouped-GEMM expert execution for MAGI-2 multi-head sparse-MoE layers.

Expert numerics reproduce SandAI's Apache-2.0 reference implementation vendored
at ``mirai/vendors/magi2_preview/model/magi2_preview.py``
(``CoreMultiHeadMoE._torch_forward``), which remains the reference path.

Tensor semantics: ``x_heads`` is ``[tokens, heads, d_head]`` and the packed
expert weights are indexed by a single flattened ``head * num_experts + expert``
axis, so one sorted token-slice layout covers every head at once. Routing,
activation, and combine stay in plain autograd; only the per-group expert matmul
is a custom Function, because its weights are frozen and carry no gradient.

The ``torch_grouped`` primitive additionally requires every operand stride to be
a multiple of 16 bytes, which for BF16 experts means ``d_head`` and
``expert_intermediate_size`` must both be multiples of 8. The expert layout
decides this before any forward runs: an ``auto`` selection resolves to ``bmm``
when the layout does not satisfy it, and an explicit ``torch_grouped`` selection
raises. There is no per-call downgrade.

The flattened ``head * num_experts`` axis makes the group count of a full-size
variant exceed the primitive's 1024-group per-call cap, so every
``torch_grouped`` call goes through ``run_grouped_mm``, which splits it into
contiguous group segments. The ``bmm`` path loops per group and has no cap.

When the routed experts are stored in NF4 (see ``quantized_experts.py``) the
same routing, activation, and combine run against segments dequantized on
demand: a contiguous group range owns a contiguous sorted-token range, so a
segment can be materialized, multiplied, and released before the next one
exists. The vendored reference loop has no dense tensor to read in that mode, so
grouped execution is the only path and an explicit ``torch`` kernel backend is
rejected rather than reinterpreted.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

import torch

from mirai.core.moe.runtime.gemm import (
    GROUPED_MM_ALIGNMENT_BYTES,
    BackendProbe,
    grouped_mm_op,
    grouped_mm_operand,
    grouped_mm_row_operand,
    grouped_mm_segments,
    grouped_mm_stride_violations,
    normalize_moe_gemm_backend,
    normalize_moe_gemm_role_backend,
    probe_backend,
    run_grouped_dx,
    run_grouped_mm,
)
from mirai.core.models.magi2_preview.quantized_experts import (
    MAGI2_ROUTED_EXPERT_TENSOR_NAMES,
    Magi2Nf4ExpertStore,
    magi2_expert_store,
)
from mirai.core.moe.runtime.specs import (
    MoEOptimizationPolicy,
    normalize_expert_weight_access_policy,
    normalize_moe_dispatch_mode,
    normalize_moe_dispatch_preprocess,
    normalize_packed_state_preload,
    normalize_packed_stream_backend,
    normalize_router_quantization_policy,
)


# Grouped execution supports the two device-portable primitives only. The
# persistent Triton kernel expects the compressed-weights slice layout and
# deepgemm_fp8 expects block-scaled FP8 experts; MAGI-2 experts are plain BF16.
MAGI2_GROUPED_GEMM_BACKENDS = ("auto", "bmm", "torch_grouped")

_GROUPED_KERNEL_BACKENDS = ("grouped",)
_DEFAULT_KERNEL_BACKENDS = ("auto", "torch")


class Magi2GroupedMoEPolicyError(ValueError):
    """Raised when a requested MoE policy is not implementable for MAGI-2."""


@dataclass(frozen=True)
class Magi2GroupedMoEPlan:
    """Resolved grouped-GEMM primitive selection for the two matmul roles."""

    forward_backend: str = "auto"
    dx_backend: str = "auto"

    def __post_init__(self) -> None:
        for field, value in (
            ("forward_backend", self.forward_backend),
            ("dx_backend", self.dx_backend),
        ):
            normalized = normalize_moe_gemm_backend(value)
            if normalized not in MAGI2_GROUPED_GEMM_BACKENDS:
                raise Magi2GroupedMoEPolicyError(
                    "MAGI-2 grouped MoE execution supports "
                    + ", ".join(MAGI2_GROUPED_GEMM_BACKENDS)
                    + f"; got '{normalized}' for the {field.split('_')[0]} role."
                )
            object.__setattr__(self, field, normalized)


def magi2_grouped_mm_alignment_violations(
    *,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> tuple[str, ...]:
    """Stride-precondition failures for every operand ``torch_grouped`` receives.

    Packed experts are ``W_gate``/``W_up`` ``[groups, d_head, d_expert]`` and
    ``W_down`` ``[groups, d_expert, d_head]``. The forward role passes the stored
    weight together with a contiguous ``[tokens, K]`` activation; the dX role
    passes a transposed weight VIEW, whose unit stride moves to the inner
    dimension and leaves the larger dimension carrying the checked stride. Both
    roles are enumerated here from the real tensors, so the verdict is derived
    from layout rather than from a shape heuristic.
    """
    operands = [
        grouped_mm_row_operand(
            label="x_sorted forward activation rows",
            columns=int(w_gate.shape[-2]),
            element_size=int(w_gate.element_size()),
        ),
        grouped_mm_operand(w_gate, label="W_gate forward expert weight"),
        grouped_mm_operand(w_up, label="W_up forward expert weight"),
        grouped_mm_row_operand(
            label="hidden forward activation rows",
            columns=int(w_down.shape[-2]),
            element_size=int(w_down.element_size()),
        ),
        grouped_mm_operand(w_down, label="W_down forward expert weight"),
        grouped_mm_row_operand(
            label="grad_output dX rows for W_gate/W_up",
            columns=int(w_gate.shape[-1]),
            element_size=int(w_gate.element_size()),
        ),
        grouped_mm_operand(
            w_gate.transpose(-2, -1), label="W_gate dX transposed weight view"
        ),
        grouped_mm_operand(
            w_up.transpose(-2, -1), label="W_up dX transposed weight view"
        ),
        grouped_mm_row_operand(
            label="grad_output dX rows for W_down",
            columns=int(w_down.shape[-1]),
            element_size=int(w_down.element_size()),
        ),
        grouped_mm_operand(
            w_down.transpose(-2, -1), label="W_down dX transposed weight view"
        ),
    ]
    return grouped_mm_stride_violations(operands)


def _alignment_rejection(role: str, violations: tuple[str, ...]) -> str:
    return (
        f"memory.moe_gemm_backend (or its '{role}' role override) selected "
        "'torch_grouped', which requires every operand stride to be a multiple "
        f"of {GROUPED_MM_ALIGNMENT_BYTES} bytes. This model's MAGI-2 expert "
        "layout does not satisfy it: "
        + " ".join(violations)
        + " Select 'bmm', or 'auto' to fall back to 'bmm' automatically."
    )


def select_grouped_backends(
    plan: Magi2GroupedMoEPlan,
    *,
    probe: Callable[[str], BackendProbe],
    alignment_violations: tuple[str, ...],
    device_label: str,
) -> tuple[str, str]:
    """Decide the forward/dX primitives once, from probes and the expert layout.

    ``auto`` falls back to ``bmm`` when ``torch_grouped`` is unavailable OR when
    the expert layout violates its stride precondition. An explicit
    ``torch_grouped`` raises instead of downgrading.
    """
    resolved: list[str] = []
    for role, requested in (
        ("forward", plan.forward_backend),
        ("dx", plan.dx_backend),
    ):
        if requested == "auto":
            usable = not alignment_violations and probe("torch_grouped").available
            resolved.append("torch_grouped" if usable else "bmm")
            continue
        if requested == "torch_grouped" and alignment_violations:
            raise Magi2GroupedMoEPolicyError(
                _alignment_rejection(role, alignment_violations)
            )
        result = probe(requested)
        if not result.available:
            raise RuntimeError(
                f"memory.moe_gemm_backend (or its '{role}' role override) "
                f"selected '{requested}', which is unavailable for MAGI-2 "
                f"grouped MoE execution on {device_label}: {result.reason}."
            )
        resolved.append(requested)
    return resolved[0], resolved[1]


def _grouped_boundaries(offsets: torch.Tensor) -> list[int]:
    return [int(value) for value in offsets.detach().to("cpu", torch.int64).tolist()]


def _grouped_linear(
    x_sorted: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """``y[i] = x_sorted[i] @ weight[group(i)]`` over the sorted token layout."""
    if backend == "torch_grouped":
        op = grouped_mm_op()
        if op is None:
            raise RuntimeError(
                "MAGI-2 grouped MoE requested torch grouped_mm, which this torch "
                "build does not provide."
            )
        return run_grouped_mm(op, x_sorted.contiguous(), weight, offsets)
    result = x_sorted.new_empty((int(x_sorted.shape[0]), int(weight.shape[-1])))
    start = 0
    for group, stop in enumerate(_grouped_boundaries(offsets)):
        if stop > start:
            result[start:stop] = x_sorted[start:stop] @ weight[group]
        start = stop
    return result


def _grouped_linear_dx(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Input gradient ``dX[i] = dY[i] @ weight[group(i)]^T``.

    The transposed weight stays a view: materializing it would copy the whole
    frozen expert tensor once per backward.
    """
    weight_t = weight.transpose(-2, -1)
    if backend == "torch_grouped":
        op = grouped_mm_op()
        if op is None:
            raise RuntimeError(
                "MAGI-2 grouped MoE requested torch grouped_mm for the dX role, "
                "which this torch build does not provide."
            )
        return run_grouped_mm(op, grad_output.contiguous(), weight_t, offsets)
    return run_grouped_dx(grad_output, weight_t, offsets, backend=backend)


class _GroupedExpertLinear(torch.autograd.Function):
    """Per-group expert matmul against frozen packed expert weights."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x_sorted: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
        forward_backend: str,
        dx_backend: str,
    ) -> torch.Tensor:
        if weight.requires_grad:
            raise RuntimeError(
                "MAGI-2 grouped MoE execution requires frozen expert weights; "
                "W_gate/W_up/W_down must not require gradients. Use "
                "memory.moe_kernel_backend='auto' for a trainable-expert path."
            )
        ctx.save_for_backward(weight, offsets)
        ctx.dx_backend = dx_backend
        return _grouped_linear(x_sorted, weight, offsets, forward_backend)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        weight, offsets = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = _grouped_linear_dx(
                grad_output, weight, offsets, ctx.dx_backend
            )
        return grad_x, None, None, None, None


class Magi2GroupedMoEBackend:
    """Grouped execution seam attached to vendored ``CoreMultiHeadMoE`` layers."""

    name = "grouped"

    def __init__(self, plan: Magi2GroupedMoEPlan) -> None:
        self.plan = plan
        self._resolved: dict[str, tuple[str, str]] = {}
        self._alignment: tuple[str, ...] | None = None

    @property
    def alignment_violations(self) -> tuple[str, ...]:
        """Operand layouts that fail the ``torch_grouped`` stride precondition.

        Empty until an expert layout has been inspected, and frozen once the
        first primitive selection has been resolved.
        """
        return self._alignment if self._alignment is not None else ()

    def inspect_expert_layout(self, module: Any) -> tuple[str, ...]:
        """Record ``module``'s expert-layout verdict; attach time calls this.

        Layers share one plan, so verdicts accumulate until the first resolution
        fixes the selection. This keeps the decision observable and stable rather
        than re-derived per call.
        """
        if self._resolved:
            return self.alignment_violations
        violations = magi2_grouped_mm_alignment_violations(
            w_gate=module.W_gate, w_up=module.W_up, w_down=module.W_down
        )
        merged = list(self._alignment or ())
        for reason in violations:
            if reason not in merged:
                merged.append(reason)
        self._alignment = tuple(merged)
        return self._alignment

    def validate_explicit_alignment(self) -> None:
        """Raise when an explicitly requested ``torch_grouped`` cannot be honored."""
        violations = self.alignment_violations
        if not violations:
            return
        for role, requested in (
            ("forward", self.plan.forward_backend),
            ("dx", self.plan.dx_backend),
        ):
            if requested == "torch_grouped":
                raise Magi2GroupedMoEPolicyError(
                    _alignment_rejection(role, violations)
                )

    def _resolve(self, device: torch.device, module: Any = None) -> tuple[str, str]:
        if module is not None and self._alignment is None:
            # Backstop for a path that never reached attach-time inspection; the
            # policy is identical, only its trigger point differs.
            self.inspect_expert_layout(module)
        key = f"{device.type}:{device.index if device.index is not None else -1}"
        cached = self._resolved.get(key)
        if cached is not None:
            return cached
        selection = select_grouped_backends(
            self.plan,
            probe=lambda name: probe_backend(name, device=device),
            alignment_violations=self.alignment_violations,
            device_label=str(device),
        )
        self._resolved[key] = selection
        return selection

    def expert_weight_dtype(self, module: Any, key: str) -> torch.dtype:
        """Storage dtype of one routed expert tensor of ``module``."""
        return getattr(module, key).dtype

    def project(
        self,
        module: Any,
        key: str,
        x_sorted: torch.Tensor,
        offsets: torch.Tensor,
        forward_backend: str,
        dx_backend: str,
    ) -> torch.Tensor:
        """One grouped expert matmul against the frozen packed tensor ``key``."""
        return _GroupedExpertLinear.apply(
            x_sorted, getattr(module, key), offsets, forward_backend, dx_backend
        )

    def execute(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        forward_backend, dx_backend = self._resolve(x_heads.device, module)
        device = x_heads.device
        tokens = int(x_heads.shape[0])
        num_heads = int(x_heads.shape[1])
        d_head = int(x_heads.shape[2])
        num_experts = int(module.num_experts)
        groups = num_heads * num_experts

        # Flatten (head, expert) onto the packed weight's single expert axis and
        # (token, head) onto the row axis of the contiguous [tokens, heads, d_head]
        # activation, then sort every routed slot by its flattened expert.
        head_axis = torch.arange(num_heads, device=device).view(num_heads, 1, 1)
        token_axis = torch.arange(tokens, device=device).view(1, tokens, 1)
        flat_experts = (topk_indices + head_axis * num_experts).reshape(-1)
        rows = (token_axis * num_heads + head_axis).expand_as(topk_indices).reshape(-1)
        order = flat_experts.argsort(stable=True)
        sorted_rows = rows[order]
        sorted_probs = topk_probs.reshape(-1)[order]
        counts = torch.bincount(flat_experts, minlength=groups)
        offsets = counts.cumsum(0).to(torch.int32)

        x_rows = x_heads.reshape(tokens * num_heads, d_head)
        x_sorted = x_rows.index_select(0, sorted_rows)
        gate = self.project(
            module, "W_gate", x_sorted, offsets, forward_backend, dx_backend
        )
        up = self.project(
            module, "W_up", x_sorted, offsets, forward_backend, dx_backend
        )
        gate = gate.float().clamp(max=7.0)
        up = up.float().clamp(min=-7.0, max=7.0)
        hidden = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
        expert_output = self.project(
            module,
            "W_down",
            hidden.to(self.expert_weight_dtype(module, "W_down")),
            offsets,
            forward_backend,
            dx_backend,
        )
        weighted = expert_output * sorted_probs.to(expert_output.dtype)[:, None]
        output = torch.zeros(
            tokens * num_heads, d_head, device=device, dtype=x_heads.dtype
        ).index_add(0, sorted_rows, weighted.to(x_heads.dtype))
        return output.view(tokens, num_heads, d_head)


def run_segmented_grouped_linear(
    x_sorted: torch.Tensor,
    offsets: torch.Tensor,
    *,
    boundaries: list[int],
    materialize: Callable[[int, int], torch.Tensor],
    backend: str,
    max_groups: int,
    columns: int,
    transposed: bool = False,
) -> torch.Tensor:
    """Run the sorted-layout grouped matmul one contiguous group range at a time.

    ``materialize(group_start, group_stop)`` returns the dense ``[groups, K, N]``
    weight for that range. The jagged-offset contract sorts rows by group, so a
    group range owns exactly one row range and the segment outputs concatenate
    back into the original row order without a scatter. Each materialized
    segment is dropped before the next one is produced, which is what bounds the
    dense working set to ``max_groups`` weight slices instead of the whole
    packed axis.

    ``transposed`` selects the input-gradient role, which multiplies by the
    transposed weight view; ``columns`` is the output width of the selected role
    and only decides the shape of an all-empty result.
    """
    outputs: list[torch.Tensor] = []
    for segment in grouped_mm_segments(boundaries, max_groups=max(1, int(max_groups))):
        if segment.row_count == 0:
            continue
        weight = materialize(segment.group_start, segment.group_stop)
        local_offsets = (
            offsets[segment.group_start : segment.group_stop] - segment.row_start
        ).contiguous()
        rows = x_sorted[segment.row_start : segment.row_stop]
        outputs.append(
            _grouped_linear_dx(rows, weight, local_offsets, backend)
            if transposed
            else _grouped_linear(rows, weight, local_offsets, backend)
        )
        del weight
    if not outputs:
        return x_sorted.new_empty((0, int(columns)))
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


class _QuantizedGroupedExpertLinear(torch.autograd.Function):
    """Grouped expert matmul against a frozen NF4 payload.

    The dequantized segment buffers are produced under ``no_grad`` inside the
    store and are never saved on the context: backward re-materializes the same
    segments from the same packed payload, so the dense expert weights exist
    only for the duration of one segment matmul and never survive in autograd.
    Re-materialization is exact rather than approximate -- NF4 dequantization is
    a deterministic function of the stored payload -- so the input gradient is
    computed against the same values the forward used.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x_sorted: torch.Tensor,
        offsets: torch.Tensor,
        store: Magi2Nf4ExpertStore,
        key: str,
        forward_backend: str,
        dx_backend: str,
    ) -> torch.Tensor:
        boundaries = _grouped_boundaries(offsets)
        _groups, rows, cols = store.expert_weight_shape(key)
        span = store.segment_group_span()
        output = run_segmented_grouped_linear(
            x_sorted,
            offsets,
            boundaries=boundaries,
            materialize=lambda start, stop: store.materialize_segment(
                key, start, stop, dtype=x_sorted.dtype, device=x_sorted.device
            ),
            backend=forward_backend,
            max_groups=span,
            columns=int(cols),
        )
        ctx.save_for_backward(offsets)
        ctx.store = store
        ctx.key = str(key)
        ctx.dx_backend = dx_backend
        ctx.boundaries = boundaries
        ctx.segment_span = int(span)
        ctx.weight_rows = int(rows)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        (offsets,) = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            store = ctx.store
            key = ctx.key
            grad_x = run_segmented_grouped_linear(
                grad_output,
                offsets,
                boundaries=ctx.boundaries,
                materialize=lambda start, stop: store.materialize_segment(
                    key, start, stop, dtype=grad_output.dtype, device=grad_output.device
                ),
                backend=ctx.dx_backend,
                max_groups=ctx.segment_span,
                columns=ctx.weight_rows,
                transposed=True,
            )
        return grad_x, None, None, None, None, None


class Magi2QuantizedGroupedMoEBackend(Magi2GroupedMoEBackend):
    """Grouped execution over NF4-packed MAGI-2 routed experts.

    The dense expert parameters no longer exist on the vendored layer, so both
    the expert-layout verdict and the matmul operands come from the layer's
    :class:`~mirai.core.models.magi2_preview.quantized_experts.Magi2Nf4ExpertStore`.
    Routing, activation, and the probability-weighted combine stay in the
    inherited native-autograd path.
    """

    name = "grouped_nf4"

    @staticmethod
    def _store(module: Any) -> Magi2Nf4ExpertStore:
        store = magi2_expert_store(module)
        if store is None:
            raise Magi2GroupedMoEPolicyError(
                "MAGI-2 quantized grouped execution requires packed NF4 experts "
                "on every multi-head MoE layer; this layer carries none."
            )
        if not store.is_fully_loaded():
            raise Magi2GroupedMoEPolicyError(
                "MAGI-2 packed experts are incomplete: "
                + ", ".join(MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
                + " must all be quantized before a forward pass."
            )
        return store

    def inspect_expert_layout(self, module: Any) -> tuple[str, ...]:
        if self._resolved:
            return self.alignment_violations
        store = self._store(module)
        violations = magi2_grouped_mm_alignment_violations(
            w_gate=store.layout_probe("W_gate"),
            w_up=store.layout_probe("W_up"),
            w_down=store.layout_probe("W_down"),
        )
        merged = list(self._alignment or ())
        for reason in violations:
            if reason not in merged:
                merged.append(reason)
        self._alignment = tuple(merged)
        return self._alignment

    def expert_weight_dtype(self, module: Any, key: str) -> torch.dtype:
        return self._store(module).expert_weight_dtype(key)

    def project(
        self,
        module: Any,
        key: str,
        x_sorted: torch.Tensor,
        offsets: torch.Tensor,
        forward_backend: str,
        dx_backend: str,
    ) -> torch.Tensor:
        return _QuantizedGroupedExpertLinear.apply(
            x_sorted, offsets, self._store(module), key, forward_backend, dx_backend
        )


# The only MoEOptimizationPolicy fields MAGI-2 grouped execution reads. Every
# other field must still hold its dataclass default, so a field added to the
# generic policy fails closed here instead of being silently ignored.
_CONSUMED_POLICY_FIELDS = frozenset(
    {
        "kernel_backend",
        "moe_gemm_backend",
        "moe_gemm_backend_forward",
        "moe_gemm_backend_dx",
    }
)

# Fields that additionally acquire a consumer once the routed experts are stored
# in a packed format: the access policy and its chunk size select the
# dequantization granularity, and the on-load flag selects the streaming
# quantized checkpoint path.
_QUANTIZED_CONSUMED_POLICY_FIELDS = _CONSUMED_POLICY_FIELDS | frozenset(
    {
        "expert_weight_access",
        "expert_dequant_chunk_size",
        "quantize_experts_on_load",
    }
)

# Non-default values that request no behavior at all for this family.
_INERT_POLICY_VALUES: dict[str, tuple[Any, ...]] = {
    "expert_weight_access": ("disabled",),
    "moe_gemm_backend_dw": ("auto",),
}

# Alias-tolerant comparison: a duck-typed policy may carry an unnormalized
# spelling of a value that the dataclass would canonicalize on construction.
_POLICY_FIELD_NORMALIZERS: dict[str, Callable[[Any], Any]] = {
    "expert_weight_access": normalize_expert_weight_access_policy,
    "router_quantization": normalize_router_quantization_policy,
    "packed_state_preload": normalize_packed_state_preload,
    "packed_stream_backend": normalize_packed_stream_backend,
    "moe_dispatch": normalize_moe_dispatch_mode,
    "moe_dispatch_preprocess": normalize_moe_dispatch_preprocess,
    "moe_gemm_backend_dw": normalize_moe_gemm_role_backend,
}

_POLICY_FIELD_NOTES: dict[str, str] = {
    "expert_weight_access": (
        "This family keeps native BF16 experts unless "
        "memory.frozen_weight_quantization='nf4' packs them."
    ),
    "expert_dequant_chunk_size": (
        "Chunked expert access exists only for packed experts; set "
        "memory.frozen_weight_quantization='nf4'."
    ),
    "quantize_experts_on_load": (
        "On-load expert quantization requires "
        "memory.frozen_weight_quantization='nf4'."
    ),
    "expert_device_cache_gib": (
        "Packed MAGI-2 experts are layer-resident state moved by the block "
        "residency subsystem, not per-expert operands streamed on demand, so "
        "they have no second byte-bounded device cache. Use "
        "training.blocks_to_swap to choose how much of the packed model stays "
        "device-resident."
    ),
    "device_residency_budget_gib": (
        "This family has no independent expert-residency owner to account "
        "against a shared ceiling."
    ),
    "moe_dispatch": "This family owns its routed dispatch.",
    "moe_dispatch_preprocess": "This family owns its routed dispatch.",
    "moe_gemm_backend_dw": (
        "Grouped execution keeps expert weights frozen, so the weight-gradient "
        "GEMM has no consumer."
    ),
}


def _reject_unconsumed_policy_fields(
    policy: Any, *, quantized_experts: bool = False
) -> None:
    """Raise for any policy field MAGI-2 neither consumes nor leaves at default.

    The consumed set is a function of the storage format: three fields acquire a
    consumer only once the routed experts are packed, and stay rejected while
    the family runs native BF16 experts.
    """
    consumed = (
        _QUANTIZED_CONSUMED_POLICY_FIELDS
        if quantized_experts
        else _CONSUMED_POLICY_FIELDS
    )
    defaults = MoEOptimizationPolicy()
    for field in fields(MoEOptimizationPolicy):
        if field.name in consumed:
            continue
        default = getattr(defaults, field.name)
        value = getattr(policy, field.name, default)
        normalizer = _POLICY_FIELD_NORMALIZERS.get(field.name)
        if normalizer is not None:
            value = normalizer(value)
        if value == default or value in _INERT_POLICY_VALUES.get(field.name, ()):
            continue
        note = _POLICY_FIELD_NOTES.get(field.name, "")
        raise Magi2GroupedMoEPolicyError(
            f"MAGI-2 Preview does not implement memory.{field.name}: got "
            f"{value!r}, this family requires its default {default!r}."
            + (f" {note}" if note else "")
        )


def resolve_magi2_moe_execution(
    policy: Any, *, quantized_experts: bool = False
) -> Magi2GroupedMoEPlan | None:
    """Validate a MoE optimization policy and return the grouped plan, if any.

    Returns ``None`` for the default kernel backends, which keep the vendored
    reference execution. Every policy field MAGI-2 cannot honor raises.

    Packed experts have no dense ``W_gate``/``W_up``/``W_down`` for the vendored
    reference loop to read, so ``auto`` resolves to grouped execution instead of
    leaving the reference path in place, and an explicit ``torch`` is rejected
    rather than silently reinterpreted.
    """
    kernel_backend = str(getattr(policy, "kernel_backend", "auto"))
    if kernel_backend not in _GROUPED_KERNEL_BACKENDS + _DEFAULT_KERNEL_BACKENDS:
        raise Magi2GroupedMoEPolicyError(
            "MAGI-2 Preview supports memory.moe_kernel_backend='auto', 'torch', "
            f"or 'grouped'; got '{kernel_backend}'."
        )
    _reject_unconsumed_policy_fields(policy, quantized_experts=quantized_experts)
    if quantized_experts and kernel_backend == "torch":
        raise Magi2GroupedMoEPolicyError(
            "memory.moe_kernel_backend='torch' selects the vendored per-expert "
            "reference loop, which reads the dense W_gate/W_up/W_down tensors "
            "that memory.frozen_weight_quantization='nf4' replaces with packed "
            "storage. Use 'auto' or 'grouped'."
        )
    if kernel_backend in _DEFAULT_KERNEL_BACKENDS and not quantized_experts:
        return None
    main = normalize_moe_gemm_backend(getattr(policy, "moe_gemm_backend", "auto"))
    forward = normalize_moe_gemm_role_backend(
        getattr(policy, "moe_gemm_backend_forward", "")
    )
    dx = normalize_moe_gemm_role_backend(getattr(policy, "moe_gemm_backend_dx", ""))
    return Magi2GroupedMoEPlan(
        forward_backend=forward or main,
        dx_backend=dx or main,
    )


def validate_grouped_moe_backend_support(
    plan: Magi2GroupedMoEPlan, *, device: torch.device | None = None
) -> None:
    """Reject a grouped plan this environment cannot run, before model execution.

    Weight residency places the transformer after the MoE policy is configured,
    so a non-CUDA ``device`` means the execution device is not yet knowable: only
    the torch build capability is decided here and the per-device architecture
    gate stays with :meth:`Magi2GroupedMoEBackend._resolve`. The operand stride
    precondition needs the expert tensors and is decided by
    :meth:`Magi2GroupedMoEBackend.inspect_expert_layout` at attach time.
    """
    probe_device = device if device is not None and device.type == "cuda" else None
    for role, requested in (
        ("forward", plan.forward_backend),
        ("dx", plan.dx_backend),
    ):
        if requested == "auto":
            continue
        if requested == "torch_grouped" and grouped_mm_op() is None:
            raise Magi2GroupedMoEPolicyError(
                "memory.moe_gemm_backend (or its "
                f"'{role}' role override) selected 'torch_grouped', which this "
                "torch build does not provide for MAGI-2 grouped MoE execution."
            )
        if probe_device is None:
            continue
        probe = probe_backend(requested, device=probe_device)
        if not probe.available:
            raise Magi2GroupedMoEPolicyError(
                f"memory.moe_gemm_backend (or its '{role}' role override) "
                f"selected '{requested}', which is unavailable for MAGI-2 "
                f"grouped MoE execution on {probe_device}: {probe.reason}."
            )


def attach_grouped_moe_backend(
    transformer: Any, backend: Magi2GroupedMoEBackend | None
) -> int:
    """Bind (or clear) the grouped execution seam on every vendored MoE module."""
    from mirai.vendors.magi2_preview.model.magi2_preview import CoreMultiHeadMoE

    attached = 0
    for module in transformer.modules():
        if isinstance(module, CoreMultiHeadMoE):
            if backend is not None:
                backend.inspect_expert_layout(module)
            module._mirai_moe_kernel_backend = backend
            attached += 1
    if backend is not None and attached == 0:
        raise Magi2GroupedMoEPolicyError(
            "memory.moe_kernel_backend='grouped' matched no MAGI-2 multi-head MoE "
            "layer in this model."
        )
    if backend is not None:
        backend.validate_explicit_alignment()
    return attached
