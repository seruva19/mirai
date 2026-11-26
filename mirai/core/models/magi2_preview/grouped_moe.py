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

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, fields
from typing import Any, Callable, ContextManager

import torch

from mirai.core.moe.runtime.gemm import (
    GROUPED_MM_ALIGNMENT_BYTES,
    BackendProbe,
    GroupedMMSegment,
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
    MAGI2_FUSED_GATE_UP,
    MAGI2_ROUTED_EXPERT_TENSOR_NAMES,
    Magi2Nf4ExpertStore,
    Magi2PackedExpertStore,
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
from mirai.core.moe.runtime.routed_gemm import normalize_routed_gemm_mode


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
    routed_gemm: str = "disabled"

    def __post_init__(self) -> None:
        for field, value in (
            ("forward_backend", self.forward_backend),
            ("dx_backend", self.dx_backend),
        ):
            normalized = normalize_moe_gemm_backend(value)
            if field == "forward_backend" and normalized == "scaled_mm_fp4":
                object.__setattr__(self, field, normalized)
                continue
            if normalized not in MAGI2_GROUPED_GEMM_BACKENDS:
                raise Magi2GroupedMoEPolicyError(
                    "MAGI-2 grouped MoE execution supports "
                    + ", ".join(MAGI2_GROUPED_GEMM_BACKENDS)
                    + f"; got '{normalized}' for the {field.split('_')[0]} role."
                )
            object.__setattr__(self, field, normalized)
        object.__setattr__(self, "routed_gemm", normalize_routed_gemm_mode(self.routed_gemm))


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
        grouped_mm_operand(w_gate.transpose(-2, -1), label="W_gate dX transposed weight view"),
        grouped_mm_operand(w_up.transpose(-2, -1), label="W_up dX transposed weight view"),
        grouped_mm_row_operand(
            label="grad_output dX rows for W_down",
            columns=int(w_down.shape[-1]),
            element_size=int(w_down.element_size()),
        ),
        grouped_mm_operand(w_down.transpose(-2, -1), label="W_down dX transposed weight view"),
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
            raise Magi2GroupedMoEPolicyError(_alignment_rejection(role, alignment_violations))
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


def _packed_group_segments(
    offsets: torch.Tensor, *, max_groups: int, routed: bool = False
) -> tuple[tuple[GroupedMMSegment, int], ...]:
    """Copy one row end and, for routed GEMM, one maximum per segment."""
    count = offsets.numel()
    if count == 0:
        return ()
    span = min(count, max(1, int(max_groups)))
    stops = list(range(span, count, span)) + [count]
    ends = torch.cat((offsets[span - 1 : count - 1 : span], offsets[-1:]))
    if routed:
        rows = offsets.diff(prepend=offsets.new_zeros(1))
        padding = (-count) % span
        maxima = torch.nn.functional.pad(rows, (0, padding)).view(-1, span).amax(dim=1)
        ends = torch.stack((ends, maxima), dim=1).flatten()
    metadata = ends.tolist()
    segments = []
    row_start = 0
    for index, stop in enumerate(stops):
        row_stop = metadata[2 * index if routed else index]
        maximum = metadata[2 * index + 1] if routed else 0
        segments.append(
            (GroupedMMSegment(index * span, stop, row_start, row_stop), maximum)
        )
        row_start = row_stop
    return tuple(segments)


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
            grad_x = _grouped_linear_dx(grad_output, weight, offsets, ctx.dx_backend)
        return grad_x, None, None, None, None


_SWIGLU7_ROW_CHUNK = 4096


def _memory_bounded_swiglu7(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    gate_rows = gate.reshape(-1, int(gate.shape[-1]))
    up_rows = up.reshape_as(gate_rows)
    output = torch.empty(gate.shape, dtype=output_dtype, device=gate.device)
    output_rows = output.reshape_as(gate_rows)
    for start in range(0, int(gate_rows.shape[0]), _SWIGLU7_ROW_CHUNK):
        stop = min(start + _SWIGLU7_ROW_CHUNK, int(gate_rows.shape[0]))
        # ``Tensor.float()`` aliases an FP32 input. Keep the saved autograd
        # operands immutable even when the reference path itself runs in FP32.
        gate_fp32 = gate_rows[start:stop].float().clamp(max=7.0)
        up_fp32 = up_rows[start:stop].float().clamp(min=-7.0, max=7.0)
        hidden = gate_fp32 * torch.sigmoid(1.702 * gate_fp32)
        hidden.mul_(up_fp32 + 1.0)
        output_rows[start:stop].copy_(hidden)
    return output


def _memory_bounded_swiglu7_backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    gate_rows = gate.reshape(-1, int(gate.shape[-1]))
    up_rows = up.reshape_as(gate_rows)
    grad_output_rows = grad_output.reshape_as(gate_rows)
    grad_gate_rows = grad_gate.reshape_as(gate_rows)
    grad_up_rows = grad_up.reshape_as(up_rows)
    for start in range(0, int(gate_rows.shape[0]), _SWIGLU7_ROW_CHUNK):
        stop = min(start + _SWIGLU7_ROW_CHUNK, int(gate_rows.shape[0]))
        raw_gate = gate_rows[start:stop].float()
        raw_up = up_rows[start:stop].float()
        gate_fp32 = raw_gate.clamp(max=7.0)
        up_fp32 = raw_up.clamp(min=-7.0, max=7.0)
        sigmoid = torch.sigmoid(1.702 * gate_fp32)
        activated_gate = gate_fp32 * sigmoid
        grad_hidden = grad_output_rows[start:stop].float()
        gate_derivative = sigmoid + (1.702 * gate_fp32 * sigmoid * (1.0 - sigmoid))
        gate_derivative.mul_(raw_gate <= 7.0)
        grad_gate_rows[start:stop].copy_(grad_hidden * gate_derivative * (up_fp32 + 1.0))
        up_derivative = (raw_up >= -7.0) & (raw_up <= 7.0)
        grad_up_rows[start:stop].copy_(grad_hidden * activated_gate * up_derivative)
    return grad_gate, grad_up


def _swiglu7_forward(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    if backend == "triton":
        from mirai.core.models.magi2_preview.triton_swiglu import triton_swiglu7

        return triton_swiglu7(gate.contiguous(), up.contiguous(), output_dtype=output_dtype)
    return _memory_bounded_swiglu7(gate, up, output_dtype=output_dtype)


def _swiglu7_backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend == "triton":
        from mirai.core.models.magi2_preview.triton_swiglu import (
            triton_swiglu7_backward,
        )

        return triton_swiglu7_backward(gate.contiguous(), up.contiguous(), grad_output)
    return _memory_bounded_swiglu7_backward(gate, up, grad_output)


class _MemoryBoundedSwiGlu7(torch.autograd.Function):
    """Evaluate MAGI-2's FP32 SwiGLU7 ladder without full-size FP32 copies."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        gate: torch.Tensor,
        up: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if gate.shape != up.shape:
            raise ValueError("MAGI-2 SwiGLU7 gate/up shapes must match.")
        ctx.save_for_backward(gate, up)
        return _memory_bounded_swiglu7(gate, up, output_dtype=output_dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = _memory_bounded_swiglu7_backward(gate, up, grad_output)
        if not ctx.needs_input_grad[0]:
            grad_gate = None
        if not ctx.needs_input_grad[1]:
            grad_up = None
        return grad_gate, grad_up, None


class _ConfiguredSwiGlu7(torch.autograd.Function):
    """Dispatch the exact SwiGLU7 contract through the configured backend."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        gate: torch.Tensor,
        up: torch.Tensor,
        output_dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        if gate.shape != up.shape:
            raise ValueError("MAGI-2 SwiGLU7 gate/up shapes must match.")
        ctx.save_for_backward(gate, up)
        ctx.backend = str(backend)
        return _swiglu7_forward(
            gate,
            up,
            output_dtype=output_dtype,
            backend=ctx.backend,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = _swiglu7_backward(
            gate,
            up,
            grad_output,
            backend=ctx.backend,
        )
        if not ctx.needs_input_grad[0]:
            grad_gate = None
        if not ctx.needs_input_grad[1]:
            grad_up = None
        return grad_gate, grad_up, None, None


class Magi2GroupedMoEBackend:
    """Grouped execution seam attached to vendored ``CoreMultiHeadMoE`` layers."""

    name = "grouped"

    def __init__(
        self,
        plan: Magi2GroupedMoEPlan,
        *,
        token_chunk_size: int = 0,
        activation_backend: str = "torch",
    ) -> None:
        if int(token_chunk_size) < 0:
            raise ValueError("MAGI-2 MoE token chunk size must be >= 0.")
        self.plan = plan
        self.token_chunk_size = int(token_chunk_size)
        self.activation_backend = str(activation_backend)
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
                raise Magi2GroupedMoEPolicyError(_alignment_rejection(role, violations))

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

    def _project_gate_up(self, module, x_sorted, offsets, forward_backend, dx_backend):
        return (
            self.project(module, "W_gate", x_sorted, offsets, forward_backend, dx_backend),
            self.project(module, "W_up", x_sorted, offsets, forward_backend, dx_backend),
        )

    def routed_project(
        self,
        module: Any,
        key: str,
        activation: torch.Tensor,
        offsets: torch.Tensor,
        *,
        gather_rows: torch.Tensor | None = None,
        scatter_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Provider-owned entry to the generic routed Triton projection."""
        from mirai.core.moe.runtime.routed_gemm_triton import routed_projection

        return routed_projection(
            activation.contiguous(),
            getattr(module, key),
            offsets,
            gather_rows=gather_rows,
            scatter_rows=scatter_rows,
        )

    @staticmethod
    def _physical_expert_layout(module, flat_experts, groups):
        aliases = getattr(module, "_mirai_expert_alias_indices", None)
        if aliases is None:
            return flat_experts, groups
        return aliases.to(flat_experts.device).index_select(0, flat_experts.long()), module._mirai_physical_expert_count

    def plan_branches(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> "Magi2ExpertBranchPlan":
        """Routed layout of one layer visit, with no expert matmul yet run.

        This is the branch decomposition
        :mod:`mirai.core.moe.runtime.expert_feature_cache` consumes. The
        canonical slot order is ``(head, token, k)`` flattened, which is the
        order ``topk_indices`` already carries, so a slot keeps its identity
        across visits of the same layer even when its routed expert changes.
        """
        forward_backend, dx_backend = self._resolve(x_heads.device, module)
        device = x_heads.device
        tokens = int(x_heads.shape[0])
        num_heads = int(x_heads.shape[1])
        num_experts = int(module.num_experts)
        head_axis = torch.arange(num_heads, device=device).view(num_heads, 1, 1)
        token_axis = torch.arange(tokens, device=device).view(1, tokens, 1)
        flat_experts, groups = self._physical_expert_layout(
            module, (topk_indices + head_axis * num_experts).reshape(-1), num_heads * num_experts,
        )
        return Magi2ExpertBranchPlan(
            backend=self,
            module=module,
            x_heads=x_heads,
            topk_probs=topk_probs,
            flat_experts=flat_experts,
            rows=(token_axis * num_heads + head_axis).expand_as(topk_indices).reshape(-1),
            groups=groups,
            forward_backend=forward_backend,
            dx_backend=dx_backend,
        )

    def branch_features(
        self,
        module: Any,
        x_heads: torch.Tensor,
        sorted_rows: torch.Tensor,
        offsets: torch.Tensor,
        forward_backend: str,
        dx_backend: str,
    ) -> torch.Tensor:
        """Pre-combine expert output of every row of a sorted routed layout."""
        tokens = int(x_heads.shape[0])
        num_heads = int(x_heads.shape[1])
        d_head = int(x_heads.shape[2])
        x_rows = x_heads.reshape(tokens * num_heads, d_head)
        x_sorted = x_rows.index_select(0, sorted_rows)
        gate, up = self._project_gate_up(module, x_sorted, offsets, forward_backend, dx_backend)
        hidden = _ConfiguredSwiGlu7.apply(
            gate,
            up,
            self.expert_weight_dtype(module, "W_down"),
            self.activation_backend,
        )
        return self.project(
            module,
            "W_down",
            hidden,
            offsets,
            forward_backend,
            dx_backend,
        )

    def execute(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        tokens = int(x_heads.shape[0])
        chunk_size = self.token_chunk_size
        if chunk_size <= 0 or tokens <= chunk_size:
            return self._execute_routed_chunk(module, x_heads, topk_probs, topk_indices)
        outputs: list[torch.Tensor] = []
        for start in range(0, tokens, chunk_size):
            stop = min(start + chunk_size, tokens)
            outputs.append(
                self._execute_routed_chunk(
                    module,
                    x_heads[start:stop],
                    topk_probs[:, start:stop],
                    topk_indices[:, start:stop],
                )
            )
        return torch.cat(outputs, dim=0)

    def _execute_routed_chunk(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Execute already-computed routes for one contiguous token range."""

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
        flat_experts, groups = self._physical_expert_layout(module, flat_experts, groups)
        rows = (token_axis * num_heads + head_axis).expand_as(topk_indices).reshape(-1)
        order = flat_experts.argsort(stable=True)
        sorted_rows = rows[order]
        sorted_probs = topk_probs.reshape(-1)[order]
        counts = torch.bincount(flat_experts, minlength=groups)
        offsets = counts.cumsum(0).to(torch.int32)

        if self.plan.routed_gemm != "disabled" and hasattr(module, "W_gate"):
            from mirai.core.moe.runtime.routed_gemm import (
                RoutedFusionSpec,
                RoutedGroupLayout,
                RoutedOutputMode,
                routed_gemm_verdict,
            )

            layout = RoutedGroupLayout(
                boundaries=offsets,
                assignment_rows=order,
                token_count=tokens * num_heads,
                top_k=int(topk_indices.shape[-1]),
                group_count=groups,
                provider_mapping=("head", "expert"),
            )
            x_rows = x_heads.reshape(tokens * num_heads, d_head)
            verdicts = [
                routed_gemm_verdict(
                    self.plan.routed_gemm,
                    x_rows,
                    getattr(module, key),
                    RoutedFusionSpec(gather_tokens=True),
                    training=torch.is_grad_enabled(),
                    resident=True,
                    quantized=False,
                    layout=layout,
                )
                for key in ("W_gate", "W_up")
            ]
            verdicts.append(
                routed_gemm_verdict(
                    self.plan.routed_gemm,
                    x_rows.new_empty((int(order.numel()), int(module.W_down.shape[1]))),
                    module.W_down,
                    RoutedFusionSpec(output=RoutedOutputMode.WEIGHTED_TOKEN_REDUCTION),
                    training=torch.is_grad_enabled(),
                    resident=True,
                    quantized=False,
                    layout=layout,
                )
            )
            verdict = next(
                (item for item in verdicts if item.selected != "triton" or not item.supported),
                verdicts[-1],
            )
            if verdict.selected == "triton" and verdict.supported:
                gate = self.routed_project(
                    module, "W_gate", x_rows, offsets, gather_rows=sorted_rows
                )
                up = self.routed_project(
                    module, "W_up", x_rows, offsets, gather_rows=sorted_rows
                )
                hidden = _ConfiguredSwiGlu7.apply(
                    gate,
                    up,
                    self.expert_weight_dtype(module, "W_down"),
                    self.activation_backend,
                )
                # ``order`` maps grouped rows to canonical assignment slots.
                from mirai.core.moe.runtime.routed_gemm_triton import (
                    routed_weighted_projection,
                )

                output = routed_weighted_projection(
                    hidden,
                    getattr(module, "W_down"),
                    offsets,
                    grouped_to_assignment=order,
                    assignment_to_token=rows,
                    coefficients=topk_probs.reshape(-1),
                    token_rows=tokens * num_heads,
                )
                return output.view(tokens, num_heads, d_head)
            if self.plan.routed_gemm == "triton":
                raise Magi2GroupedMoEPolicyError(
                    "memory.moe_routed_gemm='triton' " + verdict.reason + "."
                )

        expert_output = self._grouped_branch_features(
            module, x_heads, sorted_rows, flat_experts, order, offsets, forward_backend, dx_backend,
        )
        weighted = expert_output * sorted_probs.to(expert_output.dtype)[:, None]
        output = torch.zeros(
            tokens * num_heads, d_head, device=device, dtype=x_heads.dtype
        ).index_add(0, sorted_rows, weighted.to(x_heads.dtype))
        return output.view(tokens, num_heads, d_head)


    def _grouped_branch_features(
        self, module, x_heads, sorted_rows, experts, order, offsets, forward_backend, dx_backend,
    ):
        if getattr(module, "_mirai_expert_alias_indices", None) is None:
            return self.branch_features(module, x_heads, sorted_rows, offsets, forward_backend, dx_backend)
        # Stable grouping keeps token order inside each physical expert.
        row_count = max(1, x_heads.shape[0] * x_heads.shape[1])
        keys = experts[order] * row_count + sorted_rows
        unique, inverse = torch.unique_consecutive(keys, return_inverse=True)
        compact_rows = unique.remainder(row_count)
        compact_experts = torch.div(unique, row_count, rounding_mode="floor")
        compact_offsets = torch.bincount(compact_experts, minlength=offsets.numel()).cumsum(0).to(torch.int32)
        return self.branch_features(
            module, x_heads, compact_rows, compact_offsets, forward_backend, dx_backend,
        ).index_select(0, inverse)


@dataclass
class Magi2ExpertBranchPlan:
    """Branch decomposition of one MAGI-2 MoE layer visit.

    Implements the family-agnostic ``ExpertBranchPlan`` contract owned by
    :mod:`mirai.core.moe.runtime.expert_feature_cache`. ``compute_branch_features``
    followed by ``combine_branch_features`` reproduces
    :meth:`Magi2GroupedMoEBackend.execute` exactly: the canonical buffer is
    filled and read back through the same sort permutation, so the operand
    values, the operand order, and the accumulation order of the combine are the
    ones the uncached path uses.
    """

    backend: Magi2GroupedMoEBackend
    module: Any
    x_heads: torch.Tensor
    topk_probs: torch.Tensor
    flat_experts: torch.Tensor
    rows: torch.Tensor
    groups: int
    forward_backend: str
    dx_backend: str

    @property
    def expert_ids(self) -> torch.Tensor:
        return self.flat_experts

    def _grouped_offsets(self, experts: torch.Tensor) -> torch.Tensor:
        return torch.bincount(experts, minlength=self.groups).cumsum(0).to(torch.int32)

    def compute_branch_features(self, slot_mask: torch.Tensor | None) -> torch.Tensor:
        """Expert output of the selected routed slots, canonical order.

        Rows outside ``slot_mask`` are zero. A masked subset is sorted and
        grouped on its own, which the jagged-offset contract makes exact: a
        routed slot's expert matmul reads only that slot's row, so recomputing a
        subset yields the values a full recompute would have produced for it.
        """
        if slot_mask is None:
            selected = None
            experts = self.flat_experts
            rows = self.rows
        else:
            selected = slot_mask.reshape(-1).nonzero(as_tuple=True)[0]
            experts = self.flat_experts.index_select(0, selected)
            rows = self.rows.index_select(0, selected)
        order = experts.argsort(stable=True)
        computed = self.backend._grouped_branch_features(
            self.module,
            self.x_heads,
            rows.index_select(0, order),
            experts,
            order,
            self._grouped_offsets(experts),
            self.forward_backend,
            self.dx_backend,
        )
        target = order if selected is None else selected.index_select(0, order)
        canonical = computed.new_zeros((int(self.flat_experts.numel()), int(computed.shape[-1])))
        return canonical.index_copy(0, target, computed)

    def combine_branch_features(self, features: torch.Tensor) -> torch.Tensor:
        """Routing-weighted combine of canonical branch features."""
        order = self.flat_experts.argsort(stable=True)
        sorted_rows = self.rows.index_select(0, order)
        sorted_probs = self.topk_probs.reshape(-1).index_select(0, order)
        expert_output = features.index_select(0, order)
        weighted = expert_output * sorted_probs.to(expert_output.dtype)[:, None]
        tokens = int(self.x_heads.shape[0])
        num_heads = int(self.x_heads.shape[1])
        d_head = int(self.x_heads.shape[2])
        output = torch.zeros(
            tokens * num_heads,
            d_head,
            device=self.x_heads.device,
            dtype=self.x_heads.dtype,
        ).index_add(0, sorted_rows, weighted.to(self.x_heads.dtype))
        return output.view(tokens, num_heads, d_head)


def run_segmented_grouped_linear(
    x_sorted: torch.Tensor,
    offsets: torch.Tensor,
    *,
    boundaries: list[int] | None = None,
    materialize: Callable[[int, int], torch.Tensor],
    backend: str,
    max_groups: int,
    columns: int,
    transposed: bool = False,
    segments: tuple[GroupedMMSegment, ...] | None = None,
    prefetch: Callable[[int, int], None] | None = None,
    borrow: Callable[[int, int], ContextManager[torch.Tensor]] | None = None,
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
    if segments is None:
        segments = (
            grouped_mm_segments(boundaries, max_groups=max(1, int(max_groups)))
            if boundaries is not None
            else tuple(segment for segment, _ in _packed_group_segments(offsets, max_groups=max_groups))
        )
    if prefetch is not None:
        segments = tuple(segment for segment in segments if segment.row_count)
    for index, segment in enumerate(segments):
        if segment.row_count == 0:
            continue
        start, stop = segment.group_start, segment.group_stop
        lease = nullcontext(materialize(start, stop)) if borrow is None else borrow(start, stop)
        with lease as weight:
            if prefetch is not None and index + 1 < len(segments):
                next_segment = segments[index + 1]
                prefetch(next_segment.group_start, next_segment.group_stop)
            local_offsets = (offsets[start:stop] - segment.row_start).contiguous()
            rows = x_sorted[segment.row_start : segment.row_stop]
            outputs.append(
                _grouped_linear_dx(rows, weight, local_offsets, backend)
                if transposed
                else _grouped_linear(rows, weight, local_offsets, backend)
            )
        del weight, lease
    if not outputs:
        return x_sorted.new_empty((0, int(columns)))
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


def run_segmented_grouped_expert_mlp(
    x_sorted: torch.Tensor,
    offsets: torch.Tensor,
    *,
    boundaries: list[int] | None = None,
    materialize: Callable[[str, int, int], torch.Tensor],
    backend: str,
    max_groups: int,
    output_columns: int,
    activation_backend: str = "torch",
    fused_gate_up: bool = False,
    segments: tuple[GroupedMMSegment, ...] | None = None,
    prefetch: Callable[[str, int, int], None] | None = None,
    borrow: Callable[[str, int, int], ContextManager[torch.Tensor]] | None = None,
) -> torch.Tensor:
    """Evaluate one frozen SwiGLU7 expert segment end-to-end.

    Segmenting each projection separately retains every gate/up result until a
    full-row concatenate. Running gate, up, activation, and down for the same
    group range before advancing keeps only the final narrow expert output while
    materializing every packed weight range exactly once.
    """

    def weight_lease(key, start, stop):
        return nullcontext(materialize(key, start, stop)) if borrow is None else borrow(key, start, stop)

    outputs: list[torch.Tensor] = []
    if segments is None:
        segments = (
            grouped_mm_segments(boundaries, max_groups=max(1, int(max_groups)))
            if boundaries is not None
            else tuple(segment for segment, _ in _packed_group_segments(offsets, max_groups=max_groups))
        )
    if prefetch is not None:
        segments = tuple(segment for segment in segments if segment.row_count)
    for index, segment in enumerate(segments):
        if segment.row_count == 0:
            continue
        start, stop = segment.group_start, segment.group_stop
        local_offsets = (offsets[start:stop] - segment.row_start).contiguous()
        rows = x_sorted[segment.row_start : segment.row_stop]

        if fused_gate_up:
            with weight_lease(MAGI2_FUSED_GATE_UP, start, stop) as weight:
                if prefetch is not None:
                    prefetch("W_down", start, stop)
                gate, up = _grouped_linear(rows, weight, local_offsets, backend).chunk(2, dim=-1)
            del weight
        else:
            with weight_lease("W_gate", start, stop) as weight:
                if prefetch is not None:
                    prefetch("W_up", start, stop)
                gate = _grouped_linear(rows, weight, local_offsets, backend)
            del weight
            with weight_lease("W_up", start, stop) as weight:
                if prefetch is not None:
                    prefetch("W_down", start, stop)
                up = _grouped_linear(rows, weight, local_offsets, backend)
            del weight

        hidden = _swiglu7_forward(
            gate,
            up,
            output_dtype=rows.dtype,
            backend=activation_backend,
        )
        del gate, up

        with weight_lease("W_down", start, stop) as weight:
            if prefetch is not None and index + 1 < len(segments):
                next_segment = segments[index + 1]
                prefetch(
                    MAGI2_FUSED_GATE_UP if fused_gate_up else "W_gate",
                    next_segment.group_start, next_segment.group_stop,
                )
            outputs.append(_grouped_linear(hidden.to(weight.dtype), weight, local_offsets, backend))
        del hidden, weight
    if not outputs:
        return x_sorted.new_empty((0, int(output_columns)))
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


@dataclass(frozen=True)
class _ActiveExpertGroups:
    store: Magi2Nf4ExpertStore
    group_ids: torch.Tensor

    @property
    def has_fused_gate_up(self) -> bool:
        return self.store.has_fused_gate_up

    def segment_group_span(self) -> int:
        return self.store.segment_group_span()

    def expert_weight_shape(self, key: str) -> tuple[int, int, int]:
        _, rows, columns = self.store.expert_weight_shape(key)
        return self.group_ids.numel(), rows, columns

    def materialize_segment(
        self, key: str, start: int, stop: int, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return self.store.materialize_segment(
            key, start, stop, dtype=dtype, device=device, group_ids=self.group_ids
        )

    def prefetch_segment(self, key, start, stop, *, device, dtype) -> None:
        self.store.prefetch_segment(key, start, stop, device=device, dtype=dtype, group_ids=self.group_ids)

    def borrow_segment(self, key, start, stop, *, device, dtype):
        borrow = getattr(self.store, "borrow_segment", None)
        if borrow is None:
            return nullcontext(self.materialize_segment(key, start, stop, device=device, dtype=dtype))
        return borrow(key, start, stop, device=device, dtype=dtype, group_ids=self.group_ids)

    @property
    def packed_prefetch_enabled(self) -> bool:
        return self.store.packed_prefetch_enabled


class _CpuExpertRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, offsets, store, output):
        ctx.save_for_backward(rows, offsets)
        ctx.store = store
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rows, offsets = ctx.saved_tensors
        with torch.enable_grad():
            rows = rows.detach().requires_grad_(True)
            output = _SegmentedQuantizedExpertMlp.apply(rows, offsets, ctx.store, "bmm", "bmm", "torch")
            (grad_rows,) = torch.autograd.grad(output, rows, grad_output)
        return grad_rows, None, None, None


def _hybrid_expert_features(
    x_sorted, offsets, store, *, cpu_max_tokens, forward_backend, dx_backend, activation_backend,
):
    boundaries = [0, *_grouped_boundaries(offsets)]
    cold, hot = [], []
    for group, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        if start < stop:
            selected = cold if stop - start <= cpu_max_tokens else hot
            selected.append((group, start, stop))
    if not cold:
        active_store, active_offsets = _packed_expert_layout(store, offsets)
        return _SegmentedQuantizedExpertMlp.apply(
            x_sorted, active_offsets, active_store, forward_backend, dx_backend, activation_backend,
        )
    def subset(groups):
        group_ids = torch.tensor([group for group, _, _ in groups], dtype=torch.long)
        packed = _ActiveExpertGroups(store, group_ids)
        positions = torch.cat([torch.arange(start, stop) for _, start, stop in groups]).to(x_sorted.device)
        counts = torch.tensor([stop - start for _, start, stop in groups], dtype=offsets.dtype)
        return positions, counts.cumsum(0, dtype=offsets.dtype), packed

    cold_positions, cold_offsets, cold_store = subset(cold)
    cold_rows = x_sorted.index_select(0, cold_positions)
    copy_ready = None
    if cold_rows.device.type == "cuda":
        host_rows = torch.empty_like(cold_rows, device="cpu", pin_memory=True)
        host_rows.copy_(cold_rows, non_blocking=True)
        copy_ready = torch.cuda.Event()
        copy_ready.record(torch.cuda.current_stream(cold_rows.device))
        cold_rows = host_rows

    def execute_cold():
        if copy_ready is not None:
            copy_ready.synchronize()
        with torch.no_grad():
            return _SegmentedQuantizedExpertMlp.apply(cold_rows, cold_offsets, cold_store, "bmm", "bmm", "torch")

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mirai-magi2-cold") as executor:
        pending = executor.submit(execute_cold)
        output = torch.empty_like(x_sorted)
        if hot:
            positions, local_offsets, active_store = subset(hot)
            hot_output = _SegmentedQuantizedExpertMlp.apply(
                x_sorted.index_select(0, positions), local_offsets.to(offsets.device), active_store,
                forward_backend, dx_backend, activation_backend,
            )
            output.index_copy_(0, positions, hot_output)
        cold_output = pending.result()
    if cold_rows.requires_grad:
        cold_output = _CpuExpertRecompute.apply(cold_rows, cold_offsets, cold_store, cold_output)
    return output.index_copy_(0, cold_positions, cold_output.to(device=output.device))


def _packed_expert_layout(
    store: Magi2Nf4ExpertStore, offsets: torch.Tensor
) -> tuple[Magi2Nf4ExpertStore | _ActiveExpertGroups, torch.Tensor]:
    if store.expert_weight_access != "active_group_dequant":
        return store, offsets
    counts = offsets.diff(prepend=offsets.new_zeros(1))
    group_ids = counts.nonzero(as_tuple=True)[0]
    return _ActiveExpertGroups(store, group_ids), offsets.index_select(0, group_ids)


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
        store: Magi2Nf4ExpertStore | _ActiveExpertGroups,
        key: str,
        forward_backend: str,
        dx_backend: str,
    ) -> torch.Tensor:
        _groups, rows, cols = store.expert_weight_shape(key)
        span = store.segment_group_span()
        segments = tuple(segment for segment, _ in _packed_group_segments(offsets, max_groups=span))
        if forward_backend == "scaled_mm_fp4":
            output = _native_fp4_grouped_projection(x_sorted, offsets, store, key, cols)
        else:
            output = run_segmented_grouped_linear(
                x_sorted,
                offsets,
                segments=segments,
                materialize=lambda start, stop: store.materialize_segment(
                    key, start, stop, dtype=x_sorted.dtype, device=x_sorted.device
                ),
                borrow=(
                    (lambda start, stop: store.borrow_segment(key, start, stop, dtype=x_sorted.dtype, device=x_sorted.device))
                    if hasattr(store, "borrow_segment") else None
                ),
                prefetch=(
                    (lambda start, stop: store.prefetch_segment(key, start, stop, device=x_sorted.device, dtype=x_sorted.dtype))
                    if getattr(store, "packed_prefetch_enabled", False) else None
                ),
                backend=forward_backend,
                max_groups=span,
                columns=int(cols),
            )
        ctx.save_for_backward(offsets)
        ctx.store = store
        ctx.key = str(key)
        ctx.dx_backend = dx_backend
        ctx.segments = segments
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
                segments=ctx.segments,
                materialize=lambda start, stop: store.materialize_segment(
                    key, start, stop, dtype=grad_output.dtype, device=grad_output.device
                ),
                borrow=(
                    (lambda start, stop: store.borrow_segment(key, start, stop, dtype=grad_output.dtype, device=grad_output.device))
                    if hasattr(store, "borrow_segment") else None
                ),
                prefetch=(
                    (lambda start, stop: store.prefetch_segment(key, start, stop, device=grad_output.device, dtype=grad_output.dtype))
                    if getattr(store, "packed_prefetch_enabled", False) else None
                ),
                backend=ctx.dx_backend,
                max_groups=ctx.segment_span,
                columns=ctx.weight_rows,
                transposed=True,
            )
        return grad_x, None, None, None, None, None


def _native_fp4_grouped_projection(x_sorted, offsets, store, key, columns):
    physical_store = store.store if isinstance(store, _ActiveExpertGroups) else store
    if not isinstance(physical_store, Magi2PackedExpertStore) or not physical_store.has_native_fp4_layout:
        raise Magi2GroupedMoEPolicyError("Native MAGI-2 FP4 GEMM requires uniform contraction-major FP4 weights.")
    group_ids = store.group_ids.tolist() if isinstance(store, _ActiveExpertGroups) else range(offsets.numel())
    boundaries = [0, *_grouped_boundaries(offsets)]
    outputs = [
        physical_store.native_fp4_projection(key, group, x_sorted[start:stop])
        for group, start, stop in zip(group_ids, boundaries[:-1], boundaries[1:], strict=True)
        if start < stop
    ]
    return torch.cat(outputs, dim=0) if outputs else x_sorted.new_empty((0, columns))


def _expert_projection_prefetch(store, x):
    if not getattr(store, "packed_prefetch_enabled", False):
        return None
    return lambda key, start, stop: store.prefetch_segment(key, start, stop, device=x.device, dtype=x.dtype)


class _SegmentedQuantizedExpertMlp(torch.autograd.Function):
    """Rematerialize one NF4 expert segment at a time in forward and backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x_sorted: torch.Tensor,
        offsets: torch.Tensor,
        store: Magi2Nf4ExpertStore | _ActiveExpertGroups,
        forward_backend: str,
        dx_backend: str,
        activation_backend: str,
    ) -> torch.Tensor:
        span = store.segment_group_span()
        segments = tuple(segment for segment, _ in _packed_group_segments(offsets, max_groups=span))
        output = run_segmented_grouped_expert_mlp(
            x_sorted,
            offsets,
            segments=segments,
            materialize=lambda key, start, stop: store.materialize_segment(
                key, start, stop, dtype=x_sorted.dtype, device=x_sorted.device
            ),
            backend=forward_backend,
            max_groups=span,
            output_columns=int(x_sorted.shape[-1]),
            activation_backend=str(activation_backend),
            fused_gate_up=store.has_fused_gate_up,
            prefetch=_expert_projection_prefetch(store, x_sorted),
            borrow=(
                (lambda key, start, stop: store.borrow_segment(key, start, stop, dtype=x_sorted.dtype, device=x_sorted.device))
                if hasattr(store, "borrow_segment") else None
            ),
        )
        ctx.save_for_backward(x_sorted, offsets)
        ctx.store = store
        ctx.forward_backend = forward_backend
        ctx.dx_backend = dx_backend
        ctx.activation_backend = str(activation_backend)
        ctx.segments = segments
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        x_sorted, offsets = ctx.saved_tensors
        grad_segments: list[torch.Tensor] = []
        store = ctx.store
        prefetch = _expert_projection_prefetch(store, x_sorted)
        segments = tuple(segment for segment in ctx.segments if segment.row_count)
        borrow = getattr(store, "borrow_segment", None)

        def lease(key, start, stop):
            kwargs = {"dtype": x_sorted.dtype, "device": x_sorted.device}
            if borrow is None:
                return nullcontext(store.materialize_segment(key, start, stop, **kwargs))
            return borrow(key, start, stop, **kwargs)

        for index, segment in enumerate(segments):
            if segment.row_count == 0:
                continue
            local_offsets = (
                offsets[segment.group_start : segment.group_stop] - segment.row_start
            ).contiguous()
            rows = x_sorted[segment.row_start : segment.row_stop]
            grad_rows = grad_output[segment.row_start : segment.row_stop]

            start, stop = segment.group_start, segment.group_stop

            with ExitStack() as gate_lease, ExitStack() as up_lease:
                if store.has_fused_gate_up:
                    w_gate_up = gate_lease.enter_context(lease(MAGI2_FUSED_GATE_UP, start, stop))
                    if prefetch is not None:
                        prefetch("W_down", start, stop)
                    gate, up = _grouped_linear(rows, w_gate_up, local_offsets, ctx.forward_backend).chunk(2, dim=-1)
                else:
                    w_gate = gate_lease.enter_context(lease("W_gate", start, stop))
                    if prefetch is not None:
                        prefetch("W_up", start, stop)
                    gate = _grouped_linear(rows, w_gate, local_offsets, ctx.forward_backend)
                    w_up = up_lease.enter_context(lease("W_up", start, stop))
                    if prefetch is not None:
                        prefetch("W_down", start, stop)
                    up = _grouped_linear(rows, w_up, local_offsets, ctx.forward_backend)
                with lease("W_down", start, stop) as w_down:
                    if prefetch is not None and index + 1 < len(segments):
                        next_segment = segments[index + 1]
                        prefetch(
                            MAGI2_FUSED_GATE_UP if store.has_fused_gate_up else "W_gate",
                            next_segment.group_start, next_segment.group_stop,
                        )
                    grad_hidden = _grouped_linear_dx(grad_rows, w_down, local_offsets, ctx.dx_backend)
                del w_down
                grad_gate, grad_up = _swiglu7_backward(gate, up, grad_hidden, backend=ctx.activation_backend)
                if store.has_fused_gate_up:
                    grad_x = _grouped_linear_dx(
                        torch.cat((grad_gate, grad_up), dim=-1), w_gate_up, local_offsets, ctx.dx_backend
                    )
                    del w_gate_up
                else:
                    grad_x = _grouped_linear_dx(grad_gate, w_gate, local_offsets, ctx.dx_backend)
                    del w_gate
                    gate_lease.close()
                    grad_x.add_(_grouped_linear_dx(grad_up, w_up, local_offsets, ctx.dx_backend))
                    del w_up
            grad_segments.append(grad_x)
            del gate, up, grad_hidden, grad_gate, grad_up
        if not grad_segments:
            grad_input = torch.empty_like(x_sorted)
        elif len(grad_segments) == 1:
            grad_input = grad_segments[0]
        else:
            grad_input = torch.cat(grad_segments, dim=0)
        return grad_input, None, None, None, None, None


def _segment_routed_projection(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    gather_rows: torch.Tensor | None = None,
    max_group_rows: int | None = None,
) -> torch.Tensor:
    """Run one transient dense segment through the shared routed BF16 kernel."""
    from mirai.core.moe.runtime.routed_gemm_triton import triton_routed_grouped_mm

    return triton_routed_grouped_mm(
        activation.contiguous(),
        weight,
        boundaries,
        gather_rows=gather_rows,
        max_group_rows=max_group_rows,
    )


def _prefetch_next_routed_segment(prefetch, segments, index, fused_gate_up):
    if prefetch is not None and index + 1 < len(segments):
        next_segment, _ = segments[index + 1]
        prefetch(
            MAGI2_FUSED_GATE_UP if fused_gate_up else "W_gate",
            next_segment.group_start, next_segment.group_stop,
        )


class _SegmentedRoutedQuantizedExpertMlp(torch.autograd.Function):
    """Routed BF16 projection over one transient NF4 segment at a time."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x_rows: torch.Tensor,
        offsets: torch.Tensor,
        sorted_rows: torch.Tensor,
        store: Magi2Nf4ExpertStore | _ActiveExpertGroups,
        activation_backend: str,
    ) -> torch.Tensor:
        segments = _packed_group_segments(offsets, max_groups=store.segment_group_span(), routed=True)
        outputs: list[torch.Tensor] = []
        prefetch = _expert_projection_prefetch(store, x_rows)
        segments = tuple((segment, rows) for segment, rows in segments if segment.row_count)
        for index, (segment, max_group_rows) in enumerate(segments):
            local_offsets = (
                offsets[segment.group_start : segment.group_stop] - segment.row_start
            ).contiguous()
            local_rows = sorted_rows[segment.row_start : segment.row_stop].contiguous()
            if store.has_fused_gate_up:
                w_gate_up = store.materialize_segment(
                    MAGI2_FUSED_GATE_UP, segment.group_start, segment.group_stop,
                    dtype=x_rows.dtype, device=x_rows.device,
                )
                if prefetch is not None:
                    prefetch("W_down", segment.group_start, segment.group_stop)
                gate, up = _segment_routed_projection(
                    x_rows, w_gate_up, local_offsets, gather_rows=local_rows,
                    max_group_rows=max_group_rows,
                ).chunk(2, dim=-1)
                del w_gate_up
            else:
                w_gate = store.materialize_segment(
                    "W_gate", segment.group_start, segment.group_stop,
                    dtype=x_rows.dtype, device=x_rows.device,
                )
                if prefetch is not None:
                    prefetch("W_up", segment.group_start, segment.group_stop)
                gate = _segment_routed_projection(
                    x_rows, w_gate, local_offsets, gather_rows=local_rows,
                    max_group_rows=max_group_rows,
                )
                del w_gate
                w_up = store.materialize_segment(
                    "W_up", segment.group_start, segment.group_stop,
                    dtype=x_rows.dtype, device=x_rows.device,
                )
                if prefetch is not None:
                    prefetch("W_down", segment.group_start, segment.group_stop)
                up = _segment_routed_projection(
                    x_rows, w_up, local_offsets, gather_rows=local_rows,
                    max_group_rows=max_group_rows,
                )
                del w_up
            hidden = _swiglu7_forward(
                gate, up, output_dtype=x_rows.dtype, backend=str(activation_backend)
            )
            del gate, up
            w_down = store.materialize_segment(
                "W_down", segment.group_start, segment.group_stop,
                dtype=x_rows.dtype, device=x_rows.device,
            )
            _prefetch_next_routed_segment(prefetch, segments, index, store.has_fused_gate_up)
            outputs.append(_segment_routed_projection(
                hidden, w_down, local_offsets, max_group_rows=max_group_rows
            ))
            del hidden, w_down
        if not outputs:
            output = x_rows.new_empty((0, int(x_rows.shape[-1])))
        elif len(outputs) == 1:
            output = outputs[0]
        else:
            output = torch.cat(outputs, dim=0)
        ctx.save_for_backward(x_rows, offsets, sorted_rows)
        ctx.store = store
        ctx.activation_backend = str(activation_backend)
        ctx.segments = segments
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        x_rows, offsets, sorted_rows = ctx.saved_tensors
        grad_input = torch.zeros_like(x_rows)
        store = ctx.store
        prefetch = _expert_projection_prefetch(store, x_rows)
        segments = ctx.segments
        for index, (segment, max_group_rows) in enumerate(segments):
            local_offsets = (
                offsets[segment.group_start : segment.group_stop] - segment.row_start
            ).contiguous()
            local_rows = sorted_rows[segment.row_start : segment.row_stop].contiguous()
            grad_rows = grad_output[segment.row_start : segment.row_stop].contiguous()
            if store.has_fused_gate_up:
                w_gate_up = store.materialize_segment(
                    MAGI2_FUSED_GATE_UP, segment.group_start, segment.group_stop,
                    dtype=x_rows.dtype, device=x_rows.device,
                )
                if prefetch is not None:
                    prefetch("W_down", segment.group_start, segment.group_stop)
                gate, up = _segment_routed_projection(
                    x_rows, w_gate_up, local_offsets, gather_rows=local_rows,
                    max_group_rows=max_group_rows,
                ).chunk(2, dim=-1)
            else:
                w_gate = store.materialize_segment(
                    "W_gate", segment.group_start, segment.group_stop,
                    dtype=x_rows.dtype, device=x_rows.device,
                )
                if prefetch is not None:
                    prefetch("W_up", segment.group_start, segment.group_stop)
                gate = _segment_routed_projection(
                    x_rows, w_gate, local_offsets, gather_rows=local_rows,
                    max_group_rows=max_group_rows,
                )
                w_up = store.materialize_segment(
                    "W_up", segment.group_start, segment.group_stop,
                    dtype=x_rows.dtype, device=x_rows.device,
                )
                if prefetch is not None:
                    prefetch("W_down", segment.group_start, segment.group_stop)
                up = _segment_routed_projection(
                    x_rows, w_up, local_offsets, gather_rows=local_rows,
                    max_group_rows=max_group_rows,
                )
            w_down = store.materialize_segment(
                "W_down", segment.group_start, segment.group_stop,
                dtype=x_rows.dtype, device=x_rows.device,
            )
            _prefetch_next_routed_segment(prefetch, segments, index, store.has_fused_gate_up)
            grad_hidden = _segment_routed_projection(
                grad_rows, w_down.transpose(-2, -1), local_offsets,
                max_group_rows=max_group_rows,
            )
            grad_gate, grad_up = _swiglu7_backward(
                gate, up, grad_hidden, backend=ctx.activation_backend
            )
            if store.has_fused_gate_up:
                grad_grouped = _segment_routed_projection(
                    torch.cat((grad_gate, grad_up), dim=-1),
                    w_gate_up.transpose(-2, -1), local_offsets,
                    max_group_rows=max_group_rows,
                )
                del w_gate_up
            else:
                grad_grouped = _segment_routed_projection(
                    grad_gate.contiguous(), w_gate.transpose(-2, -1), local_offsets,
                    max_group_rows=max_group_rows,
                )
                grad_grouped.add_(
                    _segment_routed_projection(
                        grad_up.contiguous(), w_up.transpose(-2, -1), local_offsets,
                        max_group_rows=max_group_rows,
                    )
                )
                del w_gate, w_up
            grad_input.index_add_(0, local_rows.to(torch.int64), grad_grouped)
            del w_down, gate, up, grad_hidden, grad_gate, grad_up, grad_grouped
        return grad_input, None, None, None, None


class Magi2QuantizedGroupedMoEBackend(Magi2GroupedMoEBackend):
    """Grouped execution over NF4-packed MAGI-2 routed experts.

    The dense expert parameters no longer exist on the vendored layer, so both
    the expert-layout verdict and the matmul operands come from the layer's
    :class:`~mirai.core.models.magi2_preview.quantized_experts.Magi2Nf4ExpertStore`.
    Routing, activation, and the probability-weighted combine stay in the
    inherited native-autograd path.
    """

    name = "grouped_nf4"

    def __init__(
        self,
        plan: Magi2GroupedMoEPlan,
        *,
        token_chunk_size: int = 0,
        expert_autograd: str = "standard",
        activation_backend: str = "torch",
        cpu_expert_max_tokens: int = 0,
        int8_expert_checkpoint: bool = False,
    ) -> None:
        super().__init__(
            plan,
            token_chunk_size=token_chunk_size,
            activation_backend=activation_backend,
        )
        self.expert_autograd = str(expert_autograd)
        self.cpu_expert_max_tokens = cpu_expert_max_tokens
        self.int8_expert_checkpoint = int8_expert_checkpoint
        if int8_expert_checkpoint and (self.expert_autograd != "segmented_recompute" or cpu_expert_max_tokens):
            raise Magi2GroupedMoEPolicyError(
                "MAGI-2 INT8 expert checkpointing requires segmented_recompute with CUDA expert execution."
            )
        if cpu_expert_max_tokens and (
            plan.routed_gemm != "disabled" or plan.forward_backend == "scaled_mm_fp4"
            or self.expert_autograd != "segmented_recompute"
        ):
            raise Magi2GroupedMoEPolicyError(
                "Hybrid MAGI-2 experts require dequantizing grouped GEMM and segmented_recompute."
            )
        if plan.forward_backend == "scaled_mm_fp4" and (
            self.expert_autograd != "standard" or plan.routed_gemm != "disabled"
        ):
            raise Magi2GroupedMoEPolicyError(
                "Native MAGI-2 FP4 GEMM requires standard expert autograd and moe_routed_gemm='disabled'."
            )

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
        if self.plan.forward_backend == "scaled_mm_fp4":
            if not isinstance(store, Magi2PackedExpertStore) or not store.has_native_fp4_layout:
                raise Magi2GroupedMoEPolicyError("Native MAGI-2 FP4 GEMM requires uniform contraction-major FP4 weights.")
            block_size = 32 if store.quant_format == "mxfp4" else 16
            if any(store.expert_weight_shape(key)[1] % block_size for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES):
                raise Magi2GroupedMoEPolicyError(f"Native MAGI-2 FP4 GEMM requires input widths divisible by {block_size}.")
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
        store, offsets = _packed_expert_layout(self._store(module), offsets)
        return _QuantizedGroupedExpertLinear.apply(
            x_sorted, offsets, store, key, forward_backend, dx_backend
        )

    def _project_gate_up(self, module, x_sorted, offsets, forward_backend, dx_backend):
        store, offsets = _packed_expert_layout(self._store(module), offsets)
        keys = (MAGI2_FUSED_GATE_UP,) if store.has_fused_gate_up else ("W_gate", "W_up")
        projections = tuple(
            _QuantizedGroupedExpertLinear.apply(x_sorted, offsets, store, key, forward_backend, dx_backend)
            for key in keys
        )
        return projections[0].chunk(2, dim=-1) if store.has_fused_gate_up else projections

    def _routed_inference(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        from mirai.core.moe.runtime.routed_gemm_triton import (
            routed_weighted_projection,
        )

        tokens, num_heads, d_head = map(int, x_heads.shape)
        num_experts = int(module.num_experts)
        groups = num_heads * num_experts
        device = x_heads.device
        head_axis = torch.arange(num_heads, device=device).view(num_heads, 1, 1)
        token_axis = torch.arange(tokens, device=device).view(1, tokens, 1)
        flat_experts = (topk_indices + head_axis * num_experts).reshape(-1)
        flat_experts, groups = self._physical_expert_layout(module, flat_experts, groups)
        assignment_to_row = (
            token_axis * num_heads + head_axis
        ).expand_as(topk_indices).reshape(-1)
        order = flat_experts.argsort(stable=True)
        sorted_rows = assignment_to_row.index_select(0, order)
        offsets = torch.bincount(flat_experts, minlength=groups).cumsum(0).to(torch.int32)
        x_rows = x_heads.reshape(tokens * num_heads, d_head)
        store, offsets = _packed_expert_layout(self._store(module), offsets)
        output = x_rows.new_zeros((tokens * num_heads, d_head))
        segments = _packed_group_segments(offsets, max_groups=store.segment_group_span(), routed=True)
        prefetch = _expert_projection_prefetch(store, x_rows)
        segments = tuple((segment, rows) for segment, rows in segments if segment.row_count)
        for index, (segment, max_group_rows) in enumerate(segments):
            local_offsets = (
                offsets[segment.group_start : segment.group_stop] - segment.row_start
            ).contiguous()
            local_rows = sorted_rows[segment.row_start : segment.row_stop].contiguous()
            if store.has_fused_gate_up:
                w_gate_up = store.materialize_segment(
                    MAGI2_FUSED_GATE_UP, segment.group_start, segment.group_stop,
                    dtype=x_rows.dtype, device=device,
                )
                if prefetch is not None:
                    prefetch("W_down", segment.group_start, segment.group_stop)
                gate, up = _segment_routed_projection(
                    x_rows, w_gate_up, local_offsets, gather_rows=local_rows,
                    max_group_rows=max_group_rows,
                ).chunk(2, dim=-1)
                del w_gate_up
            else:
                w_gate = store.materialize_segment(
                    "W_gate", segment.group_start, segment.group_stop,
                    dtype=x_rows.dtype, device=device,
                )
                if prefetch is not None:
                    prefetch("W_up", segment.group_start, segment.group_stop)
                gate = _segment_routed_projection(
                    x_rows, w_gate, local_offsets, gather_rows=local_rows,
                    max_group_rows=max_group_rows,
                )
                del w_gate
                w_up = store.materialize_segment(
                    "W_up", segment.group_start, segment.group_stop,
                    dtype=x_rows.dtype, device=device,
                )
                if prefetch is not None:
                    prefetch("W_down", segment.group_start, segment.group_stop)
                up = _segment_routed_projection(
                    x_rows, w_up, local_offsets, gather_rows=local_rows,
                    max_group_rows=max_group_rows,
                )
                del w_up
            hidden = _swiglu7_forward(
                gate, up, output_dtype=x_rows.dtype, backend=self.activation_backend
            )
            del gate, up
            w_down = store.materialize_segment(
                "W_down", segment.group_start, segment.group_stop,
                dtype=x_rows.dtype, device=device,
            )
            _prefetch_next_routed_segment(prefetch, segments, index, store.has_fused_gate_up)
            output.add_(
                routed_weighted_projection(
                    hidden,
                    w_down,
                    local_offsets,
                    grouped_to_assignment=order[
                        segment.row_start : segment.row_stop
                    ].contiguous(),
                    assignment_to_token=assignment_to_row,
                    coefficients=topk_probs.reshape(-1),
                    token_rows=tokens * num_heads,
                    max_group_rows=max_group_rows,
                )
            )
            del hidden, w_down
        return output.view(tokens, num_heads, d_head)

    def _execute_routed_chunk(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(module, "_mirai_expert_lora", None) is not None and (
            self.plan.routed_gemm != "disabled" or self.cpu_expert_max_tokens
        ):
            raise Magi2GroupedMoEPolicyError("MAGI-2 expert LoRA requires the grouped device execution path.")
        if self.plan.routed_gemm != "disabled" and not torch.is_grad_enabled():
            supported = (
                x_heads.device.type == "cuda"
                and x_heads.dtype == torch.bfloat16
                and probe_backend("persistent", device=x_heads.device).available
            )
            if supported:
                return self._routed_inference(
                    module, x_heads, topk_probs, topk_indices
                )
            if self.plan.routed_gemm == "triton":
                raise Magi2GroupedMoEPolicyError(
                    "memory.moe_routed_gemm='triton' with packed experts requires "
                    "BF16 CUDA activations and the routed Triton backend."
                )
        return super()._execute_routed_chunk(
            module, x_heads, topk_probs, topk_indices
        )

    def branch_features(
        self,
        module: Any,
        x_heads: torch.Tensor,
        sorted_rows: torch.Tensor,
        offsets: torch.Tensor,
        forward_backend: str,
        dx_backend: str,
    ) -> torch.Tensor:
        adapters = getattr(module, "_mirai_expert_lora", None)
        if adapters is not None:
            return self._expert_lora_features(module, x_heads, sorted_rows, offsets, forward_backend, dx_backend, adapters)
        if forward_backend == "scaled_mm_fp4":
            return super().branch_features(module, x_heads, sorted_rows, offsets, forward_backend, dx_backend)
        if self.plan.routed_gemm != "disabled":
            supported = (
                x_heads.device.type == "cuda"
                and x_heads.dtype == torch.bfloat16
                and probe_backend("persistent", device=x_heads.device).available
            )
            if supported:
                tokens = int(x_heads.shape[0])
                num_heads = int(x_heads.shape[1])
                d_head = int(x_heads.shape[2])
                store, offsets = _packed_expert_layout(self._store(module), offsets)
                return self._segmented_expert_apply(
                    _SegmentedRoutedQuantizedExpertMlp,
                    x_heads.reshape(tokens * num_heads, d_head),
                    offsets,
                    sorted_rows,
                    store,
                    self.activation_backend,
                )
            if self.plan.routed_gemm == "triton":
                raise Magi2GroupedMoEPolicyError(
                    "memory.moe_routed_gemm='triton' with packed experts requires "
                    "BF16 CUDA activations and the routed Triton backend."
                )
        if torch.is_grad_enabled() and self.expert_autograd == "standard":
            return super().branch_features(
                module,
                x_heads,
                sorted_rows,
                offsets,
                forward_backend,
                dx_backend,
            )
        tokens = int(x_heads.shape[0])
        num_heads = int(x_heads.shape[1])
        d_head = int(x_heads.shape[2])
        x_rows = x_heads.reshape(tokens * num_heads, d_head)
        x_sorted = x_rows.index_select(0, sorted_rows)
        if self.cpu_expert_max_tokens and x_heads.device.type == "cuda":
            return _hybrid_expert_features(
                x_sorted, offsets, self._store(module), cpu_max_tokens=self.cpu_expert_max_tokens,
                forward_backend=forward_backend, dx_backend=dx_backend, activation_backend=self.activation_backend,
            )
        store, offsets = _packed_expert_layout(self._store(module), offsets)
        if torch.is_grad_enabled():
            return self._segmented_expert_apply(
                _SegmentedQuantizedExpertMlp,
                x_sorted,
                offsets,
                store,
                forward_backend,
                dx_backend,
                self.activation_backend,
            )
        return run_segmented_grouped_expert_mlp(
            x_sorted,
            offsets,
            materialize=lambda key, start, stop: store.materialize_segment(
                key, start, stop, dtype=x_sorted.dtype, device=x_sorted.device
            ),
            backend=forward_backend,
            max_groups=store.segment_group_span(),
            output_columns=d_head,
            activation_backend=self.activation_backend,
            fused_gate_up=store.has_fused_gate_up,
            prefetch=_expert_projection_prefetch(store, x_sorted),
        )

    def _segmented_expert_apply(self, operation, *args):
        if self.int8_expert_checkpoint and torch.is_grad_enabled():
            from mirai.core.training.residency.activation_compression import int8_saved_rows_context

            with int8_saved_rows_context():
                return operation.apply(*args)
        return operation.apply(*args)

    def _expert_lora_features(self, module, x_heads, sorted_rows, offsets, forward_backend, dx_backend, adapters):
        from contextlib import nullcontext
        from torch.utils.checkpoint import checkpoint

        from mirai.core.training.residency.activation_compression import int8_saved_rows_context

        if self.plan.routed_gemm != "disabled" or self.cpu_expert_max_tokens:
            raise Magi2GroupedMoEPolicyError("MAGI-2 expert LoRA requires the grouped device execution path.")
        store = self._store(module)
        selected = {key: set(adapter.active_expert_ids()) for key, adapter in adapters.items()}
        rows = x_heads.flatten(0, 1).index_select(0, sorted_rows)
        boundaries = [0, *_grouped_boundaries(offsets)]
        outputs = []
        for group, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
            if start == stop:
                continue
            subset = _ActiveExpertGroups(store, torch.tensor([group]))
            local_offsets = offsets.new_tensor([stop - start])

            def execute(value, expert=group, active_store=subset, bounds=local_offsets):
                def project(key, x):
                    result = _QuantizedGroupedExpertLinear.apply(
                        x, bounds, active_store, key, forward_backend, dx_backend,
                    )
                    if expert in selected[key]:
                        result = result + adapters[key](x, expert_idx=expert)
                    return result

                gate, up = project("W_gate", value), project("W_up", value)
                hidden = _ConfiguredSwiGlu7.apply(
                    gate, up, store.expert_weight_dtype("W_down"), self.activation_backend,
                )
                return project("W_down", hidden)

            group_rows = rows[start:stop]
            if torch.is_grad_enabled() and self.expert_autograd == "segmented_recompute":
                context = int8_saved_rows_context() if self.int8_expert_checkpoint else nullcontext()
                with context:
                    outputs.append(checkpoint(execute, group_rows, use_reentrant=False))
            else:
                outputs.append(execute(group_rows))
        return torch.cat(outputs, dim=0) if outputs else rows.new_empty((0, rows.shape[-1]))


# The only MoEOptimizationPolicy fields MAGI-2 grouped execution reads. Every
# other field must still hold its dataclass default, so a field added to the
# generic policy fails closed here instead of being silently ignored.
_CONSUMED_POLICY_FIELDS = frozenset(
    {
        "kernel_backend",
        "moe_gemm_backend",
        "moe_gemm_backend_forward",
        "moe_gemm_backend_dx",
        "moe_activation_backend",
        "moe_routed_gemm",
        "moe_routed_gemm_tuning",
        "moe_routed_gemm_cache_path",
        "moe_routed_gemm_architecture",
        "router_quantization",
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
        "moe_expert_autograd",
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
        "On-load expert quantization requires memory.frozen_weight_quantization='nf4'."
    ),
    "expert_device_cache_gib": (
        "Packed MAGI-2 experts are layer-resident state moved by the block "
        "residency subsystem, not per-expert operands streamed on demand, so "
        "they have no second byte-bounded device cache. Use "
        "training.blocks_to_swap to choose how much of the packed model stays "
        "device-resident."
    ),
    "device_residency_budget_gib": (
        "This family has no independent expert-residency owner to account against a shared ceiling."
    ),
    "moe_dispatch": "This family owns its routed dispatch.",
    "moe_dispatch_preprocess": "This family owns its routed dispatch.",
    "moe_gemm_backend_dw": (
        "Grouped execution keeps expert weights frozen, so the weight-gradient "
        "GEMM has no consumer."
    ),
}


def _reject_unconsumed_policy_fields(
    policy: Any, *, quantized_experts: bool = False, packed_stream_bound: bool = False,
) -> None:
    """Raise for any policy field MAGI-2 neither consumes nor leaves at default.

    The consumed set is a function of the storage format: three fields acquire a
    consumer only once the routed experts are packed, and stay rejected while
    the family runs native BF16 experts.
    """
    consumed = _QUANTIZED_CONSUMED_POLICY_FIELDS if quantized_experts else _CONSUMED_POLICY_FIELDS
    if quantized_experts and packed_stream_bound:
        consumed = consumed | {
            "packed_state_preload",
            "packed_stream_cache_gib",
            "packed_stream_backend",
            "packed_stream_prefetch_depth",
        }
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
    policy: Any, *, quantized_experts: bool = False, packed_stream_bound: bool = False,
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
    _reject_unconsumed_policy_fields(
        policy, quantized_experts=quantized_experts, packed_stream_bound=packed_stream_bound,
    )
    if quantized_experts and kernel_backend == "torch":
        raise Magi2GroupedMoEPolicyError(
            "memory.moe_kernel_backend='torch' selects the vendored per-expert "
            "reference loop, which reads the dense W_gate/W_up/W_down tensors "
            "that memory.frozen_weight_quantization='nf4' replaces with packed "
            "storage. Use 'auto' or 'grouped'."
        )
    routed_gemm = normalize_routed_gemm_mode(
        getattr(policy, "moe_routed_gemm", "disabled")
    )
    if (
        kernel_backend in _DEFAULT_KERNEL_BACKENDS
        and not quantized_experts
        and routed_gemm == "disabled"
    ):
        return None
    main = normalize_moe_gemm_backend(getattr(policy, "moe_gemm_backend", "auto"))
    forward = normalize_moe_gemm_role_backend(getattr(policy, "moe_gemm_backend_forward", ""))
    dx = normalize_moe_gemm_role_backend(getattr(policy, "moe_gemm_backend_dx", ""))
    if (forward or main) == "scaled_mm_fp4" and not quantized_experts:
        raise Magi2GroupedMoEPolicyError("Native MAGI-2 FP4 GEMM requires packed experts.")
    return Magi2GroupedMoEPlan(
        forward_backend=forward or main,
        dx_backend=dx or main,
        routed_gemm=routed_gemm,
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


def attach_grouped_moe_backend(transformer: Any, backend: Magi2GroupedMoEBackend | None) -> int:
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
