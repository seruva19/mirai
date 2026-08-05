"""Grouped-GEMM expert execution for MAGI-2 multi-head sparse-MoE layers.

Expert numerics reproduce SandAI's Apache-2.0 reference implementation vendored
at ``mirai/vendors/magi2_preview/model/magi2_preview.py``
(``CoreMultiHeadMoE._torch_forward``), which remains the reference path.

Tensor semantics: ``x_heads`` is ``[tokens, heads, d_head]`` and the packed
expert weights are indexed by a single flattened ``head * num_experts + expert``
axis, so one sorted token-slice layout covers every head at once. Routing,
activation, and combine stay in plain autograd; only the per-group expert matmul
is a custom Function, because its weights are frozen and carry no gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable

import torch

from mirai.core.moe.runtime.gemm import (
    grouped_mm_op,
    normalize_moe_gemm_backend,
    normalize_moe_gemm_role_backend,
    probe_backend,
    run_grouped_dx,
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
        return op(x_sorted.contiguous(), weight, offs=offsets)
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
        return op(grad_output.contiguous(), weight_t, offs=offsets)
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

    def _resolve(self, device: torch.device) -> tuple[str, str]:
        key = f"{device.type}:{device.index if device.index is not None else -1}"
        cached = self._resolved.get(key)
        if cached is not None:
            return cached
        resolved: list[str] = []
        for role, requested in (
            ("forward", self.plan.forward_backend),
            ("dx", self.plan.dx_backend),
        ):
            if requested == "auto":
                probe = probe_backend("torch_grouped", device=device)
                resolved.append("torch_grouped" if probe.available else "bmm")
                continue
            probe = probe_backend(requested, device=device)
            if not probe.available:
                raise RuntimeError(
                    f"memory.moe_gemm_backend (or its '{role}' role override) "
                    f"selected '{requested}', which is unavailable for MAGI-2 "
                    f"grouped MoE execution on {device}: {probe.reason}."
                )
            resolved.append(requested)
        selection = (resolved[0], resolved[1])
        self._resolved[key] = selection
        return selection

    def execute(
        self,
        module: Any,
        x_heads: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        forward_backend, dx_backend = self._resolve(x_heads.device)
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
        gate = _GroupedExpertLinear.apply(
            x_sorted, module.W_gate, offsets, forward_backend, dx_backend
        )
        up = _GroupedExpertLinear.apply(
            x_sorted, module.W_up, offsets, forward_backend, dx_backend
        )
        gate = gate.float().clamp(max=7.0)
        up = up.float().clamp(min=-7.0, max=7.0)
        hidden = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
        expert_output = _GroupedExpertLinear.apply(
            hidden.to(module.W_down.dtype),
            module.W_down,
            offsets,
            forward_backend,
            dx_backend,
        )
        weighted = expert_output * sorted_probs.to(expert_output.dtype)[:, None]
        output = torch.zeros(
            tokens * num_heads, d_head, device=device, dtype=x_heads.dtype
        ).index_add(0, sorted_rows, weighted.to(x_heads.dtype))
        return output.view(tokens, num_heads, d_head)


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
    "expert_weight_access": "This family keeps native BF16 experts.",
    "moe_dispatch": "This family owns its routed dispatch.",
    "moe_dispatch_preprocess": "This family owns its routed dispatch.",
    "moe_gemm_backend_dw": (
        "Grouped execution keeps expert weights frozen, so the weight-gradient "
        "GEMM has no consumer."
    ),
}


def _reject_unconsumed_policy_fields(policy: Any) -> None:
    """Raise for any policy field MAGI-2 neither consumes nor leaves at default."""
    defaults = MoEOptimizationPolicy()
    for field in fields(MoEOptimizationPolicy):
        if field.name in _CONSUMED_POLICY_FIELDS:
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


def resolve_magi2_moe_execution(policy: Any) -> Magi2GroupedMoEPlan | None:
    """Validate a MoE optimization policy and return the grouped plan, if any.

    Returns ``None`` for the default kernel backends, which keep the vendored
    reference execution. Every policy field MAGI-2 cannot honor raises.
    """
    kernel_backend = str(getattr(policy, "kernel_backend", "auto"))
    if kernel_backend not in _GROUPED_KERNEL_BACKENDS + _DEFAULT_KERNEL_BACKENDS:
        raise Magi2GroupedMoEPolicyError(
            "MAGI-2 Preview supports memory.moe_kernel_backend='auto', 'torch', "
            f"or 'grouped'; got '{kernel_backend}'."
        )
    _reject_unconsumed_policy_fields(policy)
    if kernel_backend in _DEFAULT_KERNEL_BACKENDS:
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
    gate stays with :meth:`Magi2GroupedMoEBackend._resolve`.
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
            module._mirai_moe_kernel_backend = backend
            attached += 1
    if backend is not None and attached == 0:
        raise Magi2GroupedMoEPolicyError(
            "memory.moe_kernel_backend='grouped' matched no MAGI-2 multi-head MoE "
            "layer in this model."
        )
    return attached
