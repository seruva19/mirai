# SPDX-License-Identifier: Apache-2.0
"""Public grouped-GEMM dispatch via ``torch.nn.functional.grouped_mm`` (probed).

The ``moe_gemm_backend="torch_grouped"`` forward primitive. It runs the routed
expert MLP on the shared SORTED-CONTIGUOUS :class:`DispatchPlan` layout (the same
device-resident offsets the persistent path builds) but uses the framework
grouped-mm op instead of the vendored Triton kernel. The public
``torch.nn.functional.grouped_mm`` is preferred; the private ``torch._grouped_mm``
is the fallback when the public name is absent.

Frozen experts are RE-DEQUANTIZED in backward (never a bf16 stack retained on the
autograd graph), matching the invariant of every other Mirai expert path. LoRA is
added through the existing ``sorted_subset_forward`` (the vendored grouped_gemm
SM80+ path), identical to the persistent dispatch adapter add — so this backend is
numerically a sibling of ``persistent``, not of the padded ``bmm`` primitive, and
is checked against both within bf16 tolerance in the CUDA equivalence suite.

This backend dequantizes the full expert stack for the
matmul (the natural ``grouped_mm`` usage, same as ``CompressedGroupedExperts.
grouped_linear``). The persistent backend is the chunk-streamed
sorted-contiguous path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from mirai.core.moe.runtime.gemm import resolve_moe_gemm_backend, run_grouped_dx

from .dispatch_preprocess import build_dispatch_plan
from .dispatch_ops import combine_routed_tokens, permute_routed_tokens


def _grouped_mm(x: torch.Tensor, weight_operand: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Call the public grouped_mm op, falling back to the private one.

    ``weight_operand`` is already in the ``[E, K, N]`` orientation the op expects
    (forward passes ``weight.transpose(-2, -1)``; the dX backward passes the
    ``[E, N, K]`` weight itself). ``offsets`` is the int32 inclusive-cumsum group
    boundary vector.
    """
    op = getattr(F, "grouped_mm", None)
    if not callable(op):
        op = getattr(torch, "_grouped_mm", None)
    if not callable(op):
        raise RuntimeError(
            "moe_gemm_backend='torch_grouped' needs torch.nn.functional.grouped_mm "
            "or torch._grouped_mm; neither exists in this torch build."
        )
    return op(x, weight_operand, offs=offsets)


class _FrozenGroupedMMLinear(torch.autograd.Function):
    """Frozen grouped linear over global offsets via the public grouped_mm op.

    Forward: ``y = grouped_mm(x, W^T, offs)`` for the full expert stack. Backward
    re-dequantizes the frozen weight and computes ``dX = grouped_mm(dY, W, offs)``.
    """

    @staticmethod
    def forward(ctx, x, owner, key, offsets):
        experts = tuple(range(int(owner.num_experts)))
        weight = owner._dequant_expert_stack(
            key, experts, dtype=x.dtype, device=x.device
        ).contiguous()
        output = _grouped_mm(x.contiguous(), weight.transpose(-2, -1).contiguous(), offsets)
        ctx.owner = owner
        ctx.key = key
        ctx.weight_dtype = x.dtype
        ctx.save_for_backward(offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (offsets,) = ctx.saved_tensors
        experts = tuple(range(int(ctx.owner.num_experts)))
        weight = ctx.owner._dequant_expert_stack(
            ctx.key, experts, dtype=ctx.weight_dtype, device=grad_output.device
        ).contiguous()
        dx_backend = resolve_moe_gemm_backend(
            "dx", device=grad_output.device
        )
        if dx_backend == "auto":
            dx_backend = "torch_grouped"
        grad_input = run_grouped_dx(
            grad_output.to(dtype=ctx.weight_dtype).contiguous(),
            weight,
            offsets,
            backend=dx_backend,
        )
        return grad_input, None, None, None


def _grouped_linear(
    owner,
    key: str,
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    route_gate=None,
) -> torch.Tensor:
    output = _FrozenGroupedMMLinear.apply(x, owner, key, offsets)
    adapter = owner.expert_lora[key] if key in owner.expert_lora else None
    if adapter is None:
        return output
    return output + adapter.sorted_subset_forward(
        x,
        expert_indices=expert_indices,
        m_offsets=offsets,
        counts=counts,
        route_gate=route_gate,
    )


def run_torch_grouped_dispatch(
    owner,
    tokens: torch.Tensor,
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    chunk_size: int,
    output: torch.Tensor,
    routed_adapter_gate=None,
) -> torch.Tensor:
    """Sorted-contiguous routed expert MLP through the public grouped_mm op.

    ``chunk_size`` is accepted for dispatch-runner signature symmetry; the public
    op consumes the full expert stack per matmul (see the module scope note), so
    it does not sub-chunk the dequant like the persistent path.
    """
    _ = int(chunk_size)
    plan = build_dispatch_plan(top_indices, top_scores, owner.num_experts)
    return run_torch_grouped_dispatch_plan(
        owner,
        tokens,
        plan,
        output=output,
        routed_adapter_gate=routed_adapter_gate,
    )


def run_torch_grouped_dispatch_plan(
    owner,
    tokens: torch.Tensor,
    plan,
    *,
    output: torch.Tensor,
    routed_adapter_gate=None,
) -> torch.Tensor:
    """Run a prebuilt sorted dispatch plan through framework grouped-mm."""
    offsets = plan.offsets  # int32 inclusive cumsum, numel == num_experts
    counts = plan.counts
    expert_ids = torch.arange(owner.num_experts, device=tokens.device, dtype=torch.long)
    sorted_tokens = permute_routed_tokens(tokens, plan.sorted_token_indices)
    route_gate = (
        None
        if routed_adapter_gate is None
        else routed_adapter_gate.resolve_sorted(plan.sorted_token_indices, counts)
    )
    primary_role = "gate" if owner.expert_mlp_spec.uses_gated_product else "input"
    primary = _grouped_linear(
        owner,
        owner.projection_for_role(primary_role),
        sorted_tokens,
        expert_ids,
        counts,
        offsets,
        route_gate=route_gate,
    )
    secondary = None
    if owner.expert_mlp_spec.uses_gated_product:
        secondary = _grouped_linear(
            owner,
            owner.projection_for_role("up"),
            sorted_tokens,
            expert_ids,
            counts,
            offsets,
            route_gate=route_gate,
        )
    hidden = owner._combine_expert_inputs(primary, secondary)
    owner._capture_routed_intermediates(hidden, plan.sort_positions)
    expert_output = _grouped_linear(
        owner,
        owner.projection_for_role("down"),
        hidden,
        expert_ids,
        counts,
        offsets,
        route_gate=route_gate,
    )
    owner._capture_routed_outputs(expert_output, plan.sort_positions)
    combined = combine_routed_tokens(
        expert_output,
        plan.sorted_scores,
        plan.sorted_token_indices,
        output_rows=int(tokens.shape[0]),
        output_dtype=tokens.dtype,
    )
    return output.to(dtype=tokens.dtype) + combined
