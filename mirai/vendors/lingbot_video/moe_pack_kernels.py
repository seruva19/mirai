# Adapted from LingBot-Video's Triton MoE token-pack path (Apache-2.0).
# Source: https://github.com/Robbyant/lingbot-video
# See the adjacent LICENSE.

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional outside GPU deployments.
    triton = None
    tl = None


if triton is not None and tl is not None:
    @triton.jit
    def _moe_route_count_slots_kernel(
        flat_indices,
        counts,
        route_slots,
        num_routes: tl.constexpr,
        block_m: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
        mask = offsets < num_routes
        experts = tl.load(flat_indices + offsets, mask=mask, other=0)
        slots = tl.atomic_add(counts + experts, 1, sem="relaxed", mask=mask)
        tl.store(route_slots + offsets, slots, mask=mask)

    @triton.jit
    def _moe_pack_tokens_kernel(
        tokens,
        flat_scores,
        flat_indices,
        offsets,
        route_slots,
        permuted_tokens,
        sorted_positions,
        sorted_scores,
        hidden_size: tl.constexpr,
        top_k: tl.constexpr,
        block_h: tl.constexpr,
    ):
        route_idx = tl.program_id(0)
        hidden_block = tl.program_id(1)
        offsets_h = hidden_block * block_h + tl.arange(0, block_h)
        hidden_mask = offsets_h < hidden_size

        expert_idx = tl.load(flat_indices + route_idx)
        slot = tl.load(route_slots + route_idx)
        expert_offset = tl.load(offsets + expert_idx)
        dest_idx = expert_offset + slot
        token_idx = route_idx // top_k

        values = tl.load(
            tokens + token_idx * hidden_size + offsets_h,
            mask=hidden_mask,
            other=0.0,
        )
        tl.store(
            permuted_tokens + dest_idx * hidden_size + offsets_h,
            values,
            mask=hidden_mask,
        )

        first_hidden_block = hidden_block == 0
        tl.store(sorted_positions + dest_idx, route_idx, mask=first_hidden_block)
        score = tl.load(flat_scores + route_idx, mask=first_hidden_block, other=0.0)
        tl.store(sorted_scores + dest_idx, score, mask=first_hidden_block)
else:
    _moe_route_count_slots_kernel = None
    _moe_pack_tokens_kernel = None


def _reorder_tokens_triton_pack_forward(
    tokens: torch.Tensor,
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    num_experts: int,
    block_h: int = 64,
    block_m: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if _moe_route_count_slots_kernel is None or _moe_pack_tokens_kernel is None:
        raise RuntimeError("model.params.moe_reorder_backend='triton_pack' requires Triton")
    if tokens.device.type != "cuda":
        raise RuntimeError("model.params.moe_reorder_backend='triton_pack' requires CUDA tensors")
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D tokens, got {tokens.ndim}D")
    if top_scores.shape != top_indices.shape:
        raise ValueError("top_scores and top_indices must have the same shape")
    if top_indices.ndim != 2:
        raise ValueError(f"Expected 2D top_indices, got {top_indices.ndim}D")

    num_tokens = tokens.shape[0]
    hidden_size = tokens.shape[1]
    top_k = top_indices.shape[1]
    num_routes = top_indices.numel()
    counts = torch.zeros(num_experts, dtype=torch.int32, device=tokens.device)

    if num_routes == 0:
        return (
            tokens.new_empty((0, hidden_size)),
            counts,
            torch.empty(0, dtype=torch.int64, device=tokens.device),
            top_scores.new_empty((0,)),
            num_tokens,
            top_k,
        )

    tokens = tokens.contiguous()
    flat_scores = top_scores.contiguous().reshape(-1)
    flat_indices = top_indices.contiguous().to(torch.int32).reshape(-1)
    route_slots = torch.empty(num_routes, dtype=torch.int32, device=tokens.device)

    _moe_route_count_slots_kernel[(triton.cdiv(num_routes, block_m),)](
        flat_indices,
        counts,
        route_slots,
        num_routes,
        block_m,
    )

    counts_i64 = counts.to(torch.int64)
    offsets = torch.empty(num_experts + 1, dtype=torch.int64, device=tokens.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts_i64, dim=0)

    permuted_tokens = torch.empty((num_routes, hidden_size), dtype=tokens.dtype, device=tokens.device)
    sorted_positions = torch.empty(num_routes, dtype=torch.int64, device=tokens.device)
    sorted_scores = torch.empty(num_routes, dtype=top_scores.dtype, device=tokens.device)
    _moe_pack_tokens_kernel[(num_routes, triton.cdiv(hidden_size, block_h))](
        tokens,
        flat_scores,
        flat_indices,
        offsets,
        route_slots,
        permuted_tokens,
        sorted_positions,
        sorted_scores,
        hidden_size,
        top_k,
        block_h,
    )
    return permuted_tokens, counts, sorted_positions, sorted_scores, num_tokens, top_k


class _ReorderTokensTritonPack(torch.autograd.Function):
    """Autograd boundary for the fused route-pack kernels.

    Triton writes into fresh output buffers, so invoking the kernels directly
    disconnects both token and router-score gradients.  The backward is a
    duplicate-safe inverse gather and score scatter over the saved route map.
    """

    @staticmethod
    def forward(ctx, tokens, top_scores, top_indices, num_experts):
        packed, counts, positions, scores, _, top_k = (
            _reorder_tokens_triton_pack_forward(
                tokens,
                top_scores,
                top_indices,
                int(num_experts),
            )
        )
        ctx.input_rows = int(tokens.shape[0])
        ctx.top_k = int(top_k)
        ctx.top_scores_shape = tuple(top_scores.shape)
        ctx.save_for_backward(positions)
        ctx.mark_non_differentiable(counts, positions)
        return packed, counts, positions, scores

    @staticmethod
    def backward(ctx, grad_packed, _grad_counts, _grad_positions, grad_scores):
        (positions,) = ctx.saved_tensors
        token_indices = torch.div(
            positions,
            int(ctx.top_k),
            rounding_mode="floor",
        )
        grad_tokens = None
        if grad_packed is not None:
            grad_tokens = grad_packed.new_zeros(
                (int(ctx.input_rows), grad_packed.shape[-1])
            )
            grad_tokens.index_add_(0, token_indices, grad_packed)
        grad_top_scores = None
        if grad_scores is not None:
            flat = grad_scores.new_zeros(int(ctx.input_rows) * int(ctx.top_k))
            flat.scatter_(0, positions, grad_scores)
            grad_top_scores = flat.reshape(ctx.top_scores_shape)
        return grad_tokens, grad_top_scores, None, None


def reorder_tokens_triton_pack(
    tokens: torch.Tensor,
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    num_tokens = int(tokens.shape[0])
    top_k = int(top_indices.shape[1])
    packed, counts, positions, scores = _ReorderTokensTritonPack.apply(
        tokens,
        top_scores,
        top_indices,
        int(num_experts),
    )
    return packed, counts, positions, scores, num_tokens, top_k
