"""Autograd-complete routed permutation and duplicate-safe combine operations."""

from __future__ import annotations

import torch


class _PermuteRoutedTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, token_indices):
        ctx.input_rows = int(tokens.shape[0])
        ctx.save_for_backward(token_indices)
        return tokens.index_select(0, token_indices).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        (token_indices,) = ctx.saved_tensors
        grad_tokens = grad_output.new_zeros((ctx.input_rows, grad_output.shape[-1]))
        grad_tokens.index_add_(0, token_indices, grad_output)
        return grad_tokens, None


class _CombineRoutedTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, scores, token_indices, output_rows):
        weighted = expert_output.float() * scores.float().unsqueeze(1)
        output = weighted.new_zeros((int(output_rows), weighted.shape[-1]))
        output.index_add_(0, token_indices, weighted)
        ctx.expert_dtype = expert_output.dtype
        ctx.score_dtype = scores.dtype
        ctx.save_for_backward(expert_output, scores, token_indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        expert_output, scores, token_indices = ctx.saved_tensors
        gathered = grad_output.index_select(0, token_indices)
        grad_expert = (gathered * scores.float().unsqueeze(1)).to(
            dtype=ctx.expert_dtype
        )
        grad_scores = (gathered * expert_output.float()).sum(dim=1).to(
            dtype=ctx.score_dtype
        )
        return grad_expert, grad_scores, None, None


def permute_routed_tokens(
    tokens: torch.Tensor, token_indices: torch.Tensor
) -> torch.Tensor:
    return _PermuteRoutedTokens.apply(tokens, token_indices)


def combine_routed_tokens(
    expert_output: torch.Tensor,
    scores: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    output_rows: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return _CombineRoutedTokens.apply(
        expert_output, scores, token_indices, int(output_rows)
    ).to(dtype=output_dtype)


__all__ = ["combine_routed_tokens", "permute_routed_tokens"]
