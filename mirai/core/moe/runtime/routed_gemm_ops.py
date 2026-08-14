# SPDX-License-Identifier: Apache-2.0
"""Compiler boundary for routed grouped GEMM.

The registered operator keeps Python validation and Triton launch construction
outside Dynamo's traced graph.  Its fake implementation exposes only the
shape/dtype contract, while the autograd registration composes the same routed
operators used by eager execution.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("mirai::routed_grouped_mm", mutates_args=())
def _routed_grouped_mm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    gather_rows: torch.Tensor,
    scatter_rows: torch.Tensor,
    output_rows: int,
    role: str,
) -> torch.Tensor:
    from .routed_gemm_triton import triton_routed_grouped_mm

    return triton_routed_grouped_mm(
        activation,
        weight,
        boundaries,
        gather_rows=gather_rows if gather_rows.numel() else None,
        scatter_rows=scatter_rows if scatter_rows.numel() else None,
        output_rows=output_rows,
        role=role,
    )


@_routed_grouped_mm.register_fake
def _routed_grouped_mm_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    gather_rows: torch.Tensor,
    scatter_rows: torch.Tensor,
    output_rows: int,
    role: str,
) -> torch.Tensor:
    del boundaries, gather_rows, scatter_rows, role
    return activation.new_empty((output_rows, weight.shape[2]))


@torch.library.custom_op("mirai::routed_grouped_dw", mutates_args=())
def _routed_grouped_dw(
    activation: torch.Tensor,
    grad_grouped: torch.Tensor,
    boundaries: torch.Tensor,
    gather_rows: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    from .routed_gemm_triton import triton_grouped_dw

    return triton_grouped_dw(
        activation,
        grad_grouped,
        boundaries,
        groups=groups,
        gather_rows=gather_rows if gather_rows.numel() else None,
    )


@_routed_grouped_dw.register_fake
def _routed_grouped_dw_fake(
    activation: torch.Tensor,
    grad_grouped: torch.Tensor,
    boundaries: torch.Tensor,
    gather_rows: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    del boundaries, gather_rows
    return activation.new_empty((groups, activation.shape[1], grad_grouped.shape[1]))


def _setup_context(ctx, inputs, output) -> None:
    del output
    activation, weight, boundaries, gather_rows, scatter_rows, _, _ = inputs
    ctx.save_for_backward(
        activation, weight, boundaries, gather_rows, scatter_rows
    )
    ctx.input_rows = activation.shape[0]


def _backward(ctx, grad_output: torch.Tensor):
    activation, weight, boundaries, gather_rows, scatter_rows = ctx.saved_tensors
    grouped_grad = (
        grad_output.index_select(0, scatter_rows.to(torch.int64))
        if scatter_rows.numel()
        else grad_output
    )
    grad_input = None
    if ctx.needs_input_grad[0]:
        empty = gather_rows.new_empty((0,))
        grouped_dx = _routed_grouped_mm(
            grouped_grad.contiguous(),
            weight.transpose(-2, -1),
            boundaries,
            empty,
            empty,
            grouped_grad.shape[0],
            "dx",
        )
        if gather_rows.numel():
            grad_input = grouped_dx.new_zeros((ctx.input_rows, grouped_dx.shape[1]))
            grad_input.index_add_(0, gather_rows.to(torch.int64), grouped_dx)
        else:
            grad_input = grouped_dx
    grad_weight = None
    if ctx.needs_input_grad[1]:
        grad_weight = _routed_grouped_dw(
            activation,
            grouped_grad.contiguous(),
            boundaries,
            gather_rows,
            weight.shape[0],
        )
    return grad_input, grad_weight, None, None, None, None, None


torch.library.register_autograd(
    "mirai::routed_grouped_mm",
    _backward,
    setup_context=_setup_context,
)


@torch.library.custom_op("mirai::routed_weighted_mm", mutates_args=())
def _routed_weighted_mm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    grouped_to_assignment: torch.Tensor,
    assignment_to_token: torch.Tensor,
    coefficients: torch.Tensor,
    token_rows: int,
) -> torch.Tensor:
    from .routed_gemm_triton import _weighted_reduce, triton_routed_grouped_mm

    grouped = triton_routed_grouped_mm(
        activation, weight, boundaries, output_rows=activation.shape[0],
        role="weighted",
    )
    return _weighted_reduce(
        grouped, grouped_to_assignment, assignment_to_token, coefficients,
        token_rows,
    )


@_routed_weighted_mm.register_fake
def _routed_weighted_mm_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    grouped_to_assignment: torch.Tensor,
    assignment_to_token: torch.Tensor,
    coefficients: torch.Tensor,
    token_rows: int,
) -> torch.Tensor:
    del boundaries, grouped_to_assignment, assignment_to_token, coefficients
    return activation.new_empty((token_rows, weight.shape[2]))


def _setup_weighted_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs[:-1])


def _weighted_backward(ctx, grad_output: torch.Tensor):
    (activation, weight, boundaries, grouped_to_assignment,
     assignment_to_token, coefficients) = ctx.saved_tensors
    empty = grouped_to_assignment.new_empty((0,))
    grouped = _routed_grouped_mm(
        activation, weight, boundaries, empty, empty, activation.shape[0], "weighted"
    )
    assignment = grouped_to_assignment.to(torch.int64)
    token_for_grouped = assignment_to_token.index_select(0, assignment).to(torch.int64)
    coefficient_for_grouped = coefficients.reshape(-1).index_select(0, assignment)
    selected_grad = grad_output.index_select(0, token_for_grouped)
    grouped_grad = selected_grad * coefficient_for_grouped[:, None].to(selected_grad.dtype)
    grad_input = (
        _routed_grouped_mm(
            grouped_grad.contiguous(), weight.transpose(-2, -1), boundaries,
            empty, empty, grouped_grad.shape[0],
            "dx",
        )
        if ctx.needs_input_grad[0] else None
    )
    grad_weight = (
        _routed_grouped_dw(
            activation, grouped_grad.contiguous(), boundaries, empty,
            weight.shape[0],
        )
        if ctx.needs_input_grad[1] else None
    )
    grad_coefficients = None
    if ctx.needs_input_grad[5]:
        contribution = (grouped.float() * selected_grad.float()).sum(1)
        grad_coefficients = torch.zeros_like(coefficients).reshape(-1)
        grad_coefficients.index_copy_(
            0, assignment, contribution.to(coefficients.dtype)
        )
        grad_coefficients = grad_coefficients.view_as(coefficients)
    return (grad_input, grad_weight, None, None, None, grad_coefficients, None)


torch.library.register_autograd(
    "mirai::routed_weighted_mm",
    _weighted_backward,
    setup_context=_setup_weighted_context,
)


def routed_projection_op(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    gather_rows: torch.Tensor,
    scatter_rows: torch.Tensor,
    output_rows: int,
    role: str = "forward",
) -> torch.Tensor:
    """Call the compile-visible routed projection operator."""
    return _routed_grouped_mm(
        activation,
        weight,
        boundaries,
        gather_rows,
        scatter_rows,
        output_rows,
        role,
    )


def routed_weighted_projection_op(
    activation: torch.Tensor,
    weight: torch.Tensor,
    boundaries: torch.Tensor,
    grouped_to_assignment: torch.Tensor,
    assignment_to_token: torch.Tensor,
    coefficients: torch.Tensor,
    token_rows: int,
) -> torch.Tensor:
    return _routed_weighted_mm(
        activation, weight, boundaries, grouped_to_assignment,
        assignment_to_token, coefficients, token_rows,
    )


__all__ = ["routed_projection_op", "routed_weighted_projection_op"]
