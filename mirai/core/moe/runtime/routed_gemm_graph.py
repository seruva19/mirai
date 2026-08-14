# SPDX-License-Identifier: Apache-2.0
"""Static-shape prewarm contract for routed GEMM CUDA Graph buckets."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutedGemmGraphBucket:
    """One graph-reusable routed projection shape and layout signature."""

    activation_rows: int
    routed_rows: int
    groups: int
    k_size: int
    n_size: int
    gather: bool
    scatter: bool

    def __post_init__(self) -> None:
        dimensions = (self.activation_rows, self.routed_rows, self.groups,
                      self.k_size, self.n_size)
        if any(int(value) < 0 for value in dimensions):
            raise ValueError("routed GEMM graph bucket dimensions must be non-negative")
        if self.groups > 4096:
            raise ValueError("routed GEMM graph buckets support at most 4096 groups")
        if self.gather and self.scatter:
            raise ValueError("one routed GEMM graph bucket cannot gather and scatter")

    @classmethod
    def from_tensors(cls, activation: torch.Tensor, weight: torch.Tensor, *,
                     routed_rows: int, gather: bool,
                     scatter: bool) -> "RoutedGemmGraphBucket":
        return cls(int(activation.shape[0]), int(routed_rows), int(weight.shape[0]),
                   int(weight.shape[1]), int(weight.shape[2]), bool(gather),
                   bool(scatter))

    def validate_call(self, activation: torch.Tensor, weight: torch.Tensor,
                      gather_rows: torch.Tensor,
                      scatter_rows: torch.Tensor) -> None:
        routed_rows = (int(gather_rows.shape[0]) if gather_rows.numel() else
                       int(scatter_rows.shape[0]) if scatter_rows.numel() else
                       int(activation.shape[0]))
        actual = self.from_tensors(
            activation, weight, routed_rows=routed_rows,
            gather=bool(gather_rows.numel()), scatter=bool(scatter_rows.numel()),
        )
        if actual != self:
            raise ValueError("routed GEMM call does not match its CUDA Graph bucket: "
                             f"expected {self}, observed {actual}")


def prewarm_routed_gemm_bucket(
    bucket: RoutedGemmGraphBucket, activation: torch.Tensor,
    weight: torch.Tensor, boundaries: torch.Tensor,
    gather_rows: torch.Tensor, scatter_rows: torch.Tensor, *,
    backward: bool = False,
) -> torch.Tensor:
    """Compile and allocate one fixed bucket before CUDA Graph capture."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("routed GEMM prewarm must run before CUDA Graph capture")
    if activation.device.type != "cuda":
        raise ValueError("routed GEMM CUDA Graph prewarm requires CUDA operands")
    bucket.validate_call(activation, weight, gather_rows, scatter_rows)
    from .routed_gemm_ops import routed_projection_op
    output = routed_projection_op(activation, weight, boundaries, gather_rows,
                                  scatter_rows, bucket.routed_rows)
    if backward:
        if not output.requires_grad:
            raise ValueError("backward prewarm requires a differentiable operand")
        torch.autograd.grad(output.sum(), (activation,), allow_unused=False)
    torch.cuda.synchronize(activation.device)
    return output


__all__ = ["RoutedGemmGraphBucket", "prewarm_routed_gemm_bucket"]
