"""K-packed MXFP4 block-scaled GEMM for consumer Blackwell."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mxfp4_mm(A, B, ScaleA, ScaleB, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    packed_lane = tl.arange(0, 64)
    scale_lane = tl.arange(0, 4)
    acc = tl.full((32, 32), 0, tl.float32)
    for tile in range(tl.cdiv(K, 128)):
        packed_k = tile * 64 + packed_lane
        scale_k = tile * 4 + scale_lane
        a = tl.load(A + rows[:, None] * (K // 2) + packed_k[None, :], (rows[:, None] < M) & (packed_k[None, :] < K // 2), 0)
        b = tl.load(B + cols[None, :] * (K // 2) + packed_k[:, None], (cols[None, :] < N) & (packed_k[:, None] < K // 2), 0)
        sa = tl.load(ScaleA + rows[:, None] * (K // 32) + scale_k[None, :], (rows[:, None] < M) & (scale_k[None, :] < K // 32), 127)
        sb = tl.load(ScaleB + cols[:, None] * (K // 32) + scale_k[None, :], (cols[:, None] < N) & (scale_k[None, :] < K // 32), 127)
        acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)
    tl.store(Out + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def native_mxfp4_matmul(a, b, sa, sb, *, rows: int, features: int, columns: int):
    """Multiply two row-major payloads, contracting their packed K dimension."""
    groups = columns // 32
    if (
        a.shape != (rows * groups, 16) or b.shape != (features * groups, 16)
        or sa.shape != (rows * groups,) or sb.shape != (features * groups,)
        or any(tensor.dtype != torch.uint8 or not tensor.is_contiguous() or tensor.device != a.device
               for tensor in (a, b, sa, sb))
    ):
        raise ValueError("Native MXFP4 payloads must match the row-aligned projection.")
    result = torch.empty((rows, features), device=a.device, dtype=torch.float32)
    _mxfp4_mm[(triton.cdiv(rows, 32), triton.cdiv(features, 32))](
        a, b, sa, sb, result, rows, features, columns, num_warps=4,
    )
    return result
