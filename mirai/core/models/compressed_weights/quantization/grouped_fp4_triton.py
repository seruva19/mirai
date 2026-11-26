"""Grouped E2M1 projections that decode only the current GEMM tile."""

from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _fp4_value(nibble):
    exponent = (nibble >> 1) & 3
    mantissa = (nibble & 1).to(tl.float32)
    power = ((exponent.to(tl.uint32) + 126) << 23).to(tl.float32, bitcast=True)
    value = tl.where(exponent == 0, 0.5 * mantissa, (1.0 + 0.5 * mantissa) * power)
    return tl.where((nibble & 8) != 0, -value, value)


@triton.jit
def _fp4_scale(code, global_scale, MX: tl.constexpr):
    if MX:
        # Code zero represents the smallest E8M0 power of two, a subnormal in FP32.
        bits = tl.where(code == 0, 0x00400000, code.to(tl.int32) << 23)
        return bits.to(tl.float32, bitcast=True)
    exponent = ((code >> 3) & 15).to(tl.int32)
    mantissa = (code & 7).to(tl.float32)
    power = ((exponent + 120) << 23).to(tl.float32, bitcast=True)
    value = tl.where(exponent == 0, mantissa * 0.001953125, (1.0 + mantissa * 0.125) * power)
    return tl.where((code & 128) != 0, -value, value) * global_scale


@triton.jit
def _grouped_fp4(
    X, Codes, Scales, Global, Ends, Y,
    K: tl.constexpr, N: tl.constexpr, BLOCKS: tl.constexpr, SCALE_WIDTH: tl.constexpr,
    TRANSPOSED_STORAGE: tl.constexpr, MX: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    group = tl.program_id(0)
    begin = tl.load(Ends + group - 1, mask=group > 0, other=0)
    end = tl.load(Ends + group)
    row_start = begin + tl.program_id(1) * BM
    if row_start >= end:
        return
    rows = row_start + tl.arange(0, BM)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    group_scale = tl.load(Global + group)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        inputs = start * BK + tl.arange(0, BK)
        valid = (inputs[:, None] < K) & (cols[None, :] < N)
        if TRANSPOSED_STORAGE:
            flat = cols[None, :] * K + inputs[:, None]
        else:
            flat = inputs[:, None] * N + cols[None, :]
        packed = tl.load(Codes + group * (BLOCKS * SCALE_WIDTH // 2) + flat // 2, valid, 0)
        nibble = (packed >> (4 * (flat & 1))) & 15
        scale = tl.load(Scales + group * BLOCKS + flat // SCALE_WIDTH, valid, 0)
        weights = (_fp4_value(nibble) * _fp4_scale(scale, group_scale, MX)).to(X.dtype.element_ty)
        x = tl.load(X + rows[:, None] * K + inputs[None, :], (rows[:, None] < end) & (inputs[None, :] < K), 0)
        acc = tl.dot(x, weights, acc)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < end) & (cols[None, :] < N))


def grouped_fp4_projection(
    x, codes, scales, global_scales, ends, *, columns: int, max_group_rows: int,
    transposed_storage: bool, format: str,
):
    """Apply packed FP4 matrices to sorted rows delimited by cumulative group ends."""
    if format not in {"nvfp4", "mxfp4"}:
        raise ValueError("Grouped FP4 needs NVFP4 or MXFP4 payloads.")
    if not x.is_cuda or x.ndim != 2 or x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("Grouped FP4 needs a CUDA FP16 or BF16 activation matrix.")
    width = 32 if format == "mxfp4" else 16
    groups = ends.numel()
    blocks = triton.cdiv(x.shape[1] * columns, width)
    if (
        codes.shape != (groups, blocks, width // 2) or codes.dtype != torch.uint8
        or scales.shape != (groups, blocks) or scales.dtype != torch.uint8
        or global_scales.shape != (groups,) or global_scales.dtype != torch.float32
        or ends.ndim != 1 or ends.dtype not in {torch.int32, torch.int64}
        or any(tensor.device != x.device for tensor in (codes, scales, global_scales, ends))
    ):
        raise ValueError("Grouped FP4 payload layout does not match the projection.")
    result = x.new_empty((x.shape[0], columns))
    if x.shape[0] == 0:
        return result
    if min(groups, columns, x.shape[1], max_group_rows) <= 0:
        raise ValueError("Grouped FP4 projection dimensions must be positive.")
    grid = (groups, triton.cdiv(max_group_rows, 16), triton.cdiv(columns, 32))
    _grouped_fp4[grid](
        x.contiguous(), codes.contiguous(), scales.contiguous(), global_scales.contiguous(), ends.contiguous(), result,
        x.shape[1], columns, blocks, width, transposed_storage, format == "mxfp4", 16, 32, 32, num_warps=4,
    )
    return result
