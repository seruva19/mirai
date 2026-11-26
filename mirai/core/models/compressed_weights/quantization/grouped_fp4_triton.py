"""Grouped packed projections that decode only the current GEMM tile."""

from __future__ import annotations

import triton
import triton.language as tl
import torch

from .gguf_quant import GGUF_TYPE_SIZE, _table, gguf_num_superblocks


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
def _gguf_word(pointer, shift: tl.constexpr, valid):
    return tl.load(pointer + shift, valid, 0).to(tl.uint32)


@triton.jit
def _gguf_value(Codes, Grid, Signs, group, flat, valid, BLOCKS: tl.constexpr, GGUF_BYTES: tl.constexpr):
    # Byte layout and tables match the reference codecs in gguf_quant.py.
    block = Codes + (group * BLOCKS + flat // 256) * GGUF_BYTES
    index = flat % 256
    scale_bits = _gguf_word(block, 0, valid) | (_gguf_word(block, 1, valid) << 8)
    d = scale_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    if GGUF_BYTES == 74:
        code = block + 2 + 2 * (index // 8)
        selector = _gguf_word(code, 0, valid) | (_gguf_word(code, 1, valid) << 8)
        packed = _gguf_word(block + index // 32, 66, valid)
        scale = (packed >> (4 * ((index // 16) & 1))) & 15
        magnitude = tl.load(Grid + (selector & 511) * 8 + index % 8, valid, 0)
        sign = tl.load(Signs + (selector >> 9), valid, 0)
        value = d * (0.5 + scale.to(tl.float32)) * 0.25 * magnitude
        return tl.where(((sign >> (index % 8)) & 1) == 0, value, -value)
    if GGUF_BYTES == 98:
        packed = block + 66 + 4 * (index // 32)
        scales = (
            _gguf_word(packed, 0, valid) | (_gguf_word(packed, 1, valid) << 8)
            | (_gguf_word(packed, 2, valid) << 16) | (_gguf_word(packed, 3, valid) << 24)
        )
        selector = _gguf_word(block + index // 4, 2, valid)
        magnitude = tl.load(Grid + selector * 4 + index % 4, valid, 0)
        sign = tl.load(Signs + ((scales >> (7 * ((index // 8) & 3))) & 127), valid, 0)
        value = d * (0.5 + (scales >> 28).to(tl.float32)) * 0.5 * magnitude
        return tl.where(((sign >> (index % 8)) & 1) == 0, value, -value)
    high = _gguf_word(block, 2, valid) | (_gguf_word(block, 3, valid) << 8)
    low = _gguf_word(block + index // 64, 4, valid)
    subblock = index // 32
    scale = ((low >> (4 * (subblock & 1))) & 15) | (((high >> (2 * subblock)) & 3) << 4)
    code = _gguf_word(block + subblock * 16 + index % 16, 8, valid)
    nibble = (code >> (4 * ((index // 16) & 1))) & 15
    magnitude = tl.load(Grid + nibble, valid, 0)
    return d * (scale.to(tl.int32) - 32).to(tl.float32) * magnitude


@triton.jit
def _grouped_fp4(
    X, Codes, Scales, Global, Ends, Y,
    K: tl.constexpr, N: tl.constexpr, BLOCKS: tl.constexpr, SCALE_WIDTH: tl.constexpr,
    TRANSPOSED_STORAGE: tl.constexpr, MX: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GGUF_BYTES: tl.constexpr = 0,
):
    group = tl.program_id(0)
    begin = tl.load(Ends + group - 1, mask=group > 0, other=0)
    end = tl.load(Ends + group)
    row_start = begin + tl.program_id(1) * BM
    if row_start >= end:
        return
    rows = row_start + tl.arange(0, BM)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    if GGUF_BYTES == 0:
        group_scale = tl.load(Global + group)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        inputs = start * BK + tl.arange(0, BK)
        valid = (inputs[:, None] < K) & (cols[None, :] < N)
        if TRANSPOSED_STORAGE:
            flat = cols[None, :] * K + inputs[:, None]
        else:
            flat = inputs[:, None] * N + cols[None, :]
        if GGUF_BYTES:
            weights = _gguf_value(Codes, Scales, Global, group, flat, valid, BLOCKS, GGUF_BYTES)
        else:
            packed = tl.load(Codes + group * (BLOCKS * SCALE_WIDTH // 2) + flat // 2, valid, 0)
            nibble = (packed >> (4 * (flat & 1))) & 15
            scale = tl.load(Scales + group * BLOCKS + flat // SCALE_WIDTH, valid, 0)
            weights = _fp4_value(nibble) * _fp4_scale(scale, group_scale, MX)
        x = tl.load(X + rows[:, None] * K + inputs[None, :], (rows[:, None] < end) & (inputs[None, :] < K), 0)
        acc = tl.dot(x, weights.to(X.dtype.element_ty), acc)
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


def grouped_gguf_projection(x, codes, ends, *, columns, max_group_rows, transposed, format):
    """Apply IQ2_XS, IQ3_XXS or IQ4_XS blocks to sorted expert rows."""
    grids = {"gguf_iq2": "grid_iq2xs", "gguf_iq3": "grid", "gguf_iq4": "kvalues"}
    if format not in grids:
        raise ValueError("Grouped GGUF needs IQ2_XS, IQ3_XXS or IQ4_XS payloads.")
    if not x.is_cuda or x.ndim != 2 or x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("Grouped GGUF needs a CUDA FP16 or BF16 activation matrix.")
    groups = ends.numel()
    blocks = gguf_num_superblocks(x.shape[1] * columns)
    width = GGUF_TYPE_SIZE[format]
    if (
        codes.shape != (groups, blocks, width) or codes.dtype != torch.uint8
        or ends.ndim != 1 or ends.dtype not in {torch.int32, torch.int64}
        or codes.device != x.device or ends.device != x.device
    ):
        raise ValueError("Grouped GGUF payload layout does not match the projection.")
    result = x.new_empty((x.shape[0], columns))
    if not x.shape[0]:
        return result
    if min(groups, columns, x.shape[1], max_group_rows) <= 0:
        raise ValueError("Grouped GGUF projection dimensions must be positive.")
    table = _table(grids[format], x.device)
    # IQ4 carries signed magnitudes in the codebook itself.
    signs = table if format == "gguf_iq4" else _table("ksigns", x.device)
    grid = (groups, triton.cdiv(max_group_rows, 16), triton.cdiv(columns, 32))
    _grouped_fp4[grid](
        x.contiguous(), codes.contiguous(), table, signs, ends.contiguous(), result,
        x.shape[1], columns, blocks, 256, transposed, False, 16, 32, 32, width, num_warps=4, enable_fp_fusion=False,
    )
    return result
