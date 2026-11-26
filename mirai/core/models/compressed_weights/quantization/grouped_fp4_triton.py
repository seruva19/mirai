"""Grouped packed projections that decode only the current GEMM tile."""

from __future__ import annotations

import triton
import triton.language as tl
import torch

from .gguf_quant import GGUF_TYPE_SIZE, _table, gguf_num_superblocks
from .int4_quant import INT4_BLOCK_SIZE, validate_int4_layout


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
def _packed_tile(
    X, Codes, Scales, Global, Y, group, row_start, end, column_tile,
    K: tl.constexpr, N: tl.constexpr, BLOCKS: tl.constexpr, SCALE_WIDTH: tl.constexpr,
    TRANSPOSED_STORAGE: tl.constexpr, MX: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GGUF_BYTES: tl.constexpr, INT4: tl.constexpr,
):
    rows = row_start + tl.arange(0, BM)
    cols = column_tile * BN + tl.arange(0, BN)
    if GGUF_BYTES == 0 and not INT4:
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
            if INT4:
                signed = nibble.to(tl.int32)
                weights = tl.where(signed < 8, signed, signed - 16).to(tl.float32) * scale
            else:
                weights = _fp4_value(nibble) * _fp4_scale(scale, group_scale, MX)
        x = tl.load(X + rows[:, None] * K + inputs[None, :], (rows[:, None] < end) & (inputs[None, :] < K), 0)
        acc = tl.dot(x, weights.to(X.dtype.element_ty), acc)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < end) & (cols[None, :] < N))


@triton.jit
def _grouped_fp4(
    X, Codes, Scales, Global, Ends, Y,
    K: tl.constexpr, N: tl.constexpr, BLOCKS: tl.constexpr, SCALE_WIDTH: tl.constexpr,
    TRANSPOSED_STORAGE: tl.constexpr, MX: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GGUF_BYTES: tl.constexpr = 0,
    INT4: tl.constexpr = False,
):
    group = tl.program_id(0)
    begin = tl.load(Ends + group - 1, mask=group > 0, other=0)
    end = tl.load(Ends + group)
    row_start = begin + tl.program_id(1) * BM
    if row_start < end:
        _packed_tile(
            X, Codes, Scales, Global, Y, group, row_start, end, tl.program_id(2),
            K, N, BLOCKS, SCALE_WIDTH, TRANSPOSED_STORAGE, MX, BM, BN, BK, GGUF_BYTES, INT4,
        )


@triton.jit
def _persistent_grouped_packed(
    X, Codes, Scales, Global, Ends, TileEnds, Y,
    GROUPS: tl.constexpr, K: tl.constexpr, N: tl.constexpr, BLOCKS: tl.constexpr, SCALE_WIDTH: tl.constexpr,
    TRANSPOSED_STORAGE: tl.constexpr, MX: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GGUF_BYTES: tl.constexpr, INT4: tl.constexpr,
):
    column_tiles: tl.constexpr = tl.cdiv(N, BN)
    tile_count = tl.load(TileEnds + GROUPS - 1)
    for task in range(tl.program_id(0), tile_count * column_tiles, tl.num_programs(0)):
        row_tile = task // column_tiles
        low = 0
        high = GROUPS - 1
        while low < high:
            middle = (low + high) // 2
            if tl.load(TileEnds + middle) <= row_tile:
                low = middle + 1
            else:
                high = middle
        begin = tl.load(Ends + low - 1, mask=low > 0, other=0)
        end = tl.load(Ends + low)
        tile_begin = tl.load(TileEnds + low - 1, mask=low > 0, other=0)
        row_start = begin + (row_tile - tile_begin) * BM
        _packed_tile(
            X, Codes, Scales, Global, Y, low, row_start, end, task % column_tiles,
            K, N, BLOCKS, SCALE_WIDTH, TRANSPOSED_STORAGE, MX, BM, BN, BK, GGUF_BYTES, INT4,
        )


def _launch_packed_projection(
    x, codes, scales, global_scales, ends, result, *, columns, blocks, width,
    transposed, mx=False, gguf_bytes=0, int4=False, max_group_rows, persistent,
):
    groups = ends.numel()
    args = (x.contiguous(), codes.contiguous(), scales.contiguous(), global_scales.contiguous(), ends.contiguous())
    dimensions = (x.shape[1], columns, blocks, width, transposed, mx, 16, 32, 32, gguf_bytes, int4)
    if persistent:
        rows = ends.diff(prepend=ends.new_zeros(1))
        tile_ends = torch.cumsum((rows + 15) // 16, dim=0, dtype=torch.int32)
        programs = min(x.shape[0] * triton.cdiv(columns, 32), torch.cuda.get_device_properties(x.device).multi_processor_count)
        _persistent_grouped_packed[(programs,)](
            *args, tile_ends, result, groups, *dimensions, num_warps=4, enable_fp_fusion=not gguf_bytes,
        )
    else:
        grid = (groups, triton.cdiv(max_group_rows, 16), triton.cdiv(columns, 32))
        _grouped_fp4[grid](*args, result, *dimensions, num_warps=4, enable_fp_fusion=not gguf_bytes)


def grouped_fp4_projection(
    x, codes, scales, global_scales, ends, *, columns: int, max_group_rows: int,
    transposed_storage: bool, format: str, persistent: bool = False,
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
    if min(groups, columns, x.shape[1]) <= 0 or (not persistent and max_group_rows <= 0):
        raise ValueError("Grouped FP4 projection dimensions must be positive.")
    _launch_packed_projection(
        x, codes, scales, global_scales, ends, result, columns=columns, blocks=blocks, width=width,
        transposed=transposed_storage, mx=format == "mxfp4", max_group_rows=max_group_rows, persistent=persistent,
    )
    return result


def grouped_gguf_projection(x, codes, ends, *, columns, max_group_rows, transposed, format, persistent=False):
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
    if min(groups, columns, x.shape[1]) <= 0 or (not persistent and max_group_rows <= 0):
        raise ValueError("Grouped GGUF projection dimensions must be positive.")
    table = _table(grids[format], x.device)
    # IQ4 carries signed magnitudes in the codebook itself.
    signs = table if format == "gguf_iq4" else _table("ksigns", x.device)
    _launch_packed_projection(
        x, codes, table, signs, ends, result, columns=columns, blocks=blocks, width=256, transposed=transposed,
        gguf_bytes=width, max_group_rows=max_group_rows, persistent=persistent,
    )
    return result


def grouped_int4_projection(x, codes, scales, ends, *, columns, max_group_rows, transposed, persistent=False):
    """Apply signed block-scaled INT4 to sorted expert rows."""
    if not x.is_cuda or x.ndim != 2 or x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("Grouped INT4 needs a CUDA FP16 or BF16 activation matrix.")
    validate_int4_layout(codes, scales, (x.shape[1], columns))
    groups = ends.numel()
    if (
        codes.ndim != 3 or codes.shape[0] != groups
        or ends.ndim != 1 or ends.dtype not in {torch.int32, torch.int64}
        or any(tensor.device != x.device for tensor in (codes, scales, ends))
    ):
        raise ValueError("Grouped INT4 payload layout does not match the projection.")
    result = x.new_empty((x.shape[0], columns))
    if not x.shape[0]:
        return result
    if min(groups, columns, x.shape[1]) <= 0 or (not persistent and max_group_rows <= 0):
        raise ValueError("Grouped INT4 projection dimensions must be positive.")
    _launch_packed_projection(
        x, codes, scales, scales, ends, result, columns=columns, blocks=codes.shape[1],
        width=INT4_BLOCK_SIZE, transposed=transposed, int4=True, max_group_rows=max_group_rows, persistent=persistent,
    )
    return result
