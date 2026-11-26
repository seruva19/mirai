"""Decode block-scaled E4M3 weights into a caller-owned CUDA buffer."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _decode_fp8_out(Codes, Scales, Out, SIZE: tl.constexpr, M: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    pos = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = pos < SIZE
    code = tl.load(Codes + pos, valid, 0).to(tl.int32)
    tile_cols: tl.constexpr = tl.cdiv(N, 128)
    tile_rows: tl.constexpr = tl.cdiv(M, 128)
    scale_index = pos // (M * N) * (tile_rows * tile_cols) + (pos // N % M) // 128 * tile_cols + (pos % N) // 128
    scale = tl.load(Scales + scale_index, valid, 0)
    exponent = (code >> 3) & 15
    mantissa = (code & 7).to(tl.float32)
    power = ((exponent + 120) << 23).to(tl.float32, bitcast=True)
    magnitude = tl.where(exponent == 0, mantissa * 0.001953125, (1.0 + mantissa * 0.125) * power)
    magnitude = tl.where((code & 127) == 127, float("nan"), magnitude)
    decoded = tl.where((code & 128) != 0, -magnitude, magnitude)
    tl.store(Out + pos, decoded * scale, valid)


def decode_fp8_out(codes, scales, out):
    if out.numel():
        _decode_fp8_out[(triton.cdiv(out.numel(), 1024),)](
            codes, scales, out, out.numel(), out.shape[-2], out.shape[-1], 1024,
        )
