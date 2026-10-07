"""Lazy Triton INT8 GEMM used by the rotated W4A8 primitive."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


_PACKED_KERNEL = None


def _get_packed_kernel(triton, tl):
    global _PACKED_KERNEL
    if _PACKED_KERNEL is not None:
        return _PACKED_KERNEL

    @triton.jit(do_not_specialize=["M"])
    def kernel(a_ptr, packed_ptr, group_ptr, row_ptr, codebook_ptr, xscale_ptr,
               out_ptr, M, N: tl.constexpr, K: tl.constexpr,
               GROUP_SIZE: tl.constexpr, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + offs_k
            av = tl.load(a_ptr + offs_m[:, None] * K + ks[None, :],
                         mask=(offs_m[:, None] < M) & (ks[None, :] < K), other=0)
            packed = tl.load(
                packed_ptr + offs_n[None, :] * (K // 2) + (ks[:, None] // 2),
                mask=(offs_n[None, :] < N) & (ks[:, None] < K), other=0,
            )
            shift = (ks[:, None] & 1) * 4
            codes = (packed >> shift) & 15
            levels = tl.load(codebook_ptr + codes)
            scale_bits = tl.load(
                group_ptr + offs_n[None, :] * (K // GROUP_SIZE)
                + (ks[:, None] // GROUP_SIZE),
                mask=(offs_n[None, :] < N) & (ks[:, None] < K), other=0,
            )
            exponent = ((scale_bits >> 3) & 15).to(tl.float32)
            mantissa = (scale_bits & 7).to(tl.float32)
            # Positive E4M3FN decode in software keeps this kernel valid on SM80/86,
            # where hardware FP8 conversion is unavailable. Quantized scales are
            # positive and validated, so sign and the all-ones NaN code cannot occur.
            subnormal = mantissa * 0.001953125  # 2^-9
            normal = (1.0 + mantissa * 0.125) * tl.exp2(exponent - 7.0)
            relative = tl.where(exponent == 0.0, subnormal, normal)
            product = levels.to(tl.float32) * relative
            decoded = libdevice.rint(product).to(tl.int32)
            decoded = tl.maximum(-127, tl.minimum(127, decoded)).to(tl.int8)
            acc += tl.dot(av, decoded, out_dtype=tl.int32)
        xscale = tl.load(xscale_ptr + offs_m, mask=offs_m < M, other=0.0)
        row = tl.load(row_ptr + offs_n, mask=offs_n < N, other=0.0)
        result = acc.to(tl.float32) * xscale[:, None] * row[None, :]
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], result,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    _PACKED_KERNEL = kernel
    return kernel


def packed_w4a8_linear(
    a: torch.Tensor,
    packed: torch.Tensor,
    group_scales: torch.Tensor,
    row_scales: torch.Tensor,
    codebook: torch.Tensor,
    x_scales: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Decode packed W4 tiles inside the INT8 GEMM and scale its epilogue."""

    if a.ndim != 2 or a.dtype != torch.int8 or packed.ndim != 2:
        raise ValueError("packed W4A8 GEMM requires 2D int8 activations and packed weights")
    m, k = (int(v) for v in a.shape)
    n = int(packed.shape[0])
    if packed.dtype != torch.uint8 or tuple(packed.shape) != (n, k // 2):
        raise ValueError("packed W4A8 weight has incompatible shape or dtype")
    if not a.is_cuda or any(t.device != a.device for t in (packed, group_scales, row_scales, codebook, x_scales)):
        raise RuntimeError("packed W4A8 GEMM requires CUDA operands on one device")
    if group_scales.dtype != torch.uint8:
        raise ValueError("group scales must contain E4M3FN bytes")
    a = a.contiguous()
    output = torch.empty((m, n), device=a.device, dtype=out_dtype)
    kernel = _get_packed_kernel(triton, tl)
    grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))
    kernel[grid](
        a, packed, group_scales, row_scales, codebook, x_scales, output,
        M=m, N=n, K=k, GROUP_SIZE=16, BLOCK_M=32, BLOCK_N=32,
        BLOCK_K=32, num_warps=4,
    )
    return output


__all__ = ["packed_w4a8_linear"]
