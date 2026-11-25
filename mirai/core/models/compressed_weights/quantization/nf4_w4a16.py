"""Fused Triton W4A16 multiplication for double-quantized NF4 weights."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .quant import _Nf4Meta, _nf4_dequantize


@triton.jit
def _nf4_w4a16(
    X, Packed, Absmax, NestedAbsmax, Offset, Code, NestedCode, Out,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK: tl.constexpr, NESTED_BLOCK: tl.constexpr,
    NESTED_DTYPE: tl.constexpr, WEIGHT_DTYPE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k_lane = tl.arange(0, BK)
    offset = tl.load(Offset).to(tl.float32)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        k = tile * BK + k_lane
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        position = n[None, :] * K + k[:, None]
        valid = (n[None, :] < N) & (k[:, None] < K)
        byte = tl.load(Packed + position // 2, valid, 0).to(tl.int32)
        # bitsandbytes stores the even index in the high nibble.
        # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/kernels.cu
        nibble = tl.where(position % 2 == 0, byte >> 4, byte & 15)
        block = position // BLOCK
        scale_code = tl.load(Absmax + block, valid, 0).to(tl.int32)
        nested = tl.load(NestedAbsmax + block // NESTED_BLOCK, valid, 0).to(tl.float32)
        scale = (tl.load(NestedCode + scale_code).to(tl.float32) * nested).to(NESTED_DTYPE)
        scale = (scale.to(tl.float32) + offset).to(NESTED_DTYPE).to(tl.float32)
        weight = (tl.load(Code + nibble).to(tl.float32) * scale).to(WEIGHT_DTYPE).to(x.dtype)
        acc = tl.dot(x, tl.where(valid, weight, 0).to(x.dtype), acc)
    tl.store(Out + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def _launch(x, packed, absmax, nested_absmax, offset, code, nested_code, meta, shape):
    formats = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}
    n, k = shape
    x = x.contiguous().reshape(-1, k)
    m = x.shape[0]
    out = x.new_empty((m, n))
    if not m:
        return out
    device = x.device
    fields = tuple(tensor.to(device).contiguous() for tensor in (
        packed, absmax, nested_absmax, offset, code, nested_code,
    ))
    _nf4_w4a16[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
        x, *fields, out, m, n, k, meta.blocksize, meta.nested_blocksize,
        formats[meta.nested_dtype], formats[meta.weight_dtype], 16, 32, 32,
    )
    return out


class _FusedNF4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, absmax, nested_absmax, offset, code, nested_code, meta, shape):
        ctx.save_for_backward(packed, absmax, nested_absmax, offset, code, nested_code)
        ctx.meta = meta
        ctx.shape = shape
        ctx.dtype = x.dtype
        return _launch(
            x, packed, absmax, nested_absmax, offset, code, nested_code, meta, shape,
        ).reshape(*x.shape[:-1], shape[0])

    @staticmethod
    def backward(ctx, grad):
        packed, absmax, nested_absmax, offset, code, nested_code = ctx.saved_tensors
        weight = _nf4_dequantize(
            dict(packed=packed, absmax=absmax, nested_absmax=nested_absmax, offset=offset),
            dict(code=code, nested_code=nested_code),
            ctx.meta, shape=ctx.shape, dtype=ctx.dtype, device=grad.device,
        )
        return (grad.float() @ weight.float()).to(ctx.dtype), None, None, None, None, None, None, None, None


def fused_nf4_linear(
    x: torch.Tensor, *, packed: torch.Tensor, absmax: torch.Tensor,
    nested_absmax: torch.Tensor, offset: torch.Tensor, code: torch.Tensor,
    nested_code: torch.Tensor, meta: _Nf4Meta, shape: tuple[int, int],
) -> torch.Tensor:
    from mirai.core.moe.runtime.gemm import probe_backend

    status = probe_backend("triton_nf4_w4a16", device=x.device)
    if not status.available:
        raise RuntimeError(status.reason)
    n, k = shape
    if n <= 0 or k <= 0 or x.shape[-1] != k or x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("Fused NF4 W4A16 requires positive matrix dimensions and matching FP16/BF16 inputs.")
    if meta.blocksize <= 0 or meta.nested_blocksize <= 0:
        raise ValueError("Fused NF4 W4A16 requires positive scale block sizes.")
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if meta.nested_dtype not in supported_dtypes or meta.weight_dtype not in supported_dtypes:
        raise ValueError("Unsupported NF4 scale or weight dtype.")
    blocks = triton.cdiv(n * k, meta.blocksize)
    if (
        packed.dtype != torch.uint8 or packed.numel() != triton.cdiv(n * k, 2)
        or absmax.dtype != torch.uint8 or absmax.numel() != blocks
        or nested_absmax.numel() != triton.cdiv(blocks, meta.nested_blocksize)
        or offset.numel() != 1 or code.numel() != 16 or nested_code.numel() != 256
    ):
        raise ValueError("Invalid double-quantized NF4 payload for fused W4A16.")
    return _FusedNF4Linear.apply(
        x, packed, absmax, nested_absmax, offset, code, nested_code, meta, shape,
    )
