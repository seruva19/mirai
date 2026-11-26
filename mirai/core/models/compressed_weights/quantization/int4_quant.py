"""Symmetric signed INT4 weights with one FP32 scale per 32 values."""

from __future__ import annotations

from math import prod

import torch

from .microscaling_quant import _pack_nibbles, _unpack_nibbles


INT4_BLOCK_SIZE = 32


def quantize_int4(weight):
    flat = weight.detach().float().reshape(-1)
    if not flat.numel() or not bool(torch.isfinite(flat).all()):
        raise ValueError("INT4 requires nonempty finite weights.")
    padded = torch.nn.functional.pad(flat, (0, (-flat.numel()) % INT4_BLOCK_SIZE))
    blocks = padded.view(-1, INT4_BLOCK_SIZE)
    scales = (blocks.abs().amax(dim=-1) / 7.0).clamp_min(torch.finfo(torch.float32).tiny)
    codes = (blocks / scales[:, None]).round().clamp(-7, 7).to(torch.int8)
    return _pack_nibbles(codes.to(torch.uint8)), scales


def validate_int4_layout(codes, scales, shape):
    blocks = (prod(shape) + INT4_BLOCK_SIZE - 1) // INT4_BLOCK_SIZE
    if (
        codes.ndim < 2 or codes.shape[-2:] != (blocks, INT4_BLOCK_SIZE // 2)
        or scales.shape != codes.shape[:-1] or codes.dtype != torch.uint8 or scales.dtype != torch.float32
    ):
        raise ValueError("INT4 codes and scales do not match the weight shape.")


def dequantize_int4(codes, scales, shape, *, dtype, device):
    validate_int4_layout(codes, scales, shape)
    nibbles = _unpack_nibbles(codes.to(device=device)).to(torch.int8)
    signed = torch.where(nibbles < 8, nibbles, nibbles - 16)
    decoded = signed.float() * scales.to(device=device).unsqueeze(-1)
    return decoded.flatten(-2)[..., :prod(shape)].unflatten(-1, tuple(shape)).to(dtype=dtype)
