"""Portable rotated W4A8 storage and exact INT8-arithmetic reference.

The 4-bit coding follows the Apache-2.0 Comfy Kitchen W4A8 format: even
columns occupy the low nibble, a shared 16-entry Gaussian Lloyd-Max codebook
maps nibbles to levels, group scales map those levels onto the INT8 grid, and
one floating scale is stored per output row.  This module is an independent
native PyTorch implementation; it has no ComfyUI or Comfy Kitchen dependency.

Mirai rotation is deliberately different from Comfy ConvRot.  Stored weights
use ``quant._rotate_last_dim`` and its normalized fixed Hadamard, in groups
selected by Mirai.  A rotation group of zero means that no rotation is used.

Algorithm provenance:
  Copyright (c) 2025 Comfy Org. All rights reserved. Apache-2.0.
  https://github.com/Comfy-Org/comfy-kitchen/blob/be003b7c23c5b01328657955b8bc5d3f073d868e/comfy_kitchen/backends/eager/w4a8_int8.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .quant import _is_power_of_four, _quantization_workspace_bytes, _rotate_last_dim


_CODEBOOK = (
    -0.980602, -0.794529, -0.638165, -0.500986,
    -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533,
    0.502636, 0.638953, 0.794876, 0.980671,
)
_SCHEME = "mirai_rotated_w4a8_int8"


@dataclass(frozen=True)
class W4A8Metadata:
    """Versioned, tensor-serialization-friendly W4A8 layout description."""

    version: int
    shape: tuple[int, ...]
    group_size: int = 16
    rotation_group_size: int = 0
    scheme: str = _SCHEME
    scale_format: str = "float8_e4m3fn_bits"


@dataclass(frozen=True)
class W4A8Weight:
    """Packed frozen weight plus the tensors needed to decode its INT8 grid."""

    packed: torch.Tensor
    group_scales: torch.Tensor
    row_scales: torch.Tensor
    codebook: torch.Tensor
    metadata: W4A8Metadata

    def validate(self) -> None:
        _validate_weight(self)


def _validate_weight(value: W4A8Weight, *, numeric: bool = True) -> tuple[int, int, int]:
    meta = value.metadata
    if meta.version != 1 or meta.scheme != _SCHEME:
        raise ValueError(
            f"unsupported W4A8 metadata version/scheme: {meta.version!r}/{meta.scheme!r}"
        )
    if meta.scale_format != "float8_e4m3fn_bits":
        raise ValueError(f"unsupported W4A8 scale format: {meta.scale_format!r}")
    if len(meta.shape) not in (2, 3) or any(int(v) <= 0 for v in meta.shape):
        raise ValueError(f"W4A8 shape must be [N,K] or [E,N,K], got {meta.shape}")
    *batch, n, k = (int(v) for v in meta.shape)
    if meta.group_size != 16 or k % 16:
        raise ValueError(f"W4A8 requires group_size=16 and K divisible by 16, got K={k}")
    rg = int(meta.rotation_group_size)
    if rg < 0 or (rg and (k % rg or not _is_power_of_four(rg))):
        raise ValueError(f"rotation_group_size={rg} must be zero or a power of four that divides K={k}")
    if k > ((1 << 31) - 1) // (127 * 127):
        raise ValueError(f"K={k} can overflow an INT32 dot product")
    expected_packed = (*batch, n, k // 2)
    expected_groups = (*batch, n, k // 16)
    expected_rows = (*batch, n, 1)
    if value.packed.dtype != torch.uint8 or tuple(value.packed.shape) != expected_packed:
        raise ValueError(f"packed must be uint8 with shape {expected_packed}")
    if value.group_scales.dtype != torch.uint8 or tuple(value.group_scales.shape) != expected_groups:
        raise ValueError(f"group_scales must be float8_e4m3fn uint8 bits with shape {expected_groups}")
    if value.row_scales.dtype != torch.float32 or tuple(value.row_scales.shape) != expected_rows:
        raise ValueError(f"row_scales must be float32 with shape {expected_rows}")
    if value.codebook.dtype != torch.float32 or tuple(value.codebook.shape) != (16,):
        raise ValueError("codebook must be float32 with shape (16,)")
    tensors = (value.packed, value.group_scales, value.row_scales, value.codebook)
    if any(t.device != value.packed.device for t in tensors[1:]):
        raise ValueError("all W4A8 tensors must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all W4A8 storage tensors must be contiguous")
    if any(t.requires_grad for t in tensors):
        raise ValueError("W4A8 storage tensors must be frozen")
    if not numeric:
        return n, k, len(batch)
    decoded_scales = _decode_group_scales(value.group_scales)
    if not bool(torch.isfinite(decoded_scales).all().item()) or not all(bool(torch.isfinite(t).all().item()) for t in tensors[2:]):
        raise ValueError("W4A8 scales and codebook must be finite")
    if bool((decoded_scales <= 0).any().item()) or bool((value.row_scales <= 0).any().item()):
        raise ValueError("W4A8 scales must be positive")
    if not bool((value.codebook[1:] > value.codebook[:-1]).all().item()):
        raise ValueError("W4A8 codebook must be strictly increasing")
    return n, k, len(batch)


def _rotate(value: torch.Tensor, group_size: int, *, inverse: bool) -> torch.Tensor:
    if group_size == 0:
        return value
    shape = value.shape
    return _rotate_last_dim(
        value.reshape(-1, shape[-1]).float(), group_size, inverse=inverse
    ).reshape(shape)


def rotate_w4a8_activations(x: torch.Tensor, rotation_group_size: int) -> torch.Tensor:
    """Rotate activations into the stored-weight basis using Mirai Hadamard."""

    return _rotate(x, int(rotation_group_size), inverse=False).to(x.dtype)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    return ((low & 15) | ((high & 15) << 4)).to(torch.uint8).contiguous()


def _encode_group_scales(scales: torch.Tensor) -> torch.Tensor:
    return scales.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def _decode_group_scales(bits: torch.Tensor) -> torch.Tensor:
    return bits.view(torch.float8_e4m3fn).float()


def _unpack_codes(packed: torch.Tensor, k: int) -> torch.Tensor:
    codes = torch.empty(*packed.shape[:-1], k, dtype=torch.long, device=packed.device)
    raw = packed.to(torch.long)
    codes[..., 0::2] = raw & 15
    codes[..., 1::2] = (raw >> 4) & 15
    return codes


def _nearest_codes(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    midpoints = (levels[:-1] + levels[1:]) * 0.5
    return torch.bucketize(values.contiguous(), midpoints)


@torch.no_grad()
def quantize_w4a8(
    weight: torch.Tensor,
    *,
    rotation_group_size: int = 0,
) -> W4A8Weight:
    """Rotate and quantize a 2D or expert-batched floating weight.

    Group scales are serialized as raw E4M3FN bytes.  Together with the packed
    nibbles this costs 4.5 bits/weight at group size 16, plus row-scale and
    shared-codebook metadata.
    """

    if weight.ndim not in (2, 3) or not weight.is_floating_point():
        raise ValueError("weight must be a floating [N,K] or [E,N,K] tensor")
    k = int(weight.shape[-1])
    rg = int(rotation_group_size)
    if k % 16 or rg < 0 or (rg and (k % rg or not _is_power_of_four(rg))):
        raise ValueError(
            f"K={k} must be divisible by 16 and rotation_group_size={rg} must be zero or a power of four that divides K"
        )
    codebook = torch.tensor(_CODEBOOK, device=weight.device, dtype=torch.float32)
    flat = weight.detach().reshape(-1, k)
    rows = int(flat.shape[0])
    row_block = max(1, _quantization_workspace_bytes() // max(k * 16, 1))
    packed = torch.empty((rows, k // 2), device=weight.device, dtype=torch.uint8)
    encoded_relative = torch.empty((rows, k // 16), device=weight.device, dtype=torch.uint8)
    row_scales = torch.empty((rows, 1), device=weight.device, dtype=torch.float32)
    max_rel = torch.finfo(torch.float8_e4m3fn).max
    min_rel = torch.finfo(torch.float8_e4m3fn).tiny
    for start in range(0, rows, row_block):
        stop = min(rows, start + row_block)
        source = flat[start:stop]
        if not bool(torch.isfinite(source).all().item()):
            raise ValueError("weight must be finite")
        rotated = _rotate(source, rg, inverse=False).float()
        grouped = rotated.reshape(stop - start, k // 16, 16)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        codes = _nearest_codes(grouped / scale, codebook)
        for _ in range(2):
            levels = codebook[codes]
            scale = ((grouped * levels).sum(-1, keepdim=True) /
                     levels.square().sum(-1, keepdim=True).clamp_min(1e-8)).clamp_min(1e-8)
            codes = _nearest_codes(grouped / scale, codebook)
        row = (codebook[codes] * scale).abs().amax(dim=(-1, -2)).div(127).clamp_min(1e-8).unsqueeze(-1)
        relative = (scale.squeeze(-1) / row).clamp(min=min_rel, max=max_rel)
        encoded = _encode_group_scales(relative)
        int8_levels = (codebook.view(1, 1, 16) * _decode_group_scales(encoded).unsqueeze(-1)).round().clamp(-127, 127)
        codes = torch.searchsorted(int8_levels.contiguous(), (grouped / row.unsqueeze(-1)).contiguous())
        codes = codes.clamp(0, 15)
        lo = (codes - 1).clamp(0, 15)
        hi = codes
        dlo = (grouped / row.unsqueeze(-1) - torch.gather(int8_levels, -1, lo)).abs()
        dhi = (grouped / row.unsqueeze(-1) - torch.gather(int8_levels, -1, hi)).abs()
        codes = torch.where(dhi < dlo, hi, lo)
        packed[start:stop] = _pack_codes(codes.reshape(stop - start, k))
        encoded_relative[start:stop] = encoded
        row_scales[start:stop] = row
    result = W4A8Weight(
        packed=packed.reshape(*weight.shape[:-1], k // 2),
        group_scales=encoded_relative.reshape(*weight.shape[:-1], k // 16),
        row_scales=row_scales.reshape(*weight.shape[:-1], 1),
        codebook=codebook,
        metadata=W4A8Metadata(
            version=1,
            shape=tuple(int(v) for v in weight.shape),
            rotation_group_size=rg,
        ),
    )
    result.validate()
    return result


def unpack_w4a8_to_int8(weight: W4A8Weight, *, _numeric_validation: bool = True) -> torch.Tensor:
    """Decode packed nibbles to the rotated INT8 weight consumed by GEMM."""

    _n, k, _ = _validate_weight(weight, numeric=_numeric_validation)
    codes = _unpack_codes(weight.packed, k)
    values = weight.codebook[codes].reshape(
        *weight.metadata.shape[:-1], k // 16, 16
    )
    return (values * _decode_group_scales(weight.group_scales).unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8).reshape(weight.metadata.shape)


def dequantize_w4a8(
    weight: W4A8Weight,
    *,
    dtype: torch.dtype = torch.float32,
    _numeric_validation: bool = True,
) -> torch.Tensor:
    """Decode storage and undo rotation, returning the original-basis weight."""

    decoded = unpack_w4a8_to_int8(
        weight, _numeric_validation=_numeric_validation
    ).float() * weight.row_scales
    return _rotate(decoded, weight.metadata.rotation_group_size, inverse=True).to(dtype)


def _quantize_activations(x_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x_rot.float().abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
    q = (x_rot.float() / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def _reference_int8_mm(a: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    """Exact integer dot product for K bounded to fit one INT32 accumulation."""

    # int32 matmul is not implemented by every torch backend. Float64 exactly
    # represents these integer products and sums for all practical model K.
    return torch.matmul(a.to(torch.float64), b_t.to(torch.float64)).to(torch.int32)


class _W4A8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, group_scales, row_scales, codebook, metadata, backend):
        stored = W4A8Weight(packed, group_scales, row_scales, codebook, metadata)
        _n, k, expert_dims = _validate_weight(stored, numeric=False)
        if int(x.shape[-1]) != k:
            raise ValueError(f"input K={x.shape[-1]} does not match weight K={k}")
        x_rot = _rotate(x, metadata.rotation_group_size, inverse=False)
        x_q, x_scale = _quantize_activations(x_rot)
        w_q = None if backend == "triton" else unpack_w4a8_to_int8(stored, _numeric_validation=False)
        if expert_dims == 0:
            flat = x_q.reshape(-1, k)
            if backend == "reference":
                assert w_q is not None
                accum = _reference_int8_mm(flat, w_q.transpose(-2, -1))
                output = accum.reshape(*x.shape[:-1], w_q.shape[-2]).float()
                output = output * x_scale.unsqueeze(-1) * row_scales.squeeze(-1).unsqueeze(-2)
            else:
                output = _triton_packed_linear(
                    flat, packed, group_scales, row_scales, codebook,
                    x_scale.reshape(-1), x.dtype,
                ).reshape(*x.shape[:-1], metadata.shape[-2])
        else:
            if x.ndim < 3 or x.shape[0] != metadata.shape[0]:
                raise ValueError("expert-batched weight requires input [E,...,K] with matching E")
            flat = x_q.reshape(x.shape[0], -1, k)
            parts = []
            for expert in range(int(x.shape[0])):
                if backend == "reference":
                    assert w_q is not None
                    parts.append(_reference_int8_mm(flat[expert], w_q[expert].transpose(-2, -1)).float()
                                 * x_scale[expert].reshape(-1, 1)
                                 * row_scales[expert].squeeze(-1).unsqueeze(0))
                else:
                    parts.append(_triton_packed_linear(
                        flat[expert], packed[expert], group_scales[expert],
                        row_scales[expert], codebook, x_scale[expert].reshape(-1), x.dtype,
                    ))
            output = torch.stack(parts).reshape(*x.shape[:-1], metadata.shape[-2])
        ctx.save_for_backward(packed, group_scales, row_scales, codebook)
        ctx.layout = metadata
        ctx.input_dtype = x.dtype
        return output.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        packed, group_scales, row_scales, codebook = ctx.saved_tensors
        stored = W4A8Weight(packed, group_scales, row_scales, codebook, ctx.layout)
        w_rot = unpack_w4a8_to_int8(stored, _numeric_validation=False).float() * row_scales
        if w_rot.ndim == 2:
            grad_rot = torch.matmul(grad_output.float(), w_rot)
        else:
            grad_rot = torch.bmm(
                grad_output.reshape(w_rot.shape[0], -1, w_rot.shape[-2]).float(), w_rot
            ).reshape(*grad_output.shape[:-1], w_rot.shape[-1])
        grad = _rotate(grad_rot, ctx.layout.rotation_group_size, inverse=True)
        return grad.to(ctx.input_dtype), None, None, None, None, None, None


def _triton_packed_linear(a, packed, group_scales, row_scales, codebook, x_scales, out_dtype):
    if not a.is_cuda:
        raise RuntimeError("W4A8 Triton backend requires CUDA tensors")
    if torch.cuda.get_device_capability(a.device) < (8, 0):
        raise RuntimeError("W4A8 Triton backend requires CUDA SM80 or newer")
    try:
        from .w4a8_triton import packed_w4a8_linear
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("W4A8 Triton backend is unavailable") from exc
    return packed_w4a8_linear(
        a, packed, group_scales, row_scales, codebook, x_scales,
        out_dtype=out_dtype,
    )


def w4a8_linear_reference(x: torch.Tensor, weight: W4A8Weight) -> torch.Tensor:
    """Portable A8 x W4 reference with exact integer dot products."""

    return _W4A8Linear.apply(
        x, weight.packed, weight.group_scales, weight.row_scales,
        weight.codebook, weight.metadata, "reference",
    )


def w4a8_linear(
    x: torch.Tensor,
    weight: W4A8Weight,
    *,
    backend: Literal["reference", "triton"] = "reference",
) -> torch.Tensor:
    """Run the explicit reference or accelerated W4A8 implementation."""

    if backend not in ("reference", "triton"):
        raise ValueError(f"unknown W4A8 backend: {backend!r}")
    return _W4A8Linear.apply(
        x, weight.packed, weight.group_scales, weight.row_scales,
        weight.codebook, weight.metadata, backend,
    )


__all__ = [
    "W4A8Metadata", "W4A8Weight", "dequantize_w4a8", "quantize_w4a8",
    "rotate_w4a8_activations", "unpack_w4a8_to_int8", "w4a8_linear",
    "w4a8_linear_reference",
]
