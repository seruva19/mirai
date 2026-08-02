"""Reference MXFP8, packed MXFP4, and NVFP4 frozen-weight quantization.

MXFP8 follows Mishra et al.'s E4M3/UE8M0 conversion recipe: 32-value blocks,
round-up power-of-two scales, saturating conversion, and round-to-nearest-even
element encoding (arXiv:2506.08027). MXFP4 follows the OCP MX E2M1/E8M0
representation, while NVFP4 uses NVIDIA's E2M1/E4M3/FP32 hierarchical scaling
contract. This is deliberately a portable reference path; hardware-specific
GEMM dispatch belongs to a backend module and must prove parity against this
owner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


MICROSCALING_FORMATS = ("mxfp8_e4m3", "mxfp4", "nvfp4")
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E4M3_MAX = 448.0


@dataclass(frozen=True)
class MicroscalingMeta:
    format: str
    block_size: int
    shape: tuple[int, ...]
    padding: int


def normalize_microscaling_format(value: str) -> str:
    fmt = str(value).strip().lower()
    if fmt not in MICROSCALING_FORMATS:
        raise ValueError(
            "microscaling format must be one of: "
            + ", ".join(MICROSCALING_FORMATS)
            + "."
        )
    return fmt


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("microscaling quantization requires torch.")


def _round_e2m1(values: "torch.Tensor") -> "torch.Tensor":
    """Encode non-negative normalized values with round-to-nearest-even ties."""
    levels = torch.tensor(_E2M1_VALUES, device=values.device, dtype=torch.float32)
    distances = (values.float().unsqueeze(-1) - levels).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    candidates = distances == minimum
    indices = torch.arange(8, device=values.device)
    even = (indices & 1) == 0
    preferred = candidates & even
    has_even = preferred.any(dim=-1, keepdim=True)
    return torch.where(has_even, preferred, candidates).to(torch.int8).argmax(dim=-1)


def _encode_e2m1(values: "torch.Tensor") -> "torch.Tensor":
    magnitude = _round_e2m1(values.abs().clamp(max=6.0))
    sign = torch.signbit(values).to(torch.uint8) << 3
    return magnitude.to(torch.uint8) | sign


def _decode_e2m1(codes: "torch.Tensor") -> "torch.Tensor":
    levels = torch.tensor(_E2M1_VALUES, device=codes.device, dtype=torch.float32)
    magnitude = levels[(codes & 0x07).long()]
    return torch.where((codes & 0x08) != 0, -magnitude, magnitude)


def _pack_nibbles(codes: "torch.Tensor") -> "torch.Tensor":
    if codes.shape[-1] % 2:
        codes = torch.nn.functional.pad(codes, (0, 1))
    return ((codes[..., 0::2] & 0x0F) | ((codes[..., 1::2] & 0x0F) << 4)).contiguous()


def _unpack_nibbles(packed: "torch.Tensor") -> "torch.Tensor":
    result = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    result[..., 0::2] = packed & 0x0F
    result[..., 1::2] = packed >> 4
    return result


def _encode_e4m3(value: "torch.Tensor") -> "torch.Tensor":
    """Encode finite values as saturating E4M3FN with RN-even rounding."""
    source = value.float()
    magnitude = source.abs().clamp(max=_E4M3_MAX)
    sign = torch.signbit(source).to(torch.uint8) << 7

    # E4M3FN subnormals have a fixed 2^-9 quantum. ``torch.round`` is
    # round-to-nearest-even, including the transition to the first normal.
    subnormal_code = torch.round(magnitude * (2.0**9)).clamp(0, 8).to(torch.int32)

    safe = magnitude.clamp(min=2.0**-6)
    exponent = torch.floor(torch.log2(safe)).to(torch.int32).clamp(-6, 8)
    mantissa = torch.round((safe / torch.pow(2.0, exponent.float()) - 1.0) * 8.0)
    carry = mantissa >= 8.0
    exponent = exponent + carry.to(torch.int32)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    normal_code = ((exponent + 7) << 3) | mantissa.to(torch.int32)
    # 0x7f is NaN in E4M3FN; finite saturation stops at 0x7e (448).
    normal_code = normal_code.clamp(0, 0x7E)
    positive = torch.where(magnitude < 2.0**-6, subnormal_code, normal_code)
    return positive.to(torch.uint8) | sign


def _decode_e4m3(codes: "torch.Tensor") -> "torch.Tensor":
    positive = codes & 0x7F
    exponent = ((positive >> 3) & 0x0F).long()
    mantissa = (positive & 0x07).float()
    decoded = torch.where(
        exponent == 0,
        mantissa * (2.0 ** -9),
        (1.0 + mantissa / 8.0) * torch.pow(2.0, exponent.float() - 7.0),
    )
    decoded = torch.where(positive == 0x7F, torch.nan, decoded)
    return torch.where((codes & 0x80) != 0, -decoded, decoded)


def _encode_e8m0(scale: "torch.Tensor") -> "torch.Tensor":
    # Decode scale rounds upward so the block maximum remains representable.
    exponent = torch.ceil(torch.log2(scale.float().clamp(min=2.0 ** -127)))
    return (exponent.clamp(-127, 127) + 127).to(torch.uint8)


def _decode_e8m0(codes: "torch.Tensor") -> "torch.Tensor":
    return torch.pow(2.0, codes.float() - 127.0)


def validate_microscaling_payload(
    packed: "torch.Tensor",
    scales: "torch.Tensor",
    global_scale: "torch.Tensor",
    meta: MicroscalingMeta,
) -> None:
    """Validate one tensor or a leading batch of tensors at an artifact boundary."""
    _require_torch()
    fmt = normalize_microscaling_format(meta.format)
    canonical_block_size = 16 if fmt == "nvfp4" else 32
    if int(meta.block_size) != canonical_block_size:
        raise ValueError("microscaling metadata declares the wrong block size.")
    numel = math.prod(meta.shape)
    expected_padding = (-numel) % canonical_block_size
    if int(meta.padding) != expected_padding:
        raise ValueError("microscaling metadata declares inconsistent padding.")
    blocks = math.ceil(numel / canonical_block_size)
    values_per_byte = 1 if fmt == "mxfp8_e4m3" else 2
    payload_width = canonical_block_size // values_per_byte
    if packed.ndim < 2 or tuple(packed.shape[-2:]) != (blocks, payload_width):
        raise ValueError("microscaling packed payload shape does not match metadata.")
    leading_shape = tuple(int(dim) for dim in packed.shape[:-2])
    if tuple(scales.shape) != (*leading_shape, blocks):
        raise ValueError("microscaling scale payload shape does not match metadata.")
    if tuple(global_scale.shape) != leading_shape:
        raise ValueError("microscaling global-scale shape does not match payload.")
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("microscaling payload and block scales must use uint8 storage.")
    if fmt in {"mxfp8_e4m3", "mxfp4"}:
        if bool((scales == 0xFF).any().item()):
            raise ValueError("UE8M0 scale code 255 is NaN and cannot decode weights.")
        if not bool((global_scale.float() == 1.0).all().item()):
            raise ValueError("MXFP payloads require a canonical unit global scale.")
    else:
        if bool(((scales & 0x7F) == 0x7F).any().item()):
            raise ValueError("NVFP4 block scales cannot contain E4M3 NaN codes.")
        global_float = global_scale.float()
        if not bool(
            (torch.isfinite(global_float) & (global_float > 0.0)).all().item()
        ):
            raise ValueError("NVFP4 global scales must be finite and positive.")
    if fmt == "mxfp8_e4m3" and bool(
        ((packed & 0x7F) == 0x7F).any().item()
    ):
        raise ValueError("MXFP8 payload cannot contain E4M3 NaN codes.")


def quantize_microscaling(
    fmt: str,
    weight: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", MicroscalingMeta]:
    """Return encoded elements, block scales, global scale, and static metadata."""
    _require_torch()
    fmt = normalize_microscaling_format(fmt)
    block_size = 16 if fmt == "nvfp4" else 32
    shape = tuple(int(dim) for dim in weight.shape)
    if not shape:
        raise ValueError("microscaling quantization requires a non-scalar tensor.")
    flat = weight.detach().float().reshape(-1)
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError("microscaling quantization requires finite weights.")
    padding = (-flat.numel()) % block_size
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    blocks = flat.reshape(-1, block_size)
    amax = blocks.abs().amax(dim=-1)
    if fmt in {"mxfp8_e4m3", "mxfp4"}:
        destination_max = _E4M3_MAX if fmt == "mxfp8_e4m3" else 6.0
        decode_scale = (amax / destination_max).clamp(min=2.0 ** -127)
        scale_codes = _encode_e8m0(decode_scale)
        decoded_scale = _decode_e8m0(scale_codes)
        global_scale = torch.ones((), dtype=torch.float32, device=weight.device)
    else:
        global_scale = (flat.abs().amax() / (448.0 * 6.0)).clamp(min=1e-30)
        local_scale = (amax / 6.0) / global_scale
        scale_codes = _encode_e4m3(local_scale)
        decoded_scale = _decode_e4m3(scale_codes) * global_scale
    safe_scale = decoded_scale.clamp(min=1e-30).unsqueeze(-1)
    normalized = blocks / safe_scale
    codes = (
        _encode_e4m3(normalized)
        if fmt == "mxfp8_e4m3"
        else _encode_e2m1(normalized)
    )
    meta = MicroscalingMeta(fmt, block_size, shape, padding)
    encoded = codes.contiguous() if fmt == "mxfp8_e4m3" else _pack_nibbles(codes)
    return encoded, scale_codes, global_scale, meta


def dequantize_microscaling(
    packed: "torch.Tensor",
    scales: "torch.Tensor",
    global_scale: "torch.Tensor",
    meta: MicroscalingMeta,
    *,
    dtype: "torch.dtype",
    device: "torch.device",
) -> "torch.Tensor":
    _require_torch()
    fmt = normalize_microscaling_format(meta.format)
    expected_blocks = math.ceil(math.prod(meta.shape) / meta.block_size)
    values_per_byte = 1 if fmt == "mxfp8_e4m3" else 2
    if packed.shape != (expected_blocks, meta.block_size // values_per_byte):
        raise ValueError("microscaling packed payload shape does not match metadata.")
    if scales.shape != (expected_blocks,):
        raise ValueError("microscaling scale payload shape does not match metadata.")
    target = torch.device(device)
    stored = packed.to(device=target)
    codes = stored if fmt == "mxfp8_e4m3" else _unpack_nibbles(stored)
    if fmt in {"mxfp8_e4m3", "mxfp4"}:
        decoded_scale = _decode_e8m0(scales.to(device=target))
    else:
        decoded_scale = _decode_e4m3(scales.to(device=target)) * global_scale.to(
            device=target, dtype=torch.float32
        )
    decoded = _decode_e4m3(codes) if fmt == "mxfp8_e4m3" else _decode_e2m1(codes)
    values = decoded * decoded_scale.unsqueeze(-1)
    return values.reshape(-1)[: math.prod(meta.shape)].reshape(meta.shape).to(dtype=dtype)


def microscaling_stored_bytes(fmt: str, numel: int) -> int:
    fmt = normalize_microscaling_format(fmt)
    if numel <= 0:
        raise ValueError("microscaling storage size requires numel > 0.")
    block_size = 16 if fmt == "nvfp4" else 32
    blocks = math.ceil(numel / block_size)
    element_bytes = (
        blocks * block_size
        if fmt == "mxfp8_e4m3"
        else blocks * block_size // 2
    )
    return element_bytes + blocks + (4 if fmt == "nvfp4" else 0)
