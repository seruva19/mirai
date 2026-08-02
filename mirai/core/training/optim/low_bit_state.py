"""Packed low-bit exponential-moving-average optimizer states.

The 4/2-bit format follows SOLO's fine-tuning configuration: signed
dynamic-exponent 4-bit first moments and unsigned logarithmic 2-bit second
moments, both with 128-element blocks.  Floating-point EMA updates remain the
reference computation; only the persistent state is quantized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Low-bit optimizer states require torch: {exc}")


SOLO_4_2_STATE_FORMAT = "solo_4_2"
SOLO_4_2_BLOCK_SIZE = 128
SOLO_4_2_QUANTILE = 0.1
SOLO_4_2_BETAS = (0.8, 0.999)

# Dynamic-exponent levels for signed 4-bit state.  These are the exact levels
# obtained from the mapping described by Li et al. and used by SOLO/torchao.
SIGNED_DE_4BIT_LEVELS = (
    -0.8875,
    -0.6625,
    -0.4375,
    -0.2125,
    -0.0775,
    -0.0325,
    -0.0055,
    0.0,
    0.0055,
    0.0325,
    0.0775,
    0.2125,
    0.4375,
    0.6625,
    0.8875,
    1.0,
)


@dataclass(frozen=True)
class PackedMomentState:
    """Typed, checkpoint-safe payload for one packed optimizer moment."""

    encoding: str
    codes: Any
    scales: Any
    bases: Any
    shape: tuple[int, ...]
    block_size: int = SOLO_4_2_BLOCK_SIZE

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in (self.codes, self.scales, self.bases)
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "codes": self.codes,
            "scales": self.scales,
            "bases": self.bases,
            "shape": self.shape,
            "block_size": self.block_size,
        }

    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        device: Any | None = None,
    ) -> "PackedMomentState":
        if not isinstance(payload, Mapping):
            raise ValueError("Packed optimizer moment must be a mapping.")
        try:
            encoding = str(payload["encoding"])
            codes = payload["codes"]
            scales = payload["scales"]
            bases = payload["bases"]
            shape = tuple(int(value) for value in payload["shape"])
            block_size = int(payload["block_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Packed optimizer moment metadata is invalid.") from exc
        if device is not None:
            codes = codes.to(device=device, dtype=torch.uint8)
            scales = scales.to(device=device, dtype=torch.float32)
            bases = bases.to(device=device, dtype=torch.float32)
        result = cls(
            encoding=encoding,
            codes=codes,
            scales=scales,
            bases=bases,
            shape=shape,
            block_size=block_size,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.encoding not in {"signed_de_4bit", "unsigned_qema_2bit"}:
            raise ValueError(f"Unsupported packed moment encoding: {self.encoding!r}.")
        if self.block_size != SOLO_4_2_BLOCK_SIZE:
            raise ValueError("SOLO 4/2-bit optimizer state requires block_size=128.")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("Packed optimizer moment shape must be non-empty and positive.")
        if not torch.is_tensor(self.codes) or self.codes.dtype is not torch.uint8:
            raise ValueError("Packed optimizer moment codes must be uint8.")
        if not torch.is_tensor(self.scales) or self.scales.dtype is not torch.float32:
            raise ValueError("Packed optimizer moment scales must be float32.")
        if not torch.is_tensor(self.bases) or self.bases.dtype is not torch.float32:
            raise ValueError("Packed optimizer moment bases must be float32.")
        blocks = math.ceil(self.numel / self.block_size)
        values_per_byte = 2 if self.encoding == "signed_de_4bit" else 4
        padded_values = blocks * self.block_size
        if self.codes.ndim != 1 or self.codes.numel() != padded_values // values_per_byte:
            raise ValueError("Packed optimizer moment code length is invalid.")
        if self.scales.ndim != 1 or self.scales.numel() != blocks:
            raise ValueError("Packed optimizer moment scale length is invalid.")
        expected_bases = 0 if self.encoding == "signed_de_4bit" else blocks
        if self.bases.ndim != 1 or self.bases.numel() != expected_bases:
            raise ValueError("Packed optimizer moment base length is invalid.")
        if not torch.isfinite(self.scales).all() or (self.scales < 0).any():
            raise ValueError("Packed optimizer moment scales must be finite and non-negative.")
        if self.bases.numel() and (
            not torch.isfinite(self.bases).all()
            or (self.bases < 0).any()
            or (self.bases > 1).any()
        ):
            raise ValueError("QEMA bases must be finite and lie in [0, 1].")


def _padded_blocks(tensor: Any, *, block_size: int) -> tuple[Any, int]:
    values = tensor.detach().to(dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        raise ValueError("Cannot quantize an empty optimizer moment.")
    if not torch.isfinite(values).all():
        raise ValueError("Optimizer moments must be finite before quantization.")
    blocks = math.ceil(values.numel() / block_size)
    padded_numel = blocks * block_size
    if padded_numel != values.numel():
        values = torch.nn.functional.pad(values, (0, padded_numel - values.numel()))
    return values.view(blocks, block_size), tensor.numel()


def _nearest_code(values: Any, levels: Any) -> Any:
    """Return nearest monotonic code, resolving exact midpoints upward."""

    insertion = torch.searchsorted(levels, values, right=False)
    upper = insertion.clamp(max=levels.numel() - 1)
    lower = (upper - 1).clamp(min=0)
    lower_distance = (values - levels[lower]).abs()
    upper_distance = (levels[upper] - values).abs()
    return torch.where(upper_distance <= lower_distance, upper, lower).to(torch.uint8)


def _pack_4bit(codes: Any) -> Any:
    flat = codes.reshape(-1).to(torch.uint8)
    return ((flat[0::2] << 4) | flat[1::2]).contiguous()


def _unpack_4bit(codes: Any) -> Any:
    return torch.stack((codes >> 4, codes & 0x0F), dim=-1).reshape(-1)


def _pack_2bit(codes: Any) -> Any:
    flat = codes.reshape(-1).to(torch.uint8)
    return (
        (flat[0::4] << 6)
        | (flat[1::4] << 4)
        | (flat[2::4] << 2)
        | flat[3::4]
    ).contiguous()


def _unpack_2bit(codes: Any) -> Any:
    return torch.stack(
        (
            codes >> 6,
            (codes >> 4) & 0x03,
            (codes >> 2) & 0x03,
            codes & 0x03,
        ),
        dim=-1,
    ).reshape(-1)


def encode_signed_de_4bit(
    tensor: Any,
    *,
    block_size: int = SOLO_4_2_BLOCK_SIZE,
) -> PackedMomentState:
    """Encode a signed first moment with SOLO's 4-bit DE mapping."""

    if block_size != SOLO_4_2_BLOCK_SIZE:
        raise ValueError("SOLO 4/2-bit optimizer state requires block_size=128.")
    blocks, _ = _padded_blocks(tensor, block_size=block_size)
    scales = blocks.abs().amax(dim=1)
    normalized = torch.where(
        scales[:, None] > 0,
        blocks / scales.clamp_min(torch.finfo(torch.float32).tiny)[:, None],
        torch.zeros_like(blocks),
    )
    levels = torch.tensor(
        SIGNED_DE_4BIT_LEVELS,
        device=blocks.device,
        dtype=torch.float32,
    )
    result = PackedMomentState(
        encoding="signed_de_4bit",
        codes=_pack_4bit(_nearest_code(normalized, levels)),
        scales=scales.to(torch.float32),
        bases=torch.empty(0, device=blocks.device, dtype=torch.float32),
        shape=tuple(int(value) for value in tensor.shape),
        block_size=block_size,
    )
    result.validate()
    return result


def decode_signed_de_4bit(payload: PackedMomentState) -> Any:
    payload.validate()
    if payload.encoding != "signed_de_4bit":
        raise ValueError("Expected a signed DE4 optimizer moment.")
    levels = torch.tensor(
        SIGNED_DE_4BIT_LEVELS,
        device=payload.codes.device,
        dtype=torch.float32,
    )
    values = levels[_unpack_4bit(payload.codes).long()].view(
        -1, payload.block_size
    )
    values = values * payload.scales[:, None]
    return values.reshape(-1)[: payload.numel].view(payload.shape)


def encode_unsigned_qema_2bit(
    tensor: Any,
    *,
    block_size: int = SOLO_4_2_BLOCK_SIZE,
    quantile: float = SOLO_4_2_QUANTILE,
) -> PackedMomentState:
    """Encode an unsigned second moment with SOLO's logarithmic QEMA map."""

    if block_size != SOLO_4_2_BLOCK_SIZE:
        raise ValueError("SOLO 4/2-bit optimizer state requires block_size=128.")
    if quantile != SOLO_4_2_QUANTILE:
        raise ValueError("SOLO 4/2-bit optimizer state requires quantile=0.1.")
    if (tensor < 0).any():
        raise ValueError("Unsigned QEMA optimizer moments must be non-negative.")
    blocks, original_numel = _padded_blocks(tensor, block_size=block_size)
    scales = blocks.amax(dim=1)
    normalized = torch.where(
        scales[:, None] > 0,
        blocks / scales.clamp_min(torch.finfo(torch.float32).tiny)[:, None],
        torch.zeros_like(blocks),
    )
    quantiles = torch.quantile(normalized, quantile, dim=1)
    tail = original_numel % block_size
    if tail:
        quantiles[-1] = torch.quantile(normalized[-1, :tail], quantile)
    bases = quantiles.clamp(0.0, 1.0).pow(1.0 / 3.0)

    base_matrix = bases[:, None]
    regular = (base_matrix > 0) & (base_matrix < 1)
    safe_values = normalized.clamp_min(torch.finfo(torch.float32).tiny)
    safe_bases = base_matrix.clamp_min(torch.finfo(torch.float32).tiny)
    exponents = torch.log2(safe_values) / torch.log2(safe_bases)
    rounded = torch.round(exponents + torch.rand_like(exponents) - 0.5)
    codes = rounded.clamp(0, 3).to(torch.uint8)
    boundary_codes = torch.where(
        normalized >= 1.0,
        torch.zeros_like(codes),
        torch.full_like(codes, 3),
    )
    codes = torch.where(regular, codes, boundary_codes)
    if tail:
        codes[-1, tail:] = 3

    result = PackedMomentState(
        encoding="unsigned_qema_2bit",
        codes=_pack_2bit(codes),
        scales=scales.to(torch.float32),
        bases=bases.to(torch.float32),
        shape=tuple(int(value) for value in tensor.shape),
        block_size=block_size,
    )
    result.validate()
    return result


def decode_unsigned_qema_2bit(payload: PackedMomentState) -> Any:
    payload.validate()
    if payload.encoding != "unsigned_qema_2bit":
        raise ValueError("Expected an unsigned QEMA2 optimizer moment.")
    codes = _unpack_2bit(payload.codes).view(-1, payload.block_size)
    values = payload.bases[:, None].pow(codes.to(torch.float32))
    values = values * payload.scales[:, None]
    zero_blocks = payload.scales == 0
    if zero_blocks.any():
        values[zero_blocks] = 0
    return values.reshape(-1)[: payload.numel].view(payload.shape)


def encode_solo_4_2_moments(first: Any, second: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        encode_signed_de_4bit(first).state_dict(),
        encode_unsigned_qema_2bit(second).state_dict(),
    )


def decode_solo_4_2_moments(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    device: Any,
) -> tuple[Any, Any]:
    first_payload = PackedMomentState.from_state_dict(first, device=device)
    second_payload = PackedMomentState.from_state_dict(second, device=device)
    return (
        decode_signed_de_4bit(first_payload),
        decode_unsigned_qema_2bit(second_payload),
    )


def packed_solo_4_2_state_nbytes(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> int:
    return (
        PackedMomentState.from_state_dict(first).nbytes
        + PackedMomentState.from_state_dict(second).nbytes
    )
