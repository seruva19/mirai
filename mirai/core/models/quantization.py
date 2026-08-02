"""Quantized linear wrappers for frozen base-model weights.

Each wrapper stores weights in a compressed dtype and dequantizes on forward.
Adapter parameters (LoRA/LoHa/LoKr) are never quantized -- they stay in full
precision.

Usage::

    from mirai.core.models.quantization import QUANT_REGISTRY, quantize_linear

    wrapped = quantize_linear("fp8", base_linear)
    # or
    cls = QUANT_REGISTRY.get("nf4")
    wrapped = cls(base_linear, block_size=64)
"""

from __future__ import annotations

from typing import Any

from mirai.core.registry import Registry
from mirai.core.models.compressed_weights.execution.linear import CompressedLinear

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Unified onto the generic Registry[T] so quant formats are discovered the same
# way as models/strategies/losses. Registration order and names are preserved.
QUANT_REGISTRY: Registry[type] = Registry("frozen-weight quantization")


def register_quant(name: str):
    """Decorator to register a quantized-linear class."""
    return QUANT_REGISTRY.decorator(name)


def quantize_linear(quant_type: str, base: Any, **kwargs: Any) -> Any:
    """Wrap an ``nn.Linear`` with the named quantization scheme."""
    key = quant_type.strip().lower()
    if not QUANT_REGISTRY.has(key):
        supported = ", ".join(QUANT_REGISTRY.names()) or "(none)"
        raise ValueError(
            f"Unknown frozen_weight_quantization '{quant_type}'. "
            f"Supported: {supported}"
        )
    return QUANT_REGISTRY.get(key)(base, **kwargs)


def is_quantized_linear(module: Any) -> bool:
    """Return True if *module* is any registered quantized wrapper."""
    registered = tuple(QUANT_REGISTRY.values())
    return bool(registered) and isinstance(module, registered)


def expert_quantization_formats() -> frozenset[str]:
    """Registered quant formats that support routed-expert on-load quantization.

    Derived from the ``supports_expert_quantization`` capability declared by each
    registered wrapper so runtime gating stays
    registry-driven instead of hardwiring ``{"int8", "nf4"}`` at the call site.
    """
    return frozenset(
        name
        for name, cls in QUANT_REGISTRY.items()
        if bool(getattr(cls, "supports_expert_quantization", False))
    )


# ---------------------------------------------------------------------------
# FP8 (float8_e4m3fn)
# ---------------------------------------------------------------------------

@register_quant("fp8")
class FP8Linear(CompressedLinear):
    """DeepSeek-style 128x128/1x128 block-scaled E4M3 reference wrapper."""

    supports_expert_quantization = True

    def __init__(self, base: nn.Linear, **kwargs: Any):
        super().__init__(base, quant_format="fp8", **kwargs)


# ---------------------------------------------------------------------------
# NF4 (4-bit NormalFloat)  --  QLoRA-style quantization
# ---------------------------------------------------------------------------

# Pre-computed NF4 quantization levels (16 values for 4-bit).
# These are the optimal quantile breakpoints for a standard normal
# distribution, as described in the QLoRA paper (Dettmers et al., 2023).
_NF4_LEVELS: list[float] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


@register_quant("nf4")
class NF4Linear(nn.Module):
    """Store frozen weights in 4-bit NormalFloat with per-block absmax scaling.

    Uses the NF4 quantization scheme from QLoRA: each weight is mapped to the
    nearest of 16 pre-computed normal-distribution quantiles, stored as uint8
    (two values packed per byte), with a per-block float16 scale factor.

    Parameters
    ----------
    base : nn.Linear
        The linear layer to quantize.
    block_size : int
        Number of elements per quantization block (default 64).
        Each block gets its own absmax scale factor.
    """

    supports_expert_quantization = True

    def __init__(self, base: nn.Linear, *, block_size: int = 64, **_kwargs: Any):
        super().__init__()
        if torch is None:  # pragma: no cover
            raise RuntimeError("NF4 wrapper requires torch.")
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.block_size = int(block_size)

        weight = base.weight.detach().float()
        self._weight_shape = weight.shape  # (out, in)

        # Flatten and pad to a multiple of block_size
        flat = weight.reshape(-1)
        numel = flat.numel()
        pad_len = (self.block_size - numel % self.block_size) % self.block_size
        if pad_len > 0:
            flat = torch.cat([flat, flat.new_zeros(pad_len)])
        self._padded_numel = flat.numel()

        blocks = flat.reshape(-1, self.block_size)  # (num_blocks, block_size)

        # Per-block absmax scaling
        absmax = blocks.abs().amax(dim=1).clamp(min=1e-12)  # (num_blocks,)
        scaled = blocks / absmax.unsqueeze(1)  # values in [-1, 1]

        # Quantize to nearest NF4 level
        nf4 = torch.tensor(_NF4_LEVELS, dtype=torch.float32)
        # (num_blocks, block_size, 1) vs (1, 1, 16) -> argmin over 16 levels
        dists = (scaled.unsqueeze(-1) - nf4.reshape(1, 1, -1)).abs()
        indices = dists.argmin(dim=-1).to(torch.uint8)  # (num_blocks, block_size)

        # Pack two 4-bit indices per byte
        indices_flat = indices.reshape(-1)
        assert indices_flat.numel() % 2 == 0, "Padded numel should be even"
        low = indices_flat[0::2]
        high = indices_flat[1::2]
        packed = (high << 4) | low  # uint8

        # Store as non-trainable buffers
        self.register_buffer("quant_packed", packed)
        self.register_buffer("quant_scale", absmax.to(torch.float16))
        self.register_buffer(
            "nf4_levels", nf4, persistent=False
        )

        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().float(), requires_grad=False)
        else:
            self.bias = None

    def _dequantize(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        packed = self.quant_packed.to(device=device)
        scale = self.quant_scale.to(device=device, dtype=dtype)
        nf4 = self.nf4_levels.to(device=device, dtype=dtype)

        # Unpack two 4-bit indices per byte
        low = (packed & 0x0F).to(torch.long)
        high = ((packed >> 4) & 0x0F).to(torch.long)
        indices = torch.stack([low, high], dim=1).reshape(-1)  # interleave

        # Look up NF4 values
        values = nf4[indices]  # (padded_numel,)

        # Reshape to blocks and rescale
        blocks = values[: self._padded_numel].reshape(-1, self.block_size)
        blocks = blocks * scale.unsqueeze(1)

        # Flatten and trim padding
        numel = self._weight_shape[0] * self._weight_shape[1]
        flat = blocks.reshape(-1)[:numel]
        return flat.reshape(self._weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize(dtype=x.dtype, device=x.device)
        bias = (
            self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        )
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"dtype=nf4, block_size={self.block_size}"
        )


# ---------------------------------------------------------------------------
# INT8 (symmetric per-channel)
# ---------------------------------------------------------------------------

@register_quant("int8")
class INT8Linear(nn.Module):
    """Store frozen weights as int8 with per-channel (per-output-row) scaling.

    Uses integer arithmetic and does not require FP8 hardware support.
    """

    supports_expert_quantization = True

    def __init__(self, base: nn.Linear, **_kwargs: Any):
        super().__init__()
        if torch is None:  # pragma: no cover
            raise RuntimeError("INT8 wrapper requires torch.")
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)

        weight = base.weight.detach().float()
        # Per-row (per-output-channel) symmetric quantization
        row_absmax = weight.abs().amax(dim=1).clamp(min=1e-12)  # (out,)
        scale = row_absmax / 127.0
        quantized = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer("weight_int8", quantized)
        self.register_buffer("weight_scale", scale.to(torch.float16))
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().float(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight_scale.to(device=x.device, dtype=x.dtype)  # (out,)
        weight = self.weight_int8.to(device=x.device, dtype=x.dtype) * scale.unsqueeze(1)
        bias = (
            self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        )
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"dtype=int8_symmetric"
        )


@register_quant("mxfp4")
class MXFP4Linear(CompressedLinear):
    """OCP MXFP4 frozen-weight wrapper using the portable reference path."""

    supports_expert_quantization = True

    def __init__(self, base: nn.Linear, **kwargs: Any):
        super().__init__(base, quant_format="mxfp4", **kwargs)


@register_quant("mxfp8_e4m3")
class MXFP8E4M3Linear(CompressedLinear):
    """MXFP8-E4M3 wrapper with round-up UE8M0 block scales."""

    supports_expert_quantization = True

    def __init__(self, base: nn.Linear, **kwargs: Any):
        super().__init__(base, quant_format="mxfp8_e4m3", **kwargs)


@register_quant("nvfp4")
class NVFP4Linear(CompressedLinear):
    """NVIDIA NVFP4 frozen-weight wrapper using hierarchical block scaling."""

    supports_expert_quantization = True

    def __init__(self, base: nn.Linear, **kwargs: Any):
        super().__init__(base, quant_format="nvfp4", **kwargs)
