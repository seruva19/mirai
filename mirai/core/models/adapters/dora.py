"""Weight-decomposed low-rank adaptation primitives.

The effective weight follows DoRA Equation 5 from arXiv:2402.09353:
``W' = m * (W + sBA) / ||W + sBA||_c``.  Mirai treats the final adapter
scale as an interpolation between the frozen base and that paper-defined
weight so a runtime scale of zero still disables the complete adapter.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


DORA_MAGNITUDE_SUFFIX = ".dora_magnitude"


def dora_row_magnitude(weight: Any) -> Any:
    """Return the output-channel magnitude in FP32."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("DoRA requires torch.")
    if not isinstance(weight, torch.Tensor) or weight.ndim not in {2, 3}:
        raise ValueError("DoRA weights must be matrices or grouped matrices.")
    return torch.linalg.vector_norm(weight.float(), dim=-1)


def dora_effective_weight(
    *,
    base_weight: Any,
    lora_a: Any,
    lora_b: Any,
    magnitude: Any,
    direction_scale: float,
    adapter_scale: float,
) -> Any:
    """Build the DoRA weight with stable FP32 normalization.

    ``direction_scale`` is the fixed LoRA alpha/rank rule inside the direction.
    ``adapter_scale`` is Mirai's runtime/schedule scale and blends the whole
    DoRA update with the frozen base, preserving an exact zero-scale bypass.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("DoRA requires torch.")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (base_weight, lora_a, lora_b, magnitude)
    ):
        raise TypeError("DoRA inputs must be tensors.")
    if base_weight.ndim not in {2, 3}:
        raise ValueError("DoRA base weights must be matrices or grouped matrices.")
    if lora_a.ndim != base_weight.ndim or lora_b.ndim != base_weight.ndim:
        raise ValueError("DoRA factors must have the base weight's rank.")
    delta = torch.matmul(lora_b.float(), lora_a.float())
    if tuple(delta.shape) != tuple(base_weight.shape):
        raise ValueError("DoRA factor product must match the base weight shape.")
    if tuple(magnitude.shape) != tuple(base_weight.shape[:-1]):
        raise ValueError(
            "DoRA magnitude shape must match every base-weight axis except input."
        )

    base_fp32 = base_weight.float()
    direction = base_fp32 + (delta * float(direction_scale))
    denominator = dora_row_magnitude(direction).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    normalized = direction * (magnitude.float() / denominator).unsqueeze(-1)
    effective = base_fp32 + (
        normalized - base_fp32
    ) * float(adapter_scale)
    return effective.to(dtype=base_weight.dtype)


__all__ = [
    "DORA_MAGNITUDE_SUFFIX",
    "dora_effective_weight",
    "dora_row_magnitude",
]
