"""Latent-space Haar supervision for rectified-flow training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.tensors import is_torch_tensor

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LatentWaveletLoss:
    per_sample_loss: Any
    low_frequency_loss: Any
    high_frequency_loss: Any


def _haar_subbands(value: Any) -> tuple[Any, Any, Any, Any]:
    if torch is None or not is_torch_tensor(value):
        raise TypeError("Latent wavelet loss requires a torch tensor.")
    if value.ndim < 3:
        raise ValueError(
            "Latent wavelet loss requires a batch and two spatial dimensions."
        )
    if int(value.shape[-2]) % 2 or int(value.shape[-1]) % 2:
        raise ValueError(
            "Latent wavelet loss requires even latent height and width."
        )
    source = value.float()
    a = source[..., 0::2, 0::2]
    b = source[..., 0::2, 1::2]
    c = source[..., 1::2, 0::2]
    d = source[..., 1::2, 1::2]
    return (
        (a + b + c + d) * 0.5,
        (a - b + c - d) * 0.5,
        (a + b - c - d) * 0.5,
        (a - b - c + d) * 0.5,
    )


def _per_sample_mse(left: Any, right: Any) -> Any:
    return (left - right).square().reshape(left.shape[0], -1).mean(dim=1)


def reconstruct_clean_rectified_flow(
    *,
    prediction: Any,
    noisy_latents: Any,
    sigmas: Any,
) -> Any:
    """Recover x0 from a rectified-flow velocity prediction."""

    if torch is None or not all(
        is_torch_tensor(value) for value in (prediction, noisy_latents, sigmas)
    ):
        raise TypeError("Clean-latent reconstruction requires torch tensors.")
    if tuple(prediction.shape) != tuple(noisy_latents.shape):
        raise ValueError("Prediction and noisy latent shapes must match.")
    sigma = sigmas.to(device=prediction.device, dtype=prediction.dtype)
    if sigma.numel() != int(prediction.shape[0]):
        raise ValueError("Expected one rectified-flow sigma per batch item.")
    sigma = sigma.reshape((-1,) + (1,) * (prediction.ndim - 1))
    return noisy_latents.to(
        device=prediction.device,
        dtype=prediction.dtype,
    ) - sigma * prediction


def compute_latent_wavelet_loss(
    *,
    predicted_clean: Any,
    clean_latents: Any,
) -> LatentWaveletLoss:
    """Apply the single-level spatial Haar objective from Nucleus-Image."""

    if not is_torch_tensor(clean_latents):
        raise TypeError("Latent wavelet loss requires tensor clean latents.")
    if tuple(predicted_clean.shape) != tuple(clean_latents.shape):
        raise ValueError("Predicted and target clean latent shapes must match.")
    predicted_bands = _haar_subbands(predicted_clean)
    target_bands = _haar_subbands(
        clean_latents.to(device=predicted_clean.device)
    )
    band_losses = tuple(
        _per_sample_mse(predicted, target)
        for predicted, target in zip(
            predicted_bands,
            target_bands,
            strict=True,
        )
    )
    low = band_losses[0]
    high = band_losses[1] + band_losses[2] + band_losses[3]
    return LatentWaveletLoss(
        per_sample_loss=low + high,
        low_frequency_loss=low,
        high_frequency_loss=high,
    )


__all__ = [
    "LatentWaveletLoss",
    "compute_latent_wavelet_loss",
    "reconstruct_clean_rectified_flow",
]
