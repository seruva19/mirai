"""Learned uncertainty weighting across flow-matching noise levels.

The parameterization follows EDM2's scalar ``u(sigma)`` head, adapted to the
linear flow path used by Mirai.  For ``x_t = (1 - t) x + t noise`` the
noise-to-signal ratio is ``sigma = t / (1 - t)``, so the EDM noise coordinate
``log(sigma) / 4`` is exactly ``logit(t) / 4``.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Adaptive noise-level weighting requires torch: {exc}")


class NoiseLevelUncertaintyWeighting(nn.Module):
    """Estimate one log-variance value for each sampled noise level."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        if int(channels) < 1:
            raise ValueError("Uncertainty weighting channels must be >= 1.")
        self.channels = int(channels)
        self.register_buffer(
            "frequencies",
            2.0 * math.pi * torch.randn(self.channels, dtype=torch.float32),
        )
        self.register_buffer(
            "phases",
            2.0 * math.pi * torch.rand(self.channels, dtype=torch.float32),
        )
        self.output_weight = nn.Parameter(
            torch.randn(1, self.channels, dtype=torch.float32)
        )

    def forward(self, timesteps: Any, *, timestep_eps: float) -> Any:
        if not torch.is_tensor(timesteps):
            raise TypeError(
                "Adaptive noise-level weighting requires tensor timesteps."
            )
        t = timesteps.to(dtype=torch.float32).reshape(-1)
        t = t.clamp(
            min=float(timestep_eps),
            max=1.0 - float(timestep_eps),
        )
        noise_coordinate = (torch.log(t) - torch.log1p(-t)) / 4.0
        features = (
            noise_coordinate[:, None] * self.frequencies[None, :]
            + self.phases[None, :]
        ).cos() * math.sqrt(2.0)

        weight = self.output_weight.to(dtype=torch.float32)
        norm = torch.linalg.vector_norm(weight, dtype=torch.float32)
        norm = norm + 1.0e-4 / math.sqrt(float(weight.numel()))
        weight = weight / norm / math.sqrt(float(weight.shape[1]))
        return (features @ weight.t()).reshape(-1)

    def weighted_loss(
        self,
        per_sample_loss: Any,
        *,
        timesteps: Any,
        timestep_eps: float,
    ) -> tuple[Any, Any, Any]:
        """Return ``L / exp(u) + u``, its multiplicative weight, and ``u``."""

        if not torch.is_tensor(per_sample_loss):
            raise TypeError(
                "Adaptive noise-level weighting requires tensor losses."
            )
        log_variance = self.forward(
            timesteps,
            timestep_eps=timestep_eps,
        )
        inverse_variance = torch.exp(-log_variance)
        weighted = per_sample_loss.to(dtype=torch.float32) * inverse_variance
        weighted = weighted + log_variance
        return weighted, inverse_variance, log_variance


__all__ = ["NoiseLevelUncertaintyWeighting"]
