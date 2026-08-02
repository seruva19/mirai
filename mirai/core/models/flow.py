"""Rectified-flow helper math for sparse-MoE diffusion training."""

from __future__ import annotations

from typing import Any

from mirai.core.tensors import is_torch_tensor


def clamp_timesteps(timesteps: Any, eps: float) -> Any:
    if is_torch_tensor(timesteps):
        return timesteps.clamp(min=eps, max=1.0 - eps)
    return [min(max(float(t), eps), 1.0 - eps) for t in timesteps]


def shifted_sigma(timesteps: Any, shift: Any) -> Any:
    if is_torch_tensor(timesteps):
        return (shift * timesteps) / (1.0 + (shift - 1.0) * timesteps)
    if isinstance(shift, (list, tuple)):
        shifts = [float(value) for value in shift]
        return [
            (value_shift * float(timestep))
            / (1.0 + (value_shift - 1.0) * float(timestep))
            for timestep, value_shift in zip(timesteps, shifts, strict=True)
        ]
    out: list[float] = []
    for t in timesteps:
        out.append((shift * float(t)) / (1.0 + (shift - 1.0) * float(t)))
    return out


def _list_apply_noise(
    clean_latents: list[float],
    noise: list[float],
    sigmas: list[float],
) -> list[float]:
    return [
        (1.0 - s) * clean + (s * n)
        for clean, n, s in zip(clean_latents, noise, sigmas, strict=True)
    ]


def apply_rectified_flow_noise(
    clean_latents: Any,
    noise: Any,
    timesteps: Any,
    *,
    shift: Any,
    timestep_eps: float,
) -> Any:
    clamped_t = clamp_timesteps(timesteps, timestep_eps)
    sigma = shifted_sigma(clamped_t, shift)

    if is_torch_tensor(clean_latents):
        if sigma.ndim == 1 and clean_latents.ndim > 1:
            view_shape = [sigma.shape[0], *([1] * (clean_latents.ndim - 1))]
            sigma = sigma.view(*view_shape)
        return (1.0 - sigma) * clean_latents + sigma * noise
    return _list_apply_noise(clean_latents, noise, sigma)


def rectified_flow_target(noise: Any, clean_latents: Any) -> Any:
    if is_torch_tensor(noise):
        return noise - clean_latents
    return [n - clean for clean, n in zip(clean_latents, noise, strict=True)]
