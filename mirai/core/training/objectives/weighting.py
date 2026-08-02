"""Loss weighting policies."""

from __future__ import annotations

from mirai.core.tensors import is_torch_tensor


def _clamp_timestep(t: float, eps: float) -> float:
    return min(max(t, eps), 1.0 - eps)


def _min_snr_weight_scalar(t: float, gamma: float, eps: float) -> float:
    tc = _clamp_timestep(t, eps)
    snr = ((1.0 - tc) ** 2) / max(tc * tc, eps)
    return min(snr, gamma) / max(snr, eps)


def compute_loss_weights(
    timesteps, mode: str, gamma: float, timestep_eps: float
):
    mode_key = mode.strip().lower()
    if is_torch_tensor(timesteps):
        if mode_key == "uniform":
            return timesteps.new_ones(timesteps.shape)
        if mode_key == "min_snr_gamma":
            tc = timesteps.clamp(min=timestep_eps, max=1.0 - timestep_eps)
            snr = ((1.0 - tc) ** 2) / (tc * tc).clamp(min=timestep_eps)
            return snr.clamp(max=gamma) / snr.clamp(min=timestep_eps)
        if mode_key == "cosmap":
            tc = timesteps.clamp(min=timestep_eps, max=1.0 - timestep_eps)
            return 0.5 + 0.5 * (1.0 - (2.0 * tc - 1.0).abs())
        raise ValueError(f"Unsupported loss_weighting '{mode}'.")

    if mode_key == "uniform":
        return [1.0] * len(timesteps)
    if mode_key == "min_snr_gamma":
        return [_min_snr_weight_scalar(t, gamma=gamma, eps=timestep_eps) for t in timesteps]
    if mode_key == "cosmap":
        return [0.5 + 0.5 * (1.0 - abs(2.0 * _clamp_timestep(t, timestep_eps) - 1.0)) for t in timesteps]
    raise ValueError(f"Unsupported loss_weighting '{mode}'.")
