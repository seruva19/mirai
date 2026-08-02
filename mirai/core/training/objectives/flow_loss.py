"""Flow-matching loss assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.tensors import is_torch_tensor
from mirai.core.training.objectives.weighting import compute_loss_weights

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class FlowLossResult:
    loss: Any
    loss_pre_accum: Any
    per_sample_loss: Any
    per_sample_loss_normalized: Any
    loss_weights: Any
    weighted_loss: Any
    diagnostics: dict[str, Any] | None = None


def apply_noise_offset(noise: Any, offset: float) -> Any:
    value = float(offset)
    if value == 0.0:
        return noise
    if is_torch_tensor(noise):
        return noise + value
    return [float(v) + value for v in noise]


def normalize_loss_across_buckets(
    per_sample_loss: Any,
    *,
    bucket_ids: list[str] | None,
    mode: str,
) -> Any:
    mode_key = mode.strip().lower()
    if mode_key == "none":
        return per_sample_loss
    if mode_key != "per_bucket_mean":
        raise ValueError(f"Unsupported loss_bucket_normalization '{mode}'.")
    if not bucket_ids:
        return per_sample_loss

    if is_torch_tensor(per_sample_loss):
        out = per_sample_loss.clone()
        unique = sorted(set(str(v) for v in bucket_ids))
        for key in unique:
            mask = torch.tensor(
                [str(v) == key for v in bucket_ids],
                device=per_sample_loss.device,
                dtype=torch.bool,
            )
            if not mask.any():
                continue
            denom = per_sample_loss[mask].mean().clamp(min=1e-8)
            out[mask] = per_sample_loss[mask] / denom
        return out

    out_list = [float(v) for v in per_sample_loss]
    groups: dict[str, list[int]] = {}
    for idx, bucket in enumerate(bucket_ids):
        groups.setdefault(str(bucket), []).append(idx)
    for indexes in groups.values():
        denom = sum(out_list[idx] for idx in indexes) / max(1, len(indexes))
        denom = max(denom, 1e-8)
        for idx in indexes:
            out_list[idx] = out_list[idx] / denom
    return out_list


def compute_flow_matching_loss(
    *,
    per_sample_loss: Any,
    timesteps: Any,
    gradient_accumulation: int,
    loss_weighting: str,
    min_snr_gamma: float,
    timestep_eps: float,
    adaptive_weighting: Any = None,
    bucket_ids: list[str] | None = None,
    loss_bucket_normalization: str = "none",
) -> FlowLossResult:
    normalized = normalize_loss_across_buckets(
        per_sample_loss,
        bucket_ids=bucket_ids,
        mode=loss_bucket_normalization,
    )
    mode_key = str(loss_weighting).strip().lower()
    diagnostics: dict[str, Any] = {}
    if mode_key == "adaptive_uncertainty":
        if adaptive_weighting is None:
            raise ValueError(
                "Adaptive uncertainty loss weighting is not configured."
            )
        weighted, weights, log_variance = adaptive_weighting.weighted_loss(
            normalized,
            timesteps=timesteps,
            timestep_eps=timestep_eps,
        )
        diagnostics = {
            "adaptive_loss_log_variance_mean": log_variance.detach().mean(),
            "adaptive_loss_log_variance_min": log_variance.detach().min(),
            "adaptive_loss_log_variance_max": log_variance.detach().max(),
        }
    else:
        weights = compute_loss_weights(
            timesteps=timesteps,
            mode=loss_weighting,
            gamma=min_snr_gamma,
            timestep_eps=timestep_eps,
        )

    # Explicit ordering: weight -> reduce -> accumulation divide.
    if is_torch_tensor(normalized):
        if mode_key != "adaptive_uncertainty":
            weighted = normalized * weights
        pre_accum = weighted.mean()
    else:
        if mode_key == "adaptive_uncertainty":
            raise TypeError(
                "Adaptive uncertainty loss weighting requires tensor losses."
            )
        weighted = [
            loss * weight
            for loss, weight in zip(normalized, weights, strict=True)
        ]
        pre_accum = sum(weighted) / len(weighted)
    accum = max(1, int(gradient_accumulation))
    loss = pre_accum / accum
    return FlowLossResult(
        loss=loss,
        loss_pre_accum=pre_accum,
        per_sample_loss=per_sample_loss,
        per_sample_loss_normalized=normalized,
        loss_weights=weights,
        weighted_loss=weighted,
        diagnostics=diagnostics,
    )
