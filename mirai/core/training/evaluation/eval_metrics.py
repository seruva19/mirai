"""Measured evaluation metrics computed from native eval tensors."""

from __future__ import annotations

import math
from typing import Any

from mirai.core.models.flow import shifted_sigma

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


BUILTIN_MEASURED_EVAL_METRICS = {
    "pred_target_mse",
    "pred_target_mae",
    "pred_target_psnr",
    "latent_recon_mse",
    "latent_recon_mae",
    "latent_recon_psnr",
    "frame_recon_mse",
    "frame_recon_mae",
    "frame_recon_psnr",
    "frame_recon_ssim",
    "frame_temporal_delta_mse",
    "frame_temporal_delta_mae",
    "frame_temporal_delta_ssim",
}
_MAX_PSNR_DB = 120.0


def _tensor_mae(prediction: Any, target: Any) -> float:
    return float(torch.mean(torch.abs(prediction - target)).item())


def _tensor_mse(prediction: Any, target: Any) -> float:
    return float(torch.mean((prediction - target) ** 2).item())


def _tensor_psnr(prediction: Any, target: Any) -> float:
    mse = _tensor_mse(prediction, target)
    if mse <= 0.0:
        return float(_MAX_PSNR_DB)
    data_range = float((target.max() - target.min()).abs().item())
    if data_range <= 0.0:
        data_range = 1.0
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def _tensor_ssim(prediction: Any, target: Any) -> float:
    pred = prediction.detach().float()
    tgt = target.detach().float()
    if torch.equal(pred, tgt):
        return 1.0
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        tgt = tgt.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
    if pred.ndim != 4 or tgt.ndim != 4:
        raise ValueError("SSIM expects [N,C,H,W]-like tensors.")

    height = int(pred.shape[-2])
    width = int(pred.shape[-1])
    kernel = max(1, min(7, height, width))
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    pad = kernel // 2

    data_min = min(float(pred.min().item()), float(tgt.min().item()))
    data_max = max(float(pred.max().item()), float(tgt.max().item()))
    data_range = max(1e-6, data_max - data_min)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(pred, kernel_size=kernel, stride=1, padding=pad)
    mu_y = F.avg_pool2d(tgt, kernel_size=kernel, stride=1, padding=pad)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(pred * pred, kernel_size=kernel, stride=1, padding=pad) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(tgt * tgt, kernel_size=kernel, stride=1, padding=pad) - mu_y_sq
    sigma_xy = F.avg_pool2d(pred * tgt, kernel_size=kernel, stride=1, padding=pad) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)
    return float(ssim_map.mean().item())


def compute_measured_eval_metrics(
    *,
    clean_latents: Any,
    noisy_latents: Any,
    prediction: Any,
    target: Any,
    timesteps: Any,
    flow_shift: float,
) -> dict[str, float]:
    sigma = shifted_sigma(timesteps, float(flow_shift))
    if sigma.ndim == 1 and clean_latents.ndim > 1:
        view_shape = [sigma.shape[0], *([1] * (clean_latents.ndim - 1))]
        sigma = sigma.view(*view_shape)
    reconstructed_clean = noisy_latents - (sigma * prediction)
    return {
        "pred_target_mse": _tensor_mse(prediction, target),
        "pred_target_mae": _tensor_mae(prediction, target),
        "pred_target_psnr": _tensor_psnr(prediction, target),
        "latent_recon_mse": _tensor_mse(reconstructed_clean, clean_latents),
        "latent_recon_mae": _tensor_mae(reconstructed_clean, clean_latents),
        "latent_recon_psnr": _tensor_psnr(reconstructed_clean, clean_latents),
    }


def average_metric_series(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({key for row in metric_rows for key in row})
    out: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in metric_rows if key in row]
        if values:
            out[key] = float(sum(values) / len(values))
    return out


def parse_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return max(1, int(value[0])), max(1, int(value[1]))
    text = str(value).strip().lower()
    if text and "x" in text:
        w_raw, h_raw = text.split("x", 1)
        try:
            return max(1, int(w_raw)), max(1, int(h_raw))
        except ValueError:
            pass
    return 64, 64


def compute_frame_reconstruction_metrics(
    *,
    clean_frames: Any,
    reconstructed_frames: Any,
) -> dict[str, float]:
    metrics = {
        "frame_recon_mse": _tensor_mse(reconstructed_frames, clean_frames),
        "frame_recon_mae": _tensor_mae(reconstructed_frames, clean_frames),
        "frame_recon_psnr": _tensor_psnr(reconstructed_frames, clean_frames),
        "frame_recon_ssim": _tensor_ssim(reconstructed_frames, clean_frames),
    }
    if int(clean_frames.shape[0]) > 1 and int(reconstructed_frames.shape[0]) > 1:
        clean_delta = clean_frames[1:] - clean_frames[:-1]
        recon_delta = reconstructed_frames[1:] - reconstructed_frames[:-1]
        metrics["frame_temporal_delta_mse"] = _tensor_mse(recon_delta, clean_delta)
        metrics["frame_temporal_delta_mae"] = _tensor_mae(recon_delta, clean_delta)
        metrics["frame_temporal_delta_ssim"] = _tensor_ssim(recon_delta, clean_delta)
    return metrics
