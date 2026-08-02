"""Prior-preservation helpers."""

from __future__ import annotations

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for prior-preservation helpers: {exc}")


def prior_preservation_loss(
    *,
    pipeline,
    noisy_latents,
    timesteps,
    text_embeds,
) -> torch.Tensor:
    """Compute adapter-vs-base regularization loss with adapter temporarily disabled."""
    original_scale = float(pipeline.get_lora_scale())
    pipeline.set_lora_scale(0.0)
    with torch.no_grad():
        base_pred = pipeline.forward(noisy_latents, timesteps, text_embeds)
    pipeline.set_lora_scale(original_scale)
    adapter_pred = pipeline.forward(noisy_latents, timesteps, text_embeds)
    return ((adapter_pred - base_pred.detach()) ** 2).mean()


def build_window_source_schedule(
    *,
    num_windows: int,
    prior_ratio: float,
) -> list[str]:
    if num_windows <= 0:
        raise ValueError("num_windows must be > 0.")
    if prior_ratio < 0.0 or prior_ratio > 1.0:
        raise ValueError("prior_ratio must be in [0, 1].")

    if prior_ratio == 0.5:
        # Deterministic alternation for a balanced prior ratio.
        return ["train" if idx % 2 == 0 else "regularization" for idx in range(num_windows)]
    if prior_ratio <= 0.0:
        return ["train"] * num_windows
    if prior_ratio >= 1.0:
        return ["regularization"] * num_windows

    prior_windows = round(num_windows * prior_ratio)
    schedule = ["regularization"] * prior_windows + ["train"] * (num_windows - prior_windows)
    return schedule


def expand_windows_to_micro_batches(
    *,
    window_schedule: list[str],
    gradient_accumulation: int,
) -> list[str]:
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be > 0.")
    out: list[str] = []
    for source in window_schedule:
        out.extend([source] * gradient_accumulation)
    return out
