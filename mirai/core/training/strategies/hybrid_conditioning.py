"""Hybrid conditioning strategy (text + optional clip embedding)."""

from __future__ import annotations

from typing import Any, Callable

from mirai.config.schema import StrategyConfig
from mirai.core.models.base import BasePipeline
from mirai.core.registry import register_strategy
from mirai.core.tensors import is_torch_tensor
from mirai.core.training.objectives.sampling import NoiseGenerator, TimestepSampler
from mirai.core.training.strategies.base import TrainingInputs, TrainingStrategy

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _text_payload(batch: dict[str, Any], text_embeds: Any) -> dict[str, Any]:
    payload = {"t5": text_embeds}
    if "text_mask" in batch:
        payload["text_mask"] = batch["text_mask"]
    return payload


@register_strategy("hybrid_conditioning")
class HybridConditioningStrategy(TrainingStrategy):
    def __init__(self, config: StrategyConfig):
        self.config = config

    def get_config_schema(self) -> set[str]:
        return {"text_weight", "clip_weight"}

    def get_extra_batch_keys(self) -> list[str]:
        return ["clip_embed"]

    def validate_config(self) -> list[str]:
        text_w = float(self.config.params.get("text_weight", 1.0))
        clip_w = float(self.config.params.get("clip_weight", 0.0))
        if text_w < 0.0 or clip_w < 0.0:
            return ["text_weight and clip_weight must be >= 0."]
        if text_w == 0.0 and clip_w == 0.0:
            return ["At least one of text_weight or clip_weight must be > 0."]
        return []

    def prepare_inputs(
        self,
        batch: dict[str, Any],
        pipeline: BasePipeline,
        timestep_sampler: TimestepSampler,
        noise_generator: NoiseGenerator,
        *,
        training: bool = True,
        objective: Any = None,
    ) -> TrainingInputs:
        _ = training
        latents = batch["latents"]
        batch_size = latents.shape[0] if is_torch_tensor(latents) else len(latents)
        timesteps = timestep_sampler.sample(batch_size, like=latents)
        noise = noise_generator.sample(latents if is_torch_tensor(latents) else batch_size)
        if objective is not None:
            noisy_latents = objective.corrupt(
                pipeline=pipeline, clean_latents=latents, noise=noise, timesteps=timesteps
            )
        else:
            noisy_latents = pipeline.apply_noise(latents, noise, timesteps)

        text_weight = float(self.config.params.get("text_weight", 1.0))
        clip_weight = float(self.config.params.get("clip_weight", 0.0))
        text = batch.get("text_embeds", [])
        clip = batch.get("clip_embed")
        if clip is None:
            mixed = text
        elif is_torch_tensor(text) and is_torch_tensor(clip):
            mixed = (text * text_weight) + (clip * clip_weight)
        else:
            mixed = [
                (float(tv) * text_weight) + (float(cv) * clip_weight)
                for tv, cv in zip(text, clip, strict=True)
            ]
        model_timesteps = pipeline.prepare_model_timesteps(
            timesteps,
            latents=latents,
        )
        return TrainingInputs(
            noisy_latents=noisy_latents,
            timestep=model_timesteps,
            noise=noise,
            clean_latents=latents,
            text_embeds=_text_payload(batch, mixed),
            objective_timestep=(
                None if model_timesteps is timesteps else timesteps
            ),
            loss_mask=batch.get("loss_mask"),
            extra_forward_kwargs={},
        )

    def compute_per_sample_loss(
        self,
        prediction: Any,
        target: Any,
        inputs: TrainingInputs,
        loss_fn: Callable[[Any, Any], Any],
    ) -> Any:
        raw = loss_fn(prediction, target)
        if is_torch_tensor(raw):
            if raw.ndim == 0:
                return raw.reshape(1)
            if raw.ndim == 1:
                return raw
            return raw.reshape(raw.shape[0], -1).mean(dim=1)
        return raw
