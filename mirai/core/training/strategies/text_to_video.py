"""Text-to-video strategy."""

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


@register_strategy("text_to_video")
class TextToVideoStrategy(TrainingStrategy):
    def __init__(self, config: StrategyConfig):
        self.config = config

    def get_config_schema(self) -> set[str]:
        return {"conditioning_dropout_p", "empty_text_embed"}

    def _conditioning_dropout_p(self) -> float:
        return float(self.config.params.get("conditioning_dropout_p", 0.0))

    def _empty_text_embed(self) -> float:
        return float(self.config.params.get("empty_text_embed", 0.0))

    def validate_config(self) -> list[str]:
        p = self._conditioning_dropout_p()
        if p < 0.0 or p > 1.0:
            return ["conditioning_dropout_p must be in [0, 1]."]
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
        text_embeds = batch.get("text_embeds", [])
        model_timesteps = pipeline.prepare_model_timesteps(
            timesteps,
            latents=latents,
        )
        p = self._conditioning_dropout_p()
        empty_embed = self._empty_text_embed()
        if training and p > 0.0:
            if is_torch_tensor(text_embeds):
                text_embeds = text_embeds.clone()
                if p >= 1.0:
                    text_embeds[:] = empty_embed
                else:
                    mask = torch.rand((batch_size,), device=text_embeds.device) < p
                    text_embeds[mask] = empty_embed
            else:
                out = list(text_embeds)
                if p >= 1.0:
                    out = [empty_embed for _ in out]
                else:
                    import random

                    step_hint = batch.get("_step", 0)
                    rng = random.Random(batch_size + step_hint)
                    out = [empty_embed if rng.random() < p else value for value in out]
                text_embeds = out
        return TrainingInputs(
            noisy_latents=noisy_latents,
            timestep=model_timesteps,
            noise=noise,
            clean_latents=latents,
            text_embeds=_text_payload(batch, text_embeds),
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
            mask = inputs.loss_mask
            if mask is not None:
                if mask.ndim == 1 and raw.ndim == 1:
                    return raw * mask.to(dtype=raw.dtype)
                if raw.ndim > 1:
                    weighted = raw * mask
                    denom = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp(min=1.0)
                    return weighted.reshape(weighted.shape[0], -1).sum(dim=1) / denom
            if raw.ndim == 1:
                return raw
            return raw.reshape(raw.shape[0], -1).mean(dim=1)
        if inputs.loss_mask is None:
            return raw
        return [
            value * mask
            for value, mask in zip(raw, inputs.loss_mask, strict=True)
        ]
