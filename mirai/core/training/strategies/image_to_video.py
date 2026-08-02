"""Image-to-video strategy."""

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


def _condition_frame(
    noisy_latents: Any,
    clean_latents: Any,
    loss_mask: Any,
    *,
    frame_dim: int,
    batch_index: int | None = None,
) -> None:
    ndim = int(getattr(clean_latents, "ndim", 0))
    dim = int(frame_dim)
    if dim <= 0 or dim >= ndim:
        raise ValueError(
            f"Invalid image-to-video conditioning frame dimension {dim} for "
            f"latent tensor rank {ndim}."
        )
    selector: list[Any] = [slice(None)] * ndim
    selector[dim] = 0
    if batch_index is not None:
        selector[0] = int(batch_index)
    key = tuple(selector)
    noisy_latents[key] = clean_latents[key]
    loss_mask[key] = 0.0


def _text_payload(batch: dict[str, Any]) -> dict[str, Any]:
    payload = {"t5": batch.get("text_embeds", [])}
    if "text_mask" in batch:
        payload["text_mask"] = batch["text_mask"]
    return payload


@register_strategy("image_to_video")
class ImageToVideoStrategy(TrainingStrategy):
    def __init__(self, config: StrategyConfig):
        self.config = config

    def _p(self) -> float:
        return float(self.config.params.get("first_frame_conditioning_p", 1.0))

    def get_config_schema(self) -> set[str]:
        return {"first_frame_conditioning_p"}

    def get_extra_batch_keys(self) -> list[str]:
        return ["clip_embed"]

    def get_required_batch_keys(self, *, model_type: str = "") -> list[str]:
        # Whether image conditioning is *required* in the cache is a model-local
        # fact declared by the pipeline through its capability contract.
        # via REQUIRED_BATCH_KEYS_BY_STRATEGY. The strategy stays family-agnostic.
        return []

    def validate_config(self) -> list[str]:
        p = self._p()
        if p < 0.0 or p > 1.0:
            return ["first_frame_conditioning_p must be in [0, 1]."]
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
            noisy = objective.corrupt(
                pipeline=pipeline, clean_latents=latents, noise=noise, timesteps=timesteps
            )
        else:
            noisy = pipeline.apply_noise(latents, noise, timesteps)

        p = self._p()
        condition_frame_indexes: list[list[int]] = [[] for _ in range(batch_size)]
        if is_torch_tensor(latents):
            noisy_latents = noisy.clone()
            loss_mask = torch.ones_like(latents, dtype=latents.dtype)
            frame_dim = int(pipeline.i2v_conditioning_frame_dim(latents=latents))
            if p >= 1.0 or (not training and p > 0.0):
                _condition_frame(
                    noisy_latents,
                    latents,
                    loss_mask,
                    frame_dim=frame_dim,
                )
                condition_frame_indexes = [[0] for _ in range(batch_size)]
            elif p <= 0.0:
                pass
            else:
                cond = (torch.rand((batch_size,)) < p).to(device=latents.device)
                for idx in range(batch_size):
                    if bool(cond[idx]):
                        _condition_frame(
                            noisy_latents,
                            latents,
                            loss_mask,
                            frame_dim=frame_dim,
                            batch_index=idx,
                        )
                        condition_frame_indexes[idx] = [0]
        else:
            noisy_latents = noisy
            loss_mask = [1.0] * len(latents)

        extra = {}
        if "clip_embed" in batch:
            extra["clip_embed"] = batch["clip_embed"]
        extra.update(
            pipeline.i2v_conditioning_forward_kwargs(
                condition_frame_indexes=condition_frame_indexes
            )
        )
        model_timesteps = pipeline.prepare_model_timesteps(
            timesteps,
            latents=latents,
        )
        return TrainingInputs(
            noisy_latents=noisy_latents,
            timestep=model_timesteps,
            noise=noise,
            clean_latents=latents,
            text_embeds=_text_payload(batch),
            objective_timestep=(
                None if model_timesteps is timesteps else timesteps
            ),
            loss_mask=loss_mask,
            extra_forward_kwargs=extra,
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
            if mask is None:
                return raw.reshape(raw.shape[0], -1).mean(dim=1)
            weighted = raw * mask
            denom = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp(min=1.0)
            return weighted.reshape(weighted.shape[0], -1).sum(dim=1) / denom
        if inputs.loss_mask is None:
            return raw
        return [
            value * mask
            for value, mask in zip(raw, inputs.loss_mask, strict=True)
        ]
