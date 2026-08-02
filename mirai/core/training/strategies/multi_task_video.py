"""Homogeneous-microbatch T2I/T2V/I2V training strategy."""

from __future__ import annotations

from typing import Any, Callable

from mirai.config.schema import StrategyConfig
from mirai.core.models.base import BasePipeline
from mirai.core.registry import register_strategy
from mirai.core.training.data.curriculum import TRAINING_TASKS
from mirai.core.training.objectives.sampling import NoiseGenerator, TimestepSampler
from mirai.core.training.strategies.base import TrainingInputs, TrainingStrategy
from mirai.core.training.strategies.image_to_video import ImageToVideoStrategy
from mirai.core.training.strategies.text_to_video import TextToVideoStrategy


@register_strategy("multi_task_video")
class MultiTaskVideoStrategy(TrainingStrategy):
    """Delegate each curriculum-selected microbatch to its task strategy.

    Task selection belongs to the data curriculum. This class only translates
    the resulting homogeneous ``training_task`` batch into model inputs, keeping
    the task-mixture mechanism independent of any model family.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self._text = TextToVideoStrategy(config)
        self._image = ImageToVideoStrategy(config)

    def get_config_schema(self) -> set[str]:
        return {
            "conditioning_dropout_p",
            "empty_text_embed",
            "first_frame_conditioning_p",
        }

    def validate_config(self) -> list[str]:
        return [
            *self._text.validate_config(),
            *self._image.validate_config(),
        ]

    @staticmethod
    def _task(batch: dict[str, Any]) -> str:
        task = str(batch.get("training_task", "")).strip().lower()
        if task not in TRAINING_TASKS:
            raise ValueError(
                "multi_task_video requires a homogeneous batch with "
                f"training_task in {list(TRAINING_TASKS)}; got {task!r}."
            )
        return task

    @staticmethod
    def _validate_text_to_image_latents(
        batch: dict[str, Any],
        pipeline: BasePipeline,
    ) -> None:
        latents = batch["latents"]
        ndim = int(getattr(latents, "ndim", 0))
        frame_dim = int(pipeline.i2v_conditioning_frame_dim(latents=latents))
        if frame_dim <= 0 or frame_dim >= ndim or int(latents.shape[frame_dim]) != 1:
            raise ValueError(
                "text_to_image batches must contain exactly one latent frame."
            )

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
        task = self._task(batch)
        if task == "image_to_video":
            return self._image.prepare_inputs(
                batch,
                pipeline,
                timestep_sampler,
                noise_generator,
                training=training,
                objective=objective,
            )
        if task == "text_to_image":
            self._validate_text_to_image_latents(batch, pipeline)
        return self._text.prepare_inputs(
            batch,
            pipeline,
            timestep_sampler,
            noise_generator,
            training=training,
            objective=objective,
        )

    def compute_per_sample_loss(
        self,
        prediction: Any,
        target: Any,
        inputs: TrainingInputs,
        loss_fn: Callable[[Any, Any], Any],
    ) -> Any:
        # TextToVideoStrategy's reduction supports both ordinary masks and the
        # dense first-frame mask produced by ImageToVideoStrategy.
        return self._text.compute_per_sample_loss(
            prediction,
            target,
            inputs,
            loss_fn,
        )
