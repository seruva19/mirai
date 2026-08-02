"""Training strategy abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from mirai.core.models.base import BasePipeline
from mirai.core.training.objectives.sampling import NoiseGenerator, TimestepSampler


@dataclass
class TrainingInputs:
    noisy_latents: Any
    timestep: Any
    noise: Any
    clean_latents: Any
    text_embeds: dict[str, Any]
    objective_timestep: Any = None
    loss_mask: Any = None
    extra_forward_kwargs: dict[str, Any] = field(default_factory=dict)


class TrainingStrategy(ABC):
    @abstractmethod
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
        """Prepare model inputs for one train step.

        ``objective`` (a ``TrainingObjective``) defines the forward/corruption
        process; strategies should build the model input via ``objective.corrupt``
        so non-diffusion objectives work. Falls back to ``pipeline.apply_noise``
        when not provided.
        """

    @abstractmethod
    def compute_per_sample_loss(
        self,
        prediction: Any,
        target: Any,
        inputs: TrainingInputs,
        loss_fn: Callable[[Any, Any], Any],
    ) -> Any:
        """Return per-sample losses before global weighting/reduction."""

    def validate_config(self) -> list[str]:
        return []

    def get_config_schema(self) -> set[str]:
        """Allowed keys under [strategy.params]."""
        return set()

    def validate_params(self) -> list[str]:
        allowed = self.get_config_schema()
        params = getattr(self, "config", None)
        payload = getattr(params, "params", {}) if params is not None else {}
        if not isinstance(payload, dict):
            return ["strategy.params must be a table."]
        unknown = sorted([key for key in payload if key not in allowed])
        if not unknown:
            return []
        return [
            "Unknown strategy.params keys: "
            + ", ".join(unknown)
            + (f". Allowed: {', '.join(sorted(allowed))}" if allowed else ".")
        ]

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        params = getattr(self, "config", None)
        strategy_type = getattr(params, "type", "")
        strategy_params = getattr(params, "params", {})
        return {
            "strategy_type": strategy_type,
            "strategy_params": strategy_params,
        }

    def get_extra_batch_keys(self) -> list[str]:
        return []

    def get_required_batch_keys(self, *, model_type: str = "") -> list[str]:
        _ = model_type
        return []
