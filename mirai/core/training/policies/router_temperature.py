"""Training-policy adapter for model-agnostic router temperature controls."""

from __future__ import annotations

import math
from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.adaptation.temperature import (
    ROUTER_TEMPERATURE_SCHEDULES,
    RouterTemperatureController,
)
from mirai.core.training.training_policy import BatchAugmentContext
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


ROUTER_TEMPERATURE_STEP_BATCH_KEY = "_mirai_router_temperature_step"


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get(
        "router_temperature", {}
    )


def _validate_router_temperature_config(config: Any) -> list[str]:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return []
    errors: list[str] = []
    temperature = float(options.get("temperature", 1.0))
    minimum = float(options.get("minimum_temperature", temperature))
    schedule = str(options.get("schedule", "constant")).strip().lower()
    start_step = int(options.get("start_step", 0))
    end_step = int(options.get("end_step", 0))
    sharpness = float(options.get("sigmoid_sharpness", 7.0))
    jitter = float(options.get("jitter_epsilon", 0.0))
    entropy_floor = float(options.get("entropy_floor", 0.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        errors.append("temperature must be finite and > 0")
    if (
        not math.isfinite(minimum)
        or minimum <= 0.0
        or minimum > temperature
    ):
        errors.append(
            "minimum_temperature must be finite, > 0, and <= temperature"
        )
    if schedule not in ROUTER_TEMPERATURE_SCHEDULES:
        errors.append("schedule must be constant, linear, or sigmoid")
    if start_step < 0 or end_step < 0:
        errors.append("schedule steps must be non-negative")
    if schedule != "constant" and end_step <= start_step:
        errors.append("annealed schedules require end_step > start_step")
    if not math.isfinite(sharpness) or sharpness <= 0.0:
        errors.append("sigmoid_sharpness must be finite and > 0")
    if not math.isfinite(jitter) or not 0.0 <= jitter < 1.0:
        errors.append("jitter_epsilon must be finite and in [0, 1)")
    num_experts = int(config.model.params.num_experts)
    max_entropy = math.log(num_experts) if num_experts > 0 else 0.0
    if (
        not math.isfinite(entropy_floor)
        or entropy_floor < 0.0
        or entropy_floor > max_entropy
    ):
        errors.append(
            "entropy_floor must be finite and in [0, log(num_experts)]"
        )
    provider = get_model_family_provider(config.model.type)
    if (
        provider is None
        or not provider.supports_router_temperature_policy(config)
    ):
        errors.append(
            f"model.type '{config.model.type}' does not support router temperature"
        )
    return errors


class RouterTemperatureTrainingPolicy(TrainingPolicy):
    name = "router_temperature"
    priority = 110

    def __init__(self, controller: RouterTemperatureController) -> None:
        self.controller = controller

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_router_temperature(self.controller)

    def augment_batch(self, context: BatchAugmentContext) -> Mapping[str, Any]:
        return {ROUTER_TEMPERATURE_STEP_BATCH_KEY: int(context.step)}

    def before_forward(
        self,
        pipeline: Any,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> None:
        _ = pipeline
        if ROUTER_TEMPERATURE_STEP_BATCH_KEY not in batch:
            raise ValueError(
                "Router temperature batch is missing its step context."
            )
        self.controller.bind_step(
            step=int(batch[ROUTER_TEMPERATURE_STEP_BATCH_KEY]),
            training=bool(training),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return self.controller.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.controller.load_state_dict(state)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "temperature": self.controller.temperature,
            "minimum_temperature": self.controller.minimum_temperature,
            "schedule": self.controller.schedule,
            "start_step": self.controller.start_step,
            "end_step": self.controller.end_step,
            "sigmoid_sharpness": self.controller.sigmoid_sharpness,
            "jitter_epsilon": self.controller.jitter_epsilon,
            "entropy_floor": self.controller.entropy_floor,
        }


@register_training_policy(
    "router_temperature",
    validate_config=_validate_router_temperature_config,
)
def build_router_temperature_training_policy(
    config: Any,
) -> RouterTemperatureTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    temperature = float(options.get("temperature", 1.0))
    return RouterTemperatureTrainingPolicy(
        RouterTemperatureController(
            temperature=temperature,
            minimum_temperature=float(
                options.get("minimum_temperature", temperature)
            ),
            schedule=str(options.get("schedule", "constant")),
            start_step=int(options.get("start_step", 0)),
            end_step=int(options.get("end_step", 0)),
            sigmoid_sharpness=float(
                options.get("sigmoid_sharpness", 7.0)
            ),
            jitter_epsilon=float(options.get("jitter_epsilon", 0.0)),
            entropy_floor=float(options.get("entropy_floor", 0.0)),
        )
    )


__all__ = [
    "ROUTER_TEMPERATURE_STEP_BATCH_KEY",
    "RouterTemperatureTrainingPolicy",
    "build_router_temperature_training_policy",
]
