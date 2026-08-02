"""Training-policy adapter for frozen-reference router distillation."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.adaptation.distillation import RouterDistillationController
from mirai.core.training.training_policy import BatchAugmentContext
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


ROUTER_DISTILLATION_STEP_BATCH_KEY = "_mirai_router_distillation_step"


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get("router_distillation", {})


def _validate_config(config: Any) -> list[str]:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return []
    weight = float(options.get("weight", 0.01))
    temperature = float(options.get("temperature", 1.0))
    start = int(options.get("start_step", 0))
    end = int(options.get("end_step", 0))
    weight_schedule = str(options.get("weight_schedule", "constant")).strip().lower()
    errors: list[str] = []
    if weight <= 0.0 or temperature <= 0.0:
        errors.append("weight and temperature must be positive")
    if start < 0 or end < 0 or (end and end <= start):
        errors.append("schedule requires non-negative steps and end_step > start_step")
    if weight_schedule not in {"constant", "linear_decay"}:
        errors.append("weight_schedule must be 'constant' or 'linear_decay'")
    if weight_schedule == "linear_decay" and end == 0:
        errors.append("weight_schedule='linear_decay' requires end_step")
    if getattr(config.adapter, "train_router", None) is False:
        errors.append("adapter.train_router=false conflicts with router distillation")
    stage = getattr(config.training, "policy_options", {}).get(
        "router_stage_schedule", {}
    )
    if bool(stage.get("enabled", False)):
        stage_start = int(stage.get("train_start_step", 0))
        stage_end = int(stage.get("freeze_step", 0))
        starts_too_early = start < stage_start
        ends_too_late = stage_end and (end == 0 or end > stage_end)
        if starts_too_early or ends_too_late:
            errors.append(
                "distillation window must stay inside the staged router "
                "adaptation window"
            )
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_router_distillation(config):
        errors.append(f"model.type '{config.model.type}' does not support router distillation")
    return errors


class RouterDistillationTrainingPolicy(TrainingPolicy):
    name = "router_distillation"
    priority = 125

    def __init__(self, controller: RouterDistillationController) -> None:
        self.controller = controller

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_training_policy(self.name, self.controller)

    def augment_batch(self, context: BatchAugmentContext) -> Mapping[str, Any]:
        return {ROUTER_DISTILLATION_STEP_BATCH_KEY: int(context.step)}

    def before_forward(
        self, pipeline: Any, batch: Mapping[str, Any], *, training: bool
    ) -> None:
        _ = pipeline
        if ROUTER_DISTILLATION_STEP_BATCH_KEY not in batch:
            raise ValueError("Router distillation batch is missing its step context.")
        self.controller.bind_step(
            step=int(batch[ROUTER_DISTILLATION_STEP_BATCH_KEY]),
            training=bool(training),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return self.controller.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.controller.load_state_dict(state)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "weight": self.controller.weight,
            "temperature": self.controller.temperature,
            "teacher_fingerprint": self.controller.teacher_fingerprint,
            "weight_schedule": self.controller.weight_schedule,
        }


@register_training_policy("router_distillation", validate_config=_validate_config)
def build_router_distillation_training_policy(
    config: Any,
) -> RouterDistillationTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return RouterDistillationTrainingPolicy(
        RouterDistillationController(
            weight=float(options.get("weight", 0.01)),
            temperature=float(options.get("temperature", 1.0)),
            start_step=int(options.get("start_step", 0)),
            end_step=int(options.get("end_step", 0)),
            weight_schedule=str(options.get("weight_schedule", "constant")),
        )
    )


__all__ = [
    "ROUTER_DISTILLATION_STEP_BATCH_KEY",
    "RouterDistillationTrainingPolicy",
    "build_router_distillation_training_policy",
]
