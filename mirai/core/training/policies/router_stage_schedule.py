"""Training-policy adapter for staged sparse-MoE router adaptation."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.adaptation.stage_schedule import RouterStageScheduleController
from mirai.core.training.training_policy import BatchAugmentContext
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


ROUTER_STAGE_STEP_BATCH_KEY = "_mirai_router_stage_step"


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get(
        "router_stage_schedule", {}
    )


def _validate_config(config: Any) -> list[str]:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return []
    start = int(options.get("train_start_step", 0))
    freeze = int(options.get("freeze_step", 0))
    errors: list[str] = []
    if start < 0 or freeze < 0:
        errors.append("schedule steps must be non-negative")
    if freeze and freeze <= start:
        errors.append("freeze_step must exceed train_start_step")
    if getattr(config.adapter, "train_router", None) is False:
        errors.append("adapter.train_router=false conflicts with staged router training")
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_router_stage_schedule(config):
        errors.append(
            f"model.type '{config.model.type}' does not support router stage schedules"
        )
    return errors


class RouterStageScheduleTrainingPolicy(TrainingPolicy):
    name = "router_stage_schedule"
    priority = 130

    def __init__(self, controller: RouterStageScheduleController) -> None:
        self.controller = controller

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_training_policy(self.name, self.controller)

    def augment_batch(self, context: BatchAugmentContext) -> Mapping[str, Any]:
        return {ROUTER_STAGE_STEP_BATCH_KEY: int(context.step)}

    def before_forward(
        self,
        pipeline: Any,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> None:
        _ = pipeline
        if ROUTER_STAGE_STEP_BATCH_KEY not in batch:
            raise ValueError("Router stage schedule batch is missing its step context.")
        self.controller.apply_step(
            step=int(batch[ROUTER_STAGE_STEP_BATCH_KEY]),
            training=bool(training),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return self.controller.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.controller.load_state_dict(state)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "train_start_step": self.controller.train_start_step,
            "freeze_step": self.controller.freeze_step,
        }


@register_training_policy("router_stage_schedule", validate_config=_validate_config)
def build_router_stage_schedule_training_policy(
    config: Any,
) -> RouterStageScheduleTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return RouterStageScheduleTrainingPolicy(
        RouterStageScheduleController(
            train_start_step=int(options.get("train_start_step", 0)),
            freeze_step=int(options.get("freeze_step", 0)),
        )
    )


__all__ = [
    "ROUTER_STAGE_STEP_BATCH_KEY",
    "RouterStageScheduleTrainingPolicy",
    "build_router_stage_schedule_training_policy",
]
