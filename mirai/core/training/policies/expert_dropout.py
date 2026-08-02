"""Training-policy adapter for sparse-MoE expert route dropout."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.adaptation.dropout import ExpertDropoutController
from mirai.core.training.training_policy import BatchAugmentContext
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


EXPERT_DROPOUT_STEP_BATCH_KEY = "_mirai_expert_dropout_step"


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get("expert_dropout", {})


def _validate_expert_dropout_config(config: Any) -> list[str]:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return []
    errors: list[str] = []
    probability = float(options.get("probability", 0.4))
    start_step = int(options.get("start_step", 0))
    end_step = int(options.get("end_step", 0))
    if not 0.0 < probability < 1.0:
        errors.append("probability must be between 0 and 1")
    if start_step < 0 or end_step < 0:
        errors.append("schedule steps must be non-negative")
    if end_step and end_step <= start_step:
        errors.append("end_step must exceed start_step")
    if int(config.model.params.experts_per_token) < 2:
        errors.append("model.params.experts_per_token must be at least 2")
    if float(config.model.params.moe_expert_orthogonality_loss_weight) != 0.0:
        errors.append("cannot be combined with expert-output orthogonality")
    if float(config.model.params.moe_swiglu_specialization_loss_weight) != 0.0:
        errors.append("cannot be combined with SwiGLU specialization")
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_expert_dropout_policy(config):
        errors.append(
            f"model.type '{config.model.type}' does not support expert dropout"
        )
    return errors


class ExpertDropoutTrainingPolicy(TrainingPolicy):
    name = "expert_dropout"
    priority = 120

    def __init__(self, controller: ExpertDropoutController) -> None:
        self.controller = controller

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_training_policy(self.name, self.controller)

    def augment_batch(self, context: BatchAugmentContext) -> Mapping[str, Any]:
        return {EXPERT_DROPOUT_STEP_BATCH_KEY: int(context.step)}

    def before_forward(
        self,
        pipeline: Any,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> None:
        _ = pipeline
        if EXPERT_DROPOUT_STEP_BATCH_KEY not in batch:
            raise ValueError("Expert dropout batch is missing its step context.")
        self.controller.bind_step(
            step=int(batch[EXPERT_DROPOUT_STEP_BATCH_KEY]),
            training=bool(training),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return self.controller.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.controller.load_state_dict(state)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "probability": self.controller.probability,
            "start_step": self.controller.start_step,
            "end_step": self.controller.end_step,
        }


@register_training_policy(
    "expert_dropout",
    validate_config=_validate_expert_dropout_config,
)
def build_expert_dropout_training_policy(
    config: Any,
) -> ExpertDropoutTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return ExpertDropoutTrainingPolicy(
        ExpertDropoutController(
            probability=float(options.get("probability", 0.4)),
            start_step=int(options.get("start_step", 0)),
            end_step=int(options.get("end_step", 0)),
        )
    )


__all__ = [
    "EXPERT_DROPOUT_STEP_BATCH_KEY",
    "ExpertDropoutTrainingPolicy",
    "build_expert_dropout_training_policy",
]
