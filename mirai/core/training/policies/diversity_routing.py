"""Lifecycle adapter for single-GPU diversity-aware MoE routing."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.adaptation.diversity import DiversityAwareRoutingController
from mirai.core.training.training_policy import BatchAugmentContext
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


DIVERSITY_ROUTING_STEP_BATCH_KEY = "_mirai_diversity_routing_step"


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get(
        "diversity_routing", {}
    )


def _validate_diversity_routing_config(config: Any) -> list[str]:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return []
    errors: list[str] = []
    if int(options.get("warmup_steps", 100)) <= 0:
        errors.append("warmup_steps must be positive")
    if float(options.get("ridge", 1e-4)) <= 0.0:
        errors.append("ridge must be positive")
    if int(options.get("token_chunk_size", 2048)) <= 0:
        errors.append("token_chunk_size must be positive")
    if int(config.model.params.experts_per_token) < 2:
        errors.append("model.params.experts_per_token must be at least 2")
    if float(config.model.params.expert_subset_fraction) != 1.0:
        errors.append("cannot be combined with stochastic expert-subset routing")
    if str(config.dataset.moe_routing.specialization_mode) != "emergent":
        errors.append("cannot be combined with dataset routing affinity")
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_diversity_routing_policy(config):
        errors.append(
            f"model.type '{config.model.type}' does not support diversity routing"
        )
    return errors


class DiversityRoutingTrainingPolicy(TrainingPolicy):
    name = "diversity_routing"
    priority = 110

    def __init__(self, controller: DiversityAwareRoutingController) -> None:
        self.controller = controller

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_diversity_routing(self.controller)

    def augment_batch(self, context: BatchAugmentContext) -> Mapping[str, Any]:
        return {DIVERSITY_ROUTING_STEP_BATCH_KEY: int(context.step)}

    def before_forward(
        self,
        pipeline: Any,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> None:
        _ = pipeline
        if DIVERSITY_ROUTING_STEP_BATCH_KEY not in batch:
            raise ValueError("Diversity routing batch is missing its step context.")
        self.controller.bind_step(
            step=int(batch[DIVERSITY_ROUTING_STEP_BATCH_KEY]),
            training=bool(training),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return self.controller.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.controller.load_state_dict(state)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "warmup_steps": self.controller.warmup_steps,
            "ridge": self.controller.ridge,
            "num_experts": self.controller.num_experts,
            "top_k": self.controller.top_k,
            "token_chunk_size": self.controller.token_chunk_size,
        }


@register_training_policy(
    "diversity_routing",
    validate_config=_validate_diversity_routing_config,
)
def build_diversity_routing_training_policy(
    config: Any,
) -> DiversityRoutingTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return DiversityRoutingTrainingPolicy(
        DiversityAwareRoutingController(
            num_experts=int(config.model.params.num_experts),
            top_k=int(config.model.params.experts_per_token),
            warmup_steps=int(options.get("warmup_steps", 100)),
            ridge=float(options.get("ridge", 1e-4)),
            token_chunk_size=int(options.get("token_chunk_size", 2048)),
        )
    )


__all__ = [
    "DIVERSITY_ROUTING_STEP_BATCH_KEY",
    "DiversityRoutingTrainingPolicy",
    "build_diversity_routing_training_policy",
]
