"""Training-policy adapter for similarity-preserving router balancing."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.adaptation.simbal import SimBalController, SimBalSpec
from mirai.core.training.training_policy import (
    TrainingPolicy,
    register_training_policy,
)


_OPTION_KEYS = frozenset({"enabled", "weight"})


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get("simbal", {})


def validate_simbal_config(config: Any) -> list[str]:
    options = _options(config)
    unknown = sorted(set(options) - _OPTION_KEYS)
    errors = [f"unknown option '{name}'" for name in unknown]
    if not bool(options.get("enabled", False)):
        return errors
    try:
        SimBalSpec(weight=float(options.get("weight", 0.1))).validate()
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    if getattr(config.adapter, "train_router", None) is False:
        errors.append("adapter.train_router=false conflicts with SIMBAL")
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_simbal_policy(config):
        errors.append(f"model.type '{config.model.type}' does not support SIMBAL")
    return errors


class SimBalTrainingPolicy(TrainingPolicy):
    name = "simbal"
    priority = 126

    def __init__(self, controller: SimBalController) -> None:
        self.controller = controller

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_training_policy(self.name, self.controller)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {"weight": float(self.controller.spec.weight), "norm": "entrywise_l1"}


@register_training_policy("simbal", validate_config=validate_simbal_config)
def build_simbal_training_policy(config: Any) -> SimBalTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return SimBalTrainingPolicy(
        SimBalController(SimBalSpec(weight=float(options.get("weight", 0.1))))
    )


__all__ = [
    "build_simbal_training_policy",
    "SimBalTrainingPolicy",
    "validate_simbal_config",
]
