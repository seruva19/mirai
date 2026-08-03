"""Training-policy state for mechanism-driven Q/K update monitoring."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.monitoring.preemptive import PreemptiveAttentionMonitor
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


def validate_preemptive_monitoring_config(config: Any) -> list[str]:
    if not bool(getattr(config.model.params, "moe_routing_health", False)):
        return []
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_preemptive_monitoring(config):
        return [
            f"model.type '{config.model.type}' does not expose mechanism-driven "
            "router and Q/K monitoring"
        ]
    return []


class PreemptiveMonitoringTrainingPolicy(TrainingPolicy):
    name = "preemptive_monitoring"
    priority = 115

    def __init__(self, monitor: PreemptiveAttentionMonitor | None = None) -> None:
        self.monitor = monitor or PreemptiveAttentionMonitor()

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_preemptive_monitoring(self.monitor)

    def state_dict(self) -> Mapping[str, Any]:
        return self.monitor.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.monitor.load_state_dict(state)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "attention_operator": "first_order_qk_product_increment",
            "spectral_alpha": 2,
            "window": "one_applied_effective_weight_update",
        }


@register_training_policy(
    "preemptive_monitoring",
    validate_config=validate_preemptive_monitoring_config,
)
def build_preemptive_monitoring_policy(
    config: Any,
) -> PreemptiveMonitoringTrainingPolicy | None:
    if not bool(getattr(config.model.params, "moe_routing_health", False)):
        return None
    return PreemptiveMonitoringTrainingPolicy()


__all__ = [
    "build_preemptive_monitoring_policy",
    "PreemptiveMonitoringTrainingPolicy",
    "validate_preemptive_monitoring_config",
]
