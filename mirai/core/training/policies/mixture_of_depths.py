"""Training-policy adapter for attention-routed Mixture-of-Depths."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.routing.depth import MixtureOfDepthsSpec
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


_OPTION_KEYS = frozenset(
    {
        "enabled",
        "capacity_fraction",
        "first_layer",
        "layer_stride",
        "attention_query_chunk_size",
    }
)


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get(
        "mixture_of_depths", {}
    )


def _spec(config: Any) -> MixtureOfDepthsSpec:
    options = _options(config)
    return MixtureOfDepthsSpec(
        capacity_fraction=float(options.get("capacity_fraction", 0.5)),
        first_layer=int(options.get("first_layer", 1)),
        layer_stride=int(options.get("layer_stride", 2)),
        attention_query_chunk_size=int(
            options.get("attention_query_chunk_size", 128)
        ),
    ).validate()


def validate_mixture_of_depths_config(config: Any) -> list[str]:
    options = _options(config)
    errors = [
        f"unknown option '{name}'" for name in sorted(set(options) - _OPTION_KEYS)
    ]
    if not bool(options.get("enabled", False)):
        return errors
    try:
        _spec(config)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_mixture_of_depths_policy(config):
        errors.append(
            f"model.type '{config.model.type}' does not support Mixture-of-Depths"
        )
    return errors


class MixtureOfDepthsTrainingPolicy(TrainingPolicy):
    name = "mixture_of_depths"
    priority = 114

    def __init__(self, spec: MixtureOfDepthsSpec) -> None:
        self.spec = spec.validate()

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_training_policy(self.name, self.spec)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "capacity_fraction": float(self.spec.capacity_fraction),
            "first_layer": int(self.spec.first_layer),
            "layer_stride": int(self.spec.layer_stride),
            "attention_query_chunk_size": int(
                self.spec.attention_query_chunk_size
            ),
            "routing": "previous_layer_received_attention",
            "capacity_scope": "provider_selected_visual_tokens",
        }


@register_training_policy(
    "mixture_of_depths",
    validate_config=validate_mixture_of_depths_config,
)
def build_mixture_of_depths_training_policy(
    config: Any,
) -> MixtureOfDepthsTrainingPolicy | None:
    if not bool(_options(config).get("enabled", False)):
        return None
    return MixtureOfDepthsTrainingPolicy(_spec(config))


__all__ = [
    "build_mixture_of_depths_training_policy",
    "MixtureOfDepthsTrainingPolicy",
    "validate_mixture_of_depths_config",
]
