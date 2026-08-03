"""Training-policy adapter for residual ProMoE prototype guidance."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.routing.prototypical import PrototypicalRoutingSpec
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


_OPTION_KEYS = frozenset(
    {
        "enabled",
        "prototype_scale",
        "contrastive_weight",
        "contrastive_temperature",
        "seed",
    }
)


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get(
        "prototypical_routing", {}
    )


def validate_prototypical_routing_config(config: Any) -> list[str]:
    options = _options(config)
    errors = [
        f"unknown option '{name}'" for name in sorted(set(options) - _OPTION_KEYS)
    ]
    if not bool(options.get("enabled", False)):
        return errors
    try:
        PrototypicalRoutingSpec(
            prototype_scale=float(options.get("prototype_scale", 1.0)),
            contrastive_weight=float(options.get("contrastive_weight", 1.0)),
            contrastive_temperature=float(
                options.get("contrastive_temperature", 0.07)
            ),
            seed=int(options.get("seed", config.training.seed)),
        ).validate()
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    params = config.model.params
    if str(params.moe_routing_mode).strip().lower() != "token_choice":
        errors.append("requires model.params.moe_routing_mode='token_choice'")
    if str(params.moe_balance_mode).strip().lower() != "off":
        errors.append("requires model.params.moe_balance_mode='off'")
    if float(params.moe_router_z_loss_weight) != 0.0:
        errors.append("requires model.params.moe_router_z_loss_weight=0")
    if float(params.moe_phi_balance_weight) != 0.0:
        errors.append("cannot be combined with population phi balancing")
    if float(params.expert_subset_fraction) != 1.0:
        errors.append("cannot be combined with stochastic expert-subset routing")
    if int(params.moe_dynamic_topk_min) != 0:
        errors.append("cannot be combined with compute-budgeted dynamic top-k")
    if int(params.moe_lightweight_top_k) != 0:
        errors.append("cannot be combined with lightweight experts")
    if bool(params.moe_chain_of_experts):
        errors.append("cannot be combined with Chain-of-Experts routing")
    if str(config.dataset.moe_routing.specialization_mode) != "emergent":
        errors.append("cannot be combined with dataset routing affinity")

    policies = getattr(config.training, "policy_options", {})
    for name in (
        "diversity_routing",
        "expert_dropout",
        "router_temperature",
        "selective_sinkhorn",
        "simbal",
    ):
        if bool(policies.get(name, {}).get("enabled", False)):
            errors.append(f"cannot be combined with training policy '{name}'")
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_prototypical_routing_policy(
        config
    ):
        errors.append(
            f"model.type '{config.model.type}' does not support prototypical routing"
        )
    return errors


class PrototypicalRoutingTrainingPolicy(TrainingPolicy):
    name = "prototypical_routing"
    priority = 112

    def __init__(self, spec: PrototypicalRoutingSpec) -> None:
        self.spec = spec.validate()

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_prototypical_routing(self.spec)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "prototype_scale": float(self.spec.prototype_scale),
            "contrastive_weight": float(self.spec.contrastive_weight),
            "contrastive_temperature": float(
                self.spec.contrastive_temperature
            ),
            "residual_scale_init": 0.0,
            "seed": int(self.spec.seed),
            "score_composition": "native_plus_beta_cosine",
            "scope": "provider_selected_visual_tokens",
        }


@register_training_policy(
    "prototypical_routing", validate_config=validate_prototypical_routing_config
)
def build_prototypical_routing_training_policy(
    config: Any,
) -> PrototypicalRoutingTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return PrototypicalRoutingTrainingPolicy(
        PrototypicalRoutingSpec(
            prototype_scale=float(options.get("prototype_scale", 1.0)),
            contrastive_weight=float(options.get("contrastive_weight", 1.0)),
            contrastive_temperature=float(
                options.get("contrastive_temperature", 0.07)
            ),
            seed=int(options.get("seed", config.training.seed)),
        )
    )


__all__ = [
    "build_prototypical_routing_training_policy",
    "PrototypicalRoutingTrainingPolicy",
    "validate_prototypical_routing_config",
]
