"""Training-policy adapter for SharpMoE saliency routing."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.routing.saliency import SharpMoESpec
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


_OPTION_KEYS = frozenset(
    {"enabled", "trajectory_steps", "router_hidden_dim", "seed"}
)


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get("sharp_moe", {})


def validate_sharp_moe_config(config: Any) -> list[str]:
    options = _options(config)
    errors = [
        f"unknown option '{name}'" for name in sorted(set(options) - _OPTION_KEYS)
    ]
    if not bool(options.get("enabled", False)):
        if str(config.training.objective).strip().lower() == "sharp_moe_trajectory":
            errors.append(
                "training.objective='sharp_moe_trajectory' requires "
                "training.policy_options.sharp_moe.enabled=true"
            )
        return errors
    try:
        SharpMoESpec(
            trajectory_steps=int(options.get("trajectory_steps", 10)),
            router_hidden_dim=int(options.get("router_hidden_dim", 128)),
            seed=int(options.get("seed", config.training.seed)),
        ).validate()
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    if str(config.training.objective).strip().lower() != "sharp_moe_trajectory":
        errors.append("requires training.objective='sharp_moe_trajectory'")
    if str(config.training.loss_function).strip().lower() != "mse":
        errors.append("requires training.loss_function='mse'")
    if str(config.training.loss_weighting).strip().lower() != "uniform":
        errors.append("requires training.loss_weighting='uniform'")
    if str(config.training.loss_bucket_normalization).strip().lower() != "none":
        errors.append("requires training.loss_bucket_normalization='none'")
    if float(config.training.contrastive_flow_weight) != 0.0:
        errors.append("cannot be combined with contrastive flow matching")
    if float(config.training.prior_ratio) != 0.0:
        errors.append("cannot be combined with prior-preservation batches")

    params = config.model.params
    if str(params.moe_routing_mode).strip().lower() != "token_choice":
        errors.append("requires model.params.moe_routing_mode='token_choice'")
    if str(params.moe_balance_mode).strip().lower() != "off":
        errors.append("requires model.params.moe_balance_mode='off'")
    for name in (
        "moe_aux_loss_weight",
        "moe_router_z_loss_weight",
        "moe_phi_balance_weight",
        "moe_router_similarity_loss_weight",
        "moe_router_variance_loss_weight",
        "moe_expert_orthogonality_loss_weight",
        "moe_swiglu_specialization_loss_weight",
        "moe_cross_layer_coupling_loss_weight",
        "moe_spatiotemporal_routing_weight",
    ):
        if float(getattr(params, name)) != 0.0:
            errors.append(f"requires model.params.{name}=0")
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
        "dispersive_loss",
        "expert_dropout",
        "prototypical_routing",
        "router_distillation",
        "router_stage_schedule",
        "selective_sinkhorn",
        "simbal",
    ):
        if bool(policies.get(name, {}).get("enabled", False)):
            errors.append(f"cannot be combined with training policy '{name}'")
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_sharp_moe_policy(config):
        errors.append(
            f"model.type '{config.model.type}' does not support SharpMoE routing"
        )
    return errors


class SharpMoETrainingPolicy(TrainingPolicy):
    name = "sharp_moe"
    priority = 113

    def __init__(self, spec: SharpMoESpec) -> None:
        self.spec = spec.validate()

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_training_policy(self.name, self.spec)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "trajectory_steps": int(self.spec.trajectory_steps),
            "router_hidden_dim": int(self.spec.router_hidden_dim),
            "seed": int(self.spec.seed),
            "score_composition": "pretrained_plus_saliency_mlp",
            "guidance": "previous_predicted_clean_latent",
        }


@register_training_policy("sharp_moe", validate_config=validate_sharp_moe_config)
def build_sharp_moe_training_policy(config: Any) -> SharpMoETrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return SharpMoETrainingPolicy(
        SharpMoESpec(
            trajectory_steps=int(options.get("trajectory_steps", 10)),
            router_hidden_dim=int(options.get("router_hidden_dim", 128)),
            seed=int(options.get("seed", config.training.seed)),
        )
    )


__all__ = [
    "build_sharp_moe_training_policy",
    "SharpMoETrainingPolicy",
    "validate_sharp_moe_config",
]
