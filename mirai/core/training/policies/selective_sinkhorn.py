"""Training-policy adapter for Selective Sinkhorn Routing."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.routing.selective_sinkhorn import (
    SelectiveSinkhornController,
    SelectiveSinkhornSpec,
)
from mirai.core.training.training_policy import BatchAugmentContext
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


SELECTIVE_SINKHORN_BATCH_INDEX_KEY = "_mirai_selective_sinkhorn_batch_index"
_OPTION_KEYS = frozenset(
    {
        "enabled",
        "probability",
        "cost_mode",
        "entropy_regularization",
        "max_iterations",
        "tolerance",
        "noise_scale",
        "seed",
    }
)


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get(
        "selective_sinkhorn", {}
    )


def validate_selective_sinkhorn_config(config: Any) -> list[str]:
    options = _options(config)
    errors = [
        f"unknown option '{name}'" for name in sorted(set(options) - _OPTION_KEYS)
    ]
    if not bool(options.get("enabled", False)):
        return errors
    try:
        SelectiveSinkhornSpec(
            probability=float(options.get("probability", 0.001)),
            cost_mode=str(options.get("cost_mode", "softmax")),
            entropy_regularization=float(
                options.get("entropy_regularization", 0.05)
            ),
            max_iterations=int(options.get("max_iterations", 100)),
            tolerance=float(options.get("tolerance", 1e-4)),
            noise_scale=float(options.get("noise_scale", 0.0)),
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
    if str(config.dataset.moe_routing.specialization_mode) != "emergent":
        errors.append("cannot be combined with dataset routing affinity")
    policies = getattr(config.training, "policy_options", {})
    for name in ("diversity_routing", "expert_dropout", "simbal"):
        if bool(policies.get(name, {}).get("enabled", False)):
            errors.append(f"cannot be combined with training policy '{name}'")
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_selective_sinkhorn_policy(config):
        errors.append(
            f"model.type '{config.model.type}' does not support Selective Sinkhorn"
        )
    return errors


class SelectiveSinkhornTrainingPolicy(TrainingPolicy):
    name = "selective_sinkhorn"
    priority = 111

    def __init__(self, controller: SelectiveSinkhornController) -> None:
        self.controller = controller

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_selective_sinkhorn(self.controller)

    def augment_batch(self, context: BatchAugmentContext) -> Mapping[str, Any]:
        index = (
            int(context.global_batch_index)
            if context.global_batch_index is not None
            else int(context.step)
        )
        return {SELECTIVE_SINKHORN_BATCH_INDEX_KEY: index}

    def before_forward(
        self,
        pipeline: Any,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> None:
        _ = pipeline
        if SELECTIVE_SINKHORN_BATCH_INDEX_KEY not in batch:
            raise ValueError(
                "Selective Sinkhorn batch is missing its absolute batch index."
            )
        self.controller.bind_batch(
            global_batch_index=int(batch[SELECTIVE_SINKHORN_BATCH_INDEX_KEY]),
            training=bool(training),
        )

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        spec = self.controller.spec
        return {
            "probability": float(spec.probability),
            "cost_mode": str(spec.cost_mode),
            "entropy_regularization": float(spec.entropy_regularization),
            "max_iterations": int(spec.max_iterations),
            "tolerance": float(spec.tolerance),
            "noise_scale": float(spec.noise_scale),
            "seed": int(spec.seed),
            "branch_rng": "blake2b(global_batch_index,layer_name)",
        }


@register_training_policy(
    "selective_sinkhorn", validate_config=validate_selective_sinkhorn_config
)
def build_selective_sinkhorn_training_policy(
    config: Any,
) -> SelectiveSinkhornTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return SelectiveSinkhornTrainingPolicy(
        SelectiveSinkhornController(
            SelectiveSinkhornSpec(
                probability=float(options.get("probability", 0.001)),
                cost_mode=str(options.get("cost_mode", "softmax")),
                entropy_regularization=float(
                    options.get("entropy_regularization", 0.05)
                ),
                max_iterations=int(options.get("max_iterations", 100)),
                tolerance=float(options.get("tolerance", 1e-4)),
                noise_scale=float(options.get("noise_scale", 0.0)),
                seed=int(options.get("seed", config.training.seed)),
            )
        )
    )


__all__ = [
    "build_selective_sinkhorn_training_policy",
    "SELECTIVE_SINKHORN_BATCH_INDEX_KEY",
    "SelectiveSinkhornTrainingPolicy",
    "validate_selective_sinkhorn_config",
]
