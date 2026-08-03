"""Training-policy adapter for dataset-domain MoE routing."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.adaptation.dataset_routing import DatasetRoutingPolicy
from mirai.core.moe.adaptation.dataset_routing import ROUTING_DOMAINS_BATCH_KEY
from mirai.core.training.training_policy import BatchAugmentContext
from mirai.core.training.training_policy import RecordSelectionContext
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


def _validate_dataset_routing_config(config: Any) -> list[str]:
    policy = DatasetRoutingPolicy.from_config(config.dataset.moe_routing)
    errors = list(policy.validation_errors())
    if policy.uses_affinity:
        provider = get_model_family_provider(config.model.type)
        if provider is None or not provider.supports_dataset_routing_policy(config):
            errors.append(
                f"model.type '{config.model.type}' does not support dataset "
                "routing affinity."
            )
        if float(config.model.params.expert_subset_fraction) != 1.0:
            errors.append(
                "dataset routing affinity cannot be combined with stochastic "
                "expert-subset routing because their expert masks may have fewer "
                "than top_k entries."
            )
    if (
        policy.specialization_mode == "domain_balanced"
        and bool(config.dataset.online_temporal_resampling)
    ):
        errors.append(
            "dataset.moe_routing domain_balanced cannot be combined with "
            "dataset.online_temporal_resampling because both own sample selection."
        )
    return errors


class DatasetRoutingTrainingPolicy(TrainingPolicy):
    name = "dataset_routing"
    priority = 100

    def __init__(self, policy: DatasetRoutingPolicy) -> None:
        self.policy = policy

    def configure_pipeline(self, pipeline: Any) -> None:
        if self.policy.uses_affinity:
            pipeline.configure_dataset_routing(self.policy)

    def validate_records(self, records) -> list[str]:
        return self.policy.validate_records(records)

    def select_records(
        self, context: RecordSelectionContext
    ) -> list[Any] | None:
        if self.policy.specialization_mode != "domain_balanced":
            return None
        return self.policy.select_domain_balanced_records(
            context.records,
            batch_size=context.batch_size,
            base_seed=context.base_seed,
            global_batch_index=context.global_batch_index,
        )

    def augment_batch(self, context: BatchAugmentContext) -> Mapping[str, Any]:
        routing_batch = self.policy.batch_from_records(
            context.records,
            step=context.step,
            training=context.training,
        )
        if routing_batch is None:
            return {}
        return {ROUTING_DOMAINS_BATCH_KEY: list(routing_batch.domains)}

    def before_forward(
        self,
        pipeline: Any,
        batch: Mapping[str, Any],
        *,
        training: bool,
    ) -> None:
        self.policy.bind_batch(pipeline, batch, training=training)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "mode": self.policy.specialization_mode,
            "domain_metadata_key": self.policy.domain_metadata_key,
            "expert_affinity": {
                domain: list(expert_ids)
                for domain, expert_ids in self.policy.expert_affinity.items()
            },
            "routing_prior_weight": self.policy.routing_prior_weight,
            "router_warmup_steps": self.policy.router_warmup_steps,
        }


@register_training_policy(
    "dataset_routing",
    validate_config=_validate_dataset_routing_config,
)
def build_dataset_routing_training_policy(
    config: Any,
) -> DatasetRoutingTrainingPolicy | None:
    policy = DatasetRoutingPolicy.from_config(config.dataset.moe_routing)
    if policy.is_disabled:
        return None
    return DatasetRoutingTrainingPolicy(policy)
