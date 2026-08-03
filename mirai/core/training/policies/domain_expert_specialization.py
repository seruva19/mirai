"""Training-policy adapter for online domain-to-expert specialization."""

from __future__ import annotations

from typing import Any, Mapping

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.adaptation.domain_specialization import (
    DomainExpertSpecializationController,
)
from mirai.core.training.training_policy import BatchAugmentContext
from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


DOMAIN_EXPERT_DOMAINS_BATCH_KEY = "_mirai_domain_expert_domains"
DOMAIN_EXPERT_STEP_BATCH_KEY = "_mirai_domain_expert_step"


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get(
        "domain_expert_specialization", {}
    )


def _record_value(record: Any, key: str) -> Any:
    value = record.get(key) if hasattr(record, "get") else None
    if value is not None:
        return value
    metadata = record.get("metadata") if hasattr(record, "get") else None
    return metadata.get(key) if isinstance(metadata, Mapping) else None


def _validate_config(config: Any) -> list[str]:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return []
    errors: list[str] = []
    if int(options.get("warmup_steps", 100)) <= 0:
        errors.append("warmup_steps must be > 0")
    threshold = float(options.get("affinity_threshold", 0.5))
    if not 0.0 < threshold <= 1.0:
        errors.append("affinity_threshold must be in (0, 1]")
    if int(options.get("min_experts", 1)) <= 0:
        errors.append("min_experts must be > 0")
    momentum = float(options.get("momentum", 0.9))
    if not 0.0 <= momentum < 1.0:
        errors.append("momentum must be in [0, 1)")
    if int(options.get("update_interval", 1)) <= 0:
        errors.append("update_interval must be > 0")
    if not str(options.get("domain_metadata_key", "")).strip():
        errors.append("domain_metadata_key must be non-empty")
    if (
        float(config.optimizer.weight_decay) != 0.0
        and str(config.optimizer.type).strip().lower() != "adamw"
    ):
        errors.append("nonzero weight decay requires optimizer.type='adamw'")
    provider = get_model_family_provider(config.model.type)
    if provider is None or not provider.supports_domain_expert_specialization(config):
        errors.append(
            f"model.type '{config.model.type}' does not support domain expert specialization"
        )
    return errors


class DomainExpertSpecializationTrainingPolicy(TrainingPolicy):
    name = "domain_expert_specialization"
    priority = 125

    def __init__(
        self, controller: DomainExpertSpecializationController, *, metadata_key: str
    ) -> None:
        self.controller = controller
        self.metadata_key = str(metadata_key)

    def configure_pipeline(self, pipeline: Any) -> None:
        pipeline.configure_domain_expert_specialization(self.controller)

    def validate_records(self, records) -> list[str]:
        missing = sum(
            not str(_record_value(record, self.metadata_key) or "").strip()
            for record in records
        )
        return (
            [f"{missing} record(s) lack domain metadata '{self.metadata_key}'"]
            if missing
            else []
        )

    def augment_batch(self, context: BatchAugmentContext) -> Mapping[str, Any]:
        domains = [
            str(_record_value(record, self.metadata_key) or "").strip()
            for record in context.records
        ]
        return {
            DOMAIN_EXPERT_DOMAINS_BATCH_KEY: domains,
            DOMAIN_EXPERT_STEP_BATCH_KEY: int(context.step),
        }

    def before_forward(
        self, pipeline: Any, batch: Mapping[str, Any], *, training: bool
    ) -> None:
        _ = pipeline
        self.controller.bind_batch(
            domains=tuple(batch[DOMAIN_EXPERT_DOMAINS_BATCH_KEY]),
            step=int(batch[DOMAIN_EXPERT_STEP_BATCH_KEY]),
            training=bool(training),
        )

    def before_optimizer_step(self, optimizer: Any) -> None:
        self.controller.before_optimizer_step(optimizer)

    def after_optimizer_step(self, optimizer: Any, *, applied: bool) -> None:
        self.controller.after_optimizer_step(optimizer, applied=applied)

    def state_dict(self) -> Mapping[str, Any]:
        return self.controller.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.controller.load_state_dict(state)

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "domain_metadata_key": self.metadata_key,
            **self.controller.checkpoint_metadata(),
        }


@register_training_policy(
    "domain_expert_specialization", validate_config=_validate_config
)
def build_domain_expert_specialization_training_policy(
    config: Any,
) -> DomainExpertSpecializationTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return DomainExpertSpecializationTrainingPolicy(
        DomainExpertSpecializationController(
            warmup_steps=int(options.get("warmup_steps", 100)),
            affinity_threshold=float(options.get("affinity_threshold", 0.5)),
            min_experts=int(options.get("min_experts", 1)),
            momentum=float(options.get("momentum", 0.9)),
            update_interval=int(options.get("update_interval", 1)),
        ),
        metadata_key=str(options.get("domain_metadata_key", "")).strip(),
    )
