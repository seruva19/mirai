"""Dataset-domain sampling and routed-expert affinity policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Mapping, Sequence


ALLOWED_MOE_DATASET_SPECIALIZATION_MODES = frozenset(
    {"emergent", "domain_balanced", "soft_affinity", "hard_affinity"}
)
ROUTING_DOMAINS_BATCH_KEY = "_moe_routing_domains"


def _record_value(record: Any, key: str) -> Any:
    value = record.get(key) if hasattr(record, "get") else None
    if value is not None:
        return value
    metadata = record.get("metadata") if hasattr(record, "get") else None
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    return None


def _stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


@dataclass(frozen=True)
class DatasetRoutingBatch:
    domains: tuple[str, ...]
    step: int
    training: bool


@dataclass(frozen=True)
class DatasetRoutingPolicy:
    specialization_mode: str
    domain_metadata_key: str
    expert_affinity: Mapping[str, tuple[int, ...]]
    routing_prior_weight: float
    router_warmup_steps: int

    @classmethod
    def from_config(cls, config: Any) -> DatasetRoutingPolicy:
        affinity = {
            str(domain).strip(): tuple(int(expert_id) for expert_id in expert_ids)
            for domain, expert_ids in dict(config.expert_affinity).items()
        }
        return cls(
            specialization_mode=str(config.specialization_mode).strip().lower(),
            domain_metadata_key=str(config.domain_metadata_key).strip(),
            expert_affinity=affinity,
            routing_prior_weight=float(config.routing_prior_weight),
            router_warmup_steps=int(config.router_warmup_steps),
        )

    @property
    def is_disabled(self) -> bool:
        return self.specialization_mode == "emergent"

    @property
    def uses_affinity(self) -> bool:
        return self.specialization_mode in {"soft_affinity", "hard_affinity"}

    def validation_errors(self, *, require_executable: bool = True) -> list[str]:
        _ = require_executable
        mode = self.specialization_mode
        errors: list[str] = []
        if mode not in ALLOWED_MOE_DATASET_SPECIALIZATION_MODES:
            errors.append(
                "dataset.moe_routing.specialization_mode must be one of: "
                "emergent, domain_balanced, soft_affinity, hard_affinity."
            )
        if mode != "emergent" and not self.domain_metadata_key:
            errors.append(
                "dataset.moe_routing.domain_metadata_key is required when "
                "specialization_mode is not 'emergent'."
            )
        if mode in {"soft_affinity", "hard_affinity"} and not self.expert_affinity:
            errors.append(
                "dataset.moe_routing.expert_affinity is required for affinity modes."
            )
        if mode == "soft_affinity" and self.routing_prior_weight <= 0.0:
            errors.append(
                "dataset.moe_routing.routing_prior_weight must be > 0 for "
                "soft_affinity mode."
            )
        if self.routing_prior_weight < 0.0:
            errors.append(
                "dataset.moe_routing.routing_prior_weight must be >= 0."
            )
        if self.router_warmup_steps < 0:
            errors.append("dataset.moe_routing.router_warmup_steps must be >= 0.")
        for domain, expert_ids in self.expert_affinity.items():
            if not domain:
                errors.append(
                    "dataset.moe_routing.expert_affinity domain names must be non-empty."
                )
            if not expert_ids:
                errors.append(
                    "dataset.moe_routing.expert_affinity groups must not be empty; "
                    f"domain '{domain}' has no experts."
                )
            if any(expert_id < 0 for expert_id in expert_ids):
                errors.append(
                    "dataset.moe_routing.expert_affinity expert ids must be >= 0; "
                    f"domain '{domain}' is invalid."
                )
            if len(set(expert_ids)) != len(expert_ids):
                errors.append(
                    "dataset.moe_routing.expert_affinity expert ids must be unique within "
                    f"domain '{domain}'."
                )
        if mode == "emergent" and self.expert_affinity:
            errors.append(
                "dataset.moe_routing.expert_affinity would be ignored in emergent mode; "
                "select soft_affinity or hard_affinity explicitly."
            )
        if mode == "domain_balanced" and self.expert_affinity:
            errors.append(
                "dataset.moe_routing.expert_affinity is not used by domain_balanced "
                "sampling; select an affinity mode or remove the mapping."
            )
        return errors

    def validate_model_contract(self, *, num_experts: int, top_k: int) -> list[str]:
        if not self.uses_affinity:
            return []
        errors: list[str] = []
        for domain, expert_ids in self.expert_affinity.items():
            invalid = [value for value in expert_ids if value >= int(num_experts)]
            if invalid:
                errors.append(
                    f"dataset.moe_routing.expert_affinity domain '{domain}' contains "
                    f"expert ids outside [0, {int(num_experts)}): {invalid}."
                )
            if self.specialization_mode == "hard_affinity" and len(expert_ids) < int(top_k):
                errors.append(
                    f"dataset.moe_routing hard_affinity domain '{domain}' provides "
                    f"{len(expert_ids)} experts but the model routes top_k={int(top_k)}."
                )
        return errors

    def domain_for_record(self, record: Any) -> str:
        domain = str(_record_value(record, self.domain_metadata_key) or "").strip()
        if not domain:
            sample_id = str(_record_value(record, "sample_id") or "<unknown>")
            raise ValueError(
                f"Cache record '{sample_id}' is missing non-empty dataset routing "
                f"metadata '{self.domain_metadata_key}'."
            )
        if self.uses_affinity and domain not in self.expert_affinity:
            raise ValueError(
                f"Dataset routing domain '{domain}' has no expert_affinity entry."
            )
        return domain

    def validate_records(self, records: Sequence[Any]) -> list[str]:
        if self.is_disabled:
            return []
        errors: list[str] = []
        for record in records:
            try:
                self.domain_for_record(record)
            except ValueError as exc:
                errors.append(str(exc))
                if len(errors) >= 8:
                    break
        return errors

    def batch_from_records(
        self,
        records: Sequence[Any],
        *,
        step: int,
        training: bool,
    ) -> DatasetRoutingBatch | None:
        if self.is_disabled or not training or not self.uses_affinity:
            return None
        return DatasetRoutingBatch(
            domains=tuple(self.domain_for_record(record) for record in records),
            step=max(0, int(step)),
            training=True,
        )

    def bind_batch(self, pipeline: Any, batch: Mapping[str, Any], *, training: bool) -> None:
        if not self.uses_affinity:
            return
        raw_domains = batch.get(ROUTING_DOMAINS_BATCH_KEY)
        domains = tuple(str(value).strip() for value in (raw_domains or ()))
        if self.uses_affinity and training and not domains:
            raise ValueError(
                f"Training batch is missing '{ROUTING_DOMAINS_BATCH_KEY}' for "
                f"dataset routing mode '{self.specialization_mode}'."
            )
        pipeline.set_training_policy_context(
            "dataset_routing",
            DatasetRoutingBatch(
                domains=domains,
                step=max(0, int(batch.get("_step", 0))),
                training=bool(training),
            ),
        )

    def warmup_scale(self, step: int) -> float:
        if self.router_warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, float(step) / float(self.router_warmup_steps)))

    def hard_affinity_active(self, step: int) -> bool:
        return self.specialization_mode == "hard_affinity" and (
            self.router_warmup_steps <= 0 or int(step) >= self.router_warmup_steps
        )

    def select_domain_balanced_records(
        self,
        records: Sequence[Any],
        *,
        batch_size: int,
        base_seed: int,
        global_batch_index: int,
    ) -> list[Any]:
        if self.specialization_mode != "domain_balanced":
            raise ValueError("Domain-balanced selection requires domain_balanced mode.")
        groups: dict[str, list[Any]] = {}
        for record in records:
            groups.setdefault(self.domain_for_record(record), []).append(record)
        if not groups:
            raise ValueError("Cannot sample from an empty dataset routing domain map.")
        domains = sorted(groups)
        size = max(1, int(batch_size))
        absolute_start = max(0, int(global_batch_index)) * size
        chosen: list[Any] = []
        for offset in range(size):
            absolute_position = absolute_start + offset
            domain_cycle = absolute_position // len(domains)
            domain_order = list(domains)
            random.Random(int(base_seed) + domain_cycle * 100_003).shuffle(domain_order)
            domain = domain_order[absolute_position % len(domains)]
            occurrence = absolute_position // len(domains)
            pool = groups[domain]
            pool_cycle = occurrence // len(pool)
            pool_order = list(range(len(pool)))
            seed = int(base_seed) + _stable_seed(domain) + pool_cycle * 1_000_003
            random.Random(seed).shuffle(pool_order)
            chosen.append(pool[pool_order[occurrence % len(pool)]])
        return chosen
