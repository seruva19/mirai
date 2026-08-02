"""Sparse-MoE capability and source contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SparseMoECapabilities:
    """Public contract for models that are true sparse MoE denoisers."""

    is_sparse_moe: bool = False
    architecture: str = ""
    routing: str = ""
    routing_granularity: str = ""
    num_routed_experts: int = 0
    num_shared_experts: int = 0
    num_activated_experts: int = 0
    emits_router_metrics: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoutingStats:
    """Serializable routing diagnostics for one MoE layer."""

    layer: str
    tokens: int
    selected_tokens: int
    expert_fraction: tuple[float, ...]
    mean_router_probability: tuple[float, ...]
    dead_experts: int
    routing_entropy: float

    @property
    def tokens_per_expert(self) -> tuple[int, ...]:
        return tuple(
            int(round(float(fraction) * float(self.selected_tokens)))
            for fraction in self.expert_fraction
        )

    @property
    def probability_mass_per_expert(self) -> tuple[float, ...]:
        return tuple(
            float(probability) * float(self.tokens)
            for probability in self.mean_router_probability
        )

    @property
    def load_cv(self) -> float:
        loads = tuple(float(value) for value in self.expert_fraction)
        if not loads:
            return 0.0
        mean = sum(loads) / len(loads)
        if mean <= 0.0:
            return 0.0
        variance = sum((value - mean) ** 2 for value in loads) / len(loads)
        return variance**0.5 / mean

    @property
    def max_to_mean_load(self) -> float:
        loads = tuple(float(value) for value in self.expert_fraction)
        if not loads:
            return 0.0
        mean = sum(loads) / len(loads)
        return max(loads) / mean if mean > 0.0 else 0.0

    @property
    def active_expert_fraction(self) -> float:
        if not self.expert_fraction:
            return 0.0
        return float(len(self.expert_fraction) - self.dead_experts) / float(
            len(self.expert_fraction)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "tokens": self.tokens,
            "selected_tokens": self.selected_tokens,
            "expert_fraction": list(self.expert_fraction),
            "mean_router_probability": list(self.mean_router_probability),
            "tokens_per_expert": list(self.tokens_per_expert),
            "probability_mass_per_expert": list(self.probability_mass_per_expert),
            "dead_experts": self.dead_experts,
            "active_expert_fraction": self.active_expert_fraction,
            "load_cv": self.load_cv,
            "max_to_mean_load": self.max_to_mean_load,
            "routing_entropy": self.routing_entropy,
        }


@dataclass(frozen=True)
class OpenSparseMoEModelSpec:
    """Tracked external MoE diffusion/model source."""

    model_id: str
    display_name: str
    modality: str
    status: str
    routing: str
    source_urls: tuple[str, ...]
    license: str
    source_code_license: str = "unknown"
    artifact_license: str = "unknown"
    native_model_type: str = ""
    integration_level: str = "planned_fail_fast"
    integration_blockers: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
