"""Sparse MoE routing primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from mirai.core.moe.routing.contracts import (
    RoutingStats,
    assignment_distribution_entropy,
)
from mirai.core.moe.routing.expert_choice import ExpertChoiceRoutingPolicy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RoutingDecision:
    topk_indices: Any
    topk_weights: Any
    load_balance_loss: Any
    z_loss: Any
    stats: RoutingStats


@dataclass(frozen=True)
class ExpertChoiceRoutingDecision:
    expert_token_indices: Any
    expert_token_weights: Any
    router_probabilities: Any
    load_balance_loss: Any
    z_loss: Any
    stats: RoutingStats
    coverage: ExpertChoiceCoverageStats


@dataclass(frozen=True)
class ExpertChoiceCoverageStats:
    """Serializable token-coverage diagnostics for expert-choice routing."""

    capacity_per_expert: int
    per_sample_capacity: tuple[int, ...]
    covered_tokens: int
    uncovered_tokens: int
    multiply_selected_tokens: int
    max_experts_per_token: int
    mean_experts_per_token: float
    per_sample_covered_tokens: tuple[int, ...]

    @property
    def coverage_fraction(self) -> float:
        total = self.covered_tokens + self.uncovered_tokens
        return float(self.covered_tokens) / float(total) if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "capacity_per_expert": self.capacity_per_expert,
            "per_sample_capacity": list(self.per_sample_capacity),
            "covered_tokens": self.covered_tokens,
            "uncovered_tokens": self.uncovered_tokens,
            "coverage_fraction": self.coverage_fraction,
            "multiply_selected_tokens": self.multiply_selected_tokens,
            "max_experts_per_token": self.max_experts_per_token,
            "mean_experts_per_token": self.mean_experts_per_token,
            "per_sample_covered_tokens": list(self.per_sample_covered_tokens),
        }


def route_expert_choice_logits(
    logits: Any,
    *,
    capacity_factor: float,
    capacity_per_sample: Any | None = None,
    route_scale: float,
    layer_name: str,
    output_dtype: Any,
    z_loss_weight: float = 0.0,
) -> ExpertChoiceRoutingDecision:
    """Build canonical per-sample Expert-Choice routes from router logits."""
    if torch is None or F is None:  # pragma: no cover
        raise RuntimeError("Expert-Choice routing requires torch.")
    if logits.ndim != 3:
        raise ValueError("Expert-Choice logits must have shape [batch, tokens, experts].")
    batch, seq_len, num_experts = logits.shape
    if int(num_experts) <= 0 or float(capacity_factor) <= 0.0:
        raise ValueError("Expert-Choice requires experts and capacity_factor > 0.")
    scores = F.softmax(logits.float(), dim=-1)
    static_capacity = min(
        int(seq_len),
        max(1, math.ceil(int(seq_len) * float(capacity_factor) / int(num_experts))),
    )
    if capacity_per_sample is None:
        sample_capacities = torch.full(
            (batch,),
            static_capacity,
            device=logits.device,
            dtype=torch.int64,
        )
    else:
        sample_capacities = torch.as_tensor(
            capacity_per_sample,
            device=logits.device,
            dtype=torch.int64,
        )
        if tuple(sample_capacities.shape) != (int(batch),):
            raise ValueError(
                "capacity_per_sample must have shape [batch] matching logits."
            )
        if bool(
            ((sample_capacities < 1) | (sample_capacities > int(seq_len))).any()
        ):
            raise ValueError(
                "capacity_per_sample values must be in [1, tokens_per_sample]."
            )
    capacity = int(sample_capacities.max().item())
    expert_scores = scores.transpose(1, 2).contiguous()
    expert_token_weights, expert_token_indices = torch.topk(
        expert_scores, k=capacity, dim=-1, sorted=False
    )
    active_slots = (
        torch.arange(capacity, device=logits.device)[None, None, :]
        < sample_capacities[:, None, None]
    )
    expert_token_weights = expert_token_weights * active_slots
    token_totals = torch.zeros(
        (batch, seq_len), device=logits.device, dtype=expert_token_weights.dtype
    )
    token_totals.scatter_add_(
        1,
        expert_token_indices.reshape(batch, -1),
        expert_token_weights.reshape(batch, -1),
    )
    expert_token_weights = expert_token_weights / torch.gather(
        token_totals, 1, expert_token_indices.reshape(batch, -1)
    ).reshape_as(expert_token_weights).clamp_min(1e-20)
    expert_token_weights = expert_token_weights * float(route_scale)
    token_multiplicity = torch.zeros(
        (batch, seq_len), device=logits.device, dtype=torch.int64
    )
    token_multiplicity.scatter_add_(
        1,
        expert_token_indices.reshape(batch, -1),
        active_slots.expand(batch, num_experts, capacity)
        .reshape(batch, -1)
        .to(torch.int64),
    )
    expert_counts = active_slots.expand(
        batch, num_experts, capacity
    ).sum(dim=(0, 2)).float()
    expert_fraction = expert_counts / expert_counts.sum().clamp_min(1.0)
    expert_fraction_values = tuple(
        float(value) for value in expert_fraction.detach().cpu().tolist()
    )
    mean_probability = scores.mean(dim=(0, 1))
    zero = logits.sum() * 0.0
    log_z = torch.logsumexp(logits.float(), dim=-1)
    z_loss = (
        log_z.square().mean() * float(z_loss_weight)
        if float(z_loss_weight)
        else zero
    )
    covered = token_multiplicity > 0
    return ExpertChoiceRoutingDecision(
        expert_token_indices=expert_token_indices,
        expert_token_weights=expert_token_weights.to(dtype=output_dtype),
        router_probabilities=scores,
        load_balance_loss=zero.to(dtype=output_dtype),
        z_loss=z_loss.to(dtype=output_dtype),
        stats=RoutingStats(
            layer=str(layer_name),
            tokens=int(batch * seq_len),
            selected_tokens=int(
                int(num_experts) * int(sample_capacities.sum().item())
            ),
            expert_fraction=expert_fraction_values,
            mean_router_probability=tuple(
                float(v) for v in mean_probability.detach().cpu()
            ),
            dead_experts=int((expert_counts == 0).sum().item()),
            routing_entropy=assignment_distribution_entropy(expert_fraction_values),
        ),
        coverage=ExpertChoiceCoverageStats(
            capacity_per_expert=capacity,
            per_sample_capacity=tuple(
                int(value)
                for value in sample_capacities.detach().cpu().tolist()
            ),
            covered_tokens=int(covered.sum().item()),
            uncovered_tokens=int((~covered).sum().item()),
            multiply_selected_tokens=int((token_multiplicity > 1).sum().item()),
            max_experts_per_token=int(token_multiplicity.max().item()),
            mean_experts_per_token=float(token_multiplicity.float().mean().item()),
            per_sample_covered_tokens=tuple(
                int(value) for value in covered.sum(dim=1).detach().cpu().tolist()
            ),
        ),
    )


class TokenChoiceRouter(nn.Module):
    """Top-k token-choice router with Switch-style balance diagnostics."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        experts_per_token: int,
        layer_name: str,
        z_loss_weight: float = 0.0,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("TokenChoiceRouter requires torch.")
        super().__init__()
        if int(num_experts) <= 0:
            raise ValueError("num_experts must be > 0.")
        if int(experts_per_token) <= 0 or int(experts_per_token) > int(num_experts):
            raise ValueError("experts_per_token must be in [1, num_experts].")
        self.num_experts = int(num_experts)
        self.experts_per_token = int(experts_per_token)
        self.layer_name = str(layer_name)
        self.z_loss_weight = float(z_loss_weight)
        self.gate = nn.Linear(int(hidden_size), self.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states: Any) -> RoutingDecision:
        if torch is None or F is None:  # pragma: no cover
            raise RuntimeError("TokenChoiceRouter requires torch.")
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, tokens, hidden].")
        batch, seq_len, hidden = hidden_states.shape
        flat = hidden_states.reshape(batch * seq_len, hidden)
        logits = self.gate(flat).float()
        scores = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(
            scores,
            k=self.experts_per_token,
            dim=-1,
            sorted=False,
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        selected = F.one_hot(topk_indices.reshape(-1), num_classes=self.num_experts).float()
        expert_fraction = selected.mean(dim=0)
        expert_fraction_values = tuple(
            float(value) for value in expert_fraction.detach().cpu().tolist()
        )
        mean_probability = scores.mean(dim=0)
        load_balance = self.num_experts * torch.sum(expert_fraction * mean_probability)
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = (log_z.square().mean() * self.z_loss_weight) if self.z_loss_weight else logits.sum() * 0.0
        stats = RoutingStats(
            layer=self.layer_name,
            tokens=int(batch * seq_len),
            selected_tokens=int(batch * seq_len * self.experts_per_token),
            expert_fraction=expert_fraction_values,
            mean_router_probability=tuple(float(v) for v in mean_probability.detach().cpu().tolist()),
            dead_experts=int((expert_fraction.detach() == 0).sum().item()),
            routing_entropy=assignment_distribution_entropy(expert_fraction_values),
        )
        return RoutingDecision(
            topk_indices=topk_indices,
            topk_weights=topk_weights.to(dtype=hidden_states.dtype),
            load_balance_loss=load_balance.to(dtype=hidden_states.dtype),
            z_loss=z_loss.to(dtype=hidden_states.dtype),
            stats=stats,
        )


class ExpertChoiceRouter(nn.Module):
    """Per-sample expert-choice router with token-normalized gate weights."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float,
        layer_name: str,
        z_loss_weight: float = 0.0,
        policy: ExpertChoiceRoutingPolicy | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("ExpertChoiceRouter requires torch.")
        super().__init__()
        if int(num_experts) <= 0:
            raise ValueError("num_experts must be > 0.")
        if float(capacity_factor) <= 0.0:
            raise ValueError("capacity_factor must be > 0.")
        self.num_experts = int(num_experts)
        self.capacity_factor = float(capacity_factor)
        self.layer_name = str(layer_name)
        self.z_loss_weight = float(z_loss_weight)
        self.policy = policy or ExpertChoiceRoutingPolicy()
        gate_input_size = int(hidden_size) + int(self.policy.timestep_hidden_size)
        self.gate = nn.Linear(gate_input_size, self.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(
        self,
        hidden_states: Any,
        *,
        timestep_embedding: Any | None = None,
        capacity_factor: float | None = None,
        capacity_per_sample: Any | None = None,
    ) -> ExpertChoiceRoutingDecision:
        if torch is None or F is None:  # pragma: no cover
            raise RuntimeError("ExpertChoiceRouter requires torch.")
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, tokens, hidden].")
        router_input = self.policy.build_router_input(hidden_states, timestep_embedding)
        logits = self.gate(router_input).float()
        active_capacity_factor = (
            self.capacity_factor if capacity_factor is None else float(capacity_factor)
        )
        return route_expert_choice_logits(
            logits,
            capacity_factor=active_capacity_factor,
            capacity_per_sample=capacity_per_sample,
            route_scale=float(self.policy.route_scale),
            layer_name=self.layer_name,
            output_dtype=hidden_states.dtype,
            z_loss_weight=self.z_loss_weight,
        )
