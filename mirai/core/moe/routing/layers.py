"""Plain PyTorch sparse MoE layers used by native MoE pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.moe.routing.contracts import RoutingStats
from mirai.core.moe.routing.dispatch import ExpertChoiceDispatchHost
from mirai.core.moe.routing.dispatch import validate_expert_choice_dispatch_host
from mirai.core.moe.routing.expert_choice import ExpertChoiceRoutingPolicy
from mirai.core.moe.routing.routers import ExpertChoiceRouter, TokenChoiceRouter

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


@dataclass(frozen=True)
class SparseMoEOutput:
    hidden_states: Any
    auxiliary_loss: Any
    stats: RoutingStats
    expert_choice_coverage: Any | None = None


class SwiGLUExpert(nn.Module):
    def __init__(self, *, hidden_size: int, intermediate_size: int) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("SwiGLUExpert requires torch.")
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_states: Any) -> Any:
        return self.down_proj(self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class SparseMoEFeedForward(nn.Module):
    """Top-k sparse FFN with optional always-on shared experts."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_routed_experts: int,
        num_shared_experts: int,
        experts_per_token: int,
        layer_name: str,
        load_balance_weight: float = 0.01,
        router_z_loss_weight: float = 0.0,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("SparseMoEFeedForward requires torch.")
        super().__init__()
        self.num_routed_experts = int(num_routed_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.experts_per_token = int(experts_per_token)
        self.load_balance_weight = float(load_balance_weight)
        self.router = TokenChoiceRouter(
            hidden_size=int(hidden_size),
            num_experts=self.num_routed_experts,
            experts_per_token=self.experts_per_token,
            layer_name=layer_name,
            z_loss_weight=float(router_z_loss_weight),
        )
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    hidden_size=int(hidden_size),
                    intermediate_size=int(intermediate_size),
                )
                for _ in range(self.num_routed_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    hidden_size=int(hidden_size),
                    intermediate_size=int(intermediate_size),
                )
                for _ in range(max(self.num_shared_experts, 0))
            ]
        )

    def forward(self, hidden_states: Any) -> SparseMoEOutput:
        decision = self.router(hidden_states)
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, orig_shape[-1])
        topk_indices = decision.topk_indices.reshape(-1)
        topk_weights = decision.topk_weights.reshape(-1, 1)
        repeated = flat.repeat_interleave(self.experts_per_token, dim=0)
        routed = torch.zeros_like(repeated)
        for expert_idx, expert in enumerate(self.experts):
            mask = topk_indices == expert_idx
            if bool(mask.any()):
                # Autocast-safe: under bf16 autocast the expert emits bf16
                # while the fp32 scatter buffer keeps its dtype (no-op cast
                # when dtypes already match).
                routed[mask] = expert(repeated[mask]).to(routed.dtype)
        routed = routed * topk_weights
        routed = routed.reshape(flat.shape[0], self.experts_per_token, -1).sum(dim=1)
        output = routed.reshape(orig_shape)
        for expert in self.shared_experts:
            output = output + expert(hidden_states)
        auxiliary = (
            decision.load_balance_loss * self.load_balance_weight
            + decision.z_loss
        )
        return SparseMoEOutput(
            hidden_states=output,
            auxiliary_loss=auxiliary,
            stats=decision.stats,
        )


class ExpertChoiceMoEFeedForward(nn.Module):
    """Expert-choice sparse FFN with optional always-on shared experts."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_routed_experts: int,
        num_shared_experts: int,
        capacity_factor: float,
        layer_name: str,
        load_balance_weight: float = 0.01,
        router_z_loss_weight: float = 0.0,
        layer_index: int = 0,
        routing_policy: ExpertChoiceRoutingPolicy | None = None,
        expert_host: ExpertChoiceDispatchHost | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("ExpertChoiceMoEFeedForward requires torch.")
        super().__init__()
        self.num_routed_experts = int(num_routed_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.load_balance_weight = float(load_balance_weight)
        self.layer_index = int(layer_index)
        self.routing_policy = routing_policy or ExpertChoiceRoutingPolicy()
        self.router = ExpertChoiceRouter(
            hidden_size=int(hidden_size),
            num_experts=self.num_routed_experts,
            capacity_factor=float(capacity_factor),
            layer_name=layer_name,
            z_loss_weight=float(router_z_loss_weight),
            policy=self.routing_policy,
        )
        self.expert_host = (
            None
            if expert_host is None
            else validate_expert_choice_dispatch_host(
                expert_host, expected_experts=self.num_routed_experts
            )
        )
        self.experts = nn.ModuleList(
            []
            if expert_host is not None
            else [
                SwiGLUExpert(
                    hidden_size=int(hidden_size),
                    intermediate_size=int(intermediate_size),
                )
                for _ in range(self.num_routed_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    hidden_size=int(hidden_size),
                    intermediate_size=int(intermediate_size),
                )
                for _ in range(max(self.num_shared_experts, 0))
            ]
        )

    def forward(
        self,
        hidden_states: Any,
        *,
        routing_hidden_states: Any | None = None,
        timestep_embedding: Any | None = None,
        training_stage: str | None = None,
    ) -> SparseMoEOutput:
        router_states = hidden_states if routing_hidden_states is None else routing_hidden_states
        if router_states.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "routing_hidden_states and hidden_states must share batch/token dimensions."
            )
        capacity_factor = self.routing_policy.resolve_capacity_factor(
            layer_index=self.layer_index,
            stage=training_stage,
            fallback=self.router.capacity_factor,
        )
        decision = self.router(
            router_states,
            timestep_embedding=timestep_embedding,
            capacity_factor=capacity_factor,
        )
        orig_shape = hidden_states.shape
        batch, seq_len, hidden_size = orig_shape
        flat = hidden_states.reshape(-1, hidden_size)
        if self.expert_host is not None:
            routed = self.expert_host.run_expert_choice_routed(
                flat,
                decision.expert_token_weights,
                decision.expert_token_indices,
                tokens_per_sample=int(seq_len),
            )
        else:
            routed = torch.zeros_like(flat)
            batch_offsets = (
                torch.arange(batch, device=flat.device, dtype=torch.long) * seq_len
            ).reshape(batch, 1)
            for expert_idx, expert in enumerate(self.experts):
                indexes = (
                    decision.expert_token_indices[:, expert_idx, :] + batch_offsets
                ).reshape(-1)
                weights = decision.expert_token_weights[:, expert_idx, :].reshape(-1, 1)
                expert_out = expert(flat[indexes]) * weights
                # Autocast-safe: index_add_ requires matching dtypes (no-op cast
                # when they already match).
                routed.index_add_(0, indexes, expert_out.to(routed.dtype))
        output = routed.reshape(orig_shape)
        for expert in self.shared_experts:
            output = output + expert(hidden_states)
        auxiliary = (
            decision.load_balance_loss * self.load_balance_weight
            + decision.z_loss
        )
        return SparseMoEOutput(
            hidden_states=output,
            auxiliary_loss=auxiliary,
            stats=decision.stats,
            expert_choice_coverage=decision.coverage,
        )
