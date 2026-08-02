"""Grouped adjugate experts for router-preserving MoE capacity expansion.

Grove MoE adds one smaller parallel expert to every disjoint group of routed
experts. The ordinary router is unchanged. When several selected experts for a
token belong to the same group, the group expert is evaluated once and scaled
by the sum of their routing weights:

    sum_i rho_i E_i(x)
    + scale * sum_g (sum_{i in g} rho_i) A_g(x)

https://arxiv.org/abs/2508.07785
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


ADJUGATE_EXPERT_STATE_PREFIX = "adjugate_experts."
ADJUGATE_EXPERT_STATE_VERSION = 1


@dataclass(frozen=True)
class AdjugateExpertTopology:
    """Model-agnostic disjoint expert grouping and Grove scale contract."""

    num_experts: int
    num_groups: int
    scale: float

    def validate(self) -> "AdjugateExpertTopology":
        if int(self.num_experts) <= 0:
            raise ValueError("Adjugate topology num_experts must be > 0.")
        if int(self.num_groups) <= 0:
            raise ValueError("Adjugate topology num_groups must be > 0.")
        if int(self.num_experts) % int(self.num_groups) != 0:
            raise ValueError(
                "Adjugate topology requires num_groups to divide num_experts."
            )
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError("Adjugate expert scale must be finite and > 0.")
        if float(self.scale) > self.maximum_scale:
            raise ValueError(
                "Adjugate expert scale must be <= num_groups / num_experts "
                f"({self.maximum_scale:g}) for the configured topology."
            )
        return self

    @property
    def experts_per_group(self) -> int:
        return int(self.num_experts) // int(self.num_groups)

    @property
    def maximum_scale(self) -> float:
        return float(self.num_groups) / float(self.num_experts)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "num_experts": int(self.num_experts),
            "num_groups": int(self.num_groups),
            "experts_per_group": int(self.experts_per_group),
            "scale": float(self.scale),
        }


@dataclass(frozen=True)
class AdjugateExpertRoutingDecision:
    """Aggregated group weights for one ordinary top-k routing decision."""

    group_weights: Any
    active_group_mask: Any


def aggregate_adjugate_group_routes(
    top_indices: Any,
    top_scores: Any,
    *,
    topology: AdjugateExpertTopology,
) -> AdjugateExpertRoutingDecision:
    """Aggregate repeated routes so each token/group pair is evaluated once."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Adjugate expert routing requires torch.")
    topology.validate()
    if (
        top_indices.ndim != 2
        or top_scores.shape != top_indices.shape
        or int(top_indices.shape[1]) <= 0
    ):
        raise ValueError(
            "Adjugate expert routes must be matching [tokens, top_k] tensors."
        )
    if top_indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("Adjugate expert indices must use an integer dtype.")
    if top_indices.device.type == "cpu" and (
        bool((top_indices < 0).any())
        or bool((top_indices >= int(topology.num_experts)).any())
    ):
        raise ValueError("Adjugate expert routes contain an invalid expert id.")
    group_indices = torch.div(
        top_indices,
        int(topology.experts_per_group),
        rounding_mode="floor",
    )
    group_weights = top_scores.new_zeros(
        (int(top_indices.shape[0]), int(topology.num_groups))
    )
    group_weights.scatter_add_(1, group_indices, top_scores)
    return AdjugateExpertRoutingDecision(
        group_weights=group_weights,
        active_group_mask=group_weights != 0,
    )


class AdjugateExpertPool(nn.Module):
    """Provider-supplied group experts behind a model-agnostic Grove executor."""

    def __init__(
        self,
        *,
        topology: AdjugateExpertTopology,
        hidden_size: int,
        intermediate_size: int,
        expert_kind: str,
        expert_factory: Callable[[int], Any],
        zero_output_initializer: Callable[[Any], None],
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("AdjugateExpertPool requires torch.")
        super().__init__()
        topology.validate()
        if int(hidden_size) <= 0:
            raise ValueError("Adjugate expert hidden_size must be > 0.")
        if int(intermediate_size) <= 0:
            raise ValueError("Adjugate expert intermediate_size must be > 0.")
        normalized_kind = str(expert_kind).strip()
        if not normalized_kind:
            raise ValueError("Adjugate expert_kind must be non-empty.")
        experts = []
        for group_index in range(int(topology.num_groups)):
            expert = expert_factory(group_index)
            if not isinstance(expert, nn.Module):
                raise TypeError("Adjugate expert_factory must return nn.Module.")
            zero_output_initializer(expert)
            experts.append(expert)
        self.topology_spec = topology
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.expert_kind = normalized_kind
        self.experts = nn.ModuleList(experts)

    def topology(self) -> dict[str, int | float | str]:
        return {
            "kind": "grouped_adjugate",
            "expert_kind": self.expert_kind,
            "hidden_size": int(self.hidden_size),
            "intermediate_size": int(self.intermediate_size),
            **self.topology_spec.as_dict(),
        }

    def output_contribution(
        self,
        tokens: Any,
        top_indices: Any,
        top_scores: Any,
    ) -> Any:
        """Compute each active adjugate expert once per token and group."""

        if tokens.ndim != 2 or int(tokens.shape[-1]) != self.hidden_size:
            raise ValueError(
                "Adjugate expert tokens must be [tokens, hidden_size]."
            )
        if int(top_indices.shape[0]) != int(tokens.shape[0]):
            raise ValueError(
                "Adjugate expert routes must have one row per input token."
            )
        decision = aggregate_adjugate_group_routes(
            top_indices,
            top_scores,
            topology=self.topology_spec,
        )
        output = torch.zeros_like(tokens)
        for group_index, expert in enumerate(self.experts):
            token_indices = torch.where(
                decision.active_group_mask[:, group_index]
            )[0]
            if int(token_indices.numel()) == 0:
                continue
            selected_tokens = tokens.index_select(0, token_indices)
            selected_weights = decision.group_weights.index_select(
                0, token_indices
            )[:, group_index]
            contribution = expert(selected_tokens)
            contribution = contribution * selected_weights.to(
                contribution.dtype
            ).unsqueeze(-1)
            output.index_add_(0, token_indices, contribution.to(output.dtype))
        return output * float(self.topology_spec.scale)

    forward = output_contribution


def _adjugate_pools(root: Any) -> dict[str, AdjugateExpertPool]:
    return {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, AdjugateExpertPool)
    }


def export_adjugate_expert_state(root: Any) -> dict[str, Any]:
    """Export every enabled pool with a versioned exact topology manifest."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Adjugate expert persistence requires torch.")
    pools = _adjugate_pools(root)
    if not pools:
        return {}
    prefix = ADJUGATE_EXPERT_STATE_PREFIX
    state: dict[str, Any] = {
        f"{prefix}schema_version": ADJUGATE_EXPERT_STATE_VERSION,
        f"{prefix}topology": {
            name: module.topology() for name, module in sorted(pools.items())
        },
    }
    for name, module in sorted(pools.items()):
        for key, value in module.state_dict().items():
            state[f"{prefix}modules.{name}.{key}"] = (
                value.detach().cpu().clone()
            )
    return state


def load_adjugate_expert_state(root: Any, state: dict[str, Any]) -> None:
    """Load enabled pools and reject absent, unknown, or mismatched topology."""

    prefix = ADJUGATE_EXPERT_STATE_PREFIX
    supplied = {
        str(key): value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    pools = _adjugate_pools(root)
    if not pools:
        if supplied:
            raise ValueError(
                "Adapter contains adjugate-expert state, but the configured "
                "model has no adjugate expert pool."
            )
        return
    if not supplied:
        raise ValueError(
            "The configured adjugate expert pool requires adapter state."
        )
    version = supplied.get(f"{prefix}schema_version")
    if int(version) != ADJUGATE_EXPERT_STATE_VERSION:
        raise ValueError(
            f"Unsupported adjugate-expert state version {version!r}."
        )
    expected_topology = {
        name: module.topology() for name, module in sorted(pools.items())
    }
    if supplied.get(f"{prefix}topology") != expected_topology:
        raise ValueError("Adjugate-expert topology does not match the model.")
    consumed = {f"{prefix}schema_version", f"{prefix}topology"}
    for name, module in sorted(pools.items()):
        module_prefix = f"{prefix}modules.{name}."
        module_state = {
            key[len(module_prefix) :]: value
            for key, value in supplied.items()
            if key.startswith(module_prefix)
        }
        expected_keys = set(module.state_dict())
        if set(module_state) != expected_keys:
            raise ValueError(
                f"Adjugate-expert state for '{name}' has keys "
                f"{sorted(module_state)}, expected {sorted(expected_keys)}."
            )
        module.load_state_dict(module_state, strict=True)
        consumed.update(module_prefix + key for key in module_state)
    unknown = sorted(set(supplied) - consumed)
    if unknown:
        raise ValueError(f"Unknown adjugate-expert state keys: {unknown}.")


__all__ = [
    "ADJUGATE_EXPERT_STATE_PREFIX",
    "ADJUGATE_EXPERT_STATE_VERSION",
    "AdjugateExpertPool",
    "AdjugateExpertRoutingDecision",
    "AdjugateExpertTopology",
    "aggregate_adjugate_group_routes",
    "export_adjugate_expert_state",
    "load_adjugate_expert_state",
]
