"""Checkpoint-preserving Chain-of-Experts adaptation for sparse MoE layers.

Chain-of-Experts applies the same expert pool repeatedly with an independent
router at each communication step.  Mirai's adapter-scale form keeps the
pretrained first step intact and represents the second router as a trainable
low-rank delta over the native router.  A zero-initialized continuation scale
makes construction an exact no-op until training enables the second step.

The source method and its ablations are described in:
https://arxiv.org/abs/2506.18945
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


CHAIN_OF_EXPERTS_STATE_PREFIX = "chain_of_experts."
CHAIN_OF_EXPERTS_STATE_VERSION = 1


@dataclass(frozen=True)
class ChainOfExpertsSpec:
    """Topology of Mirai's two-step, low-rank-router CoE adaptation."""

    hidden_size: int
    num_experts: int
    router_rank: int
    communication_steps: int = 2

    def validate(self) -> "ChainOfExpertsSpec":
        if int(self.hidden_size) <= 0:
            raise ValueError("CoE hidden_size must be > 0.")
        if int(self.num_experts) <= 0:
            raise ValueError("CoE num_experts must be > 0.")
        if int(self.router_rank) <= 0:
            raise ValueError("CoE router_rank must be > 0.")
        if int(self.router_rank) > min(int(self.hidden_size), int(self.num_experts)):
            raise ValueError(
                "CoE router_rank must not exceed hidden_size or num_experts."
            )
        if int(self.communication_steps) != 2:
            raise ValueError(
                "Mirai's checkpoint-preserving CoE adaptation supports exactly "
                "two communication steps."
            )
        return self

    def as_dict(self) -> dict[str, int | str]:
        return {
            "kind": "checkpoint_preserving_low_rank_router_delta",
            "hidden_size": int(self.hidden_size),
            "num_experts": int(self.num_experts),
            "router_rank": int(self.router_rank),
            "communication_steps": int(self.communication_steps),
        }


class ChainOfExpertsExtension(nn.Module):
    """Own the second router delta, continuation gate, and route telemetry."""

    def __init__(
        self,
        spec: ChainOfExpertsSpec,
        *,
        device: Any = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("ChainOfExpertsExtension requires torch.")
        super().__init__()
        self.spec = spec.validate()
        self.router_down = nn.Parameter(
            torch.empty(
                int(spec.router_rank),
                int(spec.hidden_size),
                dtype=torch.float32,
                device=device,
            )
        )
        self.router_up = nn.Parameter(
            torch.zeros(
                int(spec.num_experts),
                int(spec.router_rank),
                dtype=torch.float32,
                device=device,
            )
        )
        self.continuation_scale = nn.Parameter(
            torch.zeros((), dtype=torch.float32, device=device)
        )
        nn.init.kaiming_uniform_(self.router_down, a=math.sqrt(5.0))
        self.last_route_retention: Any | None = None
        self.last_route_switch_fraction: Any | None = None

    def topology(self) -> dict[str, int | str]:
        return self.spec.as_dict()

    def router_logit_delta(self, tokens: Any) -> Any:
        """Return the trainable second-router delta in FP32."""

        if tokens.ndim != 2 or int(tokens.shape[-1]) != int(self.spec.hidden_size):
            raise ValueError("CoE router tokens must be [tokens, hidden_size].")
        hidden = F.linear(tokens.float(), self.router_down.float())
        return F.linear(hidden, self.router_up.float())

    @staticmethod
    def continuation_input(hidden_states: Any, first_output: Any) -> Any:
        if hidden_states.shape != first_output.shape:
            raise ValueError("CoE inner residual tensors must have matching shapes.")
        return hidden_states + first_output

    def combine(self, first_output: Any, continuation_output: Any) -> Any:
        if first_output.shape != continuation_output.shape:
            raise ValueError("CoE step outputs must have matching shapes.")
        scale = self.continuation_scale.to(
            device=continuation_output.device,
            dtype=continuation_output.dtype,
        )
        return first_output + continuation_output * scale

    def record_routes(self, first_indices: Any, second_indices: Any) -> None:
        """Record unordered route retention and complete-switch frequency."""

        if (
            first_indices.ndim != 2
            or second_indices.ndim != 2
            or int(first_indices.shape[0]) != int(second_indices.shape[0])
            or int(first_indices.shape[1]) != int(second_indices.shape[1])
        ):
            raise ValueError("CoE route telemetry requires matching [tokens, k] tensors.")
        first = first_indices.detach().sort(dim=-1).values
        second = second_indices.detach().sort(dim=-1).values
        matches = (first.unsqueeze(-1) == second.unsqueeze(-2)).any(dim=-1)
        retention = matches.float().mean(dim=-1)
        self.last_route_retention = retention.mean().detach()
        self.last_route_switch_fraction = (retention < 1.0).float().mean().detach()


def _chain_modules(root: Any) -> dict[str, ChainOfExpertsExtension]:
    return {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, ChainOfExpertsExtension)
    }


def chain_of_experts_metrics(root: Any) -> dict[str, float]:
    """Aggregate detached transition metrics across enabled CoE layers."""

    modules = tuple(_chain_modules(root).values())
    retention = [
        module.last_route_retention
        for module in modules
        if module.last_route_retention is not None
    ]
    switches = [
        module.last_route_switch_fraction
        for module in modules
        if module.last_route_switch_fraction is not None
    ]
    metrics: dict[str, float] = {}
    if retention:
        metrics["moe_chain_route_retention"] = float(
            torch.stack(retention).mean().float().cpu().item()
        )
    if switches:
        metrics["moe_chain_route_switch_fraction"] = float(
            torch.stack(switches).mean().float().cpu().item()
        )
    if modules:
        metrics["moe_chain_continuation_scale"] = float(
            torch.stack(
                [module.continuation_scale.detach().float().cpu() for module in modules]
            ).mean().item()
        )
    return metrics


def export_chain_of_experts_state(root: Any) -> dict[str, Any]:
    """Export every enabled CoE extension with an exact topology manifest."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Chain-of-Experts persistence requires torch.")
    modules = _chain_modules(root)
    if not modules:
        return {}
    prefix = CHAIN_OF_EXPERTS_STATE_PREFIX
    state: dict[str, Any] = {
        f"{prefix}schema_version": CHAIN_OF_EXPERTS_STATE_VERSION,
        f"{prefix}topology": {
            name: module.topology() for name, module in sorted(modules.items())
        },
    }
    for name, module in sorted(modules.items()):
        for key, value in module.state_dict().items():
            state[f"{prefix}modules.{name}.{key}"] = value.detach().cpu().clone()
    return state


def load_chain_of_experts_state(root: Any, state: dict[str, Any]) -> None:
    """Load CoE state and fail on absent, unknown, or mismatched topology."""

    prefix = CHAIN_OF_EXPERTS_STATE_PREFIX
    supplied = {
        str(key): value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    modules = _chain_modules(root)
    if not modules:
        if supplied:
            raise ValueError(
                "Adapter contains Chain-of-Experts state, but CoE is disabled."
            )
        return
    if not supplied:
        raise ValueError("Enabled Chain-of-Experts requires adapter state.")
    version = supplied.get(f"{prefix}schema_version")
    if int(version) != CHAIN_OF_EXPERTS_STATE_VERSION:
        raise ValueError(f"Unsupported Chain-of-Experts state version {version!r}.")
    expected_topology = {
        name: module.topology() for name, module in sorted(modules.items())
    }
    if supplied.get(f"{prefix}topology") != expected_topology:
        raise ValueError("Chain-of-Experts topology does not match the model.")
    consumed = {f"{prefix}schema_version", f"{prefix}topology"}
    for name, module in sorted(modules.items()):
        module_prefix = f"{prefix}modules.{name}."
        module_state = {
            key[len(module_prefix) :]: value
            for key, value in supplied.items()
            if key.startswith(module_prefix)
        }
        expected_keys = set(module.state_dict())
        if set(module_state) != expected_keys:
            raise ValueError(
                f"Chain-of-Experts state for {name!r} has keys "
                f"{sorted(module_state)}, expected {sorted(expected_keys)}."
            )
        module.load_state_dict(module_state, strict=True)
        consumed.update(module_prefix + key for key in module_state)
    unknown = sorted(set(supplied) - consumed)
    if unknown:
        raise ValueError(f"Unknown Chain-of-Experts state keys: {unknown}.")


__all__ = [
    "CHAIN_OF_EXPERTS_STATE_PREFIX",
    "CHAIN_OF_EXPERTS_STATE_VERSION",
    "ChainOfExpertsExtension",
    "ChainOfExpertsSpec",
    "chain_of_experts_metrics",
    "export_chain_of_experts_state",
    "load_chain_of_experts_state",
]
