"""Timestep-aware routing separated from timestep-modulated expert input.

Nucleus-Image routes with ``[x_norm | t]`` while experts consume the AdaLN
modulated representation.  The concatenated linear map is represented here as
two additive projections so a pretrained router remains the content branch:
https://arxiv.org/abs/2604.12163
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


DECOUPLED_ROUTING_STATE_PREFIX = "decoupled_routing."
DECOUPLED_ROUTING_STATE_VERSION = 1


class DecoupledRouterConditioner(nn.Module):
    """Independent timestep projection added to unmodulated content logits."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        timestep_weight: float,
        device: Any = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("DecoupledRouterConditioner requires torch.")
        super().__init__()
        if int(hidden_size) <= 0 or int(num_experts) <= 0:
            raise ValueError("hidden_size and num_experts must be > 0.")
        if float(timestep_weight) <= 0.0:
            raise ValueError("timestep_weight must be > 0.")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.timestep_weight = float(timestep_weight)
        # Nucleus-Image does not publish an initialization rule. Zero is the
        # deterministic choice that starts from content-only routing and lets
        # the independent timestep channel enter through training.
        self.timestep_projection = nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.hidden_size,
                dtype=torch.float32,
                device=device,
            )
        )

    def forward(
        self,
        *,
        content_tokens: Any,
        timestep_hidden: Any,
        content_weight: Any,
    ) -> Any:
        if (
            content_tokens.ndim != 3
            or int(content_tokens.shape[-1]) != self.hidden_size
        ):
            raise ValueError(
                "Unmodulated router content must be [batch, tokens, hidden]."
            )
        if (
            timestep_hidden is None
            or timestep_hidden.ndim != 3
            or int(timestep_hidden.shape[0]) != int(content_tokens.shape[0])
            or int(timestep_hidden.shape[-1]) != self.hidden_size
            or int(timestep_hidden.shape[1])
            not in {1, int(content_tokens.shape[1])}
        ):
            raise ValueError(
                "Timestep router input must be [batch, 1|tokens, hidden]."
            )
        if tuple(content_weight.shape) != (
            self.num_experts,
            self.hidden_size,
        ):
            raise ValueError(
                "Content router weight must be [num_experts, hidden]."
            )
        timestep_hidden = timestep_hidden.expand_as(content_tokens)
        content_logits = F.linear(
            content_tokens.float(),
            content_weight.to(
                device=content_tokens.device,
                dtype=torch.float32,
            ),
        )
        timestep_logits = F.linear(
            timestep_hidden.float(),
            self.timestep_projection.to(
                device=content_tokens.device,
                dtype=torch.float32,
            ),
        )
        return content_logits + timestep_logits * self.timestep_weight

    def topology(self) -> dict[str, int | float]:
        return {
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "timestep_weight": self.timestep_weight,
        }


def export_decoupled_routing_state(root: Any) -> dict[str, Any]:
    """Export every conditioner with an exact, versioned topology."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Decoupled routing persistence requires torch.")
    modules = {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, DecoupledRouterConditioner)
    }
    if not modules:
        return {}
    prefix = DECOUPLED_ROUTING_STATE_PREFIX
    state: dict[str, Any] = {
        f"{prefix}schema_version": DECOUPLED_ROUTING_STATE_VERSION,
        f"{prefix}topology": {
            name: module.topology() for name, module in sorted(modules.items())
        },
    }
    for name, module in sorted(modules.items()):
        for key, value in module.state_dict().items():
            state[f"{prefix}modules.{name}.{key}"] = (
                value.detach().cpu().clone()
            )
    return state


def load_decoupled_routing_state(root: Any, state: dict[str, Any]) -> None:
    """Load conditioners and reject absent, mismatched, or unknown state."""

    prefix = DECOUPLED_ROUTING_STATE_PREFIX
    supplied = {
        str(key): value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    modules = {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, DecoupledRouterConditioner)
    }
    if not modules:
        if supplied:
            raise ValueError(
                "Adapter contains decoupled-routing state, but the configured "
                "model has no decoupled router conditioner."
            )
        return
    if not supplied:
        raise ValueError(
            "The configured decoupled router requires adapter state."
        )
    version = supplied.get(f"{prefix}schema_version")
    if int(version) != DECOUPLED_ROUTING_STATE_VERSION:
        raise ValueError(
            f"Unsupported decoupled-routing state version {version!r}."
        )
    expected_topology = {
        name: module.topology() for name, module in sorted(modules.items())
    }
    if supplied.get(f"{prefix}topology") != expected_topology:
        raise ValueError("Decoupled-routing topology does not match the model.")
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
                f"Decoupled-routing state for '{name}' has keys "
                f"{sorted(module_state)}, expected {sorted(expected_keys)}."
            )
        module.load_state_dict(module_state, strict=True)
        consumed.update(module_prefix + key for key in module_state)
    unknown = sorted(set(supplied) - consumed)
    if unknown:
        raise ValueError(f"Unknown decoupled-routing state keys: {unknown}.")


__all__ = [
    "DECOUPLED_ROUTING_STATE_PREFIX",
    "DECOUPLED_ROUTING_STATE_VERSION",
    "DecoupledRouterConditioner",
    "export_decoupled_routing_state",
    "load_decoupled_routing_state",
]
