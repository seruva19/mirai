"""Saliency-harnessing residual router for diffusion MoE models.

The module implements the additive dual-router score in SharpMoE Equation 7.
The paper specifies a two-layer SiLU MLP and zero initialization but omits its
hidden width. Mirai exposes a bounded bottleneck and uses zero-output
initialization: the first projection is deterministic while the final projection
is zero. This preserves the pretrained route exactly and keeps the final layer
trainable on the first update.

Source: https://arxiv.org/abs/2606.26938
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


SHARP_MOE_STATE_PREFIX = "sharp_moe."
SHARP_MOE_STATE_VERSION = 1


@dataclass(frozen=True)
class SharpMoESpec:
    """Model-agnostic SharpMoE post-training configuration."""

    trajectory_steps: int = 10
    router_hidden_dim: int = 128
    seed: int = 0

    def validate(self) -> "SharpMoESpec":
        if int(self.trajectory_steps) < 2:
            raise ValueError("trajectory_steps must be >= 2.")
        if int(self.router_hidden_dim) < 1:
            raise ValueError("router_hidden_dim must be > 0.")
        if int(self.seed) < 0:
            raise ValueError("seed must be >= 0.")
        return self


class SaliencyHarnessingRouter(nn.Module):
    """One zero-output residual saliency router attached to one native router."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        bottleneck_size: int,
        initialization_seed: int,
        device: Any = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("SaliencyHarnessingRouter requires torch.")
        super().__init__()
        if int(hidden_size) < 1 or int(num_experts) < 2:
            raise ValueError("hidden_size must be > 0 and num_experts must be > 1.")
        if int(bottleneck_size) < 1:
            raise ValueError("bottleneck_size must be > 0.")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.bottleneck_size = int(bottleneck_size)
        self.initialization_seed = int(initialization_seed)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.initialization_seed)
        bound = 1.0 / math.sqrt(float(self.hidden_size))
        input_weight = torch.empty(
            self.bottleneck_size,
            self.hidden_size,
            dtype=torch.float32,
        ).uniform_(-bound, bound, generator=generator)
        output_weight = torch.zeros(
            self.num_experts,
            self.bottleneck_size,
            dtype=torch.float32,
        )
        if device is not None:
            input_weight = input_weight.to(device=device)
            output_weight = output_weight.to(device=device)
        self.input_weight = nn.Parameter(input_weight)
        self.output_weight = nn.Parameter(output_weight)

    def topology(self) -> dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "bottleneck_size": self.bottleneck_size,
            "initialization_seed": self.initialization_seed,
        }

    def forward(
        self,
        saliency_tokens: Any,
        *,
        route_scope_mask: Any | None = None,
    ) -> Any:
        if saliency_tokens.ndim != 2 or int(saliency_tokens.shape[-1]) != self.hidden_size:
            raise ValueError(
                "Saliency router tokens must have shape [tokens, hidden_size]."
            )
        hidden = F.silu(
            F.linear(saliency_tokens.float(), self.input_weight)
        )
        delta = F.linear(hidden, self.output_weight)
        if route_scope_mask is not None:
            scope = torch.as_tensor(
                route_scope_mask,
                device=delta.device,
                dtype=torch.bool,
            ).reshape(-1)
            if int(scope.numel()) != int(delta.shape[0]):
                raise ValueError("route_scope_mask must share the token axis.")
            delta = delta * scope.unsqueeze(-1).to(dtype=delta.dtype)
        return delta


def export_sharp_moe_state(root: Any) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("SharpMoE persistence requires torch.")
    modules = {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, SaliencyHarnessingRouter)
    }
    if not modules:
        return {}
    prefix = SHARP_MOE_STATE_PREFIX
    state: dict[str, Any] = {
        f"{prefix}schema_version": SHARP_MOE_STATE_VERSION,
        f"{prefix}topology": {
            name: module.topology() for name, module in sorted(modules.items())
        },
    }
    for name, module in sorted(modules.items()):
        for key, value in module.state_dict().items():
            state[f"{prefix}modules.{name}.{key}"] = value.detach().cpu().clone()
    return state


def load_sharp_moe_state(root: Any, state: Mapping[str, Any]) -> None:
    prefix = SHARP_MOE_STATE_PREFIX
    supplied = {
        str(key): value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    modules = {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, SaliencyHarnessingRouter)
    }
    if not modules:
        if supplied:
            raise ValueError(
                "Adapter contains SharpMoE state, but the model has no saliency router."
            )
        return
    if not supplied:
        raise ValueError("The configured SharpMoE router requires adapter state.")
    if int(supplied.get(f"{prefix}schema_version", -1)) != SHARP_MOE_STATE_VERSION:
        raise ValueError("Unsupported SharpMoE state version.")
    expected_topology = {
        name: module.topology() for name, module in sorted(modules.items())
    }
    if supplied.get(f"{prefix}topology") != expected_topology:
        raise ValueError("SharpMoE routing topology does not match the model.")
    consumed = {f"{prefix}schema_version", f"{prefix}topology"}
    for name, module in sorted(modules.items()):
        module_prefix = f"{prefix}modules.{name}."
        module_state = {
            key[len(module_prefix) :]: value
            for key, value in supplied.items()
            if key.startswith(module_prefix)
        }
        if set(module_state) != set(module.state_dict()):
            raise ValueError(f"SharpMoE state for '{name}' is incomplete.")
        module.load_state_dict(module_state, strict=True)
        consumed.update(module_prefix + key for key in module_state)
    unknown = sorted(set(supplied) - consumed)
    if unknown:
        raise ValueError(f"Unknown SharpMoE state keys: {unknown}.")


__all__ = [
    "export_sharp_moe_state",
    "load_sharp_moe_state",
    "SaliencyHarnessingRouter",
    "SHARP_MOE_STATE_PREFIX",
    "SHARP_MOE_STATE_VERSION",
    "SharpMoESpec",
]
