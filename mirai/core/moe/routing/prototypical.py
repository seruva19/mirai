"""Learnable prototype guidance for pretrained token-choice routers.

ProMoE routes visual tokens by cosine similarity to one learnable prototype per
expert and regularizes the prototypes against the means of their assigned token
sets (Equations 4 and 6): https://arxiv.org/abs/2510.24711

The published model is trained from scratch with prototype-only scores.  Mirai
adapts the mechanism to a pretrained router by adding a zero-initialized scalar
multiple of the prototype score to both its selection and gating scores.  This
preserves the pretrained route exactly at construction while retaining task and
contrastive gradients for the new parameters.
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


PROTOTYPICAL_ROUTING_STATE_PREFIX = "prototypical_routing."
PROTOTYPICAL_ROUTING_STATE_VERSION = 1


@dataclass(frozen=True)
class PrototypicalRoutingSpec:
    """Model-agnostic parameters for residual prototypical routing."""

    prototype_scale: float = 1.0
    contrastive_weight: float = 1.0
    contrastive_temperature: float = 0.07
    seed: int = 0

    def validate(self) -> "PrototypicalRoutingSpec":
        values = {
            "prototype_scale": self.prototype_scale,
            "contrastive_weight": self.contrastive_weight,
            "contrastive_temperature": self.contrastive_temperature,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if float(self.prototype_scale) <= 0.0:
            raise ValueError("prototype_scale must be > 0.")
        if float(self.contrastive_weight) < 0.0:
            raise ValueError("contrastive_weight must be >= 0.")
        if float(self.contrastive_temperature) <= 0.0:
            raise ValueError("contrastive_temperature must be > 0.")
        if int(self.seed) < 0:
            raise ValueError("seed must be >= 0.")
        return self


@dataclass(frozen=True)
class PrototypicalRoutes:
    top_indices: Any
    top_weights: Any


def routing_contrastive_loss(
    token_embeddings: Any,
    cluster_assignments: Any,
    prototypes: Any,
    *,
    temperature: float,
) -> Any:
    """Equation 6 over experts that receive at least one assigned token.

    ``cluster_assignments`` may be top-1 ``[tokens]`` or top-k
    ``[tokens, k]``.  A token contributes once to each selected expert, matching
    the official top-k implementation.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("routing_contrastive_loss requires torch.")
    if token_embeddings.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("Token embeddings and prototypes must be rank-2 tensors.")
    if int(token_embeddings.shape[-1]) != int(prototypes.shape[-1]):
        raise ValueError("Token and prototype hidden dimensions must match.")
    if cluster_assignments.ndim not in {1, 2}:
        raise ValueError("Cluster assignments must have shape [tokens] or [tokens, k].")
    if int(cluster_assignments.shape[0]) != int(token_embeddings.shape[0]):
        raise ValueError("Cluster assignments must share the token axis.")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and > 0.")

    means: list[Any] = []
    active_ids: list[int] = []
    for expert_id in range(int(prototypes.shape[0])):
        if cluster_assignments.ndim == 1:
            selected = cluster_assignments == expert_id
        else:
            selected = (cluster_assignments == expert_id).any(dim=-1)
        if bool(selected.any().item()):
            means.append(token_embeddings[selected].float().mean(dim=0))
            active_ids.append(expert_id)

    if len(active_ids) < 2:
        # Keep a graph-connected zero so auxiliary-loss composition remains
        # valid even when a microbatch reaches fewer than two experts.
        return token_embeddings.float().sum() * 0.0 + prototypes.float().sum() * 0.0

    token_means = torch.stack(means)
    index = torch.as_tensor(active_ids, device=prototypes.device, dtype=torch.long)
    active_prototypes = prototypes.index_select(0, index).float()
    logits = F.normalize(active_prototypes, p=2, dim=-1) @ F.normalize(
        token_means, p=2, dim=-1
    ).transpose(0, 1)
    labels = torch.arange(len(active_ids), device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits / float(temperature), labels)


class PrototypicalRouterExtension(nn.Module):
    """One residual prototype router attached to one native router module."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        spec: PrototypicalRoutingSpec,
        initialization_seed: int,
        device: Any = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PrototypicalRouterExtension requires torch.")
        super().__init__()
        if int(hidden_size) <= 0 or int(num_experts) <= 1:
            raise ValueError("hidden_size must be > 0 and num_experts must be > 1.")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.spec = spec.validate()
        self.initialization_seed = int(initialization_seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.initialization_seed)
        initial = torch.randn(
            self.num_experts,
            self.hidden_size,
            generator=generator,
            dtype=torch.float32,
        ).mul_(0.02)
        if device is not None:
            initial = initial.to(device=device)
        self.prototypes = nn.Parameter(initial)
        self.residual_scale = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
                device=device,
            )
        )
        self.last_contrastive_loss: Any | None = None
        self.last_active_experts = 0
        self.last_eligible_tokens = 0

    def topology(self) -> dict[str, int | float]:
        return {
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "prototype_scale": float(self.spec.prototype_scale),
            "contrastive_weight": float(self.spec.contrastive_weight),
            "contrastive_temperature": float(self.spec.contrastive_temperature),
            "initialization_seed": self.initialization_seed,
        }

    def select(
        self,
        tokens: Any,
        native_choice_scores: Any,
        native_gate_scores: Any,
        native_top_indices: Any,
        native_top_weights: Any,
        *,
        route_scope_mask: Any | None,
        valid_token_mask: Any | None,
        norm_topk_prob: bool,
        route_scale: float,
        training: bool,
        choice_score_transform: Any | None = None,
    ) -> PrototypicalRoutes:
        """Add prototype affinities and route only provider-selected tokens."""

        if tokens.ndim != 2 or int(tokens.shape[-1]) != self.hidden_size:
            raise ValueError("Prototype router tokens must have shape [tokens, hidden].")
        expected_scores = (int(tokens.shape[0]), self.num_experts)
        if tuple(native_choice_scores.shape) != expected_scores:
            raise ValueError("Prototype choice scores must have shape [tokens, experts].")
        if tuple(native_gate_scores.shape) != expected_scores:
            raise ValueError("Prototype gate scores must have shape [tokens, experts].")
        if tuple(native_top_indices.shape) != tuple(native_top_weights.shape):
            raise ValueError("Native route indices and weights must have equal shape.")
        if int(native_top_indices.shape[0]) != int(tokens.shape[0]):
            raise ValueError("Native routes must share the prototype token axis.")

        eligible = torch.ones(
            int(tokens.shape[0]), device=tokens.device, dtype=torch.bool
        )
        if route_scope_mask is not None:
            scope = torch.as_tensor(
                route_scope_mask, device=tokens.device, dtype=torch.bool
            ).reshape(-1)
            if tuple(scope.shape) != tuple(eligible.shape):
                raise ValueError("route_scope_mask must have shape [tokens].")
            eligible &= scope
        if valid_token_mask is not None:
            valid = torch.as_tensor(
                valid_token_mask, device=tokens.device, dtype=torch.bool
            ).reshape(-1)
            if tuple(valid.shape) != tuple(eligible.shape):
                raise ValueError("valid_token_mask must have shape [tokens].")
            eligible &= valid

        self.last_contrastive_loss = None
        self.last_active_experts = 0
        eligible_indices = torch.nonzero(eligible, as_tuple=False).flatten()
        self.last_eligible_tokens = int(eligible_indices.numel())
        if self.last_eligible_tokens == 0:
            return PrototypicalRoutes(native_top_indices, native_top_weights)

        eligible_tokens = tokens.index_select(0, eligible_indices)
        token_norm = F.normalize(eligible_tokens.float(), p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes.float(), p=2, dim=-1)
        affinity = token_norm @ prototype_norm.transpose(0, 1)
        residual = (
            self.residual_scale.float()
            * float(self.spec.prototype_scale)
            * affinity
        )
        choice_scores = native_choice_scores[eligible].float() + residual
        gate_scores = native_gate_scores[eligible].float() + residual
        if choice_score_transform is not None:
            choice_scores = choice_score_transform(choice_scores)
        top_k = int(native_top_indices.shape[-1])
        selected_indices = torch.topk(
            choice_scores, k=top_k, dim=-1, sorted=False
        ).indices
        selected_weights = gate_scores.gather(-1, selected_indices)
        if top_k > 1 and bool(norm_topk_prob):
            selected_weights = selected_weights / (
                selected_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        selected_weights = selected_weights * float(route_scale)

        indices = native_top_indices.clone()
        weights = native_top_weights.clone()
        indices[eligible] = selected_indices.to(dtype=indices.dtype)
        weights[eligible] = selected_weights.to(dtype=weights.dtype)

        if bool(training) and float(self.spec.contrastive_weight) > 0.0:
            raw_loss = routing_contrastive_loss(
                eligible_tokens,
                selected_indices,
                self.prototypes,
                temperature=float(self.spec.contrastive_temperature),
            )
            self.last_contrastive_loss = (
                raw_loss * float(self.spec.contrastive_weight)
            )
            self.last_active_experts = int(
                torch.unique(selected_indices.detach()).numel()
            )
        return PrototypicalRoutes(indices, weights)

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "moe_prototypical_residual_scale": float(
                self.residual_scale.detach().float().cpu().item()
            ),
            "moe_prototypical_active_experts": int(self.last_active_experts),
            "moe_prototypical_eligible_tokens": int(self.last_eligible_tokens),
        }


def collect_prototypical_routing_losses(root: Any) -> list[Any]:
    checkpoint_terms = tuple(
        getattr(root, "_mirai_checkpoint_prototypical_routing_terms", ()) or ()
    )
    if checkpoint_terms:
        return list(checkpoint_terms)
    return [
        module.last_contrastive_loss
        for module in root.modules()
        if isinstance(module, PrototypicalRouterExtension)
        and module.last_contrastive_loss is not None
    ]


def prototypical_routing_diagnostics(root: Any) -> dict[str, int | float]:
    modules = [
        module
        for module in root.modules()
        if isinstance(module, PrototypicalRouterExtension)
    ]
    if not modules:
        return {}
    values = [module.diagnostics() for module in modules]
    return {
        "moe_prototypical_residual_scale_mean": sum(
            float(item["moe_prototypical_residual_scale"]) for item in values
        )
        / len(values),
        "moe_prototypical_active_experts_mean": sum(
            int(item["moe_prototypical_active_experts"]) for item in values
        )
        / len(values),
        "moe_prototypical_eligible_tokens": sum(
            int(item["moe_prototypical_eligible_tokens"]) for item in values
        ),
    }


def export_prototypical_routing_state(root: Any) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Prototypical routing persistence requires torch.")
    modules = {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, PrototypicalRouterExtension)
    }
    if not modules:
        return {}
    prefix = PROTOTYPICAL_ROUTING_STATE_PREFIX
    state: dict[str, Any] = {
        f"{prefix}schema_version": PROTOTYPICAL_ROUTING_STATE_VERSION,
        f"{prefix}topology": {
            name: module.topology() for name, module in sorted(modules.items())
        },
    }
    for name, module in sorted(modules.items()):
        for key, value in module.state_dict().items():
            state[f"{prefix}modules.{name}.{key}"] = value.detach().cpu().clone()
    return state


def load_prototypical_routing_state(root: Any, state: Mapping[str, Any]) -> None:
    prefix = PROTOTYPICAL_ROUTING_STATE_PREFIX
    supplied = {
        str(key): value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    modules = {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, PrototypicalRouterExtension)
    }
    if not modules:
        if supplied:
            raise ValueError(
                "Adapter contains prototypical-routing state, but the configured "
                "model has no prototypical router extension."
            )
        return
    if not supplied:
        raise ValueError("The configured prototypical router requires adapter state.")
    version = supplied.get(f"{prefix}schema_version")
    if int(version) != PROTOTYPICAL_ROUTING_STATE_VERSION:
        raise ValueError(f"Unsupported prototypical-routing state version {version!r}.")
    expected_topology = {
        name: module.topology() for name, module in sorted(modules.items())
    }
    if supplied.get(f"{prefix}topology") != expected_topology:
        raise ValueError("Prototypical-routing topology does not match the model.")
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
                f"Prototypical-routing state for '{name}' has keys "
                f"{sorted(module_state)}, expected {sorted(expected_keys)}."
            )
        module.load_state_dict(module_state, strict=True)
        consumed.update(module_prefix + key for key in module_state)
    unknown = sorted(set(supplied) - consumed)
    if unknown:
        raise ValueError(f"Unknown prototypical-routing state keys: {unknown}.")


__all__ = [
    "collect_prototypical_routing_losses",
    "export_prototypical_routing_state",
    "load_prototypical_routing_state",
    "PROTOTYPICAL_ROUTING_STATE_PREFIX",
    "PROTOTYPICAL_ROUTING_STATE_VERSION",
    "PrototypicalRouterExtension",
    "PrototypicalRoutes",
    "PrototypicalRoutingSpec",
    "prototypical_routing_diagnostics",
    "routing_contrastive_loss",
]
