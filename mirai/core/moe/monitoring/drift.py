"""Detached, checkpointable router-reference drift diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ROUTER_DRIFT_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LayerRouterLogitDrift:
    normalized_rms: float
    max_expert_normalized_abs: float
    most_drifted_expert: int


@dataclass(frozen=True)
class RouterLogitDriftObservation:
    mean_normalized_rms: float
    layers: dict[str, LayerRouterLogitDrift]


class RouterLogitReferenceTracker:
    """Track bounded per-expert raw-logit means against the first observation."""

    def __init__(self) -> None:
        self._reference: dict[str, Any] = {}

    @property
    def captured(self) -> bool:
        return bool(self._reference)

    def observe(self, logits_by_layer: Mapping[str, Any]) -> float | None:
        observation = self.observe_detailed(logits_by_layer)
        return None if observation is None else observation.mean_normalized_rms

    def observe_detailed(
        self, logits_by_layer: Mapping[str, Any]
    ) -> RouterLogitDriftObservation | None:
        if torch is None:
            return None
        summaries: dict[str, Any] = {}
        for raw_name, raw_logits in logits_by_layer.items():
            if not torch.is_tensor(raw_logits):
                continue
            logits = raw_logits.detach().float()
            if logits.ndim < 2 or int(logits.shape[-1]) < 2:
                continue
            name = str(raw_name)
            summaries[name] = logits.reshape(-1, logits.shape[-1]).mean(dim=0).cpu()
        if not summaries:
            return None
        if not self._reference:
            self._reference = summaries
            return None
        if set(summaries) != set(self._reference):
            raise ValueError("Router logit reference layer topology changed.")
        layer_drifts: dict[str, LayerRouterLogitDrift] = {}
        for name, current in summaries.items():
            reference = self._reference[name]
            if tuple(current.shape) != tuple(reference.shape):
                raise ValueError(f"Router logit width changed for layer {name!r}.")
            numerator = (current - reference).square().mean().sqrt()
            denominator = reference.square().mean().sqrt().clamp_min(1e-12)
            normalized_rms = float((numerator / denominator).item())
            expert_delta = (current - reference).abs() / reference.abs().clamp_min(1e-12)
            most_drifted = int(expert_delta.argmax().item())
            layer_drifts[name] = LayerRouterLogitDrift(
                normalized_rms=normalized_rms,
                max_expert_normalized_abs=float(expert_delta[most_drifted].item()),
                most_drifted_expert=most_drifted,
            )
        return RouterLogitDriftObservation(
            mean_normalized_rms=float(
                sum(item.normalized_rms for item in layer_drifts.values())
                / len(layer_drifts)
            ),
            layers=layer_drifts,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTER_DRIFT_STATE_SCHEMA_VERSION,
            "reference": {
                name: value.detach().cpu().clone()
                for name, value in sorted(self._reference.items())
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", 0)) != ROUTER_DRIFT_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported router drift state schema {state.get('schema_version')!r}."
            )
        raw_reference = state.get("reference")
        if not isinstance(raw_reference, Mapping):
            raise ValueError("Router drift state requires a reference mapping.")
        reference: dict[str, Any] = {}
        for raw_name, raw_value in raw_reference.items():
            value = torch.as_tensor(raw_value).detach().float().cpu()
            if value.ndim != 1 or int(value.numel()) < 2 or not torch.isfinite(value).all():
                raise ValueError(f"Invalid router drift reference for {raw_name!r}.")
            reference[str(raw_name)] = value.clone()
        self._reference = reference


def router_gradient_update_ratios(
    named_params: list[tuple[str, Any]],
    *,
    lr: float,
) -> dict[str, float]:
    """Return global router gradient/parameter and update/parameter RMS ratios."""
    if torch is None:
        return {}
    weight_sq = 0.0
    grad_sq = 0.0
    count = 0
    for name, parameter in named_params:
        if "router" not in str(name).lower():
            continue
        gradient = getattr(parameter, "_mirai_cpu_grad", None)
        if gradient is None:
            gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        weight = parameter.detach().float()
        grad = gradient.detach().float()
        if weight.numel() != grad.numel():
            raise ValueError(f"Router gradient shape mismatch for {name!r}.")
        weight_sq += float(weight.square().sum().item())
        grad_sq += float(grad.square().sum().item())
        count += int(weight.numel())
    if count == 0:
        return {}
    weight_rms = max((weight_sq / count) ** 0.5, 1e-30)
    grad_ratio = (grad_sq / count) ** 0.5 / weight_rms
    return {
        "moe_router_grad_param_ratio": float(grad_ratio),
        "moe_router_update_param_ratio": float(abs(float(lr)) * grad_ratio),
    }


def router_drift_checkpoint_state(
    tracker: RouterLogitReferenceTracker,
) -> dict[str, Any]:
    state = tracker.state_dict()
    output = {
        "router_drift.schema_version": torch.tensor(
            int(state["schema_version"]), dtype=torch.int64
        )
    }
    output.update(
        {
            f"router_drift.reference.{name}": value
            for name, value in state["reference"].items()
        }
    )
    return output


def load_router_drift_checkpoint_state(
    tracker: RouterLogitReferenceTracker,
    state: Mapping[str, Any],
) -> None:
    prefix = "router_drift.reference."
    references = {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    schema = state.get("router_drift.schema_version")
    if schema is None and not references:
        return
    tracker.load_state_dict(
        {
            "schema_version": int(torch.as_tensor(schema).item()),
            "reference": references,
        }
    )


__all__ = [
    "LayerRouterLogitDrift",
    "ROUTER_DRIFT_STATE_SCHEMA_VERSION",
    "RouterLogitDriftObservation",
    "RouterLogitReferenceTracker",
    "load_router_drift_checkpoint_state",
    "router_drift_checkpoint_state",
    "router_gradient_update_ratios",
]
