"""LingBot binding for model-agnostic routing-health diagnostics."""

from __future__ import annotations

from typing import Any

from mirai.core.moe.monitoring.drift import RouterLogitReferenceTracker
from mirai.core.moe.monitoring.health import DeadlockTracker
from mirai.core.moe.monitoring.health import expert_output_cossim
from mirai.core.moe.monitoring.fisher import summarize_fisher_specialization
from mirai.core.moe.monitoring.preemptive import summarize_router_mechanisms
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoRouter

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def collect_lingbot_routing_health(
    transformer: Any,
    *,
    deadlock_tracker: DeadlockTracker | None,
    drift_tracker: RouterLogitReferenceTracker,
    training: bool,
    timesteps: Any | None = None,
    latent_shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Translate LingBot router runtime fields into generic health contracts."""
    score_columns: list[Any] = []
    probability_layers: list[Any] = []
    router_mechanism_layers: list[tuple[Any, Any]] = []
    logits_by_layer: dict[str, Any] = {}
    monopoly_by_layer: dict[str, float] = {}
    layer_order: list[str] = []
    for name, module in transformer.named_modules():
        if not isinstance(module, LingBotVideoRouter):
            continue
        layer_order.append(name)
        scores = getattr(module, "last_scores", None)
        if scores is not None:
            score_columns.append(scores)
            probability_layers.append(scores)
            weight = module._execution_weight(
                device=scores.device,
                dtype=torch.float32,
            ).detach()
            router_mechanism_layers.append((weight, scores))
        logits = getattr(module, "training_logits", None)
        if logits is not None:
            logits_by_layer[name] = logits
        indices = getattr(module, "last_top_indices", None)
        if indices is None:
            continue
        counts = torch.bincount(
            indices.detach().reshape(-1), minlength=int(module.num_experts)
        ).float()
        monopoly_by_layer[name] = float(
            (counts / counts.sum().clamp_min(1.0)).max().item()
        )
    metrics: dict[str, Any] = {}
    cossim = expert_output_cossim(score_columns)
    if cossim is not None:
        metrics["moe_expert_output_cossim"] = cossim
    fisher = summarize_fisher_specialization(probability_layers)
    if fisher is not None:
        metrics.update(fisher.to_metrics())
    router_mechanisms = summarize_router_mechanisms(router_mechanism_layers)
    if router_mechanisms is not None:
        metrics.update(router_mechanisms.to_metrics())
    if bool(training):
        logit_drift = drift_tracker.observe_detailed(logits_by_layer)
        if logit_drift is not None:
            metrics["moe_router_logit_drift"] = logit_drift.mean_normalized_rms
            context: dict[str, Any] = {
                "layers": {
                    name: {
                        "normalized_rms": item.normalized_rms,
                        "max_expert_normalized_abs": item.max_expert_normalized_abs,
                        "most_drifted_expert": item.most_drifted_expert,
                    }
                    for name, item in sorted(logit_drift.layers.items())
                }
            }
            if torch.is_tensor(timesteps) and int(timesteps.numel()) > 0:
                values = timesteps.detach().float().reshape(-1)
                context["timestep"] = {
                    "min": float(values.min().cpu().item()),
                    "max": float(values.max().cpu().item()),
                    "mean": float(values.mean().cpu().item()),
                }
            if latent_shape is not None and len(latent_shape) == 5:
                context["latent_resolution"] = {
                    "frames": int(latent_shape[2]),
                    "height": int(latent_shape[3]),
                    "width": int(latent_shape[4]),
                }
            metrics["moe_router_logit_drift_context"] = context
        if deadlock_tracker is not None:
            metrics.update(
                deadlock_tracker.update(
                    monopoly_by_layer,
                    layer_order=layer_order,
                )
            )
    return metrics


__all__ = ["collect_lingbot_routing_health"]
