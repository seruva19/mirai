"""LingBot router runtime state, auxiliary losses, and assignment telemetry."""

from __future__ import annotations

from typing import Any

from mirai.core.moe.routing.contracts import RoutingStats
from mirai.core.moe.adaptation.losses import (
    expert_combination_usage,
    router_similarity_loss,
)
from mirai.core.moe.adaptation.phi_balance import PhiBalanceController
from mirai.core.moe.adaptation.specialization_loss import (
    active_expert_orthogonality_loss,
)
from mirai.core.moe.monitoring.capture import RoutedExpertTensorCapture
from mirai.core.moe.monitoring.gradient_ratio import (
    BalanceGradientProbe,
    RouterProbabilityTarget,
)
from mirai.core.moe.routing.expert_choice import (
    summarize_expert_choice_coverage,
)
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoRouter
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoSparseMoeBlock
from mirai.vendors.lingbot_video.transformer_lingbot_video import _block_router_auxiliary_terms

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


def _routing_stats(model: nn.Module) -> list[RoutingStats]:
    stats: list[RoutingStats] = []
    for name, module in model.named_modules():
        if not isinstance(module, LingBotVideoRouter):
            continue
        top_indices = getattr(module, "last_top_indices", None)
        top_scores = getattr(module, "last_top_scores", None)
        scores = getattr(module, "last_scores", None)
        if top_indices is None or top_scores is None or scores is None:
            continue
        num_experts = int(module.num_experts)
        active_mask = getattr(module, "last_route_active_mask", None)
        selected = (
            top_indices[active_mask]
            if active_mask is not None
            else top_indices.reshape(-1)
        )
        counts = torch.bincount(selected, minlength=num_experts).float()
        selected_tokens = max(1, int(selected.numel()))
        expert_fraction = tuple((counts / float(selected_tokens)).detach().cpu().tolist())
        mean_prob = tuple(scores.float().mean(dim=0).detach().cpu().tolist())
        token_probs = top_scores.float()
        entropy = -(token_probs * (token_probs.clamp_min(1e-12).log())).sum(dim=-1).mean()
        stats.append(
            RoutingStats(
                layer=name,
                tokens=int(top_indices.shape[0]),
                selected_tokens=int(selected.numel()),
                expert_fraction=expert_fraction,
                mean_router_probability=mean_prob,
                dead_experts=int((counts == 0).sum().item()),
                routing_entropy=float(entropy.detach().cpu().item()),
            )
        )
    return stats


def _expert_choice_coverage_metrics(
    model: nn.Module,
    *,
    alarm_threshold: float,
) -> dict[str, float]:
    """Bind provider router state to the model-agnostic coverage contract."""
    if float(alarm_threshold) <= 0.0:
        return {}
    fractions: list[float] = []
    for module in model.modules():
        if not isinstance(module, LingBotVideoRouter):
            continue
        decision = getattr(module, "training_expert_choice_decision", None)
        if decision is not None:
            fractions.append(float(decision.coverage.coverage_fraction))
    summary = summarize_expert_choice_coverage(
        fractions,
        alarm_threshold=float(alarm_threshold),
    )
    if summary is None:
        return {}
    return {
        "moe_expert_choice_coverage_fraction": summary.mean_fraction,
        "moe_expert_choice_min_coverage_fraction": summary.minimum_fraction,
        "moe_expert_choice_coverage_alarm": float(summary.below_threshold),
    }


def _build_phi_balance_runtime(
    model_params: Any,
    *,
    legacy_aux_weight: float,
    legacy_bias_rate: float,
) -> tuple[float, PhiBalanceController | None]:
    weight = float(getattr(model_params, "moe_phi_balance_weight", 0.0))
    if weight < 0.0:
        raise ValueError("model.params.moe_phi_balance_weight must be >= 0.")
    if weight == 0.0:
        return 0.0, None
    if float(legacy_aux_weight) != 0.0 or float(legacy_bias_rate) != 0.0:
        raise ValueError(
            "phi-balancing cannot be combined with the existing MoE balance "
            "objective; set moe_aux_loss_weight=0 and moe_bias_update_rate=0."
        )
    return weight, PhiBalanceController(
        ema_rate=float(model_params.moe_phi_balance_ema_rate),
        potential=str(model_params.moe_phi_balance_potential),
    )


def _build_expert_orthogonality_runtime(
    model_params: Any,
) -> tuple[float, RoutedExpertTensorCapture | None]:
    weight = float(
        getattr(model_params, "moe_expert_orthogonality_loss_weight", 0.0)
    )
    if weight < 0.0:
        raise ValueError(
            "model.params.moe_expert_orthogonality_loss_weight must be >= 0."
        )
    if weight == 0.0:
        return 0.0, None
    if int(model_params.experts_per_token) < 2:
        raise ValueError("Expert-output orthogonality requires experts_per_token >= 2.")
    return weight, RoutedExpertTensorCapture(
        max_tokens=int(model_params.moe_specialization_max_tokens),
        loss_fn=active_expert_orthogonality_loss,
    )


def _install_expert_output_observer(
    model: nn.Module, observer: RoutedExpertTensorCapture | None
) -> None:
    for module in model.modules():
        if isinstance(module, LingBotVideoSparseMoeBlock):
            module.set_expert_output_observer(observer)


def _clear_router_runtime_state(model: nn.Module) -> None:
    for module in model.modules():
        if not isinstance(module, LingBotVideoRouter):
            continue
        for name in (
            "last_top_indices",
            "last_top_scores",
            "last_scores",
            "training_top_indices",
            "training_unbiased_top_indices",
            "training_logits",
            "training_scores",
            "training_gradient_probabilities",
            "training_subset_kl",
            "training_router_distillation",
            "training_expert_choice_decision",
            "last_subset_ids",
            "last_unique_experts",
            "last_dataset_affinity_hit_rate",
            "last_route_active_mask",
            "last_logical_top_indices",
            "last_logical_top_scores",
            "runtime_lightweight_decision",
            "training_lightweight_balance_loss",
            "training_lightweight_z_loss",
            "training_aux_tokens_per_sample",
        ):
            setattr(module, name, None)


def _collect_router_auxiliary_terms(
    model: nn.Module,
) -> tuple[list[Any], list[Any], list[float | None]]:
    """Per-block (balance, z) auxiliary terms, collected exactly once.

    Uses the aggressive-checkpoint terms when present (collected inside the
    checkpointed block calls), otherwise recomputes from the routers' stored
    training logits. Terms are collected regardless of loss weights so their
    raw VALUES are available for per-step telemetry even at weight 0.
    """
    checkpoint_terms = tuple(
        getattr(model, "_mirai_checkpoint_router_auxiliary_terms", ()) or ()
    )
    if checkpoint_terms:
        sparse_routers = [
            block.ffn.router
            for block in getattr(model, "blocks", ())
            if isinstance(getattr(block, "ffn", None), LingBotVideoSparseMoeBlock)
        ]
        if len(sparse_routers) != len(checkpoint_terms):
            raise RuntimeError(
                "Checkpointed router terms disagree with sparse-MoE topology."
            )
        return (
            [item[0] for item in checkpoint_terms],
            [item[1] for item in checkpoint_terms],
            [
                getattr(router, "_mirai_z_loss_weight", None)
                for router in sparse_routers
            ],
        )
    balance_terms: list[Any] = []
    z_terms: list[Any] = []
    z_weights: list[float | None] = []
    for block in getattr(model, "blocks", ()):
        if not isinstance(getattr(block, "ffn", None), LingBotVideoSparseMoeBlock):
            continue
        logits = getattr(block.ffn.router, "training_logits", None)
        if logits is None:
            continue
        balance, z_loss = _block_router_auxiliary_terms(block, like=logits)
        balance_terms.append(balance)
        z_terms.append(z_loss)
        z_weights.append(getattr(block.ffn.router, "_mirai_z_loss_weight", None))
    return balance_terms, z_terms, z_weights


def _build_balance_gradient_probe(
    model: nn.Module,
    *,
    objectives: dict[str, Any],
) -> BalanceGradientProbe | None:
    """Bind native router state to the model-agnostic probability contract."""

    targets: list[RouterProbabilityTarget] = []
    for name, module in model.named_modules():
        if not isinstance(module, LingBotVideoRouter):
            continue
        probabilities = getattr(
            module, "training_gradient_probabilities", None
        )
        if probabilities is None:
            continue
        targets.append(
            RouterProbabilityTarget(
                layer=str(name),
                probabilities=probabilities,
            )
        )
    if not targets or not objectives:
        return None
    return BalanceGradientProbe(
        targets=tuple(targets),
        objectives=dict(objectives),
    )


def _weighted_router_auxiliary_losses(
    balance_terms: list[Any],
    z_terms: list[Any],
    *,
    load_balance_weight: float,
    z_loss_weight: float,
    z_loss_weights: list[float | None] | None = None,
) -> dict[str, Any]:
    """Weighted auxiliary loss terms in the reference operation order."""
    losses: dict[str, Any] = {}
    if balance_terms and float(load_balance_weight) != 0.0:
        losses["moe_load_balance"] = torch.stack(balance_terms).mean() * float(
            load_balance_weight
        )
    if z_terms and z_loss_weights is not None and any(
        weight is not None for weight in z_loss_weights
    ):
        if len(z_terms) != len(z_loss_weights):
            raise ValueError("Per-layer z-loss weights disagree with router terms.")
        resolved = [
            float(z_loss_weight) if weight is None else float(weight)
            for weight in z_loss_weights
        ]
        if any(weight != 0.0 for weight in resolved):
            losses["moe_router_z"] = torch.stack(
                [term * weight for term, weight in zip(z_terms, resolved, strict=True)]
            ).mean()
    elif z_terms and float(z_loss_weight) != 0.0:
        losses["moe_router_z"] = torch.stack(z_terms).mean() * float(z_loss_weight)
    return losses


def _phi_balance_auxiliary_terms(
    model: nn.Module, controller: PhiBalanceController
) -> list[Any]:
    """Bind LingBot pre-top-k soft probabilities to generic phi-balancing."""
    terms: list[Any] = []
    for name, module in model.named_modules():
        if not isinstance(module, LingBotVideoRouter):
            continue
        probabilities = getattr(module, "training_scores", None)
        if probabilities is not None:
            normalized = probabilities.float()
            normalized = normalized / normalized.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)
            terms.append(controller.loss(name, normalized))
    return terms


def _router_auxiliary_losses(
    model: nn.Module,
    *,
    load_balance_weight: float,
    z_loss_weight: float,
) -> dict[str, Any]:
    balance_terms, z_terms, z_weights = _collect_router_auxiliary_terms(model)
    return _weighted_router_auxiliary_losses(
        balance_terms,
        z_terms,
        load_balance_weight=load_balance_weight,
        z_loss_weight=z_loss_weight,
        z_loss_weights=z_weights,
    )


def _router_similarity_terms(
    model: nn.Module,
) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    terms: list[Any] = []
    telemetry: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LingBotVideoRouter):
            continue
        indices = getattr(module, "training_top_indices", None)
        logits = getattr(module, "training_logits", None)
        if indices is None or logits is None:
            continue
        terms.append(
            router_similarity_loss(
                indices,
                logits,
                num_experts=int(module.num_experts),
            )
        )
        telemetry[name] = expert_combination_usage(
            indices.detach(),
            num_experts=int(module.num_experts),
        )
    return terms, telemetry


def _spatiotemporal_routing_terms(
    model: nn.Module,
    *,
    video_offsets: tuple[int, ...],
    grid: tuple[int, int, int],
    max_edges: int,
) -> list[Any]:
    from mirai.core.moe.adaptation.spatiotemporal import (
        spatiotemporal_routing_consistency_loss,
    )

    terms: list[Any] = []
    for module in model.modules():
        if not isinstance(module, LingBotVideoRouter):
            continue
        probabilities = getattr(module, "training_scores", None)
        if probabilities is not None:
            terms.append(
                spatiotemporal_routing_consistency_loss(
                    probabilities,
                    video_offsets=video_offsets,
                    grid=grid,
                    max_edges=max_edges,
                )
            )
    return terms


def _router_assignment_counts(model: nn.Module) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LingBotVideoRouter):
            continue
        indices = getattr(module, "training_top_indices", None)
        if indices is None:
            continue
        counts[name] = torch.bincount(
            indices.detach().reshape(-1), minlength=int(module.num_experts)
        ).to(device="cpu", dtype=torch.float64)
    return counts
