"""LingBot binding for checkpoint-safe expert-specialization losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from mirai.core.moe.adaptation.specialization_loss import (
    cross_layer_topk_coupling_loss,
    deterministic_token_sample,
    router_score_variance_loss,
)
from mirai.core.moe.monitoring.capture import RoutedExpertTensorCapture
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoRouter


@dataclass(frozen=True)
class LingBotRouterSpecializationRuntime:
    variance_weight: float
    coupling_weight: float
    max_tokens: int
    top_k: int

    def checkpoint_topk_mass(self, router: LingBotVideoRouter) -> Any:
        scores = deterministic_token_sample(
            router.training_scores, max_tokens=self.max_tokens
        ).float()
        probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        return torch.topk(probabilities, k=self.top_k, dim=-1).values.sum(dim=-1)

    def auxiliary_losses(self, model: nn.Module) -> dict[str, Any]:
        routers = [
            module
            for module in model.modules()
            if isinstance(module, LingBotVideoRouter)
            and getattr(module, "training_scores", None) is not None
        ]
        losses: dict[str, Any] = {}
        if self.variance_weight > 0.0:
            terms = [
                router_score_variance_loss(
                    module.training_scores, max_tokens=self.max_tokens
                )
                for module in routers
            ]
            if terms:
                losses["moe_router_variance"] = (
                    torch.stack(terms).mean() * self.variance_weight
                )
        checkpoint_masses = list(
            getattr(model, "_mirai_checkpoint_router_specialization_terms", ()) or ()
        )
        if self.coupling_weight > 0.0 and checkpoint_masses:
            coupling = -torch.stack(
                [
                    source * target
                    for source, target in zip(
                        checkpoint_masses, checkpoint_masses[1:]
                    )
                ]
            ).mean()
            losses["moe_cross_layer_coupling"] = coupling * self.coupling_weight
        elif self.coupling_weight > 0.0 and routers:
            coupling = cross_layer_topk_coupling_loss(
                [module.training_scores for module in routers],
                top_k=self.top_k,
                max_tokens=self.max_tokens,
            )
            losses["moe_cross_layer_coupling"] = coupling * self.coupling_weight
        return losses


def build_lingbot_router_specialization_runtime(
    model_params: Any,
) -> LingBotRouterSpecializationRuntime:
    variance_weight = float(
        getattr(model_params, "moe_router_variance_loss_weight", 0.0)
    )
    coupling_weight = float(
        getattr(model_params, "moe_cross_layer_coupling_loss_weight", 0.0)
    )
    max_tokens = int(getattr(model_params, "moe_specialization_max_tokens", 256))
    if variance_weight < 0.0:
        raise ValueError("model.params.moe_router_variance_loss_weight must be >= 0.")
    if coupling_weight < 0.0:
        raise ValueError(
            "model.params.moe_cross_layer_coupling_loss_weight must be >= 0."
        )
    if max_tokens <= 0:
        raise ValueError("model.params.moe_specialization_max_tokens must be positive.")
    if coupling_weight > 0.0 and int(getattr(model_params, "num_layers", 0)) < 2:
        raise ValueError("Cross-layer coupling requires at least two MoE layers.")
    return LingBotRouterSpecializationRuntime(
        variance_weight=variance_weight,
        coupling_weight=coupling_weight,
        max_tokens=max_tokens,
        top_k=int(getattr(model_params, "experts_per_token", 0)),
    )


def expert_orthogonality_auxiliary_losses(
    model: nn.Module,
    capture: RoutedExpertTensorCapture,
    *,
    weight: float,
) -> dict[str, Any]:
    terms = list(
        getattr(model, "_mirai_checkpoint_expert_orthogonality_terms", ()) or ()
    )
    if not terms:
        terms = capture.take_losses()
    if not terms:
        return {}
    return {
        "moe_expert_orthogonality": torch.stack(terms).mean() * float(weight)
    }


def clear_checkpoint_auxiliary_terms(model: nn.Module) -> None:
    model._mirai_checkpoint_router_auxiliary_terms = ()
    model._mirai_checkpoint_expert_orthogonality_terms = ()
    model._mirai_checkpoint_router_specialization_terms = ()


__all__ = [
    "build_lingbot_router_specialization_runtime",
    "clear_checkpoint_auxiliary_terms",
    "expert_orthogonality_auxiliary_losses",
    "LingBotRouterSpecializationRuntime",
]
