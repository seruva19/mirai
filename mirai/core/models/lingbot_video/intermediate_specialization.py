"""LingBot binding for bounded SwiGLU-intermediate specialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from mirai.core.moe.adaptation.specialization_loss import (
    coactivated_intermediate_cosine_loss,
)
from mirai.core.moe.monitoring.capture import RoutedExpertTensorCapture
from mirai.vendors.lingbot_video.transformer_lingbot_video import (
    LingBotVideoSparseMoeBlock,
)


@dataclass(frozen=True)
class LingBotIntermediateSpecializationRuntime:
    weight: float
    capture: RoutedExpertTensorCapture | None

    def bind(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, LingBotVideoSparseMoeBlock):
                module.set_expert_intermediate_observer(self.capture)

    def auxiliary_losses(self, model: nn.Module) -> dict[str, Any]:
        if self.capture is None:
            return {}
        terms = list(
            getattr(
                model, "_mirai_checkpoint_expert_intermediate_terms", ()
            )
            or ()
        )
        if not terms:
            terms = self.capture.take_losses()
        if not terms:
            return {}
        return {
            "moe_swiglu_specialization": torch.stack(terms).mean() * self.weight
        }


def build_lingbot_intermediate_specialization_runtime(
    model_params: Any,
) -> LingBotIntermediateSpecializationRuntime:
    weight = float(
        getattr(model_params, "moe_swiglu_specialization_loss_weight", 0.0)
    )
    if weight < 0.0:
        raise ValueError(
            "model.params.moe_swiglu_specialization_loss_weight must be >= 0."
        )
    if weight > 0.0 and int(model_params.experts_per_token) < 2:
        raise ValueError("SwiGLU specialization requires experts_per_token >= 2.")
    capture = (
        RoutedExpertTensorCapture(
            max_tokens=int(model_params.moe_specialization_max_tokens),
            loss_fn=coactivated_intermediate_cosine_loss,
        )
        if weight > 0.0
        else None
    )
    return LingBotIntermediateSpecializationRuntime(weight=weight, capture=capture)


def clear_checkpoint_intermediate_terms(model: nn.Module) -> None:
    model._mirai_checkpoint_expert_intermediate_terms = ()


__all__ = [
    "build_lingbot_intermediate_specialization_runtime",
    "clear_checkpoint_intermediate_terms",
    "LingBotIntermediateSpecializationRuntime",
]
