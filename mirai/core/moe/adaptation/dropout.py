"""No-token-drop regularization for selected sparse-MoE expert outputs.

Conceptual source: ExPLoRe, arXiv:2606.31201, downstream Expert Dropout.
"""

from __future__ import annotations

from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


EXPERT_DROPOUT_STATE_SCHEMA_VERSION = 1


def apply_expert_route_dropout(top_scores: Any, keep_mask: Any) -> Any:
    """Mask selected expert contributions while preserving each token's gate mass."""
    if not torch.is_tensor(top_scores) or top_scores.ndim != 2:
        raise ValueError("expert dropout scores must have shape [tokens, top_k].")
    if not torch.is_tensor(keep_mask) or keep_mask.shape != top_scores.shape:
        raise ValueError("expert dropout keep mask must match selected scores.")
    if keep_mask.dtype != torch.bool:
        raise ValueError("expert dropout keep mask must be boolean.")
    if bool((keep_mask.sum(dim=-1) == 0).any()):
        raise ValueError("expert dropout must retain at least one route per token.")
    original_mass = top_scores.sum(dim=-1, keepdim=True)
    retained = top_scores * keep_mask.to(dtype=top_scores.dtype)
    retained_mass = retained.sum(dim=-1, keepdim=True)
    if bool(((original_mass != 0) & (retained_mass == 0)).any()):
        raise ValueError("expert dropout retained only zero-mass routes.")
    scale = torch.where(
        retained_mass != 0,
        original_mass / retained_mass.clamp_min(torch.finfo(top_scores.dtype).tiny),
        torch.ones_like(retained_mass),
    )
    return retained * scale


class ExpertDropoutController:
    """Step-bound stochastic route regularizer with checkpointed configuration."""

    def __init__(
        self,
        *,
        probability: float,
        start_step: int = 0,
        end_step: int = 0,
    ) -> None:
        if not 0.0 < float(probability) < 1.0:
            raise ValueError("Expert dropout probability must be between 0 and 1.")
        if int(start_step) < 0 or int(end_step) < 0:
            raise ValueError("Expert dropout schedule steps must be non-negative.")
        if int(end_step) and int(end_step) <= int(start_step):
            raise ValueError("Expert dropout end_step must exceed start_step.")
        self.probability = float(probability)
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self._step = 0
        self._training = False

    def bind_step(self, *, step: int, training: bool) -> None:
        if int(step) < 0:
            raise ValueError("Expert dropout step must be non-negative.")
        self._step = int(step)
        self._training = bool(training)

    def is_active(self, *, training: bool) -> bool:
        return bool(
            self._training
            and training
            and self._step >= self.start_step
            and (self.end_step == 0 or self._step < self.end_step)
        )

    def regularize(
        self,
        layer_name: str,
        top_indices: Any,
        top_scores: Any,
        *,
        training: bool,
    ) -> Any:
        _ = layer_name
        if not self.is_active(training=training):
            return top_scores
        if top_indices.shape != top_scores.shape or top_scores.ndim != 2:
            raise ValueError("Expert dropout routes must have shape [tokens, top_k].")
        if int(top_scores.shape[1]) < 2:
            raise ValueError("Expert dropout requires top_k >= 2.")
        keep = torch.rand(
            top_scores.shape, device=top_scores.device, dtype=torch.float32
        ) >= self.probability
        empty = keep.sum(dim=-1) == 0
        if bool(empty.any()):
            fallback = top_scores.detach().argmax(dim=-1, keepdim=True)
            fallback_keep = torch.zeros_like(keep)
            fallback_keep.scatter_(1, fallback, True)
            keep = keep | (fallback_keep & empty.unsqueeze(1))
        return apply_expert_route_dropout(top_scores, keep)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPERT_DROPOUT_STATE_SCHEMA_VERSION,
            "probability": self.probability,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "step": self._step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = (self.probability, self.start_step, self.end_step)
        observed = (
            float(state.get("probability", -1.0)),
            int(state.get("start_step", -1)),
            int(state.get("end_step", -1)),
        )
        if int(state.get("schema_version", 0)) != EXPERT_DROPOUT_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported expert-dropout state schema.")
        if observed != expected:
            raise ValueError("Expert-dropout checkpoint configuration changed.")
        self.bind_step(step=int(state.get("step", -1)), training=False)


__all__ = [
    "EXPERT_DROPOUT_STATE_SCHEMA_VERSION",
    "ExpertDropoutController",
    "apply_expert_route_dropout",
]
