"""Momentum-anchored orthogonal gradient projection.

Implements MAOP from Rosetta (arXiv:2607.00293, Eq. 4) over the complete
trainable parameter vector owned by one AdamW optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch

from mirai.core.training.training_policy import TrainingPolicy
from mirai.core.training.training_policy import register_training_policy


_POLICY_NAME = "momentum_anchor"
_ALLOWED_OPTIONS = frozenset(
    {"enabled", "start_after_steps", "epsilon", "min_anchor_norm_sq", "chunk_size"}
)


def _options(config: Any) -> Mapping[str, Any]:
    return getattr(config.training, "policy_options", {}).get(_POLICY_NAME, {})


def _validate_config(config: Any) -> list[str]:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return []
    errors: list[str] = []
    unknown = sorted(set(options) - _ALLOWED_OPTIONS)
    if unknown:
        errors.append("unknown option(s): " + ", ".join(str(key) for key in unknown))
    if str(config.optimizer.type).strip().lower() != "adamw":
        errors.append("requires optimizer.type='adamw'")
    if bool(config.optimizer.stochastic_rounding):
        errors.append("requires optimizer.stochastic_rounding=false")
    if bool(config.training.optimizer_cpu_offload):
        errors.append("requires training.optimizer_cpu_offload=false")
    if int(options.get("start_after_steps", 0)) < 0:
        errors.append("start_after_steps must be >= 0")
    epsilon = float(options.get("epsilon", 1e-12))
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        errors.append("epsilon must be finite and > 0")
    min_norm = float(options.get("min_anchor_norm_sq", 1e-12))
    if not math.isfinite(min_norm) or min_norm < 0.0:
        errors.append("min_anchor_norm_sq must be finite and >= 0")
    if int(options.get("chunk_size", 1 << 20)) <= 0:
        errors.append("chunk_size must be > 0")
    return errors


@dataclass(frozen=True)
class MAOPResult:
    projected: bool
    dot: float
    anchor_norm_sq: float
    parameter_count: int


class MomentumAnchoredOrthogonalProjection:
    """Project a mixed gradient away from an antagonistic AdamW momentum."""

    def __init__(
        self,
        *,
        epsilon: float = 1e-12,
        min_anchor_norm_sq: float = 1e-12,
        chunk_size: int = 1 << 20,
    ) -> None:
        self.epsilon = float(epsilon)
        self.min_anchor_norm_sq = float(min_anchor_norm_sq)
        self.chunk_size = int(chunk_size)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("MAOP epsilon must be finite and > 0.")
        if not math.isfinite(self.min_anchor_norm_sq) or self.min_anchor_norm_sq < 0.0:
            raise ValueError("MAOP min_anchor_norm_sq must be finite and >= 0.")
        if self.chunk_size <= 0:
            raise ValueError("MAOP chunk_size must be > 0.")

    @staticmethod
    def _gradient_anchor_pairs(optimizer: Any) -> list[tuple[Any, Any]]:
        pairs: list[tuple[Any, Any]] = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("MAOP does not support sparse gradients.")
                state = optimizer.state.get(parameter, {})
                anchor = state.get("exp_avg") if isinstance(state, Mapping) else None
                if anchor is None:
                    continue
                if not isinstance(anchor, torch.Tensor) or anchor.shape != gradient.shape:
                    raise RuntimeError(
                        "MAOP requires an AdamW exp_avg tensor matching each gradient."
                    )
                if anchor.dtype != torch.float32:
                    raise RuntimeError("MAOP requires FP32 AdamW exp_avg state.")
                if anchor.device != gradient.device:
                    raise RuntimeError("MAOP requires gradient and exp_avg on one device.")
                pairs.append((gradient, anchor))
        return pairs

    @torch.no_grad()
    def apply(self, optimizer: Any) -> MAOPResult:
        pairs = self._gradient_anchor_pairs(optimizer)
        if not pairs:
            return MAOPResult(False, 0.0, 0.0, 0)

        device = pairs[0][0].device
        dot = torch.zeros((), device=device, dtype=torch.float32)
        norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        for gradient, anchor in pairs:
            if gradient.device != device:
                raise RuntimeError("MAOP requires all optimizer gradients on one device.")
            gradient_flat = gradient.reshape(-1)
            anchor_flat = anchor.reshape(-1)
            for start in range(0, int(gradient_flat.numel()), self.chunk_size):
                end = min(start + self.chunk_size, int(gradient_flat.numel()))
                gradient_chunk = gradient_flat[start:end].float()
                anchor_chunk = anchor_flat[start:end]
                dot.add_(torch.dot(gradient_chunk, anchor_chunk))
                norm_sq.add_(torch.dot(anchor_chunk, anchor_chunk))

        dot_value = float(dot.item())
        norm_value = float(norm_sq.item())
        if dot_value >= 0.0 or norm_value < self.min_anchor_norm_sq:
            return MAOPResult(False, dot_value, norm_value, len(pairs))

        coefficient = dot_value / (norm_value + self.epsilon)
        for gradient, anchor in pairs:
            gradient_flat = gradient.reshape(-1)
            anchor_flat = anchor.reshape(-1)
            for start in range(0, int(gradient_flat.numel()), self.chunk_size):
                end = min(start + self.chunk_size, int(gradient_flat.numel()))
                gradient_flat[start:end].add_(
                    anchor_flat[start:end].to(dtype=gradient.dtype),
                    alpha=-coefficient,
                )
        return MAOPResult(True, dot_value, norm_value, len(pairs))


class MomentumAnchorTrainingPolicy(TrainingPolicy):
    name = _POLICY_NAME
    priority = 900

    def __init__(self, projection: MomentumAnchoredOrthogonalProjection, *, start_after_steps: int) -> None:
        self.projection = projection
        self.start_after_steps = int(start_after_steps)
        self.applied_steps = 0
        self.projection_steps = 0
        self._pending_projection = False

    def before_optimizer_step(self, optimizer: Any) -> None:
        if self.applied_steps < self.start_after_steps:
            return
        result = self.projection.apply(optimizer)
        self._pending_projection = bool(result.projected)

    def after_optimizer_step(self, optimizer: Any, *, applied: bool) -> None:
        _ = optimizer
        if applied:
            self.applied_steps += 1
            if self._pending_projection:
                self.projection_steps += 1
        self._pending_projection = False

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": 1,
            "start_after_steps": self.start_after_steps,
            "epsilon": self.projection.epsilon,
            "min_anchor_norm_sq": self.projection.min_anchor_norm_sq,
            "chunk_size": self.projection.chunk_size,
            "applied_steps": self.applied_steps,
            "projection_steps": self.projection_steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported MAOP policy state version.")
        expected = (
            self.start_after_steps,
            self.projection.epsilon,
            self.projection.min_anchor_norm_sq,
            self.projection.chunk_size,
        )
        observed = (
            int(state.get("start_after_steps", -1)),
            float(state.get("epsilon", float("nan"))),
            float(state.get("min_anchor_norm_sq", float("nan"))),
            int(state.get("chunk_size", -1)),
        )
        if observed != expected:
            raise ValueError("MAOP checkpoint configuration changed.")
        self.applied_steps = int(state.get("applied_steps", 0))
        self.projection_steps = int(state.get("projection_steps", 0))
        if self.applied_steps < 0 or not 0 <= self.projection_steps <= self.applied_steps:
            raise ValueError("Invalid MAOP checkpoint counters.")

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "start_after_steps": self.start_after_steps,
            "epsilon": self.projection.epsilon,
            "min_anchor_norm_sq": self.projection.min_anchor_norm_sq,
            "chunk_size": self.projection.chunk_size,
        }


@register_training_policy(_POLICY_NAME, validate_config=_validate_config)
def build_momentum_anchor_training_policy(config: Any) -> MomentumAnchorTrainingPolicy | None:
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return None
    return MomentumAnchorTrainingPolicy(
        MomentumAnchoredOrthogonalProjection(
            epsilon=float(options.get("epsilon", 1e-12)),
            min_anchor_norm_sq=float(options.get("min_anchor_norm_sq", 1e-12)),
            chunk_size=int(options.get("chunk_size", 1 << 20)),
        ),
        start_after_steps=int(options.get("start_after_steps", 0)),
    )
