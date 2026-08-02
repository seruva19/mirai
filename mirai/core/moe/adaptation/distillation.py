"""Frozen-reference distillation for sparse-MoE router adaptation."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from mirai.core.moe.adaptation.distillation_schedule import (
    router_distillation_weight_scale,
)

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


ROUTER_DISTILLATION_STATE_SCHEMA_VERSION = 2


def router_distillation_kl(
    student_logits: Any,
    teacher_logits: Any,
    *,
    temperature: float,
) -> Any:
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("Router distillation logits must have matching [..., experts] shapes.")
    if float(temperature) <= 0.0:
        raise ValueError("Router distillation temperature must be positive.")
    scale = float(temperature)
    student_log = F.log_softmax(student_logits.float() / scale, dim=-1)
    teacher_log = F.log_softmax(teacher_logits.detach().float() / scale, dim=-1)
    teacher_prob = teacher_log.exp()
    return (
        (teacher_prob * (teacher_log - student_log)).sum(dim=-1).mean()
        * (scale * scale)
    )


def router_weight_fingerprint(weights: Mapping[str, Any]) -> str:
    if not weights:
        raise ValueError("Router teacher lineage requires at least one weight.")
    digest = hashlib.sha256()
    for name, weight in sorted(weights.items()):
        tensor = weight.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return "sha256:" + digest.hexdigest()


class RouterDistillationController:
    def __init__(
        self,
        *,
        weight: float,
        temperature: float = 1.0,
        start_step: int = 0,
        end_step: int = 0,
        weight_schedule: str = "constant",
    ) -> None:
        if float(weight) <= 0.0 or float(temperature) <= 0.0:
            raise ValueError("Router distillation weight and temperature must be positive.")
        if int(start_step) < 0 or int(end_step) < 0:
            raise ValueError("Router distillation schedule steps must be non-negative.")
        if int(end_step) and int(end_step) <= int(start_step):
            raise ValueError("Router distillation end_step must exceed start_step.")
        normalized_schedule = str(weight_schedule).strip().lower()
        router_distillation_weight_scale(
            int(start_step),
            start_step=int(start_step),
            end_step=int(end_step),
            schedule=normalized_schedule,
        )
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.weight_schedule = normalized_schedule
        self._step = 0
        self._training = False
        self._teacher_fingerprint = ""
        self._loaded_fingerprint = ""

    @property
    def teacher_fingerprint(self) -> str:
        return self._teacher_fingerprint

    def bind_teacher_weights(self, weights: Mapping[str, Any]) -> None:
        observed = router_weight_fingerprint(weights)
        if self._teacher_fingerprint and observed != self._teacher_fingerprint:
            raise ValueError("Router distillation teacher weights changed after binding.")
        if self._loaded_fingerprint and observed != self._loaded_fingerprint:
            raise ValueError("Router distillation teacher lineage does not match checkpoint.")
        self._teacher_fingerprint = observed

    def bind_step(self, *, step: int, training: bool) -> None:
        if int(step) < 0:
            raise ValueError("Router distillation step must be non-negative.")
        self._step = int(step)
        self._training = bool(training)

    def loss(self, layer_name: str, student_logits: Any, teacher_logits: Any) -> Any | None:
        _ = layer_name
        active = (
            self._training
            and self._step >= self.start_step
            and (self.end_step == 0 or self._step < self.end_step)
        )
        if not active:
            return None
        if not self._teacher_fingerprint:
            raise RuntimeError("Router distillation teacher lineage is not bound.")
        return router_distillation_kl(
            student_logits,
            teacher_logits,
            temperature=self.temperature,
        ) * self.weight * router_distillation_weight_scale(
            self._step,
            start_step=self.start_step,
            end_step=self.end_step,
            schedule=self.weight_schedule,
        )

    def state_dict(self) -> dict[str, Any]:
        if not self._teacher_fingerprint:
            raise RuntimeError("Router distillation teacher lineage is not bound.")
        return {
            "schema_version": ROUTER_DISTILLATION_STATE_SCHEMA_VERSION,
            "weight": self.weight,
            "temperature": self.temperature,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "weight_schedule": self.weight_schedule,
            "step": self._step,
            "teacher_fingerprint": self._teacher_fingerprint,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        schema_version = int(state.get("schema_version", 0))
        if schema_version not in {1, ROUTER_DISTILLATION_STATE_SCHEMA_VERSION}:
            raise ValueError("Unsupported router-distillation state schema.")
        observed = (
            float(state.get("weight", -1.0)),
            float(state.get("temperature", -1.0)),
            int(state.get("start_step", -1)),
            int(state.get("end_step", -1)),
            str(state.get("weight_schedule", "constant"))
            if schema_version >= 2
            else "constant",
        )
        expected = (
            self.weight,
            self.temperature,
            self.start_step,
            self.end_step,
            self.weight_schedule,
        )
        if observed != expected:
            raise ValueError("Router-distillation checkpoint configuration changed.")
        fingerprint = str(state.get("teacher_fingerprint", ""))
        if not fingerprint.startswith("sha256:"):
            raise ValueError("Router-distillation checkpoint teacher lineage is invalid.")
        if self._teacher_fingerprint and fingerprint != self._teacher_fingerprint:
            raise ValueError("Router distillation teacher lineage does not match checkpoint.")
        self._loaded_fingerprint = fingerprint
        self._step = int(state.get("step", -1))
        if self._step < 0:
            raise ValueError("Router-distillation checkpoint step is invalid.")


__all__ = [
    "ROUTER_DISTILLATION_STATE_SCHEMA_VERSION",
    "RouterDistillationController",
    "router_distillation_kl",
    "router_weight_fingerprint",
]
