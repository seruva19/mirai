"""Optimizer-safe staged trainability for sparse-MoE router adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


ROUTER_STAGE_SCHEDULE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RouterStageStatus:
    step: int
    stage: str
    trainable_parameters: int


class RouterStageScheduleController:
    """Toggle bound router-adapter gradients without changing optimizer ownership."""

    def __init__(self, *, train_start_step: int = 0, freeze_step: int = 0) -> None:
        if int(train_start_step) < 0 or int(freeze_step) < 0:
            raise ValueError("Router stage schedule steps must be non-negative.")
        if int(freeze_step) and int(freeze_step) <= int(train_start_step):
            raise ValueError("Router freeze_step must exceed train_start_step.")
        self.train_start_step = int(train_start_step)
        self.freeze_step = int(freeze_step)
        self._step = 0
        self._parameters: tuple[Any, ...] = ()
        self._original_trainability: tuple[bool, ...] = ()

    @property
    def is_bound(self) -> bool:
        return bool(self._parameters)

    def bind_adapters(self, adapters: Iterable[Any]) -> None:
        parameters: list[Any] = []
        seen: set[int] = set()
        for adapter in adapters:
            for parameter in adapter.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        if not parameters:
            raise ValueError(
                "Router stage schedule requires at least one router adapter parameter."
            )
        if self._parameters:
            if tuple(id(item) for item in parameters) != tuple(
                id(item) for item in self._parameters
            ):
                raise ValueError(
                    "Router stage schedule cannot be rebound to different adapters."
                )
            return
        if not any(bool(parameter.requires_grad) for parameter in parameters):
            raise ValueError(
                "Router stage schedule cannot own adapters frozen by another policy."
            )
        self._parameters = tuple(parameters)
        self._original_trainability = tuple(
            bool(parameter.requires_grad) for parameter in parameters
        )

    def apply_step(self, *, step: int, training: bool) -> RouterStageStatus:
        if int(step) < 0:
            raise ValueError("Router stage schedule step must be non-negative.")
        if not self._parameters:
            raise RuntimeError("Router stage schedule has not been bound to adapters.")
        self._step = int(step)
        if not training:
            return self.status()
        active = self._step >= self.train_start_step and (
            self.freeze_step == 0 or self._step < self.freeze_step
        )
        for parameter, originally_trainable in zip(
            self._parameters, self._original_trainability, strict=True
        ):
            parameter.requires_grad_(bool(active and originally_trainable))
        return self.status()

    def status(self) -> RouterStageStatus:
        if self._step < self.train_start_step:
            stage = "pre_router"
        elif self.freeze_step and self._step >= self.freeze_step:
            stage = "post_router"
        else:
            stage = "router_adaptation"
        return RouterStageStatus(
            step=self._step,
            stage=stage,
            trainable_parameters=sum(
                int(parameter.numel())
                for parameter in self._parameters
                if parameter.requires_grad
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTER_STAGE_SCHEDULE_SCHEMA_VERSION,
            "train_start_step": self.train_start_step,
            "freeze_step": self.freeze_step,
            "step": self._step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            int(state.get("schema_version", 0))
            != ROUTER_STAGE_SCHEDULE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported router-stage-schedule state schema.")
        observed = (
            int(state.get("train_start_step", -1)),
            int(state.get("freeze_step", -1)),
        )
        if observed != (self.train_start_step, self.freeze_step):
            raise ValueError("Router-stage-schedule checkpoint configuration changed.")
        step = int(state.get("step", -1))
        if step < 0:
            raise ValueError("Router-stage-schedule checkpoint step is invalid.")
        self._step = step


__all__ = [
    "ROUTER_STAGE_SCHEDULE_SCHEMA_VERSION",
    "RouterStageScheduleController",
    "RouterStageStatus",
]
