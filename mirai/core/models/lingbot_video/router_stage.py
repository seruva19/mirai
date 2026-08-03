"""LingBot binding seam for model-agnostic staged router adaptation."""

from __future__ import annotations

from typing import Any, Iterable

from mirai.core.moe.adaptation.stage_schedule import RouterStageScheduleController
from mirai.core.moe.adaptation.router_training import RouterAdapterBinding


def configure_lingbot_router_stage_policy(
    pipeline: Any,
    policy: RouterStageScheduleController,
) -> None:
    if not isinstance(policy, RouterStageScheduleController):
        raise TypeError("router_stage_schedule requires RouterStageScheduleController.")
    pipeline._router_stage_schedule_controller = policy


def bind_lingbot_router_stage_policy(
    controller: RouterStageScheduleController | None,
    bindings: Iterable[RouterAdapterBinding],
) -> None:
    if controller is None:
        return
    controller.bind_adapters(binding.adapter for binding in bindings)


__all__ = [
    "bind_lingbot_router_stage_policy",
    "configure_lingbot_router_stage_policy",
]
