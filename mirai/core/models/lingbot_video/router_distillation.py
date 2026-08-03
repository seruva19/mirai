"""LingBot binding for frozen-base router distillation."""

from __future__ import annotations

from typing import Any

from mirai.core.moe.adaptation.distillation import RouterDistillationController

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def configure_lingbot_router_distillation(
    pipeline: Any, policy: RouterDistillationController
) -> None:
    if not isinstance(policy, RouterDistillationController):
        raise TypeError("router_distillation requires RouterDistillationController.")
    pipeline._router_distillation_controller = policy


def bind_lingbot_router_distillation(pipeline: Any) -> None:
    controller = pipeline._router_distillation_controller
    if controller is None:
        return
    teachers: dict[str, Any] = {}
    bindings: list[tuple[str, Any, Any]] = []
    for name, router in pipeline.transformer.named_modules():
        params = getattr(router, "parametrizations", None)
        if params is None or "weight" not in params:
            continue
        original = params.weight.original
        if "router" not in name:
            continue
        teachers[name] = original
        bindings.append((name, router, original))
    controller.bind_teacher_weights(teachers)
    for name, router, original in bindings:
        def teacher_loss(tokens: Any, student: Any, *, _name=name, _weight=original):
            with torch.no_grad():
                teacher = F.linear(tokens.float(), _weight.float())
            return controller.loss(_name, student, teacher)

        router.set_router_distillation_extension(teacher_loss)


def collect_lingbot_router_distillation_terms(model: Any) -> list[Any]:
    return [
        module.training_router_distillation
        for module in model.modules()
        if getattr(module, "training_router_distillation", None) is not None
    ]


__all__ = [
    "bind_lingbot_router_distillation",
    "collect_lingbot_router_distillation_terms",
    "configure_lingbot_router_distillation",
]
