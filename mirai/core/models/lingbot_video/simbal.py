"""LingBot binding for model-agnostic SIMBAL router regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.moe.adaptation.simbal import SimBalController
from mirai.vendors.lingbot_video.transformer_lingbot_video import (
    LingBotVideoRouter,
)


@dataclass
class LingBotSimBalRuntime:
    controller: SimBalController
    bindings: tuple[tuple[str, LingBotVideoRouter], ...] = ()

    def bind(self, model: Any) -> None:
        routers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, LingBotVideoRouter)
        ]
        if not routers:
            raise ValueError("SIMBAL found no sparse-MoE routers.")
        missing = []
        for name, router in routers:
            parametrizations = getattr(router, "parametrizations", None)
            if parametrizations is None or "weight" not in parametrizations:
                missing.append(name)
        if missing:
            raise ValueError(
                "SIMBAL requires a router adapter on every sparse-MoE layer; "
                "missing: " + ", ".join(missing)
            )
        self.bindings = tuple(routers)

    def auxiliary_losses(self) -> dict[str, Any]:
        weights = {
            name: router.weight
            for name, router in self.bindings
            if bool(router.weight.requires_grad)
        }
        if not weights:
            return {}
        return {"moe_simbal": self.controller.loss(weights)}

    def diagnostics(self) -> dict[str, float | int]:
        return self.controller.diagnostics()


def configure_lingbot_simbal(
    pipeline: Any,
    *,
    policy_name: str,
    policy: Any,
) -> bool:
    if str(policy_name).strip().lower() != "simbal":
        return False
    if not isinstance(policy, SimBalController):
        raise TypeError("simbal requires SimBalController.")
    runtime = LingBotSimBalRuntime(policy)
    pipeline._simbal_runtime = runtime
    if getattr(pipeline, "_lora_report", None) is not None:
        runtime.bind(pipeline.transformer)
    return True


def bind_lingbot_simbal(pipeline: Any) -> None:
    runtime = getattr(pipeline, "_simbal_runtime", None)
    if runtime is not None:
        runtime.bind(pipeline.transformer)


__all__ = [
    "bind_lingbot_simbal",
    "configure_lingbot_simbal",
    "LingBotSimBalRuntime",
]
