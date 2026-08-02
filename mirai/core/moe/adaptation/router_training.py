"""Router-adapter trainability policy independent of model-family layout."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable


@dataclass(frozen=True)
class RouterAdapterBinding:
    weight: Any
    adapter: Any

    def freeze(self) -> None:
        self.weight.requires_grad_(False)
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)


@dataclass(frozen=True)
class RouterTrainingResult:
    adapted_count: int
    frozen_count: int
    override: bool | None


@dataclass(frozen=True)
class RouterTrainingPolicy:
    override: bool | None

    def apply(
        self,
        bindings: Iterable[RouterAdapterBinding],
        *,
        target_preset: str,
        logger: logging.Logger | None = None,
    ) -> RouterTrainingResult:
        resolved = tuple(bindings)
        adapted_count = len(resolved)
        frozen_count = 0
        if self.override is False:
            for binding in resolved:
                binding.freeze()
            frozen_count = adapted_count
            if logger is not None and frozen_count:
                logger.info(
                    "adapter.train_router=false: froze %d router adapter(s).",
                    frozen_count,
                )
        elif self.override is True and not adapted_count:
            if logger is not None:
                logger.warning(
                    "adapter.train_router=true but target_preset '%s' does not "
                    "adapt the router; no router parameters are trainable.",
                    target_preset,
                )
        elif self.override is None and adapted_count and logger is not None:
            logger.warning(
                "target_preset '%s' adapts the MoE router; set "
                "adapter.train_router=false to freeze it explicitly.",
                target_preset,
            )
        return RouterTrainingResult(
            adapted_count=adapted_count,
            frozen_count=frozen_count,
            override=self.override,
        )
