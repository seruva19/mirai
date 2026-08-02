"""Model-agnostic discovery of standard LoRA factor pairs for optimizers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:
    import torch.nn as nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"LoRA optimizer pairing requires torch: {exc}")

from mirai.core.models.adapters.lora_allocation import lora_scale


@dataclass(frozen=True)
class LoRAFactorPair:
    """One dense or grouped standard-LoRA factor pair."""

    name: str
    lora_a: nn.Parameter
    lora_b: nn.Parameter
    scale: float

    @property
    def batch_size(self) -> int:
        return 1 if self.lora_a.ndim == 2 else int(self.lora_a.shape[0])

    @property
    def rank(self) -> int:
        return int(self.lora_a.shape[-2])

    @property
    def in_features(self) -> int:
        return int(self.lora_a.shape[-1])

    @property
    def out_features(self) -> int:
        return int(self.lora_b.shape[-2])

    def signature(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "a_shape": list(self.lora_a.shape),
            "b_shape": list(self.lora_b.shape),
            "scale": float(self.scale),
        }


def _as_float(value: Any) -> float:
    try:
        return float(value.detach().float().item())
    except AttributeError:
        return float(value)


def collect_lora_factor_pairs(
    root: nn.Module,
    *,
    consumer: str,
) -> tuple[LoRAFactorPair, ...]:
    """Collect complete standard LoRA pairs without model-family knowledge."""

    pairs: list[LoRAFactorPair] = []
    seen: set[tuple[int, int]] = set()
    for module_name, module in root.named_modules():
        lora_a = getattr(module, "lora_a", None)
        lora_b = getattr(module, "lora_b", None)
        if not isinstance(lora_a, nn.Parameter) or not isinstance(
            lora_b, nn.Parameter
        ):
            continue
        identity = (id(lora_a), id(lora_b))
        if identity in seen:
            continue
        seen.add(identity)
        if bool(getattr(module, "use_dora", False)):
            raise ValueError(f"{consumer} cannot optimize a DoRA factor pair.")
        if lora_a.ndim not in {2, 3} or lora_b.ndim != lora_a.ndim:
            raise ValueError(
                f"{consumer} target {module_name!r} requires rank-2 or rank-3 "
                "standard LoRA factors."
            )
        if tuple(lora_a.shape[:-2]) != tuple(lora_b.shape[:-2]):
            raise ValueError(
                f"{consumer} target {module_name!r} has incompatible batch axes."
            )
        if int(lora_a.shape[-2]) != int(lora_b.shape[-1]):
            raise ValueError(
                f"{consumer} target {module_name!r} has incompatible rank axes."
            )
        if lora_a.device != lora_b.device:
            raise ValueError(
                f"{consumer} target {module_name!r} has factors on different devices."
            )
        alpha = _as_float(getattr(module, "lora_alpha"))
        rank = int(getattr(module, "rank"))
        scale = lora_scale(
            alpha,
            rank,
            use_rslora=bool(getattr(module, "use_rslora", False)),
        )
        scale *= float(getattr(module, "_lora_scale", 1.0))
        scale *= float(getattr(module, "_rank_schedule_scale", 1.0))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"{consumer} target {module_name!r} requires a finite positive scale."
            )
        pairs.append(
            LoRAFactorPair(
                name=str(module_name),
                lora_a=lora_a,
                lora_b=lora_b,
                scale=float(scale),
            )
        )
    return tuple(pairs)


__all__ = ["LoRAFactorPair", "collect_lora_factor_pairs"]
