"""LingBot bindings for mechanism-driven attention monitoring."""

from __future__ import annotations

from typing import Any

from mirai.core.models.adapters.lora import LoRALinear
from mirai.core.models.adapters.lora_allocation import lora_scale
from mirai.core.moe.monitoring.preemptive import AttentionQKState
from mirai.core.moe.monitoring.preemptive import LowRankProjectionState
from mirai.core.moe.monitoring.preemptive import PreemptiveAttentionMonitor
from mirai.vendors.lingbot_video.transformer_lingbot_video import (
    LingBotVideoAttention,
)

try:
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    nn = object  # type: ignore[assignment]


def _static_projection_state(
    projection: Any,
    *,
    num_heads: int,
) -> LowRankProjectionState | None:
    if not isinstance(projection, LoRALinear):
        return None
    if not isinstance(projection.base, nn.Linear) or bool(projection.use_dora):
        return None
    if (
        projection._timestep_mask_per_sample is not None
        or projection._timestep_mask_uniform is not None
        or projection._tc_gate_hypernet is not None
    ):
        return None
    scale = (
        lora_scale(
            float(projection.lora_alpha.detach().float().item()),
            int(projection.rank),
            use_rslora=bool(projection.use_rslora),
        )
        * float(projection.lora_scale)
        * float(projection._rank_schedule_scale)
    )
    return LowRankProjectionState(
        base_weight=projection.base.weight,
        factor_a=projection.lora_a,
        factor_b=projection.lora_b,
        scale=float(scale),
        num_heads=int(num_heads),
    )


def collect_lingbot_attention_qk_states(transformer: Any) -> tuple[AttentionQKState, ...]:
    """Return static effective Q/K LoRA targets supported by Proposition 2."""

    states: list[AttentionQKState] = []
    for name, module in transformer.named_modules():
        if not isinstance(module, LingBotVideoAttention):
            continue
        query = _static_projection_state(
            module.to_q,
            num_heads=int(module.num_heads),
        )
        key = _static_projection_state(
            module.to_k,
            num_heads=int(module.num_heads),
        )
        if query is None or key is None:
            continue
        states.append(AttentionQKState(name=str(name), query=query, key=key))
    return tuple(states)


def collect_lingbot_attention_monitoring(
    transformer: Any,
    *,
    monitor: PreemptiveAttentionMonitor | None,
    training: bool,
) -> dict[str, float]:
    if monitor is None or not bool(training):
        return {}
    return monitor.observe(collect_lingbot_attention_qk_states(transformer))


__all__ = [
    "collect_lingbot_attention_monitoring",
    "collect_lingbot_attention_qk_states",
]
