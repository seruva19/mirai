"""Sparse expert gradient-coverage metrics independent of model family."""

from __future__ import annotations

import math
from typing import Any


def moe_gradient_health(named_params: list[tuple[str, Any]]) -> dict[str, float]:
    expert_active = 0
    expert_total = 0
    expert_sq = 0.0
    router_sq = 0.0
    for name, param in named_params:
        key = str(name).lower()
        grad = getattr(param, "_mirai_cpu_grad", None)
        if grad is None:
            grad = getattr(param, "grad", None)
        if grad is None:
            continue
        grad_float = grad.detach().float()
        if "expert" in key:
            expert_sq += float(grad_float.square().sum().item())
            if grad_float.ndim >= 3 and ("lora_a" in key or "lora_b" in key):
                per_expert = grad_float.reshape(grad_float.shape[0], -1).norm(dim=1)
                expert_active += int((per_expert > 0).sum().item())
                expert_total += int(per_expert.numel())
        if "router" in key:
            router_sq += float(grad_float.square().sum().item())
    if expert_total == 0 and expert_sq == 0.0 and router_sq == 0.0:
        return {}
    active_fraction = (
        float(expert_active) / float(expert_total) if expert_total else 0.0
    )
    inactive_slices = float(max(0, expert_total - expert_active))
    return {
        "moe_expert_grad_norm": math.sqrt(expert_sq),
        "moe_router_grad_norm": math.sqrt(router_sq),
        "moe_expert_adapter_active_slice_fraction": active_fraction,
        "moe_expert_adapter_inactive_slices": inactive_slices,
        "moe_expert_adapter_total_slices": float(expert_total),
        # Stable metric aliases consumed by checkpoint and reporting contracts.
        "moe_active_expert_fraction": active_fraction,
        "moe_inactive_expert_count": inactive_slices,
    }
