"""Step schedules for frozen-reference router distillation."""

from __future__ import annotations


ROUTER_DISTILLATION_WEIGHT_SCHEDULES = frozenset({"constant", "linear_decay"})


def router_distillation_weight_scale(
    step: int, *, start_step: int, end_step: int, schedule: str
) -> float:
    normalized = str(schedule).strip().lower()
    if normalized not in ROUTER_DISTILLATION_WEIGHT_SCHEDULES:
        raise ValueError(f"Unsupported router distillation weight schedule '{schedule}'.")
    if normalized == "constant":
        return 1.0
    if int(end_step) <= int(start_step):
        raise ValueError("linear_decay requires end_step > start_step.")
    progress = (int(step) - int(start_step)) / (int(end_step) - int(start_step))
    return max(0.0, min(1.0, 1.0 - progress))


__all__ = [
    "ROUTER_DISTILLATION_WEIGHT_SCHEDULES",
    "router_distillation_weight_scale",
]
