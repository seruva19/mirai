"""Learning-rate scheduler helpers."""

from __future__ import annotations

import math
from typing import Callable

from mirai.core.registry import Registry

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for scheduler setup: {exc}")


# Builder signature: (progress, num_cycles, power, lo) -> post-warmup decay
# multiplier in [lo, 1.0], where ``progress`` is the fraction of decay steps
# elapsed. Builtins register at import (below); third parties add a mode via
# ``@register_scheduler("name")``.
SchedulerDecay = Callable[[float, float, float, float], float]
SchedulerRegistry: Registry[SchedulerDecay] = Registry("scheduler")


def register_scheduler(name: str) -> Callable[[SchedulerDecay], SchedulerDecay]:
    return SchedulerRegistry.decorator(name)


@register_scheduler("constant")
def _decay_constant(progress: float, num_cycles: float, power: float, lo: float) -> float:
    return 1.0


@register_scheduler("linear")
def _decay_linear(progress: float, num_cycles: float, power: float, lo: float) -> float:
    return lo + (1.0 - lo) * (1.0 - progress)


@register_scheduler("cosine")
def _decay_cosine(progress: float, num_cycles: float, power: float, lo: float) -> float:
    # half-cosine: 1.0 → lo over [warmup, total_steps]
    cycles = float(num_cycles)
    cosine_val = math.cos(math.pi * progress * cycles)
    return lo + (1.0 - lo) * 0.5 * (1.0 + cosine_val)


@register_scheduler("cosine_with_restarts")
def _decay_cosine_with_restarts(
    progress: float, num_cycles: float, power: float, lo: float
) -> float:
    cycles = float(num_cycles)
    cycle_progress = math.fmod(progress * cycles, 1.0)
    cosine_val = math.cos(math.pi * cycle_progress)
    return lo + (1.0 - lo) * 0.5 * (1.0 + cosine_val)


@register_scheduler("polynomial")
def _decay_polynomial(progress: float, num_cycles: float, power: float, lo: float) -> float:
    # Decay from 1.0 to lo with exponent *power*, measured over [0, 1].
    pwr = float(power)
    return lo + (1.0 - lo) * ((1.0 - progress) ** pwr)


@register_scheduler("rex")
def _decay_rex(progress: float, num_cycles: float, power: float, lo: float) -> float:
    # Rex (Revisiting LR schedules): T-rex variant — reciprocal-exponential.
    # scale = 1 / (1 + power * progress / (1 - progress + eps)) clipped to [lo, 1.0]
    pwr = max(float(power), 0.0)
    if progress >= 1.0:
        return lo
    denom = 1.0 + pwr * progress / max(1.0 - progress, 1e-8)
    return lo + (1.0 - lo) / max(denom, 1e-8)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int = 0,
    scheduler: str = "constant",
    num_cycles: float = 1.0,
    power: float = 1.0,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a LambdaLR implementing *scheduler* composed with linear warmup.

    ``scheduler="constant"`` uses a step+1 warmup ramp and remains flat afterward.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps (0 = no warmup).
        total_steps: Total training steps.  Required for all modes except "constant".
        scheduler: One of constant | linear | cosine | cosine_with_restarts | polynomial | rex.
        num_cycles: Number of cosine cycles (cosine_with_restarts) or half-cycles (cosine).
        power: Polynomial decay exponent (polynomial) or rex temperature (rex).
        min_lr_ratio: Floor as a fraction of base LR (default 0 → decay to zero).
    """
    warmup = max(0, int(warmup_steps))
    mode = scheduler.strip().lower()
    decay = SchedulerRegistry.get(mode)
    decay_steps = max(1, int(total_steps) - warmup)
    lo = float(min_lr_ratio)  # floor multiplier (0.0 → full decay)

    def _warmup_factor(step: int) -> float:
        if warmup <= 0:
            return 1.0
        # LambdaLR is called with current step index; step+1 reaches 1.0 at warmup-1.
        return min(float(step + 1) / float(warmup), 1.0)

    def _decay_factor(step: int) -> float:
        """Return post-warmup decay multiplier in [min_lr_ratio, 1.0]."""
        progress = max(0.0, float(step - warmup) / float(decay_steps))
        progress = min(progress, 1.0)
        return decay(progress, num_cycles, power, lo)

    def lr_lambda(step: int) -> float:
        wf = _warmup_factor(step)
        if wf < 1.0:
            # Still warming up — do not apply decay yet.
            return wf
        return _decay_factor(step)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
