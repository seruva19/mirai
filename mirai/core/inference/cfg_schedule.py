"""Per-generation classifier-free guidance schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CfgSchedule:
    interval: tuple[float, float] = (0.0, 1.0)
    uncond_reuse_stride: int = 1

    def __post_init__(self) -> None:
        if len(self.interval) != 2 or not 0.0 <= self.interval[0] <= self.interval[1] <= 1.0:
            raise ValueError("inference.cfg_interval must be [start, end] within [0, 1].")
        if type(self.uncond_reuse_stride) is not int or self.uncond_reuse_stride < 1:
            raise ValueError("inference.uncond_reuse_stride must be a positive integer.")

    @property
    def enabled(self) -> bool:
        return self.interval != (0.0, 1.0) or self.uncond_reuse_stride != 1

    def applies(self, index: int, count: int) -> bool:
        progress = index / max(count - 1, 1)
        return self.interval[0] <= progress <= self.interval[1]
