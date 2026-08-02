"""Global-step schedule for relaxing auxiliary MoE balance pressure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuxiliaryBalanceLossSchedule:
    """Disable only the auxiliary load-balance loss at one exact step.

    The schedule is derived solely from ``global_step``. It therefore needs no
    mutable checkpoint state and resumes identically from the same step.
    """

    disable_step: int

    def __post_init__(self) -> None:
        if int(self.disable_step) <= 0:
            raise ValueError(
                "Auxiliary balance-loss scheduling requires disable_step > 0."
            )

    @classmethod
    def from_model_params(
        cls,
        params: Any,
    ) -> AuxiliaryBalanceLossSchedule | None:
        disable_step = int(getattr(params, "moe_balance_loss_disable_step", 0))
        return None if disable_step == 0 else cls(disable_step=disable_step)

    def is_exploration(self, *, step: int) -> bool:
        current_step = int(step)
        if current_step < 0:
            raise ValueError("Auxiliary balance-loss schedule step must be >= 0.")
        return current_step < int(self.disable_step)

    def weight(self, base_weight: float, *, step: int) -> float:
        return float(base_weight) if self.is_exploration(step=step) else 0.0
