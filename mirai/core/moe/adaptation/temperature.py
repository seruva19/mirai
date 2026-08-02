"""Model-agnostic router temperature scheduling and optional input jitter."""

from __future__ import annotations

import math
from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ROUTER_TEMPERATURE_STATE_SCHEMA_VERSION = 1
ROUTER_TEMPERATURE_SCHEDULES = frozenset({"constant", "linear", "sigmoid"})


class RouterTemperatureController:
    """Checkpointed logit temperature with an optional entropy stop.

    The controller owns policy math only. Model providers install
    :meth:`transform_logits` at their native pre-score routing seam and feed
    detached routing entropy back through :meth:`observe_entropy`.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        minimum_temperature: float = 1.0,
        schedule: str = "constant",
        start_step: int = 0,
        end_step: int = 0,
        sigmoid_sharpness: float = 7.0,
        jitter_epsilon: float = 0.0,
        entropy_floor: float = 0.0,
    ) -> None:
        mode = str(schedule).strip().lower()
        if mode not in ROUTER_TEMPERATURE_SCHEDULES:
            raise ValueError(
                "Router temperature schedule must be constant, linear, or sigmoid."
            )
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("Router temperature must be finite and > 0.")
        if (
            not math.isfinite(float(minimum_temperature))
            or float(minimum_temperature) <= 0.0
            or float(minimum_temperature) > float(temperature)
        ):
            raise ValueError(
                "Router minimum temperature must be finite, > 0, and no "
                "greater than the initial temperature."
            )
        if int(start_step) < 0 or int(end_step) < 0:
            raise ValueError("Router temperature schedule steps must be non-negative.")
        if mode != "constant" and int(end_step) <= int(start_step):
            raise ValueError(
                "Annealed router temperature requires end_step > start_step."
            )
        if (
            not math.isfinite(float(sigmoid_sharpness))
            or float(sigmoid_sharpness) <= 0.0
        ):
            raise ValueError("Router sigmoid sharpness must be finite and > 0.")
        if (
            not math.isfinite(float(jitter_epsilon))
            or not 0.0 <= float(jitter_epsilon) < 1.0
        ):
            raise ValueError("Router jitter epsilon must be finite and in [0, 1).")
        if not math.isfinite(float(entropy_floor)) or float(entropy_floor) < 0.0:
            raise ValueError("Router entropy floor must be finite and >= 0.")
        self.temperature = float(temperature)
        self.minimum_temperature = float(minimum_temperature)
        self.schedule = mode
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.sigmoid_sharpness = float(sigmoid_sharpness)
        self.jitter_epsilon = float(jitter_epsilon)
        self.entropy_floor = float(entropy_floor)
        self._step = 0
        self._training = False
        self._frozen_temperature: float | None = None
        self._last_observed_entropy: float | None = None

    def bind_step(self, *, step: int, training: bool) -> None:
        if int(step) < 0:
            raise ValueError("Router temperature step must be non-negative.")
        self._step = int(step)
        self._training = bool(training)

    def scheduled_temperature(self, step: int | None = None) -> float:
        if self._frozen_temperature is not None:
            return float(self._frozen_temperature)
        current_step = self._step if step is None else int(step)
        if self.schedule == "constant" or current_step <= self.start_step:
            return self.temperature
        if current_step >= self.end_step:
            return self.minimum_temperature
        progress = (current_step - self.start_step) / float(
            self.end_step - self.start_step
        )
        if self.schedule == "linear":
            weight = 1.0 - progress
        else:
            weight = 1.0 / (
                1.0
                + math.exp(
                    self.sigmoid_sharpness * (progress - 0.5)
                )
            )
        return self.minimum_temperature + (
            self.temperature - self.minimum_temperature
        ) * weight

    def transform_logits(
        self,
        layer_name: str,
        logits: Any,
        *,
        training: bool,
    ) -> Any:
        _ = layer_name
        resolved_temperature = self.scheduled_temperature()
        use_jitter = bool(
            self._training
            and training
            and self.jitter_epsilon > 0.0
        )
        if resolved_temperature == 1.0 and not use_jitter:
            return logits
        transformed = (
            logits / resolved_temperature
            if resolved_temperature != 1.0
            else logits
        )
        if use_jitter:
            jitter = torch.empty_like(transformed).uniform_(
                1.0 - self.jitter_epsilon,
                1.0 + self.jitter_epsilon,
            )
            transformed = transformed * jitter
        return transformed

    def observe_entropy(self, value: float, *, training: bool) -> None:
        entropy = float(value)
        if not math.isfinite(entropy):
            raise ValueError("Observed router entropy must be finite.")
        self._last_observed_entropy = entropy
        if (
            self._frozen_temperature is None
            and self.entropy_floor > 0.0
            and self._training
            and training
            and entropy < self.entropy_floor
        ):
            self._frozen_temperature = self.scheduled_temperature()

    @property
    def annealing_frozen(self) -> bool:
        return self._frozen_temperature is not None

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTER_TEMPERATURE_STATE_SCHEMA_VERSION,
            "temperature": self.temperature,
            "minimum_temperature": self.minimum_temperature,
            "schedule": self.schedule,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "sigmoid_sharpness": self.sigmoid_sharpness,
            "jitter_epsilon": self.jitter_epsilon,
            "entropy_floor": self.entropy_floor,
            "step": self._step,
            "frozen_temperature": self._frozen_temperature,
            "last_observed_entropy": self._last_observed_entropy,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            int(state.get("schema_version", 0))
            != ROUTER_TEMPERATURE_STATE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported router-temperature state schema.")
        expected = (
            self.temperature,
            self.minimum_temperature,
            self.schedule,
            self.start_step,
            self.end_step,
            self.sigmoid_sharpness,
            self.jitter_epsilon,
            self.entropy_floor,
        )
        observed = (
            float(state.get("temperature", -1.0)),
            float(state.get("minimum_temperature", -1.0)),
            str(state.get("schedule", "")),
            int(state.get("start_step", -1)),
            int(state.get("end_step", -1)),
            float(state.get("sigmoid_sharpness", -1.0)),
            float(state.get("jitter_epsilon", -1.0)),
            float(state.get("entropy_floor", -1.0)),
        )
        if observed != expected:
            raise ValueError("Router-temperature checkpoint configuration changed.")
        frozen = state.get("frozen_temperature")
        last_entropy = state.get("last_observed_entropy")
        self._frozen_temperature = None if frozen is None else float(frozen)
        self._last_observed_entropy = (
            None if last_entropy is None else float(last_entropy)
        )
        self.bind_step(step=int(state.get("step", -1)), training=False)


__all__ = [
    "ROUTER_TEMPERATURE_SCHEDULES",
    "ROUTER_TEMPERATURE_STATE_SCHEMA_VERSION",
    "RouterTemperatureController",
]
