"""Post-routing guardrail for saturated MoE expert working sets."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping


EXPERT_TOUCH_GUARD_MODES = ("off", "warn", "error")


@dataclass(frozen=True)
class ExpertTouchGuardResult:
    observed_fraction: float | None
    threshold: float
    exceeded: bool


def normalize_expert_touch_guard_mode(value: str | None) -> str:
    normalized = str(value or "off").strip().lower()
    normalized = {"": "off", "none": "off", "disabled": "off"}.get(
        normalized, normalized
    )
    if normalized not in EXPERT_TOUCH_GUARD_MODES:
        raise ValueError(
            "training.moe_expert_touch_guard must be one of: "
            + ", ".join(EXPERT_TOUCH_GUARD_MODES)
            + "."
        )
    return normalized


def max_active_expert_fraction(
    diagnostics: Mapping[str, Any] | None,
) -> float | None:
    routing = (
        diagnostics.get("moe_routing")
        if isinstance(diagnostics, Mapping)
        else None
    )
    details = (
        routing.get("layers_detail")
        if isinstance(routing, Mapping)
        else None
    )
    if not isinstance(details, (list, tuple)):
        return None
    fractions: list[float] = []
    for detail in details:
        if not isinstance(detail, Mapping):
            continue
        value = detail.get("active_expert_fraction")
        if value is None:
            continue
        fraction = float(value)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "MoE active_expert_fraction diagnostics must be in [0, 1]."
            )
        fractions.append(fraction)
    return max(fractions) if fractions else None


def enforce_expert_touch_guard(
    diagnostics: Mapping[str, Any] | None,
    *,
    mode: str,
    max_fraction: float,
) -> ExpertTouchGuardResult:
    resolved_mode = normalize_expert_touch_guard_mode(mode)
    threshold = float(max_fraction)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            "training.moe_expert_touch_max_fraction must be in (0, 1]."
        )
    if resolved_mode == "off":
        return ExpertTouchGuardResult(None, threshold, False)
    observed = max_active_expert_fraction(diagnostics)
    exceeded = observed is not None and observed > threshold
    result = ExpertTouchGuardResult(observed, threshold, exceeded)
    if not exceeded:
        return result
    message = (
        "MoE expert working set exceeded its configured limit before backward: "
        f"observed_fraction={observed:.6f}, max_fraction={threshold:.6f}."
    )
    if resolved_mode == "error":
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return result


__all__ = [
    "EXPERT_TOUCH_GUARD_MODES",
    "ExpertTouchGuardResult",
    "enforce_expert_touch_guard",
    "max_active_expert_fraction",
    "normalize_expert_touch_guard_mode",
]
