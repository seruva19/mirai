"""Versioned promotion-gate contract for release evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PROMOTION_GATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PromotionGateReport:
    status: str
    gate_count: int
    failing_gates: list[str]
    gates: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PROMOTION_GATE_SCHEMA_VERSION,
            "status": str(self.status),
            "gate_count": int(self.gate_count),
            "failing_gates": list(self.failing_gates),
            "gates": list(self.gates),
        }


def build_promotion_gate_report(**kwargs: Any) -> dict[str, object]:
    return PromotionGateReport(**kwargs).to_dict()


def validate_promotion_gate_report(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    gates = list(payload.get("gates", []) or [])
    if int(payload.get("gate_count", 0)) != len(gates):
        failures.append("promotion_gate_report: gate_count must match gates length.")
    if not gates:
        failures.append("promotion_gate_report: gates must be non-empty.")
    failing_gates = list(payload.get("failing_gates", []) or [])
    if str(payload.get("status", "")).strip().lower() == "ok":
        if failing_gates:
            failures.append("promotion_gate_report: failing_gates must be empty when status is ok.")
    for idx, gate in enumerate(gates):
        if not isinstance(gate, dict):
            failures.append(f"promotion_gate_report: gate {idx} must be an object.")
            continue
        if not str(gate.get("name", "")).strip():
            failures.append(f"promotion_gate_report: gate {idx} name is required.")
        if str(gate.get("status", "")).strip().lower() not in {"ok", "failed"}:
            failures.append(f"promotion_gate_report: gate {idx} status must be ok or failed.")
    return failures
