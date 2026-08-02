"""Promotion gate builders for release evidence."""

from __future__ import annotations

from typing import Any

from mirai.core.training.release.promotion_gate_contract import build_promotion_gate_report


def _gate(name: str, passed: bool, details: dict[str, Any]) -> dict[str, object]:
    return {
        "name": str(name),
        "status": "ok" if passed else "failed",
        "details": dict(details),
    }


def build_release_promotion_gate(
    *,
    verification_payload: dict[str, Any],
    operational_matrix_payload: dict[str, Any],
    resume_certification_payload: dict[str, Any],
    export_certification_payload: dict[str, Any],
) -> dict[str, object]:
    gates = [
        _gate(
            "strict_release_verifier",
            str(verification_payload.get("status", "")).strip().lower() == "passed",
            {
                "failure_count": len(list(verification_payload.get("failures", []) or [])),
            },
        ),
        _gate(
            "operational_matrix",
            str(operational_matrix_payload.get("status", "")).strip().lower() == "ok",
            {
                "entry_count": int(operational_matrix_payload.get("entry_count", 0)),
                "failing_entries": list(operational_matrix_payload.get("failing_entries", []) or []),
            },
        ),
        _gate(
            "resume_equivalence",
            str(resume_certification_payload.get("status", "")).strip().lower() == "ok",
            {
                "compared_tensor_count": int(
                    resume_certification_payload.get("compared_tensor_count", 0)
                ),
                "mismatched_tensor_keys": list(
                    resume_certification_payload.get("mismatched_tensor_keys", []) or []
                ),
            },
        ),
        _gate(
            "export_certification",
            str(export_certification_payload.get("status", "")).strip().lower() == "ok",
            {
                "entry_count": int(export_certification_payload.get("entry_count", 0)),
                "failing_entries": list(
                    export_certification_payload.get("failing_entries", []) or []
                ),
            },
        ),
    ]
    failing_gates = [
        str(gate["name"])
        for gate in gates
        if str(gate.get("status", "")).strip().lower() != "ok"
    ]
    return build_promotion_gate_report(
        status="ok" if not failing_gates else "failed",
        gate_count=len(gates),
        failing_gates=failing_gates,
        gates=gates,
    )
