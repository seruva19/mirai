"""Versioned operational-certification contracts for release evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OPERATIONAL_CERTIFICATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResumeEquivalenceCertificationReport:
    status: str
    train_summary: str
    resume_summary: str
    full_checkpoint: str
    resumed_checkpoint: str
    global_step_match: bool
    compared_tensor_count: int
    missing_tensor_keys: list[str]
    extra_tensor_keys: list[str]
    mismatched_tensor_keys: list[str]
    max_abs_delta: float
    tolerated_atol: float
    tolerated_rtol: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPERATIONAL_CERTIFICATION_SCHEMA_VERSION,
            "status": str(self.status),
            "train_summary": str(self.train_summary),
            "resume_summary": str(self.resume_summary),
            "full_checkpoint": str(self.full_checkpoint),
            "resumed_checkpoint": str(self.resumed_checkpoint),
            "global_step_match": bool(self.global_step_match),
            "compared_tensor_count": int(self.compared_tensor_count),
            "missing_tensor_keys": list(self.missing_tensor_keys),
            "extra_tensor_keys": list(self.extra_tensor_keys),
            "mismatched_tensor_keys": list(self.mismatched_tensor_keys),
            "max_abs_delta": float(self.max_abs_delta),
            "tolerated_atol": float(self.tolerated_atol),
            "tolerated_rtol": float(self.tolerated_rtol),
        }


def build_resume_equivalence_certification_report(**kwargs: Any) -> dict[str, object]:
    return ResumeEquivalenceCertificationReport(**kwargs).to_dict()


def validate_resume_equivalence_certification_report(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if str(payload.get("status", "")).strip().lower() != "ok":
        failures.append("resume_certification_report: status must be 'ok'.")
    if not bool(payload.get("global_step_match", False)):
        failures.append("resume_certification_report: global_step_match must be true.")
    if int(payload.get("compared_tensor_count", 0)) <= 0:
        failures.append("resume_certification_report: compared_tensor_count must be > 0.")
    if list(payload.get("missing_tensor_keys", []) or []):
        failures.append("resume_certification_report: missing_tensor_keys must be empty.")
    if list(payload.get("extra_tensor_keys", []) or []):
        failures.append("resume_certification_report: extra_tensor_keys must be empty.")
    if list(payload.get("mismatched_tensor_keys", []) or []):
        failures.append("resume_certification_report: mismatched_tensor_keys must be empty.")
    return failures
