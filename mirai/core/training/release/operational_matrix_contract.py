"""Versioned operational certification matrix contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OPERATIONAL_MATRIX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OperationalMatrixEntry:
    category: str
    scenario: str
    status: str
    report_path: str
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "category": str(self.category),
            "scenario": str(self.scenario),
            "status": str(self.status),
            "report_path": str(self.report_path),
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class OperationalCertificationMatrixReport:
    status: str
    entry_count: int
    failing_entries: list[str]
    entries: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OPERATIONAL_MATRIX_SCHEMA_VERSION,
            "status": str(self.status),
            "entry_count": int(self.entry_count),
            "failing_entries": list(self.failing_entries),
            "entries": list(self.entries),
        }


def build_operational_matrix_entry(**kwargs: Any) -> dict[str, object]:
    return OperationalMatrixEntry(**kwargs).to_dict()


def build_operational_certification_matrix_report(**kwargs: Any) -> dict[str, object]:
    return OperationalCertificationMatrixReport(**kwargs).to_dict()


def validate_operational_certification_matrix_report(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if str(payload.get("status", "")).strip().lower() != "ok":
        failures.append("operational_matrix_report: status must be 'ok'.")
    entries = list(payload.get("entries", []) or [])
    if int(payload.get("entry_count", 0)) != len(entries):
        failures.append("operational_matrix_report: entry_count must match entries length.")
    if not entries:
        failures.append("operational_matrix_report: entries must be non-empty.")
    if list(payload.get("failing_entries", []) or []):
        failures.append("operational_matrix_report: failing_entries must be empty.")
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(f"operational_matrix_report: entry {idx} must be an object.")
            continue
        if str(entry.get("status", "")).strip().lower() != "ok":
            failures.append(f"operational_matrix_report: entry {idx} status must be 'ok'.")
        if not str(entry.get("category", "")).strip():
            failures.append(f"operational_matrix_report: entry {idx} category is required.")
        if not str(entry.get("scenario", "")).strip():
            failures.append(f"operational_matrix_report: entry {idx} scenario is required.")
        if not str(entry.get("report_path", "")).strip():
            failures.append(f"operational_matrix_report: entry {idx} report_path is required.")
        if not isinstance(entry.get("summary", {}), dict):
            failures.append(f"operational_matrix_report: entry {idx} summary must be an object.")
    return failures
