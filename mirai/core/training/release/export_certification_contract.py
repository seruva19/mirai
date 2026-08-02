"""Versioned export/import certification contract for release evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXPORT_CERTIFICATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExportCertificationEntry:
    category: str
    status: str
    artifact_path: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "category": str(self.category),
            "status": str(self.status),
            "artifact_path": str(self.artifact_path),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ExportCertificationReport:
    status: str
    entry_count: int
    failing_entries: list[str]
    entries: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": EXPORT_CERTIFICATION_SCHEMA_VERSION,
            "status": str(self.status),
            "entry_count": int(self.entry_count),
            "failing_entries": list(self.failing_entries),
            "entries": list(self.entries),
        }


def build_export_certification_entry(**kwargs: Any) -> dict[str, object]:
    return ExportCertificationEntry(**kwargs).to_dict()


def build_export_certification_report(**kwargs: Any) -> dict[str, object]:
    return ExportCertificationReport(**kwargs).to_dict()


def validate_export_certification_report(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    entries = list(payload.get("entries", []) or [])
    if int(payload.get("entry_count", 0)) != len(entries):
        failures.append("export_certification_report: entry_count must match entries length.")
    if not entries:
        failures.append("export_certification_report: entries must be non-empty.")
    failing_entries = list(payload.get("failing_entries", []) or [])
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"ok", "failed"}:
        failures.append("export_certification_report: status must be ok or failed.")
    if status == "ok" and failing_entries:
        failures.append("export_certification_report: failing_entries must be empty when status is ok.")
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(f"export_certification_report: entry {idx} must be an object.")
            continue
        if not str(entry.get("category", "")).strip():
            failures.append(f"export_certification_report: entry {idx} category is required.")
        if str(entry.get("status", "")).strip().lower() not in {"ok", "failed", "skipped"}:
            failures.append(
                f"export_certification_report: entry {idx} status must be ok, failed, or skipped."
            )
        if not str(entry.get("artifact_path", "")).strip():
            failures.append(f"export_certification_report: entry {idx} artifact_path is required.")
        if not isinstance(entry.get("details", {}), dict):
            failures.append(f"export_certification_report: entry {idx} details must be an object.")
    return failures
