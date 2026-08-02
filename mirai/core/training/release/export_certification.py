"""Builders for export/import certification evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mirai.core.training.adapters import load_adapter_payload, normalize_adapter_state
from mirai.core.persistence.checkpoints import load_checkpoint
from mirai.core.training.release.export_certification_contract import (
    build_export_certification_entry,
    build_export_certification_report,
)
from mirai.core.training.release.release_evidence_contract import load_release_json


def _entry(
    *,
    category: str,
    status: str,
    artifact_path: str | Path,
    details: dict[str, Any],
) -> dict[str, object]:
    return build_export_certification_entry(
        category=category,
        status=status,
        artifact_path=str(Path(artifact_path).resolve()),
        details=dict(details),
    )


def _normalized_state_ok(path: Path, *, lora_format: str) -> tuple[bool, dict[str, Any]]:
    payload = load_adapter_payload(path)
    normalized = normalize_adapter_state(payload, lora_format=lora_format)
    keys = sorted(str(key) for key in normalized.keys())
    return True, {"normalized_keys": keys, "tensor_key_count": len(keys)}


def build_export_certification_from_artifacts(
    *,
    adapter_checkpoint_path: str | Path,
    kohya_adapter_path: str | Path,
    diffusers_export_path: str | Path,
    merged_export_path: str | Path,
    host_compatibility_report_path: str | Path,
    lycoris_export_path: str | Path | None = None,
    comfyui_report_path: str | Path | None = None,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    adapter_checkpoint = Path(adapter_checkpoint_path).resolve()
    adapter_payload = load_checkpoint(adapter_checkpoint)
    adapter_state = adapter_payload.get("adapter_state", {})
    checkpoint_ok = isinstance(adapter_state, dict) and bool(adapter_state)
    entries.append(
        _entry(
            category="adapter_checkpoint_reload",
            status="ok" if checkpoint_ok else "failed",
            artifact_path=adapter_checkpoint,
            details={
                "state_key_count": len(list(adapter_state.keys())) if isinstance(adapter_state, dict) else 0,
            },
        )
    )

    kohya_path = Path(kohya_adapter_path).resolve()
    _, kohya_details = _normalized_state_ok(kohya_path, lora_format="kohya")
    entries.append(
        _entry(
            category="kohya_reload",
            status="ok",
            artifact_path=kohya_path,
            details=kohya_details,
        )
    )

    diffusers_path = Path(diffusers_export_path).resolve()
    _, diffusers_details = _normalized_state_ok(diffusers_path, lora_format="diffusers")
    entries.append(
        _entry(
            category="diffusers_export_reload",
            status="ok",
            artifact_path=diffusers_path,
            details=diffusers_details,
        )
    )

    merged_path = Path(merged_export_path).resolve()
    merged_payload = load_checkpoint(merged_path)
    merged_state = dict(merged_payload.get("merged_base_state", {}) or {})
    merged_ok = "base_scale" in merged_state
    entries.append(
        _entry(
            category="merged_export_reload",
            status="ok" if merged_ok else "failed",
            artifact_path=merged_path,
            details={
                "merged_keys": sorted(str(key) for key in merged_state.keys()),
            },
        )
    )

    host_report_path = Path(host_compatibility_report_path).resolve()
    host_report = load_release_json(host_report_path)
    host_status = str(host_report.get("status", "")).strip().lower()
    entries.append(
        _entry(
            category="host_contract_report",
            status="ok" if host_status == "ok" else "failed",
            artifact_path=host_report_path,
            details={
                "host_count": len(list(host_report.get("hosts", []) or [])),
                "status": host_status,
            },
        )
    )

    if lycoris_export_path is not None and str(lycoris_export_path).strip():
        lycoris_path = Path(lycoris_export_path).resolve()
        _, lycoris_details = _normalized_state_ok(lycoris_path, lora_format="lycoris")
        entries.append(
            _entry(
                category="lycoris_export_reload",
                status="ok",
                artifact_path=lycoris_path,
                details=lycoris_details,
            )
        )

    if comfyui_report_path is not None and str(comfyui_report_path).strip():
        comfy_path = Path(comfyui_report_path).resolve()
        comfy_report = load_release_json(comfy_path)
        comfy_status = str(comfy_report.get("status", "")).strip().lower()
        entry_status = "ok" if comfy_status == "ok" else "failed"
        entries.append(
            _entry(
                category="comfyui_external_load",
                status=entry_status,
                artifact_path=comfy_path,
                details={
                    "comfy_status": str(comfy_report.get("comfy_status", "")),
                    "produced_files": len(list(comfy_report.get("produced_files", []) or [])),
                },
            )
        )

    failing_entries = [
        str(entry["category"])
        for entry in entries
        if str(entry.get("status", "")).strip().lower() == "failed"
    ]
    return build_export_certification_report(
        status="ok" if not failing_entries else "failed",
        entry_count=len(entries),
        failing_entries=failing_entries,
        entries=entries,
    )
