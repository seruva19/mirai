"""Run manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from mirai.core.lineage import snapshot_descriptor_for_path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot_id_for_path(
    path: str | Path,
    *,
    manifest_candidates: tuple[str, ...] = ("registration.json", "download_manifest.json"),
) -> str:
    return snapshot_descriptor_for_path(
        path,
        manifest_candidates=manifest_candidates,
    ).snapshot_id


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    created_at_utc: str
    dataset_snapshot_id: str
    cache_snapshot_id: str
    model_snapshot_id: str
    config_snapshot_id: str
    dataset_snapshot_meta: dict[str, str]
    cache_snapshot_meta: dict[str, str]
    model_snapshot_meta: dict[str, str]
    config_snapshot_meta: dict[str, str]
    resolved_config: dict[str, Any]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "cache_snapshot_id": self.cache_snapshot_id,
            "model_snapshot_id": self.model_snapshot_id,
            "config_snapshot_id": self.config_snapshot_id,
            "dataset_snapshot_meta": dict(self.dataset_snapshot_meta),
            "cache_snapshot_meta": dict(self.cache_snapshot_meta),
            "model_snapshot_meta": dict(self.model_snapshot_meta),
            "config_snapshot_meta": dict(self.config_snapshot_meta),
            "resolved_config": self.resolved_config,
            "manifest_sha256": self.manifest_sha256,
        }


def build_run_manifest(
    *,
    resolved_config: dict[str, Any],
    run_id: str,
    dataset_snapshot_id: str,
    cache_snapshot_id: str,
    model_snapshot_id: str = "",
    config_snapshot_id: str = "",
    dataset_snapshot_meta: dict[str, str] | None = None,
    cache_snapshot_meta: dict[str, str] | None = None,
    model_snapshot_meta: dict[str, str] | None = None,
    config_snapshot_meta: dict[str, str] | None = None,
    created_at_utc: str | None = None,
    schema_version: int = 2,
) -> RunManifest:
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()
    core = {
        "schema_version": schema_version,
        "run_id": run_id,
        "created_at_utc": timestamp,
        "dataset_snapshot_id": dataset_snapshot_id,
        "cache_snapshot_id": cache_snapshot_id,
        "model_snapshot_id": model_snapshot_id,
        "config_snapshot_id": config_snapshot_id,
        "dataset_snapshot_meta": dict(dataset_snapshot_meta or {}),
        "cache_snapshot_meta": dict(cache_snapshot_meta or {}),
        "model_snapshot_meta": dict(model_snapshot_meta or {}),
        "config_snapshot_meta": dict(config_snapshot_meta or {}),
        "resolved_config": resolved_config,
    }
    digest = _sha256_text(_canonical_json(core))
    return RunManifest(manifest_sha256=digest, **core)


def write_run_manifest(path: str | Path, manifest: RunManifest) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
