"""Metadata migration helpers for persisted artifacts."""

from __future__ import annotations

from typing import Any


CACHE_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
DB_SCHEMA_VERSION = 1


def migrate_cache_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Cache payload must be a mapping.")

    out = dict(payload)
    raw_records = out.get("records", [])
    records = raw_records if isinstance(raw_records, list) else []
    migrated_records: list[dict[str, Any]] = []
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        rec = dict(item)
        rec.setdefault("sample_id", f"idx_{idx}")
        rec.setdefault("caption_variant_index", 0)
        rec.setdefault("caption", "")
        rec.setdefault("base_sample_id", rec["sample_id"])
        rec.setdefault("clip_index", 0)
        rec.setdefault("text_mask", [1])
        rec.setdefault("loss_mask", None)
        migrated_records.append(rec)

    out["records"] = migrated_records
    out["num_records"] = int(out.get("num_records", len(migrated_records)))
    out["num_skipped"] = int(out.get("num_skipped", 0))
    out.setdefault("estimated_disk_bytes", 0)
    out.setdefault("skipped", [])
    out.setdefault("fp8_text_encoder", False)
    out.setdefault("cache_mode", "disk")
    out.setdefault("cache_compression", "none")
    out.setdefault("tag_shuffle_variants", 1)
    out.setdefault("clips_per_video", 1)
    out.setdefault("partial_recovery", True)
    out.setdefault("recovered_records", 0)
    out.setdefault("dataset_snapshot_id", "")
    out.setdefault("dataset_snapshot_source_path", "")
    out.setdefault("dataset_snapshot_source_kind", "")
    out.setdefault("dataset_snapshot_source_manifest", "")
    out["version"] = CACHE_SCHEMA_VERSION
    return out


def migrate_checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping.")

    out = dict(payload)
    meta_raw = out.get("metadata", {})
    meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    meta.setdefault("schema_version", CHECKPOINT_SCHEMA_VERSION)
    if "run_id" not in meta and "run_id" in out:
        meta["run_id"] = str(out.get("run_id", ""))
    if "manifest_sha256" not in meta and "manifest_sha256" in out:
        meta["manifest_sha256"] = str(out.get("manifest_sha256", ""))
    for key in (
        "dataset_snapshot_id",
        "cache_snapshot_id",
        "model_snapshot_id",
        "config_snapshot_id",
    ):
        if key not in meta and key in out:
            meta[key] = str(out.get(key, ""))
    out["metadata"] = meta
    return out


def migrate_job_row(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Job row payload must be a mapping.")

    out = dict(payload)
    out["schema_version"] = int(out.get("schema_version", DB_SCHEMA_VERSION))
    out["job_id"] = str(out.get("job_id", ""))
    out["status"] = str(out.get("status", "queued"))
    out["created_at"] = str(out.get("created_at", ""))
    out["updated_at"] = str(out.get("updated_at", out["created_at"]))
    return out
