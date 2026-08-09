"""Session artifact/lineage context builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mirai.core.lineage import (
    SnapshotDescriptor,
    bind_snapshot_component,
    normalize_snapshot_component,
    snapshot_component_id,
    snapshot_descriptor_for_path,
)
from mirai.core.persistence.run_manifest import build_run_manifest, write_run_manifest
from mirai.core.training.lifecycle.session_output_contract import prepare_session_output_plan


@dataclass(frozen=True)
class SessionArtifactContext:
    output_root: Path
    ckpt_dir: Path
    run_id: str
    manifest: Any
    manifest_path: Path
    attempt_manifest_path: Path
    dataset_snapshot: SnapshotDescriptor
    cache_snapshot: SnapshotDescriptor
    model_snapshot: SnapshotDescriptor
    config_snapshot: SnapshotDescriptor


def build_session_artifact_context(
    *,
    config: Any,
    config_path: str | Path,
    output_dir: str | Path,
    resume_path: str = "",
) -> SessionArtifactContext:
    output_plan = prepare_session_output_plan(
        config=config,
        output_dir=output_dir,
        resume_path=resume_path,
    )
    output_root = output_plan.output_root
    dataset_snapshot = snapshot_descriptor_for_path(config.dataset.path)
    cache_snapshot = snapshot_descriptor_for_path(config.dataset.cache_path)
    model_root_snapshot = snapshot_descriptor_for_path(
        config.model.path,
        hash_contents=bool(
            getattr(config.model, "hash_snapshot_contents", False)
        ),
    )
    model_component = normalize_snapshot_component(
        getattr(config.model.params, "denoiser_subfolder", "transformer") or "transformer"
    )
    model_component_id = snapshot_component_id(
        model_component,
        component_label="denoiser_subfolder",
    )
    model_snapshot = bind_snapshot_component(
        model_root_snapshot,
        component=model_component,
        component_label="denoiser_subfolder",
    )
    config_snapshot = snapshot_descriptor_for_path(config_path)
    model_snapshot_meta = model_snapshot.to_dict()
    model_snapshot_meta.update(
        {
            "base_snapshot_id": str(model_root_snapshot.snapshot_id),
            "base_source_kind": str(model_root_snapshot.source_kind),
            "model_component_id": str(model_component_id),
            "model_component_label": "denoiser_subfolder",
            "denoiser_subfolder": str(model_component),
        }
    )
    manifest = build_run_manifest(
        resolved_config=asdict(config),
        run_id=output_plan.run_id,
        dataset_snapshot_id=dataset_snapshot.snapshot_id,
        cache_snapshot_id=cache_snapshot.snapshot_id,
        model_snapshot_id=model_snapshot.snapshot_id,
        config_snapshot_id=config_snapshot.snapshot_id,
        dataset_snapshot_meta=dataset_snapshot.to_dict(),
        cache_snapshot_meta=cache_snapshot.to_dict(),
        model_snapshot_meta=model_snapshot_meta,
        config_snapshot_meta=config_snapshot.to_dict(),
    )
    attempt_manifest_path = write_run_manifest(output_plan.attempt_manifest_path, manifest)
    manifest_path = write_run_manifest(output_plan.manifest_path, manifest)
    return SessionArtifactContext(
        output_root=output_root,
        ckpt_dir=output_plan.ckpt_dir,
        run_id=output_plan.run_id,
        manifest=manifest,
        manifest_path=manifest_path,
        attempt_manifest_path=attempt_manifest_path,
        dataset_snapshot=dataset_snapshot,
        cache_snapshot=cache_snapshot,
        model_snapshot=model_snapshot,
        config_snapshot=config_snapshot,
    )
