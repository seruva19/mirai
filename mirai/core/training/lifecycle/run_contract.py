"""Typed payload contracts for run lifecycle reporting and final summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RUN_CONTRACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunStartPayload:
    config_path: str
    max_steps: int
    gradient_accumulation: int
    resumed: bool
    resume_checkpoint_path: str
    resume_checkpoint_step: int
    resume_lineage_verified: bool
    train_record_count: int
    val_record_count: int
    cache_data_access_mode: str
    indexed_cache_enabled: bool
    training_sampling_mode: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
            "config_path": str(self.config_path),
            "max_steps": int(self.max_steps),
            "gradient_accumulation": int(self.gradient_accumulation),
            "resumed": bool(self.resumed),
            "resume_checkpoint_path": str(self.resume_checkpoint_path),
            "resume_checkpoint_step": int(self.resume_checkpoint_step),
            "resume_lineage_verified": bool(self.resume_lineage_verified),
            "train_record_count": int(self.train_record_count),
            "val_record_count": int(self.val_record_count),
            "cache_data_access_mode": str(self.cache_data_access_mode),
            "indexed_cache_enabled": bool(self.indexed_cache_enabled),
            "training_sampling_mode": str(self.training_sampling_mode),
        }


@dataclass(frozen=True)
class PreviewGeneratedPayload:
    sample_name: str
    path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
            "sample_name": str(self.sample_name),
            "path": str(self.path),
        }


@dataclass(frozen=True)
class RunCompletedPayload:
    status: str
    global_step: int
    last_checkpoint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
            "status": str(self.status),
            "global_step": int(self.global_step),
            "last_checkpoint": str(self.last_checkpoint),
        }


@dataclass(frozen=True)
class RunFailedPayload:
    error: str
    error_type: str
    global_step: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
            "error": str(self.error),
            "error_type": str(self.error_type),
            "global_step": int(self.global_step),
        }


@dataclass(frozen=True)
class FinalRunSummary:
    status: str
    run_id: str
    manifest_path: str
    manifest_sha256: str
    dataset_snapshot_id: str
    cache_snapshot_id: str
    model_snapshot_id: str
    config_snapshot_id: str
    model_component_id: str
    optimizer_type: str
    optimizer_fallback_used: bool
    lr: float
    global_step: int
    skipped_steps: int
    last_checkpoint: str
    adapter_path: str
    adapter_safetensors_path: str
    adapter_type: str
    adapter_lycoris_path: str
    last_metrics: dict[str, Any]
    ema_enabled: bool
    gradient_cpu_offload: bool
    optimizer_cpu_offload: bool
    gradient_offload_ops: int
    optimizer_offload_ops: int
    compile_enabled: bool
    compile_warning: str
    cache_data_access_mode: str
    indexed_cache_enabled: bool
    indexed_cache_metadata_path: str
    indexed_cache_tensor_path: str
    training_sampling_mode: str
    resumed: bool
    resume_checkpoint_path: str
    resume_checkpoint_step: int
    resume_lineage_verified: bool
    resume_validation_warnings: list[str]
    resource_telemetry: dict[str, Any]
    runtime_policy: dict[str, Any]
    runtime_policy_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
            "status": str(self.status),
            "run_id": str(self.run_id),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": str(self.manifest_sha256),
            "dataset_snapshot_id": str(self.dataset_snapshot_id),
            "cache_snapshot_id": str(self.cache_snapshot_id),
            "model_snapshot_id": str(self.model_snapshot_id),
            "config_snapshot_id": str(self.config_snapshot_id),
            "model_component_id": str(self.model_component_id),
            "optimizer_type": str(self.optimizer_type),
            "optimizer_fallback_used": bool(self.optimizer_fallback_used),
            "lr": float(self.lr),
            "global_step": int(self.global_step),
            "skipped_steps": int(self.skipped_steps),
            "last_checkpoint": str(self.last_checkpoint),
            "adapter_path": str(self.adapter_path),
            "adapter_safetensors_path": str(self.adapter_safetensors_path),
            "adapter_type": str(self.adapter_type),
            "adapter_lycoris_path": str(self.adapter_lycoris_path),
            "last_metrics": dict(self.last_metrics),
            "ema_enabled": bool(self.ema_enabled),
            "gradient_cpu_offload": bool(self.gradient_cpu_offload),
            "optimizer_cpu_offload": bool(self.optimizer_cpu_offload),
            "gradient_offload_ops": int(self.gradient_offload_ops),
            "optimizer_offload_ops": int(self.optimizer_offload_ops),
            "compile_enabled": bool(self.compile_enabled),
            "compile_warning": str(self.compile_warning),
            "cache_data_access_mode": str(self.cache_data_access_mode),
            "indexed_cache_enabled": bool(self.indexed_cache_enabled),
            "indexed_cache_metadata_path": str(self.indexed_cache_metadata_path),
            "indexed_cache_tensor_path": str(self.indexed_cache_tensor_path),
            "training_sampling_mode": str(self.training_sampling_mode),
            "resumed": bool(self.resumed),
            "resume_checkpoint_path": str(self.resume_checkpoint_path),
            "resume_checkpoint_step": int(self.resume_checkpoint_step),
            "resume_lineage_verified": bool(self.resume_lineage_verified),
            "resume_validation_warnings": list(self.resume_validation_warnings),
            "resource_telemetry": dict(self.resource_telemetry),
            "runtime_policy": dict(self.runtime_policy),
            "runtime_policy_notes": list(self.runtime_policy_notes),
        }


def build_run_start_payload(**kwargs: Any) -> dict[str, object]:
    return RunStartPayload(**kwargs).to_dict()


def build_preview_generated_payload(**kwargs: Any) -> dict[str, object]:
    return PreviewGeneratedPayload(**kwargs).to_dict()


def build_run_completed_payload(**kwargs: Any) -> dict[str, object]:
    return RunCompletedPayload(**kwargs).to_dict()


def build_run_failed_payload(**kwargs: Any) -> dict[str, object]:
    return RunFailedPayload(**kwargs).to_dict()


def build_final_run_summary(**kwargs: Any) -> dict[str, Any]:
    return FinalRunSummary(**kwargs).to_dict()
