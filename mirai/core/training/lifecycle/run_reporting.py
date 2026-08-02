"""Run lifecycle reporting and finalization helpers."""

from __future__ import annotations

from typing import Any

from mirai.config.runtime_policy import runtime_policy_summary
from mirai.core.training.lifecycle.artifacts import (
    FinalRunArtifacts,
    finalize_training_artifacts,
)
from mirai.core.training.lifecycle.posthoc_ema import (
    save_final_posthoc_ema_snapshot,
)
from mirai.core.training.observability.dispatch import emit_event
from mirai.core.training.residency.memory import current_resource_telemetry
from mirai.core.training.lifecycle.run_contract import (
    build_preview_generated_payload,
    build_run_completed_payload,
    build_run_failed_payload,
    build_run_start_payload,
)


def resolve_training_sampling_mode(config: Any) -> str:
    return (
        "temporal_resampling"
        if bool(config.dataset.online_temporal_resampling)
        else "deterministic_epoch_shuffle"
    )


def emit_run_started(session: Any) -> None:
    run_state = session.run_state
    emit_event(
        session.event_bus,
        session.callbacks,
        event_type="run.started",
        step=run_state.global_step,
        payload=build_run_start_payload(
            config_path=str(session.config_path),
            max_steps=int(session.max_steps),
            gradient_accumulation=int(session.grad_accum),
            resumed=bool(session.resumed),
            resume_checkpoint_path=str(session.resume_checkpoint_path),
            resume_checkpoint_step=int(session.resume_checkpoint_step),
            resume_lineage_verified=bool(session.resume_lineage_verified),
            train_record_count=int(len(session.train_records)),
            val_record_count=int(len(session.val_records)),
            cache_data_access_mode=str(session.cache_data_access_mode),
            indexed_cache_enabled=bool(session.indexed_cache_enabled),
            training_sampling_mode=resolve_training_sampling_mode(session.config),
        ),
    )


def emit_preview_generated(
    session: Any,
    *,
    step: int,
    sample_name: str,
    sample_path: str,
) -> None:
    emit_event(
        session.event_bus,
        session.callbacks,
        event_type="preview.generated",
        step=step,
        payload=build_preview_generated_payload(
            sample_name=str(sample_name),
            path=str(sample_path),
        ),
    )


def finalize_run_artifacts(session: Any) -> FinalRunArtifacts:
    run_state = session.run_state
    save_final_posthoc_ema_snapshot(session)
    model_snapshot_meta = getattr(session.manifest, "model_snapshot_meta", {})
    if not isinstance(model_snapshot_meta, dict):
        model_snapshot_meta = {}
    return finalize_training_artifacts(
        config=session.config,
        run_id=session.run_id,
        output_dir=session.output_dir,
        ckpt_dir=session.ckpt_dir,
        trainer=session.trainer,
        manifest_path=session.manifest_path,
        manifest_sha256=session.manifest.manifest_sha256,
        build_ckpt_payload=session.build_ckpt_payload,
        global_step=run_state.global_step,
        stopped_early=run_state.stopped_early,
        skipped_steps=run_state.skipped_steps,
        optimizer_type=session.optimizer_result.resolved_type,
        optimizer_fallback_used=session.optimizer_result.used_fallback,
        lr=float(session.optimizer.param_groups[0]["lr"]),
        last_metrics=run_state.last_metrics,
        ema_state=run_state.ema_state,
        gradient_offload_ops=run_state.gradient_offload_ops,
        optimizer_offload_ops=run_state.optimizer_offload_ops,
        compile_enabled=bool(getattr(session.trainer, "compile_enabled", False)),
        compile_warning=str(getattr(session.trainer, "compile_warning", "")),
        runtime_policy=runtime_policy_summary(session.config),
        runtime_policy_notes=list(session.runtime_policy_notes),
        log_on_this_rank=session.log_on_this_rank,
        dataset_snapshot_id=str(session.manifest.dataset_snapshot_id),
        cache_snapshot_id=str(session.manifest.cache_snapshot_id),
        model_snapshot_id=str(getattr(session.manifest, "model_snapshot_id", "")),
        config_snapshot_id=str(getattr(session.manifest, "config_snapshot_id", "")),
        model_component_id=str(model_snapshot_meta.get("model_component_id", "")),
        cache_data_access_mode=str(session.cache_data_access_mode),
        indexed_cache_enabled=bool(session.indexed_cache_enabled),
        indexed_cache_metadata_path=str(session.indexed_cache_metadata_path),
        indexed_cache_tensor_path=str(session.indexed_cache_tensor_path),
        training_sampling_mode=resolve_training_sampling_mode(session.config),
        resumed=bool(session.resumed),
        resume_checkpoint_path=str(session.resume_checkpoint_path),
        resume_checkpoint_step=int(session.resume_checkpoint_step),
        resume_lineage_verified=bool(session.resume_lineage_verified),
        resume_validation_warnings=list(session.resume_validation_warnings),
        resource_telemetry=current_resource_telemetry(),
    )


def emit_run_completed(session: Any, *, final_artifacts: FinalRunArtifacts) -> None:
    run_state = session.run_state
    if session.log_on_this_rank:
        emit_event(
            session.event_bus,
            session.callbacks,
            event_type="checkpoint.saved",
            step=run_state.global_step,
            payload={
                "path": str(final_artifacts.final_checkpoint_path),
                "reason": "final",
            },
        )
    emit_event(
        session.event_bus,
        session.callbacks,
        event_type="run.completed",
        step=run_state.global_step,
        payload=build_run_completed_payload(
            status=str(final_artifacts.summary["status"]),
            global_step=int(run_state.global_step),
            last_checkpoint=str(final_artifacts.final_checkpoint_path),
        ),
    )


def emit_run_failed(session: Any, *, exc: BaseException) -> None:
    emit_event(
        session.event_bus,
        session.callbacks,
        event_type="run.failed",
        step=session.run_state.global_step,
        payload=build_run_failed_payload(
            error=str(exc),
            error_type=type(exc).__name__,
            global_step=int(session.run_state.global_step),
        ),
    )
