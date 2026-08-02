"""Training-session assembly for the train CLI."""

from __future__ import annotations

from dataclasses import dataclass
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from mirai.core.training.adapters import (
    load_adapter_payload,
    normalize_adapter_state,
)
from mirai.core.training.calibration.gora import maybe_initialize_gora
from mirai.core.training.calibration.esft import maybe_initialize_esft
from mirai.core.training.data.curriculum import CurriculumSchedule
from mirai.core.training.lifecycle.session_components import build_training_runtime_components
from mirai.core.training.observability.events import TrainingEventBus
from mirai.core.training.lifecycle.session_callbacks import build_session_callback_context
from mirai.core.training.lifecycle.resume_validation import ResumeValidationResult
from mirai.core.training.lifecycle.session_resume import apply_resume_checkpoint
from mirai.core.training.lifecycle.session_context import build_session_artifact_context
from mirai.core.training.lifecycle.session_state import (
    TrainingRunState,
    build_checkpoint_payload,
    init_training_run_state,
)
from mirai.core.training.control.live_control import LiveTrainingController
from mirai.core.training.control.live_control_actions import (
    TrainingRuntimeOverrides,
    build_training_runtime_overrides,
)
from mirai.core.training.data.schema import resolve_training_batch_schema
from mirai.core.training.residency.device_placement import (
    place_pipeline_on_device,
    resolve_compute_device,
    resolve_compute_dtype,
)
from mirai.core.training.data.loader import load_prepared_training_data
from mirai.core.training.data.preprocessing import ensure_cache_artifacts_for_training
from mirai.core.training.trainer import Trainer


@dataclass
class TrainingSession:
    config: Any
    config_path: str
    runtime_policy_notes: list[str]
    trainer: Any
    compute_device: Any
    compute_dtype: Any
    run_id: str
    manifest: Any
    manifest_path: Path
    output_dir: Path
    ckpt_dir: Path
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]
    temporal_base_ids: list[str]
    temporal_groups: dict[str, list[dict[str, Any]]]
    cache_data_access_mode: str
    indexed_cache_enabled: bool
    indexed_cache_metadata_path: str
    indexed_cache_tensor_path: str
    curriculum: Any
    params: list[Any]
    named_params: list[tuple[str, Any]]
    optimizer: Any
    optimizer_result: Any
    scheduler: Any
    rng: random.Random
    run_state: TrainingRunState
    grad_accum: int
    max_steps: int
    log_on_this_rank: bool
    callbacks: list[object]
    event_bus: TrainingEventBus
    live_control: LiveTrainingController | None
    runtime_overrides: TrainingRuntimeOverrides
    resumed: bool
    build_ckpt_payload: Callable[[int], dict[str, object]]
    resume_checkpoint_path: str
    resume_checkpoint_step: int
    resume_lineage_verified: bool
    resume_validation_warnings: list[str]
    gora_initialization_report: dict[str, Any] | None
    esft_calibration_report: dict[str, Any] | None

    def close_callbacks(self) -> None:
        for cb in self.callbacks:
            close_cb = getattr(cb, "close", None)
            if callable(close_cb):
                close_cb()


def create_training_session(
    *,
    config: Any,
    config_path: str,
    runtime_policy_notes: list[str],
    output_dir: str | Path,
    resume_path: str = "",
) -> TrainingSession:
    try:
        context = build_session_artifact_context(
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            resume_path=str(resume_path or ""),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    trainer = Trainer(config)
    try:
        trainer.pipeline.validate_adapter_artifact_lineage(
            dataset_snapshot_id=str(context.manifest.dataset_snapshot_id),
            model_snapshot_id=str(context.manifest.model_snapshot_id),
            config_snapshot_id=str(context.manifest.config_snapshot_id),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    compute_device = resolve_compute_device()
    compute_dtype = resolve_compute_dtype(config)
    place_pipeline_on_device(
        trainer.pipeline,
        device=compute_device,
        dtype=compute_dtype,
        residency_strategy=str(getattr(trainer, "weight_residency_strategy", "disabled")),
    )
    if config.adapter.init_from and not resume_path:
        init_payload = load_adapter_payload(config.adapter.init_from)
        trainer.pipeline.load_adapter_state(
            normalize_adapter_state(init_payload, lora_format="auto")
        )
    batch_schema = resolve_training_batch_schema(
        model_type=config.model.type,
        strategy_type=config.strategy.type,
        strategy=trainer.strategy,
        pipeline=trainer.pipeline,
    )

    try:
        ensure_cache_artifacts_for_training(config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))
    try:
        prepared_data = load_prepared_training_data(
            cache_path=config.dataset.cache_path,
            model_type=config.model.type,
            strategy_type=config.strategy.type,
            val_every_n_steps=int(config.training.val_every_n_steps),
            dataset_snapshot_id=str(context.dataset_snapshot.snapshot_id),
            model_variant=str(getattr(config.model.params, "variant", "")),
            model_component_id=str(
                context.manifest.model_snapshot_meta.get("model_component_id", "")
            ),
            batch_schema=batch_schema,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    policy_record_errors = trainer.training_policies.validation_errors(
        prepared_data.train_records
    )
    if policy_record_errors:
        raise SystemExit(
            "Training policy record validation failed:\n- "
            + "\n- ".join(policy_record_errors)
        )

    rng = random.Random(config.training.seed)
    calibration_run_state = SimpleNamespace(global_step=0)
    try:
        curriculum = CurriculumSchedule.from_config(config.training.curriculum)
        if curriculum.uses_task_mix and config.strategy.type != "multi_task_video":
            raise ValueError(
                "training.curriculum.task_mix_schedule requires "
                "strategy.type='multi_task_video'."
            )
        if config.strategy.type == "multi_task_video" and not curriculum.uses_task_mix:
            raise ValueError(
                "strategy.type='multi_task_video' requires an enabled "
                "training.curriculum.task_mix_schedule."
            )
        curriculum.validate_records(prepared_data.train_records)
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        gora_report = maybe_initialize_gora(
            trainer=trainer,
            config=config,
            prepared_data=prepared_data,
            compute_device=compute_device,
            compute_dtype=compute_dtype,
            curriculum=curriculum,
            rng=rng,
            run_state=calibration_run_state,
            grad_accum=max(1, int(config.training.gradient_accumulation)),
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc))

    try:
        esft_report = maybe_initialize_esft(
            trainer=trainer,
            config=config,
            prepared_data=prepared_data,
            compute_device=compute_device,
            compute_dtype=compute_dtype,
            curriculum=curriculum,
            rng=rng,
            run_state=calibration_run_state,
            grad_accum=max(1, int(config.training.gradient_accumulation)),
        )
    except (ValueError, RuntimeError, TypeError) as exc:
        raise SystemExit(str(exc))
    if esft_report is not None:
        selected_counts = [
            len(ids) for ids in esft_report.plan.selected_experts.values()
        ]
        print(
            "[esft] calibrated "
            f"{len(selected_counts)} layers from {esft_report.observed_samples} "
            f"samples; selected experts/layer "
            f"min={min(selected_counts)} max={max(selected_counts)}; "
            f"plan={esft_report.plan.fingerprint[:12]}",
            file=sys.stderr,
        )

    run_state = init_training_run_state(trainer=trainer, config=config)
    try:
        components = build_training_runtime_components(
            trainer=trainer,
            config=config,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    if components.warmup_warning:
        print(f"[warning] {components.warmup_warning}", file=sys.stderr)

    resume_validation = ResumeValidationResult(
        checkpoint_path="",
        checkpoint_global_step=0,
        lineage_verified=False,
        warnings=[],
    )
    if resume_path:
        try:
            resume_result = apply_resume_checkpoint(
                resume_path=str(resume_path),
                config=config,
                trainer=trainer,
                optimizer=components.optimizer_result.optimizer,
                scheduler=components.scheduler,
                rng=rng,
                run_state=run_state,
                dataset_snapshot_id=str(context.manifest.dataset_snapshot_id),
                cache_snapshot_id=str(context.manifest.cache_snapshot_id),
                model_snapshot_id=str(context.manifest.model_snapshot_id),
                config_snapshot_id=str(context.manifest.config_snapshot_id),
                model_component_id=str(
                    context.manifest.model_snapshot_meta.get("model_component_id", "")
                ),
            )
        except ValueError as exc:
            raise SystemExit(str(exc))
        resume_validation = resume_result.validation
        run_state = resume_result.run_state

    def build_ckpt(step: int) -> dict[str, object]:
        return build_checkpoint_payload(
            step=step,
            trainer=trainer,
            optimizer=components.optimizer_result.optimizer,
            scheduler=components.scheduler,
            rng=rng,
            state=run_state,
            config=config,
            manifest_sha256=context.manifest.manifest_sha256,
            dataset_snapshot_id=context.manifest.dataset_snapshot_id,
            cache_snapshot_id=context.manifest.cache_snapshot_id,
            model_snapshot_id=context.manifest.model_snapshot_id,
            config_snapshot_id=context.manifest.config_snapshot_id,
            model_component_id=str(
                context.manifest.model_snapshot_meta.get("model_component_id", "")
            ),
        )

    callback_context = build_session_callback_context(
        config=config,
        output_dir=context.output_root,
        ckpt_dir=context.ckpt_dir,
        log_on_this_rank=True,
        run_id=context.run_id,
        build_ckpt_payload=build_ckpt,
    )

    return TrainingSession(
        config=config,
        config_path=str(config_path),
        runtime_policy_notes=list(runtime_policy_notes),
        trainer=trainer,
        compute_device=compute_device,
        compute_dtype=compute_dtype,
        run_id=context.run_id,
        manifest=context.manifest,
        manifest_path=context.manifest_path,
        output_dir=context.output_root,
        ckpt_dir=context.ckpt_dir,
        train_records=prepared_data.train_records,
        val_records=prepared_data.val_records,
        temporal_base_ids=prepared_data.temporal_base_ids,
        temporal_groups=prepared_data.temporal_groups,
        cache_data_access_mode=prepared_data.cache_data_access_mode,
        indexed_cache_enabled=bool(prepared_data.indexed_cache_enabled),
        indexed_cache_metadata_path=str(prepared_data.indexed_cache_metadata_path),
        indexed_cache_tensor_path=str(prepared_data.indexed_cache_tensor_path),
        curriculum=curriculum,
        params=components.params,
        named_params=components.named_params,
        optimizer=components.optimizer_result.optimizer,
        optimizer_result=components.optimizer_result,
        scheduler=components.scheduler,
        rng=rng,
        run_state=run_state,
        grad_accum=max(1, int(config.training.gradient_accumulation)),
        max_steps=int(config.training.max_steps),
        log_on_this_rank=True,
        callbacks=callback_context.callbacks,
        event_bus=callback_context.event_bus,
        live_control=LiveTrainingController.from_env(
            run_id=context.run_id,
            event_bus=callback_context.event_bus,
            callbacks=callback_context.callbacks,
            gradient_accumulation=max(1, int(config.training.gradient_accumulation)),
        ),
        runtime_overrides=build_training_runtime_overrides(),
        resumed=bool(resume_path),
        build_ckpt_payload=build_ckpt,
        resume_checkpoint_path=str(resume_validation.checkpoint_path),
        resume_checkpoint_step=int(resume_validation.checkpoint_global_step),
        resume_lineage_verified=bool(resume_validation.lineage_verified),
        resume_validation_warnings=list(resume_validation.warnings),
        gora_initialization_report=(
            None if gora_report is None else gora_report.to_dict()
        ),
        esft_calibration_report=(
            None if esft_report is None else esft_report.to_dict()
        ),
    )
