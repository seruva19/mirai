"""Resume checkpoint application for training sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.persistence.checkpoints import load_checkpoint
from mirai.core.training.lifecycle.resume_validation import (
    ResumeValidationResult,
    validate_resume_checkpoint_compatibility,
)
from mirai.core.training.lifecycle.session_state import restore_training_run_state


@dataclass(frozen=True)
class ResumeApplicationResult:
    run_state: Any
    validation: ResumeValidationResult


def apply_resume_checkpoint(
    *,
    resume_path: str,
    config: Any,
    trainer: Any,
    optimizer: Any,
    scheduler: Any,
    rng: Any,
    run_state: Any,
    dataset_snapshot_id: str,
    cache_snapshot_id: str,
    model_snapshot_id: str,
    config_snapshot_id: str = "",
    model_component_id: str = "",
    runtime_overrides: Any | None = None,
    optimizer_result: Any | None = None,
) -> ResumeApplicationResult:
    payload = load_checkpoint(resume_path)
    validation = validate_resume_checkpoint_compatibility(
        payload=payload,
        config=config,
        resume_path=str(resume_path),
        dataset_snapshot_id=str(dataset_snapshot_id),
        cache_snapshot_id=str(cache_snapshot_id),
        model_snapshot_id=str(model_snapshot_id),
        config_snapshot_id=str(config_snapshot_id),
        model_component_id=str(model_component_id),
        optimizer_result=optimizer_result,
    )
    trainer.load_state_dict(payload.get("trainer_state", {}))
    optimizer.load_state_dict(payload.get("optimizer_state", {}))
    scheduler_state = payload.get("scheduler_state")
    if isinstance(scheduler_state, dict):
        scheduler.load_state_dict(scheduler_state)
    restored_state = restore_training_run_state(
        state=run_state,
        payload=payload,
        rng=rng,
        config=config,
    )
    if runtime_overrides is not None:
        from mirai.core.training.control.live_control_actions import (
            load_training_runtime_overrides_state,
        )

        load_training_runtime_overrides_state(
            runtime_overrides,
            payload.get("runtime_overrides"),
        )
    return ResumeApplicationResult(
        run_state=restored_state,
        validation=validation,
    )
