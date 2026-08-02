"""Training step orchestration facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.training.lifecycle.training_step_post import process_completed_training_step
from mirai.core.training.lifecycle.training_step_pre import (
    resolve_step_sampling_context,
    run_pre_postprocessed_training_step,
)


@dataclass(frozen=True)
class TrainingStepResult:
    advanced: bool
    curriculum_profile: Any
    grad_norm_value: float
    step_metrics: dict[str, Any]
    applied_control: Any | None = None


def run_training_step(session: Any) -> TrainingStepResult:
    step_result = run_pre_postprocessed_training_step(
        session,
        sampling_context=resolve_step_sampling_context(session),
    )
    applied_control = getattr(step_result, "applied_control", None)
    if not step_result.advanced:
        return TrainingStepResult(
            advanced=False,
            curriculum_profile=step_result.curriculum_profile,
            grad_norm_value=0.0,
            step_metrics={},
            applied_control=applied_control,
        )

    completed_result = process_completed_training_step(
        session,
        curriculum_profile=step_result.curriculum_profile,
        grad_norm_value=step_result.grad_norm_value,
        applied_control=applied_control,
    )
    return TrainingStepResult(
        advanced=True,
        curriculum_profile=step_result.curriculum_profile,
        grad_norm_value=float(step_result.grad_norm_value),
        step_metrics=dict(completed_result.step_metrics),
        applied_control=applied_control,
    )
