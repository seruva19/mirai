"""Forward-only EVA spectra collection and adaptive rank-plan creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.models.adapters.lora_adaptive_rank import allocate_adaptive_ranks
from mirai.core.models.adapters.lora_adaptive_rank import save_adaptive_rank_plan
from mirai.core.models.adapters.lora_eva import EVAActivationCalibration
from mirai.core.training.lifecycle.training_step_pre import _build_training_batch_factory
from mirai.core.training.lifecycle.training_step_pre import resolve_step_sampling_context
from mirai.core.training.calibration.conditioning import (
    allocation_conditioning_correlation,
    target_conditioning_diagnostics,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AdaptiveRankCalibrationReport:
    output_path: str
    calibration_steps: int
    rank_budget: int
    target_ranks: dict[str, int]
    plan_fingerprint: str
    conditioning: dict[str, dict[str, float | int]]
    allocation_conditioning_correlation: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "output": self.output_path,
            "calibration_steps": self.calibration_steps,
            "rank_budget": self.rank_budget,
            "target_ranks": dict(sorted(self.target_ranks.items())),
            "plan_fingerprint": self.plan_fingerprint,
            "conditioning": self.conditioning,
            "allocation_conditioning_correlation": (
                self.allocation_conditioning_correlation
            ),
        }


def run_adaptive_rank_calibration_session(
    session: Any,
    *,
    output_path: str | Path,
    calibration_steps: int,
    rank_budget: int,
    minimum_rank: int,
    maximum_rank: int,
    samples_per_target: int,
    convergence_threshold: float,
) -> AdaptiveRankCalibrationReport:
    """Collect activation spectra and write an immutable pre-injection plan."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("Adaptive rank calibration requires torch.")
    transformer = getattr(session.trainer.pipeline, "transformer", None)
    if transformer is None:
        raise ValueError("Adaptive rank calibration requires pipeline.transformer.")
    calibration = EVAActivationCalibration(
        transformer,
        samples_per_target=int(samples_per_target),
        convergence_threshold=float(convergence_threshold),
        components_per_target=int(maximum_rank),
    )
    steps = int(calibration_steps)
    if steps < 2:
        raise ValueError("Adaptive rank calibration_steps must be >= 2.")
    sampling_context = resolve_step_sampling_context(session)
    build_batch = _build_training_batch_factory(
        session=session,
        sampling_context=sampling_context,
    )
    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    trainer = session.trainer
    validation_state = trainer.begin_validation()
    completed_steps = 0
    calibration.install()
    try:
        with torch.no_grad():
            for step in range(steps):
                trainer.compute_loss(build_batch(step), training=False)
                completed_steps = step + 1
                if calibration.converged:
                    break
    finally:
        calibration.close()
        if rng is not None and rng_state is not None:
            rng.setstate(rng_state)
        trainer.end_validation(validation_state)
    spectra = calibration.explained_variance_spectra()
    manifest = session.manifest
    plan = allocate_adaptive_ranks(
        spectra,
        rank_budget=int(rank_budget),
        minimum_rank=int(minimum_rank),
        maximum_rank=int(maximum_rank),
        lineage={
            "dataset_snapshot_id": str(manifest.dataset_snapshot_id),
            "model_snapshot_id": str(manifest.model_snapshot_id),
            "config_snapshot_id": str(manifest.config_snapshot_id),
        },
    )
    save_adaptive_rank_plan(output_path, plan)
    conditioning = target_conditioning_diagnostics(spectra)
    return AdaptiveRankCalibrationReport(
        output_path=str(output_path),
        calibration_steps=completed_steps,
        rank_budget=plan.rank_budget,
        target_ranks=dict(plan.ranks),
        plan_fingerprint=plan.fingerprint,
        conditioning=conditioning,
        allocation_conditioning_correlation=allocation_conditioning_correlation(
            conditioning, plan.ranks
        ),
    )


__all__ = [
    "AdaptiveRankCalibrationReport",
    "run_adaptive_rank_calibration_session",
]
