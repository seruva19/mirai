"""Session driver for fixed-rank EVA LoRA initialization."""

from __future__ import annotations

import sys
from typing import Any

from mirai.core.models.adapters.lora_eva import EVAActivationCalibration
from mirai.core.models.adapters.lora_eva import EVAInitializationReport
from mirai.core.training.lifecycle.training_step_pre import _build_training_batch_factory
from mirai.core.training.lifecycle.training_step_pre import resolve_step_sampling_context

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _log(session: Any, message: str) -> None:
    if bool(getattr(session, "log_on_this_rank", True)):
        print(message, file=sys.stderr)


def maybe_initialize_lora_eva(session: Any) -> EVAInitializationReport | None:
    """Calibrate EVA factors before the first optimizer step.

    Resume restores the already-calibrated A factors from checkpoint. A fresh
    run replays validation-mode forwards while preserving both Python and torch
    sampling state, then copies converged principal directions into A.
    """
    adapter = session.config.adapter
    if str(getattr(adapter, "lora_init", "kaiming")).strip().lower() != "eva":
        return None
    if torch is None:  # pragma: no cover
        raise RuntimeError("EVA initialization requires torch.")
    if bool(getattr(session, "resumed", False)):
        _log(session, "[eva] resumed run: reusing checkpoint initialization.")
        return None
    transformer = getattr(session.trainer.pipeline, "transformer", None)
    if transformer is None:
        raise ValueError("EVA requires a pipeline transformer with LoRA targets.")
    calibration = EVAActivationCalibration(
        transformer,
        samples_per_target=int(adapter.eva_samples_per_target),
        convergence_threshold=float(adapter.eva_convergence_threshold),
    )
    max_steps = int(adapter.eva_calibration_steps)
    sampling_context = resolve_step_sampling_context(session)
    build_batch = _build_training_batch_factory(
        session=session, sampling_context=sampling_context
    )
    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    trainer = session.trainer
    validation_state = trainer.begin_validation()
    completed_steps = 0
    _log(
        session,
        "[eva] calibrating fixed-rank activation PCs: "
        f"max_steps={max_steps} samples_per_target="
        f"{int(adapter.eva_samples_per_target)} threshold="
        f"{float(adapter.eva_convergence_threshold):.6f}",
    )
    calibration.install()
    try:
        with torch.no_grad():
            for step in range(max_steps):
                trainer.compute_loss(build_batch(step), training=False)
                completed_steps = step + 1
                if calibration.converged:
                    break
    finally:
        calibration.close()
        if rng is not None and rng_state is not None:
            rng.setstate(rng_state)
        trainer.end_validation(validation_state)

    report = calibration.initialize(calibration_steps=completed_steps)
    setattr(session, "eva_initialization_report", report)
    _log(
        session,
        "[eva] initialized "
        f"{report.calibrated_ranks} rank directions across "
        f"{len(report.targets)} targets from {report.activation_samples} "
        f"sampled activation rows in {report.calibration_steps} forwards.",
    )
    return report


__all__ = ["maybe_initialize_lora_eva"]
