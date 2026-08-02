"""Single-GPU pre-optimizer ESFT affinity calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.calibration.esft import ESFTCalibrationCapture
from mirai.core.moe.calibration.esft import ESFTCalibrationTarget
from mirai.core.moe.calibration.esft import ESFTSelectionPlan
from mirai.core.moe.calibration.esft import build_esft_selection_plan
from mirai.core.training.lifecycle.training_step_pre import (
    _build_training_batch_factory,
)
from mirai.core.training.lifecycle.training_step_pre import (
    resolve_step_sampling_context,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ESFTCalibrationReport:
    """Reader-facing summary retained by the training session."""

    plan: ESFTSelectionPlan
    calibration_steps: int
    requested_samples: int
    observed_samples: int

    def to_dict(self) -> dict[str, Any]:
        payload = self.plan.to_dict()
        payload.update(
            {
                "calibration_steps": int(self.calibration_steps),
                "requested_samples": int(self.requested_samples),
                "observed_samples": int(self.observed_samples),
                "selected_expert_count": {
                    name: len(ids)
                    for name, ids in sorted(self.plan.selected_experts.items())
                },
            }
        )
        return payload


def _validate_targets(
    raw_targets: dict[str, ESFTCalibrationTarget],
) -> dict[str, ESFTCalibrationTarget]:
    if not raw_targets:
        raise ValueError("Model provider returned no ESFT calibration targets.")
    targets: dict[str, ESFTCalibrationTarget] = {}
    for raw_name, target in raw_targets.items():
        if not isinstance(target, ESFTCalibrationTarget):
            raise TypeError(
                "Model provider ESFT targets must use ESFTCalibrationTarget."
            )
        target.validate()
        name = str(raw_name)
        if name != target.name or name in targets:
            raise ValueError("ESFT target names must match and be unique.")
        targets[name] = target
    return targets


def maybe_initialize_esft(
    *,
    trainer: Any,
    config: Any,
    prepared_data: Any,
    compute_device: Any,
    compute_dtype: Any,
    curriculum: Any,
    rng: Any,
    run_state: Any,
    grad_accum: int,
) -> ESFTCalibrationReport | None:
    """Calibrate and bind a per-layer ESFT plan before optimizer creation."""

    mode = str(config.adapter.expert_selection).strip().lower()
    if mode not in {"esft_gate", "esft_token"}:
        return None
    if torch is None:  # pragma: no cover
        raise RuntimeError("ESFT calibration requires torch.")
    provider = get_model_family_provider(str(config.model.type))
    if provider is None or not provider.supports_esft_expert_selection(config):
        raise ValueError(
            f"Model provider {config.model.type!r} does not support ESFT selection."
        )
    targets = _validate_targets(
        provider.build_esft_calibration_targets(trainer.pipeline)
    )
    requested_samples = int(config.adapter.esft_calibration_samples)
    batch_size = max(1, int(config.training.batch_size))
    calibration_steps = max(1, math.ceil(requested_samples / batch_size))
    observed_samples = calibration_steps * batch_size

    calibration_session = SimpleNamespace(
        config=config,
        trainer=trainer,
        compute_device=compute_device,
        compute_dtype=compute_dtype,
        train_records=prepared_data.train_records,
        temporal_base_ids=prepared_data.temporal_base_ids,
        temporal_groups=prepared_data.temporal_groups,
        curriculum=curriculum,
        rng=rng,
        run_state=run_state,
        grad_accum=max(1, int(grad_accum)),
    )
    sampling_context = resolve_step_sampling_context(calibration_session)
    build_batch = _build_training_batch_factory(
        session=calibration_session,
        sampling_context=sampling_context,
    )

    validation_state = trainer.begin_validation()
    rng_state = rng.getstate()
    try:
        with ESFTCalibrationCapture(targets) as capture:
            with torch.no_grad():
                for step in range(calibration_steps):
                    trainer.compute_loss(build_batch(step), training=False)
        plan = build_esft_selection_plan(
            capture.accumulators,
            score_mode=mode,
            selection_mass=float(config.adapter.esft_selection_mass),
            calibration_samples=observed_samples,
        )
    finally:
        rng.setstate(rng_state)
        trainer.end_validation(validation_state)

    setter = getattr(trainer.pipeline, "set_selected_expert_plan", None)
    if not callable(setter):
        raise ValueError(
            f"Model provider {config.model.type!r} cannot bind an ESFT plan."
        )
    setter(plan.selected_experts)
    return ESFTCalibrationReport(
        plan=plan,
        calibration_steps=calibration_steps,
        requested_samples=requested_samples,
        observed_samples=observed_samples,
    )


__all__ = ["ESFTCalibrationReport", "maybe_initialize_esft"]
