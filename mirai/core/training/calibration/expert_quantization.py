"""Provider-driven execution for MoE quantization calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.calibration.quantization import ExpertAffinityAccumulator
from mirai.core.moe.calibration.quantization import ExpertAffinityEvidence
from mirai.core.moe.calibration.quantization import QuantizationCalibrationTarget
from mirai.core.moe.calibration.quantization import (
    register_quantization_calibration_hooks,
)
from mirai.core.moe.calibration.quantization import (
    save_quantization_calibration_evidence,
)
from mirai.core.training.lifecycle.training_step_pre import _build_training_batch_factory
from mirai.core.training.lifecycle.training_step_pre import resolve_step_sampling_context

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class QuantizationCalibrationRunReport:
    output_path: str
    calibration_steps: int
    sample_budget: int
    modules: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "output": self.output_path,
            "calibration_steps": self.calibration_steps,
            "sample_budget": self.sample_budget,
            "modules": self.modules,
        }


def run_moe_quantization_calibration_session(
    session: Any,
    *,
    output_path: str | Path,
    calibration_steps: int,
    sample_budget: int,
    overwrite: bool = False,
) -> QuantizationCalibrationRunReport:
    """Run forward-only route-affinity collection without perturbing training RNG."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("MoE quantization calibration requires torch.")
    config = session.config
    gate = str(
        getattr(config.model.params, "expert_quantization_calibration", "off")
    ).strip().lower()
    if gate != "affinity":
        raise ValueError(
            "MoE quantization calibration requires "
            "model.params.expert_quantization_calibration='affinity'."
        )
    steps = int(calibration_steps)
    budget = int(sample_budget)
    if steps <= 0 or budget <= 0:
        raise ValueError("Calibration steps and sample budget must be positive.")
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"Quantization calibration output already exists: {output}.")
    provider = get_model_family_provider(str(config.model.type))
    if provider is None or not provider.supports_moe_quantization_calibration(
        config
    ):
        raise ValueError(
            f"Model provider {config.model.type!r} does not support "
            "MoE quantization calibration."
        )
    raw_targets = provider.build_moe_quantization_calibration_targets(
        session.trainer.pipeline
    )
    targets: dict[str, QuantizationCalibrationTarget] = {}
    for name, target in raw_targets.items():
        if not isinstance(target, QuantizationCalibrationTarget):
            raise TypeError(
                f"Quantization calibration target {name!r} has the wrong contract."
            )
        target.validate()
        if target.name != str(name) or name in targets:
            raise ValueError("Quantization calibration target names must match and be unique.")
        targets[str(name)] = target
    if not targets:
        raise ValueError("Model provider returned no quantization calibration targets.")
    accumulators = {
        name: ExpertAffinityAccumulator(target.num_experts)
        for name, target in targets.items()
    }
    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    trainer = session.trainer
    validation_state = trainer.begin_validation()
    handles: list[Any] = []
    try:
        handles = register_quantization_calibration_hooks(targets, accumulators)
        sampling_context = resolve_step_sampling_context(session)
        build_batch = _build_training_batch_factory(
            session=session,
            sampling_context=sampling_context,
        )
        with torch.no_grad():
            for step in range(steps):
                trainer.compute_loss(build_batch(step), training=False)
    finally:
        for handle in reversed(handles):
            handle.remove()
        if rng is not None and rng_state is not None:
            rng.setstate(rng_state)
        trainer.end_validation(validation_state)
    evidence: dict[str, ExpertAffinityEvidence] = {
        name: accumulator.evidence() for name, accumulator in accumulators.items()
    }
    manifest = session.manifest
    save_quantization_calibration_evidence(
        output,
        evidence,
        dataset_snapshot_id=str(manifest.dataset_snapshot_id),
        model_snapshot_id=str(manifest.model_snapshot_id),
        config_snapshot_id=str(manifest.config_snapshot_id),
    )
    module_reports = {
        name: {
            "num_experts": item.num_experts,
            "observed_routes": int(torch.as_tensor(item.selected_count).sum().item()),
            "candidate_samples": item.num_samples,
            "balanced_sample_indices": list(
                item.balanced_sample_indices(min(budget, item.num_samples))
            ),
            "covered_experts": int(
                (torch.as_tensor(item.selected_count) > 0).sum().item()
            ),
            "affinity_weight_min": float(
                item.affinity_reconstruction_weights(
                    require_full_coverage=False
                ).min().item()
            ),
            "affinity_weight_max": float(
                item.affinity_reconstruction_weights(
                    require_full_coverage=False
                ).max().item()
            ),
        }
        for name, item in evidence.items()
    }
    return QuantizationCalibrationRunReport(
        output_path=str(output),
        calibration_steps=steps,
        sample_budget=budget,
        modules=module_reports,
    )


__all__ = [
    "QuantizationCalibrationRunReport",
    "run_moe_quantization_calibration_session",
]
