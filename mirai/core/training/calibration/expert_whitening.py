"""Provider-driven collection of expert projection-input covariances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.models.providers import get_model_family_provider
from mirai.core.models.compressed_weights import packed_artifact_fingerprint
from mirai.core.moe.calibration.whitening import ActivationCovarianceAccumulator
from mirai.core.moe.calibration.whitening import ExpertWhiteningCalibrationTarget
from mirai.core.moe.calibration.whitening import ExpertWhiteningEvidence
from mirai.core.moe.calibration.whitening import save_expert_whitening_evidence
from mirai.core.training.lifecycle.training_step_pre import _build_training_batch_factory
from mirai.core.training.lifecycle.training_step_pre import resolve_step_sampling_context

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ExpertWhiteningCalibrationRunReport:
    output_path: str
    calibration_steps: int
    covariance_budget_bytes: int
    passes: int
    modules: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "output": self.output_path,
            "calibration_steps": self.calibration_steps,
            "covariance_budget_bytes": self.covariance_budget_bytes,
            "passes": self.passes,
            "modules": self.modules,
        }


def _target_groups(
    targets: dict[str, ExpertWhiteningCalibrationTarget],
    *,
    budget_bytes: int,
) -> list[list[ExpertWhiteningCalibrationTarget]]:
    groups: list[list[ExpertWhiteningCalibrationTarget]] = []
    current: list[ExpertWhiteningCalibrationTarget] = []
    current_bytes = 0
    for target in targets.values():
        target_bytes = int(target.covariance_bytes)
        if target_bytes > int(budget_bytes):
            raise ValueError(
                f"Whitening covariance target {target.name!r} requires "
                f"{target_bytes} bytes, exceeding the {int(budget_bytes)}-byte budget."
            )
        if current and current_bytes + target_bytes > int(budget_bytes):
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(target)
        current_bytes += target_bytes
    if current:
        groups.append(current)
    return groups


def run_expert_whitening_calibration_session(
    session: Any,
    *,
    output_path: str | Path,
    calibration_steps: int,
    max_covariance_gib: float,
    overwrite: bool = False,
) -> ExpertWhiteningCalibrationRunReport:
    """Collect exact streamed outer products under an explicit memory bound."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert whitening calibration requires torch.")
    config = session.config
    gate = str(
        getattr(config.model.params, "expert_factorization_calibration", "off")
    ).strip().lower()
    if gate != "whitened":
        raise ValueError(
            "Expert whitening calibration requires "
            "model.params.expert_factorization_calibration='whitened'."
        )
    if str(config.model.params.moe_routing_mode).strip().lower() != "token_choice":
        raise ValueError(
            "Expert whitening calibration requires "
            "model.params.moe_routing_mode='token_choice'."
        )
    steps = int(calibration_steps)
    if steps <= 0:
        raise ValueError("Expert whitening calibration steps must be positive.")
    budget_gib = float(max_covariance_gib)
    if not (budget_gib > 0.0):
        raise ValueError("max_covariance_gib must be positive.")
    budget_bytes = int(budget_gib * (1024**3))
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"Expert whitening output already exists: {output}.")
    packed_path = str(config.memory.frozen_weight_packed_state_path).strip()
    if not packed_path:
        raise ValueError(
            "Expert whitening calibration requires "
            "memory.frozen_weight_packed_state_path."
        )
    provider = get_model_family_provider(str(config.model.type))
    if provider is None or not provider.supports_expert_whitening_calibration(config):
        raise ValueError(
            f"Model provider {config.model.type!r} does not support expert whitening."
        )
    raw_targets = provider.build_expert_whitening_calibration_targets(
        session.trainer.pipeline
    )
    targets: dict[str, ExpertWhiteningCalibrationTarget] = {}
    for name, target in raw_targets.items():
        if not isinstance(target, ExpertWhiteningCalibrationTarget):
            raise TypeError(f"Whitening target {name!r} has the wrong contract.")
        target.validate()
        if target.name != str(name) or name in targets:
            raise ValueError("Whitening target names must match and be unique.")
        targets[str(name)] = target
    if not targets:
        raise ValueError("Model provider returned no expert whitening targets.")
    groups = _target_groups(targets, budget_bytes=budget_bytes)

    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    trainer = session.trainer
    validation_state = trainer.begin_validation()
    sampling_context = resolve_step_sampling_context(session)
    build_batch = _build_training_batch_factory(
        session=session,
        sampling_context=sampling_context,
    )
    evidence: dict[str, ExpertWhiteningEvidence] = {}
    try:
        for group in groups:
            accumulators = {
                target.name: ActivationCovarianceAccumulator(
                    target.projection_input_dims
                )
                for target in group
            }
            attached: list[ExpertWhiteningCalibrationTarget] = []
            try:
                for target in group:
                    target.host.set_whitening_calibration_observer(
                        accumulators[target.name]
                    )
                    attached.append(target)
                with torch.no_grad():
                    for step in range(steps):
                        trainer.compute_loss(build_batch(step), training=False)
            finally:
                for target in reversed(attached):
                    target.host.clear_whitening_calibration_observer()
            for target in group:
                evidence[target.name] = accumulators[target.name].evidence()
            del accumulators
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if rng is not None and rng_state is not None:
            rng.setstate(rng_state)
        trainer.end_validation(validation_state)

    manifest = session.manifest
    save_expert_whitening_evidence(
        output,
        evidence,
        dataset_snapshot_id=str(manifest.dataset_snapshot_id),
        model_snapshot_id=str(manifest.model_snapshot_id),
        config_snapshot_id=str(manifest.config_snapshot_id),
        packed_artifact_fingerprint=packed_artifact_fingerprint(packed_path),
    )
    modules = {
        name: {
            projection: {
                "input_features": item.input_features,
                "sample_count": item.sample_count,
            }
            for projection, item in module.projections.items()
        }
        for name, module in evidence.items()
    }
    return ExpertWhiteningCalibrationRunReport(
        output_path=str(output),
        calibration_steps=steps,
        covariance_budget_bytes=budget_bytes,
        passes=len(groups),
        modules=modules,
    )


__all__ = [
    "ExpertWhiteningCalibrationRunReport",
    "run_expert_whitening_calibration_session",
]
