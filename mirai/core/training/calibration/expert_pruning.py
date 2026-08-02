"""Provider-driven calibration for structured expert-pruning criteria."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.calibration.pruning import ExpertPruningCalibrationTarget
from mirai.core.moe.calibration.pruning import ExpertPruningRoutedOutputObserver
from mirai.core.moe.calibration.pruning import ExpertPruningSaliencyAccumulator
from mirai.core.moe.calibration.pruning import normalize_expert_pruning_criterion
from mirai.core.moe.calibration.pruning import save_expert_pruning_evidence
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
class ExpertPruningCalibrationRunReport:
    output_path: str
    calibration_steps: int
    criterion: str
    modules: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "output": self.output_path,
            "calibration_steps": self.calibration_steps,
            "criterion": self.criterion,
            "modules": self.modules,
        }


def _validate_targets(
    raw_targets: dict[str, ExpertPruningCalibrationTarget],
) -> dict[str, ExpertPruningCalibrationTarget]:
    if not raw_targets:
        raise ValueError("Model provider returned no expert-pruning targets.")
    targets: dict[str, ExpertPruningCalibrationTarget] = {}
    for raw_name, target in raw_targets.items():
        if not isinstance(target, ExpertPruningCalibrationTarget):
            raise TypeError(
                "Model provider expert-pruning targets must use "
                "ExpertPruningCalibrationTarget."
            )
        target.validate()
        name = str(raw_name)
        if name != target.name or name in targets:
            raise ValueError(
                "Expert-pruning calibration target names must match and be unique."
            )
        targets[name] = target
    return targets


def run_expert_pruning_calibration_session(
    session: Any,
    *,
    output_path: str | Path,
    calibration_steps: int,
    criterion: str,
    overwrite: bool = False,
) -> ExpertPruningCalibrationRunReport:
    """Collect exact route/output statistics without optimizer or gradient work."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert-pruning calibration requires torch.")
    config = session.config
    gate = str(getattr(config.model.params, "expert_pruning", "off")).strip().lower()
    if gate != "prune":
        raise ValueError(
            "Expert-pruning calibration requires model.params.expert_pruning='prune'."
        )
    resolved_criterion = normalize_expert_pruning_criterion(criterion)
    configured = normalize_expert_pruning_criterion(
        getattr(config.model.params, "expert_pruning_criterion", "frequency")
    )
    if resolved_criterion != configured:
        raise ValueError(
            f"Requested pruning criterion {resolved_criterion!r} does not match "
            f"model.params.expert_pruning_criterion={configured!r}."
        )
    steps = int(calibration_steps)
    if steps <= 0:
        raise ValueError("Expert-pruning calibration steps must be positive.")
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"Expert-pruning calibration output already exists: {output}.")

    provider = get_model_family_provider(str(config.model.type))
    if provider is None or not provider.supports_expert_pruning_calibration(config):
        raise ValueError(
            f"Model provider {config.model.type!r} does not support "
            "expert-pruning calibration."
        )
    targets = _validate_targets(
        provider.build_expert_pruning_calibration_targets(session.trainer.pipeline)
    )
    accumulators = {
        name: ExpertPruningSaliencyAccumulator(
            target.num_experts,
            criterion=resolved_criterion,
        )
        for name, target in targets.items()
    }
    observers = {
        name: ExpertPruningRoutedOutputObserver(accumulators[name])
        for name in targets
    }
    previous_observers = {
        name: target.host.get_expert_output_observer()
        for name, target in targets.items()
    }
    if any(observer is not None for observer in previous_observers.values()):
        raise ValueError(
            "Expert-pruning calibration requires the training-only expert-output "
            "regularizer to be disabled so its observer is not replaced."
        )

    trainer = session.trainer
    training_model = trainer.pipeline.get_training_model()
    if training_model is None:
        raise ValueError("Expert-pruning calibration requires an exposed training model.")
    was_training = bool(getattr(training_model, "training", True))
    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    attached: list[ExpertPruningCalibrationTarget] = []
    try:
        for name, target in targets.items():
            target.host.set_expert_output_observer(observers[name])
            attached.append(target)
        training_model.train(True)
        sampling_context = resolve_step_sampling_context(session)
        build_batch = _build_training_batch_factory(
            session=session,
            sampling_context=sampling_context,
        )
        with torch.no_grad():
            for step in range(steps):
                trainer.compute_loss(build_batch(step), training=False)
    finally:
        for target in reversed(attached):
            target.host.set_expert_output_observer(
                previous_observers[target.name]
            )
        training_model.train(was_training)
        if rng is not None and rng_state is not None:
            rng.setstate(rng_state)

    evidence = {
        name: accumulator.evidence()
        for name, accumulator in accumulators.items()
    }
    manifest = session.manifest
    save_expert_pruning_evidence(
        output,
        evidence,
        dataset_snapshot_id=str(manifest.dataset_snapshot_id),
        model_snapshot_id=str(manifest.model_snapshot_id),
        config_snapshot_id=str(manifest.config_snapshot_id),
    )
    return ExpertPruningCalibrationRunReport(
        output_path=str(output),
        calibration_steps=steps,
        criterion=resolved_criterion,
        modules={
            name: {
                "num_experts": item.num_experts,
                "observed_routes": int(
                    torch.as_tensor(item.selected_count).sum().item()
                ),
                "covered_experts": int(
                    (torch.as_tensor(item.selected_count) > 0).sum().item()
                ),
            }
            for name, item in evidence.items()
        },
    )


__all__ = [
    "ExpertPruningCalibrationRunReport",
    "run_expert_pruning_calibration_session",
]
