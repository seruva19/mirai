"""Provider-driven offline calibration for expert prototype consolidation."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from mirai.core.lineage import bind_snapshot_component
from mirai.core.lineage import normalize_snapshot_component
from mirai.core.lineage import snapshot_descriptor_for_path
from mirai.core.models.compressed_weights import packed_artifact_fingerprint
from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.calibration.prototypes import MeanExpertOutputAccumulator
from mirai.core.moe.calibration.prototypes import PrototypeContributionAccumulator
from mirai.core.moe.calibration.prototypes import save_prototype_calibration_evidence
from mirai.core.moe.calibration.projection import PrototypeCalibrationTarget
from mirai.core.moe.calibration.projection import normalized_parameter_distance_streaming
from mirai.core.training.lifecycle.training_step_pre import _build_training_batch_factory
from mirai.core.training.lifecycle.training_step_pre import resolve_step_sampling_context

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class PrototypeCalibrationRunReport:
    output: str
    calibration_steps: int
    max_block_elements: int
    max_output_tokens_per_observation: int
    modules: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "output": self.output,
            "calibration_steps": self.calibration_steps,
            "max_block_elements": self.max_block_elements,
            "max_output_tokens_per_observation": (
                self.max_output_tokens_per_observation
            ),
            "modules": self.modules,
        }


def expected_prototype_calibration_lineage(
    config: Any,
    config_path: str | Path,
) -> dict[str, str]:
    """Resolve exact dataset, model, config, and packed-source lineage."""
    packed_path = str(config.memory.frozen_weight_packed_state_path).strip()
    if not packed_path:
        raise ValueError(
            "Expert consolidation requires "
            "memory.frozen_weight_packed_state_path so calibration and transform "
            "can be bound to the exact packed source."
        )
    model_root = snapshot_descriptor_for_path(config.model.path)
    component = normalize_snapshot_component(
        getattr(config.model.params, "denoiser_subfolder", "transformer")
        or "transformer"
    )
    model = bind_snapshot_component(
        model_root,
        component=component,
        component_label="denoiser_subfolder",
    )
    return {
        "dataset_snapshot_id": snapshot_descriptor_for_path(
            config.dataset.path
        ).snapshot_id,
        "model_snapshot_id": model.snapshot_id,
        "config_snapshot_id": snapshot_descriptor_for_path(config_path).snapshot_id,
        "packed_artifact_fingerprint": packed_artifact_fingerprint(packed_path),
    }


def run_prototype_calibration_session(
    session: Any,
    *,
    output_path: str | Path,
    calibration_steps: int,
    max_block_elements: int,
    max_output_tokens_per_observation: int = 256,
    projection_device: Any | None = None,
    distance_dtype: Any = None,
    overwrite: bool = False,
) -> PrototypeCalibrationRunReport:
    """Collect contribution evidence, stream distances, and write one artifact."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("Prototype calibration requires torch.")
    config = session.config
    gate = str(
        getattr(config.model.params, "expert_consolidation", "off")
    ).strip().lower()
    if gate not in {"prototype", "hierarchical_output"}:
        raise ValueError(
            "Expert consolidation calibration requires model.params."
            "expert_consolidation='prototype' or 'hierarchical_output'."
        )
    steps = int(calibration_steps)
    if steps < 1:
        raise ValueError("calibration_steps must be positive.")
    block_elements = int(max_block_elements)
    if block_elements < 1:
        raise ValueError("max_block_elements must be positive.")
    output_token_limit = int(max_output_tokens_per_observation)
    if output_token_limit < 1:
        raise ValueError("max_output_tokens_per_observation must be positive.")
    output = Path(output_path)
    if output.exists() and not bool(overwrite):
        raise ValueError(
            f"Prototype calibration output already exists: {output}. "
            "Pass overwrite=True to replace it."
        )

    provider = get_model_family_provider(str(config.model.type))
    if provider is None or not provider.supports_prototype_calibration(config):
        raise ValueError(
            f"model.type={config.model.type!r} does not support prototype calibration."
        )
    raw_targets = provider.build_prototype_calibration_targets(
        session.trainer.pipeline
    )
    targets = _validate_targets(raw_targets)
    if gate == "hierarchical_output":
        accumulators = {
            name: MeanExpertOutputAccumulator(
                target.projection_source.num_experts,
                max_tokens_per_observation=output_token_limit,
            )
            for name, target in targets.items()
        }
    else:
        accumulators = {
            name: PrototypeContributionAccumulator(
                target.projection_source.num_experts
            )
            for name, target in targets.items()
        }

    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    trainer = session.trainer
    validation_state = trainer.begin_validation()
    attached: list[PrototypeCalibrationTarget] = []
    try:
        for name, target in targets.items():
            target.host.set_prototype_calibration_observer(accumulators[name])
            attached.append(target)
        sampling_context = resolve_step_sampling_context(session)
        build_batch = _build_training_batch_factory(
            session=session,
            sampling_context=sampling_context,
        )
        with torch.no_grad():
            for step in range(steps):
                batch = build_batch(step)
                trainer.compute_loss(batch, training=False)
    finally:
        for target in reversed(attached):
            target.host.clear_prototype_calibration_observer()
        if rng is not None and rng_state is not None:
            rng.setstate(rng_state)
        trainer.end_validation(validation_state)

    resolved_device = (
        getattr(session, "compute_device", "cpu")
        if projection_device is None
        else projection_device
    )
    resolved_dtype = torch.float64 if distance_dtype is None else distance_dtype
    evidence = {}
    for name, target in targets.items():
        if gate == "hierarchical_output":
            evidence[name] = accumulators[name].evidence()
        else:
            distances = normalized_parameter_distance_streaming(
                target.projection_source,
                max_block_elements=block_elements,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            evidence[name] = accumulators[name].evidence(distances)
    manifest = session.manifest
    packed_path = str(config.memory.frozen_weight_packed_state_path).strip()
    if not packed_path:
        raise ValueError(
            "Expert consolidation calibration requires "
            "memory.frozen_weight_packed_state_path."
        )
    save_prototype_calibration_evidence(
        output,
        evidence,
        dataset_snapshot_id=str(manifest.dataset_snapshot_id),
        model_snapshot_id=str(manifest.model_snapshot_id),
        config_snapshot_id=str(manifest.config_snapshot_id),
        packed_artifact_fingerprint=packed_artifact_fingerprint(packed_path),
    )
    return PrototypeCalibrationRunReport(
        output=str(output),
        calibration_steps=steps,
        max_block_elements=block_elements,
        max_output_tokens_per_observation=output_token_limit,
        modules={
            name: {
                "experts": target.projection_source.num_experts,
                "selected_routes": int(accumulators[name].selected_count.sum().item()),
                **(
                    {
                        "output_tokens_per_expert": (
                            accumulators[name].output_tokens_per_expert
                        )
                    }
                    if isinstance(accumulators[name], MeanExpertOutputAccumulator)
                    else {}
                ),
            }
            for name, target in targets.items()
        },
    )


def _validate_targets(
    raw_targets: dict[str, PrototypeCalibrationTarget],
) -> dict[str, PrototypeCalibrationTarget]:
    if not raw_targets:
        raise ValueError("Model provider returned no prototype calibration targets.")
    targets: dict[str, PrototypeCalibrationTarget] = {}
    for raw_name, target in raw_targets.items():
        if not isinstance(target, PrototypeCalibrationTarget):
            raise TypeError(
                "Model provider prototype calibration targets must use "
                "PrototypeCalibrationTarget."
            )
        target.validate()
        name = str(raw_name)
        if name != target.name:
            raise ValueError(
                f"Prototype calibration target key {name!r} does not match "
                f"target.name {target.name!r}."
            )
        if name in targets:
            raise ValueError(f"Duplicate prototype calibration target {name!r}.")
        targets[name] = target
    return targets


__all__ = [
    "PrototypeCalibrationRunReport",
    "expected_prototype_calibration_lineage",
    "run_prototype_calibration_session",
]
