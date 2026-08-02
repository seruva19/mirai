"""Provider-driven imatrix collection and per-tensor precision allocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from mirai.core.models.compressed_weights.quantization.sensitivity import (
    EXPERT_TENSOR_PRECISION_FORMATS,
)
from mirai.core.models.compressed_weights.quantization.sensitivity import (
    measure_projection_format,
)
from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.calibration.imatrix import ExpertImportanceAccumulator
from mirai.core.moe.calibration.imatrix import ExpertImportanceCalibrationTarget
from mirai.core.moe.calibration.imatrix import ExpertImportanceEvidence
from mirai.core.moe.calibration.precision import TensorPrecisionEvidence
from mirai.core.moe.calibration.precision import allocate_tensor_precision
from mirai.core.moe.calibration.router_repair import router_tensor_fingerprint
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
class ExpertPrecisionCalibrationRunReport:
    output_path: str
    calibration_steps: int
    accumulator_budget_bytes: int
    passes: int
    candidate_formats: tuple[str, ...]
    estimated_bytes: int
    weighted_error: float
    modules: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "output": self.output_path,
            "calibration_steps": self.calibration_steps,
            "accumulator_budget_bytes": self.accumulator_budget_bytes,
            "passes": self.passes,
            "candidate_formats": list(self.candidate_formats),
            "estimated_bytes": self.estimated_bytes,
            "weighted_error": self.weighted_error,
            "modules": self.modules,
        }


def _normalize_targets(
    raw_targets: dict[str, ExpertImportanceCalibrationTarget],
) -> dict[str, ExpertImportanceCalibrationTarget]:
    targets: dict[str, ExpertImportanceCalibrationTarget] = {}
    for raw_name, target in raw_targets.items():
        if not isinstance(target, ExpertImportanceCalibrationTarget):
            raise TypeError(
                "Expert precision targets must use "
                "ExpertImportanceCalibrationTarget."
            )
        target.validate()
        name = str(raw_name)
        if name != target.name or name in targets:
            raise ValueError(
                "Expert precision target names must match and be unique."
            )
        targets[name] = target
    if not targets:
        raise ValueError("Model provider returned no expert precision targets.")
    return dict(sorted(targets.items()))


def _target_groups(
    targets: dict[str, ExpertImportanceCalibrationTarget],
    *,
    budget_bytes: int,
) -> list[list[ExpertImportanceCalibrationTarget]]:
    groups: list[list[ExpertImportanceCalibrationTarget]] = []
    current: list[ExpertImportanceCalibrationTarget] = []
    current_bytes = 0
    for target in targets.values():
        required = int(target.accumulator_bytes)
        if required > int(budget_bytes):
            raise ValueError(
                f"Importance target {target.name!r} requires {required} bytes, "
                f"exceeding the {int(budget_bytes)}-byte budget."
            )
        if current and current_bytes + required > int(budget_bytes):
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(target)
        current_bytes += required
    if current:
        groups.append(current)
    return groups


def _source_tensors(
    targets: dict[str, ExpertImportanceCalibrationTarget],
) -> dict[str, Any]:
    return {
        f"{name}.{projection}": target.weights[projection]
        for name, target in targets.items()
        for projection in ("w1", "w2", "w3")
    }


def _measurement_evidence(
    targets: dict[str, ExpertImportanceCalibrationTarget],
    observations: dict[str, ExpertImportanceEvidence],
    *,
    formats: tuple[str, ...],
) -> list[TensorPrecisionEvidence]:
    rows: list[TensorPrecisionEvidence] = []
    for name, target in targets.items():
        evidence = observations[name].validate()
        for projection in ("w1", "w2", "w3"):
            counts = torch.as_tensor(evidence.observation_counts[projection])
            total_count = int(counts.sum().item())
            if total_count <= 0:
                raise ValueError(
                    f"Importance calibration observed no {name}.{projection} inputs."
                )
            source = torch.as_tensor(target.weights[projection]).detach()
            for expert_id in range(target.num_experts):
                importance = evidence.mean_squares(projection, expert_id)
                format_error: dict[str, float] = {}
                format_bytes: dict[str, int] = {}
                for quant_format in formats:
                    measurement = measure_projection_format(
                        source[expert_id],
                        importance,
                        quant_format=quant_format,
                        projection=projection,
                    )
                    format_error[quant_format] = measurement.weighted_mse
                    format_bytes[quant_format] = measurement.stored_bytes
                rows.append(
                    TensorPrecisionEvidence(
                        module_name=name,
                        expert_id=expert_id,
                        projection=projection,
                        weight_numel=int(source[expert_id].numel()),
                        format_error=format_error,
                        format_bytes=format_bytes,
                        routing_frequency=(
                            float(counts[expert_id].item()) / float(total_count)
                        ),
                    )
                )
    return rows


def run_expert_precision_calibration_session(
    session: Any,
    *,
    output_path: str | Path,
    calibration_steps: int,
    max_accumulator_gib: float,
    budget_bytes: int,
    allowed_formats: Sequence[str],
    overwrite: bool = False,
) -> ExpertPrecisionCalibrationRunReport:
    """Collect routed imatrix evidence and emit a schema-v2 runtime plan."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert precision calibration requires torch.")
    config = session.config
    gate = str(
        getattr(config.model.params, "expert_precision_calibration", "off")
    ).strip().lower()
    if gate != "imatrix":
        raise ValueError(
            "Expert precision calibration requires "
            "model.params.expert_precision_calibration='imatrix'."
        )
    if str(config.memory.frozen_weight_quantization).strip().lower() not in {
        "",
        "none",
        "off",
        "disabled",
    }:
        raise ValueError(
            "Expert precision calibration must load floating-point expert weights."
        )
    if str(config.memory.frozen_weight_packed_state_path).strip() or str(
        config.memory.expert_precision_plan_path
    ).strip():
        raise ValueError(
            "Expert precision calibration cannot consume packed weights or an "
            "existing precision plan."
        )
    steps = int(calibration_steps)
    max_gib = float(max_accumulator_gib)
    if steps <= 0 or not (max_gib > 0.0) or int(budget_bytes) <= 0:
        raise ValueError("Calibration steps, accumulator budget, and plan budget must be positive.")
    formats = tuple(
        dict.fromkeys(str(value).strip().lower() for value in allowed_formats)
    )
    if not formats or any(
        value not in EXPERT_TENSOR_PRECISION_FORMATS for value in formats
    ):
        raise ValueError("Expert precision candidate format is unsupported.")
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"Expert precision output already exists: {output}.")

    provider = get_model_family_provider(str(config.model.type))
    if provider is None or not provider.supports_expert_precision_calibration(config):
        raise ValueError(
            f"Model provider {config.model.type!r} does not support expert "
            "precision calibration."
        )
    targets = _normalize_targets(
        provider.build_expert_precision_calibration_targets(
            session.trainer.pipeline
        )
    )
    source_fingerprint = router_tensor_fingerprint(_source_tensors(targets))
    accumulator_budget_bytes = int(max_gib * (1024**3))
    groups = _target_groups(targets, budget_bytes=accumulator_budget_bytes)

    trainer = session.trainer
    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    validation_state = trainer.begin_validation()
    sampling_context = resolve_step_sampling_context(session)
    build_batch = _build_training_batch_factory(
        session=session,
        sampling_context=sampling_context,
    )
    observations: dict[str, ExpertImportanceEvidence] = {}
    try:
        for group in groups:
            if rng is not None and rng_state is not None:
                rng.setstate(rng_state)
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_state:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            accumulators = {
                target.name: ExpertImportanceAccumulator(
                    num_experts=target.num_experts,
                    input_dims={
                        key: int(torch.as_tensor(target.weights[key]).shape[-1])
                        for key in ("w1", "w2", "w3")
                    },
                )
                for target in group
            }
            attached: list[ExpertImportanceCalibrationTarget] = []
            try:
                for target in group:
                    target.host.set_importance_calibration_observer(
                        accumulators[target.name]
                    )
                    attached.append(target)
                with torch.no_grad():
                    for step in range(steps):
                        trainer.compute_loss(build_batch(step), training=False)
            finally:
                for target in reversed(attached):
                    target.host.clear_importance_calibration_observer()
            for target in group:
                observations[target.name] = accumulators[target.name].evidence()
            del accumulators
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if rng is not None and rng_state is not None:
            rng.setstate(rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        trainer.end_validation(validation_state)

    rows = _measurement_evidence(targets, observations, formats=formats)
    manifest = session.manifest
    plan = allocate_tensor_precision(
        rows,
        budget_bytes=int(budget_bytes),
        allowed_formats=formats,
        dataset_snapshot_id=str(manifest.dataset_snapshot_id),
        model_snapshot_id=str(manifest.model_snapshot_id),
        config_snapshot_id=str(manifest.config_snapshot_id),
        source_weight_fingerprint=source_fingerprint,
    )
    plan.save(output)
    modules = {
        name: {
            "num_experts": target.num_experts,
            "formats": {
                projection: list(
                    plan.formats_for_module(name)[projection]
                )
                for projection in ("w1", "w2", "w3")
            },
        }
        for name, target in targets.items()
    }
    return ExpertPrecisionCalibrationRunReport(
        output_path=str(output),
        calibration_steps=steps,
        accumulator_budget_bytes=accumulator_budget_bytes,
        passes=len(groups),
        candidate_formats=formats,
        estimated_bytes=plan.estimated_bytes,
        weighted_error=plan.weighted_error,
        modules=modules,
    )


__all__ = [
    "ExpertPrecisionCalibrationRunReport",
    "run_expert_precision_calibration_session",
]
