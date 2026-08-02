"""Provider-driven EAQuant calibration for frozen router INT8 storage."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.calibration.router_quantization import (
    RouterInputAccumulator,
)
from mirai.core.moe.calibration.router_quantization import (
    RouterQuantizationCalibrationArtifact,
)
from mirai.core.moe.calibration.router_quantization import (
    RouterQuantizationCalibrationTarget,
)
from mirai.core.moe.calibration.router_quantization import (
    calibrate_symmetric_int8_router,
)
from mirai.core.moe.calibration.router_quantization import (
    save_router_quantization_calibration,
)
from mirai.core.moe.calibration.router_quantization import source_router_tensors
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
class RouterQuantizationCalibrationRunReport:
    output_path: str
    calibration_steps: int
    max_tokens_per_router: int
    input_budget_bytes: int
    passes: int
    modules: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "output": self.output_path,
            "calibration_steps": self.calibration_steps,
            "max_tokens_per_router": self.max_tokens_per_router,
            "input_budget_bytes": self.input_budget_bytes,
            "passes": self.passes,
            "modules": self.modules,
        }


def _normalize_targets(
    raw_targets: dict[str, RouterQuantizationCalibrationTarget],
) -> dict[str, RouterQuantizationCalibrationTarget]:
    targets: dict[str, RouterQuantizationCalibrationTarget] = {}
    for raw_name, target in raw_targets.items():
        if not isinstance(target, RouterQuantizationCalibrationTarget):
            raise TypeError(
                "Router quantization calibration targets must use "
                "RouterQuantizationCalibrationTarget."
            )
        target.validate()
        name = str(raw_name)
        if name != target.name or name in targets:
            raise ValueError(
                "Router quantization calibration target names must match and be unique."
            )
        targets[name] = target
    if not targets:
        raise ValueError(
            "Model provider returned no router quantization calibration targets."
        )
    return dict(sorted(targets.items()))


def _target_groups(
    targets: dict[str, RouterQuantizationCalibrationTarget],
    *,
    max_tokens: int,
    budget_bytes: int,
) -> list[list[RouterQuantizationCalibrationTarget]]:
    groups: list[list[RouterQuantizationCalibrationTarget]] = []
    current: list[RouterQuantizationCalibrationTarget] = []
    current_bytes = 0
    for target in targets.values():
        required = int(target.estimated_input_bytes_per_token) * int(max_tokens)
        if required > int(budget_bytes):
            raise ValueError(
                f"Router calibration target {target.name!r} requires {required} "
                f"input bytes, exceeding the {int(budget_bytes)}-byte budget."
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


def run_router_quantization_calibration_session(
    session: Any,
    *,
    output_path: str | Path,
    calibration_steps: int,
    max_tokens_per_router: int,
    max_input_gib: float,
    relaxation: float = 0.0,
    minimum_clipping_ratio: float = 0.35,
    grid_size: int = 101,
    coordinate_sweeps: int = 1,
    overwrite: bool = False,
) -> RouterQuantizationCalibrationRunReport:
    """Collect bounded router inputs and emit calibrated per-channel scales."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("Router quantization calibration requires torch.")
    config = session.config
    gate = str(
        getattr(config.model.params, "router_quantization_calibration", "off")
    ).strip().lower()
    if gate != "eaquant":
        raise ValueError(
            "Router quantization calibration requires "
            "model.params.router_quantization_calibration='eaquant'."
        )
    if str(config.memory.router_quantization).strip().lower() not in {
        "",
        "disabled",
        "none",
        "off",
    }:
        raise ValueError(
            "Router calibration must load floating-point source routers; set "
            "memory.router_quantization='disabled'."
        )
    if str(
        getattr(config.memory, "router_quantization_calibration_path", "")
    ).strip():
        raise ValueError(
            "Router calibration source config cannot load an existing calibration."
        )
    steps = int(calibration_steps)
    tokens = int(max_tokens_per_router)
    budget_gib = float(max_input_gib)
    if steps <= 0 or tokens <= 0 or not (budget_gib > 0.0):
        raise ValueError(
            "Calibration steps, router token budget, and input budget must be positive."
        )
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"Router calibration output already exists: {output}.")

    provider = get_model_family_provider(str(config.model.type))
    if (
        provider is None
        or not provider.supports_router_quantization_calibration(config)
    ):
        raise ValueError(
            f"Model provider {config.model.type!r} does not support router "
            "quantization calibration."
        )
    targets = _normalize_targets(
        provider.build_router_quantization_calibration_targets(
            session.trainer.pipeline
        )
    )
    source_tensors = source_router_tensors(targets)
    source_fingerprint = router_tensor_fingerprint(source_tensors)
    budget_bytes = int(budget_gib * (1024**3))
    groups = _target_groups(
        targets,
        max_tokens=tokens,
        budget_bytes=budget_bytes,
    )
    per_observation = max(1, int(math.ceil(tokens / steps)))

    trainer = session.trainer
    rng = getattr(session, "rng", None)
    rng_state = rng.getstate() if rng is not None else None
    validation_state = trainer.begin_validation()
    sampling_context = resolve_step_sampling_context(session)
    build_batch = _build_training_batch_factory(
        session=session,
        sampling_context=sampling_context,
    )
    results = {}
    try:
        for group in groups:
            accumulators = {
                target.name: RouterInputAccumulator(
                    target,
                    max_tokens=tokens,
                    max_tokens_per_observation=per_observation,
                )
                for target in group
            }
            try:
                for accumulator in accumulators.values():
                    accumulator.attach()
                with torch.no_grad():
                    for step in range(steps):
                        trainer.compute_loss(build_batch(step), training=False)
            finally:
                for accumulator in reversed(tuple(accumulators.values())):
                    accumulator.close()
            for target in group:
                weight = torch.as_tensor(target.read_weight()).detach().float()
                with torch.no_grad():
                    results[target.name] = calibrate_symmetric_int8_router(
                        weight,
                        accumulators[target.name].batch(),
                        top_k=int(target.top_k),
                        relaxation=float(relaxation),
                        minimum_clipping_ratio=float(minimum_clipping_ratio),
                        grid_size=int(grid_size),
                        coordinate_sweeps=int(coordinate_sweeps),
                    )
            del accumulators
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if rng is not None and rng_state is not None:
            rng.setstate(rng_state)
        trainer.end_validation(validation_state)

    manifest = session.manifest
    topology = {
        name: {
            "num_experts": int(target.num_experts),
            "input_features": int(target.input_features),
            "top_k": int(target.top_k),
        }
        for name, target in targets.items()
    }
    artifact = RouterQuantizationCalibrationArtifact(
        modules=results,
        topology=topology,
        dataset_snapshot_id=str(manifest.dataset_snapshot_id),
        model_snapshot_id=str(manifest.model_snapshot_id),
        config_snapshot_id=str(manifest.config_snapshot_id),
        source_router_fingerprint=source_fingerprint,
        relaxation=float(relaxation),
        minimum_clipping_ratio=float(minimum_clipping_ratio),
        grid_size=int(grid_size),
        coordinate_sweeps=int(coordinate_sweeps),
    ).validate()
    save_router_quantization_calibration(output, artifact)
    module_reports = {
        name: {
            "num_experts": int(topology[name]["num_experts"]),
            "input_features": int(topology[name]["input_features"]),
            "top_k": int(topology[name]["top_k"]),
            "token_count": int(result.token_count),
            "baseline_objective": float(result.baseline_objective),
            "calibrated_objective": float(result.calibrated_objective),
            "objective_reduction": float(
                result.baseline_objective - result.calibrated_objective
            ),
        }
        for name, result in results.items()
    }
    return RouterQuantizationCalibrationRunReport(
        output_path=str(output),
        calibration_steps=steps,
        max_tokens_per_router=tokens,
        input_budget_bytes=budget_bytes,
        passes=len(groups),
        modules=module_reports,
    )


__all__ = [
    "RouterQuantizationCalibrationRunReport",
    "run_router_quantization_calibration_session",
]
