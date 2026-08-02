"""Collect provider-owned evidence for offline expert consolidation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.schema import TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.dataset.registration import enforce_dataset_compliance
from mirai.core.moe.runtime.specs import normalize_expert_weight_access_policy
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path
from mirai.core.training.calibration.expert_prototypes import (
    run_prototype_calibration_session,
)
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.core.training.lifecycle.training_session import create_training_session

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")


def _validate_calibration_config(config: TrainingConfig) -> None:
    gate = str(config.model.params.expert_consolidation).strip().lower()
    if gate not in {"prototype", "hierarchical_output"}:
        raise ValueError(
            "Expert consolidation calibration requires model.params."
            "expert_consolidation='prototype' or 'hierarchical_output'."
        )
    access = normalize_expert_weight_access_policy(config.memory.expert_weight_access)
    if access != "active_dequant" or int(config.memory.expert_dequant_chunk_size) > 1:
        raise ValueError(
            "Expert consolidation calibration requires memory.expert_weight_access="
            "'active_dequant' and expert_dequant_chunk_size <= 1 so per-route "
            "expert outputs remain observable."
        )
    if not str(config.memory.frozen_weight_packed_state_path).strip():
        raise ValueError(
            "Expert consolidation calibration requires "
            "memory.frozen_weight_packed_state_path so evidence is bound to the "
            "exact packed source."
        )


def calibrate_expert_consolidation(
    config: TrainingConfig,
    *,
    config_path: str | Path,
    output: str | Path,
    calibration_steps: int,
    projection_block_mib: float,
    max_output_tokens_per_observation: int = 256,
    projection_device: str = "model",
    distance_dtype: str = "float64",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build an isolated training session and persist calibration evidence."""
    _validate_calibration_config(config)
    validate_training_runtime_config(config)
    block_mib = float(projection_block_mib)
    if block_mib <= 0.0:
        raise ValueError("projection_block_mib must be positive.")
    dtype_name = str(distance_dtype).strip().lower()
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("distance_dtype must be 'float32' or 'float64'.")
    requested_device = str(projection_device).strip().lower()
    if requested_device not in {"model", "cpu"}:
        raise ValueError("projection_device must be 'model' or 'cpu'.")
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    element_size = torch.empty((), dtype=dtype).element_size()
    max_block_elements = max(
        1,
        int((block_mib * 1024.0 * 1024.0) // element_size),
    )
    output_path = Path(output)
    if output_path.exists() and not bool(overwrite):
        raise ValueError(
            f"Prototype calibration output already exists: {output_path}. "
            "Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enforce_dataset_compliance(
        dataset_path=config.dataset.path,
        compliance_enabled=config.compliance.enabled,
        usage_mode=config.dataset.usage_mode,
        require_provenance=config.compliance.require_provenance,
        require_rights_attestation=config.compliance.require_rights_attestation,
    )
    original_output_dir = str(config.logging.output_dir)
    session = None
    with tempfile.TemporaryDirectory(prefix="mirai-prototype-calibration-") as temp_dir:
        config.logging.output_dir = str(Path(temp_dir) / "session")
        try:
            with acquire_gpu_lease(
                lock_path=str(resolve_lease_lock_path(ROOT)),
                timeout_seconds=float(os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0")),
            ):
                session = create_training_session(
                    config=config,
                    config_path=str(config_path),
                    runtime_policy_notes=[],
                    output_dir=config.logging.output_dir,
                )
                resolved_device = (
                    session.compute_device if requested_device == "model" else "cpu"
                )
                report = run_prototype_calibration_session(
                    session,
                    output_path=output_path,
                    calibration_steps=int(calibration_steps),
                    max_block_elements=max_block_elements,
                    max_output_tokens_per_observation=int(
                        max_output_tokens_per_observation
                    ),
                    projection_device=resolved_device,
                    distance_dtype=dtype,
                    overwrite=bool(overwrite),
                ).to_dict()
        finally:
            config.logging.output_dir = original_output_dir
            if session is not None:
                try:
                    session.close_callbacks()
                finally:
                    session.trainer.pipeline.flush_runtime_offloads()
    report.update(
        {
            "projection_block_mib": block_mib,
            "projection_device": requested_device,
            "distance_dtype": dtype_name,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--projection-block-mib", default=256.0, type=float)
    parser.add_argument(
        "--max-output-tokens-per-observation",
        default=256,
        type=int,
    )
    parser.add_argument("--projection-device", choices=("model", "cpu"), default="model")
    parser.add_argument("--distance-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    register_builtin_components()
    config, _notes = load_runtime_config(
        args.config,
        entrypoint="prototype-calibration",
    )
    summary = calibrate_expert_consolidation(
        config,
        config_path=args.config,
        output=args.output,
        calibration_steps=args.steps,
        projection_block_mib=args.projection_block_mib,
        max_output_tokens_per_observation=args.max_output_tokens_per_observation,
        projection_device=args.projection_device,
        distance_dtype=args.distance_dtype,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
