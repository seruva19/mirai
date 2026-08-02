"""Collect activation covariances for whitened shared-basis factorization."""

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
from mirai.core.training.calibration.expert_whitening import (
    run_expert_whitening_calibration_session,
)
from mirai.core.training.lifecycle.training_session import create_training_session
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path


def _validate_calibration_config(config: TrainingConfig) -> None:
    if (
        str(config.model.params.expert_factorization_calibration).strip().lower()
        != "whitened"
    ):
        raise ValueError(
            "Expert whitening calibration requires "
            "model.params.expert_factorization_calibration='whitened'."
        )
    if str(config.model.params.expert_weight_compression).strip().lower() != "off":
        raise ValueError(
            "Expert whitening calibration reads the pre-factorization packed state "
            "and requires model.params.expert_weight_compression='off'."
        )


def calibrate_expert_whitening(
    config: TrainingConfig,
    *,
    config_path: str | Path,
    output: str | Path,
    calibration_steps: int,
    max_covariance_gib: float,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build an isolated leased session and persist covariance evidence."""
    _validate_calibration_config(config)
    validate_training_runtime_config(config)
    steps = int(calibration_steps)
    if steps <= 0:
        raise ValueError("Calibration steps must be positive.")
    output_path = Path(output)
    if output_path.exists() and not bool(overwrite):
        raise ValueError(
            f"Expert whitening output already exists: {output_path}. "
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
    with tempfile.TemporaryDirectory(prefix="mirai-expert-whitening-") as temp_dir:
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
                report = run_expert_whitening_calibration_session(
                    session,
                    output_path=output_path,
                    calibration_steps=steps,
                    max_covariance_gib=float(max_covariance_gib),
                    overwrite=bool(overwrite),
                ).to_dict()
        finally:
            config.logging.output_dir = original_output_dir
            if session is not None:
                try:
                    session.close_callbacks()
                finally:
                    session.trainer.pipeline.flush_runtime_offloads()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--max-covariance-gib", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    register_builtin_components()
    config, _notes = load_runtime_config(
        args.config,
        entrypoint="expert-whitening-calibration",
    )
    report = calibrate_expert_whitening(
        config,
        config_path=args.config,
        output=args.output,
        calibration_steps=args.steps,
        max_covariance_gib=args.max_covariance_gib,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
