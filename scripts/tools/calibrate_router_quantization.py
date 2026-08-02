"""Calibrate frozen-router INT8 scales with EAQuant routing alignment."""

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
from mirai.core.training.calibration.router_quantization import (
    run_router_quantization_calibration_session,
)
from mirai.core.training.lifecycle.training_session import create_training_session
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path


def _validate_calibration_config(config: TrainingConfig) -> None:
    if (
        str(config.model.params.router_quantization_calibration).strip().lower()
        != "eaquant"
    ):
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
            "Router quantization calibration requires floating-point source "
            "routers and memory.router_quantization='disabled'."
        )


def calibrate_router_quantization(
    config: TrainingConfig,
    *,
    config_path: str | Path,
    output: str | Path,
    calibration_steps: int,
    max_tokens_per_router: int,
    max_input_gib: float,
    relaxation: float,
    minimum_clipping_ratio: float,
    grid_size: int,
    coordinate_sweeps: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build an isolated leased session and emit a calibrated scale artifact."""
    _validate_calibration_config(config)
    validate_training_runtime_config(config)
    output_path = Path(output)
    if output_path.exists() and not bool(overwrite):
        raise ValueError(
            f"Router calibration output already exists: {output_path}. "
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
    with tempfile.TemporaryDirectory(
        prefix="mirai-router-quantization-calibration-"
    ) as temp_dir:
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
                report = run_router_quantization_calibration_session(
                    session,
                    output_path=output_path,
                    calibration_steps=int(calibration_steps),
                    max_tokens_per_router=int(max_tokens_per_router),
                    max_input_gib=float(max_input_gib),
                    relaxation=float(relaxation),
                    minimum_clipping_ratio=float(minimum_clipping_ratio),
                    grid_size=int(grid_size),
                    coordinate_sweeps=int(coordinate_sweeps),
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
    parser.add_argument("--max-tokens-per-router", type=int, default=512)
    parser.add_argument("--max-input-gib", type=float, default=1.0)
    parser.add_argument("--relaxation", type=float, default=0.0)
    parser.add_argument("--minimum-clipping-ratio", type=float, default=0.35)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--coordinate-sweeps", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    register_builtin_components()
    config, _notes = load_runtime_config(
        args.config,
        entrypoint="router-quantization-calibration",
    )
    report = calibrate_router_quantization(
        config,
        config_path=args.config,
        output=args.output,
        calibration_steps=args.steps,
        max_tokens_per_router=args.max_tokens_per_router,
        max_input_gib=args.max_input_gib,
        relaxation=args.relaxation,
        minimum_clipping_ratio=args.minimum_clipping_ratio,
        grid_size=args.grid_size,
        coordinate_sweeps=args.coordinate_sweeps,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
