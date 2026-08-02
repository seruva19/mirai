"""Calibrate and write an optimizer-safe adaptive LoRA rank plan."""

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
from mirai.core.training.calibration.adaptive_rank import (
    run_adaptive_rank_calibration_session,
)
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path
from mirai.core.training.lifecycle.training_session import create_training_session


def _validate_config(config: TrainingConfig, *, output: str | Path) -> None:
    if str(config.adapter.lora_init).strip().lower() != "eva":
        raise ValueError("Adaptive rank calibration requires adapter.lora_init='eva'.")
    configured = str(config.adapter.adaptive_rank_plan_path).strip()
    if not configured:
        raise ValueError(
            "Set adapter.adaptive_rank_plan_path to the requested output before calibration."
        )
    if Path(configured).resolve() != Path(output).resolve():
        raise ValueError("Calibration output must equal adapter.adaptive_rank_plan_path.")


def calibrate_adaptive_lora_ranks(
    config: TrainingConfig,
    *,
    config_path: str | Path,
    output: str | Path,
    calibration_steps: int,
    rank_budget: int,
    minimum_rank: int,
    maximum_rank: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    _validate_config(config, output=output)
    validate_training_runtime_config(config)
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise ValueError(f"Adaptive rank plan already exists: {output_path}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enforce_dataset_compliance(
        dataset_path=config.dataset.path,
        compliance_enabled=config.compliance.enabled,
        usage_mode=config.dataset.usage_mode,
        require_provenance=config.compliance.require_provenance,
        require_rights_attestation=config.compliance.require_rights_attestation,
    )
    configured_plan_path = config.adapter.adaptive_rank_plan_path
    original_output_dir = str(config.logging.output_dir)
    session = None
    with tempfile.TemporaryDirectory(prefix="mirai-adaptive-rank-") as temp_dir:
        config.adapter.adaptive_rank_plan_path = ""
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
                report = run_adaptive_rank_calibration_session(
                    session,
                    output_path=output_path,
                    calibration_steps=int(calibration_steps),
                    rank_budget=int(rank_budget),
                    minimum_rank=int(minimum_rank),
                    maximum_rank=int(maximum_rank),
                    samples_per_target=int(config.adapter.eva_samples_per_target),
                    convergence_threshold=float(config.adapter.eva_convergence_threshold),
                ).to_dict()
        finally:
            config.adapter.adaptive_rank_plan_path = configured_plan_path
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
    parser.add_argument("--rank-budget", required=True, type=int)
    parser.add_argument("--minimum-rank", default=1, type=int)
    parser.add_argument("--maximum-rank", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    register_builtin_components()
    config, _notes = load_runtime_config(args.config, entrypoint="adaptive-rank-calibration")
    summary = calibrate_adaptive_lora_ranks(
        config,
        config_path=args.config,
        output=args.output,
        calibration_steps=args.steps,
        rank_budget=args.rank_budget,
        minimum_rank=args.minimum_rank,
        maximum_rank=args.maximum_rank,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
