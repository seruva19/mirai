"""Calibrate FlexMoE channel rankings or discrete retention actions."""

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
from mirai.core.training.calibration.flexmoe import (
    FlexMoEActionLearningSpec,
    run_flexmoe_action_session,
    run_flexmoe_ranking_session,
)
from mirai.core.training.lifecycle.training_session import create_training_session
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path


def calibrate_flexmoe(
    config: TrainingConfig,
    *,
    config_path: str | Path,
    stage: str,
    output: str | Path,
    steps: int,
    ranking: str | Path | None = None,
    action_ratios: tuple[float, ...] | None = None,
    learning_rate: float | None = None,
    thickest_logit_margin: float | None = None,
    temperature_start: float | None = None,
    temperature_end: float | None = None,
    cost_weight_start: float | None = None,
    cost_weight_end: float | None = None,
    entropy_weight_start: float | None = None,
    entropy_weight_end: float | None = None,
    teacher_loss_weight: float | None = None,
    seed: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build an isolated training session for one FlexMoE calibration stage."""

    mode = str(stage).strip().lower()
    if mode not in {"ranking", "actions"}:
        raise ValueError("FlexMoE stage must be 'ranking' or 'actions'.")
    if mode == "actions":
        required = {
            "ranking": ranking,
            "action_ratios": action_ratios,
            "learning_rate": learning_rate,
            "thickest_logit_margin": thickest_logit_margin,
            "temperature_start": temperature_start,
            "temperature_end": temperature_end,
            "cost_weight_start": cost_weight_start,
            "cost_weight_end": cost_weight_end,
            "entropy_weight_start": entropy_weight_start,
            "entropy_weight_end": entropy_weight_end,
            "teacher_loss_weight": teacher_loss_weight,
            "seed": seed,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "FlexMoE action learning requires explicit " + ", ".join(missing) + "."
            )
    validate_training_runtime_config(config)
    enforce_dataset_compliance(
        dataset_path=config.dataset.path,
        compliance_enabled=config.compliance.enabled,
        usage_mode=config.dataset.usage_mode,
        require_provenance=config.compliance.require_provenance,
        require_rights_attestation=config.compliance.require_rights_attestation,
    )
    original_output_dir = str(config.logging.output_dir)
    session = None
    with tempfile.TemporaryDirectory(prefix="mirai-flexmoe-") as temp_dir:
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
                if mode == "ranking":
                    report = run_flexmoe_ranking_session(
                        session,
                        output_path=output,
                        calibration_steps=int(steps),
                        overwrite=bool(overwrite),
                    )
                else:
                    report = run_flexmoe_action_session(
                        session,
                        ranking_path=Path(ranking),
                        output_path=output,
                        action_ratios=tuple(action_ratios),
                        spec=FlexMoEActionLearningSpec(
                            total_steps=int(steps),
                            temperature_start=float(temperature_start),
                            temperature_end=float(temperature_end),
                            cost_weight_start=float(cost_weight_start),
                            cost_weight_end=float(cost_weight_end),
                            entropy_weight_start=float(entropy_weight_start),
                            entropy_weight_end=float(entropy_weight_end),
                        ),
                        learning_rate=float(learning_rate),
                        thickest_logit_margin=float(thickest_logit_margin),
                        teacher_loss_weight=float(teacher_loss_weight),
                        seed=int(seed),
                        overwrite=bool(overwrite),
                    )
                return report.to_dict()
        finally:
            config.logging.output_dir = original_output_dir
            if session is not None:
                try:
                    session.close_callbacks()
                finally:
                    session.trainer.pipeline.flush_runtime_offloads()


def _parse_ratios(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "action ratios must be comma-separated numbers"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("ranking", "actions"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--ranking")
    parser.add_argument("--action-ratios", type=_parse_ratios)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--thickest-logit-margin", type=float)
    parser.add_argument("--temperature-start", type=float)
    parser.add_argument("--temperature-end", type=float)
    parser.add_argument("--cost-weight-start", type=float)
    parser.add_argument("--cost-weight-end", type=float)
    parser.add_argument("--entropy-weight-start", type=float)
    parser.add_argument("--entropy-weight-end", type=float)
    parser.add_argument("--teacher-loss-weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "actions":
        required = {
            "--ranking": args.ranking,
            "--action-ratios": args.action_ratios,
            "--learning-rate": args.learning_rate,
            "--thickest-logit-margin": args.thickest_logit_margin,
            "--temperature-start": args.temperature_start,
            "--temperature-end": args.temperature_end,
        }
        required.update(
            {
                "--cost-weight-start": args.cost_weight_start,
                "--cost-weight-end": args.cost_weight_end,
                "--entropy-weight-start": args.entropy_weight_start,
                "--entropy-weight-end": args.entropy_weight_end,
                "--teacher-loss-weight": args.teacher_loss_weight,
                "--seed": args.seed,
            }
        )
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("actions stage requires " + ", ".join(missing))
    register_builtin_components()
    config, _notes = load_runtime_config(
        args.config,
        entrypoint="flexmoe-calibration",
    )
    report = calibrate_flexmoe(
        config,
        config_path=args.config,
        stage=args.stage,
        output=args.output,
        steps=args.steps,
        ranking=args.ranking,
        action_ratios=args.action_ratios,
        learning_rate=args.learning_rate,
        thickest_logit_margin=args.thickest_logit_margin,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        cost_weight_start=args.cost_weight_start,
        cost_weight_end=args.cost_weight_end,
        entropy_weight_start=args.entropy_weight_start,
        entropy_weight_end=args.entropy_weight_end,
        teacher_loss_weight=args.teacher_loss_weight,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
