"""Calibrate only MoE routers after frozen expert compression.

The original model acts as teacher. Exact teacher forward inputs and final
predictions are streamed to temporary safetensors, then replayed through the
compressed student. The emitted artifact contains router weights only and is
bound to the compressed base fingerprint.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
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
from mirai.core.models.compressed_weights import packed_artifact_fingerprint
from mirai.core.moe.calibration.router_repair import (
    save_router_repair_artifact,
)
from mirai.core.training.calibration.router_repair import (
    capture_router_kd_examples,
    fit_router_kd_session,
)
from mirai.core.training.lifecycle.training_session import create_training_session
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _close_session(session: Any | None) -> None:
    if session is None:
        return
    try:
        session.close_callbacks()
    finally:
        session.trainer.pipeline.flush_runtime_offloads()


def _collect_device_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _teacher_config(config: TrainingConfig) -> TrainingConfig:
    teacher = deepcopy(config)
    teacher.memory.frozen_weight_packed_state_path = ""
    teacher.memory.frozen_weight_quantization = "none"
    teacher.memory.frozen_weight_quantization_strategy = "disabled"
    teacher.memory.quantize_experts_on_load = False
    teacher.model.params.expert_pruning = "off"
    teacher.model.params.expert_consolidation = "off"
    teacher.model.params.expert_weight_compression = "off"
    teacher.model.params.post_compression_router_repair = "off"
    teacher.model.params.router_repair_artifact_path = ""
    return teacher


def _student_config(config: TrainingConfig) -> TrainingConfig:
    student = deepcopy(config)
    student.model.params.router_repair_artifact_path = ""
    return student


def _require_gate(config: TrainingConfig) -> Path:
    mode = str(
        config.model.params.post_compression_router_repair
    ).strip().lower()
    if mode != "router_kd":
        raise ValueError(
            "repair_router_kd.py requires "
            "model.params.post_compression_router_repair='router_kd'."
        )
    packed = Path(str(config.memory.frozen_weight_packed_state_path))
    if not packed.is_file():
        raise ValueError(
            "Router-KD requires memory.frozen_weight_packed_state_path to "
            "select an existing compressed student artifact."
        )
    return packed


def repair_router_kd(
    config: TrainingConfig,
    *,
    config_path: str | Path,
    output: str | Path,
    steps: int,
    holdout_steps: int,
    learning_rate: float,
    gradient_accumulation: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    validate_training_runtime_config(config)
    enforce_dataset_compliance(
        dataset_path=config.dataset.path,
        compliance_enabled=config.compliance.enabled,
        usage_mode=config.dataset.usage_mode,
        require_provenance=config.compliance.require_provenance,
        require_rights_attestation=config.compliance.require_rights_attestation,
    )
    packed = _require_gate(config)
    output_path = Path(output)
    if output_path.resolve() == packed.resolve():
        raise ValueError("Router-KD output cannot overwrite the compressed base.")
    if output_path.exists() and not overwrite:
        raise ValueError(f"Router-KD output already exists: {output_path}.")
    train_steps = int(steps)
    heldout = int(holdout_steps)
    if train_steps < 0:
        raise ValueError("Router-KD steps cannot be negative.")
    if heldout <= 0:
        raise ValueError("Router-KD holdout_steps must be positive.")

    compressed_fingerprint = packed_artifact_fingerprint(packed)
    teacher_session = None
    student_session = None
    with tempfile.TemporaryDirectory(prefix="mirai-router-kd-") as temp_dir:
        temp_root = Path(temp_dir)
        with acquire_gpu_lease(
            lock_path=str(resolve_lease_lock_path(ROOT)),
            timeout_seconds=float(os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0")),
        ):
            try:
                teacher_cfg = _teacher_config(config)
                teacher_cfg.logging.output_dir = str(temp_root / "teacher-session")
                teacher_session = create_training_session(
                    config=teacher_cfg,
                    config_path=str(config_path),
                    runtime_policy_notes=[],
                    output_dir=teacher_cfg.logging.output_dir,
                )
                example_paths = capture_router_kd_examples(
                    teacher_session,
                    output_dir=temp_root / "examples",
                    example_count=train_steps + heldout,
                )
                teacher_model_snapshot_id = str(
                    teacher_session.manifest.model_snapshot_id
                )
            finally:
                _close_session(teacher_session)
                teacher_session = None
                _collect_device_memory()

            try:
                student_cfg = _student_config(config)
                student_cfg.logging.output_dir = str(temp_root / "student-session")
                student_session = create_training_session(
                    config=student_cfg,
                    config_path=str(config_path),
                    runtime_policy_notes=[],
                    output_dir=student_cfg.logging.output_dir,
                )
                report = fit_router_kd_session(
                    student_session,
                    example_paths=example_paths,
                    train_examples=train_steps,
                    learning_rate=float(learning_rate),
                    gradient_accumulation=int(gradient_accumulation),
                    compressed_artifact_fingerprint=compressed_fingerprint,
                    teacher_model_snapshot_id=teacher_model_snapshot_id,
                )
                save_router_repair_artifact(
                    output_path,
                    report.artifact,
                    overwrite=bool(overwrite),
                )
            finally:
                _close_session(student_session)
                student_session = None
                _collect_device_memory()

    return {
        "status": "ok",
        "output": str(output_path),
        "model_type": str(config.model.type),
        "compressed_artifact_fingerprint": compressed_fingerprint,
        **report.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--holdout-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    register_builtin_components()
    config, _notes = load_runtime_config(
        args.config,
        entrypoint="router-kd-repair",
    )
    summary = repair_router_kd(
        config,
        config_path=args.config,
        output=args.output,
        steps=args.steps,
        holdout_steps=args.holdout_steps,
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
