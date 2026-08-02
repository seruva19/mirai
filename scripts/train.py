"""Minimal trainer entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Long video buckets create variable-size routed buffers; expandable segments
# let the CUDA allocator reuse adjacent free regions instead of stranding them.
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.builtins import register_builtin_components
from mirai.core.dataset.registration import enforce_dataset_compliance
from mirai.core.training.runtime.cli import (
    emit_runtime_policy_notes,
    load_runtime_config,
)
from mirai.core.training.runtime.dry_run import build_dry_run_report
from mirai.core.training.runtime.gpu_lease import (
    GpuLeaseError,
    acquire_gpu_lease,
    resolve_lease_lock_path,
)
from mirai.core.training.data.batches import sample_batch as _core_sample_batch
from mirai.core.training.residency.memory_budget import pinned_memory_budget_warning
from mirai.core.training.residency.memory import track_resource_peaks
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.core.training.lifecycle.training_loop import run_training_loop
from mirai.core.training.lifecycle.training_session import create_training_session
from mirai.core.training.trainer import Trainer


try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")


def _sample_batch(*args, **kwargs):
    """Delegate batch construction to the core training implementation."""

    return _core_sample_batch(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--dry-run", action="store_true", help="Run one synthetic step")
    parser.add_argument("--resume", default="", help="Resume from checkpoint path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    register_builtin_components()
    config, runtime_policy_notes = load_runtime_config(
        args.config,
        entrypoint="train",
    )
    try:
        validate_training_runtime_config(config)
    except ValueError as exc:
        raise SystemExit(str(exc))
    emit_runtime_policy_notes(runtime_policy_notes)
    try:
        enforce_dataset_compliance(
            dataset_path=config.dataset.path,
            compliance_enabled=config.compliance.enabled,
            usage_mode=config.dataset.usage_mode,
            require_provenance=config.compliance.require_provenance,
            require_rights_attestation=config.compliance.require_rights_attestation,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    budget_warning = pinned_memory_budget_warning(
        blocks_to_swap=config.training.blocks_to_swap,
        num_workers=config.dataset.num_workers,
        prefetch_factor=config.dataset.prefetch_factor,
        batch_size=config.training.batch_size,
        budget_fraction=config.training.pinned_memory_budget_fraction,
    )
    if budget_warning:
        print(f"[warning] {budget_warning}", file=sys.stderr)
    if args.dry_run:
        trainer = Trainer(config)
        result = build_dry_run_report(
            trainer=trainer,
            config=config,
            runtime_policy_notes=list(runtime_policy_notes),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = str(resolve_lease_lock_path(ROOT))
    lease_timeout = float(os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0"))
    try:
        with acquire_gpu_lease(lock_path=lock_path, timeout_seconds=lease_timeout):
            with track_resource_peaks():
                session = create_training_session(
                    config=config,
                    config_path=str(args.config),
                    runtime_policy_notes=list(runtime_policy_notes),
                    output_dir=output_dir,
                    resume_path=str(args.resume or ""),
                )
                final_artifacts = run_training_loop(session)
                print(json.dumps(final_artifacts.summary, indent=2, sort_keys=True))
                return 0
    except GpuLeaseError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
