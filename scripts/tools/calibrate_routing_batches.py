"""Build a clip/timestep routing profile for microbatch scheduling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from mirai.core.training.calibration.routing_batches import collect_routing_batch_profile
from mirai.core.training.lifecycle.training_session import create_training_session
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease, resolve_lease_lock_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", required=True, type=int)
    parser.add_argument("--cache-experts", required=True, type=int)
    parser.add_argument("--timestep-bins", default=8, type=int)
    parser.add_argument("--trials", default=1, type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Routing profile already exists: {args.output}")
    if min(args.max_samples, args.timestep_bins, args.trials) < 1 or args.cache_experts < 0:
        parser.error("Calibration counts must be positive and cache-experts must be nonnegative.")
    config, notes = load_runtime_config(args.config, entrypoint="routing-batch-calibration")
    config.training.routing_batch_profile = ""
    validate_training_runtime_config(config)
    root = Path(__file__).resolve().parents[2]
    with acquire_gpu_lease(
        lock_path=str(resolve_lease_lock_path(root)),
        timeout_seconds=float(os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0")),
    ), TemporaryDirectory(prefix="mirai-routing-calibration-") as workspace:
        config.logging.output_dir = workspace
        session = create_training_session(config=config, config_path=args.config, runtime_policy_notes=notes, output_dir=workspace)
        try:
            profile = collect_routing_batch_profile(
                session, max_samples=args.max_samples, timestep_bins=args.timestep_bins,
                trials=args.trials, cache_experts=args.cache_experts,
            )
        finally:
            try:
                session.close_callbacks()
            finally:
                session.trainer.pipeline.flush_runtime_offloads()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
