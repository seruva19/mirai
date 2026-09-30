"""Capture bounded native paired-CFG evidence for DIET expert pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.inference.session import InferenceSession
from mirai.core.builtins import register_builtin_components
from mirai.core.training.calibration.diet import (
    load_diet_calibration_manifest,
    run_diet_calibration_session,
)
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, help="JSON/JSONL prompt and seed manifest")
    parser.add_argument("--output", required=True, help="Output safetensors evidence")
    parser.add_argument("--samples-per-layer", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Deterministic token-row sampling seed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    requests, manifest_id = load_diet_calibration_manifest(args.manifest)
    config_path = Path(args.config)
    config_id = hashlib.sha256(config_path.read_bytes()).hexdigest()
    register_builtin_components()
    config, _notes = load_runtime_config(str(config_path), entrypoint="diet-calibration")
    if (str(config.model.params.expert_pruning).strip().lower() != "prune"
            or str(config.model.params.expert_pruning_criterion).strip().lower() != "diet"):
        raise ValueError(
            "DIET calibration requires expert_pruning='prune' and "
            "expert_pruning_criterion='diet'."
        )
    packed_state = str(config.memory.frozen_weight_packed_state_path).strip()
    if not packed_state or not Path(packed_state).is_file():
        raise ValueError(
            "DIET calibration requires an existing "
            "memory.frozen_weight_packed_state_path."
        )
    output = Path(args.output)
    if output.resolve() == Path(packed_state).resolve():
        raise ValueError("DIET evidence output cannot overwrite its packed source state.")
    if output.exists() and not args.overwrite:
        raise ValueError(f"DIET calibration output already exists: {output}.")
    if args.samples_per_layer < len(requests) or args.seed < 0:
        raise ValueError(
            "DIET samples-per-layer must cover every manifest request and seed "
            "must be non-negative."
        )
    with acquire_gpu_lease(
        lock_path=str(resolve_lease_lock_path(ROOT)),
        timeout_seconds=float(os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0")),
    ):
        session = InferenceSession.from_config(str(config_path))
        try:
            report = run_diet_calibration_session(
                session, requests=requests, output_path=output,
                samples_per_layer=args.samples_per_layer,
                source_packed_state=packed_state,
                calibration_manifest_id=manifest_id, config_snapshot_id=config_id,
                overwrite=args.overwrite, seed=args.seed,
            )
        finally:
            session.close()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
