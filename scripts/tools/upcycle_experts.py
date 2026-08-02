"""Expand every packed physical expert with diversified Drop-Upcycling copies.

The source artifact is never modified.  The output binds the exact source
fingerprint, source model snapshot, and transform policy in its manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.loader import load_config
from mirai.config.schema import TrainingConfig
from mirai.core.lineage import snapshot_descriptor_for_path
from mirai.core.models.compressed_weights import (
    load_compressed_weights_packed_tensors,
    packed_artifact_fingerprint,
    read_compressed_weights_packed_state_manifest,
    save_compressed_weights_packed_tensors,
)
from mirai.core.moe.storage.upcycling import (
    DropUpcyclingSpec,
    drop_upcycle_packed_state,
)


def _upcycling_spec(config: TrainingConfig) -> DropUpcyclingSpec:
    params = config.model.params
    mode = str(getattr(params, "expert_upcycling", "off")).strip().lower()
    if mode != "drop":
        raise ValueError(
            "scripts/tools/upcycle_experts.py requires "
            "model.params.expert_upcycling='drop'."
        )
    return DropUpcyclingSpec(
        copies_per_expert=int(getattr(params, "expert_upcycling_copies", 0)),
        reinitialization_ratio=float(
            getattr(params, "expert_upcycling_reinit_ratio", 0.5)
        ),
        seed=int(getattr(params, "expert_upcycling_seed", 42)),
    ).validate()


def upcycle_packed_base(
    config: TrainingConfig,
    *,
    packed_state: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Apply the configured data-free transform and write a new artifact."""

    policy = _upcycling_spec(config)
    packed_path = Path(packed_state)
    configured_source_raw = str(
        config.memory.frozen_weight_packed_state_path
    ).strip()
    if not configured_source_raw:
        raise ValueError(
            "Drop-Upcycling requires memory.frozen_weight_packed_state_path."
        )
    configured_source = Path(configured_source_raw)
    source_fingerprint = packed_artifact_fingerprint(packed_path)
    if packed_artifact_fingerprint(configured_source) != source_fingerprint:
        raise ValueError(
            "--packed-state does not match memory.frozen_weight_packed_state_path."
        )
    output_path = Path(output)
    if output_path.resolve() == packed_path.resolve():
        raise ValueError("Drop-Upcycling output must differ from its input.")

    manifest = read_compressed_weights_packed_state_manifest(packed_path)
    tensors = load_compressed_weights_packed_tensors(packed_path)
    output_tensors, output_manifest = drop_upcycle_packed_state(
        tensors,
        manifest,
        policy,
    )
    output_manifest["expert_upcycling"]["lineage"] = {
        "source_artifact_fingerprint": source_fingerprint,
        "source_model_snapshot_id": snapshot_descriptor_for_path(
            config.model.path
        ).snapshot_id,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = save_compressed_weights_packed_tensors(
        output_path,
        output_tensors,
        output_manifest,
        metadata={
            "model_type": str(config.model.type),
            "model_variant": str(config.model.params.variant),
            "expert_upcycling": "drop",
        },
    )
    modules = {
        name: {
            "source_experts": int(module["source_num_experts"]),
            "expanded_experts": int(module["expanded_num_experts"]),
        }
        for name, module in output_manifest["expert_upcycling"]["modules"].items()
    }
    return {
        "status": "ok",
        "output": str(written),
        "source_artifact_fingerprint": source_fingerprint,
        "transform_fingerprint": policy.fingerprint(),
        "modules": modules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--packed-state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = upcycle_packed_base(
        load_config(args.config),
        packed_state=args.packed_state,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
