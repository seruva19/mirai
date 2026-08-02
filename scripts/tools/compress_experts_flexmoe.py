"""Export a physically ragged FlexMoE expert artifact from calibrated actions."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.lineage import sha256_file
from mirai.core.models.compressed_weights import (
    load_compressed_weights_packed_tensors,
    packed_artifact_fingerprint,
    read_compressed_weights_packed_state_manifest,
    save_compressed_weights_packed_tensors,
    transform_packed_state_flexmoe_nested,
)
from mirai.core.moe.calibration.flexmoe import (
    load_action_plans,
    load_ranking_evidence,
)
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")


def compress_expert_artifact(
    *,
    config_path: str | Path,
    packed_state: str | Path,
    ranking_evidence: str | Path,
    action_plan: str | Path,
    output: str | Path,
    device: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply verified Equation-3 permutations and Equation-11 actions."""

    config, _notes = load_runtime_config(
        config_path,
        entrypoint="flexmoe-compression",
    )
    if (
        str(config.model.params.expert_weight_compression).strip().lower()
        != "flexmoe_nested"
    ):
        raise ValueError(
            "FlexMoE compression requires "
            "model.params.expert_weight_compression='flexmoe_nested'."
        )
    source_path = Path(packed_state)
    ranking_path = Path(ranking_evidence)
    action_path = Path(action_plan)
    output_path = Path(output)
    if output_path.resolve() in {
        source_path.resolve(),
        ranking_path.resolve(),
        action_path.resolve(),
    }:
        raise ValueError("FlexMoE output must differ from every input artifact.")
    if output_path.exists() and not overwrite:
        raise ValueError(f"FlexMoE output already exists: {output_path}.")
    resolved_device = str(device).strip().lower()
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FlexMoE compression requested CUDA, but CUDA is unavailable.")
    if resolved_device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")

    source_fingerprint = packed_artifact_fingerprint(source_path)
    ranking, ranking_lineage = load_ranking_evidence(ranking_path)
    actions, action_lineage = load_action_plans(action_path)
    if ranking_lineage["model_snapshot_id"] != source_fingerprint:
        raise ValueError(
            "FlexMoE ranking evidence does not belong to the source packed artifact."
        )
    if action_lineage["model_snapshot_id"] != source_fingerprint:
        raise ValueError(
            "FlexMoE action plan does not belong to the source packed artifact."
        )
    ranking_fingerprint = "sha256:" + sha256_file(ranking_path)
    if action_lineage["ranking_snapshot_id"] != ranking_fingerprint:
        raise ValueError(
            "FlexMoE action plan does not belong to the supplied ranking artifact."
        )

    manifest = read_compressed_weights_packed_state_manifest(source_path)
    tensors = load_compressed_weights_packed_tensors(source_path)
    lease = (
        acquire_gpu_lease(
            lock_path=str(resolve_lease_lock_path(ROOT)),
            timeout_seconds=float(os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0")),
        )
        if resolved_device == "cuda"
        else nullcontext()
    )
    with lease:
        converted, converted_manifest, report = transform_packed_state_flexmoe_nested(
            tensors,
            manifest,
            ranking_by_module=ranking,
            actions_by_module=actions,
            ranking_lineage=ranking_lineage,
            action_lineage=action_lineage,
            device=torch.device(resolved_device),
        )
        written = save_compressed_weights_packed_tensors(
            output_path,
            converted,
            converted_manifest,
            metadata={
                "model_type": str(config.model.type),
                "model_variant": str(config.model.params.variant),
                "expert_weight_compression": "flexmoe_nested",
                "source_artifact_fingerprint": source_fingerprint,
                "ranking_artifact_fingerprint": ranking_fingerprint,
                "action_artifact_fingerprint": "sha256:" + sha256_file(action_path),
            },
        )
    report.update(
        {
            "status": "ok",
            "output": str(written),
            "device": resolved_device,
            "source_artifact_fingerprint": source_fingerprint,
            "ranking_artifact_fingerprint": ranking_fingerprint,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--packed-state", required=True)
    parser.add_argument("--ranking-evidence", required=True)
    parser.add_argument("--action-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = compress_expert_artifact(
        config_path=args.config,
        packed_state=args.packed_state,
        ranking_evidence=args.ranking_evidence,
        action_plan=args.action_plan,
        output=args.output,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
