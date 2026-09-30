"""Select DIET survivors and compact a lineage-matched packed expert base.

DIET: https://arxiv.org/abs/2609.37829. Calibration is produced by
``calibrate_diet.py``. Original router rows and group membership are preserved.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.loader import load_config
from mirai.config.schema import TrainingConfig
from mirai.core.lineage import sha256_file


def prune_diet_packed_base(
    config: TrainingConfig,
    *,
    packed_state: str | Path,
    calibration_file: str | Path,
    output: str | Path,
    keep_fraction: float | None = None,
    layer_counts: Mapping[str, int] | None = None,
    min_keep: int = 0,
    seed: int = 0,
    random_restarts: int = 3,
    anneal_steps: int = 1000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run offline ODL selection without changing the source packed artifact."""
    gate = str(config.model.params.expert_pruning).strip().lower()
    criterion = str(config.model.params.expert_pruning_criterion).strip().lower()
    if gate != "prune" or criterion != "diet":
        raise ValueError("DIET requires expert_pruning='prune' and expert_pruning_criterion='diet'.")
    if (keep_fraction is None) == (layer_counts is None):
        raise ValueError("DIET requires exactly one of keep_fraction or layer_counts.")
    if min_keep < 0 or seed < 0 or random_restarts < 0 or anneal_steps < 0:
        raise ValueError("DIET selection controls must be non-negative.")
    source = Path(packed_state).resolve()
    target = Path(output).resolve()
    if source == target:
        raise ValueError("DIET output must differ from the source packed state.")
    if Path(calibration_file).resolve() == target:
        raise ValueError("DIET output must differ from its calibration evidence.")
    if target.exists() and not overwrite:
        raise FileExistsError("DIET output already exists; use --overwrite explicitly.")

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from mirai.core.models.compressed_weights import read_compressed_weights_packed_state_manifest
    from mirai.core.models.compressed_weights.packed.packed_state import (
        COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY,
    )
    from mirai.core.moe.calibration.diet import (
        DietODLConfig, load_diet_evidence, odl_cost, select_odl_survivors,
        uniform_layer_counts, validate_layer_counts,
    )
    from mirai.core.moe.calibration.pruning import compact_packed_state_for_diet

    evidence, lineage = load_diet_evidence(calibration_file)
    if lineage.packed_state_sha256 != sha256_file(source):
        raise ValueError("DIET calibration belongs to a different packed source checkpoint.")
    manifest = read_compressed_weights_packed_state_manifest(source)
    grouped = {
        name: spec for name, spec in manifest["modules"].items()
        if spec.get("kind") == "grouped_experts"
    }
    if set(grouped) != set(evidence):
        raise ValueError("DIET evidence must cover exactly the packed grouped-expert modules.")
    topologies = {name: item.topology for name, item in evidence.items()}
    for name, spec in grouped.items():
        if int(spec["num_experts"]) != topologies[name].num_experts:
            raise ValueError(f"DIET source expert topology mismatch for {name!r}.")
        if topologies[name].top_k != int(config.model.params.experts_per_token):
            raise ValueError(f"DIET config top-k differs from captured topology for {name!r}.")
    counts = (
        uniform_layer_counts({name: topology.num_experts for name, topology in topologies.items()}, keep_fraction)
        if keep_fraction is not None else dict(layer_counts or {})
    )
    counts = validate_layer_counts(counts, topologies)
    if any(count < max(min_keep, topologies[name].top_k) for name, count in counts.items()):
        raise ValueError("DIET layer count is below the requested expert floor.")
    kept: dict[str, tuple[int, ...]] = {}
    costs: dict[str, float] = {}
    for index, name in enumerate(sorted(evidence)):
        item = evidence[name]
        kept[name] = select_odl_survivors(
            item.signatures, counts[name], item.topology,
            config=DietODLConfig(seed=seed + index, random_restarts=random_restarts, anneal_steps=anneal_steps),
        )
        costs[name] = odl_cost(item.signatures, kept[name])
    tensors = load_file(str(source), device="cpu")
    transformed, new_manifest = compact_packed_state_for_diet(
        tensors, manifest, kept,
        topology_by_module={name: dataclasses.asdict(value) for name, value in topologies.items()},
    )
    new_manifest["expert_pruning_transform"] = {
        "format": "mirai.moe.expert_pruning_transform",
        "schema_version": 2,
        "criterion": "diet",
        "source_lineage": lineage.to_dict(),
        "calibration_sha256": sha256_file(calibration_file),
        "kept_experts": {name: list(ids) for name, ids in kept.items()},
        "odl_costs": costs,
        "selection": {"seed": seed, "random_restarts": random_restarts, "anneal_steps": anneal_steps},
    }
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
    metadata.update({
        "expert_pruning": "prune",
        "expert_pruning_criterion": "diet",
        COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY: json.dumps(new_manifest, sort_keys=True, separators=(",", ":")),
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".diet-", suffix=".safetensors", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file({key: value.contiguous() for key, value in transformed.items()}, str(temporary), metadata=metadata)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "ok", "criterion": "diet", "output": str(target),
        "retained_experts": counts, "odl_costs": costs,
        "source_lineage": lineage.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--packed-state", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--keep-fraction", type=float)
    budget.add_argument("--layer-counts", type=Path, help="JSON object: grouped module name to retained count.")
    parser.add_argument("--min-keep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-restarts", type=int, default=3)
    parser.add_argument("--anneal-steps", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    counts = None
    if args.layer_counts is not None:
        counts = json.loads(args.layer_counts.read_text(encoding="utf-8"))
        if not isinstance(counts, dict):
            raise ValueError("--layer-counts must contain a JSON object.")
    summary = prune_diet_packed_base(
        load_config(args.config), packed_state=args.packed_state,
        calibration_file=args.calibration, output=args.output,
        keep_fraction=args.keep_fraction, layer_counts=counts, min_keep=args.min_keep,
        seed=args.seed, random_restarts=args.random_restarts, anneal_steps=args.anneal_steps,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
