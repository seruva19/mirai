"""Apply a reviewed expert-consolidation plan to a packed-state artifact.

The schema-v2 JSON plan carries method, topology, policy parameters, and exact
dataset/model/config/source-artifact lineage. The input artifact is never
modified. One-file and sharded outputs use Mirai's normal bounded-size writer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.loader import load_config
from mirai.config.schema import TrainingConfig
from mirai.core.models.compressed_weights import (
    load_compressed_weights_packed_tensors,
    packed_artifact_fingerprint,
    read_compressed_weights_packed_state_manifest,
    save_compressed_weights_packed_tensors,
)
from mirai.core.moe.storage.consolidation import consolidate_packed_state
from mirai.core.training.calibration.expert_prototypes import (
    expected_prototype_calibration_lineage,
)


@dataclass(frozen=True)
class ConsolidationPlanArtifact:
    method: str
    strategy: str
    reduction_ratio: float
    lineage: dict[str, str]
    modules: dict[str, dict[str, Any]]


def _require_consolidation_gate(config: TrainingConfig) -> str:
    gate = str(
        getattr(config.model.params, "expert_consolidation", "off")
    ).strip().lower()
    if gate not in {"prototype", "hierarchical_output"}:
        raise ValueError(
            "scripts/tools/consolidate_experts.py requires "
            "model.params.expert_consolidation='prototype' or "
            "'hierarchical_output' "
            f"(got {gate!r})."
        )
    return gate


def _load_plan(path: Path, *, expected_method: str) -> ConsolidationPlanArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Consolidation plan must be a non-empty JSON object.")
    if payload.get("format") != "mirai.moe.expert_consolidation_plan":
        raise ValueError("Unsupported expert consolidation plan format.")
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError("Unsupported expert consolidation plan schema.")
    method = str(payload.get("method", ""))
    if method != str(expected_method):
        raise ValueError(
            f"Consolidation plan method {method!r} does not match config "
            f"{expected_method!r}."
        )
    strategy = str(payload.get("strategy", "")).strip()
    if not strategy:
        raise ValueError("Consolidation plan has no strategy.")
    reduction_ratio = float(payload.get("reduction_ratio", float("nan")))
    if (
        not math.isfinite(reduction_ratio)
        or reduction_ratio <= 0.0
        or reduction_ratio >= 1.0
    ):
        raise ValueError(
            "Consolidation plan reduction_ratio must be strictly between zero and one."
        )
    raw_lineage = payload.get("lineage")
    if not isinstance(raw_lineage, Mapping):
        raise ValueError("Consolidation plan has no lineage object.")
    lineage_keys = (
        "dataset_snapshot_id",
        "model_snapshot_id",
        "config_snapshot_id",
        "packed_artifact_fingerprint",
    )
    lineage = {key: str(raw_lineage.get(key, "")).strip() for key in lineage_keys}
    if not all(lineage.values()):
        raise ValueError("Consolidation plan lineage is incomplete.")
    modules = payload.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("Consolidation plan has no module entries.")
    plan: dict[str, dict[str, Any]] = {}
    for module_name, raw_entry in modules.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"Consolidation plan entry {module_name!r} must be an object."
            )
        raw_mapping = raw_entry.get("logical_to_prototype")
        if not isinstance(raw_mapping, list):
            raise ValueError(
                f"Consolidation plan entry {module_name!r} has no integer mapping."
            )
        entry: dict[str, Any] = {
            "logical_to_prototype": [int(value) for value in raw_mapping],
        }
        raw_weights = raw_entry.get("merge_weights")
        if method == "hierarchical_output":
            if not isinstance(raw_weights, list):
                raise ValueError(
                    f"Hierarchical plan entry {module_name!r} has no merge weights."
                )
            entry["merge_weights"] = [float(value) for value in raw_weights]
        elif raw_weights is not None:
            raise ValueError("Prototype consolidation plans cannot contain merge weights.")
        plan[str(module_name)] = entry
    return ConsolidationPlanArtifact(
        method=method,
        strategy=strategy,
        reduction_ratio=reduction_ratio,
        lineage=lineage,
        modules=plan,
    )


def _validate_plan_modules(
    manifest: Mapping[str, Any],
    plan_modules: Mapping[str, Any],
) -> None:
    raw_modules = manifest.get("modules")
    if not isinstance(raw_modules, Mapping):
        raise ValueError("Packed-state manifest must contain a modules object.")
    grouped_modules = {
        str(name)
        for name, spec in raw_modules.items()
        if isinstance(spec, Mapping) and str(spec.get("kind")) == "grouped_experts"
    }
    if set(plan_modules) != grouped_modules:
        raise ValueError(
            "Consolidation plan modules do not exactly match packed "
            "grouped-expert modules."
        )


def consolidate_packed_base(
    config: TrainingConfig,
    *,
    config_path: str | Path,
    packed_state: str | Path,
    plan_file: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Apply a reviewed consolidation plan and write a new packed artifact."""
    gate = _require_consolidation_gate(config)
    packed_path = Path(packed_state)
    manifest = read_compressed_weights_packed_state_manifest(packed_path)
    tensors = load_compressed_weights_packed_tensors(packed_path)
    plan = _load_plan(Path(plan_file), expected_method=gate)
    expected_lineage = expected_prototype_calibration_lineage(config, config_path)
    actual_source_fingerprint = packed_artifact_fingerprint(packed_path)
    if (
        expected_lineage["packed_artifact_fingerprint"]
        != actual_source_fingerprint
    ):
        raise ValueError(
            "--packed-state does not match the packed source configured by "
            "memory.frozen_weight_packed_state_path."
        )
    for key, expected_value in expected_lineage.items():
        observed_value = plan.lineage[key]
        if observed_value != expected_value:
            raise ValueError(
                f"Consolidation plan {key} mismatch: expected "
                f"{expected_value!r}, found {observed_value!r}."
            )
    _validate_plan_modules(manifest, plan.modules)
    new_tensors, new_manifest = consolidate_packed_state(
        tensors,
        manifest,
        plan.modules,
    )
    new_manifest["expert_consolidation"] = {
        "schema_version": 1,
        "method": plan.method,
        "strategy": plan.strategy,
        "reduction_ratio": plan.reduction_ratio,
        "lineage": plan.lineage,
    }

    out_path = Path(output)
    if out_path.resolve() == packed_path.resolve():
        raise ValueError("Prototype consolidation output must differ from its input.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written_path = save_compressed_weights_packed_tensors(
        out_path,
        new_tensors,
        new_manifest,
        metadata={
            "model_type": str(config.model.type),
            "model_variant": str(config.model.params.variant),
            "expert_consolidation": gate,
        },
    )
    modules = {
        name: {
            "logical_experts": int(spec["logical_num_experts"]),
            "physical_experts": int(spec["num_experts"]),
        }
        for name, spec in new_manifest["modules"].items()
        if isinstance(spec, dict) and "logical_to_physical" in spec
    }
    return {
        "status": "ok",
        "output": str(written_path),
        "model_type": str(config.model.type),
        "method": gate,
        "strategy": plan.strategy,
        "reduction_ratio": plan.reduction_ratio,
        "source_artifact_fingerprint": actual_source_fingerprint,
        "modules": modules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--packed-state", required=True)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = consolidate_packed_base(
        load_config(args.config),
        config_path=args.config,
        packed_state=args.packed_state,
        plan_file=args.plan_file,
        output=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
