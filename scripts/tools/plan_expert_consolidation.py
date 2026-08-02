"""Build a reviewed expert-prototype plan from calibration evidence."""

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
from mirai.core.models.compressed_weights import (
    read_compressed_weights_packed_state_manifest,
)
from mirai.core.moe.calibration.prototypes import build_prototype_plans
from mirai.core.moe.calibration.prototypes import load_prototype_calibration_evidence
from mirai.core.moe.calibration.prototypes import select_hierarchical_output_merge
from mirai.core.training.calibration.expert_prototypes import (
    expected_prototype_calibration_lineage,
)


def _require_consolidation_gate(config: TrainingConfig) -> str:
    gate = str(
        getattr(config.model.params, "expert_consolidation", "off")
    ).strip().lower()
    if gate not in {"prototype", "hierarchical_output"}:
        raise ValueError(
            "scripts/tools/plan_expert_consolidation.py requires "
            "model.params.expert_consolidation='prototype' or "
            "'hierarchical_output' "
            f"(got {gate!r})."
        )
    return gate


def _validate_evidence_topology(
    config: TrainingConfig,
    evidence: dict[str, Any],
) -> None:
    manifest = read_compressed_weights_packed_state_manifest(
        config.memory.frozen_weight_packed_state_path
    )
    raw_modules = manifest.get("modules")
    if not isinstance(raw_modules, dict):
        raise ValueError("Packed-state manifest must contain a modules object.")
    grouped = {
        str(name): spec
        for name, spec in raw_modules.items()
        if isinstance(spec, dict) and str(spec.get("kind")) == "grouped_experts"
    }
    if set(evidence) != set(grouped):
        raise ValueError(
            "Calibration modules do not exactly match packed grouped-expert modules."
        )
    for module_name, module_evidence in evidence.items():
        expected_experts = int(grouped[module_name].get("num_experts", 0))
        if int(module_evidence.num_experts) != expected_experts:
            raise ValueError(
                f"Calibration module {module_name!r} has "
                f"{module_evidence.num_experts} experts; packed state expects "
                f"{expected_experts}."
            )


def plan_expert_consolidation(
    config: TrainingConfig,
    *,
    config_path: str | Path,
    evidence_file: str | Path,
    output: str | Path,
    reduction_ratio: float,
    strategy: str = "adaptive",
) -> dict[str, Any]:
    """Resolve registered policy and write the explicit JSON review boundary."""
    gate = _require_consolidation_gate(config)
    evidence_path = Path(evidence_file)
    output_path = Path(output)
    if output_path.resolve() == evidence_path.resolve():
        raise ValueError("Prototype plan output must differ from calibration evidence.")
    expected_lineage = expected_prototype_calibration_lineage(config, config_path)
    evidence, lineage = load_prototype_calibration_evidence(
        evidence_path,
        expected_dataset_snapshot_id=expected_lineage["dataset_snapshot_id"],
        expected_model_snapshot_id=expected_lineage["model_snapshot_id"],
        expected_config_snapshot_id=expected_lineage["config_snapshot_id"],
        expected_packed_artifact_fingerprint=expected_lineage[
            "packed_artifact_fingerprint"
        ],
    )
    _validate_evidence_topology(config, evidence)
    if gate == "hierarchical_output":
        if strategy not in {"auto", "average_linkage"}:
            raise ValueError(
                "hierarchical_output consolidation requires strategy="
                "'average_linkage' (or 'auto')."
            )
        hierarchical = {
            module_name: select_hierarchical_output_merge(
                module_evidence,
                float(reduction_ratio),
            )
            for module_name, module_evidence in evidence.items()
        }
        modules = {
            module_name: {
                "logical_to_prototype": list(
                    plan.consolidation.logical_to_prototype
                ),
                "merge_weights": list(plan.merge_weights),
            }
            for module_name, plan in hierarchical.items()
        }
        module_summary = {
            module_name: {
                "logical_experts": plan.consolidation.logical_experts,
                "physical_experts": plan.consolidation.physical_experts,
            }
            for module_name, plan in hierarchical.items()
        }
        resolved_strategy = "average_linkage"
    else:
        resolved_strategy = "adaptive" if strategy == "auto" else str(strategy)
        prototype = build_prototype_plans(
            evidence,
            reduction_ratio=float(reduction_ratio),
            strategy=resolved_strategy,
        )
        modules = {
            module_name: {
                "logical_to_prototype": list(plan.logical_to_prototype),
            }
            for module_name, plan in prototype.items()
        }
        module_summary = {
            module_name: {
                "logical_experts": plan.logical_experts,
                "physical_experts": plan.physical_experts,
            }
            for module_name, plan in prototype.items()
        }
    payload = {
        "format": "mirai.moe.expert_consolidation_plan",
        "schema_version": 2,
        "method": gate,
        "strategy": resolved_strategy,
        "reduction_ratio": float(reduction_ratio),
        "lineage": lineage,
        "modules": modules,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "output": str(output_path),
        "method": gate,
        "strategy": resolved_strategy,
        "reduction_ratio": float(reduction_ratio),
        "lineage": lineage,
        "modules": module_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evidence-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reduction-ratio", required=True, type=float)
    parser.add_argument("--strategy", default="auto")
    args = parser.parse_args()
    summary = plan_expert_consolidation(
        load_config(args.config),
        config_path=args.config,
        evidence_file=args.evidence_file,
        output=args.output,
        reduction_ratio=args.reduction_ratio,
        strategy=args.strategy,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
