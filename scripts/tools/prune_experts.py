"""Offline structured expert pruning entrypoint.

Ranks experts from either safe calibration evidence or AIMER's calibration-free
weight statistic, removes the selected tail from a frozen compressed_weights
packed base, and re-emits a smaller packed state. This is not in the hot
training path and never mutates the input base.

Use ``calibrate_expert_pruning.py`` first for frequency, REAP, MAN, or MSAN.
AIMER reads only the packed expert weights and must not receive calibration
evidence. Both paths retain a hard per-layer ``experts_per_token`` floor, slice
expert tensors and router rows, and carry source lineage into output metadata.

Requires ``model.params.expert_pruning = "prune"``.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.loader import load_config
from mirai.config.schema import TrainingConfig
from mirai.core.lineage import sha256_file
from mirai.core.moe.calibration.pruning import (
    aimer_expert_scores,
    load_expert_pruning_evidence,
    normalize_expert_pruning_criterion,
    prune_packed_state,
    select_pruned_experts,
)
from mirai.core.models.compressed_weights.artifact_source import (
    load_grouped_expert_source,
)
from mirai.core.models.compressed_weights.factorization.prototype_projection import (
    CompressedExpertProjectionSource,
)
from mirai.core.models.compressed_weights import read_compressed_weights_packed_state_manifest
from mirai.core.models.compressed_weights.packed.packed_state import (
    COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY,
)
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path


def _parent(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else ""


def plan_keep_by_module(
    saliency: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    top_k: int,
    keep_fraction: float | None = None,
    score_threshold: float | None = None,
    min_keep: int = 0,
    keep_largest: bool = True,
) -> dict[str, tuple[int, ...]]:
    """Map each grouped-experts module to kept ids from calibration scores.

    Artifacts key scores by the exact grouped-experts module. The pure
    router-frequency accumulator may supply a parent-module key.
    """
    keep_by_source = select_pruned_experts(
        saliency,
        keep_fraction=keep_fraction,
        score_threshold=score_threshold,
        min_keep=min_keep,
        top_k=top_k,
        keep_largest=keep_largest,
    )
    keep_by_parent = {_parent(name): keep for name, keep in keep_by_source.items()}
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping):
        raise ValueError("packed-state manifest has no modules object.")
    keep_by_module: dict[str, tuple[int, ...]] = {}
    for module_name, spec in modules.items():
        if not isinstance(spec, Mapping) or str(spec.get("kind")) != "grouped_experts":
            continue
        keep = keep_by_source.get(str(module_name))
        if keep is None:
            keep = keep_by_parent.get(_parent(str(module_name)))
        if keep is None:
            raise ValueError(
                f"No calibration score found for grouped-experts module "
                f"{module_name!r}."
            )
        keep_by_module[str(module_name)] = keep
    if not keep_by_module:
        raise ValueError("packed-state manifest has no grouped_experts modules to prune.")
    return keep_by_module


def _require_prune_gate(config: TrainingConfig) -> None:
    gate = str(getattr(config.model.params, "expert_pruning", "off")).strip().lower()
    if gate != "prune":
        raise ValueError(
            "scripts/tools/prune_experts.py requires model.params.expert_pruning='prune' "
            f"(got {gate!r}); pruning is an explicit, opt-in offline step."
        )


def _load_packed_tensors(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return dict(load_file(str(path)))


def _aimer_saliency(
    tensors: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    max_block_elements: int,
    metric_device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping):
        raise ValueError("packed-state manifest has no modules object.")
    scores: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for module_name, spec in modules.items():
        if not isinstance(spec, Mapping) or str(spec.get("kind")) != "grouped_experts":
            continue
        if "logical_to_physical" in spec or int(
            spec.get("logical_num_experts", spec.get("num_experts", 0))
        ) != int(spec.get("num_experts", 0)):
            raise ValueError(
                "AIMER pruning requires an unconsolidated physical expert pool."
            )
        source = CompressedExpertProjectionSource(
            load_grouped_expert_source(spec, tensors)
        )
        result = aimer_expert_scores(
            source,
            max_block_elements=int(max_block_elements),
            device=metric_device,
        )
        scores[str(module_name)] = result.scores
        summaries[str(module_name)] = {
            "elements_per_expert": int(result.elements_per_expert),
            "scores": [float(value) for value in result.scores.tolist()],
        }
    if not scores:
        raise ValueError("packed-state manifest has no grouped_experts modules to score.")
    return scores, summaries


def prune_packed_base(
    config: TrainingConfig,
    *,
    packed_state: str | Path,
    calibration_file: str | Path | None,
    output: str | Path,
    keep_fraction: float | None = None,
    score_threshold: float | None = None,
    min_keep: int = 0,
    max_block_elements: int = 4_194_304,
    metric_device: str = "cpu",
) -> dict[str, Any]:
    """Select + structurally prune + re-emit a smaller packed base artifact."""
    from safetensors.torch import save_file

    _require_prune_gate(config)
    top_k = int(config.model.params.experts_per_token)

    packed_path = Path(packed_state)
    manifest = read_compressed_weights_packed_state_manifest(packed_path)
    tensors = _load_packed_tensors(packed_path)
    criterion = normalize_expert_pruning_criterion(
        config.model.params.expert_pruning_criterion
    )
    packed_sha256 = sha256_file(packed_path)
    score_summaries: dict[str, Any] = {}
    if criterion == "aimer":
        if calibration_file is not None:
            raise ValueError(
                "AIMER is calibration-free; do not pass --calibration."
            )
        resolved_device = str(metric_device).strip().lower()
        if resolved_device not in {"cpu", "cuda"}:
            raise ValueError("AIMER metric_device must be 'cpu' or 'cuda'.")
        lease = nullcontext()
        if resolved_device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("AIMER metric_device='cuda' requires CUDA.")
            lease = acquire_gpu_lease(
                lock_path=resolve_lease_lock_path(ROOT),
                timeout_seconds=float(
                    os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0")
                ),
            )
        with lease:
            saliency, score_summaries = _aimer_saliency(
                tensors,
                manifest,
                max_block_elements=int(max_block_elements),
                metric_device=resolved_device,
            )
        source_lineage = {
            "packed_state_sha256": packed_sha256,
            "scoring": "weight_only",
        }
        keep_largest = False
    else:
        if calibration_file is None:
            raise ValueError(
                f"Criterion {criterion!r} requires --calibration evidence."
            )
        evidence, calibration_lineage = load_expert_pruning_evidence(
            calibration_file,
            expected_criterion=criterion,
        )
        saliency = {name: item.scores() for name, item in evidence.items()}
        source_lineage = {
            "packed_state_sha256": packed_sha256,
            "calibration": calibration_lineage,
        }
        keep_largest = True

    keep_by_module = plan_keep_by_module(
        saliency,
        manifest,
        top_k=top_k,
        keep_fraction=keep_fraction,
        score_threshold=score_threshold,
        min_keep=min_keep,
        keep_largest=keep_largest,
    )
    new_tensors, new_manifest = prune_packed_state(tensors, manifest, keep_by_module)
    new_manifest["expert_pruning_transform"] = {
        "format": "mirai.moe.expert_pruning_transform",
        "schema_version": 1,
        "criterion": criterion,
        "score_direction": (
            "larger_is_more_removable"
            if criterion == "aimer"
            else "larger_is_more_important"
        ),
        "source_lineage": source_lineage,
        "kept_experts": {
            name: list(indices) for name, indices in keep_by_module.items()
        },
        "score_summaries": score_summaries,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json = json.dumps(new_manifest, sort_keys=True, separators=(",", ":"))
    metadata = {
        "model_type": str(config.model.type),
        "model_variant": str(config.model.params.variant),
        "expert_pruning": "prune",
        "expert_pruning_criterion": criterion,
        "expert_pruning_source_lineage": json.dumps(
            source_lineage,
            sort_keys=True,
            separators=(",", ":"),
        ),
        COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY: manifest_json,
    }
    if criterion != "aimer":
        metadata["expert_pruning_calibration_lineage"] = json.dumps(
            source_lineage["calibration"],
            sort_keys=True,
            separators=(",", ":"),
        )
    save_file(
        {k: v.contiguous() for k, v in new_tensors.items()},
        str(out_path),
        metadata=metadata,
    )
    new_num_experts = {
        name: int(spec.get("num_experts"))
        for name, spec in new_manifest["modules"].items()
        if isinstance(spec, dict) and str(spec.get("kind")) == "grouped_experts"
    }
    return {
        "status": "ok",
        "output": str(out_path),
        "model_type": str(config.model.type),
        "pruned_modules": sorted(keep_by_module),
        "new_num_experts": new_num_experts,
        "experts_per_token": top_k,
        "criterion": criterion,
        "source_lineage": source_lineage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--packed-state", required=True, help="Input frozen compressed_weights packed base."
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help=(
            "Safetensors evidence from calibrate_expert_pruning.py. Required "
            "for frequency/REAP/MAN/MSAN and forbidden for AIMER."
        ),
    )
    parser.add_argument("--output", required=True, help="Output pruned packed artifact.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keep-fraction", type=float, default=None)
    group.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--min-keep", type=int, default=0)
    parser.add_argument("--max-block-elements", type=int, default=4_194_304)
    parser.add_argument(
        "--metric-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Dense AIMER scoring device; CUDA obtains Mirai's GPU lease.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summary = prune_packed_base(
        config,
        packed_state=args.packed_state,
        calibration_file=args.calibration,
        output=args.output,
        keep_fraction=args.keep_fraction,
        score_threshold=args.score_threshold,
        min_keep=args.min_keep,
        max_block_elements=args.max_block_elements,
        metric_device=args.metric_device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
