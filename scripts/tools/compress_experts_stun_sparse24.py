"""Create a structured-then-semi-structured packed expert artifact."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.models.compressed_weights import (
    load_compressed_weights_packed_tensors,
    read_compressed_weights_packed_state_manifest,
    save_compressed_weights_packed_tensors,
)
from mirai.core.models.compressed_weights.quantization.stun_artifact import (
    transform_packed_state_stun_sparse24,
)
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")


def _target_experts(
    manifest: dict[str, Any],
    *,
    target_experts: int | None,
    keep_fraction: float | None,
    top_k: int,
) -> dict[str, int]:
    modules = manifest.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("Packed-state manifest must contain a modules object.")
    targets: dict[str, int] = {}
    for name, spec in modules.items():
        if not isinstance(spec, dict) or str(spec.get("kind")) != "grouped_experts":
            continue
        source = int(spec.get("num_experts", 0))
        target = (
            int(target_experts)
            if target_experts is not None
            else int(math.ceil(source * float(keep_fraction)))
        )
        if target < int(top_k):
            raise ValueError(
                f"STUN target for {name!r} is {target}, below experts_per_token={top_k}."
            )
        if target >= source:
            raise ValueError(
                f"STUN target for {name!r} must remove at least one expert."
            )
        targets[str(name)] = target
    if not targets:
        raise ValueError("Packed artifact contains no grouped experts.")
    return targets


def compress_expert_artifact(
    *,
    config_path: str | Path,
    packed_state: str | Path,
    output: str | Path,
    target_experts: int | None = None,
    keep_fraction: float | None = None,
    reconstruct_below: int = 3,
    quant_group_size: int = 32,
    device: str = "auto",
    max_projection_error: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run STUN's structured stage and Mirai's compact 2:4 adaptation."""

    if (target_experts is None) == (keep_fraction is None):
        raise ValueError("Specify exactly one of target_experts or keep_fraction.")
    if keep_fraction is not None and not 0.0 < float(keep_fraction) < 1.0:
        raise ValueError("keep_fraction must be in (0, 1).")
    config, _notes = load_runtime_config(
        config_path,
        entrypoint="stun-sparse24-compression",
    )
    if str(config.model.params.expert_pruning).strip().lower() != "prune":
        raise ValueError(
            "STUN compression requires model.params.expert_pruning='prune'."
        )
    if (
        str(config.model.params.expert_weight_compression).strip().lower()
        != "stun_sparse24"
    ):
        raise ValueError(
            "STUN compression requires "
            "model.params.expert_weight_compression='stun_sparse24'."
        )
    source_path = Path(packed_state)
    output_path = Path(output)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("STUN output must differ from its source artifact.")
    if output_path.exists() and not overwrite:
        raise ValueError(f"STUN output already exists: {output_path}.")
    resolved_device = str(device).strip().lower()
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("STUN compression requested CUDA, but CUDA is unavailable.")
    if resolved_device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")

    manifest = read_compressed_weights_packed_state_manifest(source_path)
    targets = _target_experts(
        manifest,
        target_experts=target_experts,
        keep_fraction=keep_fraction,
        top_k=int(config.model.params.experts_per_token),
    )
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
        converted, converted_manifest, report = (
            transform_packed_state_stun_sparse24(
                tensors,
                manifest,
                target_experts=targets,
                device=torch.device(resolved_device),
                reconstruct_below=int(reconstruct_below),
                quant_group_size=int(quant_group_size),
            )
        )
        if max_projection_error is not None:
            maximum = max(
                float(item["relative_projection_error"])
                for item in report["modules"].values()
            )
            if maximum > float(max_projection_error):
                raise ValueError(
                    f"STUN compact projection error {maximum:.6f} exceeds "
                    f"the configured maximum {float(max_projection_error):.6f}."
                )
        written = save_compressed_weights_packed_tensors(
            output_path,
            converted,
            converted_manifest,
            metadata={
                "model_type": str(config.model.type),
                "model_variant": str(config.model.params.variant),
                "expert_pruning": "stun_router_similarity",
                "expert_weight_compression": "stun_sparse24",
            },
        )
    report.update(
        {
            "status": "ok",
            "output": str(written),
            "device": resolved_device,
            "targets": targets,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--packed-state", required=True)
    parser.add_argument("--output", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-experts", type=int)
    target.add_argument("--keep-fraction", type=float)
    parser.add_argument("--reconstruct-below", type=int, default=3)
    parser.add_argument("--quant-group-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-projection-error", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = compress_expert_artifact(
        config_path=args.config,
        packed_state=args.packed_state,
        output=args.output,
        target_experts=args.target_experts,
        keep_fraction=args.keep_fraction,
        reconstruct_below=args.reconstruct_below,
        quant_group_size=args.quant_group_size,
        device=args.device,
        max_projection_error=args.max_projection_error,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
