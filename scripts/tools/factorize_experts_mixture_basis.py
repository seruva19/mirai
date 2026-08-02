"""Create a schema-v4 optimized mixture-basis packed-state artifact."""

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

from mirai.core.builtins import register_builtin_components
from mirai.core.models.compressed_weights import (
    load_compressed_weights_packed_tensors,
)
from mirai.core.models.compressed_weights import packed_artifact_fingerprint
from mirai.core.models.compressed_weights import (
    read_compressed_weights_packed_state_manifest,
)
from mirai.core.models.compressed_weights import (
    save_compressed_weights_packed_tensors,
)
from mirai.core.models.compressed_weights.factorization.mixture_basis_artifact import (
    factorize_packed_state_mixture_basis,
)
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")


def factorize_expert_artifact(
    *,
    config_path: str | Path,
    packed_state: str | Path,
    output: str | Path,
    rank: int,
    basis_count: int,
    activation: str,
    optimization_steps: int,
    learning_rate: float,
    expert_batch_size: int,
    row_chunk_size: int,
    checkpoint_interval: int,
    factor_dtype: str,
    device: str,
    max_covariance_gib: float = 2.0,
    max_optimizer_gib: float = 24.0,
    max_reconstruction_error: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config, _notes = load_runtime_config(
        config_path,
        entrypoint="mixture-basis-factorization",
    )
    gate = str(
        config.model.params.expert_weight_compression
    ).strip().lower()
    if gate != "mixture_basis":
        raise ValueError(
            "Mixture-basis factorization requires "
            "model.params.expert_weight_compression='mixture_basis'."
        )
    source_path = Path(packed_state)
    output_path = Path(output)
    if source_path.resolve() == output_path.resolve():
        raise ValueError(
            "Mixture-basis output must differ from its source artifact."
        )
    if output_path.exists() and not overwrite:
        raise ValueError(
            f"Mixture-basis output already exists: {output_path}."
        )
    resolved_device = str(device).strip().lower()
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Mixture-basis factorization requested CUDA, but CUDA is unavailable."
        )
    if resolved_device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")

    source_fingerprint = packed_artifact_fingerprint(source_path)
    manifest = read_compressed_weights_packed_state_manifest(source_path)
    tensors = load_compressed_weights_packed_tensors(source_path)
    lease = (
        acquire_gpu_lease(
            lock_path=str(resolve_lease_lock_path(ROOT)),
            timeout_seconds=float(
                os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0")
            ),
        )
        if resolved_device == "cuda"
        else nullcontext()
    )
    with lease:
        converted, converted_manifest, report = (
            factorize_packed_state_mixture_basis(
                tensors,
                manifest,
                rank=int(rank),
                basis_count=int(basis_count),
                activation=activation,
                optimization_steps=int(optimization_steps),
                learning_rate=float(learning_rate),
                expert_batch_size=int(expert_batch_size),
                row_chunk_size=int(row_chunk_size),
                checkpoint_interval=int(checkpoint_interval),
                factor_dtype=factor_dtype,
                device=torch.device(resolved_device),
                source_artifact_fingerprint=source_fingerprint,
                max_covariance_gib=float(max_covariance_gib),
                max_optimizer_gib=float(max_optimizer_gib),
            )
        )
        if max_reconstruction_error is not None:
            maximum = max(
                float(value)
                for module in report["modules"].values()
                for value in module["relative_frobenius_error"].values()
            )
            if maximum > float(max_reconstruction_error):
                raise ValueError(
                    f"Mixture-basis reconstruction error {maximum:.6f} "
                    "exceeds the configured maximum "
                    f"{float(max_reconstruction_error):.6f}."
                )
        written = save_compressed_weights_packed_tensors(
            output_path,
            converted,
            converted_manifest,
            metadata={
                "model_type": str(config.model.type),
                "model_variant": str(config.model.params.variant),
                "expert_weight_compression": "mixture_basis",
                "source_artifact_fingerprint": source_fingerprint,
            },
        )
    report.update(
        {
            "status": "ok",
            "output": str(written),
            "device": resolved_device,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--packed-state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--basis-count", required=True, type=int)
    parser.add_argument(
        "--activation",
        choices=("silu", "tanh", "gelu"),
        default="silu",
    )
    parser.add_argument("--optimization-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--expert-batch-size", type=int, default=8)
    parser.add_argument("--row-chunk-size", type=int, default=256)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--factor-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--max-covariance-gib", type=float, default=2.0)
    parser.add_argument("--max-optimizer-gib", type=float, default=24.0)
    parser.add_argument("--max-reconstruction-error", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    register_builtin_components()
    report = factorize_expert_artifact(
        config_path=args.config,
        packed_state=args.packed_state,
        output=args.output,
        rank=args.rank,
        basis_count=args.basis_count,
        activation=args.activation,
        optimization_steps=args.optimization_steps,
        learning_rate=args.learning_rate,
        expert_batch_size=args.expert_batch_size,
        row_chunk_size=args.row_chunk_size,
        checkpoint_interval=args.checkpoint_interval,
        factor_dtype=args.factor_dtype,
        device=args.device,
        max_covariance_gib=args.max_covariance_gib,
        max_optimizer_gib=args.max_optimizer_gib,
        max_reconstruction_error=args.max_reconstruction_error,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
