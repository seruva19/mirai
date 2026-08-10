"""Export a configured model's frozen base as a packed compressed_weights artifact."""

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

from mirai.config.loader import load_config
from mirai.config.runtime_policy import (
    validate_cli_model_contract,
    validate_native_backend_availability,
)
from mirai.config.schema import TrainingConfig
from mirai.core.models.compressed_weights import save_compressed_weights_packed_state
from mirai.core.models.compressed_weights.packed.packed_storage_alignment import (
    GDS_STORAGE_ALIGNMENT_BYTES,
)
from mirai.core.models.providers import load_configured_model_provider_module
from mirai.core.models.providers import get_model_family_provider
from mirai.core.training.trainer import _instantiate_model_pipeline
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")


def _ensure_compressed_weights_config(config: TrainingConfig) -> None:
    scheme = str(config.memory.frozen_weight_quantization).strip().lower()
    strategy = str(config.memory.frozen_weight_quantization_strategy).strip().lower()
    if config.memory.frozen_weight_packed_state_path:
        raise ValueError(
            "memory.frozen_weight_packed_state_path must be empty when exporting "
            "a new packed state."
        )
    if scheme not in {
        "",
        "none",
        "int8",
        "nf4",
        "gguf_iq4",
        "gguf_iq3",
        "gguf_iq2",
        "mxfp8_e4m3",
        "mxfp4",
        "nvfp4",
    }:
        raise ValueError(
            "Packed compressed_weights export requires memory.frozen_weight_quantization "
            "to be 'none', 'int8', 'nf4', 'gguf_iq4', 'gguf_iq3', 'gguf_iq2', "
            "'mxfp8_e4m3', 'mxfp4', or 'nvfp4'."
        )
    if strategy in {"", "disabled", "none"}:
        config.memory.frozen_weight_quantization_strategy = "compressed_weights"
    elif strategy not in {"auto", "compressed_weights"}:
        raise ValueError(
            "Packed compressed-weight export requires "
            "memory.frozen_weight_quantization_strategy to be auto or "
            "compressed_weights."
        )
    config.memory.frozen_weight_quantization = "int8" if scheme in {"", "none"} else scheme


def export_compressed_weights_packed_state_from_config(
    config: TrainingConfig,
    output_path: str | Path,
    *,
    rotation_optimization_steps: int = 200,
    rotation_learning_rate: float = 0.01,
    rotation_row_chunk_size: int = 4096,
    rotation_checkpoint_interval: int = 25,
    rotation_device: str = "auto",
    rotation_max_workspace_gib: float = 2.0,
) -> dict[str, Any]:
    _ensure_compressed_weights_config(config)
    validate_cli_model_contract(config, entrypoint="export-compressed-weights")
    load_configured_model_provider_module(config)
    validate_native_backend_availability(
        config,
        entrypoint="export-compressed-weights",
    )

    provider = get_model_family_provider(config.model.type)
    if provider is None:
        raise ValueError(f"Unknown model family '{config.model.type}'.")
    model_cls = provider.require_pipeline_type()
    pipeline = _instantiate_model_pipeline(model_cls, config)
    caps = pipeline.get_memory_feature_capabilities()
    if not bool(caps.quantized_frozen_weights):
        raise ValueError(
            f"model.type='{config.model.type}' does not implement quantized_frozen_weights."
        )
    if not bool(caps.packed_frozen_weight_state):
        raise ValueError(
            f"model.type='{config.model.type}' does not implement packed_frozen_weight_state."
        )
    scheme = str(config.memory.frozen_weight_quantization)
    rotation_mode = str(
        config.model.params.expert_quantization_rotation
    ).strip().lower()
    learn_rotations = rotation_mode == "learned"
    if learn_rotations and scheme != "int8":
        raise ValueError(
            "Learned expert rotations require "
            "memory.frozen_weight_quantization='int8'."
        )
    resolved_rotation_device = "disabled"
    if learn_rotations:
        resolved_rotation_device = str(rotation_device).strip().lower()
        if resolved_rotation_device == "auto":
            resolved_rotation_device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        if resolved_rotation_device not in {"cpu", "cuda"}:
            raise ValueError("rotation_device must be auto, cpu, or cuda.")
        if resolved_rotation_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Learned rotation export requested CUDA, but CUDA is unavailable."
            )
    quant_kwargs: dict[str, Any] = {}
    if scheme == "nf4":
        quant_kwargs["block_size"] = int(config.memory.quantization_block_size)
    if config.memory.frozen_weight_quantization_strategy not in {"", "auto", "disabled", "none"}:
        quant_kwargs["strategy"] = config.memory.frozen_weight_quantization_strategy
    if learn_rotations:
        quant_kwargs.update(
            {
                "learn_expert_rotations": True,
                "rotation_optimization_steps": int(
                    rotation_optimization_steps
                ),
                "rotation_learning_rate": float(rotation_learning_rate),
                "rotation_row_chunk_size": int(rotation_row_chunk_size),
                "rotation_checkpoint_interval": int(
                    rotation_checkpoint_interval
                ),
                "rotation_device": resolved_rotation_device,
                "rotation_max_workspace_gib": float(
                    rotation_max_workspace_gib
                ),
            }
        )
    lease = (
        acquire_gpu_lease(
            lock_path=str(resolve_lease_lock_path(ROOT)),
            timeout_seconds=float(
                os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0")
            ),
        )
        if learn_rotations and resolved_rotation_device == "cuda"
        else nullcontext()
    )
    with lease:
        pipeline.enable_quantized_frozen_weights(scheme, **quant_kwargs)
    if not pipeline.has_quantized_frozen_weights():
        raise ValueError(
            f"model.type='{config.model.type}' did not produce quantized frozen weights."
        )
    training_model = pipeline.get_training_model()
    if training_model is None:
        raise ValueError(
            f"model.type='{config.model.type}' does not expose a training model for export."
        )
    artifact_metadata = {
        "model_type": str(config.model.type),
        "model_variant": str(config.model.params.variant),
        "denoiser_subfolder": str(
            getattr(config.model.params, "denoiser_subfolder", "transformer") or "transformer"
        ),
        "strategy": "compressed_weights",
        "quant_format": scheme,
    }
    saved_path = pipeline.save_packed_frozen_weight_state(
        output_path, metadata=artifact_metadata
    )
    if saved_path is None:
        saved_path = save_compressed_weights_packed_state(
            output_path,
            training_model,
            storage_alignment_bytes=(
                GDS_STORAGE_ALIGNMENT_BYTES
                if config.memory.packed_stream_backend == "gds" else 0
            ),
            metadata=artifact_metadata,
        )
    report = pipeline.get_quantized_frozen_weight_report() or {}
    return {
        "status": "ok",
        "output": str(saved_path),
        "model_type": str(config.model.type),
        "model_variant": str(config.model.params.variant),
        "denoiser_subfolder": str(
            getattr(config.model.params, "denoiser_subfolder", "transformer") or "transformer"
        ),
        "quant_format": scheme,
        "expert_quantization_rotation": rotation_mode,
        "rotation_device": (
            resolved_rotation_device
        ),
        "storage_alignment_bytes": (
            GDS_STORAGE_ALIGNMENT_BYTES
            if config.memory.packed_stream_backend == "gds" else 0
        ),
        "quantized_report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rotation-optimization-steps", type=int, default=200)
    parser.add_argument("--rotation-learning-rate", type=float, default=0.01)
    parser.add_argument("--rotation-row-chunk-size", type=int, default=4096)
    parser.add_argument("--rotation-checkpoint-interval", type=int, default=25)
    parser.add_argument(
        "--rotation-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--rotation-max-workspace-gib", type=float, default=2.0)
    args = parser.parse_args()

    config = load_config(args.config)
    summary = export_compressed_weights_packed_state_from_config(
        config,
        args.output,
        rotation_optimization_steps=args.rotation_optimization_steps,
        rotation_learning_rate=args.rotation_learning_rate,
        rotation_row_chunk_size=args.rotation_row_chunk_size,
        rotation_checkpoint_interval=args.rotation_checkpoint_interval,
        rotation_device=args.rotation_device,
        rotation_max_workspace_gib=args.rotation_max_workspace_gib,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
