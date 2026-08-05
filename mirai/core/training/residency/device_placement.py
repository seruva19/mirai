"""Compute-device and dtype placement for sparse-MoE training runs.

The cache yields CPU tensors, so this helper resolves the target device/dtype
from config and moves the model and each batch onto it. For
weight-residency/offload strategies it keeps streamed transformer blocks off-GPU
so the resident footprint reflects the configured offload budget, while
always-resident modules and the active batch live on the GPU.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mirai.core.training.residency.tensor_residency import cast_trainable_tensors


_DTYPE_BY_NAME = {
    "": "float32",
    "fp32": "float32",
    "float32": "float32",
    "f32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "f16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}

_RESIDENT_STRATEGIES = {"", "disabled", "none", "off"}


def resolve_compute_dtype(config: Any) -> "torch.dtype":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to resolve compute dtype.")
    name = str(getattr(config.model, "dtype", "") or "").strip().lower()
    if name not in _DTYPE_BY_NAME:
        allowed = ", ".join(sorted(key for key in _DTYPE_BY_NAME if key))
        raise ValueError(f"model.dtype must be one of: {allowed}; got '{name}'.")
    canonical = _DTYPE_BY_NAME[name]
    return getattr(torch, canonical)


def resolve_compute_device() -> "torch.device":
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to resolve compute device.")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def place_pipeline_on_device(
    pipeline: Any,
    *,
    device: "torch.device",
    dtype: "torch.dtype",
    residency_strategy: str,
) -> None:
    """Place a pipeline's model and precision for training.

    Model-agnostic: it resolves device/dtype and the full-resident path here, and
    delegates any offload (block-swap/stream) placement to the pipeline's own
    ``place_offloaded_modules`` contract method, so this core module needs no
    knowledge of a specific model family.
    """

    if torch is None:  # pragma: no cover
        return
    model = pipeline.get_training_model()
    if model is None:
        return

    # Record the intended mixed-precision compute dtype so the forward autocasts
    # under it even when master params stay fp32 (e.g. fp8 frozen-weight runs).
    pipeline.set_compute_autocast_dtype(dtype)

    strategy = str(residency_strategy or "disabled").strip().lower()
    # When frozen weights are quantized (fp8/nf4/int8) the packed buffers must
    # keep their storage dtype; casting them to bf16 would undo the compression.
    # The quantized linears dequantize to the activation dtype on the fly, and
    # the forward runs under a bf16 autocast, so a dtype cast is unnecessary.
    quantized = bool(pipeline.has_quantized_frozen_weights())
    preserve_native_dtypes = bool(pipeline.preserves_native_parameter_dtypes())
    if not quantized and not preserve_native_dtypes:
        # Cast precision first (cheap on CPU, avoids a transient fp32 copy on GPU).
        model.to(dtype=dtype)
    elif quantized:
        cast_trainable_tensors(model, dtype=dtype)

    if (
        device.type == "cuda"
        and strategy not in _RESIDENT_STRATEGIES
        and pipeline.supports_weight_residency_strategy()
    ):
        pipeline.place_offloaded_modules(device=device, strategy=strategy)
        return

    model.to(device=device)


def move_batch_to_device(
    batch: dict[str, Any],
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> dict[str, Any]:
    if torch is None or device.type == "cpu":
        return batch
    return {key: _move_value(value, device=device, dtype=dtype) for key, value in batch.items()}


def _move_value(value: Any, *, device: "torch.device", dtype: "torch.dtype") -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: _move_value(v, device=device, dtype=dtype) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_move_value(v, device=device, dtype=dtype) for v in value]
        return type(value)(moved) if isinstance(value, tuple) else moved
    return value
