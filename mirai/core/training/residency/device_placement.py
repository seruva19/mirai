"""Compute-device and dtype placement for sparse-MoE runs.

The cache yields CPU tensors, so this helper resolves the target device/dtype
from config and moves the model and each batch onto it. For
weight-residency/offload strategies it keeps streamed transformer blocks off-GPU
so the resident footprint reflects the configured offload budget, while
always-resident modules and the active batch live on the GPU.

The same placement seam serves the training and the inference entrypoints. The
two differ in what the surrounding execution guarantees, not in the transfer
machinery, so the difference is carried as a typed
:class:`WeightResidencyExecutionMode` established once when residency is
configured. Training keeps its adapter/recompute preconditions; inference runs
without an autograd graph, so no block weight is retained past its own forward
and those preconditions do not apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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

_OFFLOADED_STRATEGIES = ("block_swap", "stream_disk")


class WeightResidencyExecutionMode(str, Enum):
    """Which execution phase a weight-residency configuration is built for."""

    TRAINING = "training"
    INFERENCE = "inference"

    @classmethod
    def coerce(cls, value: "WeightResidencyExecutionMode | str") -> "WeightResidencyExecutionMode":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        for member in cls:
            if member.value == text:
                return member
        allowed = ", ".join(member.value for member in cls)
        raise ValueError(f"Weight-residency execution mode must be one of: {allowed}.")


@dataclass(frozen=True)
class InferenceWeightResidency:
    """Resolved inference-entrypoint weight-residency request.

    ``strategy`` is ``"disabled"`` for every configuration that does not opt in,
    in which case the remaining fields are inert and the pipeline is never
    reconfigured.
    """

    strategy: str
    blocks_to_swap: int
    mode: str
    offload_dir: str

    @property
    def enabled(self) -> bool:
        return self.strategy not in _RESIDENT_STRATEGIES and self.blocks_to_swap > 0


def resolve_inference_weight_residency(config: Any) -> InferenceWeightResidency:
    """Resolve and cross-validate the inference-entrypoint residency request.

    ``inference.blocks_to_swap`` is the sole opt-in: at ``0`` nothing here
    applies and the caller keeps whatever the trainer runtime resolved, so an
    existing config is untouched. Above ``0``,
    ``memory.weight_residency_strategy`` must name a transport, because a
    swap count with no transport would otherwise be a silently ignored key.
    """

    blocks_to_swap = int(getattr(config.inference, "blocks_to_swap", 0))
    if blocks_to_swap < 0:
        raise ValueError("inference.blocks_to_swap must be >= 0.")
    if blocks_to_swap == 0:
        return InferenceWeightResidency(
            strategy="disabled", blocks_to_swap=0, mode="sync", offload_dir=""
        )
    strategy = str(
        getattr(config.memory, "weight_residency_strategy", "disabled")
    ).strip().lower()
    if strategy in {"", "auto"}:
        strategy = "block_swap"
    if strategy not in _OFFLOADED_STRATEGIES:
        raise ValueError(
            "inference.blocks_to_swap > 0 requires "
            "memory.weight_residency_strategy='block_swap' or 'stream_disk'; "
            f"got '{strategy}'."
        )
    planner = str(getattr(config.memory, "block_residency_planner", "uniform")).strip().lower()
    if planner not in {"", "uniform", "none", "off"}:
        # Phase-aware planning pins blocks across the forward/backward turn.
        # An inference run has no backward phase, so those pins would never be
        # released and the resident set would exceed the requested budget.
        raise ValueError(
            "memory.block_residency_planner='phase_aware' is a training-phase "
            "policy and cannot be combined with inference.blocks_to_swap."
        )
    return InferenceWeightResidency(
        strategy=strategy,
        blocks_to_swap=blocks_to_swap,
        mode=str(getattr(config.inference, "block_swap_mode", "sync")).strip().lower(),
        offload_dir=str(getattr(config.logging, "output_dir", "")) + "/weight_stream",
    )


def configure_inference_weight_residency(pipeline: Any, *, config: Any) -> str | None:
    """Arm the pipeline's residency manager for a no-grad inference run.

    Returns the residency strategy the subsequent
    :func:`place_pipeline_on_device` call must use, or ``None`` when the config
    does not opt in — in which case the pipeline is left exactly as the trainer
    runtime configured it.
    """

    request = resolve_inference_weight_residency(config)
    if not request.enabled:
        return None
    if not bool(pipeline.supports_weight_residency_strategy()):
        raise ValueError(
            f"model.type='{config.model.type}' does not implement weight "
            "residency, so inference.blocks_to_swap cannot be honored."
        )
    pipeline.set_weight_residency_strategy(
        strategy=request.strategy,
        blocks_to_swap=request.blocks_to_swap,
        mode=request.mode,
        # Eviction after each block is what bounds the resident set. The flag is
        # named for the training phase it was introduced for; an inference run
        # has no backward pass whose weights would need to survive.
        block_swap_backward=True,
        offload_dir=request.offload_dir,
        block_residency_planner="uniform",
        block_swap_prefetch_depth=int(
            getattr(config.memory, "block_swap_prefetch_depth", 1)
        ),
        block_residency_priority=str(
            getattr(config.memory, "block_residency_priority", "index")
        ),
        block_swap_transfer_strategy=str(
            getattr(config.memory, "block_swap_transfer_strategy", "per_tensor")
        ),
        execution_mode=WeightResidencyExecutionMode.INFERENCE,
    )
    return request.strategy


def configure_inference_refiner_weight_residency(
    pipeline: Any,
    *,
    config: Any,
) -> bool:
    """Apply an explicit residency override to a separate refiner denoiser."""

    blocks_to_swap = int(
        getattr(config.inference, "refiner_blocks_to_swap", 0)
    )
    if blocks_to_swap <= 0:
        return False
    strategy = str(
        getattr(config.memory, "weight_residency_strategy", "disabled")
    ).strip().lower()
    if strategy in {"", "auto"}:
        strategy = "block_swap"
    if strategy not in _OFFLOADED_STRATEGIES:
        raise ValueError(
            "inference.refiner_blocks_to_swap > 0 requires "
            "memory.weight_residency_strategy='block_swap' or 'stream_disk'."
        )
    configure = getattr(pipeline, "set_refiner_weight_residency_strategy", None)
    if not callable(configure):
        raise ValueError(
            f"model.type='{config.model.type}' does not implement separate refiner "
            "weight residency."
        )
    configure(
        strategy=strategy,
        blocks_to_swap=blocks_to_swap,
        mode=str(
            getattr(config.inference, "refiner_block_swap_mode", "sync")
        ).strip().lower(),
        offload_dir=str(getattr(config.logging, "output_dir", ""))
        + "/refiner_weight_stream",
        block_swap_prefetch_depth=int(
            getattr(config.memory, "block_swap_prefetch_depth", 1)
        ),
        block_residency_priority=str(
            getattr(config.memory, "block_residency_priority", "index")
        ),
        block_swap_transfer_strategy=str(
            getattr(config.memory, "block_swap_transfer_strategy", "per_tensor")
        ),
    )
    return True


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
