"""Trainer startup/runtime setup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.config.runtime_policy import (
    apply_runtime_policy,
    validate_runtime_compatibility,
)
from mirai.config.hardware_tiers import apply_hardware_memory_plan
from mirai.config.schema import TrainingConfig
from mirai.core.models.quantization import expert_quantization_formats
from mirai.core.moe.runtime.specs import (
    MoEOptimizationPolicy,
    set_active_moe_optimization_policy,
)
from mirai.core.moe.runtime.autotune_warmup import (
    grouped_gemm_warmup_problems,
    warmup_persistent_grouped_gemm,
)
from mirai.core.training.runtime.contract import ALLOWED_ADAPTER_TYPES
from mirai.core.training.runtime.compilation import (
    CompilationPolicy,
    CompilationSession,
    prepare_training_compilation,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class TrainerRuntimeSetup:
    runtime_policy_notes: list[str]
    fp8_frozen_weights_enabled: bool
    frozen_weight_quantization: str
    frozen_weight_quantization_strategy: str
    frozen_weight_packed_state_path: str
    requested_weight_residency_strategy: str
    weight_residency_strategy: str
    expert_weight_access: str
    expert_dequant_chunk_size: int
    quantize_experts_on_load: bool
    router_quantization: str
    moe_kernel_backend: str
    trainable_parameter_offload: bool
    memory_feature_notes: list[str]
    forward_fn: Any
    compile_enabled: bool
    compile_warning: str
    compilation_session: CompilationSession


def resolve_weight_residency_strategy(config: TrainingConfig) -> str:
    requested = str(
        getattr(config.memory, "weight_residency_strategy", "auto")
    ).strip().lower()
    if requested in {"", "auto"}:
        if int(config.training.blocks_to_swap) > 0:
            return "block_swap"
        return "disabled"
    if requested not in {"disabled", "block_swap", "stream_disk"}:
        raise ValueError(
            "memory.weight_residency_strategy must be one of: "
            "auto, disabled, block_swap, stream_disk."
        )
    return requested


def initialize_trainer_runtime(*, config: TrainingConfig, pipeline: Any) -> TrainerRuntimeSetup:
    runtime_policy_notes = list(
        apply_runtime_policy(
            config,
            entrypoint="trainer",
        )
    )
    validate_runtime_compatibility(config, entrypoint="trainer")
    hardware_plan = apply_hardware_memory_plan(config.memory)
    fp8_frozen_weights_enabled = (
        str(config.memory.frozen_weight_quantization).strip().lower() == "fp8"
    )
    frozen_weight_quantization = str(
        config.memory.frozen_weight_quantization
    ).strip().lower()
    frozen_weight_quantization_strategy = str(
        getattr(config.memory, "frozen_weight_quantization_strategy", "disabled")
    ).strip().lower()
    frozen_weight_quantization_strategy = _normalize_frozen_weight_quantization_strategy(
        frozen_weight_quantization_strategy
    )
    frozen_weight_packed_state_path = str(
        getattr(config.memory, "frozen_weight_packed_state_path", "")
    ).strip()
    requested_weight_residency_strategy = str(
        getattr(config.memory, "weight_residency_strategy", "auto")
    ).strip().lower()
    weight_residency_strategy = resolve_weight_residency_strategy(config)
    moe_optimization_policy = MoEOptimizationPolicy.from_memory_config(config.memory)
    # Publish the config-derived MoE policy for compressed-weight execution.
    set_active_moe_optimization_policy(moe_optimization_policy)
    trainable_parameter_offload = bool(
        getattr(config.memory, "trainable_parameter_offload", False)
    )
    memory_feature_notes = _build_memory_feature_notes(
        fp8_frozen_weights_enabled=fp8_frozen_weights_enabled,
        frozen_weight_quantization=frozen_weight_quantization,
        frozen_weight_quantization_strategy=frozen_weight_quantization_strategy,
        frozen_weight_packed_state_path=frozen_weight_packed_state_path,
        requested_weight_residency_strategy=requested_weight_residency_strategy,
        weight_residency_strategy=weight_residency_strategy,
        moe_optimization_policy=moe_optimization_policy,
        trainable_parameter_offload=trainable_parameter_offload,
        hardware_tier=(
            hardware_plan.tier_name if hardware_plan is not None else ""
        ),
    )
    _configure_pipeline_runtime(
        pipeline=pipeline,
        config=config,
        weight_residency_strategy=weight_residency_strategy,
        frozen_weight_quantization=frozen_weight_quantization,
        frozen_weight_quantization_strategy=frozen_weight_quantization_strategy,
        frozen_weight_packed_state_path=frozen_weight_packed_state_path,
        fp8_frozen_weights_enabled=fp8_frozen_weights_enabled,
        moe_optimization_policy=moe_optimization_policy,
        trainable_parameter_offload=trainable_parameter_offload,
    )
    _warmup_moe_autotune(config=config, pipeline=pipeline)
    pipeline.train()
    compilation_session = prepare_training_compilation(
        pipeline=pipeline,
        policy=CompilationPolicy.from_training_config(config.training),
    )
    return TrainerRuntimeSetup(
        runtime_policy_notes=runtime_policy_notes,
        fp8_frozen_weights_enabled=fp8_frozen_weights_enabled,
        frozen_weight_quantization=frozen_weight_quantization,
        frozen_weight_quantization_strategy=frozen_weight_quantization_strategy,
        frozen_weight_packed_state_path=frozen_weight_packed_state_path,
        requested_weight_residency_strategy=requested_weight_residency_strategy,
        weight_residency_strategy=weight_residency_strategy,
        expert_weight_access=moe_optimization_policy.expert_weight_access,
        expert_dequant_chunk_size=moe_optimization_policy.expert_dequant_chunk_size,
        quantize_experts_on_load=moe_optimization_policy.quantize_experts_on_load,
        router_quantization=moe_optimization_policy.router_quantization,
        moe_kernel_backend=moe_optimization_policy.kernel_backend,
        trainable_parameter_offload=trainable_parameter_offload,
        memory_feature_notes=memory_feature_notes,
        forward_fn=compilation_session.forward_fn,
        compile_enabled=compilation_session.enabled,
        compile_warning=compilation_session.warning,
        compilation_session=compilation_session,
    )


def _warmup_moe_autotune(*, config: TrainingConfig, pipeline: Any) -> None:
    routed_rows = int(getattr(config.memory, "moe_autotune_warmup_rows", 0))
    if routed_rows == 0:
        return
    if routed_rows < 0:
        raise ValueError("memory.moe_autotune_warmup_rows must be >= 0.")
    dispatch = str(getattr(config.memory, "moe_dispatch", "vectorized"))
    gemm = str(getattr(config.memory, "moe_gemm_backend", "auto"))
    if dispatch != "triton_persistent" and gemm != "persistent":
        raise ValueError(
            "memory.moe_autotune_warmup_rows requires "
            "memory.moe_dispatch='triton_persistent' or "
            "memory.moe_gemm_backend='persistent'."
        )
    specs = pipeline.get_expert_tensor_specs()
    problems = grouped_gemm_warmup_problems(specs, routed_rows=routed_rows)
    if not problems:
        raise ValueError(
            "MoE autotune warm-up requires provider-declared routed expert tensors."
        )
    warmup_persistent_grouped_gemm(problems)


def _build_memory_feature_notes(
    *,
    fp8_frozen_weights_enabled: bool,
    frozen_weight_quantization: str,
    frozen_weight_quantization_strategy: str,
    frozen_weight_packed_state_path: str,
    requested_weight_residency_strategy: str,
    weight_residency_strategy: str,
    moe_optimization_policy: MoEOptimizationPolicy,
    trainable_parameter_offload: bool,
    hardware_tier: str = "",
) -> list[str]:
    notes: list[str] = []
    if hardware_tier:
        notes.append(
            f"memory.hardware_policy='tiered' resolved tier '{hardware_tier}'."
        )
    if frozen_weight_quantization not in {"", "none"}:
        notes.append(
            f"memory.frozen_weight_quantization='{frozen_weight_quantization}' enabled."
        )
    elif fp8_frozen_weights_enabled:
        notes.append(
            "memory.frozen_weight_quantization='fp8' enabled."
        )
    if frozen_weight_quantization_strategy not in {"", "disabled", "none", "auto"}:
        notes.append(
            "memory.frozen_weight_quantization_strategy="
            f"'{frozen_weight_quantization_strategy}' enabled."
        )
    if frozen_weight_packed_state_path:
        notes.append("memory.frozen_weight_packed_state_path enabled.")
    if weight_residency_strategy != "disabled":
        notes.append(
            f"memory.weight_residency_strategy='{weight_residency_strategy}' enabled."
        )
    if requested_weight_residency_strategy in {"", "auto"}:
        notes.append(
            "memory.weight_residency_strategy auto-resolved from memory settings."
        )
    if moe_optimization_policy.expert_weight_access not in {"", "auto", "disabled"}:
        notes.append(
            "memory.expert_weight_access="
            f"'{moe_optimization_policy.expert_weight_access}' enabled."
        )
    if moe_optimization_policy.quantize_experts_on_load:
        notes.append("memory.quantize_experts_on_load enabled.")
    if moe_optimization_policy.router_quantization != "disabled":
        notes.append(
            "memory.router_quantization="
            f"'{moe_optimization_policy.router_quantization}' enabled."
        )
    if moe_optimization_policy.kernel_backend not in {"", "auto", "torch"}:
        notes.append(
            "memory.moe_kernel_backend="
            f"'{moe_optimization_policy.kernel_backend}' enabled."
        )
    if trainable_parameter_offload:
        notes.append("memory.trainable_parameter_offload enabled.")
    return notes


def _configure_pipeline_runtime(
    *,
    pipeline: Any,
    config: TrainingConfig,
    weight_residency_strategy: str,
    frozen_weight_quantization: str,
    frozen_weight_quantization_strategy: str,
    frozen_weight_packed_state_path: str,
    fp8_frozen_weights_enabled: bool,
    moe_optimization_policy: MoEOptimizationPolicy,
    trainable_parameter_offload: bool,
) -> None:
    memory_caps = pipeline.get_memory_feature_capabilities()
    extension_caps = pipeline.get_model_extension_capabilities()
    timestep_axis_requested = (
        str(config.adapter.timestep_rank_schedule).strip().lower() != "none"
        or float(config.adapter.timestep_band_min) != 0.0
        or float(config.adapter.timestep_band_max) != 1.0
        or str(config.adapter.lora_init).strip().lower() == "orthogonal"
    )
    initializer_requested = str(config.adapter.lora_init).strip().lower() != "kaiming"
    adapter_training_policy_requested = bool(config.adapter.use_lora_fa)
    allocation_policy_requested = bool(
        config.adapter.rank_pattern
        or config.adapter.alpha_pattern
        or int(config.adapter.rank_budget) > 0
        or bool(config.adapter.adaptive_rank_plan_path)
        or bool(config.adapter.use_rslora)
    )
    if timestep_axis_requested and not extension_caps.adapter_target_presets:
        raise ValueError(
            f"model.type='{config.model.type}' does not support the adapter "
            "timestep-axis controls (adapter.timestep_rank_schedule, "
            "adapter.timestep_band_min/max, adapter.lora_init); they are "
            "consumed by set_adapter_config, which this model does not expose."
        )
    if allocation_policy_requested and not extension_caps.adapter_allocation_policy:
        raise ValueError(
            f"model.type='{config.model.type}' does not support static LoRA allocation "
            "controls (adapter.rank_pattern, adapter.alpha_pattern, "
            "adapter.rank_budget, adapter.adaptive_rank_plan_path, "
            "adapter.use_rslora)."
        )
    if initializer_requested and not extension_caps.adapter_initialization:
        raise ValueError(
            f"model.type='{config.model.type}' does not support registered LoRA "
            "initializers (adapter.lora_init)."
        )
    if (
        adapter_training_policy_requested
        and not extension_caps.adapter_training_policy
    ):
        raise ValueError(
            f"model.type='{config.model.type}' does not support adapter training "
            "policies (adapter.use_lora_fa)."
        )
    if str(config.adapter.lora_init).strip().lower() == "loftq":
        if frozen_weight_quantization in {"", "none"}:
            raise ValueError(
                "adapter.lora_init='loftq' requires frozen-weight quantization."
            )
        if frozen_weight_packed_state_path:
            raise ValueError(
                "adapter.lora_init='loftq' cannot use a packed frozen-weight state "
                "because the original reference weights are unavailable."
            )
    if str(config.adapter.lora_init).strip().lower() == "eva":
        if str(config.adapter.init_from).strip():
            raise ValueError(
                "adapter.lora_init='eva' cannot be combined with adapter.init_from; "
                "use the calibrated adapter directly or start a fresh EVA run."
            )
        if int(config.adapter.condenser_rank) > 0:
            raise ValueError(
                "adapter.lora_init='eva' is incompatible with shared condenser "
                "factors; set adapter.condenser_rank=0."
            )
        if bool(config.training.compile):
            raise ValueError(
                "adapter.lora_init='eva' requires training.compile=false during "
                "activation calibration."
            )
    if extension_caps.adapter_target_presets:
        pipeline.set_adapter_config(config.adapter)
        if str(config.adapter.type).strip().lower() == "selected_expert":
            selection_mode = str(config.adapter.expert_selection).strip().lower()
            if selection_mode not in {"esft_gate", "esft_token"}:
                setter = getattr(pipeline, "set_selected_expert_ids", None)
                if not callable(setter):
                    raise ValueError(
                        f"model.type='{config.model.type}' cannot bind selected expert ids."
                    )
                setter(config.optimizer.selected_expert_ids)
    if extension_caps.adapter_runtime_controls:
        pipeline.set_adapter_runtime(
            rank_dropout=float(config.adapter.rank_dropout),
            lora_parameter_dropout=float(
                config.adapter.lora_parameter_dropout
            ),
        )
    adapter_type = str(config.adapter.type).strip().lower()
    if adapter_type not in ALLOWED_ADAPTER_TYPES:
        raise ValueError(
            "adapter.type must be one of: "
            + ", ".join(sorted(ALLOWED_ADAPTER_TYPES))
            + "."
        )
    if extension_caps.adapter_type_controls:
        pipeline.set_adapter_type(adapter_type)
    pipeline.set_gradient_checkpointing(config.training.gradient_checkpointing)
    if trainable_parameter_offload:
        if not memory_caps.trainable_parameter_offload:
            raise ValueError(
                f"model.type='{config.model.type}' does not implement "
                "memory.trainable_parameter_offload."
            )
        pipeline.configure_trainable_parameter_offload(True)
    if moe_optimization_policy.requests_runtime_behavior():
        if (
            moe_optimization_policy.quantize_experts_on_load
            and frozen_weight_quantization not in expert_quantization_formats()
        ):
            raise ValueError(
                "memory.quantize_experts_on_load requires "
                "a registered expert-capable frozen-weight quantization format."
            )
        if (
            moe_optimization_policy.expert_weight_access not in {"", "auto", "disabled"}
            and not memory_caps.expert_weight_access_policy
        ):
            raise ValueError(
                f"model.type='{config.model.type}' does not implement "
                "memory.expert_weight_access controls."
            )
        if (
            moe_optimization_policy.quantize_experts_on_load
            and not memory_caps.quantize_experts_on_load
        ):
            raise ValueError(
                f"model.type='{config.model.type}' does not implement "
                "memory.quantize_experts_on_load."
            )
        if (
            moe_optimization_policy.router_quantization != "disabled"
            and not memory_caps.router_quantization_policy
        ):
            raise ValueError(
                f"model.type='{config.model.type}' does not implement "
                "memory.router_quantization controls."
            )
        if (
            moe_optimization_policy.kernel_backend not in {"", "auto", "torch"}
            and not memory_caps.moe_kernel_backend
        ):
            raise ValueError(
                f"model.type='{config.model.type}' does not implement "
                "memory.moe_kernel_backend controls."
            )
        pipeline.configure_moe_optimization_policy(moe_optimization_policy)
    if memory_caps.weight_residency_strategy:
        residency_options = dict(
            strategy=weight_residency_strategy,
            blocks_to_swap=config.training.blocks_to_swap,
            mode=config.training.block_swap_mode,
            block_swap_backward=config.training.block_swap_backward,
            offload_dir=str(config.logging.output_dir) + "/weight_stream",
            block_residency_planner=config.memory.block_residency_planner,
            block_swap_prefetch_depth=config.memory.block_swap_prefetch_depth,
            block_residency_priority=config.memory.block_residency_priority,
        )
        transfer_strategy = str(
            config.memory.block_swap_transfer_strategy
        ).strip().lower()
        if transfer_strategy != "per_tensor":
            residency_options["block_swap_transfer_strategy"] = transfer_strategy
        pipeline.set_weight_residency_strategy(**residency_options)
    elif memory_caps.block_swap:
        pipeline.set_block_swap(
            blocks_to_swap=config.training.blocks_to_swap,
            mode=config.training.block_swap_mode,
            block_swap_backward=config.training.block_swap_backward,
        )
        if weight_residency_strategy != "disabled":
            raise ValueError(
                f"model.type='{config.model.type}' does not implement "
                "memory.weight_residency_strategy controls."
            )
    elif (
        int(config.training.blocks_to_swap) > 0
        or weight_residency_strategy != "disabled"
    ):
        raise ValueError(
            f"model.type='{config.model.type}' does not implement "
            "weight residency / block swap controls."
        )
    if frozen_weight_packed_state_path and frozen_weight_quantization not in {
        "fp8",
        "int8",
        "nf4",
        "gguf_iq4",
        "gguf_iq3",
        "mxfp8_e4m3",
        "mxfp4",
        "nvfp4",
    }:
        raise ValueError(
            "memory.frozen_weight_packed_state_path requires "
            "memory.frozen_weight_quantization='fp8', 'int8', 'nf4', 'gguf_iq4', "
            "'gguf_iq3', 'mxfp8_e4m3', 'mxfp4', or 'nvfp4'."
        )
    if frozen_weight_packed_state_path and not memory_caps.packed_frozen_weight_state:
        raise ValueError(
            f"model.type='{config.model.type}' does not implement "
            "memory.frozen_weight_packed_state_path."
        )
    if frozen_weight_quantization not in {"", "none"}:
        _validate_frozen_weight_quantization_strategy(frozen_weight_quantization_strategy)
        if memory_caps.quantized_frozen_weights:
            quant_kwargs: dict[str, Any] = {}
            if frozen_weight_quantization == "nf4":
                quant_kwargs["block_size"] = int(config.memory.quantization_block_size)
            if frozen_weight_quantization_strategy not in {"", "disabled", "none", "auto"}:
                quant_kwargs["strategy"] = frozen_weight_quantization_strategy
            precision_plan_path = str(
                getattr(config.memory, "expert_precision_plan_path", "")
            ).strip()
            if precision_plan_path:
                quant_kwargs["precision_plan_path"] = precision_plan_path
            pipeline.enable_quantized_frozen_weights(
                frozen_weight_quantization,
                **quant_kwargs,
            )
        else:
            raise ValueError(
                f"model.type='{config.model.type}' does not implement "
                "quantized_frozen_weights "
                f"(requested '{frozen_weight_quantization}')."
            )
    elif fp8_frozen_weights_enabled:
        _validate_frozen_weight_quantization_strategy(frozen_weight_quantization_strategy)
        if memory_caps.quantized_frozen_weights:
            pipeline.enable_quantized_frozen_weights("fp8")
        else:
            raise ValueError(
                f"model.type='{config.model.type}' does not implement "
                "memory.frozen_weight_quantization='fp8'."
            )
    elif frozen_weight_quantization_strategy not in {"", "disabled", "none", "auto"}:
        raise ValueError(
            "memory.frozen_weight_quantization_strategy requires "
            "memory.frozen_weight_quantization to be set."
        )
    structured_sparsity = str(
        getattr(config.memory, "expert_structured_sparsity", "disabled")
    ).strip().lower()
    if structured_sparsity not in {"", "disabled", "none"}:
        enable_structured = getattr(pipeline, "enable_structured_expert_sparsity", None)
        if not callable(enable_structured):
            raise ValueError(
                f"model.type='{config.model.type}' does not implement structured expert sparsity."
            )
        enable_structured(structured_sparsity)


def _normalize_frozen_weight_quantization_strategy(strategy: str) -> str:
    normalized = str(strategy).strip().lower()
    aliases = {
        "off": "disabled",
    }
    return aliases.get(normalized, normalized)


def _validate_frozen_weight_quantization_strategy(strategy: str) -> None:
    normalized = _normalize_frozen_weight_quantization_strategy(strategy)
    if normalized in {"", "disabled", "none", "auto"}:
        return
    if normalized == "compressed_weights":
        return
    raise ValueError(
        "memory.frozen_weight_quantization_strategy must be one of: "
        "disabled, auto, compressed_weights."
    )


def _prepare_forward_fn(
    *,
    pipeline: Any,
    compile_requested: bool,
    compile_mode: str = "",
    dynamic: bool | None = None,
) -> tuple[Any, bool, str]:
    """Wrap ``pipeline.forward`` in ``torch.compile`` when requested.

    Shared by the training seam (``training.compile``, mode-less) and the
    inference seam (``inference.compile`` / ``--compile MODE``).
    ``compile_mode`` selects the ``torch.compile`` mode (e.g. ``reduce-overhead``
    for automatic per-region CUDA-graph capture); ``dynamic`` pins shape
    specialization. A failure to construct the compiled callable returns eager
    ``pipeline.forward`` and records one warning. ``fullgraph=False`` permits
    TorchDynamo to execute unsupported graph regions eagerly.
    """
    forward_fn = pipeline.forward
    compile_enabled = False
    compile_warning = ""
    if not compile_requested:
        return forward_fn, compile_enabled, compile_warning
    try:
        if torch is None or not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this runtime.")
        compile_kwargs: dict[str, Any] = {}
        mode = str(compile_mode or "").strip()
        # "default" is torch.compile's implicit mode; passing it explicitly is a
        # no-op, so omit it to keep the training path's compile call unchanged.
        if mode and mode != "default":
            compile_kwargs["mode"] = mode
        if dynamic is not None:
            compile_kwargs["dynamic"] = bool(dynamic)
        forward_fn = torch.compile(pipeline.forward, **compile_kwargs)  # type: ignore[assignment]
        compile_enabled = True
    except Exception as exc:
        forward_fn = pipeline.forward
        compile_enabled = False
        compile_warning = (
            "torch.compile requested but auto-disabled due to graph/runtime "
            f"constraints: {exc}"
        )
    return forward_fn, compile_enabled, compile_warning
