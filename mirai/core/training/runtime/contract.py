"""Authoritative training runtime-contract validation."""

from __future__ import annotations

import importlib
import math
import warnings

from mirai.config.schema import TrainingConfig
from mirai.core.models.attention_backends import ALLOWED_ATTENTION_BACKENDS
from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.runtime.touch_guard import EXPERT_TOUCH_GUARD_MODES
from mirai.core.moe.runtime.kernels import megablocks_runtime_available
from mirai.core.moe.runtime.kernels import normalize_moe_kernel_backend
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy
from mirai.core.training.optim.optimizer import (
    OptimizerRegistry,
    SELECTED_EXPERT_OPTIMIZER_TYPES,
)
from mirai.core.training.preview.preview_solvers import PreviewSolverRegistry
from mirai.core.training.optim.scheduler import SchedulerRegistry
from mirai.core.training.objectives.sampling import (
    MODE_SHIFT_MAX_SCALE,
    MODE_SHIFT_MIN_SCALE,
    TIMESTEP_SAMPLING_MODES,
)
from mirai.core.training.data.curriculum import CurriculumSchedule
from mirai.core.training.runtime.compilation import CompilationPolicy
from mirai.core.training.training_policy import validate_training_policy_configs


_PAIRED_LORA_OPTIMIZER_TYPES = frozenset({"lora_pro_adamw", "lora_muon"})
_STOCHASTIC_ROUNDING_OPTIMIZER_TYPES = frozenset({"adamw"}).union(
    _PAIRED_LORA_OPTIMIZER_TYPES,
    SELECTED_EXPERT_OPTIMIZER_TYPES,
)


ALLOWED_LOSS_FUNCTIONS = frozenset({"mse", "huber", "pseudo_huber"})
ALLOWED_LOSS_WEIGHTING = frozenset(
    {"uniform", "min_snr_gamma", "cosmap", "adaptive_uncertainty"}
)
ALLOWED_LOSS_BUCKET_NORMALIZATION = frozenset({"none", "per_bucket_mean"})
ALLOWED_NON_FINITE_POLICIES = frozenset({"abort", "skip_step"})
ALLOWED_CHECKPOINTING_MODES = frozenset(
    {"off", "standard", "selective", "aggressive"}
)
ALLOWED_BLOCK_SWAP_MODES = frozenset({"sync", "async"})
# Allowed optimizer/scheduler names are registry-driven: builtins register at
# import (optimizer.py / scheduler.py), and any out-of-tree type registered via
# @register_optimizer / @register_scheduler is accepted here automatically.
ALLOWED_OPTIMIZERS = frozenset(OptimizerRegistry.names())
ALLOWED_SCHEDULERS = frozenset(SchedulerRegistry.names())
ALLOWED_ADAPTER_TYPES = frozenset({"lora", "selected_expert", "sparse_delta"})
# Preview solver names are registry-driven: builtins register at import
# (preview_solvers.py registers "euler"), and any additional solver registered
# via @register_preview_solver is accepted here automatically.
ALLOWED_SAMPLE_SOLVERS = frozenset(PreviewSolverRegistry.names())
def require_module(module_name: str, *, reason: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(reason) from exc


def normalize_checkpointing_mode(value: bool | str) -> str:
    if isinstance(value, bool):
        return "standard" if value else "off"
    mode = str(value).strip().lower()
    if mode in {"false", "0", "off"}:
        return "off"
    if mode in {"true", "1", "standard"}:
        return "standard"
    if mode == "aggressive":
        return "aggressive"
    return mode


def validate_training_runtime_config(config: TrainingConfig) -> None:
    CompilationPolicy.from_training_config(config.training)
    errors: list[str] = []

    def _check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    try:
        curriculum = CurriculumSchedule.from_config(config.training.curriculum)
    except (TypeError, ValueError) as exc:
        curriculum = None
        errors.append(str(exc))
    if curriculum is not None:
        _check(
            not curriculum.uses_task_mix
            or str(config.strategy.type).strip().lower() == "multi_task_video",
            "training.curriculum.task_mix_schedule requires "
            "strategy.type='multi_task_video'.",
        )
        _check(
            str(config.strategy.type).strip().lower() != "multi_task_video"
            or curriculum.uses_task_mix,
            "strategy.type='multi_task_video' requires an enabled "
            "training.curriculum.task_mix_schedule.",
        )

    routing_mode = str(config.model.params.moe_routing_mode).strip().lower()
    if routing_mode == "expert_choice":
        _check(
            int(config.model.params.moe_dynamic_topk_min) == 0
            and float(config.model.params.moe_dynamic_topk_average) == 0.0,
            "Expert-Choice routing cannot be combined with token-choice dynamic top-k.",
        )
        _check(
            float(config.model.params.expert_subset_fraction) == 1.0,
            "Expert-Choice routing cannot be combined with token-choice expert subsets.",
        )
        _check(
            str(config.dataset.moe_routing.specialization_mode).strip().lower()
            == "emergent",
            "Expert-Choice routing cannot be combined with token-choice dataset affinity.",
        )
        _check(
            float(config.model.params.moe_bias_update_rate) == 0.0,
            "Expert-Choice routing has equal expert capacity and cannot use "
            "token-choice router-bias balancing.",
        )
        _check(
            float(config.model.params.moe_phi_balance_weight) == 0.0,
            "Expert-Choice routing cannot use token-choice phi balancing.",
        )
        _check(
            str(config.model.params.moe_balance_scope).strip().lower()
            == "microbatch",
            "Expert-Choice routing does not use token-choice accumulation-wide balancing.",
        )
        policy_options = config.training.policy_options
        for policy_name in ("diversity_routing", "expert_dropout", "dataset_routing"):
            options = policy_options.get(policy_name, {})
            _check(
                not bool(options.get("enabled", False)),
                f"Expert-Choice routing cannot be combined with '{policy_name}'.",
            )

    _check(int(config.training.batch_size) >= 1, "training.batch_size must be >= 1.")
    _check(
        int(config.training.gradient_accumulation) >= 1,
        "training.gradient_accumulation must be >= 1.",
    )
    _check(int(config.training.max_steps) >= 1, "training.max_steps must be >= 1.")
    _check(int(config.training.warmup_steps) >= 0, "training.warmup_steps must be >= 0.")
    _check(
        int(config.training.max_consecutive_skipped_steps) >= 0,
        "training.max_consecutive_skipped_steps must be >= 0.",
    )
    _check(
        str(config.training.moe_expert_touch_guard).strip().lower()
        in EXPERT_TOUCH_GUARD_MODES,
        "training.moe_expert_touch_guard must be one of: off, warn, error.",
    )
    _check(
        0.0 < float(config.training.moe_expert_touch_max_fraction) <= 1.0,
        "training.moe_expert_touch_max_fraction must be in (0, 1].",
    )
    _check(
        int(config.training.val_every_n_steps) >= 0,
        "training.val_every_n_steps must be >= 0.",
    )
    _check(
        int(config.training.early_stop_patience) >= 0,
        "training.early_stop_patience must be >= 0.",
    )
    _check(
        int(config.training.log_grad_breakdown_every_n_steps) >= 0,
        "training.log_grad_breakdown_every_n_steps must be >= 0.",
    )
    _check(
        float(config.training.prior_loss_weight) >= 0.0,
        "training.prior_loss_weight must be >= 0.",
    )
    _check(
        0.0 <= float(config.training.prior_ratio) <= 1.0,
        "training.prior_ratio must be in [0, 1].",
    )
    _check(
        float(config.training.timestep_eps) > 0.0 and float(config.training.timestep_eps) < 0.5,
        "training.timestep_eps must be > 0 and < 0.5.",
    )
    timestep_sampling = str(config.training.timestep_sampling).strip().lower()
    _check(
        timestep_sampling in TIMESTEP_SAMPLING_MODES,
        "training.timestep_sampling must be one of: "
        + ", ".join(sorted(TIMESTEP_SAMPLING_MODES))
        + ".",
    )
    _check(
        math.isfinite(float(config.training.timestep_sampling_mean)),
        "training.timestep_sampling_mean must be finite.",
    )
    _check(
        math.isfinite(float(config.training.timestep_sampling_std))
        and float(config.training.timestep_sampling_std) > 0.0,
        "training.timestep_sampling_std must be finite and > 0.",
    )
    _check(
        math.isfinite(float(config.training.timestep_sampling_mode_scale))
        and MODE_SHIFT_MIN_SCALE
        <= float(config.training.timestep_sampling_mode_scale)
        <= MODE_SHIFT_MAX_SCALE,
        "training.timestep_sampling_mode_scale must be finite and in "
        f"[{MODE_SHIFT_MIN_SCALE}, {MODE_SHIFT_MAX_SCALE}].",
    )
    _check(float(config.training.min_snr_gamma) > 0.0, "training.min_snr_gamma must be > 0.")
    _check(float(config.training.max_grad_norm) > 0.0, "training.max_grad_norm must be > 0.")
    _check(
        str(config.training.loss_function).strip().lower() in ALLOWED_LOSS_FUNCTIONS,
        "training.loss_function must be one of: mse, huber, pseudo_huber.",
    )
    contrastive_flow_weight = float(config.training.contrastive_flow_weight)
    _check(
        math.isfinite(contrastive_flow_weight)
        and 0.0 <= contrastive_flow_weight < 1.0,
        "training.contrastive_flow_weight must be finite and in [0, 1).",
    )
    if contrastive_flow_weight > 0.0:
        _check(
            str(config.training.objective).strip().lower() == "flow_matching",
            "training.contrastive_flow_weight requires "
            "training.objective='flow_matching'.",
        )
        _check(
            str(config.training.loss_function).strip().lower() == "mse",
            "training.contrastive_flow_weight requires training.loss_function='mse'.",
        )
        _check(
            str(config.training.loss_weighting).strip().lower() == "uniform",
            "training.contrastive_flow_weight requires "
            "training.loss_weighting='uniform'.",
        )
        _check(
            str(config.training.loss_bucket_normalization).strip().lower() == "none",
            "training.contrastive_flow_weight is incompatible with "
            "training.loss_bucket_normalization.",
        )
        _check(
            int(config.training.batch_size) >= 2,
            "training.contrastive_flow_weight requires training.batch_size >= 2 "
            "because negatives are drawn within each microbatch.",
        )
    _check(
        str(config.training.loss_weighting).strip().lower() in ALLOWED_LOSS_WEIGHTING,
        "training.loss_weighting must be one of: adaptive_uncertainty, "
        "cosmap, min_snr_gamma, uniform.",
    )
    adaptive_uncertainty = (
        str(config.training.loss_weighting).strip().lower()
        == "adaptive_uncertainty"
    )
    if adaptive_uncertainty:
        _check(
            str(config.training.objective).strip().lower() == "flow_matching",
            "training.loss_weighting='adaptive_uncertainty' requires "
            "training.objective='flow_matching'.",
        )
        _check(
            str(config.training.loss_function).strip().lower() == "mse",
            "training.loss_weighting='adaptive_uncertainty' requires "
            "training.loss_function='mse'.",
        )
        _check(
            str(config.training.loss_bucket_normalization).strip().lower()
            == "none",
            "training.loss_weighting='adaptive_uncertainty' is incompatible "
            "with training.loss_bucket_normalization.",
        )
        _check(
            str(config.optimizer.type).strip().lower()
            not in _PAIRED_LORA_OPTIMIZER_TYPES.union(
                SELECTED_EXPERT_OPTIMIZER_TYPES
            ),
            "training.loss_weighting='adaptive_uncertainty' requires a "
            "general-purpose optimizer; paired-LoRA and selected-expert "
            "optimizers cannot update the objective-owned uncertainty head.",
        )
    _check(
        str(config.training.loss_bucket_normalization).strip().lower()
        in ALLOWED_LOSS_BUCKET_NORMALIZATION,
        "training.loss_bucket_normalization must be one of: none, per_bucket_mean.",
    )
    _check(
        str(config.training.non_finite_grad_policy).strip().lower() in ALLOWED_NON_FINITE_POLICIES,
        "training.non_finite_grad_policy must be one of: abort, skip_step.",
    )
    _check(
        normalize_checkpointing_mode(config.training.gradient_checkpointing)
        in ALLOWED_CHECKPOINTING_MODES,
        "training.gradient_checkpointing must be one of: off, standard, selective, aggressive.",
    )
    _check(int(config.training.blocks_to_swap) >= 0, "training.blocks_to_swap must be >= 0.")
    _check(int(config.inference.blocks_to_swap) >= 0, "inference.blocks_to_swap must be >= 0.")
    _check(
        str(config.inference.block_swap_mode).strip().lower() in ALLOWED_BLOCK_SWAP_MODES,
        "inference.block_swap_mode must be one of: sync, async.",
    )
    _check(
        int(config.inference.blocks_to_swap) == 0
        or str(config.memory.weight_residency_strategy).strip().lower()
        in {"", "auto", "block_swap", "stream_disk"},
        "inference.blocks_to_swap > 0 requires "
        "memory.weight_residency_strategy='block_swap' or 'stream_disk'.",
    )
    _check(
        int(config.inference.blocks_to_swap) == 0
        or str(config.memory.block_residency_planner).strip().lower()
        in {"", "uniform", "none", "off"},
        "memory.block_residency_planner='phase_aware' is a training-phase policy "
        "and cannot be combined with inference.blocks_to_swap.",
    )
    _check(
        str(config.training.block_swap_mode).strip().lower() in ALLOWED_BLOCK_SWAP_MODES,
        "training.block_swap_mode must be one of: sync, async.",
    )
    _check(
        0.0 < float(config.training.pinned_memory_budget_fraction) <= 1.0,
        "training.pinned_memory_budget_fraction must be in (0, 1].",
    )
    _check(
        0.0 < float(config.memory.cuda_memory_fraction) <= 1.0,
        "memory.cuda_memory_fraction must be in (0, 1].",
    )
    _check(
        float(config.memory.minimum_system_memory_gib) >= 0.0,
        "memory.minimum_system_memory_gib must be >= 0.",
    )
    _check(
        float(config.memory.max_pinned_host_gib) > 0.0,
        "memory.max_pinned_host_gib must be > 0.",
    )
    _check(
        str(config.memory.moe_dispatch).strip().lower()
        in {"vectorized", "legacy", "triton", "triton_persistent"},
        "memory.moe_dispatch must be one of: vectorized, legacy, triton, "
        "triton_persistent.",
    )
    _check(
        str(config.memory.moe_dispatch_preprocess).strip().lower()
        in {"host", "device", "sonic"},
        "memory.moe_dispatch_preprocess must be one of: host, device, sonic.",
    )
    _gemm_backends = {"auto", "bmm", "persistent", "torch_grouped", "deepgemm_fp8"}
    _check(
        str(config.memory.moe_gemm_backend).strip().lower() in _gemm_backends,
        "memory.moe_gemm_backend must be one of: auto, bmm, persistent, "
        "torch_grouped, deepgemm_fp8.",
    )
    for _role in ("forward", "dx", "dw"):
        _value = str(getattr(config.memory, f"moe_gemm_backend_{_role}")).strip().lower()
        _check(
            _value in ({""} | _gemm_backends),
            f"memory.moe_gemm_backend_{_role} must be '' (inherit) or one of: "
            "auto, bmm, persistent, torch_grouped, deepgemm_fp8.",
        )
    _deepgemm_main = str(config.memory.moe_gemm_backend).strip().lower() == "deepgemm_fp8"
    _deepgemm_forward = (
        str(config.memory.moe_gemm_backend_forward).strip().lower() == "deepgemm_fp8"
    )
    _check(
        not _deepgemm_main,
        "deepgemm_fp8 is a forward-only frozen-expert backend; select it with "
        "memory.moe_gemm_backend_forward, not memory.moe_gemm_backend.",
    )
    _check(
        all(
            str(getattr(config.memory, f"moe_gemm_backend_{role}")).strip().lower()
            != "deepgemm_fp8"
            for role in ("dx", "dw")
        ),
        "deepgemm_fp8 may only be selected for memory.moe_gemm_backend_forward.",
    )
    _check(
        not _deepgemm_forward
        or str(config.memory.frozen_weight_quantization).strip().lower() == "fp8",
        "memory.moe_gemm_backend_forward='deepgemm_fp8' requires "
        "memory.frozen_weight_quantization='fp8'.",
    )
    if (
        str(config.memory.moe_dispatch_preprocess).strip().lower()
        in {"device", "sonic"}
        and str(config.memory.packed_state_preload).strip().lower() == "off"
    ):
        warnings.warn(
            "device-resident moe_dispatch_preprocess dequantizes zero-count "
            "experts each step; with memory.packed_state_preload='off' (disk "
            "streaming) that re-reads idle experts off disk. Prefer 'host' "
            "preprocessing or a RAM-resident packed_state_preload.",
            stacklevel=2,
        )
    _check(
        int(config.memory.packed_shard_size_mb) > 0,
        "memory.packed_shard_size_mb must be > 0.",
    )
    _check(
        int(config.memory.int8_workspace_mb) >= 0,
        "memory.int8_workspace_mb must be >= 0.",
    )
    _check(
        str(config.memory.block_residency_planner).strip().lower()
        in {"uniform", "phase_aware"},
        "memory.block_residency_planner must be one of: uniform, phase_aware.",
    )
    _check(
        str(config.memory.block_residency_priority).strip().lower()
        in {"index", "routing_hot"},
        "memory.block_residency_priority must be one of: index, routing_hot.",
    )
    _check(
        1 <= int(config.memory.block_swap_prefetch_depth) <= 4,
        "memory.block_swap_prefetch_depth must be in [1, 4].",
    )
    _check(
        str(config.memory.block_swap_transfer_strategy).strip().lower()
        in {"per_tensor", "flat_ring"},
        "memory.block_swap_transfer_strategy must be one of: per_tensor, flat_ring.",
    )
    _balance_mode = str(config.model.params.moe_balance_mode).strip().lower()
    _check(
        _balance_mode in {"aux_loss", "bias_only", "off"},
        "model.params.moe_balance_mode must be one of: aux_loss, bias_only, off.",
    )
    _check(
        not (
            _balance_mode == "bias_only"
            and float(config.model.params.moe_bias_update_rate) <= 0.0
        ),
        "model.params.moe_balance_mode='bias_only' requires "
        "model.params.moe_bias_update_rate > 0 (bias-only balancing has no "
        "pressure otherwise).",
    )
    _balance_loss_disable_step = int(
        config.model.params.moe_balance_loss_disable_step
    )
    _check(
        _balance_loss_disable_step >= 0,
        "model.params.moe_balance_loss_disable_step must be >= 0.",
    )
    if _balance_loss_disable_step > 0:
        _provider = get_model_family_provider(config.model.type)
        _check(
            _provider is not None
            and _provider.supports_balance_loss_schedule(config),
            f"model.type '{config.model.type}' does not support scheduled "
            "auxiliary balance-loss relaxation.",
        )
        _check(
            _balance_mode == "aux_loss",
            "model.params.moe_balance_loss_disable_step requires "
            "moe_balance_mode='aux_loss'.",
        )
        _check(
            float(config.model.params.moe_aux_loss_weight) > 0.0,
            "model.params.moe_balance_loss_disable_step requires "
            "moe_aux_loss_weight > 0.",
        )
        _check(
            str(config.model.params.moe_aux_loss_type).strip().lower()
            != "disabled",
            "model.params.moe_balance_loss_disable_step requires an enabled "
            "moe_aux_loss_type.",
        )
        _check(
            float(config.model.params.moe_bias_update_rate) == 0.0,
            "model.params.moe_balance_loss_disable_step schedules only the "
            "auxiliary loss and requires moe_bias_update_rate=0.",
        )
    _balance_scope = str(config.model.params.moe_balance_scope).strip().lower()
    _check(
        _balance_scope in {"microbatch", "global_batch"},
        "model.params.moe_balance_scope must be one of: microbatch, global_batch.",
    )
    if bool(config.model.params.moe_balance_grad_ratio_telemetry):
        _provider = get_model_family_provider(config.model.type)
        _check(
            _provider is not None
            and _provider.supports_balance_gradient_ratio_telemetry(config),
            f"model.type '{config.model.type}' does not expose graph-bearing "
            "router probabilities for balance-gradient telemetry.",
        )
        _check(
            normalize_checkpointing_mode(
                config.training.gradient_checkpointing
            )
            != "aggressive",
            "model.params.moe_balance_grad_ratio_telemetry=true cannot use "
            "training.gradient_checkpointing='aggressive': reentrant "
            "checkpointing does not expose graph-bearing router probabilities. "
            "Use standard, selective, or off checkpointing for this diagnostic.",
        )
    if bool(config.model.params.router_fp32_master) and (
        config.adapter.train_router is not True
    ):
        warnings.warn(
            "model.params.router_fp32_master=true has no effect while the router "
            "is not trainable (adapter.train_router is not true): no router "
            "parameters are optimized, so no fp32 master is created. Set "
            "adapter.train_router=true to train the router, or leave "
            "router_fp32_master=false.",
            stacklevel=2,
        )
    _check(
        not (bool(config.training.activation_cpu_offload) and bool(config.training.compile)),
        "training.activation_cpu_offload=true is not supported with "
        "training.compile=true because saved-tensor residency hooks are outside the "
        "compiled graph contract.",
    )
    _check(float(config.optimizer.lr) > 0.0, "optimizer.lr must be > 0.")
    _check(float(config.optimizer.weight_decay) >= 0.0, "optimizer.weight_decay must be >= 0.")
    _check(
        float(config.model.params.moe_cross_layer_coupling_loss_weight) >= 0.0,
        "model.params.moe_cross_layer_coupling_loss_weight must be >= 0.",
    )
    _check(
        float(config.model.params.moe_router_similarity_loss_weight) >= 0.0,
        "model.params.moe_router_similarity_loss_weight must be >= 0.",
    )
    _check(
        int(config.model.params.moe_dynamic_topk_min) >= 0,
        "model.params.moe_dynamic_topk_min must be >= 0.",
    )
    _check(
        float(config.model.params.moe_spatiotemporal_routing_weight) >= 0.0,
        "model.params.moe_spatiotemporal_routing_weight must be >= 0.",
    )
    _check(
        int(config.model.params.moe_spatiotemporal_routing_max_edges) > 0,
        "model.params.moe_spatiotemporal_routing_max_edges must be > 0.",
    )
    dynamic_min = int(config.model.params.moe_dynamic_topk_min)
    dynamic_average = float(config.model.params.moe_dynamic_topk_average)
    _check(
        (dynamic_min == 0 and dynamic_average == 0.0)
        or (
            dynamic_min > 0
            and dynamic_min
            <= dynamic_average
            <= int(config.model.params.experts_per_token)
        ),
        "Dynamic top-k requires min <= average <= experts_per_token, or both "
        "values zero.",
    )
    _check(
        int(config.training.activation_cpu_offload_min_mib) >= 0,
        "training.activation_cpu_offload_min_mib must be >= 0.",
    )
    _check(
        int(config.training.activation_cpu_offload_defer_layers) >= 0,
        "training.activation_cpu_offload_defer_layers must be >= 0.",
    )
    _check(
        int(config.training.activation_cpu_offload_prefetch_layers) >= 0,
        "training.activation_cpu_offload_prefetch_layers must be >= 0.",
    )
    scheduled_activation_offload = bool(
        int(config.training.activation_cpu_offload_defer_layers)
        or int(config.training.activation_cpu_offload_prefetch_layers)
        or bool(config.training.activation_cpu_offload_view_replay)
    )
    _check(
        not scheduled_activation_offload
        or bool(config.training.activation_cpu_offload),
        "Activation offload scheduling and view replay require "
        "training.activation_cpu_offload=true.",
    )
    _check(
        int(config.training.activation_cpu_offload_prefetch_layers) == 0
        or bool(config.training.activation_cpu_offload_pin_memory),
        "Activation offload prefetch requires "
        "training.activation_cpu_offload_pin_memory=true.",
    )
    _check(
        int(config.training.activation_compression_rank) > 0,
        "training.activation_compression_rank must be > 0.",
    )
    _check(
        int(config.training.activation_compression_min_mib) >= 0,
        "training.activation_compression_min_mib must be >= 0.",
    )
    _check(
        not (
            bool(config.training.activation_compression)
            and bool(config.training.activation_cpu_offload)
        ),
        "training.activation_compression and activation_cpu_offload are mutually exclusive.",
    )
    _check(
        float(config.training.activation_cpu_offload_max_gib) >= 0.0,
        "training.activation_cpu_offload_max_gib must be >= 0.",
    )
    _check(
        not bool(config.training.activation_cpu_offload)
        or float(config.training.activation_cpu_offload_max_gib) > 0.0,
        "training.activation_cpu_offload=true requires "
        "training.activation_cpu_offload_max_gib > 0.",
    )
    _check(
        not bool(config.training.activation_cpu_offload_pin_memory)
        or float(config.training.activation_cpu_offload_max_gib)
        <= float(config.memory.max_pinned_host_gib),
        "Pinned activation offload must fit within memory.max_pinned_host_gib.",
    )
    _check(
        float(config.model.params.moe_swiglu_specialization_loss_weight) >= 0.0,
        "model.params.moe_swiglu_specialization_loss_weight must be >= 0.",
    )
    _check(
        float(config.optimizer.loraplus_lr_ratio) > 0.0,
        "optimizer.loraplus_lr_ratio must be > 0.",
    )
    _check(
        str(config.optimizer.type).strip().lower() in ALLOWED_OPTIMIZERS,
        "optimizer.type must be one of: "
        + ", ".join(sorted(ALLOWED_OPTIMIZERS))
        + ".",
    )
    optimizer_type = str(config.optimizer.type).strip().lower()
    stochastic_rounding = bool(config.optimizer.stochastic_rounding)
    lora_pro = (
        optimizer_type == "lora_pro_adamw"
    )
    lora_muon = optimizer_type == "lora_muon"
    selected_expert_muon = optimizer_type in {
        "selected_expert_muon",
        "selected_expert_adamuon",
    }
    paired_lora_optimizer = lora_pro or lora_muon
    _check(
        not stochastic_rounding
        or optimizer_type in _STOCHASTIC_ROUNDING_OPTIMIZER_TYPES,
        "optimizer.stochastic_rounding=true requires optimizer.type to be one "
        "of: "
        + ", ".join(
            repr(value)
            for value in sorted(_STOCHASTIC_ROUNDING_OPTIMIZER_TYPES)
        )
        + ".",
    )
    _check(
        not stochastic_rounding
        or not bool(config.training.optimizer_cpu_offload),
        "optimizer.stochastic_rounding=true requires "
        "training.optimizer_cpu_offload=false; FP32 CPU shadow weights would "
        "bypass BF16 stochastic updates.",
    )
    try:
        MoEOptimizationPolicy.from_memory_config(config.memory)
    except ValueError as exc:
        errors.append(str(exc))
    _check(
        str(config.memory.weight_residency_strategy).strip().lower()
        in {"disabled", "block_swap", "stream_disk"},
        "memory.weight_residency_strategy must be one of: disabled, block_swap, "
        "stream_disk.",
    )
    try:
        resolved_moe_kernel = normalize_moe_kernel_backend(config.memory.moe_kernel_backend)
        expert_access = str(config.memory.expert_weight_access).strip().lower()
        if expert_access == "fused_kernel":
            _check(
                str(config.memory.frozen_weight_quantization).strip().lower() == "int8",
                "memory.expert_weight_access='fused_kernel' requires "
                "memory.frozen_weight_quantization='int8'.",
            )
            _check(
                int(config.memory.expert_dequant_chunk_size) > 1,
                "memory.expert_weight_access='fused_kernel' requires "
                "memory.expert_dequant_chunk_size > 1.",
            )
            _check(
                resolved_moe_kernel in {"auto", "rotated_int8"},
                "memory.expert_weight_access='fused_kernel' requires "
                "memory.moe_kernel_backend='auto' or 'rotated_int8'.",
            )
        if resolved_moe_kernel == "rotated_int8":
            _check(
                str(config.memory.frozen_weight_quantization).strip().lower() == "int8",
                "memory.moe_kernel_backend='rotated_int8' requires "
                "memory.frozen_weight_quantization='int8'.",
            )
            _check(
                str(config.memory.expert_weight_access).strip().lower()
                in {"chunked_dequant", "fused_kernel"},
                "memory.moe_kernel_backend='rotated_int8' requires "
                "memory.expert_weight_access='chunked_dequant' or 'fused_kernel'.",
            )
        if resolved_moe_kernel == "compiled_packed":
            _check(
                str(config.memory.frozen_weight_quantization).strip().lower()
                in {"gguf_iq4", "gguf_iq3", "mxfp8_e4m3", "mxfp4", "nvfp4"},
                "memory.moe_kernel_backend='compiled_packed' requires "
                "GGUF or microscaling frozen weights.",
            )
            _check(
                str(config.memory.expert_weight_access).strip().lower()
                in {"active_dequant", "chunked_dequant"},
                "memory.moe_kernel_backend='compiled_packed' requires "
                "active_dequant or chunked_dequant expert access.",
            )
        if resolved_moe_kernel == "megablocks" and not megablocks_runtime_available():
            errors.append(
                "memory.moe_kernel_backend='megablocks' requires importable "
                "megablocks.ops and grouped_gemm CUDA operators."
            )
        if str(config.memory.frozen_weight_quantization).strip().lower() == "nf4":
            require_module(
                "bitsandbytes.functional",
                reason="memory.frozen_weight_quantization='nf4' requires bitsandbytes "
                "with CUDA 4-bit (NF4) operators.",
            )
    except ValueError as exc:
        errors.append(str(exc))
    _check(
        not (
            str(config.optimizer.type).strip().lower()
            in {"adamw_8bit", "paged_adamw_8bit"}
            and bool(config.training.optimizer_cpu_offload)
        ),
        "optimizer.type='adamw_8bit' requires training.optimizer_cpu_offload=false; "
        "the same applies to paged_adamw_8bit because bitsandbytes kernels require "
        "CUDA-resident state.",
    )
    _check(
        not bool(config.memory.trainable_parameter_offload)
        or str(config.optimizer.type).strip().lower() == "paged_adamw_8bit"
        or bool(config.training.optimizer_cpu_offload),
        "memory.trainable_parameter_offload requires "
        "optimizer.type='paged_adamw_8bit' or training.optimizer_cpu_offload=true.",
    )
    _check(
        str(config.optimizer.scheduler).strip().lower() in ALLOWED_SCHEDULERS,
        "optimizer.scheduler must be one of: "
        + ", ".join(sorted(ALLOWED_SCHEDULERS))
        + ".",
    )
    _check(
        float(config.optimizer.min_lr_ratio) >= 0.0
        and float(config.optimizer.min_lr_ratio) < 1.0,
        "optimizer.min_lr_ratio must be in [0, 1).",
    )
    _check(
        str(config.model.attention_backend).strip().lower() in ALLOWED_ATTENTION_BACKENDS,
        "model.attention_backend must be one of: "
        + ", ".join(sorted(ALLOWED_ATTENTION_BACKENDS))
        + ".",
    )
    _check(
        str(config.adapter.type).strip().lower() in ALLOWED_ADAPTER_TYPES,
        "adapter.type must be 'lora', 'sparse_delta', or 'selected_expert'.",
    )
    _check(
        not bool(str(config.memory.expert_precision_plan_path).strip())
        or str(config.memory.frozen_weight_quantization).strip().lower()
        in {
            "fp8",
            "int8",
            "nf4",
            "gguf_iq4",
            "gguf_iq3",
            "mxfp8_e4m3",
            "mxfp4",
            "nvfp4",
        },
        "memory.expert_precision_plan_path requires frozen expert quantization.",
    )
    structured_sparsity = str(config.memory.expert_structured_sparsity).strip().lower()
    _check(
        structured_sparsity in {"disabled", "2:4"},
        "memory.expert_structured_sparsity must be 'disabled' or '2:4'.",
    )
    _check(
        structured_sparsity != "2:4"
        or str(config.memory.frozen_weight_quantization).strip().lower() == "none",
        "memory.expert_structured_sparsity='2:4' cannot be combined with frozen quantization.",
    )
    _check(
        structured_sparsity != "2:4"
        or str(config.adapter.target_preset).strip().lower() == "attn_only",
        "2:4 expert execution requires adapter.target_preset='attn_only'.",
    )
    direct_expert = str(config.adapter.type).strip().lower() == "selected_expert"
    selected_expert_optimizer = optimizer_type in SELECTED_EXPERT_OPTIMIZER_TYPES
    expert_selection = str(config.adapter.expert_selection).strip().lower()
    esft_selection = expert_selection in {"esft_gate", "esft_token"}
    sparse_delta = str(config.adapter.type).strip().lower() == "sparse_delta"
    _check(
        not sparse_delta
        or 0.0 < float(config.adapter.sparse_delta_density) <= 1.0,
        "adapter.sparse_delta_density must be in (0, 1].",
    )
    _check(
        str(config.adapter.sparse_delta_selection).strip().lower()
        in {"magnitude", "random"},
        "adapter.sparse_delta_selection must be magnitude or random.",
    )
    _check(
        not sparse_delta
        or str(config.adapter.target_preset).strip().lower() == "attn_only",
        "adapter.type='sparse_delta' requires target_preset='attn_only'.",
    )
    _check(
        not direct_expert
        or selected_expert_optimizer,
        "adapter.type='selected_expert' requires a selected-expert optimizer.",
    )
    _check(
        not selected_expert_optimizer or direct_expert,
        "Selected-expert optimizers require adapter.type='selected_expert'.",
    )
    _check(
        not direct_expert
        or esft_selection
        or bool(config.optimizer.selected_expert_ids),
        "Manual adapter.type='selected_expert' requires "
        "optimizer.selected_expert_ids; select esft_gate/esft_token to calibrate "
        "a per-layer plan automatically.",
    )
    _check(
        all(int(value) >= 0 for value in config.optimizer.selected_expert_ids)
        and len(set(config.optimizer.selected_expert_ids))
        == len(config.optimizer.selected_expert_ids),
        "optimizer.selected_expert_ids must contain unique non-negative ids.",
    )
    _check(
        not esft_selection or direct_expert,
        "adapter.expert_selection='esft_gate'/'esft_token' requires "
        "adapter.type='selected_expert'.",
    )
    _check(
        not esft_selection or not bool(config.optimizer.selected_expert_ids),
        "ESFT selection is automatic; optimizer.selected_expert_ids must be empty.",
    )
    if esft_selection:
        esft_provider = get_model_family_provider(config.model.type)
        _check(
            esft_provider is not None
            and esft_provider.supports_esft_expert_selection(config),
            f"model.type='{config.model.type}' does not expose ESFT calibration targets.",
        )
        _check(
            routing_mode == "token_choice",
            "ESFT Equations 6--8 require fixed-cardinality token-choice routing.",
        )
        _check(
            int(config.model.params.moe_zero_experts) == 0
            and int(config.model.params.moe_copy_experts) == 0
            and int(config.model.params.moe_constant_experts) == 0,
            "ESFT calibrates the physical expert bank and is incompatible with "
            "an active lightweight-expert topology.",
        )
        _check(
            not bool(config.training.compile),
            "ESFT pre-optimizer router calibration requires training.compile=false.",
        )
    _check(
        not direct_expert
        or str(config.memory.frozen_weight_quantization).strip().lower() == "none",
        "adapter.type='selected_expert' requires unquantized dense expert weights.",
    )
    _check(
        not direct_expert or not bool(config.training.optimizer_cpu_offload),
        "adapter.type='selected_expert' does not support optimizer CPU offload.",
    )
    if str(config.adapter.expert_selection).strip().lower() == "routing_topk":
        from mirai.core.models.quantization import expert_quantization_formats

        _check(
            str(config.memory.frozen_weight_quantization).strip().lower()
            in expert_quantization_formats(),
            "adapter.expert_selection='routing_topk' requires a quantized expert "
            "format supported by routed experts "
            "(routed-expert LoRA masking exists only on the quantized expert "
            "module; unquantized bf16 expert LoRA silently ignores the selection), "
            "or drop adapter.expert_selection.",
        )
    _check(int(config.adapter.rank) >= 1, "adapter.rank must be >= 1.")
    _check(float(config.adapter.alpha) > 0.0, "adapter.alpha must be > 0.")
    _check(
        0.0 <= float(config.adapter.rank_dropout) <= 1.0,
        "adapter.rank_dropout must be in [0, 1].",
    )
    _check(
        math.isfinite(float(config.adapter.lora_parameter_dropout))
        and 0.0 <= float(config.adapter.lora_parameter_dropout) < 1.0,
        "adapter.lora_parameter_dropout must be finite and in [0, 1).",
    )
    _check(
        int(config.adapter.rank_schedule_start) >= 0,
        "adapter.rank_schedule_start must be >= 0.",
    )
    _check(
        int(config.adapter.rank_schedule_end) >= 0,
        "adapter.rank_schedule_end must be >= 0.",
    )
    _check(
        int(config.adapter.rank_schedule_end) >= int(config.adapter.rank_schedule_start),
        "adapter.rank_schedule_end must be >= adapter.rank_schedule_start.",
    )
    _check(
        0.0 < float(config.adapter.rank_schedule_min_scale) <= 1.0,
        "adapter.rank_schedule_min_scale must be in (0, 1].",
    )
    _check(
        str(config.adapter.timestep_rank_schedule).strip().lower()
        in {"none", "tlora", "tc_gate"},
        "adapter.timestep_rank_schedule must be 'none', 'tlora' or 'tc_gate'.",
    )
    _check(
        int(config.adapter.tc_gate_hidden_dim) >= 1,
        "adapter.tc_gate_hidden_dim must be >= 1.",
    )
    _check(
        0.0 < float(config.adapter.timestep_rank_min_fraction) <= 1.0,
        "adapter.timestep_rank_min_fraction must be in (0, 1].",
    )
    _check(
        bool(str(config.adapter.lora_init).strip()),
        "adapter.lora_init must name a registered LoRA initializer.",
    )
    if paired_lora_optimizer:
        _check(
            str(config.adapter.type).strip().lower() == "lora",
            "Paired LoRA optimizers require adapter.type='lora'.",
        )
        _check(
            float(config.optimizer.loraplus_lr_ratio) == 1.0,
            "Paired LoRA optimizers require optimizer.loraplus_lr_ratio=1.",
        )
        _check(
            not bool(config.training.optimizer_cpu_offload),
            "Paired LoRA optimizers do not support optimizer CPU offload "
            "because their state and updates jointly own each factor pair.",
        )
        _check(
            not bool(config.adapter.use_dora)
            and not bool(config.adapter.use_lora_fa),
            "Paired LoRA optimizers cannot be combined with DoRA or LoRA-FA.",
        )
        _check(
            float(config.adapter.rank_dropout) == 0.0
            and float(config.adapter.lora_parameter_dropout) == 0.0,
            "Paired LoRA optimizers require LoRA dropout to be "
            "disabled because the correction assumes one fixed factor pair.",
        )
        _check(
            int(config.adapter.rank_schedule_start) == 0
            and int(config.adapter.rank_schedule_end) == 0
            and float(config.adapter.rank_schedule_min_scale) == 1.0,
            "Paired LoRA optimizers cannot be combined with a scalar rank schedule.",
        )
        _check(
            str(config.adapter.timestep_rank_schedule).strip().lower() == "none"
            and float(config.adapter.timestep_band_min) == 0.0
            and float(config.adapter.timestep_band_max) == 1.0,
            "Paired LoRA optimizers cannot be combined with timestep rank masks.",
        )
        _check(
            str(config.adapter.expert_selection).strip().lower() == "all",
            "Paired LoRA optimizers require expert_selection='all'.",
        )
        _check(
            int(config.adapter.condenser_rank) == 0,
            "Paired LoRA optimizers require adapter.condenser_rank=0.",
        )
    if lora_pro:
        _check(
            float(config.optimizer.weight_decay) == 0.0,
            "optimizer.type='lora_pro_adamw' requires optimizer.weight_decay=0; "
            "the published decay rule mutates the frozen base weight.",
        )
        _check(
            math.isfinite(float(config.optimizer.lora_pro_damping))
            and float(config.optimizer.lora_pro_damping) > 0.0,
            "optimizer.lora_pro_damping must be finite and > 0.",
        )
    if lora_muon:
        _check(
            math.isfinite(float(config.optimizer.muon_momentum))
            and 0.0 <= float(config.optimizer.muon_momentum) < 1.0,
            "optimizer.muon_momentum must be finite and in [0, 1).",
        )
        _check(
            int(config.optimizer.lora_muon_gauge_rebalance_interval) >= 0,
            "optimizer.lora_muon_gauge_rebalance_interval must be >= 0.",
        )
        _check(
            math.isfinite(
                float(config.optimizer.lora_muon_gauge_rebalance_alpha)
            )
            and 0.0
            < float(config.optimizer.lora_muon_gauge_rebalance_alpha)
            <= 1.0,
            "optimizer.lora_muon_gauge_rebalance_alpha must be in (0, 1].",
        )
        _check(
            float(config.optimizer.weight_decay) == 0.0
            or str(config.optimizer.weight_decay_filter).strip().lower()
            == "none",
            "optimizer.type='lora_muon' with nonzero weight decay requires "
            "optimizer.weight_decay_filter='none' so both factors receive the "
            "paper's split decay.",
        )
    if selected_expert_muon:
        _check(
            math.isfinite(float(config.optimizer.muon_momentum))
            and 0.0 <= float(config.optimizer.muon_momentum) < 1.0,
            "optimizer.muon_momentum must be finite and in [0, 1).",
        )
        _check(
            int(config.optimizer.muon_ns_steps) >= 1,
            "optimizer.muon_ns_steps must be >= 1.",
        )
        _check(
            math.isfinite(float(config.optimizer.muon_eps))
            and float(config.optimizer.muon_eps) > 0.0,
            "optimizer.muon_eps must be finite and > 0.",
        )
        _check(
            math.isfinite(float(config.optimizer.muon_rms_target))
            and float(config.optimizer.muon_rms_target) > 0.0,
            "optimizer.muon_rms_target must be finite and > 0.",
        )
    if str(config.adapter.lora_init).strip().lower() == "lora_ga":
        _check(
            int(config.adapter.lora_ga_calibration_steps) > 0,
            "adapter.lora_ga_calibration_steps must be > 0.",
        )
        _check(
            float(config.adapter.lora_ga_stable_gamma) > 0.0,
            "adapter.lora_ga_stable_gamma must be > 0.",
        )
        _check(
            str(config.adapter.target_preset).strip().lower() == "attn_only",
            "adapter.lora_init='lora_ga' requires "
            "adapter.target_preset='attn_only'.",
        )
        _check(
            str(config.memory.frozen_weight_quantization).strip().lower() == "none",
            "adapter.lora_init='lora_ga' requires unquantized dense calibration weights.",
        )
        _check(
            not bool(config.training.compile),
            "adapter.lora_init='lora_ga' requires training.compile=false.",
        )
        _check(
            not bool(str(config.adapter.init_from).strip()),
            "adapter.lora_init='lora_ga' cannot be combined with adapter.init_from.",
        )
    if str(config.adapter.lora_init).strip().lower() == "gora":
        _check(
            int(config.adapter.gora_calibration_steps) > 0,
            "adapter.gora_calibration_steps must be > 0.",
        )
        _check(
            float(config.adapter.gora_stable_gamma) > 0.0,
            "adapter.gora_stable_gamma must be > 0.",
        )
        _check(
            bool(config.adapter.use_rslora),
            "adapter.lora_init='gora' requires adapter.use_rslora=true.",
        )
        _check(
            not bool(config.adapter.rank_pattern)
            and int(config.adapter.rank_budget) == 0
            and not bool(str(config.adapter.adaptive_rank_plan_path).strip()),
            "adapter.lora_init='gora' owns rank allocation and cannot be combined "
            "with rank_pattern, rank_budget, or adaptive_rank_plan_path.",
        )
        _check(
            not bool(config.adapter.use_lora_fa),
            "adapter.lora_init='gora' cannot be combined with adapter.use_lora_fa.",
        )
        _check(
            int(config.adapter.condenser_rank) == 0,
            "adapter.lora_init='gora' requires adapter.condenser_rank=0.",
        )
        _check(
            str(config.adapter.timestep_rank_schedule).strip().lower() == "none"
            and float(config.adapter.timestep_band_min) == 0.0
            and float(config.adapter.timestep_band_max) == 1.0,
            "adapter.lora_init='gora' cannot be combined with timestep rank masks.",
        )
        _check(
            not bool(config.adapter.sparse_expert_export),
            "adapter.lora_init='gora' cannot use sparse_expert_export.",
        )
        _check(
            str(config.memory.frozen_weight_quantization).strip().lower() == "none"
            and str(config.memory.frozen_weight_quantization).strip().lower() != "fp8"
            and str(
                config.memory.frozen_weight_quantization_strategy
            ).strip().lower()
            in {"", "none", "disabled", "auto"}
            and not bool(
                str(config.memory.frozen_weight_packed_state_path).strip()
            ),
            "adapter.lora_init='gora' requires unquantized dense calibration weights.",
        )
        _check(
            not bool(config.training.compile),
            "adapter.lora_init='gora' requires training.compile=false.",
        )
        _check(
            not bool(str(config.adapter.init_from).strip()),
            "adapter.lora_init='gora' cannot be combined with adapter.init_from.",
        )
    if bool(config.adapter.use_dora):
        dora_provider = get_model_family_provider(config.model.type)
        _check(
            str(config.adapter.type).strip().lower() == "lora",
            "adapter.use_dora=true requires adapter.type='lora'.",
        )
        _check(
            dora_provider is not None and dora_provider.supports_dora(config),
            f"model.type='{config.model.type}' does not expose weight-space DoRA targets.",
        )
        _check(
            str(config.adapter.expert_tensor_lora_backend).strip().lower()
            == "weight_space",
            "adapter.use_dora=true requires "
            "adapter.expert_tensor_lora_backend='weight_space'.",
        )
        _check(
            not bool(config.adapter.use_lora_fa),
            "adapter.use_dora=true cannot be combined with adapter.use_lora_fa.",
        )
        _check(
            float(config.adapter.rank_dropout) == 0.0,
            "adapter.use_dora=true requires adapter.rank_dropout=0 because "
            "input-space rank dropout does not define one normalized weight.",
        )
        _check(
            str(config.adapter.timestep_rank_schedule).strip().lower() == "none"
            and float(config.adapter.timestep_band_min) == 0.0
            and float(config.adapter.timestep_band_max) == 1.0,
            "adapter.use_dora=true cannot be combined with per-sample timestep "
            "rank masks because DoRA normalizes one shared weight direction.",
        )
        _check(
            str(config.adapter.expert_selection).strip().lower() == "all"
            and not bool(config.adapter.sparse_expert_export),
            "adapter.use_dora=true requires expert_selection='all' and "
            "sparse_expert_export=false.",
        )
        _check(
            int(config.adapter.condenser_rank) == 0,
            "adapter.use_dora=true requires adapter.condenser_rank=0.",
        )
        _check(
            str(config.adapter.lora_init).strip().lower()
            not in {"loftq", "lora_ga", "gora"},
            "adapter.use_dora=true supports kaiming, orthogonal, "
            "PiSSA, or EVA initialization; LoftQ and gradient-derived "
            "initializers require a different calibration derivative.",
        )
        _check(
            str(config.memory.frozen_weight_quantization).strip().lower() == "none"
            and str(config.memory.frozen_weight_quantization).strip().lower() != "fp8"
            and str(
                config.memory.frozen_weight_quantization_strategy
            ).strip().lower()
            in {"", "none", "disabled", "auto"}
            and not bool(
                str(config.memory.frozen_weight_packed_state_path).strip()
            ),
            "adapter.use_dora=true requires unpacked frozen weights.",
        )
    _check(
        not bool(config.adapter.use_lora_fa)
        or (
            float(config.adapter.rank_dropout) == 0.0
            and float(config.adapter.lora_parameter_dropout) == 0.0
            and str(config.adapter.timestep_rank_schedule).strip().lower() == "none"
            and float(config.adapter.timestep_band_min) == 0.0
            and float(config.adapter.timestep_band_max) == 1.0
        ),
        "adapter.use_lora_fa requires a fixed A projection and fixed B factor; "
        "it cannot be combined with LoRA dropout or timestep rank masks.",
    )
    _check(
        0.0 <= float(config.adapter.timestep_band_min)
        and float(config.adapter.timestep_band_min)
        < float(config.adapter.timestep_band_max)
        and float(config.adapter.timestep_band_max) <= 1.0,
        "adapter timestep band requires 0 <= timestep_band_min < "
        "timestep_band_max <= 1.",
    )
    _check(
        0.0 <= float(config.dataset.online_tag_shuffle_dropout) <= 1.0,
        "dataset.online_tag_shuffle_dropout must be in [0, 1].",
    )
    _check(
        int(config.dataset.online_tag_shuffle_keep_first_n_tags) >= 0,
        "dataset.online_tag_shuffle_keep_first_n_tags must be >= 0.",
    )
    _check(int(config.dataset.num_workers) >= 0, "dataset.num_workers must be >= 0.")
    _check(int(config.dataset.prefetch_factor) >= 1, "dataset.prefetch_factor must be >= 1.")
    _check(
        0.0 <= float(config.dataset.max_cache_skip_ratio) <= 1.0,
        "dataset.max_cache_skip_ratio must be in [0, 1].",
    )
    errors.extend(validate_training_policy_configs(config))
    _check(
        int(config.logging.save_every_n_steps) >= 1,
        "logging.save_every_n_steps must be >= 1.",
    )
    _check(
        float(config.logging.async_checkpoint_max_gib) >= 0.0,
        "logging.async_checkpoint_max_gib must be >= 0.",
    )
    _check(
        not bool(config.logging.async_checkpoint)
        or float(config.logging.async_checkpoint_max_gib) > 0.0,
        "logging.async_checkpoint=true requires "
        "logging.async_checkpoint_max_gib > 0.",
    )
    _check(
        int(config.logging.sample_every_n_steps) >= 0,
        "logging.sample_every_n_steps must be >= 0.",
    )
    _check(
        int(config.logging.sample_frame_count) >= 1,
        "logging.sample_frame_count must be >= 1.",
    )
    _check(float(config.logging.sample_cfg_scale) >= 0.0, "logging.sample_cfg_scale must be >= 0.")
    _check(
        int(config.logging.sample_blocks_to_swap) >= -1,
        "logging.sample_blocks_to_swap must be >= -1.",
    )
    _check(
        str(config.logging.sample_solver).strip().lower() in ALLOWED_SAMPLE_SOLVERS,
        "logging.sample_solver must be one of: "
        + ", ".join(sorted(ALLOWED_SAMPLE_SOLVERS))
        + ".",
    )
    if bool(config.training.ema_enabled):
        _check(
            0.0 < float(config.training.ema_decay) <= 1.0,
            "training.ema_decay must be in (0, 1] when ema_enabled=true.",
        )
    _posthoc_snapshot_interval = int(
        config.training.posthoc_ema_snapshot_every_n_steps
    )
    _check(
        _posthoc_snapshot_interval >= 0,
        "training.posthoc_ema_snapshot_every_n_steps must be >= 0.",
    )
    if bool(config.training.posthoc_ema_enabled):
        from mirai.core.training.optim.posthoc_ema import normalize_profile_stds

        _check(
            _posthoc_snapshot_interval > 0,
            "training.posthoc_ema_snapshot_every_n_steps must be > 0 when "
            "posthoc_ema_enabled=true.",
        )
        try:
            normalize_profile_stds(config.training.posthoc_ema_profile_stds)
        except ValueError as exc:
            errors.append(str(exc))
    else:
        _check(
            _posthoc_snapshot_interval == 0,
            "training.posthoc_ema_snapshot_every_n_steps must be 0 when "
            "posthoc_ema_enabled=false.",
        )

    if errors:
        raise ValueError("Invalid training runtime configuration:\n- " + "\n- ".join(errors))

    if bool(config.logging.tensorboard):
        require_module(
            "torch.utils.tensorboard",
            reason="logging.tensorboard=true requires the 'tensorboard' package to be installed.",
        )
    if bool(config.logging.wandb):
        require_module(
            "wandb",
            reason="logging.wandb=true requires the 'wandb' package to be installed.",
        )
    if int(config.logging.sample_every_n_steps) > 0:
        has_video_writer = True
        try:
            importlib.import_module("av")
        except ModuleNotFoundError:
            has_video_writer = False
        if not has_video_writer:
            require_module(
                "PIL",
                reason=(
                    "logging.sample_every_n_steps > 0 requires either 'av' for MP4 previews "
                    "or 'Pillow' for PNG fallback previews."
                ),
            )
