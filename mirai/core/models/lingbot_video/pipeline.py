"""Native LingBot-Video MoE denoiser pipeline."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping

from mirai.config.schema import MemoryConfig, ModelConfig, TrainingConfig
from mirai.core.models.base import MemoryFeatureCapabilities
from mirai.core.models.base import ModelExtensionCapabilities
from mirai.core.models.flow import (
    apply_rectified_flow_noise,
    clamp_timesteps,
    rectified_flow_target,
    shifted_sigma,
)
from mirai.core.models.flow_shift import DynamicFlowShiftPolicy
from mirai.core.models.compressed_weights import CompressedGroupedExperts
from mirai.core.models.compressed_weights import CompressedWeightReport
from mirai.core.models.compressed_weights import combine_compressed_weights_reports
from mirai.core.models.compressed_weights import get_compressed_weights_packed_state_quant_formats
from mirai.core.models.compressed_weights import load_compressed_weights_packed_state_file
from mirai.core.models.compressed_weights import NF4_BLOCKSIZE
from mirai.core.models.compressed_weights import normalize_compressed_weights_strategy
from mirai.core.models.compressed_weights import normalize_quant_format
from mirai.core.models.compressed_weights import prepare_compressed_weights_modules_from_manifest
from mirai.core.models.compressed_weights import quantize_compressed_weights_modules
from mirai.core.models.compressed_weights import apply_structured_2_4_experts
from mirai.core.models.compressed_weights.execution.mixed_precision import (
    MixedPrecisionGroupedExperts,
)
from mirai.core.models.compressed_weights.quantization.learned_rotation import (
    validate_learned_rotation_selection,
)
from mirai.core.moe.calibration.precision import ExpertPrecisionPlan
from mirai.core.moe.calibration.precision import TensorPrecisionPlan
from mirai.core.moe.calibration.precision import load_precision_plan
from mirai.core.moe.calibration.router_repair import router_tensor_fingerprint
from mirai.core.moe.calibration.esft import ESFTCalibrationTarget
from mirai.core.models.compressed_weights import read_compressed_weights_packed_state_manifest
from mirai.core.models.compressed_weights import prepare_compressed_weights_modules_for_checkpoint_load
from mirai.core.models.compressed_weights.execution.expert_device_cache import (
    ExpertDeviceCache,
)
from mirai.core.models.compressed_weights.factorization.prototype_projection import (
    CompressedExpertProjectionSource,
)
from mirai.core.models.lingbot_video.checkpoints import LingBotVideoCheckpointReport
from mirai.core.models.lingbot_video.checkpoints import load_lingbot_transformer_checkpoint
from mirai.core.models.lingbot_video.checkpoints import load_lingbot_transformer_config
from mirai.core.models.lingbot_video.checkpoints import resolve_lingbot_transformer_dir
from mirai.core.models.lingbot_video.checkpoints import discover_lingbot_transformer_checkpoint_keys
from mirai.core.models.lingbot_video.cache import LingBotVideoNativeCacheEncoder
from mirai.core.models.lingbot_video.dispersive_loss import (
    configure_lingbot_dispersive_loss,
)
from mirai.core.models.lingbot_video.inference import LingBotVideoNativeInference
from mirai.core.models.lingbot_video.expert_specialization import clear_checkpoint_auxiliary_terms
from mirai.core.models.lingbot_video.expert_specialization import build_lingbot_router_specialization_runtime
from mirai.core.models.lingbot_video.expert_specialization import expert_orthogonality_auxiliary_losses as _orthogonality_losses
from mirai.core.models.lingbot_video.intermediate_specialization import (
    build_lingbot_intermediate_specialization_runtime,
    clear_checkpoint_intermediate_terms,
)
from mirai.core.models.lingbot_video.router_runtime import _clear_router_runtime_state
from mirai.core.models.lingbot_video.router_runtime import _collect_router_auxiliary_terms
from mirai.core.models.lingbot_video.router_runtime import (
    _expert_choice_coverage_metrics,
)
from mirai.core.models.lingbot_video.router_runtime import _build_phi_balance_runtime
from mirai.core.models.lingbot_video.router_runtime import (
    _build_balance_gradient_probe,
)
from mirai.core.models.lingbot_video.router_runtime import _build_expert_orthogonality_runtime
from mirai.core.models.lingbot_video.router_runtime import _phi_balance_auxiliary_terms
from mirai.core.models.lingbot_video.router_runtime import _router_assignment_counts
from mirai.core.models.lingbot_video.router_runtime import _routing_stats
from mirai.core.models.lingbot_video.router_runtime import _router_similarity_terms
from mirai.core.models.lingbot_video.router_runtime import _spatiotemporal_routing_terms
from mirai.core.models.lingbot_video.router_runtime import _install_expert_output_observer
from mirai.core.models.lingbot_video.router_runtime import _weighted_router_auxiliary_losses
from mirai.core.models.lingbot_video.routing_health import collect_lingbot_routing_health
from mirai.core.models.lingbot_video.preemptive_monitoring import (
    collect_lingbot_attention_monitoring,
)
from mirai.core.moe.monitoring.preemptive import PreemptiveAttentionMonitor
from mirai.core.models.lingbot_video.route_extensions import bind_lingbot_route_extensions
from mirai.core.models.lingbot_video.route_extensions import (
    configure_lingbot_diversity_routing,
    configure_lingbot_expert_dropout,
    configure_lingbot_prototypical_routing,
    configure_lingbot_router_temperature,
    configure_lingbot_selective_sinkhorn,
    configure_lingbot_sharp_moe,
)
from mirai.core.models.lingbot_video.router_stage import bind_lingbot_router_stage_policy
from mirai.core.models.lingbot_video.router_stage import configure_lingbot_router_stage_policy
from mirai.core.models.lingbot_video.router_distillation import bind_lingbot_router_distillation
from mirai.core.models.lingbot_video.router_distillation import collect_lingbot_router_distillation_terms
from mirai.core.models.lingbot_video.router_distillation import configure_lingbot_router_distillation
from mirai.core.models.lingbot_video.simbal import bind_lingbot_simbal
from mirai.core.models.lingbot_video.simbal import configure_lingbot_simbal
from mirai.core.models.lingbot_video.domain_specialization import configure_lingbot_domain_expert_specialization
from mirai.core.models.adapters.expert_tensor_lora import install_expert_tensor_lora_executor
from mirai.core.models.adapters.lora import LoRAApplicationReport
from mirai.core.models.adapters.lora import apply_lora_to_expert_tensors
from mirai.core.models.adapters.lora import apply_lora_to_linear_modules
from mirai.core.models.adapters.lora import collect_lora_expert_target_names
from mirai.core.models.adapters.lora import collect_lora_linear_target_names
from mirai.core.models.adapters.lora import iter_lora_modules
from mirai.core.models.adapters.lora import iter_lora_expert_tensor_modules
from mirai.core.models.adapters.lora import load_lora_state_dict
from mirai.core.models.adapters.lora import lora_state_dict
from mirai.core.models.adapters.lora import collect_lora_adapter_modules
from mirai.core.models.adapters.lora import enable_expert_lora_condensers
from mirai.core.models.adapters.lora import set_lora_rank_dropout
from mirai.core.models.adapters.lora import set_lora_parameter_dropout
from mirai.core.models.adapters.lora import set_lora_rank_schedule_scale
from mirai.core.models.adapters.lora import set_lora_scale
from mirai.core.models.adapters.lora import set_lora_timestep_rank_masks
from mirai.core.models.adapters.lora import set_lora_tc_gate
from mirai.core.models.adapters.lora_allocation import LoRAAllocationPolicy
from mirai.core.models.adapters.lora_adaptive_rank import AdaptiveRankPlanLineageHost
from mirai.core.models.adapters.lora_adaptive_rank import resolve_adaptive_rank_plan
from mirai.core.models.adapters.lora_initialization import initializer_requires_quantization_error
from mirai.core.models.adapters.lora_initialization import validate_lora_initializer
from mirai.core.models.adapters.sparse_delta import (
    SparseDeltaLinear,
    apply_sparse_delta_to_linear_modules,
)
from mirai.core.models.adapters.lora_fa import apply_lora_fa
from mirai.core.models.adapters.timestep_axis import TimestepAdapterPolicy
from mirai.core.models.adapters.timestep_axis import per_sample_rank_mask
from mirai.core.models.adapters.tc_lora import TimestepGateHypernet
from mirai.core.models.adapters.tc_lora import gate_summary
from mirai.core.models.moe_dit_common import as_latent_tensor
from mirai.core.models.native_video import NativeVideoPipeline, VideoLatentLayout
from mirai.core.models.providers import (
    ModelFamilyProvider,
    get_model_family_provider,
    register_model_family_provider,
)
from mirai.core.models.providers import NativeCacheEncoderConfig
from mirai.core.moe.routing.contracts import SparseMoECapabilities
from mirai.core.moe.routing.adjugate_experts import (
    ADJUGATE_EXPERT_STATE_PREFIX,
    AdjugateExpertPool,
    AdjugateExpertTopology,
    export_adjugate_expert_state,
    load_adjugate_expert_state,
)
from mirai.core.moe.routing.decoupled import (
    DECOUPLED_ROUTING_STATE_PREFIX,
    DecoupledRouterConditioner,
    export_decoupled_routing_state,
    load_decoupled_routing_state,
)
from mirai.core.moe.routing.chain_of_experts import (
    CHAIN_OF_EXPERTS_STATE_PREFIX,
    ChainOfExpertsExtension,
    ChainOfExpertsSpec,
    chain_of_experts_metrics,
    export_chain_of_experts_state,
    load_chain_of_experts_state,
)
from mirai.core.moe.routing.expert_choice import resolve_capacity_schedule
from mirai.core.moe.routing.layer_policy import LayerRouterPolicy
from mirai.core.moe.routing.progressive_sparsification import (
    ProgressiveSparsificationPolicy,
)
from mirai.core.moe.routing.timestep_capacity import (
    TimestepExpertChoiceCapacityPolicy,
)
from mirai.core.moe.routing.lightweight_experts import (
    LIGHTWEIGHT_EXPERT_STATE_PREFIX,
    LightweightExpertPool,
    export_lightweight_expert_state,
    load_lightweight_expert_state,
)
from mirai.core.moe.routing.routers import route_expert_choice_logits
from mirai.core.moe.monitoring.summary import (
    summarize_routing_by_diffusion_timestep,
    summarize_routing_stats,
)
from mirai.core.moe.artifacts.manifest import DEFAULT_DOWNLOAD_MANIFEST
from mirai.core.moe.runtime.specs import ExpertTensorSpec
from mirai.core.moe.runtime.specs import ExpertMLPExecutionSpec
from mirai.core.moe.runtime.specs import ExpertProjectionRole
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy
from mirai.core.moe.runtime.specs import normalize_expert_weight_access_policy
from mirai.core.training.runtime.compilation import (
    CompilationRegion,
    TokenBucketPlan,
)
from mirai.core.training.residency.activation_offload import (
    ActivationOffloadRegion,
)
from mirai.core.moe.adaptation.balance import normalize_moe_balance_mode
from mirai.core.moe.adaptation.balance import resolve_moe_balance_weights
from mirai.core.moe.adaptation.balance_schedule import (
    AuxiliaryBalanceLossSchedule,
)
from mirai.core.moe.adaptation.global_balance import GlobalBatchLoadAccumulator
from mirai.core.moe.adaptation.global_balance import dispatch_counts
from mirai.core.moe.adaptation.global_balance import normalize_moe_balance_scope
from mirai.core.moe.calibration.projection import PrototypeCalibrationTarget
from mirai.core.moe.calibration.pruning import ExpertPruningCalibrationTarget
from mirai.core.moe.calibration.flexmoe import FlexMoECalibrationTarget
from mirai.core.moe.calibration.router_repair import RouterRepairTarget
from mirai.core.moe.calibration.router_repair import (
    apply_router_repair_artifact,
)
from mirai.core.moe.calibration.router_repair import (
    load_router_repair_artifact,
)
from mirai.core.models.compressed_weights import packed_artifact_fingerprint
from mirai.core.moe.storage.upcycling import validate_drop_upcycling_selection
from mirai.core.moe.monitoring.agreement import RoutingSelectionTarget
from mirai.core.moe.monitoring.gradient_ratio import BalanceGradientProbe
from mirai.core.moe.storage.physical_weights import validate_physical_weight_provider_selection
from mirai.core.moe.calibration.quantization import QuantizationCalibrationTarget
from mirai.core.moe.calibration.router_quantization import (
    RouterLinearCalibrationBatch,
)
from mirai.core.moe.calibration.router_quantization import (
    RouterQuantizationCalibrationTarget,
)
from mirai.core.moe.calibration.router_quantization import (
    apply_router_quantization_calibration,
)
from mirai.core.moe.calibration.router_quantization import (
    load_router_quantization_calibration,
)
from mirai.core.moe.calibration.whitening import ExpertWhiteningCalibrationTarget
from mirai.core.moe.calibration.imatrix import ExpertImportanceCalibrationTarget
from mirai.core.training.optim.router_fp32_master import RouterFp32Master
from mirai.core.moe.adaptation.dataset_routing import DatasetRoutingBatch
from mirai.core.moe.adaptation.dataset_routing import DatasetRoutingPolicy
from mirai.core.moe.adaptation.domain_specialization import (
    DomainExpertSpecializationController,
)
from mirai.core.moe.adaptation.diversity import DiversityAwareRoutingController
from mirai.core.moe.adaptation.dropout import ExpertDropoutController
from mirai.core.moe.adaptation.simbal import SimBalController
from mirai.core.moe.adaptation.temperature import RouterTemperatureController
from mirai.core.moe.routing.dynamic_topk import BudgetedDynamicTopK
from mirai.core.moe.routing.selective_sinkhorn import SelectiveSinkhornController
from mirai.core.moe.routing.prototypical import (
    PROTOTYPICAL_ROUTING_STATE_PREFIX,
    PrototypicalRouterExtension,
    PrototypicalRoutingSpec,
    collect_prototypical_routing_losses,
    export_prototypical_routing_state,
    load_prototypical_routing_state,
    prototypical_routing_diagnostics,
)
from mirai.core.moe.routing.saliency import (
    SHARP_MOE_STATE_PREFIX,
    SaliencyHarnessingRouter,
    SharpMoESpec,
    export_sharp_moe_state,
    load_sharp_moe_state,
)
from mirai.core.moe.routing.depth import MixtureOfDepthsSpec
from mirai.core.models.lingbot_video.depth_extensions import (
    configure_lingbot_depth_policy,
    mixture_of_depths_diagnostics,
)
from mirai.core.moe.adaptation.stage_schedule import RouterStageScheduleController
from mirai.core.moe.adaptation.distillation import RouterDistillationController
from mirai.core.moe.runtime.specs import validate_expert_tensor_specs
from mirai.core.moe.runtime.kernels import build_moe_kernel_backend
from mirai.core.moe.runtime.token_chunking import MoETokenChunkPolicy
from mirai.core.training.policies.dispersive_loss import DispersiveLossController
from mirai.core.moe.adaptation import phi_balance as phi_balance_state
from mirai.core.moe.routing.subset import RouterSubsetPolicy
from mirai.core.moe.monitoring.health import DeadlockTracker
from mirai.core.moe.monitoring.agreement import SelectionMarginProbe
from mirai.core.moe.monitoring.drift import RouterLogitReferenceTracker
from mirai.core.moe.monitoring.drift import load_router_drift_checkpoint_state
from mirai.core.moe.monitoring.drift import router_drift_checkpoint_state
from mirai.core.moe.adaptation.router_training import RouterAdapterBinding
from mirai.core.moe.adaptation.router_training import RouterTrainingPolicy
from mirai.core.moe.artifacts.verification import verify_downloaded_snapshot
from mirai.core.training.residency.block_swap import BlockSwapManager
from mirai.core.training.residency.device_residency import DeviceResidencyPlanner
from mirai.core.training.residency.residency_plan import block_scores_from_router_loads
from mirai.core.training.residency.tensor_residency import cast_trainable_tensors
from mirai.core.training.residency.tensor_residency import move_trainable_tensors
from mirai.core.training.residency.tensor_residency import move_tensors_outside_modules
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoAttention
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoMLP
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoRouter
from mirai.vendors.lingbot_video.transformer_lingbot_video import (
    LingBotVideoRuntimeOptions,
    LingBotVideoSparseMoeBlock,
    set_lingbot_video_runtime_options,
)
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


LINGBOT_EXPERT_MLP_EXECUTION_SPEC = ExpertMLPExecutionSpec(
    projections=(
        ExpertProjectionRole("gate", "w1"),
        ExpertProjectionRole("up", "w3"),
        ExpertProjectionRole("down", "w2"),
    ),
    activation="silu",
    combiner="gated_product",
)


VALID_VARIANTS = {
    "scratch",
    "tiny-video",
    "lingbot-video-moe-30b-a3b",
}
SCRATCH_VARIANTS = {"scratch", "tiny-video"}
SUPPORTED_STRATEGIES = {
    "text_to_video",
    "image_to_video",
    "hybrid_conditioning",
    "multi_task_video",
}
LORA_TARGET_PRESETS = {
    "attn_only": [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
    ],
    "shared_mlp_only": [
        "ffn.shared_experts.gate_proj",
        "ffn.shared_experts.up_proj",
        "ffn.shared_experts.down_proj",
        "ffn.gate_proj",
        "ffn.up_proj",
        "ffn.down_proj",
    ],
    "attn_shared_mlp": [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "ffn.shared_experts.gate_proj",
        "ffn.shared_experts.up_proj",
        "ffn.shared_experts.down_proj",
        "ffn.gate_proj",
        "ffn.up_proj",
        "ffn.down_proj",
    ],
    "routed_experts_only": [
        "ffn.experts.w1",
        "ffn.experts.w2",
        "ffn.experts.w3",
    ],
    "attn_routed_experts": [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "ffn.experts.w1",
        "ffn.experts.w2",
        "ffn.experts.w3",
    ],
    "attn_router_routed_experts": [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "ffn.router.weight",
        "ffn.experts.w1",
        "ffn.experts.w2",
        "ffn.experts.w3",
    ],
    "all_linear": ["*"],
}


def _head_axes_dims(hidden_size: int, heads: int) -> tuple[int, int, int]:
    head_dim = int(hidden_size) // int(heads)
    if head_dim < 6:
        raise ValueError("LingBot-Video head_dim must be at least 6.")
    first = max(2, (head_dim // 4) // 2 * 2)
    second = max(2, (head_dim // 4) // 2 * 2)
    third = head_dim - first - second
    if third <= 0 or third % 2:
        first = 2
        second = 2
        third = head_dim - 4
    if third <= 0 or third % 2:
        raise ValueError("LingBot-Video head_dim must be decomposable into even RoPE axes.")
    return (first, second, third)


def _tuple_int(value: Any, *, length: int) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        items = tuple(int(v) for v in value)
    else:
        items = (int(value),)
    if len(items) == length:
        return items
    if len(items) == 1 and length == 3:
        return (1, items[0], items[0])
    raise ValueError(f"Expected {length} values, got {items}.")


def _parse_resolution_pair(value: Any) -> tuple[int, int] | None:
    text = str(value or "").strip().lower()
    if not text or text == "smallest_bucket" or "x" not in text:
        return None
    left, right = text.split("x", 1)
    return int(left.strip()), int(right.strip())


def _parse_dataset_bucket_resolution(value: Any) -> tuple[int, int]:
    text = str(value or "").strip().lower()
    if "x" not in text:
        raise ValueError(f"expected 'HxW', got '{value}'.")
    left, right = text.split("x", 1)
    return int(left.strip()), int(right.strip())


def _append_layout_request_error(
    errors: list[str],
    *,
    layout: VideoLatentLayout,
    field: str,
    frame_count: int,
    height: int,
    width: int,
) -> None:
    try:
        layout.validate_request(
            frame_count=frame_count,
            height=height,
            width=width,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"{field} is incompatible with LingBot-Video latent layout: {exc}"
        )


def _as_lingbot_latents(
    value: Any,
    config: dict[str, Any],
    *,
    label: str,
    dtype: Any | None = None,
    device: Any | None = None,
) -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("LingBot-Video requires torch.")
    if torch.is_tensor(value):
        tensor = value
        if dtype is not None or device is not None:
            tensor = tensor.to(
                dtype=dtype if dtype is not None else tensor.dtype,
                device=device if device is not None else tensor.device,
            )
    else:
        tensor = torch.tensor(value, dtype=dtype or torch.float32, device=device)
    if tensor.ndim != 5:
        raise ValueError(
            f"LingBot-Video {label} must be cached video latents shaped "
            f"[B,C,T,H,W]; got shape {tuple(tensor.shape)}."
        )
    batch, channels, frames, height, width = [int(v) for v in tensor.shape]
    expected_channels = int(config["in_channels"])
    if batch <= 0 or frames <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"LingBot-Video {label} must have positive B,T,H,W dimensions; "
            f"got shape {tuple(tensor.shape)}."
        )
    if channels != expected_channels:
        raise ValueError(
            f"LingBot-Video {label} has {channels} latent channels, but the "
            f"transformer is configured for {expected_channels}."
        )
    patch_t, patch_h, patch_w = _tuple_int(config.get("patch_size", (1, 2, 2)), length=3)
    if frames % patch_t != 0 or height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            f"LingBot-Video {label} shape {tuple(tensor.shape)} is incompatible "
            f"with patch_size={(patch_t, patch_h, patch_w)}; T,H,W must be divisible "
            "by the corresponding patch axes."
        )
    return tensor


def _tiny_config(params: Any) -> dict[str, Any]:
    hidden = int(params.hidden_size)
    heads = int(params.attention_heads)
    return {
        "patch_size": (1, int(params.patch_size), int(params.patch_size)),
        "in_channels": int(params.latent_channels),
        "out_channels": int(params.latent_channels),
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "depth": int(params.num_layers),
        "intermediate_size": hidden * 4,
        "text_dim": hidden,
        "freq_dim": max(8, hidden),
        "axes_dims": _head_axes_dims(hidden, heads),
        "num_experts": int(params.num_experts),
        "num_experts_per_tok": int(params.experts_per_token),
        "moe_intermediate_size": max(4, hidden * 2),
        "decoder_sparse_step": 1,
        "n_shared_experts": int(params.shared_experts),
        "score_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": None,
        "topk_group": None,
        "routed_scaling_factor": 1.0,
    }


def _transformer_config(model_config: ModelConfig) -> dict[str, Any]:
    subfolder = str(getattr(model_config.params, "denoiser_subfolder", "transformer") or "transformer")
    transformer_dir = resolve_lingbot_transformer_dir(
        model_config.path,
        subfolder=subfolder,
    )
    if transformer_dir is not None:
        config = load_lingbot_transformer_config(model_config.path, subfolder=subfolder)
    elif str(model_config.params.variant).strip().lower() in SCRATCH_VARIANTS:
        config = _tiny_config(model_config.params)
    else:
        raise FileNotFoundError(
            "LingBot-Video requires native transformer assets under model.path; "
            f"no {subfolder}/config.json or transformer directory was found at "
            f"{model_config.path!r}."
        )
    for key in ("patch_size", "axes_dims", "axes_lens", "mlp_only_layers"):
        if key in config:
            expected = 3 if key in {"patch_size", "axes_dims", "axes_lens"} else len(config[key])
            config[key] = _tuple_int(config[key], length=expected)
    return config


def _text_embeddings(text_embeds: dict[str, Any], *, batch: int, text_dim: int, like: Any) -> Any:
    value = text_embeds.get("lingbot")
    if value is None:
        value = text_embeds.get("t5")
    if torch is None:  # pragma: no cover
        raise RuntimeError("LingBot-Video requires torch.")
    if value is None:
        return torch.zeros((batch, 1, text_dim), device=like.device, dtype=like.dtype)
    tensor = as_latent_tensor(value, dtype=like.dtype, device=like.device)
    if tensor.ndim > 3:
        tensor = tensor.reshape(tensor.shape[0], -1)
    if tensor.ndim == 1:
        tensor = tensor.reshape(batch, 1, 1)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    if tensor.shape[0] != batch:
        tensor = tensor.reshape(1, -1, tensor.shape[-1]).expand(batch, -1, -1)
    if tensor.shape[-1] == text_dim:
        return tensor
    scalar = tensor.reshape(batch, -1).mean(dim=1, keepdim=True).unsqueeze(-1)
    return scalar.expand(batch, 1, text_dim).contiguous()


def _text_attention_mask(text_embeds: dict[str, Any], *, batch: int, like: Any) -> Any:
    value = text_embeds.get("text_mask")
    if value is None:
        value = text_embeds.get("attention_mask")
    if value is None:
        return None
    if torch is None:  # pragma: no cover
        raise RuntimeError("LingBot-Video requires torch.")
    if torch.is_tensor(value):
        mask = value.to(device=like.device, dtype=torch.bool)
    else:
        mask = torch.tensor(value, device=like.device, dtype=torch.bool)
    if mask.ndim > 2:
        mask = mask.reshape(mask.shape[0], -1)
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    if mask.shape[0] != batch:
        mask = mask.reshape(1, -1).expand(batch, -1)
    return mask.contiguous()


def _trim_encoder_states_to_prefix_mask(
    encoder_hidden_states: Any,
    encoder_attention_mask: Any,
) -> tuple[Any, Any]:
    if encoder_attention_mask is None:
        return encoder_hidden_states, None
    lengths = encoder_attention_mask.long().sum(dim=1)
    seq_len = int(encoder_attention_mask.shape[1])
    positions = torch.arange(seq_len, device=encoder_attention_mask.device).reshape(1, seq_len)
    expected = positions < lengths.reshape(-1, 1)
    if not torch.equal(encoder_attention_mask.bool(), expected):
        raise ValueError(
            "LingBot-Video text_mask must use prefix padding: all valid tokens "
            "must precede padded tokens."
        )
    max_len = max(1, int(lengths.max().detach().cpu().item()))
    return (
        encoder_hidden_states[:, :max_len, :].contiguous(),
        encoder_attention_mask[:, :max_len].contiguous(),
    )


def _initialize_scratch_weights(module: nn.Module) -> None:
    norm_weight_ids = {
        id(weight)
        for child in module.modules()
        if (
            isinstance(child, nn.LayerNorm)
            or type(child).__name__.endswith("RMSNorm")
        )
        for weight in (getattr(child, "weight", None),)
        if isinstance(weight, nn.Parameter)
    }
    for name, param in module.named_parameters():
        if "scale_shift_table" in name:
            nn.init.zeros_(param)
        elif id(param) in norm_weight_ids:
            nn.init.ones_(param)
        elif param.ndim >= 2:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)


def _replace_child_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def _shape_of_expert_tensor(module: nn.Module, tensor_name: str) -> tuple[int, ...]:
    shape_provider = getattr(module, "expert_weight_shape", None)
    if callable(shape_provider):
        return tuple(int(dim) for dim in shape_provider(tensor_name))
    buffer_name = f"{tensor_name}_int8"
    if hasattr(module, buffer_name):
        return tuple(int(dim) for dim in getattr(module, buffer_name).shape)
    tensor = getattr(module, tensor_name)
    return tuple(int(dim) for dim in tensor.shape)


def _dtype_of_expert_tensor(module: nn.Module, tensor_name: str) -> str:
    buffer_name = f"{tensor_name}_int8"
    if hasattr(module, buffer_name):
        return str(getattr(module, buffer_name).dtype)
    if isinstance(module, CompressedGroupedExperts) and module.has_packed_weight(tensor_name):
        return str(torch.int8)
    return str(getattr(module, tensor_name).dtype)


def _is_lingbot_grouped_experts(module: nn.Module) -> bool:
    return bool(getattr(module, "mirai_expert_tensor_host", False))


def _expert_access_from_policy(policy: MoEOptimizationPolicy) -> str:
    access = normalize_expert_weight_access_policy(policy.expert_weight_access)
    if access in {"auto", "disabled"}:
        return "full_dequant"
    return access


def _build_router_quantization_targets(
    training_model: nn.Module,
) -> dict[str, RouterQuantizationCalibrationTarget]:
    """Expose family tensor semantics through the generic calibration contract."""
    targets: dict[str, RouterQuantizationCalibrationTarget] = {}
    for module_name, module in training_model.named_modules():
        if not isinstance(module, LingBotVideoSparseMoeBlock):
            continue
        router = module.router
        name = f"{module_name}.router" if module_name else "router"

        def _read_weight(*, _router=router):
            weight = _router._parameters.get("weight")
            if weight is None:
                raise RuntimeError(
                    "Router quantization calibration requires floating-point "
                    "source weights."
                )
            return weight

        def _capture_batch(
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
            *,
            _router=router,
        ) -> RouterLinearCalibrationBatch:
            hidden_states = (
                kwargs.get("hidden_states")
                if "hidden_states" in kwargs
                else (args[0] if args else None)
            )
            if hidden_states is None:
                raise ValueError("Sparse-MoE calibration call has no hidden states.")
            additive = None
            features = hidden_states
            if (
                _router._expert_choice_extension is not None
                and _router.decoupled_routing is not None
            ):
                router_input = (
                    kwargs.get("router_input")
                    if "router_input" in kwargs
                    else (args[2] if len(args) > 2 else None)
                )
                timestep_input = (
                    kwargs.get("timestep_router_input")
                    if "timestep_router_input" in kwargs
                    else (args[3] if len(args) > 3 else None)
                )
                if router_input is None or timestep_input is None:
                    raise ValueError(
                        "Decoupled routing calibration requires content and "
                        "timestep router inputs."
                    )
                features = router_input
                expanded_timestep = timestep_input.expand_as(router_input)
                conditioner = _router.decoupled_routing
                additive = (
                    F.linear(
                        expanded_timestep.float(),
                        conditioner.timestep_projection.float(),
                    )
                    * float(conditioner.timestep_weight)
                )
            return RouterLinearCalibrationBatch(
                features=features.reshape(-1, features.shape[-1]),
                additive_logits=(
                    None
                    if additive is None
                    else additive.reshape(-1, additive.shape[-1])
                ),
            )

        def _install_scale(scale: Any, *, _router=router) -> None:
            _router.enable_int8_weight(calibrated_scale=scale)

        targets[name] = RouterQuantizationCalibrationTarget(
            name=name,
            observation_module=module,
            num_experts=int(router.num_experts),
            input_features=int(router.weight.shape[1]),
            top_k=int(router.top_k),
            read_weight=_read_weight,
            capture_batch=_capture_batch,
            install_int8_scale=_install_scale,
        ).validate()
    if not targets:
        raise ValueError(
            "LingBot router quantization calibration found no sparse-MoE routers."
        )
    return dict(sorted(targets.items()))


class LingBotVideoModelFamilyProvider(ModelFamilyProvider):
    def __init__(self, model_type: str) -> None:
        super().__init__(
            model_type=model_type,
            native=True,
            sparse_moe=True,
            strict_native_assets_by_default=True,
            native_cache_encoding=True,
            dora_supported=True,
            dataset_routing_policy=True,
            diversity_routing_policy=True,
            expert_dropout_policy=True,
            router_temperature_policy=True,
            layer_router_policy=True,
            progressive_sparsification=True,
            chain_of_experts=True,
            dispersive_loss_policy=True,
            simbal_policy=True,
            selective_sinkhorn_policy=True,
            prototypical_routing_policy=True,
            sharp_moe_policy=True,
            mixture_of_depths_policy=True,
            preemptive_monitoring=True,
            balance_loss_schedule=True,
            grouped_adjugate_experts=True,
            balance_gradient_ratio_telemetry=True,
            router_stage_schedule=True,
            router_distillation=True,
            domain_expert_specialization=True,
            esft_expert_selection=True,
            expert_pruning_calibration=True,
            flexmoe_calibration=True,
            prototype_calibration=True,
            moe_quantization_calibration=True,
            router_quantization_calibration=True,
            expert_whitening_calibration=True,
            expert_precision_calibration=True,
            routing_mode_agreement_evidence=True,
            post_compression_router_repair=True,
            config_defaults_name="lingbot_video",
            release_supported=True,
            release_eligible=True,
            inference_prompt_rewriters=("lingbot_json",),
            batched_cfg_inference=True,
            inference_tasks=(
                "text_to_video",
                "text_to_image",
                "image_to_video",
                "video_to_video",
            ),
            dataset_caption_formats=("raw", "lingbot_json"),
            expert_mlp_execution_spec=LINGBOT_EXPERT_MLP_EXECUTION_SPEC,
        )

    def resolve_dataset_caption(self, caption: str, *, caption_format: str) -> str:
        from mirai.core.models.lingbot_video.prompting import resolve_training_caption

        self.validate_dataset_caption_format(caption_format)
        return resolve_training_caption(caption, caption_format=caption_format)

    def validate_release_native_smoke_evidence(
        self,
        payload: dict[str, Any],
        *,
        runtime_policy: dict[str, Any],
        label: str = "native_smoke_report",
    ) -> list[str]:
        failures: list[str] = []
        snapshot = payload.get("snapshot_verification")
        denoiser_subfolder = str(
            runtime_policy.get("denoiser_subfolder", "transformer")
            if isinstance(runtime_policy, dict)
            else "transformer"
        ).strip() or "transformer"
        expected_component_id = f"denoiser_subfolder:{denoiser_subfolder}"
        if not isinstance(snapshot, dict):
            failures.append(
                f"{label}: LingBot release evidence requires snapshot_verification."
            )
        elif str(snapshot.get("status", "")).strip().lower() != "verified":
            failures.append(
                f"{label}: LingBot snapshot_verification.status must be 'verified'."
            )
        else:
            if not str(snapshot.get("manifest_path", "")).strip():
                failures.append(
                    f"{label}: LingBot snapshot_verification.manifest_path is required."
                )
            if int(snapshot.get("file_count", 0)) <= 0:
                failures.append(
                    f"{label}: LingBot snapshot_verification.file_count must be > 0."
                )
            if int(snapshot.get("total_bytes", 0)) <= 0:
                failures.append(
                    f"{label}: LingBot snapshot_verification.total_bytes must be > 0."
                )
            if str(snapshot.get("denoiser_subfolder", "")).strip() != denoiser_subfolder:
                failures.append(
                    f"{label}: LingBot snapshot_verification.denoiser_subfolder "
                    f"must be '{denoiser_subfolder}'."
                )
            if str(snapshot.get("model_component_id", "")).strip() != expected_component_id:
                failures.append(
                    f"{label}: LingBot snapshot_verification.model_component_id "
                    f"must be '{expected_component_id}'."
                )
        return failures

    def validate_native_backend_availability(self, cfg: TrainingConfig) -> list[str]:
        variant = str(cfg.model.params.variant).strip().lower()
        if variant not in SCRATCH_VARIANTS:
            model_path = Path(str(cfg.model.path))
            subfolder = str(getattr(cfg.model.params, "denoiser_subfolder", "transformer") or "transformer")
            transformer_dir = resolve_lingbot_transformer_dir(model_path, subfolder=subfolder)
            if transformer_dir is None:
                return [
                    "LingBot-Video strict native assets require model.path to point "
                    f"to a model root with {subfolder}/config.json, "
                    "transformer/config.json, or directly to the transformer directory."
                ]
            model_root = transformer_dir.parent if transformer_dir.name == "transformer" else transformer_dir
            has_download_manifest = (model_root / DEFAULT_DOWNLOAD_MANIFEST).exists()
            has_partial_files = any(model_root.rglob("*.part"))
            if has_download_manifest or has_partial_files:
                try:
                    verify_downloaded_snapshot(
                        model_root,
                        expected_variant=str(cfg.model.params.variant),
                    )
                except Exception as exc:
                    return [f"LingBot-Video downloaded snapshot verification failed: {exc}"]
        return []

    def build_native_cache_encoder(
        self,
        config: NativeCacheEncoderConfig,
    ) -> LingBotVideoNativeCacheEncoder | None:
        if not bool(config.enabled):
            return None
        return LingBotVideoNativeCacheEncoder(config)

    def build_prototype_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, PrototypeCalibrationTarget]:
        training_model = pipeline.get_training_model()
        if training_model is None:
            raise ValueError(
                "LingBot prototype calibration requires an exposed training model."
            )
        targets: dict[str, PrototypeCalibrationTarget] = {}
        for module_name, module in training_model.named_modules():
            if not isinstance(module, CompressedGroupedExperts):
                continue
            target = PrototypeCalibrationTarget(
                name=str(module_name),
                host=module,
                projection_source=CompressedExpertProjectionSource(module),
            ).validate()
            targets[target.name] = target
        if not targets:
            raise ValueError(
                "LingBot prototype calibration requires compressed_weights grouped experts; "
                "enable frozen expert quantization before calibration."
            )
        return targets

    def build_esft_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, ESFTCalibrationTarget]:
        training_model = pipeline.get_training_model()
        if training_model is None:
            raise ValueError(
                "LingBot ESFT calibration requires an exposed training model."
            )
        targets: dict[str, ESFTCalibrationTarget] = {}
        for module_name, module in training_model.named_modules():
            router = getattr(module, "router", None)
            experts = getattr(module, "experts", None)
            num_experts = getattr(experts, "num_experts", None)
            if router is None or num_experts is None:
                continue
            target_name = f"{module_name}.experts" if module_name else "experts"
            target = ESFTCalibrationTarget(
                name=target_name,
                router=router,
                num_experts=int(num_experts),
            ).validate()
            targets[target_name] = target
        if not targets:
            raise ValueError("LingBot ESFT calibration found no grouped expert hosts.")
        return targets

    def build_router_repair_targets(
        self,
        pipeline: Any,
    ) -> dict[str, RouterRepairTarget]:
        return pipeline._router_repair_targets()

    def build_expert_pruning_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, ExpertPruningCalibrationTarget]:
        training_model = pipeline.get_training_model()
        if training_model is None:
            raise ValueError(
                "LingBot expert-pruning calibration requires an exposed training model."
            )
        targets: dict[str, ExpertPruningCalibrationTarget] = {}
        for module_name, module in training_model.named_modules():
            router = getattr(module, "router", None)
            experts = getattr(module, "experts", None)
            setter = getattr(module, "set_expert_output_observer", None)
            num_experts = getattr(experts, "num_experts", None)
            if router is None or num_experts is None or not callable(setter):
                continue
            name = f"{module_name}.experts" if module_name else "experts"
            target = ExpertPruningCalibrationTarget(
                name=name,
                host=module,
                num_experts=int(num_experts),
            ).validate()
            targets[target.name] = target
        if not targets:
            raise ValueError(
                "LingBot expert-pruning calibration found no routed expert hosts."
            )
        return targets

    def build_flexmoe_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, FlexMoECalibrationTarget]:
        training_model = pipeline.get_training_model()
        if training_model is None:
            raise ValueError(
                "LingBot FlexMoE calibration requires an exposed training model."
            )
        targets: dict[str, FlexMoECalibrationTarget] = {}
        for module_name, module in training_model.named_modules():
            if not isinstance(module, CompressedGroupedExperts):
                continue
            shape = module.expert_weight_shape("w1")
            target = FlexMoECalibrationTarget(
                name=str(module_name),
                host=module,
                num_experts=int(shape[0]),
                intermediate_size=int(shape[1]),
            ).validate()
            targets[target.name] = target
        if not targets:
            raise ValueError(
                "LingBot FlexMoE calibration requires full packed grouped experts."
            )
        return targets

    def build_moe_quantization_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, QuantizationCalibrationTarget]:
        training_model = pipeline.get_training_model()
        if training_model is None:
            raise ValueError(
                "LingBot quantization calibration requires an exposed training model."
            )
        targets: dict[str, QuantizationCalibrationTarget] = {}
        for module_name, module in training_model.named_modules():
            router = getattr(module, "router", None)
            experts = getattr(module, "experts", None)
            if router is None or not isinstance(experts, CompressedGroupedExperts):
                continue
            name = f"{module_name}.experts" if module_name else "experts"
            target = QuantizationCalibrationTarget(
                name=name,
                router=router,
                num_experts=int(experts.num_experts),
                logical_to_physical=experts.logical_to_physical_map(),
            ).validate()
            targets[name] = target
        if not targets:
            raise ValueError(
                "LingBot quantization calibration requires compressed grouped experts."
            )
        return targets

    def build_router_quantization_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, RouterQuantizationCalibrationTarget]:
        training_model = pipeline.get_training_model()
        if training_model is None:
            raise ValueError(
                "LingBot router quantization calibration requires an exposed "
                "training model."
            )
        return _build_router_quantization_targets(training_model)

    def build_expert_whitening_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, ExpertWhiteningCalibrationTarget]:
        training_model = pipeline.get_training_model()
        if training_model is None:
            raise ValueError(
                "LingBot expert whitening requires an exposed training model."
            )
        targets: dict[str, ExpertWhiteningCalibrationTarget] = {}
        for module_name, module in training_model.named_modules():
            if not isinstance(module, CompressedGroupedExperts):
                continue
            shapes = {
                key: tuple(int(value) for value in module.expert_weight_shape(key))
                for key in ("w1", "w2", "w3")
            }
            name = str(module_name)
            target = ExpertWhiteningCalibrationTarget(
                name=name,
                host=module,
                projection_input_dims={
                    key: int(shape[-1]) for key, shape in shapes.items()
                },
            ).validate()
            targets[name] = target
        if not targets:
            raise ValueError(
                "LingBot expert whitening requires compressed grouped experts."
            )
        return targets

    def build_expert_precision_calibration_targets(
        self,
        pipeline: Any,
    ) -> dict[str, ExpertImportanceCalibrationTarget]:
        training_model = pipeline.get_training_model()
        if training_model is None:
            raise ValueError(
                "LingBot expert precision calibration requires an exposed "
                "training model."
            )
        targets: dict[str, ExpertImportanceCalibrationTarget] = {}
        for module_name, module in training_model.named_modules():
            if not isinstance(module, LingBotVideoSparseMoeBlock):
                continue
            experts = module.experts
            if not all(
                isinstance(getattr(experts, key, None), torch.Tensor)
                for key in ("w1", "w2", "w3")
            ):
                raise ValueError(
                    "Expert precision calibration requires floating grouped "
                    "expert weights."
                )
            name = f"{module_name}.experts"
            targets[name] = ExpertImportanceCalibrationTarget(
                name=name,
                host=module,
                weights={
                    key: getattr(experts, key)
                    for key in ("w1", "w2", "w3")
                },
            ).validate()
        if not targets:
            raise ValueError(
                "LingBot expert precision calibration found no grouped experts."
            )
        return targets

    def build_routing_mode_agreement_targets(
        self,
        pipeline: Any,
    ) -> dict[str, RoutingSelectionTarget]:
        training_model = pipeline.get_training_model()
        if training_model is None:
            raise ValueError(
                "LingBot routing agreement requires an exposed training model."
            )
        targets: dict[str, RoutingSelectionTarget] = {}
        for module_name, module in training_model.named_modules():
            router = getattr(module, "router", None)
            experts = getattr(module, "experts", None)
            num_experts = getattr(experts, "num_experts", None)
            if router is None or num_experts is None:
                continue
            name = f"{module_name}.router" if module_name else "router"
            target = RoutingSelectionTarget(
                name=name,
                router=router,
                num_experts=int(num_experts),
            ).validate()
            targets[name] = target
        if not targets:
            raise ValueError(
                "LingBot routing agreement found no routed expert hosts."
            )
        return targets


class LingBotVideoPipeline(nn.Module, AdaptiveRankPlanLineageHost, NativeVideoPipeline):
    """Native LingBot-Video denoiser training pipeline for cached latents/text."""

    def __init__(
        self,
        model_config: ModelConfig,
        memory_config: MemoryConfig | None = None,
        timestep_capacity_policy: TimestepExpertChoiceCapacityPolicy | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("LingBotVideoPipeline requires torch.")
        super().__init__()
        self.model_config = model_config
        self._flow_shift_policy = DynamicFlowShiftPolicy(
            mode=str(getattr(model_config.params, "flow_shift_mode", "constant")),
            base_shift=float(model_config.params.flow_shift),
            base_seq_len=int(
                getattr(model_config.params, "flow_shift_base_seq_len", 256)
            ),
            max_seq_len=int(
                getattr(model_config.params, "flow_shift_max_seq_len", 4096)
            ),
            max_shift=float(
                getattr(model_config.params, "flow_shift_max", 12.0)
            ),
        )
        self._moe_optimization_policy = (
            MoEOptimizationPolicy.from_memory_config(memory_config)
            if memory_config is not None
            else MoEOptimizationPolicy()
        )
        self._expert_device_cache = ExpertDeviceCache()
        self.transformer_config = _transformer_config(model_config)
        aux_loss_type = str(model_config.params.moe_aux_loss_type).strip().lower()
        self._moe_aux_loss_type = (
            "sequence" if aux_loss_type == "model_native" else aux_loss_type
        )
        self._moe_balance_mode = normalize_moe_balance_mode(
            getattr(model_config.params, "moe_balance_mode", "aux_loss")
        )
        _effective_aux_weight, _effective_bias_rate = resolve_moe_balance_weights(
            self._moe_balance_mode,
            aux_loss_weight=float(model_config.params.moe_aux_loss_weight),
            bias_update_rate=float(model_config.params.moe_bias_update_rate),
        )
        self._moe_aux_loss_base_weight = _effective_aux_weight
        self._moe_aux_loss_weight = _effective_aux_weight
        self._moe_balance_loss_schedule = (
            AuxiliaryBalanceLossSchedule.from_model_params(model_config.params)
        )
        self._moe_balance_loss_step = 0
        self._moe_router_z_loss_weight = float(
            model_config.params.moe_router_z_loss_weight
        )
        self._layer_router_policy = LayerRouterPolicy.from_model_params(
            model_config.params
        )
        self._progressive_sparsification_policy: (
            ProgressiveSparsificationPolicy | None
        ) = None
        self._progressive_sparsification_step = 0
        self._moe_router_similarity_loss_weight = float(
            model_config.params.moe_router_similarity_loss_weight
        )
        self._moe_spatiotemporal_routing_weight = float(
            model_config.params.moe_spatiotemporal_routing_weight
        )
        self._moe_spatiotemporal_routing_max_edges = int(
            model_config.params.moe_spatiotemporal_routing_max_edges
        )
        self._moe_bias_update_rate = _effective_bias_rate
        self._moe_bias_centering = bool(model_config.params.moe_bias_centering)
        # Global-batch load balancing (opt-in). "microbatch" (default) leaves the
        # per-micro-batch estimate untouched; "global_batch"
        # accumulates the token-fraction load across the accumulation window.
        self._moe_balance_scope = normalize_moe_balance_scope(
            getattr(model_config.params, "moe_balance_scope", "microbatch")
        )
        self._global_batch_load_accumulator: GlobalBatchLoadAccumulator | None = (
            GlobalBatchLoadAccumulator()
            if self._moe_balance_scope == "global_batch"
            else None
        )
        # Opt-in fp32 master copy of trainable router params (built lazily once
        # trainability is resolved). None disables the master-copy path.
        self._router_fp32_master_enabled = bool(
            getattr(model_config.params, "router_fp32_master", False)
        )
        self._router_fp32_master: RouterFp32Master | None = None
        self._router_fp32_master_built = False
        self._pending_router_loads: dict[str, Any] = {}
        self._gradient_checkpointing = "off"
        # Native text-encoder / VAE inference assets (built lazily at first use so
        # training paths never touch them). None until an inference hook is called.
        self._native_inference: LingBotVideoNativeInference | None = None
        # The refiner is a separate same-class checkpoint loaded on demand.
        self._refiner: Any | None = None
        self._compute_autocast_dtype = torch.float32
        self._last_auxiliary_losses: dict[str, Any] = {}
        self._last_diagnostics: dict[str, Any] = {}
        self._dispersive_loss_runtime: Any | None = None
        self._simbal_runtime: Any | None = None
        self._balance_gradient_ratio_enabled = bool(
            getattr(
                model_config.params,
                "moe_balance_grad_ratio_telemetry",
                False,
            )
        )
        self._last_balance_gradient_probe: BalanceGradientProbe | None = None
        params = model_config.params
        self._runtime_options = LingBotVideoRuntimeOptions(
            moe_expert_backend=str(params.moe_expert_backend).strip().lower(),
            moe_pad_backend=str(params.moe_pad_backend).strip().lower(),
            moe_reorder_backend=str(params.moe_reorder_backend).strip().lower(),
            moe_restore_backend=str(params.moe_restore_backend).strip().lower(),
            moe_restore_chunk_size=int(params.moe_restore_chunk_size),
            fused_qkv_linear=bool(params.moe_fused_qkv_linear),
            inference_bf16_fastmath=bool(params.inference_bf16_fastmath),
        )
        self._inference_routing_telemetry = bool(
            params.inference_routing_telemetry
        )
        self._inference_routing_telemetry_layer_stride = int(
            params.inference_routing_telemetry_layer_stride
        )
        # Optional eval routing traces are retained on CPU for offline analysis.
        self._inference_routing_trace: list[dict[str, Any]] = []
        self._inference_routing_forward_idx = 0
        self._checkpoint_report: LingBotVideoCheckpointReport | None = None
        self._compressed_weights_report: CompressedWeightReport | None = None
        self._quantized_experts_on_load = False
        self._lora_report: LoRAApplicationReport | None = None
        self._direct_expert_tuning = False
        self._selected_expert_plan: dict[str, tuple[int, ...]] = {}
        self._sparse_delta_tuning = False
        self._lora_allocation_fingerprint: tuple[tuple[str, int, float, bool], ...] = ()
        self._gora_allocation_fingerprint = ""
        self._gora_ranks: dict[str, int] = {}
        self._adaptive_rank_plan: Any | None = None
        self._lora_init = "kaiming"
        self._use_lora_fa = False
        self._use_dora = False
        self._expert_tensor_lora_backend = "weight_space"
        self._lora_scale = 1.0
        self._rank_schedule_scale = 1.0
        self._adapter_type = "lora"
        self._sparse_expert_export = False
        # Timestep-axis adapter policy (T-LoRA rank schedule / timestep bands).
        # None disables timestep adapter shaping.
        self._timestep_adapter_policy: TimestepAdapterPolicy | None = None
        self._timestep_adapter_modules: list[Any] = []
        # TC-LoRA (arXiv 2510.09561) learned timestep gate. Built only when
        # timestep_rank_schedule == "tc_gate"; None disables the gate. The
        # gate is attached as a transformer submodule so it rides device
        # placement and appears in get_trainable_parameters automatically.
        self._tc_gate: Any | None = None
        self._tc_gate_hidden_dim = 8
        # Stochastic per-step expert-subset routing (opt-in; None = disabled ->
        # default routing). Routers are cached at adapter-config
        # time and assigned a stable per-layer index for deterministic sampling.
        self._router_subset_policy: RouterSubsetPolicy | None = (
            RouterSubsetPolicy.from_model_params(model_config.params)
        )
        if (
            self._router_subset_policy is None
            and self._layer_router_policy is not None
            and any(
                band.subset_fraction is not None
                and float(band.subset_fraction) < 1.0
                for band in self._layer_router_policy.bands
            )
        ):
            self._router_subset_policy = RouterSubsetPolicy(
                fraction=float(model_config.params.expert_subset_fraction),
                pool_factor=float(model_config.params.expert_subset_pool_factor),
                anneal_steps=int(model_config.params.expert_subset_anneal_steps),
                kl_weight=float(
                    model_config.params.expert_subset_router_kl_weight
                ),
            )
        self._moe_routing_mode = str(
            getattr(model_config.params, "moe_routing_mode", "token_choice")
        ).strip().lower()
        self._chain_of_experts_enabled = bool(
            getattr(model_config.params, "moe_chain_of_experts", False)
        )
        self._chain_router_rank = int(
            getattr(model_config.params, "moe_chain_router_rank", 0)
        )
        self._expert_choice_capacity_factor = float(
            getattr(model_config.params, "moe_expert_choice_capacity_factor", 1.0)
        )
        self._expert_choice_capacity_schedule = tuple(
            dict(item)
            for item in getattr(
                model_config.params, "moe_expert_choice_capacity_schedule", ()
            )
        )
        self._timestep_capacity_policy = (
            timestep_capacity_policy
            if timestep_capacity_policy is not None
            else TimestepExpertChoiceCapacityPolicy(
                schedule=str(
                    getattr(
                        model_config.params,
                        "moe_expert_choice_timestep_capacity_schedule",
                        "disabled",
                    )
                ),
                capacity_factor_span=float(
                    getattr(
                        model_config.params,
                        "moe_expert_choice_timestep_capacity_span",
                        0.0,
                    )
                ),
                flow_shift=float(model_config.params.flow_shift),
            )
        )
        self._expert_choice_runtime_sigmas: Any | None = None
        self._expert_choice_runtime_flow_shifts: Any | None = None
        self._expert_choice_coverage_alarm_threshold = float(
            getattr(
                model_config.params,
                "moe_expert_choice_coverage_alarm_threshold",
                0.0,
            )
        )
        self._expert_choice_timestep_weight = float(
            getattr(model_config.params, "moe_router_timestep_weight", 0.0)
        )
        self._expert_choice_step = 0
        self._zero_expert_count = int(
            getattr(model_config.params, "moe_zero_experts", 0)
        )
        self._copy_expert_count = int(
            getattr(model_config.params, "moe_copy_experts", 0)
        )
        self._constant_expert_count = int(
            getattr(model_config.params, "moe_constant_experts", 0)
        )
        self._lightweight_expert_count = (
            self._zero_expert_count
            + self._copy_expert_count
            + self._constant_expert_count
        )
        self._lightweight_top_k = int(
            getattr(model_config.params, "moe_lightweight_top_k", 0)
        )
        self._adjugate_experts_enabled = bool(
            getattr(model_config.params, "moe_adjugate_experts", False)
        )
        self._adjugate_expert_groups = int(
            getattr(model_config.params, "moe_adjugate_expert_groups", 0)
        )
        self._adjugate_expert_intermediate_size = int(
            getattr(
                model_config.params,
                "moe_adjugate_expert_intermediate_size",
                128,
            )
        )
        self._adjugate_expert_scale = float(
            getattr(model_config.params, "moe_adjugate_expert_scale", 0.05)
        )
        self._diversity_routing_controller: DiversityAwareRoutingController | None = None
        self._expert_dropout_controller: ExpertDropoutController | None = None
        self._router_temperature_controller: RouterTemperatureController | None = None
        self._selective_sinkhorn_controller: SelectiveSinkhornController | None = None
        self._prototypical_routing_spec: PrototypicalRoutingSpec | None = None
        self._sharp_moe_spec: SharpMoESpec | None = None
        self._mixture_of_depths_spec: MixtureOfDepthsSpec | None = None
        dynamic_min = int(model_config.params.moe_dynamic_topk_min)
        self._dynamic_topk_controller: BudgetedDynamicTopK | None = (
            BudgetedDynamicTopK(
                min_k=dynamic_min,
                average_k=float(model_config.params.moe_dynamic_topk_average),
            )
            if dynamic_min > 0
            else None
        )
        self._router_stage_schedule_controller: RouterStageScheduleController | None = None
        self._router_distillation_controller: RouterDistillationController | None = None
        self._moe_router_modules: list[Any] = []
        self._dataset_routing_policy = DatasetRoutingPolicy(
            specialization_mode="emergent",
            domain_metadata_key="",
            expert_affinity={},
            routing_prior_weight=0.0,
            router_warmup_steps=0,
        )
        self._routing_step0_snapshot: dict[str, Any] | None = None
        # Routing-health diagnostics pack (opt-in). When disabled nothing is
        # computed. The deadlock tracker holds
        # the only cross-step state (per-layer single-expert streak counters).
        self._routing_health_enabled = bool(
            getattr(model_config.params, "moe_routing_health", False)
        )
        self._deadlock_tracker: DeadlockTracker | None = (
            DeadlockTracker() if self._routing_health_enabled else None
        )
        self._preemptive_attention_monitor: PreemptiveAttentionMonitor | None = None
        self._selection_margin_probe: SelectionMarginProbe | None = (
            SelectionMarginProbe()
            if bool(getattr(model_config.params, "moe_selection_margin", False))
            else None
        )
        self._moe_phi_balance_weight, self._phi_balance_controller = _build_phi_balance_runtime(
            model_config.params,
            legacy_aux_weight=self._moe_aux_loss_weight,
            legacy_bias_rate=self._moe_bias_update_rate,
        )
        self._router_specialization_runtime = (
            build_lingbot_router_specialization_runtime(model_config.params)
        )
        self._intermediate_specialization_runtime = (
            build_lingbot_intermediate_specialization_runtime(model_config.params)
        )
        (
            self._moe_expert_orthogonality_loss_weight,
            self._expert_output_orthogonality_capture,
        ) = _build_expert_orthogonality_runtime(model_config.params)
        self._router_drift_tracker = RouterLogitReferenceTracker()
        self._train_router_override: bool | None = None
        self._condenser_rank = 0
        self._condenser_alpha = 1.0
        self._condenser_init = "kaiming"
        self._weight_residency_strategy = "disabled"
        self._trainable_parameter_offload = False
        self._optimizer_compute_device = torch.device("cpu")
        self._block_swap_manager: BlockSwapManager | None = None
        residency_budget_gib = float(
            getattr(memory_config, "device_residency_budget_gib", 0.0)
            if memory_config is not None
            else 0.0
        )
        self._device_residency_planner = DeviceResidencyPlanner(
            int(residency_budget_gib * (1024**3))
        )
        self._block_swap_state: dict[str, Any] = {
            "enabled": False,
            "blocks_to_swap": 0,
            "mode": "sync",
            "block_swap_backward": True,
            "weight_residency_strategy": "disabled",
            "h2d_only_frozen_base": False,
            "events": [],
        }
        variant = str(model_config.params.variant).strip().lower()
        self._frozen_weight_quantization = (
            str(getattr(memory_config, "frozen_weight_quantization", "none") or "none").strip().lower()
            if memory_config is not None
            else "none"
        )
        self._nf4_blocksize = (
            int(getattr(memory_config, "quantization_block_size", NF4_BLOCKSIZE) or NF4_BLOCKSIZE)
            if memory_config is not None
            else NF4_BLOCKSIZE
        )
        packed_state_path = (
            str(getattr(memory_config, "frozen_weight_packed_state_path", "") or "").strip()
            if memory_config is not None
            else ""
        )
        router_repair_artifact_path = str(
            getattr(model_config.params, "router_repair_artifact_path", "") or ""
        ).strip()
        router_repair_mode = str(
            getattr(
                model_config.params,
                "post_compression_router_repair",
                "off",
            )
        ).strip().lower()
        if router_repair_artifact_path and router_repair_mode != "router_kd":
            raise ValueError(
                "LingBot router repair artifact loading requires "
                "post_compression_router_repair='router_kd'."
            )
        if router_repair_artifact_path and not packed_state_path:
            raise ValueError(
                "LingBot router repair artifacts require a compressed packed base."
            )
        load_quantized_experts = (
            bool(self._moe_optimization_policy.quantize_experts_on_load)
            and variant not in SCRATCH_VARIANTS
        )
        load_packed_state = bool(packed_state_path)
        if load_packed_state:
            scheme = str(getattr(memory_config, "frozen_weight_quantization", "") or "").strip().lower()
            strategy = normalize_compressed_weights_strategy(
                str(getattr(memory_config, "frozen_weight_quantization_strategy", "auto") or "")
            )
            if scheme not in {
                "fp8",
                "int8",
                "nf4",
                "gguf_iq4",
                "gguf_iq3",
                "mxfp8_e4m3",
                "mxfp4",
                "nvfp4",
            } or strategy not in {
                "",
                "auto",
                "disabled",
                "none",
                "compressed_weights",
            }:
                raise ValueError(
                    "LingBot-Video packed state requires frozen_weight_quantization to be "
                    "'fp8', 'int8', 'nf4', 'gguf_iq4', 'gguf_iq3', 'mxfp8_e4m3', "
                    "'mxfp4', or "
                    "'nvfp4' and strategy to be "
                    "compressed_weights or auto."
                )
        if load_quantized_experts or load_packed_state:
            with torch.device("meta"):
                self.transformer = LingBotVideoTransformer3DModel(**self.transformer_config)
        else:
            self.transformer = LingBotVideoTransformer3DModel(**self.transformer_config)
        set_lingbot_video_runtime_options(self.transformer, self._runtime_options)
        provider_spec = get_model_family_provider("lingbot-video").expert_mlp_execution_spec
        if provider_spec != LINGBOT_EXPERT_MLP_EXECUTION_SPEC:
            raise RuntimeError(
                "LingBot provider expert execution spec is absent or inconsistent."
            )
        for module in self.transformer.modules():
            if _is_lingbot_grouped_experts(module):
                module.mirai_expert_mlp_spec = provider_spec
        attention_backend = str(model_config.attention_backend).strip().lower()
        for module in self.transformer.modules():
            if isinstance(module, LingBotVideoAttention):
                module.backend = attention_backend
        setattr(
            self.transformer,
            "_mirai_router_specialization_runtime",
            self._router_specialization_runtime,
        )
        self._intermediate_specialization_runtime.bind(self.transformer)
        self.transformer._mirai_moe_aux_loss_type = self._moe_aux_loss_type
        if load_packed_state:
            self._load_packed_compressed_weights_state(packed_state_path)
        elif variant not in SCRATCH_VARIANTS:
            if load_quantized_experts:
                self._checkpoint_report = self._load_checkpoint_with_quantized_linear_and_experts()
                self._quantized_experts_on_load = True
            else:
                self._checkpoint_report = load_lingbot_transformer_checkpoint(
                    self.transformer,
                    model_config.path,
                    strict=True,
                    subfolder=str(getattr(model_config.params, "denoiser_subfolder", "transformer") or "transformer"),
                )
        else:
            _initialize_scratch_weights(self.transformer)
        if router_repair_artifact_path:
            apply_router_repair_artifact(
                self._router_repair_targets(),
                load_router_repair_artifact(router_repair_artifact_path),
                compressed_artifact_fingerprint=packed_artifact_fingerprint(
                    packed_state_path
                ),
            )
        self._install_decoupled_router_conditioners()
        self._install_lightweight_expert_pools()
        self._install_adjugate_expert_pools()
        self._install_chain_of_experts_extensions()
        self._collect_moe_router_modules()
        _install_expert_output_observer(
            self.transformer, self._expert_output_orthogonality_capture
        )

    @classmethod
    def from_training_config(cls, config: TrainingConfig) -> LingBotVideoPipeline:
        return cls(
            config.model,
            memory_config=config.memory,
            timestep_capacity_policy=TimestepExpertChoiceCapacityPolicy(
                schedule=str(
                    config.model.params.moe_expert_choice_timestep_capacity_schedule
                ),
                capacity_factor_span=float(
                    config.model.params.moe_expert_choice_timestep_capacity_span
                ),
                timestep_sampling=str(config.training.timestep_sampling),
                timestep_sampling_mean=float(
                    config.training.timestep_sampling_mean
                ),
                timestep_sampling_std=float(
                    config.training.timestep_sampling_std
                ),
                timestep_sampling_mode_scale=float(
                    config.training.timestep_sampling_mode_scale
                ),
                flow_shift=float(config.model.params.flow_shift),
            ),
        )

    def _load_checkpoint_with_quantized_linear_and_experts(self) -> LingBotVideoCheckpointReport:
        policy = self._moe_optimization_policy
        access = _expert_access_from_policy(policy)
        expected_keys = discover_lingbot_transformer_checkpoint_keys(
            self.model_config.path,
            subfolder=str(getattr(self.model_config.params, "denoiser_subfolder", "transformer") or "transformer"),
        )
        _fq = str(self._frozen_weight_quantization or "").strip().lower()
        quant_format = normalize_quant_format(
            _fq
            if _fq
            in {
                "fp8",
                "nf4",
                "gguf_iq4",
                "gguf_iq3",
                "mxfp8_e4m3",
                "mxfp4",
                "nvfp4",
            }
            else "int8"
        )
        prepare_report, handlers = prepare_compressed_weights_modules_for_checkpoint_load(
            self.transformer,
            group_sizes="auto",
            expert_weight_access=access,
            expert_dequant_chunk_size=policy.expert_dequant_chunk_size,
            replace_linear=True,
            replace_grouped_experts=True,
            quant_format=quant_format,
            nf4_blocksize=self._nf4_blocksize,
            expert_mlp_execution_spec=get_model_family_provider(
                "lingbot-video"
            ).expert_mlp_execution_spec,
        )
        if prepare_report.replaced_modules <= 0:
            raise ValueError(
                "LingBot-Video quantize_experts_on_load found no linear/expert tensors "
                "to quantize on load."
            )
        report = load_lingbot_transformer_checkpoint(
            self.transformer,
            self.model_config.path,
            strict=True,
            subfolder=str(getattr(self.model_config.params, "denoiser_subfolder", "transformer") or "transformer"),
            expected_keys=expected_keys,
            tensor_handlers=handlers,
        )
        self._compressed_weights_report = prepare_report
        self._bind_compressed_expert_runtime_policy()
        if prepare_report.grouped_expert_modules > 0:
            for module_name, module in list(self.transformer.named_modules()):
                if isinstance(module, CompressedGroupedExperts) and not module.is_fully_loaded():
                    raise ValueError(
                        "LingBot-Video quantize_experts_on_load did not load all routed "
                        f"expert tensors for module '{module_name}'."
                    )
        # NF4 on-load quantization uses transient CUDA workspaces. Release unused
        # allocator cache before placing the configured resident block set; packed
        # tensors remain on their checkpoint device.
        if (
            str(self._frozen_weight_quantization or "").strip().lower() == "nf4"
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        return report

    def _load_packed_compressed_weights_state(self, packed_state_path: str) -> None:
        manifest = read_compressed_weights_packed_state_manifest(packed_state_path)
        validate_drop_upcycling_selection(
            manifest,
            mode=self.model_config.params.expert_upcycling,
            copies_per_expert=self.model_config.params.expert_upcycling_copies,
            reinitialization_ratio=(
                self.model_config.params.expert_upcycling_reinit_ratio
            ),
            seed=self.model_config.params.expert_upcycling_seed,
        )
        compression = str(
            self.model_config.params.expert_weight_compression
        ).strip().lower()
        validate_physical_weight_provider_selection(manifest, compression)
        validate_learned_rotation_selection(
            manifest,
            self.model_config.params.expert_quantization_rotation,
        )
        artifact_formats = get_compressed_weights_packed_state_quant_formats(manifest)
        expected_format = normalize_quant_format(self._frozen_weight_quantization)
        if artifact_formats != {expected_format}:
            raise ValueError(
                "LingBot-Video packed-state quantization mismatch: config requests "
                f"{expected_format!r}, artifact contains {sorted(artifact_formats)!r}."
            )
        prepared = prepare_compressed_weights_modules_from_manifest(self.transformer, manifest)
        report = load_compressed_weights_packed_state_file(
            packed_state_path,
            self.transformer,
            strict=True,
            expert_weight_access_override=_expert_access_from_policy(
                self._moe_optimization_policy
            ),
            expert_dequant_chunk_size_override=(
                self._moe_optimization_policy.expert_dequant_chunk_size
            ),
            packed_state_preload=(
                self._moe_optimization_policy.packed_state_preload
            ),
            packed_stream_cache_gib=(
                self._moe_optimization_policy.packed_stream_cache_gib
            ),
            packed_stream_backend=(
                self._moe_optimization_policy.packed_stream_backend
            ),
            packed_stream_prefetch_depth=(
                self._moe_optimization_policy.packed_stream_prefetch_depth
            ),
        )
        if prepared.replaced_modules <= 0 or report.replaced_modules <= 0:
            raise ValueError("LingBot-Video packed compressed_weights state contained no modules.")
        self._bind_compressed_expert_runtime_policy()
        self._synchronize_packed_expert_topology()
        missing = [
            key
            for key, tensor in self.transformer.state_dict().items()
            if str(getattr(tensor, "device", "")) == "meta"
        ]
        if missing:
            preview = ", ".join(missing[:8])
            extra = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
            raise ValueError(
                "LingBot-Video packed compressed_weights state did not initialize all "
                f"transformer tensors: {preview}{extra}."
            )
        # Preparation and restore describe the same modules; summing the two
        # reports would double-count packed parameters in runtime telemetry.
        self._compressed_weights_report = report

    def _synchronize_packed_expert_topology(self) -> None:
        """Bind provider block metadata to the manifest-authorized expert axis."""

        for module in self.transformer.modules():
            if not isinstance(module, LingBotVideoSparseMoeBlock):
                continue
            physical_experts = int(getattr(module.experts, "num_experts", 0))
            router_experts = int(getattr(module.router, "num_experts", 0))
            if physical_experts != router_experts:
                raise ValueError(
                    "Packed grouped experts and their sibling router have different "
                    f"expert counts ({physical_experts} != {router_experts})."
                )
            module.num_experts = physical_experts

    def _native_inference_assets(self) -> LingBotVideoNativeInference:
        if self._native_inference is None:
            self._native_inference = LingBotVideoNativeInference(self.model_config)
        return self._native_inference

    def has_native_inference(self) -> bool:
        # The native text/VAE inference path exists in code for this family; asset
        # absence fails fast at load_text_encoder/load_vae with a remediation
        # message, mirroring strict native transformer loading.
        return True

    def load_text_encoder(self, *, device: str) -> None:
        self._native_inference_assets().load_text_encoder(device=device)

    def encode_prompt(self, prompt: str, *, device: str) -> Any:
        return self._native_inference_assets().encode_prompt(prompt, device=device)

    def prepare_inference_conditioning(
        self, request: Any, *, device: str, generator: Any
    ) -> Any:
        return self._native_inference_assets().prepare_conditioning(
            request, device=device, generator=generator
        )

    def encode_conditioned_prompt(
        self, prompt: str, *, prepared: Any, device: str
    ) -> Any:
        return self._native_inference_assets().encode_prompt(
            prompt,
            device=device,
            image=prepared.prompt_media,
        )

    def offload_text_encoder(self) -> None:
        self._native_inference_assets().offload_text_encoder()

    def load_vae(self, *, device: str) -> None:
        self._native_inference_assets().load_vae(device=device)

    def decode_latents_native(self, latents: list[Any]) -> Any:
        return self._native_inference_assets().decode_latents_native(latents)

    def offload_vae(self) -> None:
        self._native_inference_assets().offload_vae()

    def encode_video_native(
        self,
        video: Any,
        *,
        generator: Any | None = None,
        sample_posterior: bool = True,
    ) -> Any:
        return self._native_inference_assets().encode_video_native(
            video,
            generator=generator,
            sample_posterior=sample_posterior,
        )

    # -- refiner stage -----------------------------------------------------
    def _refiner_assets(self) -> Any:
        if self._refiner is None:
            from mirai.core.models.lingbot_video.refiner import LingBotRefiner

            subfolder = str(
                getattr(self.model_config.params, "refiner_subfolder", "refiner") or "refiner"
            )
            self._refiner = LingBotRefiner(self.model_config, subfolder=subfolder)
        return self._refiner

    def supports_refiner(self) -> bool:
        # The refiner is a video-only quality stage; validation fixtures have no
        # analogue. Real LingBot-Video families support it in code; missing
        # refiner/ weights fail fast at load_refiner()/has_refiner_weights().
        variant = str(self.model_config.params.variant).strip().lower()
        return variant not in SCRATCH_VARIANTS

    def validate_refinement_request(
        self,
        request: dict[str, Any],
        *,
        frames: int,
    ) -> None:
        _ = request
        if int(frames) <= 1:
            raise RuntimeError(
                "LingBot-Video refinement is video-only and requires frames>1; "
                f"got frames={int(frames)}."
            )
        if not self.supports_refiner():
            raise RuntimeError(
                "This LingBot-Video variant does not support the refinement stage."
            )
        if not self.has_refiner_weights():
            subfolder = str(
                getattr(self.model_config.params, "refiner_subfolder", "refiner")
                or "refiner"
            )
            raise RuntimeError(
                "LingBot-Video refinement requires a separate checkpoint under "
                f"'{Path(str(self.model_config.path)) / subfolder}'."
            )

    def refine_inference_latent(
        self,
        *,
        base_latent: Any,
        request: dict[str, Any],
        prompt: str,
        negative_prompt: str,
        seed: int,
        device: str,
        dtype: Any | None,
    ) -> Any:
        from mirai.core.models.lingbot_video.refiner import run_refine

        return run_refine(
            pipeline=self,
            refiner=self._refiner_assets(),
            base_latent=base_latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            height=int(request["height"]),
            width=int(request["width"]),
            steps=int(request["steps"]),
            cfg_scale=float(request["cfg_scale"]),
            shift=float(request["shift"]),
            t_thresh=float(request["t_thresh"]),
            sigma_tail_steps=int(request["sigma_tail_steps"]),
            scheduler=str(request["scheduler"]),
            device=str(device),
            dtype=dtype,
        )

    def has_refiner_weights(self) -> bool:
        return bool(self._refiner_assets().has_weights())

    def load_refiner(self, *, device: str, dtype: Any | None = None) -> None:
        self._refiner_assets().load(device=device, dtype=dtype)

    def release_refiner(self) -> None:
        if self._refiner is not None:
            self._refiner.release()

    def release_base_transformer(self) -> None:
        """Move the base DiT off the compute device to free VRAM for the refiner.

        The refine flow calls this before ``load_refiner``. The base transformer
        remains on CPU, so refinement terminates the current denoising session.
        """
        self.transformer.to(device=torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def refiner_forward(self, noisy_latents: Any, timesteps: Any, text_embeds: dict[str, Any]) -> Any:
        """One velocity forward through the loaded refiner DiT.

        Reuses the base forward's text-conditioning prep (the module-level
        ``_text_embeddings`` / ``_text_attention_mask`` / prefix-mask trim
        helpers) so the refiner sees identically-shaped conditioning; only the
        transformer weights differ. Timestep is ``sigma * 1000`` as upstream.
        """
        refiner = self._refiner_assets()
        if not refiner.loaded:
            raise RuntimeError(
                "LingBot-Video refiner is not loaded; call load_refiner() first."
            )
        config = refiner.transformer_config()
        latents = _as_lingbot_latents(noisy_latents, config, label="refiner_noisy_latents")
        timestep_tensor = as_latent_tensor(
            timesteps, dtype=latents.dtype, device=latents.device
        ).reshape(latents.shape[0])
        encoder_hidden_states = _text_embeddings(
            text_embeds,
            batch=int(latents.shape[0]),
            text_dim=int(config["text_dim"]),
            like=latents,
        )
        encoder_attention_mask = _text_attention_mask(
            text_embeds,
            batch=int(latents.shape[0]),
            like=latents,
        )
        encoder_hidden_states, encoder_attention_mask = _trim_encoder_states_to_prefix_mask(
            encoder_hidden_states,
            encoder_attention_mask,
        )
        set_lingbot_video_runtime_options(refiner.transformer, self._runtime_options)
        prediction = refiner.transformer(
            latents,
            timestep_tensor * 1000.0,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]
        _clear_router_runtime_state(refiner.transformer)
        return prediction

    def get_video_latent_layout(self) -> VideoLatentLayout:
        patch = _tuple_int(self.transformer_config.get("patch_size", (1, 2, 2)), length=3)
        _ = patch
        return VideoLatentLayout(
            latent_channels=int(self.transformer_config["in_channels"]),
            temporal_downsample=4,
            spatial_downsample=8,
            frame_count_modulus=4,
            frame_count_remainder=1,
            frame_count_rule="1 modulo 4 (4n+1)",
            request_spatial_multiple=16,
        )

    def apply_noise(self, clean_latents: Any, noise: Any, timesteps: Any) -> Any:
        clean = _as_lingbot_latents(
            clean_latents,
            self.transformer_config,
            label="clean_latents",
        )
        noise_tensor = _as_lingbot_latents(
            noise,
            self.transformer_config,
            label="noise",
            dtype=clean.dtype,
            device=clean.device,
        )
        timestep_tensor = as_latent_tensor(
            timesteps,
            dtype=clean.dtype,
            device=clean.device,
        ).reshape(clean.shape[0])
        shift: Any = float(self.model_config.params.flow_shift)
        if self._flow_shift_policy.enabled:
            shift = self._flow_shifts_for_latents(clean).to(
                dtype=timestep_tensor.dtype
            )
        return apply_rectified_flow_noise(
            clean_latents=clean,
            noise=noise_tensor.reshape(clean.shape),
            timesteps=timestep_tensor,
            shift=shift,
            timestep_eps=1e-5,
        )

    def _visual_token_count_from_latent_shape(
        self,
        latent_shape: tuple[int, ...],
    ) -> int:
        if len(latent_shape) != 5:
            raise ValueError(
                "LingBot-Video latent shape must be [batch, channels, frames, "
                "height, width]."
            )
        patch_t, patch_h, patch_w = (
            int(value) for value in self.transformer_config["patch_size"]
        )
        frames, height, width = (
            int(latent_shape[2]),
            int(latent_shape[3]),
            int(latent_shape[4]),
        )
        if (
            frames % patch_t != 0
            or height % patch_h != 0
            or width % patch_w != 0
        ):
            raise ValueError(
                "LingBot-Video latent dimensions must be divisible by the "
                "transformer patch size."
            )
        return (
            (frames // patch_t)
            * (height // patch_h)
            * (width // patch_w)
        )

    def _flow_shifts_for_latents(self, latents: Any) -> Any:
        token_count = self._visual_token_count_from_latent_shape(
            tuple(int(dim) for dim in latents.shape)
        )
        counts = torch.full(
            (int(latents.shape[0]),),
            token_count,
            device=latents.device,
            dtype=torch.int64,
        )
        return self._flow_shift_policy.shifts_for_token_counts(counts)

    def prepare_model_timesteps(self, timesteps: Any, *, latents: Any) -> Any:
        clean = _as_lingbot_latents(
            latents,
            self.transformer_config,
            label="clean_latents",
        )
        timestep_tensor = as_latent_tensor(
            timesteps,
            dtype=clean.dtype,
            device=clean.device,
        ).reshape(clean.shape[0])
        shifts: Any = float(self.model_config.params.flow_shift)
        if self._flow_shift_policy.enabled:
            shifts = self._flow_shifts_for_latents(clean).to(
                dtype=timestep_tensor.dtype
            )
        return shifted_sigma(
            clamp_timesteps(timestep_tensor, 1e-5),
            shifts,
        )

    def resolve_flow_shift_for_latent_shape(
        self,
        latent_shape: tuple[int, ...],
    ) -> float:
        token_count = self._visual_token_count_from_latent_shape(latent_shape)
        return self._flow_shift_policy.shift_for_token_count(token_count)

    def compute_target(self, noise: Any, clean_latents: Any, timesteps: Any) -> Any:
        _ = timesteps
        clean = _as_lingbot_latents(
            clean_latents,
            self.transformer_config,
            label="clean_latents",
        )
        noise_tensor = _as_lingbot_latents(
            noise,
            self.transformer_config,
            label="noise",
            dtype=clean.dtype,
            device=clean.device,
        )
        return rectified_flow_target(noise=noise_tensor.reshape(clean.shape), clean_latents=clean)

    def forward(
        self,
        noisy_latents: Any,
        timesteps: Any,
        text_embeds: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        routing_guidance_latents = kwargs.pop("routing_guidance_latents", None)
        self._last_balance_gradient_probe = None
        latents = _as_lingbot_latents(
            noisy_latents,
            self.transformer_config,
            label="noisy_latents",
        )
        if routing_guidance_latents is not None:
            routing_guidance_latents = _as_lingbot_latents(
                routing_guidance_latents,
                self.transformer_config,
                label="routing_guidance_latents",
                dtype=latents.dtype,
                device=latents.device,
            )
            if tuple(routing_guidance_latents.shape) != tuple(latents.shape):
                raise ValueError(
                    "routing_guidance_latents must match noisy_latents shape."
                )
        _ = kwargs
        timestep_tensor = as_latent_tensor(
            timesteps,
            dtype=latents.dtype,
            device=latents.device,
        ).reshape(latents.shape[0])
        encoder_hidden_states = _text_embeddings(
            text_embeds,
            batch=int(latents.shape[0]),
            text_dim=int(self.transformer_config["text_dim"]),
            like=latents,
        )
        encoder_attention_mask = _text_attention_mask(
            text_embeds,
            batch=int(latents.shape[0]),
            like=latents,
        )
        encoder_hidden_states, encoder_attention_mask = _trim_encoder_states_to_prefix_mask(
            encoder_hidden_states,
            encoder_attention_mask,
        )
        # ``forward`` receives the model-conditioning coordinate. Both constant
        # and dynamic training paths now pass post-shift sigma, as do inference
        # solvers, so downstream timestep-aware policies must consume it as-is.
        runtime_sigmas = timestep_tensor.detach().float().clamp(1e-5, 1.0 - 1e-5)
        timestep_adapter_stats = self._sync_timestep_adapter_masks(
            timestep_tensor,
            sigmas=runtime_sigmas,
        )
        if self._timestep_capacity_policy.enabled:
            self._expert_choice_runtime_sigmas = runtime_sigmas
            self._expert_choice_runtime_flow_shifts = (
                self._flow_shifts_for_latents(latents)
                if self._flow_shift_policy.enabled
                else None
            )
        # Global-batch load balancing: prime each router with the load fraction
        # accumulated through prior micro-batches of this window (None on the
        # first). This serves the aggressive-checkpoint path, which computes the
        # aux term inline during the forward. Inert unless scope='global_batch'.
        self._global_batch_prime_injected_fractions()
        prediction = self.transformer(
            latents,
            timestep_tensor * 1000.0,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            routing_guidance_states=routing_guidance_latents,
            return_dict=False,
        )[0]
        # Fold this micro-batch's dispatch counts into the window and refresh the
        # injected fraction so the standard (post-forward) aux collection below
        # reads the accumulated-through-current load. Inert in microbatch scope.
        self._global_batch_accumulate_and_inject()
        if not self.transformer.training:
            # Inference forwards skip the training telemetry below (routing
            # stats, aux terms, stability metrics): it costs hundreds of
            # GPU->CPU `.item()` syncs per forward and nothing in the denoise
            # loop consumes it. Training telemetry remains enabled below.
            #
            # Capture before clearing the router's transient assignment state.
            if self._inference_routing_telemetry:
                self._capture_inference_routing_trace(timestep_tensor)
            self._last_auxiliary_losses = {}
            self._last_diagnostics = {}
            _clear_router_runtime_state(self.transformer)
            return prediction
        if self._moe_bias_update_rate > 0.0:
            for name, counts in _router_assignment_counts(self.transformer).items():
                previous = self._pending_router_loads.get(name)
                self._pending_router_loads[name] = (
                    counts if previous is None else previous + counts
                )
        routing = _routing_stats(self.transformer)
        coverage_metrics = _expert_choice_coverage_metrics(
            self.transformer,
            alarm_threshold=self._expert_choice_coverage_alarm_threshold,
        )
        balance_terms, z_terms, z_weights = _collect_router_auxiliary_terms(
            self.transformer
        )
        self._last_auxiliary_losses = _weighted_router_auxiliary_losses(
            balance_terms,
            z_terms,
            load_balance_weight=self._moe_aux_loss_weight,
            z_loss_weight=self._moe_router_z_loss_weight,
            z_loss_weights=z_weights,
        )
        balance_gradient_objectives: dict[str, Any] = {}
        if self._balance_gradient_ratio_enabled and balance_terms:
            balance_gradient_objectives["load_balance"] = (
                torch.stack(balance_terms).mean() * self._moe_aux_loss_weight
            )
        similarity_terms: list[Any] = []
        combination_usage: dict[str, dict[str, Any]] = {}
        if self._moe_router_similarity_loss_weight != 0.0:
            similarity_terms, combination_usage = _router_similarity_terms(
                self.transformer
            )
        if similarity_terms:
            self._last_auxiliary_losses["moe_router_similarity"] = (
                torch.stack(similarity_terms).mean()
                * self._moe_router_similarity_loss_weight
            )
        if self._moe_spatiotemporal_routing_weight != 0.0:
            patch_t, patch_h, patch_w = (
                int(value) for value in self.transformer_config["patch_size"]
            )
            grid = (
                int(latents.shape[2]) // patch_t,
                int(latents.shape[3]) // patch_h,
                int(latents.shape[4]) // patch_w,
            )
            video_tokens = grid[0] * grid[1] * grid[2]
            text_lengths = (
                encoder_attention_mask.sum(dim=-1).detach().long().cpu().tolist()
                if encoder_attention_mask is not None
                else [int(encoder_hidden_states.shape[1])] * int(latents.shape[0])
            )
            offsets: list[int] = []
            cursor = 0
            for text_length in text_lengths:
                offsets.append(cursor)
                cursor += video_tokens + int(text_length)
            spatial_terms = _spatiotemporal_routing_terms(
                self.transformer,
                video_offsets=tuple(offsets),
                grid=grid,
                max_edges=self._moe_spatiotemporal_routing_max_edges,
            )
            if spatial_terms:
                self._last_auxiliary_losses["moe_spatiotemporal_routing"] = (
                    torch.stack(spatial_terms).mean()
                    * self._moe_spatiotemporal_routing_weight
                )
        if self._phi_balance_controller is not None:
            phi_terms = _phi_balance_auxiliary_terms(
                self.transformer, self._phi_balance_controller
            )
            if phi_terms:
                self._last_auxiliary_losses["moe_phi_balance"] = (
                    torch.stack(phi_terms).mean() * self._moe_phi_balance_weight
                )
                if self._balance_gradient_ratio_enabled:
                    balance_gradient_objectives["phi_balance"] = (
                        self._last_auxiliary_losses["moe_phi_balance"]
                    )
        self._last_auxiliary_losses.update(
            self._router_specialization_runtime.auxiliary_losses(self.transformer)
        )
        self._last_auxiliary_losses.update(
            self._intermediate_specialization_runtime.auxiliary_losses(
                self.transformer
            )
        )
        prototypical_terms = collect_prototypical_routing_losses(self.transformer)
        if prototypical_terms:
            # The source attaches one RCL term to every MoE block, so the model
            # objective is their sum rather than their layer mean.
            self._last_auxiliary_losses["moe_routing_contrastive"] = torch.stack(
                prototypical_terms
            ).sum()
        if self._dispersive_loss_runtime is not None:
            self._last_auxiliary_losses.update(
                self._dispersive_loss_runtime.auxiliary_losses(self.transformer)
            )
        if self._simbal_runtime is not None:
            self._last_auxiliary_losses.update(
                self._simbal_runtime.auxiliary_losses()
            )
        if self._expert_output_orthogonality_capture is not None:
            self._last_auxiliary_losses.update(
                _orthogonality_losses(
                    self.transformer, self._expert_output_orthogonality_capture,
                    weight=self._moe_expert_orthogonality_loss_weight,
                )
            )
        # Raw (unweighted) per-step aux VALUES for telemetry -- emitted even
        # when the corresponding loss weights are 0 (monitoring-gap request).
        # Detached: never touches the loss graph, defaults stay bit-identical.
        moe_balance_raw = (
            float(torch.stack(balance_terms).mean().detach().float().cpu().item())
            if balance_terms
            else None
        )
        moe_z_raw = (
            float(torch.stack(z_terms).mean().detach().float().cpu().item())
            if z_terms
            else None
        )
        # Stochastic expert-subset router-consistency loss (opt-in; only present
        # when a KL weight is set AND the router is trainable). Rides the same
        # auxiliary-loss seam as the balance/z terms -> added to the total loss.
        subset_kl_terms = [
            module.training_subset_kl
            for module in self.transformer.modules()
            if isinstance(module, LingBotVideoRouter)
            and getattr(module, "training_subset_kl", None) is not None
        ]
        if subset_kl_terms:
            self._last_auxiliary_losses = dict(self._last_auxiliary_losses)
            self._last_auxiliary_losses["moe_subset_router_kl"] = torch.stack(
                subset_kl_terms
            ).mean()
        distillation_terms = collect_lingbot_router_distillation_terms(self.transformer)
        if distillation_terms:
            self._last_auxiliary_losses["moe_router_distillation"] = torch.stack(
                distillation_terms
            ).mean()
        routing_summary = summarize_routing_stats(
            routing,
            include_expert_vectors=False,
        )
        # Routing-stability metrics (cheap, detached): entropy / utilization CV /
        # top-1 monopoly / KL vs a step-0 snapshot, plus the subset headline
        # (unique experts touched per layer per step).
        stability_metrics = self._routing_stability_metrics()
        if (
            self._router_temperature_controller is not None
            and "moe_routing_entropy" in stability_metrics
        ):
            self._router_temperature_controller.observe_entropy(
                stability_metrics["moe_routing_entropy"],
                training=True,
            )
        affinity_hit_rates = [
            float(module.last_dataset_affinity_hit_rate)
            for module in self.transformer.modules()
            if isinstance(module, LingBotVideoRouter)
            and getattr(module, "last_dataset_affinity_hit_rate", None) is not None
        ]
        routing_states = []
        for module in self.transformer.modules():
            if not isinstance(module, LingBotVideoRouter):
                continue
            indices = getattr(module, "last_top_indices", None)
            batch_size = int(getattr(module, "training_batch_size", 0))
            tokens_per_sample = int(getattr(module, "training_tokens_per_sample", 0))
            if indices is not None and batch_size > 0 and tokens_per_sample > 0:
                routing_states.append(
                    (indices, batch_size, tokens_per_sample, int(module.num_experts))
                )
        routing_summary["diffusion_timestep_buckets"] = (
            summarize_routing_by_diffusion_timestep(routing_states, timestep_tensor)
        )
        if combination_usage:
            routing_summary["expert_combination_usage"] = combination_usage
        self._last_diagnostics = {"moe_routing": routing_summary}
        if moe_balance_raw is not None:
            self._last_diagnostics["moe_balance_loss"] = moe_balance_raw
        if moe_z_raw is not None:
            self._last_diagnostics["moe_z_loss"] = moe_z_raw
        if timestep_adapter_stats is not None:
            self._last_diagnostics["timestep_adapter"] = timestep_adapter_stats
        for key, value in stability_metrics.items():
            self._last_diagnostics[key] = value
        for key, value in chain_of_experts_metrics(self.transformer).items():
            self._last_diagnostics[key] = value
        if self._dispersive_loss_runtime is not None:
            self._last_diagnostics.update(
                self._dispersive_loss_runtime.diagnostics()
            )
        if self._simbal_runtime is not None:
            self._last_diagnostics.update(self._simbal_runtime.diagnostics())
        if self._selective_sinkhorn_controller is not None:
            self._last_diagnostics.update(
                self._selective_sinkhorn_controller.diagnostics()
            )
        if self._prototypical_routing_spec is not None:
            self._last_diagnostics.update(
                prototypical_routing_diagnostics(self.transformer)
            )
        if self._mixture_of_depths_spec is not None:
            self._last_diagnostics.update(
                mixture_of_depths_diagnostics(self.transformer)
            )
        if self._router_temperature_controller is not None:
            self._last_diagnostics["moe_router_temperature"] = (
                self._router_temperature_controller.scheduled_temperature()
            )
            self._last_diagnostics["moe_router_temperature_frozen"] = float(
                self._router_temperature_controller.annealing_frozen
            )
        if self._progressive_sparsification_policy is not None:
            self._last_diagnostics["moe_progressive_sparsification_early"] = float(
                self._progressive_sparsification_policy.is_early(
                    step=self._progressive_sparsification_step
                )
            )
            self._last_diagnostics["moe_active_experts_per_token_mean"] = (
                sum(int(router.top_k) for router in self._moe_router_modules)
                / max(1, len(self._moe_router_modules))
            )
        if self._moe_balance_loss_schedule is not None:
            self._last_diagnostics["moe_balance_loss_exploration"] = float(
                self._moe_balance_loss_schedule.is_exploration(
                    step=self._moe_balance_loss_step
                )
            )
            self._last_diagnostics["moe_balance_loss_effective_weight"] = float(
                self._moe_aux_loss_weight
            )
        for key, value in coverage_metrics.items():
            self._last_diagnostics[key] = value
        if self._routing_health_enabled:
            for key, value in self._routing_health_metrics(
                timestep_tensor=timestep_tensor,
                latent_shape=tuple(int(dim) for dim in latents.shape),
            ).items():
                self._last_diagnostics[key] = value
        if affinity_hit_rates:
            self._last_diagnostics["moe_dataset_affinity_hit_rate"] = sum(
                affinity_hit_rates
            ) / len(affinity_hit_rates)
        if self._balance_gradient_ratio_enabled:
            self._last_balance_gradient_probe = _build_balance_gradient_probe(
                self.transformer,
                objectives=balance_gradient_objectives,
            )
        _clear_router_runtime_state(self.transformer)
        return prediction

    def _routing_stability_metrics(self) -> dict[str, float]:
        """Per-step routing-stability scalars from already-collected router loads.

        Entropy (of the per-expert assignment distribution), utilization CV,
        top-1 monopoly (max assignment fraction), KL vs a step-0 snapshot, and
        the subset headline ``moe_step_unique_experts`` (mean unique experts
        touched per layer). Detached -- never touches the loss graph. Absent when
        no router has produced a selection this forward.
        """
        entropies: list[float] = []
        cvs: list[float] = []
        monopolies: list[float] = []
        kls: list[float] = []
        uniques: list[float] = []
        capture = self._routing_step0_snapshot is None
        snapshot: dict[str, Any] = {} if capture else self._routing_step0_snapshot
        for name, module in self.transformer.named_modules():
            if not isinstance(module, LingBotVideoRouter):
                continue
            indices = getattr(module, "last_top_indices", None)
            if indices is None:
                continue
            num_experts = int(module.num_experts)
            counts = torch.bincount(
                indices.detach().reshape(-1), minlength=num_experts
            ).float()
            total = counts.sum().clamp_min(1.0)
            dist = counts / total
            probs = dist.clamp_min(1e-12)
            entropies.append(float(-(probs * probs.log()).sum().item()))
            mean = dist.mean().clamp_min(1e-20)
            cvs.append(float((dist.std(unbiased=False) / mean).item()))
            monopolies.append(float(dist.max().item()))
            # Actual working set = distinct experts that received tokens (experts
            # with zero tokens are skipped by the dispatch), i.e. the true PCIe
            # stream count -- not the subset cap.
            uniques.append(float(int((counts > 0).sum().item())))
            if capture:
                snapshot[name] = dist.detach().cpu()
            else:
                ref = snapshot.get(name)
                if ref is not None:
                    ref = ref.to(device=dist.device, dtype=dist.dtype)
                    kl = (dist * (dist.clamp_min(1e-12).log()
                                  - ref.clamp_min(1e-12).log())).sum()
                    kls.append(float(kl.item()))
            if self._selection_margin_probe is not None:
                scores = getattr(module, "last_scores", None)
                if scores is not None and int(scores.shape[0]) == int(indices.shape[0]):
                    self._selection_margin_probe.observe(
                        scores, top_k=int(indices.shape[1])
                    )
        if capture and snapshot:
            self._routing_step0_snapshot = snapshot
        if not entropies:
            return {}
        metrics = {
            "moe_routing_entropy": float(sum(entropies) / len(entropies)),
            "moe_utilization_cv": float(sum(cvs) / len(cvs)),
            "moe_top1_monopoly": float(sum(monopolies) / len(monopolies)),
            "moe_step_unique_experts": float(sum(uniques) / len(uniques)),
        }
        if kls:
            metrics["moe_routing_kl_vs_step0"] = float(sum(kls) / len(kls))
        if self._selection_margin_probe is not None:
            metrics.update(self._selection_margin_probe.summary())
        return metrics

    def _routing_health_metrics(
        self,
        *,
        timestep_tensor: Any | None = None,
        latent_shape: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        metrics = collect_lingbot_routing_health(
            self.transformer,
            deadlock_tracker=self._deadlock_tracker,
            drift_tracker=self._router_drift_tracker,
            training=bool(self.transformer.training),
            timesteps=timestep_tensor,
            latent_shape=latent_shape,
        )
        metrics.update(
            collect_lingbot_attention_monitoring(
                self.transformer,
                monitor=self._preemptive_attention_monitor,
                training=bool(self.transformer.training),
            )
        )
        return metrics

    def _sync_timestep_adapter_masks(
        self,
        timestep_tensor: Any,
        *,
        sigmas: Any | None = None,
    ) -> dict[str, Any] | None:
        """Distribute per-batch timestep-axis rank masks to adapter modules.

        Post-shift sigma via the SAME ``shifted_sigma(clamp(t, 1e-5), shift)``
        used by ``apply_noise``, so the T-LoRA schedule and the timestep band
        act on the coordinate AFTER flow_shift. No-op (None) when the policy
        is the default, adapter masks remain disabled.
        """
        policy = self._timestep_adapter_policy
        if policy is None or self._lora_report is None:
            return None
        if sigmas is None:
            sigmas = shifted_sigma(
                clamp_timesteps(timestep_tensor.detach().float(), 1e-5),
                float(self.model_config.params.flow_shift),
            )
        rank = int(self._lora_report.rank)
        # Non-differentiable schedule/band mask (F-C substrate): applied as a
        # plain buffer, checkpoint-safe exactly like tlora/bands.
        mask = per_sample_rank_mask(sigmas, rank=rank, policy=policy)
        uniform = mask.amin(dim=0)
        set_lora_timestep_rank_masks(self._timestep_adapter_modules, mask, uniform)
        gate_stats: dict[str, float] = {}
        tc_gate = getattr(self, "_tc_gate", None)
        if tc_gate is not None:
            # TC-LoRA: distribute the gate PROVIDER (hypernet + detached sigma),
            # NOT a precomputed differentiable mask. Each adapter recomputes the
            # gate g(sigma) inside its own forward, so under gradient
            # checkpointing every recomputed block owns a segment-local gate
            # subgraph -- this avoids backwarding the one shared hypernet subgraph
            # once per checkpoint segment (the double-backward crash) while
            # gradient still reaches the hypernet. Telemetry uses a detached gate.
            set_lora_tc_gate(
                self._timestep_adapter_modules, tc_gate, sigmas.detach()
            )
            gate_stats = gate_summary(tc_gate(sigmas).detach())
        else:
            set_lora_tc_gate(self._timestep_adapter_modules, None, None)
        active_fraction = mask.sum(dim=1) / float(rank)
        band_gate = (mask.amax(dim=1) > 0.0).float()
        stats: dict[str, Any] = {
            "sigma": [float(v) for v in sigmas.detach().cpu().tolist()],
            "active_rank_fraction": [
                float(v) for v in active_fraction.detach().cpu().tolist()
            ],
            "band_gate": [float(v) for v in band_gate.detach().cpu().tolist()],
            "uniform_active_rank": int(
                (uniform.detach() > 0.0).sum().cpu().item()
            ),
        }
        stats.update(gate_stats)
        return stats

    def validate_config(self, config: TrainingConfig) -> list[str]:
        params = config.model.params
        errors: list[str] = []
        variant = str(params.variant).strip().lower()
        if variant not in VALID_VARIANTS:
            errors.append(
                f"Unsupported LingBot-Video variant '{params.variant}'. "
                f"Expected one of: {', '.join(sorted(VALID_VARIANTS))}."
            )
        strategy_type = str(config.strategy.type).strip().lower()
        if strategy_type not in SUPPORTED_STRATEGIES:
            errors.append(
                f"Unsupported LingBot-Video strategy '{config.strategy.type}'. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_STRATEGIES))}."
            )
        if int(self.transformer_config.get("num_experts", 0)) <= 1:
            errors.append("LingBot-Video requires transformer num_experts > 1.")
        if int(self.transformer_config.get("num_experts_per_tok", 0)) <= 0:
            errors.append("LingBot-Video requires num_experts_per_tok > 0.")
        if self._chain_of_experts_enabled:
            native_hidden = int(self.transformer_config.get("hidden_size", 0))
            native_experts = int(self.transformer_config.get("num_experts", 0))
            if self._moe_routing_mode != "token_choice":
                errors.append("Chain-of-Experts requires token-choice routing.")
            if str(config.adapter.type).strip().lower() != "lora":
                errors.append(
                    "Chain-of-Experts requires adapter.type='lora'."
                )
            if not 1 <= self._chain_router_rank <= min(
                native_hidden,
                native_experts,
            ):
                errors.append(
                    "model.params.moe_chain_router_rank must be in "
                    "[1, min(native hidden size, native expert count)]."
                )
            if self._lightweight_expert_count > 0:
                errors.append(
                    "Chain-of-Experts cannot compose with lightweight "
                    "experts."
                )
            if float(params.moe_spatiotemporal_routing_weight) != 0.0:
                errors.append(
                    "Chain-of-Experts cannot compose with the "
                    "single-pass spatiotemporal routing objective."
                )
        if self._expert_choice_timestep_weight > 0.0:
            if self._moe_routing_mode != "expert_choice":
                errors.append(
                    "Decoupled routing requires expert-choice routing."
                )
            if str(config.adapter.type).strip().lower() != "lora":
                errors.append(
                    "Decoupled routing requires adapter.type='lora'."
                )
        if self._lightweight_expert_count > 0:
            physical_experts = int(
                self.transformer_config.get("num_experts", 0)
            )
            if not (
                1
                <= self._lightweight_top_k
                <= physical_experts + self._lightweight_expert_count
            ):
                errors.append(
                    "model.params.moe_lightweight_top_k must be in "
                    "[1, native num_experts + all lightweight experts]."
                )
            if str(config.adapter.type).strip().lower() != "lora":
                errors.append(
                    "Lightweight experts require adapter.type='lora'."
                )
            if self._moe_routing_mode != "token_choice":
                errors.append(
                    "Lightweight experts require token-choice routing."
                )
            if self._moe_balance_scope != "microbatch":
                errors.append(
                    "Lightweight experts require "
                    "moe_balance_scope='microbatch'."
                )
            if float(params.expert_subset_fraction) != 1.0:
                errors.append(
                    "Lightweight experts cannot be combined with "
                    "stochastic expert-subset routing."
                )
            if int(params.moe_dynamic_topk_min) != 0:
                errors.append(
                    "Lightweight experts cannot be combined with "
                    "compute-budgeted dynamic top-k."
                )
            if float(params.moe_bias_update_rate) != 0.0:
                errors.append(
                    "Lightweight experts require moe_bias_update_rate=0; "
                    "their compute-budget controller uses a distinct update rule."
                )
            if str(config.dataset.moe_routing.specialization_mode) != "emergent":
                errors.append(
                    "Lightweight experts cannot be combined with dataset "
                    "routing affinity."
                )
            for policy_name in ("diversity_routing", "expert_dropout"):
                options = config.training.policy_options.get(policy_name, {})
                if bool(options.get("enabled", False)):
                    errors.append(
                        "Lightweight experts cannot be combined with "
                        f"training policy '{policy_name}'."
                    )
            incompatible_losses = {
                "moe_router_similarity_loss_weight": (
                    params.moe_router_similarity_loss_weight
                ),
                "moe_spatiotemporal_routing_weight": (
                    params.moe_spatiotemporal_routing_weight
                ),
                "moe_phi_balance_weight": params.moe_phi_balance_weight,
                "moe_router_variance_loss_weight": (
                    params.moe_router_variance_loss_weight
                ),
                "moe_expert_orthogonality_loss_weight": (
                    params.moe_expert_orthogonality_loss_weight
                ),
                "moe_swiglu_specialization_loss_weight": (
                    params.moe_swiglu_specialization_loss_weight
                ),
                "moe_cross_layer_coupling_loss_weight": (
                    params.moe_cross_layer_coupling_loss_weight
                ),
            }
            enabled_losses = sorted(
                name
                for name, value in incompatible_losses.items()
                if float(value) != 0.0
            )
            if enabled_losses:
                errors.append(
                    "Lightweight experts cannot be combined with "
                    + ", ".join(enabled_losses)
                    + "."
                )
        if self._adjugate_experts_enabled:
            native_experts = int(
                self.transformer_config.get("num_experts", 0)
            )
            groups = int(self._adjugate_expert_groups)
            if (
                groups <= 0
                or groups > native_experts
                or native_experts % groups != 0
            ):
                errors.append(
                    "model.params.moe_adjugate_expert_groups must divide "
                    f"the native expert count ({native_experts})."
                )
            elif float(self._adjugate_expert_scale) > (
                float(groups) / float(native_experts)
            ):
                errors.append(
                    "model.params.moe_adjugate_expert_scale must be <= "
                    "moe_adjugate_expert_groups / native num_experts "
                    f"({float(groups) / float(native_experts):g})."
                )
            if int(self._adjugate_expert_intermediate_size) <= 0:
                errors.append(
                    "model.params.moe_adjugate_expert_intermediate_size "
                    "must be > 0."
                )
        if self._moe_aux_loss_type not in {"disabled", "global", "sequence"}:
            errors.append(
                "model.params.moe_aux_loss_type must be one of: "
                "model_native, disabled, global, sequence."
            )
        if float(params.moe_aux_loss_weight) < 0.0:
            errors.append("model.params.moe_aux_loss_weight must be >= 0.")
        if float(params.moe_router_z_loss_weight) < 0.0:
            errors.append("model.params.moe_router_z_loss_weight must be >= 0.")
        if float(params.moe_router_similarity_loss_weight) < 0.0:
            errors.append(
                "model.params.moe_router_similarity_loss_weight must be >= 0."
            )
        if float(params.moe_spatiotemporal_routing_weight) < 0.0:
            errors.append(
                "model.params.moe_spatiotemporal_routing_weight must be >= 0."
            )
        if int(params.moe_spatiotemporal_routing_max_edges) <= 0:
            errors.append(
                "model.params.moe_spatiotemporal_routing_max_edges must be > 0."
            )
        native_top_k = int(self.transformer_config.get("num_experts_per_tok", 0))
        dynamic_min = int(params.moe_dynamic_topk_min)
        dynamic_average = float(params.moe_dynamic_topk_average)
        if dynamic_min < 0:
            errors.append("model.params.moe_dynamic_topk_min must be >= 0.")
        if dynamic_min == 0 and dynamic_average != 0.0:
            errors.append(
                "model.params.moe_dynamic_topk_average must be 0 when dynamic "
                "top-k is disabled."
            )
        if dynamic_min > 0 and not (
            dynamic_min <= dynamic_average <= native_top_k
        ):
            errors.append(
                "Dynamic top-k requires moe_dynamic_topk_min <= "
                "moe_dynamic_topk_average <= experts_per_token."
            )
        if self._moe_routing_mode == "expert_choice" and dynamic_min > 0:
            errors.append(
                "Expert-Choice routing cannot be combined with token-choice dynamic top-k."
            )
        if float(params.moe_bias_update_rate) < 0.0:
            errors.append("model.params.moe_bias_update_rate must be >= 0.")
        hidden = int(self.transformer_config.get("hidden_size", 0))
        heads = int(self.transformer_config.get("num_attention_heads", 0))
        if hidden <= 0 or heads <= 0 or hidden % heads != 0:
            errors.append("LingBot-Video hidden_size must be divisible by num_attention_heads.")
        layout = self.get_video_latent_layout()
        for frame_bucket in list(getattr(config.dataset, "frame_buckets", []) or []):
            _append_layout_request_error(
                errors,
                layout=layout,
                field=f"dataset.frame_buckets[{int(frame_bucket)}]",
                frame_count=int(frame_bucket),
                height=16,
                width=16,
            )
        for resolution in list(getattr(config.dataset, "bucket_resolutions", []) or []):
            try:
                height, width = _parse_dataset_bucket_resolution(resolution)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"dataset.bucket_resolutions contains invalid value: {exc}")
                continue
            _append_layout_request_error(
                errors,
                layout=layout,
                field=f"dataset.bucket_resolutions[{resolution}]",
                frame_count=1,
                height=height,
                width=width,
            )
        if int(config.logging.sample_every_n_steps) > 0:
            _append_layout_request_error(
                errors,
                layout=layout,
                field="logging.sample_frame_count",
                frame_count=int(config.logging.sample_frame_count),
                height=16,
                width=16,
            )
            try:
                resolution = _parse_resolution_pair(config.logging.sample_resolution)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"logging.sample_resolution is invalid: {exc}")
            else:
                if resolution is not None:
                    width, height = resolution
                    _append_layout_request_error(
                        errors,
                        layout=layout,
                        field="logging.sample_resolution",
                        frame_count=1,
                        height=height,
                        width=width,
                    )
        return errors

    def get_trainable_parameters(self):
        if (
            self._lora_report is None
            and not self._direct_expert_tuning
            and not self._sparse_delta_tuning
        ):
            return self.transformer.parameters()
        return [param for param in self.transformer.parameters() if param.requires_grad]

    def get_named_trainable_parameters(self):
        if (
            self._lora_report is None
            and not self._direct_expert_tuning
            and not self._sparse_delta_tuning
        ):
            named = list(self.transformer.named_parameters())
        else:
            named = [
                (name, param)
                for name, param in self.transformer.named_parameters()
                if param.requires_grad
            ]
        # fp32 master: the optimizer groups over this list, so hand it the fp32
        # masters in place of the bf16 working router copies. Working copies keep
        # receiving grads in the forward; the pipeline bridges them each step.
        self._ensure_router_fp32_master()
        master = self._router_fp32_master
        if master:
            return master.substitute_named_params(named)
        return named

    def get_training_model(self) -> Any | None:
        return self.transformer

    def get_model_extension_capabilities(self) -> ModelExtensionCapabilities:
        return ModelExtensionCapabilities(
            adapter_target_presets=True,
            adapter_runtime_controls=True,
            rank_schedule_progress=True,
            adapter_allocation_policy=True,
            adapter_initialization=True,
            adapter_training_policy=True,
        )

    def uses_previous_clean_routing_guidance(self) -> bool:
        return self._sharp_moe_spec is not None

    def get_adapter_target_presets(self) -> dict[str, list[str]]:
        return {key: list(value) for key, value in LORA_TARGET_PRESETS.items()}

    def set_adapter_config(self, adapter_config: Any) -> None:
        adapter_type = str(getattr(adapter_config, "type", "lora")).strip().lower()
        if self._chain_of_experts_enabled and adapter_type != "lora":
            raise ValueError(
                "Chain-of-Experts requires adapter.type='lora'."
            )
        if adapter_type == "sparse_delta":
            preset = str(getattr(adapter_config, "target_preset", "attn_only")).strip()
            if preset not in LORA_TARGET_PRESETS:
                raise ValueError(f"Unsupported sparse-delta target preset '{preset}'.")
            for parameter in self.transformer.parameters():
                parameter.requires_grad_(False)
            matched = apply_sparse_delta_to_linear_modules(
                self.transformer,
                target_modules=LORA_TARGET_PRESETS[preset],
                density=float(getattr(adapter_config, "sparse_delta_density", 0.01)),
                selection=str(
                    getattr(adapter_config, "sparse_delta_selection", "magnitude")
                ),
                seed=0,
            )
            if not matched:
                raise ValueError("Sparse-delta target preset matched no dense linear modules.")
            self._sparse_delta_tuning = True
            self._adapter_type = adapter_type
            self._enable_adjugate_expert_training()
            self._enable_prototypical_routing_training()
            self._enable_sharp_moe_training()
            return
        if adapter_type == "selected_expert":
            if self.has_quantized_frozen_weights():
                raise ValueError(
                    "Direct selected-expert tuning requires dense expert weights."
                )
            for parameter in self.transformer.parameters():
                parameter.requires_grad_(False)
            matched = 0
            for module in self.transformer.modules():
                if not _is_lingbot_grouped_experts(module):
                    continue
                for key in ("w1", "w2", "w3"):
                    parameter = getattr(module, key, None)
                    if isinstance(parameter, torch.nn.Parameter):
                        parameter.requires_grad_(True)
                        matched += 1
            if not matched:
                raise ValueError("No grouped expert tensors are available for direct tuning.")
            self._direct_expert_tuning = True
            self._adapter_type = adapter_type
            self._enable_adjugate_expert_training()
            self._enable_prototypical_routing_training()
            self._enable_sharp_moe_training()
            return
        if adapter_type != "lora":
            raise ValueError(
                "LingBot-Video supports only adapter.type='lora'."
            )
        preset = str(getattr(adapter_config, "target_preset", "attn_only")).strip()
        if preset not in LORA_TARGET_PRESETS:
            raise ValueError(
                f"Unsupported LingBot-Video adapter.target_preset='{preset}'. "
                f"Expected one of: {', '.join(sorted(LORA_TARGET_PRESETS))}."
            )
        rank = int(getattr(adapter_config, "rank", 0))
        alpha = float(getattr(adapter_config, "alpha", rank))
        if rank <= 0:
            raise ValueError("adapter.rank must be > 0 for LingBot-Video LoRA.")
        self._sparse_expert_export = bool(
            getattr(adapter_config, "sparse_expert_export", False)
        )
        lora_init = validate_lora_initializer(
            str(getattr(adapter_config, "lora_init", "kaiming"))
        )
        previous_lora_init = self._lora_init
        use_lora_fa = bool(getattr(adapter_config, "use_lora_fa", False))
        use_dora = bool(getattr(adapter_config, "use_dora", False))
        expert_tensor_lora_backend = str(
            getattr(adapter_config, "expert_tensor_lora_backend", "weight_space")
        ).strip().lower()
        if expert_tensor_lora_backend not in {"weight_space", "activation"}:
            raise ValueError(
                "adapter.expert_tensor_lora_backend must be 'weight_space' or "
                "'activation'."
            )

        previous_use_lora_fa = self._use_lora_fa
        previous_use_dora = self._use_dora
        if (
            initializer_requires_quantization_error(lora_init)
            and self.has_quantized_frozen_weights()
            and self._lora_report is None
        ):
            raise ValueError(
                "adapter.lora_init='loftq' requires adapter injection before "
                "frozen-weight quantization and cannot initialize from a packed base."
            )
        condenser_rank = int(getattr(adapter_config, "condenser_rank", 0))
        condenser_alpha_raw = float(getattr(adapter_config, "condenser_alpha", 0.0))
        condenser_alpha = (
            condenser_alpha_raw
            if condenser_alpha_raw > 0.0
            else float(max(1, condenser_rank))
        )
        self._condenser_rank = condenser_rank
        self._condenser_alpha = condenser_alpha
        self._condenser_init = lora_init
        self._train_router_override = getattr(adapter_config, "train_router", None)
        self._tc_gate_hidden_dim = int(
            getattr(adapter_config, "tc_gate_hidden_dim", 8)
        )
        # Timestep-axis policy (T-LoRA schedule / bands / tc_gate): resolved on
        # every (re)configuration; None for the do-nothing default.
        self._timestep_adapter_policy = TimestepAdapterPolicy.from_adapter_config(
            adapter_config
        )
        target_modules = LORA_TARGET_PRESETS[preset]
        expert_tensor_specs = self.get_expert_tensor_specs()
        linear_targets = collect_lora_linear_target_names(
            self.transformer,
            target_modules=target_modules,
            include=lambda name, module: self._matches_lora_target(
                name,
                target_modules=target_modules,
            ),
        )
        expert_targets = collect_lora_expert_target_names(
            target_specs=target_modules,
            expert_tensor_specs=expert_tensor_specs,
        )
        target_names = (
            tuple(self._lora_report.matched_modules)
            if self._lora_report is not None
            else tuple(dict.fromkeys((*linear_targets, *expert_targets)))
        )
        adaptive_rank_plan = resolve_adaptive_rank_plan(
            getattr(adapter_config, "adaptive_rank_plan_path", ""),
            configured_budget=int(getattr(adapter_config, "rank_budget", 0)),
        )
        allocation_plan = LoRAAllocationPolicy.from_adapter_config(
            adapter_config
        ).resolve(
            target_names,
            exact_ranks=(
                adaptive_rank_plan.ranks
                if adaptive_rank_plan is not None
                else None
            ),
        )
        resolved_ranks = {item.rank for item in allocation_plan.allocations.values()}
        schedule_rank = next(iter(resolved_ranks))
        if self._timestep_adapter_policy is not None and len(resolved_ranks) != 1:
            raise ValueError(
                "Heterogeneous adapter.rank_pattern cannot be combined with "
                "adapter.timestep_rank_schedule; timestep rank masks require one rank."
            )
        if self._lora_report is not None:
            report = self._lora_report
            if (
                report.target_preset != preset
                or report.rank != rank
                or report.alpha != alpha
                or self._lora_allocation_fingerprint != allocation_plan.fingerprint
                or previous_lora_init != lora_init
                or previous_use_lora_fa != use_lora_fa
                or previous_use_dora != use_dora
                or self._expert_tensor_lora_backend != expert_tensor_lora_backend
            ):
                raise ValueError(
                    "LingBot-Video LoRA is already configured with a different "
                    "target preset, allocation policy, or initializer."
                )
            if self._timestep_adapter_policy is not None:
                self._timestep_adapter_modules = collect_lora_adapter_modules(
                    self.transformer
                )
                self._ensure_tc_gate(schedule_rank)
            self._apply_train_router_policy(preset)
            bind_lingbot_router_stage_policy(
                self._router_stage_schedule_controller,
                self._router_adapter_bindings(),
            )
            bind_lingbot_router_distillation(self)
            bind_lingbot_simbal(self)
            self._enable_lightweight_expert_training()
            self._enable_adjugate_expert_training()
            self._enable_decoupled_router_training()
            self._enable_chain_of_experts_training()
            self._enable_prototypical_routing_training()
            self._enable_sharp_moe_training()
            return

        self._lora_init = lora_init
        self._use_lora_fa = use_lora_fa
        self._use_dora = use_dora
        self._expert_tensor_lora_backend = expert_tensor_lora_backend
        for param in self.transformer.parameters():
            param.requires_grad_(False)
        linear_report = apply_lora_to_linear_modules(
            self.transformer,
            target_preset=preset,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            include=lambda name, module: self._matches_lora_target(
                name,
                target_modules=target_modules,
            ),
            require_match=False,
            init=lora_init,
            target_allocations=allocation_plan.allocations,
            use_dora=use_dora,
        )
        expert_report = apply_lora_to_expert_tensors(
            self.transformer,
            target_preset=preset,
            target_specs=target_modules,
            expert_tensor_specs=expert_tensor_specs,
            rank=rank,
            alpha=alpha,
            require_match=False,
            init=lora_init,
            target_allocations=allocation_plan.allocations,
            use_dora=use_dora,
        )
        if expert_tensor_lora_backend == "activation":
            matched_experts = set(expert_report.matched_modules)
            for module_name, module in self.transformer.named_modules():
                if not _is_lingbot_grouped_experts(module):
                    continue
                required = {f"{module_name}.{key}" for key in ("w1", "w2", "w3")}
                selected = required & matched_experts
                if selected and selected != required:
                    raise ValueError(
                        "Activation-space expert tensor LoRA requires w1, w2, and w3 "
                        f"for host {module_name!r}; selected {sorted(selected)!r}."
                    )
                if selected:
                    install_expert_tensor_lora_executor(module)
        matched = tuple(linear_report.matched_modules) + tuple(expert_report.matched_modules)
        skipped = tuple(linear_report.skipped_modules) + tuple(expert_report.skipped_modules)
        if not matched:
            raise ValueError(
                f"LoRA target_preset='{preset}' did not match any linear modules "
                "or expert tensors."
            )
        self._lora_report = LoRAApplicationReport(
            target_preset=preset,
            target_modules=tuple(target_modules),
            rank=rank,
            alpha=alpha,
            matched_modules=matched,
            skipped_modules=skipped,
            allocations=tuple(linear_report.allocations) + tuple(expert_report.allocations),
        )
        self._lora_allocation_fingerprint = allocation_plan.fingerprint
        self._adaptive_rank_plan = adaptive_rank_plan
        self._adapter_type = "lora"
        # Attach shared always-on condenser factors to routed-expert adapters.
        # BF16 parametrizations are attached after frozen-weight quantization.
        if condenser_rank > 0:
            attached = enable_expert_lora_condensers(
                self.transformer,
                rank=condenser_rank,
                alpha=condenser_alpha,
                init=lora_init,
            )
            if attached == 0:
                logger.info(
                    "adapter.condenser_rank=%d: no ActiveExpertLoRA hosts yet "
                    "(preset '%s'); condenser attach deferred to the "
                    "post-quantization migration.",
                    condenser_rank,
                    preset,
                )
        if self._use_lora_fa:
            apply_lora_fa(self.transformer)
        # Router policy under PEFT (DenseMixer analysis). Freeze/warn per the
        # train_router override; recommend freezing for the PCIe-bound regime.
        self._apply_train_router_policy(preset)
        bind_lingbot_router_stage_policy(
            self._router_stage_schedule_controller,
            self._router_adapter_bindings(),
        )
        bind_lingbot_router_distillation(self)
        bind_lingbot_simbal(self)
        self._enable_lightweight_expert_training()
        self._enable_adjugate_expert_training()
        self._enable_decoupled_router_training()
        self._enable_chain_of_experts_training()
        self._enable_prototypical_routing_training()
        self._enable_sharp_moe_training()
        # Cache MoE routers once for per-step expert-subset distribution.
        self._collect_moe_router_modules()
        if self._timestep_adapter_policy is not None:
            # Cache adapter modules once for per-step mask distribution.
            self._timestep_adapter_modules = collect_lora_adapter_modules(
                self.transformer
            )
            self._ensure_tc_gate(schedule_rank)

    def record_gora_allocation(
        self, *, ranks: dict[str, int], fingerprint: str
    ) -> None:
        """Persist calibrated ranks in provider reports without owning GoRA math."""

        if self._lora_report is None or self._lora_init != "gora":
            raise ValueError("GoRA allocation requires a configured GoRA adapter.")
        modules = dict(iter_lora_modules(self.transformer))
        modules.update(dict(iter_lora_expert_tensor_modules(self.transformer)))
        expected = set(self._lora_report.matched_modules)
        if set(ranks) != expected:
            raise ValueError(
                "GoRA allocation target set does not match configured adapters."
            )
        allocations = []
        for name in sorted(expected):
            module = modules.get(name)
            if module is None or int(module.rank) != int(ranks[name]):
                raise ValueError(
                    f"GoRA allocation rank for {name!r} was not applied."
                )
            alpha = float(module.lora_alpha.detach().float().item())
            allocations.append(
                (
                    name,
                    int(ranks[name]),
                    alpha,
                    "alpha_over_sqrt_rank",
                )
            )
        report = self._lora_report
        self._lora_report = LoRAApplicationReport(
            target_preset=report.target_preset,
            target_modules=report.target_modules,
            rank=report.rank,
            alpha=report.alpha,
            matched_modules=report.matched_modules,
            skipped_modules=report.skipped_modules,
            allocations=tuple(allocations),
        )
        self._gora_allocation_fingerprint = str(fingerprint)
        self._gora_ranks = {
            str(name): int(rank) for name, rank in sorted(ranks.items())
        }

    def get_adapter_calibration_root(self) -> nn.Module:
        return self.transformer

    def get_gora_allocation_metadata(self) -> dict[str, Any]:
        if not self._gora_ranks:
            return {}
        return {
            "ranks": dict(self._gora_ranks),
            "fingerprint": str(self._gora_allocation_fingerprint),
        }

    def _ensure_tc_gate(self, rank: int) -> None:
        """Build the TC-LoRA gate hypernetwork when the policy selects tc_gate.

        Attached as ``transformer.tc_lora_gate`` (a fresh trainable submodule
        with ``requires_grad=True`` even though the frozen base is frozen) so it
        rides ``transformer.to(device=...)`` device placement and is picked up by
        ``get_trainable_parameters`` automatically. Idempotent: an already-built
        gate (e.g. after a resume/reconfigure) is reused, never re-initialized.
        """
        policy = self._timestep_adapter_policy
        if policy is None or policy.schedule != "tc_gate":
            self._tc_gate = None
            return
        existing = getattr(self.transformer, "tc_lora_gate", None)
        if existing is None:
            existing = TimestepGateHypernet(
                rank=int(rank), hidden_dim=int(self._tc_gate_hidden_dim)
            )
            self.transformer.tc_lora_gate = existing
        self._tc_gate = existing

    def set_selected_expert_ids(self, expert_ids: Any) -> None:
        """Bind one manual expert set to every routed layer."""

        if not self._direct_expert_tuning:
            raise ValueError("Selected expert ids require direct selected-expert tuning.")
        ids = tuple(sorted(set(int(value) for value in expert_ids)))
        if not ids:
            raise ValueError("At least one selected expert id is required.")
        modules = {
            name: module
            for name, module in self.transformer.named_modules()
            if _is_lingbot_grouped_experts(module)
        }
        self.set_selected_expert_plan({name: ids for name in modules})

    def set_selected_expert_plan(self, plan: Any) -> None:
        """Bind a complete per-layer direct-update plan.

        Keys are provider-owned grouped-expert module names.  Routing remains
        unrestricted; the plan is consumed only by compact optimizer state and
        selected-row checkpoint export.
        """

        if not self._direct_expert_tuning:
            raise ValueError("Selected expert plan requires direct expert tuning.")
        if not isinstance(plan, dict) or not plan:
            raise ValueError("Selected expert plan must be a non-empty mapping.")
        modules = {
            name: module
            for name, module in self.transformer.named_modules()
            if _is_lingbot_grouped_experts(module)
        }
        if set(plan) != set(modules):
            missing = sorted(set(modules) - set(plan))
            extra = sorted(set(plan) - set(modules))
            raise ValueError(
                "Selected expert plan must cover every grouped expert module "
                f"exactly (missing={missing[:4]}, extra={extra[:4]})."
            )
        normalized: dict[str, tuple[int, ...]] = {}
        for name, values in sorted(plan.items()):
            ids = tuple(sorted(set(int(value) for value in values)))
            num_experts = int(getattr(modules[name], "num_experts"))
            if not ids or ids[0] < 0 or ids[-1] >= num_experts:
                raise ValueError(
                    f"Selected expert plan for {name!r} is empty or outside "
                    f"the [0, {num_experts}) expert axis."
                )
            normalized[str(name)] = ids
        self._selected_expert_plan = normalized

    def get_selected_expert_parameter_plan(self) -> dict[str, tuple[int, ...]]:
        """Expand the layer plan to exact named-parameter ownership."""

        return {
            f"{module_name}.{tensor_name}": ids
            for module_name, ids in sorted(self._selected_expert_plan.items())
            for tensor_name in ("w1", "w2", "w3")
        }

    def _matches_lora_target(self, module_name: str, *, target_modules: list[str]) -> bool:
        if ".adjugate_experts." in f".{module_name}.":
            return False
        if target_modules == ["*"]:
            return True
        return any(module_name.endswith(target) for target in target_modules)

    def set_lora_scale(self, scale: float) -> None:
        self._lora_scale = float(scale)
        set_lora_scale(self.transformer, scale)

    def get_lora_scale(self) -> float:
        return float(self._lora_scale)

    def set_adapter_runtime(
        self, *, rank_dropout: float, lora_parameter_dropout: float
    ) -> None:
        set_lora_rank_dropout(self.transformer, float(rank_dropout))
        set_lora_parameter_dropout(
            self.transformer, float(lora_parameter_dropout)
        )

    def set_rank_schedule_progress(
        self,
        *,
        step: int,
        start_step: int,
        end_step: int,
        min_scale: float,
    ) -> None:
        start = int(start_step)
        end = int(end_step)
        if end <= start:
            self._rank_schedule_scale = 1.0
        else:
            progress = min(1.0, max(0.0, (int(step) - start) / float(end - start)))
            floor = float(min_scale)
            self._rank_schedule_scale = floor + ((1.0 - floor) * progress)
        set_lora_rank_schedule_scale(self.transformer, self._rank_schedule_scale)

    def get_rank_schedule_scale(self) -> float:
        return float(self._rank_schedule_scale)

    def _router_adapter_bindings(self) -> tuple[RouterAdapterBinding, ...]:
        bindings: list[RouterAdapterBinding] = []
        for module in self.transformer.modules():
            if not isinstance(module, LingBotVideoRouter):
                continue
            params = getattr(module, "parametrizations", None)
            if params is not None and "weight" in params:
                for sub in params["weight"]:
                    bindings.append(
                        RouterAdapterBinding(weight=module.weight, adapter=sub)
                    )
        return tuple(bindings)

    def _apply_train_router_policy(self, preset: str) -> None:
        """Freeze/warn on router adaptation per adapter.train_router (DenseMixer).

        Default (None) preserves the preset's behavior and only warns when the
        preset adapts the router. ``False`` force-freezes any router adapter;
        ``True`` is explicit opt-in (warns only if it has no effect).
        """
        RouterTrainingPolicy(self._train_router_override).apply(
            self._router_adapter_bindings(),
            target_preset=preset,
            logger=logger,
        )

    def _collect_moe_router_modules(self) -> None:
        named_routers = [
            (name, module)
            for name, module in self.transformer.named_modules()
            if isinstance(module, LingBotVideoRouter)
        ]
        for idx, (name, router) in enumerate(named_routers):
            router._subset_layer_index = int(idx)
            layer_settings = (
                self._layer_router_policy.resolve(idx)
                if self._layer_router_policy is not None
                else None
            )
            if layer_settings is not None:
                router.top_k = int(layer_settings.top_k)
                router._mirai_z_loss_weight = float(
                    layer_settings.z_loss_weight
                )
            elif hasattr(router, "_mirai_z_loss_weight"):
                delattr(router, "_mirai_z_loss_weight")
            router.set_balance_gradient_ratio_capture(
                self._balance_gradient_ratio_enabled
            )
            if self._moe_routing_mode == "expert_choice":
                def route(
                    logits,
                    output_dtype,
                    *,
                    layer_index=idx,
                    layer_name=name,
                    route_scale=float(router.route_scale),
                ):
                    capacity = resolve_capacity_schedule(
                        self._expert_choice_capacity_schedule,
                        step=int(self._expert_choice_step),
                        layer_index=int(layer_index),
                        fallback=float(self._expert_choice_capacity_factor),
                    )
                    capacity_per_sample = None
                    if self._timestep_capacity_policy.enabled:
                        sigmas = self._expert_choice_runtime_sigmas
                        if sigmas is None:
                            raise RuntimeError(
                                "Expert-Choice timestep capacity requires the "
                                "current post-shift diffusion noise levels."
                            )
                        capacity_per_sample = (
                            self._timestep_capacity_policy.capacities(
                                sigmas,
                                tokens_per_sample=int(logits.shape[1]),
                                num_experts=int(logits.shape[2]),
                                fallback_capacity_factor=capacity,
                                flow_shifts=(
                                    self._expert_choice_runtime_flow_shifts
                                ),
                            )
                        )
                    return route_expert_choice_logits(
                        logits,
                        capacity_factor=capacity,
                        capacity_per_sample=capacity_per_sample,
                        route_scale=route_scale,
                        layer_name=layer_name,
                        output_dtype=output_dtype,
                        z_loss_weight=1.0,
                    )

                router.set_expert_choice_extension(route)
            else:
                router.set_expert_choice_extension(None)
            prototypical = None
            if self._prototypical_routing_spec is not None:
                prototypical = getattr(router, "prototypical_routing", None)
                if prototypical is None:
                    prototypical = PrototypicalRouterExtension(
                        hidden_size=int(router.router_weight_shape()[1]),
                        num_experts=int(router.num_experts),
                        spec=self._prototypical_routing_spec,
                        initialization_seed=(
                            int(self._prototypical_routing_spec.seed)
                            + int(idx) * 1_000_003
                        ),
                        device=router.e_score_correction_bias.device,
                    )
                    router.prototypical_routing = prototypical
                elif not isinstance(prototypical, PrototypicalRouterExtension):
                    raise TypeError(
                        f"Router '{name}' has an incompatible prototypical extension."
                    )
                else:
                    expected_topology = {
                        "hidden_size": int(router.router_weight_shape()[1]),
                        "num_experts": int(router.num_experts),
                        "prototype_scale": float(
                            self._prototypical_routing_spec.prototype_scale
                        ),
                        "contrastive_weight": float(
                            self._prototypical_routing_spec.contrastive_weight
                        ),
                        "contrastive_temperature": float(
                            self._prototypical_routing_spec.contrastive_temperature
                        ),
                        "initialization_seed": (
                            int(self._prototypical_routing_spec.seed)
                            + int(idx) * 1_000_003
                        ),
                    }
                    if prototypical.topology() != expected_topology:
                        raise ValueError(
                            f"Router '{name}' has mismatched prototypical topology."
                        )
            if self._sharp_moe_spec is not None:
                saliency = getattr(router, "saliency_harnessing", None)
                expected_topology = {
                    "hidden_size": int(router.router_weight_shape()[1]),
                    "num_experts": int(router.num_experts),
                    "bottleneck_size": int(self._sharp_moe_spec.router_hidden_dim),
                    "initialization_seed": (
                        int(self._sharp_moe_spec.seed) + int(idx) * 1_000_003
                    ),
                }
                if saliency is None:
                    saliency = SaliencyHarnessingRouter(
                        hidden_size=expected_topology["hidden_size"],
                        num_experts=expected_topology["num_experts"],
                        bottleneck_size=expected_topology["bottleneck_size"],
                        initialization_seed=expected_topology["initialization_seed"],
                        device=router.e_score_correction_bias.device,
                    )
                    router.saliency_harnessing = saliency
                elif not isinstance(saliency, SaliencyHarnessingRouter):
                    raise TypeError(
                        f"Router '{name}' has an incompatible saliency extension."
                    )
                elif saliency.topology() != expected_topology:
                    raise ValueError(
                        f"Router '{name}' has mismatched SharpMoE topology."
                    )
            bind_lingbot_route_extensions(
                router,
                layer_name=name,
                diversity=self._diversity_routing_controller,
                expert_dropout=self._expert_dropout_controller,
                dynamic_topk=self._dynamic_topk_controller,
                router_temperature=self._router_temperature_controller,
                selective_sinkhorn=self._selective_sinkhorn_controller,
                prototypical=prototypical,
            )
        self._moe_router_modules = [router for _name, router in named_routers]
        target_top_k = tuple(
            int(router.top_k) for router in self._moe_router_modules
        )
        num_experts = int(self.transformer_config.get("num_experts", 0))
        self._progressive_sparsification_policy = (
            ProgressiveSparsificationPolicy.from_model_params(
                self.model_config.params,
                target_top_k=target_top_k,
                num_experts=num_experts,
            )
        )
        policy = self._progressive_sparsification_policy
        if policy is not None:
            for layer_index, router in enumerate(self._moe_router_modules):
                n_group = int(router.n_group or 1)
                topk_group = int(router.topk_group or n_group)
                available = topk_group * (num_experts // n_group)
                early_top_k = policy.top_k(layer_index=layer_index, step=0)
                if early_top_k > available:
                    raise ValueError(
                        "Progressive-sparsification top_k exceeds the provider's "
                        f"group-limited candidate count in routed layer {layer_index}."
                    )
                layer_settings = (
                    self._layer_router_policy.resolve(layer_index)
                    if self._layer_router_policy is not None
                    else None
                )
                subset_fraction = (
                    float(layer_settings.subset_fraction)
                    if layer_settings is not None
                    else float(self.model_config.params.expert_subset_fraction)
                )
                if math.ceil(subset_fraction * num_experts) < early_top_k:
                    raise ValueError(
                        "Progressive-sparsification top_k exceeds the provider's "
                        f"resolved expert subset in routed layer {layer_index}."
                    )

    def _router_repair_targets(self) -> dict[str, RouterRepairTarget]:
        targets: dict[str, RouterRepairTarget] = {}
        for module_name, module in self.transformer.named_modules():
            if not isinstance(module, LingBotVideoRouter):
                continue
            parameter = module._parameters.get("weight")
            parametrizations = getattr(module, "parametrizations", None)
            if parametrizations is not None and "weight" in parametrizations:
                parameter = parametrizations.weight.original
            if not isinstance(parameter, nn.Parameter):
                raise TypeError(
                    f"LingBot router {module_name!r} has no weight parameter."
                )
            name = f"{module_name}.weight" if module_name else "weight"
            targets[name] = RouterRepairTarget(
                name=name,
                parameter=parameter,
            ).validate()
        if not targets:
            raise ValueError("LingBot router repair found no router weights.")
        return targets

    def _install_decoupled_router_conditioners(self) -> None:
        """Attach independent timestep projections after base checkpoint load."""

        if self._expert_choice_timestep_weight <= 0.0:
            return
        for module in self.transformer.modules():
            if not isinstance(module, LingBotVideoRouter):
                continue
            existing = getattr(module, "decoupled_routing", None)
            if existing is None:
                weight = module._execution_weight(
                    device=module.e_score_correction_bias.device,
                    dtype=torch.float32,
                )
                existing = DecoupledRouterConditioner(
                    hidden_size=int(weight.shape[1]),
                    num_experts=int(weight.shape[0]),
                    timestep_weight=self._expert_choice_timestep_weight,
                    device=weight.device,
                )
            module.set_decoupled_routing(existing)

    def _install_lightweight_expert_pools(self) -> None:
        """Attach model-agnostic lightweight routes after checkpoint loading."""

        if self._lightweight_expert_count <= 0:
            return
        for module in self.transformer.modules():
            if not isinstance(module, LingBotVideoSparseMoeBlock):
                continue
            router = module.router
            existing = getattr(router, "lightweight_experts", None)
            if existing is None:
                physical_weight = router._execution_weight(
                    device=router.e_score_correction_bias.device,
                    dtype=torch.float32,
                )
                existing = LightweightExpertPool.from_physical_router(
                    physical_weight,
                    zero_experts=self._zero_expert_count,
                    copy_experts=self._copy_expert_count,
                    constant_experts=self._constant_expert_count,
                    top_k=self._lightweight_top_k,
                    balance_mode=self._moe_aux_loss_type,
                )
                router.lightweight_experts = existing
            router.set_lightweight_expert_extension(existing)

    def _enable_lightweight_expert_training(self) -> None:
        for module in self.transformer.modules():
            if isinstance(module, LightweightExpertPool):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def _install_adjugate_expert_pools(self) -> None:
        """Attach Grove group experts without modifying the native router."""

        if not self._adjugate_experts_enabled:
            return
        for module in list(self.transformer.modules()):
            if not isinstance(module, LingBotVideoSparseMoeBlock):
                continue
            existing = module.adjugate_experts
            if existing is None:
                topology = AdjugateExpertTopology(
                    num_experts=int(module.num_experts),
                    num_groups=int(self._adjugate_expert_groups),
                    scale=float(self._adjugate_expert_scale),
                ).validate()

                def expert_factory(
                    _group_index: int,
                    *,
                    hidden_size=int(module.hidden_size),
                    intermediate_size=int(
                        self._adjugate_expert_intermediate_size
                    ),
                ) -> Any:
                    return LingBotVideoMLP(hidden_size, intermediate_size)

                def zero_output(expert: Any) -> None:
                    with torch.no_grad():
                        expert.down_proj.weight.zero_()

                existing = AdjugateExpertPool(
                    topology=topology,
                    hidden_size=int(module.hidden_size),
                    intermediate_size=int(
                        self._adjugate_expert_intermediate_size
                    ),
                    expert_kind="swiglu",
                    expert_factory=expert_factory,
                    zero_output_initializer=zero_output,
                )
                module.adjugate_experts = existing
            module.set_adjugate_expert_extension(existing)

    def _enable_adjugate_expert_training(self) -> None:
        for module in self.transformer.modules():
            if isinstance(module, AdjugateExpertPool):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def _enable_decoupled_router_training(self) -> None:
        for module in self.transformer.modules():
            if isinstance(module, DecoupledRouterConditioner):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def _install_chain_of_experts_extensions(self) -> None:
        """Attach one independent low-rank continuation router per MoE layer."""

        if not self._chain_of_experts_enabled:
            return
        for module in list(self.transformer.modules()):
            if not isinstance(module, LingBotVideoSparseMoeBlock):
                continue
            extension = module.chain_of_experts
            if extension is None:
                extension = ChainOfExpertsExtension(
                    ChainOfExpertsSpec(
                        hidden_size=int(module.hidden_size),
                        num_experts=int(module.num_experts),
                        router_rank=int(self._chain_router_rank),
                    ),
                    device=module.router.e_score_correction_bias.device,
                )
            module.set_chain_of_experts_extension(extension)
            for parameter in extension.parameters():
                parameter.requires_grad_(False)

    def _enable_chain_of_experts_training(self) -> None:
        for module in self.transformer.modules():
            if isinstance(module, ChainOfExpertsExtension):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def _enable_prototypical_routing_training(self) -> None:
        for module in self.transformer.modules():
            if isinstance(module, PrototypicalRouterExtension):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def _enable_sharp_moe_training(self) -> None:
        for module in self.transformer.modules():
            if isinstance(module, SaliencyHarnessingRouter):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def supports_expert_choice_progress(self) -> bool:
        return self._moe_routing_mode == "expert_choice"

    def set_expert_choice_progress(self, *, step: int) -> None:
        self._expert_choice_step = int(step)

    def configure_mixture_of_depths(self, policy: MixtureOfDepthsSpec) -> None:
        configure_lingbot_depth_policy(self, policy)

    def configure_dispersive_loss(self, policy: DispersiveLossController) -> None:
        configure_lingbot_dispersive_loss(self, policy)

    def configure_simbal(self, policy: SimBalController) -> None:
        configure_lingbot_simbal(self, policy)

    def configure_preemptive_monitoring(
        self, policy: PreemptiveAttentionMonitor
    ) -> None:
        self._preemptive_attention_monitor = policy

    def configure_moe_token_chunking(self, policy: MoETokenChunkPolicy) -> None:
        for block in self.transformer.blocks:
            setter = getattr(block.ffn, "set_token_chunk_policy", None)
            if not callable(setter):
                raise ValueError(
                    "LingBot-Video MoE block does not expose token chunking."
                )
            setter(policy)

    def configure_domain_expert_specialization(
        self, policy: DomainExpertSpecializationController
    ) -> None:
        configure_lingbot_domain_expert_specialization(self, policy)

    def configure_router_distillation(
        self, policy: RouterDistillationController
    ) -> None:
        configure_lingbot_router_distillation(self, policy)

    def configure_router_stage_schedule(
        self, policy: RouterStageScheduleController
    ) -> None:
        configure_lingbot_router_stage_policy(self, policy)

    def configure_diversity_routing(
        self, policy: DiversityAwareRoutingController
    ) -> None:
        configure_lingbot_diversity_routing(self, policy)

    def configure_expert_dropout(self, policy: ExpertDropoutController) -> None:
        configure_lingbot_expert_dropout(self, policy)

    def configure_router_temperature(
        self, policy: RouterTemperatureController
    ) -> None:
        configure_lingbot_router_temperature(self, policy)

    def configure_selective_sinkhorn(
        self, policy: SelectiveSinkhornController
    ) -> None:
        configure_lingbot_selective_sinkhorn(self, policy)

    def configure_prototypical_routing(
        self, policy: PrototypicalRoutingSpec
    ) -> None:
        configure_lingbot_prototypical_routing(self, policy)

    def configure_sharp_moe(self, policy: SharpMoESpec) -> None:
        configure_lingbot_sharp_moe(self, policy)

    def configure_dataset_routing(self, policy: DatasetRoutingPolicy) -> None:
        errors = policy.validate_model_contract(
            num_experts=int(self.transformer_config.get("num_experts", 0)),
            top_k=int(self.transformer_config.get("num_experts_per_tok", 0)),
        )
        if errors:
            raise ValueError("Invalid LingBot dataset routing policy:\n- " + "\n- ".join(errors))
        self._dataset_routing_policy = policy

    def set_dataset_routing_context(self, context: DatasetRoutingBatch) -> None:
        domains = context.domains
        step = context.step
        training = context.training
        policy = self._dataset_routing_policy
        if not self._moe_router_modules:
            self._collect_moe_router_modules()
        active = bool(training and policy.uses_affinity)
        domain_experts = tuple(
            tuple(policy.expert_affinity[domain]) for domain in domains
        ) if active else ()
        prior_scale = (
            float(policy.routing_prior_weight) * policy.warmup_scale(step)
            if active and policy.specialization_mode == "soft_affinity"
            else 0.0
        )
        hard_active = bool(active and policy.hard_affinity_active(step))
        for router in self._moe_router_modules:
            router.set_dataset_routing_runtime(
                mode=policy.specialization_mode if active else "emergent",
                domain_experts=domain_experts,
                prior_scale=prior_scale,
                hard_active=hard_active,
            )

    def supports_router_subset_progress(self) -> bool:
        return self._router_subset_policy is not None

    def supports_balance_loss_schedule_progress(self) -> bool:
        return self._moe_balance_loss_schedule is not None

    def set_balance_loss_schedule_progress(self, *, step: int) -> None:
        schedule = self._moe_balance_loss_schedule
        if schedule is None:
            return
        current_step = int(step)
        self._moe_aux_loss_weight = schedule.weight(
            self._moe_aux_loss_base_weight,
            step=current_step,
        )
        self._moe_balance_loss_step = current_step

    def supports_progressive_sparsification_progress(self) -> bool:
        return self._progressive_sparsification_policy is not None

    def set_progressive_sparsification_progress(self, *, step: int) -> None:
        policy = self._progressive_sparsification_policy
        if policy is None:
            return
        current_step = int(step)
        for layer_index, router in enumerate(self._moe_router_modules):
            router.top_k = policy.top_k(
                layer_index=layer_index,
                step=current_step,
            )
        self._progressive_sparsification_step = current_step

    def set_router_subset_progress(self, *, step: int, seed: int) -> None:
        """Distribute the per-step expert-subset routing state to all routers.

        Computes the annealed subset size once and installs a deterministic
        per-(seed, step, layer) sampling seed on each router. No-op when the
        subset policy is disabled or has annealed to spanning every expert.
        """
        policy = self._router_subset_policy
        if policy is None:
            return
        if not self._moe_router_modules:
            self._collect_moe_router_modules()
        num_experts = int(self.transformer_config.get("num_experts", 0))
        if num_experts <= 1:
            return
        base = int(seed)
        for idx, router in enumerate(self._moe_router_modules):
            settings = (
                self._layer_router_policy.resolve(idx)
                if self._layer_router_policy is not None
                else None
            )
            top_k = int(router.top_k)
            initial_fraction = (
                None if settings is None else float(settings.subset_fraction)
            )
            size = policy.subset_size(
                num_experts=num_experts,
                top_k=top_k,
                step=int(step),
                initial_fraction=initial_fraction,
            )
            active = size < num_experts
            layer_seed = (
                base * 1000003 + int(step) * 9176 + idx * 61
            ) & 0x7FFFFFFF
            router.set_subset_runtime(
                active=active,
                size=size,
                pool_factor=float(policy.pool_factor),
                seed=int(layer_seed),
                kl_weight=float(policy.kl_weight),
            )

    def _mutable_routing_state_dict(
        self,
        *,
        include_router_bias: bool = True,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if include_router_bias:
            for name, module in self.transformer.named_modules():
                if isinstance(module, LingBotVideoRouter):
                    state[f"moe_router_bias.{name}"] = (
                        module.e_score_correction_bias.detach().cpu().clone()
                    )
        self._ensure_router_fp32_master()
        if self._router_fp32_master:
            for name, tensor in self._router_fp32_master.state_dict().items():
                state[f"router_fp32_master.{name}"] = tensor
        if self._routing_health_enabled:
            state.update(router_drift_checkpoint_state(self._router_drift_tracker))
        state.update(
            phi_balance_state.optional_phi_balance_checkpoint_state(
                self._phi_balance_controller
            )
        )
        return state

    def _load_mutable_routing_state(self, state: dict[str, Any]) -> None:
        modules = dict(self.transformer.named_modules())
        adapter_type = str(state.get("adapter_type", "lora")).strip().lower()
        checkpoint_owned = str(state.get("model_type", "")).strip().lower() == "lingbot-video"
        require_router_bias = (
            adapter_type not in {"selected_expert", "sparse_delta"}
            or self._moe_bias_update_rate > 0.0
        )
        expected_bias_keys = (
            {
                f"moe_router_bias.{name}"
                for name, module in modules.items()
                if isinstance(module, LingBotVideoRouter)
            }
            if require_router_bias
            else set()
        )
        provided_bias_keys = {
            str(key) for key in state if str(key).startswith("moe_router_bias.")
        }
        if checkpoint_owned and provided_bias_keys != expected_bias_keys:
            missing = sorted(expected_bias_keys - provided_bias_keys)
            extra = sorted(provided_bias_keys - expected_bias_keys)
            raise ValueError(
                "Mutable router-bias state must match the configured routers "
                f"exactly (missing={missing[:4]}, extra={extra[:4]})."
            )
        with torch.no_grad():
            for key, value in state.items():
                if not str(key).startswith("moe_router_bias."):
                    continue
                name = str(key)[len("moe_router_bias.") :]
                router = modules.get(name)
                if not isinstance(router, LingBotVideoRouter):
                    raise ValueError(
                        f"Adapter state contains correction bias for unknown router '{name}'."
                    )
                tensor = torch.as_tensor(value).to(
                    device=router.e_score_correction_bias.device,
                    dtype=router.e_score_correction_bias.dtype,
                )
                if tensor.shape != router.e_score_correction_bias.shape:
                    raise ValueError(
                        f"Router correction bias '{name}' has shape {tuple(tensor.shape)}, "
                        f"expected {tuple(router.e_score_correction_bias.shape)}."
                    )
                router.e_score_correction_bias.copy_(tensor)
        prefix = "router_fp32_master."
        master_state = {
            str(key)[len(prefix) :]: value
            for key, value in state.items()
            if str(key).startswith(prefix)
        }
        self._ensure_router_fp32_master()
        expected_master_keys = (
            set(self._router_fp32_master.state_dict())
            if self._router_fp32_master
            else set()
        )
        if checkpoint_owned and set(master_state) != expected_master_keys:
            missing = sorted(expected_master_keys - set(master_state))
            extra = sorted(set(master_state) - expected_master_keys)
            raise ValueError(
                "Router FP32-master state must match the configured trainable "
                f"routers exactly (missing={missing[:4]}, extra={extra[:4]})."
            )
        if self._router_fp32_master:
            self._router_fp32_master.load_state_dict(master_state)
        load_router_drift_checkpoint_state(self._router_drift_tracker, state)
        phi_balance_state.load_optional_phi_balance_checkpoint_state(
            self._phi_balance_controller, state
        )

    def load_adapter_state(self, state: dict[str, Any]) -> None:
        payload = state.get("adapter_state") if isinstance(state, dict) else None
        if isinstance(payload, dict):
            state = payload
        if isinstance(state, dict) and state.get("adapter_type") in {
            "selected_expert",
            "sparse_delta",
        }:
            self.load_state_dict(state)
            return
        if self._lora_report is None:
            raise ValueError(
                "LingBot-Video adapter config must be applied before loading "
                "adapter state."
            )
        # The fp32 router master is namespaced but its leaf names embed the LoRA
        # module names (lora_a/lora_b), so strip it before the LoRA loader, which
        # rejects unknown lora-shaped keys. It is restored separately below.
        lora_state = {
            key: value
            for key, value in state.items()
            if not str(key).startswith(
                (
                    "router_fp32_master.",
                    "router_drift.",
                    "phi_balance.",
                    LIGHTWEIGHT_EXPERT_STATE_PREFIX,
                    ADJUGATE_EXPERT_STATE_PREFIX,
                    DECOUPLED_ROUTING_STATE_PREFIX,
                    CHAIN_OF_EXPERTS_STATE_PREFIX,
                    PROTOTYPICAL_ROUTING_STATE_PREFIX,
                    SHARP_MOE_STATE_PREFIX,
                )
            )
        }
        load_lora_state_dict(self.transformer, lora_state)
        load_lightweight_expert_state(self.transformer, state)
        load_adjugate_expert_state(self.transformer, state)
        load_decoupled_routing_state(self.transformer, state)
        load_chain_of_experts_state(self.transformer, state)
        load_prototypical_routing_state(self.transformer, state)
        load_sharp_moe_state(self.transformer, state)
        if self._tc_gate is not None:
            gate_prefix = "tc_gate."
            gate_state = {
                str(key)[len(gate_prefix) :]: value
                for key, value in state.items()
                if str(key).startswith(gate_prefix)
            }
            if gate_state:
                self._tc_gate.load_state_dict(gate_state)
        self._load_mutable_routing_state(state)

    def get_adapter_quantization_modules(self) -> list[Any]:
        return [module for _, module in iter_lora_modules(self.transformer)]

    def get_memory_feature_capabilities(self) -> MemoryFeatureCapabilities:
        return MemoryFeatureCapabilities(
            block_swap=True,
            quantized_frozen_weights=True,
            packed_frozen_weight_state=True,
            weight_residency_strategy=True,
            runtime_offload_flush=True,
            expert_tensor_specs=True,
            expert_weight_access_policy=True,
            quantize_experts_on_load=True,
            router_quantization_policy=True,
            moe_kernel_backend=True,
            trainable_parameter_offload=True,
        )

    def configure_trainable_parameter_offload(self, enabled: bool) -> None:
        self._trainable_parameter_offload = bool(enabled)

    def get_expert_tensor_specs(self) -> list[ExpertTensorSpec]:
        specs: list[ExpertTensorSpec] = []
        for module_name, module in self.transformer.named_modules():
            if isinstance(module, LingBotVideoRouter):
                specs.append(
                    ExpertTensorSpec(
                        name=f"{module_name}.weight",
                        owner_module=module_name,
                        tensor_name="weight",
                        role="router",
                        layout=("out", "in"),
                        shape=module.router_weight_shape(),
                        dtype=module.router_weight_dtype(),
                        quantizable=False,
                        adapter_targetable=True,
                        routed=False,
                        router=True,
                    )
                )
                continue
            if not (
                _is_lingbot_grouped_experts(module)
                or isinstance(module, CompressedGroupedExperts)
            ):
                continue
            execution_spec = getattr(
                module,
                "mirai_expert_mlp_spec",
                LINGBOT_EXPERT_MLP_EXECUTION_SPEC,
            )
            for projection in execution_spec.projections:
                tensor_name = projection.tensor_name
                role = projection.role
                specs.append(
                    ExpertTensorSpec(
                        name=f"{module_name}.{tensor_name}",
                        owner_module=module_name,
                        tensor_name=tensor_name,
                        role=role,
                        layout=("expert", "out", "in"),
                        shape=_shape_of_expert_tensor(module, tensor_name),
                        dtype=_dtype_of_expert_tensor(module, tensor_name),
                        quantizable=True,
                        adapter_targetable=True,
                        routed=True,
                    )
                )
        return list(validate_expert_tensor_specs(specs))

    def configure_moe_optimization_policy(self, policy: MoEOptimizationPolicy) -> None:
        if policy.router_quantization == "int8_per_channel":
            if self._router_fp32_master_enabled:
                raise ValueError(
                    "INT8 router storage cannot be combined with "
                    "model.params.router_fp32_master=true."
                )
            calibration_path = str(
                policy.router_quantization_calibration_path
            ).strip()
            if calibration_path:
                artifact = load_router_quantization_calibration(calibration_path)
                apply_router_quantization_calibration(
                    _build_router_quantization_targets(self.transformer),
                    artifact,
                )
            else:
                for module in self.transformer.modules():
                    if isinstance(module, LingBotVideoRouter):
                        module.enable_int8_weight()
        access = _expert_access_from_policy(policy)
        self._moe_optimization_policy = policy
        if (
            float(policy.expert_device_cache_gib) > 0.0
            and self._frozen_weight_quantization != "int8"
        ):
            raise ValueError(
                "memory.expert_device_cache_gib > 0 requires "
                "memory.frozen_weight_quantization='int8'."
            )
        if policy.kernel_backend == "megablocks" and access != "full_dequant":
            raise ValueError(
                "LingBot-Video MegaBlocks requires expert_weight_access='full_dequant'; "
                "use moe_kernel_backend='torch' for active/chunked direct routing."
            )
        if policy.kernel_backend == "rotated_int8" and access not in {
            "chunked_dequant",
            "fused_kernel",
        }:
            raise ValueError(
                "LingBot-Video rotated_int8 requires "
                "expert_weight_access='chunked_dequant'."
            )
        if policy.kernel_backend == "rotated_int8" and self._frozen_weight_quantization != "int8":
            raise ValueError(
                "memory.moe_kernel_backend='rotated_int8' requires "
                "memory.frozen_weight_quantization='int8'."
            )
        if policy.kernel_backend == "compiled_packed":
            if self._frozen_weight_quantization not in {
                "gguf_iq4",
                "gguf_iq3",
                "mxfp8_e4m3",
                "mxfp4",
                "nvfp4",
            }:
                raise ValueError(
                    "memory.moe_kernel_backend='compiled_packed' requires GGUF, "
                    "MXFP4, or NVFP4 frozen experts."
                )
            if access not in {"active_dequant", "chunked_dequant"}:
                raise ValueError(
                    "compiled_packed requires active_dequant or chunked_dequant."
                )
        effective_kernel_backend = policy.kernel_backend
        if access == "fused_kernel":
            if self._frozen_weight_quantization != "int8":
                raise ValueError(
                    "LingBot-Video fused_kernel expert access requires "
                    "memory.frozen_weight_quantization='int8'."
                )
            if policy.kernel_backend not in {"auto", "rotated_int8"}:
                raise ValueError(
                    "fused_kernel expert access owns its packed INT8 operation; "
                    "memory.moe_kernel_backend must be 'auto' or 'rotated_int8'."
                )
            effective_kernel_backend = "rotated_int8"
        kernel_backend = build_moe_kernel_backend(
            effective_kernel_backend,
            direct_routed=access in {"active_dequant", "chunked_dequant", "fused_kernel"},
        )
        for block in self.transformer.blocks:
            block.ffn._mirai_moe_kernel_backend = kernel_backend
        self._device_residency_planner = DeviceResidencyPlanner(
            int(policy.device_residency_budget_gib * (1024**3))
        )
        self._expert_device_cache = ExpertDeviceCache(
            int(policy.expert_device_cache_gib * (1024**3))
        )
        self._device_residency_planner.replace(
            "expert_device_cache",
            self._expert_device_cache.snapshot()["capacity_bytes"],
        )
        self._bind_compressed_expert_runtime_policy()

    def _bind_compressed_expert_runtime_policy(self) -> None:
        policy = self._moe_optimization_policy
        access = _expert_access_from_policy(policy)
        for module_name, module in self.transformer.named_modules():
            if isinstance(module, (CompressedGroupedExperts, MixedPrecisionGroupedExperts)):
                module.set_expert_weight_access_policy(
                    expert_weight_access=access,
                    expert_dequant_chunk_size=policy.expert_dequant_chunk_size,
                )
                bind_cache = getattr(module, "bind_expert_device_cache", None)
                if callable(bind_cache):
                    bind_cache(self._expert_device_cache, namespace=module_name)

    def enable_quantized_frozen_weights(
        self,
        quant_type: str,
        **kwargs: Any,
    ) -> None:
        scheme = str(quant_type).strip().lower()
        if scheme not in {
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
                "LingBot-Video quantized frozen weights support only "
                "memory.frozen_weight_quantization='fp8', 'int8', 'nf4', 'gguf_iq4', "
                "'gguf_iq3', 'mxfp8_e4m3', 'mxfp4', or 'nvfp4'."
            )
        quant_format = normalize_quant_format(scheme)
        strategy = normalize_compressed_weights_strategy(kwargs.pop("strategy", "auto"))
        if strategy in {"auto", "disabled", "none"}:
            strategy = "compressed_weights"
        if strategy != "compressed_weights":
            raise ValueError(
                "LingBot-Video frozen-weight quantization strategy must be "
                "'auto' or 'compressed_weights'."
            )
        policy = self._moe_optimization_policy
        if quant_format != "int8" and policy.kernel_backend == "rotated_int8":
            raise ValueError(
                "memory.moe_kernel_backend='rotated_int8' requires "
                "memory.frozen_weight_quantization='int8'."
            )
        block_size = int(kwargs.pop("block_size", self._nf4_blocksize))
        group_sizes = kwargs.pop("group_sizes", "auto")
        precision_plan_path = str(kwargs.pop("precision_plan_path", "")).strip()
        learn_expert_rotations = bool(
            kwargs.pop("learn_expert_rotations", False)
        )
        rotation_optimization_steps = int(
            kwargs.pop("rotation_optimization_steps", 200)
        )
        rotation_learning_rate = float(
            kwargs.pop("rotation_learning_rate", 0.01)
        )
        rotation_row_chunk_size = int(
            kwargs.pop("rotation_row_chunk_size", 4096)
        )
        rotation_checkpoint_interval = int(
            kwargs.pop("rotation_checkpoint_interval", 25)
        )
        rotation_device = kwargs.pop("rotation_device", "cpu")
        rotation_max_workspace_gib = float(
            kwargs.pop("rotation_max_workspace_gib", 2.0)
        )
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported LingBot-Video compressed_weights options: {unsupported}.")
        if self.has_quantized_frozen_weights():
            return
        expert_formats = None
        expert_tensor_formats = None
        if precision_plan_path:
            precision_plan = load_precision_plan(precision_plan_path)
            if isinstance(precision_plan, ExpertPrecisionPlan):
                expert_formats = precision_plan.formats
            elif isinstance(precision_plan, TensorPrecisionPlan):
                source_tensors = {
                    f"{module_name}.experts.{projection}": getattr(
                        module.experts, projection
                    )
                    for module_name, module in self.transformer.named_modules()
                    if isinstance(module, LingBotVideoSparseMoeBlock)
                    for projection in ("w1", "w2", "w3")
                }
                if (
                    router_tensor_fingerprint(source_tensors)
                    != precision_plan.source_weight_fingerprint
                ):
                    raise ValueError(
                        "Tensor precision plan source-weight fingerprint does "
                        "not match the loaded model."
                    )
                if set(precision_plan.module_names()) != {
                    name.rsplit(".", 1)[0]
                    for name in source_tensors
                    if name.endswith(".w1")
                }:
                    raise ValueError(
                        "Tensor precision plan module topology does not match "
                        "the loaded model."
                    )
                expert_tensor_formats = {
                    name: precision_plan.formats_for_module(name)
                    for name in precision_plan.module_names()
                }
            else:  # pragma: no cover - load_precision_plan is exhaustive
                raise TypeError("Unsupported expert precision plan type.")
        report = quantize_compressed_weights_modules(
            self.transformer,
            group_sizes=group_sizes,
            expert_weight_access=_expert_access_from_policy(policy),
            expert_dequant_chunk_size=policy.expert_dequant_chunk_size,
            quant_format=quant_format,
            nf4_blocksize=block_size,
            expert_formats=expert_formats,
            expert_tensor_formats=expert_tensor_formats,
            learn_expert_rotations=learn_expert_rotations,
            rotation_optimization_steps=rotation_optimization_steps,
            rotation_learning_rate=rotation_learning_rate,
            rotation_row_chunk_size=rotation_row_chunk_size,
            rotation_checkpoint_interval=rotation_checkpoint_interval,
            rotation_device=rotation_device,
            rotation_max_workspace_gib=rotation_max_workspace_gib,
            expert_mlp_execution_spec=get_model_family_provider(
                "lingbot-video"
            ).expert_mlp_execution_spec,
        )
        combined = combine_compressed_weights_reports(self._compressed_weights_report, report)
        if combined is None or combined.replaced_modules <= 0:
            raise ValueError("LingBot-Video compressed_weights found no quantizable frozen weights.")
        self._compressed_weights_report = combined
        self._bind_compressed_expert_runtime_policy()
        if precision_plan_path:
            mixed_backend = build_moe_kernel_backend(
                "torch_chunked",
                direct_routed=True,
            )
            for transformer_block in self.transformer.blocks:
                transformer_block.ffn._mirai_moe_kernel_backend = mixed_backend
        # Frozen-weight quantization exposes ActiveExpertLoRA hosts. Attach any
        # configured shared condenser factors idempotently.
        if self._condenser_rank > 0 and self._lora_report is not None:
            attached = enable_expert_lora_condensers(
                self.transformer,
                rank=self._condenser_rank,
                alpha=self._condenser_alpha,
                init=self._condenser_init,
            )
            if attached == 0:
                logger.warning(
                    "adapter.condenser_rank=%d set but no routed-expert adapters "
                    "accept a condenser after quantization (target_preset may "
                    "not include expert tensors).",
                    self._condenser_rank,
                )
            else:
                logger.info(
                    "Condenser LoRA attached to %d routed-expert adapters "
                    "post-quantization (rank=%d).",
                    attached,
                    self._condenser_rank,
                )
        if self._use_lora_fa and self._lora_report is not None:
            apply_lora_fa(self.transformer)

    def enable_structured_expert_sparsity(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized != "2:4":
            raise ValueError("LingBot-Video structured expert sparsity supports only '2:4'.")
        if self.has_quantized_frozen_weights():
            raise ValueError("2:4 expert execution cannot be combined with packed quantization.")
        replaced = apply_structured_2_4_experts(self.transformer, backend="auto")
        if replaced <= 0:
            raise ValueError("No native grouped expert tensors were available for 2:4 sparsity.")
        kernel_backend = build_moe_kernel_backend("torch_chunked", direct_routed=True)
        for block in self.transformer.blocks:
            block.ffn._mirai_moe_kernel_backend = kernel_backend

    def has_quantized_frozen_weights(self) -> bool:
        return self._compressed_weights_report is not None

    def get_quantized_frozen_weight_report(self) -> dict[str, Any] | None:
        if self._compressed_weights_report is None:
            return None
        report = self._compressed_weights_report
        quant_format = normalize_quant_format(self._frozen_weight_quantization)
        return {
            "strategy": "compressed_weights",
            "quant_format": quant_format,
            "linear_modules": report.linear_modules,
            "grouped_expert_modules": report.grouped_expert_modules,
            "quantized_tensors": report.quantized_tensors,
            "quantized_numel": report.quantized_numel,
            "expert_weight_access": report.expert_weight_access,
            "expert_dequant_chunk_size": report.expert_dequant_chunk_size,
            "quantized_experts_on_load": self._quantized_experts_on_load,
        }

    def get_sparse_moe_capabilities(self) -> SparseMoECapabilities:
        return SparseMoECapabilities(
            is_sparse_moe=True,
            architecture="lingbot_video",
            routing=(
                "expert_choice_capacity"
                if self._moe_routing_mode == "expert_choice"
                else "token_choice_top_k_group_limited"
            ),
            routing_granularity="joint_video_text_token",
            num_routed_experts=int(self.transformer_config.get("num_experts", 0)),
            num_shared_experts=int(self.transformer_config.get("n_shared_experts") or 0),
            num_activated_experts=int(self.transformer_config.get("num_experts_per_tok", 0)),
            emits_router_metrics=True,
            notes=(
                "Native Apache-2.0 LingBot-Video transformer port.",
                "Training path consumes cached video latents and cached text embeddings.",
                "Expert-parallel cross-rank dispatch is outside the supported surface.",
                "Native VAE/text encoder loading is provider-owned.",
            ),
        )

    def get_training_auxiliary_losses(self) -> dict[str, Any]:
        losses = self._last_auxiliary_losses
        self._last_auxiliary_losses = {}
        return dict(losses)

    def get_training_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def take_balance_gradient_probe(self) -> BalanceGradientProbe | None:
        probe = self._last_balance_gradient_probe
        self._last_balance_gradient_probe = None
        return probe

    def _capture_inference_routing_trace(self, timestep_tensor: Any) -> None:
        """Snapshot each router's last_top_indices for offline routing analysis.

        Called only on configured eval forwards before the router runtime state
        is cleared. Each entry is tagged with an
        auto-incremented forward index and the router's layer index (its ordinal
        among LingBotVideoRouter modules, stable across forwards). Indices go to
        CPU int16. The configured layer stride bounds retained trace volume.
        """
        stride = self._inference_routing_telemetry_layer_stride
        forward_idx = self._inference_routing_forward_idx
        # One scalar timestep per forward: the denoise loop drives a uniform
        # timestep across the batch. NaN when unavailable keeps entries alignable.
        timestep_value = float("nan")
        try:
            if torch.is_tensor(timestep_tensor) and timestep_tensor.numel() > 0:
                timestep_value = float(
                    timestep_tensor.detach().float().reshape(-1)[0].cpu().item()
                )
        except Exception:  # pragma: no cover - telemetry must never break a forward
            timestep_value = float("nan")
        layer_idx = 0
        for module in self.transformer.modules():
            if not isinstance(module, LingBotVideoRouter):
                continue
            current = layer_idx
            layer_idx += 1
            if stride > 1 and (current % stride) != 0:
                continue
            indices = getattr(module, "last_top_indices", None)
            if indices is None:
                continue
            snapshot = (
                indices.detach().to(device="cpu", dtype=torch.int16).contiguous()
            )
            self._inference_routing_trace.append(
                {
                    "forward_idx": forward_idx,
                    "layer_idx": current,
                    "timestep": timestep_value,
                    "num_experts": int(module.num_experts),
                    "top_indices": snapshot,
                }
            )
        self._inference_routing_forward_idx = forward_idx + 1

    def get_inference_routing_trace(self) -> list[dict[str, Any]]:
        """Captured inference routing entries (see _capture_inference_routing_trace)."""
        return list(self._inference_routing_trace)

    def reset_inference_routing_trace(self) -> None:
        """Drop accumulated routing entries and reset the forward counter."""
        self._inference_routing_trace = []
        self._inference_routing_forward_idx = 0

    def get_checkpoint_report(self) -> dict[str, Any] | None:
        if self._checkpoint_report is None:
            return None
        report = self._checkpoint_report
        return {
            "path": report.path,
            "matched_keys": report.matched_keys,
            "missing_keys": list(report.missing_keys),
            "unexpected_keys": list(report.unexpected_keys),
        }

    def set_compute_autocast_dtype(self, dtype: Any) -> None:
        self._compute_autocast_dtype = dtype
        setattr(self.transformer, "_mirai_compute_dtype", dtype)
        for block in self.transformer.blocks:
            setattr(block, "_mirai_compute_dtype", dtype)

    def set_gradient_checkpointing(self, enabled: bool | str) -> None:
        if enabled is True:
            mode = "standard"
        elif enabled is False:
            mode = "off"
        else:
            mode = str(enabled).strip().lower()
        self._gradient_checkpointing = mode
        setattr(self.transformer, "_mirai_gradient_checkpointing", mode)
        if mode == "selective":
            from mirai.core.training.residency.selective_checkpoint import (
                make_selective_checkpoint_context_fn,
            )

            setattr(
                self.transformer,
                "_mirai_selective_checkpoint_context_fn",
                make_selective_checkpoint_context_fn(),
            )
        elif hasattr(self.transformer, "_mirai_selective_checkpoint_context_fn"):
            delattr(self.transformer, "_mirai_selective_checkpoint_context_fn")

    def set_block_swap(
        self,
        *,
        blocks_to_swap: int,
        mode: str,
        block_swap_backward: bool = True,
    ) -> None:
        self.set_weight_residency_strategy(
            strategy="block_swap" if int(blocks_to_swap) > 0 else "disabled",
            blocks_to_swap=blocks_to_swap,
            mode=mode,
            block_swap_backward=block_swap_backward,
        )

    def _block_residency_scores(self) -> dict[int, float] | None:
        """Best-effort per-block hot-mass scores for routing_hot residency.

        Derived from accumulated router assignment counts when present (populated
        during calibration / when moe_bias_update_rate>0). Returns None at cold
        start, so block_residency_priority='routing_hot' safely degrades to
        index-ordered residency until routing stats exist.
        """
        loads = getattr(self, "_pending_router_loads", None)
        if not loads:
            return None
        as_lists: dict[str, list[float]] = {}
        for name, counts in loads.items():
            try:
                as_lists[str(name)] = [float(v) for v in counts.detach().cpu().tolist()]
            except AttributeError:
                as_lists[str(name)] = [float(v) for v in counts]
        return block_scores_from_router_loads(as_lists)

    def set_weight_residency_strategy(
        self,
        *,
        strategy: str,
        blocks_to_swap: int,
        mode: str,
        block_swap_backward: bool = True,
        offload_dir: str | None = None,
        block_residency_planner: str = "uniform",
        block_swap_prefetch_depth: int = 1,
        block_residency_priority: str = "index",
        block_swap_transfer_strategy: str = "per_tensor",
    ) -> None:
        resolved = str(strategy or "disabled").strip().lower()
        if resolved in {"", "none", "off"}:
            resolved = "disabled"
        if resolved not in {"disabled", "block_swap", "stream_disk"}:
            raise ValueError(
                "LingBot-Video weight residency supports disabled, block_swap, "
                "or stream_disk."
            )
        if (
            resolved == "stream_disk"
            and str(block_swap_transfer_strategy).strip().lower() != "per_tensor"
        ):
            raise ValueError(
                "LingBot-Video stream_disk requires "
                "memory.block_swap_transfer_strategy='per_tensor'."
            )
        block_count = len(self.transformer.blocks)
        requested_swap_count = int(blocks_to_swap)
        if resolved == "stream_disk" and requested_swap_count <= 0:
            requested_swap_count = block_count
        swap_count = max(0, min(requested_swap_count, block_count))
        enabled = resolved != "disabled" and swap_count > 0
        if enabled and self._lora_report is None:
            raise ValueError("LingBot-Video H2D residency requires a configured LoRA adapter.")
        if enabled and self._gradient_checkpointing in {"off", "false", "none", "0", ""}:
            raise ValueError(
                "LingBot-Video H2D residency requires training.gradient_checkpointing "
                "to be standard or aggressive."
            )
        self._weight_residency_strategy = resolved
        self._block_swap_manager = (
            BlockSwapManager(
                total_blocks=block_count,
                blocks_to_swap=swap_count,
                mode=str(mode),
                block_swap_backward=bool(block_swap_backward),
                block_residency_planner=str(block_residency_planner),
                block_swap_prefetch_depth=int(block_swap_prefetch_depth),
                block_residency_priority=str(block_residency_priority),
                block_swap_transfer_strategy=str(block_swap_transfer_strategy),
                block_scores=self._block_residency_scores(),
                disk_offload_dir=(
                    str(offload_dir) if resolved == "stream_disk" else None
                ),
            )
            if enabled
            else None
        )
        if self._block_swap_manager is None:
            self._device_residency_planner.replace_many(
                {"block_resident_set": 0, "block_transfer_window": 0}
            )
        self._block_swap_state = {
            "enabled": enabled,
            "blocks_to_swap": swap_count,
            "mode": str(mode).strip().lower(),
            "block_swap_backward": bool(block_swap_backward),
            "weight_residency_strategy": resolved,
            "h2d_only_frozen_base": enabled,
            "block_swap_transfer_strategy": str(
                block_swap_transfer_strategy
            ).strip().lower(),
            "events": [],
        }
        setattr(self.transformer, "_mirai_block_swap_manager", self._block_swap_manager)

    def get_block_swap_units(self) -> list[tuple[int, Any]]:
        return [(idx, block) for idx, block in enumerate(self.transformer.blocks)]

    def get_compilation_regions(self) -> list[CompilationRegion]:
        return [
            CompilationRegion(
                name=f"transformer.blocks.{idx}",
                owner=block,
            )
            for idx, block in enumerate(self.transformer.blocks)
        ]

    def get_activation_offload_regions(self) -> list[ActivationOffloadRegion]:
        return [
            ActivationOffloadRegion(
                name=f"transformer.blocks.{idx}",
                owner=block,
            )
            for idx, block in enumerate(self.transformer.blocks)
        ]

    def configure_compilation_token_buckets(
        self,
        plan: TokenBucketPlan | None,
    ) -> None:
        setattr(self.transformer, "_mirai_compile_token_bucket_plan", plan)

    def get_block_swap_state(self) -> dict[str, Any]:
        state = dict(self._block_swap_state)
        manager = self._block_swap_manager
        if manager is not None:
            state.update(manager.snapshot())
            state["enabled"] = True
            state["weight_residency_strategy"] = self._weight_residency_strategy
            state["h2d_only_frozen_base"] = True
        return state

    def get_device_residency_state(self) -> dict[str, object]:
        return self._device_residency_planner.snapshot()

    def place_offloaded_modules(self, *, device: Any, strategy: str) -> None:
        resolved = str(strategy or "disabled").strip().lower()
        if resolved in {"", "none", "off", "disabled"}:
            self.transformer.to(device=device)
            return
        if resolved not in {"block_swap", "stream_disk"}:
            raise ValueError(
                "LingBot-Video offloaded placement supports block_swap or stream_disk."
            )
        if self._lora_report is None:
            raise ValueError("LingBot-Video offloaded placement requires LoRA.")
        if not self.has_quantized_frozen_weights():
            raise ValueError(
                "LingBot-Video H2D residency requires "
                "a compressed_weights frozen-weight quantization format."
            )
        manager = self._block_swap_manager
        if manager is None:
            raise RuntimeError("Lingbot block residency manager is not configured.")
        blocks = list(self.transformer.blocks)
        self._optimizer_compute_device = torch.device(device)
        if self._trainable_parameter_offload:
            cast_trainable_tensors(
                self.transformer,
                dtype=self._compute_autocast_dtype,
            )
        else:
            move_trainable_tensors(
                self.transformer,
                device=device,
                dtype=self._compute_autocast_dtype,
            )
        move_tensors_outside_modules(
            self.transformer,
            excluded_modules=blocks,
            device=device,
        )
        manager.set_block_scores(self._block_residency_scores())
        block_units = self.get_block_swap_units()
        self._device_residency_planner.replace_many(
            manager.estimate_device_residency_reservations(block_units)
        )
        manager.bind(block_units, device=device)
        self._device_residency_planner.replace_many(
            manager.device_residency_reservations()
        )
        setattr(self.transformer, "_mirai_block_swap_manager", manager)

    def flush_runtime_offloads(self) -> None:
        if self._block_swap_manager is not None:
            self._block_swap_manager.reset_step()

    def finish_backward_offloads(self) -> None:
        _clear_router_runtime_state(self.transformer)
        clear_checkpoint_auxiliary_terms(self.transformer)
        clear_checkpoint_intermediate_terms(self.transformer)
        if self._block_swap_manager is not None:
            self._block_swap_manager.finish_backward()

    def prepare_optimizer_step(self) -> None:
        # fp32 master: the optimizer groups over the fp32 masters, so bridge the
        # (already clipped) working-copy grads up into the masters before step().
        master = self._router_fp32_master
        if master:
            master.sync_grads_to_master()
        if not self._trainable_parameter_offload:
            return
        move_trainable_tensors(
            self.transformer,
            device=self._optimizer_compute_device,
            dtype=self._compute_autocast_dtype,
        )

    def finish_optimizer_step(self) -> None:
        self._apply_router_bias_update()
        # fp32 master: re-materialize the bf16 working router copies from the
        # freshly stepped fp32 masters, then drop the consumed working grads (the
        # optimizer's zero_grad only reaches the masters it holds).
        master = self._router_fp32_master
        if master:
            master.materialize()
            master.clear_working_grads()
        # A completed optimizer step closes the accumulation window; start the
        # next global-batch load window fresh.
        if self._global_batch_load_accumulator is not None:
            self._global_batch_load_accumulator.reset()
        if self._trainable_parameter_offload:
            move_trainable_tensors(
                self.transformer,
                device="cpu",
                dtype=self._compute_autocast_dtype,
            )

    def discard_optimizer_step(self) -> None:
        """Discard mutable state accumulated by a skipped optimizer step."""
        self._pending_router_loads = {}
        master = self._router_fp32_master
        if master:
            master.clear_working_grads()
        if self._global_batch_load_accumulator is not None:
            self._global_batch_load_accumulator.reset()
        if self._trainable_parameter_offload:
            move_trainable_tensors(
                self.transformer,
                device="cpu",
                dtype=self._compute_autocast_dtype,
            )

    def _global_batch_prime_injected_fractions(self) -> None:
        """Set each router's injected load fraction from the window-so-far.

        Runs only in global_batch scope. On the first micro-batch of a window the
        accumulator is empty -> the injected fraction is cleared to None, so the
        aux term falls back to the local per-micro-batch bincount (this is what
        keeps accumulation == 1 equivalent to microbatch scope).
        """
        accumulator = self._global_batch_load_accumulator
        if accumulator is None or not self.transformer.training:
            return
        for name, module in self.transformer.named_modules():
            if isinstance(module, LingBotVideoRouter):
                module._mirai_global_batch_load_fraction = accumulator.fraction(name)

    def _global_batch_accumulate_and_inject(self) -> None:
        """Fold this micro-batch's counts into the window and refresh injections."""
        accumulator = self._global_batch_load_accumulator
        if accumulator is None or not self.transformer.training:
            return
        for name, module in self.transformer.named_modules():
            if not isinstance(module, LingBotVideoRouter):
                continue
            indices = getattr(module, "training_top_indices", None)
            if indices is not None:
                accumulator.accumulate(
                    name, dispatch_counts(indices, int(module.num_experts))
                )
            module._mirai_global_batch_load_fraction = accumulator.fraction(name)

    def _ensure_router_fp32_master(self) -> None:
        """Lazily build the fp32 router master once trainability is resolved.

        Inert (no master, no memory) unless ``router_fp32_master`` is enabled AND
        at least one trainable router parameter exists (train_router on).
        """
        if self._router_fp32_master_built or not self._router_fp32_master_enabled:
            return
        self._router_fp32_master_built = True
        named = [
            (name, param)
            for name, param in self.transformer.named_parameters()
            if param.requires_grad
        ]
        master = RouterFp32Master(named)
        self._router_fp32_master = master if master else None

    def get_router_fp32_master_names(self) -> frozenset[str]:
        """Qualified names whose effective storage is the fp32 master (or empty)."""
        self._ensure_router_fp32_master()
        master = self._router_fp32_master
        return master.mastered_names if master else frozenset()

    def _apply_router_bias_update(self) -> None:
        rate = float(self._moe_bias_update_rate)
        pending = self._pending_router_loads
        self._pending_router_loads = {}
        if rate <= 0.0 or not pending:
            return
        modules = dict(self.transformer.named_modules())
        with torch.no_grad():
            for name, counts_cpu in pending.items():
                router = modules.get(name)
                if not isinstance(router, LingBotVideoRouter):
                    raise RuntimeError(
                        f"Router load state references missing module '{name}'."
                    )
                deviation = counts_cpu - counts_cpu.mean()
                bias = router.e_score_correction_bias
                updated = bias.float().cpu() - rate * deviation.sign().float()
                if self._moe_bias_centering:
                    updated = updated - updated.mean()
                bias.copy_(updated.to(device=bias.device, dtype=bias.dtype))

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if args or kwargs:
            return super().state_dict(*args, **kwargs)
        if self._direct_expert_tuning:
            if not self._selected_expert_plan:
                raise RuntimeError("Direct expert tuning has no bound expert selection.")
            state: dict[str, Any] = {
                "model_type": "lingbot-video",
                "adapter_type": "selected_expert",
                "selected_expert_plan": {
                    name: torch.tensor(ids, dtype=torch.long)
                    for name, ids in sorted(self._selected_expert_plan.items())
                },
            }
            for name, module in self.transformer.named_modules():
                if not _is_lingbot_grouped_experts(module):
                    continue
                ids = torch.tensor(
                    self._selected_expert_plan[name],
                    dtype=torch.long,
                )
                for tensor_name in ("w1", "w2", "w3"):
                    weight = getattr(module, tensor_name)
                    state[f"selected_expert.{name}.{tensor_name}"] = (
                        weight.detach().index_select(0, ids.to(weight.device)).cpu().clone()
                    )
            state.update(export_adjugate_expert_state(self.transformer))
            state.update(export_prototypical_routing_state(self.transformer))
            state.update(export_sharp_moe_state(self.transformer))
            state.update(
                self._mutable_routing_state_dict(
                    include_router_bias=self._moe_bias_update_rate > 0.0
                )
            )
            return state
        if self._sparse_delta_tuning:
            state: dict[str, Any] = {
                "model_type": "lingbot-video",
                "adapter_type": "sparse_delta",
            }
            for name, module in self.transformer.named_modules():
                if isinstance(module, SparseDeltaLinear):
                    state[f"sparse_delta.{name}.values"] = (
                        module.values.detach().cpu().clone()
                    )
                    state[f"sparse_delta.{name}.indices"] = (
                        module.indices.detach().cpu().clone()
                    )
            state.update(export_adjugate_expert_state(self.transformer))
            state.update(export_prototypical_routing_state(self.transformer))
            state.update(export_sharp_moe_state(self.transformer))
            state.update(
                self._mutable_routing_state_dict(
                    include_router_bias=self._moe_bias_update_rate > 0.0
                )
            )
            return state
        if self._lora_report is not None:
            state = lora_state_dict(
                self.transformer,
                sparse_expert_export=self._sparse_expert_export,
            )
            state.update(export_lightweight_expert_state(self.transformer))
            state.update(export_adjugate_expert_state(self.transformer))
            state.update(export_decoupled_routing_state(self.transformer))
            state.update(export_chain_of_experts_state(self.transformer))
            state.update(export_prototypical_routing_state(self.transformer))
            state.update(export_sharp_moe_state(self.transformer))
            if self._tc_gate is not None:
                # TC-LoRA gate hypernetwork params (namespaced; not a LoRA
                # module, so lora_state_dict skips them). Round-trips on resume.
                for gate_key, gate_value in self._tc_gate.state_dict().items():
                    state[f"tc_gate.{gate_key}"] = gate_value.detach().cpu().clone()
            state.update(self._mutable_routing_state_dict())
            state["model_type"] = "lingbot-video"
            state["adapter_type"] = self._adapter_type
            state["target_preset"] = self._lora_report.target_preset
            state["rank"] = self._lora_report.rank
            state["alpha"] = self._lora_report.alpha
            return state
        return {
            "transformer": self.transformer.state_dict(),
            "model_type": "lingbot-video",
        }

    def load_state_dict(self, state: dict[str, Any], strict: bool = True):
        if isinstance(state, dict) and state.get("adapter_type") == "selected_expert":
            raw_plan = state.get("selected_expert_plan")
            if not isinstance(raw_plan, dict) or not raw_plan:
                raise ValueError("Selected-expert checkpoint has no per-layer plan.")
            checkpoint_plan: dict[str, tuple[int, ...]] = {}
            for name, ids_value in raw_plan.items():
                if not isinstance(ids_value, torch.Tensor):
                    raise ValueError(
                        "Selected-expert checkpoint plan values must be tensors."
                    )
                checkpoint_plan[str(name)] = tuple(
                    int(value) for value in ids_value.tolist()
                )
            if checkpoint_plan != self._selected_expert_plan:
                raise ValueError("Selected-expert checkpoint selection mismatch.")
            modules = dict(self.transformer.named_modules())
            expected_keys = {
                f"selected_expert.{module_name}.{tensor_name}"
                for module_name in checkpoint_plan
                for tensor_name in ("w1", "w2", "w3")
            }
            provided_keys = {
                str(key)
                for key in state
                if str(key).startswith("selected_expert.")
            }
            if provided_keys != expected_keys:
                missing = sorted(expected_keys - provided_keys)
                extra = sorted(provided_keys - expected_keys)
                raise ValueError(
                    "Selected-expert checkpoint must contain every selected "
                    f"expert tensor exactly (missing={missing[:4]}, extra={extra[:4]})."
                )
            for key in sorted(expected_keys):
                value = state[key]
                qualified = key[len("selected_expert.") :]
                module_name, tensor_name = qualified.rsplit(".", 1)
                module = modules.get(module_name)
                if module is None or tensor_name not in {"w1", "w2", "w3"}:
                    raise ValueError(f"Unknown selected-expert checkpoint target '{qualified}'.")
                weight = getattr(module, tensor_name)
                expected_shape = (
                    len(checkpoint_plan[module_name]),
                    *tuple(int(dim) for dim in weight.shape[1:]),
                )
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise ValueError(
                        f"Selected-expert checkpoint tensor '{qualified}' has "
                        f"shape {getattr(value, 'shape', None)}; expected {expected_shape}."
                    )
                index = torch.tensor(
                    checkpoint_plan[module_name],
                    dtype=torch.long,
                    device=weight.device,
                )
                weight.data.index_copy_(
                    0,
                    index,
                    value.to(device=weight.device, dtype=weight.dtype),
                )
            load_adjugate_expert_state(self.transformer, state)
            load_prototypical_routing_state(self.transformer, state)
            load_sharp_moe_state(self.transformer, state)
            self._load_mutable_routing_state(state)
            return None
        if isinstance(state, dict) and state.get("adapter_type") == "sparse_delta":
            modules = dict(self.transformer.named_modules())
            sparse_modules = {
                name: module
                for name, module in modules.items()
                if isinstance(module, SparseDeltaLinear)
            }
            expected_keys = {
                f"sparse_delta.{name}.{field}"
                for name in sparse_modules
                for field in ("values", "indices")
            }
            provided_keys = {
                str(key) for key in state if str(key).startswith("sparse_delta.")
            }
            if provided_keys != expected_keys:
                missing = sorted(expected_keys - provided_keys)
                extra = sorted(provided_keys - expected_keys)
                raise ValueError(
                    "Sparse-delta checkpoint must contain every configured target "
                    f"exactly (missing={missing[:4]}, extra={extra[:4]})."
                )
            for name, module in sorted(sparse_modules.items()):
                key = f"sparse_delta.{name}.values"
                value = state[key]
                indices_key = f"sparse_delta.{name}.indices"
                indices = state[indices_key]
                if not isinstance(indices, torch.Tensor) or not torch.equal(
                    module.indices.cpu(), indices.cpu()
                ):
                    raise ValueError(
                        f"Sparse-delta support mismatch for target '{name}'."
                    )
                if not isinstance(value, torch.Tensor) or value.shape != module.values.shape:
                    raise ValueError(
                        f"Sparse-delta values shape mismatch for target '{name}'."
                    )
                module.values.data.copy_(
                    value.to(device=module.values.device, dtype=module.values.dtype)
                )
            load_adjugate_expert_state(self.transformer, state)
            load_prototypical_routing_state(self.transformer, state)
            load_sharp_moe_state(self.transformer, state)
            self._load_mutable_routing_state(state)
            return None
        adapter_payload = state.get("adapter_state") if isinstance(state, dict) else None
        if isinstance(adapter_payload, dict):
            self.load_adapter_state(adapter_payload)
            return None
        if isinstance(state, dict) and any(
            key.endswith(".lora_a") or key.endswith(".lora_b")
            for key in state
        ):
            self.load_adapter_state(state)
            return None
        payload = state.get("transformer") if isinstance(state, dict) else None
        if isinstance(payload, dict):
            return self.transformer.load_state_dict(payload, strict=bool(strict))
        return super().load_state_dict(state, strict=bool(strict))


_lingbot_provider = LingBotVideoModelFamilyProvider("lingbot-video")
_lingbot_provider.pipeline_type = LingBotVideoPipeline
register_model_family_provider("lingbot-video", _lingbot_provider)
