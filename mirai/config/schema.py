"""Strict config schema with unknown-key rejection."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from difflib import get_close_matches
import math
from typing import Any


class ConfigError(ValueError):
    """Raised for invalid config payloads."""


def _unknown_key_error(path: str, key: str, allowed: set[str]) -> ConfigError:
    suggestion = get_close_matches(key, sorted(allowed), n=1)
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    return ConfigError(f"Unknown config key '{path}.{key}'.{hint}")


def _forbid_unknown(path: str, payload: dict[str, Any], allowed: set[str]) -> None:
    for key in payload:
        if key not in allowed:
            raise _unknown_key_error(path, key, allowed)


def _expect_table(path: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Expected table at '{path}', got {type(value).__name__}.")
    return value


def _validate_no_control_path(path_key: str, value: str) -> str:
    text = str(value)
    if any(ch in text for ch in ("\n", "\r", "\t")):
        raise ConfigError(
            f"{path_key} contains control characters. On Windows, use forward slashes "
            "(e.g. `/path/to/moe_dataset`) or escape backslashes (`\\\\`)."
        )
    return text


_MODEL_DTYPE_NAMES = frozenset(
    {
        "fp32",
        "float32",
        "f32",
        "fp16",
        "float16",
        "f16",
        "half",
        "bf16",
        "bfloat16",
    }
)


def _validate_model_dtype(value: Any) -> str:
    dtype = str(value).strip().lower()
    if dtype not in _MODEL_DTYPE_NAMES:
        allowed = ", ".join(sorted(_MODEL_DTYPE_NAMES))
        raise ConfigError(f"model.dtype must be one of: {allowed}; got '{value}'.")
    return dtype


def _parse_expert_choice_capacity_schedule(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ConfigError(
            "model.params.moe_expert_choice_capacity_schedule must be an array of tables."
        )
    allowed = {"start_step", "end_step", "first_layer", "end_layer", "capacity_factor"}
    schedule: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConfigError(
                "model.params.moe_expert_choice_capacity_schedule"
                f"[{index}] must be a table."
            )
        _forbid_unknown(
            f"model.params.moe_expert_choice_capacity_schedule[{index}]",
            item,
            allowed,
        )
        schedule.append(dict(item))
    return schedule


def _parse_layer_router_policy(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ConfigError(
            "model.params.moe_layer_router_policy must be an array of tables."
        )
    allowed = {
        "first_layer",
        "end_layer",
        "top_k",
        "subset_fraction",
        "z_loss_weight",
    }
    policy: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConfigError(
                f"model.params.moe_layer_router_policy[{index}] must be a table."
            )
        _forbid_unknown(
            f"model.params.moe_layer_router_policy[{index}]",
            item,
            allowed,
        )
        policy.append(dict(item))
    return policy


def _parse_progressive_sparsification_policy(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ConfigError(
            "model.params.moe_progressive_sparsification_policy must be an "
            "array of tables."
        )
    allowed = {"first_layer", "end_layer", "top_k"}
    policy: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConfigError(
                "model.params.moe_progressive_sparsification_policy"
                f"[{index}] must be a table."
            )
        _forbid_unknown(
            f"model.params.moe_progressive_sparsification_policy[{index}]",
            item,
            allowed,
        )
        policy.append(dict(item))
    return policy


@dataclass
class ModelParams:
    variant: str = "lingbot-video-moe-30b-a3b"
    flow_shift: float = 3.0
    flow_shift_mode: str = "constant"
    flow_shift_base_seq_len: int = 256
    flow_shift_max_seq_len: int = 4096
    flow_shift_max: float = 12.0
    vae_chunk_size: int = 16
    lora_dropout: float = 0.0
    strict_native_assets: bool = True
    moe_expert_backend: str = "grouped_mm"
    moe_pad_backend: str = "auto"
    moe_reorder_backend: str = "sort"
    moe_restore_backend: str = "chunked_scatter"
    moe_restore_chunk_size: int = 128
    moe_fused_qkv_linear: bool = False
    inference_bf16_fastmath: bool = False
    inference_routing_telemetry: bool = False
    inference_routing_telemetry_layer_stride: int = 1
    num_experts: int = 4
    experts_per_token: int = 2
    shared_experts: int = 1
    hidden_size: int = 32
    num_layers: int = 2
    attention_heads: int = 4
    latent_channels: int = 1
    patch_size: int = 1
    expert_capacity_factor: float = 1.0
    moe_routing_mode: str = "token_choice"
    moe_chain_of_experts: bool = False
    moe_chain_router_rank: int = 0
    moe_layer_router_policy: list[dict[str, Any]] = field(default_factory=list)
    moe_progressive_sparsification_transition_step: int = 0
    moe_progressive_sparsification_policy: list[dict[str, Any]] = field(
        default_factory=list
    )
    moe_expert_choice_capacity_factor: float = 1.0
    moe_expert_choice_capacity_schedule: list[dict[str, Any]] = field(
        default_factory=list
    )
    moe_expert_choice_timestep_capacity_schedule: str = "disabled"
    moe_expert_choice_timestep_capacity_span: float = 0.0
    moe_expert_choice_coverage_alarm_threshold: float = 0.0
    moe_zero_experts: int = 0
    moe_copy_experts: int = 0
    moe_constant_experts: int = 0
    moe_lightweight_top_k: int = 0
    moe_adjugate_experts: bool = False
    moe_adjugate_expert_groups: int = 0
    moe_adjugate_expert_intermediate_size: int = 128
    moe_adjugate_expert_scale: float = 0.05
    moe_router_timestep_weight: float = 0.0
    moe_aux_loss_type: str = "model_native"
    moe_aux_loss_weight: float = 0.01
    moe_router_z_loss_weight: float = 0.0
    moe_router_similarity_loss_weight: float = 0.0
    moe_dynamic_topk_min: int = 0
    moe_dynamic_topk_average: float = 0.0
    moe_spatiotemporal_routing_weight: float = 0.0
    moe_spatiotemporal_routing_max_edges: int = 4096
    moe_bias_update_rate: float = 0.0
    moe_bias_centering: bool = True
    # Load-balance mode: "aux_loss" (default, aux term in the loss graph +
    # optional online bias), "bias_only" (online bias only, aux term dropped
    # from backward), "off" (no balance pressure -- frozen-router LoRA).
    moe_balance_mode: str = "aux_loss"
    # Optional DMEP-style phase boundary: retain auxiliary balance pressure
    # before this global step, then disable only that loss term.
    moe_balance_loss_disable_step: int = 0
    # Load-balance scope: "microbatch" uses a per-micro-batch load estimate;
    # "global_batch" accumulates the
    # token-fraction load is accumulated across the gradient-accumulation window,
    # matching what one large global batch would see). At accumulation == 1 the
    # two are exactly equivalent. See mirai/core/moe/adaptation/global_balance.py.
    moe_balance_scope: str = "microbatch"
    moe_balance_grad_ratio_telemetry: bool = False
    moe_balance_grad_ratio_threshold: float = 0.1
    # Population-level phi-balancing (arXiv:2605.15403). A zero weight is fully
    # disabled. The EMA tracks pre-top-k soft routing probabilities per layer.
    moe_phi_balance_weight: float = 0.0
    moe_phi_balance_ema_rate: float = 0.01
    moe_phi_balance_potential: str = "negative_entropy"
    moe_router_variance_loss_weight: float = 0.0
    moe_expert_orthogonality_loss_weight: float = 0.0
    moe_swiglu_specialization_loss_weight: float = 0.0
    moe_cross_layer_coupling_loss_weight: float = 0.0
    moe_specialization_max_tokens: int = 256
    # Opt-in fp32 master copy of trainable router parameters (default off). When
    # the router is trainable, sub-ULP bf16 router updates truncate to zero; the
    # fp32 master accumulates them losslessly and re-materializes the bf16 working
    # copy each step. Inert (zero cost) when train_router is off/absent. See
    # mirai/core/training/router_fp32_master.py.
    router_fp32_master: bool = False
    # Stochastic per-step expert-subset routing (Mirai variant of EMoE,
    # arXiv 2509.21892). Disabled settings leave routing unchanged. Per step per MoE
    # layer, restrict routing to a hot-biased sampled subset of size
    # ceil(fraction*num_experts) so the PCIe working set shrinks by `fraction`.
    # fraction == 1.0 (default) disables it. See mirai/core/moe/routing/subset.py.
    expert_subset_fraction: float = 1.0
    # Candidate pool = top ceil(pool_factor*size) hottest experts by routing mass
    # (>= 1; 1.0 = deterministic hottest-size, larger = more warm-tail exploration).
    expert_subset_pool_factor: float = 2.0
    # Anneal fraction -> 1.0 over this many steps (0 = constant fraction).
    expert_subset_anneal_steps: int = 0
    # Optional reverse-KL router-consistency loss weight (only applied when the
    # router is trainable; default 0 = off).
    expert_subset_router_kl_weight: float = 0.0
    # Routing-health diagnostics pack (opt-in, default off). When True the active
    # model provider emits three extra detached per-step alarms alongside the
    # routing-stability metrics: expert-output homogenization cosine proxy,
    # per-depth single-expert deadlock duration, and a bf16 router-update
    # underflow fraction. Telemetry only -- never touches the loss/gradient path.
    # See mirai/core/moe/routing_health.py.
    moe_routing_health: bool = False
    moe_selection_margin: bool = False
    # Offline paired evidence: identical batch/noise/timestep, first with
    # training-time routing behavior and then with inference-time behavior.
    # "off" is inert; "report" arms only the calibration CLI/provider seam.
    moe_routing_agreement_evidence: str = "off"
    # Dataset-conditioned structured expert pruning gate (offline, default off).
    # "off" (default) = no pruning; the trainer/model do not branch
    # on this key; it is a marker consumed only by scripts/tools/prune_experts.py, which
    # removes cold experts from the frozen base and re-emits a smaller packed-state
    # + reduced num_experts). "prune" arms that offline script. Numeric knobs
    # (keep-fraction / score-threshold / min-keep / calibration-steps) are CLI args
    # on the script, keeping the hot-path schema surface to one gate. See
    # Consumed by the offline structured-pruning workflow.
    expert_pruning: str = "off"
    # Exact offline ranking criterion: route frequency, REAP, MAN, or MSAN.
    # Inert while expert_pruning="off".
    expert_pruning_criterion: str = "frequency"
    # Offline packed-expert consolidation: retained prototypes or HC-SMoE-style
    # output clustering with frequency-weighted physical weight merging.
    expert_consolidation: str = "off"
    # Data-free physical expert expansion. The dedicated offline transform
    # interleaves uniform copies, diversifies shared SwiGLU channels, and
    # expands sibling router rows in one lineage-bound packed artifact.
    expert_upcycling: str = "off"
    expert_upcycling_copies: int = 0
    expert_upcycling_reinit_ratio: float = 0.5
    expert_upcycling_seed: int = 42
    # Offline physical expert-weight compression. "off" is inert;
    # Provider-backed schema-v4 packed artifacts require an exact explicit
    # selection. Selecting such an artifact without its gate fails before load.
    expert_weight_compression: str = "off"
    # Offline FlexMoE ranking/action calibration runs against the complete
    # schema-v1/v2/v3 expert source before a ragged provider is exported.
    flexmoe_calibration: str = "off"
    # Offline Router-KD repair is armed separately from runtime patch loading.
    # The artifact path remains empty by default, preserving the packed base.
    post_compression_router_repair: str = "off"
    router_repair_artifact_path: str = ""
    # Offline MoEQuant-style route-affinity evidence collection. "off" is
    # inert; "affinity" arms only the calibration CLI/provider hook.
    expert_quantization_calibration: str = "off"
    # Offline learned orthogonal rotation of grouped expert INT8 weights.
    # "learned" is consumed only while exporting a new packed artifact.
    expert_quantization_rotation: str = "off"
    # Offline EAQuant router-scale calibration. Runtime consumption is armed
    # separately by memory.router_quantization_calibration_path.
    router_quantization_calibration: str = "off"
    # Offline projection-input covariance collection for shared-basis SVD.
    expert_factorization_calibration: str = "off"
    # Offline routed input-square collection and per-tensor precision planning.
    expert_precision_calibration: str = "off"
    # Optional path to a separate text-encoder directory for sparse-MoE families
    # that distribute text assets apart from the main checkpoint.
    text_encoder_path: str = ""
    # Component subfolder that owns the trainable denoiser checkpoint when a
    # model root packages multiple DiT components.
    denoiser_subfolder: str = "transformer"
    # Provider-owned extension namespace. Core preserves its typed TOML value
    # tree and delegates semantic validation to the active provider.
    family_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    type: str = "lingbot-video"
    path: str = "./models/lingbot_video"
    dtype: str = "bf16"
    attention_backend: str = "auto"
    hash_snapshot_contents: bool = False
    provider_module: str = ""
    params: ModelParams = field(default_factory=ModelParams)


@dataclass
class StrategyConfig:
    type: str = "text_to_video"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    task: str = "text_to_video"
    denoising_strength: float = 1.0
    prompt_rewriter: str = "none"
    cfg_mode: str = "sequential"
    keep_text_encoder_resident: bool = False
    stage_text_encoder_before_denoiser: bool = False
    text_encoder_weight_quantization: str = "none"
    keep_vae_resident: bool = False
    vae_tiling: bool = False
    vae_tile_size: int = 256
    vae_tile_stride: int = 192
    # Cross-timestep expert-branch feature reuse (default-off, lossy). See
    # mirai/core/moe/runtime/expert_feature_cache.py.
    expert_feature_cache: str = "off"
    expert_feature_cache_drift_threshold: float = 0.05
    expert_feature_cache_max_reuse_span: int = 2
    expert_feature_cache_slots: int = 2
    # Bound routed-expert activation workspace without changing routing.
    moe_token_chunk_size: int = 0
    # Inference-entrypoint frozen-weight block residency. 0 keeps the denoiser
    # fully resident; above 0 it requires memory.weight_residency_strategy.
    blocks_to_swap: int = 0
    block_swap_mode: str = "sync"
    # Optional refiner-only override. 0 inherits the denoiser residency request.
    refiner_blocks_to_swap: int = 0
    refiner_block_swap_mode: str = "sync"


@dataclass
class TrainingSection:
    seed: int = 42
    batch_size: int = 2
    gradient_accumulation: int = 1
    max_steps: int = 100
    loss_function: str = "mse"
    objective: str = "flow_matching"
    contrastive_flow_weight: float = 0.0
    latent_wavelet_loss_weight: float = 0.0
    loss_weighting: str = "uniform"
    min_snr_gamma: float = 5.0
    timestep_eps: float = 1e-5
    timestep_sampling: str = "uniform"
    timestep_sampling_mean: float = 0.0
    timestep_sampling_std: float = 1.0
    timestep_sampling_mode_scale: float = 1.29
    warmup_steps: int = 0
    non_finite_grad_policy: str = "abort"
    max_consecutive_skipped_steps: int = 5
    moe_expert_touch_guard: str = "off"
    moe_expert_touch_max_fraction: float = 1.0
    max_grad_norm: float = 1.0
    val_every_n_steps: int = 0
    early_stop_patience: int = 0
    ema_enabled: bool = False
    ema_decay: float = 0.999
    posthoc_ema_enabled: bool = False
    posthoc_ema_profile_stds: list[float] = field(
        default_factory=lambda: [0.05, 0.1]
    )
    posthoc_ema_snapshot_every_n_steps: int = 0
    prior_loss_weight: float = 1.0
    prior_ratio: float = 0.0
    log_grad_breakdown_every_n_steps: int = 0
    gradient_checkpointing: str = "standard"
    noise_offset: float = 0.0
    loss_bucket_normalization: str = "none"
    blocks_to_swap: int = 0
    block_swap_mode: str = "sync"
    optimizer_cpu_offload: bool = False
    gradient_cpu_offload: bool = False
    activation_cpu_offload: bool = False
    activation_cpu_offload_min_mib: int = 8
    activation_cpu_offload_max_gib: float = 0.0
    activation_cpu_offload_pin_memory: bool = False
    activation_cpu_offload_defer_layers: int = 0
    activation_cpu_offload_prefetch_layers: int = 0
    activation_cpu_offload_view_replay: bool = False
    masked_loss: bool = False
    compile: bool = False
    compile_scope: str = "regional"
    compile_mode: str = "default"
    compile_dynamic: bool | None = None
    compile_token_buckets: list[int] = field(default_factory=list)
    moe_token_chunk_size: int = 0
    block_swap_backward: bool = True
    pinned_memory_budget_fraction: float = 0.5
    curriculum: dict[str, Any] = field(default_factory=dict)
    policy_modules: list[str] = field(default_factory=list)
    policy_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    activation_compression: bool = False
    activation_compression_rank: int = 32
    activation_compression_min_mib: int = 8


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    weight_decay_filter: str = "lora_b_bias"
    loraplus_lr_ratio: float = 1.0
    prodigy_beta3: float = 0.0
    prodigy_decouple: bool = True
    prodigy_use_bias_correction: bool = False
    prodigy_safeguard_warmup: bool = False
    prodigy_d0: float = 1e-6
    prodigy_d_coef: float = 1.0
    prodigy_growth_rate: float = math.inf
    prodigy_slice_p: int = 1
    lora_pro_damping: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_eps: float = 1e-8
    muon_rms_target: float = 0.2
    lora_muon_gauge_rebalance_interval: int = 0
    lora_muon_gauge_rebalance_alpha: float = 1.0
    stochastic_rounding: bool = False
    allow_fallback: bool = True
    scheduler: str = "constant"
    scheduler_num_cycles: float = 1.0
    scheduler_power: float = 1.0
    min_lr_ratio: float = 0.0
    selected_expert_ids: list[int] = field(default_factory=list)

@dataclass
class AdapterConfig:
    type: str = "lora"
    target_preset: str = "attn_only"
    rank: int = 16
    alpha: float = 16.0
    rank_pattern: dict[str, int] = field(default_factory=dict)
    alpha_pattern: dict[str, float] = field(default_factory=dict)
    rank_budget: int = 0
    adaptive_rank_plan_path: str = ""
    use_rslora: bool = False
    use_lora_fa: bool = False
    use_dora: bool = False
    expert_tensor_lora_backend: str = "weight_space"
    init_from: str = ""
    rank_dropout: float = 0.0
    lora_parameter_dropout: float = 0.0
    # Offline post-optimization adapter transform. "off" is inert; "para"
    # arms the dedicated QR-subspace global rank-pruning CLI.
    posthoc_rank_compression: str = "off"
    rank_schedule_start: int = 0
    rank_schedule_end: int = 0
    rank_schedule_min_scale: float = 1.0
    lr_multipliers: dict[str, float] = field(default_factory=dict)
    expert_selection: str = "all"
    expert_topk_fraction: float = 0.25
    expert_calibration_steps: int = 8
    # ESFT (arXiv:2407.01906): select the minimum per-layer expert prefix
    # reaching this normalized task-affinity mass from this many samples.
    esft_selection_mass: float = 0.2
    esft_calibration_samples: int = 32
    # Condenser LoRA guardrail (ExpertCondenser, arXiv 2604.23036): a shared
    # always-on low-rank term (rank ~2-4) per routed-expert tensor key, NEVER
    # masked by expert_selection / expert-subset. Default 0 disables the feature.
    condenser_rank: int = 0
    # Condenser alpha; <= 0 means "use condenser_rank" (scale 1.0).
    condenser_alpha: float = 0.0
    # Router policy under PEFT (DenseMixer analysis: freeze the router for the
    # PCIe-bound regime). Default None = keep current preset-driven behavior
    # (router trainable only under a router-inclusive target preset). Set False
    # to force-freeze the router weight; True to force it trainable.
    train_router: bool | None = None
    # Timestep-axis adapter behaviors are opt-in.
    # T-LoRA (arXiv 2507.05964): timestep-dependent rank masking — at high noise
    # (post-shift sigma -> 1) only a low-rank prefix of each LoRA is active,
    # growing linearly to the full rank at sigma -> 0. "none" disables.
    timestep_rank_schedule: str = "none"
    # Fraction of the rank active at sigma == 1.0 under the tlora schedule.
    timestep_rank_min_fraction: float = 0.25
    # LoRA A-matrix initialization policy.
    lora_init: str = "kaiming"
    # Fixed-rank EVA (rho=1): maximum downstream-data forwards, per-target
    # activation rows retained per update, and principal-vector convergence.
    eva_calibration_steps: int = 32
    eva_samples_per_target: int = 256
    eva_convergence_threshold: float = 0.99
    lora_ga_calibration_steps: int = 4
    lora_ga_stable_gamma: float = 16.0
    # GoRA (arXiv:2502.12171): pre-optimizer full-weight gradient calibration,
    # sensitivity rank allocation, and pseudo-inverse factor initialization.
    gora_calibration_steps: int = 64
    gora_min_rank: int = 0
    gora_max_rank: int = 0
    gora_stable_gamma: float = 0.05
    # TC-LoRA (arXiv 2510.09561) gate hypernetwork hidden width. Only consulted
    # when timestep_rank_schedule == "tc_gate"; the gate is init-to-identity so
    # disabled schedules do not consult this value.
    tc_gate_hidden_dim: int = 8
    # Per-adapter timestep band [min, max) over the POST-shift sigma: the
    # adapter contributes zero output and zero grads for samples outside the
    # band (frozen-base forward and loss unaffected). Defaults = full range.
    timestep_band_min: float = 0.0
    timestep_band_max: float = 1.0
    # Opt-in compact adapter export for routing_topk selection: store only the
    # selected experts' LoRA slices + the mask instead of full-shape per-expert
    # tensors. False exports full-shape per-expert tensors.
    # NOTE: compact checkpoints are NOT kohya/ComfyUI-portable -- reload them in
    # Mirai (reconstruct-on-load) or expand them via scripts/export_adapter.py.
    sparse_expert_export: bool = False
    sparse_delta_density: float = 0.01
    sparse_delta_selection: str = "magnitude"


@dataclass
class MoEDatasetRoutingConfig:
    specialization_mode: str = "emergent"
    domain_metadata_key: str = ""
    expert_affinity: dict[str, list[int]] = field(default_factory=dict)
    routing_prior_weight: float = 0.0
    router_warmup_steps: int = 0


@dataclass
class DatasetConfig:
    path: str = "./data"
    cache_path: str = "./cache/dataset_cache.pt"
    auto_preprocess_cache: bool = False
    preprocess_raw_media_to_pt: bool = False
    max_cache_skip_ratio: float = 0.2
    split_seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    usage_mode: str = "internal"
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    cache_mode: str = "disk"
    cache_compression: str = "none"
    fp8_text_encoder: bool = False
    clips_per_video: int = 1
    tag_shuffle_variants: int = 1
    online_tag_shuffle: bool = False
    online_tag_shuffle_dropout: float = 0.0
    online_tag_shuffle_keep_first_n_tags: int = 1
    online_temporal_resampling: bool = False
    partial_recovery: bool = True
    mask_extension: str = ".mask.pt"
    caption_format: str = "raw"
    enable_bucketing: bool = False
    bucket_resolutions: list[str] = field(default_factory=list)
    bucket_base_resolution: int = 512
    frame_buckets: list[int] = field(default_factory=lambda: [33])
    bucket_resize_mode: str = "resize_crop"
    bucket_round_to: int = 16
    moe_routing: MoEDatasetRoutingConfig = field(
        default_factory=MoEDatasetRoutingConfig
    )


@dataclass
class LoggingConfig:
    output_dir: str = "./outputs"
    save_every_n_steps: int = 50
    async_checkpoint: bool = False
    async_checkpoint_max_gib: float = 0.0
    tensorboard: bool = False
    sample_every_n_steps: int = 0
    sample_prompt: str = "a cat walking"
    sample_seed: int = 42
    sample_prompts: list[str] = field(default_factory=list)
    sample_seeds: list[int] = field(default_factory=list)
    sample_blocks_to_swap: int = -1
    sample_cfg_scale: float = 7.5
    sample_negative_prompt: str = ""
    sample_solver: str = "euler"
    sample_resolution: str = "smallest_bucket"
    sample_frame_count: int = 16
    wandb: bool = False
    wandb_project: str = "mirai"
    wandb_run_name: str = ""


@dataclass
class ComplianceConfig:
    enabled: bool = False
    require_provenance: bool = True
    require_rights_attestation: bool = True


@dataclass
class MemoryConfig:
    hardware_policy: str = "disabled"
    frozen_weight_quantization: str = "none"
    frozen_weight_quantization_strategy: str = "disabled"
    # Empty inherits frozen_weight_quantization for a separate refiner DiT.
    refiner_frozen_weight_quantization: str = ""
    frozen_weight_packed_state_path: str = ""
    expert_precision_plan_path: str = ""
    expert_structured_sparsity: str = "disabled"
    quantization_block_size: int = 64
    weight_residency_strategy: str = "disabled"
    expert_weight_access: str = "auto"
    expert_dequant_chunk_size: int = 0
    expert_device_cache_gib: float = 0.0
    device_residency_budget_gib: float = 0.0
    quantize_experts_on_load: bool = False
    router_quantization: str = "disabled"
    router_quantization_calibration_path: str = ""
    moe_kernel_backend: str = "auto"
    cuda_memory_fraction: float = 1.0
    minimum_system_memory_gib: float = 0.0
    # Independent page-locked host-memory ceiling for block residency and packed
    # state preload. Over-budget tensors remain in pageable host memory; tensor
    # values and persistence formats are unchanged.
    max_pinned_host_gib: float = 12.0
    trainable_parameter_offload: bool = False
    # Disk-backed packed expert state is served from host RAM by default: "pinned"
    # (page-locked, enables async H2D) with an automatic degrade to "ram" (pageable)
    # only. RAM is the sole primary path -- it never auto-degrades to disk streaming.
    # "off" is an explicit opt-in to disk-backed lazy get_slice (streams every expert
    # off disk each forward/backward, causing sustained storage traffic) and must
    # be chosen by hand.
    packed_state_preload: str = "pinned"
    # Exact active-expert batch cache for explicit disk streaming. Zero keeps
    # the direct-range path allocation-identical; positive values are a strict
    # pageable/pinned host-RAM ceiling and require packed_state_preload="off".
    packed_stream_cache_gib: float = 0.0
    # Optional direct storage-to-CUDA backend. "staged" preserves the normal
    # bounded host path; "gds" is explicit and fails when true GDS is unavailable.
    packed_stream_backend: str = "staged"
    packed_stream_prefetch_depth: int = 0
    # MoE kernel and dispatch selection. Config values are the public control
    # surface; execution owners validate backend availability.
    moe_dispatch: str = "vectorized"  # vectorized|legacy|triton|triton_persistent
    moe_dispatch_preprocess: str = "host"  # host|device|sonic
    # Grouped-GEMM backend registry (below moe_dispatch, moe/runtime/gemm.py): "auto"
    # (default) inherits the moe_dispatch selection; per-role
    # keys are "" = inherit the main key. See CONFIG_REFERENCE.md for full semantics.
    moe_gemm_backend: str = "auto"  # auto|bmm|persistent|torch_grouped
    moe_gemm_backend_forward: str = ""  # inherited backend or deepgemm_fp8
    moe_gemm_backend_dx: str = ""  # ""|auto|bmm|persistent|torch_grouped
    moe_gemm_backend_dw: str = ""  # ""|auto|bmm|persistent|torch_grouped
    moe_batched_dequant: bool = True
    moe_pair_dequant: bool = True
    moe_batched_gather: bool = False
    moe_expert_autograd: str = "standard"
    moe_activation_backend: str = "torch"
    packed_shard_size_mb: int = 2048
    int8_workspace_mb: int = 0
    # Block-residency planning over the block-swap ring (all default = current
    # behavior; opt-in only). See mirai/core/training/residency_plan.py.
    block_residency_planner: str = "uniform"  # uniform | phase_aware
    block_swap_prefetch_depth: int = 1  # async prefetch ring depth (1-4)
    block_residency_priority: str = "index"  # index | routing_hot
    block_swap_transfer_strategy: str = "per_tensor"  # per_tensor | flat_ring


@dataclass
class TrainingConfig:
    preset: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingSection = field(default_factory=TrainingSection)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingConfig:
        root_allowed = {
            "preset",
            "model",
            "strategy",
            "inference",
            "training",
            "optimizer",
            "adapter",
            "dataset",
            "logging",
            "compliance",
            "memory",
        }
        _forbid_unknown("root", payload, root_allowed)

        model_table = _expect_table("model", payload.get("model", {}))
        if "variant" in model_table:
            raise ConfigError(
                "Unknown config key 'model.variant'. "
                "Use 'model.params.variant' instead."
            )
        model_allowed = {
            "type",
            "path",
            "dtype",
            "attention_backend",
            "hash_snapshot_contents",
            "provider_module",
            "params",
        }
        _forbid_unknown("model", model_table, model_allowed)
        inference_table = _expect_table("inference", payload.get("inference", {}))
        _forbid_unknown(
            "inference",
            inference_table,
            {
                "task",
                "denoising_strength",
                "prompt_rewriter",
                "cfg_mode",
                "keep_text_encoder_resident",
                "stage_text_encoder_before_denoiser",
                "text_encoder_weight_quantization",
                "keep_vae_resident",
                "vae_tiling",
                "vae_tile_size",
                "vae_tile_stride",
                "expert_feature_cache",
                "expert_feature_cache_drift_threshold",
                "expert_feature_cache_max_reuse_span",
                "expert_feature_cache_slots",
                "moe_token_chunk_size",
                "blocks_to_swap",
                "block_swap_mode",
                "refiner_blocks_to_swap",
                "refiner_block_swap_mode",
            },
        )
        inference_blocks_to_swap = int(inference_table.get("blocks_to_swap", 0))
        inference_refiner_blocks_to_swap = int(
            inference_table.get("refiner_blocks_to_swap", 0)
        )
        inference_text_encoder_quantization = str(
            inference_table.get("text_encoder_weight_quantization", "none")
        ).strip().lower()
        if inference_text_encoder_quantization not in {"none", "int8", "nf4"}:
            raise ConfigError(
                "inference.text_encoder_weight_quantization must be 'none', "
                f"'int8', or 'nf4'; got '{inference_text_encoder_quantization}'."
            )
        if (
            inference_text_encoder_quantization != "none"
            and not bool(
                inference_table.get("stage_text_encoder_before_denoiser", False)
            )
        ):
            raise ConfigError(
                "inference.text_encoder_weight_quantization requires "
                "inference.stage_text_encoder_before_denoiser=true."
            )
        if (
            bool(inference_table.get("stage_text_encoder_before_denoiser", False))
            and bool(inference_table.get("keep_text_encoder_resident", False))
        ):
            raise ConfigError(
                "inference.stage_text_encoder_before_denoiser=true is incompatible "
                "with inference.keep_text_encoder_resident=true."
            )
        if inference_blocks_to_swap < 0:
            raise ConfigError("inference.blocks_to_swap must be >= 0.")
        if inference_refiner_blocks_to_swap < 0:
            raise ConfigError("inference.refiner_blocks_to_swap must be >= 0.")
        inference_vae_tile_size = int(inference_table.get("vae_tile_size", 256))
        inference_vae_tile_stride = int(inference_table.get("vae_tile_stride", 192))
        if inference_vae_tile_size <= 0:
            raise ConfigError("inference.vae_tile_size must be > 0.")
        if not 0 < inference_vae_tile_stride <= inference_vae_tile_size:
            raise ConfigError(
                "inference.vae_tile_stride must be > 0 and no greater than "
                "inference.vae_tile_size."
            )
        inference_moe_token_chunk_size = int(
            inference_table.get("moe_token_chunk_size", 0)
        )
        if inference_moe_token_chunk_size < 0:
            raise ConfigError("inference.moe_token_chunk_size must be >= 0.")
        inference_block_swap_mode = str(
            inference_table.get("block_swap_mode", "sync")
        ).strip().lower()
        if inference_block_swap_mode not in {"sync", "async"}:
            raise ConfigError(
                "inference.block_swap_mode must be 'sync' or 'async'; "
                f"got '{inference_block_swap_mode}'."
            )
        inference_refiner_block_swap_mode = str(
            inference_table.get("refiner_block_swap_mode", "sync")
        ).strip().lower()
        if inference_refiner_block_swap_mode not in {"sync", "async"}:
            raise ConfigError(
                "inference.refiner_block_swap_mode must be 'sync' or 'async'; "
                f"got '{inference_refiner_block_swap_mode}'."
            )
        if inference_blocks_to_swap > 0 or inference_refiner_blocks_to_swap > 0:
            configured_residency_strategy = (
                str(
                    _expect_table("memory", payload.get("memory", {})).get(
                        "weight_residency_strategy", "disabled"
                    )
                )
                .strip()
                .lower()
            )
            if configured_residency_strategy not in {
                "",
                "auto",
                "block_swap",
                "stream_disk",
            }:
                raise ConfigError(
                    "inference block swapping requires "
                    "memory.weight_residency_strategy='block_swap' or "
                    f"'stream_disk'; got '{configured_residency_strategy}'."
                )
        from mirai.core.inference.conditioning import normalize_inference_task

        try:
            inference_task = normalize_inference_task(
                str(inference_table.get("task", "text_to_video"))
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        denoising_strength = float(
            inference_table.get("denoising_strength", 1.0)
        )
        if not 0.0 <= denoising_strength <= 1.0:
            raise ConfigError(
                "inference.denoising_strength must be between 0 and 1."
            )
        if inference_task != "video_to_video" and denoising_strength != 1.0:
            raise ConfigError(
                "inference.denoising_strength may differ from 1 only when "
                "inference.task='video_to_video'."
            )
        prompt_rewriter = str(
            inference_table.get("prompt_rewriter", "none")
        ).strip().lower() or "none"
        cfg_mode = str(inference_table.get("cfg_mode", "sequential")).strip().lower()
        if cfg_mode not in {"sequential", "batched"}:
            raise ConfigError(
                "inference.cfg_mode must be 'sequential' or 'batched'; "
                f"got '{cfg_mode}'."
            )
        from mirai.core.moe.runtime.expert_feature_cache import (
            ExpertFeatureCacheError,
            ExpertFeatureCachePolicy,
        )

        try:
            expert_feature_cache_policy = ExpertFeatureCachePolicy(
                mode=inference_table.get("expert_feature_cache", "off"),
                drift_threshold=float(
                    inference_table.get(
                        "expert_feature_cache_drift_threshold", 0.05
                    )
                ),
                max_reuse_span=int(
                    inference_table.get("expert_feature_cache_max_reuse_span", 2)
                ),
                slots=int(inference_table.get("expert_feature_cache_slots", 2)),
            )
        except ExpertFeatureCacheError as exc:
            raise ConfigError(str(exc)) from exc
        if expert_feature_cache_policy.enabled:
            # Branch-level reuse needs the grouped expert-execution seam: the
            # per-expert reference loop and the fused kernel both produce only a
            # combined layer output, which has no branch to cache.
            configured_kernel_backend = (
                str(
                    _expect_table("memory", payload.get("memory", {})).get(
                        "moe_kernel_backend", "auto"
                    )
                )
                .strip()
                .lower()
            )
            if configured_kernel_backend != "grouped":
                raise ConfigError(
                    "inference.expert_feature_cache requires "
                    "memory.moe_kernel_backend='grouped'; got "
                    f"'{configured_kernel_backend}'."
                )
            if inference_moe_token_chunk_size > 0:
                raise ConfigError(
                    "inference.moe_token_chunk_size is incompatible with "
                    "inference.expert_feature_cache because branch caching owns "
                    "the complete routed-slot layout."
                )
        model_params_table = _expect_table("model.params", model_table.get("params", {}))
        model_params_allowed = {
            "variant",
            "flow_shift",
            "flow_shift_mode",
            "flow_shift_base_seq_len",
            "flow_shift_max_seq_len",
            "flow_shift_max",
            "vae_chunk_size",
            "lora_dropout",
            "strict_native_assets",
            "moe_expert_backend",
            "moe_pad_backend",
            "moe_reorder_backend",
            "moe_restore_backend",
            "moe_restore_chunk_size",
            "moe_fused_qkv_linear",
            "inference_bf16_fastmath",
            "inference_routing_telemetry",
            "inference_routing_telemetry_layer_stride",
            "num_experts",
            "experts_per_token",
            "shared_experts",
            "hidden_size",
            "num_layers",
            "attention_heads",
            "latent_channels",
            "patch_size",
            "expert_capacity_factor",
            "moe_routing_mode",
            "moe_chain_of_experts",
            "moe_chain_router_rank",
            "moe_layer_router_policy",
            "moe_progressive_sparsification_transition_step",
            "moe_progressive_sparsification_policy",
            "moe_expert_choice_capacity_factor",
            "moe_expert_choice_capacity_schedule",
            "moe_expert_choice_timestep_capacity_schedule",
            "moe_expert_choice_timestep_capacity_span",
            "moe_expert_choice_coverage_alarm_threshold",
            "moe_zero_experts",
            "moe_copy_experts",
            "moe_constant_experts",
            "moe_lightweight_top_k",
            "moe_adjugate_experts",
            "moe_adjugate_expert_groups",
            "moe_adjugate_expert_intermediate_size",
            "moe_adjugate_expert_scale",
            "moe_router_timestep_weight",
            "moe_aux_loss_type",
            "moe_aux_loss_weight",
            "moe_router_z_loss_weight",
            "moe_router_similarity_loss_weight",
            "moe_dynamic_topk_min",
            "moe_dynamic_topk_average",
            "moe_spatiotemporal_routing_weight",
            "moe_spatiotemporal_routing_max_edges",
            "moe_bias_update_rate",
            "moe_bias_centering",
            "moe_balance_mode",
            "moe_balance_loss_disable_step",
            "moe_balance_scope",
            "moe_balance_grad_ratio_telemetry",
            "moe_balance_grad_ratio_threshold",
            "moe_phi_balance_weight",
            "moe_phi_balance_ema_rate",
            "moe_phi_balance_potential",
            "moe_router_variance_loss_weight",
            "moe_expert_orthogonality_loss_weight",
            "moe_swiglu_specialization_loss_weight",
            "moe_cross_layer_coupling_loss_weight",
            "moe_specialization_max_tokens",
            "router_fp32_master",
            "expert_subset_fraction",
            "expert_subset_pool_factor",
            "expert_subset_anneal_steps",
            "expert_subset_router_kl_weight",
            "moe_routing_health",
            "moe_selection_margin",
            "moe_routing_agreement_evidence",
            "expert_pruning",
            "expert_pruning_criterion",
            "expert_consolidation",
            "expert_upcycling",
            "expert_upcycling_copies",
            "expert_upcycling_reinit_ratio",
            "expert_upcycling_seed",
            "expert_weight_compression",
            "flexmoe_calibration",
            "post_compression_router_repair",
            "router_repair_artifact_path",
            "expert_quantization_calibration",
            "expert_quantization_rotation",
            "router_quantization_calibration",
            "expert_factorization_calibration",
            "expert_precision_calibration",
            "text_encoder_path",
            "denoiser_subfolder",
            "family_params",
        }
        if "moe_boundary" in model_params_table:
            raise ConfigError(
                "Config key 'model.params.moe_boundary' was removed; it was never "
                "consumed by the trainer. Remove it from your config."
            )
        _forbid_unknown("model.params", model_params_table, model_params_allowed)
        family_params = model_params_table.get("family_params", {})
        if not isinstance(family_params, dict):
            raise ConfigError("Expected table at 'model.params.family_params'.")

        strategy_table = _expect_table("strategy", payload.get("strategy", {}))
        strategy_allowed = {"type", "params"}
        _forbid_unknown("strategy", strategy_table, strategy_allowed)
        strategy_params = strategy_table.get("params", {})
        if not isinstance(strategy_params, dict):
            raise ConfigError(
                f"Expected table at 'strategy.params', got {type(strategy_params).__name__}."
            )

        training_table = _expect_table("training", payload.get("training", {}))
        training_allowed = {
            "seed",
            "batch_size",
            "gradient_accumulation",
            "max_steps",
            "loss_function",
            "objective",
            "contrastive_flow_weight",
            "latent_wavelet_loss_weight",
            "loss_weighting",
            "min_snr_gamma",
            "timestep_eps",
            "timestep_sampling",
            "timestep_sampling_mean",
            "timestep_sampling_std",
            "timestep_sampling_mode_scale",
            "warmup_steps",
            "non_finite_grad_policy",
            "max_consecutive_skipped_steps",
            "moe_expert_touch_guard",
            "moe_expert_touch_max_fraction",
            "max_grad_norm",
            "val_every_n_steps",
            "early_stop_patience",
            "ema_enabled",
            "ema_decay",
            "posthoc_ema_enabled",
            "posthoc_ema_profile_stds",
            "posthoc_ema_snapshot_every_n_steps",
            "prior_loss_weight",
            "prior_ratio",
            "log_grad_breakdown_every_n_steps",
            "gradient_checkpointing",
            "noise_offset",
            "loss_bucket_normalization",
            "blocks_to_swap",
            "block_swap_mode",
            "optimizer_cpu_offload",
            "gradient_cpu_offload",
            "activation_cpu_offload",
            "activation_cpu_offload_min_mib",
            "activation_cpu_offload_max_gib",
            "activation_cpu_offload_pin_memory",
            "activation_cpu_offload_defer_layers",
            "activation_cpu_offload_prefetch_layers",
            "activation_cpu_offload_view_replay",
            "activation_compression",
            "activation_compression_rank",
            "activation_compression_min_mib",
            "masked_loss",
            "compile",
            "compile_scope",
            "compile_mode",
            "compile_dynamic",
            "compile_token_buckets",
            "moe_token_chunk_size",
            "block_swap_backward",
            "pinned_memory_budget_fraction",
            "curriculum",
            "policy_modules",
            "policy_options",
        }
        _forbid_unknown("training", training_table, training_allowed)
        curriculum_raw = training_table.get("curriculum", {})
        if not isinstance(curriculum_raw, dict):
            raise ConfigError("Expected table at 'training.curriculum'.")
        policy_modules_raw = training_table.get("policy_modules", [])
        if not isinstance(policy_modules_raw, list) or not all(
            isinstance(value, str) and value.strip() for value in policy_modules_raw
        ):
            raise ConfigError(
                "training.policy_modules must be a list of non-empty module names."
            )
        policy_options_raw = training_table.get("policy_options", {})
        if not isinstance(policy_options_raw, dict) or not all(
            isinstance(name, str)
            and name.strip()
            and isinstance(options, dict)
            for name, options in policy_options_raw.items()
        ):
            raise ConfigError(
                "training.policy_options must be a table of policy-name tables."
            )
        compile_dynamic_raw = training_table.get("compile_dynamic")
        if compile_dynamic_raw is not None and not isinstance(
            compile_dynamic_raw, bool
        ):
            raise ConfigError("training.compile_dynamic must be a boolean.")
        compile_token_buckets_raw = training_table.get(
            "compile_token_buckets", []
        )
        if not isinstance(compile_token_buckets_raw, list) or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in compile_token_buckets_raw
        ):
            raise ConfigError(
                "training.compile_token_buckets must be a list of integers."
            )

        optimizer_table = _expect_table("optimizer", payload.get("optimizer", {}))
        optimizer_allowed = {
            "type",
            "lr",
            "weight_decay",
            "weight_decay_filter",
            "loraplus_lr_ratio",
            "prodigy_beta3",
            "prodigy_decouple",
            "prodigy_use_bias_correction",
            "prodigy_safeguard_warmup",
            "prodigy_d0",
            "prodigy_d_coef",
            "prodigy_growth_rate",
            "prodigy_slice_p",
            "lora_pro_damping",
            "muon_momentum",
            "muon_nesterov",
            "muon_ns_steps",
            "muon_eps",
            "muon_rms_target",
            "lora_muon_gauge_rebalance_interval",
            "lora_muon_gauge_rebalance_alpha",
            "stochastic_rounding",
            "allow_fallback",
            "scheduler",
            "scheduler_num_cycles",
            "scheduler_power",
            "min_lr_ratio",
            "selected_expert_ids",
        }
        _forbid_unknown("optimizer", optimizer_table, optimizer_allowed)
        adapter_table = _expect_table("adapter", payload.get("adapter", {}))
        adapter_allowed = {
            "target_preset",
            "type",
            "rank",
            "alpha",
            "rank_pattern",
            "alpha_pattern",
            "rank_budget",
            "adaptive_rank_plan_path",
            "use_rslora",
            "use_lora_fa",
            "use_dora",
            "expert_tensor_lora_backend",
            "init_from",
            "rank_dropout",
            "lora_parameter_dropout",
            "posthoc_rank_compression",
            "rank_schedule_start",
            "rank_schedule_end",
            "rank_schedule_min_scale",
            "lr_multipliers",
            "expert_selection",
            "expert_topk_fraction",
            "expert_calibration_steps",
            "esft_selection_mass",
            "esft_calibration_samples",
            "sparse_expert_export",
            "sparse_delta_density",
            "sparse_delta_selection",
            "timestep_rank_schedule",
            "timestep_rank_min_fraction",
            "lora_init",
            "eva_calibration_steps",
            "eva_samples_per_target",
            "eva_convergence_threshold",
            "lora_ga_calibration_steps",
            "lora_ga_stable_gamma",
            "gora_calibration_steps",
            "gora_min_rank",
            "gora_max_rank",
            "gora_stable_gamma",
            "tc_gate_hidden_dim",
            "timestep_band_min",
            "timestep_band_max",
            "condenser_rank",
            "condenser_alpha",
            "train_router",
        }
        _forbid_unknown("adapter", adapter_table, adapter_allowed)
        lr_multipliers_raw = adapter_table.get("lr_multipliers", {})
        if not isinstance(lr_multipliers_raw, dict):
            raise ConfigError("Expected table at 'adapter.lr_multipliers'.")
        rank_pattern_raw = adapter_table.get("rank_pattern", {})
        if not isinstance(rank_pattern_raw, dict):
            raise ConfigError("Expected table at 'adapter.rank_pattern'.")
        alpha_pattern_raw = adapter_table.get("alpha_pattern", {})
        if not isinstance(alpha_pattern_raw, dict):
            raise ConfigError("Expected table at 'adapter.alpha_pattern'.")

        dataset_table = _expect_table("dataset", payload.get("dataset", {}))
        dataset_allowed = {
            "path",
            "cache_path",
            "auto_preprocess_cache",
            "preprocess_raw_media_to_pt",
            "max_cache_skip_ratio",
            "split_seed",
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "usage_mode",
            "num_workers",
            "prefetch_factor",
            "persistent_workers",
            "cache_mode",
            "cache_compression",
            "fp8_text_encoder",
            "clips_per_video",
            "tag_shuffle_variants",
            "online_tag_shuffle",
            "online_tag_shuffle_dropout",
            "online_tag_shuffle_keep_first_n_tags",
            "online_temporal_resampling",
            "partial_recovery",
            "mask_extension",
            "caption_format",
            "enable_bucketing",
            "bucket_resolutions",
            "bucket_base_resolution",
            "frame_buckets",
            "bucket_resize_mode",
            "bucket_round_to",
            "moe_routing",
        }
        _forbid_unknown("dataset", dataset_table, dataset_allowed)
        caption_format = str(dataset_table.get("caption_format", "raw")).strip().lower()
        if not caption_format:
            raise ConfigError(
                "dataset.caption_format must be a non-empty provider-owned format name."
            )
        moe_routing_table = _expect_table(
            "dataset.moe_routing", dataset_table.get("moe_routing", {})
        )
        moe_routing_allowed = {
            "specialization_mode",
            "domain_metadata_key",
            "expert_affinity",
            "routing_prior_weight",
            "router_warmup_steps",
        }
        if "allow_unassigned_domains" in moe_routing_table:
            raise ConfigError(
                "Config key 'dataset.moe_routing.allow_unassigned_domains' was "
                "removed; it was never consumed. Remove it from your config."
            )
        _forbid_unknown(
            "dataset.moe_routing", moe_routing_table, moe_routing_allowed
        )
        expert_affinity_raw = moe_routing_table.get("expert_affinity", {})
        if not isinstance(expert_affinity_raw, dict):
            raise ConfigError(
                "Expected table at 'dataset.moe_routing.expert_affinity'."
            )
        expert_affinity: dict[str, list[int]] = {}
        for domain, expert_ids in expert_affinity_raw.items():
            if not isinstance(expert_ids, list):
                raise ConfigError(
                    "dataset.moe_routing.expert_affinity values must be arrays "
                    f"of expert ids; domain '{domain}' is not an array."
                )
            expert_affinity[str(domain)] = [int(value) for value in expert_ids]

        logging_table = _expect_table("logging", payload.get("logging", {}))
        logging_allowed = {
            "output_dir",
            "save_every_n_steps",
            "async_checkpoint",
            "async_checkpoint_max_gib",
            "tensorboard",
            "sample_every_n_steps",
            "sample_prompt",
            "sample_seed",
            "sample_prompts",
            "sample_seeds",
            "sample_blocks_to_swap",
            "sample_cfg_scale",
            "sample_negative_prompt",
            "sample_solver",
            "sample_resolution",
            "sample_frame_count",
            "wandb",
            "wandb_project",
            "wandb_run_name",
        }
        _forbid_unknown("logging", logging_table, logging_allowed)
        compliance_table = _expect_table("compliance", payload.get("compliance", {}))
        compliance_allowed = {
            "enabled",
            "require_provenance",
            "require_rights_attestation",
        }
        _forbid_unknown("compliance", compliance_table, compliance_allowed)
        memory_table = _expect_table("memory", payload.get("memory", {}))
        memory_allowed = {
            "hardware_policy",
            "frozen_weight_quantization",
            "frozen_weight_quantization_strategy",
            "refiner_frozen_weight_quantization",
            "frozen_weight_packed_state_path",
            "expert_precision_plan_path",
            "expert_structured_sparsity",
            "quantization_block_size",
            "weight_residency_strategy",
            "expert_weight_access",
            "expert_dequant_chunk_size",
            "expert_device_cache_gib",
            "device_residency_budget_gib",
            "quantize_experts_on_load",
            "router_quantization",
            "router_quantization_calibration_path",
            "moe_kernel_backend",
            "cuda_memory_fraction",
            "minimum_system_memory_gib",
            "max_pinned_host_gib",
            "trainable_parameter_offload",
            "packed_state_preload",
            "packed_stream_cache_gib",
            "packed_stream_backend",
            "packed_stream_prefetch_depth",
            "moe_dispatch",
            "moe_dispatch_preprocess",
            "moe_gemm_backend",
            "moe_gemm_backend_forward",
            "moe_gemm_backend_dx",
            "moe_gemm_backend_dw",
            "moe_batched_dequant",
            "moe_pair_dequant",
            "moe_batched_gather",
            "moe_expert_autograd",
            "moe_activation_backend",
            "packed_shard_size_mb",
            "int8_workspace_mb",
            "block_residency_planner",
            "block_swap_prefetch_depth",
            "block_residency_priority",
            "block_swap_transfer_strategy",
        }
        _forbid_unknown("memory", memory_table, memory_allowed)

        product_model_defaults = "type" not in model_table
        model_type = str(model_table.get("type", "lingbot-video")).strip().lower()
        model_path = _validate_no_control_path(
            "model.path",
            str(
                model_table.get(
                    "path",
                    "./models/lingbot_video" if product_model_defaults else "",
                )
            ),
        )
        provider_module = _validate_no_control_path(
            "model.provider_module", str(model_table.get("provider_module", ""))
        )
        dataset_path = _validate_no_control_path(
            "dataset.path", str(dataset_table.get("path", "./data"))
        )
        cache_path = _validate_no_control_path(
            "dataset.cache_path", str(dataset_table.get("cache_path", "./cache/dataset_cache.pt"))
        )
        output_dir = _validate_no_control_path(
            "logging.output_dir", str(logging_table.get("output_dir", "./outputs"))
        )
        frozen_weight_packed_state_path = _validate_no_control_path(
            "memory.frozen_weight_packed_state_path",
            str(memory_table.get("frozen_weight_packed_state_path", "")),
        )
        expert_precision_plan_path = _validate_no_control_path(
            "memory.expert_precision_plan_path",
            str(memory_table.get("expert_precision_plan_path", "")),
        )
        model = ModelConfig(
            type=model_type,
            path=model_path,
            dtype=_validate_model_dtype(model_table.get("dtype", "bf16")),
            attention_backend=str(model_table.get("attention_backend", "auto")),
            hash_snapshot_contents=bool(
                model_table.get("hash_snapshot_contents", False)
            ),
            provider_module=provider_module,
            params=ModelParams(
                variant=str(
                    model_params_table.get(
                        "variant",
                        "lingbot-video-moe-30b-a3b" if product_model_defaults else "",
                    )
                ),
                flow_shift=float(model_params_table.get("flow_shift", 3.0)),
                flow_shift_mode=str(
                    model_params_table.get("flow_shift_mode", "constant")
                ),
                flow_shift_base_seq_len=int(
                    model_params_table.get("flow_shift_base_seq_len", 256)
                ),
                flow_shift_max_seq_len=int(
                    model_params_table.get("flow_shift_max_seq_len", 4096)
                ),
                flow_shift_max=float(
                    model_params_table.get("flow_shift_max", 12.0)
                ),
                vae_chunk_size=int(model_params_table.get("vae_chunk_size", 16)),
                lora_dropout=float(model_params_table.get("lora_dropout", 0.0)),
                strict_native_assets=bool(
                    model_params_table.get("strict_native_assets", product_model_defaults)
                ),
                moe_expert_backend=str(
                    model_params_table.get("moe_expert_backend", "grouped_mm")
                ),
                moe_pad_backend=str(
                    model_params_table.get("moe_pad_backend", "auto")
                ),
                moe_reorder_backend=str(
                    model_params_table.get("moe_reorder_backend", "sort")
                ),
                moe_restore_backend=str(
                    model_params_table.get("moe_restore_backend", "chunked_scatter")
                ),
                moe_restore_chunk_size=int(
                    model_params_table.get("moe_restore_chunk_size", 128)
                ),
                moe_fused_qkv_linear=bool(
                    model_params_table.get("moe_fused_qkv_linear", False)
                ),
                inference_bf16_fastmath=bool(
                    model_params_table.get("inference_bf16_fastmath", False)
                ),
                inference_routing_telemetry=bool(
                    model_params_table.get("inference_routing_telemetry", False)
                ),
                inference_routing_telemetry_layer_stride=int(
                    model_params_table.get(
                        "inference_routing_telemetry_layer_stride",
                        1,
                    )
                ),
                num_experts=int(model_params_table.get("num_experts", 4)),
                experts_per_token=int(model_params_table.get("experts_per_token", 2)),
                shared_experts=int(model_params_table.get("shared_experts", 1)),
                hidden_size=int(model_params_table.get("hidden_size", 32)),
                num_layers=int(model_params_table.get("num_layers", 2)),
                attention_heads=int(model_params_table.get("attention_heads", 4)),
                latent_channels=int(model_params_table.get("latent_channels", 1)),
                patch_size=int(model_params_table.get("patch_size", 1)),
                expert_capacity_factor=float(
                    model_params_table.get("expert_capacity_factor", 1.0)
                ),
                moe_routing_mode=str(
                    model_params_table.get("moe_routing_mode", "token_choice")
                ),
                moe_chain_of_experts=bool(
                    model_params_table.get("moe_chain_of_experts", False)
                ),
                moe_chain_router_rank=int(
                    model_params_table.get("moe_chain_router_rank", 0)
                ),
                moe_layer_router_policy=_parse_layer_router_policy(
                    model_params_table.get("moe_layer_router_policy", [])
                ),
                moe_progressive_sparsification_transition_step=int(
                    model_params_table.get(
                        "moe_progressive_sparsification_transition_step",
                        0,
                    )
                ),
                moe_progressive_sparsification_policy=(
                    _parse_progressive_sparsification_policy(
                        model_params_table.get(
                            "moe_progressive_sparsification_policy",
                            [],
                        )
                    )
                ),
                moe_expert_choice_capacity_factor=float(
                    model_params_table.get(
                        "moe_expert_choice_capacity_factor", 1.0
                    )
                ),
                moe_expert_choice_capacity_schedule=(
                    _parse_expert_choice_capacity_schedule(
                        model_params_table.get(
                            "moe_expert_choice_capacity_schedule", []
                        )
                    )
                ),
                moe_expert_choice_timestep_capacity_schedule=str(
                    model_params_table.get(
                        "moe_expert_choice_timestep_capacity_schedule",
                        "disabled",
                    )
                ),
                moe_expert_choice_timestep_capacity_span=float(
                    model_params_table.get(
                        "moe_expert_choice_timestep_capacity_span",
                        0.0,
                    )
                ),
                moe_expert_choice_coverage_alarm_threshold=float(
                    model_params_table.get(
                        "moe_expert_choice_coverage_alarm_threshold",
                        0.0,
                    )
                ),
                moe_zero_experts=int(
                    model_params_table.get("moe_zero_experts", 0)
                ),
                moe_copy_experts=int(
                    model_params_table.get("moe_copy_experts", 0)
                ),
                moe_constant_experts=int(
                    model_params_table.get("moe_constant_experts", 0)
                ),
                moe_lightweight_top_k=int(
                    model_params_table.get("moe_lightweight_top_k", 0)
                ),
                moe_adjugate_experts=bool(
                    model_params_table.get("moe_adjugate_experts", False)
                ),
                moe_adjugate_expert_groups=int(
                    model_params_table.get("moe_adjugate_expert_groups", 0)
                ),
                moe_adjugate_expert_intermediate_size=int(
                    model_params_table.get(
                        "moe_adjugate_expert_intermediate_size",
                        128,
                    )
                ),
                moe_adjugate_expert_scale=float(
                    model_params_table.get("moe_adjugate_expert_scale", 0.05)
                ),
                moe_router_timestep_weight=float(
                    model_params_table.get("moe_router_timestep_weight", 0.0)
                ),
                moe_aux_loss_type=str(
                    model_params_table.get("moe_aux_loss_type", "model_native")
                ),
                moe_aux_loss_weight=float(
                    model_params_table.get("moe_aux_loss_weight", 0.01)
                ),
                moe_router_z_loss_weight=float(
                    model_params_table.get("moe_router_z_loss_weight", 0.0)
                ),
                moe_router_similarity_loss_weight=float(
                    model_params_table.get("moe_router_similarity_loss_weight", 0.0)
                ),
                moe_dynamic_topk_min=int(
                    model_params_table.get("moe_dynamic_topk_min", 0)
                ),
                moe_dynamic_topk_average=float(
                    model_params_table.get("moe_dynamic_topk_average", 0.0)
                ),
                moe_spatiotemporal_routing_weight=float(
                    model_params_table.get(
                        "moe_spatiotemporal_routing_weight", 0.0
                    )
                ),
                moe_spatiotemporal_routing_max_edges=int(
                    model_params_table.get(
                        "moe_spatiotemporal_routing_max_edges", 4096
                    )
                ),
                moe_bias_update_rate=float(
                    model_params_table.get("moe_bias_update_rate", 0.0)
                ),
                moe_bias_centering=bool(
                    model_params_table.get("moe_bias_centering", True)
                ),
                moe_balance_mode=str(
                    model_params_table.get("moe_balance_mode", "aux_loss")
                ),
                moe_balance_loss_disable_step=int(
                    model_params_table.get("moe_balance_loss_disable_step", 0)
                ),
                moe_balance_scope=str(
                    model_params_table.get("moe_balance_scope", "microbatch")
                ),
                moe_balance_grad_ratio_telemetry=bool(
                    model_params_table.get(
                        "moe_balance_grad_ratio_telemetry", False
                    )
                ),
                moe_balance_grad_ratio_threshold=float(
                    model_params_table.get(
                        "moe_balance_grad_ratio_threshold", 0.1
                    )
                ),
                moe_phi_balance_weight=float(
                    model_params_table.get("moe_phi_balance_weight", 0.0)
                ),
                moe_phi_balance_ema_rate=float(
                    model_params_table.get("moe_phi_balance_ema_rate", 0.01)
                ),
                moe_phi_balance_potential=str(
                    model_params_table.get(
                        "moe_phi_balance_potential", "negative_entropy"
                    )
                ),
                moe_router_variance_loss_weight=float(
                    model_params_table.get("moe_router_variance_loss_weight", 0.0)
                ),
                moe_expert_orthogonality_loss_weight=float(
                    model_params_table.get(
                        "moe_expert_orthogonality_loss_weight", 0.0
                    )
                ),
                moe_swiglu_specialization_loss_weight=float(
                    model_params_table.get(
                        "moe_swiglu_specialization_loss_weight", 0.0
                    )
                ),
                moe_cross_layer_coupling_loss_weight=float(
                    model_params_table.get(
                        "moe_cross_layer_coupling_loss_weight", 0.0
                    )
                ),
                moe_specialization_max_tokens=int(
                    model_params_table.get("moe_specialization_max_tokens", 256)
                ),
                router_fp32_master=bool(
                    model_params_table.get("router_fp32_master", False)
                ),
                expert_subset_fraction=float(
                    model_params_table.get("expert_subset_fraction", 1.0)
                ),
                expert_subset_pool_factor=float(
                    model_params_table.get("expert_subset_pool_factor", 2.0)
                ),
                expert_subset_anneal_steps=int(
                    model_params_table.get("expert_subset_anneal_steps", 0)
                ),
                expert_subset_router_kl_weight=float(
                    model_params_table.get("expert_subset_router_kl_weight", 0.0)
                ),
                moe_routing_health=bool(
                    model_params_table.get("moe_routing_health", False)
                ),
                moe_selection_margin=bool(
                    model_params_table.get("moe_selection_margin", False)
                ),
                moe_routing_agreement_evidence=str(
                    model_params_table.get(
                        "moe_routing_agreement_evidence",
                        "off",
                    )
                ),
                expert_pruning=str(
                    model_params_table.get("expert_pruning", "off")
                ),
                expert_pruning_criterion=str(
                    model_params_table.get(
                        "expert_pruning_criterion",
                        "frequency",
                    )
                ),
                expert_consolidation=str(
                    model_params_table.get("expert_consolidation", "off")
                ),
                expert_upcycling=str(
                    model_params_table.get("expert_upcycling", "off")
                ),
                expert_upcycling_copies=int(
                    model_params_table.get("expert_upcycling_copies", 0)
                ),
                expert_upcycling_reinit_ratio=float(
                    model_params_table.get(
                        "expert_upcycling_reinit_ratio", 0.5
                    )
                ),
                expert_upcycling_seed=int(
                    model_params_table.get("expert_upcycling_seed", 42)
                ),
                expert_weight_compression=str(
                    model_params_table.get("expert_weight_compression", "off")
                ),
                flexmoe_calibration=str(
                    model_params_table.get("flexmoe_calibration", "off")
                ),
                post_compression_router_repair=str(
                    model_params_table.get(
                        "post_compression_router_repair",
                        "off",
                    )
                ),
                router_repair_artifact_path=_validate_no_control_path(
                    "model.params.router_repair_artifact_path",
                    str(model_params_table.get("router_repair_artifact_path", "")),
                ),
                expert_quantization_calibration=str(
                    model_params_table.get("expert_quantization_calibration", "off")
                ),
                expert_quantization_rotation=str(
                    model_params_table.get("expert_quantization_rotation", "off")
                ),
                router_quantization_calibration=str(
                    model_params_table.get(
                        "router_quantization_calibration",
                        "off",
                    )
                ),
                expert_factorization_calibration=str(
                    model_params_table.get(
                        "expert_factorization_calibration",
                        "off",
                    )
                ),
                expert_precision_calibration=str(
                    model_params_table.get(
                        "expert_precision_calibration",
                        "off",
                    )
                ),
                text_encoder_path=str(model_params_table.get("text_encoder_path", "")),
                denoiser_subfolder=str(model_params_table.get("denoiser_subfolder", "transformer")),
                family_params=dict(family_params),
            ),
        )
        _mp = model.params
        _runtime_choices = {
            "moe_expert_backend": {
                "grouped_mm",
                "loop",
                "sglang_triton",
                "sglang_triton_fp8",
            },
            "moe_pad_backend": {"auto", "loop", "vectorized"},
            "moe_reorder_backend": {"sort", "triton_pack"},
            "moe_restore_backend": {
                "scatter",
                "chunked_scatter",
                "weighted_scatter",
                "index_add",
                "triton",
            },
        }
        for _name, _choices in _runtime_choices.items():
            _value = str(getattr(_mp, _name)).strip().lower()
            if _value not in _choices:
                raise ConfigError(
                    f"model.params.{_name} must be one of: "
                    + ", ".join(sorted(_choices))
                    + "."
                )
            setattr(_mp, _name, _value)
        if int(_mp.moe_restore_chunk_size) <= 0:
            raise ConfigError("model.params.moe_restore_chunk_size must be > 0.")
        if int(_mp.inference_routing_telemetry_layer_stride) <= 0:
            raise ConfigError(
                "model.params.inference_routing_telemetry_layer_stride must be > 0."
            )
        if not math.isfinite(float(_mp.flow_shift)) or float(_mp.flow_shift) <= 0.0:
            raise ConfigError("model.params.flow_shift must be finite and > 0.")
        _flow_shift_mode = str(_mp.flow_shift_mode).strip().lower()
        if _flow_shift_mode not in {"constant", "dynamic"}:
            raise ConfigError(
                "model.params.flow_shift_mode must be 'constant' or 'dynamic'."
            )
        if int(_mp.flow_shift_base_seq_len) <= 0:
            raise ConfigError(
                "model.params.flow_shift_base_seq_len must be > 0."
            )
        if int(_mp.flow_shift_max_seq_len) < int(_mp.flow_shift_base_seq_len):
            raise ConfigError(
                "model.params.flow_shift_max_seq_len must be >= "
                "flow_shift_base_seq_len."
            )
        if (
            not math.isfinite(float(_mp.flow_shift_max))
            or float(_mp.flow_shift_max) <= 0.0
        ):
            raise ConfigError(
                "model.params.flow_shift_max must be finite and > 0."
            )
        if _flow_shift_mode == "dynamic":
            _expected_flow_shift_max = float(_mp.flow_shift) * math.sqrt(
                float(_mp.flow_shift_max_seq_len)
                / float(_mp.flow_shift_base_seq_len)
            )
            if not math.isclose(
                float(_mp.flow_shift_max),
                _expected_flow_shift_max,
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise ConfigError(
                    "model.params.flow_shift_max must equal flow_shift * "
                    "sqrt(flow_shift_max_seq_len / flow_shift_base_seq_len) "
                    "when flow_shift_mode='dynamic'."
                )
        if str(_mp.moe_routing_mode).strip().lower() not in {
            "token_choice",
            "expert_choice",
        }:
            raise ConfigError(
                "model.params.moe_routing_mode must be 'token_choice' or "
                "'expert_choice'."
            )
        if bool(_mp.moe_chain_of_experts):
            if str(_mp.moe_routing_mode).strip().lower() != "token_choice":
                raise ConfigError(
                    "model.params.moe_chain_of_experts requires "
                    "moe_routing_mode='token_choice'."
                )
            if int(_mp.moe_chain_router_rank) < 1:
                raise ConfigError(
                    "model.params.moe_chain_router_rank must be > 0 when CoE "
                    "is enabled."
                )
            if (
                int(_mp.moe_zero_experts)
                + int(_mp.moe_copy_experts)
                + int(_mp.moe_constant_experts)
                > 0
            ):
                raise ConfigError(
                    "Chain-of-Experts cannot compose with lightweight "
                    "experts."
                )
            if float(_mp.moe_spatiotemporal_routing_weight) != 0.0:
                raise ConfigError(
                    "Chain-of-Experts cannot compose with the "
                    "single-pass spatiotemporal routing objective."
                )
        elif int(_mp.moe_chain_router_rank) != 0:
            raise ConfigError(
                "model.params.moe_chain_router_rank must be 0 while "
                "moe_chain_of_experts=false."
            )
        _layer_router_bands: list[dict[str, Any]] = []
        for _index, _band in enumerate(_mp.moe_layer_router_policy):
            _path = f"model.params.moe_layer_router_policy[{_index}]"
            if "first_layer" not in _band or "end_layer" not in _band:
                raise ConfigError(f"{_path} requires first_layer and end_layer.")
            if not any(
                key in _band
                for key in ("top_k", "subset_fraction", "z_loss_weight")
            ):
                raise ConfigError(f"{_path} requires at least one router override.")
            try:
                _first_layer = int(_band["first_layer"])
                _end_layer = int(_band["end_layer"])
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"{_path} layer boundaries must be integers."
                ) from exc
            if not 0 <= _first_layer < _end_layer <= int(_mp.num_layers):
                raise ConfigError(
                    f"{_path} requires 0 <= first_layer < end_layer <= num_layers."
                )
            _normalized_band = dict(_band)
            _normalized_band["first_layer"] = _first_layer
            _normalized_band["end_layer"] = _end_layer
            if "top_k" in _band:
                try:
                    _top_k = int(_band["top_k"])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"{_path}.top_k must be an integer.") from exc
                if not 1 <= _top_k <= int(_mp.num_experts):
                    raise ConfigError(
                        f"{_path}.top_k must be in [1, num_experts]."
                    )
                if str(_mp.moe_routing_mode).strip().lower() != "token_choice":
                    raise ConfigError(
                        f"{_path}.top_k requires moe_routing_mode='token_choice'."
                    )
                if (
                    int(_mp.moe_zero_experts)
                    + int(_mp.moe_copy_experts)
                    + int(_mp.moe_constant_experts)
                    > 0
                ):
                    raise ConfigError(
                        f"{_path}.top_k cannot compose with lightweight experts; "
                        "use moe_lightweight_top_k for that topology."
                    )
                if float(_mp.moe_cross_layer_coupling_loss_weight) > 0.0:
                    raise ConfigError(
                        f"{_path}.top_k cannot compose with fixed-width "
                        "cross-layer coupling."
                    )
                _normalized_band["top_k"] = _top_k
            if "subset_fraction" in _band:
                try:
                    _subset_fraction = float(_band["subset_fraction"])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"{_path}.subset_fraction must be numeric."
                    ) from exc
                if not math.isfinite(_subset_fraction) or not (
                    0.0 < _subset_fraction <= 1.0
                ):
                    raise ConfigError(
                        f"{_path}.subset_fraction must be finite and in (0, 1]."
                    )
                if str(_mp.moe_routing_mode).strip().lower() != "token_choice":
                    raise ConfigError(
                        f"{_path}.subset_fraction requires "
                        "moe_routing_mode='token_choice'."
                    )
                _normalized_band["subset_fraction"] = _subset_fraction
            if "z_loss_weight" in _band:
                try:
                    _z_weight = float(_band["z_loss_weight"])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"{_path}.z_loss_weight must be numeric."
                    ) from exc
                if not math.isfinite(_z_weight) or _z_weight < 0.0:
                    raise ConfigError(
                        f"{_path}.z_loss_weight must be finite and >= 0."
                    )
                _normalized_band["z_loss_weight"] = _z_weight
            _resolved_top_k = int(
                _normalized_band.get("top_k", _mp.experts_per_token)
            )
            _resolved_subset = float(
                _normalized_band.get(
                    "subset_fraction",
                    _mp.expert_subset_fraction,
                )
            )
            if math.ceil(_resolved_subset * int(_mp.num_experts)) < _resolved_top_k:
                raise ConfigError(
                    f"{_path} resolves to fewer subset experts than top_k."
                )
            if _resolved_top_k < 2 and (
                float(_mp.moe_expert_orthogonality_loss_weight) > 0.0
                or float(_mp.moe_swiglu_specialization_loss_weight) > 0.0
            ):
                raise ConfigError(
                    f"{_path}.top_k must be >= 2 when co-activated expert "
                    "specialization objectives are enabled."
                )
            _layer_router_bands.append(_normalized_band)
        _layer_router_bands.sort(
            key=lambda band: (int(band["first_layer"]), int(band["end_layer"]))
        )
        for _left, _right in zip(
            _layer_router_bands,
            _layer_router_bands[1:],
            strict=False,
        ):
            if int(_right["first_layer"]) < int(_left["end_layer"]):
                raise ConfigError(
                    "model.params.moe_layer_router_policy bands must not overlap."
                )
        _mp.moe_layer_router_policy = _layer_router_bands
        _progressive_transition = int(
            _mp.moe_progressive_sparsification_transition_step
        )
        _progressive_raw = list(_mp.moe_progressive_sparsification_policy)
        if (_progressive_transition == 0) != (not _progressive_raw):
            raise ConfigError(
                "model.params progressive sparsification requires both a "
                "positive transition step and a non-empty policy."
            )
        if _progressive_transition < 0:
            raise ConfigError(
                "model.params.moe_progressive_sparsification_transition_step "
                "must be >= 0."
            )
        _progressive_bands: list[dict[str, int]] = []
        if _progressive_raw:
            _progressive_static_topology = (
                str(_mp.variant).strip().lower() in {"scratch", "tiny-video"}
            )
            if str(_mp.moe_routing_mode).strip().lower() != "token_choice":
                raise ConfigError(
                    "Progressive sparsification requires "
                    "model.params.moe_routing_mode='token_choice'."
                )
            if int(_mp.moe_dynamic_topk_min) > 0:
                raise ConfigError(
                    "Progressive sparsification cannot compose with "
                    "moe_dynamic_topk_min."
                )
            if (
                int(_mp.moe_zero_experts)
                + int(_mp.moe_copy_experts)
                + int(_mp.moe_constant_experts)
                > 0
            ):
                raise ConfigError(
                    "Progressive sparsification cannot compose with lightweight "
                    "experts."
                )
            if float(_mp.moe_cross_layer_coupling_loss_weight) > 0.0:
                raise ConfigError(
                    "Progressive sparsification cannot compose with fixed-width "
                    "cross-layer coupling."
                )

            def _target_router_settings(
                layer_index: int,
            ) -> tuple[int, float]:
                target_top_k = int(_mp.experts_per_token)
                subset_fraction = float(_mp.expert_subset_fraction)
                for layer_band in _layer_router_bands:
                    if (
                        int(layer_band["first_layer"])
                        <= int(layer_index)
                        < int(layer_band["end_layer"])
                    ):
                        target_top_k = int(
                            layer_band.get("top_k", target_top_k)
                        )
                        subset_fraction = float(
                            layer_band.get(
                                "subset_fraction",
                                subset_fraction,
                            )
                        )
                        break
                return target_top_k, subset_fraction

            for _index, _band in enumerate(_progressive_raw):
                _path = (
                    "model.params.moe_progressive_sparsification_policy"
                    f"[{_index}]"
                )
                if not all(
                    key in _band
                    for key in ("first_layer", "end_layer", "top_k")
                ):
                    raise ConfigError(
                        f"{_path} requires first_layer, end_layer, and top_k."
                    )
                try:
                    _first_layer = int(_band["first_layer"])
                    _end_layer = int(_band["end_layer"])
                    _top_k = int(_band["top_k"])
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"{_path} layer bounds and top_k must be integers."
                    ) from exc
                if not 0 <= _first_layer < _end_layer:
                    raise ConfigError(
                        f"{_path} requires 0 <= first_layer < end_layer."
                    )
                if _top_k < 1:
                    raise ConfigError(
                        f"{_path}.top_k must be positive."
                    )
                if _progressive_static_topology:
                    if _end_layer > int(_mp.num_layers):
                        raise ConfigError(
                            f"{_path}.end_layer must be <= num_layers."
                        )
                    if _top_k > int(_mp.num_experts):
                        raise ConfigError(
                            f"{_path}.top_k must be <= num_experts."
                        )
                    for _layer_index in range(_first_layer, _end_layer):
                        _target_top_k, _subset_fraction = (
                            _target_router_settings(_layer_index)
                        )
                        if _top_k <= _target_top_k:
                            raise ConfigError(
                                f"{_path}.top_k must exceed the target top_k "
                                "for every covered layer."
                            )
                        if (
                            math.ceil(
                                _subset_fraction * int(_mp.num_experts)
                            )
                            < _top_k
                        ):
                            raise ConfigError(
                                f"{_path}.top_k exceeds the resolved expert "
                                "subset."
                            )
                _progressive_bands.append(
                    {
                        "first_layer": _first_layer,
                        "end_layer": _end_layer,
                        "top_k": _top_k,
                    }
                )
            _progressive_bands.sort(
                key=lambda band: (
                    int(band["first_layer"]),
                    int(band["end_layer"]),
                )
            )
            for _left, _right in zip(
                _progressive_bands,
                _progressive_bands[1:],
                strict=False,
            ):
                if int(_right["first_layer"]) < int(_left["end_layer"]):
                    raise ConfigError(
                        "model.params.moe_progressive_sparsification_policy "
                        "bands must not overlap."
                    )
        _mp.moe_progressive_sparsification_policy = _progressive_bands
        if float(_mp.moe_expert_choice_capacity_factor) <= 0.0:
            raise ConfigError(
                "model.params.moe_expert_choice_capacity_factor must be > 0."
            )
        _timestep_capacity_schedule = str(
            _mp.moe_expert_choice_timestep_capacity_schedule
        ).strip().lower()
        if _timestep_capacity_schedule not in {"disabled", "linear_reverse"}:
            raise ConfigError(
                "model.params.moe_expert_choice_timestep_capacity_schedule "
                "must be 'disabled' or 'linear_reverse'."
            )
        _timestep_capacity_span = float(
            _mp.moe_expert_choice_timestep_capacity_span
        )
        if (
            not math.isfinite(_timestep_capacity_span)
            or _timestep_capacity_span < 0.0
        ):
            raise ConfigError(
                "model.params.moe_expert_choice_timestep_capacity_span "
                "must be finite and >= 0."
            )
        if (
            _timestep_capacity_schedule == "disabled"
            and _timestep_capacity_span != 0.0
        ):
            raise ConfigError(
                "model.params.moe_expert_choice_timestep_capacity_span must be "
                "0 when timestep capacity scheduling is disabled."
            )
        if (
            _timestep_capacity_schedule != "disabled"
            and _timestep_capacity_span <= 0.0
        ):
            raise ConfigError(
                "Enabled Expert-Choice timestep capacity scheduling requires "
                "model.params.moe_expert_choice_timestep_capacity_span > 0."
            )
        if (
            _timestep_capacity_schedule != "disabled"
            and str(_mp.moe_routing_mode).strip().lower() != "expert_choice"
        ):
            raise ConfigError(
                "model.params.moe_expert_choice_timestep_capacity_schedule "
                "requires moe_routing_mode='expert_choice'."
            )
        if not (
            0.0
            <= float(_mp.moe_expert_choice_coverage_alarm_threshold)
            <= 1.0
        ):
            raise ConfigError(
                "model.params.moe_expert_choice_coverage_alarm_threshold "
                "must be in [0, 1]."
            )
        if (
            float(_mp.moe_expert_choice_coverage_alarm_threshold) > 0.0
            and str(_mp.moe_routing_mode).strip().lower() != "expert_choice"
        ):
            raise ConfigError(
                "model.params.moe_expert_choice_coverage_alarm_threshold "
                "requires moe_routing_mode='expert_choice'."
            )
        _lightweight_counts = {
            "moe_zero_experts": int(_mp.moe_zero_experts),
            "moe_copy_experts": int(_mp.moe_copy_experts),
            "moe_constant_experts": int(_mp.moe_constant_experts),
        }
        for _name, _count in _lightweight_counts.items():
            if _count < 0:
                raise ConfigError(f"model.params.{_name} must be >= 0.")
        _lightweight_total = sum(_lightweight_counts.values())
        if _lightweight_total == 0 and int(_mp.moe_lightweight_top_k) != 0:
            raise ConfigError(
                "model.params.moe_lightweight_top_k must be 0 when "
                "all lightweight expert counts are disabled."
            )
        if _lightweight_total > 0:
            if not (
                1
                <= int(_mp.moe_lightweight_top_k)
                <= int(_mp.num_experts) + _lightweight_total
            ):
                raise ConfigError(
                    "model.params.moe_lightweight_top_k must be in "
                    "[1, num_experts + all lightweight experts]."
                )
            if str(_mp.moe_routing_mode).strip().lower() != "token_choice":
                raise ConfigError(
                    "Lightweight experts require "
                    "model.params.moe_routing_mode='token_choice'."
                )
        _adjugate_groups = int(_mp.moe_adjugate_expert_groups)
        _adjugate_intermediate = int(
            _mp.moe_adjugate_expert_intermediate_size
        )
        _adjugate_scale = float(_mp.moe_adjugate_expert_scale)
        if _adjugate_groups < 0:
            raise ConfigError(
                "model.params.moe_adjugate_expert_groups must be >= 0."
            )
        if _adjugate_intermediate <= 0:
            raise ConfigError(
                "model.params.moe_adjugate_expert_intermediate_size must be > 0."
            )
        if not math.isfinite(_adjugate_scale) or _adjugate_scale <= 0.0:
            raise ConfigError(
                "model.params.moe_adjugate_expert_scale must be finite and > 0."
            )
        if bool(_mp.moe_adjugate_experts):
            if _adjugate_groups <= 0:
                raise ConfigError(
                    "model.params.moe_adjugate_expert_groups must be > 0 "
                    "when grouped adjugate experts are enabled."
                )
            if str(_mp.variant).strip().lower() in {"scratch", "tiny-video"}:
                _adjugate_num_experts = int(_mp.num_experts)
                if (
                    _adjugate_groups > _adjugate_num_experts
                    or _adjugate_num_experts % _adjugate_groups != 0
                ):
                    raise ConfigError(
                        "model.params.moe_adjugate_expert_groups must divide "
                        "model.params.num_experts."
                    )
                _adjugate_max_scale = (
                    float(_adjugate_groups) / float(_adjugate_num_experts)
                )
                if _adjugate_scale > _adjugate_max_scale:
                    raise ConfigError(
                        "model.params.moe_adjugate_expert_scale must be <= "
                        "moe_adjugate_expert_groups / num_experts "
                        f"({_adjugate_max_scale:g})."
                    )
        if float(_mp.moe_router_timestep_weight) < 0.0:
            raise ConfigError(
                "model.params.moe_router_timestep_weight must be >= 0."
            )
        if (
            float(_mp.moe_router_timestep_weight) > 0.0
            and str(_mp.moe_routing_mode).strip().lower() != "expert_choice"
        ):
            raise ConfigError(
                "model.params.moe_router_timestep_weight requires "
                "moe_routing_mode='expert_choice'."
            )
        _balance_grad_ratio_threshold = float(
            _mp.moe_balance_grad_ratio_threshold
        )
        if int(_mp.moe_balance_loss_disable_step) < 0:
            raise ConfigError(
                "model.params.moe_balance_loss_disable_step must be >= 0."
            )
        if (
            not math.isfinite(_balance_grad_ratio_threshold)
            or _balance_grad_ratio_threshold <= 0.0
        ):
            raise ConfigError(
                "model.params.moe_balance_grad_ratio_threshold must be "
                "finite and > 0."
            )
        for index, band in enumerate(_mp.moe_expert_choice_capacity_schedule):
            missing = {
                "start_step",
                "first_layer",
                "end_layer",
                "capacity_factor",
            } - set(band)
            if missing:
                raise ConfigError(
                    "model.params.moe_expert_choice_capacity_schedule"
                    f"[{index}] is missing {sorted(missing)}."
                )
            start_step = int(band["start_step"])
            end_step = int(band.get("end_step", -1))
            first_layer = int(band["first_layer"])
            end_layer = int(band["end_layer"])
            if start_step < 0 or (end_step != -1 and end_step <= start_step):
                raise ConfigError(
                    "Expert-Choice capacity schedule requires start_step >= 0 "
                    "and end_step=-1 or end_step > start_step."
                )
            if first_layer < 0 or end_layer <= first_layer:
                raise ConfigError(
                    "Expert-Choice capacity schedule requires "
                    "0 <= first_layer < end_layer."
                )
            if float(band["capacity_factor"]) <= 0.0:
                raise ConfigError(
                    "Expert-Choice capacity schedule capacity_factor must be > 0."
                )
        schedule = _mp.moe_expert_choice_capacity_schedule
        for left_index, left in enumerate(schedule):
            for right in schedule[left_index + 1 :]:
                left_step_end = math.inf if int(left.get("end_step", -1)) < 0 else int(
                    left["end_step"]
                )
                right_step_end = (
                    math.inf
                    if int(right.get("end_step", -1)) < 0
                    else int(right["end_step"])
                )
                steps_overlap = (
                    int(left["start_step"]) < right_step_end
                    and int(right["start_step"]) < left_step_end
                )
                layers_overlap = (
                    int(left["first_layer"]) < int(right["end_layer"])
                    and int(right["first_layer"]) < int(left["end_layer"])
                )
                if steps_overlap and layers_overlap:
                    raise ConfigError(
                        "Expert-Choice capacity schedule bands must not overlap "
                        "in both step and layer ranges."
                    )
        layer_boundaries = sorted(
            {
                int(boundary)
                for band in schedule
                for boundary in (band["first_layer"], band["end_layer"])
            }
        )
        for layer_index in layer_boundaries[:-1]:
            step_boundaries = {0}
            for band in schedule:
                if not (
                    int(band["first_layer"])
                    <= layer_index
                    < int(band["end_layer"])
                ):
                    continue
                step_boundaries.add(int(band["start_step"]))
                if int(band.get("end_step", -1)) >= 0:
                    step_boundaries.add(int(band["end_step"]))
            previous_factor: float | None = None
            for step in sorted(step_boundaries):
                factor = float(_mp.moe_expert_choice_capacity_factor)
                for band in schedule:
                    end_step = int(band.get("end_step", -1))
                    if (
                        int(band["start_step"]) <= step
                        and (end_step < 0 or step < end_step)
                        and int(band["first_layer"])
                        <= layer_index
                        < int(band["end_layer"])
                    ):
                        factor = float(band["capacity_factor"])
                        break
                if previous_factor is not None and factor > previous_factor:
                    raise ConfigError(
                        "Expert-Choice capacity schedule must resolve to a "
                        "monotone non-increasing capacity for every covered "
                        "layer, including fallback transitions."
                    )
                previous_factor = factor
        if not (0.0 < float(_mp.expert_subset_fraction) <= 1.0):
            raise ConfigError(
                "model.params.expert_subset_fraction must be in (0, 1]."
            )
        if float(_mp.expert_subset_pool_factor) < 1.0:
            raise ConfigError(
                "model.params.expert_subset_pool_factor must be >= 1."
            )
        if int(_mp.expert_subset_anneal_steps) < 0:
            raise ConfigError(
                "model.params.expert_subset_anneal_steps must be >= 0."
            )
        if float(_mp.expert_subset_router_kl_weight) < 0.0:
            raise ConfigError(
                "model.params.expert_subset_router_kl_weight must be >= 0."
            )
        if str(_mp.moe_routing_agreement_evidence).strip().lower() not in {
            "off",
            "report",
        }:
            raise ConfigError(
                "model.params.moe_routing_agreement_evidence must be 'off' "
                "(default) or 'report'."
            )
        if str(_mp.expert_pruning).strip().lower() not in {"off", "prune"}:
            raise ConfigError(
                "model.params.expert_pruning must be 'off' (default) or 'prune'."
            )
        if str(_mp.expert_pruning_criterion).strip().lower() not in {
            "frequency",
            "reap",
            "man",
            "msan",
            "aimer",
        }:
            raise ConfigError(
                "model.params.expert_pruning_criterion must be frequency, "
                "reap, man, msan, or aimer."
            )
        if str(_mp.expert_consolidation).strip().lower() not in {
            "off",
            "prototype",
            "hierarchical_output",
        }:
            raise ConfigError(
                "model.params.expert_consolidation must be 'off' (default), "
                "'prototype', or 'hierarchical_output'."
            )
        if str(_mp.expert_upcycling).strip().lower() not in {"off", "drop"}:
            raise ConfigError(
                "model.params.expert_upcycling must be 'off' (default) or 'drop'."
            )
        if int(_mp.expert_upcycling_copies) < 0:
            raise ConfigError(
                "model.params.expert_upcycling_copies must be >= 0."
            )
        if not (
            math.isfinite(float(_mp.expert_upcycling_reinit_ratio))
            and 0.0 < float(_mp.expert_upcycling_reinit_ratio) <= 1.0
        ):
            raise ConfigError(
                "model.params.expert_upcycling_reinit_ratio must be finite and "
                "in (0, 1]."
            )
        if int(_mp.expert_upcycling_seed) < 0:
            raise ConfigError("model.params.expert_upcycling_seed must be >= 0.")
        if (
            str(_mp.expert_upcycling).strip().lower() == "drop"
            and int(_mp.expert_upcycling_copies) < 1
        ):
            raise ConfigError(
                "expert_upcycling='drop' requires expert_upcycling_copies >= 1."
            )
        if (
            str(_mp.expert_upcycling).strip().lower() == "off"
            and int(_mp.expert_upcycling_copies) != 0
        ):
            raise ConfigError(
                "expert_upcycling_copies must be 0 while expert_upcycling='off'."
            )
        if str(_mp.expert_weight_compression).strip().lower() not in {
            "off",
            "flexmoe_nested",
            "mixture_basis",
            "shared_basis",
            "stun_sparse24",
        }:
            raise ConfigError(
                "model.params.expert_weight_compression must be 'off' (default) "
                "'flexmoe_nested', 'mixture_basis', 'shared_basis', or "
                "'stun_sparse24'."
            )
        if str(_mp.flexmoe_calibration).strip().lower() not in {
            "off",
            "nested",
        }:
            raise ConfigError(
                "model.params.flexmoe_calibration must be 'off' (default) or "
                "'nested'."
            )
        if (
            str(_mp.flexmoe_calibration).strip().lower() == "nested"
            and str(_mp.expert_weight_compression).strip().lower() != "off"
        ):
            raise ConfigError(
                "flexmoe_calibration='nested' requires the complete source with "
                "expert_weight_compression='off'."
            )
        if (
            str(_mp.flexmoe_calibration).strip().lower() == "nested"
            and str(_mp.moe_routing_mode).strip().lower() != "token_choice"
        ):
            raise ConfigError(
                "flexmoe_calibration='nested' requires "
                "moe_routing_mode='token_choice'."
            )
        if str(_mp.post_compression_router_repair).strip().lower() not in {
            "off",
            "router_kd",
        }:
            raise ConfigError(
                "model.params.post_compression_router_repair must be 'off' "
                "(default) or 'router_kd'."
            )
        if (
            str(_mp.router_repair_artifact_path).strip()
            and str(_mp.post_compression_router_repair).strip().lower()
            != "router_kd"
        ):
            raise ConfigError(
                "model.params.router_repair_artifact_path requires "
                "post_compression_router_repair='router_kd'."
            )
        if str(_mp.expert_quantization_calibration).strip().lower() not in {
            "off",
            "affinity",
        }:
            raise ConfigError(
                "model.params.expert_quantization_calibration must be 'off' "
                "(default) or 'affinity'."
            )
        if str(_mp.expert_quantization_rotation).strip().lower() not in {
            "off",
            "learned",
        }:
            raise ConfigError(
                "model.params.expert_quantization_rotation must be 'off' "
                "(default) or 'learned'."
            )
        if str(_mp.router_quantization_calibration).strip().lower() not in {
            "off",
            "eaquant",
        }:
            raise ConfigError(
                "model.params.router_quantization_calibration must be 'off' "
                "(default) or 'eaquant'."
            )
        if str(_mp.expert_factorization_calibration).strip().lower() not in {
            "off",
            "whitened",
        }:
            raise ConfigError(
                "model.params.expert_factorization_calibration must be 'off' "
                "(default) or 'whitened'."
            )
        if str(_mp.expert_precision_calibration).strip().lower() not in {
            "off",
            "imatrix",
        }:
            raise ConfigError(
                "model.params.expert_precision_calibration must be 'off' "
                "(default) or 'imatrix'."
            )
        if (
            str(_mp.expert_factorization_calibration).strip().lower()
            == "whitened"
            and str(_mp.expert_weight_compression).strip().lower()
            not in {"off", "shared_basis"}
        ):
            raise ConfigError(
                "expert_factorization_calibration='whitened' requires "
                "expert_weight_compression='off' for evidence collection or "
                "'shared_basis' for artifact transformation."
            )
        if (
            str(_mp.expert_factorization_calibration).strip().lower()
            == "whitened"
            and str(_mp.moe_routing_mode).strip().lower() != "token_choice"
        ):
            raise ConfigError(
                "expert_factorization_calibration='whitened' requires "
                "moe_routing_mode='token_choice'."
            )
        strategy = StrategyConfig(
            type=str(strategy_table.get("type", "text_to_video")),
            params=dict(strategy_params),
        )
        training = TrainingSection(
            seed=int(training_table.get("seed", 42)),
            batch_size=int(training_table.get("batch_size", 2)),
            gradient_accumulation=int(training_table.get("gradient_accumulation", 1)),
            max_steps=int(training_table.get("max_steps", 100)),
            loss_function=str(training_table.get("loss_function", "mse")),
            objective=str(training_table.get("objective", "flow_matching")),
            contrastive_flow_weight=float(
                training_table.get("contrastive_flow_weight", 0.0)
            ),
            latent_wavelet_loss_weight=float(
                training_table.get("latent_wavelet_loss_weight", 0.0)
            ),
            loss_weighting=str(training_table.get("loss_weighting", "uniform")),
            min_snr_gamma=float(training_table.get("min_snr_gamma", 5.0)),
            timestep_eps=float(training_table.get("timestep_eps", 1e-5)),
            timestep_sampling=str(
                training_table.get("timestep_sampling", "uniform")
            ),
            timestep_sampling_mean=float(
                training_table.get("timestep_sampling_mean", 0.0)
            ),
            timestep_sampling_std=float(
                training_table.get("timestep_sampling_std", 1.0)
            ),
            timestep_sampling_mode_scale=float(
                training_table.get("timestep_sampling_mode_scale", 1.29)
            ),
            warmup_steps=int(training_table.get("warmup_steps", 0)),
            non_finite_grad_policy=str(
                training_table.get("non_finite_grad_policy", "abort")
            ),
            max_consecutive_skipped_steps=int(
                training_table.get("max_consecutive_skipped_steps", 5)
            ),
            moe_expert_touch_guard=str(
                training_table.get("moe_expert_touch_guard", "off")
            ),
            moe_expert_touch_max_fraction=float(
                training_table.get("moe_expert_touch_max_fraction", 1.0)
            ),
            max_grad_norm=float(training_table.get("max_grad_norm", 1.0)),
            val_every_n_steps=int(training_table.get("val_every_n_steps", 0)),
            early_stop_patience=int(training_table.get("early_stop_patience", 0)),
            ema_enabled=bool(training_table.get("ema_enabled", False)),
            ema_decay=float(training_table.get("ema_decay", 0.999)),
            posthoc_ema_enabled=bool(
                training_table.get("posthoc_ema_enabled", False)
            ),
            posthoc_ema_profile_stds=[
                float(value)
                for value in training_table.get(
                    "posthoc_ema_profile_stds",
                    [0.05, 0.1],
                )
            ],
            posthoc_ema_snapshot_every_n_steps=int(
                training_table.get(
                    "posthoc_ema_snapshot_every_n_steps",
                    0,
                )
            ),
            prior_loss_weight=float(training_table.get("prior_loss_weight", 1.0)),
            prior_ratio=float(training_table.get("prior_ratio", 0.0)),
            log_grad_breakdown_every_n_steps=int(
                training_table.get("log_grad_breakdown_every_n_steps", 0)
            ),
            gradient_checkpointing=(
                "standard"
                if training_table.get("gradient_checkpointing", "standard") is True
                else (
                    "off"
                    if training_table.get("gradient_checkpointing", "standard") is False
                    else str(training_table.get("gradient_checkpointing", "standard"))
                )
            ),
            noise_offset=float(training_table.get("noise_offset", 0.0)),
            loss_bucket_normalization=str(
                training_table.get("loss_bucket_normalization", "none")
            ),
            blocks_to_swap=int(training_table.get("blocks_to_swap", 0)),
            block_swap_mode=str(training_table.get("block_swap_mode", "sync")),
            optimizer_cpu_offload=bool(
                training_table.get("optimizer_cpu_offload", False)
            ),
            gradient_cpu_offload=bool(
                training_table.get("gradient_cpu_offload", False)
            ),
            activation_cpu_offload=bool(
                training_table.get("activation_cpu_offload", False)
            ),
            activation_cpu_offload_min_mib=int(
                training_table.get("activation_cpu_offload_min_mib", 8)
            ),
            activation_cpu_offload_max_gib=float(
                training_table.get("activation_cpu_offload_max_gib", 0.0)
            ),
            activation_cpu_offload_pin_memory=bool(
                training_table.get("activation_cpu_offload_pin_memory", False)
            ),
            activation_cpu_offload_defer_layers=int(
                training_table.get("activation_cpu_offload_defer_layers", 0)
            ),
            activation_cpu_offload_prefetch_layers=int(
                training_table.get("activation_cpu_offload_prefetch_layers", 0)
            ),
            activation_cpu_offload_view_replay=bool(
                training_table.get("activation_cpu_offload_view_replay", False)
            ),
            activation_compression=bool(
                training_table.get("activation_compression", False)
            ),
            activation_compression_rank=int(
                training_table.get("activation_compression_rank", 32)
            ),
            activation_compression_min_mib=int(
                training_table.get("activation_compression_min_mib", 8)
            ),
            masked_loss=bool(training_table.get("masked_loss", False)),
            compile=bool(training_table.get("compile", False)),
            compile_scope=str(
                training_table.get("compile_scope", "regional")
            ),
            compile_mode=str(training_table.get("compile_mode", "default")),
            compile_dynamic=compile_dynamic_raw,
            compile_token_buckets=[
                int(value) for value in compile_token_buckets_raw
            ],
            moe_token_chunk_size=int(
                training_table.get("moe_token_chunk_size", 0)
            ),
            block_swap_backward=bool(training_table.get("block_swap_backward", True)),
            pinned_memory_budget_fraction=float(
                training_table.get("pinned_memory_budget_fraction", 0.5)
            ),
            curriculum=dict(curriculum_raw),
            policy_modules=[
                str(value).strip()
                for value in policy_modules_raw
                if str(value).strip()
            ],
            policy_options={
                str(name).strip().lower(): dict(options)
                for name, options in policy_options_raw.items()
            },
        )
        optimizer = OptimizerConfig(
            type=str(optimizer_table.get("type", "adamw")),
            lr=float(optimizer_table.get("lr", 1e-3)),
            weight_decay=float(optimizer_table.get("weight_decay", 0.0)),
            weight_decay_filter=str(optimizer_table.get("weight_decay_filter", "lora_b_bias")),
            loraplus_lr_ratio=float(optimizer_table.get("loraplus_lr_ratio", 1.0)),
            prodigy_beta3=float(optimizer_table.get("prodigy_beta3", 0.0)),
            prodigy_decouple=bool(optimizer_table.get("prodigy_decouple", True)),
            prodigy_use_bias_correction=bool(
                optimizer_table.get("prodigy_use_bias_correction", False)
            ),
            prodigy_safeguard_warmup=bool(
                optimizer_table.get("prodigy_safeguard_warmup", False)
            ),
            prodigy_d0=float(optimizer_table.get("prodigy_d0", 1e-6)),
            prodigy_d_coef=float(optimizer_table.get("prodigy_d_coef", 1.0)),
            prodigy_growth_rate=float(
                optimizer_table.get("prodigy_growth_rate", math.inf)
            ),
            prodigy_slice_p=int(optimizer_table.get("prodigy_slice_p", 1)),
            lora_pro_damping=float(
                optimizer_table.get("lora_pro_damping", 1e-8)
            ),
            muon_momentum=float(optimizer_table.get("muon_momentum", 0.95)),
            muon_nesterov=bool(optimizer_table.get("muon_nesterov", True)),
            muon_ns_steps=int(optimizer_table.get("muon_ns_steps", 5)),
            muon_eps=float(optimizer_table.get("muon_eps", 1e-8)),
            muon_rms_target=float(
                optimizer_table.get("muon_rms_target", 0.2)
            ),
            lora_muon_gauge_rebalance_interval=int(
                optimizer_table.get("lora_muon_gauge_rebalance_interval", 0)
            ),
            lora_muon_gauge_rebalance_alpha=float(
                optimizer_table.get("lora_muon_gauge_rebalance_alpha", 1.0)
            ),
            stochastic_rounding=bool(
                optimizer_table.get("stochastic_rounding", False)
            ),
            allow_fallback=bool(optimizer_table.get("allow_fallback", True)),
            scheduler=str(optimizer_table.get("scheduler", "constant")),
            scheduler_num_cycles=float(optimizer_table.get("scheduler_num_cycles", 1.0)),
            scheduler_power=float(optimizer_table.get("scheduler_power", 1.0)),
            min_lr_ratio=float(optimizer_table.get("min_lr_ratio", 0.0)),
            selected_expert_ids=[
                int(value)
                for value in optimizer_table.get("selected_expert_ids", [])
            ],
        )
        if not (
            optimizer.prodigy_beta3 == 0.0
            or 0.0 < optimizer.prodigy_beta3 < 1.0
        ):
            raise ConfigError("optimizer.prodigy_beta3 must be 0 or in (0, 1).")
        if not math.isfinite(optimizer.prodigy_d0) or optimizer.prodigy_d0 <= 0.0:
            raise ConfigError("optimizer.prodigy_d0 must be finite and > 0.")
        if not (
            math.isfinite(optimizer.prodigy_d_coef)
            and optimizer.prodigy_d_coef > 0.0
        ):
            raise ConfigError("optimizer.prodigy_d_coef must be finite and > 0.")
        if (
            math.isnan(optimizer.prodigy_growth_rate)
            or optimizer.prodigy_growth_rate < 1.0
        ):
            raise ConfigError("optimizer.prodigy_growth_rate must be >= 1.")
        if optimizer.prodigy_slice_p < 1:
            raise ConfigError("optimizer.prodigy_slice_p must be >= 1.")
        if not (
            math.isfinite(optimizer.lora_pro_damping)
            and optimizer.lora_pro_damping > 0.0
        ):
            raise ConfigError(
                "optimizer.lora_pro_damping must be finite and > 0."
            )
        if not (
            math.isfinite(optimizer.muon_momentum)
            and 0.0 <= optimizer.muon_momentum < 1.0
        ):
            raise ConfigError(
                "optimizer.muon_momentum must be finite and in [0, 1)."
            )
        if optimizer.muon_ns_steps < 1:
            raise ConfigError("optimizer.muon_ns_steps must be >= 1.")
        if not (
            math.isfinite(optimizer.muon_eps)
            and optimizer.muon_eps > 0.0
        ):
            raise ConfigError("optimizer.muon_eps must be finite and > 0.")
        if not (
            math.isfinite(optimizer.muon_rms_target)
            and optimizer.muon_rms_target > 0.0
        ):
            raise ConfigError(
                "optimizer.muon_rms_target must be finite and > 0."
            )
        if optimizer.lora_muon_gauge_rebalance_interval < 0:
            raise ConfigError(
                "optimizer.lora_muon_gauge_rebalance_interval must be >= 0."
            )
        if not (
            math.isfinite(optimizer.lora_muon_gauge_rebalance_alpha)
            and 0.0 < optimizer.lora_muon_gauge_rebalance_alpha <= 1.0
        ):
            raise ConfigError(
                "optimizer.lora_muon_gauge_rebalance_alpha must be finite "
                "and in (0, 1]."
            )
        adapter = AdapterConfig(
            type=str(adapter_table.get("type", "lora")),
            target_preset=str(adapter_table.get("target_preset", "attn_only")),
            rank=int(adapter_table.get("rank", 16)),
            alpha=float(adapter_table.get("alpha", 16.0)),
            rank_pattern={str(k): int(v) for k, v in rank_pattern_raw.items()},
            alpha_pattern={str(k): float(v) for k, v in alpha_pattern_raw.items()},
            rank_budget=int(adapter_table.get("rank_budget", 0)),
            adaptive_rank_plan_path=_validate_no_control_path(
                "adapter.adaptive_rank_plan_path",
                str(adapter_table.get("adaptive_rank_plan_path", "")),
            ),
            use_rslora=bool(adapter_table.get("use_rslora", False)),
            use_lora_fa=bool(adapter_table.get("use_lora_fa", False)),
            use_dora=bool(adapter_table.get("use_dora", False)),
            expert_tensor_lora_backend=str(
                adapter_table.get("expert_tensor_lora_backend", "weight_space")
            ).strip().lower(),
            init_from=str(adapter_table.get("init_from", "")),
            rank_dropout=float(adapter_table.get("rank_dropout", 0.0)),
            lora_parameter_dropout=float(
                adapter_table.get("lora_parameter_dropout", 0.0)
            ),
            posthoc_rank_compression=str(
                adapter_table.get("posthoc_rank_compression", "off")
            ).strip().lower(),
            rank_schedule_start=int(adapter_table.get("rank_schedule_start", 0)),
            rank_schedule_end=int(adapter_table.get("rank_schedule_end", 0)),
            rank_schedule_min_scale=float(adapter_table.get("rank_schedule_min_scale", 1.0)),
            lr_multipliers={
                str(k): float(v) for k, v in dict(lr_multipliers_raw).items()
            },
            expert_selection=str(
                adapter_table.get("expert_selection", "all")
            ).strip().lower(),
            expert_topk_fraction=float(
                adapter_table.get("expert_topk_fraction", 0.25)
            ),
            expert_calibration_steps=int(
                adapter_table.get("expert_calibration_steps", 8)
            ),
            esft_selection_mass=float(
                adapter_table.get("esft_selection_mass", 0.2)
            ),
            esft_calibration_samples=int(
                adapter_table.get("esft_calibration_samples", 32)
            ),
            sparse_expert_export=bool(
                adapter_table.get("sparse_expert_export", False)
            ),
            sparse_delta_density=float(
                adapter_table.get("sparse_delta_density", 0.01)
            ),
            sparse_delta_selection=str(
                adapter_table.get("sparse_delta_selection", "magnitude")
            ),
            timestep_rank_schedule=str(
                adapter_table.get("timestep_rank_schedule", "none")
            ).strip().lower(),
            timestep_rank_min_fraction=float(
                adapter_table.get("timestep_rank_min_fraction", 0.25)
            ),
            lora_init=str(adapter_table.get("lora_init", "kaiming")).strip().lower(),
            eva_calibration_steps=int(
                adapter_table.get("eva_calibration_steps", 32)
            ),
            eva_samples_per_target=int(
                adapter_table.get("eva_samples_per_target", 256)
            ),
            eva_convergence_threshold=float(
                adapter_table.get("eva_convergence_threshold", 0.99)
            ),
            lora_ga_calibration_steps=int(
                adapter_table.get("lora_ga_calibration_steps", 4)
            ),
            lora_ga_stable_gamma=float(
                adapter_table.get("lora_ga_stable_gamma", 16.0)
            ),
            gora_calibration_steps=int(
                adapter_table.get("gora_calibration_steps", 64)
            ),
            gora_min_rank=int(adapter_table.get("gora_min_rank", 0)),
            gora_max_rank=int(adapter_table.get("gora_max_rank", 0)),
            gora_stable_gamma=float(
                adapter_table.get("gora_stable_gamma", 0.05)
            ),
            tc_gate_hidden_dim=int(adapter_table.get("tc_gate_hidden_dim", 8)),
            timestep_band_min=float(adapter_table.get("timestep_band_min", 0.0)),
            timestep_band_max=float(adapter_table.get("timestep_band_max", 1.0)),
            condenser_rank=int(adapter_table.get("condenser_rank", 0)),
            condenser_alpha=float(adapter_table.get("condenser_alpha", 0.0)),
            train_router=(
                None
                if adapter_table.get("train_router", None) is None
                else bool(adapter_table.get("train_router"))
            ),
        )
        if bool(model.params.moe_chain_of_experts) and (
            str(adapter.type).strip().lower() != "lora"
        ):
            raise ConfigError(
                "model.params.moe_chain_of_experts requires adapter.type='lora'."
            )
        if adapter.condenser_rank < 0:
            raise ConfigError("adapter.condenser_rank must be >= 0.")
        if adapter.posthoc_rank_compression not in {"off", "para"}:
            raise ConfigError(
                "adapter.posthoc_rank_compression must be 'off' (default) "
                "or 'para'."
            )
        if (
            adapter.posthoc_rank_compression == "para"
            and adapter.type.strip().lower() != "lora"
        ):
            raise ConfigError(
                "adapter.posthoc_rank_compression='para' requires "
                "adapter.type='lora'."
            )
        if adapter.expert_tensor_lora_backend not in {"weight_space", "activation"}:
            raise ConfigError(
                "adapter.expert_tensor_lora_backend must be 'weight_space' or "
                "'activation'."
            )
        if adapter.rank_budget < 0:
            raise ConfigError("adapter.rank_budget must be >= 0.")
        for pattern, value in rank_pattern_raw.items():
            if isinstance(value, bool) or not str(pattern).strip() or int(value) <= 0:
                raise ConfigError(
                    "adapter.rank_pattern keys must be non-empty and values must be > 0."
                )
        for pattern, value in alpha_pattern_raw.items():
            resolved = float(value)
            if (
                isinstance(value, bool)
                or not str(pattern).strip()
                or not math.isfinite(resolved)
                or resolved <= 0.0
            ):
                raise ConfigError(
                    "adapter.alpha_pattern keys must be non-empty and values must be "
                    "finite and > 0."
                )
        if adapter.condenser_alpha < 0.0:
            raise ConfigError("adapter.condenser_alpha must be >= 0.")
        if not (
            math.isfinite(adapter.lora_parameter_dropout)
            and 0.0 <= adapter.lora_parameter_dropout < 1.0
        ):
            raise ConfigError(
                "adapter.lora_parameter_dropout must be finite and in [0, 1)."
            )
        if adapter.expert_selection not in {
            "all",
            "routing_topk",
            "esft_gate",
            "esft_token",
        }:
            raise ConfigError(
                "adapter.expert_selection must be 'all', 'routing_topk', "
                "'esft_gate', or 'esft_token'."
            )
        if adapter.expert_selection == "routing_topk":
            if not (0.0 < adapter.expert_topk_fraction <= 1.0):
                raise ConfigError(
                    "adapter.expert_topk_fraction must be in (0, 1]."
                )
            if adapter.expert_calibration_steps < 1:
                raise ConfigError(
                    "adapter.expert_calibration_steps must be >= 1."
                )
        if not (
            math.isfinite(adapter.esft_selection_mass)
            and 0.0 < adapter.esft_selection_mass <= 1.0
        ):
            raise ConfigError(
                "adapter.esft_selection_mass must be finite and in (0, 1]."
            )
        if adapter.esft_calibration_samples < 1:
            raise ConfigError("adapter.esft_calibration_samples must be >= 1.")
        if adapter.timestep_rank_schedule not in {"none", "tlora", "tc_gate"}:
            raise ConfigError(
                "adapter.timestep_rank_schedule must be 'none', 'tlora' or "
                "'tc_gate'."
            )
        if adapter.tc_gate_hidden_dim < 1:
            raise ConfigError("adapter.tc_gate_hidden_dim must be >= 1.")
        if not (0.0 < adapter.timestep_rank_min_fraction <= 1.0):
            raise ConfigError(
                "adapter.timestep_rank_min_fraction must be in (0, 1]."
            )
        if not adapter.lora_init:
            raise ConfigError(
                "adapter.lora_init must name a registered LoRA initializer."
            )
        if adapter.eva_calibration_steps < 2:
            raise ConfigError("adapter.eva_calibration_steps must be >= 2.")
        if adapter.eva_samples_per_target < 1:
            raise ConfigError("adapter.eva_samples_per_target must be >= 1.")
        if not (0.0 < adapter.eva_convergence_threshold <= 1.0):
            raise ConfigError(
                "adapter.eva_convergence_threshold must be in (0, 1]."
            )
        if adapter.gora_calibration_steps < 1:
            raise ConfigError("adapter.gora_calibration_steps must be >= 1.")
        if adapter.gora_min_rank < 0:
            raise ConfigError("adapter.gora_min_rank must be >= 0.")
        if adapter.gora_max_rank < 0:
            raise ConfigError("adapter.gora_max_rank must be >= 0.")
        if (
            adapter.gora_min_rank > 0
            and adapter.gora_max_rank > 0
            and adapter.gora_max_rank < adapter.gora_min_rank
        ):
            raise ConfigError(
                "adapter.gora_max_rank must be zero or >= gora_min_rank."
            )
        if not (
            math.isfinite(adapter.gora_stable_gamma)
            and adapter.gora_stable_gamma > 0.0
        ):
            raise ConfigError(
                "adapter.gora_stable_gamma must be finite and > 0."
            )
        if adapter.use_lora_fa and (
            adapter.rank_dropout > 0.0
            or adapter.lora_parameter_dropout > 0.0
            or adapter.timestep_rank_schedule != "none"
            or adapter.timestep_band_min != 0.0
            or adapter.timestep_band_max != 1.0
        ):
            raise ConfigError(
                "adapter.use_lora_fa cannot be combined with LoRA dropout or "
                "timestep rank masks because LoRA-FA requires a fixed A "
                "projection and fixed B factor."
            )
        if not (
            0.0 <= adapter.timestep_band_min
            and adapter.timestep_band_min < adapter.timestep_band_max
            and adapter.timestep_band_max <= 1.0
        ):
            raise ConfigError(
                "adapter timestep band requires 0 <= timestep_band_min < "
                "timestep_band_max <= 1."
            )
        dataset = DatasetConfig(
            path=dataset_path,
            cache_path=cache_path,
            auto_preprocess_cache=bool(
                dataset_table.get("auto_preprocess_cache", False)
            ),
            preprocess_raw_media_to_pt=bool(
                dataset_table.get("preprocess_raw_media_to_pt", False)
            ),
            max_cache_skip_ratio=float(dataset_table.get("max_cache_skip_ratio", 0.2)),
            split_seed=int(dataset_table.get("split_seed", 42)),
            train_ratio=float(dataset_table.get("train_ratio", 0.8)),
            val_ratio=float(dataset_table.get("val_ratio", 0.1)),
            test_ratio=float(dataset_table.get("test_ratio", 0.1)),
            usage_mode=str(dataset_table.get("usage_mode", "internal")),
            num_workers=int(dataset_table.get("num_workers", 0)),
            prefetch_factor=int(dataset_table.get("prefetch_factor", 2)),
            persistent_workers=bool(dataset_table.get("persistent_workers", True)),
            cache_mode=str(dataset_table.get("cache_mode", "disk")),
            cache_compression=str(dataset_table.get("cache_compression", "none")),
            fp8_text_encoder=bool(dataset_table.get("fp8_text_encoder", False)),
            clips_per_video=int(dataset_table.get("clips_per_video", 1)),
            tag_shuffle_variants=int(dataset_table.get("tag_shuffle_variants", 1)),
            online_tag_shuffle=bool(dataset_table.get("online_tag_shuffle", False)),
            online_tag_shuffle_dropout=float(
                dataset_table.get("online_tag_shuffle_dropout", 0.0)
            ),
            online_tag_shuffle_keep_first_n_tags=int(
                dataset_table.get("online_tag_shuffle_keep_first_n_tags", 1)
            ),
            online_temporal_resampling=bool(
                dataset_table.get("online_temporal_resampling", False)
            ),
            partial_recovery=bool(dataset_table.get("partial_recovery", True)),
            mask_extension=str(dataset_table.get("mask_extension", ".mask.pt")),
            caption_format=caption_format,
            enable_bucketing=bool(dataset_table.get("enable_bucketing", False)),
            bucket_resolutions=[str(v) for v in dataset_table.get("bucket_resolutions", [])],
            bucket_base_resolution=int(dataset_table.get("bucket_base_resolution", 512)),
            frame_buckets=[int(v) for v in dataset_table.get("frame_buckets", [33])],
            bucket_resize_mode=str(dataset_table.get("bucket_resize_mode", "resize_crop")),
            bucket_round_to=int(dataset_table.get("bucket_round_to", 16)),
            moe_routing=MoEDatasetRoutingConfig(
                specialization_mode=str(
                    moe_routing_table.get("specialization_mode", "emergent")
                ),
                domain_metadata_key=str(
                    moe_routing_table.get("domain_metadata_key", "")
                ),
                expert_affinity=expert_affinity,
                routing_prior_weight=float(
                    moe_routing_table.get("routing_prior_weight", 0.0)
                ),
                router_warmup_steps=int(
                    moe_routing_table.get("router_warmup_steps", 0)
                ),
            ),
        )
        sample_prompts_raw = logging_table.get("sample_prompts", [])
        if not isinstance(sample_prompts_raw, list):
            raise ConfigError("Expected array at 'logging.sample_prompts'.")
        sample_seeds_raw = logging_table.get("sample_seeds", [])
        if not isinstance(sample_seeds_raw, list):
            raise ConfigError("Expected array at 'logging.sample_seeds'.")
        logging_cfg = LoggingConfig(
            output_dir=output_dir,
            save_every_n_steps=int(logging_table.get("save_every_n_steps", 50)),
            async_checkpoint=bool(logging_table.get("async_checkpoint", False)),
            async_checkpoint_max_gib=float(
                logging_table.get("async_checkpoint_max_gib", 0.0)
            ),
            tensorboard=bool(logging_table.get("tensorboard", False)),
            sample_every_n_steps=int(logging_table.get("sample_every_n_steps", 0)),
            sample_prompt=str(logging_table.get("sample_prompt", "a cat walking")),
            sample_seed=int(logging_table.get("sample_seed", 42)),
            sample_prompts=[str(v) for v in sample_prompts_raw],
            sample_seeds=[int(v) for v in sample_seeds_raw],
            sample_blocks_to_swap=int(logging_table.get("sample_blocks_to_swap", -1)),
            sample_cfg_scale=float(logging_table.get("sample_cfg_scale", 7.5)),
            sample_negative_prompt=str(logging_table.get("sample_negative_prompt", "")),
            sample_solver=str(logging_table.get("sample_solver", "euler")),
            sample_resolution=str(logging_table.get("sample_resolution", "smallest_bucket")),
            sample_frame_count=int(logging_table.get("sample_frame_count", 16)),
            wandb=bool(logging_table.get("wandb", False)),
            wandb_project=str(logging_table.get("wandb_project", "mirai")),
            wandb_run_name=str(logging_table.get("wandb_run_name", "")),
        )
        compliance_cfg = ComplianceConfig(
            enabled=bool(compliance_table.get("enabled", False)),
            require_provenance=bool(compliance_table.get("require_provenance", True)),
            require_rights_attestation=bool(
                compliance_table.get("require_rights_attestation", True)
            ),
        )
        # Resolve the canonical frozen-weight quantization selector.
        _fwq_raw = str(memory_table.get("frozen_weight_quantization", "none")).strip().lower()
        _fwq_allowed = {
            "",
            "none",
            "fp8",
            "int8",
            "nf4",
            "gguf_iq4",
            "gguf_iq3",
            "mxfp8_e4m3",
            "mxfp4",
            "nvfp4",
        }
        if _fwq_raw not in _fwq_allowed:
            raise ValueError(
                "memory.frozen_weight_quantization must be one of: "
                + ", ".join(sorted(v for v in _fwq_allowed if v))
                + f"; got {_fwq_raw!r}."
            )
        _refiner_fwq_raw = str(
            memory_table.get("refiner_frozen_weight_quantization", "")
        ).strip().lower()
        if _refiner_fwq_raw not in _fwq_allowed:
            raise ValueError(
                "memory.refiner_frozen_weight_quantization must be empty or one of: "
                + ", ".join(sorted(v for v in _fwq_allowed if v))
                + f"; got {_refiner_fwq_raw!r}."
            )
        _fwq_strategy = str(
            memory_table.get("frozen_weight_quantization_strategy", "disabled")
        ).strip().lower()
        memory_cfg = MemoryConfig(
            hardware_policy=str(memory_table.get("hardware_policy", "disabled")),
            frozen_weight_quantization=_fwq_raw,
            frozen_weight_quantization_strategy=_fwq_strategy,
            refiner_frozen_weight_quantization=_refiner_fwq_raw,
            frozen_weight_packed_state_path=frozen_weight_packed_state_path,
            expert_precision_plan_path=expert_precision_plan_path,
            expert_structured_sparsity=str(
                memory_table.get("expert_structured_sparsity", "disabled")
            ),
            quantization_block_size=int(memory_table.get("quantization_block_size", 64)),
            weight_residency_strategy=str(
                memory_table.get("weight_residency_strategy", "disabled")
            ),
            expert_weight_access=str(memory_table.get("expert_weight_access", "auto")),
            expert_dequant_chunk_size=int(
                memory_table.get("expert_dequant_chunk_size", 0)
            ),
            expert_device_cache_gib=float(
                memory_table.get("expert_device_cache_gib", 0.0)
            ),
            device_residency_budget_gib=float(
                memory_table.get("device_residency_budget_gib", 0.0)
            ),
            quantize_experts_on_load=bool(
                memory_table.get("quantize_experts_on_load", False)
            ),
            router_quantization=str(memory_table.get("router_quantization", "disabled")),
            router_quantization_calibration_path=_validate_no_control_path(
                "memory.router_quantization_calibration_path",
                str(
                    memory_table.get(
                        "router_quantization_calibration_path",
                        "",
                    )
                ),
            ),
            moe_kernel_backend=str(memory_table.get("moe_kernel_backend", "auto")),
            cuda_memory_fraction=float(memory_table.get("cuda_memory_fraction", 1.0)),
            minimum_system_memory_gib=float(
                memory_table.get("minimum_system_memory_gib", 0.0)
            ),
            max_pinned_host_gib=float(
                memory_table.get("max_pinned_host_gib", 12.0)
            ),
            trainable_parameter_offload=bool(
                memory_table.get("trainable_parameter_offload", False)
            ),
            packed_state_preload=str(
                memory_table.get("packed_state_preload", "pinned")
            ),
            packed_stream_cache_gib=float(
                memory_table.get("packed_stream_cache_gib", 0.0)
            ),
            packed_stream_backend=str(
                memory_table.get("packed_stream_backend", "staged")
            ),
            packed_stream_prefetch_depth=int(memory_table.get("packed_stream_prefetch_depth", 0)),
            moe_dispatch=str(memory_table.get("moe_dispatch", "vectorized")),
            moe_dispatch_preprocess=str(
                memory_table.get("moe_dispatch_preprocess", "host")
            ),
            moe_gemm_backend=str(memory_table.get("moe_gemm_backend", "auto")),
            moe_gemm_backend_forward=str(
                memory_table.get("moe_gemm_backend_forward", "")
            ),
            moe_gemm_backend_dx=str(memory_table.get("moe_gemm_backend_dx", "")),
            moe_gemm_backend_dw=str(memory_table.get("moe_gemm_backend_dw", "")),
            moe_batched_dequant=bool(
                memory_table.get("moe_batched_dequant", True)
            ),
            moe_pair_dequant=bool(memory_table.get("moe_pair_dequant", True)),
            moe_batched_gather=bool(
                memory_table.get("moe_batched_gather", False)
            ),
            moe_expert_autograd=str(
                memory_table.get("moe_expert_autograd", "standard")
            ),
            moe_activation_backend=str(
                memory_table.get("moe_activation_backend", "torch")
            ),
            packed_shard_size_mb=int(
                memory_table.get("packed_shard_size_mb", 2048)
            ),
            int8_workspace_mb=int(memory_table.get("int8_workspace_mb", 0)),
            block_residency_planner=str(
                memory_table.get("block_residency_planner", "uniform")
            ),
            block_swap_prefetch_depth=int(
                memory_table.get("block_swap_prefetch_depth", 1)
            ),
            block_residency_priority=str(
                memory_table.get("block_residency_priority", "index")
            ),
            block_swap_transfer_strategy=str(
                memory_table.get("block_swap_transfer_strategy", "per_tensor")
            ),
        )
        if (
            str(model.params.router_repair_artifact_path).strip()
            and not str(memory_cfg.frozen_weight_packed_state_path).strip()
        ):
            raise ConfigError(
                "model.params.router_repair_artifact_path requires "
                "memory.frozen_weight_packed_state_path."
            )
        _router_quantization = str(memory_cfg.router_quantization).strip().lower()
        _router_calibration_path = str(
            memory_cfg.router_quantization_calibration_path
        ).strip()
        if _router_calibration_path and _router_quantization not in {
            "int8",
            "int8_per_channel",
            "per_channel_int8",
        }:
            raise ConfigError(
                "memory.router_quantization_calibration_path requires "
                "memory.router_quantization='int8_per_channel'."
            )
        if (
            str(model.params.router_quantization_calibration).strip().lower()
            == "eaquant"
            and (
                _router_quantization not in {"", "disabled", "none", "off"}
                or _router_calibration_path
            )
        ):
            raise ConfigError(
                "EAQuant evidence collection requires floating-point source "
                "routers: set memory.router_quantization='disabled' and leave "
                "memory.router_quantization_calibration_path empty."
            )
        _expert_precision_path = str(
            memory_cfg.expert_precision_plan_path
        ).strip()
        _frozen_quantization = str(
            memory_cfg.frozen_weight_quantization
        ).strip().lower()
        if _expert_precision_path and _frozen_quantization in {
            "",
            "none",
            "off",
            "disabled",
        }:
            raise ConfigError(
                "memory.expert_precision_plan_path requires frozen expert "
                "quantization."
            )
        if (
            str(model.params.expert_precision_calibration).strip().lower()
            == "imatrix"
            and (
                _frozen_quantization not in {"", "none", "off", "disabled"}
                or str(memory_cfg.frozen_weight_packed_state_path).strip()
                or _expert_precision_path
            )
        ):
            raise ConfigError(
                "Imatrix precision calibration requires floating-point source "
                "experts: set memory.frozen_weight_quantization='none' and "
                "leave packed-state and precision-plan paths empty."
            )
        if str(model.params.expert_upcycling).strip().lower() == "drop":
            if not str(memory_cfg.frozen_weight_packed_state_path).strip():
                raise ConfigError(
                    "expert_upcycling='drop' requires "
                    "memory.frozen_weight_packed_state_path."
                )
            if str(adapter.type).strip().lower() != "lora":
                raise ConfigError(
                    "expert_upcycling='drop' requires adapter.type='lora'."
                )
            if adapter.train_router is not True:
                raise ConfigError(
                    "expert_upcycling='drop' requires adapter.train_router=true so "
                    "the newly initialized router rows can adapt."
                )
            if str(adapter.target_preset).strip() not in {
                "routed_experts_only",
                "attn_routed_experts",
                "attn_router_routed_experts",
            }:
                raise ConfigError(
                    "expert_upcycling='drop' requires an adapter.target_preset "
                    "that includes routed experts."
                )
            if str(model.params.expert_pruning).strip().lower() != "off":
                raise ConfigError(
                    "expert_upcycling='drop' cannot be composed with structural "
                    "expert pruning in the same artifact transform."
                )
            if str(model.params.expert_consolidation).strip().lower() != "off":
                raise ConfigError(
                    "expert_upcycling='drop' cannot be composed with expert "
                    "consolidation in the same artifact transform."
                )
            if str(model.params.expert_weight_compression).strip().lower() != "off":
                raise ConfigError(
                    "expert_upcycling='drop' requires regular packed expert "
                    "weights, not a physical-weight provider artifact."
                )
        return cls(
            preset=payload.get("preset"),
            model=model,
            strategy=strategy,
            inference=InferenceConfig(
                task=inference_task,
                denoising_strength=denoising_strength,
                prompt_rewriter=prompt_rewriter,
                cfg_mode=cfg_mode,
                keep_text_encoder_resident=bool(
                    inference_table.get("keep_text_encoder_resident", False)
                ),
                stage_text_encoder_before_denoiser=bool(
                    inference_table.get("stage_text_encoder_before_denoiser", False)
                ),
                text_encoder_weight_quantization=(
                    inference_text_encoder_quantization
                ),
                keep_vae_resident=bool(
                    inference_table.get("keep_vae_resident", False)
                ),
                vae_tiling=bool(inference_table.get("vae_tiling", False)),
                vae_tile_size=inference_vae_tile_size,
                vae_tile_stride=inference_vae_tile_stride,
                expert_feature_cache=expert_feature_cache_policy.mode,
                expert_feature_cache_drift_threshold=(
                    expert_feature_cache_policy.drift_threshold
                ),
                expert_feature_cache_max_reuse_span=(
                    expert_feature_cache_policy.max_reuse_span
                ),
                expert_feature_cache_slots=expert_feature_cache_policy.slots,
                moe_token_chunk_size=inference_moe_token_chunk_size,
                blocks_to_swap=inference_blocks_to_swap,
                block_swap_mode=inference_block_swap_mode,
                refiner_blocks_to_swap=inference_refiner_blocks_to_swap,
                refiner_block_swap_mode=inference_refiner_block_swap_mode,
            ),
            training=training,
            optimizer=optimizer,
            adapter=adapter,
            dataset=dataset,
            logging=logging_cfg,
            compliance=compliance_cfg,
            memory=memory_cfg,
        )


# Nested-dataclass fields that are TOML tables of their own, not leaf config
# keys: they are documented as their own sections, so they are excluded from the
# leaf-key set of their parent table.
_SUBTABLE_FIELDS: dict[str, set[str]] = {
    "": {
        "model",
        "strategy",
        "inference",
        "training",
        "optimizer",
        "adapter",
        "dataset",
        "logging",
        "compliance",
        "memory",
    },
    "model": {"params"},
    "dataset": {"moe_routing"},
}


def all_config_keys() -> dict[str, set[str]]:
    """Authoritative leaf config keys per TOML table, keyed by dotted table path.

    The empty string key ``""`` holds root-level keys (``preset``). Nested
    dataclass container fields (``model.params``, ``dataset.moe_routing``) are
    reported under their own dotted section and stripped from the parent's leaf
    set. This is the single source of truth the CONFIG_REFERENCE.md coverage
    test asserts against; it is derived directly from the schema dataclasses so
    it can never drift from what the loader actually accepts.
    """

    def _leaves(section: str, dc: type) -> set[str]:
        return {f.name for f in fields(dc)} - _SUBTABLE_FIELDS.get(section, set())

    return {
        "": _leaves("", TrainingConfig),
        "model": _leaves("model", ModelConfig),
        "model.params": _leaves("model.params", ModelParams),
        "strategy": _leaves("strategy", StrategyConfig),
        "inference": _leaves("inference", InferenceConfig),
        "training": _leaves("training", TrainingSection),
        "optimizer": _leaves("optimizer", OptimizerConfig),
        "adapter": _leaves("adapter", AdapterConfig),
        "dataset": _leaves("dataset", DatasetConfig),
        "dataset.moe_routing": _leaves("dataset.moe_routing", MoEDatasetRoutingConfig),
        "logging": _leaves("logging", LoggingConfig),
        "compliance": _leaves("compliance", ComplianceConfig),
        "memory": _leaves("memory", MemoryConfig),
    }
