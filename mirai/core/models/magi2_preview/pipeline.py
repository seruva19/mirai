"""One-device adapter training and inference runtime for MAGI-2 Preview.

The architecture and checkpoint naming follow SandAI's Apache-2.0 reference
implementation vendored under :mod:`mirai.vendors.magi2_preview`. Heavy runtime
imports remain lazy so the rest of Mirai is usable without them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize

from mirai.core.inference.cfg_schedule import CfgSchedule
from mirai.core.inference.diffusion_cache import DiffusionPredictionCache

from mirai.core.models.base import (
    MemoryFeatureCapabilities,
    ModelExtensionCapabilities,
    SyntheticBatchSpec,
)
from mirai.core.models.magi2_preview.quantized_experts import (
    MAGI2_PACKED_EXPERT_FORMATS,
    MAGI2_ROUTED_EXPERT_TENSORS,
    Magi2Nf4ExpertStore,
    iter_magi2_moe_layers,
    magi2_expert_store,
)
from mirai.core.models.magi2_preview.refiner import (
    MAGI2_REFINER_OUTPUT_FPS,
    MAGI2_REFINER_SUBFOLDER,
    Magi2RefineSettings,
)
from mirai.core.models.magi2_preview.refiner_attention import (
    MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT,
    Magi2RefinerAttentionUnsupported,
    normalize_refiner_attention_backend,
)
from mirai.core.models.native_video import NativeVideoPipeline, VideoLatentLayout
from mirai.core.models.providers import (
    FamilyGenerationDefaults,
    ModelFamilyProvider,
    NativeCacheEncoderConfig,
    register_model_family_provider,
)
from mirai.core.moe.routing.contracts import SparseMoECapabilities
from mirai.core.moe.runtime.specs import (
    ExpertTensorSpec,
    MoEOptimizationPolicy,
    validate_expert_tensor_specs,
)
from mirai.core.moe.runtime.expert_transfer_profile import (
    load_expert_transfer_profile,
    validate_expert_transfer_profile_identity,
)
from mirai.core.training.residency.block_swap import BlockSwapManager, RoutedSwapRegion
from mirai.core.training.residency.device_placement import WeightResidencyExecutionMode
from mirai.core.training.residency.tensor_residency import (
    move_tensors_outside_modules,
    move_trainable_tensors,
)
from mirai.core.training.runtime.compilation import CompilationRegion


# Hidden width of the Qwen3.5 text encoder shipped with the MAGI-2 snapshot;
# ``ModelConfig.text_in_channels`` in the released architecture JSON.
MAGI2_TEXT_EMBED_WIDTH = 5120

# The vendored ``TransformerBlock`` carries a single whole-block recompute flag,
# so the family implements only the "off" family and "standard".
MAGI2_GRADIENT_CHECKPOINTING_OFF = frozenset({"off", "false", "none", "0", ""})
MAGI2_GRADIENT_CHECKPOINTING_ON = frozenset({"standard", "true", "1"})

# Flow-UniPC defaults to a single batched conditional/unconditional forward.
MAGI2_NATIVE_SOLVER = "unipc"
MAGI2_NATIVE_CFG_MODE = "batched"
MAGI2_RELEASED_PREVIEW_WIDTH = 896
MAGI2_RELEASED_PREVIEW_HEIGHT = 512
MAGI2_RELEASED_FRAMES = 249
MAGI2_RELEASED_PREVIEW_STEPS = 100
MAGI2_RELEASED_PREVIEW_CFG_SCALE = 5.0
# A Turbo VAE middle window carries two context latents in addition to its
# configured step. Bound that complete window from the latent geometry while
# preserving the released 7/7 schedule for the preview grid and smaller inputs.
MAGI2_VAE_MAX_WINDOW_VOLUME = 32_640

# The audio placeholder is drawn from a family-owned stream rather than from the
# latent-noise stream the training loop seeds, so the two draws stay independent
# of each other's shapes and call counts.
MAGI2_AUDIO_NOISE_SEED_OFFSET = 20_011


def resolve_magi2_audio_tokens(configured: int, latent_frames: int) -> int:
    return (latent_frames - 1) * 8 + 1 if configured < 0 else configured

# The preview transformer is positioned on a 25 fps timeline sampled at
# temporal stride 8 (``vae_stride[0] = 8``, ``time_pos_fps = 25 / 8 = 3.125``),
# so a frame request is expressed on that 25 fps timeline. The preview-only
# decode is half-rate: the Turbo VAE expands T latent frames into 4 * (T - 1) +
# 1 physical frames, which play back as the requested duration at 12.5 fps.
# Upstream reaches 25 fps by running the refiner, which interpolates the latent
# to 2T - 1 frames before its own denoise; that stage is not part of this
# preview-only path.
MAGI2_REQUEST_FPS = 25.0
MAGI2_NATIVE_OUTPUT_FPS = 12.5

# Generation envelope of the single-GPU sampling path, in latent frames. T=32
# is the horizon the preview model is trained for: it is exactly the ten
# seconds upstream resolves for its own default request. Larger T is refiner
# space rather than a longer preview - the refiner, not a longer preview,
# produces T=63 - and sampling the preview there collapses to flat output. The
# floor is empirical: T=5 is observed to degenerate and the intermediate
# lengths were not probed. Both bounds constrain generation only; training
# forwards accept any representable latent length.
MAGI2_MIN_SAMPLING_LATENT_FRAMES = 8
MAGI2_MAX_SAMPLING_LATENT_FRAMES = 32

# Frozen-weight quantization formats this family implements. Only the routed
# expert stack is packed; the router, expert bias, shared experts, hyper
# connections, and norms keep their released dtype.
MAGI2_FROZEN_WEIGHT_QUANTIZATION_FORMATS = MAGI2_PACKED_EXPERT_FORMATS
MAGI2_NF4_BLOCKSIZE = 64


def _magi2_request_frames(latent_frames: int) -> int:
    """Requested frames on the 25 fps timeline for ``latent_frames`` latents."""
    return 8 * (int(latent_frames) - 1) + 1


def _magi2_decoded_frames(latent_frames: int) -> int:
    """Physical frames the Turbo VAE decodes from ``latent_frames`` latents."""
    return 4 * (int(latent_frames) - 1) + 1


def resolve_magi2_vae_decode_chunk_size(
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    released_chunk_size: int,
    requested_chunk_size: int = 0,
) -> int:
    """Choose a deterministic temporal window that bounds decoder activations.

    The vendored decoder adds one context latent on each side of a middle
    window.  Its peak is therefore governed by ``H * W * (step + 2)`` rather
    than by decoded output size.  An explicit positive request wins; zero keeps
    the released window where it fits and shrinks only large spatial grids.
    """

    frames = int(latent_frames)
    height = int(latent_height)
    width = int(latent_width)
    released = int(released_chunk_size)
    requested = int(requested_chunk_size)
    if frames < 1 or height < 1 or width < 1:
        raise ValueError("MAGI-2 VAE latent dimensions must all be >= 1.")
    if released < 1:
        raise ValueError("MAGI-2 released VAE chunk size must be >= 1.")
    if requested < 0:
        raise ValueError("MAGI-2 requested VAE chunk size must be >= 0.")
    if requested > 0:
        return requested
    safe_step = max(
        1,
        MAGI2_VAE_MAX_WINDOW_VOLUME // (height * width) - 2,
    )
    return min(released, safe_step)


MAGI2_TARGET_PRESETS: dict[str, tuple[str, ...]] = {
    "attn_only": (".attention.linear_qkv", ".attention.linear_proj"),
    "attn_router": (
        ".attention.linear_qkv",
        ".attention.linear_proj",
        ".mlp.moe_mlp.gate",
    ),
}


class LowRankWeight(nn.Module):
    """Shape-preserving LoRA parametrization for MAGI-2 packed weights."""

    def __init__(self, shape: tuple[int, ...], *, rank: int, alpha: float) -> None:
        super().__init__()
        if len(shape) < 2:
            raise ValueError("MAGI-2 LoRA targets must have at least two dimensions.")
        self.shape = tuple(int(v) for v in shape)
        self.rows = int(torch.tensor(self.shape[:-1]).prod().item())
        self.cols = int(self.shape[-1])
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.runtime_scale = 1.0
        self.lora_a = nn.Parameter(torch.empty(self.rank, self.cols))
        self.lora_b = nn.Parameter(torch.zeros(self.rows, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        if base.numel() == 0:
            raise RuntimeError("Packed MAGI-2 adapters use activation-space execution.")
        delta = (self.lora_b @ self.lora_a).reshape(self.shape)
        return base + delta.to(dtype=base.dtype) * self.scale * self.runtime_scale


@dataclass(frozen=True)
class Magi2RuntimeOptions:
    config_path: str
    audio_tokens: int = -1
    refiner_config_path: str = ""
    refiner_subfolder: str = MAGI2_REFINER_SUBFOLDER
    refiner_attention_backend: str = MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT
    refiner_query_chunk_size: int = 0
    refiner_weight_quantization: str = "none"
    refiner_block_swap_mode: str = "sync"
    vae_decode_chunk_size: int = 0
    vae_decode_tile_size: int = 0
    vae_decode_tile_overlap: int = 4


@dataclass(frozen=True)
class Magi2ResidencyRequest:
    """The block-residency policy the family was configured with.

    Captured at configuration time so the refiner stage can stream its own
    layers under exactly the policy the preview transformer runs under, rather
    than re-deriving one from config it does not own.
    """

    enabled: bool
    blocks_to_swap: int
    mode: str
    block_residency_planner: str
    block_swap_prefetch_depth: int
    block_residency_priority: str
    block_swap_transfer_strategy: str
    offload_dir: str | None


class Magi2PreviewPipeline(nn.Module, NativeVideoPipeline):
    """Native MAGI-2 denoiser with host-resident block streaming."""

    supports_native_cfg_schedule = True
    supports_native_diffusion_cache = True
    # Class-level defaults so every policy seam answers identically before
    # __init__ has run, which is how the weightless contract probes reach them.
    _refiner: Any | None = None
    _refine_settings: "Magi2RefineSettings | None" = None
    _residency_request: "Magi2ResidencyRequest | None" = None
    _expert_stores: "dict[str, Magi2Nf4ExpertStore]" = {}
    _packed_companion_linears: tuple[nn.Module, ...] = ()
    _packed_norms: tuple[nn.Module, ...] = ()
    _companion_weight_quantization = "none"
    _router_int8_enabled = False
    _routing_collapse: Any | None = None
    _last_diagnostics: "dict[str, float]" = {}

    def __init__(
        self,
        model_config: Any,
        memory_config: Any | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        # The frozen-weight storage decision is read before the checkpoint is
        # touched: packing the routed experts is only a host-memory saving if the
        # released BF16 stack is never materialized in the first place.
        self._frozen_weight_quantization = str(
            getattr(memory_config, "frozen_weight_quantization", "none") or "none"
        ).strip().lower()
        self._quantize_experts_on_load = bool(
            getattr(memory_config, "quantize_experts_on_load", False)
        )
        self._packed_expert_state_path = str(
            getattr(memory_config, "frozen_weight_packed_state_path", "") or ""
        ).strip()
        self._expert_precision_plan_path = str(getattr(memory_config, "expert_precision_plan_path", "") or "").strip()
        if self._expert_precision_plan_path and self._packed_expert_state_path:
            raise ValueError("MAGI-2 precision plans require a source checkpoint without a packed expert artifact.")
        self._nf4_blocksize = int(
            getattr(memory_config, "quantization_block_size", MAGI2_NF4_BLOCKSIZE)
        )
        self._expert_stores: dict[str, Magi2Nf4ExpertStore] = {}
        self._packed_companion_linears: tuple[nn.Module, ...] = ()
        self._applied_expert_precision_plan = ""
        self._moe_optimization_policy = MoEOptimizationPolicy()
        self._moe_token_chunk_policy: Any | None = None
        # Armed only by inference.expert_feature_cache; None keeps the MoE seam
        # exactly as the memory policy resolved it.
        self._expert_feature_cache: Any = None
        family = dict(getattr(model_config.params, "family_params", {}) or {})
        from mirai.core.models.magi2_preview.mixed_experts import MAGI2_EXPERT_ROTATION_SIZES

        self._expert_rotation_group_size = family.get("expert_rotation_group_size", 0)
        from mirai.core.models.magi2_preview.companion_weights import MAGI2_COMPANION_FORMATS, MAGI2_NORM_FORMATS

        self._companion_weight_quantization = family.get("companion_weight_quantization", "none")
        if self._companion_weight_quantization not in MAGI2_COMPANION_FORMATS:
            raise ValueError("MAGI-2 companion_weight_quantization must be 'none', 'fp8', or 'nf4'.")
        self._norm_weight_quantization = family.get("norm_weight_quantization", "none")
        if self._norm_weight_quantization not in MAGI2_NORM_FORMATS:
            raise ValueError("MAGI-2 norm_weight_quantization must be 'none' or 'int8'.")
        if type(self._expert_rotation_group_size) is not int or self._expert_rotation_group_size not in MAGI2_EXPERT_ROTATION_SIZES:
            raise ValueError("Invalid MAGI-2 expert Hadamard group size.")
        if self._expert_rotation_group_size and not (
            self._packed_expert_state_path
            or self._expert_precision_plan_path
            or str(getattr(model_config.params, "expert_precision_calibration", "off")).strip().lower() in {"imatrix", "alphaq"}
        ):
            raise ValueError("MAGI-2 expert rotation requires precision calibration, a mixed plan, or a rotated artifact.")
        self._expert_pruning_plan_path = str(family.get("expert_pruning_plan_path", "")).strip()
        self._expert_consolidation_plan_path = str(family.get("expert_consolidation_plan_path", "")).strip()
        self._expert_shared_basis_rank = family.get("expert_shared_basis_rank", 0)
        if type(self._expert_shared_basis_rank) is not int or self._expert_shared_basis_rank < 0:
            raise ValueError("MAGI-2 expert_shared_basis_rank must be a non-negative integer.")
        self._expert_pruning_adapter_source_plan_path = str(
            family.get("expert_pruning_adapter_source_plan_path", "") or ""
        ).strip()
        if self._expert_pruning_adapter_source_plan_path and self._expert_pruning_plan_path:
            raise ValueError("Choose either pruning the MAGI-2 base or restoring a pruned adapter to the source base.")
        if self._expert_pruning_adapter_source_plan_path and (
            self._quantize_experts_on_load or self._packed_expert_state_path
        ):
            raise ValueError("MAGI-2 pruned-adapter source verification requires the dense checkpoint.")
        if self._expert_pruning_plan_path and self._expert_precision_plan_path and not self._quantizes_experts_on_load():
            raise ValueError("MAGI-2 pruning with a mixed precision plan requires quantize_experts_on_load.")
        if self._expert_consolidation_plan_path and (
            self._expert_pruning_plan_path or self._expert_pruning_adapter_source_plan_path
            or self._expert_precision_plan_path
            or (
                not self._quantizes_experts_on_load() and not self._packed_expert_state_path
                and self._frozen_weight_quantization not in {"", "none", "off", "disabled"}
            )
        ):
            raise ValueError("MAGI-2 hierarchical consolidation requires a dense source and no pruning plan.")
        if self._expert_shared_basis_rank and (
            self._expert_pruning_plan_path or self._expert_pruning_adapter_source_plan_path
            or self._expert_consolidation_plan_path or self._expert_precision_plan_path
            or self._quantize_experts_on_load
            or self._frozen_weight_quantization not in {"", "none", "off", "disabled"}
        ):
            raise ValueError("MAGI-2 shared-basis rank requires an original dense checkpoint and no other expert storage policy.")
        default_config = (
            Path(__file__).resolve().parents[3]
            / "vendors"
            / "magi2_preview"
            / "configs"
            / "magi2_preview.json"
        )
        default_refiner_config = default_config.with_name("magi2_refiner.json")
        self.options = Magi2RuntimeOptions(
            config_path=str(family.get("config_path") or default_config),
            audio_tokens=int(family.get("audio_tokens", -1)),
            refiner_config_path=str(
                family.get("refiner_config_path") or default_refiner_config
            ),
            refiner_subfolder=str(
                family.get("refiner_subfolder") or MAGI2_REFINER_SUBFOLDER
            ),
            refiner_attention_backend=normalize_refiner_attention_backend(
                family.get("refiner_attention_backend")
            ),
            refiner_query_chunk_size=int(family.get("refiner_query_chunk_size", 0)),
            refiner_weight_quantization=str(family.get("refiner_weight_quantization", "none")).strip().lower(),
            refiner_block_swap_mode=str(
                family.get("refiner_block_swap_mode", "sync") or "sync"
            ).strip().lower(),
            vae_decode_chunk_size=int(family.get("vae_decode_chunk_size", 0) or 0),
            vae_decode_tile_size=int(family.get("vae_decode_tile_size", 0)),
            vae_decode_tile_overlap=int(family.get("vae_decode_tile_overlap", 4)),
        )
        self.runtime_config, self.transformer, self.data_proxy = self._build_model()
        # Routing telemetry is opt-in: with the gate off no observer, tracker, or
        # tap exists and the monitoring module is never imported.
        self._routing_collapse: Any | None = None
        self._last_diagnostics: dict[str, float] = {}
        if bool(getattr(model_config.params, "moe_routing_health", False)):
            self._enable_routing_collapse_telemetry()
        self._block_swap_manager: BlockSwapManager | None = None
        self._weight_residency_execution_mode = WeightResidencyExecutionMode.TRAINING
        self._block_hook_handles: list[Any] = []
        self._gradient_checkpointing = "off"
        self._adapter_configured = False
        self._last_audio_prediction: torch.Tensor | None = None
        self._compute_autocast_dtype = torch.bfloat16
        self._lora_scale = 1.0
        self._text_encoder: Any | None = None
        self._text_encoder_weight_quantization = "none"
        self._vae: nn.Module | None = None
        self._last_vae_decode_chunk_size = 0
        self._inference_device = "cpu"
        self._audio_noise_generator = self._build_audio_noise_generator(seed)
        self._refiner_generator_state: torch.Tensor | None = None
        self._refiner_context_pair: tuple[torch.Tensor, torch.Tensor] | None = None
        # Refiner state stays absent until a refinement request is validated.
        self._refiner: Any | None = None
        self._refine_settings: Magi2RefineSettings | None = None
        self._residency_request: Magi2ResidencyRequest | None = None

    @staticmethod
    def _build_audio_noise_generator(seed: int | None) -> torch.Generator:
        """CPU generator owning the audio placeholder stream of one run."""
        generator = torch.Generator()
        generator.manual_seed(
            int(0 if seed is None else seed) + MAGI2_AUDIO_NOISE_SEED_OFFSET
        )
        return generator

    def refiner_noise_generator(
        self, *, seed: int, device: torch.device
    ) -> torch.Generator:
        """Continue the release pipeline's per-sample RNG stream.

        SandAI draws preview video noise, preview audio noise, and refiner noise
        from one CUDA generator in that order. The generic inference loop owns
        the generator, so the family snapshots its state after the preview draws
        and restores it here. Direct refiner calls without a preceding preview
        retain their deterministic seed fallback.
        """
        generator = torch.Generator(device=device)
        if self._refiner_generator_state is None:
            generator.manual_seed(int(seed))
        else:
            generator.set_state(self._refiner_generator_state)
        return generator

    @classmethod
    def from_training_config(cls, config: Any) -> "Magi2PreviewPipeline":
        return cls(config.model, config.memory, seed=int(config.training.seed))

    def _build_model(self) -> tuple[Any, nn.Module, Any]:
        try:
            from mirai.vendors.magi2_preview.common.magi2_config import load_config
            from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import (
                load_magi2_model_state_dict,
            )
            from mirai.vendors.magi2_preview.model.magi2_preview import Transformer
            from mirai.vendors.magi2_preview.pipeline.preview_data_proxy import Magi2DataProxy
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "MAGI-2 execution requires Torch, Triton, pydantic-settings, and "
                "unfoldNd. Optional MagiAttention, MagiCompiler, and FlashAttention "
                "packages enable the accelerated inference path."
            ) from exc
        config = load_config(self.options.config_path)
        config.engine_config.cp_size = 1
        config.engine_config.ep_size = 1
        model_path = Path(str(self.model_config.path)).expanduser()
        checkpoint_dir = model_path / "preview" if (model_path / "preview").is_dir() else model_path
        config.engine_config.load = str(checkpoint_dir)
        packed_experts = self._quantizes_experts_on_load() or bool(self._packed_expert_state_path)
        pruning_plan_path = self._expert_pruning_plan_path
        consolidation_plan_path = getattr(self, "_expert_consolidation_plan_path", "")
        shared_basis_rank = getattr(self, "_expert_shared_basis_rank", 0)
        transformer = (
            Transformer(config.arch_config, ep_size=1, routed_weight_device="meta")
            if packed_experts or pruning_plan_path or consolidation_plan_path or shared_basis_rank
            else Transformer(config.arch_config, ep_size=1)
        )
        mixed_stream = self._quantizes_experts_on_load() and self._expert_precision_plan_path
        preloaded_keys = set()
        pruning_plan = None
        if pruning_plan_path:
            from mirai.core.models.magi2_preview.expert_pruning import Magi2ExpertPruningPlan

            pruning_plan = Magi2ExpertPruningPlan.load(pruning_plan_path)
        if shared_basis_rank:
            from mirai.core.models.magi2_preview.basis_experts import fingerprint_magi2_basis_checkpoint

            self._expert_basis_source_sha256 = fingerprint_magi2_basis_checkpoint(str(checkpoint_dir))
            if self._packed_expert_state_path:
                self._restore_packed_experts(transformer, shared_basis_rank=shared_basis_rank)
            else:
                from mirai.core.models.magi2_preview.basis_experts import stream_factorize_magi2_shared_basis_experts

                self._expert_stores = stream_factorize_magi2_shared_basis_experts(
                    transformer, checkpoint_dir=str(checkpoint_dir), rank=shared_basis_rank,
                )
        elif consolidation_plan_path:
            from mirai.core.models.magi2_preview.expert_pruning import (
                Magi2ExpertConsolidationPlan,
                stream_merge_magi2_experts,
            )

            consolidation_plan = Magi2ExpertConsolidationPlan.load(consolidation_plan_path)
            restoring_packed = bool(self._packed_expert_state_path)
            preloaded_keys, self._expert_stores = stream_merge_magi2_experts(
                transformer, consolidation_plan, checkpoint_dir,
                quant_format=(
                    self._frozen_weight_quantization
                    if not restoring_packed and self._quantizes_experts_on_load() else None
                ),
                blocksize=self._nf4_blocksize,
                prepare_packed_restore=restoring_packed,
            )
            self._expert_pruning_digest = consolidation_plan.topology_digest()
            if restoring_packed:
                self._restore_packed_experts(transformer, consolidation_digest=self._expert_pruning_digest)
        elif mixed_stream:
            from mirai.core.models.magi2_preview.mixed_experts import stream_quantize_magi2_mixed_experts

            self._expert_stores, preloaded_keys = stream_quantize_magi2_mixed_experts(
                transformer, checkpoint_dir=str(checkpoint_dir), plan_path=self._expert_precision_plan_path,
                base_format=self._frozen_weight_quantization, blocksize=self._nf4_blocksize,
                rotation_group_size=self._expert_rotation_group_size, pruning_plan=pruning_plan,
            )
            self._applied_expert_precision_plan = self._expert_precision_plan_path
        elif pruning_plan is not None:
            from mirai.core.models.magi2_preview.expert_pruning import stream_prune_magi2_experts

            restoring_packed = bool(self._packed_expert_state_path)
            preloaded_keys, self._expert_stores = stream_prune_magi2_experts(
                transformer, pruning_plan, checkpoint_dir,
                quant_format=(
                    self._frozen_weight_quantization
                    if not restoring_packed and self._quantizes_experts_on_load() else None
                ),
                blocksize=getattr(self, "_nf4_blocksize", MAGI2_NF4_BLOCKSIZE),
                prepare_packed_restore=restoring_packed,
            )
            if restoring_packed:
                from mirai.core.models.magi2_preview.quantized_experts import install_magi2_nf4_expert_stores

                install_magi2_nf4_expert_stores(
                    transformer, blocksize=self._nf4_blocksize,
                    quant_format=self._frozen_weight_quantization,
                )
        elif packed_experts:
            # Remove the dense routed expert targets before reading checkpoint shards.
            from mirai.core.models.magi2_preview.quantized_experts import install_magi2_nf4_expert_stores

            install_magi2_nf4_expert_stores(
                transformer, blocksize=self._nf4_blocksize,
                quant_format=self._frozen_weight_quantization,
            )
        if pruning_plan is not None:
            self._expert_pruning_digest = pruning_plan.topology_digest()
        if self._companion_weight_quantization != "none":
            from mirai.core.models.magi2_preview.companion_weights import stream_pack_magi2_companion_weights

            self._packed_companion_linears, companion_keys = stream_pack_magi2_companion_weights(
                transformer, checkpoint_dir=str(checkpoint_dir),
                quant_format=self._companion_weight_quantization, blocksize=self._nf4_blocksize,
            )
            preloaded_keys.update(companion_keys)
        load_options = {"preloaded_keys": preloaded_keys} if preloaded_keys else {}
        state = load_magi2_model_state_dict(transformer, config.engine_config, **load_options)
        transformer.load_state_dict(state, strict=True)
        del state
        if getattr(self, "_norm_weight_quantization", "none") == "int8":
            from mirai.core.models.magi2_preview.companion_weights import pack_magi2_norm_weights

            self._packed_norms = pack_magi2_norm_weights(transformer)
        if (
            packed_experts and not mixed_stream and not consolidation_plan_path and not shared_basis_rank
            and (not pruning_plan_path or self._packed_expert_state_path)
        ):
            if self._packed_expert_state_path:
                self._restore_packed_experts(
                    transformer, pruning_digest=self._expert_pruning_digest if pruning_plan_path else None,
                )
            else:
                from mirai.core.models.magi2_preview.quantized_experts import stream_quantize_magi2_experts

                self._expert_stores = stream_quantize_magi2_experts(
                    transformer,
                    checkpoint_dir=str(checkpoint_dir),
                    blocksize=self._nf4_blocksize,
                    quant_format=self._frozen_weight_quantization,
                )
        elif getattr(self, "_expert_pruning_adapter_source_plan_path", ""):
            from mirai.core.models.magi2_preview.expert_pruning import (
                Magi2ExpertPruningPlan,
                verify_magi2_pruning_source,
            )

            source_plan = Magi2ExpertPruningPlan.load(self._expert_pruning_adapter_source_plan_path)
            self._expert_pruning_digest = verify_magi2_pruning_source(transformer, source_plan)
        transformer.train()
        self._configure_attention_backend(transformer)
        return config, transformer, Magi2DataProxy(config.evaluation_config.data_proxy_config)

    def _configure_attention_backend(self, transformer: nn.Module) -> None:
        """Select the attention execution path declared by ``model.attention_backend``.

        Only ``flex`` has a MAGI-2 implementation, because the family's
        attention carries per-head sink logits the shared SDPA and
        FlashAttention backends do not model. Every other value leaves the
        vendored dispatch in place.
        """
        from mirai.core.models.magi2_preview.flex_attention import (
            attach_flex_attention_backend,
            resolve_magi2_flex_attention,
            validate_flex_attention_support,
        )

        if str(getattr(self.model_config, "attention_backend", "auto")).strip().lower() in {"sage", "sage2"}:
            raise ValueError("MAGI-2 attention requires sink logits, which SageAttention does not support.")
        backend = resolve_magi2_flex_attention(self.model_config)
        if backend is not None:
            validate_flex_attention_support()
        attach_flex_attention_backend(transformer, backend)

    def get_video_latent_layout(self) -> VideoLatentLayout:
        """Latent geometry of the released MAGI-2 preview VAE pair.

        The transformer is positioned at temporal stride 8 on a 25 fps timeline,
        and the Wan2.2 encoder used at cache time compresses time by eight with
        a leading key frame, so ``T`` latent frames span ``8 * (T - 1) + 1``
        frames of that timeline and only counts congruent to 1 modulo 8 are
        representable. Requests are rejected rather than rounded: a rounded
        request produces a clip that is not the length the caller asked for.

        ``native_output_fps`` is 12.5 rather than 25 because the preview-only
        decode is half-rate: the Turbo VAE emits ``4 * (T - 1) + 1`` physical
        frames, which cover the requested duration only when played at 12.5 fps.
        Once a refinement request has been validated the layout reports 25
        instead: the refiner resamples the latent to ``2T - 1`` frames, which
        the same decoder expands to the full ``8 * (T - 1) + 1`` frames the
        request denotes. The rate is a property of the configured stage rather
        than of the request, so it is answered here rather than by the caller.
        """
        return VideoLatentLayout(
            latent_channels=48,
            temporal_downsample=8,
            spatial_downsample=16,
            layout="BCTHW",
            frame_count_modulus=8,
            frame_count_remainder=1,
            frame_count_rule="1 modulo 8 (8n+1)",
            request_spatial_multiple=16,
            native_output_fps=(
                MAGI2_REFINER_OUTPUT_FPS
                if self._refine_settings is not None
                else MAGI2_NATIVE_OUTPUT_FPS
            ),
        )

    def get_synthetic_batch_spec(self) -> SyntheticBatchSpec:
        """MAGI-2 rejects any batch outside its own conditioning contract.

        The forward pass requires ``[B, 48, T, H, W]`` latents and ``[B, S,
        5120]`` Qwen3.5 hidden states, so a synthetic batch carrying the generic
        placeholder widths cannot reach the denoiser.
        """
        return SyntheticBatchSpec(
            latent_channels=int(self.get_video_latent_layout().latent_channels),
            text_embed_width=MAGI2_TEXT_EMBED_WIDTH,
        )

    def preview_latent_geometry(
        self, *, frame_count: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        """Apply the 8n+1 layout rule and the sampling-path length envelope.

        The layout rule states what the VAE pair can represent; the envelope
        states what the preview-only sampling path can produce. A request
        outside it is rejected here rather than returning a clip the caller
        cannot use.
        """
        geometry = super().preview_latent_geometry(
            frame_count=frame_count, height=height, width=width
        )
        latent_frames = int(geometry[1])
        if latent_frames > MAGI2_MAX_SAMPLING_LATENT_FRAMES:
            raise ValueError(
                f"frame_count={int(frame_count)} maps to {latent_frames} latent "
                "frames, beyond the MAGI-2 Preview native horizon of "
                f"{MAGI2_MAX_SAMPLING_LATENT_FRAMES} latent frames "
                f"({_magi2_request_frames(MAGI2_MAX_SAMPLING_LATENT_FRAMES)} "
                f"frames at {MAGI2_REQUEST_FPS:g} fps, about 10 s), which is "
                "the full length the preview model is trained for. Latent "
                "lengths above it belong to the refiner stage, which resamples "
                "the preview latent in time; they are not a longer preview and "
                "sampling them on this path degenerates to flat output. The "
                "maximum supported request is "
                f"{_magi2_request_frames(MAGI2_MAX_SAMPLING_LATENT_FRAMES)} frames."
            )
        if latent_frames < MAGI2_MIN_SAMPLING_LATENT_FRAMES:
            raise ValueError(
                f"frame_count={int(frame_count)} maps to {latent_frames} latent "
                "frames, below the validated MAGI-2 Preview generation "
                f"envelope of {MAGI2_MIN_SAMPLING_LATENT_FRAMES} latent frames "
                f"({_magi2_request_frames(MAGI2_MIN_SAMPLING_LATENT_FRAMES)} "
                "frames, about 2.3 s). Short-horizon sampling below that "
                "degrades on this path and is unsupported; the bound is "
                "empirical."
            )
        return geometry

    def has_native_inference(self) -> bool:
        return True

    def resolve_flow_shift_for_latent_shape(self, latent_shape: tuple[int, ...]) -> float:
        _ = latent_shape
        return float(self.runtime_config.evaluation_config.shift)

    def _asset_root(self) -> Path:
        root = Path(str(self.model_config.path)).expanduser()
        return root.parent if root.name == "preview" else root

    def load_text_encoder(self, *, device: str) -> None:
        if self._text_encoder is None:
            from mirai.vendors.magi2_preview.model.qwen35 import Qwen35TextEncoder

            path = self._asset_root() / "text_encoder"
            if not path.is_dir():
                raise FileNotFoundError(
                    f"MAGI-2 text encoder assets are missing at {path}."
                )
            self._text_encoder = Qwen35TextEncoder(
                str(path),
                device=device,
                precision=torch.bfloat16,
                weight_quantization=self._text_encoder_weight_quantization,
            )
        else:
            self._text_encoder.to(device)
        self._inference_device = str(device)

    def encode_prompt(self, prompt: str, *, device: str) -> Any:
        if self._text_encoder is None:
            raise RuntimeError("load_text_encoder() must be called before encode_prompt().")
        if not str(prompt).strip() and bool(
            self.runtime_config.evaluation_config.use_negative_prompt
        ):
            from mirai.vendors.magi2_preview.pipeline.inference_engine import NEGATIVE_PROMPT

            prompt = NEGATIVE_PROMPT
        self._text_encoder.to(device)
        return self._text_encoder.encode(prompt).to(device=device, dtype=torch.float32)

    def offload_text_encoder(self) -> None:
        if self._text_encoder is not None:
            if self._text_encoder_weight_quantization == "none":
                self._text_encoder.to("cpu")
            else:
                self._text_encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_text_encoder_weight_quantization(self, mode: str) -> None:
        key = str(mode).strip().lower()
        if key not in {"none", "int8", "nf4"}:
            raise ValueError(
                "MAGI-2 text-encoder weight quantization must be 'none', "
                f"'int8', or 'nf4'; got '{mode}'."
            )
        if self._text_encoder is not None and key != self._text_encoder_weight_quantization:
            raise RuntimeError(
                "Text-encoder weight quantization cannot change while the encoder "
                "is loaded."
            )
        self._text_encoder_weight_quantization = key

    def load_vae(self, *, device: str) -> None:
        if self._vae is not None:
            self._vae.to(device)
            return
        from mirai.vendors.magi2_preview.model.turbo_vaed import get_turbo_vaed

        root = self._asset_root() / "turbo_vae"
        config_path = root / "TurboV3-Wan22-TinyShallow_7_7.json"
        checkpoint_path = root / "checkpoint.ckpt"
        missing = [str(path) for path in (config_path, checkpoint_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("MAGI-2 VAE assets are missing: " + ", ".join(missing))
        self._vae = get_turbo_vaed(
            str(config_path), str(checkpoint_path), device=device, weight_dtype=torch.bfloat16
        )

    def decode_latents_native(self, latents: list[Any]) -> Any:
        if self._vae is None:
            raise RuntimeError("load_vae() must be called before decode_latents_native().")
        value = torch.stack([torch.as_tensor(item) for item in latents], dim=0)
        device = next(self._vae.parameters()).device
        tile_size = int(getattr(self.options, "vae_decode_tile_size", 0))
        tiled = 0 < tile_size < max(value.shape[-2:])
        released_first = int(self._vae.first_chunk_size)
        released_step = int(self._vae.step_size)
        chunk_size = resolve_magi2_vae_decode_chunk_size(
            latent_frames=int(value.shape[-3]),
            latent_height=int(min(value.shape[-2], tile_size) if tiled else value.shape[-2]),
            latent_width=int(min(value.shape[-1], tile_size) if tiled else value.shape[-1]),
            released_chunk_size=min(released_first, released_step),
            requested_chunk_size=int(self.options.vae_decode_chunk_size),
        )
        self._last_vae_decode_chunk_size = chunk_size
        self._vae.first_chunk_size = chunk_size
        self._vae.step_size = chunk_size
        try:
            def decode(tile: torch.Tensor):
                return self._vae.decode(
                    tile.to(device=device, dtype=torch.bfloat16), output_offload=True
                )

            if tiled:
                from mirai.core.models.magi2_preview.vae_tiling import decode_spatial_tiles

                decoded = decode_spatial_tiles(
                    value, decode, tile_size=tile_size,
                    overlap=self.options.vae_decode_tile_overlap,
                )
            else:
                decoded = decode(value)
        finally:
            self._vae.first_chunk_size = released_first
            self._vae.step_size = released_step
        if isinstance(decoded, list):
            decoded = torch.cat(decoded, dim=0)
        frames = decoded[0].detach().float().mul(0.5).add(0.5).clamp(0.0, 1.0)
        return frames.permute(1, 0, 2, 3).contiguous().cpu()

    def offload_vae(self) -> None:
        if self._vae is not None:
            self._vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_extension_capabilities(self) -> ModelExtensionCapabilities:
        return ModelExtensionCapabilities(
            adapter_target_presets=True,
            adapter_runtime_controls=True,
            adapter_type_controls=True,
        )

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
            expert_on_load_formats=MAGI2_FROZEN_WEIGHT_QUANTIZATION_FORMATS,
            family_packed_weight_format=bool(getattr(self, "_expert_shared_basis_rank", 0)),
            router_quantization_policy=True,
            moe_kernel_backend=True,
        )

    def _restore_packed_experts(
        self, transformer: nn.Module, *,
        pruning_digest: bytes | None = None, consolidation_digest: bytes | None = None,
        shared_basis_rank: int | None = None,
    ) -> None:
        from mirai.core.models.magi2_preview.packed_experts import load_magi2_nf4_packed_state

        expected_metadata = {
            "model_variant": str(self.model_config.params.variant),
            "denoiser_subfolder": str(getattr(self.model_config.params, "denoiser_subfolder", "transformer") or "transformer"),
        }
        if shared_basis_rank is not None:
            from mirai.core.models.magi2_preview.basis_experts import MAGI2_BASIS_SOURCE_METADATA

            expected_metadata[MAGI2_BASIS_SOURCE_METADATA] = self._expert_basis_source_sha256
        self._expert_stores = load_magi2_nf4_packed_state(
            self._packed_expert_state_path, transformer, blocksize=self._nf4_blocksize,
            quant_format="shared_basis" if shared_basis_rank is not None else self._frozen_weight_quantization,
            expected_shared_basis_rank=shared_basis_rank,
            expected_pruning_digest=pruning_digest, expected_consolidation_digest=consolidation_digest,
            expected_metadata=expected_metadata,
        )
        if self._expert_rotation_group_size and any(
            getattr(store, "rotation_group_size", 0) != self._expert_rotation_group_size
            for store in self._expert_stores.values()
        ):
            raise ValueError("MAGI-2 packed artifact does not match the requested expert rotation.")

    def save_packed_frozen_weight_state(
        self,
        path: Any,
        *,
        metadata: dict[str, str] | None = None,
    ) -> Any:
        from mirai.core.models.magi2_preview.packed_experts import (
            save_magi2_nf4_packed_state,
        )

        if getattr(self, "_expert_shared_basis_rank", 0):
            from mirai.core.models.magi2_preview.basis_experts import MAGI2_BASIS_SOURCE_METADATA

            metadata = {**(metadata or {}), MAGI2_BASIS_SOURCE_METADATA: self._expert_basis_source_sha256}
        return save_magi2_nf4_packed_state(
            path, self.transformer, metadata=metadata,
            pruning_digest=self._expert_pruning_digest if self._expert_pruning_plan_path else None,
            consolidation_digest=(
                self._expert_pruning_digest if getattr(self, "_expert_consolidation_plan_path", "") else None
            ),
            fuse_gate_up=(getattr(self.model_config.params, "family_params", None) or {}).get(
                "nf4_fuse_gate_up", False
            ),
        )

    def _quantizes_experts_on_load(self) -> bool:
        """Whether the checkpoint load itself packs the routed expert stack."""
        return bool(
            self._quantize_experts_on_load
            and self._frozen_weight_quantization
            in MAGI2_FROZEN_WEIGHT_QUANTIZATION_FORMATS
        )

    def _transformer_device(self) -> torch.device | None:
        for parameter in self.transformer.parameters():
            return parameter.device
        return None

    def get_expert_tensor_specs(self) -> list[ExpertTensorSpec]:
        """Declare the router projection and the three routed tensors per layer.

        Everything else in a MAGI-2 MoE layer -- the expert bias buffer, the two
        shared experts, the hyper-connection state, and the norms -- is outside
        the routed expert stack and is not described here, because none of it is
        addressable on the packed ``head * num_experts + expert`` axis.
        """
        specs: list[ExpertTensorSpec] = []
        for module_name, module in iter_magi2_moe_layers(self.transformer):
            store = magi2_expert_store(module)
            packed_gate = getattr(module, "_mirai_router_weight_int8", None)
            gate = module.gate if packed_gate is None else packed_gate
            specs.append(
                ExpertTensorSpec(
                    name=f"{module_name}.gate",
                    owner_module=module_name,
                    tensor_name="gate",
                    role="router",
                    layout=("out", "in"),
                    shape=tuple(int(dim) for dim in gate.shape),
                    dtype=str(gate.dtype),
                    quantizable=True,
                    adapter_targetable=True,
                    routed=False,
                    router=True,
                )
            )
            for tensor_name, role in MAGI2_ROUTED_EXPERT_TENSORS:
                if store is not None:
                    shape = store.expert_weight_shape(tensor_name)
                    dtype = store.quant_format
                else:
                    tensor = getattr(module, tensor_name)
                    shape = tuple(int(dim) for dim in tensor.shape)
                    dtype = str(tensor.dtype)
                specs.append(
                    ExpertTensorSpec(
                        name=f"{module_name}.{tensor_name}",
                        owner_module=module_name,
                        tensor_name=tensor_name,
                        role=role,
                        layout=("expert", "out", "in"),
                        shape=tuple(int(dim) for dim in shape),
                        dtype=dtype,
                        quantizable=True,
                        # The shipped LoRA presets target attention and the
                        # router only; the packed expert axis carries no adapter.
                        adapter_targetable=False,
                        routed=True,
                    )
                )
        return list(validate_expert_tensor_specs(specs))

    def configure_moe_optimization_policy(self, policy: Any) -> None:
        router_mode = str(getattr(policy, "router_quantization", "disabled"))
        if self._router_int8_enabled and router_mode != "int8_per_channel":
            raise ValueError("Changing MAGI-2 router storage requires reloading the source model.")
        if router_mode == "int8_per_channel":
            params = self.model_config.params
            if getattr(params, "router_fp32_master", False):
                raise ValueError("MAGI-2 INT8 router storage is incompatible with an FP32 router master.")
            if str(getattr(params, "expert_precision_calibration", "off")).strip().lower() in {"imatrix", "alphaq"}:
                raise ValueError("MAGI-2 precision calibration requires the original router weights.")
        self._moe_optimization_policy = policy
        self._attach_moe_execution()

    def configure_moe_token_chunking(self, policy: Any) -> None:
        """Apply a bounded token schedule to grouped expert execution."""

        self._moe_token_chunk_policy = (
            policy if int(getattr(policy, "token_chunk_size", 0)) > 0 else None
        )
        self._attach_moe_execution()

    def configure_expert_feature_cache(self, cache: Any) -> None:
        """Arm (or clear) cross-timestep expert-branch reuse on the MoE seam."""
        self._expert_feature_cache = cache if getattr(cache, "enabled", False) else None
        self._attach_moe_execution()

    def _attach_moe_execution(self) -> None:
        """Resolve and bind the expert-execution seam for the current storage.

        Called both when the policy arrives and after quantization replaces the
        dense expert parameters, because the storage format decides which
        execution paths the policy can resolve to.
        """
        from mirai.core.models.magi2_preview.grouped_moe import (
            Magi2GroupedMoEBackend,
            Magi2GroupedMoEPolicyError,
            Magi2QuantizedGroupedMoEBackend,
            attach_grouped_moe_backend,
            resolve_magi2_moe_execution,
            validate_grouped_moe_backend_support,
        )

        quantized = bool(self._expert_stores)
        policy = self._moe_optimization_policy
        token_chunk_size = int(
            getattr(getattr(self, "_moe_token_chunk_policy", None), "token_chunk_size", 0)
        )
        plan = resolve_magi2_moe_execution(policy, quantized_experts=quantized)
        if plan is not None:
            validate_grouped_moe_backend_support(
                plan, device=self._transformer_device()
            )
        if plan is None:
            backend = None
        elif quantized:
            self._bind_quantized_expert_runtime_policy()
            backend = Magi2QuantizedGroupedMoEBackend(
                plan,
                token_chunk_size=token_chunk_size,
                expert_autograd=str(
                    getattr(policy, "moe_expert_autograd", "standard")
                ),
                activation_backend=str(
                    getattr(policy, "moe_activation_backend", "torch")
                ),
            )
        else:
            backend = Magi2GroupedMoEBackend(
                plan,
                token_chunk_size=token_chunk_size,
                activation_backend=str(
                    getattr(policy, "moe_activation_backend", "torch")
                ),
            )
        cache = getattr(self, "_expert_feature_cache", None)
        if cache is not None and getattr(cache, "enabled", False):
            from mirai.core.moe.runtime.expert_feature_cache import (
                CachedExpertExecution,
            )

            if backend is None:
                raise Magi2GroupedMoEPolicyError(
                    "inference.expert_feature_cache needs the grouped "
                    "expert-execution seam to expose per-branch features; set "
                    "memory.moe_kernel_backend='grouped'."
                )
            backend = CachedExpertExecution(inner=backend, cache=cache)
        if (
            getattr(policy, "router_quantization", "disabled") == "int8_per_channel"
            and not self._router_int8_enabled
            and (quantized or not getattr(self, "_expert_precision_plan_path", ""))
        ):
            from mirai.core.models.magi2_preview.router_weights import pack_magi2_router_weights

            pack_magi2_router_weights(self.transformer)
            self._router_int8_enabled = True
        attach_grouped_moe_backend(self.transformer, backend)
        # The grouped attach rebinds the execution seam this family's routing
        # telemetry rides on, so the tap is re-wrapped around whatever backend
        # the policy just selected.
        self._reattach_routing_collapse_tap()

    def _enable_routing_collapse_telemetry(self) -> None:
        """Arm the opt-in routing-collapse telemetry for this family.

        MAGI-2 carries no balance loss and no router statistic of its own, so
        the metrics come from a monitoring tap on the expert-execution seam.
        """
        from mirai.core.models.magi2_preview.routing_collapse import (
            Magi2RoutingCollapseState,
            attach_routing_collapse_tap,
        )

        state = Magi2RoutingCollapseState.create()
        attach_routing_collapse_tap(self.transformer, state.observer)
        self._routing_collapse = state

    def _reattach_routing_collapse_tap(self) -> None:
        """Re-wrap the execution seam after a backend change; no-op when off."""
        state = getattr(self, "_routing_collapse", None)
        if state is None:
            return
        from mirai.core.models.magi2_preview.routing_collapse import (
            attach_routing_collapse_tap,
        )

        attach_routing_collapse_tap(self.transformer, state.observer)

    def _bind_quantized_expert_runtime_policy(self) -> None:
        """Push the configured dequantization granularity onto every store."""
        policy = self._moe_optimization_policy
        for store in self._expert_stores.values():
            store.set_expert_weight_access_policy(
                expert_weight_access=getattr(policy, "expert_weight_access", "auto"),
                expert_dequant_chunk_size=int(
                    getattr(policy, "expert_dequant_chunk_size", 0)
                ),
            )

    def enable_quantized_frozen_weights(self, quant_type: str, **kwargs: Any) -> None:
        """Store the routed expert stack in a packed frozen format.

        With ``memory.quantize_experts_on_load`` the packing already happened
        while the checkpoint was read and this call only rebinds the runtime
        policy; without it the dense stack is converted layer by layer here,
        which requires the released BF16 experts to have been loaded first.
        """
        scheme = str(quant_type).strip().lower()
        if scheme not in MAGI2_FROZEN_WEIGHT_QUANTIZATION_FORMATS:
            raise ValueError(
                "MAGI-2 Preview implements memory.frozen_weight_quantization="
                + " or ".join(
                    f"'{value}'" for value in MAGI2_FROZEN_WEIGHT_QUANTIZATION_FORMATS
                )
                + f"; got '{scheme}'."
            )
        strategy = str(kwargs.pop("strategy", "auto")).strip().lower()
        if strategy not in {"", "auto", "disabled", "none", "compressed_weights"}:
            raise ValueError(
                "MAGI-2 frozen-weight quantization strategy must be 'auto' or "
                "'compressed_weights'."
            )
        block_size = int(kwargs.pop("block_size", self._nf4_blocksize))
        precision_plan_path = str(kwargs.pop("precision_plan_path", "") or "").strip()
        if kwargs:
            raise ValueError(
                "Unsupported MAGI-2 frozen-weight quantization options: "
                + ", ".join(sorted(kwargs))
                + "."
            )
        if self._expert_stores and any(store.quant_format != scheme for store in self._expert_stores.values()):
            raise ValueError("MAGI-2 expert storage already contains a different packed format.")
        self._frozen_weight_quantization = scheme
        self._nf4_blocksize = block_size
        if precision_plan_path:
            from mirai.core.models.magi2_preview.mixed_experts import (
                Magi2MixedExpertStore,
                quantize_magi2_mixed_experts,
            )

            if not self._expert_stores:
                self._expert_stores = quantize_magi2_mixed_experts(
                    self.transformer, plan_path=precision_plan_path, base_format=scheme,
                    blocksize=block_size,
                    rotation_group_size=self._expert_rotation_group_size,
                )
            elif (
                self._applied_expert_precision_plan != precision_plan_path
                or not all(isinstance(store, Magi2MixedExpertStore) for store in self._expert_stores.values())
            ):
                raise ValueError("A MAGI-2 precision plan cannot replace an existing packed store.")
            self._applied_expert_precision_plan = precision_plan_path
        elif not self._expert_stores:
            from mirai.core.models.magi2_preview.quantized_experts import (
                quantize_magi2_experts_in_place,
            )

            self._expert_stores = quantize_magi2_experts_in_place(
                self.transformer, blocksize=block_size, quant_format=scheme
            )
        self._attach_moe_execution()

    def has_quantized_frozen_weights(self) -> bool:
        return bool(self._expert_stores or self._packed_companion_linears or self._packed_norms or self._router_int8_enabled)

    def get_quantized_frozen_weight_report(self) -> dict[str, Any] | None:
        if not self.has_quantized_frozen_weights():
            return None
        stores = list(self._expert_stores.values())
        companion_projections = [
            projection
            for module in self._packed_companion_linears
            for projection in module._mirai_companion_projections
        ]
        companion_bytes = sum(
            buffer.numel() * buffer.element_size()
            for projection in companion_projections for buffer in projection.buffers()
        )
        routers = [
            module for _, module in iter_magi2_moe_layers(self.transformer)
            if hasattr(module, "_mirai_router_weight_int8")
        ] if self._router_int8_enabled else []
        router_numel = sum(module._mirai_router_weight_int8.numel() for module in routers)
        router_bytes = router_numel + sum(
            module._mirai_router_weight_scale.numel() * module._mirai_router_weight_scale.element_size()
            for module in routers
        )
        norm_numel = sum(module._mirai_norm_weight_int8.numel() for module in self._packed_norms)
        norm_bytes = norm_numel + sum(
            module._mirai_norm_weight_scale.numel() * module._mirai_norm_weight_scale.element_size()
            for module in self._packed_norms
        )
        return {
            "strategy": "compressed_weights",
            "quant_format": (
                stores[0].quant_format if stores else
                self._companion_weight_quantization if companion_projections else "int8"
            ),
            "companion_quant_format": self._companion_weight_quantization,
            "mixed_precision": bool(stores and getattr(stores[0], "is_mixed_precision", False)),
            "linear_modules": len(self._packed_companion_linears),
            "grouped_expert_modules": len(stores),
            "router_modules": len(routers),
            "norm_modules": len(self._packed_norms),
            "router_quantization": "int8_per_channel" if routers else "disabled",
            "quantized_tensors": (
                len(MAGI2_ROUTED_EXPERT_TENSORS) * len(stores)
                + len(companion_projections) + len(routers) + len(self._packed_norms)
            ),
            "quantized_numel": router_numel + norm_numel + sum(store.quantized_numel() for store in stores) + sum(
                projection.frozen_quantized_numel() for projection in companion_projections
            ),
            "packed_bytes": sum(store.payload_bytes() for store in stores) + companion_bytes + router_bytes + norm_bytes,
            "expert_weight_access": stores[0].expert_weight_access if stores else None,
            "expert_dequant_chunk_size": stores[0].expert_dequant_chunk_size if stores else None,
        }

    def preserves_native_parameter_dtypes(self) -> bool:
        return True

    def casts_trainable_tensors_for_quantized_base(self) -> bool:
        return bool(self._expert_stores or self._packed_companion_linears)

    def get_sparse_moe_capabilities(self) -> SparseMoECapabilities:
        arch = self.runtime_config.arch_config
        return SparseMoECapabilities(
            is_sparse_moe=True,
            architecture="MAGI-2 multi-head sparse MoE flow transformer",
            routing="per-head sigmoid top-k with learned expert bias",
            routing_granularity="token-per-head",
            num_routed_experts=int(arch.moe_config.num_experts),
            num_shared_experts=2,
            num_activated_experts=int(arch.moe_config.top_k),
            # The family emits no router statistic of its own; the opt-in
            # routing-collapse tap is what makes routing observable here.
            emits_router_metrics=getattr(self, "_routing_collapse", None) is not None,
        )

    def get_adapter_target_presets(self) -> dict[str, list[str]]:
        return {key: list(value) for key, value in MAGI2_TARGET_PRESETS.items()}

    def set_adapter_config(self, adapter_config: Any) -> None:
        if str(getattr(adapter_config, "type", "lora")).strip().lower() != "lora":
            raise ValueError("MAGI-2 Preview currently supports adapter.type='lora'.")
        preset = str(getattr(adapter_config, "target_preset", "attn_only")).strip()
        if (
            getattr(self, "_expert_pruning_plan_path", "")
            or getattr(self, "_expert_pruning_digest", None) is not None
        ) and preset != "attn_only":
            raise ValueError("MAGI-2 pruning lineage requires adapter.target_preset='attn_only'.")
        if preset not in MAGI2_TARGET_PRESETS:
            raise ValueError(
                f"Unsupported MAGI-2 adapter target preset '{preset}'; expected one of "
                + ", ".join(sorted(MAGI2_TARGET_PRESETS))
            )
        rank = int(getattr(adapter_config, "rank", 0))
        alpha = float(getattr(adapter_config, "alpha", rank))
        if rank <= 0:
            raise ValueError("adapter.rank must be > 0 for MAGI-2.")
        for parameter in self.transformer.parameters():
            parameter.requires_grad_(False)
        targets = MAGI2_TARGET_PRESETS[preset]
        matched: list[str] = []
        for name, module in self.transformer.named_modules():
            for tensor_name, parameter in tuple(module._parameters.items()):
                if parameter is None:
                    continue
                full_name = f"{name}.{tensor_name}" if name else tensor_name
                owner_name = f".{name}" if name else ""
                if not any(
                    (tensor_name == "weight" and owner_name.endswith(target)) or full_name.endswith(target)
                    for target in targets
                ):
                    continue
                if parametrize.is_parametrized(module, tensor_name):
                    continue
                packed_linear = tensor_name == "weight" and hasattr(module, "_mirai_companion_projections")
                packed_router = tensor_name == "gate" and hasattr(module, "_mirai_router_weight_int8")
                if packed_linear:
                    shape = (module.num_experts * module.out_features, module.in_features)
                elif packed_router:
                    shape = tuple(module._mirai_router_weight_int8.shape)
                else:
                    shape = tuple(parameter.shape)
                parameter.requires_grad_(False)
                parametrize.register_parametrization(
                    module,
                    tensor_name,
                    LowRankWeight(shape, rank=rank, alpha=alpha),
                    unsafe=packed_linear or packed_router,
                )
                module.parametrizations[tensor_name].original.requires_grad_(False)
                from mirai.core.models.magi2_preview.lora_execution import (
                    attach_magi2_lora_executor,
                )

                attach_magi2_lora_executor(module, tensor_name)
                matched.append(full_name)
        if not matched:
            raise ValueError(f"MAGI-2 LoRA preset '{preset}' matched no parameters.")
        self._adapter_configured = True

    def set_adapter_type(self, adapter_type: str) -> None:
        if str(adapter_type).strip().lower() != "lora":
            raise ValueError("MAGI-2 Preview supports only LoRA adapters.")

    def get_adapter_type(self) -> str:
        return "lora"

    def set_adapter_runtime(self, *, rank_dropout: float, lora_parameter_dropout: float) -> None:
        if float(rank_dropout) != 0.0 or float(lora_parameter_dropout) != 0.0:
            raise ValueError("MAGI-2 packed-weight LoRA does not yet support adapter dropout.")

    def set_lora_scale(self, scale: float) -> None:
        value = float(scale)
        self._lora_scale = value
        for module in self.transformer.modules():
            for parametrizations in module.parametrizations.values() if parametrize.is_parametrized(module) else ():
                for item in parametrizations:
                    if isinstance(item, LowRankWeight):
                        item.runtime_scale = value

    def get_lora_scale(self) -> float:
        return float(self._lora_scale)

    def load_adapter_state(self, state: dict[str, Any]) -> None:
        payload = state.get("adapter_state") if isinstance(state, dict) else None
        if isinstance(payload, dict):
            state = payload
        if not isinstance(state, dict):
            raise TypeError("MAGI-2 adapter state must be a mapping.")
        self.validate_adapter_payload(state)
        expected = set(self.state_dict())
        portable_alphas = {
            key for key in state
            if key.endswith(".lora_alpha") and key.removesuffix(".lora_alpha") + ".lora_a" in expected
        }
        for key in portable_alphas:
            alpha = state[key]
            adapter = self.transformer.get_submodule(key.removesuffix(".lora_alpha"))
            expected_alpha = torch.tensor(adapter.scale * adapter.rank, dtype=torch.float32)
            if not isinstance(alpha, torch.Tensor) or alpha.numel() != 1 or not torch.isclose(
                alpha.detach().cpu().float().reshape(()), expected_alpha,
            ):
                raise ValueError(f"MAGI-2 adapter alpha does not match the configured scale for {key!r}.")
        if portable_alphas:
            state = {key: value for key, value in state.items() if key not in portable_alphas}
        supplied = {str(key) for key in state}
        unknown = sorted(supplied - expected)
        missing = sorted(expected - supplied)
        if unknown or missing:
            raise ValueError(
                "MAGI-2 adapter state does not match the configured LoRA targets "
                f"(missing={missing[:4]}, unknown={unknown[:4]})."
            )
        self.transformer.load_state_dict(
            {key: value for key, value in state.items() if key != "__mirai_magi2_pruning_digest__"},
            strict=False,
        )

    def validate_adapter_payload(self, payload: dict[str, Any]) -> None:
        state = payload.get("adapter_state", payload)
        supplied = state.get("__mirai_magi2_pruning_digest__")
        expected = getattr(self, "_expert_pruning_digest", None)
        if expected is None and supplied is None:
            return
        if (
            expected is None or not isinstance(supplied, torch.Tensor)
            or supplied.dtype != torch.uint8 or supplied.shape != (len(expected),)
            or bytes(supplied.detach().cpu().tolist()) != expected
        ):
            raise ValueError("MAGI-2 adapter expert pruning topology differs from the configured base.")

    def set_gradient_checkpointing(self, enabled: bool | str) -> None:
        mode = str(enabled).strip().lower()
        if mode in MAGI2_GRADIENT_CHECKPOINTING_OFF:
            active = False
        elif mode in MAGI2_GRADIENT_CHECKPOINTING_ON:
            active = True
        else:
            raise ValueError(
                f"MAGI-2 Preview does not implement gradient_checkpointing='{mode}'. "
                "The vendored transformer block exposes one whole-block recompute "
                "switch, so only 'off' and 'standard' are accepted; 'selective' and "
                "'aggressive' would silently resolve to 'standard'."
            )
        self._gradient_checkpointing = mode
        self.transformer.block.gradient_checkpointing = active

    def get_block_swap_units(self) -> list[tuple[int, Any]]:
        return list(enumerate(self.transformer.block.layers))

    def get_compilation_regions(self) -> list[CompilationRegion]:
        # The layer call and its residency hooks stay eager. Packed attention
        # and routed experts own their dispatch and synchronization boundaries.
        return [
            CompilationRegion(
                name=f"transformer.block.layers.{index}.{method}",
                owner=layer,
                attribute=method,
            )
            for index, layer in self.get_block_swap_units()
            for method in (
                "_forward_mhc_pre_attn",
                "_forward_mhc_attn_to_mlp",
                "_forward_mhc_post_mlp",
            )
        ]

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
        execution_mode: WeightResidencyExecutionMode | str = (
            WeightResidencyExecutionMode.TRAINING
        ),
    ) -> None:
        phase = WeightResidencyExecutionMode.coerce(execution_mode)
        self._weight_residency_execution_mode = phase
        resolved = str(strategy).strip().lower()
        enabled = resolved != "disabled" and int(blocks_to_swap) > 0
        # The adapter is what keeps the streamed base frozen under a training
        # backward; an inference forward has no backward to protect.
        if (
            enabled
            and phase is WeightResidencyExecutionMode.TRAINING
            and not self._adapter_configured
        ):
            raise ValueError("MAGI-2 block residency requires a configured adapter.")
        self._residency_request = Magi2ResidencyRequest(
            enabled=enabled,
            blocks_to_swap=int(blocks_to_swap),
            mode=str(mode),
            block_residency_planner=str(block_residency_planner),
            block_swap_prefetch_depth=int(block_swap_prefetch_depth),
            block_residency_priority=str(block_residency_priority),
            block_swap_transfer_strategy=str(block_swap_transfer_strategy),
            offload_dir=offload_dir if resolved == "stream_disk" else None,
        )
        count = len(self.transformer.block.layers)
        self._block_swap_manager = BlockSwapManager(
            total_blocks=count,
            blocks_to_swap=min(count, int(blocks_to_swap)),
            mode=mode,
            block_swap_backward=block_swap_backward,
            block_residency_planner=block_residency_planner,
            block_swap_prefetch_depth=block_swap_prefetch_depth,
            block_residency_priority=block_residency_priority,
            block_swap_transfer_strategy=block_swap_transfer_strategy,
            disk_offload_dir=offload_dir if resolved == "stream_disk" else None,
        ) if enabled else None

    def validate_device_placement(self, *, device: Any) -> None:
        profile_path = str(
            self._moe_optimization_policy.expert_transfer_profile_path
        ).strip()
        if not profile_path:
            return
        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            raise ValueError(
                "memory.expert_transfer_profile_path requires CUDA placement."
            )
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(index)
        quantization = str(self._frozen_weight_quantization).strip().lower()
        validate_expert_transfer_profile_identity(
            load_expert_transfer_profile(profile_path),
            gpu_name=torch.cuda.get_device_name(index),
            compute_capability=f"{capability[0]}.{capability[1]}",
            expert_format=(
                "bf16"
                if quantization in {"", "none", "off", "disabled"}
                else quantization
            ),
        )

    def place_offloaded_modules(self, *, device: Any, strategy: str) -> None:
        manager = self._block_swap_manager
        inference_phase = (
            self._weight_residency_execution_mode
            is WeightResidencyExecutionMode.INFERENCE
        )
        if inference_phase and manager is not None and bool(self.transformer.training):
            raise RuntimeError(
                "Inference-mode MAGI-2 block residency requires the denoiser in "
                "eval(); a training-mode forward retains evicted block weights "
                "in the autograd graph."
            )
        if (
            manager is not None
            and not inference_phase
            and manager.block_swap_backward
            and str(self._gradient_checkpointing).strip().lower()
            in MAGI2_GRADIENT_CHECKPOINTING_OFF
        ):
            # A swapped block is released when its forward returns, so backward
            # reaches device weights only where the forward is recomputed.
            raise ValueError(
                "MAGI-2 block residency with training.block_swap_backward=true "
                "requires training.gradient_checkpointing='standard'; without "
                "recompute the backward pass reads host-resident block weights."
            )
        units = self.get_block_swap_units()
        routed = None
        if manager is not None and manager.uses_routed_experts:
            if torch.device(device).type != "cuda":
                raise ValueError("routed_experts block swap requires CUDA.")
            routed = {
                index: RoutedSwapRegion(experts, experts)
                for index, layer in units
                if (experts := getattr(layer.mlp, "moe_mlp", None)) is not None
            }
            manager.estimate_device_residency_reservations(units, routed_units=routed)
        excluded = (
            [region.weights for region in routed.values()]
            if routed is not None else [module for _, module in units]
        )
        move_trainable_tensors(self.transformer, device=device)
        move_tensors_outside_modules(self.transformer, excluded_modules=excluded, device=device)
        if self._block_swap_manager is None:
            self.transformer.to(device)
            return
        manager.bind(units, device=device, routed_units=routed)
        for handle in self._block_hook_handles:
            handle.remove()
        self._block_hook_handles.clear()
        for index, layer in units:
            self._block_hook_handles.append(
                layer.register_forward_pre_hook(
                    lambda _module, _args, idx=index: self._block_swap_manager.before_block(idx)
                )
            )
            self._block_hook_handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, output, idx=index: (
                        self._block_swap_manager.after_block(idx), output
                    )[1]
                )
            )

    def flush_runtime_offloads(self) -> None:
        if self._block_swap_manager is not None:
            self._block_swap_manager.finish_backward()

    def finish_backward_offloads(self) -> None:
        self.flush_runtime_offloads()

    def get_block_swap_state(self) -> dict[str, Any]:
        if self._block_swap_manager is None:
            return {"enabled": False}
        return {"enabled": True, **self._block_swap_manager.snapshot()}

    def apply_noise(self, clean_latents: Any, noise: Any, timesteps: Any) -> Any:
        sigma = torch.as_tensor(timesteps, device=clean_latents.device, dtype=clean_latents.dtype)
        while sigma.ndim < clean_latents.ndim:
            sigma = sigma.unsqueeze(-1)
        return clean_latents * (1.0 - sigma) + noise * sigma

    def compute_target(self, noise: Any, clean_latents: Any, timesteps: Any) -> Any:
        _ = timesteps
        return noise - clean_latents

    def _text_features(self, text_embeds: Any, *, batch: int, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = text_embeds
        mask = None
        if isinstance(text_embeds, dict):
            value = text_embeds.get("magi2", text_embeds.get("qwen", text_embeds.get("t5")))
            mask = text_embeds.get("text_mask", text_embeds.get("attention_mask"))
        if value is None:
            raise ValueError(
                "MAGI-2 requires Qwen3.5 text embeddings of width "
                f"{MAGI2_TEXT_EMBED_WIDTH}; the batch carried none under "
                "'magi2', 'qwen', or 't5'. Unconditional conditioning is a "
                "real encode of the empty prompt, not an absent embedding."
            )
        value = torch.as_tensor(value, device=like.device, dtype=torch.float32)
        if value.ndim == 2:
            value = value.unsqueeze(1)
        if value.ndim != 3 or int(value.shape[-1]) != MAGI2_TEXT_EMBED_WIDTH:
            raise ValueError(
                "MAGI-2 text embeddings must be [B, S, "
                f"{MAGI2_TEXT_EMBED_WIDTH}] Qwen3.5 hidden states; got "
                f"{tuple(value.shape)}. A different width means the latent cache "
                "was written by another text encoder, so its lineage does not "
                "match this model."
            )
        if int(value.shape[0]) != batch:
            raise ValueError(
                "MAGI-2 text embeddings must cover the latent batch: got "
                f"{int(value.shape[0])} rows for {batch} latents."
            )
        lengths = (
            torch.as_tensor(mask, device=like.device).long().sum(dim=1)
            if mask is not None
            else torch.full((batch,), value.shape[1], device=like.device, dtype=torch.long)
        )
        return value, lengths

    def forward(self, noisy_latents: Any, timesteps: Any, text_embeds: Any, **kwargs: Any) -> Any:
        _ = kwargs
        from mirai.vendors.magi2_preview.pipeline.preview_data_proxy import ModelInput
        latents = torch.as_tensor(noisy_latents).float()
        if latents.ndim != 5 or latents.shape[1] != 48:
            raise ValueError("MAGI-2 noisy_latents must have shape [B, 48, T, H, W].")
        batch = int(latents.shape[0])
        text, text_lengths = self._text_features(text_embeds, batch=batch, like=latents)
        configured_audio_tokens = int(self.options.audio_tokens)
        audio_tokens = resolve_magi2_audio_tokens(configured_audio_tokens, int(latents.shape[2]))
        # MAGI-2 ships without an audio encoder, so the audio track carries no
        # user signal; the reference engine feeds it Gaussian noise
        # (`pipeline/inference_engine.py`), once per generation. Those tokens
        # take part in attention and MoE routing, so this training forward draws
        # them fresh on every call rather than freezing one hidden sample - the
        # deliberate difference from the per-generation inference draw in
        # ``sample_native_preview``. The draw comes from the run-seeded family
        # generator, so a training step is reproducible under its seed and the
        # audio track length never perturbs the process RNG stream.
        audio = torch.randn(
            batch,
            audio_tokens,
            64,
            generator=self._audio_noise_generator,
            dtype=latents.dtype,
        ).to(device=latents.device)
        timestep = torch.as_tensor(timesteps, device=latents.device).reshape(batch)
        model_input = ModelInput(
            x_t=latents,
            audio_x_t=audio,
            audio_feat_len=torch.full((batch,), audio_tokens, device=latents.device, dtype=torch.long),
            txt_feat=text,
            txt_feat_len=text_lengths,
            t=timestep,
            per_token_video_t=timestep.to(latents.dtype).view(batch, 1, 1, 1, 1).expand(
                batch, 1, *latents.shape[2:]
            ),
            per_token_audio_t=timestep.to(latents.dtype).view(batch, 1, 1).expand(
                batch, audio_tokens, 1
            ),
        )
        packed = self.data_proxy.process_input(model_input)
        prediction = self.transformer(*packed)
        video, audio_prediction = self.data_proxy.process_output(prediction)
        self._last_audio_prediction = audio_prediction
        self._collect_routing_diagnostics()
        return video

    def _collect_routing_diagnostics(self) -> None:
        """Fold this forward's routing observations into the step diagnostics.

        Cross-step collapse state advances on training forwards only, so an
        inference denoise loop neither pays for the reduction nor perturbs the
        streak counters.
        """
        state = getattr(self, "_routing_collapse", None)
        if state is None:
            return
        if not bool(self.transformer.training):
            state.observer.clear()
            self._last_diagnostics = {}
            return
        self._last_diagnostics = state.collect()

    @torch.inference_mode()
    def sample_native_preview(
        self,
        *,
        noise: torch.Tensor,
        context: torch.Tensor,
        context_null: torch.Tensor,
        denoise_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        solver_name: str = MAGI2_NATIVE_SOLVER,
        cfg_mode: str = MAGI2_NATIVE_CFG_MODE,
        cfg_schedule: CfgSchedule | None = None,
        diffusion_cache: DiffusionPredictionCache | None = None,
    ) -> torch.Tensor:
        """Run the shipping joint video/audio Flow-UniPC sampler.

        Both CFG layouts use the same Flow-UniPC schedule and guidance blend.
        """
        requested_solver = str(solver_name).strip().lower()
        if requested_solver != MAGI2_NATIVE_SOLVER:
            raise ValueError(
                f"MAGI-2 Preview implements the '{MAGI2_NATIVE_SOLVER}' solver only; "
                f"got '{requested_solver}'. The native sampler is the vendored "
                "Flow-UniPC multistep scheduler and has no other schedule."
            )
        requested_cfg_mode = str(cfg_mode).strip().lower()
        if requested_cfg_mode not in {MAGI2_NATIVE_CFG_MODE, "paired"}:
            raise ValueError(
                f"MAGI-2 Preview requires batched or paired CFG; got '{requested_cfg_mode}'."
            )
        if requested_cfg_mode == "paired" and getattr(self, "_expert_feature_cache", None) is not None:
            raise ValueError("Paired MAGI-2 CFG requires inference.expert_feature_cache='off'.")
        if diffusion_cache is not None and (
            (cfg_schedule is not None and cfg_schedule.enabled)
            or getattr(self, "_expert_feature_cache", None) is not None
        ):
            raise ValueError("Diffusion caching requires default CFG scheduling and expert-feature caching off.")
        from mirai.vendors.magi2_preview.pipeline.preview_data_proxy import (
            CFGConfig,
            SamplerInput,
        )
        from mirai.vendors.magi2_preview.pipeline.sampler import (
            FlowUniPCMultistepScheduler,
            Magi2PreviewSampler,
        )

        device = noise.device
        latent = noise.unsqueeze(0)
        audio_tokens = resolve_magi2_audio_tokens(int(self.options.audio_tokens), int(noise.shape[1]))
        audio_latent = torch.randn(
            1,
            audio_tokens,
            64,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        self._refiner_generator_state = generator.get_state().detach().cpu().clone()
        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        shift = float(self.runtime_config.evaluation_config.shift)
        video_scheduler.set_timesteps(int(denoise_steps), device=device, shift=shift)
        audio_scheduler.set_timesteps(int(denoise_steps), device=device, shift=shift)
        evaluation = self.runtime_config.evaluation_config
        cfg = CFGConfig(
            use_cfg_trick=bool(evaluation.use_cfg_trick),
            cfg_trick_start_frame=int(evaluation.cfg_trick_start_frame),
            cfg_trick_value=float(evaluation.cfg_trick_value),
            use_dynamic_cfg=bool(evaluation.use_dynamic_cfg),
            dynamic_cfg_start_t=int(evaluation.dynamic_cfg_start_t),
            dynamic_cfg_cutoff_value=float(evaluation.dynamic_cfg_cutoff_value),
            video_txt_guidance_scale=float(guidance_scale),
            audio_txt_guidance_scale=float(evaluation.audio_txt_guidance_scale),
            use_ref_for_uncond=bool(evaluation.use_ref_for_uncond),
            use_skimmed_cfg_linear=bool(evaluation.use_skimmed_cfg_linear),
            skimmed_cfg_scale=float(evaluation.skimmed_cfg_scale),
            cfg_rescale=float(evaluation.cfg_rescale),
        )
        sampler = Magi2PreviewSampler(
            model=self.transformer,
            data_proxy=self.data_proxy,
            device=str(device),
            dtype=torch.bfloat16,
        )

        def resident_forward(model_input):
            packed = self.data_proxy.process_input(model_input)
            output = self.transformer(*packed)
            return self.data_proxy.process_output(output)

        if requested_cfg_mode == "paired":
            first_proxy = type(self.data_proxy)(self.data_proxy.config)
            second_proxy = type(self.data_proxy)(self.data_proxy.config)
            sampler.forward = lambda model_input: self._paired_preview_forward(
                model_input, first_proxy, second_proxy
            )
        else:
            sampler.forward = resident_forward
        if diffusion_cache is not None:
            sampler.forward = self._cached_preview_prediction(diffusion_cache, sampler.forward)
        if cfg_schedule is not None and cfg_schedule.enabled:
            if getattr(self, "_expert_feature_cache", None) is not None:
                raise ValueError("Scheduled MAGI-2 CFG requires inference.expert_feature_cache='off'.")
            sampler.prediction_step = self._scheduled_preview_prediction(
                cfg_schedule, len(video_scheduler.timesteps), sampler.forward, resident_forward
            )
        sampler.step_callback = self.flush_runtime_offloads
        sampled_video, _sampled_audio = sampler.sample(
            SamplerInput(
                video_t_list=video_scheduler.timesteps,
                audio_t_list=audio_scheduler.timesteps,
                latent=latent,
                audio_latent=audio_latent,
                txt_feat=context,
                null_txt_feat=context_null,
                ref_audio_feat=torch.zeros(1, 0, 64, device=device),
                ref_video_feat=None,
                video_scheduler=video_scheduler,
                audio_scheduler=audio_scheduler,
                cfg_config=cfg,
            )
        )
        self._refiner_context_pair = (
            context.detach().to(device="cpu"),
            context_null.detach().to(device="cpu"),
        )
        return sampled_video[0]

    @staticmethod
    def _cached_preview_prediction(cache, forward):
        def predict(model_input):
            inputs = (model_input.x_t[:1], model_input.audio_x_t[:1], model_input.t[:1])
            reused = cache.lookup(inputs)
            if reused is not None:
                return reused
            prediction = forward(model_input)
            cache.store(inputs, prediction)
            return prediction

        return predict

    @staticmethod
    def _preview_branch_input(model_input, index):
        return replace(model_input, **{
            field.name: value[index : index + 1]
            for field in fields(model_input)
            if (value := getattr(model_input, field.name)) is not None
        })

    def _scheduled_preview_prediction(self, policy, steps, pair_forward, single_forward):
        cached = None
        last_step = -1

        def predict(index, model_input):
            nonlocal cached, last_step
            if not policy.applies(index, steps):
                cached = None
                return single_forward(self._preview_branch_input(model_input, 0)), True
            if cached is None or index - last_step >= policy.uncond_reuse_stride:
                predictions = pair_forward(model_input)
                if policy.uncond_reuse_stride > 1:
                    cached = tuple(value[1:2].clone() for value in predictions)
                    last_step = index
                return predictions, False
            conditional = single_forward(self._preview_branch_input(model_input, 0))
            return tuple(
                torch.cat((cond, uncond)) for cond, uncond in zip(conditional, cached, strict=True)
            ), False

        return predict

    def _paired_preview_forward(self, model_input, first_proxy, second_proxy):
        first, second = self.transformer.forward_pair(
            first_proxy.process_input(Magi2PreviewPipeline._preview_branch_input(model_input, 0)),
            second_proxy.process_input(Magi2PreviewPipeline._preview_branch_input(model_input, 1)),
        )
        first_video, first_audio = first_proxy.process_output(first)
        second_video, second_audio = second_proxy.process_output(second)
        return torch.cat((first_video, second_video)), torch.cat((first_audio, second_audio))

    # -- refiner stage -----------------------------------------------------
    def _refiner_assets(self) -> Any:
        if self._refiner is None:
            from mirai.core.models.magi2_preview.refiner import Magi2Refiner

            self._refiner = Magi2Refiner(
                self.model_config,
                config_path=self.options.refiner_config_path,
                subfolder=self.options.refiner_subfolder,
                attention_backend=self.options.refiner_attention_backend,
                query_chunk_size=self.options.refiner_query_chunk_size,
                weight_quantization=self.options.refiner_weight_quantization,
            )
        return self._refiner

    def supports_refiner(self) -> bool:
        """MAGI-2 implements the refiner stage; assets are checked separately."""
        return True

    def has_refiner_weights(self) -> bool:
        return bool(self._refiner_assets().has_weights())

    def refiner_residency_request(self) -> Any:
        """Resolve the refiner's stage-specific block-transfer schedule."""

        request = self._residency_request
        if request is None:
            return None
        mode = str(getattr(self.options, "refiner_block_swap_mode", "sync"))
        transfer = request.block_swap_transfer_strategy
        if transfer.strip().lower() == "routed_experts":
            transfer = "per_tensor"
        return replace(request, mode=mode, block_swap_transfer_strategy=transfer)

    def discard_refiner_context(self) -> None:
        """Drop prompt embeddings retained only for the adjacent refiner stage."""

        self._refiner_context_pair = None

    def take_refiner_context(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Consume preview prompt embeddings so refinement need not re-encode."""

        pair = self._refiner_context_pair
        self._refiner_context_pair = None
        return pair

    def load_refiner(self, *, device: str) -> None:
        self._refiner_assets().load(
            device=device, residency=self.refiner_residency_request()
        )

    def release_refiner(self) -> None:
        if self._refiner is not None:
            self._refiner.release()

    def release_base_transformer(self) -> None:
        """Move the preview transformer off the compute device for the refiner.

        The preview transformer stays on CPU, so refinement terminates the
        current sampling session; the owning session restores its placement
        before the next generation.
        """
        for handle in self._block_hook_handles:
            handle.remove()
        self._block_hook_handles.clear()
        if self._block_swap_manager is not None:
            self._block_swap_manager.release_device()
        self.transformer.to(device=torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validate_refinement_request(
        self,
        request: dict[str, Any],
        *,
        frames: int,
        height: int,
        width: int,
    ) -> dict[str, Any]:
        """Resolve a refinement request against the released refiner profile.

        Absent values take the release profile, so ``--refine`` alone runs the
        shipped refinement. Keys that describe another family's mechanism are
        rejected rather than ignored: MAGI-2 enters the refiner at a fixed index
        into a zero-terminal-SNR table, not at a rectified-flow ``t_thresh``.

        Resolving here also arms the stage, which is what makes the latent
        layout report the refined output rate.
        """
        latent_frames = int(self.preview_latent_geometry(
            frame_count=int(frames), height=int(height), width=int(width)
        )[1])
        if latent_frames < 2:
            raise RuntimeError(
                "MAGI-2 refinement resamples the preview latent in time and "
                f"needs at least two latent frames; frames={int(frames)} maps to "
                f"{latent_frames}."
            )
        unsupported = sorted(
            key
            for key in ("t_thresh", "sigma_tail_steps")
            if request.get(key) is not None
        )
        if unsupported:
            raise RuntimeError(
                "The MAGI-2 refiner does not implement "
                + ", ".join(f"--refiner-{key.replace('_', '-')}" for key in unsupported)
                + ". It re-noises once at magi2_refiner_noise_value of the "
                "zero-terminal-SNR table declared by the refiner profile; change "
                "the profile through "
                "model.params.family_params.refiner_config_path instead."
            )
        refiner = self._refiner_assets()
        if not refiner.has_weights():
            raise RuntimeError(
                "MAGI-2 refinement requires a separate checkpoint under "
                f"'{refiner.checkpoint_dir()}' holding "
                "model.safetensors.index.json and its shards."
            )
        settings = refiner.settings(
            steps=request.get("steps"),
            cfg_scale=request.get("cfg_scale"),
            shift=request.get("shift"),
            height=request.get("height"),
            width=request.get("width"),
            preview_height=int(height),
            preview_width=int(width),
            scheduler=str(request.get("scheduler") or MAGI2_NATIVE_SOLVER),
        )
        self._refine_settings = settings
        return settings.as_request()

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
        conditioning: Any | None = None,
    ) -> Any:
        from mirai.core.models.magi2_preview.refiner import run_refine

        if self._refine_settings is None:
            raise RuntimeError(
                "MAGI-2 refinement must be validated before it runs; "
                "validate_refinement_request() resolves the release profile."
            )
        # MAGI-2 declares text_to_video only, so a conditioned request never
        # reaches this stage; the parameter exists to satisfy the shared
        # contract. ``dtype`` is discarded for the same reason: the vendored
        # refiner fixes its own dtype policy, so the stage has no compute dtype
        # a caller may select.
        _ = request, conditioning, dtype
        return run_refine(
            pipeline=self,
            refiner=self._refiner_assets(),
            base_latent=torch.as_tensor(base_latent),
            settings=self._refine_settings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            device=str(device),
        )

    def refiner_forward(self, latents: Any, context: Any) -> Any:
        """One velocity forward through the loaded refiner transformer.

        The refiner takes no timestep: it is entered at one fixed corruption
        level, and the solver alone advances the state. Its audio track is empty
        by design — the released profile sets
        ``magi2_refiner_audio_noise_scale = -1``, the sentinel for "no audio
        tokens at all" — so the audio velocity comes back empty and is dropped.
        """
        from mirai.vendors.magi2_preview.pipeline.inference_engine import EvalInput

        refiner = self._refiner_assets()
        if not refiner.loaded:
            raise RuntimeError("MAGI-2 refiner is not loaded; call load_refiner() first.")
        video = torch.as_tensor(latents)
        device = video.device
        audio_channels = int(
            refiner.runtime_config().magi2_refiner_arch_config.audio_in_channels
        )
        empty_audio = torch.zeros(1, 0, audio_channels, dtype=torch.float32, device=device)
        text = torch.as_tensor(context).to(device=device, dtype=torch.float32)
        packed = refiner.data_proxy.process_input(
            EvalInput(
                x_t=video,
                audio_x_t=empty_audio,
                audio_feat_len=torch.tensor([0], device=device),
                txt_feat=text,
                txt_feat_len=torch.tensor([int(text.shape[1])], device=device),
                ref_audio_feat=empty_audio,
                ref_audio_feat_len=torch.tensor([0], device=device),
                ref_video_feat=torch.empty_like(video),
                ref_video_feat_len=torch.tensor([0], device=device),
            ),
            compact=True,
        )
        prediction = refiner.transformer(*packed)
        video_velocity, audio_velocity = refiner.data_proxy.process_output(prediction)
        if audio_velocity is not None and audio_velocity.numel() > 0:
            raise RuntimeError(
                "The MAGI-2 refiner returned audio velocity for an empty audio "
                "track, so the configured refiner profile does not match the "
                "released audio-free refinement."
            )
        return video_velocity

    def refiner_cfg_forward(
        self, latents: Any, context: Any, context_null: Any
    ) -> tuple[Any, Any]:
        """Evaluate both CFG branches with one refiner layer-residency pass."""
        from mirai.vendors.magi2_preview.pipeline.inference_engine import EvalInput

        refiner = self._refiner_assets()
        if not refiner.loaded:
            raise RuntimeError("MAGI-2 refiner is not loaded; call load_refiner() first.")
        video = torch.as_tensor(latents)
        device = video.device
        audio_channels = int(
            refiner.runtime_config().magi2_refiner_arch_config.audio_in_channels
        )
        empty_audio = torch.zeros(1, 0, audio_channels, dtype=torch.float32, device=device)

        def pack(value: Any) -> tuple[tuple[Any, Any, Any, Any, Any], Any]:
            text = torch.as_tensor(value).to(device=device, dtype=torch.float32)
            packed = refiner.data_proxy.process_input(
                EvalInput(
                    x_t=video,
                    audio_x_t=empty_audio,
                    audio_feat_len=torch.tensor([0], device=device),
                    txt_feat=text,
                    txt_feat_len=torch.tensor([int(text.shape[1])], device=device),
                    ref_audio_feat=empty_audio,
                    ref_audio_feat_len=torch.tensor([0], device=device),
                    ref_video_feat=torch.empty_like(video),
                    ref_video_feat_len=torch.tensor([0], device=device),
                ),
                compact=True,
            )
            return packed, refiner.data_proxy.output_layout()

        cond_packed, cond_layout = pack(context)
        null_packed, null_layout = pack(context_null)
        cond_prediction, null_prediction = refiner.forward_pair(
            cond_packed, null_packed
        )
        cond_video, cond_audio = refiner.data_proxy.process_output(
            cond_prediction, layout=cond_layout
        )
        null_video, null_audio = refiner.data_proxy.process_output(
            null_prediction, layout=null_layout
        )
        for audio_velocity in (cond_audio, null_audio):
            if audio_velocity is not None and audio_velocity.numel() > 0:
                raise RuntimeError(
                    "The MAGI-2 refiner returned audio velocity for an empty audio track."
                )
        return cond_video, null_video

    def validate_config(self, config: Any) -> list[str]:
        errors: list[str] = []
        if str(config.model.type).strip().lower() not in {"magi2-preview", "magi-2-preview"}:
            errors.append("MAGI-2 pipeline requires model.type='magi2-preview'.")
        return errors

    def get_trainable_parameters(self):
        return [parameter for parameter in self.transformer.parameters() if parameter.requires_grad]

    def get_named_trainable_parameters(self):
        return [(name, parameter) for name, parameter in self.transformer.named_parameters() if parameter.requires_grad]

    def get_training_model(self) -> Any:
        return self.transformer

    def get_training_diagnostics(self) -> dict[str, Any]:
        return dict(getattr(self, "_last_diagnostics", {}))

    def train(self, mode: bool = True):
        self.transformer.train(mode)
        state = getattr(self, "_routing_collapse", None)
        if state is not None:
            state.observer.set_enabled(bool(mode))
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        full = self.transformer.state_dict(*args, **kwargs)
        adapter = {key: value for key, value in full.items() if "parametrizations." in key and (key.endswith("lora_a") or key.endswith("lora_b"))}
        digest = getattr(self, "_expert_pruning_digest", None)
        if digest is not None:
            adapter["__mirai_magi2_pruning_digest__"] = torch.tensor(list(digest), dtype=torch.uint8)
        return adapter

    def load_state_dict(self, state: dict[str, Any], strict: bool = True):
        """Load the adapter surface reported by :meth:`state_dict`.

        The payload is applied to the wrapped transformer non-strictly because it
        never carries the frozen base tensors. ``strict`` is enforced against the
        adapter surface itself: a payload that does not cover exactly the
        configured LoRA parameters is a lineage mismatch and fails.
        """
        self.validate_adapter_payload(state)
        if strict or getattr(self, "_expert_pruning_digest", None) is not None:
            expected = set(self.state_dict())
            supplied = {str(key) for key in state}
            missing = sorted(expected - supplied) if strict else []
            unexpected = sorted(supplied - expected)
            if missing or unexpected:
                raise ValueError(
                    "MAGI-2 adapter state does not match the configured LoRA "
                    f"targets (missing={missing[:4]}, unexpected={unexpected[:4]})."
                )
        return self.transformer.load_state_dict(
            {key: value for key, value in state.items() if key != "__mirai_magi2_pruning_digest__"},
            strict=False,
        )


class Magi2PreviewModelFamilyProvider(ModelFamilyProvider):
    _ALLOWED_FAMILY_PARAMS = {
        "nf4_fuse_gate_up",
        "expert_pruning_plan_path",
        "expert_consolidation_plan_path",
        "expert_shared_basis_rank",
        "expert_pruning_adapter_source_plan_path",
        "expert_rotation_group_size",
        "companion_weight_quantization",
        "norm_weight_quantization",
        "config_path",
        "audio_tokens",
        "refiner_config_path",
        "refiner_subfolder",
        "refiner_attention_backend",
        "refiner_query_chunk_size",
        "refiner_weight_quantization",
        "refiner_block_swap_mode",
        "vae_decode_chunk_size",
        "vae_decode_tile_size",
        "vae_decode_tile_overlap",
    }

    def __init__(self, model_type: str = "magi2-preview") -> None:
        super().__init__(
            model_type=model_type,
            native=True,
            sparse_moe=True,
            strict_native_assets_by_default=True,
            batched_cfg_inference=True,
            paired_cfg_inference=True,
            expert_pruning_calibration=True,
            diet_calibration_dense_source=True,
            prototype_calibration_dense_source=True,
            diet_calibration_cfg_mode="paired",
            expert_precision_calibration=True,
            # Raw media is encoded by the family-owned Qwen3.5/Wan2.2 encoder
            # returned from build_native_cache_encoder(); no generic pixel
            # projection is a valid MAGI-2 latent.
            native_cache_encoding=True,
            config_defaults_name="magi2_preview",
            inference_tasks=("text_to_video",),
            dataset_caption_formats=("raw",),
            pipeline_type=Magi2PreviewPipeline,
        )

    def build_diet_calibration_targets(self, pipeline: Any):
        from mirai.core.models.magi2_preview.expert_pruning import build_magi2_diet_calibration_targets

        return build_magi2_diet_calibration_targets(pipeline.transformer)

    def supports_prototype_calibration(self, cfg: Any = None) -> bool:
        return cfg is not None and str(cfg.model.params.expert_consolidation).strip().lower() == "hierarchical_output"

    def build_prototype_calibration_targets(self, pipeline: Any):
        from mirai.core.models.magi2_preview.expert_pruning import build_magi2_hierarchical_calibration_targets

        return build_magi2_hierarchical_calibration_targets(pipeline.transformer)

    def prototype_dense_source_fingerprint(self, pipeline: Any) -> str:
        return "sha256:" + self.diet_dense_source_fingerprint(pipeline)

    def diet_dense_source_fingerprint(self, pipeline: Any) -> str:
        from mirai.core.models.magi2_preview.expert_pruning import magi2_diet_source_fingerprint

        return magi2_diet_source_fingerprint(pipeline.transformer)

    def build_expert_pruning_calibration_targets(self, pipeline: Any):
        from mirai.core.models.magi2_preview.expert_pruning import build_magi2_pruning_calibration_targets

        return build_magi2_pruning_calibration_targets(pipeline.transformer)

    def build_expert_precision_calibration_targets(self, pipeline: Any):
        from mirai.core.models.magi2_preview.mixed_experts import build_magi2_precision_calibration_targets

        return build_magi2_precision_calibration_targets(
            pipeline.transformer, blocksize=pipeline._nf4_blocksize,
            audio_noise_generator=pipeline._audio_noise_generator,
            rotation_group_size=pipeline._expert_rotation_group_size,
        )

    def expert_precision_candidate_formats(self) -> tuple[str, ...]:
        return (*MAGI2_PACKED_EXPERT_FORMATS, "bf16")

    def validate_family_params(self, params: dict[str, Any]) -> list[str]:
        from mirai.core.models.magi2_preview.companion_weights import MAGI2_COMPANION_FORMATS, MAGI2_NORM_FORMATS
        from mirai.core.models.magi2_preview.mixed_experts import MAGI2_EXPERT_ROTATION_SIZES

        unknown = sorted(set(params) - self._ALLOWED_FAMILY_PARAMS)
        errors = [f"Unknown MAGI-2 family parameter '{name}'." for name in unknown]
        basis_rank = params.get("expert_shared_basis_rank", 0)
        if type(basis_rank) is not int or basis_rank < 0:
            errors.append("MAGI-2 expert_shared_basis_rank must be a non-negative integer.")
        if params.get("companion_weight_quantization", "none") not in MAGI2_COMPANION_FORMATS:
            errors.append("MAGI-2 companion_weight_quantization must be 'none', 'fp8', or 'nf4'.")
        if params.get("norm_weight_quantization", "none") not in MAGI2_NORM_FORMATS:
            errors.append("MAGI-2 norm_weight_quantization must be 'none' or 'int8'.")
        rotation = params.get("expert_rotation_group_size", 0)
        if type(rotation) is not int or rotation not in MAGI2_EXPERT_ROTATION_SIZES:
            errors.append("MAGI-2 expert_rotation_group_size must be 0, 4, 16, 64, or 256.")
        if not isinstance(params.get("nf4_fuse_gate_up", False), bool):
            errors.append("MAGI-2 family_params.nf4_fuse_gate_up must be a boolean.")
        if "audio_tokens" in params and int(params["audio_tokens"]) < -1:
            errors.append("MAGI-2 family_params.audio_tokens must be -1 (auto) or >= 0.")
        for name in (
            "expert_pruning_plan_path",
            "expert_consolidation_plan_path",
            "expert_pruning_adapter_source_plan_path",
            "refiner_config_path",
            "refiner_subfolder",
            "refiner_attention_backend",
            "refiner_block_swap_mode",
        ):
            value = params.get(name)
            if value is not None and not isinstance(value, str):
                errors.append(f"MAGI-2 family_params.{name} must be a string.")
        attention = params.get("refiner_attention_backend")
        chunk = params.get("refiner_query_chunk_size", 0)
        quantization = params.get("refiner_weight_quantization", "none")
        if not isinstance(quantization, str) or quantization.strip().lower() not in {"none", "fp8", "nf4"}:
            errors.append("MAGI-2 family_params.refiner_weight_quantization must be 'none', 'fp8', or 'nf4'.")
        if type(chunk) is not int or chunk < 0:
            errors.append("MAGI-2 family_params.refiner_query_chunk_size must be a nonnegative integer.")
        elif chunk and str(attention).strip().lower() != "native_flex":
            errors.append("MAGI-2 refiner query chunking requires refiner_attention_backend='native_flex'.")
        if isinstance(attention, str):
            try:
                normalize_refiner_attention_backend(attention)
            except Magi2RefinerAttentionUnsupported as exc:
                errors.append(str(exc))
        vae_chunk = params.get("vae_decode_chunk_size")
        tile = params.get("vae_decode_tile_size", 0)
        overlap = params.get("vae_decode_tile_overlap", 4)
        if type(tile) is not int or tile < 0:
            errors.append("MAGI-2 family_params.vae_decode_tile_size must be a nonnegative integer.")
        elif type(overlap) is not int or overlap < 1 or (tile and overlap >= tile):
            errors.append("MAGI-2 VAE tile overlap must be positive and smaller than an enabled tile size.")
        if vae_chunk is not None and (
            isinstance(vae_chunk, bool)
            or not isinstance(vae_chunk, int)
            or int(vae_chunk) < 0
        ):
            errors.append(
                "MAGI-2 family_params.vae_decode_chunk_size must be an integer >= 0."
            )
        swap_mode = params.get("refiner_block_swap_mode")
        if isinstance(swap_mode, str) and swap_mode.strip().lower() not in {
            "sync",
            "async",
        }:
            errors.append(
                "MAGI-2 family_params.refiner_block_swap_mode must be "
                "'sync' or 'async'."
            )
        subfolder = params.get("refiner_subfolder")
        # Both flavours are tested because the rule is about the value, not
        # about the host the config happens to be validated on.
        if isinstance(subfolder, str) and (
            PurePosixPath(subfolder).is_absolute()
            or PureWindowsPath(subfolder).is_absolute()
            or ".." in PurePosixPath(subfolder.replace("\\", "/")).parts
        ):
            errors.append(
                "MAGI-2 family_params.refiner_subfolder names a directory inside "
                "the snapshot root; it must be relative and must not traverse "
                "upwards."
            )
        return errors

    def generation_defaults(self) -> FamilyGenerationDefaults:
        from mirai.vendors.magi2_preview.pipeline.inference_engine import (
            NEGATIVE_PROMPT,
        )

        return FamilyGenerationDefaults(
            negative_prompt=NEGATIVE_PROMPT,
            steps=MAGI2_RELEASED_PREVIEW_STEPS,
            cfg_scale=MAGI2_RELEASED_PREVIEW_CFG_SCALE,
            scheduler=MAGI2_NATIVE_SOLVER,
            width=MAGI2_RELEASED_PREVIEW_WIDTH,
            height=MAGI2_RELEASED_PREVIEW_HEIGHT,
            frames=MAGI2_RELEASED_FRAMES,
        )

    def validate_native_backend_availability(self, cfg: Any) -> list[str]:
        root = Path(str(cfg.model.path)).expanduser()
        preview = root / "preview" if (root / "preview").is_dir() else root
        if not (preview / "model.safetensors.index.json").is_file():
            return [
                "MAGI-2 model.path must point to the Hugging Face snapshot root or "
                "its preview directory containing model.safetensors.index.json."
            ]
        return []

    def build_native_cache_encoder(
        self, config: NativeCacheEncoderConfig
    ) -> Any | None:
        if not bool(config.enabled):
            return None
        from mirai.core.dataset.native_encode import validate_native_cache_encoder
        from mirai.core.models.magi2_preview.cache import Magi2PreviewNativeCacheEncoder

        return validate_native_cache_encoder(
            Magi2PreviewNativeCacheEncoder(config),
            source="the magi2-preview model-family provider",
        )


_provider = Magi2PreviewModelFamilyProvider()
register_model_family_provider("magi2-preview", _provider)
register_model_family_provider("magi-2-preview", _provider)
