"""One-device adapter training and inference runtime for MAGI-2 Preview.

The architecture and checkpoint naming follow SandAI's Apache-2.0 reference
implementation vendored under :mod:`mirai.vendors.magi2_preview`. Heavy runtime
imports remain lazy so the rest of Mirai is usable without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize

from mirai.core.models.base import (
    MemoryFeatureCapabilities,
    ModelExtensionCapabilities,
    SyntheticBatchSpec,
)
from mirai.core.models.magi2_preview.refiner import (
    MAGI2_REFINER_OUTPUT_FPS,
    MAGI2_REFINER_SUBFOLDER,
    Magi2RefineSettings,
)
from mirai.core.models.native_video import NativeVideoPipeline, VideoLatentLayout
from mirai.core.models.providers import (
    ModelFamilyProvider,
    NativeCacheEncoderConfig,
    register_model_family_provider,
)
from mirai.core.moe.routing.contracts import SparseMoECapabilities
from mirai.core.training.residency.block_swap import BlockSwapManager
from mirai.core.training.residency.tensor_residency import (
    move_tensors_outside_modules,
    move_trainable_tensors,
)


# Hidden width of the Qwen3.5 text encoder shipped with the MAGI-2 snapshot;
# ``ModelConfig.text_in_channels`` in the released architecture JSON.
MAGI2_TEXT_EMBED_WIDTH = 5120

# The vendored ``TransformerBlock`` carries a single whole-block recompute flag,
# so the family implements only the "off" family and "standard".
MAGI2_GRADIENT_CHECKPOINTING_OFF = frozenset({"off", "false", "none", "0", ""})
MAGI2_GRADIENT_CHECKPOINTING_ON = frozenset({"standard", "true", "1"})

# The vendored sampler is Flow-UniPC and always evaluates the conditional and
# unconditional branches in one B=2 forward.
MAGI2_NATIVE_SOLVER = "unipc"
MAGI2_NATIVE_CFG_MODE = "batched"

# The audio placeholder is drawn from a family-owned stream rather than from the
# latent-noise stream the training loop seeds, so the two draws stay independent
# of each other's shapes and call counts.
MAGI2_AUDIO_NOISE_SEED_OFFSET = 20_011

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


def _magi2_request_frames(latent_frames: int) -> int:
    """Requested frames on the 25 fps timeline for ``latent_frames`` latents."""
    return 8 * (int(latent_frames) - 1) + 1


def _magi2_decoded_frames(latent_frames: int) -> int:
    """Physical frames the Turbo VAE decodes from ``latent_frames`` latents."""
    return 4 * (int(latent_frames) - 1) + 1


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
        delta = (self.lora_b @ self.lora_a).reshape(self.shape)
        return base + delta.to(dtype=base.dtype) * self.scale * self.runtime_scale


@dataclass(frozen=True)
class Magi2RuntimeOptions:
    config_path: str
    audio_tokens: int = -1
    refiner_config_path: str = ""
    refiner_subfolder: str = MAGI2_REFINER_SUBFOLDER


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

    # Class-level defaults so every policy seam answers identically before
    # __init__ has run, which is how the weightless contract probes reach them.
    _refiner: Any | None = None
    _refine_settings: "Magi2RefineSettings | None" = None
    _residency_request: "Magi2ResidencyRequest | None" = None

    def __init__(
        self,
        model_config: Any,
        memory_config: Any | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        family = dict(getattr(model_config.params, "family_params", {}) or {})
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
        )
        self.runtime_config, self.transformer, self.data_proxy = self._build_model()
        self._block_swap_manager: BlockSwapManager | None = None
        self._block_hook_handles: list[Any] = []
        self._gradient_checkpointing = "off"
        self._adapter_configured = False
        self._last_audio_prediction: torch.Tensor | None = None
        self._compute_autocast_dtype = torch.bfloat16
        self._lora_scale = 1.0
        self._text_encoder: Any | None = None
        self._vae: nn.Module | None = None
        self._inference_device = "cpu"
        self._audio_noise_generator = self._build_audio_noise_generator(seed)
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
        transformer = Transformer(config.arch_config, ep_size=1)
        state = load_magi2_model_state_dict(transformer, config.engine_config)
        transformer.load_state_dict(state, strict=True)
        del state
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
                str(path), device=device, precision=torch.bfloat16
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
            self._text_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        decoded = self._vae.decode(value.to(device=device, dtype=torch.bfloat16), output_offload=True)
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
            weight_residency_strategy=True,
            runtime_offload_flush=True,
            moe_kernel_backend=True,
        )

    def _transformer_device(self) -> torch.device | None:
        for parameter in self.transformer.parameters():
            return parameter.device
        return None

    def configure_moe_optimization_policy(self, policy: Any) -> None:
        from mirai.core.models.magi2_preview.grouped_moe import (
            Magi2GroupedMoEBackend,
            attach_grouped_moe_backend,
            resolve_magi2_moe_execution,
            validate_grouped_moe_backend_support,
        )

        plan = resolve_magi2_moe_execution(policy)
        if plan is not None:
            validate_grouped_moe_backend_support(
                plan, device=self._transformer_device()
            )
        attach_grouped_moe_backend(
            self.transformer,
            Magi2GroupedMoEBackend(plan) if plan is not None else None,
        )

    def preserves_native_parameter_dtypes(self) -> bool:
        return True

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
            emits_router_metrics=False,
        )

    def get_adapter_target_presets(self) -> dict[str, list[str]]:
        return {key: list(value) for key, value in MAGI2_TARGET_PRESETS.items()}

    def set_adapter_config(self, adapter_config: Any) -> None:
        if str(getattr(adapter_config, "type", "lora")).strip().lower() != "lora":
            raise ValueError("MAGI-2 Preview currently supports adapter.type='lora'.")
        preset = str(getattr(adapter_config, "target_preset", "attn_only")).strip()
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
                    owner_name.endswith(target) or full_name.endswith(target)
                    for target in targets
                ):
                    continue
                if parametrize.is_parametrized(module, tensor_name):
                    continue
                parameter.requires_grad_(False)
                parametrize.register_parametrization(
                    module,
                    tensor_name,
                    LowRankWeight(tuple(parameter.shape), rank=rank, alpha=alpha),
                    unsafe=False,
                )
                module.parametrizations[tensor_name].original.requires_grad_(False)
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
        expected = set(self.state_dict())
        supplied = {str(key) for key in state}
        unknown = sorted(supplied - expected)
        missing = sorted(expected - supplied)
        if unknown or missing:
            raise ValueError(
                "MAGI-2 adapter state does not match the configured LoRA targets "
                f"(missing={missing[:4]}, unknown={unknown[:4]})."
            )
        self.transformer.load_state_dict(state, strict=False)

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
        resolved = str(strategy).strip().lower()
        enabled = resolved != "disabled" and int(blocks_to_swap) > 0
        if enabled and not self._adapter_configured:
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

    def place_offloaded_modules(self, *, device: Any, strategy: str) -> None:
        manager = self._block_swap_manager
        if (
            manager is not None
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
        layers = [module for _, module in self.get_block_swap_units()]
        move_trainable_tensors(self.transformer, device=device)
        move_tensors_outside_modules(self.transformer, excluded_modules=layers, device=device)
        if self._block_swap_manager is None:
            self.transformer.to(device)
            return
        self._block_swap_manager.bind(self.get_block_swap_units(), device=device)
        for handle in self._block_hook_handles:
            handle.remove()
        self._block_hook_handles.clear()
        for index, layer in self.get_block_swap_units():
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
        audio_tokens = (
            (int(latents.shape[2]) - 1) * 8 + 1
            if configured_audio_tokens < 0
            else configured_audio_tokens
        )
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
        return video

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
    ) -> torch.Tensor:
        """Run the shipping joint video/audio Flow-UniPC sampler.

        The vendored sampler owns its schedule and its CFG execution, so the two
        requested policies are checked rather than applied: it implements
        Flow-UniPC only, and every step evaluates the conditional and
        unconditional branches together in one ``B=2`` forward.
        """
        requested_solver = str(solver_name).strip().lower()
        if requested_solver != MAGI2_NATIVE_SOLVER:
            raise ValueError(
                f"MAGI-2 Preview implements the '{MAGI2_NATIVE_SOLVER}' solver only; "
                f"got '{requested_solver}'. The native sampler is the vendored "
                "Flow-UniPC multistep scheduler and has no other schedule."
            )
        requested_cfg_mode = str(cfg_mode).strip().lower()
        if requested_cfg_mode != MAGI2_NATIVE_CFG_MODE:
            raise ValueError(
                f"MAGI-2 Preview runs '{MAGI2_NATIVE_CFG_MODE}' CFG only; got "
                f"'{requested_cfg_mode}'. The native sampler packs the conditional "
                "and unconditional branches into one B=2 forward per step."
            )
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
        audio_tokens = (
            (int(noise.shape[1]) - 1) * 8 + 1
            if int(self.options.audio_tokens) < 0
            else int(self.options.audio_tokens)
        )
        audio_latent = torch.randn(
            1,
            audio_tokens,
            64,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
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

        sampler.forward = resident_forward
        def release_step_memory() -> None:
            self.flush_runtime_offloads()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        sampler.step_callback = release_step_memory
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
        return sampled_video[0]

    # -- refiner stage -----------------------------------------------------
    def _refiner_assets(self) -> Any:
        if self._refiner is None:
            from mirai.core.models.magi2_preview.refiner import Magi2Refiner

            self._refiner = Magi2Refiner(
                self.model_config,
                config_path=self.options.refiner_config_path,
                subfolder=self.options.refiner_subfolder,
            )
        return self._refiner

    def supports_refiner(self) -> bool:
        """MAGI-2 implements the refiner stage; assets are checked separately."""
        return True

    def has_refiner_weights(self) -> bool:
        return bool(self._refiner_assets().has_weights())

    def refiner_residency_request(self) -> Any:
        """The block-residency policy the refiner stage should stream under."""
        return self._residency_request

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
        if self._block_swap_manager is not None:
            self._block_swap_manager.finish_backward()
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
    ) -> Any:
        from mirai.core.models.magi2_preview.refiner import run_refine

        if self._refine_settings is None:
            raise RuntimeError(
                "MAGI-2 refinement must be validated before it runs; "
                "validate_refinement_request() resolves the release profile."
            )
        _ = request
        return run_refine(
            pipeline=self,
            refiner=self._refiner_assets(),
            base_latent=torch.as_tensor(base_latent),
            settings=self._refine_settings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            device=str(device),
            dtype=dtype,
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
            )
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

    def train(self, mode: bool = True):
        self.transformer.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        full = self.transformer.state_dict(*args, **kwargs)
        return {key: value for key, value in full.items() if "parametrizations." in key and (key.endswith("lora_a") or key.endswith("lora_b"))}

    def load_state_dict(self, state: dict[str, Any], strict: bool = True):
        """Load the adapter surface reported by :meth:`state_dict`.

        The payload is applied to the wrapped transformer non-strictly because it
        never carries the frozen base tensors. ``strict`` is enforced against the
        adapter surface itself: a payload that does not cover exactly the
        configured LoRA parameters is a lineage mismatch and fails.
        """
        if strict:
            expected = set(self.state_dict())
            supplied = {str(key) for key in state}
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            if missing or unexpected:
                raise ValueError(
                    "MAGI-2 adapter state does not match the configured LoRA "
                    f"targets (missing={missing[:4]}, unexpected={unexpected[:4]}). "
                    "Pass strict=False to load a partial adapter surface."
                )
        return self.transformer.load_state_dict(state, strict=False)


class Magi2PreviewModelFamilyProvider(ModelFamilyProvider):
    _ALLOWED_FAMILY_PARAMS = {
        "config_path",
        "audio_tokens",
        "refiner_config_path",
        "refiner_subfolder",
    }

    def __init__(self, model_type: str = "magi2-preview") -> None:
        super().__init__(
            model_type=model_type,
            native=True,
            sparse_moe=True,
            strict_native_assets_by_default=True,
            batched_cfg_inference=True,
            # Raw media is encoded by the family-owned Qwen3.5/Wan2.2 encoder
            # returned from build_native_cache_encoder(); no generic pixel
            # projection is a valid MAGI-2 latent.
            native_cache_encoding=True,
            config_defaults_name="magi2_preview",
            inference_tasks=("text_to_video",),
            dataset_caption_formats=("raw",),
            pipeline_type=Magi2PreviewPipeline,
        )

    def validate_family_params(self, params: dict[str, Any]) -> list[str]:
        unknown = sorted(set(params) - self._ALLOWED_FAMILY_PARAMS)
        errors = [f"Unknown MAGI-2 family parameter '{name}'." for name in unknown]
        if "audio_tokens" in params and int(params["audio_tokens"]) < -1:
            errors.append("MAGI-2 family_params.audio_tokens must be -1 (auto) or >= 0.")
        for name in ("refiner_config_path", "refiner_subfolder"):
            value = params.get(name)
            if value is not None and not isinstance(value, str):
                errors.append(f"MAGI-2 family_params.{name} must be a string.")
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
