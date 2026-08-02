"""Native sparse-MoE pipeline used by executable CPU contracts only.

The family is available solely through explicit ``provider_module`` selection;
it is not a production model provider.
"""

from __future__ import annotations

from typing import Any

from mirai.config.schema import ModelConfig, TrainingConfig
from mirai.core.models.adapters.lora_parameter_dropout import (
    validate_lora_parameter_dropout,
)
from mirai.core.models.base import (
    MemoryFeatureCapabilities,
    ModelExtensionCapabilities,
)
from mirai.core.models.flow import apply_rectified_flow_noise, rectified_flow_target
from mirai.core.models.native_video import NativeVideoPipeline, VideoLatentLayout
from mirai.core.dataset.native_encode import ValidationCacheEncoder
from mirai.core.models.providers import (
    ModelFamilyProvider,
    NativeCacheEncoderConfig,
    register_model_family_provider,
)
from mirai.core.moe.routing.contracts import RoutingStats, SparseMoECapabilities
from mirai.core.moe.monitoring.summary import summarize_routing_stats
from mirai.core.moe.adaptation.balance import resolve_moe_balance_weights
from mirai.core.moe.routing.layers import SparseMoEFeedForward
from mirai.core.registry import register_model
from mirai.core.tensors import is_torch_tensor
from mirai.core.training.residency.block_swap import BlockSwapManager

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


VALID_VARIANTS = {"tiny-video", "tiny-video-moe"}


class SparseMoETransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        shared_experts: int,
        experts_per_token: int,
        layer_name: str,
        load_balance_weight: float,
        router_z_loss_weight: float,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("SparseMoETransformerBlock requires torch.")
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.moe = SparseMoEFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_routed_experts=num_experts,
            num_shared_experts=shared_experts,
            experts_per_token=experts_per_token,
            layer_name=layer_name,
            load_balance_weight=load_balance_weight,
            router_z_loss_weight=router_z_loss_weight,
        )

    def forward(self, hidden_states: Any) -> tuple[Any, Any, RoutingStats]:
        out = self.moe(self.norm(hidden_states))
        return hidden_states + out.hidden_states, out.auxiliary_loss, out.stats


class TinySparseMoEDenoiser(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("TinySparseMoEDenoiser requires torch.")
        super().__init__()
        params = config.params
        hidden_size = int(params.hidden_size)
        intermediate_size = hidden_size * 4
        # moe_balance_mode gates the effective aux-loss weight: bias_only/off
        # drop the load-balance term (this test model has no online bias, so
        # both zero the aux weight). Default aux_loss leaves it unchanged.
        effective_load_balance_weight, _ = resolve_moe_balance_weights(
            getattr(params, "moe_balance_mode", "aux_loss"),
            aux_loss_weight=float(params.moe_aux_loss_weight),
            bias_update_rate=float(params.moe_bias_update_rate),
        )
        self.input_proj = nn.Linear(1, hidden_size)
        self.timestep_proj = nn.Linear(1, hidden_size)
        self.text_proj = nn.Linear(1, hidden_size)
        self.blocks = nn.ModuleList(
            [
                SparseMoETransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=int(params.num_experts),
                    shared_experts=int(params.shared_experts),
                    experts_per_token=int(params.experts_per_token),
                    layer_name=f"blocks.{idx}",
                    load_balance_weight=float(effective_load_balance_weight),
                    router_z_loss_weight=float(params.moe_router_z_loss_weight),
                )
                for idx in range(int(params.num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, 1)

    @staticmethod
    def _text_scalar(text_embeds: dict[str, Any], *, batch: int, like: Any) -> Any:
        value = text_embeds.get("t5", None)
        device = like.device
        dtype = like.dtype
        if is_torch_tensor(value):
            tensor = value.to(device=device, dtype=dtype)
            if tensor.ndim == 0:
                return tensor.reshape(1, 1).expand(batch, 1)
            if tensor.shape[0] != batch:
                return tensor.reshape(1, -1).mean(dim=1, keepdim=True).expand(batch, 1)
            return tensor.reshape(batch, -1).mean(dim=1, keepdim=True)
        if isinstance(value, (list, tuple)):
            tensor = torch.tensor(value, device=device, dtype=dtype)
            if tensor.ndim == 1 and tensor.numel() == batch:
                return tensor.reshape(batch, 1)
            return tensor.reshape(1, -1).mean(dim=1, keepdim=True).expand(batch, 1)
        return torch.zeros((batch, 1), device=device, dtype=dtype)

    def embed_inputs(
        self, noisy_latents: Any, timesteps: Any, text_embeds: dict[str, Any]
    ) -> tuple[Any, Any]:
        batch = int(noisy_latents.shape[0])
        original_shape = noisy_latents.shape
        dtype = self.input_proj.weight.dtype
        tokens = noisy_latents.reshape(batch, -1, 1).to(dtype=dtype)
        hidden = self.input_proj(tokens)
        timestep = timesteps.to(device=noisy_latents.device, dtype=dtype).reshape(
            batch, 1
        )
        hidden = hidden + self.timestep_proj(timestep).unsqueeze(1)
        hidden = hidden + self.text_proj(
            self._text_scalar(text_embeds, batch=batch, like=hidden)
        ).unsqueeze(1)
        return hidden, original_shape

    def forward(
        self,
        noisy_latents: Any,
        timesteps: Any,
        text_embeds: dict[str, Any],
    ) -> tuple[Any, Any, list[RoutingStats]]:
        hidden, original_shape = self.embed_inputs(
            noisy_latents, timesteps, text_embeds
        )
        aux_loss = None
        stats: list[RoutingStats] = []
        for block in self.blocks:
            hidden, block_aux, block_stats = block(hidden)
            aux_loss = block_aux if aux_loss is None else aux_loss + block_aux
            stats.append(block_stats)
        output = self.output_proj(self.output_norm(hidden)).reshape(original_shape)
        zero_touch = output.sum() * 0.0
        for param in self.parameters():
            zero_touch = zero_touch + (param.sum() * 0.0)
        output = output + zero_touch
        if aux_loss is None:
            aux_loss = output.sum() * 0.0
        else:
            aux_loss = aux_loss + zero_touch
        return output, aux_loss, stats


@register_model("sparse_moe_test")
class SparseMoETestPipeline(nn.Module, NativeVideoPipeline):
    """Small native sparse-MoE denoiser used to validate trainer semantics."""

    REQUIRED_BATCH_KEYS_BY_STRATEGY = {"image_to_video": ["clip_embed"]}

    def __init__(self, model_config: ModelConfig) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("SparseMoETestPipeline requires torch.")
        super().__init__()
        self.model_config = model_config
        self.model = TinySparseMoEDenoiser(model_config)
        self.lora_a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("lora_alpha", torch.tensor(1.0, dtype=torch.float32))
        self._lora_scale = 1.0
        self._rank_dropout = 0.0
        self._lora_parameter_dropout = 0.0
        self._rank_schedule_scale = 1.0
        self._adapter_merged = False
        self._vae_loaded = False
        self._text_encoder_loaded = False
        self._block_swap_manager: BlockSwapManager | None = None
        self._weight_residency_strategy = "disabled"
        self._block_swap_state: dict[str, Any] = {
            "enabled": False,
            "blocks_to_swap": 0,
            "mode": "sync",
            "block_swap_backward": True,
            "weight_residency_strategy": "disabled",
            "events": [],
        }
        self._block_placements = ["meta" for _ in range(int(model_config.params.num_layers))]
        self._stashed_cpu_state: dict[str, Any] = {}
        self._gradient_checkpointing = "off"
        self._last_auxiliary_losses: dict[str, Any] = {}
        self._last_diagnostics: dict[str, Any] = {}

    def get_adapter_target_presets(self) -> dict[str, list[str]]:
        # The toy scalar adapter modulates the prediction head only. Both
        # preset names exist for config compatibility (schema default is
        # "attn_only") and truthfully resolve to that single surface, so
        # exported adapter metadata carries non-empty target_modules.
        return {"attn_only": ["output_proj"], "full": ["output_proj"]}

    def validate_adapter_artifact_lineage(
        self,
        *,
        dataset_snapshot_id: str,
        model_snapshot_id: str,
        config_snapshot_id: str,
    ) -> None:
        """The testbed has no adapter-shaping artifact to validate."""
        _ = dataset_snapshot_id, model_snapshot_id, config_snapshot_id

    def get_video_latent_layout(self) -> VideoLatentLayout:
        return VideoLatentLayout(
            latent_channels=int(self.model_config.params.latent_channels),
            temporal_downsample=1,
            spatial_downsample=8,
        )

    def apply_noise(self, clean_latents: Any, noise: Any, timesteps: Any) -> Any:
        return apply_rectified_flow_noise(
            clean_latents=clean_latents,
            noise=noise,
            timesteps=timesteps,
            shift=float(self.model_config.params.flow_shift),
            timestep_eps=1e-5,
        )

    def compute_target(self, noise: Any, clean_latents: Any, timesteps: Any) -> Any:
        _ = timesteps
        return rectified_flow_target(noise=noise, clean_latents=clean_latents)

    def forward(
        self,
        noisy_latents: Any,
        timesteps: Any,
        text_embeds: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        _ = kwargs
        prediction, aux_loss, stats = self._model_forward(noisy_latents, timesteps, text_embeds)
        adapter_scale = float(self._lora_scale) * float(self._rank_schedule_scale)
        if bool(self.training):
            adapter_scale *= max(0.0, 1.0 - float(self.model_config.params.lora_dropout))
            adapter_scale *= max(0.0, 1.0 - float(self._rank_dropout))
        adapter_delta = noisy_latents * self.lora_a * self.lora_b * self.lora_alpha
        if bool(self.training) and self._lora_parameter_dropout > 0.0:
            keep = 1.0 - float(self._lora_parameter_dropout)
            input_mask = torch.empty(
                (), device=adapter_delta.device, dtype=adapter_delta.dtype
            ).bernoulli_(keep)
            output_mask = torch.empty(
                (), device=adapter_delta.device, dtype=adapter_delta.dtype
            ).bernoulli_(keep)
            adapter_delta = adapter_delta * input_mask * output_mask
        prediction = prediction + (adapter_delta * adapter_scale)
        self._last_auxiliary_losses = {"moe_router_aux": aux_loss}
        self._last_diagnostics = {"moe_routing": summarize_routing_stats(stats)}
        return prediction

    def _model_forward(
        self,
        noisy_latents: Any,
        timesteps: Any,
        text_embeds: dict[str, Any],
    ) -> tuple[Any, Any, list[RoutingStats]]:
        manager = self._block_swap_manager
        if manager is None:
            return self._checkpointed_model_forward(noisy_latents, timesteps, text_embeds)
        manager.reset_step()
        return self._checkpointed_model_forward(
            noisy_latents,
            timesteps,
            text_embeds,
            block_manager=manager,
        )

    def _checkpointed_model_forward(
        self,
        noisy_latents: Any,
        timesteps: Any,
        text_embeds: dict[str, Any],
        *,
        block_manager: BlockSwapManager | None = None,
    ) -> tuple[Any, Any, list[RoutingStats]]:
        if self._gradient_checkpointing in {"off", "false", "none", "0"} and block_manager is None:
            return self.model(noisy_latents, timesteps, text_embeds)

        hidden, original_shape = self.model.embed_inputs(
            noisy_latents, timesteps, text_embeds
        )

        use_checkpoint = self._gradient_checkpointing not in {"off", "false", "none", "0"}
        aggressive = self._gradient_checkpointing == "aggressive"
        aux_loss = None
        stats: list[RoutingStats] = []
        for idx, block in enumerate(self.model.blocks):
            if block_manager is not None:
                block_manager.before_block(idx)
            if use_checkpoint:
                import torch.utils.checkpoint as checkpoint

                def run_hidden(input_hidden: Any, *, current_block=block) -> Any:
                    return current_block(input_hidden)[0]

                hidden = checkpoint.checkpoint(run_hidden, hidden, use_reentrant=False)
                if aggressive:
                    hidden = checkpoint.checkpoint(lambda value: value, hidden, use_reentrant=False)
                _, block_aux, block_stats = block(hidden)
            else:
                hidden, block_aux, block_stats = block(hidden)
            if block_manager is not None:
                block_manager.after_block(idx)
            aux_loss = block_aux if aux_loss is None else aux_loss + block_aux
            stats.append(block_stats)
        output = self.model.output_proj(self.model.output_norm(hidden)).reshape(original_shape)
        zero_touch = output.sum() * 0.0
        for param in self.model.parameters():
            zero_touch = zero_touch + (param.sum() * 0.0)
        output = output + zero_touch
        if aux_loss is None:
            aux_loss = output.sum() * 0.0
        else:
            aux_loss = aux_loss + zero_touch
        return output, aux_loss, stats

    def validate_config(self, config: TrainingConfig) -> list[str]:
        params = config.model.params
        errors: list[str] = []
        if params.variant not in VALID_VARIANTS:
            errors.append(
                f"Unsupported sparse MoE test variant '{params.variant}'. "
                f"Expected one of: {', '.join(sorted(VALID_VARIANTS))}."
            )
        if int(params.num_experts) <= 1:
            errors.append("model.params.num_experts must be > 1 for sparse MoE.")
        if int(params.experts_per_token) <= 0:
            errors.append("model.params.experts_per_token must be > 0.")
        if int(params.experts_per_token) > int(params.num_experts):
            errors.append("model.params.experts_per_token must be <= num_experts.")
        if int(params.shared_experts) < 0:
            errors.append("model.params.shared_experts must be >= 0.")
        if int(params.hidden_size) <= 0:
            errors.append("model.params.hidden_size must be > 0.")
        if int(params.num_layers) <= 0:
            errors.append("model.params.num_layers must be > 0.")
        if float(params.moe_aux_loss_weight) < 0.0:
            errors.append("model.params.moe_aux_loss_weight must be >= 0.")
        if float(params.moe_router_z_loss_weight) < 0.0:
            errors.append("model.params.moe_router_z_loss_weight must be >= 0.")
        return errors

    def get_trainable_parameters(self):
        return [*self.model.parameters(), self.lora_a, self.lora_b]

    def get_named_trainable_parameters(self):
        return [
            *self.model.named_parameters(),
            ("lora_a", self.lora_a),
            ("lora_b", self.lora_b),
        ]

    def get_model_extension_capabilities(self) -> ModelExtensionCapabilities:
        return ModelExtensionCapabilities(
            adapter_runtime_controls=True,
            adapter_merge_unmerge=True,
            rank_schedule_progress=True,
            preview_latent_decode=True,
            validation_inference=True,
        )

    def get_memory_feature_capabilities(self) -> MemoryFeatureCapabilities:
        return MemoryFeatureCapabilities(
            block_swap=True,
            weight_residency_strategy=True,
            runtime_offload_flush=True,
        )

    def get_sparse_moe_capabilities(self) -> SparseMoECapabilities:
        params = self.model_config.params
        return SparseMoECapabilities(
            is_sparse_moe=True,
            architecture="tiny_video_moe",
            routing="token_choice_top_k",
            routing_granularity="token",
            num_routed_experts=int(params.num_experts),
            num_shared_experts=int(params.shared_experts),
            num_activated_experts=int(params.experts_per_token),
            emits_router_metrics=True,
            notes=("Executable sparse-MoE trainer contract fixture.",),
        )

    def get_training_auxiliary_losses(self) -> dict[str, Any]:
        return dict(self._last_auxiliary_losses)

    def get_training_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def set_gradient_checkpointing(self, enabled: bool | str) -> None:
        if enabled is True:
            self._gradient_checkpointing = "standard"
        elif enabled is False:
            self._gradient_checkpointing = "off"
        else:
            self._gradient_checkpointing = str(enabled).strip().lower()

    def set_lora_scale(self, scale: float) -> None:
        self._lora_scale = float(scale)

    def get_lora_scale(self) -> float:
        return float(self._lora_scale)

    def set_adapter_runtime(
        self, *, rank_dropout: float, lora_parameter_dropout: float
    ) -> None:
        value = float(rank_dropout)
        if value < 0.0 or value > 1.0:
            raise ValueError("adapter.rank_dropout must be in [0, 1].")
        self._rank_dropout = value
        self._lora_parameter_dropout = validate_lora_parameter_dropout(
            lora_parameter_dropout
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
        current = int(step)
        floor = float(min_scale)
        if end <= start:
            self._rank_schedule_scale = 1.0
            return
        progress = min(1.0, max(0.0, (current - start) / float(end - start)))
        self._rank_schedule_scale = floor + ((1.0 - floor) * progress)

    def get_rank_schedule_scale(self) -> float:
        return float(self._rank_schedule_scale)

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
        _ = offload_dir
        resolved = str(strategy).strip().lower() or "disabled"
        block_count = len(self.model.blocks)
        swap_count = max(0, min(int(blocks_to_swap), block_count))
        enabled = bool(resolved != "disabled" and swap_count > 0)
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
            )
            if enabled
            else None
        )
        self._block_swap_state = {
            "enabled": enabled,
            "blocks_to_swap": swap_count,
            "mode": str(mode).strip().lower(),
            "block_swap_backward": bool(block_swap_backward),
            "weight_residency_strategy": resolved,
            "events": [],
        }

    def get_block_swap_units(self) -> list[tuple[int, Any]]:
        return [(idx, block) for idx, block in enumerate(self.model.blocks)]

    def get_block_swap_state(self) -> dict[str, Any]:
        state = dict(self._block_swap_state)
        manager = self._block_swap_manager
        if manager is not None:
            state.update(manager.snapshot())
            state["enabled"] = True
            state["weight_residency_strategy"] = self._weight_residency_strategy
        return state

    def flush_runtime_offloads(self) -> None:
        if self._block_swap_manager is not None:
            self._block_swap_manager.reset_step()

    def merge_adapter(self) -> bool:
        self._adapter_merged = True
        return True

    def unmerge_adapter(self) -> bool:
        was_merged = bool(self._adapter_merged)
        self._adapter_merged = False
        return was_merged

    def is_adapter_merged(self) -> bool:
        return bool(self._adapter_merged)

    def load_vae_to_gpu(self) -> float:
        self._vae_loaded = True
        return self.get_vae_scaling_factor()

    def unload_vae_from_gpu(self) -> None:
        self._vae_loaded = False

    def is_vae_loaded(self) -> bool:
        return bool(self._vae_loaded)

    # ---- Deterministic validation inference hooks (text encoder + VAE) -----
    # Concrete overrides of the model-agnostic inference contract declared on
    # ``NativeVideoPipeline`` (load_text_encoder/encode_prompt/offload_text_encoder,
    # load_vae/decode_latents_native/offload_vae). They exercise inference,
    # preview, and evaluation contracts end-to-end on CPU for this test family.
    #
    # They are deterministic validation fixtures, not a generative model:
    #   * the text encoder is a hash stand-in whose reduced scalar matches the
    #     family's asset-free cache text encoding (``_hash_scalar``), preserving
    #     inference/cache embedding parity;
    #   * the "VAE" is a fixed learned-free latent->pixel projection.
    # ``has_native_inference()`` stays False so this family resolves to the
    # explicit validation-fixture mode and cannot enter release evidence.

    def load_text_encoder(self, *, device: str) -> None:
        _ = device
        self._text_encoder_loaded = True

    def offload_text_encoder(self) -> None:
        self._text_encoder_loaded = False

    def encode_prompt(self, prompt: str, *, device: str) -> Any:
        from mirai.core.dataset.native_encode import _hash_scalar

        value = float(_hash_scalar(str(prompt)))
        return torch.tensor([[value]], dtype=torch.float32, device=str(device))

    def load_vae(self, *, device: str) -> None:
        _ = device
        self._vae_loaded = True

    def offload_vae(self) -> None:
        self._vae_loaded = False

    def decode_latents_native(self, latents: list[Any]) -> Any:
        if torch is None:  # pragma: no cover
            raise RuntimeError("decode_latents_native requires torch.")
        tensor = latents[0] if isinstance(latents, (list, tuple)) else latents
        if tensor is None:
            raise ValueError("decode_latents_native requires a latent tensor.")
        x = (
            tensor.detach().float()
            if is_torch_tensor(tensor)
            else torch.as_tensor(tensor, dtype=torch.float32)
        )
        if x.ndim == 5:
            x = x[0]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        while x.ndim < 4:
            x = x.unsqueeze(0)
        if x.ndim > 4:
            x = x.reshape(int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), -1)
        channels, frames = int(x.shape[0]), int(x.shape[1])
        h_lat, w_lat = int(x.shape[2]), int(x.shape[3])
        # Fixed, seed-independent channel->RGB mixing (learned-free projection),
        # built on the latent's device so GPU-resident inference decodes too.
        idx_c = torch.arange(channels, dtype=torch.float32, device=x.device).reshape(1, channels)
        phases = torch.tensor(
            [[0.0], [2.0943951], [4.1887902]], dtype=torch.float32, device=x.device
        )  # 0, 2*pi/3, 4*pi/3
        weight = torch.cos(idx_c * 0.7 + phases) / float(max(1, channels))  # (3, C)
        flat = x.reshape(channels, frames * h_lat * w_lat)  # (C, N)
        rgb = (weight @ flat).reshape(3, frames, h_lat, w_lat)
        rgb = rgb.permute(1, 0, 2, 3).contiguous()  # (T, 3, H, W)
        upscale = max(1, int(self.get_video_latent_layout().spatial_downsample))
        if upscale > 1:
            rgb = rgb.repeat_interleave(upscale, dim=-2).repeat_interleave(upscale, dim=-1)
        return torch.sigmoid(rgb)

    def decode_latents(
        self,
        latents: Any,
        *,
        frame_count: int,
        height: int,
        width: int,
        chunk_size: int,
        temporal_overlap: int = 0,
    ) -> tuple[Any, int]:
        _ = temporal_overlap
        value = float(latents.detach().float().mean().item()) if is_torch_tensor(latents) else float(latents)
        frames = torch.zeros(
            (max(1, int(frame_count)), 3, max(1, int(height)), max(1, int(width))),
            dtype=torch.float32,
        )
        for idx in range(int(frames.shape[0])):
            frames[idx].fill_((((value * 127.0) + idx * 3.0) % 255.0) / 255.0)
        chunk = max(1, int(chunk_size))
        chunk_count = max(1, (int(frames.shape[0]) + chunk - 1) // chunk)
        return frames, chunk_count

    def meta_loading_status(self) -> dict[str, Any]:
        return {
            "meta_initialized": True,
            "block_placements": list(self._block_placements),
            "stashed_cpu_keys": sorted(self._stashed_cpu_state),
        }

    def materialize_blocks(self, *, gpu_resident_blocks: int) -> dict[str, int]:
        total = len(self._block_placements)
        resident = max(0, min(int(gpu_resident_blocks), total))
        self._block_placements = [
            "cuda" if idx < resident else "cpu_pinned"
            for idx in range(total)
        ]
        return {
            "gpu_resident_blocks": resident,
            "cpu_pinned_blocks": max(0, total - resident),
        }

    def get_block_placements(self) -> list[str]:
        return list(self._block_placements)

    def stash_state_dict_on_cpu(self, state: dict[str, Any]) -> None:
        self._stashed_cpu_state = dict(state)

    def load_adapter_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("Adapter state must be a dictionary.")
        if "lora_a" not in state or "lora_b" not in state:
            raise ValueError("Adapter state must contain lora_a and lora_b.")
        with torch.no_grad():
            self.lora_a.copy_(self._scalar_adapter_tensor(state["lora_a"]))
            self.lora_b.copy_(self._scalar_adapter_tensor(state["lora_b"]))
            if "lora_alpha" in state:
                self.lora_alpha.copy_(self._scalar_adapter_tensor(state["lora_alpha"]))

    def _scalar_adapter_tensor(self, value: Any) -> Any:
        if is_torch_tensor(value):
            return value.detach().to(device=self.lora_a.device, dtype=self.lora_a.dtype).reshape(-1).mean()
        return torch.tensor(float(value), device=self.lora_a.device, dtype=self.lora_a.dtype)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if args or kwargs:
            return super().state_dict(*args, **kwargs)
        return {
            "model": self.model.state_dict(),
            "lora_a": self.lora_a.detach().clone(),
            "lora_b": self.lora_b.detach().clone(),
            "lora_alpha": self.lora_alpha.detach().clone(),
            "model_type": "sparse_moe_test",
        }

    def load_state_dict(self, state: dict[str, Any], strict: bool = True):
        payload = state.get("model") if isinstance(state, dict) else None
        if isinstance(payload, dict):
            result = self.model.load_state_dict(payload, strict=bool(strict))
            if "lora_a" in state or "lora_b" in state:
                self.load_adapter_state(state)
            return result
        if isinstance(state, dict) and ("lora_a" in state or "lora_b" in state):
            self.load_adapter_state(state)
            return None
        return super().load_state_dict(state, strict=bool(strict))


class SparseMoETestProvider(ModelFamilyProvider):
    """Provider for isolated, asset-free executable contracts."""

    def build_native_cache_encoder(
        self,
        config: NativeCacheEncoderConfig,
    ) -> ValidationCacheEncoder:
        return ValidationCacheEncoder(
            enabled=config.enabled,
            model_type=config.model_type,
            variant=config.variant,
            model_path=config.model_path,
            dtype_name=config.dtype_name,
            max_frames=config.max_frames,
            denoiser_subfolder=config.denoiser_subfolder,
            strict_assets=config.strict_assets,
            text_encoder_path=config.text_encoder_path,
            enable_bucketing=config.enable_bucketing,
            resolution_buckets=config.resolution_buckets,
            frame_buckets=config.frame_buckets,
            bucket_resize_mode=config.bucket_resize_mode,
        )


register_model_family_provider(
    "sparse_moe_test",
    SparseMoETestProvider(
        model_type="sparse_moe_test",
        native=True,
        sparse_moe=True,
        strict_native_assets_by_default=False,
        native_cache_encoding=True,
        asset_free_cache_encoding=True,
        dora_supported=False,
        config_defaults_name="moe",
        release_supported=False,
        release_eligible=False,
        batched_cfg_inference=True,
    ),
)
