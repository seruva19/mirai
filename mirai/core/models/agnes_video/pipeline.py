"""Native Agnes provider and joint audio-video adapter training."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import torch
from torch import nn

from mirai.core.models.base import MemoryFeatureCapabilities, ModelExtensionCapabilities, SyntheticBatchSpec
from mirai.core.models.native_video import NativeVideoPipeline, VideoLatentLayout
from mirai.core.models.providers import ModelFamilyProvider, register_model_family_provider
from mirai.core.models.agnes_video.checkpoints import iter_agnes_safetensors, load_agnes_pt_checkpoint
from mirai.core.models.agnes_video.lora import AgnesLoRALinear, ModalityLoRALinear, adapter_modules
from mirai.core.models.agnes_video.packing import pack_inputs
from mirai.vendors.agnes_video.dit import AgnesDiT, AgnesDiTConfig, BaseLinear, SparseMoEMLP
from mirai.core.training.residency.block_swap import BlockSwapManager
from mirai.core.training.residency.device_placement import WeightResidencyExecutionMode
from mirai.core.training.residency.tensor_residency import move_tensors_outside_modules, move_trainable_tensors


TARGET_PRESETS = {
    "attn_only": ("linear_qkv", "linear_proj"),
    "attn_mlp": ("linear_qkv", "linear_proj", "up_gate_proj", "down_proj"),
    "all": ("linear_qkv", "linear_proj", "up_gate_proj", "down_proj",
            "video_embedder", "audio_embedder", "text_embedder",
            "final_linear_video", "final_linear_audio"),
}


# The released Agnes inference pipeline uses this family-owned conditioning for
# classifier-free guidance when the caller does not supply a negative prompt.
AGNES_DEFAULT_NEGATIVE_PROMPT_PATH = (
    Path(__file__).resolve().parent / "default_negative_prompt.txt"
)
_AGNES_DEFAULT_NEGATIVE_PROMPT: str | None = None


def agnes_default_negative_prompt() -> str:
    """Read the released Agnes negative prompt once and cache the text."""
    global _AGNES_DEFAULT_NEGATIVE_PROMPT
    if _AGNES_DEFAULT_NEGATIVE_PROMPT is None:
        _AGNES_DEFAULT_NEGATIVE_PROMPT = AGNES_DEFAULT_NEGATIVE_PROMPT_PATH.read_text(
            encoding="utf-8"
        ).strip()
    return _AGNES_DEFAULT_NEGATIVE_PROMPT


class AgnesVideoPipeline(nn.Module, NativeVideoPipeline):
    REQUIRED_BATCH_KEYS_BY_STRATEGY = {"text_to_video": ["audio_latents"], "image_to_video": ["audio_latents"]}

    def __init__(self, model_config: Any, memory_config: Any = None, *, seed: int = 0,
                 transformer: AgnesDiT | None = None) -> None:
        super().__init__()
        self.model_config = model_config
        self.options = dict(getattr(model_config.params, "family_params", {}) or {})
        self._lora_scale = 1.0
        self._adapter_spec: dict[str, Any] | None = None
        self._assets = None
        self._last_diagnostics: dict[str, Any] = {}
        self._seed = seed
        self._block_swap_manager: BlockSwapManager | None = None
        self._block_hook_handles: list[Any] = []
        self._weight_residency_execution_mode = WeightResidencyExecutionMode.TRAINING
        from mirai.core.training.residency.device_placement import resolve_compute_dtype
        from types import SimpleNamespace
        requested = getattr(model_config, "dtype", "bf16")
        self._compute_dtype = resolve_compute_dtype(SimpleNamespace(model=SimpleNamespace(dtype=requested)))
        if transformer is None:
            self.runtime_config = AgnesDiTConfig(params_dtype=self._compute_dtype)
            with torch.device("meta"):
                transformer = AgnesDiT(self.runtime_config)
            path = Path(model_config.path)
            if path.is_dir() or path.name.endswith(".index.json"):
                expected = set(transformer.state_dict())
                loaded = set()
                for name, tensor in iter_agnes_safetensors(path):
                    if name not in expected:
                        raise ValueError(f"Unknown Agnes checkpoint tensor: {name}.")
                    transformer.load_state_dict({name: tensor}, strict=False, assign=True)
                    loaded.add(name)
                if loaded != expected:
                    raise ValueError(f"Missing Agnes checkpoint tensors: {sorted(expected-loaded)[:4]}.")
            else:
                state = load_agnes_pt_checkpoint(path)
                transformer.load_state_dict(state, strict=True, assign=True)
                del state
        else:
            self.runtime_config = transformer.config
        self.transformer = transformer
        self.set_compute_autocast_dtype(self._compute_dtype)
        self.transformer.requires_grad_(False)
        self._configure_attention_backend()

    def _configure_attention_backend(self):
        from mirai.core.models.attention_backends import normalize_attention_backend
        backend = normalize_attention_backend(getattr(self.model_config, "attention_backend", "auto"))
        for layer in self.transformer.block.layers:
            layer.attention.backend = backend

    def set_compute_autocast_dtype(self, dtype):
        from dataclasses import replace
        if dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError("Agnes compute dtype must be BF16, FP16, or FP32.")
        self._compute_dtype = dtype
        self.runtime_config = replace(self.runtime_config, params_dtype=dtype)
        self.transformer.config = self.runtime_config
        self.transformer.adapter.config = self.runtime_config

    @classmethod
    def from_training_config(cls, config):
        return cls(config.model, config.memory, seed=int(config.training.seed))

    def get_training_model(self):
        return self.transformer

    def preserves_native_parameter_dtypes(self):
        return True

    @torch.no_grad()
    def prepare_inference_runtime(self) -> None:
        # The released entry casts the full base DiT before inference, including
        # norm affine parameters and RoPE bands. Adapter matrices remain FP32.
        for parameter in self.transformer.parameters():
            if not parameter.requires_grad and parameter.is_floating_point():
                parameter.data = parameter.data.to(dtype=self._compute_dtype)
        for module in self.transformer.modules():
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.is_floating_point() and name != "lora_alpha":
                    setattr(module, name, buffer.to(dtype=self._compute_dtype))
        for layer in self.transformer.block.layers:
            layer.attention.released_inference_attention = True
            if isinstance(layer.mlp, SparseMoEMLP):
                layer.mlp.released_inference_grouped = True

    def get_memory_feature_capabilities(self):
        return MemoryFeatureCapabilities(block_swap=True, weight_residency_strategy=True,
                                         runtime_offload_flush=True)

    def get_block_swap_units(self):
        return list(enumerate(self.transformer.block.layers))

    def set_weight_residency_strategy(
        self, *, strategy, blocks_to_swap, mode, block_swap_backward=True,
        offload_dir=None, block_residency_planner="uniform", block_swap_prefetch_depth=1,
        block_residency_priority="index", block_swap_transfer_strategy="per_tensor",
        execution_mode=WeightResidencyExecutionMode.TRAINING,
    ):
        resolved = str(strategy).strip().lower()
        if resolved not in {"disabled", "block_swap"}:
            raise ValueError("Agnes weight residency supports disabled or RAM block_swap.")
        if int(blocks_to_swap) < 0:
            raise ValueError("Agnes blocks_to_swap must be nonnegative.")
        phase = WeightResidencyExecutionMode.coerce(execution_mode)
        enabled = resolved == "block_swap" and int(blocks_to_swap) > 0
        if enabled and phase is WeightResidencyExecutionMode.TRAINING and self._adapter_spec is None:
            raise ValueError("Agnes training block residency requires a configured adapter.")
        if enabled and phase is WeightResidencyExecutionMode.INFERENCE and block_residency_planner == "phase_aware":
            raise ValueError("Agnes inference residency does not support phase_aware planning.")
        manager = BlockSwapManager(
            total_blocks=len(self.transformer.block.layers), blocks_to_swap=int(blocks_to_swap),
            mode=mode, block_swap_backward=block_swap_backward,
            block_residency_planner=block_residency_planner,
            block_swap_prefetch_depth=block_swap_prefetch_depth,
            block_residency_priority=block_residency_priority,
            block_swap_transfer_strategy=block_swap_transfer_strategy,
        ) if enabled else None
        for handle in self._block_hook_handles:
            handle.remove()
        self._block_hook_handles.clear()
        if self._block_swap_manager is not None:
            self._block_swap_manager.release_device()
        self._block_swap_manager = manager
        self._weight_residency_execution_mode = phase

    def place_offloaded_modules(self, *, device, strategy):
        manager = self._block_swap_manager
        inference = self._weight_residency_execution_mode is WeightResidencyExecutionMode.INFERENCE
        if manager is not None and inference and self.transformer.training:
            raise ValueError("Agnes inference block residency requires eval mode.")
        if manager is not None and not inference and manager.block_swap_backward and not self.transformer.gradient_checkpointing:
            raise ValueError("Agnes training backward block swaps require standard gradient checkpointing.")
        layers = [module for _, module in self.get_block_swap_units()]
        move_trainable_tensors(self.transformer, device=device)
        move_tensors_outside_modules(self.transformer, excluded_modules=layers, device=device)
        for handle in self._block_hook_handles:
            handle.remove()
        self._block_hook_handles.clear()
        if manager is None:
            self.transformer.to(device)
            return
        manager.bind(self.get_block_swap_units(), device=device)
        for index, layer in self.get_block_swap_units():
            self._block_hook_handles.append(layer.register_forward_pre_hook(
                lambda _module, _args, idx=index: manager.before_block(idx)))
            self._block_hook_handles.append(layer.register_forward_hook(
                lambda _module, _args, output, idx=index: (manager.after_block(idx), output)[1]))

    def flush_runtime_offloads(self):
        if self._block_swap_manager is not None:
            self._block_swap_manager.finish_backward()

    def finish_backward_offloads(self):
        self.flush_runtime_offloads()

    def get_block_swap_state(self):
        return ({"enabled": False} if self._block_swap_manager is None
                else {"enabled": True, **self._block_swap_manager.snapshot()})

    def get_video_latent_layout(self):
        return VideoLatentLayout(latent_channels=48, temporal_downsample=4,
                                 spatial_downsample=16, frame_count_modulus=1,
                                 frame_count_remainder=0, request_spatial_multiple=32,
                                 native_output_fps=float(self.options.get("fps", 30)))

    def get_synthetic_batch_spec(self):
        return SyntheticBatchSpec(latent_channels=self.runtime_config.video_in_channels // 4,
                                  text_embed_width=self.runtime_config.text_in_channels)

    def build_synthetic_batch_extras(self, *, batch_size, frame_count, height, width, dtype):
        tokens = max(1, int(frame_count))
        return {"audio_latents": torch.zeros(batch_size, tokens, self.runtime_config.audio_in_channels,
                                             dtype=dtype),
                "audio_lengths": torch.full((batch_size,), tokens, dtype=torch.long)}

    def get_sparse_moe_capabilities(self):
        from mirai.core.moe.routing.contracts import SparseMoECapabilities
        return SparseMoECapabilities(is_sparse_moe=True, architecture="multimodal_dit",
                                    routing="token_choice", routing_granularity="token",
                                    num_routed_experts=self.runtime_config.num_routed_experts,
                                    num_activated_experts=self.runtime_config.moe_top_k)

    def _native_assets(self):
        if self._assets is None:
            from mirai.core.models.agnes_video.assets import AgnesAssets
            self._assets = AgnesAssets(
                {**self.options, "model_path": str(self.model_config.path)},
                weight_dtype=self._compute_dtype,
            )
        return self._assets

    def load_text_encoder(self, *, device):
        self._native_assets().load_text_encoder(device)

    def encode_prompt(self, prompt, *, device):
        return self._native_assets().encode_prompt(prompt, device).float()

    def prepare_inference_conditioning(self, request, *, device, generator):
        from mirai.core.models.agnes_video.conditioning import prepare_agnes_conditioning
        return prepare_agnes_conditioning(self, request, device=device, generator=generator)

    def sample_native_conditioned_media(self, *, request, prepared, **kwargs):
        return self.sample_native_preview(request=request, prepared=prepared, **kwargs)

    def offload_text_encoder(self):
        if self._assets is not None:
            self._assets.offload_text_encoder()

    def has_native_inference(self):
        return True

    def load_vae(self, *, device):
        self._decode_device = str(device)
        self._native_assets().load_video_vae(device)

    def offload_vae(self):
        if self._assets is not None:
            self._assets.offload_video_vae()
            self._assets.offload_audio_vae()

    def decode_latents_native(self, latents):
        latent = latents[0]
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        assets = self._native_assets()
        decoded = assets.load_video_vae(self._decode_device).decode(
            latent.to(device=self._decode_device, dtype=self._compute_dtype)
        )
        return decoded[0].permute(1, 0, 2, 3).float().add(1).mul(0.5).clamp(0, 1).cpu()

    def decode_media_native(self, latents):
        from mirai.core.models.native_video import DecodedMedia, JointMediaLatents
        from mirai.core.models.agnes_video.sampler import (
            agnes_export_frame_count,
            agnes_sample_count,
        )
        if not isinstance(latents, JointMediaLatents):
            raise ValueError("Agnes decoding requires joint video/audio latents.")
        frames = self.decode_latents_native([latents.video])
        assets = self._native_assets()
        assets.offload_video_vae()
        audio = assets.decode_audio(latents.audio, self._decode_device).detach().cpu()
        rate, _, _ = assets.audio_geometry()
        fps = float(latents.fps or self.options.get("fps", 30))
        requested_frames = getattr(latents, "requested_frame_count", None) or frames.shape[0]
        samples = agnes_sample_count(requested_frames, fps, rate)
        exported_frames = agnes_export_frame_count(requested_frames)
        if frames.shape[0] < exported_frames:
            frames = torch.cat((frames, frames[-1:].expand(
                exported_frames - frames.shape[0], *frames.shape[1:])), dim=0)
        else:
            frames = frames[:exported_frames]
        audio = audio.reshape(-1)[:samples]
        if audio.numel() < samples:
            audio = torch.nn.functional.pad(audio, (0, samples-audio.numel()))
        return DecodedMedia(frames=frames, audio=audio, sample_rate=rate)

    @torch.no_grad()
    def sample_native_preview(self, *, noise, context, context_null, denoise_steps,
                              guidance_scale, generator, solver_name, cfg_mode="sequential",
                              request=None, prepared=None):
        from mirai.core.models.native_video import JointMediaLatents
        from mirai.core.models.agnes_video.sampler import agnes_audio_token_count, build_agnes_solver
        if cfg_mode not in {"sequential", "batched"}:
            raise ValueError("Unknown Agnes CFG mode.")
        device = noise.device
        fps = float(self.options.get("fps", 30))
        rate, hop, dim = self._native_assets().audio_geometry()
        self._native_assets().offload_audio_vae()
        if dim != self.runtime_config.audio_in_channels:
            raise ValueError("OmniVAE audio channels do not match the Agnes checkpoint.")
        frames = (int(request.frame_count) if request is not None
                  else 1 + 4 * (noise.shape[-3]-1))
        video = noise.unsqueeze(0) if noise.ndim == 4 else noise
        media = {} if prepared is None else prepared.forward_media
        freeze_audio = bool(media.get("freeze_audio", False))
        audio_count = agnes_audio_token_count(
            frames, fps, rate, hop, conditioned="audio_latents" in media
        )
        if "audio_latents" in media:
            audio = media["audio_latents"].to(device=device, dtype=torch.float32).clone()
            if audio.shape != (1, audio_count, dim):
                raise ValueError("Conditioned audio does not cover the requested Agnes timeline.")
        elif "audio_noise" in media:
            audio = media["audio_noise"].to(device=device, dtype=torch.float32).clone()
            if audio.shape != (1, audio_count, dim):
                raise ValueError("Prepared audio noise does not cover the requested Agnes timeline.")
        else:
            audio = torch.randn((1, audio_count, dim), device=device, dtype=torch.float32, generator=generator)
        if solver_name != "unipc":
            raise ValueError("Agnes native inference requires the released unipc sampler.")
        solver_options = {
            "num_inference_steps": denoise_steps,
            "shift": self.resolve_flow_shift_for_latent_shape(tuple(video.shape)),
            "device": str(device),
            "stochastic": float(guidance_scale) == 1.0,
        }
        video_solver = build_agnes_solver(**solver_options)
        audio_solver = build_agnes_solver(**solver_options)
        if prepared is not None and prepared.source_latent is not None:
            from mirai.core.inference.conditioning import blend_source_with_noise, denoising_schedule_start
            begin = denoising_schedule_start(denoise_steps, prepared.denoising_strength)
            video_solver.set_begin_index(begin)
            audio_solver.set_begin_index(begin)
            source = prepared.source_latent.to(device=device, dtype=video.dtype)
            if source.shape != video.shape:
                raise ValueError("Agnes source video latent geometry does not match the request.")
            if len(video_solver.timesteps) == 0:
                return JointMediaLatents(video=source[0], audio=audio[0], fps=fps, sample_rate=rate,
                                         requested_frame_count=frames)
            video = blend_source_with_noise(source, video, video_solver.timesteps[0])
        for step_index, timestep in enumerate(video_solver.timesteps):
            if prepared is not None:
                prepared.pin_condition(video)
            kwargs = {"audio_latents": audio}
            audio_scale = float(self.options.get("audio_guidance_scale", 5.0))
            source_timestep = int(video_solver.source_timesteps[step_index].item())
            video_scale = (float(guidance_scale) if source_timestep > 500
                           else float(self.options.get("guidance_low_scale", 2.0)))
            if guidance_scale == 1:
                video_scale = audio_scale = 1.0
            if cfg_mode == "batched" and (video_scale != 1 or audio_scale != 1):
                texts = [context[0], context_null[0]]
                lengths = torch.tensor([sample.shape[0] for sample in texts], device=device)
                text = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
                mask = torch.arange(text.shape[1], device=device)[None, :] < lengths[:, None]
                output = self(video.expand(2, *video.shape[1:]), timestep.expand(2),
                              {"t5": text, "text_mask": mask},
                              audio_latents=audio.expand(2, *audio.shape[1:]))
                conditional = tuple(value[:1] for value in output)
                unconditional = tuple(value[1:] for value in output)
            else:
                conditional = self(video, timestep.reshape(1), {"t5": context}, **kwargs)
                unconditional = (self(video, timestep.reshape(1), {"t5": context_null}, **kwargs)
                                 if video_scale != 1 or audio_scale != 1 else None)
            if unconditional is not None:
                prediction = tuple(u + scale*(c-u) for u, c, scale in
                                   zip(unconditional, conditional, (video_scale, audio_scale)))
            else:
                prediction = conditional
            video = video_solver.step(prediction[0], timestep, video, generator=generator).prev_sample
            if not freeze_audio:
                audio = audio_solver.step(prediction[1], timestep, audio, generator=generator).prev_sample
        if prepared is not None:
            prepared.pin_condition(video)
        return JointMediaLatents(video=video[0], audio=audio[0], fps=fps, sample_rate=rate,
                                 requested_frame_count=frames)

    def get_adapter_target_presets(self):
        return {key: list(value) for key, value in TARGET_PRESETS.items()}

    def get_model_extension_capabilities(self):
        return ModelExtensionCapabilities(adapter_target_presets=True,
                                          adapter_runtime_controls=True,
                                          adapter_type_controls=True)

    def set_adapter_config(self, config):
        if config.type != "lora":
            raise ValueError("Agnes supports LoRA adapters.")
        if (str(getattr(config, "timestep_rank_schedule", "none")).lower() != "none"
                or float(getattr(config, "timestep_band_min", 0.0)) != 0.0
                or float(getattr(config, "timestep_band_max", 1.0)) != 1.0):
            raise ValueError("Agnes does not support adapter timestep schedules or timestep bands.")
        preset = str(config.target_preset)
        if preset not in TARGET_PRESETS:
            raise ValueError(f"Unsupported Agnes LoRA target preset: {preset}.")
        spec = {"rank": int(config.rank), "alpha": float(config.alpha), "target_preset": preset,
                "train_router": bool(getattr(config, "train_router", False))}
        if self._adapter_spec is not None:
            if self._adapter_spec != spec:
                raise ValueError("Agnes adapters are already configured with different targets or ranks.")
            return
        self._adapter_lineage = self._make_adapter_lineage(spec)
        self.transformer.requires_grad_(False)
        replacements = []
        for name, module in self.transformer.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in TARGET_PRESETS[preset] and not (spec["train_router"] and leaf == "router"):
                continue
            if isinstance(module, BaseLinear):
                replacement = ModalityLoRALinear(module, rank=spec["rank"], alpha=spec["alpha"])
            elif isinstance(module, nn.Linear):
                replacement = AgnesLoRALinear(module, rank=spec["rank"], alpha=spec["alpha"],
                                         init=str(getattr(config, "init", "kaiming")))
            else:
                continue
            replacements.append((name, replacement))
        if not replacements:
            raise ValueError("Agnes LoRA targets matched no projections.")
        for name, replacement in replacements:
            parent_name, _, leaf = name.rpartition(".")
            parent = self.transformer.get_submodule(parent_name) if parent_name else self.transformer
            setattr(parent, leaf, replacement)
        self._adapter_spec = spec
        self.set_lora_scale(self._lora_scale)

    def set_adapter_type(self, adapter_type):
        if adapter_type != "lora":
            raise ValueError("Agnes supports only LoRA adapters.")

    def set_adapter_runtime(self, *, rank_dropout, lora_parameter_dropout):
        if not 0 <= rank_dropout < 1 or not 0 <= lora_parameter_dropout < 1:
            raise ValueError("Agnes adapter dropout must be in [0,1).")
        for module in adapter_modules(self.transformer):
            if isinstance(module, ModalityLoRALinear):
                module.rank_dropout, module.parameter_dropout = rank_dropout, lora_parameter_dropout
            else:
                module.set_rank_dropout(rank_dropout)
                module.set_lora_parameter_dropout(lora_parameter_dropout)

    def set_lora_scale(self, scale):
        self._lora_scale = float(scale)
        for module in adapter_modules(self.transformer):
            if isinstance(module, ModalityLoRALinear):
                module.runtime_scale = float(scale)
            else:
                module.set_lora_scale(float(scale))

    def get_lora_scale(self):
        return self._lora_scale

    def set_gradient_checkpointing(self, enabled):
        key = str(enabled).lower()
        if key not in {"true", "false", "off", "on", "standard"}:
            raise ValueError("Agnes gradient checkpointing supports off or standard.")
        self.transformer.gradient_checkpointing = key in {"true", "on", "standard"}

    def apply_noise(self, clean_latents, noise, timesteps):
        sigma = timesteps.to(clean_latents).reshape((-1,) + (1,)*(clean_latents.ndim-1))
        return clean_latents * (1-sigma) + noise * sigma

    def compute_target(self, noise, clean_latents, timesteps):
        return noise - clean_latents

    def i2v_conditioning_forward_kwargs(self, *, condition_frame_indexes):
        return {"condition_frame_indexes": condition_frame_indexes}

    def prepare_training_inputs(self, batch, *, timestep_sampler, noise_generator, objective, training, conditioning=None):
        from mirai.core.training.strategies.base import TrainingInputs

        video, audio = batch["latents"], batch["audio_latents"]
        lengths = batch["audio_lengths"].to(device=video.device)
        clean = torch.cat((video.flatten(1), audio.flatten(1)), dim=1)
        sigma = timestep_sampler.sample(video.shape[0], like=video)
        # Independent noise per modality shares one sampled corruption sigma.
        noise = torch.cat((noise_generator.sample(video).flatten(1),
                           noise_generator.sample(audio).flatten(1)), dim=1)
        noisy = (objective.corrupt(pipeline=self, clean_latents=clean, noise=noise, timesteps=sigma)
                 if objective is not None else self.apply_noise(clean, noise, sigma))
        valid_audio = torch.arange(audio.shape[1], device=video.device)[None, :] < lengths[:, None]
        mask = torch.cat((torch.ones_like(video).flatten(1),
                          valid_audio[..., None].expand_as(audio).flatten(1)), dim=1)
        source_mask = batch.get("loss_mask")
        if source_mask is not None:
            source_mask = torch.as_tensor(source_mask, device=video.device, dtype=mask.dtype)
            if source_mask.ndim == 1 and source_mask.shape[0] == video.shape[0]:
                mask = mask * source_mask[:, None]
            elif source_mask.shape == mask.shape:
                mask = mask * source_mask
            else:
                try:
                    video_mask = source_mask.expand_as(video).flatten(1)
                except RuntimeError as exc:
                    raise ValueError("Agnes loss_mask must cover samples, joint elements, or video elements.") from exc
                mask = torch.cat((video_mask, mask[:, video.flatten(1).shape[1]:]), dim=1)
        conditioned_indexes = None
        if conditioning is not None:
            from mirai.core.training.strategies.base import VideoTrainingConditioning
            if not isinstance(conditioning, VideoTrainingConditioning):
                raise TypeError("Agnes video training requires VideoTrainingConditioning.")
            if video.ndim != 5 or video.shape[2] < 1:
                raise ValueError("Agnes image-to-video training requires BCTHW video latents.")
            p = conditioning.first_frame_conditioning_p
            if p >= 1.0 or (not training and p > 0.0):
                selected = torch.ones(video.shape[0], device=video.device, dtype=torch.bool)
            elif p <= 0:
                selected = torch.zeros(video.shape[0], device=video.device, dtype=torch.bool)
            else:
                selected = (torch.rand((video.shape[0],)) < p).to(video.device)
            noisy = noisy.clone()
            video_elements = video.flatten(1).shape[1]
            noisy_video = noisy[:, :video_elements].reshape_as(video)
            video_loss_mask = mask[:, :video_elements].reshape_as(video)
            noisy_video[selected, :, 0] = video[selected, :, 0]
            video_loss_mask[selected, :, 0] = 0
            conditioned_indexes = [[0] if value else [] for value in selected.tolist()]
        return TrainingInputs(noisy_latents=noisy, timestep=sigma, clean_latents=clean, noise=noise,
                              text_embeds={"t5": batch["text_embeds"],
                                           **({"text_mask": batch["text_mask"]} if "text_mask" in batch else {})},
                              loss_mask=mask,
                              extra_forward_kwargs={"video_shape": tuple(video.shape),
                                                    "audio_shape": tuple(audio.shape),
                                                    "audio_lengths": lengths,
                                                    **(self.i2v_conditioning_forward_kwargs(condition_frame_indexes=conditioned_indexes)
                                                       if conditioned_indexes is not None else {})})

    def forward(self, noisy_latents, timesteps, text_embeds, **kwargs):
        with torch.autocast(device_type=noisy_latents.device.type, dtype=self._compute_dtype,
                            enabled=self._compute_dtype != torch.float32):
            return self._forward_native(noisy_latents, timesteps, text_embeds, **kwargs)

    def _forward_native(self, noisy_latents, timesteps, text_embeds, **kwargs):
        del timesteps  # Released Agnes conditions through corrupted modalities and coordinates.
        joint = noisy_latents.ndim == 2
        if joint:
            video_shape, audio_shape = kwargs["video_shape"], kwargs["audio_shape"]
            video_elements = int(torch.tensor(video_shape[1:]).prod())
            video = noisy_latents[:, :video_elements].reshape(video_shape)
            audio = noisy_latents[:, video_elements:].reshape(audio_shape)
        else:
            video, audio = noisy_latents, kwargs["audio_latents"]
        text = text_embeds["t5"] if isinstance(text_embeds, dict) else text_embeds
        mask = text_embeds.get("text_mask") if isinstance(text_embeds, dict) else None
        if text.ndim == 2:
            text = text.unsqueeze(0)
        if mask is not None:
            mask = torch.as_tensor(mask, device=text.device, dtype=torch.bool)
            if mask.shape != text.shape[:2]:
                raise ValueError("Agnes text_mask must match the text token dimensions.")
            texts = [sample[sample_mask] for sample, sample_mask in zip(text, mask)]
            text = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
        packed = pack_inputs(video, audio, text, audio_lengths=kwargs.get("audio_lengths"),
                             text_lengths=None if mask is None else mask.sum(dim=-1))
        prediction = self.transformer(packed.tokens, packed.coordinates, packed.modalities, packed.cu_seqlens)
        predicted_video, predicted_audio = packed.unpack(prediction)
        if predicted_audio.shape[1] < audio.shape[1]:
            predicted_audio = torch.nn.functional.pad(predicted_audio, (0, 0, 0, audio.shape[1]-predicted_audio.shape[1]))
        self._last_diagnostics = {}
        return (torch.cat((predicted_video.flatten(1), predicted_audio.flatten(1)), dim=1)
                if joint else (predicted_video, predicted_audio))

    def get_training_auxiliary_losses(self):
        losses = [module.last_aux_loss for module in self.transformer.modules()
                  if isinstance(module, SparseMoEMLP) and module.last_aux_loss is not None]
        coefficient = float(getattr(self.model_config.params, "moe_aux_loss_weight", 0.01))
        return {"moe_aux_loss": torch.stack(losses).mean() * coefficient} if losses and coefficient else {}

    def get_trainable_parameters(self):
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def get_named_trainable_parameters(self):
        return [(name, p) for name, p in self.transformer.named_parameters() if p.requires_grad]

    def state_dict(self, *args, **kwargs):
        import json
        state = self.transformer.state_dict()
        adapter = {key: value for key, value in state.items() if "lora_" in key}
        if self._adapter_spec is not None:
            encoded = json.dumps(self._adapter_lineage, sort_keys=True, separators=(",", ":")).encode()
            adapter["__mirai_adapter_lineage__"] = torch.tensor(list(encoded), dtype=torch.uint8)
        return adapter

    def load_state_dict(self, state, strict=True):
        self.load_adapter_state(state)

    def load_adapter_state(self, state):
        self.validate_adapter_payload(state)
        state = state.get("adapter_state", state)
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("Agnes adapter targets/ranks differ from the saved adapter.")
        for key, value in state.items():
            if value.shape != expected[key].shape:
                raise ValueError(f"Agnes adapter tensor shape mismatch: {key}.")
        weights = {key: value for key, value in state.items() if key != "__mirai_adapter_lineage__"}
        self.transformer.load_state_dict(weights, strict=False)

    def _make_adapter_lineage(self, spec):
        """Version 1 binds adapters to native tensor contents and target topology."""
        import hashlib
        import json
        from dataclasses import asdict
        from mirai.core.models.agnes_video.checkpoints import AGNES_SOURCE_REVISION
        fingerprint = getattr(self, "_adapter_base_fingerprint", None)
        if fingerprint is None:
            digest = hashlib.sha256()
            for name, tensor in sorted(self.transformer.state_dict().items()):
                header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
                digest.update(header)
                digest.update(b"\0")
                raw = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy()
                digest.update(memoryview(raw))
            fingerprint = digest.hexdigest()
            self._adapter_base_fingerprint = fingerprint
        architecture = asdict(self.runtime_config)
        architecture.pop("params_dtype")
        return {"schema_version": 1, "family": "agnes-video",
                "source_revision": AGNES_SOURCE_REVISION, "base_sha256": fingerprint,
                "architecture": json.loads(json.dumps(architecture, default=str)),
                "adapter": spec}

    def validate_adapter_payload(self, payload):
        import json
        if self._adapter_spec is None:
            raise ValueError("Agnes adapter loading requires configured adapter targets.")
        state = payload.get("adapter_state", payload)
        encoded = state.get("__mirai_adapter_lineage__")
        metadata = payload.get("_metadata", {})
        if encoded is not None:
            if not isinstance(encoded, torch.Tensor) or encoded.dtype != torch.uint8 or encoded.ndim != 1:
                raise ValueError("Invalid Agnes adapter lineage tensor.")
            raw = bytes(encoded.detach().cpu().tolist()).decode("utf-8")
        else:
            raw = metadata.get("mirai_adapter_lineage") if isinstance(metadata, dict) else None
        if not raw:
            raise ValueError("Agnes adapter payload is missing required versioned base lineage.")
        try:
            lineage = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid Agnes adapter lineage metadata.") from exc
        if lineage != self._adapter_lineage:
            raise ValueError("Agnes adapter family, base checkpoint, source revision, or topology differs.")

    def validate_config(self, config):
        from mirai.core.models.attention_backends import normalize_attention_backend
        from mirai.core.moe.runtime.specs import MoEOptimizationPolicy
        errors = []
        try:
            normalize_attention_backend(getattr(config.model, "attention_backend", "auto"))
        except ValueError as exc:
            errors.append(str(exc))
        adapter = config.adapter
        if (str(getattr(adapter, "timestep_rank_schedule", "none")).lower() != "none"
                or float(getattr(adapter, "timestep_band_min", 0.0)) != 0.0
                or float(getattr(adapter, "timestep_band_max", 1.0)) != 1.0):
            errors.append("Agnes does not support adapter timestep schedules or timestep bands.")
        if config.strategy.type not in {"text_to_video", "image_to_video"}:
            errors.append("Agnes joint training requires text_to_video or image_to_video strategy.")
        params = config.model.params
        aux_loss_type = str(getattr(params, "moe_aux_loss_type", "model_native")).strip().lower()
        if aux_loss_type not in {"model_native", "switch"}:
            errors.append(
                "Agnes implements only the native Switch load-balance loss; "
                "model.params.moe_aux_loss_type must be 'model_native' or 'switch'."
            )
        if str(getattr(params, "moe_balance_scope", "microbatch")).strip().lower() != "microbatch":
            errors.append("Agnes native load balancing requires model.params.moe_balance_scope='microbatch'.")
        for field in ("moe_router_z_loss_weight", "moe_router_similarity_loss_weight"):
            if float(getattr(params, field, 0.0)) != 0.0:
                errors.append(f"Agnes does not implement model.params.{field}; it must remain 0.0.")
        try:
            self.configure_moe_optimization_policy(
                MoEOptimizationPolicy.from_memory_config(config.memory)
            )
        except ValueError as exc:
            errors.append(str(exc))
        return errors

    def configure_moe_optimization_policy(self, policy):
        from dataclasses import fields
        from mirai.core.moe.runtime.specs import MoEOptimizationPolicy

        defaults = MoEOptimizationPolicy()
        for field in fields(MoEOptimizationPolicy):
            default = getattr(defaults, field.name)
            value = getattr(policy, field.name, default)
            if value != default:
                raise ValueError(
                    "Agnes uses its vendored token-choice expert loop and does not "
                    f"implement memory.{field.name}; got {value!r}, expected {default!r}."
                )


class AgnesVideoModelFamilyProvider(ModelFamilyProvider):
    def __init__(self):
        super().__init__(model_type="agnes-video", native=True, sparse_moe=True,
                         strict_native_assets_by_default=True, native_cache_encoding=True,
                         config_defaults_name="agnes_video", pipeline_type=AgnesVideoPipeline,
                         batched_cfg_inference=True,
                         inference_tasks=("text_to_video", "image_to_video", "video_to_video", "audio_to_video"))

    def validate_family_params(self, params):
        allowed = {"text_encoder_path", "video_vae_path", "audio_vae_path",
                   "audio_vae_code_root", "fps", "allow_missing_audio",
                   "audio_guidance_scale", "guidance_low_scale"}
        unknown = set(params) - allowed
        errors = [f"Unknown Agnes family parameters: {', '.join(sorted(unknown))}."] if unknown else []
        for key in ("fps", "audio_guidance_scale", "guidance_low_scale"):
            if key not in params:
                continue
            value = params[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or value < 0
                    or (key == "fps" and value == 0)):
                errors.append(f"Agnes family_params.{key} must be finite and {'positive' if key == 'fps' else 'nonnegative'}.")
        if "allow_missing_audio" in params and not isinstance(params["allow_missing_audio"], bool):
            errors.append("Agnes family_params.allow_missing_audio must be a boolean.")
        for key in ("text_encoder_path", "video_vae_path", "audio_vae_path", "audio_vae_code_root"):
            if key in params and not isinstance(params[key], str):
                errors.append(f"Agnes family_params.{key} must be a path string.")
        return errors

    def generation_defaults(self):
        from mirai.core.models.providers import FamilyGenerationDefaults
        return FamilyGenerationDefaults(negative_prompt=agnes_default_negative_prompt(),
                                        steps=32, cfg_scale=5.0, scheduler="unipc",
                                        width=448, height=256, frames=121)

    def build_native_cache_encoder(self, config):
        from mirai.core.models.agnes_video.cache import AgnesNativeCacheEncoder
        return AgnesNativeCacheEncoder(config)


register_model_family_provider("agnes-video", AgnesVideoModelFamilyProvider())
