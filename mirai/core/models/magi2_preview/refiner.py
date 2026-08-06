"""Provider-owned MAGI-2 refiner stage.

The refiner is a SEPARATE checkpoint of a DIFFERENT architecture class from the
preview denoiser: a dense 30-layer transformer
(:class:`mirai.vendors.magi2_preview.model.magi2_refiner.Transformer`) with no
routed experts, living in the ``refiner/`` subfolder of the released snapshot.
It consumes the finished preview latent rather than re-running generation.

Stage shape, mirroring SandAI's reference ``evaluate_magi2_refiner_with_latent``
(``mirai/vendors/magi2_preview/pipeline/inference_engine.py``):

1. The preview latent ``[C, T, H, W]`` is resampled by trilinear interpolation
   to ``[C, 2T - 1, H', W']``. The temporal rule is ``2T - 1``, not a factor-two
   resize, and ``align_corners=True`` is load-bearing: it pins the first and
   last preview latent frames to the first and last refiner frames, so the
   inserted frames are true midpoints of neighbouring preview frames.
2. The resampled latent is re-noised once, variance-preserving, at a FIXED
   index into a zero-terminal-SNR ``sqrt(alphas_cumprod)`` table:
   ``x * sigma + n * sqrt(1 - sigma^2)``. ``sigma`` is the SIGNAL coefficient,
   so a larger index means less signal.
3. A short Flow-UniPC denoise runs over the full timestep grid.

The preview transformer is positioned at temporal stride 8 while the shared
Turbo VAE decoder expands ``4 * (T - 1) + 1`` frames, which is why a preview-only
clip plays at half rate. The refiner's own stride is 4
(``magi2_refiner_vae_stride = [4, 16, 16]``), and ``2T - 1`` latent frames decode
to ``4 * (2T - 2) + 1 == 8 * (T - 1) + 1`` physical frames — exactly the frame
count the request denotes on its 25 fps timeline. Refining is therefore what
makes the file play at 25 fps rather than 12.5.

This module owns two things:

1. :class:`Magi2Refiner` — the on-demand lifecycle of the refiner transformer
   (build from the vendored refiner architecture JSON, strict-load the
   ``refiner/`` shards, place it on the compute device through the family's own
   block-residency mechanism, release afterwards).
2. :func:`run_refine` — the family-owned inference *policy* loop. It reaches the
   model only through the pipeline's provider hooks
   (``release_base_transformer`` / ``load_text_encoder`` / ``encode_prompt`` /
   ``offload_text_encoder`` / ``refiner_forward``), never model internals.

The pure geometry and re-noise math (:func:`magi2_refiner_latent_frames`,
:func:`magi2_refiner_upsample`, :func:`magi2_refiner_renoise`) is unit-tested on
CPU without any released weights.

Attribution: SandAI MAGI-2-preview, Apache-2.0
(https://github.com/SandAI-org/MAGI-2-preview).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mirai.core.models.magi2_preview.refiner_attention import (
    MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT,
    normalize_refiner_attention_backend,
)


# Subfolder of the released snapshot holding the refiner shards. The upstream
# config states it as ``${MAGI2_CKPT_ROOT}/refiner``; the path is resolved from
# ``model.path`` here instead of from the environment.
MAGI2_REFINER_SUBFOLDER = "refiner"

# The refiner's own temporal VAE stride (``magi2_refiner_vae_stride[0]``). The
# preview stride is 8, so the ``2T - 1`` resample is what lands the latent on the
# decoder's native grid.
MAGI2_REFINER_VAE_TEMPORAL_STRIDE = 4

# Playback rate of a refined clip: the decoder emits the full requested frame
# count instead of the preview's half-rate expansion.
MAGI2_REFINER_OUTPUT_FPS = 25.0

# The vendored refiner sampler is Flow-UniPC, as the preview sampler is.
MAGI2_REFINER_SOLVER = "unipc"

# Length of the zero-terminal-SNR discretization the re-noise index addresses.
MAGI2_REFINER_SIGMA_TABLE_SIZE = 1000

_MAGI2_REFINER_ASSET_HINT = (
    "The MAGI-2 refiner is a separate checkpoint in the 'refiner/' subfolder of "
    "the released snapshot, holding model.safetensors.index.json and its shards. "
    "Fetch it from the release and point model.path at a snapshot root that "
    "contains it."
)


# --------------------------------------------------------------------------- #
# Pure geometry and re-noise math
# --------------------------------------------------------------------------- #
def magi2_refiner_latent_frames(preview_latent_frames: int) -> int:
    """Refiner latent length for ``preview_latent_frames`` preview latents.

    ``2T - 1`` keeps both endpoints and inserts one frame between each
    neighbouring pair, so the covered interval is unchanged while the sampling
    rate doubles.
    """
    frames = int(preview_latent_frames)
    if frames < 1:
        raise ValueError(
            f"preview latent frame count must be >= 1, got {frames}."
        )
    return 2 * frames - 1


def magi2_refiner_decoded_frames(preview_latent_frames: int) -> int:
    """Physical frames a refined clip decodes to, at ``MAGI2_REFINER_OUTPUT_FPS``.

    The shared Turbo VAE expands ``4 * (T' - 1) + 1`` frames from ``T'`` latent
    frames, and ``T' = 2T - 1``, so the result is ``8 * (T - 1) + 1`` — the frame
    count the request itself denotes.
    """
    refined = magi2_refiner_latent_frames(preview_latent_frames)
    return MAGI2_REFINER_VAE_TEMPORAL_STRIDE * (refined - 1) + 1


def magi2_refiner_sigma_table(device: Any | None = None) -> torch.Tensor:
    """The zero-terminal-SNR ``sqrt(alphas_cumprod)`` table the index addresses.

    Built by the vendored discretization so the table this stage re-noises
    against is the released one rather than a restatement of it. Descending:
    entry 0 keeps almost all signal and the final entry keeps none.
    """
    from mirai.vendors.magi2_preview.pipeline.inference_engine import (
        ZeroSNRDDPMDiscretization,
    )

    table = ZeroSNRDDPMDiscretization()(
        MAGI2_REFINER_SIGMA_TABLE_SIZE, do_append_zero=False, flip=True
    )
    return table if device is None else table.to(device=device)


def magi2_refiner_renoise_sigma(noise_index: int) -> float:
    """Signal coefficient at ``noise_index`` of the zero-terminal-SNR table."""
    index = int(noise_index)
    if not (0 <= index < MAGI2_REFINER_SIGMA_TABLE_SIZE):
        raise ValueError(
            f"refiner noise index must lie in [0, {MAGI2_REFINER_SIGMA_TABLE_SIZE}), "
            f"got {index}."
        )
    return float(magi2_refiner_sigma_table()[index])


def magi2_refiner_upsample(
    latent: torch.Tensor, *, latent_height: int, latent_width: int
) -> torch.Tensor:
    """Resample a ``[B, C, T, H, W]`` preview latent to ``[B, C, 2T-1, H', W']``.

    ``align_corners=True`` is required rather than incidental: it fixes the
    endpoints of the preview trajectory, so the inserted frames are midpoints of
    real preview frames. With ``align_corners=False`` the whole trajectory is
    shifted by half a preview frame and the endpoints are extrapolated.
    """
    import torch.nn.functional as F

    if latent.ndim != 5:
        raise ValueError(
            f"magi2_refiner_upsample expects [B,C,T,H,W], got {tuple(latent.shape)}."
        )
    height = int(latent_height)
    width = int(latent_width)
    if height < 1 or width < 1:
        raise ValueError(
            f"refiner latent size must be positive, got {height}x{width}."
        )
    frames = magi2_refiner_latent_frames(int(latent.shape[2]))
    return F.interpolate(
        latent,
        size=(frames, height, width),
        mode="trilinear",
        align_corners=True,
    )


def magi2_refiner_renoise(
    latent: torch.Tensor, noise: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Variance-preserving single-shot re-noise at ``sigma``.

    ``sigma`` is the SIGNAL coefficient of the zero-terminal-SNR table, so the
    noise weight is ``sqrt(1 - sigma^2)``. This is not the rectified-flow blend
    the preview training path uses; the refiner is entered at a fixed corruption
    level rather than at a flow timestep.
    """
    value = float(sigma)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"refiner re-noise sigma must lie in [0, 1], got {value}.")
    if tuple(latent.shape) != tuple(noise.shape):
        raise ValueError(
            "refiner re-noise requires matching shapes, got "
            f"{tuple(latent.shape)} and {tuple(noise.shape)}."
        )
    return latent * value + noise * (1.0 - value**2) ** 0.5


# --------------------------------------------------------------------------- #
# Resolved request
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Magi2RefineSettings:
    """The refinement actually applied, after release-profile resolution.

    Every field is reported back to the caller, so a run states the values that
    drove it rather than the values that were typed.
    """

    steps: int
    cfg_scale: float
    shift: float
    height: int
    width: int
    noise_index: int
    scheduler: str

    def as_request(self) -> dict[str, Any]:
        return {
            "steps": int(self.steps),
            "cfg_scale": float(self.cfg_scale),
            "shift": float(self.shift),
            "height": int(self.height),
            "width": int(self.width),
            "noise_index": int(self.noise_index),
            "scheduler": str(self.scheduler),
        }


# --------------------------------------------------------------------------- #
# Refiner transformer lifecycle (on-demand load / release)
# --------------------------------------------------------------------------- #
class Magi2Refiner:
    """On-demand owner of the refiner transformer and its data proxy.

    Built lazily for the refine stage, placed on the compute device through the
    same block-residency mechanism the preview transformer uses, and released
    afterwards so it never co-resides with the preview denoiser.
    """

    def __init__(
        self,
        model_config: Any,
        *,
        config_path: str,
        subfolder: str = MAGI2_REFINER_SUBFOLDER,
        attention_backend: str = MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT,
    ) -> None:
        self.model_config = model_config
        self.config_path = str(config_path)
        self.subfolder = str(subfolder or MAGI2_REFINER_SUBFOLDER)
        self.attention_backend = normalize_refiner_attention_backend(attention_backend)
        self._transformer: Any | None = None
        self._data_proxy: Any | None = None
        self._runtime_config: Any | None = None
        self._block_swap_manager: Any | None = None
        self._block_hook_handles: list[Any] = []

    # -- asset location ----------------------------------------------------
    def snapshot_root(self) -> Path:
        root = Path(str(self.model_config.path)).expanduser()
        return root.parent if root.name == "preview" else root

    def checkpoint_dir(self) -> Path:
        return self.snapshot_root() / self.subfolder

    def has_weights(self) -> bool:
        """True when the refiner subfolder carries a sharded checkpoint index."""
        return (self.checkpoint_dir() / "model.safetensors.index.json").is_file()

    # -- release profile ---------------------------------------------------
    def runtime_config(self) -> Any:
        """The vendored refiner profile, with the checkpoint path resolved.

        ``magi2_refiner_model_path`` is stated upstream as an environment
        expansion; it is rewritten here from ``model.path`` so the released
        profile can be read without an environment contract.
        """
        if self._runtime_config is None:
            from mirai.vendors.magi2_preview.common.magi2_config import load_config

            config = load_config(self.config_path)
            if config.magi2_refiner_arch_config is None:
                raise RuntimeError(
                    f"'{self.config_path}' declares no magi2_refiner_arch_config, so "
                    "it is a preview-only profile and cannot drive the refiner "
                    "stage. Point family_params.refiner_config_path at the "
                    "refiner architecture JSON."
                )
            config.evaluation_config.magi2_refiner_model_path = str(
                self.checkpoint_dir()
            )
            self._runtime_config = config
        return self._runtime_config

    def settings(
        self,
        *,
        steps: int | None,
        cfg_scale: float | None,
        shift: float | None,
        height: int | None,
        width: int | None,
        preview_height: int,
        preview_width: int,
        scheduler: str,
    ) -> Magi2RefineSettings:
        """Resolve a request against the release profile.

        An absent value takes the released refiner profile; a supplied value
        overrides it. The target resolution falls back to the preview request,
        which makes the default refinement purely temporal.
        """
        evaluation = self.runtime_config().evaluation_config
        resolved_scheduler = str(scheduler or MAGI2_REFINER_SOLVER).strip().lower()
        if resolved_scheduler != MAGI2_REFINER_SOLVER:
            raise RuntimeError(
                f"The MAGI-2 refiner implements the '{MAGI2_REFINER_SOLVER}' solver "
                f"only; got '{resolved_scheduler}'. Its sampler is the vendored "
                "Flow-UniPC multistep scheduler and has no other schedule."
            )
        resolved_shift = (
            float(evaluation.magi2_refiner_shift)
            if evaluation.magi2_refiner_shift is not None
            else float(evaluation.shift)
        )
        settings = Magi2RefineSettings(
            steps=int(
                evaluation.magi2_refiner_num_inference_steps if steps is None else steps
            ),
            cfg_scale=float(
                evaluation.magi2_refiner_video_txt_guidance_scale
                if cfg_scale is None
                else cfg_scale
            ),
            shift=resolved_shift if shift is None else float(shift),
            height=int(preview_height if height is None else height),
            width=int(preview_width if width is None else width),
            noise_index=int(evaluation.magi2_refiner_noise_value),
            scheduler=resolved_scheduler,
        )
        if settings.steps < 1:
            raise RuntimeError(
                f"refiner steps must be >= 1, got {settings.steps}."
            )
        if settings.shift <= 0.0:
            raise RuntimeError(
                f"refiner flow shift must be > 0, got {settings.shift}."
            )
        if settings.cfg_scale < 0.0:
            raise RuntimeError(
                f"refiner CFG scale must be >= 0, got {settings.cfg_scale}."
            )
        stride = tuple(int(value) for value in evaluation.magi2_refiner_vae_stride)
        for label, value in (("height", settings.height), ("width", settings.width)):
            multiple = stride[1] if label == "height" else stride[2]
            if value < multiple or value % multiple:
                raise RuntimeError(
                    f"refiner {label}={value} must be a positive multiple of "
                    f"{multiple}, the refiner VAE spatial stride."
                )
        if not (0 <= settings.noise_index < MAGI2_REFINER_SIGMA_TABLE_SIZE):
            raise RuntimeError(
                "family_params.refiner_config_path states "
                f"magi2_refiner_noise_value={settings.noise_index}, outside the "
                f"[0, {MAGI2_REFINER_SIGMA_TABLE_SIZE}) zero-terminal-SNR table."
            )
        return settings

    def latent_size(self, settings: Magi2RefineSettings) -> tuple[int, int]:
        """Refiner latent height and width for a resolved request."""
        stride = tuple(
            int(value)
            for value in self.runtime_config().evaluation_config.magi2_refiner_vae_stride
        )
        return settings.height // stride[1], settings.width // stride[2]

    # -- lifecycle ---------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._transformer is not None

    @property
    def transformer(self) -> Any:
        if self._transformer is None:
            raise RuntimeError("MAGI-2 refiner is not loaded; call load() first.")
        return self._transformer

    @property
    def data_proxy(self) -> Any:
        if self._data_proxy is None:
            raise RuntimeError("MAGI-2 refiner is not loaded; call load() first.")
        return self._data_proxy

    def load(self, *, device: str, residency: Any | None = None) -> None:
        """Build (once) and place the refiner transformer on ``device``.

        The architecture comes from the vendored refiner JSON rather than from a
        ``config.json`` beside the shards, because the release ships none; the
        shards are then loaded strictly, so a checkpoint whose key set does not
        match the declared architecture fails instead of loading partially.

        The refiner's attention is dispatched through operators MagiCompiler
        registers, falling back to the eager implementations vendored beside
        them when the operator namespace is empty. That at least one of the two
        is reachable is a precondition of the stage rather than something to
        discover mid-forward. The precondition covers only the operators the
        configured architecture actually reaches: a bound native attention
        backend serves the local-attention operator itself, and a profile whose
        layers are all local never reaches the dense one.
        """
        from mirai.core.models.magi2_preview.refiner_attention import (
            attach_refiner_attention_backend,
            refiner_required_magi2_ops,
            resolve_magi2_refiner_attention,
            validate_refiner_flex_support,
        )
        from mirai.vendors.magi2_preview.common.magi_compiler_compat import (
            require_magi2_custom_ops,
        )

        config = self.runtime_config()
        backend = resolve_magi2_refiner_attention(self.attention_backend)
        if backend is not None:
            validate_refiner_flex_support()
        required = refiner_required_magi2_ops(
            config.magi2_refiner_arch_config, backend
        )
        if required:
            require_magi2_custom_ops("The MAGI-2 refiner stage", required)
        if not self.has_weights():
            raise RuntimeError(
                "--refine requested but no MAGI-2 refiner weights were found under "
                f"'{self.checkpoint_dir()}'. {_MAGI2_REFINER_ASSET_HINT}"
            )
        if self._transformer is None:
            from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import (
                load_safetensors_dir,
            )
            from mirai.vendors.magi2_preview.model.magi2_refiner import (
                Transformer as Magi2RefinerTransformer,
            )
            from mirai.vendors.magi2_preview.pipeline.refiner_data_proxy import (
                Magi2RefinerDataProxy,
            )

            transformer = Magi2RefinerTransformer(config.magi2_refiner_arch_config)
            state = load_safetensors_dir(
                str(self.checkpoint_dir()), desc="Loading MAGI-2 refiner shards"
            )
            transformer.load_state_dict(state, strict=True)
            del state
            transformer.eval()
            for parameter in transformer.parameters():
                parameter.requires_grad_(False)
            self._transformer = transformer
            self._data_proxy = Magi2RefinerDataProxy(
                config.evaluation_config.magi2_refiner_data_proxy_config
            )
        attach_refiner_attention_backend(self._transformer, backend)
        self._place(device=device, residency=residency)

    def _place(self, *, device: str, residency: Any | None) -> None:
        """Place the refiner using the family's own residency mechanism.

        With no residency request the dense refiner is simply resident. With
        one, its layers stream host-to-device exactly as the preview blocks do:
        the refiner is a different architecture but the same
        ``block.layers`` stack shape, so the family's block-swap manager binds
        to it unchanged.
        """
        target = torch.device(device)
        transformer = self.transformer
        if residency is None or not bool(getattr(residency, "enabled", False)):
            transformer.to(target)
            return

        from mirai.core.training.residency.block_swap import BlockSwapManager
        from mirai.core.training.residency.tensor_residency import (
            move_tensors_outside_modules,
        )

        units = list(enumerate(transformer.block.layers))
        layers = [module for _index, module in units]
        move_tensors_outside_modules(
            transformer, excluded_modules=layers, device=target
        )
        manager = BlockSwapManager(
            total_blocks=len(units),
            blocks_to_swap=min(len(units), int(residency.blocks_to_swap)),
            mode=str(residency.mode),
            # Refinement is inference-only, so no backward window exists to
            # keep a block resident for.
            block_swap_backward=False,
            block_residency_planner=str(residency.block_residency_planner),
            block_swap_prefetch_depth=int(residency.block_swap_prefetch_depth),
            block_residency_priority=str(residency.block_residency_priority),
            block_swap_transfer_strategy=str(residency.block_swap_transfer_strategy),
            disk_offload_dir=residency.offload_dir,
        )
        manager.bind(units, device=target)
        self._release_hooks()
        for index, layer in units:
            self._block_hook_handles.append(
                layer.register_forward_pre_hook(
                    lambda _module, _args, idx=index: manager.before_block(idx)
                )
            )
            self._block_hook_handles.append(
                layer.register_forward_hook(
                    lambda _module, _args, output, idx=index: (
                        manager.after_block(idx),
                        output,
                    )[1]
                )
            )
        self._block_swap_manager = manager

    def _release_hooks(self) -> None:
        for handle in self._block_hook_handles:
            handle.remove()
        self._block_hook_handles.clear()

    def release(self) -> None:
        """Drop the refiner off the compute device and free its VRAM."""
        self._release_hooks()
        self._block_swap_manager = None
        if self._transformer is not None:
            self._transformer.to(device=torch.device("cpu"))
            self._transformer = None
        self._data_proxy = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Refine policy loop (family-owned; model access via provider hooks only)
# --------------------------------------------------------------------------- #
def run_refine(
    *,
    pipeline: Any,
    refiner: Magi2Refiner,
    base_latent: torch.Tensor,
    settings: Magi2RefineSettings,
    prompt: str,
    negative_prompt: str,
    seed: int,
    device: str,
    dtype: Any | None = None,
) -> torch.Tensor:
    """Resample, re-noise and short-denoise a preview latent into a refined one.

    Returns ``[C, 2T-1, H', W']`` so the result flows straight into the standard
    native VAE decode.

    The preview transformer is released before any refiner state is allocated.
    This operation does not restore it; the owning inference session restores
    placement before a subsequent preview denoise.
    """
    if base_latent.ndim != 4:
        raise RuntimeError(
            "MAGI-2 refinement expects a [C,T,H,W] preview latent, got "
            f"{tuple(base_latent.shape)}."
        )
    compute_device = torch.device(device)
    generator = torch.Generator(device=compute_device)
    generator.manual_seed(int(seed))

    # The preview latent is complete, so its sparse-MoE transformer leaves the
    # device before the refiner's weights are allocated.
    pipeline.release_base_transformer()

    # Text conditioning is produced before the refiner is resident so the Qwen3.5
    # encoder and the refiner never share the device.
    pipeline.load_text_encoder(device=str(compute_device))
    try:
        context = torch.as_tensor(
            pipeline.encode_prompt(prompt, device=str(compute_device))
        )
        context_null = torch.as_tensor(
            pipeline.encode_prompt(negative_prompt or "", device=str(compute_device))
        )
    finally:
        pipeline.offload_text_encoder()
    context = _as_batched_context(context, compute_device)
    context_null = _as_batched_context(context_null, compute_device)

    latent_height, latent_width = refiner.latent_size(settings)
    latents = magi2_refiner_upsample(
        base_latent.unsqueeze(0).to(device=compute_device, dtype=torch.float32),
        latent_height=latent_height,
        latent_width=latent_width,
    )
    noise = torch.randn(
        latents.shape,
        generator=generator,
        device=compute_device,
        dtype=latents.dtype,
    )
    latents = magi2_refiner_renoise(
        latents, noise, magi2_refiner_renoise_sigma(settings.noise_index)
    )

    from mirai.vendors.magi2_preview.pipeline.sampler import (
        FlowUniPCMultistepScheduler,
    )

    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(
        int(settings.steps), device=compute_device, shift=float(settings.shift)
    )

    try:
        refiner.load(
            device=str(compute_device),
            residency=pipeline.refiner_residency_request(),
        )
        scale = float(settings.cfg_scale)
        autocast_ctx: Any = (
            torch.amp.autocast("cuda", dtype=dtype)
            if compute_device.type == "cuda"
            and dtype in {torch.float16, torch.bfloat16}
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            for timestep in scheduler.timesteps:
                # The refiner evaluates the conditional and unconditional
                # branches as two separate forwards. Unlike the preview sampler
                # it is NOT packed into one B=2 forward: its window attention
                # asserts batch size 1.
                if scale <= 0.0:
                    velocity = pipeline.refiner_forward(latents, context_null)
                else:
                    v_cond = pipeline.refiner_forward(latents, context)
                    v_uncond = pipeline.refiner_forward(latents, context_null)
                    velocity = v_uncond + scale * (v_cond - v_uncond)
                # Upstream expands the guidance scale over the frame axis so a
                # per-frame schedule can lower it on the leading frames. The
                # released refiner profile leaves that schedule off, so the
                # scalar blend above is the released behavior; a per-frame
                # schedule would have to enter as its own explicit option.
                latents = scheduler.step(
                    velocity, timestep, latents, return_dict=False
                )[0]
    finally:
        refiner.release()
    refined = latents.squeeze(0) if latents.ndim == 5 else latents
    return refined.detach().float()


def _as_batched_context(context: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Coerce an encoded prompt to the ``[1, S, W]`` the refiner proxy packs."""
    value = context.to(device=device, dtype=torch.float32)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or int(value.shape[0]) != 1:
        raise RuntimeError(
            "MAGI-2 refinement conditions on one prompt at a time; got context "
            f"of shape {tuple(value.shape)}."
        )
    return value


__all__ = [
    "MAGI2_REFINER_OUTPUT_FPS",
    "MAGI2_REFINER_SIGMA_TABLE_SIZE",
    "MAGI2_REFINER_SOLVER",
    "MAGI2_REFINER_SUBFOLDER",
    "MAGI2_REFINER_VAE_TEMPORAL_STRIDE",
    "Magi2RefineSettings",
    "Magi2Refiner",
    "magi2_refiner_decoded_frames",
    "magi2_refiner_latent_frames",
    "magi2_refiner_renoise",
    "magi2_refiner_renoise_sigma",
    "magi2_refiner_sigma_table",
    "magi2_refiner_upsample",
    "run_refine",
]
