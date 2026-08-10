"""Provider-owned LingBot-Video refiner stage.

The refiner is a SEPARATE DiT checkpoint of the SAME architecture class
(:class:`LingBotVideoTransformer3DModel`) living in the ``refiner/`` subfolder of
the model root. The base pass generates a low-resolution video, the decoded
video is upscaled in PIXEL space (bicubic) and VAE re-encoded, the latent is
re-noised to ``t_thresh`` (rectified-flow blend), and the refiner DiT denoises
the low-noise tail with classifier-free guidance to the high-res output.

This module owns two things:

1. :class:`LingBotRefiner` — the on-demand lifecycle of the refiner transformer
   (load from ``model_root/refiner`` reusing the shared transformer loader,
   place on the compute device, release afterwards). Held by
   :class:`~mirai.core.models.lingbot_video.pipeline.LingBotVideoPipeline` alongside its
   native text/VAE inference assets.
2. :func:`run_refine` — the family-owned inference *policy* loop: upscale ->
   re-noise -> tail denoise via the existing preview solver registry. It touches
   the model only through the pipeline's provider hooks (``decode_latents_native``
   / ``encode_video_native`` / ``load_refiner`` / ``refiner_forward`` /
   ``release_base_transformer``), never model internals.

Two conditioning rules are specific to this stage and differ from the base pass:

- The refiner conditions on TEXT ONLY. It calls the text-only ``encode_prompt``
  hook, never ``encode_conditioned_prompt``, so the condition image does not
  reach the VLM text encoder here.
- For an image-conditioned task the condition frame is re-encoded at the
  REFINER's geometry (see :func:`resolve_refiner_conditioning`) and frame 0 is
  re-pinned to it before the loop, before every forward, and after every solver
  step. Text-only refines prepare no conditioning and pin nothing.

The pure math (:func:`compute_refiner_sigmas`, :func:`prepare_refiner_latent`,
:func:`resize_video_bicubic`) mirrors upstream ``lingbot_video/utils.py`` 1:1 and
is unit-tested on CPU without any real weights.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from mirai.core.models.lingbot_video.checkpoints import load_lingbot_transformer_checkpoint
from mirai.core.models.lingbot_video.checkpoints import load_lingbot_transformer_config
from mirai.core.models.lingbot_video.checkpoints import resolve_lingbot_transformer_dir
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# Released refinement profile of this family. A request that leaves a value
# unset takes the entry below, which is what ``--refine`` alone runs.
LINGBOT_REFINER_DEFAULTS: dict[str, Any] = {
    "steps": 8,
    "cfg_scale": 3.0,
    "shift": 3.0,
    "t_thresh": 0.85,
    "sigma_tail_steps": 2,
    "height": 1088,
    "width": 1920,
}

# The tail extends the sigma grid between the last shifted grid point and 0.0.
# Beyond this the tail dominates the schedule instead of refining it, which is
# a request error rather than a supported configuration.
MAX_REFINER_SIGMA_TAIL_STEPS = 64


_HF_REFINER_HINT = (
    "The LingBot-Video refiner is a separate DiT checkpoint in the 'refiner/' "
    "subfolder of the model root. Fetch it, e.g. "
    "`hf download robbyant/lingbot-video-moe-30b-a3b --include 'refiner/*'`, or "
    "point model.path at a snapshot that contains refiner/config.json + "
    "refiner/diffusion_pytorch_model.safetensors*."
)


# --------------------------------------------------------------------------- #
# Pure math (mirrors upstream lingbot_video/utils.py verbatim)
# --------------------------------------------------------------------------- #
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


def validate_refiner_sigmas(sigmas: "torch.Tensor", t_thresh: float | None = None) -> "torch.Tensor":
    """Mirror upstream ``validate_refiner_sigmas``: descending tail in [0,1]."""
    arr = sigmas.to(dtype=torch.float64).reshape(-1)
    if arr.numel() == 0:
        raise ValueError("refiner sigma schedule must be a non-empty 1D list")
    if not bool(torch.isfinite(arr).all()):
        raise ValueError("refiner sigma schedule contains non-finite values")
    if bool((arr < 0.0).any()) or bool((arr > 1.0).any()):
        raise ValueError(
            f"refiner sigma schedule values must be in [0, 1], got {arr.tolist()}"
        )
    if arr.numel() > 1 and not bool((arr[1:] - arr[:-1] < 0.0).all()):
        raise ValueError(
            f"refiner sigma schedule must be strictly descending, got {arr.tolist()}"
        )
    if t_thresh is not None and abs(float(arr[0]) - float(t_thresh)) > 1e-6:
        raise ValueError(
            f"refiner sigma schedule must start at t_thresh={float(t_thresh)}, "
            f"got {float(arr[0])}"
        )
    return arr


def compute_refiner_sigmas(
    *,
    num_inference_steps: int,
    shift: float,
    t_thresh: float,
    tail_steps: int = 0,
    sigma_max: float = 1.0,
    sigma_min: float = 0.0,
) -> "torch.Tensor":
    """Build the refiner tail sigma grid — verbatim port of upstream.

    The full step grid ``linspace(sigma_max, sigma_min, steps+1)[:-1]`` is
    flow-shifted (``shift * s / (1 + (shift-1) * s)``), FILTERED to the low-noise
    tail (``<= t_thresh``), forced to start exactly at ``t_thresh``, and — when
    ``tail_steps > 0`` — extended with extra low-noise sigmas between the last
    grid value and ``sigma_min``. ``sigma_max``/``sigma_min`` default to the
    rectified-flow scheduler's ``1.0``/``0.0`` (our fm_solvers convention).
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for the LingBot-Video refiner.")
    t_value = float(t_thresh)
    if not (0.0 < t_value <= 1.0):
        raise ValueError(f"refiner t_thresh must lie in (0, 1], got {t_value}")
    steps = int(num_inference_steps)
    if steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
    tail = int(tail_steps or 0)
    if tail < 0:
        raise ValueError(f"refiner_sigma_tail_steps must be >= 0, got {tail}")

    base = torch.linspace(float(sigma_max), float(sigma_min), steps + 1, dtype=torch.float64)[:-1]
    shift_value = float(shift)
    shifted = shift_value * base / (1.0 + (shift_value - 1.0) * base)
    eps = 1e-6
    sigmas = shifted[shifted <= t_value + eps]
    if sigmas.numel() == 0 or abs(float(sigmas[0]) - t_value) > eps:
        sigmas = torch.cat([torch.tensor([t_value], dtype=torch.float64), sigmas])
    if tail > 0:
        start = float(sigmas[-1])
        stop = min(float(sigma_min), start)
        extra = torch.linspace(start, stop, tail + 2, dtype=torch.float64)[1:-1]
        sigmas = torch.cat([sigmas, extra])
    validate_refiner_sigmas(sigmas, t_value)
    return sigmas.to(dtype=torch.float32)


def prepare_refiner_latent(
    x_up: "torch.Tensor",
    noise: "torch.Tensor",
    t_thresh: "float | torch.Tensor",
) -> "torch.Tensor":
    """Re-noise the upscaled latent at sigma == ``t_thresh`` (rectified flow).

    Verbatim upstream: ``(1 - t_thresh) * x_up + t_thresh * noise``. ``t_thresh``
    is the sigma DIRECTLY (no flow-shift is applied to the re-noise itself).
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for the LingBot-Video refiner.")
    if not torch.is_tensor(t_thresh):
        t_thresh = torch.tensor(float(t_thresh), device=x_up.device, dtype=x_up.dtype)
    while t_thresh.ndim < x_up.ndim:
        t_thresh = t_thresh.view(*t_thresh.shape, *([1] * (x_up.ndim - t_thresh.ndim)))
    return (1.0 - t_thresh) * x_up + t_thresh * noise


def resize_video_bicubic(video: "torch.Tensor", height: int, width: int) -> "torch.Tensor":
    """Bicubic per-frame resize of a ``[B,C,T,H,W]`` video (upstream semantics).

    Upstream ``resize_video_tensor`` flattens frames into the batch and calls
    ``F.interpolate(..., mode="bicubic", align_corners=False)`` (no antialias).
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for the LingBot-Video refiner.")
    import torch.nn.functional as F

    if video.ndim != 5:
        raise ValueError(
            f"resize_video_bicubic expects [B,C,T,H,W], got {tuple(video.shape)}."
        )
    b, c, t, h, w = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    resized = F.interpolate(
        flat, size=(int(height), int(width)), mode="bicubic", align_corners=False
    )
    resized = resized.reshape(b, t, c, int(height), int(width)).permute(0, 2, 1, 3, 4)
    return resized.contiguous()


def build_refiner_solver(scheduler_name: str, sigmas: "torch.Tensor", *, shift: float, device: str) -> Any:
    """Build a preview solver and OVERRIDE its grid with the refiner tail sigmas.

    Reuses the single-owner preview solver registry (``euler`` / ``unipc``) so the
    per-step update math is exactly the family's denoise step; only the timestep
    grid is replaced by the tail schedule. The shift is ALREADY baked into
    ``sigmas`` by :func:`compute_refiner_sigmas`, so the solver must not re-shift.
    """
    from mirai.core.training.preview.preview_solvers import PreviewSolverSpec, resolve_preview_solver

    solver = resolve_preview_solver(
        scheduler_name,
        PreviewSolverSpec(
            num_inference_steps=int(sigmas.numel()),
            flow_shift=float(shift),
            device=str(device),
            num_train_timesteps=1000,
        ),
    )
    grid = sigmas.to(device=torch.device(device), dtype=torch.float32)
    solver.timesteps = grid
    # UniPC keeps an internal extended grid (tail + trailing 0.0) and multistep
    # history; euler steps straight off ``.timesteps``. Reset both so the tail
    # run is self-contained and lands exactly on x0.
    if hasattr(solver, "_sigmas"):
        zero = torch.zeros(1, dtype=grid.dtype, device=grid.device)
        solver._sigmas = torch.cat([grid, zero])
    if hasattr(solver, "_reset_multistep_state"):
        solver._reset_multistep_state()
    solver._step_index = 0
    return solver


def resolve_refiner_conditioning(
    pipeline: Any,
    conditioning: Any | None,
    *,
    height: int,
    width: int,
    device: str,
    generator: Any,
) -> Any | None:
    """Prepare the frame-0 condition latent at the REFINER's output geometry.

    Returns ``None`` for an unconditioned request, so a text-only refine keeps
    its previous behaviour exactly. For an image-conditioned request the
    condition image is re-encoded at ``height``/``width`` — the base pass encoded
    it at the base geometry, and that latent has the wrong spatial shape for the
    upscaled tail — and the resulting
    :class:`~mirai.core.inference.conditioning.PreparedInferenceConditioning`
    carries the ``pin_condition`` used by the tail loop.

    A source-video request needs no pin: the refiner already starts from the
    upscaled base result, which is the source-derived state itself.
    """
    from mirai.core.inference.conditioning import (
        IMAGE_TO_VIDEO,
        InferenceConditioningRequest,
    )

    if conditioning is None:
        return None
    if not isinstance(conditioning, InferenceConditioningRequest):
        raise TypeError(
            "LingBot-Video refinement expects an InferenceConditioningRequest, "
            f"got {type(conditioning).__name__}."
        )
    if conditioning.task != IMAGE_TO_VIDEO:
        return None

    prepare = getattr(pipeline, "prepare_inference_conditioning", None)
    if not callable(prepare):
        raise RuntimeError(
            "Image-conditioned LingBot-Video refinement requires the pipeline's "
            "prepare_inference_conditioning() hook to re-encode the condition "
            "frame at the refiner geometry."
        )
    request = InferenceConditioningRequest(
        task=conditioning.task,
        input_image=conditioning.input_image,
        denoising_strength=1.0,
        frame_count=int(conditioning.frame_count),
        height=int(height),
        width=int(width),
    ).validate()
    prepared = prepare(request, device=str(device), generator=generator)
    if prepared is None or getattr(prepared, "condition_latent", None) is None:
        raise RuntimeError(
            "Image-conditioned LingBot-Video refinement produced no condition "
            "latent; the refined frame 0 would drift from the input image."
        )
    return prepared


# --------------------------------------------------------------------------- #
# Refiner transformer lifecycle (on-demand load / release)
# --------------------------------------------------------------------------- #
class LingBotRefiner:
    """On-demand owner of the refiner DiT loaded from ``model_root/<subfolder>``.

    Loaded lazily for the refine stage, placed on the compute device, and
    released (moved off-device + dropped) afterwards so it never co-resides with
    the base DiT.
    """

    def __init__(self, model_config: Any, *, subfolder: str = "refiner") -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for the LingBot-Video refiner.")
        self.model_config = model_config
        self.model_root = Path(str(model_config.path))
        self.subfolder = str(subfolder or "refiner")
        self._transformer: Any | None = None
        self._transformer_config: dict[str, Any] | None = None
        self._block_swap_manager: Any | None = None
        self._block_hook_handles: list[Any] = []

    def has_weights(self) -> bool:
        return resolve_lingbot_transformer_dir(self.model_root, subfolder=self.subfolder) is not None

    def transformer_config(self) -> dict[str, Any]:
        if self._transformer_config is None:
            config = load_lingbot_transformer_config(self.model_root, subfolder=self.subfolder)
            for key in ("patch_size", "axes_dims", "axes_lens", "mlp_only_layers"):
                if key in config:
                    length = 3 if key in {"patch_size", "axes_dims", "axes_lens"} else len(config[key])
                    config[key] = _tuple_int(config[key], length=length)
            self._transformer_config = config
        return self._transformer_config

    @property
    def loaded(self) -> bool:
        return self._transformer is not None

    @property
    def transformer(self) -> Any:
        if self._transformer is None:
            raise RuntimeError("LingBot-Video refiner is not loaded; call load() first.")
        return self._transformer

    def load(
        self,
        *,
        device: str,
        dtype: Any | None = None,
        residency: Any | None = None,
        transformer_loader: Any | None = None,
    ) -> None:
        """Build (once) and place the refiner transformer on ``device``.

        Reuses the shared transformer loader with the refiner subfolder. Built and
        strict-loaded on first use; subsequent calls only re-place it.
        """
        if not self.has_weights():
            raise RuntimeError(
                f"--refine requested but no refiner weights were found under "
                f"'{self.model_root / self.subfolder}'. {_HF_REFINER_HINT}"
            )
        if self._transformer is None:
            config = self.transformer_config()
            if callable(transformer_loader):
                transformer = transformer_loader(
                    config,
                    subfolder=self.subfolder,
                    dtype=dtype,
                )
            else:
                transformer = LingBotVideoTransformer3DModel(**config)
                load_lingbot_transformer_checkpoint(
                    transformer,
                    self.model_root,
                    strict=True,
                    subfolder=self.subfolder,
                )
            transformer.eval()
            for param in transformer.parameters():
                param.requires_grad_(False)
            self._transformer = transformer
        if dtype is not None and not callable(transformer_loader):
            self._transformer.to(dtype=dtype)
        self._place(device=device, residency=residency)

    def _place(self, *, device: str, residency: Any | None) -> None:
        target = torch.device(device)
        transformer = self.transformer
        if residency is None or not bool(getattr(residency, "enabled", False)):
            transformer.to(device=target)
            return

        from mirai.core.training.residency.block_swap import BlockSwapManager
        from mirai.core.training.residency.tensor_residency import (
            move_tensors_outside_modules,
        )

        units = list(enumerate(transformer.blocks))
        blocks = [block for _index, block in units]
        move_tensors_outside_modules(
            transformer,
            excluded_modules=blocks,
            device=target,
        )
        manager = BlockSwapManager(
            total_blocks=len(units),
            blocks_to_swap=min(len(units), int(residency.blocks_to_swap)),
            mode=str(residency.mode),
            # In inference this flag means "evict after the forward". There is
            # no backward pass, so keeping it false would accumulate every
            # streamed block on the device during the first model call.
            block_swap_backward=True,
            block_residency_planner=str(residency.block_residency_planner),
            block_swap_prefetch_depth=int(residency.block_swap_prefetch_depth),
            block_residency_priority=str(residency.block_residency_priority),
            block_swap_transfer_strategy=str(residency.block_swap_transfer_strategy),
            disk_offload_dir=residency.offload_dir,
        )
        manager.bind(units, device=target)
        self._release_hooks()
        for index, block in units:
            self._block_hook_handles.append(
                block.register_forward_pre_hook(
                    lambda _module, _args, idx=index: manager.before_block(idx)
                )
            )
            self._block_hook_handles.append(
                block.register_forward_hook(
                    lambda _module, _args, output, idx=index: (
                        manager.after_block(idx), output
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
        if self._block_swap_manager is not None:
            self._block_swap_manager.release_device()
        self._block_swap_manager = None
        if self._transformer is not None:
            self._transformer.to(device=torch.device("cpu"))
            self._transformer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Refine policy loop (family-owned; model access via provider hooks only)
# --------------------------------------------------------------------------- #
def run_refine(
    *,
    pipeline: Any,
    refiner: LingBotRefiner,
    base_latent: "torch.Tensor",
    prompt: str,
    negative_prompt: str,
    seed: int,
    height: int,
    width: int,
    steps: int,
    cfg_scale: float,
    shift: float,
    t_thresh: float,
    sigma_tail_steps: int,
    scheduler: str,
    device: str,
    dtype: Any | None = None,
    conditioning: Any | None = None,
) -> "torch.Tensor":
    """Upscale -> re-noise -> tail-denoise a base latent into a refined latent.

    Returns the refined latent as ``[C,T,H,W]`` (DiT space) so it flows straight
    into the standard native VAE decode path. Model access is entirely through
    the pipeline's provider hooks.

    ``conditioning`` is the media request that drove the base pass. For an
    image-conditioned task the condition image is re-encoded at the REFINER's
    geometry — the base-resolution condition latent has the wrong spatial shape
    — and frame 0 is pinned to it before the loop, before every forward, and
    after every solver step, so the conditioned frame cannot drift while the
    remainder of the clip denoises against it.

    The base DiT is released before refiner assets are loaded. This operation
    does not restore it; the owning inference session must restore placement
    before a subsequent base denoise.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for the LingBot-Video refiner.")
    from mirai.core.training.preview.preview import _as_context_tensor

    if not refiner.has_weights():
        raise RuntimeError(
            f"--refine requested but no refiner weights were found under "
            f"'{refiner.model_root / refiner.subfolder}'. {_HF_REFINER_HINT}"
        )

    compute_device = torch.device(device)
    # One generator, seeded with the base seed, drives BOTH the VAE-encode
    # posterior sample and the re-noise draw — exactly upstream's single
    # ``torch.Generator(device).manual_seed(args.seed)``.
    generator = torch.Generator(device=compute_device)
    generator.manual_seed(int(seed))

    # The base latent is complete, so release its transformer before allocating
    # VAE encoder and refiner state.
    pipeline.release_base_transformer()

    # (a) upscale: decode base latent -> pixels -> bicubic resize -> VAE re-encode.
    pipeline.load_vae(device=str(compute_device))
    try:
        frames = pipeline.decode_latents_native([base_latent])  # [T,3,H,W] in [0,1]
        video = frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1,3,T,H,W]
        # Bicubic overshoots at sharp edges (values slip just outside [0,1]), and
        # the VAE encoder's pixel-range detection rejects such frames — clamp back
        # to the decode range before re-encoding.
        video = resize_video_bicubic(video, int(height), int(width)).clamp_(0.0, 1.0)
        x_up = pipeline.encode_video_native(video, generator=generator)  # [1,C,T,H,W] DiT space
    finally:
        pipeline.offload_vae()
    x_up = x_up.to(device=compute_device)

    # (b) re-noise at t_thresh with the SAME seeded generator.
    noise = torch.randn(
        x_up.shape, generator=generator, device=compute_device, dtype=x_up.dtype
    )
    latents = prepare_refiner_latent(x_up, noise, float(t_thresh))

    # (c) condition pinning: re-encode the condition media at the refiner's own
    # geometry. Drawn AFTER the re-noise so the unconditioned schedule consumes
    # exactly the same generator sequence as before.
    prepared = resolve_refiner_conditioning(
        pipeline,
        conditioning,
        height=int(height),
        width=int(width),
        device=str(compute_device),
        generator=generator,
    )
    if prepared is not None:
        prepared.pin_condition(latents)

    # Base DiT was already released above (before the VAE stage).
    try:
        refiner.load(
            device=str(compute_device),
            dtype=dtype,
            residency=pipeline.refiner_residency_request(),
            transformer_loader=pipeline.load_refiner_transformer,
        )
        # (e) TEXT-ONLY conditioning, by contract. The base pass may feed the
        # condition image to the VLM text encoder via ``encode_conditioned_prompt``;
        # the refiner deliberately does not, and re-anchors frame 0 through
        # ``pin_condition`` instead. ``encode_prompt`` is the text-only hook.
        pipeline.load_text_encoder(device="cpu")
        try:
            context = _as_context_tensor(
                pipeline.encode_prompt(prompt, device=str(compute_device)), compute_device
            )
            context_null = _as_context_tensor(
                pipeline.encode_prompt(negative_prompt or "", device=str(compute_device)),
                compute_device,
            )
        finally:
            pipeline.offload_text_encoder()

        # (d) tail grid via the preview solver registry, restricted to [t_thresh, 0].
        sigmas = compute_refiner_sigmas(
            num_inference_steps=int(steps),
            shift=float(shift),
            t_thresh=float(t_thresh),
            tail_steps=int(sigma_tail_steps),
        )
        solver = build_refiner_solver(
            str(scheduler),
            sigmas,
            shift=float(shift),
            device=str(compute_device),
        )

        scale = float(cfg_scale)
        autocast_ctx: Any = (
            torch.amp.autocast("cuda", dtype=dtype)
            if compute_device.type == "cuda"
            and dtype in {torch.float16, torch.bfloat16}
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            for sigma in solver.timesteps:
                if prepared is not None:
                    prepared.pin_condition(latents)
                timestep = sigma.reshape(1).to(compute_device)
                if scale <= 1.0:
                    v_out = pipeline.refiner_forward(latents, timestep, {"t5": context})
                else:
                    v_cond = pipeline.refiner_forward(latents, timestep, {"t5": context})
                    v_uncond = pipeline.refiner_forward(
                        latents, timestep, {"t5": context_null}
                    )
                    v_out = v_uncond + scale * (v_cond - v_uncond)
                result = solver.step(v_out.squeeze(0), timestep, latents.squeeze(0))
                prev = result.prev_sample
                latents = prev.unsqueeze(0) if prev.ndim == 4 else prev
                if prepared is not None:
                    prepared.pin_condition(latents)
    finally:
        refiner.release()
    refined = latents.squeeze(0) if latents.ndim == 5 else latents  # [C,T,H,W]
    return refined.detach().float()


def resolve_refiner_request(
    request: dict[str, Any], *, scheduler: str
) -> dict[str, Any]:
    """Fill an incoming request from :data:`LINGBOT_REFINER_DEFAULTS`.

    ``None`` means "the family decides", so the resolved dict states every
    parameter the stage will actually run with.
    """
    resolved: dict[str, Any] = {}
    for key, default in LINGBOT_REFINER_DEFAULTS.items():
        value = request.get(key)
        resolved[key] = type(default)(default if value is None else value)
    resolved["scheduler"] = str(request.get("scheduler") or scheduler)

    # Pre-flight range checks: the schedule is built after the base denoise and
    # after the refiner weights are placed, so an out-of-range value must be
    # rejected here rather than deep inside the tail loop.
    t_value = float(resolved["t_thresh"])
    if not (0.0 < t_value <= 1.0):
        raise RuntimeError(
            f"refiner t_thresh must lie in (0, 1], got {t_value}."
        )
    tail = int(resolved["sigma_tail_steps"])
    if not 0 <= tail <= MAX_REFINER_SIGMA_TAIL_STEPS:
        raise RuntimeError(
            "refiner sigma_tail_steps must lie in "
            f"[0, {MAX_REFINER_SIGMA_TAIL_STEPS}], got {tail}."
        )
    if int(resolved["steps"]) < 1:
        raise RuntimeError(
            f"refiner steps must be >= 1, got {int(resolved['steps'])}."
        )
    return resolved


__all__ = [
    "LINGBOT_REFINER_DEFAULTS",
    "MAX_REFINER_SIGMA_TAIL_STEPS",
    "LingBotRefiner",
    "resolve_refiner_conditioning",
    "resolve_refiner_request",
    "build_refiner_solver",
    "compute_refiner_sigmas",
    "prepare_refiner_latent",
    "resize_video_bicubic",
    "run_refine",
    "validate_refiner_sigmas",
]
