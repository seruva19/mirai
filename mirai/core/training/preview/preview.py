"""Training preview sampling helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from mirai.core.training.preview.preview_runtime import (
    activate_preview_runtime_assets,
    begin_preview_runtime,
    restore_preview_runtime,
)

try:
    import av
except ModuleNotFoundError:  # pragma: no cover
    av = None  # type: ignore[assignment]

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for preview generation: {exc}")


def _prompt_embed(prompt: str) -> float:
    digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(digest, "little") % 10000) / 10000.0


def _parse_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return max(1, int(value[0])), max(1, int(value[1]))
    text = str(value).strip().lower()
    if text and "x" in text:
        w_raw, h_raw = text.split("x", 1)
        try:
            return max(1, int(w_raw)), max(1, int(h_raw))
        except ValueError:
            pass
    return 64, 64


def run_cfg_denoise_loop(
    *,
    pipeline: Any,
    prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    seed: int,
    step: int,
    denoise_steps: int,
    scheduler: str,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    # Place validation-fixture tensors on the model's compute device. Block-swap /
    # residency may have materialised the model on CUDA while the preview inputs
    # would otherwise default to CPU, so resolve the device the same way the
    # native denoise loop does to avoid a cross-device forward.
    device = _resolve_model_compute_device(pipeline)
    dtype = _resolve_model_compute_dtype(pipeline)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + int(step))
    latent = torch.randn((1,), generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    steps = max(1, int(denoise_steps))
    dt = 1.0 / float(steps)
    scale = float(cfg_scale)
    schedule_key = str(scheduler).strip().lower()
    cond_embed = {"t5": torch.tensor([_prompt_embed(prompt)], dtype=dtype, device=device)}
    uncond_embed = {"t5": torch.tensor([_prompt_embed(negative_prompt)], dtype=dtype, device=device)}
    extra_forward_kwargs: dict[str, Any] = dict(pipeline.preview_extra_forward_kwargs())
    stats: list[dict[str, float]] = []
    with torch.no_grad():
        for idx in range(steps):
            t = torch.tensor([1.0 - (idx / float(steps))], dtype=torch.float32)
            if scale <= 1.0:
                v_out = pipeline.forward(latent, t, cond_embed, **extra_forward_kwargs)
            else:
                v_cond = pipeline.forward(latent, t, cond_embed, **extra_forward_kwargs)
                v_uncond = pipeline.forward(latent, t, uncond_embed, **extra_forward_kwargs)
                v_out = v_uncond + scale * (v_cond - v_uncond)
            if schedule_key == "euler":
                latent = latent - (v_out * dt)
            else:
                latent = latent - (v_out * dt)
            stats.append(
                {
                    "step": float(idx),
                    "latent_mean": float(latent.detach().float().mean().item()),
                    "latent_std": float(latent.detach().float().std(unbiased=False).item()),
                }
            )
    return latent, stats


def _as_context_tensor(context: Any, device: "torch.device") -> "torch.Tensor":
    """Normalise an ``encode_prompt`` result to a single (1, S, D) tensor.

    Native text encoders may return a list of per-prompt tensors or a single
    tensor. The denoise loop encodes one prompt at a time, so collapse either
    form to ``(1, S, D)`` on ``device``.
    """
    if isinstance(context, (list, tuple)):
        context = context[0] if len(context) == 1 else torch.stack(list(context), dim=0)
    if not torch.is_tensor(context):
        context = torch.as_tensor(context)
    context = context.to(device)
    if context.ndim == 2:
        context = context.unsqueeze(0)
    return context


def _resolve_model_compute_device(pipeline: Any) -> "torch.device":
    """Return the device the model actually lives on for preview sampling.

    The preview noise must be created on the same device as the transformer
    weights. Block-swap / residency may leave some blocks on CPU while the
    resident parameters (e.g. patch embedding) stay on GPU, so prefer a CUDA
    parameter when present. Falls back to the cuda/cpu heuristic when no
    parameters are found.
    """
    candidates = []
    model = pipeline.get_training_model()
    if model is not None:
        candidates.append(model)
    for attr in ("low_noise_model", "high_noise_model"):
        expert = getattr(pipeline, attr, None)
        if expert is not None:
            candidates.append(expert)
    fallback: Any = None
    for module in candidates:
        try:
            params = module.parameters()
        except Exception:
            continue
        for param in params:
            dev = param.device
            if dev.type == "cuda":
                return dev
            if fallback is None:
                fallback = dev
    if fallback is not None:
        return fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_model_compute_dtype(pipeline: Any) -> "torch.dtype":
    """Return the dtype of the model's leading parameters for preview sampling.

    The trainer casts the model to its compute dtype (e.g. bf16), so validation
    preview inputs built here must match that dtype or the first matmul fails
    with a Float/BFloat16 mismatch. Falls back to float32 when no parameters
    are found.
    """
    model = pipeline.get_training_model()
    if model is not None:
        try:
            for param in model.parameters():
                return param.dtype
        except Exception:  # pragma: no cover - defensive
            pass
    return torch.float32


def run_native_denoise_loop(
    *,
    pipeline: Any,
    prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    seed: int,
    step: int,
    denoise_steps: int,
    scheduler: str,
    frame_count: int = 17,
    height: int = 480,
    width: int = 832,
    solver_name: str = "euler",
    cfg_mode: str = "sequential",
    forward_fn: Any = None,
    conditioning: Any | None = None,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Real denoise loop using native T5 encoding and flow matching solver.

    ``forward_fn`` overrides the per-step model call; default ``None`` uses
    ``pipeline.forward`` directly. An ``InferenceSession`` may pass a
    ``torch.compile``-wrapped forward and reuses it across steps.
    """
    fwd = forward_fn if forward_fn is not None else pipeline.forward
    from mirai.core.training.preview.preview_solvers import (
        PreviewSolverSpec,
        resolve_preview_solver,
    )

    scale = float(cfg_scale)
    cfg_mode_key = str(cfg_mode).strip().lower()
    if cfg_mode_key not in {"sequential", "batched"}:
        raise ValueError(f"Unknown CFG mode '{cfg_mode}'.")
    device = _resolve_model_compute_device(pipeline)

    from mirai.core.inference.conditioning import (
        VIDEO_TO_VIDEO,
        InferenceConditioningRequest,
        PreparedInferenceConditioning,
        blend_source_with_noise,
        denoising_schedule_start,
    )

    request = conditioning or InferenceConditioningRequest(
        frame_count=int(frame_count),
        height=int(height),
        width=int(width),
    )
    request.validate()

    # Model-agnostic latent geometry: [C, T_lat, H_lat, W_lat]
    channels, t_lat, h_lat, w_lat = pipeline.preview_latent_geometry(
        frame_count=frame_count, height=height, width=width
    )
    target_shape = (channels, t_lat, h_lat, w_lat)

    # A dedicated per-run generator owns both provider-side posterior sampling
    # and the initial diffusion noise without touching global RNG state.
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + int(step))
    prepare_conditioning = getattr(
        pipeline, "prepare_inference_conditioning", None
    )
    prepared = (
        prepare_conditioning(
            request,
            device=str(device),
            generator=g,
        )
        if callable(prepare_conditioning)
        else PreparedInferenceConditioning.unconditioned(request)
    )
    if not isinstance(prepared, PreparedInferenceConditioning):
        raise TypeError(
            "prepare_inference_conditioning() must return "
            "PreparedInferenceConditioning."
        )

    # Encode prompts via the native text encoder. Different families may return
    # different tensor/list shapes, so normalise to a single (1, S, D) tensor.
    pipeline.load_text_encoder(device="cpu")
    encode_conditioned = getattr(pipeline, "encode_conditioned_prompt", None)
    encode = (
        (
            lambda text: encode_conditioned(
                text,
                prepared=prepared,
                device=str(device),
            )
        )
        if callable(encode_conditioned)
        else (lambda text: pipeline.encode_prompt(text, device=str(device)))
    )
    context = _as_context_tensor(encode(prompt), device)
    context_null = _as_context_tensor(
        encode(negative_prompt or ""),
        device,
    )
    pipeline.offload_text_encoder()

    # Generate initial noise
    noise = torch.randn(target_shape, dtype=torch.float32, device=device, generator=g)

    # Resolve the configured solver through the preview solver registry.
    steps = max(1, int(denoise_steps))
    flow_shift = float(
        pipeline.resolve_flow_shift_for_latent_shape((1, *target_shape))
    )
    solver = resolve_preview_solver(
        solver_name,
        PreviewSolverSpec(
            num_inference_steps=steps,
            flow_shift=flow_shift,
            device=str(device),
            num_train_timesteps=1000,
        ),
    )
    if request.task == VIDEO_TO_VIDEO:
        source = torch.as_tensor(prepared.source_latent)
        if source.ndim == 5 and int(source.shape[0]) == 1:
            source = source[0]
        if tuple(source.shape) != target_shape:
            raise ValueError(
                "V2V source latent does not match requested geometry: "
                f"{tuple(source.shape)} vs {target_shape}."
            )
        source = source.to(device=device, dtype=noise.dtype)
        begin_index = denoising_schedule_start(
            steps, float(prepared.denoising_strength)
        )
        set_begin_index = getattr(solver, "set_begin_index", None)
        if not callable(set_begin_index):
            raise ValueError(
                f"Solver '{solver_name}' does not support V2V schedule truncation."
            )
        set_begin_index(begin_index)
        if len(solver.timesteps) == 0:
            return source, []
        sigma = solver.timesteps[0].to(device=device, dtype=noise.dtype)
        latents = [blend_source_with_noise(source, noise, sigma)]
    else:
        latents = [noise]
    prepared.pin_condition(latents[0])

    # Accumulate scalar statistics on device and transfer them together after
    # denoising to avoid synchronizing the device inside every step.
    stat_accum: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    saliency_guidance = None
    uses_saliency_guidance = bool(
        getattr(pipeline, "uses_previous_clean_routing_guidance", lambda: False)()
    )
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for idx, t_val in enumerate(solver.timesteps):
            prepared.pin_condition(latents[0])
            timestep = t_val.unsqueeze(0).to(device)
            x_list = latents
            sample_a = torch.stack(x_list, dim=0)
            if uses_saliency_guidance and saliency_guidance is None:
                saliency_guidance = sample_a.detach()

            text_cond = {"t5": context}
            if scale <= 1.0:
                v_out = fwd(
                    sample_a,
                    timestep,
                    text_cond,
                    **(
                        {"routing_guidance_latents": saliency_guidance}
                        if uses_saliency_guidance
                        else {}
                    ),
                )
            else:
                if cfg_mode_key == "batched":
                    from mirai.core.inference.cfg_batching import (
                        build_batched_cfg_inputs,
                        split_batched_cfg_output,
                    )

                    batch = build_batched_cfg_inputs(
                        sample=sample_a,
                        timestep=timestep,
                        conditional=context,
                        unconditional=context_null,
                    )
                    v_uncond, v_cond = split_batched_cfg_output(
                        fwd(
                            batch.sample,
                            batch.timestep,
                            batch.text_embeds,
                            **(
                                {
                                    "routing_guidance_latents": torch.cat(
                                        (saliency_guidance, saliency_guidance),
                                        dim=0,
                                    )
                                }
                                if uses_saliency_guidance
                                else {}
                            ),
                        )
                    )
                else:
                    text_uncond = {"t5": context_null}
                    guidance_kwargs = (
                        {"routing_guidance_latents": saliency_guidance}
                        if uses_saliency_guidance
                        else {}
                    )
                    v_cond = fwd(sample_a, timestep, text_cond, **guidance_kwargs)
                    v_uncond = fwd(sample_a, timestep, text_uncond, **guidance_kwargs)
                v_out = v_uncond + scale * (v_cond - v_uncond)

            if uses_saliency_guidance:
                sigma = timestep.detach().float()
                sigma = sigma.reshape(
                    int(sigma.shape[0]),
                    *((1,) * (int(v_out.ndim) - 1)),
                ).to(device=v_out.device, dtype=v_out.dtype)
                saliency_guidance = (sample_a - sigma * v_out).detach()

            # Solver step
            sample = torch.stack(x_list, dim=0) if not isinstance(x_list[0], torch.Tensor) or x_list[0].ndim < 4 else x_list[0].unsqueeze(0)
            if v_out.ndim == 5 and sample.ndim == 5:
                result = solver.step(v_out.squeeze(0), timestep, sample.squeeze(0))
                latents = [result.prev_sample]
            else:
                result = solver.step(v_out, timestep, sample)
                latents = [result.prev_sample.squeeze(0)] if result.prev_sample.ndim == 5 else [result.prev_sample]
            prepared.pin_condition(latents[0])

            latent_stat = latents[0].detach().float()
            stat_accum.append(
                (idx, latent_stat.mean(), latent_stat.std(unbiased=False))
            )

    if stat_accum:
        means = torch.stack([m for _, m, _ in stat_accum]).cpu().tolist()
        stds = torch.stack([s for _, _, s in stat_accum]).cpu().tolist()
        stats: list[dict[str, float]] = [
            {
                "step": float(idx),
                "latent_mean": float(means[i]),
                "latent_std": float(stds[i]),
            }
            for i, (idx, _mean_t, _std_t) in enumerate(stat_accum)
        ]
    else:
        stats = []

    return latents[0], stats


def _write_mp4(path: Path, frames: torch.Tensor, fps: int = 8) -> bool:
    if av is None:
        return False
    if frames.ndim != 4 or int(frames.shape[1]) != 3:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count, _, height, width = frames.shape
    try:
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=int(fps))
            stream.width = int(width)
            stream.height = int(height)
            stream.pix_fmt = "yuv420p"
            for idx in range(int(frame_count)):
                rgb = (
                    frames[idx]
                    .detach()
                    .float()
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception:
        return False
    return True


def _write_png_fallback(dir_path: Path, stem: str, frames: torch.Tensor) -> Path:
    if Image is None:
        raise RuntimeError("Pillow is required for PNG fallback preview writing.")
    dir_path.mkdir(parents=True, exist_ok=True)
    out_dir = dir_path / f"{stem}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(int(frames.shape[0])):
        rgb = (
            frames[idx]
            .detach()
            .float()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        Image.fromarray(rgb, mode="RGB").save(out_dir / f"{stem}_{idx:04d}.png")
    return out_dir


def _decode_validation_preview_frames(
    *,
    pipeline: Any,
    pred: torch.Tensor,
    scaling_factor: float,
    frame_count: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, int]:
    unscaled = pred.detach().float() / max(float(scaling_factor), 1e-8)
    supports_decode = getattr(pipeline, "supports_preview_latent_decode", None)
    if callable(supports_decode) and bool(supports_decode()):
        decode_fn = getattr(pipeline, "decode_latents", None)
        if not callable(decode_fn):
            raise RuntimeError(
                "Pipeline declares preview_latent_decode but does not expose decode_latents."
            )
        frames, decode_chunk_count = decode_fn(
            unscaled,
            frame_count=frame_count,
            height=height,
            width=width,
            chunk_size=getattr(pipeline, "model_config", None).params.vae_chunk_size
            if getattr(getattr(pipeline, "model_config", None), "params", None) is not None
            else 16,
            temporal_overlap=1,
        )
        return frames, int(decode_chunk_count)

    value = float(unscaled.detach().float().mean().item())
    frames = torch.zeros((frame_count, 3, height, width), dtype=torch.float32)
    for idx in range(frame_count):
        frames[idx].fill_((((value * 127.0) + idx * 3.0) % 255.0) / 255.0)
    return frames, max(1, (frame_count + max(1, 16) - 1) // max(1, 16))


def _resolve_preview_denoise_mode(pipeline: Any) -> str:
    if bool(pipeline.has_native_inference()):
        return "native"
    supports_validation = getattr(pipeline, "supports_validation_inference", None)
    if callable(supports_validation) and bool(supports_validation()):
        return "validation"
    raise RuntimeError(
        f"{type(pipeline).__name__} does not expose native inference assets "
        "and does not declare validation-fixture preview/inference support."
    )


def generate_preview(
    *,
    trainer,
    step: int,
    output_dir: str | Path,
    prompt: str,
    seed: int,
    sample_blocks_to_swap: int,
    training_blocks_to_swap: int,
    block_swap_mode: str,
    block_swap_backward: bool,
    sample_name: str = "preview",
    sample_cfg_scale: float = 7.5,
    sample_negative_prompt: str = "",
    sample_solver: str = "euler",
    sample_resolution: str = "smallest_bucket",
    sample_frame_count: int = 16,
    denoise_steps: int = 6,
) -> dict[str, Any]:
    pipeline = trainer.pipeline
    runtime = begin_preview_runtime(
        pipeline=pipeline,
        sample_blocks_to_swap=sample_blocks_to_swap,
        training_blocks_to_swap=training_blocks_to_swap,
        block_swap_mode=block_swap_mode,
        block_swap_backward=block_swap_backward,
    )
    decode_chunk_count = 1
    try:
        activate_preview_runtime_assets(runtime)

        uncond_prompt = (
            str(sample_negative_prompt) if str(sample_negative_prompt).strip() else ""
        )
        width, height = _parse_resolution(sample_resolution)
        frame_count = max(1, int(sample_frame_count))

        # Validation fixtures opt into the synthetic path explicitly; supported
        # model families must complete their native preview path or fail.
        preview_mode = _resolve_preview_denoise_mode(pipeline)
        use_native = preview_mode == "native"
        native_preview_error = ""
        if use_native:
            pred, denoise_stats = run_native_denoise_loop(
                pipeline=pipeline,
                prompt=prompt,
                negative_prompt=uncond_prompt,
                cfg_scale=float(sample_cfg_scale),
                seed=int(seed),
                step=int(step),
                denoise_steps=int(denoise_steps),
                scheduler=str(sample_solver),
                frame_count=frame_count,
                height=height,
                width=width,
                solver_name=str(sample_solver),
            )
        if not use_native:
            pred, denoise_stats = run_cfg_denoise_loop(
                pipeline=pipeline,
                prompt=prompt,
                negative_prompt=uncond_prompt,
                cfg_scale=float(sample_cfg_scale),
                seed=int(seed),
                step=int(step),
                denoise_steps=int(denoise_steps),
                scheduler=str(sample_solver),
            )

        out_dir = Path(output_dir) / "samples" / f"step_{step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = str(sample_name).strip() or "preview"
        out_path = out_dir / f"{name}.pt"
        torch.save(pred.detach().cpu(), out_path)

        # Decode latents to pixel frames. A native decode failure invalidates the
        # preview and propagates to the caller; fabricated frames are never emitted.
        if use_native:
            pipeline.load_vae(device="cuda" if torch.cuda.is_available() else "cpu")
            try:
                frames = pipeline.decode_latents_native([pred])
                decode_chunk_count = 1
            finally:
                pipeline.offload_vae()
        else:
            frames, decode_chunk_count = _decode_validation_preview_frames(
                pipeline=pipeline,
                pred=pred,
                scaling_factor=runtime.scaling_factor,
                frame_count=frame_count,
                height=height,
                width=width,
            )

        video_path = out_dir / f"{name}.mp4"
        wrote_mp4 = _write_mp4(video_path, frames)
        png_dir = ""
        if not wrote_mp4:
            video_path = Path()
            png_dir = str(_write_png_fallback(out_dir, name, frames))
    finally:
        restore_preview_runtime(runtime)

    payload = {
        "sample_path": str(out_path),
        "sample_video_path": str(video_path) if str(video_path) else "",
        "sample_frames_dir": png_dir,
        "sample_blocks_to_swap": int(sample_blocks_to_swap),
        "sample_block_swap_overridden": bool(runtime.block_swap_overridden),
        "sample_cfg_scale": float(sample_cfg_scale),
        "sample_cfg_enabled": bool(float(sample_cfg_scale) > 1.0),
        "sample_solver": str(sample_solver),
        "sample_resolution": f"{int(width)}x{int(height)}",
        "sample_frame_count": int(sample_frame_count),
        "sample_adapter_merged": bool(runtime.adapter_merged),
        "sample_adapter_unmerged": bool(runtime.adapter_unmerged),
        "sample_vae_loaded": bool(runtime.vae_loaded),
        "sample_vae_unloaded": bool(runtime.vae_unloaded),
        "sample_decode_chunk_count": int(decode_chunk_count),
        "sample_vae_scaling_factor": float(runtime.scaling_factor),
        "sample_latents_unscaled": True,
        "sample_denoise_steps": int(len(denoise_stats)),
        "sample_denoise_final_mean": float(denoise_stats[-1]["latent_mean"]) if denoise_stats else 0.0,
        "sample_denoise_final_std": float(denoise_stats[-1]["latent_std"]) if denoise_stats else 0.0,
        "sample_native_used": bool(use_native),
        "sample_native_preview_error": native_preview_error,
    }
    return payload
