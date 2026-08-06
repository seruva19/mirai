"""Load-once, generate-many inference session.

``InferenceSession.from_config`` runs the one-time build that
``scripts/infer.py`` performs per process today: register builtin components,
load the runtime config, construct the :class:`Trainer`, load an optional
checkpoint/adapter, switch to eval, set the LoRA scale, resolve the inference
mode, and place the pipeline on the compute device. ``session.generate`` then
runs a single denoise loop + latent dump + VAE decode and returns the same
payload dict ``scripts/infer.py`` prints.

Building the model once and calling ``generate`` many times amortizes the fixed
~15 s build/placement cost across N prompts (N x (build + denoise) collapses to
build + N x denoise). The numerics of a single ``generate`` call are identical
to a single ``scripts/infer.py`` run for the same seed and geometry.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from mirai.config.runtime_policy import runtime_policy_summary
from mirai.core.builtins import register_builtin_components
from mirai.core.inference.prompt_rewriter import (
    PromptRewriteRequest,
    rewrite_inference_prompts,
)
from mirai.core.inference.component_residency import (
    resolve_inference_component_residency,
)
from mirai.core.models.native_video import resolve_output_fps
from mirai.core.models.providers import get_model_family_provider
from mirai.core.training.adapters import load_adapter_payload, normalize_adapter_state
from mirai.core.persistence.checkpoints import load_checkpoint
from mirai.core.training.runtime.cli import (
    emit_runtime_policy_notes,
    load_runtime_config,
)
from mirai.core.training.residency.device_placement import (
    place_pipeline_on_device,
    resolve_compute_device,
    resolve_compute_dtype,
)
from mirai.core.training.trainer import Trainer
from mirai.core.training.runtime.trainer import _prepare_forward_fn

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")


def _sync_perf_counter() -> float:
    """Synchronize pending CUDA work before reading ``perf_counter``."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def decode_pipeline_video(
    *, pipeline, pred: torch.Tensor, video_path: Path, fps: float | None = None
) -> None:
    from mirai.core.training.preview.preview import _write_mp4

    fps = resolve_output_fps(pipeline=pipeline, requested=fps)
    try:
        pipeline.load_vae(
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        frames = pipeline.decode_latents_native([pred])
        written = bool(_write_mp4(video_path, frames, fps=float(fps)))
    except Exception as exc:
        raise RuntimeError(f"Native VAE decode failed: {exc}") from exc
    finally:
        offload_vae = getattr(pipeline, "offload_vae", None)
        if callable(offload_vae):
            try:
                offload_vae()
            except Exception:
                pass
    # _write_mp4 is best-effort for training previews and signals failure via
    # its return value instead of raising; here a missing file is a failure.
    if not written:
        raise RuntimeError(
            f"Video write produced no file at {video_path}: PyAV ('av') is not "
            f"installed or decoded frames have an unexpected shape "
            f"{tuple(frames.shape)} (expected (T, 3, H, W))."
        )


def decode_pipeline_media(
    *,
    pipeline,
    pred: torch.Tensor,
    out_path: Path,
    task: str,
    fps: float | None = None,
) -> Path:
    """Decode one latent and write the task's native PNG or MP4 artifact."""
    from mirai.core.inference.conditioning import TEXT_TO_IMAGE
    from mirai.core.training.preview.preview import _write_mp4

    fps = resolve_output_fps(pipeline=pipeline, requested=fps)
    task_key = str(task)
    media_path = (
        Path(out_path).with_suffix(".png")
        if task_key == TEXT_TO_IMAGE
        else Path(out_path).with_suffix(".mp4")
    )
    try:
        pipeline.load_vae(
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        frames = pipeline.decode_latents_native([pred])
        if task_key == TEXT_TO_IMAGE:
            if int(frames.ndim) != 4 or tuple(frames.shape[:2]) != (1, 3):
                raise ValueError(
                    "T2I decode must produce exactly one RGB frame; got "
                    f"{tuple(frames.shape)}."
                )
            from PIL import Image

            media_path.parent.mkdir(parents=True, exist_ok=True)
            rgb = (
                frames[0]
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
            Image.fromarray(rgb, mode="RGB").save(media_path)
            written = media_path.is_file()
        else:
            written = bool(_write_mp4(media_path, frames, fps=float(fps)))
    except Exception as exc:
        raise RuntimeError(f"Native VAE decode failed: {exc}") from exc
    finally:
        offload_vae = getattr(pipeline, "offload_vae", None)
        if callable(offload_vae):
            try:
                offload_vae()
            except Exception:
                pass
    if not written:
        raise RuntimeError(
            f"Media write produced no file at {media_path}; decoded shape was "
            f"{tuple(frames.shape)}."
        )
    return media_path


def resolve_inference_mode(pipeline) -> str:
    if bool(pipeline.has_native_inference()):
        return "native"
    supports_validation = getattr(pipeline, "supports_validation_inference", None)
    if callable(supports_validation) and bool(supports_validation()):
        return "validation"
    raise RuntimeError(
        f"{type(pipeline).__name__} does not expose native inference or an "
        "explicit validation-fixture inference contract."
    )


class InferenceSession:
    """A built-once pipeline that can service many ``generate`` calls.

    Construct via :meth:`from_config`; do not instantiate directly.
    """

    def __init__(
        self,
        *,
        trainer: Trainer,
        cfg: Any,
        runtime_policy_notes: list,
        inference_mode: str,
        checkpoint: str,
        adapter: str,
        lora_scale: float,
        effective_scale: float,
        lora_format: str,
        merge: bool,
        compile_mode: str = "",
        forward_fn: Any = None,
        compile_enabled: bool = False,
        compile_warning: str = "",
        place_on_device: bool = True,
        place_fn: Any = None,
        compute_device: Any = None,
        compute_dtype: Any = None,
        residency_strategy: str = "disabled",
    ) -> None:
        self.trainer = trainer
        self.pipeline = trainer.pipeline
        self.cfg = cfg
        self.runtime_policy_notes = runtime_policy_notes
        self.inference_mode = inference_mode
        self.use_native = inference_mode == "native"
        self.checkpoint = checkpoint
        self.adapter = adapter
        self.lora_scale = lora_scale
        self.effective_scale = effective_scale
        self.lora_format = lora_format
        self.merge = merge
        # Default-off compilation leaves the direct forward path unchanged.
        self.compile_mode = str(compile_mode or "")
        self.compile_enabled = bool(compile_enabled)
        self.compile_warning = str(compile_warning or "")
        self._forward_fn = forward_fn
        self._resident_text_encoder = False
        self._resident_vae = False
        self._place_on_device = bool(place_on_device)
        self._place_fn = place_fn
        self._compute_device = compute_device
        self._compute_dtype = compute_dtype
        self._residency_strategy = str(residency_strategy)
        self._base_placement_dirty = False
        # Armed by from_config only when inference.expert_feature_cache is on;
        # None means no cache object exists anywhere in the run.
        self._expert_feature_cache: Any = None

    # -- construction -------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config_path: str,
        *,
        checkpoint: str = "",
        adapter: str = "",
        lora_format: str = "auto",
        lora_scale: float = 1.0,
        merge: bool = False,
        place_on_device: bool = True,
        keep_text_encoder_resident: bool | None = None,
        keep_vae_resident: bool | None = None,
        compile_mode: str = "",
        place_fn=None,
        device_fn=None,
        dtype_fn=None,
    ) -> "InferenceSession":
        """Perform the one-time build ``scripts/infer.py`` does per process.

        ``place_fn`` / ``device_fn`` / ``dtype_fn`` default to the shared
        ``device_placement`` helpers; ``scripts/infer.py`` passes its own
        module-level references. ``place_on_device=False`` supports latent decode,
        where the VAE owns device placement.

        ``compile_mode`` is the opt-in ``torch.compile`` inference seam:
        ``""`` (default) leaves the denoise loop calling ``pipeline.forward``
        directly; ``"default"`` /
        ``"reduce-overhead"`` / ``"max-autotune"`` resolve a compiled forward
        ONCE on the session (``dynamic=False``). ``reduce-overhead`` lets
        torch.compile apply CUDA graphs per-region where safe. Any compile
        construction failure returns the eager callable and records one warning.
        """
        _place_fn = place_fn or place_pipeline_on_device
        _device_fn = device_fn or resolve_compute_device
        _dtype_fn = dtype_fn or resolve_compute_dtype

        adapter_path = str(adapter or "").strip()
        register_builtin_components()
        cfg, runtime_policy_notes = load_runtime_config(
            config_path,
            entrypoint="infer",
        )
        emit_runtime_policy_notes(runtime_policy_notes)
        trainer = Trainer(cfg)

        if checkpoint:
            ckpt = load_checkpoint(checkpoint)
            trainer.load_state_dict(ckpt.get("trainer_state", {}))
        if adapter_path:
            adapter_payload = load_adapter_payload(adapter_path)
            trainer.pipeline.load_adapter_state(
                normalize_adapter_state(adapter_payload, lora_format=lora_format)
            )
        trainer.pipeline.eval()
        trainer.pipeline.set_lora_scale(lora_scale)
        merge_applied = False
        if merge:
            if not bool(trainer.pipeline.supports_adapter_merge_unmerge()):
                raise RuntimeError(
                    f"{type(trainer.pipeline).__name__} does not support adapter "
                    "merge/unmerge; --merge cannot be honored."
                )
            merge_applied = bool(trainer.pipeline.merge_adapter())
            if not merge_applied:
                raise RuntimeError("Adapter merge was requested but did not complete.")
        effective_scale = 1.0 if merge_applied else lora_scale

        inference_mode = resolve_inference_mode(trainer.pipeline)
        compute_device = _device_fn()
        compute_dtype = _dtype_fn(cfg)
        residency_strategy = str(
            getattr(trainer, "weight_residency_strategy", "disabled")
        )
        if place_on_device:
            # Same placement contract as training: without it the denoiser
            # stays in host RAM and the denoise loop silently runs on CPU
            # (hours instead of minutes for a real MoE family). CPU-only
            # hosts keep the pre-placement fp32 semantics.
            if compute_device.type == "cuda":
                _place_fn(
                    trainer.pipeline,
                    device=compute_device,
                    dtype=compute_dtype,
                    residency_strategy=residency_strategy,
                )

        # Resolve the compiled forward once after placement so torch.compile
        # traces the on-device module. When disabled, forward_fn stays None.
        # A construction failure yields a
        # single warning and falls back to eager (forward_fn None).
        forward_fn = None
        compile_enabled = False
        compile_warning = ""
        requested_mode = str(compile_mode or "").strip()
        if requested_mode:
            # Belt-and-suspenders over the construction-level try/except below:
            # make torch.compile fall back to eager per-region on ANY runtime
            # backend/graph-break failure (e.g. no C++ toolchain on a CPU host,
            # or a D2H sync that can't be captured) instead of crashing the run.
            try:  # pragma: no cover - depends on torch build
                import torch._dynamo as _dynamo

                _dynamo.config.suppress_errors = True
            except Exception:
                pass
            resolved_fn, compile_enabled, compile_warning = _prepare_forward_fn(
                pipeline=trainer.pipeline,
                compile_requested=True,
                compile_mode=requested_mode,
                dynamic=False,
            )
            forward_fn = resolved_fn if compile_enabled else None
            if compile_warning:
                print(f"[warning] {compile_warning}", file=sys.stderr)

        session = cls(
            trainer=trainer,
            cfg=cfg,
            runtime_policy_notes=runtime_policy_notes,
            inference_mode=inference_mode,
            checkpoint=str(checkpoint) if checkpoint else "",
            adapter=adapter_path,
            lora_scale=lora_scale,
            effective_scale=effective_scale,
            lora_format=lora_format,
            merge=merge_applied,
            compile_mode=requested_mode,
            forward_fn=forward_fn,
            compile_enabled=compile_enabled,
            compile_warning=compile_warning,
            place_on_device=place_on_device,
            place_fn=_place_fn,
            compute_device=compute_device,
            compute_dtype=compute_dtype,
            residency_strategy=residency_strategy,
        )
        component_residency = resolve_inference_component_residency(
            config_text_encoder=bool(cfg.inference.keep_text_encoder_resident),
            config_vae=bool(cfg.inference.keep_vae_resident),
            text_encoder_override=keep_text_encoder_resident,
            vae_override=keep_vae_resident,
        )
        session._arm_expert_feature_cache()
        if component_residency.text_encoder:
            session._make_text_encoder_resident()
        if component_residency.vae:
            session._make_vae_resident()
        return session

    # -- expert-branch feature cache ----------------------------------------
    def _arm_expert_feature_cache(self) -> None:
        """Build and attach the cross-timestep branch cache when configured.

        Nothing is imported or constructed while
        ``inference.expert_feature_cache='off'``, so the default run holds no
        cache object and the MoE seam is the one the memory policy resolved.
        """
        inference_cfg = self.cfg.inference
        if str(getattr(inference_cfg, "expert_feature_cache", "off")) == "off":
            return
        from mirai.core.moe.runtime.expert_feature_cache import (
            ExpertFeatureCache,
            ExpertFeatureCachePolicy,
        )

        cache = ExpertFeatureCache(
            ExpertFeatureCachePolicy(
                mode=str(inference_cfg.expert_feature_cache),
                drift_threshold=float(
                    inference_cfg.expert_feature_cache_drift_threshold
                ),
                max_reuse_span=int(
                    inference_cfg.expert_feature_cache_max_reuse_span
                ),
                slots=int(inference_cfg.expert_feature_cache_slots),
            )
        )
        self.pipeline.configure_expert_feature_cache(cache)
        self._expert_feature_cache = cache

    # -- resident-weight seams ---------------------------------------------
    # These opt-in flags leave the pipeline unchanged when disabled. When
    # enabled, the load/offload hooks are replaced so the text encoder or VAE remains
    # on the compute device across generate() calls. close() restores the hooks
    # and offloads the resident modules.
    def _make_text_encoder_resident(self) -> None:
        if self._resident_text_encoder:
            return
        self.pipeline.load_text_encoder(device=str(self._compute_device))
        self._orig_load_text_encoder = self.pipeline.load_text_encoder
        self._orig_offload_text_encoder = self.pipeline.offload_text_encoder
        self.pipeline.load_text_encoder = lambda *, device=None: None
        self.pipeline.offload_text_encoder = lambda: None
        self._resident_text_encoder = True

    def _make_vae_resident(self) -> None:
        if self._resident_vae:
            return
        self._orig_offload_vae = self.pipeline.offload_vae
        self.pipeline.offload_vae = lambda: None
        self._resident_vae = True

    def close(self) -> None:
        """Restore any resident-weight hooks and offload the held modules."""
        if self._resident_text_encoder:
            del self.pipeline.load_text_encoder
            del self.pipeline.offload_text_encoder
            self._resident_text_encoder = False
            try:
                self.pipeline.offload_text_encoder()
            except Exception:
                pass
        if self._resident_vae:
            del self.pipeline.offload_vae
            self._resident_vae = False
            try:
                self.pipeline.offload_vae()
            except Exception:
                pass

    def _ensure_base_placement(self) -> None:
        """Restore the base transformer after an on-demand refiner released it."""

        if not self._base_placement_dirty:
            return
        if (
            self._place_on_device
            and self._place_fn is not None
            and getattr(self._compute_device, "type", "cpu") == "cuda"
        ):
            self._place_fn(
                self.pipeline,
                device=self._compute_device,
                dtype=self._compute_dtype,
                residency_strategy=self._residency_strategy,
            )
        self._base_placement_dirty = False

    # -- refiner pre-flight ------------------------------------------------
    def _validate_refine_request(
        self, refine: dict, *, frames: int, height: int, width: int
    ) -> dict:
        """Fail fast on an unusable --refine request before the base denoise.

        Rejects invalid requests before a base denoise run is started, and
        returns the request the family resolved, so the run reports the values
        that drove it rather than the values that were typed.
        """
        resolved = self.pipeline.validate_refinement_request(
            refine, frames=int(frames), height=int(height), width=int(width)
        )
        if not isinstance(resolved, dict):
            raise RuntimeError(
                f"{type(self.pipeline).__name__}.validate_refinement_request must "
                "return the resolved refinement request."
            )
        return resolved

    def __enter__(self) -> "InferenceSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- generation ---------------------------------------------------------
    def generate(
        self,
        *,
        prompt: str,
        out_path,
        negative_prompt: str = "",
        seed: int = 42,
        steps: int = 20,
        cfg_scale: float = 5.0,
        frames: int = 17,
        height: int = 480,
        width: int = 832,
        fps: float | None = None,
        scheduler: str = "euler",
        decode_latent: str = "",
        allow_latent_output_only: bool = False,
        timings: dict | None = None,
        refine: dict | None = None,
        prompt_rewriter: str | None = None,
        cfg_mode: str | None = None,
        task: str | None = None,
        input_image: Any | None = None,
        input_video: Any | None = None,
        denoising_strength: float | None = None,
    ) -> dict:
        """Run one denoise + decode and return the ``scripts/infer.py`` payload.

        This is the per-run body of ``scripts/infer.py`` (denoise loop, latent
        dump, VAE decode, file write, payload assembly), unchanged except that
        the model build/placement it shared has already happened in
        :meth:`from_config`.

        ``refine=None`` leaves the base payload and behavior unchanged. When a
        dict of refiner parameters is supplied, the model family resolves it
        against its own released refinement profile — an entry left ``None``
        means "the family decides" — and then hands the base final latent to its
        on-demand refiner before the standard VAE decode. Which parameters are
        meaningful, and what the stage does with them, is family-owned; the base
        transformer is released off the compute device before the refiner is
        loaded (VRAM realism).
        """
        output_fps = resolve_output_fps(pipeline=self.pipeline, requested=fps)
        decode_latent_path = str(decode_latent or "").strip()
        if not decode_latent_path:
            self._ensure_base_placement()
        from mirai.core.inference.conditioning import (
            TEXT_TO_IMAGE,
            TEXT_TO_VIDEO,
            InferenceConditioningRequest,
            normalize_inference_task,
        )

        task_key = normalize_inference_task(
            task if task is not None else self.cfg.inference.task
        )
        effective_frames = 1 if task_key == TEXT_TO_IMAGE else int(frames)
        strength = float(
            denoising_strength
            if denoising_strength is not None
            else self.cfg.inference.denoising_strength
        )
        rewriter_name = str(
            prompt_rewriter
            if prompt_rewriter is not None
            else self.cfg.inference.prompt_rewriter
        ).strip().lower() or "none"
        provider = get_model_family_provider(str(self.cfg.model.type))
        if provider is None:
            raise ValueError(
                f"No model-family provider registered for '{self.cfg.model.type}'."
            )
        provider.validate_inference_prompt_rewriter(rewriter_name)
        if not provider.supports_inference_task(task_key):
            supported = ", ".join(provider.inference_tasks)
            raise ValueError(
                f"Model family '{provider.model_type}' does not support inference "
                f"task '{task_key}'; supported: {supported}."
            )
        cfg_mode_key = str(
            cfg_mode if cfg_mode is not None else self.cfg.inference.cfg_mode
        ).strip().lower() or "sequential"
        if cfg_mode_key == "batched" and not provider.supports_batched_cfg_inference():
            raise ValueError(
                f"Model family '{provider.model_type}' does not support batched CFG inference."
            )
        # Invariant 4: a family that ships a default negative prompt degrades
        # measurably under an empty one, so no path may reach the denoise loop
        # with that combination silently. This is a warning, not a failure: an
        # empty unconditional is a legitimate deliberate request (ablation,
        # cfg_scale=1). It is unreachable from a correct run, which omits the
        # negative prompt and receives the family default.
        empty_negative_warning = provider.generation_defaults().empty_negative_prompt_warning(
            str(negative_prompt), model_type=str(provider.model_type)
        )
        if empty_negative_warning is not None:
            print(f"[warning] {empty_negative_warning}", file=sys.stderr)
        rewritten = rewrite_inference_prompts(
            rewriter_name,
            PromptRewriteRequest(prompt=prompt, negative_prompt=str(negative_prompt)),
        )
        original_prompt = prompt
        original_negative_prompt = str(negative_prompt)
        prompt = rewritten.prompt
        negative_prompt = rewritten.negative_prompt
        use_native = self.use_native
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Refine pre-flight (fail fast BEFORE the expensive base denoise). Only
        # runs when --refine is set; the default path is untouched.
        if refine and not decode_latent_path:
            refine = self._validate_refine_request(
                refine,
                frames=effective_frames,
                height=int(height),
                width=int(width),
            )
            # Arming the stage can change the rate the family decodes at, so the
            # output rate is resolved again against the now-configured pipeline.
            output_fps = resolve_output_fps(pipeline=self.pipeline, requested=fps)

        # Provider hooks drive text encoding, denoising, VAE decode, and media output.
        from mirai.core.training.preview.preview import (
            run_cfg_denoise_loop,
            run_native_denoise_loop,
        )

        if self._expert_feature_cache is not None:
            # Branch features are only comparable within one denoising
            # trajectory, so nothing survives across generate() calls.
            self._expert_feature_cache.reset()

        ran_denoise_loop = True
        video_written = False
        media_written = False
        media_path: Path | None = None
        if decode_latent_path:
            pred = torch.load(decode_latent_path, map_location="cpu", weights_only=True)
            if not isinstance(pred, torch.Tensor):
                raise SystemExit(
                    f"--decode-latent expects a tensor .pt file, got {type(pred).__name__}"
                )
        else:
            _denoise_t0 = _sync_perf_counter() if timings is not None else 0.0
            if use_native:
                conditioning = InferenceConditioningRequest(
                    task=task_key,
                    input_image=input_image,
                    input_video=input_video,
                    denoising_strength=strength,
                    frame_count=effective_frames,
                    height=int(height),
                    width=int(width),
                ).validate()
                pred, _denoise_stats = run_native_denoise_loop(
                    pipeline=self.pipeline,
                    prompt=prompt,
                    negative_prompt=str(negative_prompt),
                    cfg_scale=float(cfg_scale),
                    seed=int(seed),
                    step=0,
                    denoise_steps=int(steps),
                    scheduler=str(scheduler),
                    frame_count=effective_frames,
                    height=int(height),
                    width=int(width),
                    solver_name=str(scheduler),
                    cfg_mode=cfg_mode_key,
                    forward_fn=self._forward_fn,
                    conditioning=conditioning,
                )
            else:
                if task_key != TEXT_TO_VIDEO:
                    raise ValueError(
                        "Validation-fixture inference supports text_to_video only."
                    )
                pred, _denoise_stats = run_cfg_denoise_loop(
                    pipeline=self.pipeline,
                    prompt=prompt,
                    negative_prompt=str(negative_prompt),
                    cfg_scale=float(cfg_scale),
                    seed=int(seed),
                    step=0,
                    denoise_steps=int(steps),
                    scheduler=str(scheduler),
                )
            if timings is not None:
                timings["denoise_s"] = _sync_perf_counter() - _denoise_t0
            torch.save(pred.detach().cpu(), out_path.with_suffix(".pt"))
        refined = False
        if refine and ran_denoise_loop and not decode_latent_path:
            device = self._compute_device
            dtype = self._compute_dtype if device.type == "cuda" else None
            _refine_t0 = _sync_perf_counter() if timings is not None else 0.0
            # The family refiner releases the base transformer before loading its
            # own weights. Mark placement dirty before entering so a failed refine
            # is also recoverable by the next generate() call.
            self._base_placement_dirty = True
            pred = self.pipeline.refine_inference_latent(
                base_latent=pred,
                request=refine,
                prompt=prompt,
                negative_prompt=str(negative_prompt),
                seed=int(seed),
                device=str(device),
                dtype=dtype,
            )
            if timings is not None:
                timings["refine_s"] = _sync_perf_counter() - _refine_t0
            torch.save(pred.detach().cpu(), out_path.with_suffix(".pt"))
            refined = True
        if ran_denoise_loop:
            # VAE decode and task-native media write.
            _decode_t0 = _sync_perf_counter() if timings is not None else 0.0
            try:
                media_path = decode_pipeline_media(
                    pipeline=self.pipeline,
                    pred=pred,
                    out_path=out_path,
                    task=task_key,
                    fps=output_fps,
                )
                media_written = True
                video_written = task_key != TEXT_TO_IMAGE
            except Exception as exc:
                if not allow_latent_output_only:
                    raise SystemExit(str(exc)) from exc
                print(f"[warning] {exc}", file=sys.stderr)
                media_written = False
                video_written = False
            if timings is not None:
                timings["decode_s"] = _sync_perf_counter() - _decode_t0

        payload = {
            "status": (
                "partial"
                if bool(use_native and not media_written and allow_latent_output_only)
                else "ok"
            ),
            "checkpoint": self.checkpoint,
            "adapter": self.adapter,
            "decode_latent": decode_latent_path,
            "output_path": str(out_path),
            "seed": int(seed),
            "lora_scale": self.lora_scale,
            "effective_lora_scale": self.effective_scale,
            "lora_format": self.lora_format,
            "merge": bool(self.merge),
            "prompt": prompt,
            "negative_prompt": str(negative_prompt),
            "fps": output_fps,
            "inference_mode": self.inference_mode,
            "native_inference": use_native,
            "video_written": video_written,
            "runtime_policy": runtime_policy_summary(self.cfg),
            "runtime_policy_notes": list(self.runtime_policy_notes),
        }
        if task_key != TEXT_TO_VIDEO:
            payload["task"] = task_key
            payload["media_written"] = media_written
            payload["media_path"] = str(media_path) if media_path is not None else ""
            if task_key == TEXT_TO_IMAGE:
                payload["image_written"] = media_written
            if task_key == "video_to_video":
                payload["denoising_strength"] = strength
        if refined:
            # Only present when --refine ran; default payload stays byte-identical.
            payload["refined"] = True
            # ``refine`` is the family-resolved request, so the payload names the
            # parameters the stage ran with and nothing the family does not own.
            payload["refiner"] = dict(refine)
        if rewriter_name != "none":
            payload["prompt_rewriter"] = rewriter_name
            payload["original_prompt"] = original_prompt
            payload["original_negative_prompt"] = original_negative_prompt
        if cfg_mode_key != "sequential":
            payload["cfg_mode"] = cfg_mode_key
        if self._expert_feature_cache is not None:
            payload["expert_feature_cache"] = {
                "mode": self._expert_feature_cache.policy.mode,
                "drift_threshold": self._expert_feature_cache.policy.drift_threshold,
                "max_reuse_span": self._expert_feature_cache.policy.max_reuse_span,
                "slots": self._expert_feature_cache.policy.slots,
                **self._expert_feature_cache.telemetry.snapshot(),
            }
        return payload


__all__ = [
    "InferenceSession",
    "decode_pipeline_media",
    "decode_pipeline_video",
    "resolve_inference_mode",
    "resolve_output_fps",
]
