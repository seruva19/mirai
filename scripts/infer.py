"""Deterministic thin-slice inference from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.inference.session import InferenceSession
from mirai.core.models.providers import resolve_family_generation_defaults

# Family-agnostic fallbacks. They apply only to a model family that declares no
# generation defaults of its own; a family that declares them wins whenever the
# corresponding flag was omitted.
GENERIC_STEPS = 20
GENERIC_CFG_SCALE = 5.0
GENERIC_SCHEDULER = "euler"

# These module-level seams let callers supply placement behavior to the
# load-once InferenceSession without coupling the session to this CLI.
from mirai.core.training.residency.device_placement import (  # noqa: F401
    place_pipeline_on_device,
    resolve_compute_device,
    resolve_compute_dtype,
)

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")

def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[arg-type]


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to TOML config")
    p.add_argument("--checkpoint", default="", help="Checkpoint path")
    p.add_argument("--adapter", default="", help="Adapter-only path")
    p.add_argument("--prompt", required=True, help="Prompt text")
    p.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt text. Unset applies the negative prompt the model "
        "family declares, if any; an explicit empty string is honored but "
        "warns, because families trained with a default negative prompt "
        "degrade under an empty one.",
    )
    p.add_argument(
        "--prompt-rewriter",
        default="",
        help="Prompt rewriter override (default: inference.prompt_rewriter from config).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video frame rate. Unset uses the rate the model family "
        "declares for its decoder, or 8.0 when it declares none.",
    )
    p.add_argument(
        "--task",
        choices=["t2v", "t2i", "ti2v", "i2v", "v2v", "text_to_video",
                 "text_to_image", "image_to_video", "video_to_video"],
        default="",
        help="Generation task override (default: inference.task).",
    )
    p.add_argument("--input-image", default="", help="Condition image for TI2V.")
    p.add_argument("--input-video", default="", help="Source video for V2V.")
    p.add_argument(
        "--denoising-strength",
        type=float,
        default=None,
        help="V2V schedule fraction in [0,1] (default: config).",
    )
    p.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    p.add_argument("--lora-scale", type=float, default=1.0, help="Inference-time LoRA scale")
    p.add_argument(
        "--lora-format",
        choices=["auto", "kohya", "diffusers", "peft", "lycoris"],
        default="auto",
        help="Adapter key format",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Merge adapter into model for inference.",
    )
    p.add_argument("--out", default="./outputs/infer_output.pt", help="Output path (.pt or .mp4)")
    p.add_argument("--width", type=int, default=832, help="Output width")
    p.add_argument("--height", type=int, default=480, help="Output height")
    p.add_argument(
        "--frames",
        type=int,
        default=17,
        help="Number of frames; the model family's latent layout states the "
        "rule the count must satisfy.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"Number of denoise steps (unset: the model family's declared "
        f"value, or {GENERIC_STEPS} when it declares none).",
    )
    p.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help=f"Classifier-free guidance scale (unset: the model family's "
        f"declared value, or {GENERIC_CFG_SCALE} when it declares none).",
    )
    p.add_argument(
        "--cfg-mode",
        choices=["sequential", "batched"],
        default="",
        help="Single-device CFG execution override (default: inference.cfg_mode).",
    )
    p.add_argument(
        "--keep-text-encoder-resident",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep the text encoder resident between session generations.",
    )
    p.add_argument(
        "--keep-vae-resident",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep the VAE resident between session generations.",
    )
    p.add_argument(
        "--scheduler",
        default="",
        help=f"Denoise scheduler (unset: the model family's declared solver, or "
        f"'{GENERIC_SCHEDULER}' when it declares none).",
    )
    p.add_argument(
        "--compile",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="",
        help=(
            "Opt-in torch.compile inference. Unset (default) "
            "runs eager without compiled regions. 'reduce-overhead' applies "
            "per-region CUDA graphs to the launch-bound denoise loop; any compile "
            "failure or graph break degrades to eager with a warning."
        ),
    )
    p.add_argument(
        "--allow-latent-output-only",
        action="store_true",
        help="Allow native inference to succeed without decoded video output.",
    )
    # The refiner is disabled by default. When enabled, the model family hands
    # its final latent to its own second-stage refiner before the VAE decode.
    # Every parameter below defaults to unset, which selects the family's
    # released refinement profile; the run reports the values it resolved.
    p.add_argument(
        "--refine",
        action="store_true",
        help="Run the model family's refiner stage after the base denoise. "
        "Video-only; requires the family's refiner weights in the model root.",
    )
    p.add_argument(
        "--refiner-steps",
        type=int,
        default=None,
        help="Refiner denoise steps (unset: the family's released value).",
    )
    p.add_argument(
        "--refiner-cfg",
        type=float,
        default=None,
        help="Refiner CFG scale (unset: the family's released value).",
    )
    p.add_argument(
        "--refiner-shift",
        type=float,
        default=None,
        help="Refiner flow shift (unset: the family's released value).",
    )
    p.add_argument(
        "--refiner-t-thresh",
        type=float,
        default=None,
        help="Re-noise sigma / tail start in (0,1], for families whose refiner "
        "re-enters the flow at a threshold.",
    )
    p.add_argument(
        "--refiner-height",
        type=int,
        default=None,
        help="Refiner output height (unset: the family's released target).",
    )
    p.add_argument(
        "--refiner-width",
        type=int,
        default=None,
        help="Refiner output width (unset: the family's released target).",
    )
    p.add_argument(
        "--refiner-sigma-tail-steps",
        type=int,
        default=None,
        help="Extra low-noise sigmas appended to the tail, for families whose "
        "refiner extends its sigma grid. Out-of-range values are rejected "
        "before the base denoise runs.",
    )
    p.add_argument(
        "--refiner-scheduler",
        default="",
        help="Refiner solver (registry name); empty falls back to --scheduler.",
    )
    p.add_argument(
        "--decode-latent",
        default="",
        help=(
            "Decode an existing latent .pt (the latent dump of a previous run) "
            "to video, skipping the denoise loop."
        ),
    )
    p.add_argument(
        "--timings-out",
        default="",
        help=(
            "If set, write a per-phase timings JSON "
            '{"load_s","denoise_s","decode_s","peak_vram_mb"} to this path. '
            "Consumed by inference/bench.py."
        ),
    )
    p.add_argument(
        "--routing-trace-out",
        default="",
        help=(
            "Write configured inference routing telemetry as "
            "stacked per-forward router top-k assignments + metadata to this "
            "path (.pt or .npz). Requires "
            "model.params.inference_routing_telemetry=true. Analyze with "
            "inference/analyze_routing.py."
        ),
    )
    return p.parse_args()


def _sync_perf_counter() -> float:
    """Read ``perf_counter`` after synchronizing pending CUDA work.

    On CPU-only hosts the sync is a no-op and this is a plain wall clock read.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _write_routing_trace(pipeline, path: Path, *, metadata: dict) -> str:
    """Stack the captured routing entries and write them with metadata.

    Entries within a fixed-shape denoise run share [N, K]; they stack into a
    single [E, N, K] int16 tensor with parallel forward_idx / layer_idx /
    timestep / num_experts vectors. Falls back to an object array when shapes
    differ (e.g. a layer filter that changed token count mid-run -- unexpected).
    """
    trace = pipeline.get_inference_routing_trace()
    forward_idx = np.asarray([e["forward_idx"] for e in trace], dtype=np.int32)
    layer_idx = np.asarray([e["layer_idx"] for e in trace], dtype=np.int32)
    timestep = np.asarray([e["timestep"] for e in trace], dtype=np.float32)
    num_experts = np.asarray(
        [int(e.get("num_experts") or 0) for e in trace], dtype=np.int32
    )
    stacks = [e["top_indices"] for e in trace]
    shapes = {tuple(t.shape) for t in stacks}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        top = (
            np.stack([t.cpu().numpy() for t in stacks]).astype(np.int16)
            if len(shapes) == 1 and stacks
            else np.asarray([t.cpu().numpy() for t in stacks], dtype=object)
        )
        np.savez_compressed(
            path,
            top_indices=top,
            forward_idx=forward_idx,
            layer_idx=layer_idx,
            timestep=timestep,
            num_experts=num_experts,
            metadata=json.dumps(metadata),
        )
    else:
        top = (
            torch.stack([t.to(torch.int16) for t in stacks])
            if len(shapes) == 1 and stacks
            else [t.to(torch.int16) for t in stacks]
        )
        torch.save(
            {
                "top_indices": top,
                "forward_idx": torch.from_numpy(forward_idx),
                "layer_idx": torch.from_numpy(layer_idx),
                "timestep": torch.from_numpy(timestep),
                "num_experts": torch.from_numpy(num_experts),
                "metadata": metadata,
            },
            path,
        )
    return f"{len(trace)} entries, {sorted(shapes)} shapes -> {path}"


def main() -> int:
    args = parse_args()
    adapter_path = str(args.adapter).strip()
    config_path = str(args.config).strip()
    decode_latent_path = str(args.decode_latent).strip()
    timings_out_path = str(args.timings_out).strip()
    collect_timings = bool(timings_out_path)
    # Per-phase wall times (seconds). Left at 0.0 for phases a given run skips
    # (e.g. decode_latent skips the denoise phase).
    load_s = 0.0
    denoise_s = 0.0
    decode_s = 0.0
    refine_s = 0.0
    peak_vram_mb: float | None = None
    refine_request = None
    if bool(args.refine):
        # ``None`` is preserved rather than filled in: the model family owns the
        # default for every value the caller did not state.
        refine_request = {
            "height": _optional_int(args.refiner_height),
            "width": _optional_int(args.refiner_width),
            "steps": _optional_int(args.refiner_steps),
            "cfg_scale": _optional_float(args.refiner_cfg),
            "shift": _optional_float(args.refiner_shift),
            "t_thresh": _optional_float(args.refiner_t_thresh),
            "sigma_tail_steps": _optional_int(args.refiner_sigma_tail_steps),
            # Filled in once the base scheduler has been resolved against the
            # model family's declared solver.
            "scheduler": "",
        }
    if collect_timings and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    routing_trace_out = str(args.routing_trace_out).strip()
    routing_trace_stride = 1
    session: InferenceSession | None = None
    try:
        # InferenceSession.from_config owns registration, configuration,
        # checkpoint and adapter loading, evaluation mode, and placement. The
        # CLI supplies its placement seams; decode-latent delegates placement to
        # the VAE.
        _load_t0 = _sync_perf_counter() if collect_timings else 0.0
        session = InferenceSession.from_config(
            config_path,
            checkpoint=str(args.checkpoint),
            adapter=adapter_path,
            lora_format=str(args.lora_format),
            lora_scale=float(args.lora_scale),
            merge=bool(args.merge),
            place_on_device=not decode_latent_path,
            compile_mode=str(args.compile),
            keep_text_encoder_resident=args.keep_text_encoder_resident,
            keep_vae_resident=args.keep_vae_resident,
            place_fn=place_pipeline_on_device,
            device_fn=resolve_compute_device,
            dtype_fn=resolve_compute_dtype,
        )
        # The model family declares what it was released with; the generic CLI
        # never names a family. A flag left unset takes the family value, an
        # explicitly passed flag always wins -- including an empty negative
        # prompt, which the session then reports as a degraded run.
        family_defaults = resolve_family_generation_defaults(
            str(session.cfg.model.type)
        )
        negative_prompt = family_defaults.resolve_negative_prompt(
            args.negative_prompt
        )
        steps = family_defaults.resolve_steps(args.steps, fallback=GENERIC_STEPS)
        cfg_scale = family_defaults.resolve_cfg_scale(
            args.cfg_scale, fallback=GENERIC_CFG_SCALE
        )
        scheduler = family_defaults.resolve_scheduler(
            args.scheduler, fallback=GENERIC_SCHEDULER
        )
        if refine_request is not None:
            refine_request["scheduler"] = str(args.refiner_scheduler or scheduler)
        routing_params = session.cfg.model.params
        routing_trace_stride = int(
            routing_params.inference_routing_telemetry_layer_stride
        )
        if routing_trace_out and not bool(routing_params.inference_routing_telemetry):
            raise SystemExit(
                "--routing-trace-out requires "
                "model.params.inference_routing_telemetry=true."
            )
        if collect_timings:
            load_s = _sync_perf_counter() - _load_t0
        if routing_trace_out and hasattr(
            session.pipeline, "reset_inference_routing_trace"
        ):
            session.pipeline.reset_inference_routing_trace()
        timings_box: dict | None = {} if collect_timings else None
        payload = session.generate(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            seed=int(args.seed),
            steps=steps,
            cfg_scale=cfg_scale,
            frames=int(args.frames),
            height=int(args.height),
            width=int(args.width),
            out_path=args.out,
            fps=None if args.fps is None else float(args.fps),
            scheduler=scheduler,
            decode_latent=decode_latent_path,
            allow_latent_output_only=bool(args.allow_latent_output_only),
            timings=timings_box,
            refine=refine_request,
            prompt_rewriter=str(args.prompt_rewriter) or None,
            cfg_mode=str(args.cfg_mode) or None,
            task=str(args.task) or None,
            input_image=str(args.input_image) or None,
            input_video=str(args.input_video) or None,
            denoising_strength=args.denoising_strength,
        )
        if collect_timings and timings_box is not None:
            denoise_s = float(timings_box.get("denoise_s", 0.0))
            decode_s = float(timings_box.get("decode_s", 0.0))
            refine_s = float(timings_box.get("refine_s", 0.0))
        routing_trace_summary = ""
        if routing_trace_out and hasattr(
            session.pipeline, "get_inference_routing_trace"
        ):
            routing_trace_summary = _write_routing_trace(
                session.pipeline,
                Path(routing_trace_out),
                metadata={
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "scheduler": scheduler,
                    "layer_stride": routing_trace_stride,
                    "seed": int(args.seed),
                    "prompt": str(payload["prompt"]),
                    "negative_prompt": str(payload["negative_prompt"]),
                    "frames": int(args.frames),
                    "height": int(args.height),
                    "width": int(args.width),
                },
            )
            print(f"[routing-trace] {routing_trace_summary}", file=sys.stderr)
        if routing_trace_out:
            payload["routing_trace_out"] = routing_trace_out
            payload["routing_trace"] = routing_trace_summary
        if collect_timings:
            if torch.cuda.is_available():
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            Path(timings_out_path).write_text(
                json.dumps(
                    {
                        "load_s": load_s,
                        "denoise_s": denoise_s,
                        "decode_s": decode_s,
                        "refine_s": refine_s,
                        "peak_vram_mb": peak_vram_mb,
                    }
                ),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())
