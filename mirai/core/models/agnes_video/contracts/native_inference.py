"""Run prompt-to-joint-media inference for the full released Agnes runtime."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
from typing import Any

import torch

from mirai.core.builtins import register_builtin_components
from mirai.core.inference.session import InferenceSession, deserialize_inference_latent
from mirai.core.models.agnes_video.contracts._runtime import (
    ContractUnavailable,
    checkpoint_digest,
    require_cuda,
    runtime_identity,
    unavailable_reason,
    write_report,
)
from mirai.core.models.agnes_video.contracts.report import (
    SCHEMA_VERSION,
    validate_inference_report,
)
from mirai.core.models.native_video import JointMediaLatents
from mirai.core.training.runtime.cli import load_runtime_config


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _install_input_probe(pipeline: Any, *, seed: int):
    evidence: dict[str, Any] = {"seed": int(seed)}
    had_instance_override = "sample_native_preview" in pipeline.__dict__
    previous_instance_override = pipeline.__dict__.get("sample_native_preview")
    original = pipeline.sample_native_preview

    def capture_audio(_module, _args, kwargs):
        audio = kwargs.get("audio_latents")
        if "initial_audio_sha256" not in evidence and isinstance(audio, torch.Tensor):
            evidence["initial_audio_sha256"] = _tensor_digest(audio)

    handle = pipeline.register_forward_pre_hook(capture_audio, with_kwargs=True)

    def capture_inputs(*args, **kwargs):
        noise = kwargs.get("noise")
        context = kwargs.get("context")
        context_null = kwargs.get("context_null")
        if not all(isinstance(value, torch.Tensor) for value in (noise, context, context_null)):
            raise RuntimeError("Agnes sampling hook did not receive tensor inputs.")
        evidence["initial_video_sha256"] = _tensor_digest(noise)
        evidence["conditioned_context_sha256"] = _tensor_digest(context)
        evidence["unconditional_context_sha256"] = _tensor_digest(context_null)
        from mirai.core.models.agnes_video.sampler import build_agnes_solver

        shift = pipeline.resolve_flow_shift_for_latent_shape(
            tuple(noise.unsqueeze(0).shape if noise.ndim == 4 else noise.shape)
        )
        solver = build_agnes_solver(
            num_inference_steps=max(1, int(kwargs["denoise_steps"])),
            shift=float(shift),
            device=str(noise.device),
            stochastic=float(kwargs.get("guidance_scale", 1.0)) == 1.0,
        )
        evidence["sigma_schedule_sha256"] = _tensor_digest(solver.timesteps)
        return original(*args, **kwargs)

    pipeline.sample_native_preview = capture_inputs
    return evidence, handle, (had_instance_override, previous_instance_override)


def _remove_input_probe(pipeline: Any, handle: Any, restore_state: tuple[bool, Any]) -> None:
    handle.remove()
    had_instance_override, previous_instance_override = restore_state
    if had_instance_override:
        pipeline.sample_native_preview = previous_instance_override
    else:
        del pipeline.sample_native_preview


def _latent(path: Path) -> JointMediaLatents:
    value = deserialize_inference_latent(torch.load(path, map_location="cpu", weights_only=True))
    if not isinstance(value, JointMediaLatents):
        raise RuntimeError("Agnes inference did not emit joint video/audio latents.")
    if value.audio.numel() == 0:
        raise RuntimeError("Agnes inference dropped the audio latent stream.")
    return value


def _probe_media(path: Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise ContractUnavailable("ffprobe is required to verify the released audio/video artifact.")
    completed = subprocess.run(
        [executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not isinstance(video, dict) or not isinstance(audio, dict):
        raise RuntimeError("The Agnes media artifact must contain video and audio streams.")

    def duration(stream: dict[str, Any]) -> float:
        raw = stream.get("duration", payload.get("format", {}).get("duration", 0))
        return float(raw or 0)

    frame_count = int(video.get("nb_frames", 0) or 0)
    if frame_count <= 0:
        rate = str(video.get("avg_frame_rate", "0/1"))
        numerator, denominator = (int(value) for value in rate.split("/", 1))
        frame_count = round(duration(video) * numerator / max(1, denominator))
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    return {
        "path": str(path),
        "width": int(video.get("width", 0) or 0),
        "height": int(video.get("height", 0) or 0),
        "frames": frame_count,
        "sample_rate": sample_rate,
        "audio_samples": round(duration(audio) * sample_rate),
        "video_duration_seconds": duration(video),
        "audio_duration_seconds": duration(audio),
        "fps": (
            float(str(video.get("avg_frame_rate", "0/1")).split("/", 1)[0])
            / max(1.0, float(str(video.get("avg_frame_rate", "0/1")).split("/", 1)[1]))
        ),
    }


def _generate(*, config_path: str, adapter: str, scale: float, out_path: Path,
              prompt: str, seed: int, steps: int, frames: int, height: int, width: int):
    session = InferenceSession.from_config(config_path, adapter=adapter, lora_scale=scale)
    evidence, probe_handle, restore_state = _install_input_probe(session.pipeline, seed=seed)
    try:
        payload = session.generate(prompt=prompt, out_path=out_path, seed=seed, steps=steps,
                                   cfg_scale=5.0, frames=frames, height=height, width=width,
                                   scheduler="unipc", allow_latent_output_only=False)
        latent = _latent(out_path.with_suffix(".pt"))
        runtime = runtime_identity(session.pipeline, require_cuda())
        text_encoder = type(session.pipeline._native_assets().text_encoder).__name__
        required = {
            "initial_video_sha256", "initial_audio_sha256",
            "conditioned_context_sha256", "unconditional_context_sha256",
            "sigma_schedule_sha256",
        }
        missing = required.difference(evidence)
        if missing:
            raise RuntimeError(f"Agnes paired-input probe missed: {sorted(missing)}")
        return payload, latent, runtime, text_encoder, evidence
    finally:
        _remove_input_probe(session.pipeline, probe_handle, restore_state)
        session.close()
        del probe_handle, restore_state, session
        gc.collect()
        torch.cuda.empty_cache()


def run(config_path: str, output_path: str, *, prompt: str, seed: int, steps: int,
        frames: int, height: int, width: int) -> dict[str, Any]:
    require_cuda()
    register_builtin_components()
    config, _ = load_runtime_config(config_path, entrypoint="infer")
    if str(config.model.type) != "agnes-video":
        raise ContractUnavailable("The Agnes inference contract requires model.type='agnes-video'.")
    adapter = str(config.adapter.init_from or "").strip()
    if not adapter or not Path(adapter).is_file():
        raise ContractUnavailable("A configured Agnes adapter.init_from artifact is required for the scale proof.")
    source_sha = checkpoint_digest(config.model.path)
    evidence = Path(output_path).with_suffix("")
    baseline_path = evidence.with_name(evidence.name + "-scale0")
    candidate_path = evidence.with_name(evidence.name + "-scale1")
    baseline_payload, baseline, _, _, baseline_inputs = _generate(
        config_path=config_path, adapter=adapter, scale=0.0, out_path=baseline_path,
        prompt=prompt, seed=seed, steps=steps, frames=frames, height=height, width=width,
    )
    candidate_payload, candidate, runtime, text_encoder, candidate_inputs = _generate(
        config_path=config_path, adapter=adapter, scale=1.0, out_path=candidate_path,
        prompt=prompt, seed=seed, steps=steps, frames=frames, height=height, width=width,
    )
    video_difference = float((candidate.video.float() - baseline.video.float()).abs().max())
    audio_difference = float((candidate.audio.float() - baseline.audio.float()).abs().max())
    maximum_difference = max(video_difference, audio_difference)
    paired_fields = (
        "initial_video_sha256", "initial_audio_sha256",
        "conditioned_context_sha256", "unconditional_context_sha256",
        "sigma_schedule_sha256",
    )
    paired_inputs_match = all(
        baseline_inputs[field] == candidate_inputs[field] for field in paired_fields
    )
    media_path = candidate_path.with_suffix(".mp4")
    media = _probe_media(media_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": "agnes_native_inference",
        "status": "passed",
        "runtime": runtime,
        "snapshot": {"source": str(config.model.path), "sha256": source_sha},
        "seed": seed,
        "proof": {
            "native_inference": candidate_payload.get("native_inference") is True,
            "text_encoder": text_encoder,
            "joint_denoising": isinstance(candidate, JointMediaLatents),
            "video_decoded": bool(candidate_payload.get("video_written")),
            "audio_decoded": candidate.audio.numel() > 0,
            "audio_muxed": media["audio_samples"] > 0,
            "adapter_scale_effect": (
                maximum_difference > 0
                and paired_inputs_match
                and baseline_payload.get("seed") == candidate_payload.get("seed") == seed
            ),
        },
        "adapter_scale": {
            "baseline": 0.0,
            "candidate": 1.0,
            "max_abs_latent_difference": maximum_difference,
            "video_max_abs_latent_difference": video_difference,
            "audio_max_abs_latent_difference": audio_difference,
        },
        "paired_inputs": {
            "method": "separate_sessions_same_seed_with_input_digests",
            "sigma_schedule_basis": "family_sampler_factory_from_captured_hook_arguments",
            "determinism_scope": (
                "paired input equivalence; final latent repeatability across devices "
                "or kernel implementations is not claimed"
            ),
            "baseline": baseline_inputs,
            "candidate": candidate_inputs,
        },
        "media": media,
    }
    failures = validate_inference_report(report)
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("MIRAI_AGNES_INFERENCE_CONFIG"))
    parser.add_argument("--output", default=os.environ.get("MIRAI_AGNES_INFERENCE_REPORT"))
    parser.add_argument("--prompt", default="A quiet ocean sunrise with gentle waves.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    args = parser.parse_args(argv)
    if not args.config:
        report = {"schema_version": SCHEMA_VERSION, "contract": "agnes_native_inference",
                  "status": "incomplete",
                  "failures": ["Set MIRAI_AGNES_INFERENCE_CONFIG or pass --config."]}
    elif not args.output:
        report = {"schema_version": SCHEMA_VERSION, "contract": "agnes_native_inference",
                  "status": "incomplete",
                  "failures": ["Set MIRAI_AGNES_INFERENCE_REPORT or pass --output."]}
    else:
        try:
            report = run(args.config, args.output, prompt=args.prompt, seed=args.seed,
                         steps=args.steps, frames=args.frames, height=args.height, width=args.width)
        except ContractUnavailable as exc:
            report = {"schema_version": SCHEMA_VERSION, "contract": "agnes_native_inference",
                      "status": "incomplete", "failures": [str(exc)]}
        except BaseException as exc:
            unavailable = unavailable_reason(exc)
            report = {"schema_version": SCHEMA_VERSION, "contract": "agnes_native_inference",
                      "status": "incomplete" if unavailable else "failed",
                      "failures": [unavailable or f"{type(exc).__name__}: {exc}"]}
            if unavailable is None:
                report["traceback"] = traceback.format_exc()
    if args.output:
        write_report(args.output, report)
    else:
        print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else (2 if report["status"] == "incomplete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
