"""Matched-seed audio-token ablation for native MAGI-2 Preview inference."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from mirai.core.inference.session import InferenceSession
from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline, resolve_magi2_audio_tokens


def run_audio_ab(session: InferenceSession, *, output: Path, seeds: list[int], **sampling) -> dict:
    pipeline = session.pipeline
    if not session.use_native or not isinstance(pipeline, Magi2PreviewPipeline):
        raise ValueError("Audio-token ablation requires a native MAGI-2 Preview session.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one seed and use each seed only once.")
    output.mkdir(parents=True, exist_ok=False)
    options = pipeline.options
    device = torch.device(session._compute_device)
    pairs = []
    try:
        for index, seed in enumerate(seeds):
            modes = (-1, 0) if index % 2 == 0 else (0, -1)
            cases = []
            latents = {}
            for mode in modes:
                pipeline.options = replace(options, audio_tokens=mode)
                destination = output / f"seed_{seed}_{'auto' if mode == -1 else 'zero'}.mp4"
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                timings = {}
                payload = session.generate(
                    seed=seed, out_path=destination, timings=timings, **sampling
                )
                peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                latent_path = destination.with_suffix(".pt")
                latent = torch.load(latent_path, map_location="cpu", weights_only=True).float()
                latents[mode] = latent
                cases.append({
                    "setting": mode,
                    "audio_tokens": resolve_magi2_audio_tokens(mode, latent.shape[-3]),
                    "timings": timings,
                    "generation_peak_vram_bytes": peak,
                    "video": destination.name,
                    "latent": latent_path.name,
                    "status": payload["status"],
                })
            if latents[-1].shape != latents[0].shape:
                raise ValueError("Matched audio-token runs produced different video latent shapes.")
            delta = latents[-1] - latents[0]
            pairs.append({
                "seed": seed, "cases": cases,
                "video_latent_rmse": delta.square().mean().sqrt().item(),
            })
    finally:
        pipeline.options = options
    report = {"sampling": sampling, "checkpoint": session.checkpoint, "pairs": pairs}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()
    session = InferenceSession.from_config(args.config)
    try:
        run_audio_ab(
            session, output=args.output, seeds=args.seed, prompt=args.prompt,
            steps=args.steps, frames=args.frames, width=args.width, height=args.height,
            scheduler="unipc", cfg_mode="batched", cfg_scale=5.0,
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
