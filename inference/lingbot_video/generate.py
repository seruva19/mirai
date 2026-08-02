"""LingBot-Video family inference CLI.

Wraps the model-agnostic ``scripts/infer.py`` with the LingBot prompt contract
and family defaults. Model access uses the same provider hooks as training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.models.lingbot_video.prompting import (  # noqa: E402
    is_valid_json_caption,
    wrap_caption_as_lingbot_json,
)

DEFAULT_NEGATIVE_PATH = Path(__file__).resolve().parent / "default_negative_prompt.json"


class LingBotInferenceProfile(NamedTuple):
    scheduler: str = "unipc"
    steps: int = 25
    cfg_scale: float = 3.0
    frames: int = 33
    height: int = 480
    width: int = 832
    fps: int = 24


DEFAULT_INFERENCE_PROFILE = LingBotInferenceProfile()
DISTILLED_4STEP_PROFILE = LingBotInferenceProfile(
    scheduler="euler",
    steps=4,
    cfg_scale=1.0,
)
INFERENCE_PROFILES = {
    "standard": DEFAULT_INFERENCE_PROFILE,
    "distilled-4step": DISTILLED_4STEP_PROFILE,
}


CONFIG_TEMPLATE = """\
preset = "lingbot_video"

[model]
type = "lingbot-video"
path = "{model_root}"

[model.params]
variant = "lingbot-video-moe-30b-a3b"
strict_native_assets = true

[memory]
frozen_weight_quantization = "{quant}"
frozen_weight_quantization_strategy = "{quant_strategy}"
"""

ADAPTER_TEMPLATE = """
[adapter]
type = "lora"
target_preset = "{preset}"
rank = {rank}
alpha = {alpha}
"""


def resolve_prompt_text(prompt: str, *, raw: bool) -> tuple[str, bool]:
    """Resolve ``@file`` references and wrap plain NL prompts into the minimal
    JSON caption the DiT was trained on. Returns (text, was_wrapped).

    The wrapping logic is owned by ``mirai.core.models.lingbot_video.prompting`` and
    shared verbatim with the training cache-encode path so the two never drift.
    """
    text = prompt.strip()
    if text.startswith("@"):
        text = Path(text[1:]).read_text(encoding="utf-8").strip()
    if raw:
        return text, False
    if is_valid_json_caption(text):
        return text, False  # already structured
    return wrap_caption_as_lingbot_json(text), True


def default_negative_prompt() -> str:
    return DEFAULT_NEGATIVE_PATH.read_text(encoding="utf-8").strip()


def build_infer_argv(args: argparse.Namespace, *, config_path: str) -> list[str]:
    prompt_text, wrapped = resolve_prompt_text(args.prompt, raw=bool(args.raw))
    if wrapped:
        print(
            "[lingbot] plain-text prompt auto-wrapped into a minimal JSON caption; "
            "pass a full Rewriter-schema JSON (or @file) to preserve every structured field, "
            "--raw to bypass.",
            file=sys.stderr,
        )
    negative = args.negative_prompt
    if negative is None:
        negative = default_negative_prompt()
    argv = [
        "infer.py",
        "--config",
        config_path,
        "--prompt",
        prompt_text,
        "--negative-prompt",
        negative,
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--cfg-scale",
        str(args.cfg_scale),
        "--frames",
        str(args.frames),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--fps",
        str(args.fps),
        "--scheduler",
        str(args.scheduler),
        "--out",
        args.out,
    ]
    if str(args.task) != "t2v":
        argv += ["--task", str(args.task)]
    if str(args.input_image).strip():
        argv += ["--input-image", str(args.input_image)]
    if str(args.input_video).strip():
        argv += ["--input-video", str(args.input_video)]
    if args.denoising_strength is not None:
        argv += ["--denoising-strength", str(args.denoising_strength)]
    if args.adapter:
        argv += ["--adapter", args.adapter, "--lora-scale", str(args.lora_scale)]
    if args.decode_latent:
        argv += ["--decode-latent", args.decode_latent]
    if getattr(args, "compile", ""):
        argv += ["--compile", args.compile]
    if getattr(args, "refine", False):
        argv += [
            "--refine",
            "--refiner-steps", str(args.refiner_steps),
            "--refiner-cfg", str(args.refiner_cfg),
            "--refiner-shift", str(args.refiner_shift),
            "--refiner-t-thresh", str(args.refiner_t_thresh),
            "--refiner-height", str(args.refiner_height),
            "--refiner-width", str(args.refiner_width),
            "--refiner-sigma-tail-steps", str(args.refiner_sigma_tail_steps),
            "--refiner-scheduler", str(args.refiner_scheduler),
        ]
    return argv


def _refine_dict(args: argparse.Namespace) -> dict | None:
    """Build the InferenceSession refine kwargs (batch path); None when off."""
    if not getattr(args, "refine", False):
        return None
    return {
        "height": int(args.refiner_height),
        "width": int(args.refiner_width),
        "steps": int(args.refiner_steps),
        "cfg_scale": float(args.refiner_cfg),
        "shift": float(args.refiner_shift),
        "t_thresh": float(args.refiner_t_thresh),
        "sigma_tail_steps": int(args.refiner_sigma_tail_steps),
        "scheduler": str(args.refiner_scheduler or args.scheduler),
    }


def _generate_kwargs(
    args: argparse.Namespace,
    *,
    prompt: str,
    out: str,
    negative: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    frames: int,
    height: int,
    width: int,
    fps: int,
    decode_latent: str,
    refine: dict | None = None,
    task: str | None = None,
    input_image: str | None = None,
    input_video: str | None = None,
    denoising_strength: float | None = None,
) -> dict:
    """Map resolved family args to InferenceSession.generate() kwargs."""
    kwargs = {
        "prompt": prompt,
        "out_path": out,
        "negative_prompt": negative,
        "seed": int(seed),
        "steps": int(steps),
        "cfg_scale": float(cfg_scale),
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "fps": int(fps),
        "scheduler": str(
            getattr(args, "scheduler", DEFAULT_INFERENCE_PROFILE.scheduler)
        ),
        "decode_latent": str(decode_latent or ""),
        "allow_latent_output_only": False,
        "task": str(task if task is not None else getattr(args, "task", "t2v")),
        "input_image": str(
            input_image
            if input_image is not None
            else getattr(args, "input_image", "")
        ).strip()
        or None,
        "input_video": str(
            input_video
            if input_video is not None
            else getattr(args, "input_video", "")
        ).strip()
        or None,
        "denoising_strength": (
            denoising_strength
            if denoising_strength is not None
            else getattr(args, "denoising_strength", None)
        ),
    }
    if refine is not None:
        kwargs["refine"] = refine
    return kwargs


def _resolve_prompt_for_run(args: argparse.Namespace, prompt: str) -> str:
    prompt_text, wrapped = resolve_prompt_text(prompt, raw=bool(args.raw))
    if wrapped:
        print(
            "[lingbot] plain-text prompt auto-wrapped into a minimal JSON caption; "
            "pass a full Rewriter-schema JSON (or @file) to preserve every structured field, "
            "--raw to bypass.",
            file=sys.stderr,
        )
    return prompt_text


def _load_batch_entries(path: str) -> list[dict]:
    """Parse the --batch-prompts JSONL file (one object per non-empty line).

    Each object must carry ``prompt`` and ``out``; the rest of the generation
    knobs (seed/steps/cfg_scale/frames/height/width/fps/negative_prompt/
    decode_latent) are optional per-line overrides that fall back to the CLI
    defaults.
    """
    entries: list[dict] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"--batch-prompts line {lineno}: invalid JSON ({exc})")
        if not isinstance(obj, dict):
            raise SystemExit(f"--batch-prompts line {lineno}: expected a JSON object")
        if "prompt" not in obj or "out" not in obj:
            raise SystemExit(
                f"--batch-prompts line {lineno}: each entry needs 'prompt' and 'out'"
            )
        entries.append(obj)
    if not entries:
        raise SystemExit(f"--batch-prompts file {path} has no prompt entries")
    return entries


def run_batch(args: argparse.Namespace, *, config_path: str) -> int:
    """Build ONE InferenceSession and loop generate() over the batch file.

    Amortizes the ~15 s model build/placement across every prompt (the whole
    point of A4) instead of paying it per prompt as the single-shot CLI does.
    """
    from mirai.core.inference.session import InferenceSession

    entries = _load_batch_entries(args.batch_prompts)
    negative = args.negative_prompt
    if negative is None:
        negative = default_negative_prompt()

    session = InferenceSession.from_config(
        config_path,
        checkpoint="",
        adapter=str(args.adapter),
        lora_format="auto",
        lora_scale=float(args.lora_scale),
        merge=False,
        place_on_device=True,
        keep_text_encoder_resident=bool(args.keep_text_encoder_resident),
        keep_vae_resident=bool(args.keep_vae_resident),
        compile_mode=str(getattr(args, "compile", "")),
    )
    results: list[dict] = []
    try:
        for entry in entries:
            prompt_text = _resolve_prompt_for_run(args, str(entry["prompt"]))
            payload = session.generate(
                **_generate_kwargs(
                    args,
                    prompt=prompt_text,
                    out=str(entry["out"]),
                    negative=str(entry.get("negative_prompt", negative)),
                    seed=entry.get("seed", args.seed),
                    steps=entry.get("steps", args.steps),
                    cfg_scale=entry.get("cfg_scale", args.cfg_scale),
                    frames=entry.get("frames", args.frames),
                    height=entry.get("height", args.height),
                    width=entry.get("width", args.width),
                    fps=entry.get("fps", args.fps),
                    decode_latent=entry.get("decode_latent", ""),
                    refine=_refine_dict(args),
                    task=entry.get("task"),
                    input_image=entry.get("input_image"),
                    input_video=entry.get("input_video"),
                    denoising_strength=entry.get("denoising_strength"),
                )
            )
            results.append(payload)
    finally:
        session.close()
    print(json.dumps({"status": "ok", "count": len(results), "results": results}, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", help="JSON caption, @file, or plain text")
    p.add_argument(
        "--batch-prompts",
        default="",
        help=(
            "JSONL file of {'prompt':..., 'out':..., <optional overrides>} entries. "
            "Builds ONE session and loops generate() over every line, amortizing "
            "the model build across prompts. Mutually exclusive with --prompt."
        ),
    )
    p.add_argument(
        "--keep-text-encoder-resident",
        action="store_true",
        help=(
            "Keep the text encoder resident on the compute device across batch "
            "prompts. When disabled, the encoder returns to CPU after each prompt."
        ),
    )
    p.add_argument(
        "--keep-vae-resident",
        action="store_true",
        help=(
            "Keep the VAE resident on the compute device across batch prompts. "
            "When disabled, the VAE returns to CPU after each prompt."
        ),
    )
    p.add_argument(
        "--negative-prompt",
        default=None,
        help="Override the vendored upstream JSON negative (default: vendored)",
    )
    p.add_argument("--raw", action="store_true", help="Do not auto-wrap NL prompts")
    p.add_argument(
        "--compile",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="",
        help=(
            "Opt-in torch.compile inference. Unset (default) "
            "runs eager. 'reduce-overhead' applies per-region CUDA graphs to the "
            "launch-bound denoise loop; failures degrade to eager with a warning."
        ),
    )
    p.add_argument("--config", default="", help="Explicit trainer TOML (else generated)")
    p.add_argument(
        "--model-root",
        default="./models/lingbot-video-moe-30b-a3b",
        help="Model root for the generated config",
    )
    p.add_argument(
        "--quant",
        choices=["none", "nf4"],
        default="none",
        help="Frozen-weight quantization: none = resident BF16; nf4 = compressed frozen weights",
    )
    p.add_argument("--adapter", default="", help="LoRA adapter path")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument(
        "--inference-profile",
        choices=sorted(INFERENCE_PROFILES),
        default="standard",
        help="Family-owned denoise and adapter defaults.",
    )
    p.add_argument("--adapter-rank", type=int)
    p.add_argument("--adapter-alpha", type=float)
    p.add_argument("--adapter-preset")
    p.add_argument("--decode-latent", default="", help="Decode an existing latent .pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--task",
        choices=["t2v", "t2i", "ti2v", "v2v"],
        default="t2v",
        help="Text-to-video, text-to-image, text+image-to-video, or video-to-video.",
    )
    p.add_argument("--input-image", default="", help="Condition image for --task ti2v.")
    p.add_argument("--input-video", default="", help="Source video for --task v2v.")
    p.add_argument(
        "--denoising-strength",
        type=float,
        default=None,
        help="V2V schedule fraction in [0,1].",
    )
    p.add_argument("--steps", type=int)
    p.add_argument(
        "--scheduler",
        default=None,
        help="Denoise solver (registry name). The default profile uses UniPC "
        "with 25 steps; use --scheduler euler --steps 40 for the reference "
        "trajectory.",
    )
    p.add_argument("--cfg-scale", type=float)
    p.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_INFERENCE_PROFILE.frames,
        help="4n+1",
    )
    p.add_argument("--height", type=int, default=DEFAULT_INFERENCE_PROFILE.height)
    p.add_argument("--width", type=int, default=DEFAULT_INFERENCE_PROFILE.width)
    p.add_argument("--fps", type=int, default=DEFAULT_INFERENCE_PROFILE.fps)
    p.add_argument("--out", default="./outputs/lingbot_generate.mp4")
    # The optional refiner leaves the base path unchanged when disabled.
    # When set, the base 480p latent is upscaled -> re-noised at t_thresh ->
    # tail-denoised by the on-demand refiner DiT (model_root/refiner) to 1088p.
    p.add_argument(
        "--refine",
        action="store_true",
        help="Run the refiner stage after the base denoise (video-only; needs "
        "refiner/ weights).",
    )
    p.add_argument("--refiner-steps", type=int, default=8)
    p.add_argument("--refiner-cfg", type=float, default=3.0)
    p.add_argument("--refiner-shift", type=float, default=3.0)
    p.add_argument("--refiner-t-thresh", type=float, default=0.85)
    p.add_argument("--refiner-height", type=int, default=1088)
    p.add_argument("--refiner-width", type=int, default=1920)
    p.add_argument("--refiner-sigma-tail-steps", type=int, default=2)
    p.add_argument(
        "--refiner-scheduler",
        default="unipc",
        help="Refiner tail solver (registry name); family default unipc.",
    )
    args = p.parse_args(argv)
    profile = INFERENCE_PROFILES[str(args.inference_profile)]
    args.steps = profile.steps if args.steps is None else args.steps
    args.scheduler = profile.scheduler if args.scheduler is None else args.scheduler
    args.cfg_scale = profile.cfg_scale if args.cfg_scale is None else args.cfg_scale
    distilled = str(args.inference_profile) == "distilled-4step"
    if args.adapter_rank is None:
        args.adapter_rank = 128 if distilled else 32
    if args.adapter_alpha is None:
        args.adapter_alpha = 128.0 if distilled else 32.0
    if args.adapter_preset is None:
        args.adapter_preset = (
            "attn_shared_mlp" if distilled else "attn_routed_experts"
        )
    if distilled and not str(args.adapter).strip():
        p.error("--inference-profile distilled-4step requires --adapter")
    if args.task == "t2i":
        args.frames = 1
        if args.refine:
            p.error("--refine is video-only and cannot be used with --task t2i")
    if args.task == "ti2v" and not str(args.input_image).strip():
        p.error("--task ti2v requires --input-image")
    if args.task == "v2v" and not str(args.input_video).strip():
        p.error("--task v2v requires --input-video")
    if args.task not in {"ti2v"} and str(args.input_image).strip():
        p.error("--input-image is only valid with --task ti2v")
    if args.task not in {"v2v"} and str(args.input_video).strip():
        p.error("--input-video is only valid with --task v2v")
    if args.denoising_strength is not None and args.task != "v2v":
        p.error("--denoising-strength is only valid with --task v2v")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    batch_path = str(args.batch_prompts).strip()
    if batch_path and args.prompt:
        raise SystemExit("Provide either --prompt or --batch-prompts, not both.")
    if not batch_path and not args.prompt:
        raise SystemExit("Provide --prompt or --batch-prompts.")
    temp_cfg: Path | None = None
    config_path = str(args.config).strip()
    if not config_path:
        quant = str(args.quant)
        rendered = CONFIG_TEMPLATE.format(
            model_root=str(args.model_root).replace("\\", "/"),
            quant=quant if quant != "none" else "none",
            quant_strategy="auto" if quant != "none" else "disabled",
        )
        if str(args.adapter).strip():
            rendered += ADAPTER_TEMPLATE.format(
                preset=str(args.adapter_preset),
                rank=int(args.adapter_rank),
                alpha=float(args.adapter_alpha),
            )
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".toml", encoding="utf-8"
        ) as tmp:
            tmp.write(rendered)
            temp_cfg = Path(tmp.name)
        config_path = str(temp_cfg)
    try:
        if batch_path:
            # Batch mode reuses one session across generate() calls.
            return run_batch(args, config_path=config_path)
        # A single prompt uses one model build and one generation call while
        # preserving the entrypoint's documented output payload.
        spec = importlib.util.spec_from_file_location(
            "mirai_scripts_infer", ROOT / "scripts" / "infer.py"
        )
        assert spec is not None and spec.loader is not None
        infer_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(infer_mod)
        old_argv = sys.argv
        try:
            sys.argv = build_infer_argv(args, config_path=config_path)
            return int(infer_mod.main())
        finally:
            sys.argv = old_argv
    finally:
        if temp_cfg is not None:
            temp_cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
