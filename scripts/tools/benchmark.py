"""Benchmark and validation harness for cache, training, and preview paths."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.loader import load_config
from mirai.config.runtime_policy import runtime_policy_summary
from mirai.core.builtins import register_builtin_components
from mirai.core.dataset.cache import build_cache, load_cache
from mirai.core.models.providers import model_supports_native_cache_encoding
from mirai.core.registry import ModelRegistry
from mirai.core.training.data.batches import sample_batch
from mirai.core.training.data.schema import resolve_training_batch_schema
from mirai.core.training.runtime.cli import (
    emit_runtime_policy_notes,
    load_runtime_config,
)
from mirai.core.training.residency.memory import current_vram_used_mb
from mirai.core.training.optim.optimizer import build_optimizer
from mirai.core.training.preview.preview import generate_preview
from mirai.core.training.residency.device_placement import (
    move_batch_to_device,
    place_pipeline_on_device,
    resolve_compute_device,
    resolve_compute_dtype,
)
from mirai.core.training.optim.scheduler import build_scheduler
from mirai.core.training.trainer import Trainer

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")

_MEDIA_SUFFIXES = {
    ".pt",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
}


def _text_embed_scalar(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    return float(value)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to TOML config.")
    p.add_argument("--out", default="", help="Optional path for JSON report.")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run benchmark harness (cache + train + preview).")
    run.add_argument("--steps", type=int, default=30, help="Short train benchmark steps.")

    cache_validate = sub.add_parser(
        "cache-validate",
        help="Validate caption/text cache invalidation and disk footprint.",
    )

    loss_cmp = sub.add_parser(
        "loss-weighting",
        help="Compare loss curves for uniform vs min_snr_gamma weighting.",
    )
    loss_cmp.add_argument("--steps", type=int, default=300, help="Comparison steps.")

    tol = sub.add_parser(
        "loss-tolerance",
        help="Compare baseline vs memory-feature loss curves and report %% delta.",
    )
    tol.add_argument("--steps", type=int, default=500, help="Comparison steps.")

    profile = sub.add_parser(
        "module-profile",
        help="Profile target module presets (step time/VRAM) on short runs.",
    )
    profile.add_argument("--steps", type=int, default=10, help="Short run steps per preset.")
    profile.add_argument("--rank", type=int, default=32, help="Adapter rank metadata override.")
    return p.parse_args()

def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return float(ordered[idx])


def _assert_native_i2v_clip_ready(*, cfg, records: list[dict]) -> None:
    schema = resolve_training_batch_schema(
        model_type=str(cfg.model.type),
        strategy_type=str(cfg.strategy.type),
    )
    if not schema.requires("clip_embed"):
        return
    if all(rec.get("clip_embed") is not None for rec in records):
        return
    raise SystemExit(
        "Cache is missing clip_embed records required for this model/strategy benchmark. "
        "Rebuild cache with strict native assets enabled and conditioning checkpoints present."
    )


def _run_short_train(
    *,
    cfg,
    records: list[dict],
    steps: int,
) -> dict:
    trainer = Trainer(cfg)
    compute_device = resolve_compute_device()
    compute_dtype = resolve_compute_dtype(cfg)
    place_pipeline_on_device(
        trainer.pipeline,
        device=compute_device,
        dtype=compute_dtype,
        residency_strategy=str(trainer.weight_residency_strategy),
    )
    params = list(trainer.get_trainable_parameters())
    named = list(trainer.pipeline.get_named_trainable_parameters())
    opt_result = build_optimizer(
        params=params,
        named_params=named,
        optimizer_type=cfg.optimizer.type,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        weight_decay_filter=cfg.optimizer.weight_decay_filter,
        loraplus_lr_ratio=cfg.optimizer.loraplus_lr_ratio,
        module_lr_multipliers=cfg.adapter.lr_multipliers,
        allow_fallback=cfg.optimizer.allow_fallback,
        stochastic_rounding=cfg.optimizer.stochastic_rounding,
    )
    optimizer = opt_result.optimizer
    scheduler = build_scheduler(optimizer, warmup_steps=cfg.training.warmup_steps)
    rng = random.Random(cfg.training.seed)

    timings_ms: list[float] = []
    losses: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    warmup_steps = 1
    measured_steps = max(1, int(steps))
    for step_index in range(warmup_steps + measured_steps):
        batch = sample_batch(
            records,
            int(cfg.training.batch_size),
            rng,
            masked_loss=bool(cfg.training.masked_loss),
        )
        batch = move_batch_to_device(
            batch,
            device=compute_device,
            dtype=compute_dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        if compute_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss, _ = trainer.compute_loss(batch)
        if torch.is_tensor(loss):
            loss.backward()
        optimizer.step()
        scheduler.step()
        if compute_device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if step_index < warmup_steps:
            continue
        timings_ms.append(float(elapsed_ms))
        losses.append(float(loss.item()) if torch.is_tensor(loss) else float(loss))

    avg_step_ms = float(sum(timings_ms) / max(1, len(timings_ms)))
    vram_peak_mb = 0.0
    if torch.cuda.is_available():
        vram_peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    return {
        "trainer": trainer,
        "warmup_steps": warmup_steps,
        "timings_ms": timings_ms,
        "losses": losses,
        "p95_step_time_ms": _p95(timings_ms),
        "avg_step_time_ms": avg_step_ms,
        "step_1000_wall_clock_s": avg_step_ms,
        "vram_peak_mb": float(max(vram_peak_mb, current_vram_used_mb())),
    }


def _resolve_output_path(args: argparse.Namespace, default_name: str) -> Path:
    if str(args.out).strip():
        return Path(str(args.out)).resolve()
    out_dir = ROOT / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    return (out_dir / default_name).resolve()


def _ensure_cache(cfg) -> tuple[dict, float]:
    t0 = time.perf_counter()
    payload = build_cache(
        cfg.dataset.path,
        cfg.dataset.cache_path,
        max_cache_skip_ratio=cfg.dataset.max_cache_skip_ratio,
        cache_mode=cfg.dataset.cache_mode,
        cache_compression=cfg.dataset.cache_compression,
        fp8_text_encoder=cfg.dataset.fp8_text_encoder,
        clips_per_video=cfg.dataset.clips_per_video,
        tag_shuffle_variants=cfg.dataset.tag_shuffle_variants,
        partial_recovery=cfg.dataset.partial_recovery,
        mask_extension=cfg.dataset.mask_extension,
        native_encode=model_supports_native_cache_encoding(str(cfg.model.type), cfg),
        model_type=cfg.model.type,
        model_variant=cfg.model.params.variant,
        model_path=cfg.model.path,
        model_dtype=cfg.model.dtype,
        model_denoiser_subfolder=str(getattr(cfg.model.params, "denoiser_subfolder", "transformer")),
        native_max_frames=33,
        strict_native_assets=bool(cfg.model.params.strict_native_assets),
    )
    elapsed = max(1e-9, time.perf_counter() - t0)
    return payload, elapsed


def _cmd_run(args: argparse.Namespace, cfg) -> dict:
    cache_payload, cache_elapsed = _ensure_cache(cfg)
    records = list(cache_payload.get("records", []))
    if not records:
        raise SystemExit("Cache has no records.")
    _assert_native_i2v_clip_ready(cfg=cfg, records=records)
    bench = _run_short_train(cfg=cfg, records=records, steps=int(args.steps))
    trainer = bench["trainer"]
    t0 = time.perf_counter()
    preview = generate_preview(
        trainer=trainer,
        step=int(args.steps),
        output_dir=cfg.logging.output_dir,
        prompt=cfg.logging.sample_prompt,
        seed=cfg.logging.sample_seed,
        sample_name="benchmark_preview",
        sample_blocks_to_swap=cfg.logging.sample_blocks_to_swap,
        training_blocks_to_swap=cfg.training.blocks_to_swap,
        block_swap_mode=cfg.training.block_swap_mode,
        block_swap_backward=cfg.training.block_swap_backward,
        sample_cfg_scale=cfg.logging.sample_cfg_scale,
        sample_negative_prompt=cfg.logging.sample_negative_prompt,
        sample_solver=cfg.logging.sample_solver,
        sample_resolution=cfg.logging.sample_resolution,
        sample_frame_count=cfg.logging.sample_frame_count,
    )
    preview_elapsed = time.perf_counter() - t0
    metrics = {
        "p95_step_time_ms": float(bench["p95_step_time_ms"]),
        "step_1000_wall_clock_s": float(bench["step_1000_wall_clock_s"]),
        "cache_throughput_samples_per_s": float(
            int(cache_payload.get("num_records", len(records))) / max(1e-9, cache_elapsed)
        ),
        "time_to_first_preview_s": float(preview_elapsed),
        "peak_vram_mb": float(bench["vram_peak_mb"]),
    }
    return {
        "status": "ok",
        "profile_id": "sparse_moe_test_ref",
        "num_records": int(cache_payload.get("num_records", len(records))),
        "cache_elapsed_s": float(cache_elapsed),
        "short_train_steps": int(args.steps),
        "warmup_steps": int(bench["warmup_steps"]),
        "metrics": metrics,
        "preview": preview,
    }


def _cmd_cache_validate(args: argparse.Namespace, cfg) -> dict:
    cache_payload, _ = _ensure_cache(cfg)
    records_before = list(cache_payload.get("records", []))
    if not records_before:
        raise SystemExit("Cache has no records.")

    dataset_path = Path(cfg.dataset.path)
    media_files = sorted(
        [path for path in dataset_path.iterdir() if path.is_file() and path.suffix.lower() in _MEDIA_SUFFIXES]
    )
    if not media_files:
        raise SystemExit("Dataset has no media files.")
    sample_file = media_files[0]
    caption_path = sample_file.with_suffix(".txt")
    original_caption = caption_path.read_text(encoding="utf-8") if caption_path.exists() else ""
    try:
        before_embed = None
        sample_id = sample_file.stem
        for rec in records_before:
            if str(rec.get("sample_id", "")).startswith(sample_id):
                before_embed = _text_embed_scalar(rec.get("text_embed", 0.0))
                break
        caption_path.write_text((original_caption.strip() + ", drift_check\n").lstrip(", "), encoding="utf-8")
        cache_payload_after = build_cache(
            cfg.dataset.path,
            cfg.dataset.cache_path,
            max_cache_skip_ratio=cfg.dataset.max_cache_skip_ratio,
            cache_mode=cfg.dataset.cache_mode,
            cache_compression=cfg.dataset.cache_compression,
            fp8_text_encoder=cfg.dataset.fp8_text_encoder,
            clips_per_video=cfg.dataset.clips_per_video,
            tag_shuffle_variants=cfg.dataset.tag_shuffle_variants,
            partial_recovery=False,
            mask_extension=cfg.dataset.mask_extension,
            native_encode=model_supports_native_cache_encoding(str(cfg.model.type), cfg),
            model_type=cfg.model.type,
            model_variant=cfg.model.params.variant,
            model_path=cfg.model.path,
            model_dtype=cfg.model.dtype,
            model_denoiser_subfolder=str(getattr(cfg.model.params, "denoiser_subfolder", "transformer")),
            native_max_frames=33,
            strict_native_assets=bool(cfg.model.params.strict_native_assets),
        )
        after_embed = None
        for rec in cache_payload_after.get("records", []):
            if str(rec.get("sample_id", "")).startswith(sample_id):
                after_embed = _text_embed_scalar(rec.get("text_embed", 0.0))
                break
    finally:
        caption_path.write_text(original_caption, encoding="utf-8")

    cache_file = Path(cfg.dataset.cache_path)
    disk_bytes = int(cache_file.stat().st_size) if cache_file.exists() else 0
    return {
        "status": "ok",
        "sample_id": sample_id,
        "before_text_embed": before_embed,
        "after_text_embed": after_embed,
        "invalidation_detected": bool(before_embed != after_embed),
        "num_records": int(cache_payload_after.get("num_records", 0)),
        "estimated_disk_bytes": int(cache_payload_after.get("estimated_disk_bytes", 0)),
        "cache_file_bytes": int(disk_bytes),
    }


def _loss_curve(cfg, records: list[dict], *, mode: str, steps: int) -> list[float]:
    cfg_local = deepcopy(cfg)
    cfg_local.training.loss_weighting = str(mode)
    bench = _run_short_train(cfg=cfg_local, records=records, steps=steps)
    return [float(v) for v in bench["losses"]]


def _cmd_loss_weighting(args: argparse.Namespace, cfg) -> dict:
    cache = load_cache(cfg.dataset.cache_path) if Path(cfg.dataset.cache_path).exists() else _ensure_cache(cfg)[0]
    records = list(cache.get("records", []))
    if not records:
        raise SystemExit("Cache has no records.")
    _assert_native_i2v_clip_ready(cfg=cfg, records=records)
    steps = max(1, int(args.steps))
    uniform = _loss_curve(cfg, records, mode="uniform", steps=steps)
    min_snr = _loss_curve(cfg, records, mode="min_snr_gamma", steps=steps)
    return {
        "status": "ok",
        "steps": steps,
        "uniform": {
            "losses": uniform,
            "loss_mean": float(sum(uniform) / max(1, len(uniform))),
            "loss_last": float(uniform[-1]),
        },
        "min_snr_gamma": {
            "losses": min_snr,
            "loss_mean": float(sum(min_snr) / max(1, len(min_snr))),
            "loss_last": float(min_snr[-1]),
        },
    }


def _cmd_module_profile(args: argparse.Namespace, cfg) -> dict:
    cache = load_cache(cfg.dataset.cache_path) if Path(cfg.dataset.cache_path).exists() else _ensure_cache(cfg)[0]
    records = list(cache.get("records", []))
    if not records:
        raise SystemExit("Cache has no records.")
    _assert_native_i2v_clip_ready(cfg=cfg, records=records)
    base_trainer = Trainer(cfg)
    presets = sorted(base_trainer.pipeline.get_adapter_target_presets().keys())
    output: dict[str, dict[str, float]] = {}
    for preset in presets:
        cfg_local = deepcopy(cfg)
        cfg_local.adapter.target_preset = str(preset)
        cfg_local.adapter.rank = int(args.rank)
        bench = _run_short_train(cfg=cfg_local, records=records, steps=max(1, int(args.steps)))
        output[str(preset)] = {
            "p95_step_time_ms": float(bench["p95_step_time_ms"]),
            "peak_vram_mb": float(bench["vram_peak_mb"]),
        }
    return {
        "status": "ok",
        "rank": int(args.rank),
        "steps_per_preset": int(args.steps),
        "results": output,
    }


def _family_supports_quantized_frozen_weights(cfg) -> bool:
    """Probe whether the configured family implements quantized frozen weights."""
    try:
        model_cls = ModelRegistry.get(str(cfg.model.type))
        factory = getattr(model_cls, "from_training_config", None)
        pipeline = factory(cfg) if callable(factory) else model_cls(cfg.model)
        return bool(pipeline.supports_quantized_frozen_weights())
    except Exception:
        return False


def _cmd_loss_tolerance(args: argparse.Namespace, cfg) -> dict:
    cache = load_cache(cfg.dataset.cache_path) if Path(cfg.dataset.cache_path).exists() else _ensure_cache(cfg)[0]
    records = list(cache.get("records", []))
    if not records:
        raise SystemExit("Cache has no records.")
    steps = max(1, int(args.steps))
    baseline_losses = _loss_curve(cfg, records, mode=str(cfg.training.loss_weighting), steps=steps)
    cfg_flags = deepcopy(cfg)
    # Apply only memory features declared by the configured provider and record
    # the resolved surface in the report.
    fp8_applied = _family_supports_quantized_frozen_weights(cfg)
    if fp8_applied:
        cfg_flags.memory.frozen_weight_quantization = "fp8"
    cfg_flags.memory.weight_residency_strategy = "stream_disk"
    flagged_losses = _loss_curve(
        cfg_flags,
        records,
        mode=str(cfg_flags.training.loss_weighting),
        steps=steps,
    )
    baseline_mean = float(sum(baseline_losses) / max(1, len(baseline_losses)))
    flagged_mean = float(sum(flagged_losses) / max(1, len(flagged_losses)))
    delta_pct = (
        abs(flagged_mean - baseline_mean) / max(1e-9, abs(baseline_mean)) * 100.0
    )
    return {
        "status": "ok",
        "steps": steps,
        "fp8_frozen_weight_quantization_applied": bool(fp8_applied),
        "weight_residency_strategy": "stream_disk",
        "baseline_loss_mean": baseline_mean,
        "feature_flags_loss_mean": flagged_mean,
        "loss_delta_pct": float(delta_pct),
        "within_5pct_tolerance": bool(delta_pct <= 5.0),
    }


def main() -> int:
    args = _parse_args()
    register_builtin_components()
    cfg, runtime_policy_notes = load_runtime_config(
        args.config,
        entrypoint="benchmark",
    )
    emit_runtime_policy_notes(runtime_policy_notes)

    if args.cmd == "run":
        report = _cmd_run(args, cfg)
        default_name = "benchmark_latest.json"
    elif args.cmd == "cache-validate":
        report = _cmd_cache_validate(args, cfg)
        default_name = "cache_validation.json"
    elif args.cmd == "loss-weighting":
        report = _cmd_loss_weighting(args, cfg)
        default_name = "loss_weighting_compare.json"
    elif args.cmd == "module-profile":
        report = _cmd_module_profile(args, cfg)
        default_name = "target_module_profile.json"
    elif args.cmd == "loss-tolerance":
        report = _cmd_loss_tolerance(args, cfg)
        default_name = "loss_tolerance.json"
    else:
        raise SystemExit(f"Unknown command '{args.cmd}'.")

    out_path = _resolve_output_path(args, default_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_policy"] = runtime_policy_summary(cfg)
    report["runtime_policy_notes"] = list(runtime_policy_notes)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(out_path), **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
