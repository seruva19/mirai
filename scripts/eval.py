"""Deterministic evaluation for thin-slice checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.runtime_policy import runtime_policy_summary
from mirai.core.builtins import register_builtin_components
from mirai.core.dataset.cache import load_cache
from mirai.core.dataset.record_batch import (
    pad_text_embeddings,
    stack_latent_tensors,
)
from mirai.core.tensors import to_python_scalar
from mirai.core.persistence.checkpoints import load_checkpoint
from mirai.core.training.runtime.cli import (
    emit_runtime_policy_notes,
    load_runtime_config,
)
from mirai.core.training.data.schema import resolve_training_batch_schema
from mirai.core.training.evaluation.eval_gate import (
    handle_baseline_mode,
    load_measured_metrics_json,
    validate_eval_thresholds,
)
from mirai.core.training.evaluation.eval_metrics import (
    average_metric_series,
    compute_measured_eval_metrics,
    compute_frame_reconstruction_metrics,
    parse_resolution,
)
from mirai.core.training.objectives.weighting import compute_loss_weights
from mirai.core.training.trainer import Trainer

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to TOML config")
    p.add_argument("--checkpoint", default="", help="Checkpoint path")
    p.add_argument("--adapter", default="", help="Adapter-only path")
    p.add_argument("--seed", type=int, default=1337, help="Deterministic evaluation seed")
    p.add_argument("--out", default="", help="Optional path to write JSON report")
    p.add_argument(
        "--thresholds",
        default="artifacts/eval/absolute_thresholds.json",
        help="JSON file with absolute eval thresholds.",
    )
    p.add_argument(
        "--baseline-mode",
        default="",
        help="Optional baseline mode: establish|require_existing.",
    )
    p.add_argument(
        "--baseline-path",
        default="artifacts/eval/baseline.json",
        help="Baseline JSON path used with --baseline-mode.",
    )
    p.add_argument(
        "--measured-metrics-json",
        default="",
        help="Optional JSON file with externally measured numeric eval metrics.",
    )
    p.add_argument(
        "--allow-missing-frame-metrics",
        action="store_true",
        help="Allow eval to continue when latent decode/frame metrics fail.",
    )
    return p.parse_args()


def _load_thresholds(path: str | Path) -> dict[str, object]:
    in_path = Path(path)
    if not in_path.exists():
        return {}
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _passes_absolute_thresholds(
    *,
    metrics: dict[str, float],
    thresholds: dict[str, object],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, raw in thresholds.items():
        name = str(key)
        try:
            value = float(raw)
        except Exception:
            failures.append(f"{name}: non-numeric threshold")
            continue
        if name.endswith("_max"):
            metric_name = name[: -len("_max")]
            if metric_name not in metrics:
                failures.append(f"{metric_name}: missing metric")
                continue
            observed = float(metrics.get(metric_name, float("nan")))
            if not math.isfinite(observed):
                failures.append(f"{metric_name}: non-finite metric")
                continue
            if observed > value:
                failures.append(f"{metric_name}={observed:.6f} > max {value:.6f}")
        elif name.endswith("_min"):
            metric_name = name[: -len("_min")]
            if metric_name not in metrics:
                failures.append(f"{metric_name}: missing metric")
                continue
            observed = float(metrics.get(metric_name, float("nan")))
            if not math.isfinite(observed):
                failures.append(f"{metric_name}: non-finite metric")
                continue
            if observed < value:
                failures.append(f"{metric_name}={observed:.6f} < min {value:.6f}")
    return len(failures) == 0, failures


def _partition_thresholds_for_partial_frame_metrics(
    *,
    thresholds: dict[str, object],
    allow_missing_frame_metrics: bool,
    frame_metric_decode_failures: list[str],
) -> tuple[dict[str, object], list[str]]:
    if not allow_missing_frame_metrics or not frame_metric_decode_failures:
        return dict(thresholds), []
    active: dict[str, object] = {}
    skipped: list[str] = []
    for key, value in thresholds.items():
        name = str(key)
        metric_name = name
        if name.endswith("_max"):
            metric_name = name[: -len("_max")]
        elif name.endswith("_min"):
            metric_name = name[: -len("_min")]
        if metric_name.startswith("frame_"):
            skipped.append(name)
            continue
        active[name] = value
    return active, skipped


def _infer_eval_decode_shape(
    *,
    cfg,
    record: dict[str, object],
    latent_tensor: torch.Tensor,
) -> tuple[int, int, int]:
    frame_count = int(record.get("frame_count", record.get("frames", cfg.logging.sample_frame_count)))
    resolution_value = record.get(
        "bucket_resolution",
        record.get("resolution", cfg.logging.sample_resolution),
    )
    width, height = parse_resolution(resolution_value)
    if latent_tensor.ndim >= 5:
        frame_count = max(frame_count, int(latent_tensor.shape[2]))
    elif latent_tensor.ndim >= 4:
        frame_count = max(frame_count, int(latent_tensor.shape[1]))
    return int(frame_count), int(height), int(width)


def _broadcast_low_rank_eval_latent(
    *,
    pipeline,
    sample: torch.Tensor,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Shape a scalar/low-rank eval latent into a native [C,T,H,W] latent.

    Test families (and any asset-free cache) may cache latents as scalars with
    no temporal/spatial axes. The native VAE decode contract determines its
    output geometry from the latent alone, so frame-count-dependent eval metrics
    (e.g. ``frame_temporal_delta_*``) require a latent whose temporal axis
    reflects the requested ``frame_count``. Real 4D/5D latents pass through
    untouched. Uses the model-owned video latent layout for channels and the
    spatial/temporal compression ratios.
    """
    layout = pipeline.get_video_latent_layout()
    channels = max(1, int(layout.latent_channels))
    spatial = max(1, int(layout.spatial_downsample))
    temporal = max(1, int(getattr(layout, "temporal_downsample", 1)))
    t_lat = max(1, (int(frame_count) + temporal - 1) // temporal)
    h_lat = max(1, int(height) // spatial)
    w_lat = max(1, int(width) // spatial)
    value = float(sample.reshape(-1).float().mean().item())
    return torch.full((channels, t_lat, h_lat, w_lat), value, dtype=torch.float32)


def _decode_eval_frames(
    *,
    pipeline,
    latents: torch.Tensor,
    frame_count: int,
    height: int,
    width: int,
):
    if not hasattr(pipeline, "decode_latents_native") or not hasattr(pipeline, "load_vae"):
        raise RuntimeError(
            "Eval frame metrics require native VAE decode via "
            "load_vae/decode_latents_native; decode_latents is unsupported."
        )

    offload_fn = getattr(pipeline, "offload_vae", None)
    try:
        pipeline.load_vae(device="cuda" if torch.cuda.is_available() else "cpu")
        sample = latents[0] if latents.ndim >= 5 else latents
        sample = sample.detach().float()
        if sample.ndim < 4:
            sample = _broadcast_low_rank_eval_latent(
                pipeline=pipeline,
                sample=sample,
                frame_count=frame_count,
                height=height,
                width=width,
            )
        frames = pipeline.decode_latents_native([sample])
        return frames.detach().float()
    except NotImplementedError as exc:
        raise RuntimeError(
            "Eval frame metrics require a pipeline with native VAE decode; "
            "this model family does not implement it."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Native eval latent decode failed: {exc}") from exc
    finally:
        if callable(offload_fn):
            try:
                offload_fn()
            except Exception:
                pass


def _compute_frame_metrics_for_record(
    *,
    cfg,
    pipeline,
    record: dict[str, object],
    clean_latents: torch.Tensor,
    reconstructed_latents: torch.Tensor,
) -> dict[str, float]:
    frame_count, height, width = _infer_eval_decode_shape(
        cfg=cfg,
        record=record,
        latent_tensor=clean_latents,
    )
    clean_frames = _decode_eval_frames(
        pipeline=pipeline,
        latents=clean_latents,
        frame_count=frame_count,
        height=height,
        width=width,
    )
    reconstructed_frames = _decode_eval_frames(
        pipeline=pipeline,
        latents=reconstructed_latents,
        frame_count=frame_count,
        height=height,
        width=width,
    )
    return compute_frame_reconstruction_metrics(
        clean_frames=clean_frames,
        reconstructed_frames=reconstructed_frames,
    )


def main() -> int:
    args = parse_args()
    register_builtin_components()
    cfg, runtime_policy_notes = load_runtime_config(
        args.config,
        entrypoint="eval",
    )
    emit_runtime_policy_notes(runtime_policy_notes)
    trainer = Trainer(cfg)

    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint)
        trainer.load_state_dict(payload.get("trainer_state", {}))
    if args.adapter:
        payload = load_checkpoint(args.adapter)
        trainer.pipeline.load_state_dict(payload.get("adapter_state", {}))
    trainer.pipeline.eval()

    cache_path = Path(cfg.dataset.cache_path)
    if not cache_path.exists():
        raise SystemExit(f"Cache file not found: {cache_path}. Run scripts/cache.py first.")
    cache = load_cache(cache_path)
    records = cache.get("records", [])
    if not records:
        raise SystemExit("Cache has no records.")
    test_records = [r for r in records if str(r.get("split", "train")) == "test"]
    eval_records = test_records if test_records else list(records)
    batch_schema = resolve_training_batch_schema(
        model_type=str(cfg.model.type),
        strategy_type=str(cfg.strategy.type),
        pipeline=trainer.pipeline,
    )
    if batch_schema.requires("clip_embed") and not all(
        rec.get("clip_embed") is not None for rec in eval_records
    ):
        raise SystemExit(
            "Cache is missing clip_embed records required for image_to_video eval. "
            "Rebuild cache with strict native assets enabled and conditioning checkpoints present."
        )

    losses: list[float] = []
    measured_metric_rows: list[dict[str, float]] = []
    frame_metric_decode_failures: list[str] = []
    eps = cfg.training.timestep_eps
    for idx, record in enumerate(eval_records):
        seed = args.seed + idx
        g = torch.Generator()
        g.manual_seed(seed)

        clean = stack_latent_tensors([record["latent"]])
        text_embeds = {"t5": pad_text_embeddings([record["text_embed"]])}
        extra_forward_kwargs: dict[str, object] = {}
        if record.get("clip_embed") is not None:
            extra_forward_kwargs["clip_embed"] = pad_text_embeddings([record["clip_embed"]])
        timestep = torch.rand((1,), generator=g, dtype=torch.float32).clamp(min=eps, max=1.0 - eps)
        noise = torch.randn(clean.shape, generator=g, dtype=torch.float32)
        noisy = trainer.pipeline.apply_noise(clean, noise, timestep)
        prediction = trainer.pipeline.forward(noisy, timestep, text_embeds, **extra_forward_kwargs)
        target = trainer.pipeline.compute_target(noise=noise, clean_latents=clean, timesteps=timestep)
        per_sample = trainer.loss_fn(prediction, target)
        if isinstance(per_sample, torch.Tensor) and per_sample.ndim > 1:
            per_sample = per_sample.reshape(per_sample.shape[0], -1).mean(dim=1)
        weights = compute_loss_weights(
            timestep,
            mode=cfg.training.loss_weighting,
            gamma=cfg.training.min_snr_gamma,
            timestep_eps=eps,
        )
        losses.append(to_python_scalar((per_sample * weights).mean()))
        metrics_row = compute_measured_eval_metrics(
            clean_latents=clean,
            noisy_latents=noisy,
            prediction=prediction,
            target=target,
            timesteps=timestep,
            flow_shift=float(cfg.model.params.flow_shift),
        )
        reconstructed_clean = noisy - (
            trainer.pipeline.apply_noise(
                torch.zeros_like(clean),
                prediction,
                timestep,
            )
        )
        try:
            metrics_row.update(
                _compute_frame_metrics_for_record(
                    cfg=cfg,
                    pipeline=trainer.pipeline,
                    record=record,
                    clean_latents=clean,
                    reconstructed_latents=reconstructed_clean,
                )
            )
        except Exception as exc:
            sample_id = str(record.get("sample_id", f"idx_{idx}"))
            if not args.allow_missing_frame_metrics:
                raise SystemExit(
                    "Frame reconstruction metric generation failed for "
                    f"sample '{sample_id}': {exc}"
                ) from exc
            frame_metric_decode_failures.append(f"{sample_id}: {exc}")
        measured_metric_rows.append(metrics_row)

    report = {
        "status": "ok",
        "num_samples": len(losses),
        "dataset_split_used": ("test" if test_records else "all"),
        "seed": args.seed,
        "checkpoint": str(args.checkpoint) if args.checkpoint else "",
        "adapter": str(args.adapter) if args.adapter else "",
        "test_loss_mean": float(sum(losses) / len(losses)),
        "test_losses": losses,
        "runtime_policy": runtime_policy_summary(cfg),
        "runtime_policy_notes": list(runtime_policy_notes),
    }
    tail = losses[-min(len(losses), 10) :] if losses else []
    report["loss_tail_mean"] = float(sum(tail) / max(1, len(tail)))
    report["val_loss_tail_mean"] = float(report["loss_tail_mean"])
    measured_metrics = average_metric_series(measured_metric_rows)
    if str(args.measured_metrics_json).strip():
        measured_metrics.update(load_measured_metrics_json(args.measured_metrics_json))
    report["evaluation_contract"] = (
        "hybrid_measured" if measured_metrics else "loss_only"
    )
    report["measured_metrics"] = dict(measured_metrics)
    report["frame_metrics_complete"] = not frame_metric_decode_failures
    report["frame_metric_decode_failures"] = list(frame_metric_decode_failures)
    if frame_metric_decode_failures:
        report["status"] = "partial"
    for key, value in measured_metrics.items():
        report[str(key)] = float(value)

    thresholds = _load_thresholds(args.thresholds)
    active_thresholds, skipped_thresholds = _partition_thresholds_for_partial_frame_metrics(
        thresholds=thresholds,
        allow_missing_frame_metrics=bool(args.allow_missing_frame_metrics),
        frame_metric_decode_failures=frame_metric_decode_failures,
    )
    validate_eval_thresholds(
        active_thresholds,
        allowed_metrics=set(measured_metrics),
    )
    abs_passed, abs_failures = _passes_absolute_thresholds(
        metrics={str(k): float(v) for k, v in report.items() if isinstance(v, (int, float))},
        thresholds=active_thresholds,
    )
    report["absolute_thresholds_path"] = str(args.thresholds)
    report["absolute_thresholds_passed"] = bool(abs_passed)
    report["absolute_threshold_failures"] = abs_failures
    report["skipped_absolute_thresholds"] = skipped_thresholds
    if not abs_passed:
        report["status"] = "threshold_failed"
    if str(args.baseline_mode).strip():
        baseline_path = handle_baseline_mode(
            baseline_mode=str(args.baseline_mode),
            baseline_path=args.baseline_path,
            metrics_payload=report,
            absolute_thresholds_passed=bool(abs_passed),
        )
        report["baseline_path"] = str(baseline_path) if baseline_path is not None else ""

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if abs_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
