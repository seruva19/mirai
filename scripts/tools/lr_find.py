"""Learning-rate range finder for AdamW-style training."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.config.loader import load_config
from mirai.config.runtime_policy import runtime_policy_summary
from mirai.core.builtins import register_builtin_components
from mirai.core.dataset.cache import load_cache
from mirai.core.training.data.batches import sample_batch
from mirai.core.training.data.schema import resolve_training_batch_schema
from mirai.core.training.runtime.cli import (
    emit_runtime_policy_notes,
    load_runtime_config,
)
from mirai.core.training.runtime.gpu_lease import (
    GpuLeaseError,
    acquire_gpu_lease,
    resolve_lease_lock_path,
)
from mirai.core.training.optim.optimizer import build_optimizer
from mirai.core.training.trainer import Trainer

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to TOML config")
    p.add_argument("--steps", type=int, default=200, help="Number of LR sweep steps")
    p.add_argument("--lr-min", type=float, default=1e-6, help="Minimum learning rate")
    p.add_argument("--lr-max", type=float, default=1e-2, help="Maximum learning rate")
    p.add_argument("--seed", type=int, default=1234, help="Deterministic seed")
    p.add_argument(
        "--out",
        default="",
        help="Optional JSON output path. Defaults to {output_dir}/lr_find.json.",
    )
    return p.parse_args()

def _compute_recommended_lr(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return float(points[0][0]) if points else 0.0
    best_lr = float(points[0][0])
    best_slope = 0.0
    for idx in range(1, len(points)):
        prev_lr, prev_loss = points[idx - 1]
        curr_lr, curr_loss = points[idx]
        if prev_lr <= 0.0 or curr_lr <= 0.0:
            continue
        dx = math.log10(curr_lr) - math.log10(prev_lr)
        if dx == 0.0:
            continue
        slope = (curr_loss - prev_loss) / dx
        if slope < best_slope:
            best_slope = slope
            best_lr = curr_lr
    return float(best_lr)


def _lr_for_step(*, lr_min: float, lr_max: float, index: int, total: int) -> float:
    if total <= 1:
        return float(lr_min)
    ratio = max(0.0, min(1.0, float(index) / float(total - 1)))
    return float(lr_min * ((lr_max / lr_min) ** ratio))


def main() -> int:
    args = parse_args()
    if args.lr_min <= 0.0 or args.lr_max <= 0.0:
        raise SystemExit("--lr-min and --lr-max must be > 0.")
    if args.lr_min >= args.lr_max:
        raise SystemExit("--lr-min must be < --lr-max.")
    steps = max(1, int(args.steps))

    register_builtin_components()
    cfg, runtime_policy_notes = load_runtime_config(
        args.config,
        entrypoint="lr_find",
    )
    emit_runtime_policy_notes(runtime_policy_notes)
    trainer = Trainer(cfg)
    trainer.pipeline.train()

    cache_path = Path(cfg.dataset.cache_path)
    if not cache_path.exists():
        raise SystemExit(f"Cache file not found: {cache_path}. Run scripts/cache.py first.")
    cache = load_cache(cache_path)
    records = [r for r in cache.get("records", []) if str(r.get("split", "train")) == "train"]
    if not records:
        raise SystemExit("Cache has no train records.")
    batch_schema = resolve_training_batch_schema(
        model_type=str(cfg.model.type),
        strategy_type=str(cfg.strategy.type),
        pipeline=trainer.pipeline,
    )
    if batch_schema.requires("clip_embed") and not all(
        rec.get("clip_embed") is not None for rec in records
    ):
        raise SystemExit(
            "Cache is missing clip_embed records required for image_to_video LR finder. "
            "Rebuild cache with strict native assets enabled and conditioning checkpoints present."
        )

    params = list(trainer.get_trainable_parameters())
    named_params = list(trainer.pipeline.get_named_trainable_parameters())
    opt_result = build_optimizer(
        params=params,
        named_params=named_params,
        optimizer_type="adamw",
        lr=float(args.lr_min),
        weight_decay=cfg.optimizer.weight_decay,
        weight_decay_filter=cfg.optimizer.weight_decay_filter,
        loraplus_lr_ratio=cfg.optimizer.loraplus_lr_ratio,
        allow_fallback=True,
        stochastic_rounding=cfg.optimizer.stochastic_rounding,
    )
    optimizer = opt_result.optimizer
    rng = random.Random(int(args.seed))

    lock_path = resolve_lease_lock_path(ROOT)
    timeout_s = float(os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0"))

    points: list[tuple[float, float]] = []
    aborted = False
    abort_reason = ""
    try:
        with acquire_gpu_lease(lock_path=lock_path, timeout_seconds=timeout_s):
            for idx in range(steps):
                lr = _lr_for_step(
                    lr_min=float(args.lr_min),
                    lr_max=float(args.lr_max),
                    index=idx,
                    total=steps,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                batch = sample_batch(
                    records,
                    cfg.training.batch_size,
                    rng,
                    masked_loss=bool(cfg.training.masked_loss),
                )
                loss, _ = trainer.compute_loss(batch)
                if not torch.is_tensor(loss):
                    loss = torch.tensor(float(loss), dtype=torch.float32)
                if not torch.isfinite(loss).all():
                    aborted = True
                    abort_reason = "non_finite_loss"
                    break
                loss.backward()
                optimizer.step()
                points.append((float(lr), float(loss.detach().cpu().item())))
    except GpuLeaseError as exc:
        raise SystemExit(str(exc))

    recommended_lr = _compute_recommended_lr(points)
    out_path = (
        Path(args.out).resolve()
        if str(args.out).strip()
        else (Path(cfg.logging.output_dir).resolve() / "lr_find.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "ok" if not aborted else "aborted",
        "abort_reason": abort_reason,
        "steps_requested": int(steps),
        "steps_completed": int(len(points)),
        "lr_min": float(args.lr_min),
        "lr_max": float(args.lr_max),
        "recommended_lr": float(recommended_lr),
        "points": [{"lr": float(lr), "loss": float(loss)} for lr, loss in points],
        "runtime_policy": runtime_policy_summary(cfg),
        "runtime_policy_notes": list(runtime_policy_notes),
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "out": str(out_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
