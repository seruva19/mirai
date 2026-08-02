"""Lifecycle ownership for power-function EMA snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mirai.core.training.optim.posthoc_ema import (
    save_posthoc_ema_snapshot,
    update_posthoc_ema_state,
)


def _snapshot_metadata(session: Any) -> dict[str, Any]:
    manifest = session.manifest
    return {
        "dataset_snapshot_id": str(manifest.dataset_snapshot_id),
        "cache_snapshot_id": str(manifest.cache_snapshot_id),
        "model_snapshot_id": str(getattr(manifest, "model_snapshot_id", "")),
        "config_snapshot_id": str(getattr(manifest, "config_snapshot_id", "")),
        "manifest_sha256": str(manifest.manifest_sha256),
    }


def posthoc_ema_snapshot_path(session: Any, *, step: int) -> Path:
    return Path(session.ckpt_dir) / "posthoc_ema" / f"step_{int(step):08d}.pt"


def update_and_maybe_save_posthoc_ema(session: Any) -> Path | None:
    config = session.config
    state = session.run_state
    if not bool(config.training.posthoc_ema_enabled):
        return None
    if state.posthoc_ema_state is None:
        raise RuntimeError("Post-hoc EMA is enabled but its run state is absent.")
    state.posthoc_ema_state = update_posthoc_ema_state(
        state.posthoc_ema_state,
        session.trainer.pipeline.state_dict(),
        next_step=int(state.global_step),
    )
    interval = int(config.training.posthoc_ema_snapshot_every_n_steps)
    if (
        not bool(session.log_on_this_rank)
        or interval <= 0
        or int(state.global_step) % interval != 0
    ):
        return None
    path = posthoc_ema_snapshot_path(session, step=int(state.global_step))
    return save_posthoc_ema_snapshot(
        path,
        state.posthoc_ema_state,
        metadata=_snapshot_metadata(session),
    )


def save_final_posthoc_ema_snapshot(session: Any) -> Path | None:
    config = session.config
    state = session.run_state
    if (
        not bool(config.training.posthoc_ema_enabled)
        or not bool(session.log_on_this_rank)
        or state.posthoc_ema_state is None
        or int(state.global_step) <= 0
    ):
        return None
    path = posthoc_ema_snapshot_path(session, step=int(state.global_step))
    return save_posthoc_ema_snapshot(
        path,
        state.posthoc_ema_state,
        metadata=_snapshot_metadata(session),
    )
