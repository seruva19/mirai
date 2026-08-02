"""Mutable training-loop state helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mirai.core.training.evaluation.early_stop import EarlyStopState
from mirai.core.training.optim.ema import init_ema_state
from mirai.core.training.optim.posthoc_ema import (
    init_posthoc_ema_state,
    normalize_posthoc_ema_state,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class TrainingRunState:
    global_step: int
    skipped_steps: int
    consecutive_skipped_steps: int
    early_stop_state: EarlyStopState
    ema_state: dict[str, Any] | None
    posthoc_ema_state: dict[str, Any] | None
    last_metrics: dict[str, object]
    stopped_early: bool
    gradient_offload_ops: int
    optimizer_offload_ops: int


def init_training_run_state(*, trainer: Any, config: Any) -> TrainingRunState:
    ema_state: dict[str, Any] | None = None
    posthoc_ema_state: dict[str, Any] | None = None
    if bool(config.training.ema_enabled):
        ema_state = init_ema_state(trainer.pipeline.state_dict())
    if bool(config.training.posthoc_ema_enabled):
        posthoc_ema_state = init_posthoc_ema_state(
            trainer.pipeline.state_dict(),
            profile_stds=config.training.posthoc_ema_profile_stds,
        )
    return TrainingRunState(
        global_step=0,
        skipped_steps=0,
        consecutive_skipped_steps=0,
        early_stop_state=EarlyStopState(
            best_val_loss=float("inf"),
            best_step=-1,
            patience_counter=0,
            best_checkpoint_path="",
        ),
        ema_state=ema_state,
        posthoc_ema_state=posthoc_ema_state,
        last_metrics={},
        stopped_early=False,
        gradient_offload_ops=0,
        optimizer_offload_ops=0,
    )


def capture_torch_rng_state() -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        return {}
    state: dict[str, Any] = {
        "cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_torch_rng_state(state: Any) -> None:
    if torch is None or not isinstance(state, dict):  # pragma: no cover
        return
    cpu_state = state.get("cpu")
    if isinstance(cpu_state, torch.Tensor):
        torch.set_rng_state(cpu_state)
    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        cuda_tensors = [item for item in cuda_state if isinstance(item, torch.Tensor)]
        if cuda_tensors:
            torch.cuda.set_rng_state_all(cuda_tensors)


def restore_training_run_state(
    *,
    state: TrainingRunState,
    payload: dict[str, Any],
    rng: Any,
    config: Any,
) -> TrainingRunState:
    state.global_step = int(payload.get("global_step", 0))
    state.skipped_steps = int(payload.get("skipped_steps", 0))
    state.consecutive_skipped_steps = int(payload.get("consecutive_skipped_steps", 0))
    early_stop_payload = payload.get("early_stop_state")
    if isinstance(early_stop_payload, dict):
        state.early_stop_state = EarlyStopState.from_dict(early_stop_payload)
    rng_state = payload.get("rng_state")
    if rng_state is not None:
        rng.setstate(rng_state)
    restore_torch_rng_state(payload.get("torch_rng_state"))
    if bool(config.training.ema_enabled) and isinstance(payload.get("ema_state"), dict):
        if torch is None:  # pragma: no cover
            state.ema_state = dict(payload["ema_state"])
        else:
            state.ema_state = {
                str(k): v.detach().cpu().float()
                for k, v in payload["ema_state"].items()
                if isinstance(v, torch.Tensor)
            }
    if bool(config.training.posthoc_ema_enabled) and isinstance(
        payload.get("posthoc_ema_state"),
        dict,
    ):
        state.posthoc_ema_state = normalize_posthoc_ema_state(
            payload["posthoc_ema_state"]
        )
    return state


def build_checkpoint_payload(
    *,
    step: int,
    trainer: Any,
    optimizer: Any,
    scheduler: Any,
    rng: Any,
    state: TrainingRunState,
    config: Any,
    manifest_sha256: str,
    dataset_snapshot_id: str = "",
    cache_snapshot_id: str = "",
    model_snapshot_id: str = "",
    config_snapshot_id: str = "",
    model_component_id: str = "",
) -> dict[str, object]:
    training_policies = getattr(trainer, "training_policies", None)
    policy_metadata = (
        training_policies.checkpoint_metadata()
        if training_policies is not None
        else {"active": [], "policies": {}}
    )
    active_policies = list(
        getattr(training_policies, "active_names", ())
    )
    payload = {
        "global_step": int(step),
        "trainer_state": trainer.state_dict(),
        "strategy_metadata": trainer.strategy.get_checkpoint_metadata(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": rng.getstate(),
        "torch_rng_state": capture_torch_rng_state(),
        "skipped_steps": int(state.skipped_steps),
        "consecutive_skipped_steps": int(state.consecutive_skipped_steps),
        "early_stop_state": state.early_stop_state.to_dict(),
        "ema_state": state.ema_state if config.training.ema_enabled else None,
        "ema_decay": float(config.training.ema_decay),
        "posthoc_ema_state": (
            state.posthoc_ema_state
            if config.training.posthoc_ema_enabled
            else None
        ),
        "config": asdict(config),
        "manifest_sha256": str(manifest_sha256),
        "dataset_snapshot_id": str(dataset_snapshot_id),
        "cache_snapshot_id": str(cache_snapshot_id),
        "model_snapshot_id": str(model_snapshot_id),
        "config_snapshot_id": str(config_snapshot_id),
        "model_component_id": str(model_component_id),
        "metadata": {
            "schema_version": 1,
            "manifest_sha256": str(manifest_sha256),
            "dataset_snapshot_id": str(dataset_snapshot_id),
            "cache_snapshot_id": str(cache_snapshot_id),
            "model_snapshot_id": str(model_snapshot_id),
            "config_snapshot_id": str(config_snapshot_id),
            "model_component_id": str(model_component_id),
        },
    }
    if active_policies:
        payload["training_policy_metadata"] = policy_metadata
        payload["metadata"]["training_policies"] = active_policies
    return payload
