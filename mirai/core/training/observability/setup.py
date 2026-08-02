"""Rank-local callback assembly for training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from mirai.config.schema import TrainingConfig
from mirai.core.registry import Registry
from mirai.core.training.observability.callbacks import (
    CheckpointSaver,
    EventJsonlWriter,
    MetricsJsonlWriter,
    StructlogWriter,
    TensorBoardCallback,
    WandbCallback,
)
from mirai.core.training.observability.events import read_last_event_seq


@dataclass(frozen=True)
class CallbackSetup:
    callbacks: list[object]
    event_stream_path: Path
    start_event_seq: int


@dataclass(frozen=True)
class RankCallbackContext:
    """Inputs shared by every rank-local callback factory."""

    config: TrainingConfig
    output_root: Path
    ckpt_dir: str | Path
    event_stream_path: Path
    run_id: str
    build_ckpt_payload: Callable[[int], dict[str, Any]]
    on_checkpoint_saved: Callable[[int, Path], None]


# A callback factory returns the callbacks it contributes (possibly none, when
# its config gate is off). Builtins register at import in the order below; a
# third party appends its own by calling ``@register_rank_callback("name")`` on
# a factory, which slots in after the builtins (registration order preserved).
RankCallbackFactory = Callable[[RankCallbackContext], list[object]]
RankCallbackRegistry: Registry[RankCallbackFactory] = Registry("rank_callback")


def register_rank_callback(name: str) -> Callable[[RankCallbackFactory], RankCallbackFactory]:
    return RankCallbackRegistry.decorator(name)


@register_rank_callback("metrics_jsonl")
def _metrics_jsonl(ctx: RankCallbackContext) -> list[object]:
    return [MetricsJsonlWriter(ctx.output_root / "metrics.jsonl")]


@register_rank_callback("event_jsonl")
def _event_jsonl(ctx: RankCallbackContext) -> list[object]:
    return [EventJsonlWriter(ctx.event_stream_path)]


@register_rank_callback("structlog")
def _structlog(ctx: RankCallbackContext) -> list[object]:
    return [StructlogWriter()]


@register_rank_callback("tensorboard")
def _tensorboard(ctx: RankCallbackContext) -> list[object]:
    if not ctx.config.logging.tensorboard:
        return []
    return [TensorBoardCallback(ctx.output_root / "tb_logs")]


@register_rank_callback("wandb")
def _wandb(ctx: RankCallbackContext) -> list[object]:
    if not ctx.config.logging.wandb:
        return []
    return [
        WandbCallback(
            project=ctx.config.logging.wandb_project,
            run_name=(ctx.config.logging.wandb_run_name or ctx.run_id),
            config=asdict(ctx.config),
        )
    ]


@register_rank_callback("checkpoint")
def _checkpoint(ctx: RankCallbackContext) -> list[object]:
    return [
        CheckpointSaver(
            output_dir=ctx.ckpt_dir,
            save_every_n_steps=ctx.config.logging.save_every_n_steps,
            async_checkpoint=ctx.config.logging.async_checkpoint,
            async_checkpoint_max_gib=ctx.config.logging.async_checkpoint_max_gib,
            build_payload=ctx.build_ckpt_payload,
            on_checkpoint_saved=ctx.on_checkpoint_saved,
        )
    ]


def build_rank_callbacks(
    *,
    config: TrainingConfig,
    output_dir: str | Path,
    ckpt_dir: str | Path,
    log_on_this_rank: bool,
    event_stream_raw: str,
    run_id: str,
    build_ckpt_payload: Callable[[int], dict[str, Any]],
    on_checkpoint_saved: Callable[[int, Path], None],
) -> CallbackSetup:
    output_root = Path(output_dir)
    event_stream_path = (
        Path(str(event_stream_raw).strip())
        if str(event_stream_raw).strip()
        else (output_root / "events.jsonl")
    )
    start_event_seq = read_last_event_seq(event_stream_path) if log_on_this_rank else 0
    callbacks: list[object] = []
    if log_on_this_rank:
        ctx = RankCallbackContext(
            config=config,
            output_root=output_root,
            ckpt_dir=ckpt_dir,
            event_stream_path=event_stream_path,
            run_id=run_id,
            build_ckpt_payload=build_ckpt_payload,
            on_checkpoint_saved=on_checkpoint_saved,
        )
        for factory in RankCallbackRegistry.values():
            callbacks.extend(factory(ctx))
    return CallbackSetup(
        callbacks=callbacks,
        event_stream_path=event_stream_path,
        start_event_seq=int(start_event_seq),
    )
