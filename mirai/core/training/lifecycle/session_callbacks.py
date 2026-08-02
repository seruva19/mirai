"""Callback and event-bus assembly for training sessions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from mirai.core.training.observability.dispatch import emit_event
from mirai.core.training.observability.events import TrainingEventBus
from mirai.core.training.observability.setup import build_rank_callbacks


@dataclass
class SessionCallbackContext:
    callbacks: list[object]
    event_bus: TrainingEventBus


def build_session_callback_context(
    *,
    config: Any,
    output_dir: str,
    ckpt_dir: str,
    log_on_this_rank: bool,
    run_id: str,
    build_ckpt_payload: Callable[[int], dict[str, object]],
) -> SessionCallbackContext:
    callbacks: list[object] = []
    event_bus: TrainingEventBus | None = None

    callback_setup = build_rank_callbacks(
        config=config,
        output_dir=output_dir,
        ckpt_dir=ckpt_dir,
        log_on_this_rank=bool(log_on_this_rank),
        event_stream_raw=str(os.environ.get("MIRAI_JOB_EVENTS_PATH", "")).strip(),
        run_id=run_id,
        build_ckpt_payload=build_ckpt_payload,
        on_checkpoint_saved=lambda step, path: emit_event(
            event_bus,  # type: ignore[arg-type]
            callbacks,
            event_type="checkpoint.saved",
            step=int(step),
            payload={"path": str(path)},
        ),
    )
    event_bus = TrainingEventBus(
        run_id=run_id,
        job_id=str(os.environ.get("MIRAI_JOB_ID", "")).strip(),
        start_seq=int(callback_setup.start_event_seq),
    )
    callbacks.extend(callback_setup.callbacks)
    return SessionCallbackContext(
        callbacks=callbacks,
        event_bus=event_bus,
    )
