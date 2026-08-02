"""Event-bus and callback dispatch helpers for training loops."""

from __future__ import annotations

from mirai.core.training.observability.events import TrainingEvent, TrainingEventBus


def dispatch_step_callbacks(
    callbacks: list[object],
    *,
    step: int,
    metrics: dict[str, object],
) -> None:
    for cb in callbacks:
        on_step = getattr(cb, "on_step", None)
        if callable(on_step):
            try:
                on_step(step=step, metrics=metrics)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Callback %s.on_step failed at step %d: %s",
                    type(cb).__name__,
                    step,
                    exc,
                )


def dispatch_event_callbacks(callbacks: list[object], *, event: TrainingEvent) -> None:
    for cb in callbacks:
        on_event = getattr(cb, "on_event", None)
        if callable(on_event):
            try:
                on_event(event=event)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Callback %s.on_event failed: %s",
                    type(cb).__name__,
                    exc,
                )


def emit_event(
    event_bus: TrainingEventBus,
    callbacks: list[object],
    *,
    event_type: str,
    step: int | None = None,
    payload: dict[str, object] | None = None,
) -> TrainingEvent:
    event = event_bus.emit(
        event_type=event_type,
        step=step,
        payload=(payload or {}),
    )
    dispatch_event_callbacks(callbacks, event=event)
    return event
