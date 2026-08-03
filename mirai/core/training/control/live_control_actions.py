"""Runtime application of live training-control commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.persistence.checkpoints import save_checkpoint
from mirai.core.training.observability.dispatch import emit_event


@dataclass
class TrainingRuntimeOverrides:
    sample_every_n_steps_override: int | None = None
    val_every_n_steps_override: int | None = None


@dataclass(frozen=True)
class LiveControlActionResult:
    force_validation: bool = False
    force_preview: bool = False
    preview_name: str = ""
    preview_prompt: str = ""
    stop_requested: bool = False
    failed: bool = False
    error: str = ""


def build_training_runtime_overrides() -> TrainingRuntimeOverrides:
    return TrainingRuntimeOverrides()


def training_runtime_overrides_state_dict(
    overrides: TrainingRuntimeOverrides,
) -> dict[str, int | None]:
    return {
        "sample_every_n_steps_override": overrides.sample_every_n_steps_override,
        "val_every_n_steps_override": overrides.val_every_n_steps_override,
    }


def load_training_runtime_overrides_state(
    overrides: TrainingRuntimeOverrides,
    state: Any,
) -> None:
    if not isinstance(state, dict):
        return
    for key in (
        "sample_every_n_steps_override",
        "val_every_n_steps_override",
    ):
        value = state.get(key)
        if value is not None:
            value = int(value)
            if value < 0:
                raise ValueError(
                    f"Checkpoint runtime override '{key}' must be non-negative."
                )
        setattr(overrides, key, value)


def resolve_effective_sample_every_n_steps(session: Any) -> int:
    override = getattr(session.runtime_overrides, "sample_every_n_steps_override", None)
    if override is not None:
        return int(override)
    return int(session.config.logging.sample_every_n_steps)


def resolve_effective_val_every_n_steps(session: Any) -> int:
    override = getattr(session.runtime_overrides, "val_every_n_steps_override", None)
    if override is not None:
        return int(override)
    return int(session.config.training.val_every_n_steps)


def build_live_runtime_state_payload(session: Any) -> dict[str, object]:
    sample_override = getattr(session.runtime_overrides, "sample_every_n_steps_override", None)
    val_override = getattr(session.runtime_overrides, "val_every_n_steps_override", None)
    return {
        "sample_interval": {
            "command": "set_sample_interval",
            "mutable": True,
            "base_every_n_steps": int(session.config.logging.sample_every_n_steps),
            "override_every_n_steps": (
                None
                if sample_override is None
                else int(sample_override)
            ),
            "effective_every_n_steps": int(resolve_effective_sample_every_n_steps(session)),
        },
        "validation_interval": {
            "command": "set_validation_interval",
            "mutable": True,
            "base_every_n_steps": int(session.config.training.val_every_n_steps),
            "override_every_n_steps": (
                None
                if val_override is None
                else int(val_override)
            ),
            "effective_every_n_steps": int(resolve_effective_val_every_n_steps(session)),
        },
    }


def apply_live_training_control(
    session: Any,
    *,
    applied_control: Any | None,
    step: int,
) -> LiveControlActionResult:
    if applied_control is None:
        return LiveControlActionResult()
    command = str(getattr(applied_control, "command", "")).strip().lower()
    arguments = dict(getattr(applied_control, "arguments", {}))
    live_control = getattr(session, "live_control", None)
    try:
        if command == "save":
            if session.log_on_this_rank:
                manual_ckpt_path = (
                    session.ckpt_dir
                    / f"manual_step_{int(step)}_seq_{int(applied_control.seq)}.pt"
                )
                save_checkpoint(manual_ckpt_path, session.build_ckpt_payload(step))
                emit_event(
                    session.event_bus,
                    session.callbacks,
                    event_type="checkpoint.saved",
                    step=int(step),
                    payload={
                        "path": str(manual_ckpt_path),
                        "reason": "manual_control",
                        "control_seq": int(applied_control.seq),
                    },
                )
            if live_control is not None:
                live_control.mark_request_applied_with_result(
                    applied_control=applied_control,
                    global_step=int(step),
                    result={
                        "checkpoint_path": (
                            str(manual_ckpt_path)
                            if session.log_on_this_rank
                            else ""
                        ),
                        "step": int(step),
                    },
                )
                live_control.publish_capabilities(
                    global_step=int(step),
                    active=True,
                    runtime_state=build_live_runtime_state_payload(session),
                )
            return LiveControlActionResult()
        if command == "preview":
            preview_name = str(arguments.get("sample_name", "")).strip() or "manual_preview"
            preview_prompt = str(arguments.get("prompt", "")).strip()
            if live_control is not None:
                live_control.mark_request_applied_with_result(
                    applied_control=applied_control,
                    global_step=int(step),
                    result={
                        "preview_requested": True,
                        "sample_name": preview_name,
                        "prompt": preview_prompt,
                        "step": int(step),
                    },
                )
                live_control.publish_capabilities(
                    global_step=int(step),
                    active=True,
                    runtime_state=build_live_runtime_state_payload(session),
                )
            return LiveControlActionResult(
                force_preview=True,
                preview_name=preview_name,
                preview_prompt=preview_prompt,
            )
        if command == "stop":
            emit_event(
                session.event_bus,
                session.callbacks,
                event_type="run.stop_requested",
                step=int(step),
                payload={
                    "reason": str(arguments.get("reason", "")),
                    "control_seq": int(applied_control.seq),
                },
            )
            if live_control is not None:
                live_control.mark_request_applied_with_result(
                    applied_control=applied_control,
                    global_step=int(step),
                    result={
                        "stop_requested": True,
                        "step": int(step),
                    },
                )
                live_control.publish_capabilities(
                    global_step=int(step),
                    active=True,
                    runtime_state=build_live_runtime_state_payload(session),
                )
            return LiveControlActionResult(stop_requested=True)
        if command == "validate":
            emit_event(
                session.event_bus,
                session.callbacks,
                event_type="control.runtime_mutation.applied",
                step=int(step),
                payload={
                    "command": "validate",
                    "control_seq": int(applied_control.seq),
                },
            )
            if live_control is not None:
                live_control.mark_request_applied_with_result(
                    applied_control=applied_control,
                    global_step=int(step),
                    result={
                        "validation_forced": True,
                        "step": int(step),
                    },
                )
                live_control.publish_capabilities(
                    global_step=int(step),
                    active=True,
                    runtime_state=build_live_runtime_state_payload(session),
                )
            return LiveControlActionResult(force_validation=True)
        if command == "set_sample_interval":
            every_n_steps = int(arguments.get("every_n_steps", 0))
            session.runtime_overrides.sample_every_n_steps_override = int(every_n_steps)
            emit_event(
                session.event_bus,
                session.callbacks,
                event_type="control.runtime_mutation.applied",
                step=int(step),
                payload={
                    "command": "set_sample_interval",
                    "control_seq": int(applied_control.seq),
                    "every_n_steps": int(every_n_steps),
                },
            )
            if live_control is not None:
                live_control.mark_request_applied_with_result(
                    applied_control=applied_control,
                    global_step=int(step),
                    result={
                        "sample_interval_every_n_steps": int(every_n_steps),
                        "effective_sample_interval_every_n_steps": int(
                            resolve_effective_sample_every_n_steps(session)
                        ),
                    },
                )
                live_control.publish_capabilities(
                    global_step=int(step),
                    active=True,
                    runtime_state=build_live_runtime_state_payload(session),
                )
            return LiveControlActionResult()
        if command == "set_validation_interval":
            every_n_steps = int(arguments.get("every_n_steps", 0))
            session.runtime_overrides.val_every_n_steps_override = int(every_n_steps)
            emit_event(
                session.event_bus,
                session.callbacks,
                event_type="control.runtime_mutation.applied",
                step=int(step),
                payload={
                    "command": "set_validation_interval",
                    "control_seq": int(applied_control.seq),
                    "every_n_steps": int(every_n_steps),
                },
            )
            if live_control is not None:
                live_control.mark_request_applied_with_result(
                    applied_control=applied_control,
                    global_step=int(step),
                    result={
                        "validation_interval_every_n_steps": int(every_n_steps),
                        "effective_validation_interval_every_n_steps": int(
                            resolve_effective_val_every_n_steps(session)
                        ),
                    },
                )
                live_control.publish_capabilities(
                    global_step=int(step),
                    active=True,
                    runtime_state=build_live_runtime_state_payload(session),
                )
            return LiveControlActionResult()
        raise ValueError(f"Unsupported applied live control command '{command}'.")
    except Exception as exc:
        if live_control is not None:
            live_control.mark_request_failed(
                applied_control=applied_control,
                global_step=int(step),
                error=str(exc),
            )
            live_control.publish_capabilities(
                global_step=int(step),
                active=True,
                runtime_state=build_live_runtime_state_payload(session),
            )
        return LiveControlActionResult(
            failed=True,
            error=str(exc),
        )
