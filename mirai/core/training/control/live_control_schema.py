"""Versioned persisted payloads for live training control."""

from __future__ import annotations

from typing import Any

from mirai.core.training.control.training_control_contract import normalize_training_control_request


LIVE_TRAINING_CONTROL_SCHEMA_FAMILY = "mirai.live_training_control"
LIVE_TRAINING_CONTROL_SCHEMA_VERSION = 1
LIVE_TRAINING_CONTROL_CAPABILITIES_PAYLOAD_TYPE = "capabilities"
LIVE_TRAINING_CONTROL_REQUEST_PAYLOAD_TYPE = "request"
LIVE_TRAINING_CONTROL_STATUS_PAYLOAD_TYPE = "status"
LIVE_TRAINING_CONTROL_REQUEST_STATES = (
    "requested",
    "accepted",
    "rejected",
    "applied",
    "failed",
)


def _base_payload(*, payload_type: str) -> dict[str, object]:
    return {
        "schema_family": LIVE_TRAINING_CONTROL_SCHEMA_FAMILY,
        "schema_version": LIVE_TRAINING_CONTROL_SCHEMA_VERSION,
        "payload_type": str(payload_type),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_command_specs(value: Any) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command", "")).strip().lower()
        if not command:
            continue
        normalized: dict[str, object] = {"command": command}
        description = str(item.get("description", "")).strip()
        if description:
            normalized["description"] = description
        boundary = str(item.get("boundary", "")).strip()
        if boundary:
            normalized["boundary"] = boundary
        command_class = str(item.get("command_class", "")).strip()
        if command_class:
            normalized["command_class"] = command_class
        safety_class = str(item.get("safety_class", "")).strip()
        if safety_class:
            normalized["safety_class"] = safety_class
        allowed = item.get("allowed_argument_keys", [])
        if isinstance(allowed, (list, tuple)):
            normalized["allowed_argument_keys"] = [
                str(key)
                for key in allowed
                if str(key).strip()
            ]
        argument_schema = item.get("argument_schema")
        if isinstance(argument_schema, dict):
            normalized["argument_schema"] = dict(argument_schema)
        result_schema = item.get("result_schema")
        if isinstance(result_schema, dict):
            normalized["result_schema"] = dict(result_schema)
        out.append(normalized)
    return out


def _normalize_runtime_interval(value: Any) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    command = str(value.get("command", "")).strip().lower()
    if not command:
        return None
    base_every_n_steps = _optional_int(value.get("base_every_n_steps"))
    effective_every_n_steps = _optional_int(value.get("effective_every_n_steps"))
    if base_every_n_steps is None or base_every_n_steps < 0:
        return None
    if effective_every_n_steps is None or effective_every_n_steps < 0:
        return None
    normalized: dict[str, object] = {
        "command": command,
        "mutable": bool(value.get("mutable", False)),
        "base_every_n_steps": int(base_every_n_steps),
        "effective_every_n_steps": int(effective_every_n_steps),
    }
    override_every_n_steps = _optional_int(value.get("override_every_n_steps"))
    if override_every_n_steps is not None:
        if override_every_n_steps < 0:
            return None
        normalized["override_every_n_steps"] = int(override_every_n_steps)
    else:
        normalized["override_every_n_steps"] = None
    return normalized


def _normalize_runtime_state(value: Any) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    sample_interval = _normalize_runtime_interval(value.get("sample_interval"))
    validation_interval = _normalize_runtime_interval(value.get("validation_interval"))
    if sample_interval is None and validation_interval is None:
        return None
    return {
        "sample_interval": sample_interval,
        "validation_interval": validation_interval,
    }


def _normalize_pending_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        request = normalize_training_control_request(
            seq=int(value.get("seq", 0)),
            command=str(value.get("command", "")),
            arguments=dict(value.get("arguments", {})),
        )
    except (TypeError, ValueError):
        return None
    return {
        "seq": int(request.seq),
        "command": str(request.command),
        "arguments": dict(request.arguments),
    }


def build_live_training_control_request_payload(
    *,
    job_id: str,
    seq: int,
    command: str,
    arguments: dict[str, Any] | None = None,
    state: str = "requested",
    updated_at: float,
    run_id: str = "",
    requested_at: float | None = None,
    boundary: str = "",
    dispatched_at_optimizer_step: int | None = None,
    applied_at_optimizer_step: int | None = None,
    error: str = "",
    reason: str = "",
    supported_commands: list[dict[str, object]] | None = None,
    pending_seq: int | None = None,
    pending_command: str = "",
    pending_arguments: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        **_base_payload(payload_type=LIVE_TRAINING_CONTROL_REQUEST_PAYLOAD_TYPE),
        "job_id": str(job_id),
        "seq": int(seq),
        "command": str(command),
        "arguments": dict(arguments or {}),
        "state": str(state),
        "updated_at": float(updated_at),
    }
    if run_id:
        payload["run_id"] = str(run_id)
    if requested_at is not None:
        payload["requested_at"] = float(requested_at)
    if boundary:
        payload["boundary"] = str(boundary)
    if dispatched_at_optimizer_step is not None:
        payload["dispatched_at_optimizer_step"] = int(dispatched_at_optimizer_step)
    if applied_at_optimizer_step is not None:
        payload["applied_at_optimizer_step"] = int(applied_at_optimizer_step)
    if error:
        payload["error"] = str(error)
    if reason:
        payload["reason"] = str(reason)
    if supported_commands:
        payload["supported_commands"] = _normalize_command_specs(list(supported_commands))
    if pending_seq is not None:
        payload["pending_seq"] = int(pending_seq)
    if pending_command:
        payload["pending_command"] = str(pending_command)
    if pending_arguments:
        payload["pending_arguments"] = dict(pending_arguments)
    if result is not None:
        payload["result"] = dict(result)
    normalized = normalize_live_training_control_request_payload(payload)
    if normalized is None:
        raise ValueError("Invalid live training control request payload.")
    return normalized


def normalize_live_training_control_request_payload(
    payload: dict[str, Any] | None,
    *,
    default_job_id: str = "",
    default_run_id: str = "",
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    payload_type = str(payload.get("payload_type", "")).strip().lower()
    if payload_type and payload_type != LIVE_TRAINING_CONTROL_REQUEST_PAYLOAD_TYPE:
        return None
    job_id = str(payload.get("job_id") or default_job_id).strip()
    if not job_id:
        return None
    try:
        request = normalize_training_control_request(
            seq=int(payload.get("seq", 0)),
            command=str(payload.get("command", "")),
            arguments=dict(payload.get("arguments", {})),
        )
    except (TypeError, ValueError):
        return None
    state = str(payload.get("state", "requested")).strip().lower() or "requested"
    if state not in set(LIVE_TRAINING_CONTROL_REQUEST_STATES):
        return None
    normalized: dict[str, Any] = {
        **_base_payload(payload_type=LIVE_TRAINING_CONTROL_REQUEST_PAYLOAD_TYPE),
        "job_id": job_id,
        "seq": int(request.seq),
        "command": str(request.command),
        "arguments": dict(request.arguments),
        "state": state,
        "updated_at": float(_optional_float(payload.get("updated_at")) or 0.0),
    }
    run_id = str(payload.get("run_id") or default_run_id).strip()
    if run_id:
        normalized["run_id"] = run_id
    requested_at = _optional_float(payload.get("requested_at"))
    if requested_at is not None:
        normalized["requested_at"] = requested_at
    boundary = str(payload.get("boundary", "")).strip()
    if boundary:
        normalized["boundary"] = boundary
    dispatched_at_optimizer_step = _optional_int(payload.get("dispatched_at_optimizer_step"))
    if dispatched_at_optimizer_step is not None:
        normalized["dispatched_at_optimizer_step"] = dispatched_at_optimizer_step
    applied_at_optimizer_step = _optional_int(payload.get("applied_at_optimizer_step"))
    if applied_at_optimizer_step is not None:
        normalized["applied_at_optimizer_step"] = applied_at_optimizer_step
    error = str(payload.get("error", "")).strip()
    if error:
        normalized["error"] = error
    reason = str(payload.get("reason", "")).strip()
    if reason:
        normalized["reason"] = reason
    supported_commands = _normalize_command_specs(payload.get("supported_commands"))
    if supported_commands:
        normalized["supported_commands"] = supported_commands
    pending_seq = _optional_int(payload.get("pending_seq"))
    if pending_seq is not None:
        normalized["pending_seq"] = pending_seq
    pending_command = str(payload.get("pending_command", "")).strip()
    if pending_command:
        normalized["pending_command"] = pending_command
    pending_arguments = payload.get("pending_arguments")
    if isinstance(pending_arguments, dict):
        normalized["pending_arguments"] = dict(pending_arguments)
    result = payload.get("result")
    if isinstance(result, dict):
        normalized["result"] = dict(result)
    return normalized


def build_live_training_control_status_payload(
    *,
    job_id: str,
    seq: int,
    command: str,
    arguments: dict[str, Any] | None = None,
    state: str,
    updated_at: float,
    run_id: str = "",
    boundary: str = "",
    applied_at_optimizer_step: int | None = None,
    error: str = "",
    reason: str = "",
    supported_commands: list[dict[str, object]] | None = None,
    pending_seq: int | None = None,
    pending_command: str = "",
    pending_arguments: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        **_base_payload(payload_type=LIVE_TRAINING_CONTROL_STATUS_PAYLOAD_TYPE),
        "job_id": str(job_id),
        "seq": int(seq),
        "command": str(command),
        "arguments": dict(arguments or {}),
        "state": str(state),
        "updated_at": float(updated_at),
    }
    if run_id:
        payload["run_id"] = str(run_id)
    if boundary:
        payload["boundary"] = str(boundary)
    if applied_at_optimizer_step is not None:
        payload["applied_at_optimizer_step"] = int(applied_at_optimizer_step)
    if error:
        payload["error"] = str(error)
    if reason:
        payload["reason"] = str(reason)
    if supported_commands:
        payload["supported_commands"] = _normalize_command_specs(list(supported_commands))
    if pending_seq is not None:
        payload["pending_seq"] = int(pending_seq)
    if pending_command:
        payload["pending_command"] = str(pending_command)
    if pending_arguments:
        payload["pending_arguments"] = dict(pending_arguments)
    if result is not None:
        payload["result"] = dict(result)
    normalized = normalize_live_training_control_status_payload(payload)
    if normalized is None:
        raise ValueError("Invalid live training control status payload.")
    return normalized


def normalize_live_training_control_status_payload(
    payload: dict[str, Any] | None,
    *,
    default_job_id: str = "",
    default_run_id: str = "",
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    payload_type = str(payload.get("payload_type", "")).strip().lower()
    if payload_type and payload_type != LIVE_TRAINING_CONTROL_STATUS_PAYLOAD_TYPE:
        return None
    job_id = str(payload.get("job_id") or default_job_id).strip()
    if not job_id:
        return None
    try:
        request = normalize_training_control_request(
            seq=int(payload.get("seq", 0)),
            command=str(payload.get("command", "")),
            arguments=dict(payload.get("arguments", {})),
        )
    except (TypeError, ValueError):
        return None
    state = str(payload.get("state", "requested")).strip().lower() or "requested"
    if state not in set(LIVE_TRAINING_CONTROL_REQUEST_STATES):
        return None
    normalized: dict[str, Any] = {
        **_base_payload(payload_type=LIVE_TRAINING_CONTROL_STATUS_PAYLOAD_TYPE),
        "job_id": job_id,
        "seq": int(request.seq),
        "command": str(request.command),
        "arguments": dict(request.arguments),
        "state": state,
        "updated_at": float(_optional_float(payload.get("updated_at")) or 0.0),
    }
    run_id = str(payload.get("run_id") or default_run_id).strip()
    if run_id:
        normalized["run_id"] = run_id
    boundary = str(payload.get("boundary", "")).strip()
    if boundary:
        normalized["boundary"] = boundary
    applied_at_optimizer_step = _optional_int(payload.get("applied_at_optimizer_step"))
    if applied_at_optimizer_step is not None:
        normalized["applied_at_optimizer_step"] = applied_at_optimizer_step
    error = str(payload.get("error", "")).strip()
    if error:
        normalized["error"] = error
    reason = str(payload.get("reason", "")).strip()
    if reason:
        normalized["reason"] = reason
    supported_commands = _normalize_command_specs(payload.get("supported_commands"))
    if supported_commands:
        normalized["supported_commands"] = supported_commands
    pending_seq = _optional_int(payload.get("pending_seq"))
    if pending_seq is not None:
        normalized["pending_seq"] = pending_seq
    pending_command = str(payload.get("pending_command", "")).strip()
    if pending_command:
        normalized["pending_command"] = pending_command
    pending_arguments = payload.get("pending_arguments")
    if isinstance(pending_arguments, dict):
        normalized["pending_arguments"] = dict(pending_arguments)
    result = payload.get("result")
    if isinstance(result, dict):
        normalized["result"] = dict(result)
    return normalized


def build_live_training_control_capabilities_payload(
    *,
    job_id: str,
    run_id: str,
    active: bool,
    global_step: int,
    boundary: str,
    gradient_accumulation: int,
    supported_commands: list[dict[str, object]],
    pending: dict[str, Any] | None,
    last_seen_seq: int,
    last_dispatched_seq: int,
    last_applied_seq: int,
    updated_at: float,
    last_request_status: dict[str, Any] | None = None,
    runtime_state: dict[str, object] | None = None,
) -> dict[str, Any]:
    payload = {
        **_base_payload(payload_type=LIVE_TRAINING_CONTROL_CAPABILITIES_PAYLOAD_TYPE),
        "job_id": str(job_id),
        "run_id": str(run_id),
        "active": bool(active),
        "global_step": int(global_step),
        "boundary": str(boundary),
        "gradient_accumulation": int(gradient_accumulation),
        "supported_commands": _normalize_command_specs(supported_commands),
        "pending": _normalize_pending_payload(pending),
        "last_seen_seq": int(last_seen_seq),
        "last_dispatched_seq": int(last_dispatched_seq),
        "last_applied_seq": int(last_applied_seq),
        "updated_at": float(updated_at),
    }
    if last_request_status is not None:
        payload["last_request_status"] = normalize_live_training_control_status_payload(
            dict(last_request_status),
            default_job_id=str(job_id),
            default_run_id=str(run_id),
        )
    if runtime_state is not None:
        payload["runtime_state"] = _normalize_runtime_state(runtime_state)
    normalized = normalize_live_training_control_capabilities_payload(payload)
    if normalized is None:
        raise ValueError("Invalid live training control capabilities payload.")
    return normalized


def normalize_live_training_control_capabilities_payload(
    payload: dict[str, Any] | None,
    *,
    default_job_id: str = "",
    default_run_id: str = "",
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    payload_type = str(payload.get("payload_type", "")).strip().lower()
    if payload_type and payload_type != LIVE_TRAINING_CONTROL_CAPABILITIES_PAYLOAD_TYPE:
        return None
    job_id = str(payload.get("job_id") or default_job_id).strip()
    if not job_id:
        return None
    run_id = str(payload.get("run_id") or default_run_id).strip()
    boundary = str(payload.get("boundary", "accumulation")).strip() or "accumulation"
    gradient_accumulation = _optional_int(payload.get("gradient_accumulation"))
    global_step = _optional_int(payload.get("global_step"))
    if gradient_accumulation is None or gradient_accumulation <= 0:
        return None
    if global_step is None:
        global_step = 0
    if global_step < 0:
        return None
    normalized: dict[str, Any] = {
        **_base_payload(payload_type=LIVE_TRAINING_CONTROL_CAPABILITIES_PAYLOAD_TYPE),
        "job_id": job_id,
        "run_id": run_id,
        "active": bool(payload.get("active", False)),
        "global_step": int(global_step),
        "boundary": boundary,
        "gradient_accumulation": int(gradient_accumulation),
        "supported_commands": _normalize_command_specs(payload.get("supported_commands")),
        "pending": _normalize_pending_payload(payload.get("pending")),
        "last_seen_seq": max(0, int(_optional_int(payload.get("last_seen_seq")) or 0)),
        "last_dispatched_seq": max(
            0,
            int(_optional_int(payload.get("last_dispatched_seq")) or 0),
        ),
        "last_applied_seq": max(
            0,
            int(_optional_int(payload.get("last_applied_seq")) or 0),
        ),
        "updated_at": float(_optional_float(payload.get("updated_at")) or 0.0),
    }
    last_request_status = normalize_live_training_control_status_payload(
        payload.get("last_request_status"),
        default_job_id=job_id,
        default_run_id=run_id,
    )
    normalized["last_request_status"] = last_request_status
    normalized["runtime_state"] = _normalize_runtime_state(payload.get("runtime_state"))
    return normalized
