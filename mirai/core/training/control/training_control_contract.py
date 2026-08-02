"""Typed in-process control contract for training-time interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingControlCommandSpec:
    command: str
    description: str
    boundary: str = "accumulation"
    command_class: str = "runtime_mutation"
    safety_class: str = "boundary_safe"
    allowed_argument_keys: tuple[str, ...] = ()
    argument_schema: dict[str, object] | None = None
    result_schema: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "command": str(self.command),
            "description": str(self.description),
            "boundary": str(self.boundary),
            "command_class": str(self.command_class),
            "safety_class": str(self.safety_class),
            "allowed_argument_keys": list(self.allowed_argument_keys),
        }
        if self.argument_schema is not None:
            payload["argument_schema"] = dict(self.argument_schema)
        if self.result_schema is not None:
            payload["result_schema"] = dict(self.result_schema)
        return payload


@dataclass(frozen=True)
class TrainingControlRequest:
    seq: int
    command: str
    arguments: dict[str, Any]


TRAINING_CONTROL_COMMANDS: dict[str, TrainingControlCommandSpec] = {
    "save": TrainingControlCommandSpec(
        command="save",
        description="Persist a checkpoint or savepoint at the next safe accumulation boundary.",
        command_class="artifact_producing",
        allowed_argument_keys=("reason", "tag"),
        argument_schema={
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "tag": {"type": "string"},
            },
            "additionalProperties": False,
        },
        result_schema={
            "type": "object",
            "properties": {
                "checkpoint_path": {"type": "string"},
                "step": {"type": "integer", "minimum": 0},
            },
        },
    ),
    "pause": TrainingControlCommandSpec(
        command="pause",
        description="Pause training at the next safe accumulation boundary.",
        command_class="loop_control",
        allowed_argument_keys=("mode", "reason"),
        argument_schema={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["hold", "unload"]},
                "reason": {"type": "string"},
            },
            "additionalProperties": False,
        },
    ),
    "stop": TrainingControlCommandSpec(
        command="stop",
        description="Stop training cleanly at the next safe accumulation boundary.",
        command_class="loop_control",
        allowed_argument_keys=("reason",),
        argument_schema={
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
            },
            "additionalProperties": False,
        },
        result_schema={
            "type": "object",
            "properties": {
                "stop_requested": {"type": "boolean"},
                "step": {"type": "integer", "minimum": 0},
            },
        },
    ),
    "cancel": TrainingControlCommandSpec(
        command="cancel",
        description="Abort training at the next safe accumulation boundary.",
        command_class="loop_control",
        allowed_argument_keys=("reason",),
        argument_schema={
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
            },
            "additionalProperties": False,
        },
    ),
    "preview": TrainingControlCommandSpec(
        command="preview",
        description="Request preview/sample generation at the next safe accumulation boundary.",
        command_class="artifact_producing",
        allowed_argument_keys=("sample_name", "prompt"),
        argument_schema={
            "type": "object",
            "properties": {
                "sample_name": {"type": "string"},
                "prompt": {"type": "string"},
            },
            "additionalProperties": False,
        },
        result_schema={
            "type": "object",
            "properties": {
                "preview_requested": {"type": "boolean"},
                "sample_name": {"type": "string"},
                "prompt": {"type": "string"},
                "step": {"type": "integer", "minimum": 0},
            },
        },
    ),
    "validate": TrainingControlCommandSpec(
        command="validate",
        description="Force one validation pass at the next safe accumulation boundary.",
        command_class="observational",
        allowed_argument_keys=("reason",),
        argument_schema={
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
            },
            "additionalProperties": False,
        },
        result_schema={
            "type": "object",
            "properties": {
                "validation_forced": {"type": "boolean"},
                "step": {"type": "integer", "minimum": 0},
            },
        },
    ),
    "set_sample_interval": TrainingControlCommandSpec(
        command="set_sample_interval",
        description="Override the run-local preview/sample cadence for the active training session.",
        command_class="runtime_mutation",
        allowed_argument_keys=("every_n_steps",),
        argument_schema={
            "type": "object",
            "properties": {
                "every_n_steps": {"type": "integer", "minimum": 0},
            },
            "required": ["every_n_steps"],
            "additionalProperties": False,
        },
        result_schema={
            "type": "object",
            "properties": {
                "sample_interval_every_n_steps": {"type": "integer", "minimum": 0},
                "effective_sample_interval_every_n_steps": {
                    "type": "integer",
                    "minimum": 0,
                },
            },
        },
    ),
    "set_validation_interval": TrainingControlCommandSpec(
        command="set_validation_interval",
        description="Override the run-local validation cadence for the active training session.",
        command_class="runtime_mutation",
        allowed_argument_keys=("every_n_steps",),
        argument_schema={
            "type": "object",
            "properties": {
                "every_n_steps": {"type": "integer", "minimum": 0},
            },
            "required": ["every_n_steps"],
            "additionalProperties": False,
        },
        result_schema={
            "type": "object",
            "properties": {
                "validation_interval_every_n_steps": {"type": "integer", "minimum": 0},
                "effective_validation_interval_every_n_steps": {
                    "type": "integer",
                    "minimum": 0,
                },
            },
        },
    ),
}


def get_training_control_command_spec(command: str) -> TrainingControlCommandSpec:
    key = str(command).strip().lower()
    spec = TRAINING_CONTROL_COMMANDS.get(key)
    if spec is None:
        supported = ", ".join(sorted(TRAINING_CONTROL_COMMANDS))
        raise ValueError(
            f"Unsupported training control command '{command}'. Supported commands: {supported}."
        )
    return spec


def normalize_training_control_request(
    *,
    seq: int,
    command: str,
    arguments: dict[str, Any] | None = None,
) -> TrainingControlRequest:
    normalized_seq = int(seq)
    if normalized_seq <= 0:
        raise ValueError("seq must be > 0.")
    spec = get_training_control_command_spec(command)
    raw_arguments = dict(arguments or {})
    unknown_keys = sorted(
        key for key in raw_arguments.keys() if key not in set(spec.allowed_argument_keys)
    )
    if unknown_keys:
        allowed = ", ".join(spec.allowed_argument_keys) or "<none>"
        raise ValueError(
            f"{spec.command} arguments contain unsupported keys: {', '.join(unknown_keys)}. "
            f"Allowed keys: {allowed}."
        )
    normalized_arguments = {
        str(key): value
        for key, value in raw_arguments.items()
    }
    if spec.command == "pause" and "mode" in normalized_arguments:
        normalized_arguments["mode"] = str(normalized_arguments["mode"]).strip().lower()
    if spec.command in {"set_sample_interval", "set_validation_interval"} and "every_n_steps" in normalized_arguments:
        value = int(normalized_arguments["every_n_steps"])
        if value < 0:
            raise ValueError(f"{spec.command}.every_n_steps must be >= 0.")
        normalized_arguments["every_n_steps"] = int(value)
    return TrainingControlRequest(
        seq=normalized_seq,
        command=str(spec.command),
        arguments=normalized_arguments,
    )


def training_control_command_specs() -> list[dict[str, object]]:
    return [spec.to_dict() for spec in TRAINING_CONTROL_COMMANDS.values()]
