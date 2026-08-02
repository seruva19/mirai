"""Control-command deferral to accumulation boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.training.control.training_control_contract import (
    TrainingControlRequest,
    normalize_training_control_request,
    training_control_command_specs,
)


@dataclass
class PendingControl:
    request: TrainingControlRequest


class AccumulationBoundaryController:
    def __init__(
        self,
        *,
        gradient_accumulation: int,
        supported_commands: tuple[str, ...] | None = None,
    ):
        self.gradient_accumulation = max(1, int(gradient_accumulation))
        self.pending: PendingControl | None = None
        self.supported_commands = tuple(
            str(command).strip().lower()
            for command in (supported_commands or tuple())
            if str(command).strip()
        )

    def request(
        self,
        *,
        seq: int,
        command: str,
        arguments: dict[str, Any] | None = None,
    ) -> tuple[bool, dict]:
        request = normalize_training_control_request(
            seq=seq,
            command=command,
            arguments=arguments,
        )
        if self.supported_commands and request.command not in set(self.supported_commands):
            return False, {
                "status": "rejected",
                "reason": "unsupported_command",
                "boundary": "accumulation",
                "supported_commands": list(self.supported_commands),
            }
        if self.pending is not None:
            return False, {
                "status": "rejected",
                "reason": "pending_command",
                "boundary": "accumulation",
                "pending_seq": self.pending.request.seq,
                "pending_command": self.pending.request.command,
                "pending_arguments": dict(self.pending.request.arguments),
            }
        self.pending = PendingControl(request=request)
        return True, {
            "status": "accepted",
            "deferred": True,
            "boundary": "accumulation",
            "seq": int(request.seq),
            "command": str(request.command),
            "arguments": dict(request.arguments),
        }

    def on_microstep(self, *, microstep_index: int, optimizer_steps_committed: int) -> dict | None:
        boundary = ((int(microstep_index) + 1) % self.gradient_accumulation) == 0
        if not boundary or self.pending is None:
            return None
        pending = self.pending.request
        self.pending = None
        return {
            "seq": int(pending.seq),
            "command": str(pending.command),
            "arguments": dict(pending.arguments),
            "boundary": "accumulation",
            "applied_at_optimizer_step": int(optimizer_steps_committed),
        }

    def capabilities(self) -> dict[str, object]:
        specs = training_control_command_specs()
        if self.supported_commands:
            specs = [
                spec
                for spec in specs
                if str(spec.get("command", "")).strip().lower()
                in set(self.supported_commands)
            ]
        return {
            "boundary": "accumulation",
            "gradient_accumulation": int(self.gradient_accumulation),
            "supported_commands": specs,
            "pending": (
                None
                if self.pending is None
                else {
                    "seq": int(self.pending.request.seq),
                    "command": str(self.pending.request.command),
                    "arguments": dict(self.pending.request.arguments),
                }
            ),
        }
