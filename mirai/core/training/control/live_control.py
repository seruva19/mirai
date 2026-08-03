"""Live training-control bridge for active subprocess runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

from mirai.core.training.control.control_boundary import AccumulationBoundaryController
from mirai.core.training.observability.dispatch import emit_event
from mirai.core.training.control.live_control_schema import (
    build_live_training_control_capabilities_payload,
    build_live_training_control_request_payload,
    build_live_training_control_status_payload,
    normalize_live_training_control_capabilities_payload,
    normalize_live_training_control_request_payload,
    normalize_live_training_control_status_payload,
)
from mirai.core.training.control.training_control_contract import (
    TrainingControlRequest,
    normalize_training_control_request,
)


TRAINING_CONTROL_REQUESTS_NAMESPACE = "training_control_requests"
TRAINING_CONTROL_CAPABILITIES_NAMESPACE = "training_control_capabilities"
TRAINING_CONTROL_STATUS_NAMESPACE = "training_control_status"
LIVE_TRAINING_CONTROL_COMMANDS = (
    "save",
    "preview",
    "stop",
    "validate",
    "set_sample_interval",
    "set_validation_interval",
)


def _load_control_plane_state_row(
    *,
    db_path: str | Path,
    namespace: str,
    state_key: str,
) -> dict[str, Any] | None:
    path = Path(db_path)
    if not path.exists():
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT payload_json
            FROM control_plane_state
            WHERE namespace = ? AND state_key = ?
            """,
            (str(namespace), str(state_key)),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        payload = json.loads(str(row["payload_json"]))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _set_control_plane_state_row(
    *,
    db_path: str | Path,
    namespace: str,
    state_key: str,
    payload: dict[str, Any],
    updated_at: float,
) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS control_plane_state (
              namespace TEXT NOT NULL,
              state_key TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              updated_at REAL NOT NULL DEFAULT 0,
              PRIMARY KEY(namespace, state_key)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO control_plane_state(namespace, state_key, payload_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, state_key) DO UPDATE SET
              payload_json = excluded.payload_json,
              updated_at = excluded.updated_at
            """,
            (
                str(namespace),
                str(state_key),
                json.dumps(payload, sort_keys=True),
                float(updated_at),
            ),
        )
        conn.commit()
    finally:
        conn.close()


@dataclass(frozen=True)
class AppliedTrainingControl:
    seq: int
    command: str
    arguments: dict[str, Any]
    boundary: str
    applied_at_optimizer_step: int


class LiveTrainingController:
    def __init__(
        self,
        *,
        db_path: str | Path,
        job_id: str,
        run_id: str,
        event_bus: Any,
        callbacks: list[object],
        gradient_accumulation: int,
        supported_commands: tuple[str, ...] = LIVE_TRAINING_CONTROL_COMMANDS,
    ) -> None:
        self.db_path = Path(db_path)
        self.job_id = str(job_id)
        self.run_id = str(run_id)
        self.event_bus = event_bus
        self.callbacks = callbacks
        self.boundary = AccumulationBoundaryController(
            gradient_accumulation=int(gradient_accumulation),
            supported_commands=tuple(supported_commands),
        )
        self.highest_seen_seq = 0
        self.last_dispatched_seq = 0
        self.last_applied_seq = 0
        self.last_request_status: dict[str, Any] | None = None
        self.last_runtime_state: dict[str, object] | None = None
        prior_status = normalize_live_training_control_status_payload(
            _load_control_plane_state_row(
                db_path=self.db_path,
                namespace=TRAINING_CONTROL_STATUS_NAMESPACE,
                state_key=self.job_id,
            ),
            default_job_id=self.job_id,
            default_run_id=self.run_id,
        )
        if prior_status is not None:
            self.last_request_status = dict(prior_status)
            prior_seq = int(prior_status.get("seq", 0) or 0)
            self.highest_seen_seq = max(self.highest_seen_seq, prior_seq)
            if str(prior_status.get("state", "")).strip().lower() in {
                "applied",
                "rejected",
                "failed",
            }:
                self.last_dispatched_seq = max(self.last_dispatched_seq, prior_seq)
                self.last_applied_seq = max(self.last_applied_seq, prior_seq)

    def _publish_request_status(self, payload: dict[str, Any]) -> None:
        normalized = normalize_live_training_control_status_payload(
            payload,
            default_job_id=self.job_id,
            default_run_id=self.run_id,
        )
        if normalized is None:
            raise ValueError("Invalid live training control status payload.")
        self.last_request_status = dict(normalized)
        _set_control_plane_state_row(
            db_path=self.db_path,
            namespace=TRAINING_CONTROL_STATUS_NAMESPACE,
            state_key=self.job_id,
            payload=dict(normalized),
            updated_at=float(normalized.get("updated_at", time.time()) or time.time()),
        )

    @classmethod
    def from_env(
        cls,
        *,
        run_id: str,
        event_bus: Any,
        callbacks: list[object],
        gradient_accumulation: int,
    ) -> "LiveTrainingController" | None:
        job_id = str(os.environ.get("MIRAI_JOB_ID", "")).strip()
        db_path = str(os.environ.get("MIRAI_API_DB_PATH", "")).strip()
        if not job_id or not db_path:
            return None
        return cls(
            db_path=db_path,
            job_id=job_id,
            run_id=run_id,
            event_bus=event_bus,
            callbacks=callbacks,
            gradient_accumulation=int(gradient_accumulation),
        )

    def capabilities_payload(
        self,
        *,
        global_step: int,
        active: bool,
        runtime_state: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        if runtime_state is not None:
            self.last_runtime_state = dict(runtime_state)
        payload = self.boundary.capabilities()
        return build_live_training_control_capabilities_payload(
            job_id=self.job_id,
            run_id=self.run_id,
            active=bool(active),
            global_step=int(global_step),
            boundary=str(payload.get("boundary", "accumulation")),
            gradient_accumulation=int(payload.get("gradient_accumulation", 1)),
            supported_commands=list(payload.get("supported_commands", [])),
            pending=(
                dict(payload["pending"])
                if isinstance(payload.get("pending"), dict)
                else None
            ),
            last_seen_seq=int(self.highest_seen_seq),
            last_dispatched_seq=int(self.last_dispatched_seq),
            last_applied_seq=int(self.last_applied_seq),
            updated_at=float(time.time()),
            last_request_status=(
                None
                if self.last_request_status is None
                else dict(self.last_request_status)
            ),
            runtime_state=self.last_runtime_state,
        )

    def publish_capabilities(
        self,
        *,
        global_step: int,
        active: bool = True,
        runtime_state: dict[str, object] | None = None,
    ) -> None:
        payload = self.capabilities_payload(
            global_step=int(global_step),
            active=bool(active),
            runtime_state=runtime_state,
        )
        _set_control_plane_state_row(
            db_path=self.db_path,
            namespace=TRAINING_CONTROL_CAPABILITIES_NAMESPACE,
            state_key=self.job_id,
            payload=payload,
            updated_at=float(payload.get("updated_at", time.time()) or time.time()),
        )

    def poll_pending_request(self, *, global_step: int) -> dict[str, Any] | None:
        payload = normalize_live_training_control_request_payload(
            _load_control_plane_state_row(
                db_path=self.db_path,
                namespace=TRAINING_CONTROL_REQUESTS_NAMESPACE,
                state_key=self.job_id,
            ),
            default_job_id=self.job_id,
            default_run_id=self.run_id,
        )
        if not payload:
            return None
        try:
            request = normalize_training_control_request(
                seq=int(payload.get("seq", 0)),
                command=str(payload.get("command", "")),
                arguments=dict(payload.get("arguments", {})),
            )
        except (TypeError, ValueError):
            return None
        if int(request.seq) <= int(self.highest_seen_seq):
            return None
        self.highest_seen_seq = int(request.seq)
        accepted, meta = self.boundary.request(
            seq=int(request.seq),
            command=str(request.command),
            arguments=dict(request.arguments),
        )
        updated_at = float(time.time())
        status_payload = build_live_training_control_status_payload(
            job_id=self.job_id,
            run_id=self.run_id,
            seq=int(request.seq),
            command=str(request.command),
            arguments=dict(request.arguments),
            state="accepted" if accepted else "rejected",
            boundary=str(meta.get("boundary", "")),
            updated_at=updated_at,
            reason=str(meta.get("reason", "")),
            supported_commands=(
                list(meta.get("supported_commands", []))
                if isinstance(meta.get("supported_commands"), list)
                else None
            ),
            pending_seq=(
                int(meta["pending_seq"])
                if meta.get("pending_seq") is not None
                else None
            ),
            pending_command=str(meta.get("pending_command", "")),
            pending_arguments=(
                dict(meta["pending_arguments"])
                if isinstance(meta.get("pending_arguments"), dict)
                else None
            ),
        )
        _set_control_plane_state_row(
            db_path=self.db_path,
            namespace=TRAINING_CONTROL_REQUESTS_NAMESPACE,
            state_key=self.job_id,
            payload=build_live_training_control_request_payload(
                job_id=self.job_id,
                run_id=self.run_id,
                seq=int(request.seq),
                command=str(request.command),
                arguments=dict(request.arguments),
                state=str(status_payload["state"]),
                boundary=str(meta.get("boundary", "")),
                updated_at=updated_at,
                reason=str(meta.get("reason", "")),
                supported_commands=(
                    list(meta.get("supported_commands", []))
                    if isinstance(meta.get("supported_commands"), list)
                    else None
                ),
                pending_seq=(
                    int(meta["pending_seq"])
                    if meta.get("pending_seq") is not None
                    else None
                ),
                pending_command=str(meta.get("pending_command", "")),
                pending_arguments=(
                    dict(meta["pending_arguments"])
                    if isinstance(meta.get("pending_arguments"), dict)
                    else None
                ),
            ),
            updated_at=updated_at,
        )
        self._publish_request_status(status_payload)
        emit_event(
            self.event_bus,
            self.callbacks,
            event_type=(
                "control.request.accepted"
                if accepted
                else "control.request.rejected"
            ),
            payload={
                "seq": int(request.seq),
                "command": str(request.command),
                "arguments": dict(request.arguments),
                **dict(meta),
            },
        )
        self.publish_capabilities(global_step=int(global_step), active=True)
        return dict(meta)

    def on_microstep(
        self,
        *,
        microstep_index: int,
        optimizer_steps_committed: int,
        global_step: int,
    ) -> AppliedTrainingControl | None:
        applied = self.boundary.on_microstep(
            microstep_index=int(microstep_index),
            optimizer_steps_committed=int(optimizer_steps_committed),
        )
        if applied is None:
            return None
        self.last_dispatched_seq = max(
            int(self.last_dispatched_seq),
            int(applied["seq"]),
        )
        updated_at = float(time.time())
        _set_control_plane_state_row(
            db_path=self.db_path,
            namespace=TRAINING_CONTROL_REQUESTS_NAMESPACE,
            state_key=self.job_id,
            payload=build_live_training_control_request_payload(
                job_id=self.job_id,
                run_id=self.run_id,
                seq=int(applied["seq"]),
                command=str(applied["command"]),
                arguments=dict(applied["arguments"]),
                state="accepted",
                boundary=str(applied["boundary"]),
                dispatched_at_optimizer_step=int(applied["applied_at_optimizer_step"]),
                updated_at=updated_at,
            ),
            updated_at=updated_at,
        )
        emit_event(
            self.event_bus,
            self.callbacks,
            event_type="control.request.dispatched",
            step=int(global_step),
            payload=dict(applied),
        )
        self.publish_capabilities(global_step=int(global_step), active=True)
        return AppliedTrainingControl(
            seq=int(applied["seq"]),
            command=str(applied["command"]),
            arguments=dict(applied["arguments"]),
            boundary=str(applied["boundary"]),
            applied_at_optimizer_step=int(applied["applied_at_optimizer_step"]),
        )

    def mark_request_applied(self, *, applied_control: AppliedTrainingControl, global_step: int) -> None:
        self.mark_request_applied_with_result(
            applied_control=applied_control,
            global_step=global_step,
            result=None,
        )

    def mark_request_applied_with_result(
        self,
        *,
        applied_control: AppliedTrainingControl,
        global_step: int,
        result: dict[str, Any] | None,
    ) -> None:
        self.last_applied_seq = max(
            int(self.last_applied_seq),
            int(applied_control.seq),
        )
        updated_at = float(time.time())
        payload = build_live_training_control_status_payload(
            job_id=self.job_id,
            run_id=self.run_id,
            seq=int(applied_control.seq),
            command=str(applied_control.command),
            arguments=dict(applied_control.arguments),
            state="applied",
            boundary=str(applied_control.boundary),
            applied_at_optimizer_step=int(applied_control.applied_at_optimizer_step),
            updated_at=updated_at,
            result=result,
        )
        self._publish_request_status(payload)
        _set_control_plane_state_row(
            db_path=self.db_path,
            namespace=TRAINING_CONTROL_REQUESTS_NAMESPACE,
            state_key=self.job_id,
            payload=build_live_training_control_request_payload(
                job_id=self.job_id,
                run_id=self.run_id,
                seq=int(applied_control.seq),
                command=str(applied_control.command),
                arguments=dict(applied_control.arguments),
                state="applied",
                boundary=str(applied_control.boundary),
                applied_at_optimizer_step=int(applied_control.applied_at_optimizer_step),
                updated_at=updated_at,
                result=result,
            ),
            updated_at=updated_at,
        )
        emit_event(
            self.event_bus,
            self.callbacks,
            event_type="control.request.applied",
            step=int(global_step),
            payload=dict(payload),
        )
        self.publish_capabilities(global_step=int(global_step), active=True)

    def mark_request_failed(
        self,
        *,
        applied_control: AppliedTrainingControl,
        global_step: int,
        error: str,
    ) -> None:
        updated_at = float(time.time())
        payload = build_live_training_control_status_payload(
            job_id=self.job_id,
            run_id=self.run_id,
            seq=int(applied_control.seq),
            command=str(applied_control.command),
            arguments=dict(applied_control.arguments),
            state="failed",
            boundary=str(applied_control.boundary),
            applied_at_optimizer_step=int(applied_control.applied_at_optimizer_step),
            error=str(error),
            updated_at=updated_at,
        )
        self._publish_request_status(payload)
        _set_control_plane_state_row(
            db_path=self.db_path,
            namespace=TRAINING_CONTROL_REQUESTS_NAMESPACE,
            state_key=self.job_id,
            payload=build_live_training_control_request_payload(
                job_id=self.job_id,
                run_id=self.run_id,
                seq=int(applied_control.seq),
                command=str(applied_control.command),
                arguments=dict(applied_control.arguments),
                state="failed",
                boundary=str(applied_control.boundary),
                applied_at_optimizer_step=int(applied_control.applied_at_optimizer_step),
                error=str(error),
                updated_at=updated_at,
            ),
            updated_at=updated_at,
        )
        emit_event(
            self.event_bus,
            self.callbacks,
            event_type="control.request.failed",
            step=int(global_step),
            payload=dict(payload),
        )
        self.publish_capabilities(global_step=int(global_step), active=True)

    def close(
        self,
        *,
        global_step: int,
        runtime_state: dict[str, object] | None = None,
    ) -> None:
        self.publish_capabilities(
            global_step=int(global_step),
            active=False,
            runtime_state=runtime_state,
        )


def load_live_training_control_payload(
    *,
    db_path: str | Path,
    job_id: str,
) -> dict[str, Any] | None:
    payload = normalize_live_training_control_capabilities_payload(
        _load_control_plane_state_row(
            db_path=db_path,
            namespace=TRAINING_CONTROL_CAPABILITIES_NAMESPACE,
            state_key=str(job_id),
        ),
        default_job_id=str(job_id),
    )
    if not isinstance(payload, dict):
        return None
    return payload


def load_live_training_control_request_payload(
    *,
    db_path: str | Path,
    job_id: str,
) -> dict[str, Any] | None:
    payload = normalize_live_training_control_request_payload(
        _load_control_plane_state_row(
            db_path=db_path,
            namespace=TRAINING_CONTROL_REQUESTS_NAMESPACE,
            state_key=str(job_id),
        ),
        default_job_id=str(job_id),
    )
    if not isinstance(payload, dict):
        return None
    return payload


def load_live_training_control_status_payload(
    *,
    db_path: str | Path,
    job_id: str,
) -> dict[str, Any] | None:
    payload = normalize_live_training_control_status_payload(
        _load_control_plane_state_row(
            db_path=db_path,
            namespace=TRAINING_CONTROL_STATUS_NAMESPACE,
            state_key=str(job_id),
        ),
        default_job_id=str(job_id),
    )
    if not isinstance(payload, dict):
        return None
    return payload


def build_training_control_request_payload(
    *,
    seq: int,
    command: str,
    arguments: dict[str, Any] | None = None,
) -> TrainingControlRequest:
    return normalize_training_control_request(
        seq=int(seq),
        command=str(command),
        arguments=arguments,
    )
