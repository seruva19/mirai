"""Typed training event schema and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

EVENT_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


@dataclass(frozen=True)
class TrainingEvent:
    seq: int
    event_type: str
    timestamp: str
    run_id: str
    job_id: str
    payload: dict[str, Any]
    step: int | None = None
    schema_version: int = EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "seq": int(self.seq),
            "event_type": str(self.event_type),
            "timestamp": str(self.timestamp),
            "run_id": str(self.run_id),
            "job_id": str(self.job_id),
            "step": (None if self.step is None else int(self.step)),
            "payload": _json_safe(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingEvent":
        return cls(
            schema_version=int(payload.get("schema_version", EVENT_SCHEMA_VERSION)),
            seq=int(payload.get("seq", 0)),
            event_type=str(payload.get("event_type", "")),
            timestamp=str(payload.get("timestamp", "")),
            run_id=str(payload.get("run_id", "")),
            job_id=str(payload.get("job_id", "")),
            step=(
                None
                if payload.get("step") is None
                else int(payload.get("step", 0))
            ),
            payload=dict(payload.get("payload", {})),
        )


class TrainingEventBus:
    """In-process event bus with strictly monotonic sequence ids."""

    def __init__(
        self,
        *,
        run_id: str,
        job_id: str = "",
        start_seq: int = 0,
        subscribers: Iterable[Callable[[TrainingEvent], None]] | None = None,
    ):
        self.run_id = str(run_id)
        self.job_id = str(job_id)
        self._seq = max(0, int(start_seq))
        self._subscribers = list(subscribers or [])

    @property
    def current_seq(self) -> int:
        return int(self._seq)

    def subscribe(self, callback: Callable[[TrainingEvent], None]) -> None:
        self._subscribers.append(callback)

    def emit(
        self,
        event_type: str,
        *,
        step: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> TrainingEvent:
        self._seq += 1
        event = TrainingEvent(
            seq=int(self._seq),
            event_type=str(event_type),
            timestamp=_utc_now_iso(),
            run_id=self.run_id,
            job_id=self.job_id,
            step=(None if step is None else int(step)),
            payload=dict(payload or {}),
        )
        for callback in self._subscribers:
            callback(event)
        return event


def read_events_since(
    path: str | Path,
    *,
    after_seq: int = 0,
    limit: int = 1000,
) -> list[TrainingEvent]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return []
    out: list[TrainingEvent] = []
    max_items = max(1, int(limit))
    for line in p.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        event = TrainingEvent.from_dict(parsed)
        if int(event.seq) <= int(after_seq):
            continue
        out.append(event)
        if len(out) >= max_items:
            break
    return out


def read_last_event_seq(path: str | Path) -> int:
    events = read_events_since(path, after_seq=-1, limit=1_000_000)
    if not events:
        return 0
    return max(int(e.seq) for e in events)
