"""Step-based curriculum schedule helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any


TRAINING_TASK_METADATA_KEY = "training_task"
TRAINING_TASKS = ("text_to_image", "text_to_video", "image_to_video")


@dataclass(frozen=True)
class CurriculumProfile:
    resolution: str | None = None
    frame_count: int | None = None
    task_weights: tuple[tuple[str, float], ...] = ()


def _parse_schedule(raw: Any, *, value_type: str) -> dict[int, Any]:
    out: dict[int, Any] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            out[int(key)] = value
        return out
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and ":" in item:
                step_text, _, value_text = item.partition(":")
                out[int(step_text.strip())] = value_text.strip()
                continue
            if isinstance(item, dict):
                if "step" in item and "value" in item:
                    out[int(item["step"])] = item["value"]
                    continue
                if len(item) == 1:
                    key, value = next(iter(item.items()))
                    out[int(key)] = value
                    continue
            raise ValueError(f"Invalid curriculum {value_type} schedule item: {item!r}")
        return out
    if raw in {None, ""}:
        return {}
    raise ValueError(f"Invalid curriculum {value_type} schedule payload: {raw!r}")


def _normalize_task_weights(raw: Any, *, step: int) -> tuple[tuple[str, float], ...]:
    if not isinstance(raw, dict):
        raise ValueError(
            "Invalid curriculum task-mix schedule at step "
            f"{int(step)}: expected a task-to-weight table."
        )
    unknown = sorted(str(task) for task in raw if str(task) not in TRAINING_TASKS)
    if unknown:
        raise ValueError(
            "Invalid curriculum task-mix schedule at step "
            f"{int(step)}: unsupported tasks {unknown}; expected only "
            f"{list(TRAINING_TASKS)}."
        )
    weights: list[tuple[str, float]] = []
    for task in TRAINING_TASKS:
        weight = float(raw.get(task, 0.0))
        if weight < 0.0:
            raise ValueError(
                "Invalid curriculum task-mix schedule at step "
                f"{int(step)}: weight for '{task}' must be >= 0."
            )
        if weight > 0.0:
            weights.append((task, weight))
    if not weights:
        raise ValueError(
            "Invalid curriculum task-mix schedule at step "
            f"{int(step)}: at least one task weight must be > 0."
        )
    return tuple(weights)


def record_training_task(
    record: Any,
    *,
    metadata_key: str = TRAINING_TASK_METADATA_KEY,
) -> str:
    """Read one canonical training task from cache-record metadata."""

    value = record.get(metadata_key)
    if value is None:
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict):
            value = metadata.get(metadata_key)
    task = str(value or "").strip().lower()
    if task and task not in TRAINING_TASKS:
        raise ValueError(
            f"Record has unsupported curriculum task '{task}' in metadata "
            f"key '{metadata_key}'; expected one of {list(TRAINING_TASKS)}."
        )
    return task


def _record_frame_count(record: Any) -> int | None:
    value = record.get(
        "bucket_frames",
        record.get("frame_count", record.get("frames")),
    )
    if value is None:
        return None
    return int(value)


def _record_resolution(record: Any) -> str:
    value = record.get("bucket_resolution", record.get("resolution"))
    if value is not None and str(value).strip():
        return str(value).strip().lower()
    bucket_h = int(record.get("bucket_h", 0) or 0)
    bucket_w = int(record.get("bucket_w", 0) or 0)
    if bucket_h > 0 and bucket_w > 0:
        return f"{bucket_h}x{bucket_w}"
    return ""


class CurriculumSchedule:
    def __init__(
        self,
        *,
        enabled: bool,
        resolution_schedule: dict[int, str],
        frame_schedule: dict[int, int],
        task_mix_schedule: dict[int, tuple[tuple[str, float], ...]],
    ):
        self.enabled = bool(enabled)
        self._resolution_schedule = dict(sorted(resolution_schedule.items()))
        self._frame_schedule = dict(sorted(frame_schedule.items()))
        self._task_mix_schedule = dict(sorted(task_mix_schedule.items()))
        self.task_metadata_key = TRAINING_TASK_METADATA_KEY

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> CurriculumSchedule:
        payload = raw if isinstance(raw, dict) else {}
        allowed = {
            "enabled",
            "resolution_schedule",
            "frame_schedule",
            "task_mix_schedule",
        }
        unknown = sorted(str(key) for key in payload if str(key) not in allowed)
        if unknown:
            raise ValueError(
                "Unknown training.curriculum keys: "
                + ", ".join(unknown)
                + "."
            )
        enabled = bool(payload.get("enabled", False))
        resolution_raw = _parse_schedule(
            payload.get("resolution_schedule", {}),
            value_type="resolution",
        )
        frame_raw = _parse_schedule(
            payload.get("frame_schedule", {}),
            value_type="frame",
        )
        task_mix_raw = _parse_schedule(
            payload.get("task_mix_schedule", {}),
            value_type="task-mix",
        )
        resolution_schedule = {
            int(step): str(value)
            for step, value in resolution_raw.items()
            if str(value).strip()
        }
        frame_schedule = {
            int(step): int(value)
            for step, value in frame_raw.items()
            if int(value) > 0
        }
        task_mix_schedule = {
            int(step): _normalize_task_weights(value, step=int(step))
            for step, value in task_mix_raw.items()
        }
        return cls(
            enabled=enabled,
            resolution_schedule=resolution_schedule,
            frame_schedule=frame_schedule,
            task_mix_schedule=task_mix_schedule,
        )

    def profile_for_step(self, step: int) -> CurriculumProfile:
        if not self.enabled:
            return CurriculumProfile()
        current_resolution: str | None = None
        current_frames: int | None = None
        current_task_weights: tuple[tuple[str, float], ...] = ()
        for start_step, resolution in self._resolution_schedule.items():
            if int(step) >= int(start_step):
                current_resolution = str(resolution)
        for start_step, frame_count in self._frame_schedule.items():
            if int(step) >= int(start_step):
                current_frames = int(frame_count)
        for start_step, task_weights in self._task_mix_schedule.items():
            if int(step) >= int(start_step):
                current_task_weights = tuple(task_weights)
        return CurriculumProfile(
            resolution=current_resolution,
            frame_count=current_frames,
            task_weights=current_task_weights,
        )

    @property
    def uses_task_mix(self) -> bool:
        return bool(self.enabled and self._task_mix_schedule)

    def select_task(
        self,
        *,
        step: int,
        seed: int,
        global_batch_index: int,
    ) -> str | None:
        """Choose one task deterministically for a homogeneous microbatch."""

        weights = self.profile_for_step(step).task_weights
        if not weights:
            return None
        digest = hashlib.blake2b(
            f"{int(seed)}:{int(global_batch_index)}".encode("utf-8"),
            digest_size=8,
        ).digest()
        unit = int.from_bytes(digest, "little") / float(2**64)
        total = sum(weight for _, weight in weights)
        target = unit * total
        cumulative = 0.0
        for task, weight in weights:
            cumulative += weight
            if target < cumulative:
                return task
        return weights[-1][0]

    def filter_records_for_batch(
        self,
        records: list[dict[str, Any]],
        *,
        step: int,
        seed: int,
        global_batch_index: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        task = self.select_task(
            step=step,
            seed=seed,
            global_batch_index=global_batch_index,
        )
        if task is None:
            return records, None
        filtered = [
            record
            for record in records
            if record_training_task(
                record,
                metadata_key=self.task_metadata_key,
            )
            == task
        ]
        if not filtered:
            raise ValueError(
                "Curriculum task mix selected "
                f"'{task}' at step {int(step)}, but the active stage has no "
                f"matching records in metadata key '{self.task_metadata_key}'."
            )
        return filtered, task

    def validate_records(self, records: list[dict[str, Any]]) -> None:
        """Fail before training when any configured stage/task pool is unusable."""

        if not self.uses_task_mix:
            return
        transition_steps = sorted(
            {
                *self._resolution_schedule,
                *self._frame_schedule,
                *self._task_mix_schedule,
            }
        )
        for step in transition_steps:
            stage_records = self.filter_records(records, step=step)
            for task, _weight in self.profile_for_step(step).task_weights:
                task_records = [
                    record
                    for record in stage_records
                    if record_training_task(
                        record,
                        metadata_key=self.task_metadata_key,
                    )
                    == task
                ]
                if not task_records:
                    raise ValueError(
                        "Curriculum stage at step "
                        f"{int(step)} assigns positive weight to '{task}', "
                        "but no matching records remain after the stage filters."
                    )
                if task == "text_to_image":
                    non_image = [
                        record
                        for record in task_records
                        if _record_frame_count(record) not in {None, 1}
                    ]
                    if non_image:
                        raise ValueError(
                            "Curriculum text_to_image records must have exactly "
                            "one frame."
                        )

    def filter_records(
        self,
        records: list[dict[str, Any]],
        *,
        step: int,
    ) -> list[dict[str, Any]]:
        profile = self.profile_for_step(step)
        resolution = profile.resolution
        frame_count = profile.frame_count
        if resolution is None and frame_count is None:
            return records

        def _match(rec: dict[str, Any]) -> bool:
            if resolution is not None:
                rec_resolution = _record_resolution(rec)
                if rec_resolution != str(resolution).strip().lower():
                    return False
            if frame_count is not None:
                rec_frames = _record_frame_count(rec)
                if rec_frames is None or int(rec_frames) != int(frame_count):
                    return False
            return True

        filtered = [rec for rec in records if _match(rec)]
        if not filtered and records:
            # Falling back to the unfiltered set would silently train the stage
            # on every bucket, which is the opposite of the requested schedule.
            raise ValueError(
                "Curriculum stage at step "
                f"{int(step)} matches no dataset records "
                f"(resolution={resolution!r}, frame_count={frame_count!r}). "
                "Align the schedule with the cached bucket resolutions and "
                "frame counts, or remove the stage."
            )
        return filtered
