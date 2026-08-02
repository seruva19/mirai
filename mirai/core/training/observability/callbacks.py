"""Training callback utilities."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import importlib
from typing import Any, Callable, Protocol

from mirai.core.persistence.checkpoints import AsyncCheckpointWriter, CheckpointPolicy
from mirai.core.training.observability.events import TrainingEvent

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for callbacks: {exc}")


def _load_summary_writer():
    try:
        module = importlib.import_module("torch.utils.tensorboard")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "logging.tensorboard=true requires the 'tensorboard' package to be installed."
        ) from exc
    return module.SummaryWriter


class TrainingCallback(Protocol):
    def on_step(self, *, step: int, metrics: dict[str, Any]) -> None:
        """Called after each optimizer step."""

    def on_event(self, *, event: TrainingEvent) -> None:
        """Called for typed training lifecycle events."""

    def close(self) -> None:
        """Close callback resources."""


class TensorBoardCallback:
    def __init__(self, logdir: str | Path):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        summary_writer = _load_summary_writer()
        self.writer = summary_writer(log_dir=str(self.logdir))

    def on_step(self, *, step: int, metrics: dict[str, Any]) -> None:
        self.log_step(step=step, metrics=metrics)

    def log_step(self, *, step: int, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            try:
                scalar = float(value)
            except Exception:
                continue
            self.writer.add_scalar(key, scalar, global_step=step)
        self.writer.flush()

    def log_sample_image(self, *, step: int, tag: str, image: torch.Tensor) -> None:
        if image.ndim == 2:
            image = image.unsqueeze(0)
        self.writer.add_image(tag, image, global_step=step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class WandbCallback:
    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        config: dict[str, Any] | None = None,
    ):
        try:
            self._wandb = importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "logging.wandb=true requires the 'wandb' package to be installed."
            ) from exc
        self._run = self._wandb.init(
            project=str(project).strip() or "mirai",
            name=(str(run_name).strip() or None),
            config=(config or {}),
        )

    def on_step(self, *, step: int, metrics: dict[str, Any]) -> None:
        payload: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                payload[str(key)] = float(value)
            except Exception:
                continue
        if payload:
            self._run.log(payload, step=int(step))

    def close(self) -> None:
        self._run.finish()


class MetricsJsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def on_step(self, *, step: int, metrics: dict[str, Any]) -> None:
        payload = {"step": int(step), **metrics}
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class EventJsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def on_event(self, *, event: TrainingEvent) -> None:
        self._fh.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class StructlogWriter:
    def on_step(self, *, step: int, metrics: dict[str, Any]) -> None:
        payload = {"event": "train_step", "step": int(step), **metrics}
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)

    def close(self) -> None:
        return


class CheckpointSaver:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        save_every_n_steps: int,
        build_payload: Callable[[int], dict[str, Any]],
        on_checkpoint_saved: Callable[[int, Path], None] | None = None,
        policy: CheckpointPolicy | None = None,
        async_checkpoint: bool = False,
        async_checkpoint_max_gib: float = 0.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy or CheckpointPolicy.from_save_interval(save_every_n_steps)
        self.save_every_n_steps = self.policy.save_every_n_steps
        self.build_payload = build_payload
        self.last_checkpoint_path = ""
        self.on_checkpoint_saved = on_checkpoint_saved
        self._async_writer = (
            AsyncCheckpointWriter(
                max_snapshot_bytes=int(float(async_checkpoint_max_gib) * 1024**3)
            )
            if bool(async_checkpoint)
            else None
        )
        self._pending_notification: tuple[int, Path] | None = None

    def _finish_pending(self) -> None:
        if self._async_writer is None or self._pending_notification is None:
            return
        self._async_writer.wait()
        step, path = self._pending_notification
        self.last_checkpoint_path = str(path)
        if self.on_checkpoint_saved is not None:
            self.on_checkpoint_saved(step, path)
        self._pending_notification = None

    def on_step(self, *, step: int, metrics: dict[str, Any]) -> None:
        _ = metrics
        if not self.policy.should_save(step):
            return
        path = self.policy.checkpoint_path(self.output_dir, step)
        payload = self.build_payload(step)
        if self._async_writer is not None:
            self._finish_pending()
            self._async_writer.submit(self.policy, path, payload)
            self._pending_notification = (int(step), path)
        else:
            self.policy.save(path, payload)
            self.last_checkpoint_path = str(path)
            if self.on_checkpoint_saved is not None:
                self.on_checkpoint_saved(int(step), path)

    def close(self) -> None:
        self._finish_pending()
        if self._async_writer is not None:
            self._async_writer.close()
