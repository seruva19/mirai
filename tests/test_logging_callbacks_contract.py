from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from mirai.core.persistence.checkpoints import load_checkpoint
from mirai.core.training.observability.callbacks import (
    CheckpointSaver,
    TensorBoardCallback,
    WandbCallback,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class _FakeRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def log(self, payload: dict[str, float], step: int) -> None:
        self.logs.append((dict(payload), int(step)))

    def finish(self) -> None:
        self.finished = True


class _FakeWandb(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.inits: list[dict[str, object]] = []
        self.run = _FakeRun()

    def init(self, *, project: str, name: str | None, config: dict) -> _FakeRun:
        self.inits.append({"project": project, "name": name, "config": dict(config)})
        return self.run


class LoggingCallbackContractTests(unittest.TestCase):
    def test_wandb_callback_logs_numeric_metrics_and_finishes(self) -> None:
        fake = _FakeWandb()
        previous = sys.modules.get("wandb")
        sys.modules["wandb"] = fake  # type: ignore[assignment]
        try:
            callback = WandbCallback(
                project="project",
                run_name="run",
                config={"rank": 8},
            )
            callback.on_step(
                step=3,
                metrics={"loss": 0.5, "lr": 1e-3, "text": "ignored"},
            )
            callback.close()
        finally:
            if previous is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = previous

        self.assertEqual(fake.inits[0]["project"], "project")
        payload, step = fake.run.logs[0]
        self.assertEqual(step, 3)
        self.assertEqual(set(payload), {"loss", "lr"})
        self.assertTrue(fake.run.finished)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_tensorboard_callback_writes_scalar_and_image_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logdir = Path(tmp) / "tensorboard"
            callback = TensorBoardCallback(logdir)
            callback.log_step(
                step=1,
                metrics={"loss": 0.5, "lr": 1e-3, "grad_norm": 0.2},
            )
            callback.log_sample_image(
                step=1,
                tag="samples/preview",
                image=torch.rand((3, 8, 8), dtype=torch.float32),
            )
            callback.close()
            event_files = list(logdir.glob("events.out.tfevents.*"))
            self.assertTrue(event_files)
            self.assertTrue(all(path.stat().st_size > 0 for path in event_files))

    def test_missing_tensorboard_dependency_fails_explicitly(self) -> None:
        real_import_module = importlib.import_module

        def _import_without_tensorboard(name: str):
            if name == "torch.utils.tensorboard":
                raise ModuleNotFoundError("tensorboard")
            return real_import_module(name)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "mirai.core.training.observability.callbacks.importlib.import_module",
                side_effect=_import_without_tensorboard,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"logging\.tensorboard=true requires the 'tensorboard' package",
                ):
                    TensorBoardCallback(Path(tmp) / "tensorboard")

    @unittest.skipIf(torch is None, "torch not installed")
    def test_async_checkpoint_uses_immutable_bounded_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = torch.tensor([1.0])
            callback = CheckpointSaver(
                output_dir=tmp,
                save_every_n_steps=1,
                build_payload=lambda step: {"step": step, "value": source},
                async_checkpoint=True,
                async_checkpoint_max_gib=0.001,
            )
            callback.on_step(step=1, metrics={})
            source.add_(5.0)
            callback.close()
            payload = load_checkpoint(Path(tmp) / "step_1.pt")
            torch.testing.assert_close(payload["value"], torch.tensor([1.0]))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_async_checkpoint_rejects_snapshot_over_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = CheckpointSaver(
                output_dir=tmp,
                save_every_n_steps=1,
                build_payload=lambda step: {"value": torch.zeros(1024)},
                async_checkpoint=True,
                async_checkpoint_max_gib=1e-9,
            )
            with self.assertRaisesRegex(MemoryError, "exceeding"):
                callback.on_step(step=1, metrics={})
            callback.close()


if __name__ == "__main__":
    unittest.main()
