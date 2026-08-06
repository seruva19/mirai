from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from mirai.config.schema import TrainingConfig
from mirai.core.persistence.checkpoints import load_checkpoint, save_checkpoint
from mirai.core.training.data.curriculum import CurriculumSchedule
from mirai.core.training.evaluation.early_stop import EarlyStopState
from mirai.core.training.optim.gradients import resolve_non_finite_policy
from mirai.core.training.observability.metrics import apply_validation_policy
from mirai.core.training.lifecycle.session_state import seed_torch_rng
from mirai.core.training.lifecycle.resume_validation import (
    optimizer_implementation_identity,
    validate_resume_checkpoint_compatibility,
)
from mirai.core.training.runtime.execution import (
    accumulate_training_gradients,
    apply_optimizer_update,
)
from mirai.core.training.objectives.engine import TrainingLossResult
from mirai.core.training.control.live_control import (
    TRAINING_CONTROL_REQUESTS_NAMESPACE,
    LiveTrainingController,
    _set_control_plane_state_row,
    load_live_training_control_status_payload,
)
from mirai.core.training.control.live_control_schema import (
    build_live_training_control_request_payload,
)
from mirai.core.training.control.live_control_actions import (
    build_training_runtime_overrides,
    load_training_runtime_overrides_state,
    training_runtime_overrides_state_dict,
)
from mirai.core.training.observability.events import TrainingEventBus
from scripts.tools.lr_find import _compute_recommended_lr


class TrainingRuntimePolicyTests(unittest.TestCase):
    @staticmethod
    def _resume_payload(config: TrainingConfig) -> dict:
        return {
            "global_step": 1,
            "config": asdict(config),
            "dataset_snapshot_id": "dataset",
            "cache_snapshot_id": "cache",
            "model_snapshot_id": "model",
            "metadata": {"schema_version": 1},
        }

    def test_resume_contract_rejects_behavior_changes(self) -> None:
        cases = (
            ("seed", 99),
            ("warmup_steps", 3),
            ("max_grad_norm", 0.5),
            ("ema_enabled", True),
            ("activation_compression", True),
            ("curriculum", {"profiles": [{"start_step": 0}]}),
        )
        for field, changed in cases:
            with self.subTest(field=field):
                checkpoint_config = TrainingConfig()
                current = TrainingConfig()
                setattr(current.training, field, changed)
                with self.assertRaisesRegex(ValueError, "training"):
                    validate_resume_checkpoint_compatibility(
                        payload=self._resume_payload(checkpoint_config),
                        config=current,
                        resume_path="checkpoint.pt",
                        dataset_snapshot_id="dataset",
                        cache_snapshot_id="cache",
                        model_snapshot_id="model",
                    )

    def test_resume_contract_binds_nonconstant_scheduler_horizon(self) -> None:
        checkpoint_config = TrainingConfig()
        checkpoint_config.optimizer.scheduler = "cosine"
        current = TrainingConfig()
        current.optimizer.scheduler = "cosine"
        current.training.max_steps += 1
        with self.assertRaisesRegex(ValueError, "training"):
            validate_resume_checkpoint_compatibility(
                payload=self._resume_payload(checkpoint_config),
                config=current,
                resume_path="checkpoint.pt",
                dataset_snapshot_id="dataset",
                cache_snapshot_id="cache",
                model_snapshot_id="model",
            )

    def test_resume_rejects_optimizer_implementation_change(self) -> None:
        import torch

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        result = SimpleNamespace(
            optimizer=torch.optim.AdamW([parameter]),
            resolved_type="adamw",
            used_fallback=False,
        )
        payload = self._resume_payload(TrainingConfig())
        payload["optimizer_identity"] = {
            **optimizer_implementation_identity(result),
            "resolved_type": "came",
        }
        with self.assertRaisesRegex(ValueError, "optimizer implementation mismatch"):
            validate_resume_checkpoint_compatibility(
                payload=payload,
                config=TrainingConfig(),
                resume_path="checkpoint.pt",
                dataset_snapshot_id="dataset",
                cache_snapshot_id="cache",
                model_snapshot_id="model",
                optimizer_result=result,
            )

    def test_non_finite_loss_is_returned_to_the_configured_skip_policy(self) -> None:
        import torch

        class Trainer:
            @staticmethod
            def compute_loss_result(_batch):
                scalar = torch.tensor(float("nan"), requires_grad=True)
                vector = scalar.reshape(1)
                return TrainingLossResult(
                    loss=scalar,
                    loss_pre_accum=scalar,
                    per_sample_loss=vector,
                    per_sample_loss_normalized=vector,
                    loss_weights=torch.ones_like(vector),
                    weighted_loss=vector,
                    timesteps=torch.zeros_like(vector),
                )

        result = accumulate_training_gradients(
            trainer=Trainer(),
            grad_accum=1,
            build_batch=lambda _index: {},
            params=[],
            gradient_cpu_offload=False,
        )

        self.assertTrue(result.non_finite_loss)
        self.assertTrue(result.last_metrics["non_finite_loss"])

    def test_fatal_step_error_reaches_stderr_before_the_exit_code(self) -> None:
        """An OOM must leave its stack behind, not only a guidance message.

        ``SystemExit`` carries no frames, so without an explicit report the only
        record of a failing step is its exit status. The raising frame, the
        exception chain, and the unchanged exit code are all checked together.
        """
        import contextlib
        import io

        def _raise_inside_a_named_frame(_batch):
            raise RuntimeError("CUDA out of memory. Tried to allocate 40.90 GiB")

        class Trainer:
            @staticmethod
            def compute_loss_result(batch):
                return _raise_inside_a_named_frame(batch)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                accumulate_training_gradients(
                    trainer=Trainer(),
                    grad_accum=1,
                    build_batch=lambda _index: {},
                    params=[],
                    gradient_cpu_offload=False,
                    oom_error_predicate=lambda exc: "out of memory" in str(exc),
                )

        # SystemExit(str) keeps the interpreter's exit status at 1 for automation.
        self.assertIsInstance(raised.exception.code, str)
        self.assertIn("CUDA out of memory detected.", str(raised.exception.code))

        report = stderr.getvalue()
        self.assertIn("Traceback (most recent call last)", report)
        self.assertIn("_raise_inside_a_named_frame", report)
        self.assertIn("RuntimeError: CUDA out of memory", report)
        self.assertIn(__file__.rsplit("\\", 1)[-1].rsplit("/", 1)[-1], report)
        # The chained cause survives too: SystemExit is raised ``from`` the OOM.
        self.assertIs(raised.exception.__cause__.__class__, RuntimeError)

    def test_non_oom_step_error_propagates_with_its_own_traceback(self) -> None:
        """Only the OOM path is converted; anything else keeps its exception."""

        class Trainer:
            @staticmethod
            def compute_loss_result(_batch):
                raise RuntimeError("unrelated kernel failure")

        import traceback as traceback_module

        try:
            accumulate_training_gradients(
                trainer=Trainer(),
                grad_accum=1,
                build_batch=lambda _index: {},
                params=[],
                gradient_cpu_offload=False,
                oom_error_predicate=lambda exc: "out of memory" in str(exc),
            )
        except RuntimeError as exc:
            self.assertIn("unrelated kernel failure", str(exc))
            frames = traceback_module.format_exception(
                type(exc), exc, exc.__traceback__
            )
            self.assertIn("compute_loss_result", "".join(frames))
        else:
            self.fail("a non-OOM RuntimeError must propagate")

    def test_live_runtime_overrides_round_trip_through_checkpoint_state(self) -> None:
        source = build_training_runtime_overrides()
        source.sample_every_n_steps_override = 7
        source.val_every_n_steps_override = 11
        restored = build_training_runtime_overrides()

        load_training_runtime_overrides_state(
            restored,
            training_runtime_overrides_state_dict(source),
        )

        self.assertEqual(restored.sample_every_n_steps_override, 7)
        self.assertEqual(restored.val_every_n_steps_override, 11)

    def test_fresh_run_torch_seed_is_deterministic(self) -> None:
        import torch

        seed_torch_rng(731)
        first = torch.rand(4)
        seed_torch_rng(731)
        torch.testing.assert_close(torch.rand(4), first, rtol=0.0, atol=0.0)

    def test_non_finite_skip_discards_pipeline_owned_step_state(self) -> None:
        import torch

        class Pipeline:
            discarded = False

            def discard_optimizer_step(self):
                self.discarded = True

            @staticmethod
            def supports_runtime_offload_flush():
                return False

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        parameter.grad = torch.tensor(float("inf"))
        pipeline = Pipeline()
        result = apply_optimizer_update(
            params=[parameter],
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0),
            pipeline=pipeline,
            max_grad_norm=1.0,
            optimizer_cpu_offload=False,
            non_finite_gradients=True,
            non_finite_grad_policy="skip_step",
            consecutive_skipped_steps=0,
            max_consecutive_skipped_steps=5,
            skipped_steps=0,
            global_step=0,
        )
        self.assertFalse(result.advanced)
        self.assertTrue(pipeline.discarded)
        self.assertIsNone(parameter.grad)

    def test_curriculum_switches_profiles_and_filters_records(self) -> None:
        schedule = CurriculumSchedule.from_config(
            {
                "enabled": True,
                "resolution_schedule": {"0": "512x512", "100": "768x768"},
                "frame_schedule": {"0": 16, "200": 33},
            }
        )
        self.assertEqual(schedule.profile_for_step(0).resolution, "512x512")
        self.assertEqual(schedule.profile_for_step(150).resolution, "768x768")
        self.assertEqual(schedule.profile_for_step(150).frame_count, 16)
        self.assertEqual(schedule.profile_for_step(250).frame_count, 33)

        records = [
            {"sample_id": "a", "bucket_resolution": "768x768", "frame_count": 16},
            {"sample_id": "b", "bucket_resolution": "512x512", "frame_count": 16},
        ]
        self.assertEqual(
            [record["sample_id"] for record in schedule.filter_records(records, step=150)],
            ["a"],
        )

    def test_curriculum_stage_matching_no_records_fails_explicitly(self) -> None:
        """An unsatisfiable stage must not degrade into the unfiltered set."""
        schedule = CurriculumSchedule.from_config(
            {
                "enabled": True,
                "resolution_schedule": {"0": "1024x1024"},
            }
        )
        records = [
            {"sample_id": "a", "bucket_resolution": "512x512", "frame_count": 16},
        ]
        with self.assertRaises(ValueError) as ctx:
            schedule.filter_records(records, step=0)
        self.assertIn("1024x1024", str(ctx.exception))

    def test_early_stop_state_round_trips_through_checkpoint(self) -> None:
        state = EarlyStopState(
            best_val_loss=0.123,
            best_step=42,
            patience_counter=3,
            best_checkpoint_path="outputs/checkpoints/step_42.pt",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, {"early_stop_state": state.to_dict()})
            restored = EarlyStopState.from_dict(
                load_checkpoint(path)["early_stop_state"]
            )
        self.assertEqual(restored.best_step, 42)
        self.assertEqual(restored.patience_counter, 3)
        self.assertEqual(restored.best_checkpoint_path, "outputs/checkpoints/step_42.pt")
        self.assertAlmostEqual(restored.best_val_loss, 0.123, places=12)

    def test_best_checkpoint_contains_the_new_early_stop_state(self) -> None:
        initial = EarlyStopState(
            best_val_loss=float("inf"),
            best_step=-1,
            patience_counter=0,
            best_checkpoint_path="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = apply_validation_policy(
                step_metrics={},
                val_loss=0.25,
                early_stop_state=initial,
                global_step=7,
                early_stop_patience=3,
                log_on_this_rank=True,
                ckpt_dir=tmp,
                build_ckpt_payload=lambda step: {
                    "global_step": step,
                    "early_stop_state": initial.to_dict(),
                },
            )
            payload = load_checkpoint(Path(tmp) / "best.pt")

        restored = EarlyStopState.from_dict(payload["early_stop_state"])
        self.assertTrue(result.best_checkpoint_saved)
        self.assertEqual(restored.best_step, 7)
        self.assertEqual(restored.patience_counter, 0)
        self.assertEqual(restored.best_checkpoint_path, str(Path(tmp) / "best.pt"))
        self.assertAlmostEqual(restored.best_val_loss, 0.25, places=12)

    def test_non_finite_skip_policy_has_a_stall_guard(self) -> None:
        consecutive = 0
        max_consecutive = 3
        for _ in range(max_consecutive):
            action, consecutive = resolve_non_finite_policy(
                policy="skip_step",
                consecutive_skips=consecutive,
                max_consecutive_skipped_steps=max_consecutive,
            )
            self.assertEqual(action, "skip_step")
        with self.assertRaises(RuntimeError):
            resolve_non_finite_policy(
                policy="skip_step",
                consecutive_skips=consecutive,
                max_consecutive_skipped_steps=max_consecutive,
            )

    def test_sqlite_live_control_applies_only_at_accumulation_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "control.sqlite"
            request = build_live_training_control_request_payload(
                job_id="job",
                run_id="run",
                seq=1,
                command="set_sample_interval",
                arguments={"every_n_steps": 7},
                updated_at=1.0,
            )
            _set_control_plane_state_row(
                db_path=db_path,
                namespace=TRAINING_CONTROL_REQUESTS_NAMESPACE,
                state_key="job",
                payload=request,
                updated_at=1.0,
            )
            controller = LiveTrainingController(
                db_path=db_path,
                job_id="job",
                run_id="run",
                event_bus=TrainingEventBus(run_id="run", job_id="job"),
                callbacks=[],
                gradient_accumulation=2,
            )
            accepted = controller.poll_pending_request(global_step=0)
            self.assertEqual(accepted["status"], "accepted")
            self.assertIsNone(
                controller.on_microstep(
                    microstep_index=0,
                    optimizer_steps_committed=0,
                    global_step=0,
                )
            )
            applied = controller.on_microstep(
                microstep_index=1,
                optimizer_steps_committed=1,
                global_step=1,
            )
            self.assertIsNotNone(applied)
            controller.mark_request_applied_with_result(
                applied_control=applied,
                global_step=1,
                result={"effective_sample_interval_every_n_steps": 7},
            )
            status = load_live_training_control_status_payload(
                db_path=db_path,
                job_id="job",
            )
            self.assertEqual(status["state"], "applied")
            self.assertEqual(status["result"]["effective_sample_interval_every_n_steps"], 7)

            restarted = LiveTrainingController(
                db_path=db_path,
                job_id="job",
                run_id="run-restarted",
                event_bus=TrainingEventBus(run_id="run-restarted", job_id="job"),
                callbacks=[],
                gradient_accumulation=2,
            )
            self.assertIsNone(restarted.poll_pending_request(global_step=1))
            self.assertEqual(restarted.highest_seen_seq, 1)

    def test_lr_finder_selects_the_steepest_log_space_loss_drop(self) -> None:
        points = [
            (1e-6, 3.0),
            (1e-5, 2.8),
            (1e-4, 1.2),
            (1e-3, 1.1),
        ]
        self.assertEqual(_compute_recommended_lr(points), 1e-4)


if __name__ == "__main__":
    unittest.main()
