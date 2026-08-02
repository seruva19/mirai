from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mirai.core.persistence.checkpoints import load_checkpoint, save_checkpoint
from mirai.core.training.data.curriculum import CurriculumSchedule
from mirai.core.training.evaluation.early_stop import EarlyStopState
from mirai.core.training.optim.gradients import resolve_non_finite_policy
from mirai.core.training.control.live_control import (
    TRAINING_CONTROL_REQUESTS_NAMESPACE,
    LiveTrainingController,
    _set_control_plane_state_row,
    load_live_training_control_status_payload,
)
from mirai.core.training.control.live_control_schema import (
    build_live_training_control_request_payload,
)
from mirai.core.training.observability.events import TrainingEventBus
from scripts.tools.lr_find import _compute_recommended_lr


class TrainingRuntimePolicyTests(unittest.TestCase):
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
