from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mirai.core.training.release.release_evidence_contract import (
    build_release_evidence_bundle,
    build_release_verification_report,
    checkpoint_from_train_summary,
    validate_native_smoke_report,
    validate_release_evidence_bundle,
    validate_train_summary,
)
from mirai.core.training.release.operational_certification import (
    _behavior_state,
    _flatten_state_tree,
)


class ReleaseEvidenceContractTests(unittest.TestCase):
    def test_resume_certification_inventory_includes_non_pipeline_state(self) -> None:
        flattened = _flatten_state_tree(
            _behavior_state(
                {
                    "global_step": 7,
                    "trainer_state": {"pipeline": {"weight": 1}},
                    "optimizer_state": {"state": {0: {"step": 7}}},
                    "scheduler_state": {"last_epoch": 7},
                    "rng_state": (3, (1, 2), None),
                    "runtime_overrides": {"val_every_n_steps_override": 4},
                }
            )
        )
        self.assertEqual(flattened["global_step"], 7)
        self.assertEqual(flattened["optimizer_state.state.0.step"], 7)
        self.assertEqual(flattened["scheduler_state.last_epoch"], 7)
        self.assertEqual(flattened["runtime_overrides.val_every_n_steps_override"], 4)

    @staticmethod
    def _native_report(*, model_type: str) -> dict[str, object]:
        return {
            "status": "ok",
            "checkpoint_path": __file__,
            "eval_path": __file__,
            "infer_path": __file__,
            "runtime_policy": {
                "model_type": model_type,
                "strict_native_assets": True,
                "denoiser_subfolder": "transformer",
            },
            "steps": [{"returncode": 0}],
        }

    def test_verification_report_declares_only_single_gpu_evidence(self) -> None:
        report = build_release_verification_report(
            status="passed",
            train_summary="train.json",
            eval_report="eval.json",
            resume_summary="resume.json",
            native_smoke_report="native.json",
            failures=[],
        )
        self.assertEqual(report["status"], "passed")
        self.assertNotIn("distributed_smoke_report", report)

    def test_train_summary_requires_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "last.pt"
            adapter = root / "adapter.pt"
            checkpoint.touch()
            adapter.touch()
            failures = validate_train_summary(
                {
                    "status": "completed",
                    "cache_data_access_mode": "indexed_safetensors",
                    "indexed_cache_enabled": True,
                    "training_sampling_mode": "deterministic_epoch_shuffle",
                    "last_checkpoint": str(checkpoint),
                    "adapter_path": str(adapter),
                },
                label="train_summary",
            )
        self.assertEqual(failures, [])

    def test_builder_resumes_from_checkpoint_reported_by_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoints" / "last.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            resolved = checkpoint_from_train_summary(
                {"last_checkpoint": "checkpoints/last.pt"},
                repo_root=root,
                label="partial train",
            )
            self.assertEqual(resolved, checkpoint.resolve())

            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                checkpoint_from_train_summary(
                    {"last_checkpoint": "checkpoints/step_1.pt"},
                    repo_root=root,
                    label="partial train",
                )

    def test_bundle_rejects_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = build_release_evidence_bundle(
                status="failed",
                config="config.toml",
                evidence_root=tmp,
                cache_summary="missing-cache.json",
                train_summary="missing-train.json",
                eval_report="missing-eval.json",
                partial_summary="missing-partial.json",
                resume_summary="missing-resume.json",
                resume_certification_report="missing-resume-cert.json",
                export_certification_report="missing-export-cert.json",
                operational_matrix_report="missing-matrix.json",
                promotion_gate_report="missing-promotion.json",
                native_smoke_report="missing-native.json",
                verify_release_report="missing-verification.json",
                split_step=1,
                commands=[{"command": ["train"], "returncode": 0}],
            )
            failures = validate_release_evidence_bundle(bundle)
        self.assertTrue(any("does not exist" in failure for failure in failures))

    def test_lingbot_native_smoke_requires_matching_verified_snapshot(self) -> None:
        report = self._native_report(model_type="lingbot-video")
        self.assertTrue(
            any(
                "snapshot_verification" in failure
                for failure in validate_native_smoke_report(report)
            )
        )

        report["snapshot_verification"] = {
            "status": "verified",
            "manifest_path": __file__,
            "file_count": 1,
            "total_bytes": 1,
            "denoiser_subfolder": "transformer",
            "model_component_id": "denoiser_subfolder:transformer",
        }
        self.assertEqual(validate_native_smoke_report(report), [])

        report["runtime_policy"]["denoiser_subfolder"] = "refiner"  # type: ignore[index]
        self.assertTrue(
            any(
                "denoiser_subfolder" in failure
                for failure in validate_native_smoke_report(report)
            )
        )

    def test_validation_fixture_native_smoke_is_not_release_supported(self) -> None:
        report = self._native_report(model_type="sparse_moe_test")
        failures = validate_native_smoke_report(report)
        self.assertTrue(failures)
        self.assertTrue(any("model_type" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
