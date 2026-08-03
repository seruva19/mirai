from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mirai.core.moe.artifacts.manifest import write_download_manifest
from mirai.core.moe.artifacts.verification import verify_downloaded_snapshot
from scripts.download import _snapshot_file_inventory


class SnapshotVerificationTests(unittest.TestCase):
    def test_download_inventory_excludes_local_cache_and_prior_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transformer").mkdir()
            (root / "transformer" / "config.json").write_bytes(b"abc")
            (root / ".cache" / "huggingface").mkdir(parents=True)
            (root / ".cache" / "huggingface" / "metadata").write_bytes(b"local")
            (root / "download_manifest.json").write_text("{}", encoding="utf-8")

            self.assertEqual(
                _snapshot_file_inventory(root),
                [
                    {
                        "path": "transformer/config.json",
                        "size": 3,
                        "status": "downloaded",
                    }
                ],
            )

    def test_verifies_complete_lingbot_snapshot_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transformer").mkdir()
            (root / "transformer" / "config.json").write_bytes(b"abc")
            (root / "transformer" / "model.safetensors").write_bytes(b"weights")
            write_download_manifest(
                root,
                {
                    "schema_version": 1,
                    "status": "downloaded",
                    "variant": "lingbot-video-moe-30b-a3b",
                    "repo_id": "robbyant/lingbot-video-moe-30b-a3b",
                    "files": [
                        {"path": "transformer/config.json", "size": 3, "status": "downloaded"},
                        {"path": "transformer/model.safetensors", "size": 7, "status": "downloaded"},
                    ],
                },
            )

            report = verify_downloaded_snapshot(
                root,
                expected_variant="lingbot-video-moe-30b-a3b",
            )

            self.assertEqual(report.variant, "lingbot-video-moe-30b-a3b")
            self.assertEqual(report.repo_id, "robbyant/lingbot-video-moe-30b-a3b")
            self.assertEqual(report.file_count, 2)
            self.assertEqual(report.total_bytes, 10)

    def test_rejects_leftover_part_files_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "refiner").mkdir()
            (root / "refiner" / "shard.safetensors.part").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "incomplete partial files"):
                verify_downloaded_snapshot(root)

    def test_rejects_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(b"short")
            write_download_manifest(
                root,
                {
                    "status": "downloaded",
                    "variant": "lingbot-video-moe-30b-a3b",
                    "repo_id": "robbyant/lingbot-video-moe-30b-a3b",
                    "files": [{"path": "model.safetensors", "size": 10}],
                },
            )
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_downloaded_snapshot(root)

    def test_rejects_unsafe_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_download_manifest(
                root,
                {
                    "status": "downloaded",
                    "variant": "lingbot-video-moe-30b-a3b",
                    "repo_id": "robbyant/lingbot-video-moe-30b-a3b",
                    "files": [{"path": "../escape.safetensors", "size": 1}],
                },
            )
            with self.assertRaisesRegex(ValueError, "unsafe file path"):
                verify_downloaded_snapshot(root)

    def test_cli_outputs_verified_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(b"weights")
            write_download_manifest(
                root,
                {
                    "status": "downloaded",
                    "variant": "lingbot-video-moe-30b-a3b",
                    "repo_id": "robbyant/lingbot-video-moe-30b-a3b",
                    "files": [{"path": "model.safetensors", "size": 7}],
                },
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/verify_model_snapshot.py",
                    "--model-dir",
                    str(root),
                    "--variant",
                    "lingbot-video-moe-30b-a3b",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "verified")
            self.assertEqual(payload["variant"], "lingbot-video-moe-30b-a3b")


if __name__ == "__main__":
    unittest.main()
