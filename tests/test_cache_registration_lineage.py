from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from mirai.core.dataset.cache import build_cache
from mirai.core.dataset.registration import register_dataset

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class CacheRegistrationLineageTests(unittest.TestCase):
    def test_cache_preserves_configured_routing_domain_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            cache_path = Path(tmp) / "cache.pt"
            registration_path = data_dir / "registration.json"
            data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor([0.1]), data_dir / "sample.pt")
            (data_dir / "sample.txt").write_text("caption", encoding="utf-8")
            registration = register_dataset(
                dataset_path=data_dir,
                output_path=registration_path,
                split_seed=1,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
                compliance_enabled=False,
                usage_mode="internal",
            )
            registration["samples"][0]["visual_domain"] = "animation"
            registration_path.write_text(
                json.dumps(registration), encoding="utf-8"
            )

            payload = build_cache(
                data_dir,
                cache_path,
                routing_domain_metadata_key="visual_domain",
            )

            self.assertEqual(
                payload["records"][0]["metadata"]["visual_domain"],
                "animation",
            )

    def test_cache_records_inherit_registered_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            registration_path = data_dir / "registration.json"
            data_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(6):
                torch.save(torch.tensor([0.1 + (idx * 0.1)], dtype=torch.float32), data_dir / f"s{idx}.pt")
                (data_dir / f"s{idx}.txt").write_text(f"caption {idx}", encoding="utf-8")

            registration = register_dataset(
                dataset_path=data_dir,
                output_path=registration_path,
                split_seed=123,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                compliance_enabled=False,
                usage_mode="internal",
            )
            expected_splits = {
                str(sample["sample_id"]): str(sample["split"])
                for sample in registration["samples"]
            }

            payload = build_cache(data_dir, cache_path)

            self.assertTrue(bool(payload["dataset_registration_present"]))
            self.assertEqual(str(payload["dataset_registration_path"]), str(registration_path))
            self.assertTrue(str(payload["dataset_snapshot_id"]).startswith("registration.json:sha256:"))
            self.assertEqual(str(payload["dataset_snapshot_source_kind"]), "manifest")
            self.assertEqual(
                str(payload["dataset_snapshot_source_manifest"]),
                registration_path.resolve().as_posix(),
            )
            observed_splits = {
                str(record["sample_id"]): str(record["split"])
                for record in payload["records"]
            }
            self.assertEqual(observed_splits, expected_splits)

    def test_cache_rejects_stale_registration_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            registration_path = data_dir / "registration.json"
            data_dir.mkdir(parents=True, exist_ok=True)

            torch.save(torch.tensor([0.1], dtype=torch.float32), data_dir / "s0.pt")
            (data_dir / "s0.txt").write_text("caption 0", encoding="utf-8")
            register_dataset(
                dataset_path=data_dir,
                output_path=registration_path,
                split_seed=123,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                compliance_enabled=False,
                usage_mode="internal",
            )

            torch.save(torch.tensor([0.2], dtype=torch.float32), data_dir / "s1.pt")
            (data_dir / "s1.txt").write_text("caption 1", encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                build_cache(data_dir, cache_path)
            self.assertIn("Dataset registration does not match current dataset contents", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
