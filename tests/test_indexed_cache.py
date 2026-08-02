from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mirai.core.dataset.cache import (
    build_cache,
    indexed_cache_metadata_path,
    indexed_cache_tensor_path,
    load_cache,
    load_indexed_cache_records,
)
from mirai.core.dataset.record_batch import (
    latent_value_to_tensor,
    text_embed_value_to_tensor,
)
from mirai.core.training.data.loader import load_prepared_training_data

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class IndexedCacheTests(unittest.TestCase):
    def test_build_cache_writes_indexed_sidecar_and_round_trips_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor([0.1], dtype=torch.float32), data_dir / "s0.pt")
            (data_dir / "s0.txt").write_text("a", encoding="utf-8")
            torch.save(torch.tensor([0.2], dtype=torch.float32), data_dir / "s1.pt")
            (data_dir / "s1.txt").write_text("bb", encoding="utf-8")

            payload = build_cache(data_dir, cache_path)
            loaded = load_cache(cache_path)
            indexed = load_indexed_cache_records(cache_path)

            self.assertTrue(bool(payload["indexed_cache"]["enabled"]))
            self.assertTrue(str(payload["dataset_snapshot_id"]).startswith("tree-sha256:"))
            self.assertTrue(indexed_cache_metadata_path(cache_path).exists())
            self.assertTrue(indexed_cache_tensor_path(cache_path).exists())
            self.assertIsNotNone(indexed)
            assert indexed is not None
            self.assertEqual(len(indexed), len(loaded["records"]))
            for indexed_record, loaded_record in zip(indexed, loaded["records"], strict=True):
                self.assertEqual(indexed_record["sample_id"], loaded_record["sample_id"])
                self.assertTrue(
                    torch.allclose(
                        indexed_record["latent"],
                        latent_value_to_tensor(loaded_record["latent"]),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        indexed_record["text_embed"],
                        text_embed_value_to_tensor(loaded_record["text_embed"]),
                    )
                )

    def test_load_prepared_training_data_prefers_indexed_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor([0.1], dtype=torch.float32), data_dir / "train0.pt")
            (data_dir / "train0.txt").write_text("train", encoding="utf-8")
            torch.save(torch.tensor([0.2], dtype=torch.float32), data_dir / "val0.pt")
            (data_dir / "val0.txt").write_text("val", encoding="utf-8")
            registration = {
                "samples": [
                    {"sample_id": "train0", "split": "train"},
                    {"sample_id": "val0", "split": "val"},
                ]
            }
            (data_dir / "registration.json").write_text(
                json.dumps(registration, indent=2) + "\n",
                encoding="utf-8",
            )

            build_cache(data_dir, cache_path)

            with mock.patch("mirai.core.training.data.loader.load_cache", side_effect=AssertionError("full cache should not be loaded")):
                prepared = load_prepared_training_data(
                    cache_path=cache_path,
                    model_type="sparse_moe_test",
                    strategy_type="text_to_video",
                    val_every_n_steps=1,
                )

            self.assertEqual([record["sample_id"] for record in prepared.train_records], ["train0"])
            self.assertEqual([record["sample_id"] for record in prepared.val_records], ["val0"])
            self.assertEqual(prepared.cache_data_access_mode, "indexed_safetensors")
            self.assertTrue(bool(prepared.indexed_cache_enabled))
            self.assertTrue(torch.is_tensor(prepared.train_records[0]["latent"]))

    def test_load_prepared_training_data_rejects_dataset_lineage_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor([0.1], dtype=torch.float32), data_dir / "train0.pt")
            (data_dir / "train0.txt").write_text("train", encoding="utf-8")

            payload = build_cache(data_dir, cache_path)

            with self.assertRaisesRegex(ValueError, "Cache dataset lineage mismatch"):
                load_prepared_training_data(
                    cache_path=cache_path,
                    model_type="sparse_moe_test",
                    strategy_type="text_to_video",
                    val_every_n_steps=0,
                    dataset_snapshot_id=str(payload["dataset_snapshot_id"]) + "::different",
                )


if __name__ == "__main__":
    unittest.main()
