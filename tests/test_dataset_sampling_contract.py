from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from mirai.core.dataset.cache import build_cache
from mirai.core.dataset.multi_source import choose_weighted_source
from mirai.core.dataset.media.video import select_evenly_spaced_indices_from_pts
from mirai.core.training.data.online import (
    build_temporal_groups,
    online_caption_embed,
    pick_temporal_variant,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class DatasetSelectionTests(unittest.TestCase):
    def test_vfr_selection_uses_timestamps(self) -> None:
        pts = [0.0, 33.0, 67.0, 100.0, 150.0, 210.0, 300.0]
        indices = select_evenly_spaced_indices_from_pts(pts, frame_count=4)
        self.assertEqual(indices, [0, 3, 5, 6])
        self.assertEqual([pts[index] for index in indices], [0.0, 100.0, 210.0, 300.0])

    def test_source_sampling_tracks_declared_weights(self) -> None:
        rng = random.Random(123)
        weights = {"source_a": 1.0, "source_b": 3.0}
        counts = {"source_a": 0, "source_b": 0}
        for _ in range(1000):
            counts[choose_weighted_source(weights, rng)] += 1
        ratio = counts["source_b"] / max(counts["source_a"], 1)
        self.assertGreater(ratio, 2.5)
        self.assertLess(ratio, 3.5)

    def test_online_caption_and_temporal_selection_are_deterministic(self) -> None:
        record = {
            "sample_id": "s0",
            "caption": "trigger, red, blue, green",
            "text_embed": 0.1234,
        }
        first_caption = online_caption_embed(
            record=record,
            step=0,
            seed=42,
            enabled=True,
            dropout_rate=0.5,
            keep_first_n_tags=1,
        )
        next_caption = online_caption_embed(
            record=record,
            step=1,
            seed=42,
            enabled=True,
            dropout_rate=0.5,
            keep_first_n_tags=1,
        )
        self.assertNotEqual(first_caption, next_caption)

        records = [
            {"sample_id": "s0::clip0", "base_sample_id": "s0", "clip_index": 0},
            {"sample_id": "s0::clip1", "base_sample_id": "s0", "clip_index": 1},
            {"sample_id": "s1::clip0", "base_sample_id": "s1", "clip_index": 0},
        ]
        base_ids, groups = build_temporal_groups(records)
        self.assertEqual(base_ids, ["s0", "s1"])
        first = pick_temporal_variant(groups["s0"], epoch=3, seed=7, sample_position=0)
        repeated = pick_temporal_variant(groups["s0"], epoch=3, seed=7, sample_position=0)
        self.assertEqual(first["sample_id"], repeated["sample_id"])


@unittest.skipIf(torch is None, "torch not installed")
class DatasetCacheModeTests(unittest.TestCase):
    def test_caption_variants_remain_distinct_cache_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            data_dir.mkdir(parents=True)
            torch.save(torch.tensor([0.2], dtype=torch.float32), data_dir / "sample0.pt")
            (data_dir / "sample0.json").write_text(
                json.dumps({"captions": ["short", "long detailed caption"]}),
                encoding="utf-8",
            )

            payload = build_cache(data_dir, cache_path)
            self.assertEqual(int(payload["num_records"]), 2)
            self.assertEqual(
                sorted(record["sample_id"] for record in payload["records"]),
                ["sample0::cap0", "sample0::cap1"],
            )
            self.assertEqual(
                {int(record["caption_variant_index"]) for record in payload["records"]},
                {0, 1},
            )
            self.assertEqual(
                len({float(record["text_embed"]) for record in payload["records"]}),
                2,
            )

    def test_ram_cache_preserves_clip_and_encoder_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            data_dir.mkdir(parents=True)
            torch.save(torch.tensor([0.1], dtype=torch.float32), data_dir / "s0.pt")
            (data_dir / "s0.txt").write_text("caption", encoding="utf-8")
            payload = build_cache(
                data_dir,
                cache_path,
                cache_mode="ram",
                clips_per_video=2,
                fp8_text_encoder=True,
            )
            self.assertFalse(cache_path.exists())
            self.assertEqual(payload["cache_mode"], "ram")
            self.assertTrue(payload["fp8_text_encoder"])
            self.assertEqual(int(payload["num_records"]), 2)

    def test_empty_dataset_rejects_synthetic_runtime_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            data_dir.mkdir(parents=True)
            with self.assertRaises(ValueError):
                build_cache(data_dir, cache_path)


if __name__ == "__main__":
    unittest.main()
