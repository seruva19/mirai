"""Behavioral contract for aspect-ratio, resolution, and frame bucketing."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from mirai.core.dataset.bucketing.bucket_resolve import (
    bucket_id,
    choose_frame_bucket,
    choose_resolution_bucket,
    parse_resolution_buckets,
    validate_frame_buckets,
)

try:
    from mirai.core.dataset.media.media_resize import resize_crop_tensor, select_frames
except RuntimeError:  # pragma: no cover  # torch unavailable
    resize_crop_tensor = None  # type: ignore[assignment]
    select_frames = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# bucket_resolve helpers
# ---------------------------------------------------------------------------


class TestChooseResolutionBucket(unittest.TestCase):
    def setUp(self) -> None:
        self.buckets = [(512, 512), (768, 512), (512, 768), (720, 1280), (1280, 720)]

    def test_square_source_picks_square_bucket(self) -> None:
        bh, bw = choose_resolution_bucket(480, 480, self.buckets)
        self.assertEqual((bh, bw), (512, 512))

    def test_landscape_source_picks_landscape_bucket(self) -> None:
        bh, bw = choose_resolution_bucket(360, 640, self.buckets)
        # 360/640 ≈ 0.5625;  512/768 ≈ 0.667;  720/1280 = 0.5625 — exact match
        self.assertEqual((bh, bw), (720, 1280))

    def test_portrait_source_picks_portrait_bucket(self) -> None:
        bh, bw = choose_resolution_bucket(640, 360, self.buckets)
        self.assertEqual((bh, bw), (1280, 720))

    def test_near_3_4_picks_768x512(self) -> None:
        # 768/512 = 1.5; source 600x400 = 1.5 — exact AR match
        bh, bw = choose_resolution_bucket(600, 400, self.buckets)
        self.assertEqual((bh, bw), (768, 512))

    def test_raises_on_empty_buckets(self) -> None:
        with self.assertRaises(ValueError):
            choose_resolution_bucket(480, 640, [])


class TestChooseFrameBucket(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertEqual(choose_frame_bucket(33, [17, 33, 49]), 33)

    def test_picks_largest_leq(self) -> None:
        self.assertEqual(choose_frame_bucket(40, [17, 33, 49]), 33)

    def test_clamps_to_min_when_below_all(self) -> None:
        self.assertEqual(choose_frame_bucket(5, [17, 33, 49]), 17)

    def test_single_bucket(self) -> None:
        self.assertEqual(choose_frame_bucket(100, [33]), 33)

    def test_raises_on_empty(self) -> None:
        with self.assertRaises(ValueError):
            choose_frame_bucket(33, [])


class TestBucketId(unittest.TestCase):
    def test_format(self) -> None:
        self.assertEqual(bucket_id(512, 768, 33), "512x768x33")

    def test_distinct(self) -> None:
        ids = {bucket_id(h, w, f) for h, w, f in [(512, 512, 33), (768, 512, 33), (512, 768, 17)]}
        self.assertEqual(len(ids), 3)


class TestParseResolutionBuckets(unittest.TestCase):
    def _make_cfg(self, **overrides):
        from dataclasses import dataclass, field

        @dataclass
        class FakeCfg:
            bucket_resolutions: list = field(default_factory=list)
            bucket_base_resolution: int = 512
            bucket_round_to: int = 16

        return FakeCfg(**overrides)

    def test_parses_explicit(self) -> None:
        cfg = self._make_cfg(bucket_resolutions=["512x512", "768x512", "512x768"])
        buckets = parse_resolution_buckets(cfg)
        self.assertIn((512, 512), buckets)
        self.assertIn((768, 512), buckets)
        self.assertIn((512, 768), buckets)

    def test_derives_default_when_empty(self) -> None:
        cfg = self._make_cfg(bucket_resolutions=[])
        buckets = parse_resolution_buckets(cfg)
        self.assertTrue(len(buckets) > 0)
        # All must be divisible by bucket_round_to
        for bh, bw in buckets:
            self.assertEqual(bh % 16, 0, f"H={bh} not divisible by 16")
            self.assertEqual(bw % 16, 0, f"W={bw} not divisible by 16")

    def test_raises_on_bad_format(self) -> None:
        cfg = self._make_cfg(bucket_resolutions=["512-512"])
        with self.assertRaises(ValueError):
            parse_resolution_buckets(cfg)

    def test_raises_on_non_divisible(self) -> None:
        cfg = self._make_cfg(bucket_resolutions=["513x512"], bucket_round_to=16)
        with self.assertRaises(ValueError):
            parse_resolution_buckets(cfg)


class TestValidateFrameBuckets(unittest.TestCase):
    def test_valid_4k_plus_1_no_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_frame_buckets([1, 5, 9, 17, 33, 49])
        self.assertEqual(len(w), 0)

    def test_rejects_non_positive_buckets(self) -> None:
        with self.assertRaises(ValueError):
            validate_frame_buckets([0, 33])

    def test_accepts_model_agnostic_positive_buckets(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_frame_buckets([10, 33])
        self.assertEqual(len(w), 0)


# ---------------------------------------------------------------------------
# media_resize helpers
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None or resize_crop_tensor is None, "torch not installed")
class TestResizeCropTensor(unittest.TestCase):
    def _make_frames(self, c: int, t: int, h: int, w: int) -> "torch.Tensor":
        return torch.rand(c, t, h, w)

    def test_resize_crop_returns_exact_target(self) -> None:
        frames = self._make_frames(3, 5, 480, 640)
        out = resize_crop_tensor(frames, 256, 256, mode="resize_crop")
        self.assertEqual(out.shape, (3, 5, 256, 256))

    def test_resize_mode_returns_exact_target(self) -> None:
        frames = self._make_frames(3, 5, 480, 640)
        out = resize_crop_tensor(frames, 256, 384, mode="resize")
        self.assertEqual(out.shape, (3, 5, 256, 384))

    def test_pad_mode_returns_exact_target(self) -> None:
        frames = self._make_frames(3, 5, 100, 100)
        out = resize_crop_tensor(frames, 256, 256, mode="pad")
        self.assertEqual(out.shape, (3, 5, 256, 256))

    def test_crop_mode_returns_exact_target(self) -> None:
        frames = self._make_frames(3, 5, 512, 512)
        out = resize_crop_tensor(frames, 256, 256, mode="crop")
        self.assertEqual(out.shape, (3, 5, 256, 256))

    def test_divisibility_holds_for_bucket_round_to(self) -> None:
        frames = self._make_frames(3, 8, 480, 854)
        target_h, target_w = 480, 848  # 848 = 53*16
        out = resize_crop_tensor(frames, target_h, target_w, mode="resize_crop")
        self.assertEqual(out.shape[-2], target_h)
        self.assertEqual(out.shape[-1], target_w)

    def test_raises_on_bad_ndim(self) -> None:
        bad = torch.rand(3, 256, 256)
        with self.assertRaises(ValueError):
            resize_crop_tensor(bad, 256, 256)

    def test_raises_on_unknown_mode(self) -> None:
        frames = self._make_frames(3, 5, 256, 256)
        with self.assertRaises(ValueError):
            resize_crop_tensor(frames, 256, 256, mode="invalid_mode")

    def test_output_is_float(self) -> None:
        frames = self._make_frames(3, 2, 64, 64)
        out = resize_crop_tensor(frames, 32, 32)
        self.assertEqual(out.dtype, torch.float32)


@unittest.skipIf(torch is None or select_frames is None, "torch not installed")
class TestSelectFrames(unittest.TestCase):
    def test_trim(self) -> None:
        frames = torch.rand(3, 10, 64, 64)
        out = select_frames(frames, 5)
        self.assertEqual(out.shape, (3, 5, 64, 64))

    def test_exact(self) -> None:
        frames = torch.rand(3, 33, 64, 64)
        out = select_frames(frames, 33)
        self.assertEqual(out.shape, (3, 33, 64, 64))

    def test_loop_extension(self) -> None:
        frames = torch.rand(3, 5, 64, 64)
        out = select_frames(frames, 17)
        self.assertEqual(out.shape, (3, 17, 64, 64))

    def test_single_frame_looped(self) -> None:
        frames = torch.rand(3, 1, 32, 32)
        out = select_frames(frames, 9)
        self.assertEqual(out.shape, (3, 9, 32, 32))

    def test_raises_on_bad_ndim(self) -> None:
        bad = torch.rand(3, 64, 64)
        with self.assertRaises(ValueError):
            select_frames(bad, 5)


# ---------------------------------------------------------------------------
# build_cache: disabling bucketing preserves the unbucketed reference path.
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch not installed")
class TestBucketingDisabledPreservesLegacyBehavior(unittest.TestCase):
    def test_records_match_without_bucket_fields_populated(self) -> None:
        from mirai.core.dataset.cache import build_cache

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data = tmpdir / "data"
            data.mkdir(parents=True, exist_ok=True)
            video = torch.rand((3, 5, 32, 32), dtype=torch.float32)
            torch.save(video, data / "sample0.pt")
            (data / "sample0.txt").write_text("test caption", encoding="utf-8")

            cache_path = tmpdir / "cache.pt"
            payload = build_cache(
                data,
                cache_path,
                native_encode=True,
                model_type="sparse_moe_test",
                model_variant="tiny-video",
                model_path=str(tmpdir / "missing-model"),
                model_dtype="bf16",
                enable_bucketing=False,
            )
            rec = payload["records"][0]
            # With bucketing off, bucket fields are empty/zero.
            self.assertEqual(rec["bucket_id"], "")
            self.assertEqual(rec["bucket_h"], 0)
            self.assertEqual(rec["bucket_w"], 0)
            self.assertEqual(rec["bucket_frames"], 0)
            # Latent should still be a finite tensor.
            self.assertIsInstance(rec["latent"], torch.Tensor)
            self.assertTrue(torch.isfinite(rec["latent"]).all().item())


# ---------------------------------------------------------------------------
# build_cache: bucketing enabled — records carry bucket_id, shaped correctly
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch not installed")
class TestBucketingEnabledEndToEnd(unittest.TestCase):
    def _make_pixel_tensor(self, c: int, t: int, h: int, w: int) -> "torch.Tensor":
        return (torch.rand(c, t, h, w) * 255.0)

    def test_records_carry_bucket_id_and_latents_have_bucket_shape(self) -> None:
        from mirai.core.dataset.cache import build_cache

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data = tmpdir / "data"
            data.mkdir(parents=True, exist_ok=True)

            # Three media of deliberately different shapes.
            samples = [
                ("sq", self._make_pixel_tensor(3, 9, 64, 64)),    # square
                ("land", self._make_pixel_tensor(3, 9, 48, 64)),   # landscape
                ("port", self._make_pixel_tensor(3, 9, 64, 48)),   # portrait
            ]
            for name, tensor in samples:
                torch.save(tensor, data / f"{name}.pt")
                (data / f"{name}.txt").write_text(f"caption for {name}", encoding="utf-8")

            # Buckets: square + landscape + portrait, all divisible by 16.
            resolution_buckets = [(64, 64), (48, 64), (64, 48)]
            frame_buckets = [9]

            cache_path = tmpdir / "cache.pt"
            payload = build_cache(
                data,
                cache_path,
                native_encode=True,
                model_type="sparse_moe_test",
                model_variant="tiny-video",
                model_path=str(tmpdir / "missing-model"),
                model_dtype="bf16",
                enable_bucketing=True,
                resolution_buckets=resolution_buckets,
                frame_buckets=frame_buckets,
                bucket_resize_mode="resize_crop",
            )
            self.assertEqual(int(payload["num_records"]), 3)
            records = {r["sample_id"]: r for r in payload["records"]}

            # Each record must carry a non-empty bucket_id.
            for sid, rec in records.items():
                self.assertNotEqual(rec["bucket_id"], "", f"sample {sid} missing bucket_id")
                self.assertGreater(rec["bucket_h"], 0)
                self.assertGreater(rec["bucket_w"], 0)
                self.assertGreater(rec["bucket_frames"], 0)

            # Distinct shapes → distinct bucket_ids.
            bucket_ids = {r["bucket_id"] for r in records.values()}
            self.assertGreater(len(bucket_ids), 1, "All samples ended up in same bucket")

    def test_different_shapes_produce_distinct_bucket_ids(self) -> None:
        from mirai.core.dataset.cache import build_cache

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data = tmpdir / "data"
            data.mkdir(parents=True, exist_ok=True)

            torch.save(self._make_pixel_tensor(3, 5, 480, 640), data / "land.pt")
            torch.save(self._make_pixel_tensor(3, 5, 640, 480), data / "port.pt")
            (data / "land.txt").write_text("landscape", encoding="utf-8")
            (data / "port.txt").write_text("portrait", encoding="utf-8")

            resolution_buckets = [(480, 640), (640, 480)]
            frame_buckets = [5]

            cache_path = tmpdir / "cache.pt"
            payload = build_cache(
                data,
                cache_path,
                native_encode=True,
                model_type="sparse_moe_test",
                model_variant="tiny-video",
                model_path=str(tmpdir / "missing-model"),
                model_dtype="bf16",
                enable_bucketing=True,
                resolution_buckets=resolution_buckets,
                frame_buckets=frame_buckets,
            )
            bucket_ids = {r["bucket_id"] for r in payload["records"]}
            self.assertEqual(len(bucket_ids), 2)


# ---------------------------------------------------------------------------
# batch_sampling: group_records_by_bucket and sample_batch_bucketed
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch not installed")
class TestGroupRecordsByBucket(unittest.TestCase):
    def test_groups_correctly(self) -> None:
        from mirai.core.training.data.batches import group_records_by_bucket

        records = [
            {"sample_id": "a", "bucket_id": "512x512x33"},
            {"sample_id": "b", "bucket_id": "768x512x33"},
            {"sample_id": "c", "bucket_id": "512x512x33"},
        ]
        groups = group_records_by_bucket(records)
        self.assertIn("512x512x33", groups)
        self.assertIn("768x512x33", groups)
        self.assertEqual(len(groups["512x512x33"]), 2)
        self.assertEqual(len(groups["768x512x33"]), 1)

    def test_no_bucket_id_falls_under_empty_string(self) -> None:
        from mirai.core.training.data.batches import group_records_by_bucket

        records = [{"sample_id": "x"}, {"sample_id": "y"}]
        groups = group_records_by_bucket(records)
        self.assertIn("", groups)
        self.assertEqual(len(groups[""]), 2)


if __name__ == "__main__":
    unittest.main()
