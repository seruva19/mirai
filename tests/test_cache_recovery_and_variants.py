from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from mirai.core.dataset.cache import _write_checksum, build_cache, load_cache
from mirai.core.dataset.native_encode import (
    ValidationCacheEncoder,
    validate_native_cache_encoder,
)
from mirai.core.models.providers import (
    ModelFamilyProvider,
    NativeCacheEncoderConfig,
    register_model_family_provider,
)
from mirai.core.persistence.migrations import migrate_cache_payload

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class CacheRecoveryAndVariantsTests(unittest.TestCase):
    def test_malformed_records_fail_migration_and_load(self) -> None:
        cases = (
            (None, r"'records' must be a list"),
            ({"bad": 1}, r"'records' must be a list"),
            ("bad", r"'records' must be a list"),
            ([None], r"record\[0\] is not a table"),
            ([{"sample_id": "kept"}, None], r"record\[1\] is not a table"),
            ([{"sample_id": "kept"}, []], r"record\[1\] is not a table"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.pt"
            for records, message in cases:
                with self.subTest(records=records):
                    payload = {"records": records, "num_records": 2}
                    original = deepcopy(payload)
                    with self.assertRaisesRegex(ValueError, message):
                        migrate_cache_payload(payload)
                    self.assertEqual(payload, original)
                    torch.save(payload, cache_path)
                    with self.assertRaisesRegex(ValueError, message):
                        load_cache(cache_path, verify_checksum=False)

    def test_legacy_and_empty_cache_migrate_without_mutating_input(self) -> None:
        legacy = {"records": [{"caption": "legacy caption"}], "partial_recovery": False}
        original = deepcopy(legacy)
        migrated = migrate_cache_payload(legacy)
        self.assertEqual(legacy, original)
        self.assertEqual(migrated["num_records"], 1)
        self.assertFalse(migrated["partial_recovery"])
        self.assertEqual(
            migrated["records"],
            [{
                "sample_id": "idx_0",
                "caption_variant_index": 0,
                "caption": "legacy caption",
                "base_sample_id": "idx_0",
                "clip_index": 0,
                "text_mask": [1],
                "loss_mask": None,
            }],
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.pt"
            for payload in (legacy, {}, {"records": []}):
                with self.subTest(payload=payload):
                    torch.save(payload, cache_path)
                    loaded = load_cache(cache_path, verify_checksum=False)
                    self.assertEqual(loaded["records"], migrate_cache_payload(payload)["records"])
                    self.assertEqual(loaded["num_records"], len(loaded["records"]))
                    self.assertEqual(loaded["partial_recovery"], payload.get("partial_recovery", True))

    def test_partial_recovery_rebuilds_after_malformed_cache_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            data_dir.mkdir()
            for index in range(2):
                torch.save(torch.tensor([index], dtype=torch.float32), data_dir / f"s{index}.pt")
                (data_dir / f"s{index}.txt").write_text(f"caption {index}", encoding="utf-8")
            original = build_cache(data_dir, cache_path, partial_recovery=False)
            original["records"][1] = None
            torch.save(original, cache_path)
            _write_checksum(cache_path)
            with self.assertRaisesRegex(ValueError, r"record\[1\] is not a table"):
                load_cache(cache_path)

            rebuilt = build_cache(data_dir, cache_path, partial_recovery=True)
            self.assertEqual(rebuilt["recovered_records"], 0)
            self.assertEqual(len(rebuilt["records"]), 2)
            self.assertEqual(len(load_cache(cache_path)["records"]), 2)

    def test_tag_shuffle_variants_expand_caption_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor([0.1], dtype=torch.float32), data_dir / "s0.pt")
            (data_dir / "s0.txt").write_text("cat, walking, sunset", encoding="utf-8")

            payload = build_cache(
                data_dir,
                cache_path,
                tag_shuffle_variants=3,
            )
            self.assertEqual(int(payload["num_records"]), 3)
            self.assertEqual(int(payload["tag_shuffle_variants"]), 3)
            embeds = {float(r["text_embed"]) for r in payload["records"]}
            self.assertGreaterEqual(len(embeds), 2)
            self.assertIn("estimated_disk_bytes", payload)
            rec0 = payload["records"][0]
            self.assertIn("caption", rec0)
            self.assertIn("base_sample_id", rec0)
            self.assertIn("clip_index", rec0)

    def test_partial_recovery_rebuilds_text_when_caption_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor([0.1], dtype=torch.float32), data_dir / "s0.pt")
            (data_dir / "s0.txt").write_text("caption one", encoding="utf-8")
            first = build_cache(data_dir, cache_path, partial_recovery=False)
            first_embed = float(first["records"][0]["text_embed"])

            (data_dir / "s0.txt").write_text("caption changed", encoding="utf-8")
            second = build_cache(data_dir, cache_path, partial_recovery=True)
            second_embed = float(second["records"][0]["text_embed"])
            self.assertEqual(int(second["recovered_records"]), 1)
            self.assertNotEqual(first_embed, second_embed)

            third = build_cache(data_dir, cache_path, partial_recovery=False)
            third_embed = float(third["records"][0]["text_embed"])
            self.assertEqual(second_embed, third_embed)
            loaded = load_cache(cache_path)
            self.assertEqual(int(loaded["num_records"]), 1)

    def test_partial_recovery_rebuilds_latent_when_media_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            cache_path = tmpdir / "cache.pt"
            data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor([0.1], dtype=torch.float32), data_dir / "s0.pt")
            (data_dir / "s0.txt").write_text("caption one", encoding="utf-8")

            first = build_cache(data_dir, cache_path, partial_recovery=False)
            first_latent = float(torch.as_tensor(first["records"][0]["latent"]).float().mean().item())

            torch.save(torch.tensor([0.9], dtype=torch.float32), data_dir / "s0.pt")
            second = build_cache(data_dir, cache_path, partial_recovery=True)
            second_latent = float(torch.as_tensor(second["records"][0]["latent"]).float().mean().item())

            self.assertEqual(int(second["recovered_records"]), 1)
            self.assertNotEqual(first_latent, second_latent)


_CLIP_FAILING_STEMS: set[str] = set()


class _ClipFailingEncoder(ValidationCacheEncoder):
    """Encoder whose per-sample CLIP encoding fails on selected media."""

    def encode_clip(self, media_path: Path):
        if media_path.stem in _CLIP_FAILING_STEMS:
            raise RuntimeError("clip tower rejected this media")
        return None


class _ClipFailingProvider(ModelFamilyProvider):
    def build_native_cache_encoder(self, config: NativeCacheEncoderConfig):
        return _ClipFailingEncoder(
            enabled=config.enabled,
            model_type=config.model_type,
            variant=config.variant,
            model_path=config.model_path,
            dtype_name=config.dtype_name,
            max_frames=config.max_frames,
        )


register_model_family_provider(
    "cache_contract_test",
    _ClipFailingProvider(
        model_type="cache_contract_test",
        native=True,
        sparse_moe=True,
        native_cache_encoding=True,
    ),
)


class _MissingClipEncoder:
    """Encoder that breaks the contract by omitting ``encode_clip``."""

    latent_channels = 1

    def status(self):
        raise AssertionError("status must not be reached")

    def encode_text(self, caption: str):
        raise AssertionError("encode_text must not be reached")

    def encode_latent(self, media_path: Path):
        raise AssertionError("encode_latent must not be reached")


@unittest.skipIf(torch is None, "torch not installed")
class NativeCacheEncoderContractTests(unittest.TestCase):
    def test_missing_encoder_method_fails_at_construction_naming_the_method(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_native_cache_encoder(
                _MissingClipEncoder(), source="test model family"
            )
        message = str(ctx.exception)
        self.assertIn("encode_clip()", message)
        self.assertIn("NativeCacheEncoder contract", message)

    def test_conforming_encoder_passes_construction_validation(self) -> None:
        encoder = ValidationCacheEncoder(
            enabled=False,
            model_type="precomputed",
            variant="precomputed",
            model_path="",
            dtype_name="bf16",
            max_frames=1,
        )
        self.assertIs(validate_native_cache_encoder(encoder, source="test"), encoder)

    def _build_dataset(self, data_dir: Path, stems: list[str]) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            torch.save(torch.tensor([0.1], dtype=torch.float32), data_dir / f"{stem}.pt")
            (data_dir / f"{stem}.txt").write_text("a caption", encoding="utf-8")

    def test_per_sample_clip_failure_is_skipped_visibly(self) -> None:
        _CLIP_FAILING_STEMS.clear()
        _CLIP_FAILING_STEMS.add("bad")
        self.addCleanup(_CLIP_FAILING_STEMS.clear)
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            self._build_dataset(data_dir, ["good", "bad"])
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                payload = build_cache(
                    data_dir,
                    tmpdir / "cache.pt",
                    native_encode=True,
                    model_type="cache_contract_test",
                    model_variant="test",
                    max_cache_skip_ratio=0.9,
                )
            self.assertEqual(int(payload["num_records"]), 1)
            self.assertEqual(int(payload["num_skipped"]), 1)
            self.assertEqual(
                payload["skipped"],
                [{"sample_id": "bad", "status": "skipped_clip_error"}],
            )
            warning = stderr.getvalue()
            self.assertIn("bad", warning)
            self.assertIn("skipped_clip_error", warning)
            self.assertIn("clip tower rejected this media", warning)

    def test_all_samples_skipped_reports_the_skip_breakdown(self) -> None:
        _CLIP_FAILING_STEMS.clear()
        _CLIP_FAILING_STEMS.update({"s0", "s1"})
        self.addCleanup(_CLIP_FAILING_STEMS.clear)
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            data_dir = tmpdir / "data"
            self._build_dataset(data_dir, ["s0", "s1"])
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(ValueError) as ctx:
                    build_cache(
                        data_dir,
                        tmpdir / "cache.pt",
                        native_encode=True,
                        model_type="cache_contract_test",
                        model_variant="test",
                    )
            message = str(ctx.exception)
            self.assertIn("2/2 samples skipped", message)
            self.assertIn("2 skipped_clip_error", message)
            self.assertIn("RuntimeError: clip tower rejected this media", message)


if __name__ == "__main__":
    unittest.main()
