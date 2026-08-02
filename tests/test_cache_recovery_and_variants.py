from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mirai.core.dataset.cache import build_cache, load_cache

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class CacheRecoveryAndVariantsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
