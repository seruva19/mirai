from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mirai.core.inference.session import deserialize_inference_latent
from mirai.core.models.native_video import DecodedMedia, JointMediaLatents
from mirai.core.training.runtime.dry_run import _build_synthetic_moe_batch


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(batch_size=2),
        dataset=SimpleNamespace(frame_buckets=[3]),
        model=SimpleNamespace(params=SimpleNamespace(latent_channels=4, patch_size=2)),
    )


def test_synthetic_batch_merges_family_owned_audio_fields() -> None:
    class Pipeline:
        def get_synthetic_batch_spec(self):
            from mirai.core.models.base import SyntheticBatchSpec

            return SyntheticBatchSpec(latent_channels=4, text_embed_width=8, text_embed_tokens=2)

        def build_synthetic_batch_extras(self, **geometry):
            assert geometry["batch_size"] == 2
            assert geometry["frame_count"] == 3
            return {
                "audio_latents": torch.zeros(2, 6, 12, dtype=geometry["dtype"]),
                "audio_lengths": torch.tensor([12, 12]),
            }

    batch = _build_synthetic_moe_batch(
        _config(), pipeline=Pipeline(), dtype=torch.float32
    )

    assert batch["latents"].shape == (2, 4, 3, 4, 4)
    assert batch["audio_latents"].shape == (2, 6, 12)
    assert batch["audio_lengths"].tolist() == [12, 12]


def test_synthetic_batch_extras_cannot_replace_core_fields() -> None:
    class Pipeline:
        def build_synthetic_batch_extras(self, **geometry):
            return {"latents": torch.zeros(1)}

    with pytest.raises(ValueError, match="may not replace core fields: latents"):
        _build_synthetic_moe_batch(
            _config(), pipeline=Pipeline(), dtype=torch.float32
        )


def test_joint_preview_saves_typed_latents_and_muxes_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mirai.core.training.preview.preview as preview
    import mirai.core.inference.session as session

    joint = JointMediaLatents(
        video=torch.zeros(4, 2, 2),
        audio=torch.zeros(8, 10),
        audio_lengths=torch.tensor([10]),
        fps=24.0,
        sample_rate=16_000,
    )
    decoded = DecodedMedia(
        frames=torch.zeros(2, 3, 4, 4),
        audio=torch.zeros(1600),
        sample_rate=16_000,
    )

    class Pipeline:
        def has_native_inference(self):
            return True

        def load_vae(self, *, device):
            self.loaded = device

        def offload_vae(self):
            self.offloaded = True

        def decode_media_native(self, pred):
            assert pred is joint
            return decoded

    pipeline = Pipeline()
    trainer = SimpleNamespace(pipeline=pipeline)
    runtime = SimpleNamespace(
        scaling_factor=1.0,
        block_swap_overridden=False,
        adapter_merged=False,
        adapter_unmerged=False,
        vae_loaded=True,
        vae_unloaded=True,
    )
    monkeypatch.setattr(preview, "begin_preview_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(preview, "activate_preview_runtime_assets", lambda runtime: None)
    monkeypatch.setattr(preview, "restore_preview_runtime", lambda runtime: None)
    monkeypatch.setattr(
        preview,
        "run_native_denoise_loop",
        lambda **kwargs: (joint, []),
    )
    observed = {}

    def fake_write(path, media, *, fps):
        observed.update(path=Path(path), media=media, fps=fps)
        Path(path).write_bytes(b"muxed")
        return True

    monkeypatch.setattr(session, "_write_decoded_video", fake_write)

    payload = preview.generate_preview(
        trainer=trainer,
        step=4,
        output_dir=tmp_path,
        prompt="joint",
        seed=3,
        sample_blocks_to_swap=0,
        training_blocks_to_swap=0,
        block_swap_mode="off",
        block_swap_backward=False,
    )

    latent_payload = torch.load(
        payload["sample_path"], map_location="cpu", weights_only=True
    )
    restored = deserialize_inference_latent(latent_payload)
    assert isinstance(restored, JointMediaLatents)
    assert restored.sample_rate == 16_000
    assert observed["media"] is decoded
    assert observed["path"].read_bytes() == b"muxed"
    assert payload["sample_video_path"].endswith("preview.mp4")
    assert payload["sample_frames_dir"] == ""
