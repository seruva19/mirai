from __future__ import annotations

import torch
import pytest

from mirai.core.inference.conditioning import InferenceConditioningRequest
from mirai.core.models.agnes_video.conditioning import prepare_agnes_conditioning


class _VideoVAE:
    def __init__(self):
        self.conv = torch.nn.Conv3d(3, 1, 1, dtype=torch.bfloat16)

    def encode(self, video):
        assert tuple(video.shape) == (1, 3, 1, 32, 64)
        assert self.conv(video).dtype == torch.bfloat16
        return [torch.full((48, 1, 2, 4), 3.0, device=video.device)]


class _Assets:
    def __init__(self):
        self.video = _VideoVAE()
        self.weight_dtype = torch.bfloat16

    def offload_audio_vae(self):
        pass

    def offload_video_vae(self):
        pass

    def load_video_vae(self, device):
        _ = device
        return self.video

    def audio_geometry(self):
        return 16, 4, 2

    def encode_audio(self, waveform, device):
        assert waveform.shape == (16,)
        return torch.arange(8, device=device, dtype=torch.float32).reshape(4, 2)


class _Pipeline:
    options = {"fps": 1}

    def __init__(self):
        self.assets = _Assets()

    def _native_assets(self):
        return self.assets


def test_agnes_i2v_prepares_pinned_first_frame_and_frozen_audio() -> None:
    prepared = prepare_agnes_conditioning(
        _Pipeline(),
        InferenceConditioningRequest(
            task="i2v",
            input_image=torch.zeros(3, 1, 20, 40),
            input_audio=torch.ones(16),
            frame_count=1,
            height=32,
            width=64,
        ),
        device="cpu",
        generator=torch.Generator().manual_seed(3),
    )
    assert prepared.condition_latent is not None
    assert tuple(prepared.condition_latent.shape) == (1, 48, 1, 2, 4)
    assert tuple(prepared.forward_media["audio_latents"].shape) == (1, 4, 2)
    assert prepared.forward_media["freeze_audio"] is True
    assert prepared.forward_media["sample_rate"] == 16


def test_agnes_a2v_prepares_audio_without_video_condition() -> None:
    prepared = prepare_agnes_conditioning(
        _Pipeline(),
        InferenceConditioningRequest(
            task="a2v",
            input_audio=torch.ones(16),
            frame_count=1,
            height=32,
            width=64,
        ),
        device="cpu",
        generator=torch.Generator().manual_seed(3),
    )
    assert prepared.condition_latent is None
    assert prepared.source_latent is None
    assert torch.equal(
        prepared.forward_media["audio_lengths"], torch.tensor([4])
    )


def test_t2v_initial_noise_matches_source_audio_then_video_order():
    pipeline = _Pipeline()
    pipeline.options = {"fps": 30}
    pipeline.assets.audio_geometry = lambda: (48_000, 960, 128)
    request = InferenceConditioningRequest(
        task="text_to_video", frame_count=151, height=256, width=448,
    )
    generator = torch.Generator().manual_seed(42)
    prepared = prepare_agnes_conditioning(
        pipeline, request, device="cpu", generator=generator,
    )
    video = torch.randn((48, 38, 16, 28), generator=generator)
    reference = torch.Generator().manual_seed(42)
    source_audio = torch.randn((1, 250, 128), generator=reference)
    source_video = torch.randn((1, 48, 38, 16, 28), generator=reference)
    torch.testing.assert_close(prepared.forward_media["audio_noise"], source_audio, rtol=0, atol=0)
    torch.testing.assert_close(video, source_video[0], rtol=0, atol=0)
    assert "audio_latents" not in prepared.forward_media
    assert not prepared.forward_media.get("freeze_audio", False)


def test_tensor_audio_uses_native_first_channel_without_concatenating_channels():
    from mirai.core.models.agnes_video.conditioning import _load_audio

    first = torch.arange(6).float()
    stereo = torch.stack((first, torch.full_like(first, 99)))
    for value in (first, stereo, stereo.unsqueeze(0)):
        torch.testing.assert_close(_load_audio(value, sample_rate=48000, sample_count=8),
                                   torch.cat((first, torch.zeros(2))), rtol=0, atol=0)
    for value in (torch.ones(2, 2, 6), torch.ones(16, 6), torch.empty(0), torch.tensor([float("nan")])):
        with pytest.raises(ValueError):
            _load_audio(value, sample_rate=48000, sample_count=8)
    stereo[1, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        _load_audio(stereo, sample_rate=48000, sample_count=8)


def test_tensor_video_rejects_nonfinite_and_empty_media():
    from mirai.core.models.agnes_video.conditioning import _as_video_tensor

    for value in (torch.full((3, 1, 4, 4), float("nan")), torch.empty(3, 0, 4, 4)):
        with pytest.raises(ValueError, match="finite non-empty"):
            _as_video_tensor(value, frame_count=1, fps=30)


def test_file_video_resamples_source_fps_to_match_audio_clock(tmp_path):
    import shutil
    import subprocess
    from mirai.core.models.agnes_video.conditioning import _as_video_tensor, _load_audio

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe unavailable")
    path = tmp_path / "clock.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=black:size=32x32:rate=24:duration=3,geq=lum=16+219*T/3:cb=128:cr=128",
                    "-f", "lavfi", "-i", "aevalsrc=t/3:s=300:d=3", "-c:v", "ffv1", "-c:a", "pcm_f32le", str(path)],
                   check=True, capture_output=True)
    video = _as_video_tensor(path, frame_count=61, fps=30)
    audio = _load_audio(path, sample_rate=300, sample_count=610)
    assert video.shape == (3, 61, 32, 32)
    # Last output frame is at two seconds, whereas taking 61 source frames
    # without resampling would reach 2.5 seconds. Both modalities encode t/3.
    normalized_video_clock = (video[:, -1].mean().item() + 1) / 2
    assert normalized_video_clock == pytest.approx(2 / 3, abs=.02)
    assert audio.mean().item() == pytest.approx((609 / 300) / 6, abs=.005)
