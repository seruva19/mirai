from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace
import wave

import pytest
import torch

from mirai.core.inference.session import (
    decode_pipeline_media,
    deserialize_inference_latent,
    serialize_inference_latent,
)
from mirai.core.models.native_video import (
    DecodedMedia,
    JointMediaLatents,
    NativeVideoPipeline,
)


def _joint() -> JointMediaLatents:
    return JointMediaLatents(
        video=torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2),
        audio=torch.linspace(-1.0, 1.0, 800),
        audio_lengths=torch.tensor([800]),
        fps=24.0,
        sample_rate=16_000,
        requested_frame_count=151,
    )


def test_joint_latent_artifact_roundtrip_is_weights_only_safe(tmp_path: Path) -> None:
    path = tmp_path / "joint.pt"
    torch.save(serialize_inference_latent(_joint()), path)

    raw = torch.load(path, map_location="cpu", weights_only=True)
    restored = deserialize_inference_latent(raw)

    assert isinstance(restored, JointMediaLatents)
    assert restored.fps == 24.0
    assert restored.sample_rate == 16_000
    assert restored.requested_frame_count == 151
    torch.testing.assert_close(restored.video, _joint().video)
    torch.testing.assert_close(restored.audio, _joint().audio)
    torch.testing.assert_close(restored.audio_lengths, _joint().audio_lengths)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"version": 3}, "version"),
        ({"extra": True}, "fields"),
        ({"sample_rate": 0}, "sample_rate"),
        ({"audio": "not-a-tensor"}, "must be tensors"),
        ({"requested_frame_count": 0}, "requested_frame_count"),
        ({"requested_frame_count": 1.5}, "requested_frame_count"),
        ({"requested_frame_count": True}, "requested_frame_count"),
    ],
)
def test_joint_latent_payload_rejects_malformed_data(change: dict, message: str) -> None:
    payload = _joint().to_payload()
    payload.update(change)

    with pytest.raises((TypeError, ValueError), match=message):
        JointMediaLatents.from_payload(payload)


def test_legacy_tensor_serialization_is_unchanged() -> None:
    tensor = torch.randn(2, 3)

    serialized = serialize_inference_latent(tensor)

    assert isinstance(serialized, torch.Tensor)
    torch.testing.assert_close(deserialize_inference_latent(serialized), tensor)


def test_version_one_joint_latent_remains_readable() -> None:
    payload = _joint().to_payload()
    payload["version"] = 1
    del payload["requested_frame_count"]
    restored = JointMediaLatents.from_payload(payload)
    assert restored.requested_frame_count is None
    torch.testing.assert_close(restored.video, _joint().video)


def test_default_decode_media_hook_delegates_video_latent() -> None:
    seen = []
    pipeline = SimpleNamespace(
        decode_latents_native=lambda latents: seen.append(latents) or torch.ones(1, 3, 2, 2)
    )

    decoded = NativeVideoPipeline.decode_media_native(pipeline, _joint())

    assert isinstance(decoded, DecodedMedia)
    assert decoded.audio is None
    assert seen[0][0] is _joint().video or torch.equal(seen[0][0], _joint().video)


def test_decode_pipeline_media_muxes_audio_and_removes_exact_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mirai.core.training.preview import preview
    import mirai.core.inference.session as session_module

    frames = torch.zeros(2, 3, 4, 4)
    audio = torch.sin(torch.linspace(0, 20, 1600))

    class Pipeline:
        def load_vae(self, *, device: str) -> None:
            self.device = device

        def offload_vae(self) -> None:
            self.offloaded = True

        def decode_media_native(self, pred):
            return DecodedMedia(frames=frames, audio=audio, sample_rate=16_000)

    monkeypatch.setattr(session_module.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        preview,
        "_write_mp4",
        lambda path, decoded_frames, fps: Path(path).write_bytes(b"video") or True,
    )
    observed = {}

    def fake_run(command, **kwargs):
        video_path = Path(command[command.index("-i") + 1])
        audio_index = command.index("-i", command.index("-i") + 1)
        audio_path = Path(command[audio_index + 1])
        assert video_path.name == ".result.video.tmp.mp4"
        assert audio_path.name == ".result.audio.tmp.wav"
        assert video_path.read_bytes() == b"video"
        with wave.open(str(audio_path), "rb") as handle:
            assert handle.getframerate() == 16_000
            assert handle.getnchannels() == 1
        Path(command[-1]).write_bytes(b"muxed")
        observed["command"] = command
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(session_module.subprocess, "run", fake_run)
    pipeline = Pipeline()

    media_path = decode_pipeline_media(
        pipeline=pipeline,
        pred=_joint(),
        out_path=tmp_path / "result.any",
        task="text_to_video",
        fps=24.0,
    )

    assert media_path == tmp_path / "result.mp4"
    assert media_path.read_bytes() == b"muxed"
    assert "-shortest" in observed["command"]
    assert not (tmp_path / ".result.video.tmp.mp4").exists()
    assert not (tmp_path / ".result.audio.tmp.wav").exists()
    assert pipeline.offloaded is True


def test_decode_never_silently_drops_audio_without_sample_rate(tmp_path: Path) -> None:
    class Pipeline:
        def load_vae(self, *, device: str) -> None:
            pass

        def offload_vae(self) -> None:
            pass

        def decode_media_native(self, pred):
            return DecodedMedia(
                frames=torch.zeros(1, 3, 2, 2),
                audio=torch.zeros(10),
                sample_rate=None,
            )

    with pytest.raises(RuntimeError, match="positive sample_rate"):
        decode_pipeline_media(
            pipeline=Pipeline(),
            pred=torch.zeros(1),
            out_path=tmp_path / "result.mp4",
            task="text_to_video",
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_real_ffmpeg_mux_writes_joint_media(tmp_path: Path) -> None:
    pytest.importorskip("av")

    class Pipeline:
        def load_vae(self, *, device: str) -> None:
            pass

        def offload_vae(self) -> None:
            pass

        def decode_media_native(self, pred):
            frames = torch.zeros(2, 3, 16, 16)
            frames[1, 0] = 1.0
            return DecodedMedia(
                frames=frames,
                audio=torch.sin(torch.linspace(0, 100, 4_000)),
                sample_rate=16_000,
            )

    media_path = decode_pipeline_media(
        pipeline=Pipeline(),
        pred=_joint(),
        out_path=tmp_path / "real.mp4",
        task="text_to_video",
        fps=8.0,
    )

    assert media_path.is_file()
    assert media_path.stat().st_size > 0
