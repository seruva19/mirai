"""Native image and audio conditioning for Agnes inference.

The first-frame and frozen reference-audio semantics follow the released Agnes
inference pipeline pinned at fc3971e. Media decoding remains outside the model
forward so the denoising path consumes only validated native VAE latents.
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Any

import torch
import torch.nn.functional as F

from mirai.core.inference.conditioning import (
    AUDIO_TO_VIDEO,
    IMAGE_TO_VIDEO,
    VIDEO_TO_VIDEO,
    InferenceConditioningRequest,
    PreparedInferenceConditioning,
)


def _as_video_tensor(media: Any, *, frame_count: int, fps: float = 30.) -> torch.Tensor:
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError("Condition video fps and frame_count must be positive.")
    if isinstance(media, torch.Tensor):
        value = media.detach().cpu().float()
    else:
        path = Path(str(media)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Condition media does not exist: {path}")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            from mirai.core.dataset.native_encode import _load_image_media
            value = _load_image_media(path)
        else:
            import json
            import numpy as np
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height", "-of", "json", str(path)], check=True, capture_output=True)
            streams = json.loads(probe.stdout).get("streams", [])
            if not streams:
                raise ValueError("Condition video has no video stream.")
            width, height = int(streams[0]["width"]), int(streams[0]["height"])
            result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", f"fps={fps}",
                                     "-frames:v", str(frame_count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                                    check=True, capture_output=True)
            pixels = np.frombuffer(result.stdout, dtype=np.uint8).copy()
            frame_size = height * width * 3
            if not pixels.size or pixels.size % frame_size:
                raise ValueError("Condition video decoded invalid RGB frames.")
            value = torch.from_numpy(pixels.reshape(-1, height, width, 3)).permute(3, 0, 1, 2).float()
    if value.ndim == 5 and int(value.shape[0]) == 1:
        value = value[0]
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4:
        raise ValueError("Agnes condition media must be a CTHW or 1CTHW tensor.")
    if not value.numel() or not bool(torch.isfinite(value).all()) or any(dim <= 0 for dim in value.shape):
        raise ValueError("Agnes condition video must contain finite non-empty frames.")
    if int(value.shape[0]) not in {1, 3} and int(value.shape[-1]) in {1, 3}:
        value = value.permute(3, 0, 1, 2).contiguous()
    if int(value.shape[0]) not in {1, 3}:
        raise ValueError("Agnes condition media must have one or three channels.")
    if int(value.shape[0]) == 1:
        value = value.expand(3, -1, -1, -1)
    if float(value.max()) > 1.0 or float(value.min()) < -1.0:
        value = value.div(127.5).sub(1.0)
    elif float(value.min()) >= 0.0:
        value = value.mul(2.0).sub(1.0)
    return value.clamp(-1.0, 1.0)


def _resize_video(value: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    from mirai.core.dataset.media.media_resize import resize_crop_tensor
    return resize_crop_tensor(value, height, width, mode="resize_crop").contiguous()


def _load_audio(media: Any, *, sample_rate: int, sample_count: int) -> torch.Tensor:
    if sample_rate <= 0 or sample_count <= 0:
        raise ValueError("Condition audio sample_rate and sample_count must be positive.")
    if isinstance(media, torch.Tensor):
        waveform = media.detach().cpu().float()
        if not waveform.numel() or not bool(torch.isfinite(waveform).all()):
            raise ValueError("Condition audio must be finite and non-empty.")
        # Match the native OmniVAE tensor interface: channel-first mono uses the
        # first channel; file decoding explicitly downmixes through FFmpeg -ac 1.
        if waveform.ndim == 2 and 1 <= waveform.shape[0] <= 8:
            waveform = waveform[0]
        elif waveform.ndim == 3 and waveform.shape[0] == 1 and 1 <= waveform.shape[1] <= 8:
            waveform = waveform[0, 0]
        elif waveform.ndim != 1:
            raise ValueError("Condition audio tensor must be T, CT, or 1CT with at most eight channels.")
    else:
        path = Path(str(media)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Condition audio does not exist: {path}")
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-i", str(path), "-vn", "-map", "0:a:0",
                    "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
                ],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"Could not decode condition audio '{path}'.") from exc
        import numpy as np

        waveform = torch.from_numpy(np.frombuffer(result.stdout, dtype=np.float32).copy())
    if waveform.numel() == 0 or not bool(torch.isfinite(waveform).all()):
        raise ValueError("Condition audio must be finite and non-empty.")
    return F.pad(waveform[:sample_count], (0, max(0, sample_count - waveform.numel())))


@torch.no_grad()
def prepare_agnes_conditioning(
    pipeline: Any,
    request: InferenceConditioningRequest,
    *,
    device: str,
    generator: torch.Generator,
) -> PreparedInferenceConditioning:
    """Encode explicit Agnes source modalities into native inference latents."""
    request.validate()
    prepared = PreparedInferenceConditioning.unconditioned(request)
    assets = pipeline._native_assets()
    fps = float(pipeline.options.get("fps", 30))
    from mirai.core.models.agnes_video.sampler import (
        agnes_audio_token_count,
        agnes_sample_count,
    )

    video_media = request.input_image if request.task == IMAGE_TO_VIDEO else request.input_video
    if video_media is not None:
        frames = _as_video_tensor(video_media, frame_count=int(request.frame_count), fps=fps)
        if request.task == IMAGE_TO_VIDEO:
            frames = frames[:, :1]
        elif int(frames.shape[1]) != int(request.frame_count):
            raise ValueError(
                "Agnes V2V source frame count does not match the request: "
                f"{int(frames.shape[1])} vs {int(request.frame_count)}."
            )
        frames = _resize_video(frames, height=int(request.height), width=int(request.width))
        assets.offload_audio_vae()
        latent = assets.load_video_vae(device).encode(
            frames.unsqueeze(0).to(device=device, dtype=assets.weight_dtype)
        )[0].float()
        assets.offload_video_vae()
        if request.task == IMAGE_TO_VIDEO:
            if int(latent.shape[1]) != 1:
                raise ValueError("Agnes I2V image encoding must produce one latent frame.")
            prepared.condition_latent = latent.unsqueeze(0)
        elif request.task == VIDEO_TO_VIDEO:
            prepared.source_latent = latent.unsqueeze(0)

    has_audio = request.input_audio is not None and (
        not isinstance(request.input_audio, str) or bool(request.input_audio.strip())
    )
    if has_audio:
        assets.offload_video_vae()
        rate, hop, channels = assets.audio_geometry()
        sample_count = agnes_sample_count(int(request.frame_count), fps, rate)
        waveform = _load_audio(
            request.input_audio,
            sample_rate=rate,
            sample_count=sample_count,
        )
        audio = assets.encode_audio(waveform, device).float()
        assets.offload_audio_vae()
        expected = agnes_audio_token_count(
            int(request.frame_count), fps, rate, hop, conditioned=True
        )
        if int(audio.shape[0]) != expected or int(audio.shape[1]) != channels:
            raise ValueError(
                "Agnes audio latent shape does not match OmniVAE geometry: "
                f"{tuple(audio.shape)} vs ({expected}, {channels})."
            )
        prepared.forward_media.update(
            audio_latents=audio.unsqueeze(0),
            audio_lengths=torch.tensor([expected], device=audio.device, dtype=torch.long),
            sample_rate=rate,
            fps=fps,
            freeze_audio=True,
        )
    elif request.task == AUDIO_TO_VIDEO:
        raise ValueError("Agnes audio_to_video requires input_audio.")
    else:
        # The released runner draws audio noise before initial video noise.
        # This hook runs before the generic preview creates the video tensor.
        rate, hop, channels = assets.audio_geometry()
        count = agnes_audio_token_count(
            int(request.frame_count), fps, rate, hop, conditioned=False
        )
        prepared.forward_media["audio_noise"] = torch.randn(
            (1, count, channels), device=device, dtype=torch.float32,
            generator=generator,
        )
        assets.offload_audio_vae()
    return prepared


__all__ = ["prepare_agnes_conditioning"]
