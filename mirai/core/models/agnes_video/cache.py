"""Synchronized native Agnes video/audio cache encoding."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from mirai.core.dataset.native_encode import BucketInfo, NativeCacheStatus
from mirai.core.dataset.cache import EncodedMediaVariant
from mirai.core.models.providers import NativeCacheEncoderConfig
from mirai.core.lineage import snapshot_descriptor_for_path
from .assets import AgnesAssets


class AgnesNativeCacheEncoder:
    latent_channels = 48

    def __init__(self, config: NativeCacheEncoderConfig):
        self.config = config
        options = dict(config.family_params)
        options["model_path"] = config.model_path
        if config.text_encoder_path:
            options["text_encoder_path"] = config.text_encoder_path
        self.assets = AgnesAssets(options)
        self.fps = float(options.get("fps", 30))
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Agnes fps must be finite and positive.")
        self.allow_missing_audio = bool(options.get("allow_missing_audio", False))
        self._selected: tuple[Path, float, float] | None = None
        self._precomputed_audio: torch.Tensor | None = None
        self._selection_start = 0.0
        self._variant_selection: dict = {}

    def status(self) -> NativeCacheStatus:
        return NativeCacheStatus(self.config.enabled, self.assets.video_vae is not None,
                                 self.assets.text_encoder is not None, False,
                                 "native_wan22_vae", "t5gemma_hidden_state", "none")

    def cache_asset_fingerprints(self) -> dict[str, str]:
        """Hash component contents even when a directory has a download manifest."""
        defaults = {"text_encoder_path": "text_encoder", "video_vae_path": "vae/Wan2.2_VAE.pth",
                    "audio_vae_path": "audio_vae", "audio_vae_code_root": "OmniVAE"}
        result = {}
        for name, default in defaults.items():
            path = self.assets._path(name, default)
            if name == "audio_vae_code_root":
                path = path / "vae" / "omnivae" if (path / "vae" / "omnivae").is_dir() else path / "omnivae"
                source_files = {file.relative_to(path).as_posix(): snapshot_descriptor_for_path(file).snapshot_id
                                for file in sorted(path.rglob("*")) if file.is_file()
                                and file.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}}
                if not source_files:
                    raise FileNotFoundError(f"OmniVAE source files are missing at {path}.")
                result[name] = "source-sha256:" + hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
                continue
            result[name] = snapshot_descriptor_for_path(path, manifest_candidates=()).snapshot_id
        return result

    def encode_text(self, caption: str) -> torch.Tensor:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            return self.assets.encode_prompt(caption, device=device)[0].detach().cpu().float()
        finally:
            self.assets.offload_text_encoder()

    def encode_clip(self, media_path: Path) -> None:
        return None

    def cache_media_metadata(self, media_path: Path) -> dict:
        if self._selected is None or self._selected[0] != media_path.resolve():
            raise ValueError("Cache timeline requires the selected video input.")
        _, start, duration = self._selected
        return {"media_timeline": {"start_seconds": start, "duration_seconds": duration,
                                   "fps": self.fps, "selection": "prefix" if duration else "precomputed",
                                   **self._variant_selection}}

    def encode_media_variant(self, media_path: Path, *, clip_index: int, clip_count: int) -> EncodedMediaVariant:
        if clip_count < 1 or not 0 <= clip_index < clip_count:
            raise ValueError("Invalid media clip index/count.")
        if media_path.suffix.lower() == ".pt":
            if clip_count != 1:
                raise ValueError("Precomputed Agnes joint latents do not support temporal clip variants.")
            start, span = 0.0, 0.0
        else:
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=duration:format=duration", "-of", "json", str(media_path)],
                                   check=True, capture_output=True)
            info = json.loads(probe.stdout)
            streams = info.get("streams", [])
            stream_duration = streams[0].get("duration") if streams else None
            try:
                duration = float(stream_duration)
            except (TypeError, ValueError):
                duration = float(info.get("format", {}).get("duration", 0))
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Agnes temporal clip selection requires a positive video duration.")
            frames = 1 + 4 * ((max(1, self.config.max_frames) - 1) // 4)
            if self.config.enable_bucketing and self.config.frame_buckets:
                choices = [int(value) for value in self.config.frame_buckets if int(value) <= frames and int(value) % 4 == 1]
                if not choices:
                    raise ValueError("Agnes frame buckets require a fitting 4n+1 frame count.")
                frames = max(choices)
            span = max(0.0, duration - frames / self.fps)
            start = span * clip_index / (clip_count - 1) if clip_count > 1 else 0.0
        self._selection_start = start
        self._variant_selection = {"selection": "precomputed" if media_path.suffix.lower() == ".pt" else "uniform_windows", "clip_index": clip_index,
                                   "clip_count": clip_count, "identical_short_clip": clip_count > 1 and span == 0.0}
        try:
            latent, bucket = self.encode_latent(media_path)
            tensors, schema = self.encode_extra_tensors(media_path)
            metadata = self.cache_media_metadata(media_path)
            return EncodedMediaVariant(latent, bucket, tensors, schema, metadata)
        finally:
            self._selection_start = 0.0
            self._variant_selection = {}

    def _load_video(self, path: Path) -> torch.Tensor:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height", "-of", "json", str(path)],
                               check=True, capture_output=True)
        streams = json.loads(probe.stdout)["streams"]
        if not streams:
            raise ValueError("Agnes cache input has no video stream.")
        width, height = int(streams[0]["width"]), int(streams[0]["height"])
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(self._selection_start), "-i", str(path), "-an", "-vf", f"fps={self.fps}",
                                  "-frames:v", str(self.config.max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                                 check=True, capture_output=True)
        pixels = np.frombuffer(decoded.stdout, dtype=np.uint8).copy()
        frame_size = width * height * 3
        if not pixels.size or pixels.size % frame_size:
            raise ValueError("Agnes video decoder returned invalid RGB frames.")
        frames = torch.from_numpy(pixels.reshape(-1, height, width, 3)).permute(3, 0, 1, 2).float()
        count = 1 + 4 * ((frames.shape[1] - 1) // 4)
        if self.config.enable_bucketing and self.config.frame_buckets:
            choices = [int(value) for value in self.config.frame_buckets if int(value) <= count and int(value) % 4 == 1]
            if not choices:
                raise ValueError("Agnes frame buckets require a fitting 4n+1 frame count.")
            count = max(choices)
        self._selected = (path.resolve(), self._selection_start, count / self.fps)
        return frames[:, :count]

    def encode_latent(self, media_path: Path) -> tuple[torch.Tensor, BucketInfo | None]:
        self._precomputed_audio = None
        if media_path.suffix.lower() == ".pt":
            payload = torch.load(media_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or "audio_latents" not in payload:
                raise ValueError("Agnes precomputed .pt input requires video_latents and audio_latents.")
            latent = payload.get("video_latents", payload.get("latent"))
            audio = payload["audio_latents"]
            if not isinstance(latent, torch.Tensor) or latent.ndim != 4 or latent.shape[0] != 48 or latent.shape[-1] % 2 or latent.shape[-2] % 2:
                raise ValueError("Agnes video latents must have shape [48,T,H,W] with even H and W.")
            if not isinstance(audio, torch.Tensor) or audio.ndim != 2 or audio.shape[1] != 128 or audio.shape[0] <= 0:
                raise ValueError("Agnes audio latents must have shape [L,128] with L > 0.")
            self._precomputed_audio = audio.detach().cpu().float()
            self._selected = (media_path.resolve(), 0.0, 0.0)
            return latent.detach().cpu().float().contiguous(), None
        video = self._load_video(media_path)
        height, width = video.shape[-2:]
        target_h, target_w = height // 32 * 32, width // 32 * 32
        if self.config.enable_bucketing and self.config.resolution_buckets:
            from mirai.core.dataset.bucketing.bucket_resolve import choose_resolution_bucket
            target_h, target_w = choose_resolution_bucket(height, width, self.config.resolution_buckets)
        if min(target_h, target_w) < 32 or target_h % 32 or target_w % 32:
            raise ValueError("Agnes video dimensions must be positive multiples of 32.")
        from mirai.core.dataset.media.media_resize import resize_crop_tensor
        video = resize_crop_tensor(video, target_h, target_w, mode=self.config.bucket_resize_mode)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assets.offload_text_encoder()
        self.assets.offload_audio_vae()
        try:
            with torch.no_grad():
                latent = self.assets.load_video_vae(device).encode(video.unsqueeze(0).to(device).div(127.5).sub(1.0))[0]
        finally:
            self.assets.offload_video_vae()
        bucket = BucketInfo(f"{target_h}x{target_w}x{video.shape[1]}", target_h, target_w, video.shape[1])
        return latent.detach().cpu().float().contiguous(), bucket

    def encode_extra_tensors(self, media_path: Path) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, str]]]:
        if self._selected is None or self._selected[0] != media_path.resolve():
            raise ValueError("Audio encoding requires the same immediately selected video input.")
        if self._precomputed_audio is not None:
            audio = self._precomputed_audio
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.assets.offload_video_vae()
            self.assets.offload_text_encoder()
            self.assets.load_audio_vae(device)
            rate, _, _ = self.assets.audio_geometry()
            _, start, duration = self._selected
            samples = max(1, math.ceil(duration * rate))
            try:
                try:
                    result = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(media_path),
                                             "-t", str(duration), "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(rate),
                                             "-f", "f32le", "pipe:1"], check=True, capture_output=True)
                    waveform = np.frombuffer(result.stdout, dtype=np.float32).copy()
                    if not waveform.size:
                        raise ValueError("Audio stream is empty.")
                except (subprocess.CalledProcessError, ValueError):
                    if not self.allow_missing_audio:
                        raise ValueError(f"Agnes joint training requires audio in '{media_path}'.")
                    waveform = np.zeros(samples, dtype=np.float32)
                waveform = np.pad(waveform[:samples], (0, max(0, samples - waveform.size)))
                audio = self.assets.encode_audio(torch.from_numpy(waveform), device).detach().cpu().float()
            finally:
                self.assets.offload_audio_vae()
        return {"audio_latents": audio}, {"audio_latents": {"collation": "pad_sequence", "lengths_key": "audio_lengths"}}
