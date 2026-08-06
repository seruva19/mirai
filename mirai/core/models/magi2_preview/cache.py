"""Provider-owned Qwen and VAE cache encoding for MAGI-2 Preview."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mirai.core.dataset.native_encode import (
    VIDEO_MEDIA_SUFFIXES,
    BucketInfo,
    NativeCacheStatus,
    _load_video_media,
    _to_cthw,
)
from mirai.core.models.providers import NativeCacheEncoderConfig


def magi2_cache_frame_trim(frame_count: int) -> int:
    """Largest ``8n + 1`` frame count that fits ``frame_count``.

    The Wan2.2 encoder compresses time by eight with a leading key frame, which
    is the same grid the generation path declares in its latent layout, so a
    cached clip and a generated clip express lengths identically.
    """
    return max(1, 1 + 8 * max(0, (int(frame_count) - 1) // 8))


class Magi2PreviewNativeCacheEncoder:
    """Sequential native encoder; large components never share device residency."""

    def __init__(self, config: NativeCacheEncoderConfig) -> None:
        self.config = config
        self.root = Path(config.model_path).expanduser()
        if self.root.name == "preview":
            self.root = self.root.parent
        self.max_frames = max(1, int(config.max_frames))
        self.latent_channels = 48
        self._text: Any | None = None
        self._vae: Any | None = None

    def status(self) -> NativeCacheStatus:
        return NativeCacheStatus(
            enabled=bool(self.config.enabled),
            vae_loaded=self._vae is not None,
            text_loaded=self._text is not None,
            clip_loaded=False,
            vae_mode="native_wan22_vae",
            text_mode="qwen35_hidden_state",
            clip_mode="none",
        )

    def _load_text(self) -> Any:
        if self._text is None:
            from mirai.vendors.magi2_preview.model.qwen35 import Qwen35TextEncoder

            path = self.root / "text_encoder"
            if not path.is_dir():
                raise FileNotFoundError(f"MAGI-2 text encoder assets are missing at {path}.")
            self._text = Qwen35TextEncoder(
                str(path), device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._text

    def encode_text(self, caption: str) -> torch.Tensor:
        return self._load_text().encode(str(caption))[0].detach().cpu().float()

    def _load_vae(self) -> Any:
        if self._vae is None:
            from mirai.vendors.magi2_preview.model.vae2_2 import get_vae2_2

            path = self.root / "vae" / "Wan2.2_VAE.pth"
            if not path.is_file():
                raise FileNotFoundError(f"MAGI-2 VAE assets are missing at {path}.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._vae = get_vae2_2(str(path), device=device, weight_dtype=torch.float32)
        return self._vae

    def encode_latent(self, media_path: Path) -> tuple[torch.Tensor, BucketInfo | None]:
        if media_path.suffix.lower() == ".pt":
            payload = torch.load(media_path, map_location="cpu", weights_only=True)
            if isinstance(payload, dict):
                payload = payload.get("latents", payload.get("latent", payload))
            latent = _to_cthw(torch.as_tensor(payload)).float()
            if latent.ndim != 4 or int(latent.shape[0]) != self.latent_channels:
                raise ValueError(
                    "MAGI-2 precomputed latents must have shape [48,T,H,W]."
                )
            return latent.contiguous(), None
        if media_path.suffix.lower() not in VIDEO_MEDIA_SUFFIXES:
            raise ValueError("MAGI-2 training cache accepts videos or native .pt latents.")
        video = _load_video_media(media_path, max_frames=self.max_frames)
        valid_frames = magi2_cache_frame_trim(int(video.shape[1]))
        video = video[:, :valid_frames]
        height = max(16, int(video.shape[-2]) // 16 * 16)
        width = max(16, int(video.shape[-1]) // 16 * 16)
        video = video[:, :, :height, :width].div(127.5).sub(1.0)
        vae = self._load_vae()
        device = next(vae.vae.parameters()).device
        latent = vae.encode(video.unsqueeze(0).to(device=device, dtype=torch.float32))
        bucket = BucketInfo(
            bucket_id=f"{height}x{width}x{valid_frames}",
            bucket_h=height,
            bucket_w=width,
            bucket_frames=valid_frames,
        )
        return latent[0].detach().cpu().float().contiguous(), bucket

    def encode_clip(self, media_path: Path) -> None:
        # MAGI-2 conditions on Qwen text hidden states only; there is no CLIP tower.
        _ = media_path
        return None

