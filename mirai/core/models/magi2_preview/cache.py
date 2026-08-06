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


# Spatial stride of the MAGI-2 preview latent grid. Cached frames must be a
# multiple of it for the same reason generated frames are: the patch grid cannot
# express a fractional latent cell. It mirrors the ``spatial_downsample`` the
# pipeline's ``VideoLatentLayout`` declares, and is duplicated here so cache
# encoding stays free of the vendored runtime import.
MAGI2_SPATIAL_STRIDE = 16


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

    def _prepare_video_for_vae(self, media_path: Path) -> tuple[torch.Tensor, BucketInfo]:
        """Trim frames and land the clip on the configured resolution bucket.

        The spatial target is the dataset bucket when bucketing is configured, so
        a cached sample carries the token count the training geometry declares
        rather than the source resolution. Without buckets the clip keeps its own
        size, cropped to the spatial stride the transformer patch grid requires.
        """
        video = _load_video_media(media_path, max_frames=self.max_frames)
        valid_frames = magi2_cache_frame_trim(int(video.shape[1]))
        video = video[:, :valid_frames]
        height = int(video.shape[-2])
        width = int(video.shape[-1])
        stride = MAGI2_SPATIAL_STRIDE
        target_height = max(stride, (height // stride) * stride)
        target_width = max(stride, (width // stride) * stride)
        if bool(self.config.enable_bucketing) and self.config.resolution_buckets:
            from mirai.core.dataset.bucketing.bucket_resolve import choose_resolution_bucket
            from mirai.core.dataset.media.media_resize import resize_crop_tensor

            target_height, target_width = choose_resolution_bucket(
                height,
                width,
                list(self.config.resolution_buckets),
            )
            if (
                target_height % MAGI2_SPATIAL_STRIDE
                or target_width % MAGI2_SPATIAL_STRIDE
            ):
                raise ValueError(
                    f"MAGI-2 resolution bucket {target_height}x{target_width} is not a "
                    f"multiple of {MAGI2_SPATIAL_STRIDE}; the preview latent grid cannot "
                    "express it. Set bucket_resolutions and bucket_round_to accordingly."
                )
            video = resize_crop_tensor(
                video,
                target_height,
                target_width,
                mode=str(self.config.bucket_resize_mode),
            )
        elif target_height != height or target_width != width:
            top = max(0, (height - target_height) // 2)
            left = max(0, (width - target_width) // 2)
            video = video[:, :, top : top + target_height, left : left + target_width]
        return video, BucketInfo(
            bucket_id=f"{target_height}x{target_width}x{valid_frames}",
            bucket_h=target_height,
            bucket_w=target_width,
            bucket_frames=valid_frames,
        )

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
        video, bucket = self._prepare_video_for_vae(media_path)
        video = video.div(127.5).sub(1.0)
        vae = self._load_vae()
        device = next(vae.vae.parameters()).device
        latent = vae.encode(video.unsqueeze(0).to(device=device, dtype=torch.float32))
        return latent[0].detach().cpu().float().contiguous(), bucket

    def encode_clip(self, media_path: Path) -> None:
        # MAGI-2 conditions on Qwen text hidden states only; there is no CLIP tower.
        _ = media_path
        return None

