"""Sparse-MoE cache encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for cache encoding: {exc}") from exc


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_MEDIA_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


def _hash_scalar(text: str) -> float:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return float((value % 10000) / 10000.0)


def _load_pt_media(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in (
            "video",
            "frames",
            "pixel_values",
            "tensor",
            "latent",
            "latents",
            "dit_latent",
            "video_latent",
        ):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, torch.Tensor):
        payload = torch.as_tensor(payload, dtype=torch.float32)
    return payload.detach().cpu().float()


def _load_image_media(path: Path) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    arr = torch.from_numpy(__import__("numpy").array(image)).float()
    return arr.permute(2, 0, 1).unsqueeze(1)


def _load_video_media(path: Path, *, max_frames: int) -> torch.Tensor:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames: list[torch.Tensor] = []
    try:
        while len(frames) < int(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb).float())
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Could not decode any frames from media file: {path}")
    stacked = torch.stack(frames, dim=0)
    return stacked.permute(3, 0, 1, 2).contiguous()


def _to_cthw(tensor: torch.Tensor) -> torch.Tensor:
    x = tensor.detach().cpu().float()
    if x.ndim == 5:
        x = x[0]
    if x.ndim == 4:
        if int(x.shape[0]) in {1, 3, 4, 16, 48}:
            return x
        if int(x.shape[-1]) in {1, 3, 4}:
            return x.permute(3, 0, 1, 2).contiguous()
        return x
    if x.ndim == 3:
        if int(x.shape[0]) in {1, 3, 4, 16, 48}:
            return x.unsqueeze(1)
        if int(x.shape[-1]) in {1, 3, 4}:
            return x.permute(2, 0, 1).unsqueeze(1).contiguous()
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 1:
        return x.reshape(1, 1, 1, int(x.shape[0]))
    return x.reshape(1, 1, 1, 1)


def _normalize_pixel_range(x: torch.Tensor) -> torch.Tensor:
    out = x.float()
    if out.numel() == 0:
        return out
    mx = float(out.max().item())
    mn = float(out.min().item())
    if mn >= -1.0 and mx <= 1.0:
        return out
    if mn >= 0.0 and mx <= 1.0:
        return out * 2.0 - 1.0
    return out / 127.5 - 1.0


def _looks_like_latent(x: torch.Tensor, *, latent_channels: int) -> bool:
    return x.ndim == 4 and int(x.shape[0]) == int(latent_channels)


def _project_to_latent(pixel: torch.Tensor, *, latent_channels: int) -> torch.Tensor:
    x = _normalize_pixel_range(pixel)
    if x.ndim != 4:
        x = _to_cthw(x)
    base = x.mean(dim=0, keepdim=True).unsqueeze(0)
    target_h = max(1, int(base.shape[-2]) // 8)
    target_w = max(1, int(base.shape[-1]) // 8)
    pooled = F.adaptive_avg_pool3d(
        base,
        output_size=(int(base.shape[-3]), target_h, target_w),
    )
    out = pooled.squeeze(0).repeat(latent_channels, 1, 1, 1)
    return out.contiguous().float()


@dataclass
class NativeCacheStatus:
    enabled: bool
    vae_loaded: bool
    text_loaded: bool
    clip_loaded: bool
    vae_mode: str
    text_mode: str
    clip_mode: str


@dataclass
class BucketInfo:
    """Bucket assignment for one encoded media sample."""

    bucket_id: str
    bucket_h: int
    bucket_w: int
    bucket_frames: int


@runtime_checkable
class NativeCacheEncoder(Protocol):
    """Object contract every provider-owned cache encoder must satisfy.

    The cache builder calls exactly these members on the encoder it receives
    from ``build_native_cache_encoder``. ``encode_clip`` is part of the contract
    even for families without CLIP-style conditioning; such families return
    ``None``. ``encode_text_for_media`` is the one optional extension and is
    probed explicitly at its call site.
    """

    latent_channels: int

    def status(self) -> NativeCacheStatus: ...

    def encode_text(self, caption: str) -> Any: ...

    def encode_latent(self, media_path: Path) -> tuple[torch.Tensor, "BucketInfo | None"]: ...

    def encode_clip(self, media_path: Path) -> Any | None: ...


NATIVE_CACHE_ENCODER_REQUIRED_METHODS: tuple[str, ...] = (
    "status",
    "encode_text",
    "encode_latent",
    "encode_clip",
)
NATIVE_CACHE_ENCODER_REQUIRED_ATTRIBUTES: tuple[str, ...] = ("latent_channels",)


def validate_native_cache_encoder(encoder: Any, *, source: str) -> NativeCacheEncoder:
    """Fail before encoding starts when an encoder breaks the object contract.

    Duck typing here is unsafe: a missing method surfaces only inside the
    per-sample loop, where it is indistinguishable from a per-sample data error.
    """

    missing: list[str] = []
    for name in NATIVE_CACHE_ENCODER_REQUIRED_METHODS:
        if not callable(getattr(encoder, name, None)):
            missing.append(f"{name}()")
    for name in NATIVE_CACHE_ENCODER_REQUIRED_ATTRIBUTES:
        if getattr(encoder, name, None) is None:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Native cache encoder {type(encoder).__name__} from {source} does not "
            "satisfy the NativeCacheEncoder contract; missing: "
            f"{', '.join(missing)}."
        )
    return encoder


def native_cache_encoder_contract_error(exc: BaseException, encoder: Any) -> bool:
    """True when ``exc`` is a missing member on the encoder itself.

    A contract violation is a code defect and must abort the build; an error
    raised while processing one media file is a data problem and may be skipped.
    """

    if not isinstance(exc, AttributeError):
        return False
    if getattr(exc, "obj", None) is encoder:
        return True
    name = str(getattr(exc, "name", "") or "")
    return bool(
        name
        and name
        in set(NATIVE_CACHE_ENCODER_REQUIRED_METHODS)
        | set(NATIVE_CACHE_ENCODER_REQUIRED_ATTRIBUTES)
        and not hasattr(encoder, name)
    )


class ValidationCacheEncoder:
    """Deterministic encoder for the explicit sparse-MoE validation provider.

    Supported model families never construct this class. Their cache semantics
    remain provider-owned through ``build_native_cache_encoder``.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        model_type: str,
        variant: str,
        model_path: str,
        dtype_name: str,
        max_frames: int,
        denoiser_subfolder: str = "transformer",
        strict_assets: bool = False,
        text_encoder_path: str = "",
        enable_bucketing: bool = False,
        resolution_buckets: list[tuple[int, int]] | None = None,
        frame_buckets: list[int] | None = None,
        bucket_resize_mode: str = "resize_crop",
    ):
        _ = model_path, dtype_name, text_encoder_path, denoiser_subfolder
        self.model_type = str(model_type).strip().lower()
        self.variant = str(variant).strip().lower()
        self.enabled = bool(enabled)
        self.strict_assets = bool(strict_assets and self.enabled)
        self.max_frames = max(1, int(max_frames))
        self.latent_channels = 1
        self._text_mode = "hash"
        self._vae_mode = "deterministic_projection"
        self._clip_mode = "none"
        self.enable_bucketing = bool(enable_bucketing)
        self._resolution_buckets: list[tuple[int, int]] = list(resolution_buckets or [])
        self._frame_buckets: list[int] = list(frame_buckets or [int(max_frames)])
        self._bucket_resize_mode = str(bucket_resize_mode)
        if self.strict_assets:
            raise ValueError(
                "Strict cache assets require a sparse-MoE model-family provider "
                "with build_native_cache_encoder()."
            )

    def status(self) -> NativeCacheStatus:
        return NativeCacheStatus(
            enabled=bool(self.enabled),
            vae_loaded=False,
            text_loaded=False,
            clip_loaded=False,
            vae_mode=str(self._vae_mode),
            text_mode=str(self._text_mode),
            clip_mode=str(self._clip_mode),
        )

    def encode_text(self, caption: str) -> Any:
        return _hash_scalar(str(caption))

    def load_media_tensor(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        if suffix == ".pt":
            return _load_pt_media(path)
        if suffix in _IMAGE_SUFFIXES:
            return _load_image_media(path)
        if suffix in VIDEO_MEDIA_SUFFIXES:
            return _load_video_media(path, max_frames=self.max_frames)
        raise ValueError(f"Unsupported media extension for cache encoding: {path.suffix}")

    def _apply_bucket_transforms(
        self,
        tensor: torch.Tensor,
        *,
        is_image: bool,
    ) -> tuple[torch.Tensor, BucketInfo | None]:
        if not self.enable_bucketing:
            return tensor, None

        from mirai.core.dataset.bucketing.bucket_resolve import (
            bucket_id as make_bucket_id,
            choose_frame_bucket,
            choose_resolution_bucket,
        )
        from mirai.core.dataset.media.media_resize import resize_crop_tensor, select_frames

        h = int(tensor.shape[-2])
        w = int(tensor.shape[-1])
        t = int(tensor.shape[1])

        if self._resolution_buckets:
            bh, bw = choose_resolution_bucket(h, w, self._resolution_buckets)
        else:
            bh, bw = h, w

        if is_image:
            bf = sorted(int(f) for f in self._frame_buckets)[0]
        else:
            bf = choose_frame_bucket(t, self._frame_buckets)

        if bh != h or bw != w:
            tensor = resize_crop_tensor(tensor, bh, bw, mode=self._bucket_resize_mode)
        tensor = select_frames(tensor, bf)

        info = BucketInfo(
            bucket_id=make_bucket_id(bh, bw, bf),
            bucket_h=bh,
            bucket_w=bw,
            bucket_frames=bf,
        )
        return tensor, info

    def encode_latent(self, media_path: Path) -> tuple[torch.Tensor, BucketInfo | None]:
        raw = self.load_media_tensor(media_path)
        tensor = _to_cthw(raw)
        if _looks_like_latent(tensor, latent_channels=self.latent_channels):
            return tensor.float(), None

        is_image = media_path.suffix.lower() in _IMAGE_SUFFIXES
        tensor, bucket_info = self._apply_bucket_transforms(tensor, is_image=is_image)
        pixel = _normalize_pixel_range(tensor)
        return _project_to_latent(pixel, latent_channels=self.latent_channels), bucket_info

    def encode_clip(self, media_path: Path) -> torch.Tensor | None:
        _ = media_path
        return None
