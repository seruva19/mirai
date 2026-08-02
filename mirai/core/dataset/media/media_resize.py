"""CPU-side media resize/crop helpers for bucket-aligned preprocessing."""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for media_resize: {exc}")

_VALID_MODES = {"resize_crop", "resize", "crop", "pad"}


def _bilinear_resize(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Bilinear resize of [C, T, H, W] to (target_h, target_w).  CPU float."""
    c, t, h, w = frames.shape
    merged = frames.permute(1, 0, 2, 3).reshape(t, c, h, w)  # [T, C, H, W]
    resized = F.interpolate(
        merged.float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )  # [T, C, H, W]
    return resized.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]


def _center_crop(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Center-crop [C, T, H, W] to (target_h, target_w)."""
    h, w = frames.shape[-2], frames.shape[-1]
    start_h = max(0, (h - target_h) // 2)
    start_w = max(0, (w - target_w) // 2)
    return frames[..., start_h : start_h + target_h, start_w : start_w + target_w]


def _pad_to(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Zero-pad [C, T, H, W] to at least (target_h, target_w), then center-crop."""
    h, w = frames.shape[-2], frames.shape[-1]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if top or bottom or left or right:
        # F.pad expects [..., H, W] with padding in (left, right, top, bottom) order.
        frames = F.pad(frames.float(), (left, right, top, bottom), mode="constant", value=0.0)
    return _center_crop(frames, target_h, target_w)


def resize_crop_tensor(
    frames: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "resize_crop",
) -> torch.Tensor:
    """Resize/crop *frames* to exactly (target_h, target_w) using *mode*.

    *frames* must be [C, T, H, W] on CPU, float.  The output is always float.

    Modes:
      resize_crop — scale so the shorter side matches the target, then center-crop.
      resize      — stretch both axes to match the target (may distort AR).
      crop        — center-crop without any scaling (pads if source is smaller).
      pad         — pad to cover target, then center-crop.
    """
    mode = str(mode).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown bucket_resize_mode '{mode}'. "
            f"Valid options: {sorted(_VALID_MODES)}."
        )
    if frames.ndim != 4:
        raise ValueError(
            f"resize_crop_tensor expects [C, T, H, W], got shape {tuple(frames.shape)}."
        )
    frames = frames.float().cpu()
    h, w = int(frames.shape[-2]), int(frames.shape[-1])

    if mode == "resize":
        return _bilinear_resize(frames, target_h, target_w)

    if mode in ("crop", "pad"):
        return _pad_to(frames, target_h, target_w)

    # resize_crop: scale so shorter side matches, then center-crop.
    scale = max(target_h / h, target_w / w)
    scaled_h = max(target_h, int(math.ceil(h * scale)))
    scaled_w = max(target_w, int(math.ceil(w * scale)))
    if scaled_h != h or scaled_w != w:
        frames = _bilinear_resize(frames, scaled_h, scaled_w)
    return _center_crop(frames, target_h, target_w)


def select_frames(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Trim or loop *frames* along dim-1 (T) to reach exactly *target_frames*.

    *frames* must be [C, T, H, W].
    """
    if frames.ndim != 4:
        raise ValueError(
            f"select_frames expects [C, T, H, W], got shape {tuple(frames.shape)}."
        )
    t = int(frames.shape[1])
    target = max(1, int(target_frames))
    if t >= target:
        return frames[:, :target].contiguous()
    # Loop to fill: tile then trim.
    repeats = math.ceil(target / t)
    looped = frames.repeat(1, repeats, 1, 1)
    return looped[:, :target].contiguous()
