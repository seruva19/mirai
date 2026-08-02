"""Resolution and frame bucket selection helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirai.config.schema import DatasetConfig


_DEFAULT_BASE_ASPECTS = [
    (1, 1),
    (4, 3),
    (3, 4),
    (16, 9),
    (9, 16),
    (3, 2),
    (2, 3),
]


def _round_to(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return max(multiple, int(round(value / multiple)) * multiple)


def _derive_default_buckets(base: int, round_to: int) -> list[tuple[int, int]]:
    """Generate a symmetric aspect-ratio bucket set from *base* resolution."""
    b = _round_to(int(base), int(round_to))
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for (r_h, r_w) in _DEFAULT_BASE_ASPECTS:
        area = b * b
        raw_w = math.sqrt(area * r_w / r_h)
        raw_h = math.sqrt(area * r_h / r_w)
        bh = _round_to(int(round(raw_h)), round_to)
        bw = _round_to(int(round(raw_w)), round_to)
        if (bh, bw) not in seen:
            seen.add((bh, bw))
            out.append((bh, bw))
    return out


def parse_resolution_buckets(cfg: "DatasetConfig") -> list[tuple[int, int]]:
    """Parse *cfg.bucket_resolutions* strings into (H, W) pairs.

    Falls back to a symmetric set derived from *cfg.bucket_base_resolution*
    when *bucket_resolutions* is empty.  Each bucket is validated to be
    divisible by *cfg.bucket_round_to*.
    """
    round_to = max(1, int(cfg.bucket_round_to))
    raw = list(cfg.bucket_resolutions)
    if not raw:
        return _derive_default_buckets(int(cfg.bucket_base_resolution), round_to)

    buckets: list[tuple[int, int]] = []
    for spec in raw:
        spec = str(spec).strip()
        if "x" not in spec.lower():
            raise ValueError(
                f"Invalid bucket_resolution '{spec}': expected 'HxW' (e.g. '512x512')."
            )
        parts = spec.lower().split("x", 1)
        try:
            h, w = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid bucket_resolution '{spec}': H and W must be integers."
            )
        if h <= 0 or w <= 0:
            raise ValueError(
                f"Invalid bucket_resolution '{spec}': H and W must be positive."
            )
        if h % round_to != 0 or w % round_to != 0:
            raise ValueError(
                f"Bucket resolution '{spec}' is not divisible by bucket_round_to={round_to}."
            )
        buckets.append((h, w))
    return buckets


def choose_resolution_bucket(
    h: int, w: int, buckets: list[tuple[int, int]]
) -> tuple[int, int]:
    """Return the bucket from *buckets* closest to the (h, w) aspect ratio.

    Tie-break by closest total area.
    """
    if not buckets:
        raise ValueError("buckets cannot be empty.")
    src_h, src_w = max(1, int(h)), max(1, int(w))
    src_ar = src_h / src_w

    def _score(bkt: tuple[int, int]) -> tuple[float, float]:
        bh, bw = bkt
        ar_diff = abs(bh / bw - src_ar)
        area_diff = abs(bh * bw - src_h * src_w)
        return ar_diff, area_diff

    return min(buckets, key=_score)


def choose_frame_bucket(num_frames: int, frame_buckets: list[int]) -> int:
    """Return the largest frame bucket ≤ *num_frames*; fall back to min bucket."""
    if not frame_buckets:
        raise ValueError("frame_buckets cannot be empty.")
    ordered = sorted(int(f) for f in frame_buckets)
    eligible = [f for f in ordered if f <= int(num_frames)]
    return eligible[-1] if eligible else ordered[0]


def bucket_id(bh: int, bw: int, frames: int) -> str:
    """Stable string identifier for a (H, W, T) bucket triple."""
    return f"{int(bh)}x{int(bw)}x{int(frames)}"


def validate_frame_buckets(frame_buckets: list[int]) -> None:
    """Validate configured frame bucket values."""
    if not frame_buckets:
        raise ValueError("frame_buckets cannot be empty.")
    for f in frame_buckets:
        if int(f) <= 0:
            raise ValueError(f"frame_bucket {f} must be positive.")
