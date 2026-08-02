"""Online sampling helpers for caption and temporal variants."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from mirai.core.training.data.tags import generate_tag_variants

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def caption_embed_scalar(caption: str) -> float:
    digest = hashlib.blake2b(str(caption).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return (value % 10000) / 10000.0


def online_caption_embed(
    *,
    record: dict[str, Any],
    step: int,
    seed: int,
    enabled: bool,
    dropout_rate: float,
    keep_first_n_tags: int,
    base_value: Any | None = None,
) -> Any:
    if base_value is None:
        base_value = record.get("text_embed", 0.0)
    base_tensor = None
    if torch is not None and isinstance(base_value, torch.Tensor):
        base_tensor = base_value.detach().float()
    if not enabled:
        if base_tensor is not None:
            return base_tensor
        return float(base_value)
    caption = str(record.get("caption", "")).strip()
    if not caption:
        if base_tensor is not None:
            return base_tensor
        return float(base_value)
    sample_id = str(record.get("sample_id", ""))
    seed_bytes = hashlib.blake2b(
        f"{sample_id}:{int(step)}:{int(seed)}".encode("utf-8"),
        digest_size=8,
    ).digest()
    variant = generate_tag_variants(
        caption,
        tag_shuffle_variants=1,
        tag_dropout_rate=float(dropout_rate),
        keep_first_n_tags=int(keep_first_n_tags),
        seed=int.from_bytes(seed_bytes, "little"),
    )[0]
    variant_scalar = caption_embed_scalar(variant)
    if base_tensor is None:
        return variant_scalar
    baseline_scalar = caption_embed_scalar(caption)
    delta = float(variant_scalar - baseline_scalar)
    return base_tensor + delta


def _clip_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    clip_index = int(record.get("clip_index", 0))
    sample_id = str(record.get("sample_id", ""))
    return clip_index, sample_id


def build_temporal_groups(
    records: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        base_id = str(record.get("base_sample_id", "")).strip()
        if not base_id:
            sample_id = str(record.get("sample_id", ""))
            marker = "::clip"
            if marker in sample_id:
                base_id = sample_id.split(marker, 1)[0]
            else:
                base_id = sample_id
        groups.setdefault(base_id, []).append(record)
    for key in list(groups):
        groups[key] = sorted(groups[key], key=_clip_sort_key)
    base_ids = sorted(groups)
    return base_ids, groups


def pick_temporal_variant(
    group: list[dict[str, Any]],
    *,
    epoch: int,
    seed: int,
    sample_position: int,
) -> dict[str, Any]:
    if len(group) <= 1:
        return group[0]
    rng = random.Random(int(seed) + int(epoch * 100_003) + int(sample_position * 101))
    idx = int(rng.randrange(len(group)))
    return group[idx]
