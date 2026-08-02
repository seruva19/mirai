"""Tag shuffle/dropout helpers."""

from __future__ import annotations

import random


def _split_tags(caption: str) -> list[str]:
    return [tag.strip() for tag in caption.split(",") if tag.strip()]


def generate_tag_variants(
    caption: str,
    *,
    tag_shuffle_variants: int,
    tag_dropout_rate: float,
    keep_first_n_tags: int,
    seed: int,
) -> list[str]:
    if tag_shuffle_variants <= 0:
        raise ValueError("tag_shuffle_variants must be > 0.")
    if tag_dropout_rate < 0.0 or tag_dropout_rate > 1.0:
        raise ValueError("tag_dropout_rate must be in [0, 1].")

    # Natural-language captions are passed through unchanged.
    if "," not in caption:
        return [caption]

    tags = _split_tags(caption)
    if not tags:
        return [caption]
    keep_n = max(0, min(int(keep_first_n_tags), len(tags)))
    fixed = tags[:keep_n]
    tail = tags[keep_n:]

    out: list[str] = []
    rng = random.Random(seed)
    for _ in range(tag_shuffle_variants):
        shuffled = list(tail)
        rng.shuffle(shuffled)
        kept_tail = []
        for tag in shuffled:
            if rng.random() >= tag_dropout_rate:
                kept_tail.append(tag)
        merged = fixed + kept_tail
        if not merged:
            merged = tags[:1]
        out.append(", ".join(merged))
    return out


def resolve_conditioning_caption(
    *,
    caption_variant: str,
    conditioning_dropout_applied: bool,
) -> str:
    if conditioning_dropout_applied:
        return ""
    return caption_variant
