"""Bucket merge helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MergeDecision:
    chosen: int
    warning: str | None


def choose_merge_target(source_tokens: int, candidates: list[int]) -> MergeDecision:
    if not candidates:
        raise ValueError("candidates cannot be empty.")
    source = int(source_tokens)
    ordered = sorted(candidates)
    chosen = min(
        ordered,
        key=lambda c: (abs(int(c) - source), 0 if int(c) <= source else 1, int(c)),
    )
    warning = None
    if int(chosen) > source:
        warning = (
            f"Merging source bucket ({source}) into larger bucket ({int(chosen)}): "
            "may increase VRAM usage."
        )
    return MergeDecision(chosen=int(chosen), warning=warning)
