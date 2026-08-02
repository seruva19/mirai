"""Deterministic dataset split helpers."""

from __future__ import annotations

import hashlib
from typing import Any


SPLIT_ALGO_VERSION = "blake2b-v1"


def _hash_to_unit_interval(group_id: str, split_seed: int) -> float:
    payload = f"{SPLIT_ALGO_VERSION}:{split_seed}:{group_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value / float(2**64)


def assign_group_split(
    group_id: str,
    split_seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> str:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")

    x = _hash_to_unit_interval(group_id=group_id, split_seed=split_seed)
    if x < train_ratio:
        return "train"
    if x < train_ratio + val_ratio:
        return "val"
    return "test"


def build_split_assignments(
    samples: list[dict[str, Any]],
    *,
    split_seed: int,
    sample_id_key: str = "sample_id",
    group_id_key: str = "group_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    group_split: dict[str, str] = {}
    for sample in samples:
        sample_id = str(sample[sample_id_key])
        group_id = str(sample.get(group_id_key, sample_id))
        split = group_split.get(group_id)
        if split is None:
            split = assign_group_split(
                group_id=group_id,
                split_seed=split_seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
            group_split[group_id] = split
        assignments[sample_id] = split
    return assignments


def split_ids(assignments: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    train = sorted([sample_id for sample_id, split in assignments.items() if split == "train"])
    val = sorted([sample_id for sample_id, split in assignments.items() if split == "val"])
    test = sorted([sample_id for sample_id, split in assignments.items() if split == "test"])
    return train, val, test
