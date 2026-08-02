"""Multi-dataset source helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any

from mirai.core.dataset.cache import build_cache
from mirai.core.dataset.fingerprint import file_sha256


def choose_weighted_source(weights: dict[str, float], rng: random.Random) -> str:
    if not weights:
        raise ValueError("weights cannot be empty.")
    names = list(weights.keys())
    vals = [float(weights[name]) for name in names]
    if any(v < 0.0 for v in vals):
        raise ValueError("weights must be non-negative.")
    total = sum(vals)
    if total <= 0.0:
        raise ValueError("at least one weight must be > 0.")
    pick = rng.random() * total
    acc = 0.0
    for name, val in zip(names, vals, strict=True):
        acc += val
        if pick <= acc:
            return name
    return names[-1]


@dataclass
class SourceCacheResult:
    name: str
    cache_path: str
    manifest_path: str
    cache_sha256: str
    num_records: int


def build_source_caches(
    sources: list[dict[str, Any]],
    *,
    max_cache_skip_ratio: float = 0.2,
) -> list[SourceCacheResult]:
    results: list[SourceCacheResult] = []
    for source in sources:
        name = str(source["name"])
        dataset_path = source["path"]
        cache_path = Path(source["cache_path"])
        payload = build_cache(
            dataset_path,
            cache_path,
            max_cache_skip_ratio=max_cache_skip_ratio,
        )
        checksum = file_sha256(cache_path)
        manifest = {
            "name": name,
            "dataset_path": str(dataset_path),
            "cache_path": str(cache_path),
            "cache_sha256": checksum,
            "num_records": int(payload["num_records"]),
        }
        manifest_path = cache_path.with_name(cache_path.name + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        results.append(
            SourceCacheResult(
                name=name,
                cache_path=str(cache_path),
                manifest_path=str(manifest_path),
                cache_sha256=checksum,
                num_records=int(payload["num_records"]),
            )
        )
    return results
