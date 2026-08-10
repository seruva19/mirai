"""Offline local-routing consistency evidence for expert-cache planning.

The segment-cache oracle follows the SCH priority used by Liang et al.: experts
are ordered by their frequency in a bounded look-ahead segment, with recency as
the deterministic tie-breaker.  This is diagnostic evidence, not a realizable
online cache policy, because it uses future routing decisions.

Adapted from https://github.com/ljcleo/moe-lrc (MIT, Copyright 2025 Leo Liang).
Paper: https://arxiv.org/abs/2505.16056
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "mirai.local_routing_cache_evidence.v1"


def _validated_sequence(
    routes: Sequence[Sequence[int]], *, num_experts: int
) -> tuple[tuple[int, ...], ...]:
    experts = int(num_experts)
    if experts <= 0:
        raise ValueError("num_experts must be positive.")
    if not routes:
        raise ValueError("routes must contain at least one token.")
    result: list[tuple[int, ...]] = []
    width: int | None = None
    for token in routes:
        row = tuple(int(expert) for expert in token)
        if not row:
            raise ValueError("every token must select at least one expert.")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("routes must use a fixed active-expert width.")
        if len(set(row)) != len(row):
            raise ValueError("a token must not select the same expert twice.")
        if min(row) < 0 or max(row) >= experts:
            raise ValueError("route contains an expert outside num_experts.")
        result.append(row)
    return tuple(result)


def segment_cache_best_hit_rates(
    routes: Sequence[Sequence[int]],
    *,
    num_experts: int,
    segment_length: int,
    cache_sizes: Iterable[int],
) -> dict[int, float]:
    """Return SCH oracle hit rates for one ordered routing sequence.

    The cache ordering at each route uses expert frequency in the inclusive
    look-ahead window and LRU recency for equal frequencies.  The result for a
    cache size is the fraction of routed expert slots whose rank is in-cache.
    """
    sequence = _validated_sequence(routes, num_experts=num_experts)
    window = int(segment_length)
    if window <= 0:
        raise ValueError("segment_length must be positive.")
    sizes = sorted({int(size) for size in cache_sizes})
    if not sizes or sizes[0] <= 0 or sizes[-1] > int(num_experts):
        raise ValueError("cache_sizes must be within [1, num_experts].")

    future: Counter[int] = Counter()
    for row in sequence[:window]:
        future.update(row)
    recency = [-1] * int(num_experts)
    clock = 0
    hits = {size: 0 for size in sizes}
    total = 0

    for token_index, row in enumerate(sequence):
        for expert in row:
            order = sorted(
                range(int(num_experts)),
                key=lambda candidate: (future[candidate], recency[candidate], -candidate),
                reverse=True,
            )
            rank = order.index(expert)
            for size in sizes:
                hits[size] += int(rank < size)
            total += 1
            future[expert] -= 1
            clock += 1
            recency[expert] = clock
        entering = token_index + window
        if entering < len(sequence):
            future.update(sequence[entering])

    return {size: hits[size] / total for size in sizes}


def build_local_routing_cache_evidence(
    observations: Mapping[str, Any],
    *,
    segment_lengths: Iterable[int],
    cache_sizes: Iterable[int],
    dataset_fingerprint: str,
    model_fingerprint: str,
) -> dict[str, Any]:
    """Build a lineage-bound per-layer SCH report from detached route ids."""
    if not dataset_fingerprint.strip() or not model_fingerprint.strip():
        raise ValueError("dataset and model fingerprints must be non-empty.")
    num_experts = int(observations.get("num_experts", 0))
    raw_layers = observations.get("layers")
    if not isinstance(raw_layers, Mapping) or not raw_layers:
        raise ValueError("observations.layers must be a non-empty mapping.")
    windows = sorted({int(value) for value in segment_lengths})
    sizes = sorted({int(value) for value in cache_sizes})
    if not windows or windows[0] <= 0:
        raise ValueError("segment_lengths must contain positive integers.")

    layers: dict[str, Any] = {}
    aggregate: dict[tuple[int, int], list[float]] = {}
    for name in sorted(raw_layers):
        sequences = raw_layers[name]
        if not isinstance(sequences, list) or not sequences:
            raise ValueError(f"layer {name!r} must contain routing sequences.")
        layer_metrics: dict[str, Any] = {}
        for window in windows:
            per_sequence = [
                segment_cache_best_hit_rates(
                    sequence,
                    num_experts=num_experts,
                    segment_length=window,
                    cache_sizes=sizes,
                )
                for sequence in sequences
            ]
            values = {
                str(size): sum(item[size] for item in per_sequence) / len(per_sequence)
                for size in sizes
            }
            layer_metrics[str(window)] = values
            for size in sizes:
                aggregate.setdefault((window, size), []).append(values[str(size)])
        layers[str(name)] = layer_metrics

    return {
        "schema": SCHEMA,
        "model_fingerprint": model_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "num_experts": num_experts,
        "segment_lengths": windows,
        "cache_sizes": sizes,
        "layers": layers,
        "mean_sch": {
            str(window): {
                str(size): sum(aggregate[(window, size)]) / len(aggregate[(window, size)])
                for size in sizes
            }
            for window in windows
        },
        "oracle": True,
    }


def save_local_routing_cache_evidence(
    path: str | Path, evidence: Mapping[str, Any], *, overwrite: bool = False
) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing evidence: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
