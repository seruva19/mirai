"""Plan microbatches from predicted packed-expert reads."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path


ROUTING_TIMESTEPS_KEY = "_routing_batch_timesteps"


@dataclass(frozen=True)
class RoutingBatchProfile:
    cache_experts: int
    timestep_bins: int
    samples: dict[str, tuple[frozenset[int] | None, ...]]

    @classmethod
    def load(cls, path: str) -> RoutingBatchProfile:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Routing batch profile must use schema_version 1.")
        cache_experts, bins, samples = (payload.get(key) for key in ("cache_experts", "timestep_bins", "samples"))
        if type(cache_experts) is not int or cache_experts < 0 or type(bins) is not int or bins < 1 or not isinstance(samples, dict):
            raise ValueError("Routing batch profile requires cache_experts, timestep_bins, and a samples map.")
        fitted = {}
        for sample, routes in samples.items():
            if not isinstance(routes, list) or len(routes) != bins:
                raise ValueError(f"Routing batch profile sample {sample!r} must have {bins} timestep bins.")
            if any(route is not None and (
                not isinstance(route, list) or any(type(expert) is not int or expert < 0 for expert in route)
            ) for route in routes):
                raise ValueError(f"Routing batch profile sample {sample!r} has invalid expert IDs.")
            fitted[sample] = tuple(None if route is None else frozenset(route) for route in routes)
        return cls(cache_experts, bins, fitted)

    def experts(self, sample: str, timestep: float) -> frozenset[int] | None:
        routes = self.samples.get(sample)
        if routes is None or not 0.0 <= timestep <= 1.0:
            return None
        return routes[min(int(timestep * self.timestep_bins), self.timestep_bins - 1)]


@dataclass(frozen=True)
class RoutingBatchItem:
    """A sample position with a known route and a collation compatibility key."""

    compatibility: Hashable
    experts: frozenset[int] | None


def _cold_reads(batches, items, capacity):
    cache = OrderedDict()
    reads = 0
    for batch in batches:
        touched = set().union(*(items[index].experts for index in batch))
        for expert in sorted(touched):
            if expert in cache:
                cache.move_to_end(expert)
            else:
                reads += 1
                cache[expert] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    return reads


def plan_routing_microbatches(
    batches: Sequence[Sequence[RoutingBatchItem]], *, cache_experts: int,
) -> tuple[tuple[int, ...], ...]:
    """Return positions into the flattened original window, preserving batch sizes.

    Only contiguous runs of homogeneous, compatible microbatches with complete
    signatures are eligible for regrouping. Unknown routes keep their original
    positions. A candidate must improve the estimated cold-read count.
    """
    if cache_experts < 0:
        raise ValueError("The routing microbatch cache capacity must be non-negative.")
    items = [item for batch in batches for item in batch]
    start = 0
    original = []
    for batch in batches:
        original.append(tuple(range(start, start + len(batch))))
        start += len(batch)

    def compatible(batch):
        if not batch or batch[0].experts is None:
            return None
        key = batch[0].compatibility
        if any(item.compatibility != key or item.experts is None for item in batch):
            return None
        return (key,)

    result = list(original)
    first = 0
    while first < len(batches):
        key = compatible(batches[first])
        stop = first + 1
        while key is not None and stop < len(batches) and compatible(batches[stop]) == key:
            stop += 1
        if stop - first > 1:
            old = original[first:stop]
            remaining = dict.fromkeys(index for batch in old for index in batch)
            candidate = []
            recent: set[int] = set()
            for batch in old:
                chosen = []
                touched: set[int] = set()
                for _ in batch:
                    best = min(
                        remaining,
                        key=lambda index: (
                            len(items[index].experts - touched - recent),
                            len(items[index].experts - touched),
                            index,
                        ),
                    )
                    del remaining[best]
                    chosen.append(best)
                    touched.update(items[best].experts)
                candidate.append(tuple(chosen))
                recent = set(sorted(touched)[-cache_experts:]) if cache_experts else set()
            if _cold_reads(candidate, items, cache_experts) < _cold_reads(old, items, cache_experts):
                result[first:stop] = candidate
        first = stop
    return tuple(result)
