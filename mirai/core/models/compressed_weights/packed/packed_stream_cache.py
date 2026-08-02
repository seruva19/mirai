"""Bounded host cache for packed-stream slice and canonical expert-set requests."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any


PackedStreamCacheKey = tuple[str, str, tuple[int, ...]]
PackedStreamCacheValue = tuple[Any, ...]


class PackedStreamHostCache:
    """Byte-bounded LRU over immutable host tensors returned by range reads."""

    def __init__(self, capacity_bytes: int = 0) -> None:
        capacity = int(capacity_bytes)
        if capacity < 0:
            raise ValueError("Packed stream cache capacity must be >= 0 bytes.")
        self._capacity_bytes = capacity
        self._entries: OrderedDict[
            PackedStreamCacheKey, tuple[PackedStreamCacheValue, int]
        ] = OrderedDict()
        self._resident_bytes = 0
        self._peak_bytes = 0
        self._hits = 0
        self._misses = 0
        self._hit_bytes = 0
        self._evictions = 0
        self._evicted_bytes = 0
        self._oversize_rejections = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._capacity_bytes > 0

    def get(
        self, key: PackedStreamCacheKey
    ) -> PackedStreamCacheValue | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                self._misses += 1
                return None
            self._entries[key] = entry
            tensors, nbytes = entry
            self._hits += 1
            self._hit_bytes += int(nbytes)
            return tensors

    def put(
        self,
        key: PackedStreamCacheKey,
        tensors: Sequence[Any],
    ) -> bool:
        if not self.enabled:
            return False
        value = tuple(tensors)
        nbytes = sum(_tensor_nbytes(tensor) for tensor in value)
        if nbytes > self._capacity_bytes:
            with self._lock:
                self._oversize_rejections += 1
            return False
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._resident_bytes -= int(previous[1])
            while (
                self._entries
                and self._resident_bytes + nbytes > self._capacity_bytes
            ):
                _, (_, evicted_bytes) = self._entries.popitem(last=False)
                self._resident_bytes -= int(evicted_bytes)
                self._evictions += 1
                self._evicted_bytes += int(evicted_bytes)
            self._entries[key] = (value, nbytes)
            self._resident_bytes += nbytes
            self._peak_bytes = max(self._peak_bytes, self._resident_bytes)
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._resident_bytes = 0

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "cache_capacity_bytes": int(self._capacity_bytes),
                "cache_resident_bytes": int(self._resident_bytes),
                "cache_peak_bytes": int(self._peak_bytes),
                "cache_entries": len(self._entries),
                "cache_hits": int(self._hits),
                "cache_misses": int(self._misses),
                "cache_hit_bytes": int(self._hit_bytes),
                "cache_evictions": int(self._evictions),
                "cache_evicted_bytes": int(self._evicted_bytes),
                "cache_oversize_rejections": int(self._oversize_rejections),
            }


def packed_stream_cache_key(
    tensor_key: str,
    indices: Sequence[int],
    *,
    access: str = "batch",
) -> PackedStreamCacheKey:
    return str(access), str(tensor_key), tuple(int(index) for index in indices)


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())
