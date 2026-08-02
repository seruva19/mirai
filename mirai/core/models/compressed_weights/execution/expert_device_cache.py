"""Byte-bounded device cache for immutable quantized expert operands."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


ExpertDeviceCacheKey = tuple[str, int, str]
ExpertDeviceCacheValue = tuple[Any, Any]


def _tensor_bytes(value: Any) -> int:
    return int(value.numel()) * int(value.element_size())


class ExpertDeviceCache:
    """LRU cache that never exceeds its explicit device-memory capacity."""

    def __init__(self, capacity_bytes: int = 0) -> None:
        capacity = int(capacity_bytes)
        if capacity < 0:
            raise ValueError("Expert device cache capacity must be >= 0 bytes.")
        self._capacity_bytes = capacity
        self._entries: OrderedDict[
            ExpertDeviceCacheKey, tuple[ExpertDeviceCacheValue, int]
        ] = OrderedDict()
        self._resident_bytes = 0
        self._peak_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._oversize_rejections = 0

    @property
    def enabled(self) -> bool:
        return self._capacity_bytes > 0

    @staticmethod
    def key(namespace: str, expert: int, device: Any) -> ExpertDeviceCacheKey:
        return str(namespace), int(expert), str(device)

    def get(self, key: ExpertDeviceCacheKey) -> ExpertDeviceCacheValue | None:
        if not self.enabled:
            return None
        entry = self._entries.pop(key, None)
        if entry is None:
            self._misses += 1
            return None
        self._entries[key] = entry
        self._hits += 1
        return entry[0]

    def put(
        self,
        key: ExpertDeviceCacheKey,
        value: ExpertDeviceCacheValue,
    ) -> bool:
        if not self.enabled:
            return False
        immutable_value = tuple(tensor.detach() for tensor in value)
        size = sum(_tensor_bytes(tensor) for tensor in immutable_value)
        if size > self._capacity_bytes:
            self._oversize_rejections += 1
            return False
        prior = self._entries.pop(key, None)
        if prior is not None:
            self._resident_bytes -= prior[1]
        while self._entries and self._resident_bytes + size > self._capacity_bytes:
            _old_key, (_old_value, old_size) = self._entries.popitem(last=False)
            self._resident_bytes -= old_size
            self._evictions += 1
        self._entries[key] = (immutable_value, size)
        self._resident_bytes += size
        self._peak_bytes = max(self._peak_bytes, self._resident_bytes)
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._resident_bytes = 0

    def snapshot(self) -> dict[str, int | bool]:
        return {
            "enabled": self.enabled,
            "capacity_bytes": self._capacity_bytes,
            "resident_bytes": self._resident_bytes,
            "peak_bytes": self._peak_bytes,
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "oversize_rejections": self._oversize_rejections,
        }


__all__ = ["ExpertDeviceCache", "ExpertDeviceCacheKey", "ExpertDeviceCacheValue"]
