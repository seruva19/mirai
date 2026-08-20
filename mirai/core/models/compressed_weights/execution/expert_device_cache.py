"""Byte-bounded device cache for immutable quantized expert operands."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


ExpertDeviceCacheKey = tuple[str, int, str]
ExpertDeviceCacheValue = tuple[Any, Any]
_MAX_FREQUENCY = (1 << 31) - 1


def _tensor_bytes(value: Any) -> int:
    return int(value.numel()) * int(value.element_size())


class ExpertDeviceCache:
    """LRU cache that never exceeds its explicit device-memory capacity."""

    def __init__(
        self,
        capacity_bytes: int = 0,
        *,
        admission_policy: str = "lru",
        frequency_capacity: int = 4096,
    ) -> None:
        capacity = int(capacity_bytes)
        if capacity < 0:
            raise ValueError("Expert device cache capacity must be >= 0 bytes.")
        policy = str(admission_policy).strip().lower()
        if policy not in {"lru", "routing_frequency"}:
            raise ValueError(
                "Expert device cache admission policy must be lru or "
                "routing_frequency."
            )
        frequency_capacity = int(frequency_capacity)
        if frequency_capacity <= 0:
            raise ValueError("Expert device cache frequency capacity must be > 0.")
        self._capacity_bytes = capacity
        self._admission_policy = policy
        self._frequency_capacity = frequency_capacity
        self._entries: OrderedDict[
            ExpertDeviceCacheKey, tuple[ExpertDeviceCacheValue, int]
        ] = OrderedDict()
        self._frequencies: OrderedDict[ExpertDeviceCacheKey, int] = OrderedDict()
        self._resident_bytes = 0
        self._peak_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._oversize_rejections = 0
        self._admission_rejections = 0
        self._request_calls = 0
        self._requested_rows = 0
        self._request_hit_rows = 0
        self._request_miss_rows = 0
        self._unique_transferred_rows = 0
        self._deduplicated_miss_rows = 0
        self._transferred_bytes = 0
        self._coalesced_requests = 0
        self._transfer_fallback_requests = 0

    def _observe(self, key: ExpertDeviceCacheKey) -> int:
        frequency = min(_MAX_FREQUENCY, self._frequencies.pop(key, 0) + 1)
        self._frequencies[key] = frequency
        while len(self._frequencies) > self._frequency_capacity:
            self._frequencies.popitem(last=False)
        return frequency

    @property
    def enabled(self) -> bool:
        return self._capacity_bytes > 0

    @staticmethod
    def key(namespace: str, expert: int, device: Any) -> ExpertDeviceCacheKey:
        return str(namespace), int(expert), str(device)

    def get(self, key: ExpertDeviceCacheKey) -> ExpertDeviceCacheValue | None:
        if not self.enabled:
            return None
        self._observe(key)
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
        if key not in self._frequencies:
            self._observe(key)
        immutable_value = tuple(tensor.detach() for tensor in value)
        size = sum(_tensor_bytes(tensor) for tensor in immutable_value)
        if size > self._capacity_bytes:
            self._oversize_rejections += 1
            return False
        prior = self._entries.pop(key, None)
        if prior is not None:
            self._resident_bytes -= prior[1]
        candidate_frequency = max(1, self._frequencies.get(key, 0))
        if (
            self._admission_policy == "routing_frequency"
            and prior is None
            and self._entries
            and self._resident_bytes + size > self._capacity_bytes
        ):
            victim_key = min(
                self._entries,
                key=lambda resident_key: self._frequencies.get(resident_key, 0),
            )
            if candidate_frequency <= self._frequencies.get(victim_key, 0):
                self._admission_rejections += 1
                return False
        while self._entries and self._resident_bytes + size > self._capacity_bytes:
            if self._admission_policy == "routing_frequency":
                old_key = min(
                    self._entries,
                    key=lambda resident_key: self._frequencies.get(resident_key, 0),
                )
                _old_value, old_size = self._entries.pop(old_key)
            else:
                _old_key, (_old_value, old_size) = self._entries.popitem(last=False)
            self._resident_bytes -= old_size
            self._evictions += 1
        self._entries[key] = (immutable_value, size)
        self._resident_bytes += size
        self._peak_bytes = max(self._peak_bytes, self._resident_bytes)
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._frequencies.clear()
        self._resident_bytes = 0

    def record_transfer_request(
        self,
        *,
        requested_rows: int,
        hit_rows: int,
        miss_rows: int,
        unique_transferred_rows: int,
        transferred_bytes: int,
        coalesced: bool,
        fallback: bool,
    ) -> None:
        """Record scalar transfer accounting without retaining route identities."""

        requested = int(requested_rows)
        hits = int(hit_rows)
        misses = int(miss_rows)
        unique = int(unique_transferred_rows)
        moved = int(transferred_bytes)
        if min(requested, hits, misses, unique, moved) < 0:
            raise ValueError("Expert transfer telemetry values must be non-negative.")
        if hits + misses != requested:
            raise ValueError("Expert transfer hit and miss rows must equal requested rows.")
        if unique > misses:
            raise ValueError("Unique transferred rows cannot exceed miss rows.")
        if coalesced and fallback:
            raise ValueError("A transfer request cannot be coalesced and fallback.")
        self._request_calls += 1
        self._requested_rows += requested
        self._request_hit_rows += hits
        self._request_miss_rows += misses
        self._unique_transferred_rows += unique
        self._deduplicated_miss_rows += misses - unique
        self._transferred_bytes += moved
        self._coalesced_requests += int(bool(coalesced))
        self._transfer_fallback_requests += int(bool(fallback))

    def snapshot(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "admission_policy": self._admission_policy,
            "capacity_bytes": self._capacity_bytes,
            "resident_bytes": self._resident_bytes,
            "peak_bytes": self._peak_bytes,
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "oversize_rejections": self._oversize_rejections,
            "admission_rejections": self._admission_rejections,
            "frequency_entries": len(self._frequencies),
            "frequency_capacity": self._frequency_capacity,
            "transfer_request_calls": self._request_calls,
            "transfer_requested_rows": self._requested_rows,
            "transfer_hit_rows": self._request_hit_rows,
            "transfer_miss_rows": self._request_miss_rows,
            "transfer_unique_rows": self._unique_transferred_rows,
            "transfer_deduplicated_rows": self._deduplicated_miss_rows,
            "transfer_bytes": self._transferred_bytes,
            "transfer_coalesced_requests": self._coalesced_requests,
            "transfer_fallback_requests": self._transfer_fallback_requests,
        }


__all__ = ["ExpertDeviceCache", "ExpertDeviceCacheKey", "ExpertDeviceCacheValue"]
