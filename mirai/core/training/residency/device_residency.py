"""Shared byte-budget planning for single-device runtime residency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DeviceResidencyReservation:
    owner: str
    bytes: int


class DeviceResidencyPlanner:
    """Atomically account independent VRAM-residency owners under one ceiling."""

    def __init__(self, capacity_bytes: int = 0) -> None:
        capacity = int(capacity_bytes)
        if capacity < 0:
            raise ValueError("Device residency capacity must be >= 0 bytes.")
        self._capacity_bytes = capacity
        self._reservations: dict[str, int] = {}
        self._peak_reserved_bytes = 0

    @property
    def enabled(self) -> bool:
        return self._capacity_bytes > 0

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    def replace(self, owner: str, size_bytes: int) -> None:
        self.replace_many({str(owner): int(size_bytes)})

    def replace_many(self, reservations: Mapping[str, int]) -> None:
        candidate = dict(self._reservations)
        for owner, raw_size in reservations.items():
            name = str(owner).strip()
            size = int(raw_size)
            if not name:
                raise ValueError("Device residency reservation owner must be non-empty.")
            if size < 0:
                raise ValueError(
                    f"Device residency reservation '{name}' must be >= 0 bytes."
                )
            if size == 0:
                candidate.pop(name, None)
            else:
                candidate[name] = size
        total = sum(candidate.values())
        if self.enabled and total > self._capacity_bytes:
            detail = ", ".join(
                f"{name}={size}" for name, size in sorted(candidate.items())
            )
            raise MemoryError(
                "Device residency plan exceeds its configured ceiling: "
                f"reserved={total}, capacity={self._capacity_bytes} bytes "
                f"({detail})."
            )
        self._reservations = candidate
        self._peak_reserved_bytes = max(self._peak_reserved_bytes, total)

    def release(self, owner: str) -> None:
        self._reservations.pop(str(owner), None)

    def snapshot(self) -> dict[str, object]:
        reserved = sum(self._reservations.values())
        return {
            "enabled": self.enabled,
            "capacity_bytes": self._capacity_bytes,
            "reserved_bytes": reserved,
            "available_bytes": (
                max(0, self._capacity_bytes - reserved)
                if self.enabled
                else 0
            ),
            "peak_reserved_bytes": self._peak_reserved_bytes,
            "reservations": dict(sorted(self._reservations.items())),
        }


__all__ = ["DeviceResidencyPlanner", "DeviceResidencyReservation"]
