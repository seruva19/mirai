"""Platform I/O and request-planning helpers for packed disk streaming."""

from __future__ import annotations

import errno
import io
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class PackedShardReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file, self.noatime = open_read_only(path)
        self.lock = threading.Lock()
        self.random_advice = advise_random(self.file.fileno())

    def readinto(self, buffer: memoryview, offset: int) -> None:
        target = buffer.cast("B")
        if hasattr(os, "preadv"):
            position = 0
            while position < len(target):
                count = os.preadv(
                    self.file.fileno(), [target[position:]], int(offset) + position
                )
                if count <= 0:
                    raise EOFError(
                        f"Packed tensor read ended at {position} of {len(target)} bytes."
                    )
                position += int(count)
            return
        with self.lock:
            self.file.seek(int(offset))
            position = 0
            while position < len(target):
                count = self.file.readinto(target[position:])
                if count is None or count <= 0:
                    raise EOFError(
                        f"Packed tensor read ended at {position} of {len(target)} bytes."
                    )
                position += int(count)

    def close(self) -> None:
        self.file.close()

    def evict_file_cache(self) -> bool:
        advise = getattr(os, "posix_fadvise", None)
        advice = getattr(os, "POSIX_FADV_DONTNEED", None)
        if advise is None or advice is None:
            return False
        advise(self.file.fileno(), 0, 0, advice)
        return True


def pin_budget_bytes() -> int:
    try:
        from mirai.core.training.residency.memory_safety import current_memory_safety_policy

        gib = float(current_memory_safety_policy().max_pinned_host_gib)
    except Exception:  # pragma: no cover
        return 0
    return max(0, int(gib * (1024**3)))


def open_read_only(path: Path) -> tuple[io.FileIO, bool]:
    """Open a shard without access-time writes when the platform permits it."""

    base_flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    close_on_exec = int(getattr(os, "O_CLOEXEC", 0))
    noatime = int(getattr(os, "O_NOATIME", 0))
    if noatime:
        try:
            descriptor = os.open(path, base_flags | close_on_exec | noatime)
            return io.FileIO(descriptor, mode="rb", closefd=True), True
        except OSError as exc:
            fallback_errors = {
                errno.EACCES,
                errno.EINVAL,
                errno.EPERM,
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if exc.errno not in fallback_errors:
                raise
    descriptor = os.open(path, base_flags | close_on_exec)
    return io.FileIO(descriptor, mode="rb", closefd=True), False


def advise_random(descriptor: int) -> bool:
    """Disable speculative sequential read-ahead for sparse expert access."""

    advise = getattr(os, "posix_fadvise", None)
    advice = getattr(os, "POSIX_FADV_RANDOM", None)
    if advise is None or advice is None:
        return False
    try:
        advise(int(descriptor), 0, 0, int(advice))
    except OSError:
        return False
    return True


def contiguous_runs(
    indices: Sequence[int], *, max_count: int
) -> tuple[tuple[int, int], ...]:
    if not indices:
        return ()
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    count = 1
    for raw in indices[1:]:
        current = int(raw)
        if current == previous + 1 and count < int(max_count):
            count += 1
        else:
            runs.append((start, count))
            start = current
            count = 1
        previous = current
    runs.append((start, count))
    return tuple(runs)


def restore_order(
    requested: Sequence[int],
    unique_indices: Sequence[int],
) -> tuple[int, ...]:
    positions = {int(index): position for position, index in enumerate(unique_indices)}
    return tuple(positions[int(index)] for index in requested)


def stack_tensors(tensors: Sequence[Any]) -> Any:
    import torch

    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tuple(tensors), dim=0)
