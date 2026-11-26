"""Platform I/O and request-planning helpers for packed disk streaming."""

from __future__ import annotations

import errno
import io
import mmap
import os
import sys
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


class DirectPackedShardReader:
    """Linux O_DIRECT reader using aligned anonymous memory for each read."""

    noatime = False
    random_advice = False

    def __init__(self, path: Path) -> None:
        from .packed_storage_alignment import read_safetensors_storage_alignment

        if sys.platform != "linux" or not hasattr(os, "O_DIRECT") or not hasattr(os, "preadv"):
            raise RuntimeError("Packed direct_io requires Linux with O_DIRECT and preadv.")
        self.alignment = read_safetensors_storage_alignment(path)
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)

    def readinto(self, buffer: memoryview, offset: int) -> int:
        from .packed_storage_alignment import plan_aligned_read

        target = buffer.cast("B")
        if not len(target):
            return 0
        window = plan_aligned_read(
            offset=offset, nbytes=len(target), file_size=self.alignment.file_size,
            alignment_bytes=self.alignment.alignment_bytes,
        )
        with mmap.mmap(-1, window.read_bytes) as scratch, memoryview(scratch) as aligned:
            count = self._read_aligned(aligned, window.file_offset)
            if count != window.read_bytes:
                raise EOFError(f"Packed O_DIRECT read returned {count} of {window.read_bytes} bytes.")
            start = window.payload_offset
            target[:] = aligned[start:start + len(target)]
        return window.read_bytes

    def _read_aligned(self, buffer: memoryview, offset: int) -> int:
        return os.preadv(self.fd, [buffer], offset)

    def close(self) -> None:
        os.close(self.fd)

    def evict_file_cache(self) -> bool:
        return False


class IoUringPackedShardReader(DirectPackedShardReader):
    _FIXED_BYTES = 2 * 1024 * 1024

    @property
    def fixed_buffer_bytes(self) -> int:
        return self._FIXED_BYTES if self._fixed is not None else 0

    def __init__(self, path: Path, *, shared: IoUringPackedShardReader | None = None) -> None:
        import ctypes

        super().__init__(path)
        try:
            self._owns_ring = shared is None
            if shared is not None:
                self._bridge = shared._bridge
                self._handle = shared._handle
                self._ring_lock = shared._ring_lock
                self._fixed = shared._fixed
                self.mode = shared.mode
                return
            library = os.environ.get("MIRAI_PACKED_IO_URING_LIBRARY")
            if not library:
                raise RuntimeError("Set MIRAI_PACKED_IO_URING_LIBRARY to the compiled liburing bridge path.")
            self.mode = os.environ.get("MIRAI_PACKED_IO_URING_MODE", "legacy")
            if self.mode not in {"legacy", "fixed", "fixed_sqpoll", "fixed_sqpoll_hugepages"}:
                raise ValueError(f"Unsupported MIRAI_PACKED_IO_URING_MODE {self.mode!r}.")
            self._bridge = ctypes.CDLL(library)
            self._bridge.mirai_uring_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            self._bridge.mirai_uring_open.restype = ctypes.c_int
            self._bridge.mirai_uring_read.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64,
            ]
            self._bridge.mirai_uring_read.restype = ctypes.c_int
            self._bridge.mirai_uring_close.argtypes = [ctypes.c_void_p]
            self._bridge.mirai_uring_close.restype = None
            self._handle = ctypes.c_void_p()
            self._ring_lock = threading.Lock()
            self._fixed = None
            if self.mode == "legacy":
                result = self._bridge.mirai_uring_open(ctypes.byref(self._handle))
            else:
                try:
                    open_fixed = self._bridge.mirai_uring_open_fixed
                    fixed_buffer = self._bridge.mirai_uring_fixed_buffer
                    read_fixed = self._bridge.mirai_uring_read_fixed
                except AttributeError as exc:
                    raise RuntimeError("Fixed io_uring mode requires a bridge with registered-buffer support.") from exc
                open_fixed.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
                open_fixed.restype = ctypes.c_int
                fixed_buffer.argtypes = [ctypes.c_void_p]
                fixed_buffer.restype = ctypes.c_void_p
                read_fixed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_uint64]
                read_fixed.restype = ctypes.c_int
                result = open_fixed(
                    ctypes.byref(self._handle), self.mode != "fixed",
                    self.mode == "fixed_sqpoll_hugepages",
                )
                if result == 0:
                    address = fixed_buffer(self._handle)
                    if not address:
                        raise RuntimeError("io_uring bridge returned no registered buffer.")
                    self._fixed = (ctypes.c_ubyte * self._FIXED_BYTES).from_address(address)
            if result < 0:
                raise OSError(-result, f"io_uring {self.mode} initialization failed: {os.strerror(-result)}")
        except BaseException:
            if self._owns_ring and getattr(self, "_handle", None):
                self._bridge.mirai_uring_close(self._handle)
            super().close()
            raise

    def readinto(self, buffer: memoryview, offset: int) -> int:
        if self._fixed is None:
            return super().readinto(buffer, offset)
        from .packed_storage_alignment import plan_aligned_read

        target = buffer.cast("B")
        if not len(target):
            return 0
        window = plan_aligned_read(
            offset=offset, nbytes=len(target), file_size=self.alignment.file_size,
            alignment_bytes=self.alignment.alignment_bytes,
        )
        with self._ring_lock, memoryview(self._fixed).cast("B") as fixed:
            payload_start = window.payload_offset
            payload_end = payload_start + len(target)
            for start in range(0, window.read_bytes, self._FIXED_BYTES):
                length = min(self._FIXED_BYTES, window.read_bytes - start)
                count = self._bridge.mirai_uring_read_fixed(
                    self._handle, self.fd, length, window.file_offset + start,
                )
                if count < 0:
                    raise OSError(-count, f"io_uring fixed read failed: {os.strerror(-count)}")
                if count != length:
                    raise EOFError(f"Packed io_uring fixed read returned {count} of {length} bytes.")
                first = max(start, payload_start)
                last = min(start + length, payload_end)
                target[first - payload_start:last - payload_start] = fixed[first - start:last - start]
        return window.read_bytes

    def _read_aligned(self, buffer: memoryview, offset: int) -> int:
        import ctypes

        with self._ring_lock:
            address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
            result = self._bridge.mirai_uring_read(self._handle, self.fd, address, len(buffer), offset)
        if result < 0:
            raise OSError(-result, f"io_uring read failed: {os.strerror(-result)}")
        return result

    def close(self) -> None:
        with self._ring_lock:
            if self._owns_ring:
                self._bridge.mirai_uring_close(self._handle)
            super().close()


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
