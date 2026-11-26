"""Strict optional GPUDirect Storage reads for packed expert ranges."""

from __future__ import annotations

import time
import threading
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.models.compressed_weights.packed.packed_storage_alignment import (
    GDS_STORAGE_ALIGNMENT_BYTES,
    plan_aligned_read,
    read_safetensors_storage_alignment,
)
from mirai.core.models.compressed_weights.packed.packed_stream_prefetch import PackedCudaTransfer


class GdsUnavailableError(RuntimeError):
    """Raised when explicit GDS mode cannot prove a direct storage path."""


@dataclass
class _GdsSlot:
    stream: Any | None = None
    buffer: Any | None = None
    busy: bool = False


class KvikioGdsReader:
    """Read shard ranges directly into CUDA storage through KvikIO/cuFile.

    KvikIO is an optional execution dependency. Explicit GDS mode
    rejects both KvikIO's POSIX backend and cuFile compatibility mode so a CPU
    bounce-buffer fallback cannot be reported as GPUDirect Storage.
    """

    def __init__(self, shard_paths: Mapping[str, Path], *, slots: int, slot_bytes: int) -> None:
        if slots < 1 or slot_bytes < GDS_STORAGE_ALIGNMENT_BYTES or slot_bytes % GDS_STORAGE_ALIGNMENT_BYTES:
            raise ValueError("GPUDirect Storage requires a positive number of aligned destination slots.")
        try:
            import kvikio
            import kvikio.cufile_driver
            import kvikio.defaults
        except (ImportError, OSError) as exc:
            raise GdsUnavailableError(
                "memory.packed_stream_backend='gds' requires optional "
                "'kvikio-cu12' (or the matching CUDA major)."
            ) from exc

        self._kvikio = kvikio
        self._driver = kvikio.cufile_driver
        self._defaults = kvikio.defaults
        self._compat_context: Any | None = None
        self._files: dict[str, Any] = {}
        self._file_sizes: dict[str, int] = {}
        self._closed = False
        self._stats_lock = threading.Lock()
        self._slots = [_GdsSlot() for _ in range(slots)]
        self._slot_bytes = slot_bytes
        self._raw_arena = os.environ.get("MIRAI_PACKED_CUDA_ARENA", "") == "raw"
        self._arena = None
        self._arena_lock = threading.Lock()
        self._slot_device: int | None = None
        self._slot_condition = threading.Condition()
        self._async_ops = 0
        self._read_ops = 0
        self._read_bytes = 0
        self._requested_bytes = 0
        self._overread_bytes = 0
        self._read_seconds = 0.0
        self._trimmed_ops = 0

        try:
            self._compat_context = self._defaults.set("compat_mode", False)
            self._compat_context.__enter__()
            try:
                self._driver.initialize()
            except (OSError, RuntimeError) as exc:
                raise GdsUnavailableError("Could not initialize the cuFile driver.") from exc
            if not bool(self._driver.get("is_gds_available")):
                raise GdsUnavailableError(
                    "cuFile reports that GPUDirect Storage is unavailable. "
                    "Install/configure the GDS driver or CUDA P2P path for this "
                    "GPU, filesystem, and mount."
                )
            if bool(self._driver.get("allow_compat_mode")):
                raise GdsUnavailableError(
                    "cuFile compatibility mode is enabled; Mirai refuses to "
                    "label its CPU POSIX fallback as GPUDirect Storage. Disable "
                    "allow_compat_mode in the active cufile configuration."
                )
            for name, path in shard_paths.items():
                try:
                    storage = read_safetensors_storage_alignment(path)
                except (OSError, ValueError) as exc:
                    raise GdsUnavailableError(str(exc)) from exc
                handle = self._kvikio.CuFile(str(path), "r")
                if not callable(getattr(handle, "raw_read_async", None)):
                    handle.close()
                    raise GdsUnavailableError("GPUDirect Storage requires KvikIO's raw_read_async API.")
                if not bool(handle.is_direct_io_supported()):
                    handle.close()
                    raise GdsUnavailableError(
                        f"Packed shard {path.name!r} does not support direct I/O "
                        "on its current filesystem/mount."
                    )
                self._files[str(name)] = handle
                self._file_sizes[str(name)] = storage.file_size
        except Exception:
            self.close()
            raise

    def read_device(
        self,
        shard: str,
        *,
        offset: int,
        nbytes: int,
        device: Any,
        destination: Any | None = None,
    ) -> Any:
        """Read one byte range and wait only for its issuing storage stream."""

        if self._closed:
            raise RuntimeError("GPUDirect Storage reader is closed.")
        if int(nbytes) <= 0:
            raise ValueError("GPUDirect Storage read size must be positive.")
        import torch

        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("GPUDirect Storage requires a CUDA target device.")
        device_index = (
            int(target.index)
            if target.index is not None
            else int(torch.cuda.current_device())
        )
        try:
            handle = self._files[str(shard)]
            file_size = self._file_sizes[str(shard)]
        except KeyError as exc:
            raise KeyError(str(shard)) from exc
        window = plan_aligned_read(
            offset=int(offset),
            nbytes=int(nbytes),
            file_size=file_size,
        )

        started = time.perf_counter()
        if window.read_bytes > self._slot_bytes:
            raise ValueError("GPUDirect Storage range exceeds its aligned slot capacity.")
        consumer = None
        writable = None
        if destination is not None:
            consumer = torch.cuda.current_stream(target)
            if (
                destination.device != consumer.device or destination.dtype != torch.uint8
                or destination.shape != (window.payload_bytes,) or not destination.is_contiguous()
            ):
                raise ValueError("GPUDirect Storage destination must be a contiguous CUDA byte range.")
            writable = torch.cuda.Event()
            writable.record(consumer)
        with self._acquire_slot(device_index) as (index, slot):
            if slot.stream is None:
                slot.stream = torch.cuda.Stream(device=device_index)
            stream = slot.stream
            with torch.cuda.stream(stream):
                if slot.buffer is None:
                    slot.buffer = self._allocate_slot(index, target)
                buffer = slot.buffer[:window.read_bytes]
                try:
                    # Keep the request arguments alive through local stream completion.
                    pending = handle.raw_read_async(
                        buffer, raw_stream=int(stream.cuda_stream),
                        size=window.read_bytes, file_offset=window.file_offset,
                    )
                    count = int(pending.check_bytes_done())
                except BaseException:
                    stream.synchronize()
                    raise
                if count != window.read_bytes:
                    raise EOFError(
                        f"GPUDirect Storage read returned {count} of {window.read_bytes} aligned bytes."
                    )
                view = buffer[window.payload_offset:window.payload_offset + window.payload_bytes]
                try:
                    if writable is None:
                        payload = view.clone()
                    else:
                        stream.wait_event(writable)
                        destination.copy_(view)
                        payload = destination
                    ready = torch.cuda.Event()
                    ready.record(stream)
                    if consumer is not None:
                        consumer.wait_event(ready)
                except BaseException:
                    stream.synchronize()
                    raise
        elapsed = time.perf_counter() - started
        with self._stats_lock:
            self._read_ops += 1
            self._async_ops += 1
            self._read_bytes += window.read_bytes
            self._requested_bytes += window.payload_bytes
            self._overread_bytes += window.overread_bytes
            self._read_seconds += float(elapsed)
            self._trimmed_ops += int(window.overread_bytes > 0)
        return PackedCudaTransfer(payload, ready)

    @contextmanager
    def _acquire_slot(self, device_index: int):
        with self._slot_condition:
            if self._slot_device is not None and self._slot_device != device_index:
                raise ValueError("GPUDirect Storage slots are bound to one CUDA device.")
            self._slot_condition.wait_for(lambda: self._closed or any(not slot.busy for slot in self._slots))
            if self._closed:
                raise RuntimeError("GPUDirect Storage reader is closed.")
            self._slot_device = device_index
            index, slot = next((index, slot) for index, slot in enumerate(self._slots) if not slot.busy)
            slot.busy = True
        try:
            yield index, slot
        finally:
            with self._slot_condition:
                slot.busy = False
                self._slot_condition.notify_all()

    def _allocate_slot(self, index: int, target: Any):
        if self._raw_arena:
            from .fixed_cuda_arena import FixedCudaArena

            with self._arena_lock:
                if self._arena is None:
                    self._arena = FixedCudaArena(
                        device=target, slots=len(self._slots), slot_bytes=self._slot_bytes,
                        alignment=GDS_STORAGE_ALIGNMENT_BYTES,
                    )
                return self._arena.slot(index)
        import torch

        allocation = torch.empty(
            self._slot_bytes + GDS_STORAGE_ALIGNMENT_BYTES - 1,
            dtype=torch.uint8, device=target,
        )
        padding = -allocation.data_ptr() % GDS_STORAGE_ALIGNMENT_BYTES
        return allocation[padding:padding + self._slot_bytes]

    def snapshot(self) -> dict[str, int | float | str]:
        version = self._kvikio.cufile_driver.libcufile_version()
        with self._stats_lock:
            return {
                "gds_provider": "kvikio",
                "gds_libcufile_version": ".".join(str(value) for value in version),
                "gds_compatibility_mode": 0,
                "gds_direct_shards": len(self._files),
                "gds_read_ops": int(self._read_ops),
                "gds_async_read_ops": int(self._async_ops),
                "gds_read_bytes": int(self._read_bytes),
                "gds_requested_bytes": int(self._requested_bytes),
                "gds_overread_bytes": int(self._overread_bytes),
                "gds_read_seconds": float(self._read_seconds),
                "read_seconds": float(self._read_seconds),
                "gds_alignment_bytes": GDS_STORAGE_ALIGNMENT_BYTES,
                "gds_trimmed_ops": int(self._trimmed_ops),
                "gds_ring_slots": len(self._slots),
                "gds_ring_slot_bytes": self._slot_bytes,
                "gds_raw_arena": int(self._raw_arena),
            }

    def close(self) -> None:
        with self._slot_condition:
            if self._closed:
                return
            self._closed = True
            self._slot_condition.notify_all()
            self._slot_condition.wait_for(lambda: not any(slot.busy for slot in self._slots))
        try:
            for slot in self._slots:
                if slot.stream is not None:
                    slot.stream.synchronize()
        finally:
            self._slots.clear()
            self._arena = None
            for handle in self._files.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self._files.clear()
            self._file_sizes.clear()
            if self._compat_context is not None:
                try:
                    self._compat_context.__exit__(None, None, None)
                except Exception:
                    pass
                self._compat_context = None
