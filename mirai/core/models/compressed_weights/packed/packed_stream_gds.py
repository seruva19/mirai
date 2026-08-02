"""Strict optional GPUDirect Storage reads for packed expert ranges."""

from __future__ import annotations

import time
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mirai.core.models.compressed_weights.packed.packed_storage_alignment import (
    GDS_STORAGE_ALIGNMENT_BYTES,
    plan_aligned_read,
    read_safetensors_storage_alignment,
)


class GdsUnavailableError(RuntimeError):
    """Raised when explicit GDS mode cannot prove a direct storage path."""


class KvikioGdsReader:
    """Read shard ranges directly into CUDA storage through KvikIO/cuFile.

    KvikIO and CuPy are optional execution dependencies. Explicit GDS mode
    rejects both KvikIO's POSIX backend and cuFile compatibility mode so a CPU
    bounce-buffer fallback cannot be reported as GPUDirect Storage.
    """

    def __init__(self, shard_paths: Mapping[str, Path]) -> None:
        try:
            import cupy
            import kvikio
            import kvikio.cufile_driver
            import kvikio.defaults
        except (ImportError, OSError) as exc:
            raise GdsUnavailableError(
                "memory.packed_stream_backend='gds' requires optional "
                "'kvikio-cu12' (or the matching CUDA major) and CuPy."
            ) from exc

        self._cupy = cupy
        self._kvikio = kvikio
        self._driver = kvikio.cufile_driver
        self._defaults = kvikio.defaults
        self._compat_context: Any | None = None
        self._files: dict[str, Any] = {}
        self._file_sizes: dict[str, int] = {}
        self._closed = False
        self._stats_lock = threading.Lock()
        self._read_ops = 0
        self._read_bytes = 0
        self._requested_bytes = 0
        self._overread_bytes = 0
        self._read_seconds = 0.0
        self._trimmed_ops = 0

        try:
            self._compat_context = self._defaults.set("compat_mode", False)
            self._compat_context.__enter__()
            self._driver.initialize()
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
    ) -> Any:
        """Synchronously read one exact byte range into a CUDA uint8 tensor."""

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
        with self._cupy.cuda.Device(device_index):
            allocation = self._cupy.empty(
                window.read_bytes + GDS_STORAGE_ALIGNMENT_BYTES - 1,
                dtype=self._cupy.uint8,
            )
            pointer_padding = (
                -int(allocation.data.ptr)
            ) % GDS_STORAGE_ALIGNMENT_BYTES
            buffer = allocation[
                pointer_padding:pointer_padding + window.read_bytes
            ]
            count = int(
                handle.read(
                    buffer,
                    size=window.read_bytes,
                    file_offset=window.file_offset,
                )
            )
            self._cupy.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - started
        if count != window.read_bytes:
            raise EOFError(
                f"GPUDirect Storage read returned {count} of "
                f"{window.read_bytes} aligned bytes."
            )
        with self._stats_lock:
            self._read_ops += 1
            self._read_bytes += window.read_bytes
            self._requested_bytes += window.payload_bytes
            self._overread_bytes += window.overread_bytes
            self._read_seconds += float(elapsed)
            self._trimmed_ops += int(window.overread_bytes > 0)
        payload = buffer[
            window.payload_offset:window.payload_offset + window.payload_bytes
        ]
        return torch.from_dlpack(payload)

    def snapshot(self) -> dict[str, int | float | str]:
        version = self._kvikio.cufile_driver.libcufile_version()
        with self._stats_lock:
            return {
                "gds_provider": "kvikio",
                "gds_libcufile_version": ".".join(str(value) for value in version),
                "gds_compatibility_mode": 0,
                "gds_direct_shards": len(self._files),
                "gds_read_ops": int(self._read_ops),
                "gds_read_bytes": int(self._read_bytes),
                "gds_requested_bytes": int(self._requested_bytes),
                "gds_overread_bytes": int(self._overread_bytes),
                "gds_read_seconds": float(self._read_seconds),
                "read_seconds": float(self._read_seconds),
                "gds_alignment_bytes": GDS_STORAGE_ALIGNMENT_BYTES,
                "gds_trimmed_ops": int(self._trimmed_ops),
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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
