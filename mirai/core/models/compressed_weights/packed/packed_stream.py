"""Bounded direct-range streaming for packed safetensors expert weights."""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .packed_stream_layout import PackedTensorRegion
from .packed_stream_layout import load_packed_regions
from .packed_stream_layout import torch_dtype
from .packed_stream_prefetch import PackedStreamPrefetchHost
from .packed_stream_prefetch import PackedStreamPrefetchRing
from .packed_stream_cache import PackedStreamHostCache
from .packed_stream_cache import packed_stream_cache_key
from .packed_stream_io import contiguous_runs as _contiguous_runs
from .packed_stream_io import PackedShardReader
from .packed_stream_io import pin_budget_bytes as _pin_budget_bytes
from .packed_stream_io import restore_order as _restore_order
from .packed_stream_io import stack_tensors as _stack_tensors

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_DEFAULT_READ_WORKERS = 4
_MAX_RANGE_BYTES = 32 * 1024 * 1024


class StreamingPackedTensorMapping(PackedStreamPrefetchHost, Mapping[str, Any]):
    """Stream expert-aligned safetensors ranges without resident payloads."""

    def __init__(
        self,
        path: str | Path,
        *,
        read_workers: int = _DEFAULT_READ_WORKERS,
        cache_capacity_bytes: int = 0,
        backend: str = "staged",
        prefetch_depth: int = 0,
    ):
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for packed tensor streaming.")
        self._path = Path(path)
        self._regions, shard_paths = load_packed_regions(self._path)
        self._backend = _normalize_backend(backend)
        depth = int(prefetch_depth)
        if depth < 0 or depth > 16:
            raise ValueError("Packed stream prefetch depth must be between 0 and 16.")
        if self._backend == "gds" and int(cache_capacity_bytes) > 0:
            raise ValueError(
                "GPUDirect Storage bypasses host RAM, so "
                "memory.packed_stream_cache_gib must be 0 when "
                "memory.packed_stream_backend='gds'."
            )
        self._shards = {
            name: PackedShardReader(shard_path)
            for name, shard_path in shard_paths.items()
        }
        self._gds: Any | None = None
        if self._backend == "gds":
            from .packed_stream_gds import KvikioGdsReader

            try:
                self._gds = KvikioGdsReader(shard_paths)
            except Exception:
                for shard in self._shards.values():
                    shard.close()
                raise
        workers = max(1, min(int(read_workers), _DEFAULT_READ_WORKERS))
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="mirai-packed-read"
        )
        self._closed = False
        self._stats_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._streams: dict[str, Any] = {}
        self._cache = PackedStreamHostCache(cache_capacity_bytes)
        self._prefetch: Any | None = None
        if depth:
            self._prefetch = PackedStreamPrefetchRing(depth)
        self._pin_cap_bytes = _pin_budget_bytes()
        self._pinned_in_flight = 0
        self._pinned_peak = 0
        self._read_ops = 0
        self._read_bytes = 0
        self._read_seconds = 0.0
        self._h2d_ops = 0
        self._h2d_bytes = 0
        self._batch_requests = 0
        self._requested_slices = 0
        self._unique_slices = 0
        self._reordered_batches = 0

    def __iter__(self) -> Iterator[str]:
        return iter(self._regions)

    def __len__(self) -> int:
        return len(self._regions)

    def __contains__(self, key: object) -> bool:
        return str(key) in self._regions

    def __getitem__(self, key: str) -> Any:
        region = self._region(key)
        return self._read_region(
            region,
            offset=region.offset,
            shape=region.shape,
            nbytes=region.nbytes,
        )

    def get_slice(self, key: str, index: int) -> Any:
        region = self._region(key)
        if not region.shape:
            raise IndexError(f"Packed tensor {key!r} is scalar and cannot be sliced.")
        extent = int(region.shape[0])
        resolved = int(index)
        if resolved < 0:
            resolved += extent
        if resolved < 0 or resolved >= extent:
            raise IndexError(index)
        cache_key = packed_stream_cache_key(
            str(key), (resolved,), access="slice"
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached[0]
        slice_nbytes = region.nbytes // max(extent, 1)
        tensor = self._read_region(
            region,
            offset=region.offset + resolved * slice_nbytes,
            shape=region.shape[1:],
            nbytes=slice_nbytes,
        )
        self._cache.put(cache_key, (tensor,))
        return tensor

    def tensor_nbytes(self, key: str) -> int:
        return int(self._region(key).nbytes)

    def _get_slices_to_device_now(
        self,
        key: str,
        indices: Sequence[int],
        *,
        device: Any,
        dtype: Any | None = None,
    ) -> Any:
        requested = self._normalize_indices(str(key), indices)
        if not requested:
            raise ValueError("Packed stream batch requires at least one expert index.")
        unique_indices = tuple(sorted(set(requested)))
        restore_order = _restore_order(requested, unique_indices)
        reordered = restore_order != tuple(range(len(requested)))
        with self._stats_lock:
            self._batch_requests += 1
            self._requested_slices += len(requested)
            self._unique_slices += len(unique_indices)
            self._reordered_batches += int(reordered)
        target = torch.device(device)
        runs = self._bounded_runs(str(key), unique_indices)
        if self._backend == "gds":
            if target.type != "cuda":
                raise ValueError(
                    "memory.packed_stream_backend='gds' requires a CUDA target."
                )
            device_tensors = [
                future.result()
                for future in (
                    self._executor.submit(
                        self._read_slice_run_to_device,
                        str(key),
                        start,
                        count,
                        target,
                    )
                    for start, count in runs
                )
            ]
            stacked = _stack_tensors(device_tensors)
            if dtype is not None:
                stacked = stacked.to(dtype=dtype)
            if reordered:
                stacked = stacked.index_select(
                    0,
                    torch.as_tensor(
                        restore_order,
                        device=target,
                        dtype=torch.long,
                    ),
                )
            return stacked

        cache_key = packed_stream_cache_key(str(key), unique_indices)
        cached = self._cache.get(cache_key)
        if cached is None:
            futures = [
                self._executor.submit(self._read_slice_run, str(key), start, count)
                for start, count in runs
            ]
            host_tensors = tuple(future.result() for future in futures)
            self._cache.put(cache_key, host_tensors)
        else:
            host_tensors = cached
        if target.type != "cuda":
            stacked = _stack_tensors(host_tensors)
            if reordered:
                stacked = stacked.index_select(
                    0, torch.as_tensor(restore_order, dtype=torch.long)
                )
            return (
                stacked.to(device=target, dtype=dtype)
                if dtype is not None
                else stacked
            )

        stream = self._cuda_stream(target)
        with torch.cuda.stream(stream):
            device_tensors = [
                tensor.to(
                    device=target,
                    dtype=dtype,
                    non_blocking=bool(tensor.is_pinned()),
                )
                for tensor in host_tensors
            ]
            stacked = _stack_tensors(device_tensors)
            if reordered:
                stacked = stacked.index_select(
                    0,
                    torch.as_tensor(
                        restore_order,
                        device=target,
                        dtype=torch.long,
                    ),
                )
            event = torch.cuda.Event()
            event.record(stream)
        current = torch.cuda.current_stream(target)
        current.wait_event(event)
        stacked.record_stream(current)
        transferred = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in host_tensors
        )
        with self._stats_lock:
            self._h2d_ops += len(host_tensors)
            self._h2d_bytes += transferred
        return stacked

    def snapshot(self) -> dict[str, int | float | str]:
        with self._stats_lock:
            snapshot: dict[str, int | float | str] = {
                "backend": "gds" if self._backend == "gds" else "direct_range",
                "read_ops": int(self._read_ops),
                "read_bytes": int(self._read_bytes),
                "read_seconds": float(self._read_seconds),
                "h2d_ops": int(self._h2d_ops),
                "h2d_bytes": int(self._h2d_bytes),
                "batch_requests": int(self._batch_requests),
                "requested_slices": int(self._requested_slices),
                "unique_slices": int(self._unique_slices),
                "reordered_batches": int(self._reordered_batches),
                "pinned_in_flight_bytes": int(self._pinned_in_flight),
                "pinned_peak_bytes": int(self._pinned_peak),
                "noatime_shards": sum(
                    int(shard.noatime) for shard in self._shards.values()
                ),
                "random_advice_shards": sum(
                    int(shard.random_advice) for shard in self._shards.values()
                ),
            }
        snapshot.update(self._cache.snapshot())
        if self._gds is not None:
            snapshot.update(self._gds.snapshot())
        if self._prefetch is not None:
            snapshot.update(self._prefetch.snapshot())
        return snapshot

    def _adopt_prefetched_result(self, value: Any, *, device: Any) -> Any:
        target = torch.device(device)
        if target.type != "cuda" or self._backend == "gds":
            return value
        current = torch.cuda.current_stream(target)
        current.wait_stream(self._cuda_stream(target))
        value.record_stream(current)
        return value

    def evict_file_cache(self) -> bool:
        """Request cold-cache reads where the OS exposes POSIX fadvise."""
        return self._backend != "gds" and all(
            shard.evict_file_cache() for shard in self._shards.values()
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._prefetch is not None:
            self._prefetch.close()
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._cache.clear()
        if self._gds is not None:
            self._gds.close()
        for shard in self._shards.values():
            shard.close()

    def __enter__(self) -> StreamingPackedTensorMapping:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive interpreter cleanup
        try:
            self.close()
        except Exception:
            pass

    def _region(self, key: str) -> PackedTensorRegion:
        if self._closed:
            raise RuntimeError("Packed tensor stream is closed.")
        try:
            return self._regions[str(key)]
        except KeyError as exc:
            raise KeyError(str(key)) from exc

    def _read_region(
        self,
        region: PackedTensorRegion,
        *,
        offset: int,
        shape: tuple[int, ...],
        nbytes: int,
    ) -> Any:
        buffer, pinned = self._allocate_bytes(int(nbytes))
        started = time.perf_counter()
        try:
            self._shards[region.shard].readinto(
                memoryview(buffer.numpy()), int(offset)
            )
        except Exception:
            if pinned:
                self._release_pinned(int(nbytes))
            raise
        elapsed = time.perf_counter() - started
        if pinned:
            weakref.finalize(buffer, self._release_pinned, int(nbytes))
        with self._stats_lock:
            self._read_ops += 1
            self._read_bytes += int(nbytes)
            self._read_seconds += float(elapsed)
        dtype = torch_dtype(region.dtype_name)
        return buffer.view(dtype).reshape(shape)

    def _read_slice_run(self, key: str, start: int, count: int) -> Any:
        region = self._region(key)
        if not region.shape:
            raise IndexError(f"Packed tensor {key!r} is scalar and cannot be sliced.")
        extent = int(region.shape[0])
        if start < 0 or count <= 0 or start + count > extent:
            raise IndexError((start, count))
        slice_nbytes = region.nbytes // max(extent, 1)
        return self._read_region(
            region,
            offset=region.offset + start * slice_nbytes,
            shape=(count, *region.shape[1:]),
            nbytes=count * slice_nbytes,
        )

    def _read_slice_run_to_device(
        self,
        key: str,
        start: int,
        count: int,
        device: Any,
    ) -> Any:
        region = self._region(key)
        if self._gds is None:
            raise RuntimeError("GPUDirect Storage reader is not configured.")
        if not region.shape:
            raise IndexError(f"Packed tensor {key!r} is scalar and cannot be sliced.")
        extent = int(region.shape[0])
        if start < 0 or count <= 0 or start + count > extent:
            raise IndexError((start, count))
        slice_nbytes = region.nbytes // max(extent, 1)
        nbytes = count * slice_nbytes
        buffer = self._gds.read_device(
            region.shard,
            offset=region.offset + start * slice_nbytes,
            nbytes=nbytes,
            device=device,
        )
        with self._stats_lock:
            self._read_ops += 1
            self._read_bytes += int(nbytes)
        return buffer.view(torch_dtype(region.dtype_name)).reshape(
            count, *region.shape[1:]
        )

    def _bounded_runs(
        self, key: str, indices: Sequence[int]
    ) -> tuple[tuple[int, int], ...]:
        region = self._region(key)
        if not region.shape:
            raise IndexError(f"Packed tensor {key!r} is scalar and cannot be sliced.")
        slice_nbytes = region.nbytes // max(int(region.shape[0]), 1)
        max_count = max(1, _MAX_RANGE_BYTES // max(slice_nbytes, 1))
        return _contiguous_runs(indices, max_count=max_count)

    def _normalize_indices(
        self, key: str, indices: Sequence[int]
    ) -> tuple[int, ...]:
        region = self._region(key)
        if not region.shape:
            raise IndexError(f"Packed tensor {key!r} is scalar and cannot be sliced.")
        extent = int(region.shape[0])
        normalized: list[int] = []
        for index in indices:
            resolved = int(index)
            if resolved < 0:
                resolved += extent
            if resolved < 0 or resolved >= extent:
                raise IndexError(index)
            normalized.append(resolved)
        return tuple(normalized)

    def _allocate_bytes(self, nbytes: int) -> tuple[Any, bool]:
        want_pin = False
        with self._stats_lock:
            if (
                torch.cuda.is_available()
                and self._pin_cap_bytes > 0
                and self._pinned_in_flight + int(nbytes) <= self._pin_cap_bytes
            ):
                self._pinned_in_flight += int(nbytes)
                self._pinned_peak = max(self._pinned_peak, self._pinned_in_flight)
                want_pin = True
        if want_pin:
            try:
                return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True), True
            except (RuntimeError, MemoryError, OSError):
                self._release_pinned(int(nbytes))
        return torch.empty(nbytes, dtype=torch.uint8), False

    def _release_pinned(self, nbytes: int) -> None:
        with self._stats_lock:
            self._pinned_in_flight = max(
                0, int(self._pinned_in_flight) - int(nbytes)
            )

    def _cuda_stream(self, device: Any) -> Any:
        key = str(device)
        with self._stream_lock:
            stream = self._streams.get(key)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                self._streams[key] = stream
            return stream


def _normalize_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {"direct_range": "staged", "kvikio": "gds"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"staged", "gds"}:
        raise ValueError("memory.packed_stream_backend must be one of: staged, gds.")
    return normalized
