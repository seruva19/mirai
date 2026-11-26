"""Bounded direct-range streaming for packed safetensors expert weights."""

from __future__ import annotations

import threading
import time
import os
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from .packed_stream_layout import PackedTensorRegion
from .packed_stream_layout import load_packed_regions
from .packed_stream_layout import torch_dtype
from .packed_stream_prefetch import PackedCudaSlotTransfer, PackedCudaTransfer, PackedStreamPrefetchHost
from .packed_stream_prefetch import PackedStreamPrefetchRing
from .packed_stream_prefetch import packed_prefetch_identity
from .packed_stream_cache import PackedStreamHostCache
from .packed_stream_cache import packed_stream_cache_key
from .packed_stream_io import contiguous_runs as _contiguous_runs
from .packed_stream_io import DirectPackedShardReader, IoUringPackedShardReader, PackedShardReader
from .packed_stream_io import pin_budget_bytes as _pin_budget_bytes
from .packed_stream_io import restore_order as _restore_order
from .packed_stream_io import stack_tensors as _stack_tensors

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_DEFAULT_READ_WORKERS = 4
_MAX_RANGE_BYTES = 32 * 1024 * 1024
# Backward holds three dense projections while decoding the last packed input.
_CUDA_STAGING_SLOTS = 4
_CUDA_PREFETCH_DENSE_SLOTS = 3


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
        tensor_aliases: Mapping[str, tuple[str, str]] | None = None,
        cache_expert_slices: bool = False,
    ):
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for packed tensor streaming.")
        self._path = Path(path)
        self._regions, shard_paths = load_packed_regions(self._path, tensor_aliases=tensor_aliases)
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
        self._shards = {}
        shard_reader = {"direct_io": DirectPackedShardReader, "io_uring": IoUringPackedShardReader}.get(
            self._backend, PackedShardReader,
        )
        fixed_reader = None
        try:
            for name, shard_path in shard_paths.items():
                if self._backend == "io_uring":
                    reader = IoUringPackedShardReader(shard_path, shared=fixed_reader)
                    if reader.mode != "legacy" and fixed_reader is None:
                        fixed_reader = reader
                else:
                    reader = shard_reader(shard_path)
                self._shards[name] = reader
        except Exception:
            for shard in self._shards.values():
                shard.close()
            raise
        self._gds: Any | None = None
        workers = max(1, min(int(read_workers), _DEFAULT_READ_WORKERS))
        if self._backend == "gds":
            from .packed_stream_gds import KvikioGdsReader
            from .packed_storage_alignment import GDS_STORAGE_ALIGNMENT_BYTES

            cap = max(
                (max(min(region.nbytes, _MAX_RANGE_BYTES), region.nbytes // region.shape[0])
                 for region in self._regions.values() if region.shape and region.shape[0]),
                default=0,
            )
            alignment = GDS_STORAGE_ALIGNMENT_BYTES
            slot_bytes = ((cap + 2 * alignment - 2) // alignment) * alignment
            try:
                self._gds = KvikioGdsReader(shard_paths, slots=workers, slot_bytes=slot_bytes)
            except Exception:
                for shard in self._shards.values():
                    shard.close()
                raise
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="mirai-packed-read"
        )
        self._closed = False
        self._stats_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._streams: dict[str, Any] = {}
        self._cache = PackedStreamHostCache(cache_capacity_bytes)
        self._cache_expert_slices = bool(cache_expert_slices)
        self._warm_lock = threading.Lock()
        self._warm_executor: ThreadPoolExecutor | None = None
        self._warm_future: Future[None] | None = None
        self._warm_submitted = 0
        self._warm_dropped = 0
        self._warm_failed = 0
        self._prefetch: Any | None = None
        if depth:
            self._prefetch = PackedStreamPrefetchRing(depth)
        self._uring_fixed_bytes = 0 if fixed_reader is None else fixed_reader.fixed_buffer_bytes
        self._pin_cap_bytes = max(0, _pin_budget_bytes() - self._uring_fixed_bytes)
        self._staging_lock = threading.RLock()
        self._staging_pool = None
        self._raw_staging = os.environ.get("MIRAI_PACKED_CUDA_ARENA", "") == "raw"
        self._staging_arena = None
        self._staging_arena_device = None
        self._prefetch_dense_lock = threading.RLock()
        self._prefetch_dense_arena = None
        self._prefetch_dense_device = None
        self._prefetch_packed_lock = threading.Lock()
        self._prefetch_packed_arena = None
        self._prefetch_packed_device = None
        self._staging_unavailable = False
        self._pinned_in_flight = 0
        self._pinned_peak = 0
        self._read_ops = 0
        self._read_bytes = 0
        self._direct_io_read_bytes = 0
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

    @property
    def prefetch_enabled(self) -> bool:
        return self._prefetch is not None

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
            futures = [
                self._executor.submit(self._read_slice_run_to_device, str(key), start, count, target)
                for start, count in runs
            ]
            transfers = [future.result() for future in futures]
            stream = self._cuda_stream(target)
            with torch.cuda.stream(stream):
                for transfer in transfers:
                    stream.wait_event(transfer.ready)
                    transfer.tensor.record_stream(stream)
                stacked = _stack_tensors([transfer.tensor for transfer in transfers])
                if dtype is not None:
                    stacked = stacked.to(dtype=dtype)
                if reordered:
                    stacked = stacked.index_select(
                        0, torch.as_tensor(restore_order, device=target, dtype=torch.long),
                    )
                event = torch.cuda.Event()
                event.record(stream)
            return PackedCudaTransfer(stacked, event)

        if self._cache_expert_slices and self._cache.enabled:
            host_tensors = (self._host_expert_slice_batch(str(key), unique_indices),)
        else:
            cache_key = packed_stream_cache_key(str(key), unique_indices)
            host_tensors = self._cache.get(cache_key)
            if host_tensors is None:
                futures = [
                    self._executor.submit(self._read_slice_run, str(key), start, count)
                    for start, count in runs
                ]
                host_tensors = tuple(future.result() for future in futures)
                self._cache.put(cache_key, host_tensors)
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
            device_tensors = self._copy_staged_to_device(host_tensors, target, dtype, stream)
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
        transferred = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in host_tensors
        )
        with self._stats_lock:
            self._h2d_ops += len(host_tensors)
            self._h2d_bytes += transferred
        return PackedCudaTransfer(stacked, event)

    @contextmanager
    def borrow_cuda_weight(self, shape, *, dtype, device):
        target = torch.device(device)
        if not self._raw_staging or target.type != "cuda":
            yield None
            return
        size = torch.Size(shape).numel() * torch.empty((), device="meta", dtype=dtype).element_size()
        if size > _MAX_RANGE_BYTES:
            yield None
            return
        # A demand consumer may wait for a prefetch worker that uses staging.
        lock = self._prefetch_dense_lock if self.prefetch_enabled else self._staging_lock
        with lock:
            if self._closed:
                raise RuntimeError("Packed stream mapping is closed.")
            stream = torch.cuda.current_stream(target)
            arena = (
                self._cuda_prefetch_dense_arena(target, stream)
                if self.prefetch_enabled else self._cuda_staging_arena(target, stream)
            )
            with arena.lease(stream) as slot:
                yield slot[:size].view(dtype).view(shape)

    @property
    def cuda_weight_slot_bytes(self) -> int:
        return _MAX_RANGE_BYTES if self._raw_staging else 0

    @contextmanager
    def borrow_packed_fields(self, keys, indices, *, device):
        """Supply short-lived packed fields on the current CUDA stream."""
        target = torch.device(device)
        if self._raw_staging and target.type == "cuda" and self.prefetch_enabled:
            with self._prefetch_dense_lock, ExitStack() as leases:
                if self._closed:
                    raise RuntimeError("Packed stream mapping is closed.")
                yield [
                    leases.enter_context(self._borrow_slices_to_device(key, indices, device=target))[0]
                    for key in keys
                ]
            return
        if self._raw_staging and target.type == "cuda" and not self.prefetch_enabled:
            indices = tuple(indices)
            regions = [self._region(key) for key in keys]
            offsets = []
            size = 0
            for key, region in zip(keys, regions, strict=True):
                if not region.shape:
                    raise IndexError(f"Packed tensor {key!r} is scalar and cannot be sliced.")
                size = (size + 15) // 16 * 16
                offsets.append(size)
                size += region.nbytes // max(region.shape[0], 1) * len(indices)
            if size <= _MAX_RANGE_BYTES:
                with self._staging_lock:
                    if self._closed:
                        raise RuntimeError("Packed stream mapping is closed.")
                    stream = torch.cuda.current_stream(target)
                    arena = self._cuda_staging_arena(target, stream)
                    with arena.lease(stream) as slot:
                        if self._gds is not None:
                            yield [
                                self._read_borrowed_gds_field(key, region, indices, slot[offset:], target)
                                for key, region, offset in zip(keys, regions, offsets, strict=True)
                            ]
                            return
                        sources = [self.get_slices_to_device(key, indices, device="cpu") for key in keys]
                        fields = []
                        for source, offset in zip(sources, offsets, strict=True):
                            packed_bytes = source.numel() * source.element_size()
                            field = slot[offset:offset + packed_bytes].view(source.dtype).view(source.shape)
                            field.copy_(source)
                            fields.append(field)
                        with self._stats_lock:
                            self._h2d_ops += len(sources)
                            self._h2d_bytes += sum(source.numel() * source.element_size() for source in sources)
                        yield fields
                return
        yield [self.get_slices_to_device(key, indices, device=target) for key in keys]

    def get_slices_to_device(self, key, indices, *, device, dtype=None):
        if self._raw_staging and self.prefetch_enabled:
            with self._prefetch_dense_lock:
                if self._closed:
                    raise RuntimeError("Packed stream mapping is closed.")
                return super().get_slices_to_device(key, indices, device=device, dtype=dtype)
        return super().get_slices_to_device(key, indices, device=device, dtype=dtype)

    def prefetch_slices_to_device(self, key, indices, *, device, dtype=None):
        target = torch.device(device)
        if not self._raw_staging or self._prefetch is None or target.type != "cuda" or dtype is not None:
            return super().prefetch_slices_to_device(key, indices, device=device, dtype=dtype)
        requested = self._normalize_indices(str(key), indices)
        if not requested:
            raise ValueError("Packed stream prefetch requires at least one expert.")
        region = self._region(str(key))
        nbytes = region.nbytes // region.shape[0] * len(requested)
        if nbytes > _MAX_RANGE_BYTES:
            return super().prefetch_slices_to_device(key, indices, device=device, dtype=dtype)
        return self._prefetch.submit(
            packed_prefetch_identity(str(key), requested, device=target, dtype=None),
            lambda: self._load_prefetched_slot(str(key), region, requested, target, nbytes),
            discard=lambda transfer: transfer.release() if transfer is not None else None,
        )

    def _load_prefetched_slot(self, key, region, indices, target, nbytes):
        from .fixed_cuda_arena import FixedCudaArena

        producer = self._cuda_stream(target)
        with self._prefetch_packed_lock:
            if self._prefetch_packed_arena is None:
                self._prefetch_packed_arena = FixedCudaArena(
                    device=target, slots=_CUDA_STAGING_SLOTS, slot_bytes=_MAX_RANGE_BYTES, alignment=4096,
                )
                self._prefetch_packed_device = producer.device
            elif self._prefetch_packed_device != producer.device:
                raise ValueError("Packed CUDA prefetch slots are bound to one device.")
            arena = self._prefetch_packed_arena
        with torch.cuda.stream(producer), ExitStack() as ownership:
            slot = ownership.enter_context(arena.lease(producer, wait=False))
            if slot is None:
                return None
            if self._gds is not None:
                tensor = self._read_borrowed_gds_field(key, region, indices, slot, target)
            else:
                source = self._get_slices_to_device_now(key, indices, device="cpu")
                tensor = slot[:nbytes].view(source.dtype).view(source.shape)
                tensor.copy_(source)
                with self._stats_lock:
                    self._h2d_ops += 1
                    self._h2d_bytes += nbytes
            ready = torch.cuda.Event()
            ready.record(producer)
            return PackedCudaSlotTransfer(tensor, ready, producer, ownership.pop_all().close)

    def _read_borrowed_gds_field(self, key, region, indices, slot, device):
        requested = self._normalize_indices(key, indices)
        if not requested:
            raise ValueError("Packed stream batch requires at least one expert index.")
        slice_bytes = region.nbytes // region.shape[0]
        destination = slot[:len(requested) * slice_bytes]
        cursor = 0
        for start, count in self._bounded_runs(key, requested):
            nbytes = count * slice_bytes
            self._gds.read_device(
                region.shard, offset=region.offset + start * slice_bytes, nbytes=nbytes, device=device,
                destination=destination[cursor:cursor + nbytes],
            )
            cursor += nbytes
            with self._stats_lock:
                self._read_ops += 1
                self._read_bytes += nbytes
        return destination.view(torch_dtype(region.dtype_name)).reshape(len(requested), *region.shape[1:])

    def _host_expert_slice_batch(self, key, indices):
        cache_keys = {index: packed_stream_cache_key(key, (index,), access="slice") for index in indices}
        tensors = {index: self._cache.get(cache_key) for index, cache_key in cache_keys.items()}
        missing = tuple(index for index in indices if tensors[index] is None)
        runs = [
            (start, self._executor.submit(self._read_slice_run, key, start, count))
            for start, count in self._bounded_runs(key, missing)
        ]
        for start, future in runs:
            for offset, tensor in enumerate(future.result()):
                index = start + offset
                # Each cached slice must own its storage so byte accounting
                # stays accurate after other slices from the same run expire.
                entry = (tensor.clone(),)
                tensors[index] = entry
                self._cache.put(cache_keys[index], entry)
        return torch.stack([tensors[index][0] for index in indices])

    def warm_expert_cache(self, requests: Mapping[str, Sequence[int]]) -> bool:
        """Offer one bounded, best-effort batch of speculative host reads."""
        if self._gds is not None or not self._cache_expert_slices or not self._cache.enabled:
            return False
        normalized = {
            key: tuple(sorted(set(self._normalize_indices(key, ids))))
            for key, ids in requests.items() if ids
        }
        if not normalized:
            return False
        with self._warm_lock:
            if self._closed:
                return False
            if self._warm_future is not None and not self._warm_future.done():
                self._warm_dropped += 1
                return False
            if self._warm_executor is None:
                self._warm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mirai-packed-warm")
            self._warm_future = self._warm_executor.submit(self._warm_expert_cache, normalized)
            self._warm_submitted += 1
            return True

    def _warm_expert_cache(self, requests: Mapping[str, Sequence[int]]) -> None:
        try:
            for key, indices in requests.items():
                missing = tuple(
                    index for index in indices
                    if not self._cache.contains(packed_stream_cache_key(key, (index,), access="slice"))
                )
                for start, count in self._bounded_runs(key, missing):
                    if self._closed:
                        return
                    for offset, tensor in enumerate(self._read_slice_run(key, start, count)):
                        cache_key = packed_stream_cache_key(key, (start + offset,), access="slice")
                        self._cache.put(cache_key, (tensor.clone(),))
        except Exception:
            # The demand path retries the read and owns error reporting for required weights.
            with self._warm_lock:
                self._warm_failed += 1

    def snapshot(self) -> dict[str, int | float | str]:
        with self._stats_lock:
            snapshot: dict[str, int | float | str] = {
                "backend": "direct_range" if self._backend == "staged" else self._backend,
                "read_ops": int(self._read_ops),
                "read_bytes": int(self._read_bytes),
                "read_seconds": float(self._read_seconds),
                "h2d_ops": int(self._h2d_ops),
                "h2d_bytes": int(self._h2d_bytes),
                "batch_requests": int(self._batch_requests),
                "requested_slices": int(self._requested_slices),
                "unique_slices": int(self._unique_slices),
                "reordered_batches": int(self._reordered_batches),
                "pinned_in_flight_bytes": self._pinned_in_flight,
                "pinned_peak_bytes": int(self._pinned_peak),
                "noatime_shards": sum(
                    int(shard.noatime) for shard in self._shards.values()
                ),
                "random_advice_shards": sum(
                    int(shard.random_advice) for shard in self._shards.values()
                ),
            }
            if self._backend in {"direct_io", "io_uring"}:
                snapshot["direct_io_read_bytes"] = self._direct_io_read_bytes
                snapshot["direct_io_overread_bytes"] = self._direct_io_read_bytes - self._read_bytes
            if self._backend == "io_uring":
                snapshot["io_uring_mode"] = next(iter(self._shards.values())).mode
                snapshot["io_uring_fixed_buffer_bytes"] = self._uring_fixed_bytes
            if self._raw_staging:
                snapshot["cuda_staging_arena_bytes"] = (
                    _CUDA_STAGING_SLOTS * _MAX_RANGE_BYTES if self._staging_arena is not None else 0
                )
                snapshot["cuda_prefetch_dense_arena_bytes"] = (
                    _CUDA_PREFETCH_DENSE_SLOTS * _MAX_RANGE_BYTES if self._prefetch_dense_arena is not None else 0
                )
                snapshot["cuda_prefetch_packed_arena_bytes"] = (
                    _CUDA_STAGING_SLOTS * _MAX_RANGE_BYTES if self._prefetch_packed_arena is not None else 0
                )
        snapshot.update(self._cache.snapshot())
        if self._gds is not None:
            snapshot.update(self._gds.snapshot())
        if self._prefetch is not None:
            snapshot.update(self._prefetch.snapshot())
        with self._warm_lock:
            if self._warm_submitted:
                snapshot.update(
                    speculative_warm_submitted=self._warm_submitted,
                    speculative_warm_dropped=self._warm_dropped,
                    speculative_warm_failed=self._warm_failed,
                )
        return snapshot

    def evict_file_cache(self) -> bool:
        """Request cold-cache reads where the OS exposes POSIX fadvise."""
        return self._backend != "gds" and all(
            shard.evict_file_cache() for shard in self._shards.values()
        )

    def close(self) -> None:
        with self._warm_lock:
            if self._closed:
                return
            self._closed = True
            warmer = self._warm_executor
        if warmer is not None:
            warmer.shutdown(wait=True, cancel_futures=True)
        with self._prefetch_dense_lock:
            if self._prefetch_dense_arena is not None:
                self._prefetch_dense_arena.synchronize()
                self._prefetch_dense_arena = None
        if self._prefetch is not None:
            self._prefetch.close()
        with self._prefetch_packed_lock:
            if self._prefetch_packed_arena is not None:
                self._prefetch_packed_arena.synchronize()
                self._prefetch_packed_arena = None
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._staging_lock:
            if self._staging_arena is not None:
                self._staging_arena.synchronize()
                for stream in self._streams.values():
                    stream.synchronize()
                self._staging_arena = None
            if self._staging_pool is not None:
                self._staging_pool.close()
                self._staging_pool = None
                with self._stats_lock:
                    self._pinned_in_flight = 0
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
        buffer = torch.empty(int(nbytes), dtype=torch.uint8)
        started = time.perf_counter()
        physical_bytes = self._shards[region.shard].readinto(memoryview(buffer.numpy()), int(offset))
        elapsed = time.perf_counter() - started
        with self._stats_lock:
            self._read_ops += 1
            self._read_bytes += int(nbytes)
            if self._backend in {"direct_io", "io_uring"}:
                self._direct_io_read_bytes += int(physical_bytes)
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
        return PackedCudaTransfer(
            buffer.tensor.view(torch_dtype(region.dtype_name)).reshape(count, *region.shape[1:]),
            buffer.ready,
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

    def _copy_staged_to_device(self, host_tensors, target, dtype, stream):
        from mirai.core.training.residency.tensor_residency import PinnedHostStagingPool

        with self._staging_lock:
            budget = min(self._pin_cap_bytes, 3 * _MAX_RANGE_BYTES)
            if self._staging_pool is None and budget >= 3 and not self._staging_unavailable:
                try:
                    self._staging_pool = PinnedHostStagingPool(budget)
                except (RuntimeError, MemoryError, OSError):
                    self._staging_unavailable = True
                else:
                    with self._stats_lock:
                        self._pinned_in_flight = self._staging_pool.pinned_bytes
                        self._pinned_peak = self._pinned_in_flight
            if self._staging_pool is not None:
                if self._raw_staging:
                    return self._copy_through_cuda_arena(host_tensors, target, dtype, stream)
                copies = self._staging_pool.copy_to_device(list(host_tensors), device=target, stream=stream)
                return copies if dtype is None else [tensor.to(dtype=dtype) for tensor in copies]
            if self._raw_staging:
                return self._copy_through_cuda_arena(host_tensors, target, dtype, stream)
        return [tensor.to(device=target, dtype=dtype) for tensor in host_tensors]

    def _cuda_staging_arena(self, target, stream):
        from .fixed_cuda_arena import FixedCudaArena

        if self._staging_arena is None:
            self._staging_arena = FixedCudaArena(
                device=target, slots=_CUDA_STAGING_SLOTS, slot_bytes=_MAX_RANGE_BYTES, alignment=4096,
            )
            self._staging_arena_device = stream.device
        elif self._staging_arena_device != stream.device:
            raise ValueError("Packed CUDA staging slots are bound to one device.")
        return self._staging_arena

    def _cuda_prefetch_dense_arena(self, target, stream):
        from .fixed_cuda_arena import FixedCudaArena

        if self._prefetch_dense_arena is None:
            self._prefetch_dense_arena = FixedCudaArena(
                device=target, slots=_CUDA_PREFETCH_DENSE_SLOTS, slot_bytes=_MAX_RANGE_BYTES, alignment=4096,
            )
            self._prefetch_dense_device = stream.device
        elif self._prefetch_dense_device != stream.device:
            raise ValueError("Packed CUDA prefetch dense slots are bound to one device.")
        return self._prefetch_dense_arena

    def _copy_through_cuda_arena(self, host_tensors, target, dtype, stream):
        arena = self._cuda_staging_arena(target, stream)
        copies = []
        for source in host_tensors:
            result = torch.empty_like(source, device=target)
            source_bytes = source.view(-1).view(torch.uint8)
            result_bytes = result.view(-1).view(torch.uint8)
            for start in range(0, source_bytes.numel(), _MAX_RANGE_BYTES):
                chunk = source_bytes[start:start + _MAX_RANGE_BYTES]
                with arena.lease(stream) as storage:
                    slot = storage[:chunk.numel()]
                    if self._staging_pool is not None:
                        self._staging_pool.copy_into([chunk], [slot], stream=stream)
                    else:
                        slot.copy_(chunk)
                    result_bytes[start:start + chunk.numel()].copy_(slot)
            copies.append(result if dtype is None else result.to(dtype=dtype))
        return copies

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
    if normalized not in {"staged", "direct_io", "io_uring", "gds"}:
        raise ValueError("memory.packed_stream_backend must be one of: staged, direct_io, io_uring, gds.")
    return normalized
