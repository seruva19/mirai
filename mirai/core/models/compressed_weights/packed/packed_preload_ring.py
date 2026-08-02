"""Reusable host-batch ring for preloaded packed expert tensors."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .packed_stream_prefetch import PackedStreamPrefetchHost
from .packed_stream_prefetch import PackedStreamPrefetchRing

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class _HostSlot:
    tensor: Any | None = None
    busy: bool = False
    copy_event: Any | None = None
    pinned_bytes: int = 0


class PackedPreloadedHostRing:
    """Bound reusable host batches until their asynchronous H2D copy completes."""

    def __init__(self, depth: int, *, pin_budget_bytes: int | None) -> None:
        resolved = int(depth)
        if resolved <= 0 or resolved > 16:
            raise ValueError("Packed preload ring depth must be between 1 and 16.")
        budget = None if pin_budget_bytes is None else int(pin_budget_bytes)
        if budget is not None and budget < 0:
            raise ValueError("Packed preload pin budget must be >= 0 bytes.")
        self._slots = [_HostSlot() for _ in range(resolved)]
        self._pin_budget_bytes = budget
        self._pinned_bytes = 0
        self._peak_pinned_bytes = 0
        self._cursor = 0
        self._allocations = 0
        self._reuses = 0
        self._waits = 0
        self._wait_seconds = 0.0
        self._condition = threading.Condition()
        self._closed = False

    def acquire(self, shape: Sequence[int], dtype: Any) -> tuple[int, Any]:
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("Packed preload host ring is closed.")
                candidates = [
                    offset
                    for offset in range(len(self._slots))
                    if not self._slots[(self._cursor + offset) % len(self._slots)].busy
                ]
                if candidates:
                    index = (self._cursor + candidates[0]) % len(self._slots)
                    self._cursor = (index + 1) % len(self._slots)
                    slot = self._slots[index]
                    slot.busy = True
                    event = slot.copy_event
                    slot.copy_event = None
                    break
                self._condition.wait()
        if event is not None:
            started = time.perf_counter()
            event.synchronize()
            with self._condition:
                self._waits += 1
                self._wait_seconds += time.perf_counter() - started
        resolved_shape = tuple(int(dim) for dim in shape)
        with self._condition:
            tensor = slot.tensor
            if (
                tensor is not None
                and tuple(tensor.shape) == resolved_shape
                and tensor.dtype == dtype
            ):
                self._reuses += 1
                return index, tensor
            self._pinned_bytes -= int(slot.pinned_bytes)
            slot.pinned_bytes = 0
            nbytes = _shape_nbytes(resolved_shape, dtype)
            want_pin = (
                self._pin_budget_bytes is None
                or self._pinned_bytes + nbytes <= self._pin_budget_bytes
            )
            slot.tensor = _allocate_host(resolved_shape, dtype, pin=want_pin)
            if bool(slot.tensor.is_pinned()):
                slot.pinned_bytes = nbytes
                self._pinned_bytes += nbytes
                self._peak_pinned_bytes = max(
                    self._peak_pinned_bytes, self._pinned_bytes
                )
            self._allocations += 1
            return index, slot.tensor

    def release(self, index: int, *, copy_event: Any | None) -> None:
        with self._condition:
            slot = self._slots[int(index)]
            slot.copy_event = copy_event
            slot.busy = False
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            events = tuple(
                slot.copy_event
                for slot in self._slots
                if slot.copy_event is not None
            )
            self._condition.notify_all()
        for event in events:
            event.synchronize()
        with self._condition:
            self._slots.clear()
            self._pinned_bytes = 0

    def snapshot(self) -> dict[str, int | float]:
        with self._condition:
            return {
                "preload_ring_slots": len(self._slots),
                "preload_ring_allocations": self._allocations,
                "preload_ring_reuses": self._reuses,
                "preload_ring_pinned_bytes": self._pinned_bytes,
                "preload_ring_peak_pinned_bytes": self._peak_pinned_bytes,
                "preload_ring_waits": self._waits,
                "preload_ring_wait_seconds": self._wait_seconds,
            }


class PrefetchedPreloadedPackedTensorMapping(
    PackedStreamPrefetchHost, Mapping[str, Any]
):
    """Add exact-request asynchronous H2D to a fully preloaded packed mapping."""

    def __init__(
        self,
        base: Mapping[str, Any],
        *,
        prefetch_depth: int,
        pin_budget_bytes: int | None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for packed preload prefetch.")
        self._base = base
        self._prefetch = PackedStreamPrefetchRing(prefetch_depth, inline=True)
        self._host_ring = PackedPreloadedHostRing(
            prefetch_depth, pin_budget_bytes=pin_budget_bytes
        )
        self._stream_lock = threading.Lock()
        self._streams: dict[str, Any] = {}
        self._stats_lock = threading.Lock()
        self._h2d_ops = 0
        self._h2d_bytes = 0
        self._direct_pinned_batches = 0
        self._fragmented_bypasses = 0
        self._closed = False

    def __getitem__(self, key: str) -> Any:
        return self._base[str(key)]

    def get_slice(self, key: str, index: int) -> Any:
        get_slice = getattr(self._base, "get_slice", None)
        if callable(get_slice):
            return get_slice(str(key), int(index))
        return self._base[str(key)][int(index)]

    def cached_tensor(self, key: str) -> Any | None:
        cached = getattr(self._base, "cached_tensor", None)
        return cached(str(key)) if callable(cached) else None

    def tensor_nbytes(self, key: str) -> int:
        tensor = self._base[str(key)]
        return int(tensor.numel()) * int(tensor.element_size())

    def __iter__(self) -> Iterator[str]:
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)

    def __contains__(self, key: object) -> bool:
        return key in self._base

    def _normalize_indices(
        self, key: str, indices: Sequence[int]
    ) -> tuple[int, ...]:
        tensor = self._base[str(key)]
        if not tensor.shape:
            raise IndexError(f"Packed tensor {key!r} is scalar and cannot be sliced.")
        extent = int(tensor.shape[0])
        normalized: list[int] = []
        for index in indices:
            resolved = int(index)
            if resolved < 0:
                resolved += extent
            if resolved < 0 or resolved >= extent:
                raise IndexError(index)
            normalized.append(resolved)
        return tuple(normalized)

    def prefetch_slices_to_device(
        self,
        key: str,
        indices: Sequence[int],
        *,
        device: Any,
        dtype: Any | None = None,
    ) -> bool:
        requested = self._normalize_indices(str(key), indices)
        source = self._base[str(key)]
        target = torch.device(device)
        if (
            target.type == "cuda"
            and bool(source.is_pinned())
            and len(_contiguous_runs(tuple(sorted(set(requested)))))
            == len(set(requested))
        ):
            with self._stats_lock:
                self._fragmented_bypasses += 1
            return False
        return PackedStreamPrefetchHost.prefetch_slices_to_device(
            self, str(key), requested, device=target, dtype=dtype
        )

    def get_slices_to_device(
        self,
        key: str,
        indices: Sequence[int],
        *,
        device: Any,
        dtype: Any | None = None,
    ) -> Any:
        requested = self._normalize_indices(str(key), indices)
        source = self._base[str(key)]
        target = torch.device(device)
        if self._is_fragmented_pinned(source, requested, target):
            return self._copy_fragmented_pinned(
                source, requested, target=target, dtype=dtype
            )
        return PackedStreamPrefetchHost.get_slices_to_device(
            self, str(key), requested, device=target, dtype=dtype
        )

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
            raise ValueError("Packed preload batch requires at least one expert.")
        source = self._base[str(key)]
        target = torch.device(device)
        if target.type != "cuda":
            index = torch.as_tensor(requested, dtype=torch.long)
            batch = source.index_select(0, index)
            return batch.to(device=target, dtype=dtype) if dtype is not None else batch
        stream = self._cuda_stream(target)
        if bool(source.is_pinned()):
            unique = tuple(sorted(set(requested)))
            runs = _contiguous_runs(unique)
            if len(runs) == len(unique):
                return self._copy_fragmented_pinned(
                    source, requested, target=target, dtype=dtype
                )
            with torch.cuda.stream(stream):
                restore = tuple(unique.index(item) for item in requested)
                parts = [
                    source[start : start + count].to(
                        device=target, dtype=dtype, non_blocking=True
                    )
                    for start, count in runs
                ]
                result = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
                if restore != tuple(range(len(requested))):
                    result = result.index_select(
                        0,
                        torch.as_tensor(
                            restore, device=target, dtype=torch.long
                        ),
                    )
                event = torch.cuda.Event()
                event.record(stream)
            h2d_ops = len(parts)
            with self._stats_lock:
                self._direct_pinned_batches += 1
        else:
            shape = (len(requested), *tuple(source.shape[1:]))
            slot_index, host_batch = self._host_ring.acquire(shape, source.dtype)
            try:
                index = torch.as_tensor(requested, dtype=torch.long)
                torch.index_select(source, 0, index, out=host_batch)
                with torch.cuda.stream(stream):
                    result = host_batch.to(
                        device=target,
                        dtype=dtype,
                        non_blocking=bool(host_batch.is_pinned()),
                    )
                    event = torch.cuda.Event()
                    event.record(stream)
                self._host_ring.release(slot_index, copy_event=event)
                h2d_ops = 1
            except Exception:
                self._host_ring.release(slot_index, copy_event=None)
                raise
        current = torch.cuda.current_stream(target)
        current.wait_event(event)
        result.record_stream(current)
        nbytes = int(source[0].numel()) * int(source.element_size()) * len(requested)
        with self._stats_lock:
            self._h2d_ops += h2d_ops
            self._h2d_bytes += nbytes
        return result

    @staticmethod
    def _is_fragmented_pinned(
        source: Any, requested: Sequence[int], target: Any
    ) -> bool:
        unique = tuple(sorted(set(requested)))
        return (
            target.type == "cuda"
            and bool(source.is_pinned())
            and len(_contiguous_runs(unique)) == len(unique)
        )

    def _copy_fragmented_pinned(
        self,
        source: Any,
        requested: Sequence[int],
        *,
        target: Any,
        dtype: Any | None,
    ) -> Any:
        parts = [
            source[item].to(device=target, dtype=dtype, non_blocking=True)
            for item in requested
        ]
        result = torch.stack(parts, dim=0)
        nbytes = int(source[0].numel()) * int(source.element_size()) * len(requested)
        with self._stats_lock:
            self._h2d_ops += len(parts)
            self._h2d_bytes += nbytes
        return result

    def _adopt_prefetched_result(self, value: Any, *, device: Any) -> Any:
        target = torch.device(device)
        if target.type != "cuda":
            return value
        current = torch.cuda.current_stream(target)
        current.wait_stream(self._cuda_stream(target))
        value.record_stream(current)
        return value

    def snapshot(self) -> dict[str, int | float | str]:
        with self._stats_lock:
            snapshot: dict[str, int | float | str] = {
                "backend": "preloaded_ring",
                "h2d_ops": self._h2d_ops,
                "h2d_bytes": self._h2d_bytes,
                "preload_direct_pinned_batches": self._direct_pinned_batches,
                "preload_fragmented_bypasses": self._fragmented_bypasses,
            }
        snapshot.update(self._host_ring.snapshot())
        snapshot.update(self._prefetch.snapshot())
        return snapshot

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._prefetch.close()
        self._host_ring.close()

    def __enter__(self) -> PrefetchedPreloadedPackedTensorMapping:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _cuda_stream(self, device: Any) -> Any:
        key = str(device)
        with self._stream_lock:
            stream = self._streams.get(key)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                self._streams[key] = stream
            return stream


def _shape_nbytes(shape: Sequence[int], dtype: Any) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * int(torch.empty((), dtype=dtype).element_size())


def _allocate_host(shape: Sequence[int], dtype: Any, *, pin: bool) -> Any:
    try:
        return torch.empty(tuple(shape), dtype=dtype, pin_memory=bool(pin))
    except (RuntimeError, MemoryError, OSError):
        return torch.empty(tuple(shape), dtype=dtype)


def _contiguous_runs(indices: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if not indices:
        return ()
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous - start + 1))
        start = previous = index
    runs.append((start, previous - start + 1))
    return tuple(runs)
