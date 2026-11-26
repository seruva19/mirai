"""Bounded asynchronous request ring for packed expert streaming."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Hashable


@dataclass
class _PrefetchEntry:
    loader: Callable[[], Any]
    discard: Callable[[Any], None] | None = None
    future: Future[Any] | None = None


class PackedStreamPrefetchRing:
    """Keep a bounded number of packed requests in flight until consumption."""

    def __init__(self, depth: int, *, inline: bool = False) -> None:
        resolved = int(depth)
        if resolved <= 0 or resolved > 16:
            raise ValueError("Packed stream prefetch depth must be between 1 and 16.")
        self._depth = resolved
        self._queue_limit = resolved * 4
        self._inline = bool(inline)
        self._executor = (
            None
            if self._inline
            else ThreadPoolExecutor(
                max_workers=resolved,
                thread_name_prefix="mirai-packed-prefetch",
            )
        )
        self._lock = threading.Lock()
        self._entries: dict[Hashable, _PrefetchEntry] = {}
        self._pending: deque[Hashable] = deque()
        self._active = 0
        self._claimed = 0
        self._closed = False
        self._submitted = 0
        self._deduplicated = 0
        self._consumed = 0
        self._misses = 0
        self._rejected = 0
        self._peak_entries = 0
        self._wait_seconds = 0.0

    def submit(
        self, identity: Hashable, loader: Callable[[], Any], *, discard: Callable[[Any], None] | None = None,
    ) -> bool:
        with self._lock:
            if self._closed:
                raise RuntimeError("Packed stream prefetch ring is closed.")
            if identity in self._entries:
                self._deduplicated += 1
                return True
            if len(self._entries) + self._claimed >= self._queue_limit:
                self._rejected += 1
                return False
            self._entries[identity] = _PrefetchEntry(loader=loader, discard=discard)
            self._pending.append(identity)
            self._submitted += 1
            self._peak_entries = max(self._peak_entries, len(self._entries) + self._claimed)
            self._fill_slots_locked()
            return True

    def take(self, identity: Hashable) -> tuple[bool, Any | None]:
        with self._lock:
            entry = self._entries.pop(identity, None)
            if entry is None:
                self._misses += 1
                return False, None
            if entry.future is None:
                self._remove_pending_locked(identity)
                self._misses += 1
                return False, None
            future = entry.future
            self._claimed += 1
        started = time.perf_counter()
        try:
            value = future.result()
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self._active -= 1
                self._claimed -= 1
                self._consumed += 1
                self._wait_seconds += float(elapsed)
                if not self._closed:
                    self._fill_slots_locked()
        return True, value

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "prefetch_depth": self._depth,
                "prefetch_inline": int(self._inline),
                "prefetch_submitted": self._submitted,
                "prefetch_deduplicated": self._deduplicated,
                "prefetch_consumed": self._consumed,
                "prefetch_misses": self._misses,
                "prefetch_rejected": self._rejected,
                "prefetch_active": self._active,
                "prefetch_queued": len(self._entries) + self._claimed,
                "prefetch_peak_queued": self._peak_entries,
                "prefetch_wait_seconds": self._wait_seconds,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = tuple(
                (future, entry.discard) for entry in self._entries.values() if (future := entry.future) is not None
            )
            self._entries.clear()
            self._active -= len(entries)
            self._pending.clear()
        for future, _ in entries:
            future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        with ExitStack() as cleanup:
            for future, discard in entries:
                if discard is not None and not future.cancelled() and future.exception() is None:
                    cleanup.callback(discard, future.result())

    def _fill_slots_locked(self) -> None:
        while self._active < self._depth and self._pending:
            identity = self._pending.popleft()
            entry = self._entries.get(identity)
            if entry is None:
                continue
            if self._executor is None:
                entry.future = Future()
                try:
                    entry.future.set_result(entry.loader())
                except BaseException as exc:
                    entry.future.set_exception(exc)
            else:
                entry.future = self._executor.submit(entry.loader)
            self._active += 1

    def _remove_pending_locked(self, identity: Hashable) -> None:
        try:
            self._pending.remove(identity)
        except ValueError:
            pass


@dataclass(frozen=True)
class PackedCudaTransfer:
    tensor: Any
    ready: Any


@dataclass(frozen=True)
class PackedCudaSlotTransfer:
    tensor: Any
    ready: Any
    producer: Any
    release: Callable[[], None]

    @contextmanager
    def borrow(self):
        import torch

        consumer = torch.cuda.current_stream(self.tensor.device)
        try:
            consumer.wait_event(self.ready)
            yield self.tensor
        finally:
            try:
                consumed = torch.cuda.Event()
                consumed.record(consumer)
                self.producer.wait_event(consumed)
            except Exception:
                consumer.synchronize()
                self.release()
                raise
            self.release()


class PackedStreamPrefetchHost:
    """Request-consumption contract mixed into the packed stream mapping."""

    _prefetch: PackedStreamPrefetchRing | None

    def get_slices_to_device(
        self,
        key: str,
        indices: Sequence[int],
        *,
        device: Any,
        dtype: Any | None = None,
    ) -> Any:
        with self._borrow_slices_to_device(key, indices, device=device, dtype=dtype) as (tensor, borrowed):
            return tensor.clone() if borrowed else tensor

    @contextmanager
    def _borrow_slices_to_device(self, key, indices, *, device, dtype=None):
        import torch

        requested = self._normalize_indices(str(key), indices)
        found, value = False, None
        if self._prefetch is not None:
            identity = packed_prefetch_identity(
                str(key), requested, device=torch.device(device), dtype=dtype
            )
            found, value = self._prefetch.take(identity)
        if not found or value is None:
            value = self._get_slices_to_device_now(
                str(key), requested, device=device, dtype=dtype
            )
        if isinstance(value, PackedCudaTransfer):
            current = torch.cuda.current_stream(value.tensor.device)
            current.wait_event(value.ready)
            value.tensor.record_stream(current)
            yield value.tensor, False
        elif isinstance(value, PackedCudaSlotTransfer):
            with value.borrow() as tensor:
                yield tensor, True
        else:
            yield value, False

    def prefetch_slices_to_device(
        self,
        key: str,
        indices: Sequence[int],
        *,
        device: Any,
        dtype: Any | None = None,
    ) -> bool:
        if self._prefetch is None:
            return False
        import torch

        requested = self._normalize_indices(str(key), indices)
        if not requested:
            raise ValueError("Packed stream prefetch requires at least one expert.")
        target = torch.device(device)
        identity = packed_prefetch_identity(
            str(key), requested, device=target, dtype=dtype
        )
        return self._prefetch.submit(
            identity,
            lambda: self._get_slices_to_device_now(
                str(key), requested, device=target, dtype=dtype
            ),
        )

    def prefetch_expert_tensors(
        self,
        tensor_names: Mapping[str, str],
        *,
        quant_format: str,
        keys: Sequence[str],
        expert_indices: Sequence[int],
        device: Any,
    ) -> int:
        if self._prefetch is None:
            return 0
        return prefetch_expert_tensors(
            self,
            tensor_names,
            quant_format=quant_format,
            keys=keys,
            expert_indices=expert_indices,
            device=device,
        )


def packed_prefetch_identity(
    key: str,
    indices: Sequence[int],
    *,
    device: Any,
    dtype: Any | None,
) -> tuple[str, tuple[int, ...], str, str]:
    return (
        str(key),
        tuple(int(index) for index in indices),
        str(device),
        str(dtype),
    )


def prefetch_expert_tensors(
    source: Any | None,
    tensor_names: Mapping[str, str],
    *,
    quant_format: str,
    keys: Sequence[str],
    expert_indices: Sequence[int],
    device: Any,
) -> int:
    """Submit fields whose exact expert set is already known by dispatch."""

    submit = getattr(source, "prefetch_slices_to_device", None)
    if not callable(submit):
        return 0
    if str(quant_format) == "nf4":
        fields: tuple[tuple[str, Any | None], ...] = (
            ("nf4", None),
            ("nf4_absmax", None),
            ("nf4_nabsmax", None),
            ("nf4_offset", None),
        )
    else:
        try:
            import torch
        except ModuleNotFoundError:  # pragma: no cover
            return 0
        fields = (("int8", None), ("scale", torch.float32))
    submitted = 0
    for key in keys:
        for suffix, dtype in fields:
            tensor_name = tensor_names.get(f"{key}_{suffix}")
            if tensor_name is None:
                continue
            submitted += int(
                bool(
                    submit(
                        tensor_name,
                        expert_indices,
                        device=device,
                        dtype=dtype,
                    )
                )
            )
    return submitted
