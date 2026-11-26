"""Fixed-size CUDA staging slots backed by a single raw allocation."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Condition
from typing import Any


class FixedCudaArena:
    """Own aligned byte slots outside the PyTorch caching allocator.

    The caller orders all uses of a slot and drains its streams before
    dropping the arena. Slot views must stay inside that ownership boundary.
    """

    def __init__(self, *, device: Any, slots: int, slot_bytes: int, alignment: int) -> None:
        if min(slots, slot_bytes, alignment) <= 0 or slot_bytes % alignment:
            raise ValueError("CUDA arena requires a positive number of aligned slots.")
        import torch

        try:
            import cupy
        except (ImportError, OSError) as exc:
            raise RuntimeError("The raw packed CUDA arena requires CuPy for the active CUDA major.") from exc
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("The raw packed arena requires a CUDA device.")
        stream = torch.cuda.current_stream(target)
        total_bytes = slots * slot_bytes
        with cupy.cuda.Device(stream.device.index), cupy.cuda.ExternalStream(stream.cuda_stream):
            memory = cupy.cuda.Memory(total_bytes + alignment - 1)
            padding = -int(memory.ptr) % alignment
            pointer = cupy.cuda.MemoryPointer(memory, padding)
            array = cupy.ndarray((total_bytes,), dtype=cupy.uint8, memptr=pointer)
            self._storage = torch.from_dlpack(array)
        self._slots = slots
        self._slot_bytes = slot_bytes
        self._available = list(range(slots))
        self._release_events: dict[int, Any] = {}
        self._leases = Condition()
        self._failed = False

    def slot(self, index: int):
        if not 0 <= index < self._slots:
            raise IndexError(index)
        start = index * self._slot_bytes
        return self._storage[start:start + self._slot_bytes]

    @contextmanager
    def lease(self, stream, *, wait: bool = True):
        """Borrow a slot for reads and writes enqueued on the given stream."""
        import torch

        with self._leases:
            if wait:
                self._leases.wait_for(lambda: self._failed or bool(self._available))
            if self._failed:
                raise RuntimeError("CUDA arena slot completion could not be verified.")
            index = self._available.pop(0) if self._available else None
            previous = self._release_events.get(index)
        if index is None:
            yield None
            return
        try:
            if previous is not None:
                stream.wait_event(previous)
            yield self.slot(index)
        finally:
            try:
                released = torch.cuda.Event()
                released.record(stream)
            except Exception:
                try:
                    stream.synchronize()
                except Exception:
                    with self._leases:
                        self._failed = True
                        self._leases.notify_all()
                    raise
                self._return_slot(index, None)
                raise
            self._return_slot(index, released)

    def _return_slot(self, index: int, event: Any) -> None:
        with self._leases:
            self._release_events[index] = event
            self._available.append(index)
            self._leases.notify()

    def synchronize(self) -> None:
        """Drain returned leases after the owner has stopped new submissions."""
        with self._leases:
            if self._failed or len(self._available) != self._slots:
                raise RuntimeError("CUDA arena has outstanding slot usage.")
            for event in self._release_events.values():
                if event is not None:
                    event.synchronize()
