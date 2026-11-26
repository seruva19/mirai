"""Fixed-size CUDA staging slots backed by a single raw allocation."""

from __future__ import annotations

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

    def slot(self, index: int):
        if not 0 <= index < self._slots:
            raise IndexError(index)
        start = index * self._slot_bytes
        return self._storage[start:start + self._slot_bytes]
