"""Immutable module-tensor residency for adapter training."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class _TensorSlot:
    owner: Any
    name: str
    kind: str
    cpu_tensor: Any

    def assign(self, tensor: Any) -> None:
        if self.kind == "parameter":
            parameter = self.owner._parameters[self.name]
            parameter.data = tensor
        else:
            self.owner._buffers[self.name] = tensor


class PinnedHostStagingPool:
    """Stage contiguous CPU tensor bytes in a bounded, reusable host ring."""

    def __init__(self, budget_bytes: int, *, slots: int = 3) -> None:
        if slots < 2 or int(budget_bytes) < slots:
            raise ValueError("Pinned staging requires at least two nonempty slots.")
        # The CUDA host allocator can round each allocation up to a power of
        # two. Size each slot accordingly before applying the aggregate cap.
        capacity = 1 << ((int(budget_bytes) // slots).bit_length() - 1)
        self._buffers = [
            torch.empty(capacity, dtype=torch.uint8, device="cpu", pin_memory=True)
            for _ in range(slots)
        ]
        self._events = [None] * slots
        self._next_slot = 0
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mirai-host-staging")

    @property
    def pinned_bytes(self) -> int:
        return sum(buffer.numel() for buffer in self._buffers)

    def _stage_chunk(self, index: int, source: Any) -> Any:
        event = self._events[index]
        if event is not None:
            event.synchronize()
        staged = self._buffers[index][:source.numel()]
        with torch.inference_mode():
            staged.copy_(source)
        return staged

    def copy_to_device(self, sources: list[Any], *, device: Any, stream: Any) -> list[Any]:
        # The caller owns CUDA allocation and submission. A single CPU worker
        # prepares the next chunk while the transfer stream reads the previous one.
        copies = [torch.empty_like(source, device=device) for source in sources]
        capacity = self._buffers[0].numel()

        def chunks():
            for source, target in zip(sources, copies, strict=True):
                source_bytes = source.view(-1).view(torch.uint8)
                target_bytes = target.view(-1).view(torch.uint8)
                for start in range(0, source_bytes.numel(), capacity):
                    yield source_bytes[start:start + capacity], target_bytes[start:start + capacity]

        iterator = iter(chunks())
        pending = None
        try:
            current = next(iterator, None)
            while current is not None:
                source, target = current
                index = self._next_slot
                future = pending if pending is not None else self._worker.submit(self._stage_chunk, index, source)
                pending = None
                staged = future.result()
                self._next_slot = (index + 1) % len(self._buffers)
                current = next(iterator, None)
                if current is not None:
                    pending = self._worker.submit(self._stage_chunk, self._next_slot, current[0])
                target.copy_(staged, non_blocking=True)
                event = self._events[index]
                if event is None:
                    event = torch.cuda.Event()
                event.record(stream)
                self._events[index] = event
        except BaseException:
            stream.synchronize()
            if pending is not None:
                pending.exception()
            raise
        return copies

    def close(self) -> None:
        self._worker.shutdown(wait=True)
        for event in self._events:
            if event is not None:
                event.synchronize()
        self._buffers.clear()
        self._events.clear()


class ImmutableModuleResidency:
    """Keep immutable tensors on CPU and materialize device copies on demand."""

    def __init__(
        self,
        module: Any,
        *,
        pin_budget_bytes: int | None = None,
        disk_path: str | Path | None = None,
        staging_pool: PinnedHostStagingPool | None = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for tensor residency.")
        self.module = module
        self._slots: list[_TensorSlot] = []
        self._device_tensors: list[Any] = []
        self._device: torch.device | None = None
        self._pin_budget_bytes = pin_budget_bytes
        self._pinned_bytes = 0
        self._disk_path = Path(disk_path) if disk_path is not None else None
        self._staging_pool = staging_pool
        self._pinning_enabled = self._disk_path is None and staging_pool is None
        for owner in module.modules():
            for name, parameter in owner._parameters.items():
                if parameter is None or bool(parameter.requires_grad):
                    continue
                cpu_tensor = self._stage(parameter.detach().to(device="cpu").contiguous())
                parameter.data = cpu_tensor
                self._slots.append(_TensorSlot(owner, str(name), "parameter", cpu_tensor))
            for name, buffer in owner._buffers.items():
                if buffer is None or not torch.is_tensor(buffer):
                    continue
                cpu_tensor = self._stage(buffer.detach().to(device="cpu").contiguous())
                owner._buffers[name] = cpu_tensor
                self._slots.append(_TensorSlot(owner, str(name), "buffer", cpu_tensor))
        if self._disk_path is not None:
            self._bind_disk_backing()

    def _bind_disk_backing(self) -> None:
        """Atomically replace anonymous CPU staging with safetensors mappings."""

        from safetensors.torch import load_file, save_file

        path = self._disk_path
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        payload = {
            f"tensor_{index:06d}": slot.cpu_tensor.contiguous().clone()
            for index, slot in enumerate(self._slots)
        }
        try:
            save_file(payload, str(temporary))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        mapped = load_file(str(path), device="cpu")
        for index, slot in enumerate(self._slots):
            tensor = mapped[f"tensor_{index:06d}"]
            slot.cpu_tensor = tensor
            slot.assign(tensor)

    def _stage(self, tensor: Any) -> Any:
        """Pin the CPU staging tensor when it fits the budget, else keep it pageable.

        Pinned host memory lets ``non_blocking=True`` copies overlap with compute
        on the transfer stream; pageable tensors silently fall back to a blocking
        bounce-buffer copy. Pinning degrades gracefully when the host allocator or
        the module budget is exhausted.
        """

        if self._staging_pool is not None:
            if tensor.is_pinned():
                return torch.empty_like(tensor, pin_memory=False).copy_(tensor)
            return tensor
        if not self._pinning_enabled:
            return tensor
        nbytes = int(tensor.numel()) * int(tensor.element_size())
        if (
            self._pin_budget_bytes is not None
            and self._pinned_bytes + nbytes > int(self._pin_budget_bytes)
        ):
            self._pinning_enabled = False
            return tensor
        try:
            pinned = tensor.pin_memory()
        except (RuntimeError, OSError):
            self._pinning_enabled = False
            return tensor
        self._pinned_bytes += nbytes
        return pinned

    @property
    def pinned_bytes(self) -> int:
        return int(self._pinned_bytes)

    @property
    def backing_store_bytes(self) -> int:
        if self._disk_path is None:
            return 0
        return int(self._disk_path.stat().st_size)

    @property
    def bytes(self) -> int:
        return sum(
            int(slot.cpu_tensor.numel()) * int(slot.cpu_tensor.element_size())
            for slot in self._slots
        )

    @property
    def device(self) -> str:
        return "cpu" if self._device is None else str(self._device)

    def load(self, device: Any, *, stream: Any | None = None) -> Any | None:
        target = torch.device(device)
        if self._device == target:
            return None
        if target.type != "cuda" or stream is None:
            copies = [
                slot.cpu_tensor.to(device=target, non_blocking=False)
                for slot in self._slots
            ]
            for slot, value in zip(self._slots, copies, strict=True):
                slot.assign(value)
            self._device_tensors = copies
            self._device = target
            return None
        with torch.cuda.stream(stream):
            if self._staging_pool is None:
                copies = [
                    slot.cpu_tensor.to(device=target, non_blocking=True)
                    for slot in self._slots
                ]
            else:
                copies = self._staging_pool.copy_to_device(
                    [slot.cpu_tensor for slot in self._slots], device=target, stream=stream,
                )
            for slot, value in zip(self._slots, copies, strict=True):
                slot.assign(value)
            event = torch.cuda.Event()
            event.record(stream)
        self._device_tensors = copies
        self._device = target
        return event

    def record_stream(self, stream: Any) -> None:
        """Mark device copies as in-use by ``stream`` for the caching allocator.

        Device copies are allocated on the transfer stream; without this the
        allocator may reuse their memory for a later transfer-stream allocation
        as soon as ``offload`` drops the references, while the consumer stream
        is still reading the weights.
        """

        for tensor in self._device_tensors:
            tensor.record_stream(stream)

    def offload(self) -> None:
        if self._device is None:
            return
        for slot in self._slots:
            slot.assign(slot.cpu_tensor)
        self._device_tensors.clear()
        self._device = None


def move_trainable_tensors(
    module: Any,
    *,
    device: Any,
    dtype: Any | None = None,
) -> None:
    target = torch.device(device)
    for parameter in module.parameters():
        if not bool(parameter.requires_grad):
            continue
        target_dtype = (
            dtype
            if dtype is not None and bool(parameter.is_floating_point())
            else parameter.dtype
        )
        if parameter.device != target or parameter.dtype != target_dtype:
            parameter.data = parameter.data.to(device=target, dtype=target_dtype)
        if parameter.grad is not None and parameter.grad.device != target:
            parameter.grad.data = parameter.grad.data.to(device=target)


def cast_trainable_tensors(module: Any, *, dtype: Any) -> None:
    for parameter in module.parameters():
        if bool(parameter.requires_grad) and bool(parameter.is_floating_point()):
            if parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)


def move_tensors_outside_modules(
    root: Any,
    *,
    excluded_modules: list[Any],
    device: Any,
) -> None:
    target = torch.device(device)
    excluded_ids: set[int] = set()
    for excluded in excluded_modules:
        excluded_ids.update(id(module) for module in excluded.modules())
    for module in root.modules():
        if id(module) in excluded_ids:
            continue
        for parameter in module.parameters(recurse=False):
            if parameter.device != target:
                parameter.data = parameter.data.to(device=target)
        for name, buffer in module._buffers.items():
            if buffer is not None and torch.is_tensor(buffer) and buffer.device != target:
                module._buffers[name] = buffer.to(device=target)
