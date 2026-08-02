"""Immutable module-tensor residency for adapter training."""

from __future__ import annotations

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


class ImmutableModuleResidency:
    """Keep immutable tensors on CPU and materialize device copies on demand."""

    def __init__(
        self,
        module: Any,
        *,
        pin_budget_bytes: int | None = None,
        disk_path: str | Path | None = None,
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
        self._pinning_enabled = self._disk_path is None
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
            copies = [
                slot.cpu_tensor.to(device=target, non_blocking=True)
                for slot in self._slots
            ]
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
