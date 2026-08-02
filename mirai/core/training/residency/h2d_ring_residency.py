"""Reusable flat-buffer H2D residency for immutable module blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_BUFFER_ALIGNMENT = 16


def _align(value: int, alignment: int = _BUFFER_ALIGNMENT) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


@dataclass
class _TensorBinding:
    owner: Any
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: Any
    offset: int
    nbytes: int

    def view(self, flat: Any) -> Any:
        raw = flat.narrow(0, self.offset, self.nbytes)
        return raw.view(self.dtype).view(self.shape)

    def assign(self, tensor: Any) -> None:
        if self.kind == "parameter":
            self.owner._parameters[self.name].data = tensor
        else:
            self.owner._buffers[self.name] = tensor


@dataclass
class _RingSlot:
    flat: Any
    owner: FlatRingModuleResidency | None = None
    free_event: Any | None = None


class FlatRingModuleResidency:
    """One immutable block backed by a permanent flat CPU master."""

    def __init__(
        self,
        pool: FlatRingResidencyPool,
        *,
        module: Any,
        rank: int,
        bindings: list[_TensorBinding],
        cpu_flat: Any,
        pinned_bytes: int,
    ) -> None:
        self._pool = pool
        self.module = module
        self.rank = int(rank)
        self._bindings = bindings
        self._cpu_flat = cpu_flat
        self._pinned_bytes = int(pinned_bytes)
        self._device: Any | None = None
        self._slot_index: int | None = None
        self._device_tensors: list[Any] = []

    @property
    def bytes(self) -> int:
        return int(self._cpu_flat.numel())

    @property
    def pinned_bytes(self) -> int:
        return self._pinned_bytes

    @property
    def backing_store_bytes(self) -> int:
        return 0

    @property
    def device(self) -> str:
        return "cpu" if self._device is None else str(self._device)

    def _assign_cpu(self) -> None:
        for binding in self._bindings:
            binding.assign(binding.view(self._cpu_flat))

    def _assign_device(self, flat: Any) -> None:
        tensors = [binding.view(flat) for binding in self._bindings]
        for binding, tensor in zip(self._bindings, tensors, strict=True):
            binding.assign(tensor)
        self._device_tensors = tensors

    def load(self, device: Any, *, stream: Any | None = None) -> Any | None:
        return self._pool.load(self, device=device, stream=stream)

    def record_stream(self, stream: Any) -> None:
        for tensor in self._device_tensors:
            tensor.record_stream(stream)

    def mark_consumed(self, stream: Any) -> None:
        self._pool.mark_consumed(self, stream=stream)

    def offload(self) -> None:
        self._pool.offload(self)


class FlatRingResidencyPool:
    """Pack immutable blocks on CPU and stream them through reusable GPU slots."""

    def __init__(
        self,
        modules: Iterable[Any],
        *,
        ring_size: int,
        pin_budget_bytes: int | None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for flat H2D residency.")
        self.ring_size = max(1, int(ring_size))
        self._target: Any | None = None
        self._slots: list[_RingSlot] = []
        self._h2d_copies = 0
        self._h2d_bytes = 0
        self.units: list[FlatRingModuleResidency] = []
        remaining = pin_budget_bytes
        for rank, module in enumerate(modules):
            bindings, sources, total_bytes = self._inventory(module)
            use_pinned = remaining is None or total_bytes <= int(remaining)
            cpu_flat = self._allocate_cpu(total_bytes, pin=use_pinned)
            pinned_bytes = total_bytes if bool(cpu_flat.is_pinned()) else 0
            for binding, source in zip(bindings, sources, strict=True):
                binding.view(cpu_flat).copy_(source)
            unit = FlatRingModuleResidency(
                self,
                module=module,
                rank=rank,
                bindings=bindings,
                cpu_flat=cpu_flat,
                pinned_bytes=pinned_bytes,
            )
            unit._assign_cpu()
            self.units.append(unit)
            if remaining is not None:
                remaining = max(0, int(remaining) - pinned_bytes)
        self._slot_bytes = max((unit.bytes for unit in self.units), default=0)

    @staticmethod
    def _inventory(module: Any) -> tuple[list[_TensorBinding], list[Any], int]:
        bindings: list[_TensorBinding] = []
        sources: list[Any] = []
        offset = 0
        for owner in module.modules():
            for name, parameter in owner._parameters.items():
                if parameter is None or bool(parameter.requires_grad):
                    continue
                source = parameter.detach().to(device="cpu").contiguous()
                offset = _align(offset)
                nbytes = int(source.numel()) * int(source.element_size())
                bindings.append(
                    _TensorBinding(
                        owner=owner,
                        name=str(name),
                        kind="parameter",
                        shape=tuple(source.shape),
                        dtype=source.dtype,
                        offset=offset,
                        nbytes=nbytes,
                    )
                )
                sources.append(source)
                offset += nbytes
            for name, buffer in owner._buffers.items():
                if buffer is None or not torch.is_tensor(buffer):
                    continue
                source = buffer.detach().to(device="cpu").contiguous()
                offset = _align(offset)
                nbytes = int(source.numel()) * int(source.element_size())
                bindings.append(
                    _TensorBinding(
                        owner=owner,
                        name=str(name),
                        kind="buffer",
                        shape=tuple(source.shape),
                        dtype=source.dtype,
                        offset=offset,
                        nbytes=nbytes,
                    )
                )
                sources.append(source)
                offset += nbytes
        return bindings, sources, _align(offset)

    @staticmethod
    def _allocate_cpu(size: int, *, pin: bool) -> Any:
        try:
            return torch.empty(int(size), dtype=torch.uint8, pin_memory=bool(pin))
        except (RuntimeError, OSError):
            return torch.empty(int(size), dtype=torch.uint8)

    @property
    def pinned_bytes(self) -> int:
        return sum(unit.pinned_bytes for unit in self.units)

    @property
    def ring_buffer_bytes(self) -> int:
        return int(self.ring_size * self._slot_bytes)

    @property
    def h2d_copies(self) -> int:
        return self._h2d_copies

    @property
    def h2d_bytes(self) -> int:
        return self._h2d_bytes

    def _ensure_slots(self, target: Any) -> None:
        if target.type != "cuda":
            raise ValueError("flat_ring block swap requires a CUDA device.")
        if self._target is not None and self._target != target:
            raise ValueError("A flat H2D ring cannot span multiple CUDA devices.")
        if not self._slots:
            self._slots = [
                _RingSlot(torch.empty(self._slot_bytes, dtype=torch.uint8, device=target))
                for _ in range(self.ring_size)
            ]
            self._target = target

    def load(
        self,
        unit: FlatRingModuleResidency,
        *,
        device: Any,
        stream: Any | None,
    ) -> Any | None:
        target = torch.device(device)
        if unit._device == target:
            return None
        self._ensure_slots(target)
        slot_index = unit.rank % self.ring_size
        slot = self._slots[slot_index]
        if slot.owner is not None and slot.owner is not unit:
            raise RuntimeError(
                "flat H2D ring slot collision; increase prefetch ring capacity or "
                "reduce memory.block_swap_prefetch_depth."
            )

        def _copy() -> Any | None:
            if slot.free_event is not None:
                torch.cuda.current_stream(target).wait_event(slot.free_event)
            slot.flat.narrow(0, 0, unit.bytes).copy_(
                unit._cpu_flat, non_blocking=bool(unit.pinned_bytes)
            )
            unit._assign_device(slot.flat)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(target))
            return event

        if stream is None:
            event = _copy()
            if event is not None:
                event.synchronize()
                event = None
        else:
            with torch.cuda.stream(stream):
                event = _copy()
        slot.owner = unit
        unit._slot_index = slot_index
        unit._device = target
        self._h2d_copies += 1
        self._h2d_bytes += unit.bytes
        return event

    def mark_consumed(self, unit: FlatRingModuleResidency, *, stream: Any) -> None:
        if unit._slot_index is None:
            return
        slot = self._slots[unit._slot_index]
        if slot.owner is not unit:
            return
        event = torch.cuda.Event()
        event.record(stream)
        slot.free_event = event

    def offload(self, unit: FlatRingModuleResidency) -> None:
        if unit._device is None:
            return
        unit._assign_cpu()
        if unit._slot_index is not None:
            slot = self._slots[unit._slot_index]
            if slot.owner is unit:
                slot.owner = None
        unit._device_tensors.clear()
        unit._slot_index = None
        unit._device = None
