"""Block swap scheduling helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mirai.core.training.residency.tensor_residency import ImmutableModuleResidency
from mirai.core.training.residency.h2d_ring_residency import FlatRingResidencyPool
from mirai.core.training.residency.memory_safety import assert_system_memory_headroom
from mirai.core.training.residency.memory_safety import current_memory_safety_policy
from mirai.core.training.residency.residency_plan import ResidencyPlan, build_residency_plan

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class BlockSwapEvent:
    kind: str
    block_idx: int


def bounded_pin_budget_bytes(
    available: int, floor_bytes: int, cap_bytes: int | None
) -> int:
    """Host-pin budget = ``min(available - floor, cap)``, never negative.

    ``available`` is free host RAM; ``floor_bytes`` is the system-memory guard
    that must stay unpinned; ``cap_bytes`` is the absolute pinned-host ceiling.
    ``None`` permits the full available headroom.
    """

    headroom = max(0, int(available) - int(floor_bytes))
    if cap_bytes is None:
        return headroom
    return min(headroom, max(0, int(cap_bytes)))


class BlockSwapManager:
    def __init__(
        self,
        *,
        total_blocks: int,
        blocks_to_swap: int,
        mode: str = "sync",
        block_swap_backward: bool = True,
        block_residency_planner: str = "uniform",
        block_swap_prefetch_depth: int = 1,
        block_residency_priority: str = "index",
        block_swap_transfer_strategy: str = "per_tensor",
        block_scores: Mapping[int, float] | Sequence[float] | None = None,
        disk_offload_dir: str | None = None,
    ):
        self.total_blocks = max(0, int(total_blocks))
        self.blocks_to_swap = max(0, min(int(blocks_to_swap), self.total_blocks))
        self.mode = mode.strip().lower()
        if self.mode not in {"sync", "async"}:
            raise ValueError(f"Unsupported block swap mode '{mode}'.")
        self.block_swap_backward = bool(block_swap_backward)
        self._planner = str(block_residency_planner)
        self._priority = str(block_residency_priority)
        self._prefetch_depth = int(block_swap_prefetch_depth)
        self._block_scores = block_scores
        self._transfer_strategy = str(block_swap_transfer_strategy).strip().lower()
        self._disk_offload_dir = (
            Path(disk_offload_dir) if str(disk_offload_dir or "").strip() else None
        )
        if self._transfer_strategy not in {"per_tensor", "flat_ring"}:
            raise ValueError(
                "block_swap_transfer_strategy must be one of: per_tensor, flat_ring."
            )
        self._plan = self._build_plan()
        # ``gpu_blocks`` is the count of blocks that remain resident; the plan
        # owns their concrete indices.
        self.gpu_blocks = len(self._plan.resident_blocks)
        self.events: list[BlockSwapEvent] = []
        self._prefetch_q: deque[int] = deque()
        self._units: dict[int, ImmutableModuleResidency] = {}
        self._device: Any | None = None
        self._transfer_stream: Any | None = None
        self._prefetch_events: dict[int, Any] = {}
        self._backward_hooks: list[Any] = []
        self._backward_active = False
        self._flat_ring: FlatRingResidencyPool | None = None

    def _build_plan(self) -> ResidencyPlan:
        return build_residency_plan(
            total_blocks=self.total_blocks,
            blocks_to_swap=self.blocks_to_swap,
            planner=self._planner,
            prefetch_depth=self._prefetch_depth,
            priority=self._priority,
            block_scores=self._block_scores,
        )

    @property
    def plan(self) -> ResidencyPlan:
        return self._plan

    def set_block_scores(
        self, block_scores: Mapping[int, float] | Sequence[float] | None
    ) -> None:
        """Provide hot-block scores and re-plan residency (call before bind)."""
        self._block_scores = block_scores
        self._plan = self._build_plan()
        self.gpu_blocks = len(self._plan.resident_blocks)

    def _is_resident(self, idx: int) -> bool:
        return idx in self._plan.resident_set

    def _host_pin_budget_bytes(self) -> int | None:
        if torch is None or self._device is None or self._device.type != "cuda":
            return 0
        policy = current_memory_safety_policy()
        cap_gib = float(policy.max_pinned_host_gib)
        cap_bytes = int(cap_gib * (1024**3)) if cap_gib > 0.0 else None
        try:
            import psutil
        except ImportError:
            # Without a free-memory reading, retain the configured absolute cap.
            # ``None`` would represent an unbounded pinning budget.
            return cap_bytes if cap_bytes is not None else 0
        available = int(psutil.virtual_memory().available)
        floor = int(policy.minimum_system_memory_gib * (1024**3))
        return bounded_pin_budget_bytes(available, floor, cap_bytes)

    def bind(self, units: list[tuple[int, Any]], *, device: Any) -> None:
        for handle in self._backward_hooks:
            handle.remove()
        self._backward_hooks.clear()
        self._device = torch.device(device) if torch is not None else device
        self._transfer_stream = (
            torch.cuda.Stream(device=self._device)
            if torch is not None and self._device.type == "cuda" and self.mode == "async"
            else None
        )
        remaining = self._host_pin_budget_bytes()
        self._units = {}
        swap_modules: list[tuple[int, Any]] = []
        for index, module in units:
            idx = int(index)
            resident = self._is_resident(idx)
            if resident or self._transfer_strategy == "per_tensor":
                unit_budget = 0 if resident else remaining
                disk_path = (
                    self._disk_offload_dir / f"block_{idx:04d}.safetensors"
                    if self._disk_offload_dir is not None and not resident
                    else None
                )
                unit = ImmutableModuleResidency(
                    module,
                    pin_budget_bytes=unit_budget,
                    disk_path=disk_path,
                )
                if remaining is not None and not resident:
                    remaining = max(0, remaining - unit.pinned_bytes)
                self._units[idx] = unit
            else:
                swap_modules.append((idx, module))
        self._flat_ring = None
        if swap_modules:
            if self._device.type != "cuda":
                raise ValueError("flat_ring block swap requires a CUDA device.")
            self._flat_ring = FlatRingResidencyPool(
                [module for _, module in swap_modules],
                ring_size=self._prefetch_depth + 1,
                pin_budget_bytes=remaining,
            )
            for (idx, _), unit in zip(
                swap_modules, self._flat_ring.units, strict=True
            ):
                self._units[idx] = unit
        for index, unit in self._units.items():
            if self._is_resident(index):
                unit.load(self._device)
            else:
                unit.offload()
                module = unit.module

                def offload_after_backward(
                    _module: Any,
                    _grad_input: Any,
                    _grad_output: Any,
                    *,
                    block_index: int = index,
                ) -> None:
                    self._backward_active = True
                    # Head swap blocks are needed first in the NEXT forward; keep
                    # them resident across the bwd->fwd (step) turn instead of
                    # evicting and immediately re-fetching.
                    if block_index in self._plan.backward_pin_set:
                        return
                    managed = self._units.get(block_index)
                    if managed is not None and managed.device != "cpu":
                        if self._device.type == "cuda" and hasattr(managed, "mark_consumed"):
                            managed.mark_consumed(torch.cuda.current_stream(self._device))
                        managed.offload()
                        self.events.append(
                            BlockSwapEvent(kind="backward_swap_out", block_idx=block_index)
                        )

                self._backward_hooks.append(
                    module.register_full_backward_hook(offload_after_backward)
                )

    def reset_step(self) -> None:
        self.finish_backward()
        self.events.clear()
        self._prefetch_q.clear()
        self._prefetch_events.clear()

    def finish_backward(self) -> None:
        for index, unit in self._units.items():
            if self._is_resident(index):
                continue
            # Head pins stay resident across the step turn for the next forward.
            if index in self._plan.backward_pin_set:
                continue
            unit.offload()
        self._prefetch_q.clear()
        self._prefetch_events.clear()
        self._backward_active = False

    def _prefetch_one(self, nxt: int) -> None:
        if not (0 <= nxt < self.total_blocks) or self._is_resident(nxt):
            return
        if nxt in self._prefetch_events:
            return
        self._prefetch_q.append(nxt)
        self.events.append(BlockSwapEvent(kind="prefetch", block_idx=nxt))
        next_unit = self._units.get(nxt)
        if next_unit is not None and next_unit.device == "cpu":
            event = next_unit.load(self._device, stream=self._transfer_stream)
            if event is not None:
                self._prefetch_events[nxt] = event

    def before_block(self, block_idx: int) -> None:
        assert_system_memory_headroom(context=f"block {int(block_idx)} swap-in")
        idx = int(block_idx)
        if idx < 0 or idx >= self.total_blocks:
            raise IndexError(idx)
        if self._is_resident(idx):
            return
        unit = self._units.get(idx)
        if unit is not None:
            event = self._prefetch_events.pop(idx, None)
            if event is not None and torch is not None and self._device.type == "cuda":
                current = torch.cuda.current_stream(self._device)
                current.wait_event(event)
                unit.record_stream(current)
            elif unit.device == "cpu":
                unit.load(self._device)
        self.events.append(BlockSwapEvent(kind="swap_in", block_idx=idx))
        if self.mode == "async":
            step = -1 if self._backward_active else 1
            for depth in range(1, self._prefetch_depth + 1):
                self._prefetch_one(idx + step * depth)

    def after_block(self, block_idx: int) -> None:
        idx = int(block_idx)
        if self._is_resident(idx):
            return
        self.events.append(BlockSwapEvent(kind="swap_out", block_idx=idx))
        # Tail swap blocks are needed FIRST in backward; keep them resident
        # through the fwd->bwd turn rather than evict + immediate re-fetch. The
        # hold is a scheduling decision (independent of device binding).
        pin_hold = (
            self.block_swap_backward
            and not self._backward_active
            and idx in self._plan.forward_pin_set
        )
        if pin_hold:
            self.events.append(BlockSwapEvent(kind="phase_pin_hold", block_idx=idx))
        unit = self._units.get(idx)
        if unit is not None and self.block_swap_backward and not pin_hold:
            if self._device.type == "cuda" and hasattr(unit, "mark_consumed"):
                unit.mark_consumed(torch.cuda.current_stream(self._device))
            unit.offload()
        if self.mode == "async" and self._prefetch_q:
            done = self._prefetch_q.popleft()
            self.events.append(BlockSwapEvent(kind="prefetch_done", block_idx=done))
        assert_system_memory_headroom(context=f"block {idx} swap-out")

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_blocks": self.total_blocks,
            "blocks_to_swap": self.blocks_to_swap,
            "mode": self.mode,
            "block_swap_backward": self.block_swap_backward,
            "gpu_blocks": self.gpu_blocks,
            "block_residency_planner": self._plan.planner,
            "block_residency_priority": self._plan.priority,
            "block_swap_prefetch_depth": self._plan.prefetch_depth,
            "block_swap_transfer_strategy": self._transfer_strategy,
            "ring_slots": 0 if self._flat_ring is None else self._flat_ring.ring_size,
            "ring_buffer_bytes": (
                0 if self._flat_ring is None else self._flat_ring.ring_buffer_bytes
            ),
            "h2d_copies": 0 if self._flat_ring is None else self._flat_ring.h2d_copies,
            "h2d_bytes": 0 if self._flat_ring is None else self._flat_ring.h2d_bytes,
            "resident_blocks": list(self._plan.resident_blocks),
            "forward_to_backward_pins": list(self._plan.forward_to_backward_pins),
            "backward_to_forward_pins": list(self._plan.backward_to_forward_pins),
            "resident_bytes": sum(
                unit.bytes for unit in self._units.values() if unit.device != "cpu"
            ),
            "offloaded_bytes": sum(
                unit.bytes for unit in self._units.values() if unit.device == "cpu"
            ),
            "disk_backing_bytes": sum(
                unit.backing_store_bytes for unit in self._units.values()
            ),
            "events": [{"kind": e.kind, "block_idx": e.block_idx} for e in self.events],
        }

    def device_residency_reservations(self) -> dict[str, int]:
        """Return the conservative steady/transient VRAM promises of this manager."""
        sizes = {index: int(unit.bytes) for index, unit in self._units.items()}
        return self._device_residency_reservations_from_sizes(sizes)

    def estimate_device_residency_reservations(
        self, units: Sequence[tuple[int, Any]]
    ) -> dict[str, int]:
        """Estimate the same promises without moving or pinning module tensors."""
        sizes = {
            int(index): _immutable_module_bytes(module)
            for index, module in units
        }
        return self._device_residency_reservations_from_sizes(sizes)

    def _device_residency_reservations_from_sizes(
        self, sizes: Mapping[int, int]
    ) -> dict[str, int]:
        resident_bytes = sum(
            int(size) for index, size in sizes.items() if self._is_resident(index)
        )
        if self._flat_ring is not None:
            transfer_bytes = int(self._flat_ring.ring_buffer_bytes)
        else:
            swap_sizes = sorted(
                (
                    int(size)
                    for index, size in sizes.items()
                    if not self._is_resident(index)
                ),
                reverse=True,
            )
            slots = min(
                len(swap_sizes),
                max(
                    1 + self._prefetch_depth,
                    len(self._plan.forward_to_backward_pins),
                    len(self._plan.backward_to_forward_pins),
                ),
            )
            transfer_bytes = sum(swap_sizes[:slots])
        return {
            "block_resident_set": int(resident_bytes),
            "block_transfer_window": int(transfer_bytes),
        }


def _immutable_module_bytes(module: Any) -> int:
    seen: set[int] = set()
    total = 0
    for owner in module.modules():
        for parameter in owner._parameters.values():
            if parameter is None or bool(parameter.requires_grad):
                continue
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                total += int(parameter.numel()) * int(parameter.element_size())
        for buffer in owner._buffers.values():
            if buffer is None or not bool(getattr(buffer, "numel", None)):
                continue
            identity = id(buffer)
            if identity not in seen:
                seen.add(identity)
                total += int(buffer.numel()) * int(buffer.element_size())
    return total
