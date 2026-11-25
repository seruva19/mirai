"""Stage-boundary contracts for block-residency device teardown."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from mirai.core.training.residency import block_swap
from mirai.core.training.residency.block_swap import BlockSwapEvent, BlockSwapManager
from mirai.core.training.residency.memory_safety import MemorySafetyPolicy

torch = pytest.importorskip("torch")


def test_memory_check_interval_retries_failed_check_and_observes_policy_changes(monkeypatch):
    policy = MemorySafetyPolicy(block_swap_memory_check_interval_seconds=2)
    now = 0.0
    available = True
    calls = []

    def check(*, context):
        calls.append(context)
        if not available:
            raise MemoryError("low memory")

    monkeypatch.setattr(block_swap, "current_memory_safety_policy", lambda: policy)
    monkeypatch.setattr(block_swap, "monotonic", lambda: now)
    monkeypatch.setattr(block_swap, "assert_system_memory_headroom", check)
    manager = BlockSwapManager(total_blocks=1, blocks_to_swap=1)
    manager.bind([], device="cpu")
    manager.before_block(0)
    manager.after_block(0)
    assert calls == ["block swap bind"]

    now = 2.0
    available = False
    for _ in range(2):
        with pytest.raises(MemoryError, match="low memory"):
            manager.before_block(0)
    available = True
    manager.before_block(0)
    manager.after_block(0)
    assert calls[1:] == ["block 0 swap-in"] * 3

    policy = MemorySafetyPolicy(block_swap_memory_check_interval_seconds=5)
    manager.before_block(0)
    assert len(calls) == 5
    policy = MemorySafetyPolicy()
    manager.before_block(0)
    manager.after_block(0)
    assert calls[-2:] == ["block 0 swap-in", "block 0 swap-out"]


class _RecordingUnit:
    def __init__(self, index: int) -> None:
        self.index = int(index)
        self.offloads = 0

    def offload(self) -> None:
        self.offloads += 1


class _RecordingHandle:
    def __init__(self) -> None:
        self.removed = 0

    def remove(self) -> None:
        self.removed += 1


class _RecordingRing:
    def __init__(self) -> None:
        self.releases = 0

    def release_device(self) -> None:
        self.releases += 1


def test_release_device_drops_resident_prefetched_and_ring_state() -> None:
    manager = BlockSwapManager(total_blocks=3, blocks_to_swap=1, mode="async")
    unit_list = [_RecordingUnit(index) for index in range(3)]
    units = {unit.index: unit for unit in unit_list}
    handle = _RecordingHandle()
    ring = _RecordingRing()
    manager._device = SimpleNamespace(type="cpu")
    manager._units = units
    manager._transfer_stream = object()
    manager._prefetch_q = deque([1, 2])
    manager._prefetch_events = {1: object(), 2: object()}
    manager._backward_hooks = [handle]
    manager._backward_active = True
    manager._flat_ring = ring
    manager.events = [BlockSwapEvent(kind="prefetch", block_idx=1)]
    plan = manager.plan

    manager.release_device()

    assert [unit.offloads for unit in unit_list] == [1, 1, 1]
    assert handle.removed == 1
    assert ring.releases == 1
    assert manager.plan is plan
    assert manager._units == {}
    assert manager._device is None
    assert manager._transfer_stream is None
    assert manager._flat_ring is None
    assert manager._prefetch_events == {}
    assert list(manager._prefetch_q) == []
    assert manager._backward_hooks == []
    assert manager._backward_active is False
    assert manager.events == []


def test_release_device_can_rebind_the_same_modules() -> None:
    blocks = [torch.nn.Linear(4, 4, bias=False) for _ in range(2)]
    for block in blocks:
        block.weight.requires_grad_(False)
    inputs = torch.randn(2, 4)
    expected = blocks[1](blocks[0](inputs))
    manager = BlockSwapManager(total_blocks=2, blocks_to_swap=1, mode="sync")
    indexed = list(enumerate(blocks))

    manager.bind(indexed, device=torch.device("cpu"))
    manager.release_device()
    assert manager._units == {}
    assert all(parameter.device.type == "cpu" for block in blocks for parameter in block.parameters())

    manager.bind(indexed, device=torch.device("cpu"))
    actual = blocks[1](blocks[0](inputs))
    assert len(manager._units) == 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    manager.release_device()
