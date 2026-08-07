"""Stage-boundary contracts for block-residency device teardown."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from mirai.core.training.residency.block_swap import BlockSwapEvent, BlockSwapManager

torch = pytest.importorskip("torch")


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
