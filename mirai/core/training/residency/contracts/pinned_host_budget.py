"""Host-pin budget invariants for block swap.

The block-swap residency pins page-locked host memory to speed H2D copies.
These CPU-only tests enforce two invariants without allocating CUDA memory:

* the budget is ``min(free_ram - floor, cap)`` -- a huge ``available`` no
  longer yields a huge budget, the cap is configurable, the floor is kept;
* pinned vs pageable staging is numerically identical, so lowering the cap
  changes footprint only, never training outputs.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

from mirai.config.schema import MemoryConfig
from mirai.core.training.residency.block_swap import (
    BlockSwapManager,
    bounded_pin_budget_bytes,
)
from mirai.core.training.residency.memory_safety import configure_memory_safety

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

_GIB = 1024**3


class BoundedBudgetMathTests(unittest.TestCase):
    def test_cap_bounds_a_huge_available(self) -> None:
        # The configured cap bounds a larger available-memory value.
        budget = bounded_pin_budget_bytes(110 * _GIB, 0, 8 * _GIB)
        self.assertEqual(budget, 8 * _GIB)

    def test_headroom_wins_when_below_cap(self) -> None:
        # Available memory above the floor is smaller than the configured cap.
        budget = bounded_pin_budget_bytes(19 * _GIB, 16 * _GIB, 8 * _GIB)
        self.assertEqual(budget, 3 * _GIB)

    def test_floor_is_respected(self) -> None:
        budget = bounded_pin_budget_bytes(20 * _GIB, 16 * _GIB, 100 * _GIB)
        self.assertEqual(budget, 4 * _GIB)

    def test_never_negative_when_floor_exceeds_available(self) -> None:
        self.assertEqual(bounded_pin_budget_bytes(4 * _GIB, 16 * _GIB, 8 * _GIB), 0)

    def test_none_cap_is_legacy_unbounded(self) -> None:
        self.assertEqual(bounded_pin_budget_bytes(110 * _GIB, 0, None), 110 * _GIB)

    def test_cap_is_configurable(self) -> None:
        self.assertEqual(bounded_pin_budget_bytes(110 * _GIB, 0, 2 * _GIB), 2 * _GIB)
        self.assertEqual(bounded_pin_budget_bytes(110 * _GIB, 0, 32 * _GIB), 32 * _GIB)


@unittest.skipIf(torch is None, "torch not installed")
class HostPinBudgetPolicyTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_memory_safety(MemoryConfig())

    def _manager_on_fake_cuda(self) -> BlockSwapManager:
        mgr = BlockSwapManager(total_blocks=4, blocks_to_swap=2)
        # _host_pin_budget_bytes only needs a device that reports type "cuda"
        # and a live torch import -- no real CUDA context is created.
        mgr._device = types.SimpleNamespace(type="cuda")
        return mgr

    def test_huge_available_is_capped_by_policy(self) -> None:
        configure_memory_safety(
            MemoryConfig(minimum_system_memory_gib=0.0, max_pinned_host_gib=8.0)
        )
        mgr = self._manager_on_fake_cuda()
        with mock.patch("psutil.virtual_memory") as vm:
            vm.return_value.available = int(110 * _GIB)
            self.assertEqual(mgr._host_pin_budget_bytes(), 8 * _GIB)

    def test_cap_follows_config(self) -> None:
        configure_memory_safety(
            MemoryConfig(minimum_system_memory_gib=0.0, max_pinned_host_gib=2.0)
        )
        mgr = self._manager_on_fake_cuda()
        with mock.patch("psutil.virtual_memory") as vm:
            vm.return_value.available = int(110 * _GIB)
            self.assertEqual(mgr._host_pin_budget_bytes(), 2 * _GIB)

    def test_floor_still_applies_under_the_cap(self) -> None:
        configure_memory_safety(
            MemoryConfig(minimum_system_memory_gib=16.0, max_pinned_host_gib=8.0)
        )
        mgr = self._manager_on_fake_cuda()
        with mock.patch("psutil.virtual_memory") as vm:
            vm.return_value.available = int(19 * _GIB)
            self.assertEqual(mgr._host_pin_budget_bytes(), 3 * _GIB)

    def test_non_cuda_device_pins_nothing(self) -> None:
        mgr = BlockSwapManager(total_blocks=4, blocks_to_swap=2)
        mgr._device = types.SimpleNamespace(type="cpu")
        self.assertEqual(mgr._host_pin_budget_bytes(), 0)

    def test_missing_psutil_falls_back_to_bounded_cap_not_unlimited(self) -> None:
        # When free RAM cannot be read, the budget remains bounded by the cap.
        import sys

        configure_memory_safety(
            MemoryConfig(minimum_system_memory_gib=0.0, max_pinned_host_gib=8.0)
        )
        mgr = self._manager_on_fake_cuda()
        with mock.patch.dict(sys.modules, {"psutil": None}):
            budget = mgr._host_pin_budget_bytes()
        self.assertIsNotNone(budget)
        self.assertEqual(budget, 8 * _GIB)


@unittest.skipIf(torch is None, "torch not installed")
class PinAmountIsNumericallyInvariantTests(unittest.TestCase):
    """Pinned vs pageable staging must be bit-identical (only H2D speed differs)."""

    def _module(self) -> "nn.Module":
        torch.manual_seed(0)
        module = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return module

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "pinned host allocation requires an available accelerator",
    )
    def test_staged_values_match_across_pin_budgets(self) -> None:
        from mirai.core.training.residency.tensor_residency import ImmutableModuleResidency

        pageable = ImmutableModuleResidency(self._module(), pin_budget_bytes=0)
        pinned = ImmutableModuleResidency(
            self._module(), pin_budget_bytes=64 * 1024 * 1024
        )
        # Same seed -> identical source weights; pinning must not perturb bytes.
        page_state = dict(pageable.module.state_dict())
        pin_state = dict(pinned.module.state_dict())
        self.assertEqual(page_state.keys(), pin_state.keys())
        for key in page_state:
            torch.testing.assert_close(
                page_state[key], pin_state[key], rtol=0, atol=0
            )
        # pin_budget=0 keeps everything pageable; the large budget pins.
        self.assertEqual(pageable.pinned_bytes, 0)
        self.assertGreater(pinned.pinned_bytes, 0)

    def test_cpu_load_result_is_identical(self) -> None:
        from mirai.core.training.residency.tensor_residency import ImmutableModuleResidency

        pageable = ImmutableModuleResidency(self._module(), pin_budget_bytes=0)
        pinned = ImmutableModuleResidency(
            self._module(), pin_budget_bytes=64 * 1024 * 1024
        )
        pageable.load("cpu")
        pinned.load("cpu")
        for key, value in pageable.module.state_dict().items():
            torch.testing.assert_close(
                value, pinned.module.state_dict()[key], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
