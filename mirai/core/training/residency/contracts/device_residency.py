"""Behavioral contract for shared single-device residency planning."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from mirai.config.hardware_tiers import resolve_hardware_memory_plan
from mirai.config.schema import MemoryConfig
from mirai.core.models.compressed_weights.execution.expert_device_cache import (
    ExpertDeviceCache,
)
from mirai.core.training.residency.block_swap import BlockSwapManager
from mirai.core.training.residency.device_residency import DeviceResidencyPlanner
from mirai.core.training.residency.tensor_residency import move_trainable_tensors


class DeviceResidencyPlannerTests(unittest.TestCase):
    def test_disabled_planner_records_without_enforcing_a_ceiling(self) -> None:
        planner = DeviceResidencyPlanner()
        planner.replace("experts", 100)
        self.assertFalse(planner.snapshot()["enabled"])
        self.assertEqual(planner.snapshot()["reserved_bytes"], 100)

    def test_shared_ceiling_is_atomic_across_owners(self) -> None:
        planner = DeviceResidencyPlanner(100)
        planner.replace("experts", 40)
        planner.replace_many({"blocks": 50, "transfer_ring": 10})
        self.assertEqual(planner.snapshot()["reserved_bytes"], 100)
        with self.assertRaisesRegex(MemoryError, "experts=40"):
            planner.replace("blocks", 51)
        self.assertEqual(planner.snapshot()["reservations"]["blocks"], 50)

    def test_replace_many_releases_zero_sized_owners(self) -> None:
        planner = DeviceResidencyPlanner(100)
        planner.replace_many({"blocks": 60, "transfer_ring": 20})
        planner.replace_many({"blocks": 0, "transfer_ring": 10})
        self.assertEqual(
            planner.snapshot()["reservations"], {"transfer_ring": 10}
        )

    def test_invalid_capacity_owner_and_size_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity"):
            DeviceResidencyPlanner(-1)
        planner = DeviceResidencyPlanner(10)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            planner.replace("", 1)
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            planner.replace("blocks", -1)

    def test_block_swap_reports_resident_and_transfer_promises(self) -> None:
        modules = [(index, nn.Linear(4, 4)) for index in range(4)]
        for _index, module in modules:
            module.requires_grad_(False)
        manager = BlockSwapManager(
            total_blocks=4,
            blocks_to_swap=2,
            mode="sync",
            block_swap_prefetch_depth=1,
        )
        estimated = manager.estimate_device_residency_reservations(modules)
        manager.bind(modules, device=torch.device("cpu"))
        reservations = manager.device_residency_reservations()
        one_block = sum(
            tensor.numel() * tensor.element_size()
            for tensor in modules[0][1].state_dict().values()
        )
        self.assertEqual(reservations["block_resident_set"], 2 * one_block)
        self.assertEqual(reservations["block_transfer_window"], 2 * one_block)
        self.assertEqual(estimated, reservations)

    def test_expert_device_cache_is_byte_bounded_and_lru(self) -> None:
        cache = ExpertDeviceCache(capacity_bytes=16)
        first = cache.key("layer", 0, "cuda:0")
        second = cache.key("layer", 1, "cuda:0")
        value = (torch.zeros(2), torch.zeros(1))
        self.assertTrue(cache.put(first, value))
        self.assertIsNotNone(cache.get(first))
        self.assertTrue(cache.put(second, value))
        self.assertIsNone(cache.get(first))
        snapshot = cache.snapshot()
        self.assertLessEqual(snapshot["resident_bytes"], snapshot["capacity_bytes"])
        self.assertEqual(snapshot["evictions"], 1)

    def test_hardware_tier_plan_respects_explicit_memory_values(self) -> None:
        memory = MemoryConfig(
            hardware_policy="tiered",
            frozen_weight_quantization="int8",
            device_residency_budget_gib=7.0,
            expert_device_cache_gib=2.0,
            expert_dequant_chunk_size=3,
        )
        plan = resolve_hardware_memory_plan(memory, profile=((9, 0), 80.0))
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.device_residency_budget_gib, 7.0)
        self.assertEqual(plan.expert_device_cache_gib, 2.0)
        self.assertEqual(plan.expert_dequant_chunk_size, 3)

    def test_trainable_tensor_move_preserves_frozen_residency(self) -> None:
        module = nn.Linear(4, 4)
        module.weight.requires_grad_(False)
        module.bias.requires_grad_(True)
        module.bias.grad = torch.ones_like(module.bias)
        frozen_before = module.weight.data_ptr()
        move_trainable_tensors(module, device="cpu", dtype=torch.float64)
        self.assertEqual(module.weight.data_ptr(), frozen_before)
        self.assertEqual(module.weight.dtype, torch.float32)
        self.assertEqual(module.bias.dtype, torch.float64)
        self.assertEqual(module.bias.grad.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
