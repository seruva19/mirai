"""Behavioral contract for the bounded host-to-device transfer ring."""

from __future__ import annotations

import unittest

from mirai.config.schema import TrainingConfig
from mirai.core.training.residency.block_swap import BlockSwapManager
from mirai.core.training.residency.h2d_ring_residency import FlatRingResidencyPool
from mirai.core.training.runtime.contract import validate_training_runtime_config

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


@unittest.skipIf(torch is None, "torch not installed")
class FlatRingPackingTests(unittest.TestCase):
    def test_cpu_master_is_contiguous_and_trainable_state_is_untouched(self) -> None:
        module = nn.Sequential(nn.Linear(5, 7), nn.LayerNorm(7))
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        trainable = nn.Parameter(torch.randn(3), requires_grad=True)
        module.register_parameter("adapter_weight", trainable)
        expected = {
            name: tensor.detach().clone()
            for name, tensor in module.state_dict().items()
        }

        pool = FlatRingResidencyPool(
            [module], ring_size=2, pin_budget_bytes=0
        )

        self.assertEqual(len(pool.units), 1)
        self.assertEqual(pool.ring_size, 2)
        self.assertIs(module.adapter_weight, trainable)
        self.assertTrue(module.adapter_weight.requires_grad)
        for name, tensor in module.state_dict().items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
        frozen_ptrs = {
            parameter.untyped_storage().data_ptr()
            for parameter in module.parameters()
            if not parameter.requires_grad
        }
        frozen_ptrs.update(
            buffer.untyped_storage().data_ptr() for buffer in module.buffers()
        )
        self.assertEqual(len(frozen_ptrs), 1)

    def test_manager_default_and_opt_in_are_distinct(self) -> None:
        default = BlockSwapManager(total_blocks=4, blocks_to_swap=2)
        flat = BlockSwapManager(
            total_blocks=4,
            blocks_to_swap=2,
            block_swap_transfer_strategy="flat_ring",
        )
        self.assertEqual(
            default.snapshot()["block_swap_transfer_strategy"], "per_tensor"
        )
        self.assertEqual(
            flat.snapshot()["block_swap_transfer_strategy"], "flat_ring"
        )

    def test_invalid_strategy_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "per_tensor, flat_ring"):
            BlockSwapManager(
                total_blocks=2,
                blocks_to_swap=1,
                block_swap_transfer_strategy="implicit",
            )


class FlatRingConfigTests(unittest.TestCase):
    def test_config_default_preserves_per_tensor_path(self) -> None:
        config = TrainingConfig.from_dict({})
        self.assertEqual(config.memory.block_swap_transfer_strategy, "per_tensor")

    def test_runtime_contract_rejects_unknown_strategy(self) -> None:
        config = TrainingConfig.from_dict(
            {"memory": {"block_swap_transfer_strategy": "implicit"}}
        )
        with self.assertRaisesRegex(ValueError, "per_tensor, flat_ring"):
            validate_training_runtime_config(config)


if __name__ == "__main__":
    unittest.main()
