from __future__ import annotations

import unittest

from mirai.config.schema import ModelConfig, ModelParams, StrategyConfig, TrainingConfig, TrainingSection
from mirai.core.builtins import register_builtin_components
from mirai.core.parity.gradient_harness import run_gradient_harness

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class GradientHarnessTests(unittest.TestCase):
    def test_parameter_deltas_are_deterministic(self) -> None:
        register_builtin_components()
        cfg = TrainingConfig(
            preset=None,
            model=ModelConfig(
                type="sparse_moe_test",
                path="./models/sparse_moe_test",
                dtype="bf16",
                attention_backend="auto",
                params=ModelParams(variant="tiny-video", flow_shift=3.0, vae_chunk_size=16),
            ),
            strategy=StrategyConfig(type="text_to_video", params={}),
            training=TrainingSection(
                seed=99,
                batch_size=2,
                gradient_accumulation=1,
                max_steps=8,
                loss_function="mse",
                loss_weighting="uniform",
                min_snr_gamma=5.0,
                timestep_eps=1e-5,
            ),
        )
        first = run_gradient_harness(cfg, steps=8)
        second = run_gradient_harness(cfg, steps=8)

        self.assertEqual(first.steps, 8)
        self.assertEqual(first.param_deltas.keys(), second.param_deltas.keys())
        changed = []
        for key in first.param_deltas:
            self.assertAlmostEqual(first.param_deltas[key], second.param_deltas[key], places=10)
            changed.append(first.param_deltas[key] > 0.0)
        self.assertTrue(any(changed))


if __name__ == "__main__":
    unittest.main()
