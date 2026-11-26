from __future__ import annotations

import unittest

from mirai.config.schema import ModelConfig, ModelParams, StrategyConfig, TrainingConfig, TrainingSection
from mirai.core.builtins import register_builtin_components
from mirai.core.parity.gradient_harness import run_gradient_harness
from mirai.core.training.optim.gradients import PendingGradientOffload, _pack_sparse_gradient, offload_gradients_to_cpu

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class GradientHarnessTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
    def test_sparse_gradient_bitpacking_preserves_bits_and_skips_dense_payloads(self) -> None:
        grad = torch.zeros(16385, device="cuda", dtype=torch.float32)
        grad[::32] = 0.75
        grad[3] = -0.0
        grad[-1] = float("nan")
        packed = _pack_sparse_gradient(grad)
        self.assertIsNotNone(packed.mask)
        self.assertLess(packed.values.nbytes + packed.mask.nbytes, grad.nbytes)
        self.assertIsNone(_pack_sparse_gradient(torch.ones_like(grad)).mask)

        parameter = torch.nn.Parameter(torch.empty_like(grad))
        parameter.grad = grad
        transfers = PendingGradientOffload(bitpack=True)
        self.assertEqual(offload_gradients_to_cpu([parameter], pending=transfers), 1)
        transfers.finish()
        torch.testing.assert_close(parameter._mirai_cpu_grad.view(torch.int32), grad.cpu().view(torch.int32), rtol=0, atol=0)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
    def test_cuda_gradient_offload_drains_previous_microbatch_in_order(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(5, device="cuda", dtype=torch.bfloat16))
        transfers = PendingGradientOffload()
        (parameter * 2).sum().backward()
        self.assertEqual(offload_gradients_to_cpu([parameter], pending=transfers), 1)
        self.assertIsNone(parameter.grad)
        self.assertFalse(hasattr(parameter, "_mirai_cpu_grad"))

        (parameter * 3).sum().backward()
        transfers.finish()
        self.assertEqual(offload_gradients_to_cpu([parameter], pending=transfers), 1)
        transfers.finish()
        transfers.finish()
        self.assertIsNone(parameter.grad)
        self.assertFalse(parameter._mirai_cpu_grad.is_pinned())
        torch.testing.assert_close(parameter._mirai_cpu_grad, torch.full((5,), 5.0), rtol=0, atol=0)

    def test_cpu_gradient_offload_accumulates_each_microbatch_once(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        (parameter * 2.0).backward()
        offload_gradients_to_cpu([parameter])
        self.assertIsNone(parameter.grad)
        (parameter * 3.0).backward()
        offload_gradients_to_cpu([parameter])
        torch.testing.assert_close(
            parameter._mirai_cpu_grad,  # type: ignore[attr-defined]
            torch.tensor(5.0),
        )

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
