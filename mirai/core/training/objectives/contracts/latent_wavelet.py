"""Behavioral contract for latent-space Haar supervision."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from mirai.config.schema import TrainingConfig
from mirai.core.training.objectives.latent_wavelet import (
    compute_latent_wavelet_loss,
    reconstruct_clean_rectified_flow,
)
from mirai.core.training.objectives.flow_matching import FlowMatchingObjective
from mirai.core.training.runtime.contract import validate_training_runtime_config

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class LatentWaveletFormulaContract(unittest.TestCase):
    @staticmethod
    def _strategy():
        class Strategy:
            @staticmethod
            def compute_per_sample_loss(*, prediction, target, inputs, loss_fn):
                _ = inputs
                return loss_fn(prediction, target).flatten(1).mean(dim=1)

        return Strategy()

    def test_exact_orthonormal_haar_formula_and_gradient(self) -> None:
        predicted = torch.tensor(
            [[[[1.0, 2.0], [3.0, 5.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.zeros_like(predicted)
        observed = compute_latent_wavelet_loss(
            predicted_clean=predicted,
            clean_latents=target,
        )
        expected_bands = torch.tensor([5.5, -1.5, -2.5, 0.5])
        expected = expected_bands.square().sum()
        torch.testing.assert_close(
            observed.per_sample_loss,
            expected.reshape(1),
            rtol=0,
            atol=0,
        )
        observed.per_sample_loss.sum().backward()
        observed_gradient = predicted.grad.detach().clone()
        predicted.grad = None
        expected_pixel_loss = 4.0 * predicted.square().mean()
        expected_pixel_loss.backward()
        torch.testing.assert_close(
            observed_gradient,
            predicted.grad,
            rtol=0,
            atol=0,
        )

    def test_rectified_flow_clean_reconstruction_is_exact(self) -> None:
        clean = torch.randn(2, 3, 2, 4, 6, dtype=torch.float64)
        noise = torch.randn_like(clean)
        sigma = torch.tensor([0.2, 0.75], dtype=torch.float64)
        view = sigma.reshape(2, 1, 1, 1, 1)
        noisy = clean * (1.0 - view) + noise * view
        velocity = noise - clean
        observed = reconstruct_clean_rectified_flow(
            prediction=velocity,
            noisy_latents=noisy,
            sigmas=sigma,
        )
        torch.testing.assert_close(observed, clean, rtol=0, atol=1e-15)

    def test_video_frames_are_not_mixed_by_spatial_transform(self) -> None:
        predicted = torch.zeros(1, 1, 2, 2, 2)
        predicted[:, :, 1] = 1.0
        observed = compute_latent_wavelet_loss(
            predicted_clean=predicted,
            clean_latents=torch.zeros_like(predicted),
        )
        self.assertEqual(float(observed.per_sample_loss), 2.0)

    def test_disabled_objective_preserves_loss_and_gradient_exactly(self) -> None:
        prediction = torch.randn(2, 1, 2, 4, 4, requires_grad=True)
        target = torch.randn_like(prediction)
        objective = FlowMatchingObjective()
        result = objective.compute_per_sample_loss(
            prediction=prediction,
            target=target,
            inputs=SimpleNamespace(),
            loss_fn=lambda left, right: (left - right).square(),
            strategy=self._strategy(),
            config=SimpleNamespace(
                training=SimpleNamespace(
                    contrastive_flow_weight=0.0,
                    latent_wavelet_loss_weight=0.0,
                )
            ),
        )
        expected = (prediction - target).square().flatten(1).mean(dim=1)
        torch.testing.assert_close(result.per_sample_loss, expected, rtol=0, atol=0)
        result.per_sample_loss.sum().backward()
        observed_gradient = prediction.grad.detach().clone()
        prediction.grad = None
        expected.sum().backward()
        torch.testing.assert_close(
            observed_gradient,
            prediction.grad,
            rtol=0,
            atol=0,
        )

    def test_enabled_objective_adds_weighted_wavelet_term(self) -> None:
        clean = torch.randn(2, 1, 2, 4, 4)
        noise = torch.randn_like(clean)
        sigma = torch.tensor([0.25, 0.75])
        view = sigma.reshape(2, 1, 1, 1, 1)
        noisy = clean * (1.0 - view) + noise * view
        prediction = torch.zeros_like(clean, requires_grad=True)
        target = noise - clean
        weight = 0.1
        objective = FlowMatchingObjective()
        result = objective.compute_per_sample_loss(
            prediction=prediction,
            target=target,
            inputs=SimpleNamespace(
                noisy_latents=noisy,
                timestep=sigma,
                clean_latents=clean,
            ),
            loss_fn=lambda left, right: (left - right).square(),
            strategy=self._strategy(),
            config=SimpleNamespace(
                training=SimpleNamespace(
                    contrastive_flow_weight=0.0,
                    latent_wavelet_loss_weight=weight,
                )
            ),
        )
        predicted_clean = noisy - view * prediction
        wavelet = compute_latent_wavelet_loss(
            predicted_clean=predicted_clean,
            clean_latents=clean,
        )
        expected = target.square().flatten(1).mean(dim=1)
        expected = expected + weight * wavelet.per_sample_loss
        torch.testing.assert_close(result.per_sample_loss, expected, rtol=0, atol=0)
        self.assertIn("latent_wavelet_high_frequency_loss", result.diagnostics)
        result.per_sample_loss.sum().backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))


class LatentWaveletConfigContract(unittest.TestCase):
    def test_disabled_default_and_supported_configuration_validate(self) -> None:
        config = TrainingConfig()
        self.assertEqual(config.training.latent_wavelet_loss_weight, 0.0)
        validate_training_runtime_config(config)
        config.training.latent_wavelet_loss_weight = 0.1
        validate_training_runtime_config(config)

    def test_invalid_or_incompatible_configurations_fail(self) -> None:
        cases = (
            ("weight", -0.1, "finite and >= 0"),
            ("objective", "regression", "objective='flow_matching'"),
            ("loss_function", "huber", "loss_function='mse'"),
            ("strategy", "image_to_video", "strategy.type='text_to_video'"),
            ("contrastive", 0.05, "contrastive_flow_weight"),
            ("masked_loss", True, "masked_loss"),
        )
        for field, value, expected in cases:
            with self.subTest(field=field):
                config = TrainingConfig()
                config.training.latent_wavelet_loss_weight = 0.1
                if field == "weight":
                    config.training.latent_wavelet_loss_weight = value
                elif field == "strategy":
                    config.strategy.type = value
                elif field == "contrastive":
                    config.training.batch_size = 2
                    config.training.contrastive_flow_weight = value
                else:
                    setattr(config.training, field, value)
                with self.assertRaisesRegex(ValueError, expected):
                    validate_training_runtime_config(config)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
