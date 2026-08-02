"""Behavioral contract for Contrastive Flow Matching."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from mirai.config.schema import StrategyConfig, TrainingConfig
from mirai.core.training.objectives.contrastive_flow import (
    compute_contrastive_flow_loss,
    sample_negative_indices,
)
from mirai.core.training.objectives.engine import compute_training_loss
from mirai.core.training.objectives.flow_matching import FlowMatchingObjective
from mirai.core.training.objectives.sampling import (
    NoiseGenerator,
    UniformTimestepSampler,
)
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.core.training.strategies.text_to_video import TextToVideoStrategy

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class ContrastiveFlowFormulaContract(unittest.TestCase):
    @staticmethod
    def _mse(prediction, target):
        return (prediction - target).square().flatten(1).mean(dim=1)

    def test_exact_published_formula_and_gradient(self) -> None:
        prediction = torch.tensor(
            [[0.2, -0.5], [1.1, 0.7], [-0.3, 0.4]],
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor(
            [[0.0, 0.5], [0.9, -0.2], [0.4, 0.1]],
            dtype=torch.float64,
        )
        negatives = torch.tensor([1, 2, 0])
        weight = 0.05
        observed = compute_contrastive_flow_loss(
            prediction=prediction,
            target=target,
            loss_evaluator=self._mse,
            weight=weight,
            negative_indices=negatives,
        )
        expected = self._mse(prediction, target) - weight * self._mse(
            prediction,
            target.index_select(0, negatives),
        )
        torch.testing.assert_close(
            observed.per_sample_loss,
            expected,
            rtol=0,
            atol=0,
        )

        observed.per_sample_loss.mean().backward()
        observed_gradient = prediction.grad.detach().clone()
        prediction.grad = None
        expected.mean().backward()
        torch.testing.assert_close(
            observed_gradient,
            prediction.grad,
            rtol=0,
            atol=0,
        )

    def test_negative_sampling_is_uniformly_non_self(self) -> None:
        torch.manual_seed(73)
        observed = sample_negative_indices(4096)
        anchors = torch.arange(4096)
        self.assertTrue(bool((observed != anchors).all()))
        self.assertTrue(bool((observed >= 0).all()))
        self.assertTrue(bool((observed < 4096).all()))

    def test_singleton_batch_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            sample_negative_indices(1)

    def test_zero_weight_preserves_loss_and_rng_exactly(self) -> None:
        prediction = torch.randn(4, 3)
        target = torch.randn(4, 3)
        calls = []

        def evaluator(pred, expected):
            calls.append((pred, expected))
            return self._mse(pred, expected)

        rng_before = torch.get_rng_state().clone()
        observed = compute_contrastive_flow_loss(
            prediction=prediction,
            target=target,
            loss_evaluator=evaluator,
            weight=0.0,
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], prediction)
        self.assertIs(calls[0][1], target)
        self.assertIsNone(observed.negative_distance)
        torch.testing.assert_close(
            observed.per_sample_loss,
            self._mse(prediction, target),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            torch.get_rng_state(),
            rng_before,
            rtol=0,
            atol=0,
        )

    def test_flow_objective_exposes_detached_terms_only_when_enabled(self) -> None:
        class Strategy:
            @staticmethod
            def compute_per_sample_loss(*, prediction, target, inputs, loss_fn):
                _ = inputs
                return loss_fn(prediction, target).flatten(1).mean(dim=1)

        prediction = torch.randn(3, 2, requires_grad=True)
        target = torch.randn(3, 2)
        inputs = SimpleNamespace()
        objective = FlowMatchingObjective()
        disabled = objective.compute_per_sample_loss(
            prediction=prediction,
            target=target,
            inputs=inputs,
            loss_fn=lambda pred, expected: (pred - expected).square(),
            strategy=Strategy(),
            config=SimpleNamespace(
                training=SimpleNamespace(contrastive_flow_weight=0.0)
            ),
        )
        self.assertEqual(disabled.diagnostics, {})

        enabled = objective.compute_per_sample_loss(
            prediction=prediction,
            target=target,
            inputs=inputs,
            loss_fn=lambda pred, expected: (pred - expected).square(),
            strategy=Strategy(),
            config=SimpleNamespace(
                training=SimpleNamespace(contrastive_flow_weight=0.05)
            ),
        )
        self.assertEqual(
            set(enabled.diagnostics),
            {
                "contrastive_flow_positive_loss",
                "contrastive_flow_negative_distance",
                "contrastive_flow_weight",
            },
        )
        self.assertFalse(
            enabled.diagnostics["contrastive_flow_positive_loss"].requires_grad
        )
        self.assertTrue(enabled.per_sample_loss.requires_grad)

    def test_enabled_objective_runs_through_training_loss_engine(self) -> None:
        class Pipeline:
            @staticmethod
            def prepare_model_timesteps(timesteps, *, latents):
                _ = latents
                return timesteps

            @staticmethod
            def apply_noise(clean_latents, noise, timesteps):
                view = timesteps.reshape((-1,) + (1,) * (clean_latents.ndim - 1))
                return clean_latents * (1.0 - view) + noise * view

            @staticmethod
            def compute_target(noise, clean_latents, timesteps):
                _ = timesteps
                return noise - clean_latents

            @staticmethod
            def get_training_auxiliary_losses():
                return {}

            @staticmethod
            def get_training_diagnostics():
                return {}

        config = TrainingConfig()
        config.training.batch_size = 2
        config.training.contrastive_flow_weight = 0.05
        strategy = TextToVideoStrategy(StrategyConfig())
        prediction_holder = []

        def predict(inputs):
            prediction = torch.zeros_like(
                inputs.noisy_latents,
                requires_grad=True,
            )
            prediction_holder.append(prediction)
            return prediction

        result = compute_training_loss(
            config=config,
            batch={
                "latents": torch.tensor(
                    [[0.5, -0.25], [1.0, 0.75]],
                    dtype=torch.float64,
                ),
                "text_embeds": torch.zeros(2, 1, dtype=torch.float64),
            },
            pipeline=Pipeline(),
            strategy=strategy,
            objective=FlowMatchingObjective(),
            loss_fn=lambda pred, expected: (pred - expected).square(),
            timestep_sampler=UniformTimestepSampler(seed=11),
            noise_generator=NoiseGenerator(seed=13),
            predict=predict,
            strategy_prepare_accepts_training=True,
            strategy_prepare_accepts_objective=True,
        )
        self.assertTrue(bool(torch.isfinite(result.loss)))
        self.assertIn("contrastive_flow_negative_distance", result.diagnostics)
        result.loss.backward()
        self.assertIsNotNone(prediction_holder[0].grad)
        self.assertTrue(bool(torch.isfinite(prediction_holder[0].grad).all()))


class ContrastiveFlowConfigContract(unittest.TestCase):
    def test_supported_configuration_validates(self) -> None:
        config = TrainingConfig()
        config.training.batch_size = 2
        config.training.contrastive_flow_weight = 0.05
        validate_training_runtime_config(config)

    def test_invalid_or_incompatible_configurations_fail(self) -> None:
        cases = (
            ("contrastive_flow_weight", 1.0, "must be finite and in"),
            ("batch_size", 1, "batch_size >= 2"),
            ("objective", "regression", "objective='flow_matching'"),
            ("loss_function", "huber", "loss_function='mse'"),
            ("loss_weighting", "cosmap", "loss_weighting='uniform'"),
            (
                "loss_bucket_normalization",
                "per_bucket_mean",
                "loss_bucket_normalization",
            ),
        )
        for field, value, expected in cases:
            with self.subTest(field=field):
                config = TrainingConfig()
                config.training.batch_size = 2
                config.training.contrastive_flow_weight = 0.05
                setattr(config.training, field, value)
                with self.assertRaisesRegex(ValueError, expected):
                    validate_training_runtime_config(config)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
