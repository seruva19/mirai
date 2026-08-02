"""Behavioral contract for rectified-flow timestep sampling."""

from __future__ import annotations

import math
import unittest

from mirai.config.schema import TrainingConfig
from mirai.core.training.objectives.sampling import (
    LogitNormalTimestepSampler,
    ModeShiftTimestepSampler,
    UniformTimestepSampler,
    build_timestep_sampler,
)
from mirai.core.training.runtime.contract import validate_training_runtime_config

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class TimestepSamplingConfigContract(unittest.TestCase):
    def test_invalid_distribution_parameters_fail_before_training(self) -> None:
        config = TrainingConfig()
        config.training.timestep_sampling = "unknown"
        config.training.timestep_sampling_std = 0.0
        config.training.timestep_sampling_mode_scale = 2.0
        with self.assertRaises(ValueError) as context:
            validate_training_runtime_config(config)
        message = str(context.exception)
        self.assertIn("training.timestep_sampling must be one of", message)
        self.assertIn("training.timestep_sampling_std", message)
        self.assertIn("training.timestep_sampling_mode_scale", message)


@unittest.skipIf(torch is None, "torch not installed")
class TimestepSamplingBehaviorContract(unittest.TestCase):
    def test_uniform_factory_preserves_existing_seeded_sequence(self) -> None:
        direct_python = UniformTimestepSampler(seed=17, eps=1e-5)
        factory_python = build_timestep_sampler(
            mode="uniform",
            seed=17,
            eps=1e-5,
        )
        self.assertEqual(
            factory_python.sample(128),
            direct_python.sample(128),
        )

        direct_torch = UniformTimestepSampler(seed=17, eps=1e-5)
        factory_torch = build_timestep_sampler(
            mode="uniform",
            seed=17,
            eps=1e-5,
        )
        like = torch.empty(128, dtype=torch.float64)
        torch.testing.assert_close(
            factory_torch.sample(128, like=like),
            direct_torch.sample(128, like=like),
            rtol=0,
            atol=0,
        )

    def test_logit_normal_matches_configured_moments_and_bounds(self) -> None:
        sampler = LogitNormalTimestepSampler(
            seed=23,
            eps=1e-9,
            mean=0.3,
            std=1.2,
        )
        values = sampler.sample(
            100_000,
            like=torch.empty(1, dtype=torch.float64),
        )
        logits = torch.logit(values)
        self.assertAlmostEqual(float(logits.mean()), 0.3, delta=0.015)
        self.assertAlmostEqual(
            float(logits.std(unbiased=False)),
            1.2,
            delta=0.015,
        )
        self.assertGreaterEqual(float(values.min()), 1e-9)
        self.assertLessEqual(float(values.max()), 1.0 - 1e-9)

    def test_mode_sampler_implements_the_published_transform(self) -> None:
        seed = 31
        scale = 1.29
        sampler = ModeShiftTimestepSampler(
            seed=seed,
            eps=1e-5,
            scale=scale,
        )
        like = torch.empty(16, dtype=torch.float64)
        observed = sampler.sample(16, like=like)

        generator = torch.Generator()
        generator.manual_seed(seed)
        uniform = torch.rand(
            (16,),
            generator=generator,
            dtype=torch.float64,
        )
        expected = (
            1.0
            - uniform
            - scale
            * (
                torch.cos((math.pi / 2.0) * uniform).square()
                - 1.0
                + uniform
            )
        ).clamp(min=1e-5, max=1.0 - 1e-5)
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    def test_opt_in_sampler_state_round_trips_exactly(self) -> None:
        factories = (
            lambda seed: LogitNormalTimestepSampler(
                seed=seed,
                mean=-0.2,
                std=0.7,
            ),
            lambda seed: ModeShiftTimestepSampler(seed=seed, scale=1.1),
        )
        for factory in factories:
            with self.subTest(sampler=factory(0).__class__.__name__):
                source = factory(43)
                source.sample(7)
                source.sample(7, like=torch.empty(1, dtype=torch.float32))
                state = source.state_dict()
                expected_python = source.sample(64)
                expected_torch = source.sample(
                    64,
                    like=torch.empty(1, dtype=torch.float32),
                )

                restored = factory(999)
                restored.load_state_dict(state)
                self.assertEqual(restored.sample(64), expected_python)
                torch.testing.assert_close(
                    restored.sample(
                        64,
                        like=torch.empty(1, dtype=torch.float32),
                    ),
                    expected_torch,
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
