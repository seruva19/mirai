"""Behavioral contracts for router temperature scheduling and jitter."""

from __future__ import annotations

import copy
import math
import unittest

import torch
import torch.nn.functional as F

from mirai.config.schema import ModelConfig, ModelParams, TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.adaptation.temperature import RouterTemperatureController
from mirai.core.training.training_policy import TrainingPolicySet
from mirai.core.training.training_policy import validate_training_policy_configs


class RouterTemperatureControllerTests(unittest.TestCase):
    def test_identity_configuration_returns_same_tensor(self) -> None:
        logits = torch.randn(8, 4, requires_grad=True)
        controller = RouterTemperatureController()
        controller.bind_step(step=7, training=True)

        actual = controller.transform_logits(
            "layer", logits, training=True
        )

        self.assertIs(actual, logits)

    def test_linear_and_sigmoid_schedules_follow_declared_math(self) -> None:
        linear = RouterTemperatureController(
            temperature=1.0,
            minimum_temperature=0.2,
            schedule="linear",
            start_step=10,
            end_step=20,
        )
        self.assertEqual(linear.scheduled_temperature(10), 1.0)
        self.assertAlmostEqual(linear.scheduled_temperature(15), 0.6)
        self.assertEqual(linear.scheduled_temperature(20), 0.2)

        sigmoid = RouterTemperatureController(
            temperature=1.0,
            minimum_temperature=0.2,
            schedule="sigmoid",
            start_step=0,
            end_step=10,
            sigmoid_sharpness=7.0,
        )
        self.assertAlmostEqual(sigmoid.scheduled_temperature(5), 0.6)

    def test_jitter_is_training_only_and_checkpoint_replayable(self) -> None:
        logits = torch.ones(128, 4)
        controller = RouterTemperatureController(jitter_epsilon=0.01)
        controller.bind_step(step=0, training=True)
        rng = torch.get_rng_state()
        first = controller.transform_logits("layer", logits, training=True)
        torch.set_rng_state(rng)
        second = controller.transform_logits("layer", logits, training=True)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(first, logits))

        controller.bind_step(step=0, training=False)
        self.assertIs(
            controller.transform_logits("layer", logits, training=False),
            logits,
        )

    def test_entropy_floor_freezes_current_temperature(self) -> None:
        controller = RouterTemperatureController(
            temperature=1.0,
            minimum_temperature=0.2,
            schedule="linear",
            start_step=0,
            end_step=10,
            entropy_floor=0.5,
        )
        controller.bind_step(step=4, training=True)
        self.assertAlmostEqual(controller.scheduled_temperature(), 0.68)
        controller.observe_entropy(0.4, training=True)
        controller.bind_step(step=9, training=True)

        self.assertTrue(controller.annealing_frozen)
        self.assertAlmostEqual(controller.scheduled_temperature(), 0.68)

    def test_state_roundtrip_preserves_frozen_trajectory(self) -> None:
        source = RouterTemperatureController(
            temperature=1.0,
            minimum_temperature=0.3,
            schedule="linear",
            start_step=0,
            end_step=10,
            entropy_floor=0.5,
        )
        source.bind_step(step=3, training=True)
        source.observe_entropy(0.25, training=True)
        payload = copy.deepcopy(source.state_dict())

        restored = RouterTemperatureController(
            temperature=1.0,
            minimum_temperature=0.3,
            schedule="linear",
            start_step=0,
            end_step=10,
            entropy_floor=0.5,
        )
        restored.load_state_dict(payload)

        self.assertEqual(restored.state_dict(), payload)
        self.assertEqual(
            restored.scheduled_temperature(),
            source.scheduled_temperature(),
        )


class RouterTemperatureTrainingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin_components()

    @staticmethod
    def _config() -> TrainingConfig:
        config = TrainingConfig()
        config.model.type = "lingbot-video"
        config.model.params.num_experts = 4
        config.training.policy_options = {
            "router_temperature": {
                "enabled": True,
                "temperature": 1.0,
                "minimum_temperature": 0.3,
                "schedule": "linear",
                "start_step": 0,
                "end_step": 10,
                "jitter_epsilon": 0.01,
                "entropy_floor": 0.4,
            }
        }
        return config

    def test_default_has_no_policy_and_enabled_policy_is_checkpointed(self) -> None:
        default = TrainingPolicySet.from_config(TrainingConfig())
        self.assertNotIn("router_temperature", default.active_names)

        enabled = TrainingPolicySet.from_config(self._config())
        self.assertIn("router_temperature", enabled.active_names)
        self.assertIn(
            "router_temperature",
            enabled.checkpoint_metadata()["policies"],
        )

    def test_validation_rejects_entropy_floor_above_router_maximum(self) -> None:
        config = self._config()
        config.training.policy_options["router_temperature"][
            "entropy_floor"
        ] = math.log(4) + 0.1

        errors = validate_training_policy_configs(config)

        self.assertTrue(any("log(num_experts)" in error for error in errors))

    def test_lingbot_provider_applies_temperature_before_sigmoid(self) -> None:
        pipeline = LingBotVideoPipeline(
            ModelConfig(
                type="lingbot-video",
                path="./models/lingbot_video",
                params=ModelParams(
                    variant="tiny-video",
                    latent_channels=2,
                    num_experts=4,
                    experts_per_token=2,
                    shared_experts=1,
                    hidden_size=16,
                    num_layers=1,
                    attention_heads=2,
                    patch_size=1,
                ),
            )
        )
        controller = RouterTemperatureController(
            temperature=2.0,
            minimum_temperature=2.0,
        )
        pipeline.configure_training_policy("router_temperature", controller)
        pipeline.train()
        controller.bind_step(step=0, training=True)
        router = pipeline._moe_router_modules[0]
        tokens = torch.randn(8, 16)

        _, _, raw_logits, scores, _ = router(tokens)

        expected_raw = F.linear(tokens.float(), router.weight.float())
        torch.testing.assert_close(raw_logits, expected_raw)
        torch.testing.assert_close(scores, (expected_raw / 2.0).sigmoid())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
