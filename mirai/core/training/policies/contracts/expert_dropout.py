"""Behavioral contracts for no-token-drop expert-output regularization."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from mirai.config.schema import TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.moe.adaptation.dropout import ExpertDropoutController
from mirai.core.moe.adaptation.dropout import apply_expert_route_dropout
from mirai.core.training.training_policy import TrainingPolicySet
from mirai.core.training.training_policy import validate_training_policy_configs


class ExpertRouteDropoutFormulaTests(unittest.TestCase):
    def test_mask_preserves_token_mass_and_zeroes_dropped_routes(self) -> None:
        scores = torch.tensor(
            [[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]], requires_grad=True
        )
        keep = torch.tensor([[True, False, True], [False, True, False]])
        actual = apply_expert_route_dropout(scores, keep)
        torch.testing.assert_close(actual.sum(dim=-1), scores.sum(dim=-1))
        self.assertEqual(actual[0, 1].item(), 0.0)
        self.assertEqual(actual[1, 0].item(), 0.0)
        self.assertEqual(actual[1, 2].item(), 0.0)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_all_dropped_mask_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "retain at least one"):
            apply_expert_route_dropout(
                torch.ones(2, 2), torch.tensor([[True, False], [False, False]])
            )

    def test_controller_retains_highest_route_when_rng_drops_all(self) -> None:
        controller = ExpertDropoutController(probability=0.4)
        controller.bind_step(step=0, training=True)
        scores = torch.tensor([[0.2, 0.7, 0.1]])
        with patch("torch.rand", return_value=torch.zeros_like(scores)):
            actual = controller.regularize(
                "layer", torch.tensor([[0, 1, 2]]), scores, training=True
            )
        torch.testing.assert_close(actual, torch.tensor([[0.0, 1.0, 0.0]]))

    def test_schedule_and_eval_return_same_tensor(self) -> None:
        scores = torch.tensor([[0.4, 0.6]])
        controller = ExpertDropoutController(
            probability=0.4, start_step=2, end_step=4
        )
        for step, training in ((1, True), (2, False), (4, True)):
            controller.bind_step(step=step, training=training)
            self.assertIs(
                controller.regularize(
                    "layer", torch.tensor([[0, 1]]), scores, training=training
                ),
                scores,
            )

    def test_checkpoint_configuration_mismatch_fails(self) -> None:
        source = ExpertDropoutController(probability=0.4)
        source.bind_step(step=7, training=True)
        target = ExpertDropoutController(probability=0.5)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            target.load_state_dict(source.state_dict())

    def test_reentrant_checkpoint_replays_the_same_dropout_mask(self) -> None:
        controller = ExpertDropoutController(probability=0.4)
        controller.bind_step(step=0, training=True)
        indices = torch.tensor([[0, 1], [1, 2], [2, 0]])

        def objective(values):
            routed = controller.regularize(
                "layer", indices, values, training=True
            )
            return routed.square().sum()

        reference_scores = torch.tensor(
            [[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]], requires_grad=True
        )
        torch.manual_seed(17)
        reference = objective(reference_scores)
        reference.backward()

        checkpoint_scores = reference_scores.detach().clone().requires_grad_(True)
        torch.manual_seed(17)
        actual = torch.utils.checkpoint.checkpoint(
            objective, checkpoint_scores, use_reentrant=True
        )
        actual.backward()
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(checkpoint_scores.grad, reference_scores.grad)


class ExpertDropoutTrainingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin_components()

    def _config(self) -> TrainingConfig:
        config = TrainingConfig()
        config.model.type = "lingbot-video"
        config.model.params.experts_per_token = 2
        config.training.policy_options = {
            "expert_dropout": {
                "enabled": True,
                "probability": 0.4,
                "start_step": 1,
                "end_step": 5,
            }
        }
        return config

    def test_default_config_has_no_policy(self) -> None:
        policies = TrainingPolicySet.from_config(TrainingConfig())
        self.assertNotIn("expert_dropout", policies.active_names)

    def test_enabled_policy_is_registered_and_checkpointed(self) -> None:
        policies = TrainingPolicySet.from_config(self._config())
        self.assertIn("expert_dropout", policies.active_names)
        self.assertIn("expert_dropout", policies.checkpoint_metadata()["policies"])

    def test_top_one_and_orthogonality_combinations_fail(self) -> None:
        config = self._config()
        config.model.params.experts_per_token = 1
        config.model.params.moe_expert_orthogonality_loss_weight = 0.1
        config.model.params.moe_swiglu_specialization_loss_weight = 0.1
        errors = validate_training_policy_configs(config)
        self.assertTrue(any("at least 2" in error for error in errors))
        self.assertTrue(any("orthogonality" in error for error in errors))
        self.assertTrue(any("SwiGLU specialization" in error for error in errors))

    def test_lingbot_router_applies_policy_only_during_bound_window(self) -> None:
        from mirai.config.schema import ModelConfig, ModelParams
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

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
        controller = ExpertDropoutController(
            probability=0.4, start_step=1, end_step=2
        )
        pipeline.configure_expert_dropout(controller)
        pipeline.train()
        router = pipeline._moe_router_modules[0]
        tokens = torch.randn(8, 16)

        controller.bind_step(step=0, training=True)
        torch.manual_seed(9)
        _, baseline_scores, *_ = router(tokens)
        controller.bind_step(step=1, training=True)
        torch.manual_seed(9)
        _, dropped_scores, *_ = router(tokens)

        torch.testing.assert_close(
            dropped_scores.sum(dim=-1), baseline_scores.sum(dim=-1)
        )
        self.assertTrue(bool((dropped_scores == 0).any()))
        self.assertIsNotNone(router.last_route_active_mask)
        self.assertTrue(
            bool((router.last_route_active_mask.sum(dim=-1) >= 1).all())
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
