"""Reference contracts for diversity-aware co-occurrence routing."""

from __future__ import annotations

# Colocated behavioral contract for diversity-aware routing.

import unittest

import torch

from mirai.config.schema import TrainingConfig
from mirai.core.moe.adaptation.diversity import (
    DiversityAwareRoutingController,
    ExpertCooccurrenceCovariance,
    diversity_aware_batched_topk,
    diversity_aware_greedy_topk,
)
from mirai.core.training.training_policy import TrainingPolicySet
from mirai.core.training.training_policy import validate_training_policy_configs


class ExpertCooccurrenceCovarianceTests(unittest.TestCase):
    def test_exact_population_covariance(self) -> None:
        tracker = ExpertCooccurrenceCovariance(3)
        tracker.update(torch.tensor([[0, 1], [0, 2]]))
        indicators = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float64
        )
        expected = indicators.T @ indicators / 2 - torch.outer(
            indicators.mean(dim=0), indicators.mean(dim=0)
        )
        torch.testing.assert_close(tracker.covariance(), expected)

    def test_checkpoint_continuation_is_exact(self) -> None:
        source = ExpertCooccurrenceCovariance(3)
        source.update(torch.tensor([[0, 1]]))
        restored = ExpertCooccurrenceCovariance(3)
        restored.load_state_dict(source.state_dict())
        source.update(torch.tensor([[1, 2]]))
        restored.update(torch.tensor([[1, 2]]))
        torch.testing.assert_close(source.covariance(), restored.covariance())

    def test_duplicate_expert_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ExpertCooccurrenceCovariance(3).update(torch.tensor([[1, 1]]))


class DiversityAwareGreedyTopKTests(unittest.TestCase):
    def test_identity_covariance_reduces_to_highest_scores(self) -> None:
        selected = diversity_aware_greedy_topk(
            torch.tensor([[0.9, 0.2, 0.7]]), torch.eye(3), top_k=2
        )
        self.assertEqual(selected.tolist(), [[0, 2]])

    def test_correlated_second_expert_is_replaced_by_diverse_candidate(self) -> None:
        covariance = torch.tensor(
            [[1.0, 0.99, 0.0], [0.99, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        selected = diversity_aware_greedy_topk(
            torch.tensor([[1.0, 0.9, 0.8]]), covariance, top_k=2, ridge=1e-3
        )
        self.assertEqual(selected.tolist(), [[0, 2]])

    def test_singular_covariance_is_regularized_and_ties_are_deterministic(self) -> None:
        selected = diversity_aware_greedy_topk(
            torch.ones(2, 3), torch.zeros(3, 3), top_k=2, ridge=1e-3
        )
        self.assertEqual(selected.tolist(), [[0, 1], [0, 1]])

    def test_expert_permutation_is_equivariant_without_ties(self) -> None:
        scores = torch.tensor([[0.9, 0.4, 0.7]])
        covariance = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
        permutation = torch.tensor([2, 0, 1])
        original = diversity_aware_greedy_topk(scores, covariance, top_k=2)
        permuted = diversity_aware_greedy_topk(
            scores[:, permutation], covariance[permutation][:, permutation], top_k=2
        )
        mapped = permutation[permuted]
        self.assertEqual(mapped.tolist(), original.tolist())

    def test_chunked_device_selector_matches_scalar_reference(self) -> None:
        torch.manual_seed(17)
        scores = torch.rand(11, 5)
        basis = torch.randn(5, 5)
        covariance = basis @ basis.T
        expected = diversity_aware_greedy_topk(
            scores, covariance, top_k=3, ridge=1e-3
        )
        actual = diversity_aware_batched_topk(
            scores,
            covariance,
            top_k=3,
            ridge=1e-3,
            token_chunk_size=4,
        )
        self.assertTrue(torch.equal(actual, expected))


class DiversityAwareRoutingControllerTests(unittest.TestCase):
    def _controller(self) -> DiversityAwareRoutingController:
        return DiversityAwareRoutingController(
            num_experts=3, top_k=2, warmup_steps=2, ridge=1e-3
        )

    def test_warmup_preserves_native_routes_then_replays_covariance(self) -> None:
        controller = self._controller()
        native = torch.tensor([[0, 1], [0, 1], [0, 2], [1, 2]])
        scores = torch.tensor([[1.0, 0.9, 0.8]]).repeat(4, 1)
        controller.bind_step(step=0, training=True)
        warmup = controller.select(
            "blocks.0.router", scores, native, training=True
        )
        self.assertIs(warmup, native)
        controller.bind_step(step=2, training=True)
        replay = controller.select(
            "blocks.0.router", scores, native, training=True
        )
        self.assertEqual(replay.tolist(), [[0, 2]] * 4)

    def test_eval_never_observes_or_rewrites_routes(self) -> None:
        controller = self._controller()
        native = torch.tensor([[0, 1]])
        controller.bind_step(step=0, training=False)
        self.assertIs(
            controller.select(
                "blocks.0.router",
                torch.tensor([[1.0, 0.9, 0.8]]),
                native,
                training=False,
            ),
            native,
        )
        self.assertEqual(controller.state_dict()["layers"], {})

    def test_checkpoint_resume_preserves_replay_exactly(self) -> None:
        source = self._controller()
        scores = torch.tensor([[1.0, 0.9, 0.8]])
        native = torch.tensor([[0, 1]])
        source.bind_step(step=0, training=True)
        source.select("blocks.0.router", scores, native, training=True)
        restored = self._controller()
        restored.load_state_dict(source.state_dict())
        for controller in (source, restored):
            controller.bind_step(step=2, training=True)
        torch.testing.assert_close(
            source.select("blocks.0.router", scores, native, training=True),
            restored.select("blocks.0.router", scores, native, training=True),
        )


class DiversityRoutingTrainingPolicyTests(unittest.TestCase):
    def _config(self) -> TrainingConfig:
        config = TrainingConfig()
        config.model.type = "lingbot-video"
        config.model.params.num_experts = 3
        config.model.params.experts_per_token = 2
        config.training.policy_options = {
            "diversity_routing": {
                "enabled": True,
                "warmup_steps": 2,
                "ridge": 1e-3,
            }
        }
        return config

    def test_default_config_has_no_policy_or_state(self) -> None:
        policies = TrainingPolicySet.from_config(TrainingConfig())
        self.assertNotIn("diversity_routing", policies.active_names)

    def test_enabled_policy_is_registered_and_checkpointed(self) -> None:
        policies = TrainingPolicySet.from_config(self._config())
        self.assertIn("diversity_routing", policies.active_names)
        self.assertIn("diversity_routing", policies.checkpoint_metadata()["policies"])

    def test_conflicting_subset_policy_fails_validation(self) -> None:
        config = self._config()
        config.model.params.expert_subset_fraction = 0.5
        errors = validate_training_policy_configs(config)
        self.assertTrue(any("expert-subset" in error for error in errors))

    def test_lingbot_pipeline_binds_warmup_and_replay_runtime(self) -> None:
        from mirai.config.schema import ModelConfig, ModelParams
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

        pipeline = LingBotVideoPipeline(
            ModelConfig(
                type="lingbot-video",
                path="./models/lingbot_video",
                params=ModelParams(
                    variant="tiny-video",
                    latent_channels=2,
                    num_experts=8,
                    experts_per_token=2,
                    shared_experts=1,
                    hidden_size=16,
                    num_layers=2,
                    attention_heads=2,
                    patch_size=1,
                ),
            )
        )
        controller = DiversityAwareRoutingController(
            num_experts=8, top_k=2, warmup_steps=1
        )
        pipeline.configure_diversity_routing(controller)
        pipeline.train()
        inputs = (
            torch.randn(1, 2, 4, 8, 8),
            torch.rand(1),
            {"lingbot": torch.randn(1, 3, 16)},
        )
        controller.bind_step(step=0, training=True)
        warmup = pipeline.forward(*inputs)
        self.assertEqual(len(controller.state_dict()["layers"]), 2)
        controller.bind_step(step=1, training=True)
        replay = pipeline.forward(*inputs)
        self.assertTrue(bool(torch.isfinite(warmup).all()))
        self.assertTrue(bool(torch.isfinite(replay).all()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
