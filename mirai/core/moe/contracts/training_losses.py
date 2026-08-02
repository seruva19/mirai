from __future__ import annotations

# Colocated behavioral contract for MoE training objectives.

import math
import unittest

from mirai.config.schema import ConfigError, TrainingConfig
from mirai.core.moe.monitoring.gradients import moe_gradient_health
from mirai.core.moe.monitoring.gradient_ratio import (
    BalanceGradientProbe,
    RouterProbabilityTarget,
    measure_balance_task_gradient_ratios,
)
from mirai.core.moe.adaptation.losses import (
    TokenChoiceRouterState,
    expert_combination_usage,
    router_similarity_loss,
    token_choice_auxiliary_losses,
)
from mirai.core.moe.monitoring.summary import summarize_routing_by_diffusion_timestep
from mirai.core.moe.monitoring.summary import summarize_routing_stats
from mirai.core.moe.routing.contracts import RoutingStats
from mirai.core.training.runtime.contract import validate_training_runtime_config

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class MoETrainingLossTests(unittest.TestCase):
    @staticmethod
    def _gradient_probe(
        probabilities,
        *,
        objective,
    ) -> BalanceGradientProbe:
        return BalanceGradientProbe(
            targets=(
                RouterProbabilityTarget(
                    layer="blocks.0.ffn.router",
                    probabilities=probabilities,
                ),
            ),
            objectives={"load_balance": objective},
        )

    def test_balance_task_gradient_ratio_matches_equation_9(self) -> None:
        probabilities = torch.tensor(
            [[0.8, 0.2], [0.3, 0.7]],
            dtype=torch.float64,
            requires_grad=True,
        )
        task_coefficients = torch.tensor(
            [[1.0, 0.0], [0.0, 2.0]],
            dtype=torch.float64,
        )
        frequency = torch.tensor([3.0, 4.0], dtype=torch.float64)
        alpha = 0.2
        task_loss = (probabilities * task_coefficients).sum()
        balance_loss = alpha * (probabilities.mean(dim=0) * frequency).sum()

        metrics = measure_balance_task_gradient_ratios(
            task_loss=task_loss,
            probe=self._gradient_probe(
                probabilities,
                objective=balance_loss,
            ),
            alarm_threshold=0.1,
        )

        expected = float(
            torch.linalg.vector_norm(alpha * frequency)
            / torch.linalg.vector_norm(task_coefficients.sum(dim=0))
        )
        self.assertAlmostEqual(metrics["moe_balance_grad_ratio"], expected)
        self.assertEqual(metrics["moe_balance_grad_ratio_alarm"], 1.0)
        self.assertIsNone(probabilities.grad)

    def test_ratio_is_linear_in_weight_and_monitor_does_not_change_backward(self) -> None:
        def run(alpha: float, *, monitor: bool):
            probabilities = torch.tensor(
                [[0.8, 0.2], [0.3, 0.7]],
                requires_grad=True,
            )
            task = (
                probabilities
                * torch.tensor([[1.0, -0.5], [0.25, 2.0]])
            ).sum()
            balance = (
                float(alpha)
                * (probabilities.mean(dim=0) * torch.tensor([3.0, 4.0])).sum()
            )
            ratio = None
            if monitor:
                ratio = measure_balance_task_gradient_ratios(
                    task_loss=task,
                    probe=self._gradient_probe(
                        probabilities,
                        objective=balance,
                    ),
                )["moe_balance_grad_ratio"]
            (task + balance).backward()
            return ratio, probabilities.grad.detach().clone()

        ratio_one, monitored_grad = run(0.1, monitor=True)
        ratio_two, _ = run(0.2, monitor=True)
        _, reference_grad = run(0.1, monitor=False)
        self.assertAlmostEqual(float(ratio_two), 2.0 * float(ratio_one))
        torch.testing.assert_close(monitored_grad, reference_grad, rtol=0, atol=0)

    def test_zero_weight_is_zero_and_missing_task_path_is_absent(self) -> None:
        probabilities = torch.tensor(
            [[0.6, 0.4], [0.4, 0.6]],
            requires_grad=True,
        )
        task = probabilities[:, 0].sum()
        zero_balance = probabilities.sum() * 0.0
        metrics = measure_balance_task_gradient_ratios(
            task_loss=task,
            probe=self._gradient_probe(
                probabilities,
                objective=zero_balance,
            ),
        )
        self.assertEqual(metrics["moe_balance_grad_ratio"], 0.0)

        unrelated = torch.tensor(1.0, requires_grad=True)
        self.assertEqual(
            measure_balance_task_gradient_ratios(
                task_loss=unrelated,
                probe=self._gradient_probe(
                    probabilities,
                    objective=zero_balance,
                ),
            ),
            {},
        )

    def test_gradient_ratio_config_is_default_off_and_validated(self) -> None:
        default = TrainingConfig.from_dict({})
        self.assertFalse(default.model.params.moe_balance_grad_ratio_telemetry)
        self.assertEqual(
            default.model.params.moe_balance_grad_ratio_threshold,
            0.1,
        )
        with self.assertRaisesRegex(ConfigError, "must be finite and > 0"):
            TrainingConfig.from_dict(
                {
                    "model": {
                        "params": {
                            "moe_balance_grad_ratio_threshold": 0.0,
                        }
                    }
                }
            )
        aggressive = TrainingConfig.from_dict(
            {
                "model": {
                    "params": {
                        "moe_balance_grad_ratio_telemetry": True,
                    }
                },
                "training": {"gradient_checkpointing": "aggressive"},
            }
        )
        with self.assertRaisesRegex(ValueError, "aggressive"):
            validate_training_runtime_config(aggressive)

    def test_lingbot_provider_exposes_one_shot_probability_probe(self) -> None:
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
        from mirai.core.models.providers import get_model_family_provider

        base_params = {
            "variant": "tiny-video",
            "hidden_size": 12,
            "attention_heads": 2,
            "num_layers": 1,
            "num_experts": 2,
            "experts_per_token": 2,
            "shared_experts": 0,
            "latent_channels": 1,
            "patch_size": 1,
        }
        latents = torch.randn(1, 1, 2, 2, 2)
        timesteps = torch.tensor([0.4])
        text = {"t5": torch.randn(1, 1, 12)}

        torch.manual_seed(7)
        disabled = LingBotVideoPipeline.from_training_config(
            TrainingConfig.from_dict(
                {"model": {"params": dict(base_params)}}
            )
        )
        disabled.set_adapter_config(TrainingConfig.from_dict({}).adapter)
        disabled_output = disabled(latents, timesteps, text)
        self.assertIsNone(disabled.take_balance_gradient_probe())

        torch.manual_seed(7)
        enabled_config = TrainingConfig.from_dict(
            {
                "model": {
                    "params": {
                        **base_params,
                        "moe_balance_grad_ratio_telemetry": True,
                    }
                }
            }
        )
        enabled = LingBotVideoPipeline.from_training_config(enabled_config)
        enabled.set_adapter_config(enabled_config.adapter)
        enabled_output = enabled(latents, timesteps, text)
        torch.testing.assert_close(
            enabled_output,
            disabled_output,
            rtol=0,
            atol=0,
        )
        probe = enabled.take_balance_gradient_probe()
        self.assertIsNotNone(probe)
        metrics = measure_balance_task_gradient_ratios(
            task_loss=enabled_output.square().mean(),
            probe=probe,
        )
        self.assertIn("moe_balance_grad_ratio", metrics)
        self.assertIsNone(enabled.take_balance_gradient_probe())

        provider = get_model_family_provider(enabled_config.model.type)
        self.assertIsNotNone(provider)
        self.assertTrue(
            provider.supports_balance_gradient_ratio_telemetry(enabled_config)
        )

    def test_expert_choice_probe_reports_zero_balance_ratio(self) -> None:
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

        config = TrainingConfig.from_dict(
            {
                "model": {
                    "params": {
                        "variant": "tiny-video",
                        "hidden_size": 12,
                        "attention_heads": 2,
                        "num_layers": 1,
                        "num_experts": 2,
                        "experts_per_token": 2,
                        "shared_experts": 0,
                        "latent_channels": 1,
                        "patch_size": 1,
                        "moe_routing_mode": "expert_choice",
                        "moe_balance_grad_ratio_telemetry": True,
                    }
                }
            }
        )
        pipeline = LingBotVideoPipeline.from_training_config(config)
        pipeline.set_adapter_config(config.adapter)
        output = pipeline(
            torch.randn(1, 1, 2, 2, 2),
            torch.tensor([0.4]),
            {"t5": torch.randn(1, 1, 12)},
        )
        probe = pipeline.take_balance_gradient_probe()
        self.assertIsNotNone(probe)
        metrics = measure_balance_task_gradient_ratios(
            task_loss=output.square().mean(),
            probe=probe,
        )
        self.assertEqual(metrics["moe_balance_grad_ratio"], 0.0)

    def test_training_engine_emits_ratio_and_preserves_main_backward(self) -> None:
        from mirai.config.schema import StrategyConfig
        from mirai.core.training.objectives.engine import compute_training_loss
        from mirai.core.training.objectives.flow_matching import (
            FlowMatchingObjective,
        )
        from mirai.core.training.objectives.sampling import (
            NoiseGenerator,
            UniformTimestepSampler,
        )
        from mirai.core.training.strategies.text_to_video import (
            TextToVideoStrategy,
        )

        class Pipeline:
            probe = None
            auxiliary = {}

            @staticmethod
            def prepare_model_timesteps(timesteps, *, latents):
                _ = latents
                return timesteps

            @staticmethod
            def apply_noise(clean_latents, noise, timesteps):
                view = timesteps.reshape(
                    (-1,) + (1,) * (clean_latents.ndim - 1)
                )
                return clean_latents * (1.0 - view) + noise * view

            @staticmethod
            def compute_target(noise, clean_latents, timesteps):
                _ = timesteps
                return noise - clean_latents

            @classmethod
            def get_training_auxiliary_losses(cls):
                result = dict(cls.auxiliary)
                cls.auxiliary = {}
                return result

            @staticmethod
            def get_training_diagnostics():
                return {}

            @classmethod
            def take_balance_gradient_probe(cls):
                result = cls.probe
                cls.probe = None
                return result

        config = TrainingConfig()
        config.training.batch_size = 2
        config.model.params.moe_balance_grad_ratio_telemetry = True
        pipeline = Pipeline()
        logits_holder = []

        def predict(inputs):
            logits = torch.tensor(
                [[1.0, -0.5], [-0.25, 0.75]],
                requires_grad=True,
            )
            probabilities = torch.softmax(logits, dim=-1)
            prediction = (
                inputs.noisy_latents
                * probabilities[:, :1].to(inputs.noisy_latents.dtype)
            )
            balance = (
                0.01
                * 2.0
                * (
                    probabilities.mean(dim=0)
                    * probabilities.new_tensor([0.75, 0.25])
                ).sum()
            )
            Pipeline.auxiliary = {"moe_load_balance": balance}
            Pipeline.probe = self._gradient_probe(
                probabilities,
                objective=balance,
            )
            logits_holder.append(logits)
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
            pipeline=pipeline,
            strategy=TextToVideoStrategy(StrategyConfig()),
            objective=FlowMatchingObjective(),
            loss_fn=lambda pred, expected: (pred - expected).square(),
            timestep_sampler=UniformTimestepSampler(seed=11),
            noise_generator=NoiseGenerator(seed=13),
            predict=predict,
            strategy_prepare_accepts_training=True,
            strategy_prepare_accepts_objective=True,
        )
        self.assertIn("moe_balance_grad_ratio", result.diagnostics)
        self.assertGreaterEqual(
            result.diagnostics["moe_balance_grad_ratio"],
            0.0,
        )
        result.loss.backward()
        self.assertIsNotNone(logits_holder[0].grad)
        self.assertTrue(torch.isfinite(logits_holder[0].grad).all())

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "native gradient-ratio contract requires CUDA",
    )
    def test_native_cuda_probe_and_adapter_backward_are_finite(self) -> None:
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

        config = TrainingConfig.from_dict(
            {
                "model": {
                    "params": {
                        "variant": "tiny-video",
                        "hidden_size": 16,
                        "attention_heads": 2,
                        "num_layers": 1,
                        "num_experts": 2,
                        "experts_per_token": 2,
                        "shared_experts": 0,
                        "latent_channels": 1,
                        "patch_size": 1,
                        "moe_balance_grad_ratio_telemetry": True,
                    }
                }
            }
        )
        pipeline = LingBotVideoPipeline.from_training_config(config)
        pipeline.set_adapter_config(config.adapter)
        pipeline.transformer.to(device=torch.device("cuda"))
        output = pipeline(
            torch.randn(1, 1, 2, 2, 2, device="cuda"),
            torch.tensor([0.4], device="cuda"),
            {"t5": torch.randn(1, 1, 16, device="cuda")},
        )
        auxiliary = pipeline.get_training_auxiliary_losses()
        probe = pipeline.take_balance_gradient_probe()
        self.assertIsNotNone(probe)
        metrics = measure_balance_task_gradient_ratios(
            task_loss=output.square().mean(),
            probe=probe,
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(
            math.isfinite(metrics["moe_balance_grad_ratio"])
        )
        (output.square().mean() + sum(auxiliary.values())).backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in pipeline.get_trainable_parameters()
            )
        )

    def test_router_similarity_matches_equations_9_to_11(self) -> None:
        logits = torch.tensor(
            [[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [2.0, 0.0, 1.0]],
            requires_grad=True,
        )
        indices = torch.tensor([[0, 1], [1, 2], [0, 2]])
        actual = router_similarity_loss(indices, logits, num_experts=3)

        probabilities = torch.softmax(logits, dim=-1)
        indicator = torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
        )
        selection = indicator.T @ indicator
        probability = probabilities.T @ probabilities
        diagonal = torch.diag(torch.diag(selection))
        off_diagonal = selection - diagonal
        weights = diagonal / diagonal.sum() * 3
        weights = weights + off_diagonal / off_diagonal.sum() * 6
        expected = (weights * probability).sum() / 3

        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_combination_usage_detects_pair_collapse(self) -> None:
        report = expert_combination_usage(
            torch.tensor([[0, 1], [0, 1], [0, 1], [2, 3]]),
            num_experts=4,
            minimum_fraction=0.5,
        )
        self.assertEqual(report["active_pairs"], 1)
        self.assertEqual(report["possible_pairs"], 6)
        self.assertAlmostEqual(report["usage_ratio"], 1 / 6)

    def test_router_losses_are_differentiable(self) -> None:
        logits = torch.randn(12, 4, requires_grad=True)
        indices = torch.topk(logits, k=2, dim=-1).indices
        losses = token_choice_auxiliary_losses(
            [
                TokenChoiceRouterState(
                    top_indices=indices,
                    logits=logits,
                    probabilities=torch.sigmoid(logits),
                    num_experts=4,
                )
            ],
            load_balance_weight=0.01,
            z_loss_weight=0.001,
        )
        total = sum(losses.values())
        total.backward()
        self.assertIn("moe_load_balance", losses)
        self.assertIn("moe_router_z", losses)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_expert_gradient_health_counts_active_experts(self) -> None:
        param = torch.nn.Parameter(torch.zeros(4, 2, 3))
        param.grad = torch.zeros_like(param)
        param.grad[1] = 1.0
        param.grad[3] = 2.0
        metrics = moe_gradient_health([("blocks.0.experts.w1.lora_b", param)])
        self.assertEqual(metrics["moe_active_expert_fraction"], 0.5)
        self.assertEqual(metrics["moe_inactive_expert_count"], 2.0)
        self.assertEqual(metrics["moe_expert_adapter_active_slice_fraction"], 0.5)
        self.assertEqual(metrics["moe_expert_adapter_inactive_slices"], 2.0)
        self.assertEqual(metrics["moe_expert_adapter_total_slices"], 4.0)

    def test_routing_load_metrics_are_per_physical_expert(self) -> None:
        stats = RoutingStats(
            layer="blocks.0.ffn.router",
            tokens=10,
            selected_tokens=20,
            expert_fraction=(0.5, 0.25, 0.25, 0.0),
            mean_router_probability=(0.4, 0.3, 0.2, 0.1),
            dead_experts=1,
            routing_entropy=0.7,
        )

        detail = stats.as_dict()
        summary = summarize_routing_stats([stats], include_expert_vectors=False)

        self.assertEqual(detail["tokens_per_expert"], [10, 5, 5, 0])
        self.assertEqual(detail["probability_mass_per_expert"], [4.0, 3.0, 2.0, 1.0])
        self.assertEqual(detail["active_expert_fraction"], 0.75)
        self.assertAlmostEqual(detail["load_cv"], 0.70710678)
        self.assertEqual(detail["max_to_mean_load"], 2.0)
        self.assertEqual(summary["mean_active_expert_fraction"], 0.75)
        self.assertAlmostEqual(summary["mean_load_cv"], 0.70710678)
        self.assertEqual(summary["max_to_mean_load"], 2.0)

    def test_routing_is_summarized_by_diffusion_timestep(self) -> None:
        indices = torch.tensor(
            [[[0], [0]], [[1], [1]], [[2], [2]], [[3], [3]]]
        ).reshape(8, 1)
        summary = summarize_routing_by_diffusion_timestep(
            [(indices, 4, 2, 4)],
            torch.tensor([0.1, 0.4, 0.6, 0.9]),
        )
        self.assertEqual(summary["0.00-0.25"]["dead_experts"], 3)
        self.assertEqual(summary["0.75-1.00"]["max_expert_fraction"], 1.0)
        self.assertEqual(summary["0.00-0.25"]["active_expert_fraction"], 0.25)
        self.assertEqual(summary["0.00-0.25"]["max_to_mean_load"], 4.0)


if __name__ == "__main__":
    unittest.main()
