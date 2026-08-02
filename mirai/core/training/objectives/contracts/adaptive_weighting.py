"""Behavioral contract for EDM2 adaptive noise-level loss weighting."""

from __future__ import annotations

import copy
import math
import unittest

from mirai.config.schema import TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.models.testbed import SparseMoETestPipeline as _SparseMoETestPipeline
from mirai.core.training.objectives.adaptive_weighting import (
    NoiseLevelUncertaintyWeighting,
)
from mirai.core.training.objectives.flow_loss import (
    compute_flow_matching_loss,
)
from mirai.core.training.objectives.flow_matching import FlowMatchingObjective
from mirai.core.training.lifecycle.session_components import (
    build_training_runtime_components,
)
from mirai.core.training.runtime.contract import (
    validate_training_runtime_config,
)
from mirai.core.training.trainer import Trainer

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _tiny_testbed_config() -> TrainingConfig:
    return TrainingConfig.from_dict(
        {
            "model": {
                "type": "sparse_moe_test",
                "path": "",
                "params": {
                    "variant": "tiny-video",
                    "hidden_size": 12,
                    "attention_heads": 2,
                    "num_layers": 1,
                    "num_experts": 4,
                    "experts_per_token": 2,
                    "shared_experts": 0,
                },
            },
            "adapter": {
                "type": "lora",
                "target_preset": "attn_only",
                "rank": 2,
                "alpha": 2,
            },
        }
    )


@unittest.skipIf(torch is None, "torch not installed")
class AdaptiveWeightingFormulaContract(unittest.TestCase):
    _ = _SparseMoETestPipeline

    def test_flow_noise_coordinate_and_fourier_head_match_edm2_adaptation(
        self,
    ) -> None:
        torch.manual_seed(17)
        head = NoiseLevelUncertaintyWeighting(channels=8)
        timesteps = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)

        observed = head(timesteps, timestep_eps=1.0e-5)

        coordinate = torch.logit(timesteps.float()) / 4.0
        features = (
            coordinate[:, None] * head.frequencies[None, :]
            + head.phases[None, :]
        ).cos() * math.sqrt(2.0)
        weight = head.output_weight.float()
        weight = weight / (
            torch.linalg.vector_norm(weight)
            + 1.0e-4 / math.sqrt(float(weight.numel()))
        )
        weight = weight / math.sqrt(float(weight.shape[1]))
        expected = (features @ weight.t()).reshape(-1)
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    def test_exact_uncertainty_loss_and_gradients(self) -> None:
        torch.manual_seed(23)
        head = NoiseLevelUncertaintyWeighting(channels=16)
        per_sample = torch.tensor(
            [0.25, 1.0, 4.0],
            dtype=torch.float32,
            requires_grad=True,
        )
        timesteps = torch.tensor([0.1, 0.5, 0.9])

        weighted, inverse_variance, log_variance = head.weighted_loss(
            per_sample,
            timesteps=timesteps,
            timestep_eps=1.0e-5,
        )
        expected = per_sample * torch.exp(-log_variance) + log_variance
        torch.testing.assert_close(weighted, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            inverse_variance,
            torch.exp(-log_variance),
            rtol=0,
            atol=0,
        )

        weighted.mean().backward()
        self.assertIsNotNone(per_sample.grad)
        self.assertIsNotNone(head.output_weight.grad)
        self.assertTrue(bool(torch.isfinite(head.output_weight.grad).all()))

    def test_disabled_static_path_is_unchanged(self) -> None:
        per_sample = torch.tensor([0.25, 1.0, 4.0])
        timesteps = torch.tensor([0.1, 0.5, 0.9])
        observed = compute_flow_matching_loss(
            per_sample_loss=per_sample,
            timesteps=timesteps,
            gradient_accumulation=2,
            loss_weighting="uniform",
            min_snr_gamma=5.0,
            timestep_eps=1.0e-5,
        )
        torch.testing.assert_close(
            observed.weighted_loss,
            per_sample,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            observed.loss,
            per_sample.mean() / 2.0,
            rtol=0,
            atol=0,
        )
        self.assertEqual(observed.diagnostics, {})

    def test_objective_state_roundtrip_is_exact(self) -> None:
        config = TrainingConfig()
        config.training.loss_weighting = "adaptive_uncertainty"
        torch.manual_seed(29)
        source = FlowMatchingObjective()
        source.configure(config)
        source_state = copy.deepcopy(source.state_dict())

        torch.manual_seed(31)
        restored = FlowMatchingObjective()
        restored.configure(config)
        restored.load_state_dict(source_state)
        restored_state = restored.state_dict()
        self.assertEqual(source_state.keys(), restored_state.keys())
        for name, tensor in source_state["adaptive_weighting"].items():
            torch.testing.assert_close(
                tensor,
                restored_state["adaptive_weighting"][name],
                rtol=0,
                atol=0,
            )

        with self.assertRaisesRegex(ValueError, "missing"):
            restored.load_state_dict({})

    def test_trainer_owns_optimizer_parameter_and_checkpoint_state(self) -> None:
        config = _tiny_testbed_config()
        config.training.loss_weighting = "adaptive_uncertainty"
        config.optimizer.weight_decay = 0.1
        register_builtin_components()
        trainer = Trainer(config)

        named = dict(trainer.get_named_trainable_parameters())
        uncertainty_name = "objective.adaptive_weighting.output_weight"
        self.assertIn(uncertainty_name, named)
        uncertainty_parameter = named[uncertainty_name]

        components = build_training_runtime_components(
            trainer=trainer,
            config=config,
        )
        objective_groups = [
            group
            for group in components.optimizer_result.optimizer.param_groups
            if any(
                parameter is uncertainty_parameter
                for parameter in group["params"]
            )
        ]
        self.assertEqual(len(objective_groups), 1)
        self.assertEqual(float(objective_groups[0]["weight_decay"]), 0.0)

        checkpoint = copy.deepcopy(trainer.state_dict())
        expected = uncertainty_parameter.detach().clone()
        with torch.no_grad():
            uncertainty_parameter.add_(1.0)
        trainer.load_state_dict(checkpoint)
        torch.testing.assert_close(
            uncertainty_parameter,
            expected,
            rtol=0,
            atol=0,
        )

    def test_objective_and_optimizer_resume_produce_exact_next_update(
        self,
    ) -> None:
        config = _tiny_testbed_config()
        config.training.loss_weighting = "adaptive_uncertainty"
        register_builtin_components()
        source = Trainer(config)
        source_components = build_training_runtime_components(
            trainer=source,
            config=config,
        )
        source_optimizer = source_components.optimizer_result.optimizer
        per_sample = torch.tensor([0.5, 2.0])
        timesteps = torch.tensor([0.25, 0.75])

        def update(trainer, optimizer) -> None:
            optimizer.zero_grad(set_to_none=True)
            result = trainer.objective.reduce(
                per_sample_loss=per_sample,
                timesteps=timesteps,
                gradient_accumulation=1,
                config=config,
                bucket_ids=None,
            )
            result.loss.backward()
            optimizer.step()

        update(source, source_optimizer)
        trainer_state = copy.deepcopy(source.state_dict())
        optimizer_state = copy.deepcopy(source_optimizer.state_dict())
        update(source, source_optimizer)
        expected = dict(source.get_named_trainable_parameters())[
            "objective.adaptive_weighting.output_weight"
        ].detach().clone()

        restored = Trainer(config)
        restored_components = build_training_runtime_components(
            trainer=restored,
            config=config,
        )
        restored_optimizer = restored_components.optimizer_result.optimizer
        restored.load_state_dict(trainer_state)
        restored_optimizer.load_state_dict(optimizer_state)
        update(restored, restored_optimizer)
        observed = dict(restored.get_named_trainable_parameters())[
            "objective.adaptive_weighting.output_weight"
        ]
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


class AdaptiveWeightingConfigContract(unittest.TestCase):
    def test_supported_configuration_validates(self) -> None:
        config = TrainingConfig()
        config.training.loss_weighting = "adaptive_uncertainty"
        validate_training_runtime_config(config)

    def test_incompatible_configuration_fails_before_training(self) -> None:
        cases = (
            ("training", "objective", "regression", "flow_matching"),
            ("training", "loss_function", "huber", "loss_function"),
            (
                "training",
                "loss_bucket_normalization",
                "per_bucket_mean",
                "loss_bucket_normalization",
            ),
            (
                "optimizer",
                "type",
                "selected_expert_adamw",
                "general-purpose optimizer",
            ),
            (
                "optimizer",
                "type",
                "lora_muon",
                "general-purpose optimizer",
            ),
        )
        for section, field, value, expected in cases:
            with self.subTest(section=section, field=field):
                config = TrainingConfig()
                config.training.loss_weighting = "adaptive_uncertainty"
                setattr(getattr(config, section), field, value)
                with self.assertRaisesRegex(ValueError, expected):
                    validate_training_runtime_config(config)
