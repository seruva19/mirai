"""Behavioral contracts for frozen-reference router distillation."""

from __future__ import annotations

# Colocated behavioral contract for router distillation.

import unittest

import torch
import torch.nn.functional as F

from mirai.config.schema import AdapterConfig
from mirai.config.schema import ModelConfig
from mirai.config.schema import ModelParams
from mirai.config.schema import TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.adaptation.distillation import RouterDistillationController
from mirai.core.moe.adaptation.distillation import router_distillation_kl
from mirai.core.moe.adaptation.distillation_schedule import (
    router_distillation_weight_scale,
)
from mirai.core.training.training_policy import TrainingPolicySet
from mirai.core.training.training_policy import validate_training_policy_configs


class RouterDistillationFormulaTests(unittest.TestCase):
    def test_temperature_scaled_forward_kl_matches_reference(self) -> None:
        student = torch.tensor([[1.0, -0.5, 0.2]], requires_grad=True)
        teacher = torch.tensor([[0.1, 0.8, -0.3]], requires_grad=True)
        temperature = 2.0
        actual = router_distillation_kl(
            student, teacher, temperature=temperature
        )
        teacher_prob = F.softmax(teacher.detach() / temperature, dim=-1)
        expected = F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            teacher_prob,
            reduction="batchmean",
        ) * temperature**2
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad)

    def test_identical_logits_have_zero_loss_and_gradient(self) -> None:
        logits = torch.tensor([[0.2, -0.1, 0.7]], requires_grad=True)
        loss = router_distillation_kl(logits, logits.detach(), temperature=1.5)
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_teacher_lineage_rejects_changed_weights(self) -> None:
        source = RouterDistillationController(weight=0.1)
        source.bind_teacher_weights({"layer": torch.eye(2)})
        source.bind_step(step=3, training=True)
        state = source.state_dict()
        target = RouterDistillationController(weight=0.1)
        target.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "lineage"):
            target.bind_teacher_weights({"layer": torch.ones(2, 2)})

    def test_schedule_returns_no_term_outside_training_window(self) -> None:
        controller = RouterDistillationController(
            weight=0.2, start_step=2, end_step=4
        )
        controller.bind_teacher_weights({"layer": torch.eye(2)})
        student = torch.randn(2, 2)
        teacher = torch.randn(2, 2)
        for step, training in ((1, True), (2, False), (4, True)):
            controller.bind_step(step=step, training=training)
            self.assertIsNone(controller.loss("layer", student, teacher))

    def test_linear_decay_weight_matches_des_moe_schedule(self) -> None:
        self.assertEqual(
            router_distillation_weight_scale(
                2, start_step=2, end_step=6, schedule="linear_decay"
            ),
            1.0,
        )
        self.assertEqual(
            router_distillation_weight_scale(
                4, start_step=2, end_step=6, schedule="linear_decay"
            ),
            0.5,
        )
        controller = RouterDistillationController(
            weight=0.2,
            start_step=2,
            end_step=6,
            weight_schedule="linear_decay",
        )
        controller.bind_teacher_weights({"layer": torch.eye(2)})
        student = torch.tensor([[1.0, -1.0]], requires_grad=True)
        teacher = torch.tensor([[-1.0, 1.0]])
        controller.bind_step(step=2, training=True)
        initial = controller.loss("layer", student, teacher)
        controller.bind_step(step=4, training=True)
        middle = controller.loss("layer", student, teacher)
        torch.testing.assert_close(middle, initial * 0.5)

    def test_v1_checkpoint_migrates_only_to_constant_schedule(self) -> None:
        source = RouterDistillationController(weight=0.1, end_step=4)
        source.bind_teacher_weights({"layer": torch.eye(2)})
        state = source.state_dict()
        state["schema_version"] = 1
        state.pop("weight_schedule")
        target = RouterDistillationController(weight=0.1, end_step=4)
        target.load_state_dict(state)
        decayed = RouterDistillationController(
            weight=0.1, end_step=4, weight_schedule="linear_decay"
        )
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            decayed.load_state_dict(state)


class RouterDistillationPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin_components()

    def _config(self) -> TrainingConfig:
        config = TrainingConfig()
        config.model.type = "lingbot-video"
        config.adapter.train_router = True
        config.training.policy_options = {
            "router_distillation": {
                "enabled": True,
                "weight": 0.1,
                "temperature": 2.0,
            }
        }
        return config

    def test_default_absent_and_enabled_policy_registered(self) -> None:
        self.assertNotIn(
            "router_distillation",
            TrainingPolicySet.from_config(TrainingConfig()).active_names,
        )
        self.assertIn(
            "router_distillation",
            TrainingPolicySet.from_config(self._config()).active_names,
        )

    def test_frozen_router_and_invalid_values_fail_validation(self) -> None:
        config = self._config()
        config.adapter.train_router = False
        config.training.policy_options["router_distillation"]["weight"] = 0.0
        errors = validate_training_policy_configs(config)
        self.assertTrue(any("train_router=false" in error for error in errors))
        self.assertTrue(any("must be positive" in error for error in errors))
        config.adapter.train_router = True
        config.training.policy_options["router_distillation"].update(
            {"weight": 0.1, "weight_schedule": "linear_decay", "end_step": 0}
        )
        errors = validate_training_policy_configs(config)
        self.assertTrue(any("linear_decay" in error for error in errors))

    def test_distillation_window_must_fit_staged_router_window(self) -> None:
        config = self._config()
        config.training.policy_options["router_stage_schedule"] = {
            "enabled": True,
            "train_start_step": 2,
            "freeze_step": 5,
        }
        errors = validate_training_policy_configs(config)
        self.assertTrue(any("must stay inside" in error for error in errors))
        config.training.policy_options["router_distillation"].update(
            {"start_step": 2, "end_step": 5}
        )
        errors = validate_training_policy_configs(config)
        self.assertFalse(any("must stay inside" in error for error in errors))

    def _pipeline(self) -> LingBotVideoPipeline:
        return LingBotVideoPipeline(
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

    def test_lingbot_uses_frozen_original_weight_on_same_tokens(self) -> None:
        pipeline = self._pipeline()
        controller = RouterDistillationController(weight=0.5, temperature=2.0)
        pipeline.configure_router_distillation(controller)
        pipeline.set_adapter_config(
            AdapterConfig(
                target_preset="attn_router_routed_experts",
                rank=2,
                alpha=2.0,
                train_router=True,
            )
        )
        controller.bind_step(step=0, training=True)
        router = pipeline._moe_router_modules[0]
        adapter = pipeline._router_adapter_bindings()[0].adapter
        with torch.no_grad():
            generator = torch.Generator().manual_seed(19)
            for parameter in adapter.parameters():
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                    )
                    * 0.25
                )
        tokens = torch.randn(6, 16)
        router.train()
        router(tokens)
        term = router.training_router_distillation
        self.assertIsNotNone(term)
        self.assertGreater(float(term.detach()), 0.0)
        term.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in adapter.parameters()
            )
        )
        original = router.parametrizations.weight.original
        self.assertIsNone(original.grad)

    def test_router_free_preset_fails_teacher_binding(self) -> None:
        pipeline = self._pipeline()
        pipeline.configure_router_distillation(RouterDistillationController(weight=0.1))
        with self.assertRaisesRegex(ValueError, "at least one weight"):
            pipeline.set_adapter_config(
                AdapterConfig(target_preset="attn_only", rank=2, alpha=2.0)
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
