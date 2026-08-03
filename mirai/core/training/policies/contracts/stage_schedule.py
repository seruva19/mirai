"""Behavioral contracts for staged sparse-MoE router adaptation."""

from __future__ import annotations

# Colocated behavioral contract for staged router adaptation.

import unittest

import torch
from torch import nn

from mirai.config.schema import AdapterConfig
from mirai.config.schema import ModelConfig
from mirai.config.schema import ModelParams
from mirai.config.schema import TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.adaptation.stage_schedule import RouterStageScheduleController
from mirai.core.training.training_policy import TrainingPolicySet
from mirai.core.training.training_policy import validate_training_policy_configs


class RouterStageScheduleControllerTests(unittest.TestCase):
    def test_schedule_changes_only_bound_router_parameters(self) -> None:
        router = nn.Linear(3, 2, bias=False)
        expert = nn.Linear(3, 2, bias=False)
        controller = RouterStageScheduleController(
            train_start_step=2, freeze_step=4
        )
        controller.bind_adapters([router])

        self.assertEqual(
            controller.apply_step(step=1, training=True).stage, "pre_router"
        )
        self.assertFalse(router.weight.requires_grad)
        self.assertTrue(expert.weight.requires_grad)

        self.assertEqual(
            controller.apply_step(step=2, training=True).stage,
            "router_adaptation",
        )
        self.assertTrue(router.weight.requires_grad)
        self.assertTrue(expert.weight.requires_grad)

        self.assertEqual(
            controller.apply_step(step=4, training=True).stage, "post_router"
        )
        self.assertFalse(router.weight.requires_grad)
        self.assertTrue(expert.weight.requires_grad)

    def test_optimizer_owns_parameter_across_frozen_stages(self) -> None:
        router = nn.Linear(2, 1, bias=False)
        expert = nn.Linear(2, 1, bias=False)
        controller = RouterStageScheduleController(
            train_start_step=1, freeze_step=2
        )
        controller.bind_adapters([router])
        optimizer = torch.optim.AdamW(
            [*router.parameters(), *expert.parameters()],
            lr=0.1,
            weight_decay=0.2,
        )
        inputs = torch.ones(1, 2)

        controller.apply_step(step=0, training=True)
        before = router.weight.detach().clone()
        expert(inputs).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(router.weight, before, rtol=0, atol=0)
        self.assertNotIn(router.weight, optimizer.state)

        controller.apply_step(step=1, training=True)
        (router(inputs) + expert(inputs)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(router.weight, before))
        self.assertIn(router.weight, optimizer.state)

        controller.apply_step(step=2, training=True)
        frozen = router.weight.detach().clone()
        expert(inputs).sum().backward()
        optimizer.step()
        torch.testing.assert_close(router.weight, frozen, rtol=0, atol=0)

    def test_eval_does_not_mutate_trainability(self) -> None:
        router = nn.Linear(2, 1, bias=False)
        controller = RouterStageScheduleController(train_start_step=2)
        controller.bind_adapters([router])
        controller.apply_step(step=0, training=False)
        self.assertTrue(router.weight.requires_grad)

    def test_unbound_and_rebound_contracts_fail_explicitly(self) -> None:
        controller = RouterStageScheduleController()
        with self.assertRaisesRegex(RuntimeError, "not been bound"):
            controller.apply_step(step=0, training=True)
        controller.bind_adapters([nn.Linear(2, 1, bias=False)])
        with self.assertRaisesRegex(ValueError, "different adapters"):
            controller.bind_adapters([nn.Linear(2, 1, bias=False)])

    def test_checkpoint_configuration_mismatch_fails(self) -> None:
        source = RouterStageScheduleController(
            train_start_step=2, freeze_step=5
        )
        source.load_state_dict(
            {
                "schema_version": 1,
                "train_start_step": 2,
                "freeze_step": 5,
                "step": 3,
            }
        )
        target = RouterStageScheduleController(
            train_start_step=1, freeze_step=5
        )
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            target.load_state_dict(source.state_dict())


class RouterStageSchedulePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin_components()

    def _config(self) -> TrainingConfig:
        config = TrainingConfig()
        config.model.type = "lingbot-video"
        config.adapter.train_router = True
        config.training.policy_options = {
            "router_stage_schedule": {
                "enabled": True,
                "train_start_step": 2,
                "freeze_step": 5,
            }
        }
        return config

    def test_default_is_absent_and_enabled_policy_is_checkpointed(self) -> None:
        self.assertNotIn(
            "router_stage_schedule",
            TrainingPolicySet.from_config(TrainingConfig()).active_names,
        )
        policies = TrainingPolicySet.from_config(self._config())
        self.assertIn("router_stage_schedule", policies.active_names)
        metadata = policies.checkpoint_metadata()["policies"]
        self.assertEqual(metadata["router_stage_schedule"]["freeze_step"], 5)

    def test_conflicting_router_freeze_and_invalid_window_are_rejected(self) -> None:
        config = self._config()
        config.adapter.train_router = False
        config.training.policy_options["router_stage_schedule"][
            "freeze_step"
        ] = 1
        errors = validate_training_policy_configs(config)
        self.assertTrue(any("train_router=false" in error for error in errors))
        self.assertTrue(any("must exceed" in error for error in errors))

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

    def test_lingbot_binds_router_lora_and_leaves_expert_lora_trainable(self) -> None:
        pipeline = self._pipeline()
        controller = RouterStageScheduleController(
            train_start_step=1, freeze_step=2
        )
        pipeline.configure_router_stage_schedule(controller)
        pipeline.set_adapter_config(
            AdapterConfig(
                target_preset="attn_router_routed_experts",
                rank=2,
                alpha=2.0,
                train_router=True,
            )
        )
        router_parameters = tuple(
            parameter
            for binding in pipeline._router_adapter_bindings()
            for parameter in binding.adapter.parameters()
        )
        expert_parameters = tuple(
            parameter
            for name, parameter in pipeline.transformer.named_parameters()
            if "experts" in name and parameter.requires_grad
        )
        self.assertTrue(router_parameters)
        self.assertTrue(expert_parameters)

        controller.apply_step(step=0, training=True)
        self.assertTrue(all(not item.requires_grad for item in router_parameters))
        self.assertTrue(all(item.requires_grad for item in expert_parameters))
        controller.apply_step(step=1, training=True)
        self.assertTrue(all(item.requires_grad for item in router_parameters))

    def test_router_free_target_preset_fails_at_binding(self) -> None:
        pipeline = self._pipeline()
        pipeline.configure_router_stage_schedule(RouterStageScheduleController())
        with self.assertRaisesRegex(ValueError, "router adapter parameter"):
            pipeline.set_adapter_config(
                AdapterConfig(target_preset="attn_only", rank=2, alpha=2.0)
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
