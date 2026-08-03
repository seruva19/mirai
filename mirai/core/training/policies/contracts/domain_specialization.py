from __future__ import annotations

# Colocated behavioral contract for dataset-domain specialization.

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn.utils import parametrize

from mirai.config.schema import TrainingConfig
from mirai.core.models.compressed_weights.execution.active_expert_lora import ActiveExpertLoRA
from mirai.core.moe.adaptation.domain_specialization import (
    DomainExpertSpecializationController,
)
from mirai.core.training.training_policy import validate_training_policy_configs


class _Router(nn.Module):
    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.last_top_indices = None

    def forward(self, indices):
        self.last_top_indices = indices
        return indices


class _Experts(nn.Module):
    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.expert_lora = nn.ModuleDict(
            {
                "w1": ActiveExpertLoRA(
                    adapter_name="w1",
                    num_experts=num_experts,
                    in_features=3,
                    out_features=4,
                    rank=2,
                    alpha=2.0,
                )
            }
        )
        self.route_gate = None

    def set_routed_adapter_gate(self, gate) -> None:
        self.route_gate = gate


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = _Router(4)
        self.experts = _Experts(4)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()


class DomainExpertSpecializationTests(unittest.TestCase):
    def _controller(self) -> tuple[DomainExpertSpecializationController, _Model]:
        model = _Model()
        controller = DomainExpertSpecializationController(
            warmup_steps=2,
            affinity_threshold=0.5,
            min_experts=2,
            momentum=0.5,
            update_interval=1,
        )
        controller.bind_model(model)
        return controller, model

    def test_warmup_learns_affinity_then_masks_expert_adapters(self) -> None:
        controller, model = self._controller()
        for step in (0, 1):
            controller.bind_batch(domains=("animals",), step=step, training=True)
            model.block.router(torch.tensor([[3, 3], [3, 1]]))
            controller.after_optimizer_step(None, applied=True)
        controller.bind_batch(domains=("animals",), step=2, training=True)
        adapter = model.block.experts.expert_lora["w1"]
        self.assertEqual(adapter.active_expert_ids(), [1, 3])

    def test_domains_are_learned_independently_and_state_round_trips(self) -> None:
        controller, model = self._controller()
        controller.bind_batch(domains=("a",), step=0, training=True)
        model.block.router(torch.tensor([[0, 0], [0, 1]]))
        controller.after_optimizer_step(None, applied=True)
        controller.bind_batch(domains=("b",), step=1, training=True)
        model.block.router(torch.tensor([[2, 2], [2, 3]]))
        controller.after_optimizer_step(None, applied=True)
        state = controller.state_dict()

        restored, restored_model = self._controller()
        restored.load_state_dict(state)
        restored.bind_batch(domains=("b",), step=2, training=True)
        self.assertEqual(
            restored_model.block.experts.expert_lora["w1"].active_expert_ids(),
            [2, 3],
        )

    def test_mixed_domain_batch_learns_and_installs_per_sample_gate(self) -> None:
        controller, model = self._controller()
        for step in (0, 1):
            controller.bind_batch(
                domains=("animals", "motion"),
                step=step,
                training=True,
            )
            model.block.router(
                torch.tensor(
                    [
                        [0, 0],
                        [0, 1],
                        [2, 2],
                        [2, 3],
                    ]
                )
            )
            controller.after_optimizer_step(None, applied=True)

        controller.bind_batch(
            domains=("animals", "motion"),
            step=2,
            training=True,
        )
        gate = model.block.experts.route_gate
        self.assertIsNotNone(gate)
        gate.bind_tokens_per_sample(2)
        actual = gate.resolve(
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 2, 0, 2]),
        )
        torch.testing.assert_close(
            actual,
            torch.tensor([True, False, False, True]),
        )
        self.assertEqual(
            model.block.experts.expert_lora["w1"].active_expert_ids(),
            [0, 1, 2, 3],
        )

    def test_mixed_domain_batch_rejects_unlearned_domain(self) -> None:
        controller, model = self._controller()
        controller.bind_batch(domains=("animals",), step=0, training=True)
        model.block.router(torch.tensor([[0, 0], [0, 1]]))
        controller.after_optimizer_step(None, applied=True)
        controller.bind_batch(domains=("animals",), step=1, training=True)
        model.block.router(torch.tensor([[0, 1]]))
        controller.after_optimizer_step(None, applied=True)
        with self.assertRaisesRegex(RuntimeError, "no learned affinity"):
            controller.bind_batch(
                domains=("animals", "unseen"),
                step=2,
                training=True,
            )

    def test_lingbot_binding_installs_gate_on_native_activation_executor(self) -> None:
        from mirai.core.models.adapters.expert_tensor_lora import (
            install_expert_tensor_lora_executor,
        )
        from mirai.core.models.lingbot_video.domain_specialization import (
            configure_lingbot_domain_expert_specialization,
        )
        from mirai.core.models.adapters.lora import LoRAExpertTensorParametrization
        from mirai.vendors.lingbot_video.transformer_lingbot_video import (
            LingBotVideoGroupedExperts,
        )

        root = nn.Module()
        root.block = nn.Module()
        root.block.router = _Router(4)
        root.block.experts = LingBotVideoGroupedExperts(4, 3, 4)
        for key in ("w1", "w2", "w3"):
            weight = getattr(root.block.experts, key)
            parametrize.register_parametrization(
                root.block.experts,
                key,
                LoRAExpertTensorParametrization(
                    adapter_name=key,
                    shape=tuple(weight.shape),
                    layout=("expert", "out", "in"),
                    rank=2,
                    alpha=2.0,
                ),
            )
        install_expert_tensor_lora_executor(root.block.experts)
        controller = DomainExpertSpecializationController(
            warmup_steps=2,
            affinity_threshold=0.5,
            min_experts=1,
        )
        configure_lingbot_domain_expert_specialization(
            SimpleNamespace(transformer=root),
            controller,
        )
        controller.load_state_dict(
            {
                "scores": {
                    "block": {
                        "a": [4.0, 0.0, 0.0, 0.0],
                        "b": [0.0, 0.0, 4.0, 0.0],
                    }
                }
            }
        )
        controller.bind_batch(domains=("a", "b"), step=2, training=True)
        extension = root.block.experts.linear_extension()
        self.assertIsNotNone(extension._routed_adapter_gate)

    def test_optimizer_moments_are_cleared_for_inactive_slices(self) -> None:
        controller, model = self._controller()
        controller.bind_batch(domains=("a",), step=0, training=True)
        model.block.router(torch.tensor([[0, 0], [0, 1]]))
        controller.after_optimizer_step(None, applied=True)
        controller.bind_batch(domains=("a",), step=1, training=True)
        model.block.router(torch.tensor([[0, 1]]))
        controller.after_optimizer_step(None, applied=True)
        controller.bind_batch(domains=("a",), step=2, training=True)
        adapter = model.block.experts.expert_lora["w1"]
        optimizer = torch.optim.AdamW(
            [adapter.lora_a, adapter.lora_b], lr=0.1, weight_decay=0.0
        )
        for param in (adapter.lora_a, adapter.lora_b):
            optimizer.state[param]["step"] = torch.tensor(1.0)
            optimizer.state[param]["exp_avg"] = torch.ones_like(param)
            optimizer.state[param]["exp_avg_sq"] = torch.ones_like(param)
        controller.before_optimizer_step(optimizer)
        for param in (adapter.lora_a, adapter.lora_b):
            state = optimizer.state[param]
            self.assertTrue(torch.all(state["exp_avg"][2:] == 0))
            self.assertTrue(torch.all(state["exp_avg_sq"][2:] == 0))
            self.assertTrue(torch.all(state["exp_avg"][:2] == 1))

    def test_accumulation_window_uses_union_of_domain_experts(self) -> None:
        controller, model = self._controller()
        controller.bind_batch(domains=("a",), step=0, training=True)
        model.block.router(torch.tensor([[0, 0], [0, 1]]))
        controller.after_optimizer_step(None, applied=True)
        controller.bind_batch(domains=("b",), step=1, training=True)
        model.block.router(torch.tensor([[2, 2], [2, 3]]))
        controller.after_optimizer_step(None, applied=True)
        adapter = model.block.experts.expert_lora["w1"]
        optimizer = torch.optim.AdamW(
            [adapter.lora_a, adapter.lora_b], lr=0.1, weight_decay=0.0
        )
        for param in (adapter.lora_a, adapter.lora_b):
            optimizer.state[param]["exp_avg"] = torch.ones_like(param)
        controller.bind_batch(domains=("a",), step=2, training=True)
        controller.bind_batch(domains=("b",), step=2, training=True)
        controller.before_optimizer_step(optimizer)
        for param in (adapter.lora_a, adapter.lora_b):
            self.assertTrue(torch.all(optimizer.state[param]["exp_avg"] == 1))

    def test_adamw_decay_applies_only_to_window_active_slices(self) -> None:
        controller, model = self._controller()
        controller.bind_batch(domains=("a",), step=0, training=True)
        model.block.router(torch.tensor([[0, 0], [0, 1]]))
        controller.after_optimizer_step(None, applied=True)
        controller.bind_batch(domains=("a",), step=1, training=True)
        model.block.router(torch.tensor([[0, 1]]))
        controller.after_optimizer_step(None, applied=True)
        controller.bind_batch(domains=("a",), step=2, training=True)
        adapter = model.block.experts.expert_lora["w1"]
        with torch.no_grad():
            adapter.lora_a.fill_(1.0)
        adapter.lora_a.grad = torch.zeros_like(adapter.lora_a)
        optimizer = torch.optim.AdamW(
            [adapter.lora_a], lr=0.1, weight_decay=0.2
        )
        controller.before_optimizer_step(optimizer)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)
        optimizer.step()
        controller.after_optimizer_step(optimizer, applied=True)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.2)
        self.assertTrue(torch.allclose(adapter.lora_a[:2], torch.full_like(adapter.lora_a[:2], 0.98)))
        self.assertTrue(torch.equal(adapter.lora_a[2:], torch.ones_like(adapter.lora_a[2:])))

    def test_counts_commit_once_per_successful_optimizer_window(self) -> None:
        controller, model = self._controller()
        controller.bind_batch(domains=("a",), step=0, training=True)
        model.block.router(torch.tensor([[0, 1]]))
        model.block.router(torch.tensor([[0, 2]]))
        controller.after_optimizer_step(None, applied=True)
        scores = controller.state_dict()["scores"]["block"]["a"]
        self.assertEqual(scores, [2.0, 1.0, 1.0, 0.0])

        controller.bind_batch(domains=("a",), step=1, training=True)
        model.block.router(torch.tensor([[3, 3]]))
        controller.after_optimizer_step(None, applied=False)
        self.assertEqual(
            controller.state_dict()["scores"]["block"]["a"],
            [2.0, 1.0, 1.0, 0.0],
        )

    def test_config_allows_accumulation_and_gates_non_adamw_decay(self) -> None:
        config = TrainingConfig()
        config.model.type = "lingbot-video"
        config.training.gradient_accumulation = 4
        config.training.policy_options = {
            "domain_expert_specialization": {
                "enabled": True,
                "domain_metadata_key": "domain",
            }
        }
        self.assertFalse(
            any(
                "gradient_accumulation" in error
                for error in validate_training_policy_configs(config)
            )
        )
        config.optimizer.type = "lion"
        config.optimizer.weight_decay = 0.1
        self.assertTrue(
            any(
                "requires optimizer.type='adamw'" in error
                for error in validate_training_policy_configs(config)
            )
        )


if __name__ == "__main__":
    unittest.main()
