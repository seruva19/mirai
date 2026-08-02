"""Behavioral contracts for structured LoRA parameter dropout."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from mirai.core.models.adapters.lora import LoRAExpertTensorParametrization
from mirai.core.models.adapters.lora import LoRALinear
from mirai.core.models.adapters.lora import set_lora_parameter_dropout
from mirai.core.models.adapters.lora_parameter_dropout import (
    apply_lora_parameter_dropout,
)
from mirai.core.models.compressed_weights.execution.active_expert_lora import (
    ActiveExpertLoRA,
)


class LoRAParameterDropoutTests(unittest.TestCase):
    def test_disabled_and_eval_paths_preserve_identity_and_rng(self) -> None:
        a = torch.randn(3, 5)
        b = torch.randn(7, 3)
        before = torch.random.get_rng_state().clone()
        disabled_a, disabled_b = apply_lora_parameter_dropout(
            a, b, probability=0.0, training=True
        )
        self.assertIs(disabled_a, a)
        self.assertIs(disabled_b, b)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

        eval_a, eval_b = apply_lora_parameter_dropout(
            a, b, probability=0.5, training=False
        )
        self.assertIs(eval_a, a)
        self.assertIs(eval_b, b)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_masks_input_columns_and_output_rows_not_rank_channels(self) -> None:
        torch.manual_seed(7)
        a = torch.ones(4, 5, 17)
        b = torch.ones(4, 13, 5)
        masked_a, masked_b = apply_lora_parameter_dropout(
            a, b, probability=0.5, training=True
        )

        self.assertTrue(torch.equal(masked_a, masked_a[:, :1, :].expand_as(a)))
        self.assertTrue(torch.equal(masked_b, masked_b[:, :, :1].expand_as(b)))
        self.assertGreater(int((masked_a == 0).sum()), 0)
        self.assertGreater(int((masked_b == 0).sum()), 0)
        self.assertEqual(set(masked_a.unique().tolist()), {0.0, 1.0})
        self.assertEqual(set(masked_b.unique().tolist()), {0.0, 1.0})

    def test_masked_factor_entries_receive_exact_zero_gradients(self) -> None:
        seed = 19
        torch.manual_seed(seed)
        mask_a, mask_b = apply_lora_parameter_dropout(
            torch.ones(3, 11),
            torch.ones(7, 3),
            probability=0.5,
            training=True,
        )
        a = torch.randn(3, 11, requires_grad=True)
        b = torch.randn(7, 3, requires_grad=True)
        torch.manual_seed(seed)
        dropped_a, dropped_b = apply_lora_parameter_dropout(
            a, b, probability=0.5, training=True
        )
        (dropped_a.sum() + dropped_b.sum()).backward()

        self.assertTrue(torch.equal(a.grad, mask_a))
        self.assertTrue(torch.equal(b.grad, mask_b))
        self.assertGreater(int((a.grad == 0).sum()), 0)
        self.assertGreater(int((b.grad == 0).sum()), 0)

    def test_linear_host_matches_paper_equation(self) -> None:
        torch.manual_seed(31)
        base = nn.Linear(6, 4, bias=False)
        base.weight.data.zero_()
        module = LoRALinear(base, rank=3, alpha=3.0)
        module.lora_a.data.normal_()
        module.lora_b.data.normal_()
        module.set_lora_parameter_dropout(0.4)
        module.train()
        x = torch.randn(5, 6)

        seed = 41
        torch.manual_seed(seed)
        expected_a, expected_b = apply_lora_parameter_dropout(
            module.lora_a,
            module.lora_b,
            probability=0.4,
            training=True,
        )
        expected = F.linear(F.linear(x, expected_a), expected_b)
        torch.manual_seed(seed)
        actual = module(x)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_weight_and_activation_space_expert_hosts_share_semantics(self) -> None:
        torch.manual_seed(47)
        adapter = LoRAExpertTensorParametrization(
            adapter_name="experts.w1",
            shape=(3, 5, 7),
            layout=("expert", "out", "in"),
            rank=2,
            alpha=2.0,
        )
        adapter.lora_a.data.normal_()
        adapter.lora_b.data.normal_()
        adapter.set_lora_parameter_dropout(0.35)
        adapter.train()

        base = torch.randn(3, 5, 7)
        seed = 53
        torch.manual_seed(seed)
        expected_a, expected_b = apply_lora_parameter_dropout(
            adapter.lora_a,
            adapter.lora_b,
            probability=0.35,
            training=True,
        )
        expected_weight = base + torch.matmul(expected_b, expected_a)
        torch.manual_seed(seed)
        torch.testing.assert_close(
            adapter(base), expected_weight, rtol=0.0, atol=0.0
        )

        torch.manual_seed(seed)
        actual_a, actual_b, scale = adapter.activation_factors(base)
        torch.testing.assert_close(actual_a, expected_a, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_b, expected_b, rtol=0.0, atol=0.0)
        self.assertEqual(scale, 1.0)

    def test_active_expert_subset_host_matches_factor_reference(self) -> None:
        torch.manual_seed(59)
        adapter = ActiveExpertLoRA(
            adapter_name="experts.w1",
            num_experts=4,
            in_features=6,
            out_features=5,
            rank=3,
            alpha=3.0,
        )
        adapter.lora_a.data.normal_()
        adapter.lora_b.data.normal_()
        adapter.set_lora_parameter_dropout(0.45)
        adapter.train()
        expert_indices = torch.tensor([1, 3], dtype=torch.long)
        x = torch.randn(2, 4, 6)
        selected_a = adapter.lora_a.index_select(0, expert_indices)
        selected_b = adapter.lora_b.index_select(0, expert_indices)

        seed = 61
        torch.manual_seed(seed)
        expected_a, expected_b = apply_lora_parameter_dropout(
            selected_a,
            selected_b,
            probability=0.45,
            training=True,
        )
        expected = torch.bmm(
            torch.bmm(x, expected_a.transpose(-2, -1)),
            expected_b.transpose(-2, -1),
        )
        torch.manual_seed(seed)
        actual = adapter.batched_subset_forward(
            x, expert_indices=expert_indices
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_runtime_setter_reaches_dense_and_expert_hosts(self) -> None:
        root = nn.Module()
        root.linear = LoRALinear(nn.Linear(3, 2), rank=2, alpha=2.0)
        root.expert = ActiveExpertLoRA(
            adapter_name="expert",
            num_experts=2,
            in_features=3,
            out_features=2,
            rank=2,
            alpha=2.0,
        )
        set_lora_parameter_dropout(root, 0.2)
        self.assertEqual(root.linear._lora_parameter_dropout, 0.2)
        self.assertEqual(root.expert._lora_parameter_dropout, 0.2)

    def test_invalid_probability_and_factor_shapes_fail(self) -> None:
        a = torch.ones(2, 3)
        b = torch.ones(4, 2)
        for value in (-0.1, 1.0, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be in"):
                    apply_lora_parameter_dropout(
                        a, b, probability=value, training=True
                    )
        with self.assertRaisesRegex(ValueError, "leading dimensions"):
            apply_lora_parameter_dropout(
                torch.ones(2, 2, 3),
                torch.ones(3, 4, 2),
                probability=0.2,
                training=True,
            )
        with self.assertRaisesRegex(ValueError, "rank dimension"):
            apply_lora_parameter_dropout(
                torch.ones(2, 3),
                torch.ones(4, 5),
                probability=0.2,
                training=True,
            )
