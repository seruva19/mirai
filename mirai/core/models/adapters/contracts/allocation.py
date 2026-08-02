"""Static rank/alpha allocation and rsLoRA scaling contracts."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

# Colocated behavioral contract for adapter allocation and diagnostics.

from mirai.core.models.adapters.lora_allocation import LoRAAllocationPolicy
from mirai.core.models.adapters.lora_allocation import lora_scale
from mirai.core.training.calibration.conditioning import (
    allocation_conditioning_correlation,
    target_conditioning_diagnostics,
)

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


class LoRAAllocationPolicyTests(unittest.TestCase):
    def _policy(self, **overrides):
        values = {
            "rank": 8,
            "alpha": 16.0,
            "rank_pattern": {},
            "alpha_pattern": {},
            "rank_budget": 0,
            "use_rslora": False,
            **overrides,
        }
        return LoRAAllocationPolicy.from_adapter_config(SimpleNamespace(**values))

    def test_default_plan_is_uniform_and_legacy_scaled(self) -> None:
        plan = self._policy().resolve(["blocks.0.to_q", "blocks.1.to_q"])
        self.assertEqual(plan.total_rank, 16)
        self.assertEqual(plan.for_target("blocks.0.to_q").rank, 8)
        self.assertEqual(plan.for_target("blocks.0.to_q").scale, 2.0)

    def test_patterns_resolve_before_mutation(self) -> None:
        plan = self._policy(
            rank_pattern={"blocks.0.*": 4, "*.experts.w1": 12},
            alpha_pattern={"blocks.0.*": 8.0},
            use_rslora=True,
        ).resolve(["blocks.0.to_q", "blocks.2.experts.w1"])
        first = plan.for_target("blocks.0.to_q")
        expert = plan.for_target("blocks.2.experts.w1")
        self.assertEqual((first.rank, first.alpha), (4, 8.0))
        self.assertEqual((expert.rank, expert.alpha), (12, 16.0))
        self.assertEqual(first.scaling_rule, "alpha_over_sqrt_rank")
        self.assertAlmostEqual(first.scale, 4.0)

    def test_unmatched_ambiguous_and_over_budget_patterns_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "matched no targets"):
            self._policy(rank_pattern={"missing.*": 4}).resolve(["blocks.0.to_q"])
        with self.assertRaisesRegex(ValueError, "ambiguously"):
            self._policy(
                rank_pattern={"blocks.*": 4, "*.to_q": 6}
            ).resolve(["blocks.0.to_q"])
        with self.assertRaisesRegex(ValueError, "exceeds rank_budget"):
            self._policy(rank_budget=15).resolve(["a", "b"])

    def test_exact_adaptive_ranks_require_identical_target_topology(self) -> None:
        plan = self._policy(rank_budget=6).resolve(
            ["experts.0", "experts.1"],
            exact_ranks={"experts.0": 2, "experts.1": 4},
        )
        self.assertEqual(plan.for_target("experts.0").rank, 2)
        self.assertEqual(plan.total_rank, 6)
        with self.assertRaisesRegex(ValueError, "target topology mismatch"):
            self._policy().resolve(
                ["experts.0", "experts.1"],
                exact_ranks={"experts.0": 2},
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self._policy(rank_pattern={"experts.*": 2}).resolve(
                ["experts.0"], exact_ranks={"experts.0": 2}
            )

    def test_rslora_scale_is_rank_stabilized(self) -> None:
        self.assertEqual(lora_scale(16.0, 16, use_rslora=False), 1.0)
        self.assertEqual(lora_scale(16.0, 16, use_rslora=True), 4.0)

    def test_conditioning_diagnostics_are_bounded_and_correlate_with_ranks(self) -> None:
        diagnostics = target_conditioning_diagnostics(
            {
                "well": [4.0, 2.0, 1.0],
                "ill": [16.0, 1.0],
                "empty": [0.0, 1e-15],
            }
        )
        self.assertEqual(set(diagnostics), {"ill", "well"})
        self.assertEqual(diagnostics["well"]["condition_number"], 4.0)
        self.assertEqual(diagnostics["well"]["effective_rank"], 3)
        self.assertEqual(diagnostics["well"]["stable_rank"], 1.75)
        correlation = allocation_conditioning_correlation(
            diagnostics,
            {"well": 4, "ill": 16},
        )
        self.assertIsNotNone(correlation)
        self.assertAlmostEqual(float(correlation), 1.0)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_linear_rslora_forward_and_state_contract(self) -> None:
        from mirai.core.models.adapters.lora import LoRALinear
        from mirai.core.models.adapters.lora import load_lora_state_dict
        from mirai.core.models.adapters.lora import lora_state_dict
        from mirai.core.models.adapters.lora_allocation import RSLORA_STATE_SUFFIX

        def root(*, use_rslora: bool):
            module = nn.Module()
            base = nn.Linear(4, 3, bias=False)
            base.weight.data.zero_()
            module.proj = LoRALinear(
                base,
                rank=4,
                alpha=4.0,
                use_rslora=use_rslora,
            )
            module.proj.lora_a.data.fill_(1.0)
            module.proj.lora_b.data.fill_(1.0)
            return module

        classic = root(use_rslora=False)
        stabilized = root(use_rslora=True)
        sample = torch.ones(2, 4)
        torch.testing.assert_close(stabilized.proj(sample), classic.proj(sample) * 2.0)

        classic_state = lora_state_dict(classic)
        stabilized_state = lora_state_dict(stabilized)
        key = f"proj{RSLORA_STATE_SUFFIX}"
        self.assertNotIn(key, classic_state)
        self.assertEqual(int(stabilized_state[key].item()), 1)
        load_lora_state_dict(root(use_rslora=True), stabilized_state)
        with self.assertRaisesRegex(ValueError, "scaling rule mismatch"):
            load_lora_state_dict(root(use_rslora=False), stabilized_state)
        with self.assertRaisesRegex(ValueError, "scaling rule mismatch"):
            load_lora_state_dict(root(use_rslora=True), classic_state)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_expert_tensor_rslora_uses_same_scaling_contract(self) -> None:
        from mirai.core.models.adapters.lora import LoRAExpertTensorParametrization

        classic = LoRAExpertTensorParametrization(
            adapter_name="experts.w1",
            shape=(2, 3, 4),
            layout=("expert", "out", "in"),
            rank=4,
            alpha=4.0,
        )
        stabilized = LoRAExpertTensorParametrization(
            adapter_name="experts.w1",
            shape=(2, 3, 4),
            layout=("expert", "out", "in"),
            rank=4,
            alpha=4.0,
            use_rslora=True,
        )
        for module in (classic, stabilized):
            module.lora_a.data.fill_(1.0)
            module.lora_b.data.fill_(1.0)
        base = torch.zeros(2, 3, 4)
        torch.testing.assert_close(stabilized(base), classic(base) * 2.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
