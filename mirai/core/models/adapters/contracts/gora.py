"""Behavioral contracts for GoRA allocation, initialization, and persistence."""

from __future__ import annotations

import unittest

from mirai.core.models.adapters.lora_gora import (
    allocate_gora_ranks,
    gora_sensitivity_importance,
    initialize_gora_module,
)

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


@unittest.skipIf(torch is None, "torch not installed")
class GoRAContracts(unittest.TestCase):
    def test_sensitivity_importance_uses_matrix_means_and_group_sum(self) -> None:
        matrix = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        gradient = torch.tensor([[2.0, 1.0], [-1.0, 0.5]])
        expected = (2.0 + 2.0 + 3.0 + 2.0) / 4.0
        self.assertEqual(
            gora_sensitivity_importance(matrix, gradient),
            expected,
        )
        grouped = torch.stack((matrix, matrix * 2.0))
        grouped_gradient = torch.stack((gradient, gradient))
        self.assertEqual(
            gora_sensitivity_importance(grouped, grouped_gradient),
            expected * 3.0,
        )

    def test_rank_allocation_is_group_cost_aware_and_deterministic(self) -> None:
        plan = allocate_gora_ranks(
            {
                "dense": (1, 4, 4),
                "experts": (2, 4, 4),
            },
            {
                "dense": 3.0,
                "experts": 1.0,
            },
            reference_rank=2,
            minimum_rank=1,
            maximum_rank=4,
        )
        self.assertEqual(plan.ranks, {"dense": 4, "experts": 1})
        repeated = allocate_gora_ranks(
            {
                "experts": (2, 4, 4),
                "dense": (1, 4, 4),
            },
            {
                "experts": 1.0,
                "dense": 3.0,
            },
            reference_rank=2,
            minimum_rank=1,
            maximum_rank=4,
        )
        self.assertEqual(plan.fingerprint, repeated.fingerprint)
        self.assertEqual(plan.actual_reference_parameters, 48)
        self.assertEqual(plan.actual_allocated_parameters, 48)

    def test_pseudoinverse_initialization_matches_paper_geometry(self) -> None:
        from mirai.core.models.adapters.lora import LoRALinear

        base = nn.Linear(4, 5, bias=False)
        original = base.weight.detach().clone()
        module = LoRALinear(
            base,
            rank=3,
            alpha=3.0,
            init="gora",
            use_rslora=True,
        )
        gradient = torch.tensor(
            [
                [0.1, 0.2, -0.3, 0.4],
                [0.2, -0.1, 0.5, 0.3],
                [-0.4, 0.1, 0.2, -0.2],
                [0.3, 0.2, 0.1, -0.5],
                [0.5, -0.4, 0.3, 0.2],
            ]
        )
        diagnostic = initialize_gora_module(
            module,
            gradient,
            rank=2,
            stable_gamma=0.05,
            seed=17,
        )
        expected_a = (
            -torch.linalg.pinv(module.lora_b.detach().float())
            @ gradient.float()
            * (0.05 * (5.0**0.5) / 3.0)
        )
        torch.testing.assert_close(module.lora_a, expected_a)
        torch.testing.assert_close(base.weight, original)
        self.assertGreater(diagnostic.cosine_similarity, 0.0)
        self.assertEqual(module.rank, 2)

    def test_grouped_expert_initialization_is_per_physical_expert(self) -> None:
        from mirai.core.models.adapters.lora import (
            LoRAExpertTensorParametrization,
        )

        base = nn.Parameter(torch.zeros(2, 4, 3), requires_grad=False)
        module = LoRAExpertTensorParametrization(
            adapter_name="blocks.0.experts.w1",
            shape=(2, 4, 3),
            layout=("expert", "out", "in"),
            rank=2,
            alpha=2.0,
            init="gora",
            use_rslora=True,
            base_weight=base,
        )
        gradient = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 24
        initialize_gora_module(
            module,
            gradient,
            rank=2,
            stable_gamma=0.05,
            seed=3,
        )
        self.assertIs(module.gora_base_weight(), base)
        for expert in range(2):
            expected = (
                -torch.linalg.pinv(module.lora_b[expert].detach().float())
                @ gradient[expert]
                * (0.05 * (4.0**0.5) / 2.0)
            )
            torch.testing.assert_close(module.lora_a[expert], expected)

    def test_gora_state_load_reconstructs_calibrated_rank(self) -> None:
        from mirai.core.models.adapters.lora import (
            LoRALinear,
            load_lora_state_dict,
            lora_state_dict,
        )

        def root(rank: int):
            result = nn.Module()
            result.proj = LoRALinear(
                nn.Linear(4, 5, bias=False),
                rank=rank,
                alpha=3.0,
                init="gora",
                use_rslora=True,
            )
            return result

        source = root(3)
        initialize_gora_module(
            source.proj,
            torch.randn(5, 4),
            rank=2,
            stable_gamma=0.05,
            seed=9,
        )
        state = lora_state_dict(source)
        restored = root(3)
        load_lora_state_dict(restored, state)
        self.assertEqual(restored.proj.rank, 2)
        torch.testing.assert_close(restored.proj.lora_a, source.proj.lora_a)
        torch.testing.assert_close(restored.proj.lora_b, source.proj.lora_b)

    def test_target_discovery_ignores_provider_lifecycle_metadata(self) -> None:
        from mirai.core.models.adapters.lora import LoRALinear
        from mirai.core.training.calibration.gora import _collect_gora_targets

        root = nn.Module()
        root._lora_init = "gora"
        root.proj = LoRALinear(
            nn.Linear(4, 5, bias=False),
            rank=2,
            alpha=2.0,
            init="gora",
            use_rslora=True,
        )

        targets = _collect_gora_targets(root)

        self.assertEqual([target.name for target in targets], ["proj"])

    def test_zero_total_importance_fails_instead_of_inventing_ranks(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero total"):
            allocate_gora_ranks(
                {"a": (1, 4, 4)},
                {"a": 0.0},
                reference_rank=2,
                minimum_rank=1,
                maximum_rank=4,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
