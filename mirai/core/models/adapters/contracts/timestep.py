"""Timestep-axis adapter tests: T-LoRA rank masking, bands, init, telemetry."""

from __future__ import annotations

import unittest

# Colocated behavioral contract for timestep-conditioned adapters.

from mirai.config.schema import ConfigError, TrainingConfig
from mirai.core.models.adapters.timestep_axis import (
    TimestepAdapterPolicy,
    band_is_active,
    full_range_band,
    tlora_active_rank,
    tlora_rank_fraction,
)
from mirai.core.models.adapters.sparse_delta import SparseDeltaLinear

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


class TLoRAScheduleMathTests(unittest.TestCase):
    def test_min_fraction_at_full_noise(self) -> None:
        self.assertAlmostEqual(tlora_rank_fraction(1.0, 0.25), 0.25)

    def test_full_rank_at_zero_noise(self) -> None:
        self.assertAlmostEqual(tlora_rank_fraction(0.0, 0.25), 1.0)

    def test_linear_midpoint(self) -> None:
        self.assertAlmostEqual(tlora_rank_fraction(0.5, 0.5), 0.75)

    def test_active_rank_bounds(self) -> None:
        self.assertEqual(tlora_active_rank(1.0, rank=16, min_fraction=0.25), 4)
        self.assertEqual(tlora_active_rank(0.0, rank=16, min_fraction=0.25), 16)
        # Never below 1 even for tiny fractions/ranks.
        self.assertEqual(tlora_active_rank(1.0, rank=2, min_fraction=0.1), 1)

    def test_band_membership(self) -> None:
        self.assertTrue(band_is_active(0.9, 0.875, 1.0))
        self.assertTrue(band_is_active(1.0, 0.875, 1.0))  # inclusive top at 1.0
        self.assertFalse(band_is_active(0.5, 0.875, 1.0))
        self.assertFalse(band_is_active(0.5, 0.0, 0.5))  # half-open [min, max)
        self.assertTrue(band_is_active(0.4999, 0.0, 0.5))

    def test_full_range_detection(self) -> None:
        self.assertTrue(full_range_band(0.0, 1.0))
        self.assertFalse(full_range_band(0.0, 0.99))

    def test_policy_noop_default(self) -> None:
        policy = TimestepAdapterPolicy()
        self.assertTrue(policy.is_noop())

        class _Cfg:
            timestep_rank_schedule = "none"
            timestep_rank_min_fraction = 0.25
            timestep_band_min = 0.0
            timestep_band_max = 1.0

        self.assertIsNone(TimestepAdapterPolicy.from_adapter_config(_Cfg()))

    def test_policy_validation(self) -> None:
        with self.assertRaises(ValueError):
            TimestepAdapterPolicy(schedule="cosine")
        with self.assertRaises(ValueError):
            TimestepAdapterPolicy(min_fraction=0.0)
        with self.assertRaises(ValueError):
            TimestepAdapterPolicy(band_min=0.5, band_max=0.5)
        with self.assertRaises(ValueError):
            TimestepAdapterPolicy(band_min=-0.1, band_max=1.0)


@unittest.skipIf(torch is None, "torch not installed")
class PerSampleRankMaskTests(unittest.TestCase):
    def test_tlora_prefix_mask(self) -> None:
        from mirai.core.models.adapters.timestep_axis import per_sample_rank_mask

        policy = TimestepAdapterPolicy(schedule="tlora", min_fraction=0.25)
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        mask = per_sample_rank_mask(sigmas, rank=16, policy=policy)
        self.assertEqual(tuple(mask.shape), (3, 16))
        # sigma=1 -> 4 active; sigma=0.5 -> ceil(0.625*16)=10; sigma=0 -> 16.
        self.assertEqual(int(mask[0].sum().item()), 4)
        self.assertEqual(int(mask[1].sum().item()), 10)
        self.assertEqual(int(mask[2].sum().item()), 16)
        # Prefix property: mask rows are non-increasing along the rank axis.
        diffs = mask[:, 1:] - mask[:, :-1]
        self.assertTrue(bool((diffs <= 0).all().item()))

    def test_band_zeroes_out_of_band_rows(self) -> None:
        from mirai.core.models.adapters.timestep_axis import per_sample_rank_mask

        policy = TimestepAdapterPolicy(band_min=0.875, band_max=1.0)
        sigmas = torch.tensor([0.5, 0.9, 1.0])
        mask = per_sample_rank_mask(sigmas, rank=8, policy=policy)
        self.assertEqual(float(mask[0].abs().sum().item()), 0.0)
        self.assertEqual(int(mask[1].sum().item()), 8)
        self.assertEqual(int(mask[2].sum().item()), 8)  # sigma == 1.0 kept

    def test_batch_uniform_reduction_is_conservative(self) -> None:
        from mirai.core.models.adapters.timestep_axis import batch_uniform_rank_mask

        policy = TimestepAdapterPolicy(schedule="tlora", min_fraction=0.25)
        sigmas = torch.tensor([1.0, 0.0])
        uniform = batch_uniform_rank_mask(sigmas, rank=16, policy=policy)
        self.assertEqual(int(uniform.sum().item()), 4)  # min active rank wins
        banded = TimestepAdapterPolicy(band_min=0.875, band_max=1.0)
        uniform2 = batch_uniform_rank_mask(
            torch.tensor([0.9, 0.5]), rank=16, policy=banded
        )
        self.assertEqual(float(uniform2.abs().sum().item()), 0.0)


@unittest.skipIf(torch is None, "torch not installed")
class LoRALinearTimestepMaskTests(unittest.TestCase):
    def _module(self, *, rank: int = 8, init: str = "kaiming"):
        from mirai.core.models.adapters.lora import LoRALinear

        torch.manual_seed(7)
        base = nn.Linear(6, 5)
        module = LoRALinear(base, rank=rank, alpha=float(rank), init=init)
        with torch.no_grad():
            module.lora_b.normal_(0.0, 0.05)  # nonzero so masking is observable
        return module

    def _masks(self, sigmas, *, rank: int, policy):
        from mirai.core.models.adapters.timestep_axis import per_sample_rank_mask

        mask = per_sample_rank_mask(sigmas, rank=rank, policy=policy)
        return mask, mask.amin(dim=0)

    def test_per_sample_output_exactness(self) -> None:
        module = self._module(rank=8)
        policy = TimestepAdapterPolicy(schedule="tlora", min_fraction=0.5)
        sigmas = torch.tensor([1.0, 0.0])
        mask, uniform = self._masks(sigmas, rank=8, policy=policy)
        x = torch.randn(2, 3, 6)
        module.set_timestep_rank_masks(mask, uniform)
        out = module(x)
        # Manual reference: sample 0 uses rank prefix 4, sample 1 full rank.
        scale = 1.0  # alpha == rank
        base_out = torch.nn.functional.linear(x, module.base.weight, module.base.bias)
        a, b = module.lora_a, module.lora_b
        ref0 = base_out[0] + (x[0] @ a[:4].t()) @ b[:, :4].t() * scale
        ref1 = base_out[1] + (x[1] @ a.t()) @ b.t() * scale
        torch.testing.assert_close(out[0], ref0)
        torch.testing.assert_close(out[1], ref1)

    def test_masked_ranks_zero_grads(self) -> None:
        module = self._module(rank=8)
        policy = TimestepAdapterPolicy(schedule="tlora", min_fraction=0.5)
        sigmas = torch.tensor([1.0])  # active rank = 4
        mask, uniform = self._masks(sigmas, rank=8, policy=policy)
        module.set_timestep_rank_masks(mask, uniform)
        out = module(torch.randn(1, 3, 6))
        out.sum().backward()
        self.assertTrue(bool((module.lora_a.grad[4:] == 0).all().item()))
        self.assertTrue(bool((module.lora_b.grad[:, 4:] == 0).all().item()))
        # Active prefix does receive gradient.
        self.assertGreater(float(module.lora_a.grad[:4].abs().sum().item()), 0.0)

    def test_full_rank_at_zero_noise_matches_unmasked(self) -> None:
        module = self._module(rank=8)
        policy = TimestepAdapterPolicy(schedule="tlora", min_fraction=0.25)
        x = torch.randn(2, 3, 6)
        with torch.no_grad():
            reference = module(x).clone()
        mask, uniform = self._masks(torch.tensor([0.0, 0.0]), rank=8, policy=policy)
        module.set_timestep_rank_masks(mask, uniform)
        with torch.no_grad():
            masked = module(x)
        torch.testing.assert_close(masked, reference)

    def test_band_gates_sample_output_and_grads(self) -> None:
        module = self._module(rank=8)
        policy = TimestepAdapterPolicy(band_min=0.875, band_max=1.0)
        sigmas = torch.tensor([0.5, 0.9])  # sample 0 out-of-band
        mask, uniform = self._masks(sigmas, rank=8, policy=policy)
        x = torch.randn(2, 3, 6)
        base_out = torch.nn.functional.linear(x, module.base.weight, module.base.bias)
        module.set_timestep_rank_masks(mask, uniform)
        out = module(x)
        torch.testing.assert_close(out[0], base_out[0])  # adapter fully silent
        self.assertGreater(float((out[1] - base_out[1]).abs().sum().item()), 0.0)
        # Out-of-band-only batch -> all adapter grads exactly zero.
        module.zero_grad()
        mask2, uniform2 = self._masks(torch.tensor([0.5]), rank=8, policy=policy)
        module.set_timestep_rank_masks(mask2, uniform2)
        module(torch.randn(1, 3, 6)).sum().backward()
        self.assertTrue(bool((module.lora_a.grad == 0).all().item()))
        self.assertTrue(bool((module.lora_b.grad == 0).all().item()))

    def test_uniform_fallback_when_batch_axis_mismatch(self) -> None:
        module = self._module(rank=8)
        policy = TimestepAdapterPolicy(schedule="tlora", min_fraction=0.5)
        mask, uniform = self._masks(torch.tensor([1.0, 0.0]), rank=8, policy=policy)
        module.set_timestep_rank_masks(mask, uniform)
        x = torch.randn(5, 6)  # leading dim 5 != batch 2 -> uniform (4 ranks)
        out = module(x)
        base_out = torch.nn.functional.linear(x, module.base.weight, module.base.bias)
        a, b = module.lora_a, module.lora_b
        ref = base_out + (x @ a[:4].t()) @ b[:, :4].t()
        torch.testing.assert_close(out, ref)

    def test_clearing_masks_restores_legacy_path(self) -> None:
        module = self._module(rank=8)
        x = torch.randn(2, 3, 6)
        with torch.no_grad():
            reference = module(x).clone()
        policy = TimestepAdapterPolicy(band_min=0.5, band_max=1.0)
        mask, uniform = self._masks(torch.tensor([0.1, 0.2]), rank=8, policy=policy)
        module.set_timestep_rank_masks(mask, uniform)
        module.set_timestep_rank_masks(None, None)
        with torch.no_grad():
            restored = module(x)
        torch.testing.assert_close(restored, reference)


@unittest.skipIf(torch is None, "torch not installed")
class ActiveExpertLoRATimestepMaskTests(unittest.TestCase):
    def _adapter(self, *, rank: int = 8, init: str = "kaiming"):
        from mirai.core.models.compressed_weights.execution.experts import ActiveExpertLoRA

        torch.manual_seed(11)
        adapter = ActiveExpertLoRA(
            adapter_name="w1",
            num_experts=3,
            in_features=6,
            out_features=5,
            rank=rank,
            alpha=float(rank),
            init=init,
        )
        with torch.no_grad():
            adapter.lora_b.normal_(0.0, 0.05)
        return adapter

    def _uniform(self, sigmas, *, rank: int, policy):
        from mirai.core.models.adapters.timestep_axis import batch_uniform_rank_mask

        return batch_uniform_rank_mask(sigmas, rank=rank, policy=policy)

    def test_masked_ranks_zero_output_and_grads(self) -> None:
        adapter = self._adapter(rank=8)
        policy = TimestepAdapterPolicy(schedule="tlora", min_fraction=0.5)
        uniform = self._uniform(torch.tensor([1.0]), rank=8, policy=policy)  # 4 active
        adapter.set_timestep_rank_masks(None, uniform)
        x = torch.randn(3, 4, 6)
        out = adapter.batched_forward(x)
        a, b = adapter.lora_a, adapter.lora_b
        ref = torch.bmm(torch.bmm(x, a[:, :4].transpose(-2, -1)), b[:, :, :4].transpose(-2, -1))
        torch.testing.assert_close(out, ref)
        out.sum().backward()
        self.assertTrue(bool((adapter.lora_a.grad[:, 4:] == 0).all().item()))
        self.assertTrue(bool((adapter.lora_b.grad[:, :, 4:] == 0).all().item()))
        self.assertGreater(float(adapter.lora_a.grad[:, :4].abs().sum().item()), 0.0)

    def test_band_gate_zeroes_everything(self) -> None:
        adapter = self._adapter(rank=8)
        policy = TimestepAdapterPolicy(band_min=0.875, band_max=1.0)
        uniform = self._uniform(torch.tensor([0.5]), rank=8, policy=policy)
        adapter.set_timestep_rank_masks(None, uniform)
        x = torch.randn(3, 4, 6)
        out = adapter.batched_forward(x)
        self.assertEqual(float(out.abs().sum().item()), 0.0)
        out.sum().backward()
        self.assertTrue(bool((adapter.lora_a.grad == 0).all().item()))
        self.assertTrue(bool((adapter.lora_b.grad == 0).all().item()))

    def test_subset_forward_masked(self) -> None:
        adapter = self._adapter(rank=8)
        policy = TimestepAdapterPolicy(schedule="tlora", min_fraction=0.5)
        uniform = self._uniform(torch.tensor([1.0]), rank=8, policy=policy)
        adapter.set_timestep_rank_masks(None, uniform)
        x = torch.randn(2, 4, 6)
        indices = torch.tensor([0, 2])
        out = adapter.batched_subset_forward(x, expert_indices=indices)
        a = adapter.lora_a.index_select(0, indices)
        b = adapter.lora_b.index_select(0, indices)
        ref = torch.bmm(
            torch.bmm(x, a[:, :4].transpose(-2, -1)), b[:, :, :4].transpose(-2, -1)
        )
        torch.testing.assert_close(out, ref)

    def test_per_expert_forward_masked(self) -> None:
        adapter = self._adapter(rank=8)
        policy = TimestepAdapterPolicy(schedule="tlora", min_fraction=0.5)
        uniform = self._uniform(torch.tensor([1.0]), rank=8, policy=policy)
        adapter.set_timestep_rank_masks(None, uniform)
        x = torch.randn(4, 6)
        out = adapter(x, expert_idx=1)
        a, b = adapter.lora_a[1], adapter.lora_b[1]
        ref = (x @ a[:4].t()) @ b[:, :4].t()
        torch.testing.assert_close(out, ref)


@unittest.skipIf(torch is None, "torch not installed")
class ExpertTensorParametrizationTimestepMaskTests(unittest.TestCase):
    def test_weight_space_mask_zero_grads(self) -> None:
        from mirai.core.models.adapters.lora import LoRAExpertTensorParametrization

        torch.manual_seed(3)
        module = LoRAExpertTensorParametrization(
            adapter_name="experts.w1",
            shape=(3, 5, 6),
            layout=("expert", "out", "in"),
            rank=8,
            alpha=8.0,
        )
        with torch.no_grad():
            module.lora_b.normal_(0.0, 0.05)
        mask = torch.zeros(8)
        mask[:4] = 1.0
        module.set_timestep_rank_masks(None, mask)
        base = torch.randn(3, 5, 6)
        out = module(base)
        ref = base + torch.matmul(module.lora_b[:, :, :4], module.lora_a[:, :4])
        torch.testing.assert_close(out, ref)
        out.sum().backward()
        self.assertTrue(bool((module.lora_a.grad[:, 4:] == 0).all().item()))
        self.assertTrue(bool((module.lora_b.grad[:, :, 4:] == 0).all().item()))


@unittest.skipIf(torch is None, "torch not installed")
class LoraInitTests(unittest.TestCase):
    def test_orthogonal_init_linear(self) -> None:
        from mirai.core.models.adapters.lora import LoRALinear

        torch.manual_seed(5)
        module = LoRALinear(nn.Linear(64, 32), rank=8, alpha=8.0, init="orthogonal")
        gram = module.lora_a @ module.lora_a.t()
        torch.testing.assert_close(gram, torch.eye(8), atol=1e-5, rtol=1e-5)
        self.assertTrue(bool((module.lora_b == 0).all().item()))

    def test_orthogonal_init_expert_stack(self) -> None:
        from mirai.core.models.compressed_weights.execution.experts import ActiveExpertLoRA

        adapter = ActiveExpertLoRA(
            adapter_name="w1",
            num_experts=3,
            in_features=64,
            out_features=32,
            rank=8,
            alpha=8.0,
            init="orthogonal",
        )
        for expert_idx in range(3):
            gram = adapter.lora_a[expert_idx] @ adapter.lora_a[expert_idx].t()
            torch.testing.assert_close(gram, torch.eye(8), atol=1e-5, rtol=1e-5)

    def test_kaiming_default_unchanged(self) -> None:
        from mirai.core.models.adapters.lora import LoRALinear

        torch.manual_seed(9)
        default = LoRALinear(nn.Linear(16, 8), rank=4, alpha=4.0)
        torch.manual_seed(9)
        explicit = LoRALinear(nn.Linear(16, 8), rank=4, alpha=4.0, init="kaiming")
        torch.testing.assert_close(default.lora_a, explicit.lora_a)

    def test_pissa_preserves_initial_base_function(self) -> None:
        from mirai.core.models.adapters.lora import LoRALinear

        torch.manual_seed(12)
        base = nn.Linear(7, 5, bias=False)
        original_weight = base.weight.detach().clone()
        module = LoRALinear(base, rank=3, alpha=3.0, init="pissa")
        reconstructed = module.base.weight.detach() + (
            module.lora_b.detach() @ module.lora_a.detach()
        )
        torch.testing.assert_close(reconstructed, original_weight)
        self.assertGreater(float(module.lora_b.norm().item()), 0.0)

    def test_lora_ga_preserves_initial_base_function(self) -> None:
        from mirai.core.models.adapters.lora import LoRALinear

        torch.manual_seed(13)
        base = nn.Linear(8, 8, bias=False)
        original_weight = base.weight.detach().clone()
        module = LoRALinear(base, rank=2, alpha=2.0, init="lora_ga")
        module.initialize_from_full_gradient(
            torch.randn_like(original_weight),
            stable_gamma=16.0,
        )
        reconstructed = module.base.weight.detach() + (
            module.lora_b.detach() @ module.lora_a.detach()
        )
        torch.testing.assert_close(reconstructed, original_weight)

    def test_invalid_init_rejected(self) -> None:
        from mirai.core.models.adapters.lora import LoRALinear

        with self.assertRaises(ValueError):
            LoRALinear(nn.Linear(4, 4), rank=2, alpha=2.0, init="xavier")


@unittest.skipIf(torch is None, "torch not installed")
class MaskDistributionTests(unittest.TestCase):
    def test_set_masks_reaches_all_adapter_kinds(self) -> None:
        from mirai.core.models.compressed_weights.execution.experts import ActiveExpertLoRA
        from mirai.core.models.adapters.lora import (
            LoRALinear,
            collect_lora_adapter_modules,
            set_lora_timestep_rank_masks,
        )

        root = nn.Module()
        root.linear = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2.0)
        root.expert = ActiveExpertLoRA(
            adapter_name="w1",
            num_experts=2,
            in_features=4,
            out_features=4,
            rank=2,
            alpha=2.0,
        )
        modules = collect_lora_adapter_modules(root)
        self.assertEqual(len(modules), 2)
        mask = torch.ones(1, 2)
        uniform = torch.ones(2)
        set_lora_timestep_rank_masks(modules, mask, uniform)
        self.assertIs(root.linear._timestep_mask_uniform, uniform)
        self.assertIs(root.expert._timestep_mask_uniform, uniform)
        set_lora_timestep_rank_masks(root, None, None)
        self.assertIsNone(root.linear._timestep_mask_uniform)
        self.assertIsNone(root.expert._timestep_mask_uniform)


@unittest.skipIf(torch is None, "torch not installed")
class LingbotPipelineSyncTests(unittest.TestCase):
    """Wiring tests for the pipeline-side mask sync (no full model needed)."""

    def _pipeline_stub(self, *, policy, flow_shift: float = 3.0, rank: int = 8):
        from mirai.config.schema import ModelConfig, ModelParams
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
        from mirai.core.models.adapters.lora import LoRAApplicationReport, LoRALinear

        pipe = LingBotVideoPipeline.__new__(LingBotVideoPipeline)
        pipe.model_config = ModelConfig(params=ModelParams(flow_shift=flow_shift))
        pipe._timestep_adapter_policy = policy
        pipe._lora_report = LoRAApplicationReport(
            target_preset="attn_only",
            target_modules=("to_q",),
            rank=rank,
            alpha=float(rank),
            matched_modules=("m",),
        )
        module = LoRALinear(nn.Linear(4, 4), rank=rank, alpha=float(rank))
        pipe._timestep_adapter_modules = [module]
        return pipe, module

    def test_noop_policy_returns_none_and_leaves_modules_untouched(self) -> None:
        pipe, module = self._pipeline_stub(policy=None)
        stats = pipe._sync_timestep_adapter_masks(torch.tensor([0.5]))
        self.assertIsNone(stats)
        self.assertIsNone(module._timestep_mask_per_sample)

    def test_sigma_is_post_shift(self) -> None:
        from mirai.core.models.flow import shifted_sigma

        policy = TimestepAdapterPolicy(schedule="tlora", min_fraction=0.25)
        pipe, module = self._pipeline_stub(policy=policy, flow_shift=3.0)
        t = torch.tensor([0.25, 0.75])
        stats = pipe._sync_timestep_adapter_masks(t)
        expected = shifted_sigma(t, 3.0)
        for got, want in zip(stats["sigma"], expected.tolist(), strict=True):
            self.assertAlmostEqual(got, want, places=5)
        self.assertIsNotNone(module._timestep_mask_per_sample)
        self.assertEqual(tuple(module._timestep_mask_per_sample.shape), (2, 8))

    def test_band_stats_and_uniform_gate(self) -> None:
        policy = TimestepAdapterPolicy(band_min=0.875, band_max=1.0)
        pipe, module = self._pipeline_stub(policy=policy, flow_shift=3.0)
        # flow_shift=3: t=0.9 -> sigma ~0.964 (in band), t=0.2 -> ~0.43 (out).
        stats = pipe._sync_timestep_adapter_masks(torch.tensor([0.9, 0.2]))
        self.assertEqual(stats["band_gate"], [1.0, 0.0])
        # Mixed batch -> conservative uniform mask fully gated.
        self.assertEqual(stats["uniform_active_rank"], 0)
        self.assertEqual(
            float(module._timestep_mask_uniform.abs().sum().item()), 0.0
        )


@unittest.skipIf(torch is None, "torch not installed")
class RouterAuxTermCollectionTests(unittest.TestCase):
    def test_weighted_losses_match_legacy_and_raw_available_at_zero_weight(
        self,
    ) -> None:
        from mirai.core.models.lingbot_video.pipeline import (
            _collect_router_auxiliary_terms,
            _weighted_router_auxiliary_losses,
        )
        from mirai.vendors.lingbot_video.transformer_lingbot_video import (
            LingBotVideoSparseMoeBlock,
        )

        model = nn.Module()
        block = nn.Module()
        block.ffn = LingBotVideoSparseMoeBlock(
            hidden_size=4,
            intermediate_size=8,
            num_experts=2,
            top_k=1,
            moe_intermediate_size=8,
            score_func="sigmoid",
            norm_topk_prob=True,
            n_group=None,
            topk_group=None,
            routed_scaling_factor=1.0,
            n_shared_experts=0,
        )
        model.blocks = nn.ModuleList([block])
        balance = torch.tensor(2.0, requires_grad=True)
        z_term = torch.tensor(3.0, requires_grad=True)
        model._mirai_checkpoint_router_auxiliary_terms = [(balance, z_term)]
        balance_terms, z_terms, z_weights = _collect_router_auxiliary_terms(model)
        self.assertEqual(len(balance_terms), 1)
        # Weight 0 -> no loss terms (loss graph unchanged) ...
        losses = _weighted_router_auxiliary_losses(
            balance_terms,
            z_terms,
            load_balance_weight=0.0,
            z_loss_weight=0.0,
            z_loss_weights=z_weights,
        )
        self.assertEqual(losses, {})
        # ... but the raw values are still collectable for telemetry.
        self.assertAlmostEqual(
            float(torch.stack(balance_terms).mean().item()), 2.0
        )
        self.assertAlmostEqual(float(torch.stack(z_terms).mean().item()), 3.0)
        # Nonzero weights reproduce the reference weighted terms.
        weighted = _weighted_router_auxiliary_losses(
            balance_terms,
            z_terms,
            load_balance_weight=0.01,
            z_loss_weight=0.0001,
            z_loss_weights=z_weights,
        )
        self.assertAlmostEqual(float(weighted["moe_load_balance"].item()), 0.02)
        self.assertAlmostEqual(float(weighted["moe_router_z"].item()), 0.0003)


class SchemaTimestepKeysTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = TrainingConfig.from_dict({})
        self.assertEqual(cfg.adapter.timestep_rank_schedule, "none")
        self.assertAlmostEqual(cfg.adapter.timestep_rank_min_fraction, 0.25)
        self.assertEqual(cfg.adapter.lora_init, "kaiming")
        self.assertAlmostEqual(cfg.adapter.timestep_band_min, 0.0)
        self.assertAlmostEqual(cfg.adapter.timestep_band_max, 1.0)

    def test_parse_values(self) -> None:
        cfg = TrainingConfig.from_dict(
            {
                "adapter": {
                    "timestep_rank_schedule": "tlora",
                    "timestep_rank_min_fraction": 0.5,
                    "lora_init": "orthogonal",
                    "timestep_band_min": 0.875,
                    "timestep_band_max": 1.0,
                }
            }
        )
        self.assertEqual(cfg.adapter.timestep_rank_schedule, "tlora")
        self.assertAlmostEqual(cfg.adapter.timestep_rank_min_fraction, 0.5)
        self.assertEqual(cfg.adapter.lora_init, "orthogonal")
        self.assertAlmostEqual(cfg.adapter.timestep_band_min, 0.875)

    def test_invalid_schedule_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            TrainingConfig.from_dict(
                {"adapter": {"timestep_rank_schedule": "cosine"}}
            )

    def test_invalid_min_fraction_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            TrainingConfig.from_dict(
                {"adapter": {"timestep_rank_min_fraction": 0.0}}
            )
        with self.assertRaises(ConfigError):
            TrainingConfig.from_dict(
                {"adapter": {"timestep_rank_min_fraction": 1.5}}
            )

    def test_invalid_band_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            TrainingConfig.from_dict(
                {"adapter": {"timestep_band_min": 0.5, "timestep_band_max": 0.5}}
            )
        with self.assertRaises(ConfigError):
            TrainingConfig.from_dict({"adapter": {"timestep_band_max": 1.5}})
        with self.assertRaises(ConfigError):
            TrainingConfig.from_dict({"adapter": {"timestep_band_min": -0.1}})

    def test_initializer_name_is_resolved_by_registry_at_model_boundary(self) -> None:
        cfg = TrainingConfig.from_dict({"adapter": {"lora_init": "xavier"}})
        self.assertEqual(cfg.adapter.lora_init, "xavier")

    def test_runtime_contract_checks(self) -> None:
        from mirai.core.training.runtime.contract import (
            validate_training_runtime_config,
        )

        cfg = TrainingConfig.from_dict({})
        cfg.adapter.timestep_band_min = 0.9
        cfg.adapter.timestep_band_max = 0.5
        with self.assertRaises(Exception):
            validate_training_runtime_config(cfg)


class StepMetricsAuxTelemetryTests(unittest.TestCase):
    def test_aux_values_lifted_from_diagnostics(self) -> None:
        from mirai.core.training.observability.metrics import build_step_metrics

        cfg = TrainingConfig.from_dict({})
        metrics = build_step_metrics(
            config=cfg,
            last_metrics={
                "loss": 1.0,
                "diagnostics": {
                    "moe_balance_loss": 0.5,
                    "moe_z_loss": 2.25,
                    "moe_routing": {},
                },
            },
            lr=1e-4,
            grad_norm=0.1,
            skipped_steps=0,
            vram_used_mb=0.0,
        )
        self.assertAlmostEqual(metrics["moe_balance_loss"], 0.5)
        self.assertAlmostEqual(metrics["moe_z_loss"], 2.25)

    def test_timestep_adapter_summary_lifted(self) -> None:
        from mirai.core.training.observability.metrics import build_step_metrics

        cfg = TrainingConfig.from_dict({})
        metrics = build_step_metrics(
            config=cfg,
            last_metrics={
                "loss": 1.0,
                "diagnostics": {
                    "timestep_adapter": {
                        "sigma": [0.9, 0.5],
                        "active_rank_fraction": [0.25, 1.0],
                        "band_gate": [1.0, 0.0],
                        "uniform_active_rank": 8,
                    }
                },
            },
            lr=1e-4,
            grad_norm=0.1,
            skipped_steps=0,
            vram_used_mb=0.0,
        )
        self.assertAlmostEqual(metrics["timestep_sigma_mean"], 0.7)
        self.assertAlmostEqual(metrics["timestep_active_rank_fraction"], 0.625)
        self.assertAlmostEqual(metrics["timestep_band_gate_fraction"], 0.5)

    def test_keys_absent_without_diagnostics(self) -> None:
        from mirai.core.training.observability.metrics import build_step_metrics

        cfg = TrainingConfig.from_dict({})
        metrics = build_step_metrics(
            config=cfg,
            last_metrics={"loss": 1.0},
            lr=1e-4,
            grad_norm=0.1,
            skipped_steps=0,
            vram_used_mb=0.0,
        )
        self.assertNotIn("moe_balance_loss", metrics)
        self.assertNotIn("moe_z_loss", metrics)


@unittest.skipIf(torch is None, "torch is required")
class SparseDeltaTests(unittest.TestCase):
    def test_sparse_delta_support_is_persistent_and_only_values_train(self) -> None:
        torch.manual_seed(5)
        base = nn.Linear(8, 6, bias=False)
        adapter = SparseDeltaLinear(base, density=0.25, selection="magnitude")
        inputs = torch.randn(3, 8)
        torch.testing.assert_close(adapter(inputs), base(inputs))
        adapter.values.data.fill_(0.1)
        loss = adapter(inputs).square().mean()
        loss.backward()
        self.assertIsNotNone(adapter.values.grad)
        self.assertIsNone(base.weight.grad)
        self.assertEqual(adapter.indices.numel(), 12)
        restored = SparseDeltaLinear(
            nn.Linear(8, 6, bias=False),
            density=0.25,
            selection="magnitude",
        )
        restored.load_state_dict(adapter.state_dict())
        torch.testing.assert_close(restored.indices, adapter.indices)
        torch.testing.assert_close(restored.values, adapter.values)


if __name__ == "__main__":
    unittest.main()
