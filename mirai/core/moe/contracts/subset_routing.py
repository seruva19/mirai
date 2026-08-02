"""Stochastic expert-subset routing, condenser LoRA, freeze-router, metrics."""

from __future__ import annotations

# Colocated behavioral contract for expert working-set routing.

import copy
import unittest

from mirai.config.schema import ConfigError, TrainingConfig
from mirai.core.moe.routing.subset import (
    RouterSubsetPolicy,
    routing_mass_from_selection,
    select_expert_subset,
    subset_mask_from_ids,
    subset_reverse_kl,
    validate_subset_params,
)

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


_BASE_CONFIG = {
    "model": {
        "type": "lingbot-video",
        "params": {"num_experts": 8, "experts_per_token": 2},
    },
    "adapter": {"type": "lora", "target_preset": "routed_experts_only", "rank": 8, "alpha": 8.0},
    "dataset": {"path": "./x", "cache_path": "./x/c.pt"},
}


class SubsetPolicyMathTests(unittest.TestCase):
    def test_validation_bounds(self) -> None:
        with self.assertRaises(ValueError):
            validate_subset_params(fraction=0.0, pool_factor=2.0, anneal_steps=0, kl_weight=0.0)
        with self.assertRaises(ValueError):
            validate_subset_params(fraction=1.5, pool_factor=2.0, anneal_steps=0, kl_weight=0.0)
        with self.assertRaises(ValueError):
            validate_subset_params(fraction=0.5, pool_factor=0.5, anneal_steps=0, kl_weight=0.0)
        with self.assertRaises(ValueError):
            validate_subset_params(fraction=0.5, pool_factor=2.0, anneal_steps=-1, kl_weight=0.0)
        with self.assertRaises(ValueError):
            validate_subset_params(fraction=0.5, pool_factor=2.0, anneal_steps=0, kl_weight=-0.1)

    def test_noop_at_full_fraction(self) -> None:
        self.assertTrue(RouterSubsetPolicy(fraction=1.0).is_noop())
        self.assertFalse(RouterSubsetPolicy(fraction=0.5).is_noop())

    def test_from_model_params_noop_default(self) -> None:
        class _P:
            expert_subset_fraction = 1.0
            expert_subset_pool_factor = 2.0
            expert_subset_anneal_steps = 0
            expert_subset_router_kl_weight = 0.0

        self.assertIsNone(RouterSubsetPolicy.from_model_params(_P()))

    def test_effective_fraction_anneal(self) -> None:
        p = RouterSubsetPolicy(fraction=0.5, anneal_steps=10)
        self.assertAlmostEqual(p.effective_fraction(0), 0.5)
        self.assertAlmostEqual(p.effective_fraction(5), 0.75)
        self.assertAlmostEqual(p.effective_fraction(10), 1.0)
        self.assertAlmostEqual(p.effective_fraction(100), 1.0)

    def test_effective_fraction_constant_when_no_anneal(self) -> None:
        p = RouterSubsetPolicy(fraction=0.5, anneal_steps=0)
        self.assertAlmostEqual(p.effective_fraction(0), 0.5)
        self.assertAlmostEqual(p.effective_fraction(1000), 0.5)

    def test_subset_size_clamped_to_topk(self) -> None:
        p = RouterSubsetPolicy(fraction=0.1)
        # ceil(0.1*8)=1 but top_k=2 forces at least 2.
        self.assertEqual(p.subset_size(num_experts=8, top_k=2, step=0), 2)

    def test_subset_size_exact(self) -> None:
        p = RouterSubsetPolicy(fraction=0.5)
        self.assertEqual(p.subset_size(num_experts=8, top_k=2, step=0), 4)

    def test_subset_inactive_at_anneal_end(self) -> None:
        p = RouterSubsetPolicy(fraction=0.5, anneal_steps=10)
        self.assertTrue(p.is_active(num_experts=8, top_k=2, step=0))
        self.assertFalse(p.is_active(num_experts=8, top_k=2, step=10))


@unittest.skipIf(torch is None, "torch not installed")
class SubsetSelectionTests(unittest.TestCase):
    def _mass(self) -> "torch.Tensor":
        return torch.tensor([10.0, 1.0, 5.0, 0.1, 8.0, 3.0, 0.5, 2.0])

    def test_deterministic_given_seed(self) -> None:
        g1 = torch.Generator(); g1.manual_seed(123)
        g2 = torch.Generator(); g2.manual_seed(123)
        s1 = select_expert_subset(self._mass(), size=4, pool_factor=2.0, generator=g1)
        s2 = select_expert_subset(self._mass(), size=4, pool_factor=2.0, generator=g2)
        self.assertTrue(torch.equal(s1, s2))

    def test_exact_size_and_distinct(self) -> None:
        g = torch.Generator(); g.manual_seed(7)
        s = select_expert_subset(self._mass(), size=4, pool_factor=2.0, generator=g)
        self.assertEqual(int(s.numel()), 4)
        self.assertEqual(len(set(s.tolist())), 4)

    def test_rejects_non_finite_or_empty_mass(self) -> None:
        for invalid in (
            torch.tensor([1.0, float("nan")]),
            torch.tensor([1.0, float("inf")]),
            torch.empty(0),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError, "at least one expert|must be finite"
                ):
                    select_expert_subset(
                        invalid,
                        size=1,
                        pool_factor=1.0,
                    )

    def test_pool_factor_one_is_deterministic_hottest(self) -> None:
        # pool_factor 1 => top-`size` hottest experts, no randomness.
        s = select_expert_subset(self._mass(), size=3, pool_factor=1.0)
        self.assertEqual(s.tolist(), [0, 2, 4])  # masses 10, 5, 8

    def test_hot_bias_includes_hottest(self) -> None:
        # Over many draws the hottest expert (idx 0) is almost always present.
        present = 0
        for seed in range(20):
            g = torch.Generator(); g.manual_seed(seed)
            s = select_expert_subset(self._mass(), size=4, pool_factor=2.0, generator=g)
            present += int(0 in s.tolist())
        self.assertGreaterEqual(present, 18)

    def test_mask_from_ids(self) -> None:
        mask = subset_mask_from_ids(8, torch.tensor([0, 2, 4, 5]))
        self.assertEqual(int(mask.sum()), 4)
        self.assertTrue(bool(mask[0]) and bool(mask[4]))
        self.assertFalse(bool(mask[1]))

    def test_routing_mass_from_selection(self) -> None:
        scores = torch.tensor([[0.6, 0.3, 0.1, 0.0], [0.0, 0.5, 0.5, 0.0]])
        top = torch.tensor([[0, 1], [1, 2]])
        mass = routing_mass_from_selection(scores, top, num_experts=4)
        self.assertAlmostEqual(float(mass[0]), 0.6, places=5)
        self.assertAlmostEqual(float(mass[1]), 0.8, places=5)
        self.assertAlmostEqual(float(mass[2]), 0.5, places=5)
        self.assertAlmostEqual(float(mass[3]), 0.0, places=5)

    def test_reverse_kl_finite_nonneg_differentiable(self) -> None:
        logits = torch.randn(50, 8, requires_grad=True)
        kl = subset_reverse_kl(logits, torch.tensor([0, 1, 2, 3]))
        self.assertTrue(bool(torch.isfinite(kl)))
        self.assertGreaterEqual(float(kl.detach()), 0.0)
        kl.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))


@unittest.skipIf(torch is None, "torch not installed")
class RouterMaskingTests(unittest.TestCase):
    def _router(self):
        from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoRouter

        torch.manual_seed(0)
        r = LingBotVideoRouter(16, 8, 2, "sigmoid", True, None, None, 1.0)
        torch.nn.init.normal_(r.weight, mean=0.0, std=0.02)
        r.train()
        return r

    def test_masked_experts_get_zero_tokens(self) -> None:
        r = self._router()
        tokens = torch.randn(300, 16)
        r.set_subset_runtime(active=True, size=4, pool_factor=2.0, seed=42, kl_weight=0.0)
        top_indices, *_ = r(tokens)
        subset = set(r.last_subset_ids.tolist())
        touched = set(torch.unique(top_indices).tolist())
        self.assertTrue(touched.issubset(subset))
        self.assertLessEqual(len(subset), 4)

    def test_determinism_given_seed(self) -> None:
        r = self._router()
        tokens = torch.randn(300, 16)
        r.set_subset_runtime(active=True, size=4, pool_factor=2.0, seed=42, kl_weight=0.0)
        ti1, *_ = r(tokens)
        r.set_subset_runtime(active=True, size=4, pool_factor=2.0, seed=42, kl_weight=0.0)
        ti2, *_ = r(tokens)
        self.assertTrue(torch.equal(ti1, ti2))

    def test_eval_mode_no_masking(self) -> None:
        r = self._router()
        tokens = torch.randn(300, 16)
        ti_base, *_ = r(tokens)
        r.eval()
        r.set_subset_runtime(active=True, size=4, pool_factor=2.0, seed=42, kl_weight=0.0)
        ti_eval, *_ = r(tokens)
        self.assertTrue(torch.equal(ti_base, ti_eval))

    def test_kl_gated_by_router_trainability(self) -> None:
        r = self._router()
        tokens = torch.randn(80, 16)
        r.weight.requires_grad_(False)
        r.set_subset_runtime(active=True, size=4, pool_factor=2.0, seed=1, kl_weight=0.5)
        r(tokens)
        self.assertIsNone(r.training_subset_kl)
        r.weight.requires_grad_(True)
        r.set_subset_runtime(active=True, size=4, pool_factor=2.0, seed=1, kl_weight=0.5)
        r(tokens)
        self.assertIsNotNone(r.training_subset_kl)
        # weight 0 disables it even when trainable.
        r.set_subset_runtime(active=True, size=4, pool_factor=2.0, seed=1, kl_weight=0.0)
        r(tokens)
        self.assertIsNone(r.training_subset_kl)


@unittest.skipIf(torch is None, "torch not installed")
class CondenserTests(unittest.TestCase):
    def _adapter(self, condenser_rank: int = 0):
        from mirai.core.models.compressed_weights import ActiveExpertLoRA

        torch.manual_seed(0)
        ad = ActiveExpertLoRA(
            adapter_name="w1", num_experts=6, in_features=8, out_features=10, rank=4, alpha=8.0
        )
        if condenser_rank > 0:
            ad.enable_condenser(rank=condenser_rank, alpha=float(condenser_rank))
            ad.cond_a.data.normal_()
            ad.cond_b.data.normal_()
        return ad

    def test_rank_zero_is_byte_identical(self) -> None:
        ad = self._adapter(condenser_rank=0)
        self.assertFalse(ad.has_condenser())
        x = torch.randn(6, 5, 8)
        out = ad.batched_forward(x)
        # lora_b == 0 at init -> exactly zero, no condenser term added.
        self.assertEqual(float(out.detach().abs().sum()), 0.0)

    def test_condenser_ungated_under_sieve(self) -> None:
        ad = self._adapter(condenser_rank=2)
        x = torch.randn(6, 5, 8)
        ad.set_active_experts([0])  # mask out all but expert 0
        out = ad.batched_forward(x)
        # lora_b == 0 so per-expert term is zero; the shared condenser must still
        # fire for every expert (including the masked ones).
        cond = torch.nn.functional.linear(
            torch.nn.functional.linear(x, ad.cond_a), ad.cond_b
        ) * (float(ad.condenser_alpha) / ad.condenser_rank)
        torch.testing.assert_close(out, cond, atol=1e-5, rtol=1e-4)
        self.assertGreater(float(out[3].detach().norm()), 0.0)  # masked, still nonzero

    def test_condenser_grads_flow(self) -> None:
        ad = self._adapter(condenser_rank=2)
        ad.train()
        x = torch.randn(6, 5, 8)
        ad.batched_forward(x).sum().backward()
        self.assertGreater(float(ad.cond_a.grad.norm()), 0.0)
        self.assertGreater(float(ad.cond_b.grad.norm()), 0.0)

    def test_condenser_state_dict_roundtrip(self) -> None:
        from mirai.core.models.adapters.lora import lora_state_dict

        # Build a small container with an expert adapter to exercise export/import.
        from mirai.core.models.compressed_weights import ActiveExpertLoRA

        class _Host(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.expert_lora = nn.ModuleDict()
                self.expert_lora["w1"] = ActiveExpertLoRA(
                    adapter_name="blocks.0.ffn.experts.w1",
                    num_experts=4, in_features=8, out_features=10, rank=4, alpha=8.0,
                )

        src = _Host()
        src.expert_lora["w1"].enable_condenser(rank=2, alpha=2.0)
        src.expert_lora["w1"].cond_a.data.normal_()
        src.expert_lora["w1"].cond_b.data.normal_()
        state = lora_state_dict(src)
        cond_keys = [k for k in state if "condenser" in k]
        self.assertEqual(len(cond_keys), 3)

        from mirai.core.models.adapters.lora import load_lora_state_dict

        dst = _Host()
        dst.expert_lora["w1"].enable_condenser(rank=2, alpha=2.0)
        load_lora_state_dict(dst, state)
        torch.testing.assert_close(
            dst.expert_lora["w1"].cond_a, src.expert_lora["w1"].cond_a
        )
        torch.testing.assert_close(
            dst.expert_lora["w1"].cond_b, src.expert_lora["w1"].cond_b
        )


@unittest.skipIf(torch is None, "torch not installed")
class PipelineIntegrationTests(unittest.TestCase):
    """End-to-end wiring: metrics surface, subset drives routing, defaults clean."""

    def _pipeline(self, *, subset_fraction=1.0, condenser_rank=0, preset="attn_routed_experts"):
        from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams
        from mirai.core.builtins import register_builtin_components
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

        register_builtin_components()
        mc = ModelConfig(
            type="lingbot-video",
            path="./models/lingbot_video",
            params=ModelParams(
                variant="tiny-video", latent_channels=2, num_experts=8,
                experts_per_token=2, shared_experts=1, hidden_size=16, num_layers=2,
                attention_heads=2, patch_size=1, expert_subset_fraction=subset_fraction,
            ),
        )
        p = LingBotVideoPipeline(mc)
        p.set_adapter_config(
            AdapterConfig(
                type="lora", target_preset=preset, rank=4, alpha=4.0,
                condenser_rank=condenser_rank, condenser_alpha=float(condenser_rank or 1),
            )
        )
        p.train()
        return p

    def _forward(self, p, step=0):
        torch.manual_seed(0)
        if p.supports_router_subset_progress():
            p.set_router_subset_progress(step=step, seed=42)
        out = p.forward(
            torch.randn(1, 2, 4, 8, 8), torch.rand(1), {"lingbot": torch.randn(1, 3, 16)}
        )
        return out, p.get_training_diagnostics()

    def test_defaults_no_subset_progress_support(self) -> None:
        p = self._pipeline(subset_fraction=1.0)
        self.assertFalse(p.supports_router_subset_progress())

    def test_stability_metrics_present(self) -> None:
        p = self._pipeline(subset_fraction=1.0)
        _, diag = self._forward(p)
        for key in (
            "moe_routing_entropy",
            "moe_utilization_cv",
            "moe_top1_monopoly",
            "moe_step_unique_experts",
        ):
            self.assertIn(key, diag)
        # kl_vs_step0 is absent on the first forward (snapshot just captured).
        self.assertNotIn("moe_routing_kl_vs_step0", diag)
        _, diag2 = self._forward(p, step=1)
        self.assertIn("moe_routing_kl_vs_step0", diag2)

    def test_subset_active_finite_and_bounded(self) -> None:
        p = self._pipeline(subset_fraction=0.5)
        self.assertTrue(p.supports_router_subset_progress())
        out, diag = self._forward(p)
        self.assertTrue(bool(torch.isfinite(out).all()))
        # working set never exceeds the subset cap (ceil(0.5*8)=4).
        self.assertLessEqual(diag["moe_step_unique_experts"], 4.0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "NF4 quantization requires CUDA",
    )
    def test_condenser_attaches_in_trainer_order(self) -> None:
        """Quantization preserves condenser attachment and serialized state."""
        from mirai.core.models.compressed_weights import ActiveExpertLoRA
        from mirai.core.models.adapters.lora import lora_state_dict

        p = self._pipeline(condenser_rank=2, preset="attn_router_routed_experts")
        # No ActiveExpertLoRA hosts yet (bf16 weight-space parametrizations).
        p.enable_quantized_frozen_weights("nf4")
        adapters = [
            m for m in p.transformer.modules() if isinstance(m, ActiveExpertLoRA)
        ]
        self.assertGreater(len(adapters), 0)
        self.assertTrue(all(m.has_condenser() for m in adapters))
        state = lora_state_dict(p.transformer)
        self.assertTrue(any("condenser_a" in k for k in state))


class ConfigPlumbingTests(unittest.TestCase):
    def _cfg(self, *, model_overrides=None, adapter_overrides=None) -> TrainingConfig:
        payload = copy.deepcopy(_BASE_CONFIG)
        if model_overrides:
            payload["model"]["params"].update(model_overrides)
        if adapter_overrides:
            payload["adapter"].update(adapter_overrides)
        return TrainingConfig.from_dict(payload)

    def test_defaults_disable_subset_and_condenser(self) -> None:
        cfg = self._cfg()
        self.assertEqual(cfg.model.params.expert_subset_fraction, 1.0)
        self.assertEqual(cfg.model.params.expert_subset_pool_factor, 2.0)
        self.assertEqual(cfg.model.params.expert_subset_anneal_steps, 0)
        self.assertEqual(cfg.model.params.expert_subset_router_kl_weight, 0.0)
        self.assertEqual(cfg.adapter.condenser_rank, 0)
        self.assertIsNone(cfg.adapter.train_router)

    def test_parse_subset_params(self) -> None:
        cfg = self._cfg(
            model_overrides={
                "expert_subset_fraction": 0.5,
                "expert_subset_pool_factor": 3.0,
                "expert_subset_anneal_steps": 50,
                "expert_subset_router_kl_weight": 0.1,
            }
        )
        self.assertEqual(cfg.model.params.expert_subset_fraction, 0.5)
        self.assertEqual(cfg.model.params.expert_subset_pool_factor, 3.0)
        self.assertEqual(cfg.model.params.expert_subset_anneal_steps, 50)
        self.assertEqual(cfg.model.params.expert_subset_router_kl_weight, 0.1)

    def test_subset_fraction_bounds(self) -> None:
        with self.assertRaises(ConfigError):
            self._cfg(model_overrides={"expert_subset_fraction": 0.0})
        with self.assertRaises(ConfigError):
            self._cfg(model_overrides={"expert_subset_fraction": 1.5})

    def test_pool_factor_bound(self) -> None:
        with self.assertRaises(ConfigError):
            self._cfg(model_overrides={"expert_subset_pool_factor": 0.5})

    def test_kl_weight_bound(self) -> None:
        with self.assertRaises(ConfigError):
            self._cfg(model_overrides={"expert_subset_router_kl_weight": -0.1})

    def test_parse_condenser_and_train_router(self) -> None:
        cfg = self._cfg(
            adapter_overrides={"condenser_rank": 2, "condenser_alpha": 2.0, "train_router": False}
        )
        self.assertEqual(cfg.adapter.condenser_rank, 2)
        self.assertEqual(cfg.adapter.condenser_alpha, 2.0)
        self.assertIs(cfg.adapter.train_router, False)

    def test_condenser_rank_bound(self) -> None:
        with self.assertRaises(ConfigError):
            self._cfg(adapter_overrides={"condenser_rank": -1})

    def test_unknown_subset_key_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            self._cfg(model_overrides={"expert_subset_bogus": 1})


if __name__ == "__main__":
    unittest.main()
