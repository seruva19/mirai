"""Global-batch MoE load-balancing contracts.

Behavioral invariants:
* accumulation-window == 1 (single micro-batch) is bit-exact to microbatch scope;
* the global-batch aux load term blends the whole window on the 2nd micro-batch
  where microbatch scope sees only the local batch (documented direction);
* the accumulator's fraction is exactly the window's global load;
* bias_only composes: the online bias update uses the accumulated window counts;
* defaults (scope='microbatch') are byte-identical.
"""

from __future__ import annotations

# Colocated behavioral contract for accumulation-wide routing balance.

import unittest

from mirai.config.schema import TrainingConfig, all_config_keys
from mirai.core.moe.adaptation.global_balance import (
    GlobalBatchLoadAccumulator,
    dispatch_counts,
    normalize_moe_balance_scope,
)
from mirai.core.training.runtime.contract import validate_training_runtime_config

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class ScopeResolverTests(unittest.TestCase):
    def test_normalize_aliases_and_default(self) -> None:
        self.assertEqual(normalize_moe_balance_scope(None), "microbatch")
        self.assertEqual(normalize_moe_balance_scope(""), "microbatch")
        self.assertEqual(normalize_moe_balance_scope("global"), "global_batch")
        self.assertEqual(normalize_moe_balance_scope("GLOBAL_BATCH"), "global_batch")

    def test_reject_unknown(self) -> None:
        with self.assertRaises(ValueError):
            normalize_moe_balance_scope("per_token")


@unittest.skipIf(torch is None, "torch not installed")
class AccumulatorTests(unittest.TestCase):
    def test_dispatch_counts_matches_bincount(self) -> None:
        idx = torch.tensor([[0, 1], [0, 2]])
        counts = dispatch_counts(idx, 4)
        self.assertEqual(counts.tolist(), [2.0, 1.0, 1.0, 0.0])

    def test_single_window_fraction_equals_local(self) -> None:
        # Accumulation == 1: the accumulated fraction is the local fraction, the
        # exact quantity the per-micro-batch aux load term uses.
        acc = GlobalBatchLoadAccumulator()
        idx = torch.tensor([[0, 1], [0, 2]])
        counts = dispatch_counts(idx, 4)
        acc.accumulate("L0", counts)
        local = counts / float(max(1, idx.reshape(-1).numel()))
        self.assertTrue(torch.equal(acc.fraction("L0"), local))

    def test_fraction_is_window_global_load(self) -> None:
        acc = GlobalBatchLoadAccumulator()
        acc.accumulate("L0", dispatch_counts(torch.tensor([0, 0]), 2))  # -> [2,0]
        acc.accumulate("L0", dispatch_counts(torch.tensor([1, 1]), 2))  # -> [0,2]
        # Window load blends both micro-batches: 4 selections, 2 per expert.
        self.assertEqual(acc.fraction("L0").tolist(), [0.5, 0.5])

    def test_reset_clears_window(self) -> None:
        acc = GlobalBatchLoadAccumulator()
        acc.accumulate("L0", dispatch_counts(torch.tensor([0]), 2))
        acc.reset()
        self.assertIsNone(acc.fraction("L0"))
        self.assertFalse(acc.has("L0"))


class ScopeValidationTests(unittest.TestCase):
    def _cfg(self, **params) -> TrainingConfig:
        return TrainingConfig.from_dict({"model": {"params": params}})

    def test_default_is_microbatch(self) -> None:
        self.assertEqual(TrainingConfig().model.params.moe_balance_scope, "microbatch")
        validate_training_runtime_config(self._cfg())

    def test_global_batch_accepted(self) -> None:
        validate_training_runtime_config(self._cfg(moe_balance_scope="global_batch"))

    def test_bad_scope_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_training_runtime_config(self._cfg(moe_balance_scope="per_token"))

    def test_key_registered(self) -> None:
        self.assertIn("moe_balance_scope", all_config_keys()["model.params"])

    def test_from_dict_round_trip(self) -> None:
        cfg = TrainingConfig.from_dict(
            {"model": {"params": {"moe_balance_scope": "global_batch"}}}
        )
        self.assertEqual(cfg.model.params.moe_balance_scope, "global_batch")


@unittest.skipIf(torch is None, "torch not installed")
class PipelineAuxScopeTests(unittest.TestCase):
    """Global-batch token-fraction aux vs the per-micro-batch estimate."""

    def _pipeline(self, *, scope: str):
        from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams
        from mirai.core.builtins import register_builtin_components
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

        register_builtin_components()
        torch.manual_seed(0)
        mc = ModelConfig(
            type="lingbot-video",
            path="./models/lingbot_video",
            params=ModelParams(
                variant="tiny-video", latent_channels=2, num_experts=8,
                experts_per_token=2, shared_experts=1, hidden_size=16, num_layers=2,
                attention_heads=2, patch_size=1,
                moe_aux_loss_type="global", moe_balance_scope=scope,
            ),
        )
        p = LingBotVideoPipeline(mc)
        p.set_adapter_config(
            AdapterConfig(type="lora", target_preset="attn_routed_experts", rank=4, alpha=4.0)
        )
        p.train()
        return p

    def _balance(self, p, latents) -> float:
        torch.manual_seed(123)
        text = {"lingbot": torch.randn(1, 3, 16)}
        p.forward(latents, torch.rand(1), text)
        return float(p.get_training_diagnostics()["moe_balance_loss"])

    def test_first_microbatch_is_bit_exact_across_scope(self) -> None:
        # The accumulation-window == 1 invariant: one micro-batch, the global
        # scope must reproduce the microbatch aux term exactly.
        torch.manual_seed(7)
        x1 = torch.randn(1, 2, 4, 8, 8)
        pg = self._pipeline(scope="global_batch")
        pm = self._pipeline(scope="microbatch")
        self.assertEqual(self._balance(pg, x1), self._balance(pm, x1))

    def test_optimizer_step_resets_window(self) -> None:
        torch.manual_seed(7)
        x1 = torch.randn(1, 2, 4, 8, 8)
        pg = self._pipeline(scope="global_batch")
        first = self._balance(pg, x1)
        self._balance(pg, x1)  # window now holds two copies
        pg.finish_optimizer_step()  # closes the window
        # After reset, a lone micro-batch again equals the single-batch value.
        self.assertAlmostEqual(self._balance(pg, x1), first, places=6)


@unittest.skipIf(torch is None, "torch not installed")
class VendoredGlobalAuxInjectionTests(unittest.TestCase):
    """The `global` aux load term consumes the injected accumulated fraction.

    Deterministic (untrained tiny routers collapse routing, so the difference is
    crafted directly on the router state a forward would produce).
    """

    def _block(self):
        from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams
        from mirai.core.builtins import register_builtin_components
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

        register_builtin_components()
        torch.manual_seed(0)
        mc = ModelConfig(
            type="lingbot-video", path="./m",
            params=ModelParams(
                variant="tiny-video", latent_channels=2, num_experts=4,
                experts_per_token=2, shared_experts=1, hidden_size=16, num_layers=1,
                attention_heads=2, patch_size=1, moe_aux_loss_type="global",
            ),
        )
        p = LingBotVideoPipeline(mc)
        p.set_adapter_config(
            AdapterConfig(type="lora", target_preset="attn_routed_experts", rank=4, alpha=4.0)
        )
        return p.transformer.blocks[0]

    def test_injected_fraction_changes_balance_in_documented_direction(self) -> None:
        from mirai.vendors.lingbot_video.transformer_lingbot_video import (
            _block_router_auxiliary_terms,
        )

        block = self._block()
        block._mirai_moe_aux_loss_type = "global"
        router = block.ffn.router
        experts = int(router.num_experts)  # 4
        # Two tokens routed to experts {0,1}; differentiable router scores.
        router.training_top_indices = torch.tensor([[0, 1], [0, 1]])
        logits = torch.randn(2, experts, requires_grad=True)
        router.training_logits = logits
        router.training_scores = torch.softmax(logits, dim=-1)

        # Microbatch scope (no injection): local load concentrates on {0,1}.
        router._mirai_global_batch_load_fraction = None
        balance_local, _ = _block_router_auxiliary_terms(block, like=logits)

        # Global scope: window-accumulated load is uniform across experts.
        router._mirai_global_batch_load_fraction = torch.full((experts,), 1.0 / experts)
        balance_global, _ = _block_router_auxiliary_terms(block, like=logits)

        self.assertFalse(
            torch.isclose(balance_local, balance_global, atol=1e-6).item()
        )
        # Documented: uniform window load -> balance = sum(mean prob) = 1.0; the
        # concentrated local load overweights the {0,1} probability mass.
        self.assertAlmostEqual(float(balance_global.detach()), 1.0, places=5)
        pmean = router.training_scores.detach().float().mean(dim=0)
        expected_local = experts * float((pmean[0] + pmean[1]) * 0.5)
        self.assertAlmostEqual(float(balance_local.detach()), expected_local, places=5)

    def test_injection_is_gradient_safe(self) -> None:
        from mirai.vendors.lingbot_video.transformer_lingbot_video import (
            _block_router_auxiliary_terms,
        )

        block = self._block()
        block._mirai_moe_aux_loss_type = "global"
        router = block.ffn.router
        experts = int(router.num_experts)
        router.training_top_indices = torch.tensor([[0, 1], [2, 3]])
        logits = torch.randn(2, experts, requires_grad=True)
        router.training_logits = logits
        router.training_scores = torch.softmax(logits, dim=-1)
        router._mirai_global_batch_load_fraction = torch.full((experts,), 1.0 / experts)
        balance, _ = _block_router_auxiliary_terms(block, like=logits)
        balance.backward()
        # The injected load is a detached constant weight; gradient still flows
        # through the differentiable mean-probability factor to the logits.
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_sequence_form_consumes_global_window_frequency(self) -> None:
        from mirai.vendors.lingbot_video.transformer_lingbot_video import (
            _block_router_auxiliary_terms,
        )

        block = self._block()
        block._mirai_moe_aux_loss_type = "sequence"
        router = block.ffn.router
        experts = int(router.num_experts)
        router.training_batch_size = 2
        router.training_tokens_per_sample = 2
        router.training_top_indices = torch.tensor(
            [[0, 1], [0, 1], [2, 3], [2, 3]]
        )
        router.training_unbiased_top_indices = (
            router.training_top_indices.clone()
        )
        logits = torch.randn(4, experts, requires_grad=True)
        router.training_logits = logits
        router.training_scores = torch.softmax(logits, dim=-1)

        router._mirai_global_batch_load_fraction = None
        sequence_local, _ = _block_router_auxiliary_terms(
            block, like=logits
        )
        injected = torch.tensor([0.7, 0.1, 0.1, 0.1])
        router._mirai_global_batch_load_fraction = injected
        sequence_global, _ = _block_router_auxiliary_terms(
            block, like=logits
        )

        expected = experts * torch.sum(
            injected * router.training_scores.float().mean(dim=0)
        )
        torch.testing.assert_close(sequence_global, expected)
        self.assertFalse(
            torch.isclose(
                sequence_local,
                sequence_global,
                atol=1.0e-6,
            ).item()
        )


@unittest.skipIf(torch is None, "torch not installed")
class PipelineBiasScopeTests(unittest.TestCase):
    """bias_only composes: the online bias update uses accumulated window loads."""

    def _pipeline(self):
        from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams
        from mirai.core.builtins import register_builtin_components
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

        register_builtin_components()
        torch.manual_seed(0)
        mc = ModelConfig(
            type="lingbot-video",
            path="./models/lingbot_video",
            params=ModelParams(
                variant="tiny-video", latent_channels=2, num_experts=8,
                experts_per_token=2, shared_experts=1, hidden_size=16, num_layers=2,
                attention_heads=2, patch_size=1,
                moe_balance_mode="bias_only", moe_bias_update_rate=0.1,
                moe_bias_centering=True, moe_balance_scope="global_batch",
            ),
        )
        p = LingBotVideoPipeline(mc)
        p.set_adapter_config(
            AdapterConfig(type="lora", target_preset="attn_routed_experts", rank=4, alpha=4.0)
        )
        p.train()
        return p

    def test_bias_update_uses_accumulated_window_counts(self) -> None:
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoRouter

        p = self._pipeline()
        text = {"lingbot": torch.randn(1, 3, 16)}
        for seed in (11, 22):
            torch.manual_seed(seed)
            p.forward(torch.randn(1, 2, 4, 8, 8), torch.rand(1), text)
        # Snapshot the accumulated per-router loads and pre-update bias.
        pending = {k: v.clone() for k, v in p._pending_router_loads.items()}
        self.assertTrue(pending, "expected accumulated router loads over the window")
        routers = {n: m for n, m in p.transformer.named_modules()
                   if isinstance(m, LingBotVideoRouter)}
        before = {n: routers[n].e_score_correction_bias.detach().clone() for n in pending}
        p.finish_optimizer_step()
        rate = 0.1
        for name, counts in pending.items():
            deviation = counts - counts.mean()
            expected = before[name].float().cpu() - rate * deviation.sign().float()
            expected = expected - expected.mean()  # centering
            actual = routers[name].e_score_correction_bias.detach().float().cpu()
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6), name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
