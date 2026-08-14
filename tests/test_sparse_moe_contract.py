from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import unittest

from mirai.config.runtime_policy import validate_cli_model_contract
from mirai.config.schema import (
    ModelConfig,
    ModelParams,
    StrategyConfig,
    TrainingConfig,
    TrainingSection,
)
from mirai.core.builtins import register_builtin_components
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.models.lingbot_video.routed_moe import LingBotRoutedTritonBackend
from mirai.core.models.adapters.lora import LoRAExpertTensorParametrization
from mirai.core.models.lingbot_video.router_runtime import _routing_stats
from mirai.core.models.providers import get_model_family_provider
from mirai.core.models.testbed import TinySparseMoEDenoiser
from mirai.core.moe.calibration.pruning import ExpertPruningRoutedOutputObserver
from mirai.core.moe.calibration.pruning import ExpertPruningSaliencyAccumulator
from mirai.core.moe.artifacts.catalog import get_open_sparse_moe_model_specs
from mirai.core.moe.monitoring.agreement import RoutingSelectionCapture
from mirai.core.moe.routing.layers import ExpertChoiceMoEFeedForward, SparseMoEFeedForward
from mirai.core.moe.routing.expert_choice import (
    resolve_capacity_schedule,
    summarize_expert_choice_coverage,
)
from mirai.core.moe.routing.decoupled import DecoupledRouterConditioner
from mirai.core.moe.routing.routers import (
    ExpertChoiceRouter,
    TokenChoiceRouter,
    route_expert_choice_logits,
)
from mirai.core.training.trainer import Trainer
from mirai.vendors.lingbot_video.transformer_lingbot_video import (
    LingBotVideoAttention,
    LingBotVideoRouter,
    LingBotVideoRuntimeOptions,
    LingBotVideoSparseMoeBlock,
    _moe_expert_backend,
    set_lingbot_video_runtime_options,
)

try:
    import torch
    from torch.nn.utils import parametrize
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class SparseMoEContractTests(unittest.TestCase):
    def test_lingbot_routed_backend_auto_falls_back_and_explicit_fails_on_cpu(self) -> None:
        block = LingBotVideoSparseMoeBlock(
            hidden_size=8, intermediate_size=16, num_experts=4, top_k=2,
            moe_intermediate_size=12, score_func="softmax", norm_topk_prob=True,
            n_group=None, topk_group=None, routed_scaling_factor=1.0,
            n_shared_experts=0,
        )
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.normal_(std=0.02)
        tokens = torch.randn(3, 8)
        scores = torch.softmax(torch.randn(3, 2), dim=-1)
        indices = torch.tensor([[0, 1], [2, 3], [0, 2]])
        block._mirai_moe_kernel_backend = LingBotRoutedTritonBackend("auto")
        automatic = block._run_selected_experts(tokens, scores, indices)
        block._mirai_moe_kernel_backend = None
        reference = block._run_selected_experts(tokens, scores, indices)
        torch.testing.assert_close(automatic, reference)
        block._mirai_moe_kernel_backend = LingBotRoutedTritonBackend("triton")
        with self.assertRaisesRegex(RuntimeError, "CUDA activation.*BF16"):
            block._run_selected_experts(tokens, scores, indices)

    def test_lingbot_routed_layout_preserves_zero_score_assignment_space(self) -> None:
        block = LingBotVideoSparseMoeBlock(
            hidden_size=8, intermediate_size=16, num_experts=4, top_k=2,
            moe_intermediate_size=12, score_func="softmax", norm_topk_prob=True,
            n_group=None, topk_group=None, routed_scaling_factor=1.0,
            n_shared_experts=0,
        )
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.normal_(std=0.02)
        tokens = torch.randn(3, 8)
        scores = torch.tensor([[1.0, 0.0], [0.3, 0.7], [0.0, 1.0]])
        indices = torch.tensor([[0, 1], [2, 3], [0, 2]])
        backend = LingBotRoutedTritonBackend("auto")
        block._mirai_moe_kernel_backend = backend
        automatic = block._run_selected_experts(tokens, scores, indices)
        block._mirai_moe_kernel_backend = None
        reference = block._run_selected_experts(tokens, scores, indices)
        torch.testing.assert_close(automatic, reference)

    def test_lingbot_late_observer_auto_falls_back_and_explicit_rejects(self) -> None:
        class Observer:
            is_enabled = True

            def capture_sorted(self, *args, **kwargs):
                pass

        block = LingBotVideoSparseMoeBlock(
            hidden_size=8, intermediate_size=16, num_experts=4, top_k=2,
            moe_intermediate_size=12, score_func="softmax", norm_topk_prob=True,
            n_group=None, topk_group=None, routed_scaling_factor=1.0,
            n_shared_experts=0,
        )
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.normal_(std=0.02)
        tokens = torch.randn(3, 8)
        scores = torch.softmax(torch.randn(3, 2), dim=-1)
        indices = torch.tensor([[0, 1], [2, 3], [0, 2]])
        block.set_expert_output_observer(Observer())
        auto = LingBotRoutedTritonBackend("auto")
        object.__setattr__(auto, "execute_direct", lambda *args: (_ for _ in ()).throw(
            AssertionError("late observer must prevent routed kernel execution")
        ))
        block._mirai_moe_kernel_backend = auto
        observed = block._run_selected_experts(tokens, scores, indices)
        block._mirai_moe_kernel_backend = None
        expected = block._run_selected_experts(tokens, scores, indices)
        torch.testing.assert_close(observed, expected)
        block._mirai_moe_kernel_backend = LingBotRoutedTritonBackend("triton")
        with self.assertRaisesRegex(RuntimeError, "observers"):
            block._run_selected_experts(tokens, scores, indices)

    @unittest.skipUnless(
        torch is not None
        and torch.cuda.is_available()
        and os.environ.get("MIRAI_REMOTE_GPU_TESTS") == "1",
        "configured remote CUDA validation required",
    )
    def test_lingbot_routed_triton_full_training_and_selected_expert_parity(self) -> None:
        def build():
            torch.manual_seed(71)
            block = LingBotVideoSparseMoeBlock(
                hidden_size=17, intermediate_size=32, num_experts=5, top_k=3,
                moe_intermediate_size=23, score_func="softmax", norm_topk_prob=True,
                n_group=None, topk_group=None, routed_scaling_factor=1.0,
                n_shared_experts=0,
            ).to(device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                for parameter in block.parameters():
                    parameter.normal_(std=0.02)
            return block

        def run(*, routed):
            block = build()
            block.train()
            if not routed:
                set_lingbot_video_runtime_options(
                    block, LingBotVideoRuntimeOptions(moe_expert_backend="loop")
                )
                self.assertEqual(_moe_expert_backend(block), "loop")
            block._mirai_moe_kernel_backend = (
                LingBotRoutedTritonBackend("triton") if routed else None
            )
            torch.manual_seed(72)
            tokens = torch.randn(
                7, 17, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            logits = torch.randn(7, 3, device="cuda", dtype=torch.float32, requires_grad=True)
            scores = torch.softmax(logits, dim=-1).to(torch.bfloat16)
            # Expert 4 is empty; expert 0 is deliberately dominant.
            indices = torch.tensor(
                [[0, 1, 2], [0, 2, 3], [0, 1, 3], [0, 2, 3],
                 [0, 1, 2], [0, 2, 3], [0, 1, 3]], device="cuda",
            )
            output = block._run_selected_experts(tokens, scores, indices)
            loss = output.float().square().mean()
            loss.backward()
            optimizer = torch.optim.SGD(
                [block.experts.w1, block.experts.w2, block.experts.w3], lr=1e-2
            )
            observed = (
                output.detach(), loss.detach(), tokens.grad.detach(), logits.grad.detach(),
                block.experts.w1.grad.detach(), block.experts.w2.grad.detach(),
                block.experts.w3.grad.detach(),
            )
            for tensor in observed:
                self.assertTrue(torch.isfinite(tensor).all())
            optimizer.step()
            stepped = tuple(
                value.detach().clone()
                for value in (block.experts.w1, block.experts.w2, block.experts.w3)
            )
            if routed:
                import io
                payload = io.BytesIO()
                torch.save(block.state_dict(), payload)
                payload.seek(0)
                restored = build()
                restored.load_state_dict(
                    torch.load(payload, map_location="cuda", weights_only=True)
                )
                for expected, actual in zip(stepped, (
                    restored.experts.w1, restored.experts.w2, restored.experts.w3
                ), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            return observed + stepped

        reference = run(routed=False)
        candidate = run(routed=True)
        for expected, actual in zip(reference, candidate, strict=True):
            torch.testing.assert_close(actual.float(), expected.float(), rtol=4e-2, atol=4e-2)
        self.assertTrue(torch.equal(candidate[4][4], torch.zeros_like(candidate[4][4])))

    @unittest.skipUnless(
        torch is not None
        and torch.cuda.is_available()
        and os.environ.get("MIRAI_REMOTE_GPU_TESTS") == "1",
        "configured remote CUDA validation required",
    )
    def test_lingbot_routed_triton_lora_and_dora_adapter_parity(self) -> None:
        def build(use_dora):
            torch.manual_seed(81)
            block = LingBotVideoSparseMoeBlock(
                hidden_size=17, intermediate_size=32, num_experts=4, top_k=2,
                moe_intermediate_size=23, score_func="softmax", norm_topk_prob=True,
                n_group=None, topk_group=None, routed_scaling_factor=1.0,
                n_shared_experts=0,
            ).to(device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                for parameter in block.parameters():
                    parameter.normal_(std=0.02)
            adapters = []
            for key in ("w1", "w2", "w3"):
                base = getattr(block.experts, key)
                adapter = LoRAExpertTensorParametrization(
                    adapter_name=key, shape=tuple(base.shape),
                    layout=("expert", "out", "in"), rank=3, alpha=3.0,
                    use_dora=use_dora, base_weight=base if use_dora else None,
                ).to(device="cuda", dtype=torch.bfloat16)
                if use_dora:
                    adapter.initialize_dora_magnitude(base)
                with torch.no_grad():
                    adapter.lora_b.normal_(std=0.03)
                parametrize.register_parametrization(block.experts, key, adapter)
                block.experts.parametrizations[key].original.requires_grad_(False)
                adapters.append(adapter)
            return block, adapters

        def run(use_dora, routed):
            block, adapters = build(use_dora)
            if not routed:
                set_lingbot_video_runtime_options(
                    block, LingBotVideoRuntimeOptions(moe_expert_backend="loop")
                )
                self.assertEqual(_moe_expert_backend(block), "loop")
            block._mirai_moe_kernel_backend = (
                LingBotRoutedTritonBackend("triton") if routed else None
            )
            torch.manual_seed(82)
            tokens = torch.randn(6, 17, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            logits = torch.randn(6, 2, device="cuda", requires_grad=True)
            scores = torch.softmax(logits, -1).to(torch.bfloat16)
            indices = torch.tensor([[0, 1], [0, 2], [3, 0], [1, 3], [0, 2], [3, 1]], device="cuda")
            output = block._run_selected_experts(tokens, scores, indices)
            loss = output.float().square().mean()
            loss.backward()
            parameters = [parameter for adapter in adapters for parameter in adapter.parameters()]
            observed = [output.detach(), loss.detach(), tokens.grad.detach(), logits.grad.detach()]
            observed.extend(parameter.grad.detach() for parameter in parameters)
            for tensor in observed:
                self.assertTrue(torch.isfinite(tensor).all())
            optimizer = torch.optim.AdamW(parameters, lr=1e-2)
            optimizer.step()
            observed.extend(parameter.detach().clone() for parameter in parameters)
            if routed:
                import io
                payload = io.BytesIO()
                torch.save(block.state_dict(), payload)
                payload.seek(0)
                restored, _ = build(use_dora)
                restored.load_state_dict(torch.load(payload, map_location="cuda", weights_only=True))
                for key, value in block.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            return observed

        for use_dora in (False, True):
            reference = run(use_dora, False)
            candidate = run(use_dora, True)
            for expected, actual in zip(reference, candidate, strict=True):
                torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)

    @unittest.skipUnless(
        torch is not None
        and torch.cuda.is_available()
        and os.environ.get("MIRAI_REMOTE_GPU_TESTS") == "1",
        "configured remote CUDA validation required",
    )
    def test_lingbot_routed_triton_inference_token_chunk_parity(self) -> None:
        torch.manual_seed(91)
        block = LingBotVideoSparseMoeBlock(
            hidden_size=17, intermediate_size=32, num_experts=4, top_k=2,
            moe_intermediate_size=23, score_func="softmax", norm_topk_prob=True,
            n_group=None, topk_group=None, routed_scaling_factor=1.0,
            n_shared_experts=0,
        ).to(device="cuda", dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.normal_(std=0.02)
        block._mirai_moe_kernel_backend = LingBotRoutedTritonBackend("triton")
        tokens = torch.randn(7, 17, device="cuda", dtype=torch.bfloat16)
        scores = torch.softmax(torch.randn(7, 2, device="cuda"), -1).to(torch.bfloat16)
        indices = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2], [0, 1]], device="cuda")
        with torch.no_grad():
            whole = block._run_selected_experts(tokens, scores, indices, drop_slots=False)
            chunks = torch.cat([
                block._run_selected_experts(
                    tokens[start:start + 2], scores[start:start + 2],
                    indices[start:start + 2], drop_slots=False,
                )
                for start in range(0, 7, 2)
            ])
        self.assertTrue(torch.isfinite(whole).all())
        self.assertTrue(torch.isfinite(chunks).all())
        torch.testing.assert_close(chunks.float(), whole.float(), rtol=3e-2, atol=3e-2)

    def _lingbot_tiny_config(self, *, strategy: StrategyConfig | None = None) -> TrainingConfig:
        return TrainingConfig(
            model=ModelConfig(
                type="lingbot-video",
                path="./models/lingbot_video",
                params=ModelParams(
                    variant="tiny-video",
                    flow_shift=3.0,
                    latent_channels=2,
                    num_experts=4,
                    experts_per_token=2,
                    shared_experts=1,
                    hidden_size=16,
                    num_layers=1,
                    attention_heads=2,
                    patch_size=1,
                ),
            ),
            strategy=strategy or StrategyConfig(type="text_to_video", params={}),
            training=TrainingSection(seed=12, batch_size=1, gradient_checkpointing="off"),
        )

    def test_token_choice_router_emits_topk_and_stats(self) -> None:
        router = TokenChoiceRouter(
            hidden_size=4,
            num_experts=3,
            experts_per_token=2,
            layer_name="test",
        )
        hidden = torch.randn(2, 5, 4)
        decision = router(hidden)

        self.assertEqual(tuple(decision.topk_indices.shape), (10, 2))
        self.assertEqual(tuple(decision.topk_weights.shape), (10, 2))
        self.assertTrue(torch.isfinite(decision.load_balance_loss).all())
        self.assertEqual(decision.stats.tokens, 10)
        self.assertEqual(len(decision.stats.expert_fraction), 3)

    def test_native_routing_entropy_uses_assignment_distribution(self) -> None:
        router = LingBotVideoRouter(
            hidden_size=4,
            num_experts=3,
            top_k=2,
            score_func="softmax",
            norm_topk_prob=True,
            n_group=None,
            topk_group=None,
            route_scale=7.0,
        )
        router.last_top_indices = torch.tensor([[0, 0], [0, 1], [0, 2]])
        router.last_top_scores = torch.tensor(
            [[6.0, 1.0], [3.5, 3.5], [0.25, 6.75]]
        )
        router.last_scores = torch.tensor(
            [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.7, 0.15, 0.15]]
        )

        stats = _routing_stats(torch.nn.Sequential(router))[0]
        expected_fractions = (4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
        expected_entropy = -sum(
            fraction * math.log(fraction) for fraction in expected_fractions
        )

        self.assertEqual(stats.tokens_per_expert, (4, 1, 1))
        self.assertAlmostEqual(stats.routing_entropy, expected_entropy)

    def test_cpu_bfloat16_testbed_accepts_float32_inputs(self) -> None:
        config = self._lingbot_tiny_config().model
        model = TinySparseMoEDenoiser(config).to(dtype=torch.bfloat16)
        noisy_latents = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        timesteps = torch.rand(2, dtype=torch.float32)
        prediction, auxiliary_loss, _ = model(
            noisy_latents,
            timesteps,
            {"t5": torch.randn(2, 3, 4, dtype=torch.float32)},
        )

        self.assertEqual(prediction.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(auxiliary_loss).all())
        (prediction.float().mean() + auxiliary_loss.float()).backward()
        self.assertTrue(torch.isfinite(model.input_proj.weight.grad).all())

    def test_sparse_moe_ffn_preserves_shape_and_aux_loss(self) -> None:
        layer = SparseMoEFeedForward(
            hidden_size=8,
            intermediate_size=16,
            num_routed_experts=4,
            num_shared_experts=1,
            experts_per_token=2,
            layer_name="blocks.0",
        )
        hidden = torch.randn(2, 6, 8)
        out = layer(hidden)

        self.assertEqual(tuple(out.hidden_states.shape), tuple(hidden.shape))
        self.assertTrue(torch.isfinite(out.auxiliary_loss).all())
        self.assertEqual(out.stats.layer, "blocks.0")

    def test_expert_choice_router_selects_capacity_per_expert(self) -> None:
        router = ExpertChoiceRouter(
            hidden_size=4,
            num_experts=3,
            capacity_factor=1.5,
            layer_name="expert_choice.blocks.0",
        )
        hidden = torch.randn(2, 6, 4)
        decision = router(hidden)

        self.assertEqual(tuple(decision.expert_token_indices.shape), (2, 3, 3))
        self.assertEqual(tuple(decision.expert_token_weights.shape), (2, 3, 3))
        self.assertTrue(torch.isfinite(decision.load_balance_loss).all())
        self.assertEqual(float(decision.load_balance_loss.detach()), 0.0)
        self.assertEqual(decision.stats.tokens, 12)
        self.assertEqual(decision.stats.selected_tokens, 18)
        self.assertEqual(
            decision.stats.tokens_per_expert,
            (6, 6, 6),
        )
        self.assertAlmostEqual(decision.stats.routing_entropy, math.log(3.0))
        self.assertEqual(decision.coverage.capacity_per_expert, 3)
        self.assertEqual(
            decision.coverage.covered_tokens + decision.coverage.uncovered_tokens,
            12,
        )

    def test_expert_choice_is_per_sample_and_normalizes_per_token(self) -> None:
        router = ExpertChoiceRouter(
            hidden_size=2,
            num_experts=2,
            capacity_factor=1.0,
            layer_name="expert_choice.exact",
        )
        with torch.no_grad():
            router.gate.weight.copy_(torch.eye(2))
        hidden = torch.tensor(
            [
                [[4.0, 0.0], [0.0, 4.0], [1.0, 1.0]],
                [[3.0, 0.0], [2.0, 0.0], [0.0, 3.0]],
            ]
        )

        decision = router(hidden)

        self.assertEqual(tuple(decision.expert_token_indices.shape), (2, 2, 2))
        self.assertEqual(set(decision.expert_token_indices[0, 0].tolist()), {0, 2})
        self.assertEqual(set(decision.expert_token_indices[0, 1].tolist()), {1, 2})
        for token_index in range(3):
            selected_weights = []
            for expert_index in range(2):
                mask = decision.expert_token_indices[0, expert_index] == token_index
                selected_weights.extend(
                    decision.expert_token_weights[0, expert_index][mask].tolist()
                )
            if selected_weights:
                self.assertAlmostEqual(sum(selected_weights), 1.0, places=6)
        self.assertEqual(decision.coverage.per_sample_covered_tokens[0], 3)
        self.assertGreaterEqual(decision.coverage.multiply_selected_tokens, 1)

    def test_expert_choice_padding_is_excluded_from_assignment_stats(self) -> None:
        logits = torch.tensor(
            [
                [[20.0, -20.0, -20.0]] * 4,
                [[20.0, -20.0, -20.0]] * 4,
            ]
        )
        decision = route_expert_choice_logits(
            logits,
            capacity_factor=1.0,
            capacity_per_sample=torch.tensor([1, 2]),
            route_scale=1.0,
            layer_name="expert_choice.padding",
            output_dtype=torch.float32,
        )

        self.assertEqual(decision.stats.selected_tokens, 9)
        self.assertEqual(decision.stats.tokens_per_expert, (3, 3, 3))
        self.assertEqual(decision.stats.dead_experts, 0)
        self.assertAlmostEqual(decision.stats.routing_entropy, math.log(3.0))
        self.assertLess(decision.coverage.coverage_fraction, 1.0)

    def test_expert_choice_ffn_preserves_shape_and_aux_loss(self) -> None:
        layer = ExpertChoiceMoEFeedForward(
            hidden_size=8,
            intermediate_size=16,
            num_routed_experts=4,
            num_shared_experts=1,
            capacity_factor=1.0,
            layer_name="expert_choice.blocks.0",
        )
        hidden = torch.randn(2, 8, 8)
        out = layer(hidden)

        self.assertEqual(tuple(out.hidden_states.shape), tuple(hidden.shape))
        self.assertTrue(torch.isfinite(out.auxiliary_loss).all())
        self.assertEqual(out.stats.layer, "expert_choice.blocks.0")
        self.assertIsNotNone(out.expert_choice_coverage)

    def test_expert_choice_ffn_preserves_input_and_adapter_gradients(self) -> None:
        layer = ExpertChoiceMoEFeedForward(
            hidden_size=4,
            intermediate_size=8,
            num_routed_experts=3,
            num_shared_experts=1,
            capacity_factor=1.5,
            layer_name="expert_choice.gradients",
            router_z_loss_weight=1e-4,
        )
        hidden = torch.randn(2, 5, 4, requires_grad=True)

        out = layer(hidden)
        (out.hidden_states.square().mean() + out.auxiliary_loss).backward()

        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertIsNotNone(layer.router.gate.weight.grad)
        self.assertTrue(torch.isfinite(layer.router.gate.weight.grad).all())
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for expert in layer.experts
            for parameter in expert.parameters()
        ))

    def test_expert_choice_capacity_schedule_resolves_step_and_layer(self) -> None:
        schedule = (
            {
                "start_step": 0,
                "end_step": 10,
                "first_layer": 0,
                "end_layer": 4,
                "capacity_factor": 1.5,
            },
            {
                "start_step": 10,
                "end_step": -1,
                "first_layer": 0,
                "end_layer": 4,
                "capacity_factor": 0.75,
            },
        )
        self.assertEqual(
            resolve_capacity_schedule(
                schedule, step=4, layer_index=2, fallback=1.0
            ),
            1.5,
        )
        self.assertEqual(
            resolve_capacity_schedule(
                schedule, step=20, layer_index=2, fallback=1.0
            ),
            0.75,
        )
        self.assertEqual(
            resolve_capacity_schedule(
                schedule, step=4, layer_index=8, fallback=1.0
            ),
            1.0,
        )

    def test_expert_choice_coverage_alarm_uses_worst_observed_layer(self) -> None:
        summary = summarize_expert_choice_coverage(
            (1.0, 0.75, 0.9),
            alarm_threshold=0.8,
        )

        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary.mean_fraction, 0.8833333333)
        self.assertEqual(summary.minimum_fraction, 0.75)
        self.assertTrue(summary.below_threshold)
        self.assertIsNone(
            summarize_expert_choice_coverage((), alarm_threshold=0.8)
        )

    def test_lingbot_expert_choice_executes_and_backpropagates(self) -> None:
        block = LingBotVideoSparseMoeBlock(
            hidden_size=4,
            intermediate_size=8,
            num_experts=3,
            top_k=2,
            moe_intermediate_size=8,
            score_func="softmax",
            norm_topk_prob=True,
            n_group=None,
            topk_group=None,
            routed_scaling_factor=1.0,
            n_shared_experts=1,
        )
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.normal_(mean=0.0, std=0.02)

        def route(logits, output_dtype):
            return route_expert_choice_logits(
                logits,
                capacity_factor=1.5,
                route_scale=1.0,
                layer_name="lingbot.blocks.0",
                output_dtype=output_dtype,
                z_loss_weight=1e-4,
            )

        block.router.set_expert_choice_extension(route)
        conditioner = DecoupledRouterConditioner(
            hidden_size=4,
            num_experts=3,
            timestep_weight=0.25,
        )
        with torch.no_grad():
            conditioner.timestep_projection.normal_(mean=0.0, std=0.02)
        block.router.set_decoupled_routing(conditioner)
        hidden = torch.randn(2, 5, 4, requires_grad=True)
        router_hidden = torch.randn_like(hidden, requires_grad=True)
        timestep_hidden = torch.randn_like(hidden)
        output = block(
            hidden,
            router_input=router_hidden,
            timestep_router_input=timestep_hidden,
        )
        balance, z_loss = block.router.training_expert_choice_decision.load_balance_loss, (
            block.router.training_expert_choice_decision.z_loss
        )
        (output.square().mean() + balance + z_loss).backward()

        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(torch.isfinite(router_hidden.grad).all())
        self.assertTrue(torch.isfinite(block.router.weight.grad).all())
        self.assertTrue(
            torch.isfinite(conditioner.timestep_projection.grad).all()
        )
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in block.experts.parameters()
        ))
        self.assertIsNotNone(block.router.last_route_active_mask)
        self.assertTrue((block.router.last_top_scores >= 0).all())

    def test_lingbot_vectorized_group_padding_matches_eager_reference(self) -> None:
        counts = torch.tensor([3, 0, 5, 1], dtype=torch.int64)
        tokens = torch.arange(
            int(counts.sum()) * 4,
            dtype=torch.float32,
        ).reshape(-1, 4)

        reference = LingBotVideoSparseMoeBlock._pad_grouped_tokens_loop(
            tokens,
            counts,
            align=4,
        )
        vectorized = LingBotVideoSparseMoeBlock._pad_grouped_tokens_vectorized(
            tokens,
            counts,
            align=4,
        )

        self.assertEqual(reference[0], vectorized[0])
        for reference_tensor, vectorized_tensor in zip(
            reference[1:],
            vectorized[1:],
        ):
            self.assertTrue(torch.equal(reference_tensor, vectorized_tensor))

    def test_sparse_moe_test_model_trains_with_auxiliary_metrics(self) -> None:
        register_builtin_components()
        cfg = TrainingConfig(
            model=ModelConfig(
                type="sparse_moe_test",
                path="./models/sparse_moe_test",
                params=ModelParams(
                    variant="tiny-video",
                    flow_shift=3.0,
                    num_experts=4,
                    experts_per_token=2,
                    shared_experts=1,
                    hidden_size=16,
                    num_layers=2,
                    moe_aux_loss_weight=0.01,
                ),
            ),
            training=TrainingSection(seed=7, batch_size=2, gradient_checkpointing="off"),
        )
        trainer = Trainer(cfg)
        batch = {
            "latents": torch.randn(2, 1, 2, 2, 2),
            "text_embeds": torch.ones(2, 4),
        }
        loss, raw = trainer.compute_loss(batch)
        caps = trainer.pipeline.get_sparse_moe_capabilities()

        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(caps.is_sparse_moe)
        self.assertEqual(caps.num_routed_experts, 4)
        self.assertIn("moe_router_aux", raw["auxiliary_losses"])
        self.assertIn("moe_routing", raw["diagnostics"])

    def test_lingbot_video_backend_trains_with_real_video_moe_router(self) -> None:
        register_builtin_components()
        cfg = self._lingbot_tiny_config()
        trainer = Trainer(cfg)
        batch = {
            "latents": torch.randn(1, 2, 1, 2, 2),
            "text_embeds": torch.randn(1, 3, 16),
        }
        loss, raw = trainer.compute_loss(batch)
        caps = trainer.pipeline.get_sparse_moe_capabilities()

        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(caps.is_sparse_moe)
        self.assertEqual(caps.architecture, "lingbot_video")
        self.assertEqual(caps.routing_granularity, "joint_video_text_token")
        self.assertIn("moe_load_balance", raw["auxiliary_losses"])
        self.assertEqual(raw["diagnostics"]["moe_routing"]["layers"], 1)

    def test_lingbot_provider_exposes_native_pruning_calibration(self) -> None:
        register_builtin_components()
        cfg = self._lingbot_tiny_config()
        cfg.model.params.expert_pruning = "prune"
        cfg.model.params.expert_pruning_criterion = "msan"
        trainer = Trainer(cfg)
        provider = get_model_family_provider(cfg.model.type)
        self.assertIsNotNone(provider)
        targets = provider.build_expert_pruning_calibration_targets(trainer.pipeline)
        self.assertEqual(len(targets), 1)
        accumulators = {
            name: ExpertPruningSaliencyAccumulator(
                target.num_experts,
                criterion="msan",
            )
            for name, target in targets.items()
        }
        try:
            for name, target in targets.items():
                target.host.set_expert_output_observer(
                    ExpertPruningRoutedOutputObserver(accumulators[name])
                )
            trainer.compute_loss(
                {
                    "latents": torch.randn(1, 2, 1, 2, 2),
                    "text_embeds": torch.randn(1, 3, 16),
                },
                training=False,
            )
        finally:
            for target in targets.values():
                target.host.set_expert_output_observer(None)
        for accumulator in accumulators.values():
            evidence = accumulator.evidence()
            self.assertGreater(int(evidence.selected_count.sum().item()), 0)
            self.assertTrue(torch.isfinite(evidence.scores()).all())

    def test_lingbot_provider_exposes_routing_agreement_targets(self) -> None:
        register_builtin_components()
        cfg = self._lingbot_tiny_config()
        cfg.model.params.moe_routing_agreement_evidence = "report"
        trainer = Trainer(cfg)
        provider = get_model_family_provider(cfg.model.type)
        self.assertIsNotNone(provider)
        self.assertTrue(provider.supports_routing_mode_agreement_evidence(cfg))
        targets = provider.build_routing_mode_agreement_targets(trainer.pipeline)
        self.assertEqual(len(targets), 1)
        with RoutingSelectionCapture(targets) as capture:
            trainer.compute_loss(
                {
                    "latents": torch.randn(1, 2, 1, 2, 2),
                    "text_embeds": torch.randn(1, 3, 16),
                },
                training=False,
            )
        snapshots = capture.snapshots()
        self.assertEqual(set(snapshots), set(targets))
        self.assertEqual(len(next(iter(snapshots.values()))), 1)

    def test_lingbot_attention_backend_reaches_native_modules(self) -> None:
        cfg = self._lingbot_tiny_config()
        cfg.model.attention_backend = "flash"
        pipeline = LingBotVideoPipeline.from_training_config(cfg)
        backends = {
            module.backend
            for module in pipeline.transformer.modules()
            if isinstance(module, LingBotVideoAttention)
        }
        self.assertEqual(backends, {"flash"})

    def test_frozen_router_int8_storage_preserves_routing_and_adapter_gradients(self) -> None:
        torch.manual_seed(91)
        router = LingBotVideoRouter(
            16, 8, 2, "softmax", True, None, None, 1.0
        ).eval()
        with torch.no_grad():
            router.weight.normal_(mean=0.0, std=0.05)
        tokens = torch.randn(13, 16)
        reference = router(tokens)
        float_bytes = router.weight.numel() * router.weight.element_size()
        router.weight.requires_grad_(False)
        router.enable_int8_weight()
        observed = router(tokens)
        quantized_bytes = (
            router.weight_int8.numel() * router.weight_int8.element_size()
            + router.weight_scale.numel() * router.weight_scale.element_size()
        )
        self.assertNotIn("weight", dict(router.named_parameters()))
        self.assertLess(quantized_bytes, float_bytes)
        self.assertTrue(torch.equal(observed[0], reference[0]))
        self.assertTrue(torch.allclose(observed[2], reference[2], atol=2e-2, rtol=2e-2))

        register_builtin_components()
        cfg = self._lingbot_tiny_config()
        cfg.memory.router_quantization = "int8_per_channel"
        trainer = Trainer(cfg)
        loss, _raw = trainer.compute_loss(
            {
                "latents": torch.randn(1, 2, 3, 2, 2),
                "text_embeds": torch.randn(1, 3, 16),
            }
        )
        loss.backward()
        routed = [
            module
            for module in trainer.pipeline.transformer.modules()
            if isinstance(module, LingBotVideoRouter)
        ]
        self.assertTrue(routed)
        self.assertTrue(all(hasattr(module, "weight_int8") for module in routed))
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in trainer.pipeline.get_trainable_parameters()
            )
        )

    def test_lingbot_video_backend_trains_with_configured_expert_choice(self) -> None:
        register_builtin_components()
        cfg = self._lingbot_tiny_config()
        cfg.model.params.moe_routing_mode = "expert_choice"
        cfg.model.params.moe_expert_choice_capacity_factor = 1.5
        cfg.model.params.moe_expert_choice_coverage_alarm_threshold = 0.9
        cfg.model.params.moe_expert_choice_capacity_schedule = [
            {
                "start_step": 0,
                "end_step": 1,
                "first_layer": 0,
                "end_layer": 1,
                "capacity_factor": 2.0,
            },
            {
                "start_step": 1,
                "end_step": -1,
                "first_layer": 0,
                "end_layer": 1,
                "capacity_factor": 1.0,
            },
        ]
        cfg.model.params.moe_router_timestep_weight = 0.25
        cfg.model.params.moe_router_z_loss_weight = 1e-4
        trainer = Trainer(cfg)
        batch = {
            "latents": torch.randn(1, 2, 1, 2, 2),
            "text_embeds": torch.randn(1, 3, 16),
        }
        result = trainer.train_step(batch)
        caps = trainer.pipeline.get_sparse_moe_capabilities()

        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])).all())
        self.assertEqual(caps.routing, "expert_choice_capacity")
        self.assertIn("moe_router_z", result["auxiliary_losses"])
        self.assertEqual(result["diagnostics"]["moe_routing"]["layers"], 1)
        self.assertIn(
            "moe_expert_choice_coverage_fraction",
            result["diagnostics"],
        )
        self.assertIn(
            "moe_expert_choice_min_coverage_fraction",
            result["diagnostics"],
        )
        self.assertIn("moe_expert_choice_coverage_alarm", result["diagnostics"])
        conditioners = [
            module
            for module in trainer.pipeline.transformer.modules()
            if isinstance(module, DecoupledRouterConditioner)
        ]
        self.assertTrue(conditioners)
        loss, _metrics = trainer.compute_loss(batch)
        loss.backward()
        self.assertTrue(
            all(
                module.timestep_projection.grad is not None
                and torch.isfinite(module.timestep_projection.grad).all()
                for module in conditioners
            )
        )

    def test_lingbot_video_backend_trains_multi_frame_bcthw_latents(self) -> None:
        register_builtin_components()
        trainer = Trainer(self._lingbot_tiny_config())
        result = trainer.train_step(
            {
                "latents": torch.randn(1, 2, 3, 2, 2),
                "text_embeds": torch.randn(1, 3, 16),
            }
        )

        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])).all())
        self.assertIn("moe_load_balance", result["auxiliary_losses"])
        self.assertEqual(result["diagnostics"]["moe_routing"]["layers"], 1)

    def test_lingbot_video_i2v_trains_multi_frame_bcthw_latents(self) -> None:
        register_builtin_components()
        cfg = self._lingbot_tiny_config(
            strategy=StrategyConfig(
                type="image_to_video",
                params={"first_frame_conditioning_p": 1.0},
            )
        )
        trainer = Trainer(cfg)
        batch = {
            "latents": torch.randn(1, 2, 3, 2, 2),
            "text_embeds": torch.randn(1, 3, 16),
        }

        loss, raw = trainer.compute_loss(batch)
        prepared = trainer.strategy.prepare_inputs(
            batch=batch,
            pipeline=trainer.pipeline,
            timestep_sampler=trainer.timestep_sampler,
            noise_generator=trainer.noise_generator,
        )

        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.allclose(prepared.noisy_latents[:, :, 0], batch["latents"][:, :, 0]))
        self.assertTrue(torch.allclose(prepared.loss_mask[:, :, 0], torch.zeros_like(prepared.loss_mask[:, :, 0])))
        self.assertIn("moe_load_balance", raw["auxiliary_losses"])

    def test_lingbot_video_hybrid_conditioning_trains(self) -> None:
        register_builtin_components()
        cfg = self._lingbot_tiny_config(
            strategy=StrategyConfig(
                type="hybrid_conditioning",
                params={"text_weight": 0.7, "clip_weight": 0.3},
            )
        )
        trainer = Trainer(cfg)
        text = torch.randn(1, 3, 16)
        clip = torch.randn_like(text)
        loss, raw = trainer.compute_loss(
            {
                "latents": torch.randn(1, 2, 3, 2, 2),
                "text_embeds": text,
                "clip_embed": clip,
            }
        )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in trainer.pipeline.get_trainable_parameters()
        ]
        self.assertTrue(torch.isfinite(loss).all())
        self.assertIn("moe_load_balance", raw["auxiliary_losses"])
        self.assertTrue(
            any(
                gradient is not None and bool(torch.isfinite(gradient).all())
                for gradient in gradients
            )
        )

    def test_lingbot_video_rejects_malformed_cached_latents(self) -> None:
        register_builtin_components()
        trainer = Trainer(self._lingbot_tiny_config())

        with self.assertRaisesRegex(ValueError, "\\[B,C,T,H,W\\]"):
            trainer.compute_loss(
                {
                    "latents": torch.randn(1, 2, 2, 2),
                    "text_embeds": torch.randn(1, 3, 16),
                }
            )

        with self.assertRaisesRegex(ValueError, "latent channels"):
            trainer.compute_loss(
                {
                    "latents": torch.randn(1, 3, 3, 2, 2),
                    "text_embeds": torch.randn(1, 3, 16),
                }
            )

    def test_lingbot_video_rejects_latents_not_divisible_by_patch_size(self) -> None:
        register_builtin_components()
        cfg = self._lingbot_tiny_config()
        cfg.model.params.patch_size = 2
        trainer = Trainer(cfg)

        with self.assertRaisesRegex(ValueError, "patch_size"):
            trainer.compute_loss(
                {
                    "latents": torch.randn(1, 2, 3, 3, 2),
                    "text_embeds": torch.randn(1, 3, 16),
                }
            )

    def test_lingbot_video_passes_text_mask_to_native_transformer(self) -> None:
        register_builtin_components()
        trainer = Trainer(self._lingbot_tiny_config())
        seen: dict[str, torch.Tensor | None] = {}
        original_forward = trainer.pipeline.transformer.forward

        def wrapped_forward(*args, **kwargs):
            seen["encoder_attention_mask"] = kwargs.get("encoder_attention_mask")
            return original_forward(*args, **kwargs)

        trainer.pipeline.transformer.forward = wrapped_forward
        text_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
        loss, _raw = trainer.compute_loss(
            {
                "latents": torch.randn(1, 2, 3, 2, 2),
                "text_embeds": torch.randn(1, 3, 16),
                "text_mask": text_mask,
            }
        )

        self.assertTrue(torch.isfinite(loss).all())
        self.assertIsNotNone(seen["encoder_attention_mask"])
        self.assertTrue(
            torch.equal(
                seen["encoder_attention_mask"].cpu(),
                torch.tensor([[True, True]], dtype=torch.bool),
            )
        )

    def test_lingbot_video_rejects_non_prefix_text_mask(self) -> None:
        register_builtin_components()
        trainer = Trainer(self._lingbot_tiny_config())

        with self.assertRaisesRegex(ValueError, "prefix padding"):
            trainer.compute_loss(
                {
                    "latents": torch.randn(1, 2, 3, 2, 2),
                    "text_embeds": torch.randn(1, 3, 16),
                    "text_mask": torch.tensor([[True, False, True]], dtype=torch.bool),
                }
            )

    def test_open_sparse_moe_catalog_tracks_current_targets(self) -> None:
        specs = {spec.model_id: spec for spec in get_open_sparse_moe_model_specs()}

        self.assertEqual(set(specs), {"lingbot_video"})
        self.assertEqual(specs["lingbot_video"].modality, "video")
        self.assertEqual(specs["lingbot_video"].license, "apache-2.0")
        self.assertEqual(specs["lingbot_video"].integration_level, "native_training_and_inference")
        self.assertEqual(specs["lingbot_video"].integration_blockers, ())

    def test_lingbot_video_strict_assets_requires_transformer_dir(self) -> None:
        register_builtin_components()
        cfg = TrainingConfig(
            model=ModelConfig(
                type="lingbot-video",
                path="./models/lingbot_video_missing",
                params=ModelParams(
                    variant="lingbot-video-moe-30b-a3b",
                    strict_native_assets=True,
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "transformer/config.json"):
            Trainer(cfg)

    def test_lingbot_video_strict_assets_rejects_incomplete_download(self) -> None:
        register_builtin_components()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transformer_dir = root / "transformer"
            transformer_dir.mkdir(parents=True)
            (transformer_dir / "config.json").write_text("{}", encoding="utf-8")
            (root / "refiner").mkdir()
            (root / "refiner" / "shard.safetensors.part").write_bytes(b"partial")

            cfg = TrainingConfig(
                model=ModelConfig(
                    type="lingbot-video",
                    path=str(root),
                    params=ModelParams(
                        variant="lingbot-video-moe-30b-a3b",
                        strict_native_assets=True,
                    ),
                )
            )

            with self.assertRaisesRegex(ValueError, "downloaded snapshot verification failed"):
                Trainer(cfg)

    def test_lingbot_video_strict_assets_loads_native_transformer_safetensors(self) -> None:
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("safetensors not installed")
        register_builtin_components()
        params = ModelParams(
            variant="tiny-video",
            strict_native_assets=False,
            latent_channels=2,
            num_experts=4,
            experts_per_token=2,
            shared_experts=1,
            hidden_size=16,
            num_layers=1,
            attention_heads=2,
            patch_size=1,
        )
        seed = LingBotVideoPipeline(
            ModelConfig(
                type="lingbot-video",
                path="./models/lingbot_seed",
                params=params,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transformer_dir = root / "transformer"
            transformer_dir.mkdir(parents=True)
            (transformer_dir / "config.json").write_text(
                __import__("json").dumps(seed.transformer_config),
                encoding="utf-8",
            )
            save_file(seed.transformer.state_dict(), str(transformer_dir / "diffusion_pytorch_model.safetensors"))
            strict_params = ModelParams(
                variant="lingbot-video-moe-30b-a3b",
                strict_native_assets=True,
                latent_channels=2,
                num_experts=4,
                experts_per_token=2,
                shared_experts=1,
                hidden_size=16,
                num_layers=1,
                attention_heads=2,
                patch_size=1,
            )
            trainer = Trainer(
                TrainingConfig(
                    model=ModelConfig(
                        type="lingbot-video",
                        path=str(root),
                        params=strict_params,
                    )
                )
            )

        report = trainer.pipeline.get_checkpoint_report()
        self.assertIsNotNone(report)
        self.assertGreater(report["matched_keys"], 0)
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["unexpected_keys"], [])

    def test_lingbot_video_strict_assets_can_load_refiner_subfolder(self) -> None:
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("safetensors not installed")
        register_builtin_components()
        params = ModelParams(
            variant="tiny-video",
            strict_native_assets=False,
            latent_channels=2,
            num_experts=4,
            experts_per_token=2,
            shared_experts=1,
            hidden_size=16,
            num_layers=1,
            attention_heads=2,
            patch_size=1,
        )
        seed = LingBotVideoPipeline(
            ModelConfig(
                type="lingbot-video",
                path="./models/lingbot_seed",
                params=params,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refiner_dir = root / "refiner"
            refiner_dir.mkdir(parents=True)
            (refiner_dir / "config.json").write_text(
                __import__("json").dumps(seed.transformer_config),
                encoding="utf-8",
            )
            save_file(seed.transformer.state_dict(), str(refiner_dir / "diffusion_pytorch_model.safetensors"))
            strict_params = ModelParams(
                variant="lingbot-video-moe-30b-a3b",
                strict_native_assets=True,
                denoiser_subfolder="refiner",
                latent_channels=2,
                num_experts=4,
                experts_per_token=2,
                shared_experts=1,
                hidden_size=16,
                num_layers=1,
                attention_heads=2,
                patch_size=1,
            )
            trainer = Trainer(
                TrainingConfig(
                    model=ModelConfig(
                        type="lingbot-video",
                        path=str(root),
                        params=strict_params,
                    )
                )
            )

        report = trainer.pipeline.get_checkpoint_report()
        self.assertIsNotNone(report)
        self.assertTrue(str(report["path"]).endswith("refiner"))
        self.assertGreater(report["matched_keys"], 0)
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["unexpected_keys"], [])

    def test_cli_policy_rejects_unregistered_models(self) -> None:
        register_builtin_components()
        cfg = TrainingConfig(
            model=ModelConfig(
                type="unknown-diffusion",
                path="./models/unknown",
                params=ModelParams(variant="scratch"),
            )
        )

        with self.assertRaisesRegex(ValueError, "not a registered sparse-MoE"):
            validate_cli_model_contract(cfg, entrypoint="train")


if __name__ == "__main__":
    unittest.main()
