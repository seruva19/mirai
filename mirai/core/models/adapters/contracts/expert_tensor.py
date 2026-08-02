from __future__ import annotations

import unittest

# Colocated behavioral contract for expert-tensor adapter execution.
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.utils import parametrize

from mirai.core.models.adapters.dora import DORA_MAGNITUDE_SUFFIX
from mirai.core.models.adapters.expert_tensor_lora import ExpertTensorLoRAExecutor
from mirai.core.models.adapters.lora import LoRAExpertTensorParametrization
from mirai.core.models.adapters.lora import LoRALinear
from mirai.core.models.adapters.lora import load_lora_state_dict
from mirai.core.models.adapters.lora import lora_state_dict
from mirai.core.moe.adaptation.adapter_gate import RoutedAdapterGate


def _sm90_grouped_mm_available() -> bool:
    if not torch.cuda.is_available() or not hasattr(torch, "_grouped_mm"):
        return False
    try:
        return torch.cuda.get_device_capability()[0] >= 9
    except (AssertionError, RuntimeError):
        return False


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(3, 7, 5, dtype=torch.float64))
        self.w2 = nn.Parameter(torch.randn(3, 5, 7, dtype=torch.float64))
        self.w3 = nn.Parameter(torch.randn(3, 7, 5, dtype=torch.float64))


def _attach(
    owner: nn.Module, key: str, *, rank: int = 2
) -> LoRAExpertTensorParametrization:
    weight = getattr(owner, key)
    adapter = LoRAExpertTensorParametrization(
        adapter_name=key,
        shape=tuple(weight.shape),
        layout=("expert", "out", "in"),
        rank=rank,
        alpha=4.0,
    ).to(dtype=weight.dtype)
    parametrize.register_parametrization(owner, key, adapter)
    with torch.no_grad():
        adapter.lora_b.normal_()
    return adapter


class ExpertTensorLoRAExecutorTests(unittest.TestCase):
    def test_dora_dense_matches_weight_decomposition_and_has_gradients(
        self,
    ) -> None:
        torch.manual_seed(101)
        base = nn.Linear(5, 4, bias=True, dtype=torch.float32)
        module = LoRALinear(
            base,
            rank=2,
            alpha=4.0,
            use_dora=True,
        )
        with torch.no_grad():
            module.lora_a.normal_(std=0.2)
            module.lora_b.normal_(std=0.2)
            module.dora_magnitude.mul_(1.1)
        inputs = torch.randn(3, 5, dtype=torch.float32, requires_grad=True)

        actual = module(inputs)
        direction = base.weight + 2.0 * (module.lora_b @ module.lora_a)
        expected_weight = direction * (
            module.dora_magnitude
            / torch.linalg.vector_norm(direction, dim=1)
        ).unsqueeze(1)
        expected = torch.nn.functional.linear(inputs, expected_weight, base.bias)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        actual.square().mean().backward()
        for parameter in (
            inputs,
            module.lora_a,
            module.lora_b,
            module.dora_magnitude,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_dora_grouped_matches_decomposition_and_zero_scale_bypasses(
        self,
    ) -> None:
        torch.manual_seed(103)
        base = nn.Parameter(torch.randn(3, 4, 5, dtype=torch.float32))
        module = LoRAExpertTensorParametrization(
            adapter_name="experts.w1",
            shape=tuple(base.shape),
            layout=("expert", "out", "in"),
            rank=2,
            alpha=4.0,
            use_dora=True,
            base_weight=base,
        )
        module.initialize_dora_magnitude(base)
        with torch.no_grad():
            module.lora_a.normal_(std=0.2)
            module.lora_b.normal_(std=0.2)
            module.dora_magnitude.mul_(0.9)

        direction = base + 2.0 * torch.matmul(module.lora_b, module.lora_a)
        expected = direction * (
            module.dora_magnitude
            / torch.linalg.vector_norm(direction, dim=-1)
        ).unsqueeze(-1)
        torch.testing.assert_close(module(base), expected, rtol=1e-5, atol=1e-6)

        module.set_lora_scale(0.0)
        self.assertTrue(torch.equal(module(base), base))

    def test_dora_native_state_round_trips_and_rejects_mismatch(self) -> None:
        torch.manual_seed(107)
        source = nn.Module()
        source.proj = LoRALinear(
            nn.Linear(5, 4, bias=False),
            rank=2,
            alpha=2.0,
            use_dora=True,
        )
        with torch.no_grad():
            source.proj.lora_b.normal_()
            source.proj.dora_magnitude.add_(0.25)
        state = lora_state_dict(source)
        self.assertIn(f"proj{DORA_MAGNITUDE_SUFFIX}", state)

        restored = nn.Module()
        restored.proj = LoRALinear(
            nn.Linear(5, 4, bias=False),
            rank=2,
            alpha=2.0,
            use_dora=True,
        )
        load_lora_state_dict(restored, state)
        self.assertTrue(
            torch.equal(
                restored.proj.dora_magnitude,
                source.proj.dora_magnitude,
            )
        )

        plain = nn.Module()
        plain.proj = LoRALinear(
            nn.Linear(5, 4, bias=False),
            rank=2,
            alpha=2.0,
        )
        with self.assertRaisesRegex(ValueError, "DoRA state mismatch"):
            load_lora_state_dict(plain, state)

    def test_plain_lora_allocates_no_dora_state(self) -> None:
        module = LoRALinear(nn.Linear(3, 2), rank=1, alpha=1.0)
        self.assertFalse(module.use_dora)
        self.assertIsNone(module.dora_magnitude)
        self.assertFalse(any("dora" in key for key in module.state_dict()))

    def test_lingbot_provider_installs_weight_space_dora(self) -> None:
        from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams
        from mirai.core.builtins import register_builtin_components
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
        from mirai.core.models.providers import get_model_family_provider
        from mirai.vendors.lingbot_video.transformer_lingbot_video import (
            LingBotVideoGroupedExperts,
        )

        register_builtin_components()
        provider = get_model_family_provider("lingbot-video")
        self.assertIsNotNone(provider)
        self.assertTrue(provider.supports_dora())
        pipeline = LingBotVideoPipeline(
            ModelConfig(
                type="lingbot-video",
                path="./nonexistent/lingbot-video",
                params=ModelParams(
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
                ),
            )
        )
        root = nn.Module()
        root.blocks = nn.ModuleList([nn.Module()])
        root.blocks[0].ffn = nn.Module()
        root.blocks[0].ffn.experts = LingBotVideoGroupedExperts(4, 16, 32)
        pipeline.transformer = root
        pipeline.set_adapter_config(
            AdapterConfig(
                target_preset="routed_experts_only",
                rank=4,
                alpha=4.0,
                use_dora=True,
            )
        )
        adapters = [
            module
            for module in pipeline.transformer.modules()
            if isinstance(module, LoRAExpertTensorParametrization)
        ]
        self.assertEqual(len(adapters), 3)
        self.assertTrue(all(module.use_dora for module in adapters))
        self.assertTrue(
            all(
                tuple(module.dora_magnitude.shape)
                == tuple(module.gora_base_weight().shape[:-1])
                for module in adapters
            )
        )

    def test_lingbot_provider_installs_opt_in_host_extension(self) -> None:
        from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams
        from mirai.core.builtins import register_builtin_components
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
        from mirai.vendors.lingbot_video.transformer_lingbot_video import (
            LingBotVideoGroupedExperts,
        )

        register_builtin_components()
        pipeline = LingBotVideoPipeline(
            ModelConfig(
                type="lingbot-video",
                path="./nonexistent/lingbot-video",
                params=ModelParams(
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
                ),
            )
        )
        root = nn.Module()
        root.blocks = nn.ModuleList([nn.Module()])
        root.blocks[0].ffn = nn.Module()
        root.blocks[0].ffn.experts = LingBotVideoGroupedExperts(4, 16, 32)
        pipeline.transformer = root
        pipeline.set_adapter_config(
            AdapterConfig(
                target_preset="routed_experts_only",
                rank=4,
                alpha=4.0,
                expert_tensor_lora_backend="activation",
            )
        )
        hosts = [
            module
            for module in pipeline.transformer.modules()
            if bool(getattr(module, "mirai_expert_tensor_host", False))
        ]
        self.assertTrue(hosts)
        self.assertTrue(all(host.linear_extension() is not None for host in hosts))
        state_keys = tuple(pipeline.transformer.state_dict())
        self.assertTrue(
            any(key.endswith("parametrizations.w1.0.lora_a") for key in state_keys)
        )
        self.assertFalse(any("linear_extension" in key for key in state_keys))

    def test_loop_matches_weight_space_outputs_and_gradients(self) -> None:
        torch.manual_seed(17)
        experts = _Experts()
        adapters = {key: _attach(experts, key) for key in ("w1", "w2", "w3")}
        counts = torch.tensor([2, 0, 3], dtype=torch.int64)
        actual_tokens = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
        reference_tokens = actual_tokens.detach().clone().requires_grad_(True)

        executor = ExpertTensorLoRAExecutor()
        actual = executor.run_for_loop(experts, actual_tokens, counts)

        chunks = torch.split(reference_tokens, counts.tolist(), dim=0)
        expected_parts = []
        for expert_index, chunk in enumerate(chunks):
            if chunk.numel() == 0:
                continue
            gate = torch.nn.functional.silu(
                torch.nn.functional.linear(chunk, experts.w1[expert_index])
            )
            up = torch.nn.functional.linear(chunk, experts.w3[expert_index])
            expected_parts.append(
                torch.nn.functional.linear(gate * up, experts.w2[expert_index])
            )
        expected = torch.cat(expected_parts, dim=0)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

        actual.sum().backward()
        actual_grads = {
            (key, factor): getattr(adapter, factor).grad.detach().clone()
            for key, adapter in adapters.items()
            for factor in ("lora_a", "lora_b")
        }
        actual_input_grad = actual_tokens.grad.detach().clone()
        for adapter in adapters.values():
            adapter.lora_a.grad = None
            adapter.lora_b.grad = None
        expected.sum().backward()
        torch.testing.assert_close(actual_input_grad, reference_tokens.grad, rtol=1e-10, atol=1e-10)
        for key, adapter in adapters.items():
            for factor in ("lora_a", "lora_b"):
                torch.testing.assert_close(
                    actual_grads[(key, factor)],
                    getattr(adapter, factor).grad,
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_empty_dispatch_preserves_shape(self) -> None:
        experts = _Experts()
        for key in ("w1", "w2", "w3"):
            _attach(experts, key)
        tokens = torch.empty(0, 5, dtype=torch.float64)
        output = ExpertTensorLoRAExecutor().run_for_loop(
            experts, tokens, torch.zeros(3, dtype=torch.int64)
        )
        self.assertEqual(tuple(output.shape), (0, 5))

    def test_loop_route_gate_masks_only_adapter_math(self) -> None:
        torch.manual_seed(29)
        experts = _Experts()
        with torch.no_grad():
            experts.w1.zero_()
            experts.w2.zero_()
            experts.w3.zero_()
        adapters = {key: _attach(experts, key) for key in ("w1", "w2", "w3")}
        tokens = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
        counts = torch.tensor([2, 2, 0], dtype=torch.int64)
        route_token_indices = torch.tensor([0, 2, 1, 3])
        expected_gate = torch.tensor([True, False, False, True])
        executor = ExpertTensorLoRAExecutor()
        executor.set_routed_adapter_gate(
            RoutedAdapterGate(
                torch.tensor(
                    [
                        [True, False, False],
                        [False, True, False],
                    ]
                ),
                tokens_per_sample=2,
            )
        )

        output = executor.run_for_loop(
            experts,
            tokens,
            counts,
            route_token_indices=route_token_indices,
        )
        self.assertTrue(
            torch.equal(
                output[~expected_gate],
                torch.zeros_like(output[~expected_gate]),
            )
        )
        output.sum().backward()
        self.assertTrue(
            torch.equal(
                tokens.grad[~expected_gate],
                torch.zeros_like(tokens.grad[~expected_gate]),
            )
        )
        self.assertTrue(torch.isfinite(tokens.grad[expected_gate]).all())
        for adapter in adapters.values():
            self.assertTrue(torch.isfinite(adapter.lora_a.grad).all())
            self.assertTrue(torch.isfinite(adapter.lora_b.grad).all())

    def test_enabled_route_gate_requires_native_token_identity(self) -> None:
        experts = _Experts()
        for key in ("w1", "w2", "w3"):
            _attach(experts, key)
        executor = ExpertTensorLoRAExecutor()
        executor.set_routed_adapter_gate(
            RoutedAdapterGate(torch.ones(1, 3), tokens_per_sample=2)
        )
        with self.assertRaisesRegex(RuntimeError, "token identity"):
            executor.run_for_loop(
                experts,
                torch.randn(2, 5, dtype=torch.float64),
                torch.tensor([2, 0, 0]),
            )

    def test_native_host_passes_sorted_original_token_identity(self) -> None:
        from mirai.vendors.lingbot_video.transformer_lingbot_video import (
            LingBotVideoRuntimeOptions,
            LingBotVideoSparseMoeBlock,
        )

        block = LingBotVideoSparseMoeBlock.__new__(LingBotVideoSparseMoeBlock)
        nn.Module.__init__(block)
        block.router = SimpleNamespace(num_experts=2)
        block.experts = SimpleNamespace()
        block._mirai_moe_kernel_backend = None
        block._mirai_expert_output_observer = None
        block._mirai_expert_intermediate_observer = None
        captured: dict[str, torch.Tensor] = {}

        def _runner(tokens, counts, *, route_token_indices=None):
            captured["tokens"] = tokens
            captured["counts"] = counts
            captured["route_token_indices"] = route_token_indices
            return torch.zeros_like(tokens)

        block._run_experts_for_loop = _runner
        tokens = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        top_indices = torch.tensor([[1, 0], [0, 1]])
        block._mirai_runtime_options = LingBotVideoRuntimeOptions(
            moe_expert_backend="loop"
        )
        output = block._run_selected_experts(
            tokens,
            torch.ones(2, 2),
            top_indices,
        )

        torch.testing.assert_close(
            captured["route_token_indices"],
            torch.tensor([0, 1, 0, 1]),
        )
        torch.testing.assert_close(
            captured["tokens"],
            tokens.index_select(0, captured["route_token_indices"]),
        )
        torch.testing.assert_close(captured["counts"], torch.tensor([2, 2]))
        self.assertEqual(tuple(output.shape), (2, 4))

    @unittest.skipUnless(
        _sm90_grouped_mm_available(),
        "CUDA SM90+ grouped_mm unavailable",
    )
    def test_cuda_grouped_matches_weight_space_without_delta_materialization(self) -> None:
        torch.manual_seed(23)
        experts = nn.Module()
        experts.w1 = nn.Parameter(
            torch.randn(3, 32, 16, device="cuda", dtype=torch.bfloat16)
        )
        adapter = _attach(experts, "w1", rank=8)
        tokens = torch.randn(12, 16, device="cuda", dtype=torch.bfloat16)
        offsets = torch.tensor([4, 7, 12], device="cuda", dtype=torch.int32)
        expected = torch.cat(
            [
                torch.nn.functional.linear(chunk, experts.w1[index])
                for index, chunk in enumerate(torch.split(tokens, [4, 3, 5]))
            ],
            dim=0,
        )
        actual = ExpertTensorLoRAExecutor().grouped_linear(
            experts, "w1", tokens, offsets=offsets
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        loss = actual.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(adapter.lora_a.grad).all())
        self.assertTrue(torch.isfinite(adapter.lora_b.grad).all())

    @unittest.skipUnless(
        _sm90_grouped_mm_available(),
        "CUDA SM90+ grouped_mm unavailable",
    )
    def test_cuda_grouped_supports_low_rank_unaligned_transpose(self) -> None:
        torch.manual_seed(37)
        experts = nn.Module()
        experts.w1 = nn.Parameter(
            torch.randn(3, 32, 16, device="cuda", dtype=torch.bfloat16)
        )
        adapter = _attach(experts, "w1", rank=2)
        tokens = torch.randn(
            12,
            16,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        expected = torch.cat(
            [
                torch.nn.functional.linear(chunk, experts.w1[index])
                for index, chunk in enumerate(torch.split(tokens, [4, 3, 5]))
            ],
            dim=0,
        )
        actual = ExpertTensorLoRAExecutor().grouped_linear(
            experts,
            "w1",
            tokens,
            offsets=torch.tensor([4, 7, 12], device="cuda", dtype=torch.int32),
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        actual.float().square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(torch.isfinite(adapter.lora_a.grad).all())
        self.assertTrue(torch.isfinite(adapter.lora_b.grad).all())

    @unittest.skipUnless(
        _sm90_grouped_mm_available(),
        "CUDA SM90+ grouped_mm unavailable",
    )
    def test_cuda_grouped_route_gate_zeroes_masked_adapter_gradients(self) -> None:
        torch.manual_seed(31)
        experts = nn.Module()
        experts.w1 = nn.Parameter(
            torch.zeros(2, 32, 16, device="cuda", dtype=torch.bfloat16)
        )
        adapter = _attach(experts, "w1", rank=8)
        tokens = torch.randn(
            12,
            16,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        route_gate = torch.tensor(
            [
                True,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
            ],
            device="cuda",
        )
        output = ExpertTensorLoRAExecutor().grouped_linear(
            experts,
            "w1",
            tokens,
            offsets=torch.tensor([4, 12], device="cuda", dtype=torch.int32),
            route_gate=route_gate,
        )
        self.assertTrue(
            torch.equal(
                output[~route_gate],
                torch.zeros_like(output[~route_gate]),
            )
        )
        output.float().sum().backward()
        self.assertTrue(
            torch.equal(
                tokens.grad[~route_gate],
                torch.zeros_like(tokens.grad[~route_gate]),
            )
        )
        self.assertTrue(torch.isfinite(tokens.grad[route_gate]).all())
        self.assertTrue(torch.isfinite(adapter.lora_a.grad).all())
        self.assertTrue(torch.isfinite(adapter.lora_b.grad).all())


if __name__ == "__main__":
    unittest.main()
