from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch.nn as nn

from mirai.config.schema import (
    MemoryConfig,
    ModelConfig,
    ModelParams,
    TrainingConfig,
    TrainingSection,
)
from mirai.core.builtins import register_builtin_components
from mirai.core.models.compressed_weights import CompressedGroupedExperts
from mirai.core.models.compressed_weights.execution.mixed_precision import (
    MixedPrecisionGroupedExperts,
)
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.runtime.specs import ExpertTensorSpec
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy
from mirai.core.moe.runtime.specs import active_moe_optimization_policy
from mirai.core.moe.runtime.specs import validate_expert_tensor_specs
from mirai.core.models.base import BasePipeline
from mirai.core.training.trainer import Trainer
from mirai.core.training.runtime.trainer import initialize_trainer_runtime

try:
    import torch
    from mirai.core.models.compressed_weights.execution.dispatch_ops import (
        combine_routed_tokens,
        permute_routed_tokens,
    )
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class _NoMemorySupportPipeline(BasePipeline):
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    def apply_noise(self, clean_latents, noise, timesteps):
        return clean_latents

    def compute_target(self, noise, clean_latents, timesteps):
        return clean_latents

    def forward(self, noisy_latents, timesteps, text_embeds, **kwargs):
        return noisy_latents

    def validate_config(self, config: TrainingConfig) -> list[str]:
        return []

    def get_trainable_parameters(self):
        return []

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        _ = state


@unittest.skipIf(torch is None, "torch not installed")
class MoEExpertOptimizationTests(unittest.TestCase):
    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "SonicMoE metadata parity requires remote CUDA execution",
    )
    def test_sonic_dispatch_metadata_matches_device_reference(self) -> None:
        from mirai.core.models.compressed_weights.execution.dispatch_preprocess import (
            build_dispatch_plan,
            sonic_dispatch_preprocess_available,
        )

        if not sonic_dispatch_preprocess_available():
            self.skipTest("sonic-moe is not installed")
        torch.manual_seed(11)
        indices = torch.randint(0, 23, (257, 4), device="cuda")
        reference_scores = torch.randn(
            257,
            4,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        sonic_scores = reference_scores.detach().clone().requires_grad_(True)
        reference = build_dispatch_plan(indices, reference_scores, 23, backend="device")
        sonic = build_dispatch_plan(indices, sonic_scores, 23, backend="sonic")
        for field in (
            "counts",
            "starts",
            "offsets",
            "sort_positions",
            "sorted_token_indices",
            "sorted_scores",
            "inverse_order",
        ):
            torch.testing.assert_close(
                getattr(sonic, field),
                getattr(reference, field),
                rtol=0.0,
                atol=0.0,
            )

        reference_tokens = torch.randn(
            257,
            8,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        sonic_tokens = reference_tokens.detach().clone().requires_grad_(True)

        def consume_plan(plan, tokens):
            sorted_routes = tokens.index_select(0, plan.sorted_token_indices)
            sorted_routes = sorted_routes * plan.sorted_scores.float().unsqueeze(1)
            restored = sorted_routes.index_select(0, plan.inverse_order)
            return restored.view(257, 4, 8).sum(dim=1)

        reference_output = consume_plan(reference, reference_tokens)
        sonic_output = consume_plan(sonic, sonic_tokens)
        torch.testing.assert_close(sonic_output, reference_output, rtol=0.0, atol=0.0)
        reference_output.square().sum().backward()
        sonic_output.square().sum().backward()
        torch.testing.assert_close(
            sonic_tokens.grad,
            reference_tokens.grad,
            # Both backwards atomically accumulate four route contributions
            # into each token. Separate CUDA launches may reduce those
            # contributions in a different order despite identical metadata.
            rtol=2e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            sonic_scores.grad,
            reference_scores.grad,
            rtol=0.0,
            atol=0.0,
        )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "DeepGEMM integration parity requires remote CUDA execution",
    )
    def test_deepgemm_fp8_preserves_expert_lora_and_input_gradients(self) -> None:
        from mirai.core.moe.runtime.gemm import probe_backend

        device = torch.device("cuda")
        verdict = probe_backend("deepgemm_fp8", device=device)
        if not verdict.available:
            self.skipTest(verdict.reason)
        torch.manual_seed(12)
        module = CompressedGroupedExperts.from_empty(
            num_experts=3,
            group_sizes=128,
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=3,
            quant_format="fp8",
        )
        for key in ("w1", "w2", "w3"):
            module.load_dense_weight(key, torch.randn(3, 256, 256) * 0.04)
            adapter = module.attach_expert_lora(
                tensor_name=key,
                adapter_name=key,
                rank=4,
                alpha=4.0,
            )
            with torch.no_grad():
                adapter.lora_b.normal_(std=0.02)
        module.to(device)
        base_scores = torch.softmax(
            torch.randn(29, 2, device=device, dtype=torch.float32), dim=-1
        ).to(torch.bfloat16)
        # Expert 1 is empty; expert 0 is deliberately much heavier than expert 2.
        indices = torch.zeros((29, 2), device=device, dtype=torch.long)
        indices[::7, 1] = 2
        base_tokens = torch.randn(29, 256, device=device, dtype=torch.bfloat16)

        def run(backend: str, token_chunk: int = 0):
            module.zero_grad(set_to_none=True)
            tokens = base_tokens.clone().requires_grad_(True)
            scores = base_scores.clone().requires_grad_(True)
            with active_moe_optimization_policy(
                MoEOptimizationPolicy(moe_gemm_backend_forward=backend)
            ):
                if token_chunk:
                    output = torch.cat(
                        [
                            module.run_direct_routed(
                                tokens[start : start + token_chunk],
                                scores[start : start + token_chunk],
                                indices[start : start + token_chunk],
                            )
                            for start in range(0, int(tokens.shape[0]), token_chunk)
                        ]
                    )
                else:
                    output = module.run_direct_routed(tokens, scores, indices)
                loss = output.float().square().mean()
                loss.backward()
            adapter_grads = {
                (key, name): getattr(module.expert_lora[key], name).grad.detach().clone()
                for key in ("w1", "w2", "w3")
                for name in ("lora_a", "lora_b")
            }
            return (
                output.detach(), loss.detach(), tokens.grad.detach(),
                scores.grad.detach(), adapter_grads,
            )

        reference = run("auto")
        native = run("deepgemm_fp8")
        native_chunked = run("deepgemm_fp8", token_chunk=7)
        torch.testing.assert_close(native[0], reference[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(native[1], reference[1], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(native[2], reference[2], rtol=3e-2, atol=3e-3)
        torch.testing.assert_close(native[3], reference[3], rtol=3e-2, atol=3e-3)
        for key in reference[4]:
            torch.testing.assert_close(
                native[4][key], reference[4][key], rtol=5e-2, atol=5e-3
            )
        for candidate, expected, rtol, atol in (
            (native_chunked[0], native[0], 3e-2, 3e-2),
            (native_chunked[1], native[1], 3e-2, 3e-2),
            (native_chunked[2], native[2], 3e-2, 3e-3),
            (native_chunked[3], native[3], 3e-2, 3e-3),
        ):
            torch.testing.assert_close(candidate, expected, rtol=rtol, atol=atol)
        for key in native[4]:
            torch.testing.assert_close(
                native_chunked[4][key], native[4][key], rtol=5e-2, atol=5e-3
            )

    @unittest.skipIf(torch is None, "torch is required")
    def test_mixed_precision_experts_match_per_host_reference(self) -> None:
        class DenseExperts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 2
                self.w1 = nn.Parameter(torch.randn(2, 64, 64), requires_grad=False)
                self.w2 = nn.Parameter(torch.randn(2, 64, 64), requires_grad=False)
                self.w3 = nn.Parameter(torch.randn(2, 64, 64), requires_grad=False)

        torch.manual_seed(13)
        mixed = MixedPrecisionGroupedExperts(
            DenseExperts(),
            formats=("int8", "mxfp4"),
        )
        tokens = torch.randn(5, 64, requires_grad=True)
        scores = torch.softmax(torch.randn(5, 2), dim=-1).requires_grad_(True)
        indices = torch.tensor([[0, 1]] * 5)
        actual = mixed.run_direct_routed(tokens, scores, indices)
        expected = torch.zeros_like(actual)
        for expert_id, host in enumerate(mixed.hosts):
            local = host.run_direct_routed(
                tokens,
                torch.ones(5, 1),
                torch.zeros(5, 1, dtype=torch.long),
            )
            expected = expected + local * scores[:, expert_id : expert_id + 1]
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(tokens.grad).all()))
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        payload = mixed.state_dict()
        restored = MixedPrecisionGroupedExperts(
            DenseExperts(),
            formats=("int8", "mxfp4"),
        )
        restored.load_state_dict(payload)
        torch.testing.assert_close(
            restored.run_direct_routed(
                tokens.detach(), scores.detach(), indices
            ),
            actual.detach(),
        )

    @classmethod
    def setUpClass(cls) -> None:
        register_builtin_components()

    def _lingbot_model_config(self, *, path: str = "./models/lingbot_video") -> ModelConfig:
        return ModelConfig(
            type="lingbot-video",
            path=path,
            params=ModelParams(
                variant="tiny-video",
                strict_native_assets=False,
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
        )

    def test_expert_tensor_spec_rejects_duplicate_names(self) -> None:
        spec = ExpertTensorSpec(
            name="blocks.0.ffn.experts.w1",
            owner_module="blocks.0.ffn.experts",
            tensor_name="w1",
            role="gate",
            layout=("expert", "out", "in"),
            shape=(4, 32, 16),
        )

        with self.assertRaisesRegex(ValueError, "Duplicate expert tensor specs"):
            validate_expert_tensor_specs([spec, spec])

    def test_lingbot_exposes_concrete_expert_tensor_specs(self) -> None:
        pipeline = LingBotVideoPipeline(self._lingbot_model_config())
        specs = pipeline.get_expert_tensor_specs()
        by_name = {spec.name: spec for spec in specs}

        self.assertEqual(len(specs), 4)
        self.assertTrue(by_name["blocks.0.ffn.router.weight"].router)
        self.assertEqual(by_name["blocks.0.ffn.router.weight"].layout, ("out", "in"))
        self.assertEqual(by_name["blocks.0.ffn.experts.w1"].role, "gate")
        self.assertEqual(by_name["blocks.0.ffn.experts.w2"].role, "down")
        self.assertEqual(by_name["blocks.0.ffn.experts.w3"].role, "up")
        self.assertEqual(by_name["blocks.0.ffn.experts.w1"].layout, ("expert", "out", "in"))
        self.assertEqual(by_name["blocks.0.ffn.experts.w1"].shape, (4, 32, 16))

    def test_compressed_weights_active_and_chunked_access_match_full_dequant(self) -> None:
        torch.manual_seed(5)
        pipeline = LingBotVideoPipeline(self._lingbot_model_config())
        base = pipeline.transformer.blocks[0].ffn.experts
        tokens = torch.randn(6, 16)
        counts = torch.tensor([2, 0, 3, 1], dtype=torch.int64)

        full = CompressedGroupedExperts(
            base,
            group_sizes=16,
            expert_weight_access="full_dequant",
        )
        active = CompressedGroupedExperts(
            base,
            group_sizes=16,
            expert_weight_access="active_dequant",
        )
        chunked = CompressedGroupedExperts(
            base,
            group_sizes=16,
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
        )

        expected = full.run_for_loop(tokens, counts)
        torch.testing.assert_close(active.run_for_loop(tokens, counts), expected)
        torch.testing.assert_close(chunked.run_for_loop(tokens, counts), expected)

    def test_runtime_rejects_expert_policy_without_model_support(self) -> None:
        cfg = TrainingConfig(
            model=ModelConfig(type="lingbot-video", path="./models/lingbot_video"),
            training=TrainingSection(seed=7, batch_size=1),
            memory=MemoryConfig(expert_weight_access="active_dequant"),
        )

        with self.assertRaisesRegex(ValueError, "does not implement memory.expert_weight_access"):
            initialize_trainer_runtime(
                config=cfg,
                pipeline=_NoMemorySupportPipeline(cfg.model),
            )

    def test_lingbot_strict_load_quantizes_experts_on_load(self) -> None:
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("safetensors not installed")
        seed_pipeline = LingBotVideoPipeline(self._lingbot_model_config())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transformer_dir = root / "transformer"
            transformer_dir.mkdir(parents=True)
            (transformer_dir / "config.json").write_text(
                __import__("json").dumps(seed_pipeline.transformer_config),
                encoding="utf-8",
            )
            save_file(
                seed_pipeline.transformer.state_dict(),
                str(transformer_dir / "diffusion_pytorch_model.safetensors"),
            )
            strict_params = self._lingbot_model_config(path=str(root)).params
            strict_params.variant = "lingbot-video-moe-30b-a3b"
            strict_params.strict_native_assets = True
            trainer = Trainer(
                TrainingConfig(
                    model=ModelConfig(
                        type="lingbot-video",
                        path=str(root),
                        params=strict_params,
                    ),
                    training=TrainingSection(seed=13, batch_size=1, gradient_checkpointing="off"),
                    memory=MemoryConfig(
                        frozen_weight_quantization="int8",
                        frozen_weight_quantization_strategy="compressed_weights",
                        expert_weight_access="chunked_dequant",
                        expert_dequant_chunk_size=2,
                        quantize_experts_on_load=True,
                    ),
                )
            )

        report = trainer.pipeline.get_quantized_frozen_weight_report()
        self.assertIsNotNone(report)
        self.assertTrue(report["quantized_experts_on_load"])
        self.assertEqual(report["expert_weight_access"], "chunked_dequant")
        self.assertGreater(report["grouped_expert_modules"], 0)
        self.assertFalse(
            any(name.endswith("experts.w1") for name, _ in trainer.pipeline.named_parameters())
        )
        self.assertEqual(
            sum(1 for module in trainer.pipeline.modules() if isinstance(module, nn.Linear)),
            0,
        )
        self.assertGreater(
            sum(
                1
                for module in trainer.pipeline.modules()
                if isinstance(module, CompressedGroupedExperts) and module.prefers_for_loop()
            ),
            0,
        )
        loss, _raw = trainer.compute_loss(
            {
                "latents": torch.randn(1, 2, 1, 2, 2),
                "text_embeds": torch.randn(1, 3, 16),
            }
        )
        self.assertTrue(math.isfinite(float(loss.detach().cpu().item())))


class RoutedDispatchOperationTests(unittest.TestCase):
    def test_duplicate_routes_match_reference_forward_and_backward(self) -> None:
        torch.manual_seed(7)
        tokens = torch.randn(3, 4, requires_grad=True)
        indices = torch.tensor([2, 0, 2, 1, 0])
        scores = torch.randn(5, requires_grad=True)
        sorted_tokens = permute_routed_tokens(tokens, indices)
        expert_output = sorted_tokens.square()
        actual = combine_routed_tokens(
            expert_output,
            scores,
            indices,
            output_rows=3,
            output_dtype=tokens.dtype,
        )
        reference = torch.zeros_like(actual).index_add(
            0, indices, expert_output * scores.unsqueeze(1)
        )
        torch.testing.assert_close(actual, reference)
        grad = torch.randn_like(actual)
        actual.backward(grad, retain_graph=True)
        actual_token_grad = tokens.grad.detach().clone()
        actual_score_grad = scores.grad.detach().clone()
        tokens.grad = None
        scores.grad = None
        reference.backward(grad)
        torch.testing.assert_close(tokens.grad, actual_token_grad)
        torch.testing.assert_close(scores.grad, actual_score_grad)


if __name__ == "__main__":
    unittest.main()
