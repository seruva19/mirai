from __future__ import annotations

import copy
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from mirai.config.schema import (
    AdapterConfig,
    MemoryConfig,
    ModelConfig,
    ModelParams,
    StrategyConfig,
    TrainingConfig,
    TrainingSection,
)
from mirai.core.builtins import register_builtin_components
from mirai.core.models.compressed_weights import CompressedGroupedExperts
from mirai.core.models.compressed_weights import CompressedLinear
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.models.adapters.lora import (
    LoRAExpertTensorParametrization,
    LoRALinear,
    lora_state_dict,
)
from mirai.core.models.adapters.lora_adaptive_rank import allocate_adaptive_ranks
from mirai.core.models.adapters.lora_adaptive_rank import save_adaptive_rank_plan
from mirai.core.training.calibration.gora import maybe_initialize_gora
from mirai.core.training.calibration.esft import maybe_initialize_esft
from mirai.core.training.data.curriculum import CurriculumSchedule
from mirai.core.training.lifecycle.session_components import (
    build_training_runtime_components,
)
from mirai.core.training.lifecycle.session_state import init_training_run_state
from mirai.core.training.optim.lora_muon import LoRAMuon
from mirai.core.training.optim.lora_pro import LoRAProAdamW
from mirai.core.training.trainer import Trainer
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoMLP
from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoRouter
from mirai.vendors.lingbot_video.transformer_lingbot_video import _block_router_auxiliary_terms

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class LingBotLoRAResidencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_builtin_components()

    def _config(
        self,
        *,
        target_preset: str = "attn_only",
        gradient_checkpointing: str = "off",
        blocks_to_swap: int = 0,
        quantized: bool = False,
        expert_weight_access: str = "auto",
        block_swap_transfer_strategy: str = "per_tensor",
    ) -> TrainingConfig:
        return TrainingConfig(
            model=ModelConfig(
                type="lingbot-video",
                path="./nonexistent/lingbot-video",
                params=ModelParams(
                    variant="tiny-video",
                    flow_shift=3.0,
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
            ),
            strategy=StrategyConfig(type="text_to_video", params={}),
            training=TrainingSection(
                seed=23,
                batch_size=1,
                max_steps=1,
                gradient_checkpointing=gradient_checkpointing,
                blocks_to_swap=blocks_to_swap,
            ),
            adapter=AdapterConfig(
                target_preset=target_preset,
                rank=4,
                alpha=4.0,
            ),
            memory=MemoryConfig(
                frozen_weight_quantization="int8" if quantized else "none",
                frozen_weight_quantization_strategy="compressed_weights" if quantized else "disabled",
                expert_weight_access=expert_weight_access,
                weight_residency_strategy=(
                    "block_swap" if blocks_to_swap > 0 else "disabled"
                ),
                block_swap_transfer_strategy=block_swap_transfer_strategy,
            ),
        )

    def _batch(self) -> dict[str, object]:
        return {
            "latents": torch.randn(1, 2, 1, 1, 1),
            "text_embeds": torch.randn(1, 16),
        }

    def test_lora_injection_targets_attention_and_shared_mlp(self) -> None:
        trainer = Trainer(self._config(target_preset="attn_shared_mlp"))

        trainable = dict(trainer.pipeline.get_named_trainable_parameters())
        lora_modules = [
            name
            for name, module in trainer.pipeline.transformer.named_modules()
            if isinstance(module, LoRALinear)
        ]
        loss, _metrics = trainer.compute_loss(self._batch())
        loss.backward()

        self.assertIn("blocks.0.attn.to_q.lora_a", trainable)
        self.assertIn("blocks.0.ffn.shared_experts.gate_proj.lora_b", trainable)
        self.assertTrue(all(".lora_" in name for name in trainable))
        self.assertIn("blocks.0.attn.to_q", lora_modules)
        self.assertTrue(math.isfinite(float(loss.detach().cpu().item())))
        self.assertTrue(
            any(
                param.grad is not None
                and bool(torch.isfinite(param.grad).all())
                for name, param in trainable.items()
                if name.endswith(".lora_b")
            )
        )

    def test_regional_compile_uses_provider_owned_transformer_blocks(self) -> None:
        config = self._config()
        config.training.compile = True
        config.training.compile_scope = "regional"
        config.training.compile_dynamic = True
        config.training.compile_token_buckets = [8, 32]
        with mock.patch(
            "mirai.core.training.runtime.compilation.torch.compile",
            side_effect=lambda target, **_kwargs: target,
        ) as compiler:
            trainer = Trainer(config)

        diagnostics = trainer.get_compilation_diagnostics()
        self.assertTrue(trainer.compile_enabled)
        self.assertEqual(compiler.call_count, len(trainer.pipeline.transformer.blocks))
        self.assertEqual(
            diagnostics["regions"],
            [
                f"transformer.blocks.{idx}"
                for idx in range(len(trainer.pipeline.transformer.blocks))
            ],
        )
        self.assertEqual(
            diagnostics["token_buckets"]["upper_bounds"],
            [8, 32],
        )

    def test_moe_token_chunking_installs_and_preserves_adapter_gradients(self) -> None:
        config = self._config(target_preset="routed_experts_only")
        config.training.moe_token_chunk_size = 1
        trainer = Trainer(config)

        policies = {
            id(block.ffn._mirai_token_chunk_policy)
            for block in trainer.pipeline.transformer.blocks
        }
        self.assertEqual(len(policies), 1)
        self.assertIn("moe_token_chunking", trainer.training_policies.active_names)

        loss, _ = trainer.compute_loss(self._batch())
        loss.backward()

        expert_gradients = [
            parameter.grad
            for name, parameter in trainer.pipeline.get_named_trainable_parameters()
            if "ffn.experts" in name
        ]
        self.assertTrue(expert_gradients)
        self.assertTrue(
            all(
                gradient is not None and bool(torch.isfinite(gradient).all())
                for gradient in expert_gradients
            )
        )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA required",
    )
    def test_layered_activation_offload_uses_provider_regions(self) -> None:
        config = self._config(target_preset="attn_router_routed_experts")
        config.model.params.num_layers = 3
        config.training.activation_cpu_offload = True
        config.training.activation_cpu_offload_min_mib = 0
        config.training.activation_cpu_offload_max_gib = 0.25
        config.training.activation_cpu_offload_pin_memory = True
        config.training.activation_cpu_offload_defer_layers = 1
        config.training.activation_cpu_offload_prefetch_layers = 1
        config.training.activation_cpu_offload_view_replay = True
        trainer = Trainer(config)
        trainer.pipeline.to(device="cuda", dtype=torch.bfloat16)
        batch = {
            "latents": torch.randn(
                1, 2, 1, 1, 1, device="cuda", dtype=torch.bfloat16
            ),
            "text_embeds": torch.randn(
                1, 16, device="cuda", dtype=torch.bfloat16
            ),
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = trainer.compute_loss(batch)
        loss.backward()
        torch.cuda.synchronize()

        diagnostics = trainer.get_activation_offload_diagnostics()
        self.assertEqual(
            diagnostics["regions"],
            [
                f"transformer.blocks.{idx}"
                for idx in range(config.model.params.num_layers)
            ],
        )
        self.assertGreater(diagnostics["offloaded_tensors"], 0)
        self.assertGreater(diagnostics["prefetched_tensors"], 0)
        self.assertEqual(diagnostics["reserved_bytes"], 0)
        self.assertTrue(math.isfinite(float(loss.detach().cpu().item())))
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in trainer.pipeline.get_trainable_parameters()
            )
        )

    def test_static_target_allocation_and_rslora_are_applied(self) -> None:
        config = self._config(target_preset="attn_only")
        config.adapter.rank_pattern = {"blocks.0.attn.to_q": 2}
        config.adapter.alpha_pattern = {"blocks.0.attn.to_q": 8.0}
        config.adapter.use_rslora = True
        trainer = Trainer(config)

        modules = dict(trainer.pipeline.transformer.named_modules())
        to_q = modules["blocks.0.attn.to_q"]
        to_k = modules["blocks.0.attn.to_k"]
        self.assertIsInstance(to_q, LoRALinear)
        self.assertEqual((to_q.rank, float(to_q.lora_alpha)), (2, 8.0))
        self.assertTrue(to_q.use_rslora)
        self.assertEqual((to_k.rank, float(to_k.lora_alpha)), (4, 4.0))
        self.assertTrue(to_k.use_rslora)
        allocations = dict(
            (name, (rank, alpha, rule))
            for name, rank, alpha, rule in trainer.pipeline._lora_report.allocations
        )
        self.assertEqual(
            allocations["blocks.0.attn.to_q"],
            (2, 8.0, "alpha_over_sqrt_rank"),
        )

    def test_adaptive_rank_plan_shapes_are_fixed_with_zero_initial_delta(self) -> None:
        reference = Trainer(self._config(target_preset="attn_only"))
        names = tuple(reference.pipeline._lora_report.matched_modules)
        spectra = {
            name: ([9.0, 8.0, 7.0, 6.0] if name.endswith("to_q") else [5.0, 4.0, 3.0, 2.0])
            for name in names
        }
        lineage = {
            "dataset_snapshot_id": "dataset-v1",
            "model_snapshot_id": "model-v1",
            "config_snapshot_id": "config-v1",
        }
        plan = allocate_adaptive_ranks(
            spectra,
            rank_budget=len(names) + 2,
            lineage=lineage,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank-plan.json"
            save_adaptive_rank_plan(path, plan)
            config = self._config(target_preset="attn_only")
            config.adapter.adaptive_rank_plan_path = str(path)
            config.adapter.rank_budget = plan.rank_budget
            adaptive = Trainer(config)

        modules = dict(adaptive.pipeline.transformer.named_modules())
        for name, rank in plan.ranks.items():
            self.assertEqual(modules[name].rank, rank)
            self.assertTrue(bool(torch.count_nonzero(modules[name].lora_b) == 0))
        batch = self._batch()
        reference_loss, _ = reference.compute_loss(batch)
        adaptive_loss, _ = adaptive.compute_loss(batch)
        torch.testing.assert_close(adaptive_loss, reference_loss, rtol=0.0, atol=0.0)
        adaptive.pipeline.validate_adapter_artifact_lineage(
            dataset_snapshot_id="dataset-v1",
            model_snapshot_id="model-v1",
            config_snapshot_id="config-v1",
        )
        with self.assertRaisesRegex(ValueError, "model_snapshot_id mismatch"):
            adaptive.pipeline.validate_adapter_artifact_lineage(
                dataset_snapshot_id="dataset-v1",
                model_snapshot_id="other",
                config_snapshot_id="config-v1",
            )

    def test_adaptive_rank_plan_optimizer_resume_is_exact(self) -> None:
        probe = Trainer(self._config(target_preset="attn_only"))
        names = tuple(probe.pipeline._lora_report.matched_modules)
        plan = allocate_adaptive_ranks(
            {
                name: tuple(float(8 - index) for index in range(4))
                for name in names
            },
            rank_budget=len(names) + 3,
            lineage={
                "dataset_snapshot_id": "dataset-v1",
                "model_snapshot_id": "model-v1",
                "config_snapshot_id": "config-v1",
            },
        )

        def step(trainer: Trainer, optimizer: torch.optim.Optimizer) -> None:
            optimizer.zero_grad(set_to_none=True)
            loss = sum(
                parameter.float().square().mean()
                for parameter in trainer.get_trainable_parameters()
            )
            loss.backward()
            optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank-plan.json"
            save_adaptive_rank_plan(path, plan)
            config = self._config(target_preset="attn_only")
            config.adapter.adaptive_rank_plan_path = str(path)
            config.adapter.rank_budget = plan.rank_budget
            uninterrupted = Trainer(config)
            optimizer = torch.optim.AdamW(
                uninterrupted.get_trainable_parameters(), lr=1e-3
            )
            step(uninterrupted, optimizer)
            adapter_state = copy.deepcopy(uninterrupted.pipeline.state_dict())
            optimizer_state = copy.deepcopy(optimizer.state_dict())

            resumed = Trainer(config)
            resumed.pipeline.load_adapter_state(adapter_state)
            resumed_optimizer = torch.optim.AdamW(
                resumed.get_trainable_parameters(), lr=1e-3
            )
            resumed_optimizer.load_state_dict(optimizer_state)
            step(uninterrupted, optimizer)
            step(resumed, resumed_optimizer)

        uninterrupted_params = dict(uninterrupted.pipeline.get_named_trainable_parameters())
        resumed_params = dict(resumed.pipeline.get_named_trainable_parameters())
        self.assertEqual(uninterrupted_params.keys(), resumed_params.keys())
        for name in uninterrupted_params:
            torch.testing.assert_close(
                resumed_params[name], uninterrupted_params[name], rtol=0.0, atol=0.0
            )

    def test_heterogeneous_rank_rejects_timestep_schedule_before_injection(self) -> None:
        config = self._config(target_preset="attn_only")
        config.adapter.rank_pattern = {"blocks.0.attn.to_q": 2}
        config.adapter.timestep_rank_schedule = "tlora"
        with self.assertRaisesRegex(ValueError, "Heterogeneous adapter.rank_pattern"):
            Trainer(config)

    def test_lora_injection_targets_routed_expert_tensors(self) -> None:
        trainer = Trainer(self._config(target_preset="routed_experts_only"))

        trainable = dict(trainer.pipeline.get_named_trainable_parameters())
        expert_adapters = [
            module.adapter_name
            for module in trainer.pipeline.transformer.modules()
            if isinstance(module, LoRAExpertTensorParametrization)
        ]
        loss, _metrics = trainer.compute_loss(self._batch())
        loss.backward()

        self.assertIn("blocks.0.ffn.experts.w1", expert_adapters)
        self.assertIn("blocks.0.ffn.experts.w2", expert_adapters)
        self.assertIn("blocks.0.ffn.experts.w3", expert_adapters)
        self.assertTrue(
            any(
                name.endswith("parametrizations.w1.0.lora_b")
                and param.grad is not None
                and bool(torch.isfinite(param.grad).all())
                for name, param in trainable.items()
            )
        )
        self.assertTrue(math.isfinite(float(loss.detach().cpu().item())))

    def test_router_lora_receives_auxiliary_gradient(self) -> None:
        trainer = Trainer(self._config(target_preset="attn_router_routed_experts"))
        loss, raw = trainer.compute_loss(self._batch())
        loss.backward()
        router_b = [
            param
            for name, param in trainer.pipeline.get_named_trainable_parameters()
            if "router" in name and name.endswith("lora_b")
        ]
        self.assertIn("moe_load_balance", raw["auxiliary_losses"])
        self.assertTrue(router_b)
        self.assertTrue(any(float(param.grad.abs().sum()) > 0.0 for param in router_b))

    def test_scale_zero_restores_base_and_optimizer_changes_only_adapters(self) -> None:
        config = self._config(target_preset="attn_router_routed_experts")
        config.adapter.expert_tensor_lora_backend = "activation"
        trainer = Trainer(config)
        batch = self._batch()
        trainable = dict(trainer.pipeline.get_named_trainable_parameters())
        frozen = {
            name: parameter
            for name, parameter in trainer.pipeline.transformer.named_parameters()
            if not parameter.requires_grad
        }
        frozen_before = {
            name: parameter.detach().clone()
            for name, parameter in list(frozen.items())[:8]
        }
        linear = next(
            module
            for module in trainer.pipeline.transformer.modules()
            if isinstance(module, LoRALinear)
        )
        linear_input = torch.randn(2, linear.in_features)
        base_output = linear(linear_input).detach().clone()
        with torch.no_grad():
            for name, parameter in trainable.items():
                if name.endswith("lora_b"):
                    parameter.normal_()

        trainer.pipeline.set_lora_scale(0.0)
        scale_zero_output = linear(linear_input)
        torch.testing.assert_close(scale_zero_output, base_output, rtol=0, atol=0)

        trainer.pipeline.set_lora_scale(1.0)
        scale_one_output = linear(linear_input)
        self.assertFalse(torch.equal(scale_one_output, base_output))

        before_update = {
            name: parameter.detach().clone()
            for name, parameter in trainable.items()
        }
        loss, _ = trainer.compute_loss(batch)
        loss.backward()
        optimizer = torch.optim.SGD(trainable.values(), lr=1e-3)
        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(before_update[name], parameter.detach())
                for name, parameter in trainable.items()
            )
        )
        for name, before in frozen_before.items():
            torch.testing.assert_close(frozen[name], before, rtol=0, atol=0)

    def test_aggressive_checkpoint_preserves_router_auxiliary_gradient(self) -> None:
        trainer = Trainer(
            self._config(
                target_preset="attn_router_routed_experts",
                gradient_checkpointing="aggressive",
            )
        )
        loss, raw = trainer.compute_loss(self._batch())
        loss.backward()
        router_b = [
            param
            for name, param in trainer.pipeline.get_named_trainable_parameters()
            if "router" in name and name.endswith("lora_b")
        ]

        self.assertIn("moe_load_balance", raw["auxiliary_losses"])
        self.assertTrue(router_b)
        self.assertTrue(any(float(param.grad.abs().sum()) > 0.0 for param in router_b))

    def test_aggressive_checkpoint_supports_all_linear_timestep_lora(self) -> None:
        trainer = Trainer(
            self._config(
                target_preset="all_linear",
                gradient_checkpointing="aggressive",
            )
        )

        loss, _raw = trainer.compute_loss(self._batch())
        loss.backward()

        time_modulation_grads = [
            param.grad
            for name, param in trainer.pipeline.get_named_trainable_parameters()
            if name.startswith("time_modulation")
        ]
        self.assertTrue(time_modulation_grads)
        self.assertTrue(any(grad is not None for grad in time_modulation_grads))

    def test_router_graph_state_is_released_after_backward_boundary(self) -> None:
        trainer = Trainer(
            self._config(
                target_preset="attn_router_routed_experts",
                gradient_checkpointing="aggressive",
            )
        )
        loss, _raw = trainer.compute_loss(self._batch())
        self.assertEqual(trainer.pipeline._last_auxiliary_losses, {})

        loss.backward()
        trainer.pipeline.finish_backward_offloads()

        routers = [
            module
            for module in trainer.pipeline.transformer.modules()
            if isinstance(module, LingBotVideoRouter)
        ]
        self.assertTrue(routers)
        for router in routers:
            self.assertIsNone(router.training_logits)
            self.assertIsNone(router.training_scores)
            self.assertIsNone(router.training_top_indices)

    def test_sequence_auxiliary_loss_uses_unbiased_per_video_assignments(self) -> None:
        trainer = Trainer(self._config(target_preset="attn_router_routed_experts"))
        block = trainer.pipeline.transformer.blocks[0]
        router = block.ffn.router
        block._mirai_moe_aux_loss_type = "sequence"
        probabilities = torch.tensor(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.6, 0.2, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.6, 0.2],
            ],
            requires_grad=True,
        )
        router.training_scores = probabilities
        router.training_logits = probabilities.logit().requires_grad_(True)
        router.training_top_indices = torch.zeros((4, 2), dtype=torch.long)
        router.training_unbiased_top_indices = torch.tensor(
            [[0, 1], [0, 1], [2, 3], [2, 3]], dtype=torch.long
        )
        router.training_batch_size = 2
        router.training_tokens_per_sample = 2

        balance, _z_loss = _block_router_auxiliary_terms(
            block, like=probabilities
        )
        normalized = probabilities / probabilities.sum(dim=-1, keepdim=True)
        probability_mean = normalized.reshape(2, 2, 4).mean(dim=1)
        expected_frequency = torch.tensor(
            [[2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 2.0]]
        )
        expected = (expected_frequency * probability_mean).sum(dim=-1).mean()

        torch.testing.assert_close(balance, expected)

    def test_online_router_bias_update_is_centered_and_checkpointed(self) -> None:
        config = self._config(target_preset="attn_router_routed_experts")
        config.model.params.moe_bias_update_rate = 0.1
        trainer = Trainer(config)
        name, router = next(
            (name, module)
            for name, module in trainer.pipeline.transformer.named_modules()
            if isinstance(module, LingBotVideoRouter)
        )
        trainer.pipeline._pending_router_loads = {
            name: torch.tensor([4.0, 2.0, 2.0, 0.0], dtype=torch.float64)
        }

        trainer.pipeline.finish_optimizer_step()

        expected = torch.tensor([-0.1, 0.0, 0.0, 0.1])
        torch.testing.assert_close(router.e_score_correction_bias, expected)
        state = trainer.pipeline.state_dict()
        bias_key = f"moe_router_bias.{name}"
        self.assertIn(bias_key, state)
        with torch.no_grad():
            router.e_score_correction_bias.zero_()
        trainer.pipeline.load_adapter_state(state)
        torch.testing.assert_close(router.e_score_correction_bias, expected)

    def test_aggressive_checkpoint_auxiliary_terms_exclude_dense_blocks(self) -> None:
        config = self._config(
            target_preset="attn_router_routed_experts",
            gradient_checkpointing="aggressive",
        )
        config.model.params.num_layers = 2
        trainer = Trainer(config)
        dense_ffn = LingBotVideoMLP(16, 64)
        dense_ffn.requires_grad_(False)
        trainer.pipeline.transformer.blocks[0].ffn = dense_ffn

        loss, raw = trainer.compute_loss(self._batch())
        terms = trainer.pipeline.transformer._mirai_checkpoint_router_auxiliary_terms

        self.assertEqual(len(terms), 1)
        self.assertIn("moe_load_balance", raw["auxiliary_losses"])
        self.assertTrue(math.isfinite(float(loss.detach().cpu().item())))

    def test_gradient_checkpointing_invokes_native_block_checkpoints(self) -> None:
        import torch.utils.checkpoint as checkpoint

        trainer = Trainer(self._config(gradient_checkpointing="standard"))
        calls: list[bool] = []
        original = checkpoint.checkpoint

        def fake_checkpoint(fn, *args, **kwargs):
            calls.append(bool(kwargs.get("use_reentrant") is False))
            return fn(*args)

        checkpoint.checkpoint = fake_checkpoint
        try:
            loss, _metrics = trainer.compute_loss(self._batch())
        finally:
            checkpoint.checkpoint = original

        self.assertGreater(len(calls), 0)
        self.assertTrue(all(calls))
        self.assertTrue(math.isfinite(float(loss.detach().cpu().item())))

    def test_timestep_modulation_stays_per_sample_on_single_gpu(self) -> None:
        trainer = Trainer(self._config(target_preset="attn_only"))
        observed: dict[str, tuple[int, ...]] = {}

        def capture(name):
            def hook(_module, inputs):
                observed[name] = tuple(inputs[0].shape)

            return hook

        handles = [
            trainer.pipeline.transformer.time_modulation.register_forward_pre_hook(
                capture("time")
            ),
            trainer.pipeline.transformer.norm_out_modulation.register_forward_pre_hook(
                capture("output")
            ),
        ]
        batch = {
            "latents": torch.randn(1, 2, 1, 2, 2),
            "text_embeds": torch.randn(1, 16),
        }
        try:
            loss, _metrics = trainer.compute_loss(batch)
            loss.backward()
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(observed["time"][0], 1)
        self.assertEqual(observed["output"][0], 1)
        self.assertTrue(math.isfinite(float(loss.detach().cpu().item())))

    def test_lingbot_state_dict_is_adapter_only_after_lora_config(self) -> None:
        trainer = Trainer(self._config(target_preset="attn_shared_mlp"))
        with torch.no_grad():
            for name, param in trainer.pipeline.get_named_trainable_parameters():
                if name.endswith(".lora_b"):
                    param.add_(torch.randn_like(param))
        state = trainer.pipeline.state_dict()
        reloaded = Trainer(self._config(target_preset="attn_shared_mlp"))

        reloaded.pipeline.load_adapter_state(state)
        roundtrip = reloaded.pipeline.state_dict()

        self.assertNotIn("transformer", state)
        self.assertEqual(state["model_type"], "lingbot-video")
        self.assertIn("blocks.0.attn.to_q.lora_a", state)
        self.assertIn("blocks.0.ffn.shared_experts.down_proj.lora_b", state)
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(roundtrip[key], value)

    def test_lingbot_expert_tensor_lora_state_roundtrips(self) -> None:
        trainer = Trainer(self._config(target_preset="routed_experts_only"))
        with torch.no_grad():
            for name, param in trainer.pipeline.get_named_trainable_parameters():
                if name.endswith(".lora_b"):
                    param.add_(torch.randn_like(param))
        state = trainer.pipeline.state_dict()
        reloaded = Trainer(self._config(target_preset="routed_experts_only"))

        reloaded.pipeline.load_adapter_state(state)
        roundtrip = reloaded.pipeline.state_dict()

        self.assertIn("blocks.0.ffn.experts.w1.lora_a", state)
        self.assertIn("blocks.0.ffn.experts.w2.lora_b", state)
        self.assertNotIn("transformer", state)
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(roundtrip[key], value)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA required",
    )
    def test_cuda_dora_dense_and_grouped_lifecycle_roundtrips(self) -> None:
        config = self._config(target_preset="attn_routed_experts")
        config.adapter.use_dora = True
        trainer = Trainer(config)
        trainer.pipeline.to(device="cuda", dtype=torch.bfloat16)
        trainable = dict(trainer.pipeline.get_named_trainable_parameters())
        magnitudes = {
            name: parameter
            for name, parameter in trainable.items()
            if name.endswith("dora_magnitude")
        }
        self.assertTrue(magnitudes)
        self.assertTrue(
            any(
                isinstance(module, LoRALinear) and module.use_dora
                for module in trainer.pipeline.transformer.modules()
            )
        )
        self.assertTrue(
            any(
                isinstance(module, LoRAExpertTensorParametrization)
                and module.use_dora
                for module in trainer.pipeline.transformer.modules()
            )
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in magnitudes.items()
        }
        optimizer = torch.optim.AdamW(trainable.values(), lr=1e-2)
        batch = {
            "latents": torch.randn(
                1,
                2,
                1,
                1,
                1,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "text_embeds": torch.randn(
                1,
                16,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        }

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = trainer.compute_loss(batch)
        loss.backward()
        for parameter in magnitudes.values():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in magnitudes.items()
            )
        )

        state = trainer.pipeline.state_dict()
        restored = Trainer(config)
        restored.pipeline.to(device="cuda", dtype=torch.bfloat16)
        restored.pipeline.load_adapter_state(state)
        reloaded = restored.pipeline.state_dict()
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(reloaded[key], value))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is required for the native LoRA-Pro lifecycle contract.",
    )
    def test_cuda_lora_pro_dense_and_grouped_lifecycle_resumes(self) -> None:
        config = self._config(target_preset="attn_routed_experts")
        config.optimizer.type = "lora_pro_adamw"
        config.optimizer.lr = 0.01
        config.optimizer.weight_decay = 0.0
        trainer = Trainer(config)
        trainer.pipeline.to(device="cuda", dtype=torch.bfloat16)
        components = build_training_runtime_components(
            trainer=trainer,
            config=config,
        )
        optimizer = components.optimizer_result.optimizer
        self.assertIsInstance(optimizer, LoRAProAdamW)
        assert isinstance(optimizer, LoRAProAdamW)
        self.assertTrue(any(pair.lora_a.ndim == 2 for pair in optimizer.pairs))
        self.assertTrue(any(pair.lora_a.ndim == 3 for pair in optimizer.pairs))
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer.pipeline.get_named_trainable_parameters()
        }
        batch = {
            "latents": torch.randn(
                1,
                2,
                1,
                1,
                1,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "text_embeds": torch.randn(
                1,
                16,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = trainer.compute_loss(batch)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in trainer.pipeline.get_named_trainable_parameters()
            )
        )
        self.assertEqual(
            optimizer.estimated_state_bytes,
            sum(
                int(state["exp_avg"].numel() + state["exp_avg_sq"].numel())
                * torch.empty((), dtype=torch.float32).element_size()
                for state in optimizer.state.values()
                if "exp_avg" in state
            ),
        )
        self.assertTrue(
            all(
                state["exp_avg"].dtype == torch.float32
                and state["exp_avg_sq"].dtype == torch.float32
                for state in optimizer.state.values()
                if "exp_avg" in state
            )
        )

        adapter_state = trainer.pipeline.state_dict()
        optimizer_state = optimizer.state_dict()
        restored = Trainer(copy.deepcopy(config))
        restored.pipeline.to(device="cuda", dtype=torch.bfloat16)
        restored.pipeline.load_adapter_state(adapter_state)
        restored_components = build_training_runtime_components(
            trainer=restored,
            config=config,
        )
        restored_optimizer = restored_components.optimizer_result.optimizer
        self.assertIsInstance(restored_optimizer, LoRAProAdamW)
        assert isinstance(restored_optimizer, LoRAProAdamW)
        restored_optimizer.load_state_dict(optimizer_state)
        for source_pair, restored_pair in zip(
            optimizer.pairs,
            restored_optimizer.pairs,
            strict=True,
        ):
            source = optimizer.state.get(source_pair.lora_a)
            target = restored_optimizer.state.get(restored_pair.lora_a)
            if not source:
                self.assertFalse(target)
                continue
            self.assertEqual(target["exp_avg"].dtype, torch.float32)
            torch.testing.assert_close(target["exp_avg"], source["exp_avg"])
            torch.testing.assert_close(target["exp_avg_sq"], source["exp_avg_sq"])

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is required for the native LoRA-Muon lifecycle contract.",
    )
    def test_cuda_lora_muon_dense_and_grouped_lifecycle_resumes(self) -> None:
        config = self._config(target_preset="attn_routed_experts")
        config.optimizer.type = "lora_muon"
        config.optimizer.lr = 0.01
        config.optimizer.weight_decay = 0.01
        config.optimizer.weight_decay_filter = "none"
        config.optimizer.muon_momentum = 0.9
        trainer = Trainer(config)
        trainer.pipeline.to(device="cuda", dtype=torch.bfloat16)
        components = build_training_runtime_components(
            trainer=trainer,
            config=config,
        )
        optimizer = components.optimizer_result.optimizer
        self.assertIsInstance(optimizer, LoRAMuon)
        assert isinstance(optimizer, LoRAMuon)
        self.assertTrue(any(pair.lora_a.ndim == 2 for pair in optimizer.pairs))
        self.assertTrue(any(pair.lora_a.ndim == 3 for pair in optimizer.pairs))
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainer.pipeline.get_named_trainable_parameters()
        }
        batch = {
            "latents": torch.randn(
                1,
                2,
                1,
                1,
                1,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "text_embeds": torch.randn(
                1,
                16,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = trainer.compute_loss(batch)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in trainer.pipeline.get_named_trainable_parameters()
            )
        )
        self.assertTrue(
            all(
                state["moment_a"].dtype == torch.float32
                and state["moment_b"].dtype == torch.float32
                for state in optimizer.state.values()
                if "moment_a" in state
            )
        )

        adapter_state = trainer.pipeline.state_dict()
        optimizer_state = optimizer.state_dict()
        restored = Trainer(copy.deepcopy(config))
        restored.pipeline.to(device="cuda", dtype=torch.bfloat16)
        restored.pipeline.load_adapter_state(adapter_state)
        restored_components = build_training_runtime_components(
            trainer=restored,
            config=config,
        )
        restored_optimizer = restored_components.optimizer_result.optimizer
        self.assertIsInstance(restored_optimizer, LoRAMuon)
        assert isinstance(restored_optimizer, LoRAMuon)
        restored_optimizer.load_state_dict(optimizer_state)
        for source_pair, restored_pair in zip(
            optimizer.pairs,
            restored_optimizer.pairs,
            strict=True,
        ):
            source = optimizer.state.get(source_pair.lora_a)
            target = restored_optimizer.state.get(restored_pair.lora_a)
            if not source:
                self.assertFalse(target)
                continue
            self.assertEqual(target["moment_a"].dtype, torch.float32)
            torch.testing.assert_close(target["moment_a"], source["moment_a"])
            torch.testing.assert_close(target["moment_b"], source["moment_b"])

    def test_quantized_routed_expert_lora_receives_gradients_and_updates(self) -> None:
        trainer = Trainer(
            self._config(
                target_preset="routed_experts_only",
                quantized=True,
                expert_weight_access="active_dequant",
            )
        )
        trainable = dict(trainer.pipeline.get_named_trainable_parameters())
        b_params = {
            name: param
            for name, param in trainable.items()
            if "expert_lora" in name and name.endswith("lora_b")
        }
        before = {name: param.detach().clone() for name, param in b_params.items()}
        optimizer = torch.optim.AdamW(trainable.values(), lr=1e-2)
        expert_module = next(
            module
            for module in trainer.pipeline.transformer.modules()
            if isinstance(module, CompressedGroupedExperts)
        )
        tokens = torch.randn(6, 16)
        counts = torch.tensor([2, 1, 0, 3], dtype=torch.int64)
        loss = expert_module.run_for_loop(tokens, counts).square().mean()
        loss.backward()
        optimizer.step()

        self.assertTrue(b_params)
        self.assertTrue(
            any(
                param.grad is not None and bool(torch.isfinite(param.grad).all())
                for param in b_params.values()
            )
        )
        self.assertTrue(
            any(not torch.equal(before[name], param.detach()) for name, param in b_params.items())
        )

    def test_quantization_migration_preserves_rslora_rule(self) -> None:
        config = self._config(
            target_preset="routed_experts_only",
            quantized=True,
            expert_weight_access="active_dequant",
        )
        config.adapter.use_rslora = True
        trainer = Trainer(config)
        adapters = [
            adapter
            for module in trainer.pipeline.transformer.modules()
            if isinstance(module, CompressedGroupedExperts)
            for adapter in module.expert_lora.values()
        ]
        self.assertTrue(adapters)
        self.assertTrue(all(adapter.use_rslora for adapter in adapters))
        state = trainer.pipeline.state_dict()
        self.assertTrue(any(key.endswith(".lora_use_rslora") for key in state))

    def test_loftq_initializes_dense_and_expert_adapters_before_quantization(self) -> None:
        config = self._config(
            target_preset="attn_routed_experts",
            quantized=True,
            expert_weight_access="active_dequant",
        )
        config.adapter.lora_init = "loftq"
        trainer = Trainer(config)

        trainable = dict(trainer.pipeline.get_named_trainable_parameters())
        b_tensors = [
            value
            for name, value in trainable.items()
            if name.endswith("lora_b")
        ]
        self.assertTrue(b_tensors)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in b_tensors))
        self.assertTrue(any(bool(torch.count_nonzero(value)) for value in b_tensors))
        loss, _metrics = trainer.compute_loss(self._batch())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_loftq_requires_live_reference_weights_and_quantization(self) -> None:
        unquantized = self._config(target_preset="attn_only")
        unquantized.adapter.lora_init = "loftq"
        with self.assertRaisesRegex(ValueError, "requires frozen-weight quantization"):
            Trainer(unquantized)

        config = self._config(target_preset="attn_only")
        pipeline = LingBotVideoPipeline(config.model)
        pipeline.enable_quantized_frozen_weights("int8", strategy="compressed_weights")
        config.adapter.lora_init = "loftq"
        with self.assertRaisesRegex(ValueError, "requires adapter injection before"):
            pipeline.set_adapter_config(config.adapter)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is required for the native GoRA lifecycle contract.",
    )
    def test_gora_calibrates_before_optimizer_and_reloads_dynamic_ranks(self) -> None:
        config = self._config(target_preset="attn_routed_experts")
        config.adapter.lora_init = "gora"
        config.adapter.use_rslora = True
        config.adapter.gora_calibration_steps = 2
        config.adapter.gora_min_rank = 1
        config.adapter.gora_max_rank = 3
        config.training.compile = False
        trainer = Trainer(config)
        trainer.pipeline.to(device="cuda", dtype=torch.float32)
        records = [
            {
                "latent": torch.randn(2, 1, 1, 1),
                "text_embed": torch.randn(3, 16),
            }
            for _ in range(2)
        ]
        prepared = SimpleNamespace(
            train_records=records,
            temporal_base_ids=[],
            temporal_groups={},
        )
        rng = random.Random(config.training.seed)
        run_state = init_training_run_state(trainer=trainer, config=config)
        report = maybe_initialize_gora(
            trainer=trainer,
            config=config,
            prepared_data=prepared,
            compute_device=torch.device("cuda"),
            compute_dtype=torch.float32,
            curriculum=CurriculumSchedule.from_config({}),
            rng=rng,
            run_state=run_state,
            grad_accum=1,
        )
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.calibration_steps, 2)
        self.assertTrue(report.target_ranks)
        self.assertTrue(
            any(".experts." in name for name in report.target_ranks)
        )
        self.assertTrue(
            all(1 <= rank <= 3 for rank in report.target_ranks.values())
        )

        components = build_training_runtime_components(
            trainer=trainer,
            config=config,
        )
        optimizer_parameter_ids = {
            id(parameter)
            for group in components.optimizer_result.optimizer.param_groups
            for parameter in group["params"]
        }
        trainable = dict(trainer.pipeline.get_named_trainable_parameters())
        factor_parameters = {
            id(parameter)
            for name, parameter in trainable.items()
            if name.endswith(("lora_a", "lora_b"))
        }
        self.assertTrue(factor_parameters)
        self.assertTrue(factor_parameters <= optimizer_parameter_ids)
        loss, _ = trainer.compute_loss(
            {
                "latents": records[0]["latent"].unsqueeze(0).cuda(),
                "text_embeds": records[0]["text_embed"].unsqueeze(0).cuda(),
            }
        )
        loss.backward()
        components.optimizer_result.optimizer.step()
        self.assertTrue(torch.isfinite(loss))

        source_state = trainer.pipeline.state_dict()
        source_lora = lora_state_dict(trainer.pipeline.transformer)
        restored = Trainer(copy.deepcopy(config))
        restored.pipeline.to(device="cuda", dtype=torch.float32)
        restored.pipeline.load_adapter_state(source_state)
        restored_lora = lora_state_dict(restored.pipeline.transformer)
        self.assertEqual(set(source_lora), set(restored_lora))
        for key, value in source_lora.items():
            torch.testing.assert_close(restored_lora[key], value)
        restored_ranks = {
            (
                name
                if isinstance(module, LoRALinear)
                else str(module.adapter_name)
            ): int(module.rank)
            for name, module in restored.pipeline.transformer.named_modules()
            if isinstance(module, (LoRALinear, LoRAExpertTensorParametrization))
        }
        self.assertEqual(
            {name: int(rank) for name, rank in report.target_ranks.items()},
            restored_ranks,
        )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is required for the native ESFT lifecycle contract.",
    )
    def test_esft_calibrates_per_layer_before_optimizer_and_resumes(self) -> None:
        config = self._config()
        config.adapter.type = "selected_expert"
        config.adapter.expert_selection = "esft_gate"
        config.adapter.esft_selection_mass = 0.2
        config.adapter.esft_calibration_samples = 2
        config.optimizer.type = "selected_expert_adamw"
        config.optimizer.selected_expert_ids = []
        config.optimizer.weight_decay_filter = "none"
        config.optimizer.lr = 0.01
        config.training.compile = False

        torch.manual_seed(101)
        trainer = Trainer(config)
        trainer.pipeline.to(device="cuda", dtype=torch.float32)
        latent = torch.randn(2, 1, 1, 1)
        text_embed = torch.randn(3, 16)
        records = [
            {
                "latent": latent.clone(),
                "text_embed": text_embed.clone(),
            }
            for _ in range(2)
        ]
        prepared = SimpleNamespace(
            train_records=records,
            temporal_base_ids=[],
            temporal_groups={},
        )
        rng = random.Random(config.training.seed)
        report = maybe_initialize_esft(
            trainer=trainer,
            config=config,
            prepared_data=prepared,
            compute_device=torch.device("cuda"),
            compute_dtype=torch.float32,
            curriculum=CurriculumSchedule.from_config({}),
            rng=rng,
            run_state=SimpleNamespace(global_step=0),
            grad_accum=1,
        )
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.calibration_steps, 2)
        self.assertEqual(len(report.plan.selected_experts), 1)

        parameter_plan = trainer.pipeline.get_selected_expert_parameter_plan()
        trainable = dict(trainer.pipeline.get_named_trainable_parameters())
        self.assertEqual(set(parameter_plan), set(trainable))
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainable.items()
        }
        components = build_training_runtime_components(
            trainer=trainer,
            config=config,
        )
        batch = {
            "latents": records[0]["latent"].unsqueeze(0).cuda(),
            "text_embeds": records[0]["text_embed"].unsqueeze(0).cuda(),
        }
        loss, _ = trainer.compute_loss(batch)
        loss.backward()
        selected_grad_norms = {
            name: [
                float(parameter.grad[expert_id].float().norm().item())
                for expert_id in parameter_plan[name]
            ]
            for name, parameter in trainable.items()
            if parameter.grad is not None
        }
        self.assertTrue(
            any(
                value > 0.0
                for values in selected_grad_norms.values()
                for value in values
            ),
            f"ESFT-selected rows received no task gradient: {selected_grad_norms}",
        )
        components.optimizer_result.optimizer.step()
        self.assertTrue(torch.isfinite(loss))

        changed_selected = 0
        for name, parameter in trainable.items():
            selected = set(parameter_plan[name])
            for expert_id in range(int(parameter.shape[0])):
                changed = not torch.equal(
                    parameter[expert_id].detach(),
                    before[name][expert_id],
                )
                if expert_id in selected:
                    changed_selected += int(changed)
                else:
                    self.assertFalse(
                        changed,
                        f"unselected row changed: {name}[{expert_id}]",
                    )
        self.assertGreater(changed_selected, 0)

        model_state = trainer.pipeline.state_dict()
        optimizer_state = components.optimizer_result.optimizer.state_dict()
        restored = Trainer(copy.deepcopy(config))
        restored.pipeline.to(device="cuda", dtype=torch.float32)
        restored.pipeline.set_selected_expert_plan(
            report.plan.selected_experts
        )
        restored_components = build_training_runtime_components(
            trainer=restored,
            config=config,
        )
        restored.pipeline.load_state_dict(model_state)
        restored_components.optimizer_result.optimizer.load_state_dict(
            optimizer_state
        )
        restored_state = restored.pipeline.state_dict()
        self.assertEqual(set(model_state), set(restored_state))
        for key, value in model_state.items():
            if isinstance(value, dict):
                self.assertEqual(set(value), set(restored_state[key]))
                for nested_key, nested_value in value.items():
                    torch.testing.assert_close(
                        restored_state[key][nested_key],
                        nested_value,
                    )
            elif isinstance(value, torch.Tensor):
                torch.testing.assert_close(restored_state[key], value)
            else:
                self.assertEqual(restored_state[key], value)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is required for selected-expert AdaMuon lifecycle.",
    )
    def test_selected_expert_adamuon_cuda_lifecycle(self) -> None:
        config = self._config()
        config.adapter.type = "selected_expert"
        config.adapter.expert_selection = "all"
        config.optimizer.type = "selected_expert_adamuon"
        config.optimizer.selected_expert_ids = list(
            range(int(config.model.params.num_experts))
        )
        config.optimizer.weight_decay_filter = "none"
        config.optimizer.lr = 0.01
        config.optimizer.muon_nesterov = False
        config.optimizer.stochastic_rounding = True
        config.training.compile = False

        torch.manual_seed(211)
        trainer = Trainer(config)
        trainer.pipeline.to(device="cuda", dtype=torch.bfloat16)
        parameter_plan = trainer.pipeline.get_selected_expert_parameter_plan()
        trainable = dict(trainer.pipeline.get_named_trainable_parameters())
        self.assertEqual(set(parameter_plan), set(trainable))
        before = {
            name: parameter.detach().clone()
            for name, parameter in trainable.items()
        }
        components = build_training_runtime_components(
            trainer=trainer,
            config=config,
        )
        batch = {
            key: value.to(device="cuda")
            for key, value in self._batch().items()
        }
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = trainer.compute_loss(batch)
        loss.backward()
        components.optimizer_result.optimizer.step()
        self.assertTrue(torch.isfinite(loss))

        changed_selected = 0
        for name, parameter in trainable.items():
            selected = set(parameter_plan[name])
            for expert_id in range(int(parameter.shape[0])):
                changed = not torch.equal(
                    parameter[expert_id].detach(),
                    before[name][expert_id],
                )
                if expert_id in selected:
                    changed_selected += int(changed)
                else:
                    self.assertFalse(
                        changed,
                        f"unselected row changed: {name}[{expert_id}]",
                    )
            state = components.optimizer_result.optimizer.state[parameter]
            self.assertEqual(state["momentum_buffer"].dtype, torch.float32)
            self.assertEqual(state["second_moment"].dtype, torch.float32)
        self.assertGreater(changed_selected, 0)

        optimizer_state = components.optimizer_result.optimizer.state_dict()
        restored = Trainer(copy.deepcopy(config))
        restored.pipeline.to(device="cuda", dtype=torch.bfloat16)
        restored_components = build_training_runtime_components(
            trainer=restored,
            config=config,
        )
        restored_components.optimizer_result.optimizer.load_state_dict(
            optimizer_state
        )
        for state in restored_components.optimizer_result.optimizer.state.values():
            self.assertEqual(state["momentum_buffer"].dtype, torch.float32)
            self.assertEqual(state["second_moment"].dtype, torch.float32)

    def test_lora_fa_freezes_dense_expert_and_condenser_a_factors(self) -> None:
        config = self._config(
            target_preset="attn_routed_experts",
            quantized=True,
            expert_weight_access="active_dequant",
        )
        config.adapter.use_lora_fa = True
        config.adapter.condenser_rank = 2
        trainer = Trainer(config)

        named = dict(trainer.pipeline.transformer.named_parameters())
        a_factors = {
            name: parameter
            for name, parameter in named.items()
            if name.endswith(("lora_a", "cond_a"))
        }
        b_factors = {
            name: parameter
            for name, parameter in named.items()
            if name.endswith(("lora_b", "cond_b"))
        }
        self.assertTrue(a_factors)
        self.assertTrue(b_factors)
        self.assertTrue(all(not parameter.requires_grad for parameter in a_factors.values()))
        self.assertTrue(all(parameter.requires_grad for parameter in b_factors.values()))

        loss, _metrics = trainer.compute_loss(self._batch())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None for parameter in a_factors.values()))
        self.assertTrue(
            any(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in b_factors.values()
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_resident_quantized_experts_use_grouped_lora_backward(self) -> None:
        trainer = Trainer(
            self._config(
                target_preset="routed_experts_only",
                quantized=True,
                expert_weight_access="full_dequant",
            )
        )
        trainer.pipeline.set_compute_autocast_dtype(torch.bfloat16)
        trainer.pipeline.transformer.to(device="cuda")
        block = trainer.pipeline.transformer.blocks[0].ffn
        tokens = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([2, 2, 2, 2], device="cuda", dtype=torch.int64)

        output = block._run_grouped_experts(tokens, counts)
        output.float().square().mean().backward()

        grads = [
            param.grad
            for name, param in block.experts.named_parameters()
            if "lora_b" in name
        ]
        self.assertTrue(grads)
        self.assertTrue(any(grad is not None and bool(torch.isfinite(grad).all()) for grad in grads))


    def test_h2d_residency_requires_gradient_checkpointing(self) -> None:
        with self.assertRaisesRegex(ValueError, "gradient_checkpointing"):
            Trainer(self._config(blocks_to_swap=1, quantized=True))

    def test_shared_device_budget_rejects_expert_cache_overcommit(self) -> None:
        config = self._config(
            quantized=True,
            expert_weight_access="active_dequant",
        )
        config.memory.expert_device_cache_gib = 0.001
        config.memory.device_residency_budget_gib = 0.0005
        with self.assertRaisesRegex(MemoryError, "expert_device_cache"):
            Trainer(config)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_tiered_hardware_policy_reaches_runtime_residency(self) -> None:
        config = self._config(
            quantized=True,
            expert_weight_access="active_dequant",
        )
        config.memory.hardware_policy = "tiered"
        trainer = Trainer(config)
        state = trainer.pipeline.get_device_residency_state()
        self.assertTrue(state["enabled"])
        self.assertGreater(state["capacity_bytes"], 0)
        self.assertGreater(
            state["reservations"]["expert_device_cache"], 0
        )
        self.assertGreater(config.memory.expert_dequant_chunk_size, 0)

    def test_quantized_h2d_residency_keeps_packed_frozen_storage_cpu(self) -> None:
        trainer = Trainer(
            self._config(
                gradient_checkpointing="standard",
                blocks_to_swap=1,
                quantized=True,
                expert_weight_access="active_dequant",
            )
        )

        trainer.pipeline.place_offloaded_modules(device=torch.device("cpu"), strategy="block_swap")
        state = trainer.pipeline.get_block_swap_state()
        packed_linears = [
            module
            for module in trainer.pipeline.transformer.modules()
            if isinstance(module, CompressedLinear)
        ]
        packed_experts = [
            module
            for module in trainer.pipeline.transformer.modules()
            if isinstance(module, CompressedGroupedExperts)
        ]

        self.assertTrue(state["enabled"])
        self.assertTrue(state["h2d_only_frozen_base"])
        self.assertTrue(trainer.pipeline.has_quantized_frozen_weights())
        self.assertGreater(len(packed_linears), 0)
        self.assertGreater(len(packed_experts), 0)
        self.assertTrue(all(module.weight_int8.device.type == "cpu" for module in packed_linears))
        self.assertTrue(all(module.w1_int8.device.type == "cpu" for module in packed_experts))

    def test_disk_backed_lingbot_residency_preserves_adapter_gradients(self) -> None:
        trainer = Trainer(
            self._config(
                target_preset="routed_experts_only",
                gradient_checkpointing="standard",
                quantized=True,
                expert_weight_access="active_dequant",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            trainer.pipeline.set_weight_residency_strategy(
                strategy="stream_disk",
                blocks_to_swap=0,
                mode="sync",
                offload_dir=str(Path(tmp) / "blocks"),
            )
            trainer.pipeline.place_offloaded_modules(
                device=torch.device("cpu"),
                strategy="stream_disk",
            )
            loss, _ = trainer.compute_loss(self._batch())
            loss.backward()
            state = trainer.pipeline.get_block_swap_state()
            gradients = [
                parameter.grad
                for parameter in trainer.pipeline.get_trainable_parameters()
            ]
            self.assertEqual(
                state["blocks_to_swap"],
                len(trainer.pipeline.transformer.blocks),
            )
            self.assertGreater(state["disk_backing_bytes"], 0)
            self.assertTrue(
                any(
                    gradient is not None and bool(torch.isfinite(gradient).all())
                    for gradient in gradients
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_quantized_offload_uses_compute_dtype_for_trainables(self) -> None:
        trainer = Trainer(
            self._config(
                target_preset="routed_experts_only",
                gradient_checkpointing="standard",
                blocks_to_swap=1,
                quantized=True,
                expert_weight_access="active_dequant",
            )
        )
        trainer.pipeline.set_compute_autocast_dtype(torch.bfloat16)
        trainer.pipeline.place_offloaded_modules(device="cuda", strategy="block_swap")
        self.assertTrue(
            all(
                param.dtype == torch.bfloat16 and param.device.type == "cuda"
                for param in trainer.pipeline.get_trainable_parameters()
            )
        )

    def test_h2d_placement_requires_compressed_weights_quantization(self) -> None:
        trainer = Trainer(
            self._config(
                gradient_checkpointing="standard",
                blocks_to_swap=1,
                quantized=False,
            )
        )

        with self.assertRaisesRegex(ValueError, "compressed_weights"):
            trainer.pipeline.place_offloaded_modules(
                device=torch.device("cpu"),
                strategy="block_swap",
            )

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
    def test_cuda_block_residency_survives_checkpoint_backward(self) -> None:
        cfg = self._config(
            gradient_checkpointing="standard",
            blocks_to_swap=1,
            quantized=True,
            expert_weight_access="active_dequant",
        )
        trainer = Trainer(cfg)
        trainer.pipeline.place_offloaded_modules(
            device=torch.device("cuda"), strategy="block_swap"
        )
        batch = {
            "latents": torch.randn(1, 2, 1, 1, 1, device="cuda"),
            "text_embeds": torch.randn(1, 16, device="cuda"),
        }
        loss, _ = trainer.compute_loss(batch)
        loss.backward()
        trainer.pipeline.finish_backward_offloads()

        state = trainer.pipeline.get_block_swap_state()
        block = trainer.pipeline.transformer.blocks[0]
        packed = [
            module
            for module in block.modules()
            if isinstance(module, (CompressedLinear, CompressedGroupedExperts))
        ]
        self.assertEqual(state["resident_bytes"], 0)
        self.assertGreater(state["offloaded_bytes"], 0)
        self.assertTrue(all(param.device.type == "cuda" for param in trainer.get_trainable_parameters()))
        self.assertTrue(
            all(
                next(module.buffers()).device.type == "cpu"
                for module in packed
            )
        )

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
    def test_cuda_flat_ring_uses_bounded_contiguous_h2d_copies(self) -> None:
        cfg = self._config(
            gradient_checkpointing="standard",
            blocks_to_swap=1,
            quantized=True,
            expert_weight_access="active_dequant",
            block_swap_transfer_strategy="flat_ring",
        )
        trainer = Trainer(cfg)
        trainer.pipeline.place_offloaded_modules(
            device=torch.device("cuda"), strategy="block_swap"
        )
        batch = {
            "latents": torch.randn(1, 2, 1, 1, 1, device="cuda"),
            "text_embeds": torch.randn(1, 16, device="cuda"),
        }
        loss, _ = trainer.compute_loss(batch)
        loss.backward()
        trainer.pipeline.finish_backward_offloads()

        state = trainer.pipeline.get_block_swap_state()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(state["block_swap_transfer_strategy"], "flat_ring")
        self.assertEqual(state["ring_slots"], 2)
        self.assertGreater(state["ring_buffer_bytes"], 0)
        self.assertGreater(state["h2d_copies"], 0)
        self.assertGreater(state["h2d_bytes"], 0)
        self.assertTrue(
            all(
                parameter.device.type == "cuda"
                for parameter in trainer.get_trainable_parameters()
            )
        )

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
    def test_cuda_flat_ring_matches_per_tensor_loss_and_adapter_gradients(self) -> None:
        batch_cpu = {
            "latents": torch.randn(1, 2, 1, 1, 1),
            "text_embeds": torch.randn(1, 16),
        }

        def run(strategy: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.manual_seed(101)
            trainer = Trainer(
                self._config(
                    gradient_checkpointing="standard",
                    blocks_to_swap=1,
                    quantized=True,
                    expert_weight_access="active_dequant",
                    block_swap_transfer_strategy=strategy,
                )
            )
            trainer.pipeline.place_offloaded_modules(
                device=torch.device("cuda"), strategy="block_swap"
            )
            batch = {key: value.to("cuda") for key, value in batch_cpu.items()}
            torch.manual_seed(202)
            loss, _ = trainer.compute_loss(batch)
            loss.backward()
            gradients = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in trainer.pipeline.get_named_trainable_parameters()
                if parameter.grad is not None
            }
            trainer.pipeline.finish_backward_offloads()
            return loss.detach().cpu(), gradients

        reference_loss, reference_gradients = run("per_tensor")
        ring_loss, ring_gradients = run("flat_ring")

        torch.testing.assert_close(ring_loss, reference_loss, rtol=0, atol=0)
        self.assertEqual(ring_gradients.keys(), reference_gradients.keys())
        for name in reference_gradients:
            torch.testing.assert_close(
                ring_gradients[name], reference_gradients[name], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
