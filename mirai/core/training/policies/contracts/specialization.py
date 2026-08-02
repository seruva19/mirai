"""Formula and bounded-sampling tests for expert-specialization objectives."""

from __future__ import annotations

# Colocated behavioral contract for expert-specialization objectives.

import unittest
from unittest.mock import patch

import torch

from mirai.core.moe.adaptation.specialization_loss import (
    active_expert_orthogonality_loss,
    coactivated_intermediate_cosine_loss,
    cross_layer_topk_coupling_loss,
    deterministic_token_sample,
    router_score_variance_loss,
)
from mirai.core.moe.monitoring.capture import (
    RoutedExpertTensorCapture,
    sampled_outputs_from_sorted_dispatch,
)
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy
from mirai.core.moe.runtime.specs import active_moe_optimization_policy


class ExpertOrthogonalityLossTests(unittest.TestCase):
    def test_orthogonal_outputs_have_zero_loss(self) -> None:
        outputs = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        self.assertEqual(
            float(active_expert_orthogonality_loss(outputs, max_tokens=8)), 0.0
        )

    def test_identical_outputs_match_projection_formula(self) -> None:
        outputs = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]], requires_grad=True)
        loss = active_expert_orthogonality_loss(outputs, max_tokens=8)
        self.assertAlmostEqual(float(loss.detach()), 25.0)
        loss.backward()
        self.assertTrue(torch.isfinite(outputs.grad).all())

    def test_top_one_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            active_expert_orthogonality_loss(torch.ones(2, 1, 4), max_tokens=2)

    def test_expert_permutation_is_invariant(self) -> None:
        outputs = torch.randn(5, 3, 7)
        left = active_expert_orthogonality_loss(outputs, max_tokens=4)
        right = active_expert_orthogonality_loss(outputs[:, [2, 0, 1]], max_tokens=4)
        torch.testing.assert_close(left, right)


class IntermediateSpecializationLossTests(unittest.TestCase):
    def test_squared_cosine_matches_equation_four(self) -> None:
        intermediates = torch.tensor(
            [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 2.0], [3.0, 0.0]]],
            requires_grad=True,
        )
        loss = coactivated_intermediate_cosine_loss(
            intermediates, max_tokens=8
        )
        self.assertAlmostEqual(float(loss.detach()), 0.25)
        loss.backward()
        self.assertTrue(torch.isfinite(intermediates.grad).all())

    def test_sign_and_expert_permutation_are_invariant(self) -> None:
        intermediates = torch.randn(5, 3, 7)
        expected = coactivated_intermediate_cosine_loss(
            intermediates, max_tokens=4
        )
        actual = coactivated_intermediate_cosine_loss(
            -intermediates[:, [2, 0, 1]], max_tokens=4
        )
        torch.testing.assert_close(actual, expected)


class RouterVarianceLossTests(unittest.TestCase):
    def test_uniform_across_tokens_has_zero_loss(self) -> None:
        scores = torch.tensor([[0.2, 0.8], [0.2, 0.8]])
        self.assertEqual(float(router_score_variance_loss(scores, max_tokens=8)), 0.0)

    def test_discriminative_scores_match_formula(self) -> None:
        scores = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        loss = router_score_variance_loss(scores, max_tokens=8)
        self.assertAlmostEqual(float(loss.detach()), -0.25)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_expert_permutation_is_invariant(self) -> None:
        scores = torch.randn(7, 4)
        torch.testing.assert_close(
            router_score_variance_loss(scores, max_tokens=5),
            router_score_variance_loss(scores[:, [3, 1, 0, 2]], max_tokens=5),
        )


class CrossLayerCouplingLossTests(unittest.TestCase):
    def test_joint_topk_probability_matches_equation(self) -> None:
        first = torch.tensor(
            [[3.0, 1.0], [1.0, 3.0]], requires_grad=True
        )
        second = torch.tensor(
            [[2.0, 2.0], [4.0, 0.0]], requires_grad=True
        )
        loss = cross_layer_topk_coupling_loss(
            [first, second], top_k=1, max_tokens=8
        )
        self.assertAlmostEqual(float(loss.detach()), -0.5625)
        loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())

    def test_requires_aligned_adjacent_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical token/expert shapes"):
            cross_layer_topk_coupling_loss(
                [torch.ones(2, 3), torch.ones(3, 3)],
                top_k=2,
                max_tokens=8,
            )

    def test_independent_expert_permutations_preserve_path_mass(self) -> None:
        first = torch.randn(7, 4).softmax(dim=-1)
        second = torch.randn(7, 4).softmax(dim=-1)
        expected = cross_layer_topk_coupling_loss(
            [first, second], top_k=2, max_tokens=5
        )
        actual = cross_layer_topk_coupling_loss(
            [first[:, [2, 0, 3, 1]], second[:, [1, 3, 0, 2]]],
            top_k=2,
            max_tokens=5,
        )
        torch.testing.assert_close(actual, expected)


class DeterministicSamplingTests(unittest.TestCase):
    def test_evenly_spaced_bounded_selection(self) -> None:
        values = torch.arange(10)
        self.assertEqual(deterministic_token_sample(values, max_tokens=4).tolist(), [0, 2, 5, 7])

    def test_sorted_dispatch_capture_reconstructs_selected_outputs(self) -> None:
        unsorted = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        sorted_positions = torch.tensor([4, 0, 3, 1, 5, 2])
        expert_output = unsorted[sorted_positions]
        actual = sampled_outputs_from_sorted_dispatch(
            expert_output,
            sorted_positions,
            num_tokens=3,
            top_k=2,
            max_tokens=2,
        )
        torch.testing.assert_close(actual, unsorted.reshape(3, 2, 4)[[0, 1]])

    def test_capture_sink_returns_and_clears_differentiable_losses(self) -> None:
        outputs = torch.randn(4, 3, requires_grad=True)
        capture = RoutedExpertTensorCapture(
            max_tokens=2, loss_fn=active_expert_orthogonality_loss
        )
        capture.capture_sorted(
            outputs, torch.tensor([0, 1, 2, 3]), num_tokens=2, top_k=2
        )
        losses = capture.take_losses()
        self.assertEqual(len(losses), 1)
        self.assertTrue(losses[0].requires_grad)
        self.assertEqual(capture.take_losses(), [])

    def test_compressed_vectorized_dispatch_captures_bounded_adapter_loss(self) -> None:
        from mirai.core.models.compressed_weights import CompressedGroupedExperts

        torch.manual_seed(23)
        module = CompressedGroupedExperts.from_empty(
            num_experts=3,
            group_sizes=4,
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
            quant_format="int8",
        )
        for key, shape in {
            "w1": (3, 12, 8),
            "w2": (3, 8, 12),
            "w3": (3, 12, 8),
        }.items():
            module.load_dense_weight(key, torch.randn(shape) * 0.1)
            adapter = module.attach_expert_lora(
                tensor_name=key,
                adapter_name=key,
                rank=2,
                alpha=2.0,
            )
            with torch.no_grad():
                adapter.lora_b.normal_(std=0.05)
        indices = torch.tensor(
            [[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1]]
        )
        scores = torch.full((6, 2), 0.5)
        tokens = torch.randn(6, 8, requires_grad=True)
        with active_moe_optimization_policy(
            MoEOptimizationPolicy(moe_dispatch="vectorized")
        ):
            expected = module.run_direct_routed(
                tokens.detach(), scores, indices
            )
            capture = RoutedExpertTensorCapture(
                max_tokens=3, loss_fn=active_expert_orthogonality_loss
            )
            module.set_routed_output_observer(capture)
            actual = module.run_direct_routed(tokens, scores, indices)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        losses = capture.take_losses()
        self.assertEqual(len(losses), 1)
        for mode in ("vectorized", "legacy"):
            device_capture = RoutedExpertTensorCapture(
                max_tokens=3, loss_fn=active_expert_orthogonality_loss
            )
            module.set_routed_output_observer(device_capture)
            with active_moe_optimization_policy(
                MoEOptimizationPolicy(
                    moe_dispatch=mode,
                    moe_dispatch_preprocess="device",
                )
            ):
                device_output = module.run_direct_routed(
                    tokens.detach(), scores, indices
                )
            torch.testing.assert_close(device_output, expected)
            torch.testing.assert_close(device_capture.take_losses()[0], losses[0])
        losses[0].backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        for adapter in module.expert_lora.values():
            self.assertTrue(torch.isfinite(adapter.lora_a.grad).all())
            self.assertTrue(torch.isfinite(adapter.lora_b.grad).all())

    def test_compressed_direct_dispatch_captures_swiglu_intermediates(self) -> None:
        from mirai.core.models.compressed_weights import CompressedGroupedExperts

        torch.manual_seed(29)
        module = CompressedGroupedExperts.from_empty(
            num_experts=3,
            group_sizes=4,
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
            quant_format="int8",
        )
        for key, shape in {
            "w1": (3, 12, 8),
            "w2": (3, 8, 12),
            "w3": (3, 12, 8),
        }.items():
            module.load_dense_weight(key, torch.randn(shape) * 0.1)
            adapter = module.attach_expert_lora(
                tensor_name=key,
                adapter_name=key,
                rank=2,
                alpha=2.0,
            )
            with torch.no_grad():
                adapter.lora_b.normal_(std=0.05)
        indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]])
        scores = torch.full((4, 2), 0.5)
        tokens = torch.randn(4, 8, requires_grad=True)
        capture = RoutedExpertTensorCapture(
            max_tokens=3, loss_fn=coactivated_intermediate_cosine_loss
        )
        with active_moe_optimization_policy(
            MoEOptimizationPolicy(
                moe_dispatch="vectorized",
                moe_dispatch_preprocess="host",
            )
        ):
            expected = module.run_direct_routed(
                tokens.detach(), scores, indices
            )
            original_chunk = module._run_expert_chunk
            seen_chunks = 0

            def strict_chunk_signature(
                padded,
                active,
                expert_ids,
                *,
                device,
                rotated=False,
            ):
                nonlocal seen_chunks
                seen_chunks += 1
                return original_chunk(
                    padded,
                    active,
                    expert_ids,
                    device=device,
                    rotated=rotated,
                )

            module._run_expert_chunk = strict_chunk_signature
            module.set_routed_intermediate_observer(capture)
            output = module.run_direct_routed(tokens, scores, indices)
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)
        self.assertGreater(seen_chunks, 0)
        vectorized_loss = capture.take_losses()[0]
        legacy_capture = RoutedExpertTensorCapture(
            max_tokens=3, loss_fn=coactivated_intermediate_cosine_loss
        )
        module.set_routed_intermediate_observer(legacy_capture)
        with active_moe_optimization_policy(
            MoEOptimizationPolicy(
                moe_dispatch="legacy",
                moe_dispatch_preprocess="host",
            )
        ):
            legacy_output = module.run_direct_routed(tokens, scores, indices)
        legacy_loss = legacy_capture.take_losses()[0]
        torch.testing.assert_close(legacy_output, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(legacy_loss, vectorized_loss)
        for mode in ("vectorized", "legacy"):
            device_capture = RoutedExpertTensorCapture(
                max_tokens=3, loss_fn=coactivated_intermediate_cosine_loss
            )
            module.set_routed_intermediate_observer(device_capture)
            with active_moe_optimization_policy(
                MoEOptimizationPolicy(
                    moe_dispatch=mode,
                    moe_dispatch_preprocess="device",
                )
            ):
                device_output = module.run_direct_routed(tokens, scores, indices)
            device_loss = device_capture.take_losses()[0]
            torch.testing.assert_close(device_output, expected)
            torch.testing.assert_close(device_loss, vectorized_loss)
        loss = legacy_loss
        self.assertTrue(torch.isfinite(output).all())
        loss.backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        for key in ("w1", "w3"):
            adapter = module.expert_lora[key]
            self.assertTrue(torch.isfinite(adapter.lora_a.grad).all())
            self.assertTrue(torch.isfinite(adapter.lora_b.grad).all())

    def test_sorted_contiguous_backends_share_routed_capture_contract(self) -> None:
        from mirai.core.models.compressed_weights import CompressedGroupedExperts

        module = CompressedGroupedExperts.from_empty(
            num_experts=3,
            group_sizes=4,
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
            quant_format="int8",
        )
        indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]])
        scores = torch.full((4, 2), 0.5)
        tokens = torch.randn(4, 8, requires_grad=True)
        results = {}
        losses = {}

        def persistent_pair(_owner, _key_a, _key_b, x, *args, **kwargs):
            return x, x

        def persistent_linear(_owner, _key, x, *args, **kwargs):
            return x

        def grouped_linear(_owner, _key, x, *args, **kwargs):
            return x

        for backend in ("persistent", "torch_grouped"):
            capture = RoutedExpertTensorCapture(
                max_tokens=3, loss_fn=coactivated_intermediate_cosine_loss
            )
            module.set_routed_intermediate_observer(capture)
            patches = [
                patch(
                    "mirai.core.models.compressed_weights.execution.experts.resolve_moe_gemm_backend",
                    return_value=backend,
                )
            ]
            if backend == "persistent":
                patches.extend(
                    [
                        patch(
                            "mirai.core.models.compressed_weights.execution.persistent.streamed_linear_pair",
                            side_effect=persistent_pair,
                        ),
                        patch(
                            "mirai.core.models.compressed_weights.execution.persistent.streamed_linear",
                            side_effect=persistent_linear,
                        ),
                    ]
                )
            else:
                patches.append(
                    patch(
                        "mirai.core.models.compressed_weights.execution.torch_grouped._grouped_linear",
                        side_effect=grouped_linear,
                    )
                )
            with patches[0]:
                with patches[1]:
                    if len(patches) == 3:
                        with patches[2]:
                            results[backend] = module.run_direct_routed(
                                tokens, scores, indices
                            )
                    else:
                        results[backend] = module.run_direct_routed(
                            tokens, scores, indices
                        )
            losses[backend] = capture.take_losses()[0]

        torch.testing.assert_close(results["persistent"], results["torch_grouped"])
        torch.testing.assert_close(losses["persistent"], losses["torch_grouped"])
        losses["persistent"].backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())


@unittest.skipUnless(torch.cuda.is_available(), "optimized grouped dispatch requires CUDA")
class OptimizedDispatchCaptureTests(unittest.TestCase):
    def test_persistent_and_framework_grouped_capture_real_intermediates(self) -> None:
        from mirai.core.models.compressed_weights import CompressedGroupedExperts
        from mirai.core.moe.runtime.gemm import probe_backend

        device = torch.device("cuda")
        for backend in ("persistent", "torch_grouped"):
            verdict = probe_backend(backend, device=device)
            if not verdict.available:
                self.skipTest(f"{backend} unavailable: {verdict.reason}")
        torch.manual_seed(37)
        module = CompressedGroupedExperts.from_empty(
            num_experts=3,
            group_sizes=16,
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
            quant_format="int8",
        )
        for key, shape in {
            "w1": (3, 64, 32),
            "w2": (3, 32, 64),
            "w3": (3, 64, 32),
        }.items():
            module.load_dense_weight(key, torch.randn(shape) * 0.1)
            adapter = module.attach_expert_lora(
                tensor_name=key,
                adapter_name=key,
                rank=2,
                alpha=2.0,
            )
            with torch.no_grad():
                adapter.lora_b.normal_(std=0.05)
        module.to(device)
        indices = torch.tensor(
            [[0, 1], [1, 2], [2, 0], [0, 2]], device=device
        )
        scores = torch.full((4, 2), 0.5, device=device, dtype=torch.bfloat16)
        base_tokens = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
        reference_loss = None
        for backend in ("persistent", "torch_grouped"):
            tokens = base_tokens.clone().requires_grad_(True)
            capture = RoutedExpertTensorCapture(
                max_tokens=3, loss_fn=coactivated_intermediate_cosine_loss
            )
            module.set_routed_intermediate_observer(capture)
            module.zero_grad(set_to_none=True)
            with active_moe_optimization_policy(
                MoEOptimizationPolicy(moe_gemm_backend=backend)
            ):
                output = module.run_direct_routed(tokens, scores, indices)
            loss = capture.take_losses()[0]
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(tokens.grad).all())
            for key in ("w1", "w3"):
                adapter = module.expert_lora[key]
                self.assertTrue(torch.isfinite(adapter.lora_a.grad).all())
                self.assertTrue(torch.isfinite(adapter.lora_b.grad).all())
            if reference_loss is None:
                reference_loss = loss.detach()
            else:
                torch.testing.assert_close(
                    loss.detach(), reference_loss, rtol=3e-2, atol=3e-2
                )


class RouterVariancePipelineTests(unittest.TestCase):
    def _pipeline(
        self,
        *,
        weight: float,
        orthogonality_weight: float = 0.0,
        swiglu_weight: float = 0.0,
        coupling_weight: float = 0.0,
    ):
        from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams
        from mirai.core.builtins import register_builtin_components
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

        register_builtin_components()
        pipeline = LingBotVideoPipeline(
            ModelConfig(
                type="lingbot-video",
                path="./models/lingbot_video",
                params=ModelParams(
                    variant="tiny-video",
                    latent_channels=2,
                    num_experts=4,
                    experts_per_token=2,
                    shared_experts=1,
                    hidden_size=16,
                    num_layers=2 if coupling_weight > 0.0 else 1,
                    attention_heads=2,
                    patch_size=1,
                    moe_router_variance_loss_weight=weight,
                    moe_expert_orthogonality_loss_weight=orthogonality_weight,
                    moe_swiglu_specialization_loss_weight=swiglu_weight,
                    moe_cross_layer_coupling_loss_weight=coupling_weight,
                    moe_specialization_max_tokens=5,
                ),
            )
        )
        pipeline.set_adapter_config(
            AdapterConfig(
                type="lora",
                target_preset=(
                    "attn_router_routed_experts"
                    if coupling_weight > 0.0
                    else "attn_routed_experts"
                ),
                rank=2,
                alpha=2.0,
                train_router=True if coupling_weight > 0.0 else None,
            )
        )
        pipeline.train()
        return pipeline

    def _forward(self, pipeline) -> None:
        torch.manual_seed(17)
        pipeline.forward(
            torch.randn(1, 2, 2, 4, 4),
            torch.tensor([0.4]),
            {"lingbot": torch.randn(1, 3, 16)},
        )

    def test_default_off_has_no_variance_loss(self) -> None:
        pipeline = self._pipeline(weight=0.0)
        self._forward(pipeline)
        self.assertNotIn(
            "moe_router_variance", pipeline.get_training_auxiliary_losses()
        )

    def test_enabled_surfaces_differentiable_variance_loss(self) -> None:
        pipeline = self._pipeline(weight=0.2)
        self._forward(pipeline)
        loss = pipeline.get_training_auxiliary_losses()["moe_router_variance"]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)

    def test_enabled_surfaces_differentiable_orthogonality_loss(self) -> None:
        pipeline = self._pipeline(weight=0.0, orthogonality_weight=0.2)
        self._forward(pipeline)
        loss = pipeline.get_training_auxiliary_losses()["moe_expert_orthogonality"]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)

    def test_enabled_surfaces_differentiable_swiglu_specialization(self) -> None:
        pipeline = self._pipeline(weight=0.0, swiglu_weight=0.2)
        self._forward(pipeline)
        loss = pipeline.get_training_auxiliary_losses()[
            "moe_swiglu_specialization"
        ]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad
            for name, parameter in pipeline.named_parameters()
            if (".w1." in name or ".w3." in name)
            and "lora_" in name
            and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))

    def test_enabled_surfaces_differentiable_cross_layer_coupling(self) -> None:
        pipeline = self._pipeline(weight=0.0, coupling_weight=0.2)
        self._forward(pipeline)
        loss = pipeline.get_training_auxiliary_losses()[
            "moe_cross_layer_coupling"
        ]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        loss.backward()
        router_gradients = [
            parameter.grad
            for name, parameter in pipeline.named_parameters()
            if ".router." in name and "lora_" in name and parameter.grad is not None
        ]
        self.assertTrue(router_gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in router_gradients))

    def test_aggressive_checkpoint_preserves_cross_layer_coupling_graph(self) -> None:
        pipeline = self._pipeline(weight=0.0, coupling_weight=0.2)
        pipeline.set_gradient_checkpointing("aggressive")
        self._forward(pipeline)
        loss = pipeline.get_training_auxiliary_losses()[
            "moe_cross_layer_coupling"
        ]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_aggressive_checkpoint_surfaces_recomputed_orthogonality_loss(self) -> None:
        pipeline = self._pipeline(weight=0.0, orthogonality_weight=0.2)
        pipeline.set_gradient_checkpointing("aggressive")
        self._forward(pipeline)
        loss = pipeline.get_training_auxiliary_losses()["moe_expert_orthogonality"]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.requires_grad)
        loss.backward()
        adapter_gradients = [
            parameter.grad
            for name, parameter in pipeline.named_parameters()
            if ("lora_a" in name or "lora_b" in name) and parameter.grad is not None
        ]
        self.assertTrue(adapter_gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in adapter_gradients))
        self.assertEqual(
            pipeline._expert_output_orthogonality_capture.take_losses(), []
        )

    def test_aggressive_checkpoint_preserves_swiglu_specialization(self) -> None:
        pipeline = self._pipeline(weight=0.0, swiglu_weight=0.2)
        pipeline.set_gradient_checkpointing("aggressive")
        self._forward(pipeline)
        loss = pipeline.get_training_auxiliary_losses()[
            "moe_swiglu_specialization"
        ]
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertEqual(
            pipeline._intermediate_specialization_runtime.capture.take_losses(),
            [],
        )

    def test_standard_checkpoint_clears_swiglu_recompute_capture(self) -> None:
        pipeline = self._pipeline(weight=0.0, swiglu_weight=0.2)
        pipeline.set_gradient_checkpointing("standard")
        self._forward(pipeline)
        loss = pipeline.get_training_auxiliary_losses()[
            "moe_swiglu_specialization"
        ]
        loss.backward()
        self.assertEqual(
            pipeline._intermediate_specialization_runtime.capture.take_losses(),
            [],
        )

    def test_standard_checkpoint_does_not_leave_recompute_capture(self) -> None:
        pipeline = self._pipeline(weight=0.0, orthogonality_weight=0.2)
        pipeline.set_gradient_checkpointing("standard")
        self._forward(pipeline)
        loss = pipeline.get_training_auxiliary_losses()["moe_expert_orthogonality"]
        loss.backward()
        self.assertEqual(
            pipeline._expert_output_orthogonality_capture.take_losses(), []
        )

    def test_config_fields_parse_and_are_registered(self) -> None:
        from mirai.config.schema import TrainingConfig, all_config_keys

        config = TrainingConfig.from_dict(
            {
                "model": {
                    "params": {
                        "moe_router_variance_loss_weight": 0.3,
                        "moe_expert_orthogonality_loss_weight": 0.4,
                        "moe_swiglu_specialization_loss_weight": 0.45,
                        "moe_cross_layer_coupling_loss_weight": 0.5,
                        "moe_specialization_max_tokens": 17,
                    }
                }
            }
        )
        self.assertEqual(config.model.params.moe_router_variance_loss_weight, 0.3)
        self.assertEqual(
            config.model.params.moe_expert_orthogonality_loss_weight, 0.4
        )
        self.assertEqual(
            config.model.params.moe_swiglu_specialization_loss_weight, 0.45
        )
        self.assertEqual(
            config.model.params.moe_cross_layer_coupling_loss_weight, 0.5
        )
        self.assertEqual(config.model.params.moe_specialization_max_tokens, 17)
        keys = all_config_keys()["model.params"]
        self.assertIn("moe_router_variance_loss_weight", keys)
        self.assertIn("moe_expert_orthogonality_loss_weight", keys)
        self.assertIn("moe_swiglu_specialization_loss_weight", keys)
        self.assertIn("moe_cross_layer_coupling_loss_weight", keys)
        self.assertIn("moe_specialization_max_tokens", keys)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
