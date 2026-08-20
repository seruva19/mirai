from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mirai.config.schema import (
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
from mirai.core.models.compressed_weights import export_compressed_weights_packed_state
from mirai.core.models.compressed_weights import load_compressed_weights_packed_state
from mirai.core.models.compressed_weights import load_compressed_weights_packed_state_file
from mirai.core.models.compressed_weights import LazyPackedTensorMapping
from mirai.core.models.compressed_weights import materialize_packed_tensors
from mirai.core.models.compressed_weights import PreloadedPackedTensorMapping
from mirai.core.models.compressed_weights.execution.expert_device_cache import (
    ExpertDeviceCache,
)
from mirai.core.models.compressed_weights import prepare_compressed_weights_modules_from_manifest
from mirai.core.models.compressed_weights import quantize_compressed_weights_modules
from mirai.core.models.compressed_weights import save_compressed_weights_packed_state
from mirai.core.models.compressed_weights.artifact_source import (
    load_grouped_expert_source,
)
from mirai.core.models.compressed_weights.quantization.quant import _quantize_weight
from mirai.core.models.compressed_weights.quantization.gguf_quant import (
    dequantize_gguf,
    gguf_stored_bytes,
    normalize_gguf_format,
    quantize_gguf,
    validate_gguf_blocks,
)
from mirai.core.models.compressed_weights.quantization.learned_rotation import (
    learn_groupwise_expert_rotation,
    validate_learned_rotation_selection,
)
from mirai.core.models.compressed_weights.quantization.rotated_int8 import (
    rotated_int8_linear,
)
from mirai.core.moe.runtime.specs import (
    ExpertMLPExecutionSpec,
    ExpertProjectionRole,
    MoEOptimizationPolicy,
)
from mirai.core.moe.runtime.specs import active_moe_optimization_policy
from mirai.core.models.compressed_weights.quantization.microscaling_quant import (
    dequantize_microscaling,
    microscaling_stored_bytes,
    quantize_microscaling,
    validate_microscaling_payload,
)
from mirai.core.models.compressed_weights.quantization.blockwise_fp8 import (
    blockwise_fp8_batched_linear,
    blockwise_fp8_linear,
    dequantize_blockwise_fp8_weight,
    quantize_blockwise_fp8_weight,
)
from mirai.core.models.compressed_weights.quantization.deepgemm_fp8 import (
    deepgemm_blockwise_fp8_batched_linear,
    deepgemm_blockwise_fp8_routed_linear,
)
from mirai.core.moe.runtime.routed_gemm import (
    RoutedFusionSpec,
    RoutedGroupLayout,
    RoutedOutputMode,
)
from mirai.core.models.compressed_weights.quantization.structured_sparsity import (
    StructuredSparse24GroupedExperts,
    prune_to_2_4,
    sparse_2_4_linear,
    validate_2_4,
)
from mirai.core.moe.calibration.precision import (
    ExpertPrecisionEvidence,
    ExpertPrecisionPlan,
    allocate_expert_precision,
)
from mirai.core.moe.runtime.kernels import (
    RotatedInt8KernelBackend,
    build_moe_kernel_backend,
    normalize_moe_kernel_backend,
)
from mirai.core.models.compressed_weights.packed.packed_storage_alignment import (
    read_safetensors_storage_alignment,
)
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.training.trainer import Trainer

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


@unittest.skipIf(torch is None, "torch not installed")
class CompressedWeightQuantizationTests(unittest.TestCase):
    def test_iq2_xs_canonical_layout_and_reference_decode(self) -> None:
        self.assertEqual(gguf_stored_bytes("gguf_iq2", 256), 74)
        self.assertEqual(normalize_gguf_format("iq2_xs"), "gguf_iq2")
        self.assertEqual(normalize_gguf_format("gguf_iq2_xs"), "gguf_iq2")

        # Canonical block with fp16 d=1, grid row zero (eight 8s), scale nibble
        # zero, and sign index zero decodes to 1 * (0.5 / 4) * 8 = 1.
        block = torch.zeros((1, 74), dtype=torch.uint8)
        block[:, :2] = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
        decoded = dequantize_gguf(
            "gguf_iq2",
            block,
            shape=(16, 16),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(decoded, torch.ones_like(decoded))

    def test_iq2_xs_roundtrip_is_deterministic_and_finite(self) -> None:
        torch.manual_seed(126)
        weight = torch.randn(16, 16)
        first = quantize_gguf("gguf_iq2", weight)
        second = quantize_gguf("iq2_xs", weight)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (1, 74))
        self.assertEqual(first.dtype, torch.uint8)
        decoded = dequantize_gguf(
            "gguf_iq2",
            first,
            shape=tuple(weight.shape),
            dtype=weight.dtype,
            device=weight.device,
        )
        self.assertTrue(bool(torch.isfinite(decoded).all()))
        self.assertLess(float((weight - decoded).square().mean()), 0.25)

        zeros = torch.zeros_like(weight)
        zero_blocks = quantize_gguf("gguf_iq2", zeros)
        zero_decoded = dequantize_gguf(
            "gguf_iq2",
            zero_blocks,
            shape=tuple(zeros.shape),
            dtype=zeros.dtype,
            device=zeros.device,
        )
        self.assertEqual(float(zero_decoded.abs().max()), 0.0)

    def test_iq2_xs_rejects_malformed_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "74"):
            validate_gguf_blocks("gguf_iq2", torch.zeros((1, 73), dtype=torch.uint8))

    def test_prepare_supports_provider_declared_two_projection_gelu_experts(self) -> None:
        spec = ExpertMLPExecutionSpec(
            projections=(
                ExpertProjectionRole("input", "fc_in"),
                ExpertProjectionRole("down", "fc_out"),
            ),
            activation="gelu",
            combiner="activated",
        )

        class _AlternativeExperts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 3
                self.mirai_expert_mlp_spec = spec
                self.fc_in = nn.Parameter(
                    torch.randn(3, 24, 16), requires_grad=False
                )
                self.fc_out = nn.Parameter(
                    torch.randn(3, 16, 24), requires_grad=False
                )

        class _Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.experts = _AlternativeExperts()

        torch.manual_seed(125)
        counts = torch.tensor([2, 1, 2])
        for access in ("full_dequant", "active_dequant", "chunked_dequant"):
            with self.subTest(expert_weight_access=access):
                root = _Root()
                reference_in = root.experts.fc_in.detach().clone()
                reference_out = root.experts.fc_out.detach().clone()
                report = quantize_compressed_weights_modules(
                    root,
                    group_sizes=16,
                    expert_weight_access=access,
                    expert_dequant_chunk_size=2,
                    expert_mlp_execution_spec=spec,
                )
                self.assertEqual(report.grouped_expert_modules, 1)
                self.assertEqual(report.quantized_tensors, 2)
                self.assertIsInstance(root.experts, CompressedGroupedExperts)
                self.assertEqual(root.experts.expert_mlp_spec, spec)

                inputs = torch.randn(5, 16, requires_grad=True)
                reference_inputs = inputs.detach().clone().requires_grad_(True)
                actual = root.experts.run_for_loop(inputs, counts)
                expected_parts = []
                for expert_idx, expert_inputs in enumerate(
                    torch.split(reference_inputs, [2, 1, 2])
                ):
                    hidden = torch.nn.functional.gelu(
                        torch.nn.functional.linear(
                            expert_inputs, reference_in[expert_idx]
                        )
                    )
                    expected_parts.append(
                        torch.nn.functional.linear(hidden, reference_out[expert_idx])
                    )
                expected = torch.cat(expected_parts)
                torch.testing.assert_close(actual, expected, rtol=0.12, atol=0.35)
                actual.square().mean().backward()
                expected.square().mean().backward()
                torch.testing.assert_close(
                    inputs.grad,
                    reference_inputs.grad,
                    rtol=0.12,
                    atol=0.35,
                )

        with self.assertRaisesRegex(RuntimeError, "canonical w1/w3/w2"):
            export_compressed_weights_packed_state(root)
        with self.assertRaisesRegex(ValueError, "canonical gated-product"):
            StructuredSparse24GroupedExperts(_AlternativeExperts())
        with self.assertRaisesRegex(ValueError, "canonical gated-product"):
            quantize_compressed_weights_modules(
                _Root(),
                group_sizes=16,
                learn_expert_rotations=True,
                expert_mlp_execution_spec=spec,
            )

    def test_provider_and_module_execution_specs_must_match(self) -> None:
        alternative_spec = ExpertMLPExecutionSpec(
            projections=(
                ExpertProjectionRole("input", "fc_in"),
                ExpertProjectionRole("down", "fc_out"),
            ),
            activation="gelu",
            combiner="activated",
        )

        class _CanonicalExperts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 2
                self.w1 = nn.Parameter(torch.randn(2, 16, 16), requires_grad=False)
                self.w2 = nn.Parameter(torch.randn(2, 16, 16), requires_grad=False)
                self.w3 = nn.Parameter(torch.randn(2, 16, 16), requires_grad=False)

        class _AlternativeExperts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 2
                self.mirai_expert_mlp_spec = alternative_spec
                self.fc_in = nn.Parameter(torch.randn(2, 16, 16), requires_grad=False)
                self.fc_out = nn.Parameter(torch.randn(2, 16, 16), requires_grad=False)

        source = nn.Module()
        source.experts = _CanonicalExperts()
        quantize_compressed_weights_modules(source, group_sizes=16)
        _tensors, manifest = export_compressed_weights_packed_state(source)

        target = nn.Module()
        target.experts = _AlternativeExperts()
        with self.assertRaisesRegex(ValueError, "provider-declared spec"):
            prepare_compressed_weights_modules_from_manifest(target, manifest)

    def test_megablocks_linear_rematerializes_frozen_weight_in_backward(self) -> None:
        class _Experts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 3
                self.w1 = nn.Parameter(torch.randn(3, 16, 16), requires_grad=False)
                self.w2 = nn.Parameter(torch.randn(3, 16, 16), requires_grad=False)
                self.w3 = nn.Parameter(torch.randn(3, 16, 16), requires_grad=False)

        class _GroupedGemmOps:
            @staticmethod
            def gmm(inputs, weights, batch_sizes, *, trans_b):
                assert trans_b
                return torch.cat(
                    [
                        part @ weights[index].transpose(-2, -1)
                        for index, part in enumerate(
                            torch.split(inputs, batch_sizes.tolist())
                        )
                    ]
                )

        torch.manual_seed(127)
        module = CompressedGroupedExperts(_Experts(), group_sizes=16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        counts = torch.tensor([2, 0, 3], dtype=torch.int64, device=device)
        inputs = torch.randn(5, 16, device=device, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        weight = module._dequantize("w1", dtype=inputs.dtype, device=inputs.device)
        expected = torch.cat(
            [
                part @ weight[index].transpose(-2, -1)
                for index, part in enumerate(
                    torch.split(reference_inputs, counts.tolist())
                )
            ]
        )
        saved: list[torch.Tensor] = []

        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: saved.append(tensor) or tensor,
            lambda tensor: tensor,
        ):
            actual = module.megablocks_linear(
                "w1",
                inputs,
                counts=counts,
                grouped_gemm_ops=_GroupedGemmOps(),
            )

        self.assertEqual(saved, [])
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        torch.testing.assert_close(
            inputs.grad,
            reference_inputs.grad,
            rtol=0.0,
            atol=0.0,
        )

    def test_dense_compressed_linear_does_not_save_materialized_weight(self) -> None:
        torch.manual_seed(121)
        for quant_format in ("int8", "gguf_iq2", "mxfp8_e4m3"):
            with self.subTest(quant_format=quant_format):
                module = CompressedLinear(
                    nn.Linear(32, 16, bias=True),
                    group_sizes=16,
                    quant_format=quant_format,
                )
                inputs = torch.randn(2, 3, 32, requires_grad=True)
                saved: list[torch.Tensor] = []

                def pack(
                    tensor: torch.Tensor,
                    saved_tensors: list[torch.Tensor] = saved,
                ) -> torch.Tensor:
                    saved_tensors.append(tensor)
                    return tensor

                with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                    output = module(inputs)

                self.assertEqual(saved, [])
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_per_expert_linear_does_not_save_dequantized_weight(self) -> None:
        class _Experts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 2
                self.w1 = torch.randn(2, 16, 32)
                self.w2 = torch.randn(2, 32, 16)
                self.w3 = torch.randn(2, 16, 32)

        torch.manual_seed(122)
        module = CompressedGroupedExperts(
            _Experts(),
            group_sizes=16,
            expert_weight_access="active_dequant",
        )
        inputs = torch.randn(5, 32, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        weight = module._dequantize_expert(  # noqa: SLF001
            "w1", 1, dtype=inputs.dtype, device=inputs.device
        )
        saved: list[torch.Tensor] = []

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            observed = module._expert_linear("w1", inputs, weight, 1)  # noqa: SLF001
        expected = reference_inputs @ weight.transpose(-2, -1)
        grad_output = torch.randn_like(observed)
        observed.backward(grad_output)
        expected.backward(grad_output)

        self.assertEqual(saved, [])
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0.0, atol=0.0)

    def test_full_dequant_batched_linear_does_not_save_expert_stack(self) -> None:
        class _Experts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 3
                self.w1 = torch.randn(3, 16, 32)
                self.w2 = torch.randn(3, 32, 16)
                self.w3 = torch.randn(3, 16, 32)

        torch.manual_seed(124)
        module = CompressedGroupedExperts(
            _Experts(),
            group_sizes=16,
            expert_weight_access="full_dequant",
        )
        inputs = torch.randn(3, 5, 32, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        weight = module._dequantize("w1", dtype=inputs.dtype, device=inputs.device)  # noqa: SLF001
        saved: list[torch.Tensor] = []

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            observed = module.batched_linear("w1", inputs)
        expected = torch.bmm(reference_inputs, weight.transpose(-2, -1))
        grad_output = torch.randn_like(observed)
        observed.backward(grad_output)
        expected.backward(grad_output)

        self.assertEqual(saved, [])
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0.0, atol=0.0)

    def test_rotated_int8_does_not_save_fp32_weight_operand(self) -> None:
        torch.manual_seed(123)
        inputs = torch.randn(2, 5, 16, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        quantized = torch.randint(-127, 128, (2, 8, 16), dtype=torch.int8)
        scale = torch.rand(2, 8, 1)
        rotation, _ = torch.linalg.qr(torch.randn(16, 16))
        saved: list[torch.Tensor] = []

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            observed = rotated_int8_linear(
                inputs,
                quantized,
                scale,
                16,
                rotation=rotation,
            )
        reference_rotated = reference_inputs.float() @ rotation
        expected = torch.bmm(
            reference_rotated,
            quantized.float().transpose(-2, -1),
        ) * scale.reshape(2, 1, 8)
        grad_output = torch.randn_like(observed)
        observed.backward(grad_output)
        expected.backward(grad_output)

        self.assertEqual(saved, [])
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0.0, atol=0.0)

    def test_blockwise_fp8_matches_e4m3_tile_oracle_and_high_precision_dgrad(
        self,
    ) -> None:
        torch.manual_seed(125)
        weight = torch.randn(130, 257, dtype=torch.float32) * 0.17
        codes, scales, meta = quantize_blockwise_fp8_weight(weight)
        self.assertEqual(tuple(codes.shape), (130, 257))
        self.assertEqual(tuple(scales.shape), (2, 3))
        self.assertEqual(meta.weight_block, (128, 128))
        self.assertEqual(meta.activation_block, 128)

        padded_weight = torch.nn.functional.pad(weight, (0, 127, 0, 126))
        weight_tiles = padded_weight.reshape(2, 128, 3, 128)
        oracle_scales = (weight_tiles.abs().amax(dim=(1, 3)) / 448.0).clamp(
            min=1e-30
        )
        oracle_codes = (
            (weight_tiles / oracle_scales[:, None, :, None])
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .reshape(256, 384)[:130, :257]
        )
        torch.testing.assert_close(scales, oracle_scales, rtol=0.0, atol=0.0)
        self.assertTrue(torch.equal(codes, oracle_codes))

        inputs = torch.randn(5, 257, requires_grad=True)
        observed = blockwise_fp8_linear(inputs, codes, scales, meta)
        padded_inputs = torch.nn.functional.pad(inputs.detach(), (0, 127))
        activation_tiles = padded_inputs.reshape(5, 3, 128)
        activation_scales = (
            activation_tiles.abs().amax(dim=-1).clamp(min=1e-4) / 448.0
        )
        activation_quantized = (
            (activation_tiles / activation_scales.unsqueeze(-1))
            .to(torch.float8_e4m3fn)
            .float()
        )
        weight_quantized = (
            (weight_tiles / oracle_scales[:, None, :, None])
            .to(torch.float8_e4m3fn)
            .float()
            .reshape(256, 384)
        )
        expected = torch.zeros(5, 256)
        expanded_scales = oracle_scales.repeat_interleave(128, dim=0)
        for block_index in range(3):
            x_block = (
                activation_quantized[:, block_index]
                * activation_scales[:, block_index].unsqueeze(-1)
            )
            w_block = weight_quantized[
                :, block_index * 128 : (block_index + 1) * 128
            ] * expanded_scales[:, block_index].unsqueeze(-1)
            expected.addmm_(x_block, w_block.transpose(0, 1))
        torch.testing.assert_close(observed, expected[:, :130], rtol=0.0, atol=0.0)

        grad_output = torch.randn_like(observed)
        observed.backward(grad_output)
        dequantized = dequantize_blockwise_fp8_weight(
            codes,
            scales,
            meta,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(
            inputs.grad,
            grad_output.float() @ dequantized,
            rtol=0.0,
            atol=0.0,
        )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "DeepGEMM parity requires remote CUDA execution",
    )
    def test_deepgemm_grouped_fp8_matches_reference_and_high_precision_dgrad(
        self,
    ) -> None:
        from mirai.core.moe.runtime.gemm import probe_backend

        device = torch.device("cuda")
        verdict = probe_backend("deepgemm_fp8", device=device)
        if not verdict.available:
            self.skipTest(verdict.reason)
        torch.manual_seed(126)
        weights = torch.randn(3, 256, 256, device=device) * 0.08
        payloads = [quantize_blockwise_fp8_weight(weight) for weight in weights]
        codes = torch.stack([payload[0] for payload in payloads])
        scales = torch.stack([payload[1] for payload in payloads])
        meta = payloads[0][2]
        reference_input = torch.randn(
            3, 17, 256, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        native_input = reference_input.detach().clone().requires_grad_(True)
        reference = blockwise_fp8_batched_linear(
            reference_input, codes, scales, meta
        )
        native = deepgemm_blockwise_fp8_batched_linear(
            native_input, codes, scales, meta
        )
        torch.testing.assert_close(native, reference, rtol=2e-2, atol=2e-2)
        grad = torch.randn_like(reference)
        reference.backward(grad)
        native.backward(grad)
        torch.testing.assert_close(
            native_input.grad,
            reference_input.grad,
            rtol=1e-5,
            atol=2e-6,
        )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "DeepGEMM routed parity requires remote CUDA execution",
    )
    def test_deepgemm_routed_fp8_gather_empty_groups_and_weighted_gradients(
        self,
    ) -> None:
        from mirai.core.moe.runtime.gemm import probe_backend

        device = torch.device("cuda")
        verdict = probe_backend("deepgemm_fp8", device=device)
        if not verdict.available:
            self.skipTest(verdict.reason)
        torch.manual_seed(127)
        weights = torch.randn(4, 256, 256, device=device) * 0.06
        payloads = [quantize_blockwise_fp8_weight(weight) for weight in weights]
        codes = torch.stack([payload[0] for payload in payloads])
        scales = torch.stack([payload[1] for payload in payloads])
        meta = payloads[0][2]
        assignment_rows = torch.tensor(
            [3, 0, 5, 1, 2, 4], device=device, dtype=torch.int64
        )
        layout = RoutedGroupLayout(
            boundaries=torch.tensor([1, 1, 5, 6], device=device, dtype=torch.int32),
            assignment_rows=assignment_rows,
            token_count=3,
            top_k=2,
            group_count=4,
        )
        reference_input = torch.randn(
            3, 256, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        reference_scores = torch.randn(
            3, 2, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        native_scores = reference_scores.detach().clone().requires_grad_(True)
        grouped_input = reference_input.index_select(
            0, torch.div(assignment_rows, 2, rounding_mode="floor")
        )
        reference_grouped = blockwise_fp8_batched_linear(
            torch.stack(
                (
                    torch.nn.functional.pad(grouped_input[:1], (0, 0, 0, 3)),
                    grouped_input.new_zeros((4, 256)),
                    grouped_input[1:5],
                    torch.nn.functional.pad(grouped_input[5:], (0, 0, 0, 3)),
                )
            ),
            codes,
            scales,
            meta,
        )
        reference_grouped = torch.cat(
            (reference_grouped[0, :1], reference_grouped[2, :4], reference_grouped[3, :1])
        )
        reference_grouped.retain_grad()
        assignment = torch.empty_like(reference_grouped).index_copy(
            0, assignment_rows, reference_grouped
        )
        reference = (
            assignment.view(3, 2, -1) * reference_scores[..., None]
        ).sum(1)
        gathered_native_input = reference_input.detach().clone().requires_grad_(True)
        native_grouped = deepgemm_blockwise_fp8_routed_linear(
            gathered_native_input,
            codes,
            scales,
            meta,
            layout,
            RoutedFusionSpec(gather_tokens=True),
        )
        torch.testing.assert_close(native_grouped, reference_grouped, rtol=2e-2, atol=2e-2)
        native_grouped_input = grouped_input.detach().clone().requires_grad_(True)
        native = deepgemm_blockwise_fp8_routed_linear(
            native_grouped_input, codes, scales, meta, layout,
            RoutedFusionSpec(output=RoutedOutputMode.WEIGHTED_TOKEN_REDUCTION),
            routing_weights=native_scores,
        )
        torch.testing.assert_close(native, reference, rtol=2e-2, atol=2e-2)
        grad = torch.randn_like(native)
        reference.backward(grad)
        native.backward(grad)
        native_grouped.backward(reference_grouped.grad)
        torch.testing.assert_close(
            gathered_native_input.grad, reference_input.grad, rtol=1e-5, atol=2e-5
        )
        torch.testing.assert_close(native_scores.grad, reference_scores.grad, rtol=2e-2, atol=2e-2)

    def test_blockwise_fp8_packed_grouped_experts_execute_exactly_after_restore(
        self,
    ) -> None:
        class DenseExperts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 2
                self.w1 = nn.Parameter(torch.randn(2, 130, 129), requires_grad=False)
                self.w2 = nn.Parameter(torch.randn(2, 129, 130), requires_grad=False)
                self.w3 = nn.Parameter(torch.randn(2, 130, 129), requires_grad=False)

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.experts = DenseExperts()

        torch.manual_seed(126)
        source = Root()
        target = Root()
        for root in (source, target):
            quantize_compressed_weights_modules(
                root,
                quant_format="fp8",
                expert_weight_access="active_dequant",
            )
        sample = torch.randn(3, 129, requires_grad=True)
        counts = torch.tensor([2, 1], dtype=torch.int64)
        expected = source.experts.run_for_loop(sample, counts)
        expected.square().sum().backward()
        expected_grad = sample.grad.detach().clone()

        tensors, manifest = export_compressed_weights_packed_state(source)
        spec = manifest["modules"]["experts"]
        self.assertEqual(spec["quant_format"], "fp8")
        self.assertEqual(
            set(spec["tensors"]),
            {
                f"{key}_{suffix}"
                for key in ("w1", "w2", "w3")
                for suffix in ("fp8", "fp8_scale")
            },
        )
        offline = load_grouped_expert_source(spec, tensors)
        load_compressed_weights_packed_state(target, tensors, manifest)
        restored_sample = sample.detach().clone().requires_grad_(True)
        observed = target.experts.run_for_loop(restored_sample, counts)
        observed.square().sum().backward()
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            restored_sample.grad,
            expected_grad,
            rtol=0.0,
            atol=0.0,
        )
        for key in ("w1", "w2", "w3"):
            torch.testing.assert_close(
                offline._dequantize(key, dtype=torch.float32),
                source.experts._dequantize(key, dtype=torch.float32),
                rtol=0.0,
                atol=0.0,
            )

    def test_mxfp8_e4m3_uses_round_up_ue8m0_and_rn_even(self) -> None:
        block = torch.zeros(32, dtype=torch.float32)
        block[0] = 448.0
        block[1] = 1.0625
        block[2] = 1.1875
        encoded, scales, global_scale, meta = quantize_microscaling(
            "mxfp8_e4m3",
            block,
        )

        self.assertEqual(tuple(encoded.shape), (1, 32))
        self.assertEqual(int(scales[0]), 127)
        self.assertEqual(int(encoded[0, 1]), 0x38)
        self.assertEqual(int(encoded[0, 2]), 0x3A)
        self.assertEqual(float(global_scale), 1.0)
        self.assertEqual(microscaling_stored_bytes("mxfp8_e4m3", 32), 33)
        self.assertEqual(microscaling_stored_bytes("mxfp8_e4m3", 33), 66)

        restored = dequantize_microscaling(
            encoded,
            scales,
            global_scale,
            meta,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(float(restored[0]), 448.0)
        self.assertEqual(float(restored[1]), 1.0)
        self.assertEqual(float(restored[2]), 1.25)

        overflow_block = torch.zeros(32, dtype=torch.float32)
        overflow_block[0] = 449.0
        _, overflow_scale, _, _ = quantize_microscaling(
            "mxfp8_e4m3",
            overflow_block,
        )
        self.assertEqual(int(overflow_scale[0]), 128)

        invalid = encoded.clone()
        invalid[0, 0] = 0x7F
        with self.assertRaisesRegex(ValueError, "E4M3 NaN"):
            validate_microscaling_payload(
                invalid,
                scales,
                global_scale,
                meta,
            )

    def test_mxfp8_e4m3_packed_state_roundtrip_uses_generic_mx_roles(
        self,
    ) -> None:
        torch.manual_seed(127)
        source = nn.Sequential(
            CompressedLinear(
                nn.Linear(32, 8, bias=False),
                quant_format="mxfp8_e4m3",
            )
        )
        target = nn.Sequential(
            CompressedLinear(
                nn.Linear(32, 8, bias=False),
                quant_format="mxfp8_e4m3",
            )
        )
        sample = torch.randn(3, 32, requires_grad=True)
        expected = source(sample)
        expected.square().sum().backward()
        expected_grad = sample.grad.detach().clone()

        tensors, manifest = export_compressed_weights_packed_state(source)
        tensor_roles = manifest["modules"]["0"]["tensors"]
        self.assertEqual(manifest["modules"]["0"]["quant_format"], "mxfp8_e4m3")
        self.assertEqual(
            set(tensor_roles),
            {"weight_mx", "weight_mx_scale", "weight_mx_global"},
        )
        self.assertNotIn("fp4", " ".join(tensor_roles))

        load_compressed_weights_packed_state(target, tensors, manifest)
        restored_sample = sample.detach().clone().requires_grad_(True)
        observed = target(restored_sample)
        observed.square().sum().backward()
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            restored_sample.grad,
            expected_grad,
            rtol=0.0,
            atol=0.0,
        )

    def test_mxfp8_e4m3_grouped_expert_artifact_roundtrip(self) -> None:
        class DenseExperts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 2
                self.w1 = nn.Parameter(torch.randn(2, 8, 32), requires_grad=False)
                self.w2 = nn.Parameter(torch.randn(2, 32, 8), requires_grad=False)
                self.w3 = nn.Parameter(torch.randn(2, 8, 32), requires_grad=False)

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.experts = DenseExperts()

        torch.manual_seed(129)
        source = Root()
        target = Root()
        quantize_compressed_weights_modules(
            source,
            quant_format="mxfp8_e4m3",
            expert_weight_access="active_dequant",
        )
        quantize_compressed_weights_modules(
            target,
            quant_format="mxfp8_e4m3",
            expert_weight_access="active_dequant",
        )
        expected = tuple(
            source.experts._dequantize(key, dtype=torch.float32)
            for key in ("w1", "w2", "w3")
        )

        tensors, manifest = export_compressed_weights_packed_state(source)
        spec = manifest["modules"]["experts"]
        self.assertEqual(
            set(spec["tensors"]),
            {
                f"{key}_{suffix}"
                for key in ("w1", "w2", "w3")
                for suffix in ("mx", "mx_scale", "mx_global")
            },
        )
        offline = load_grouped_expert_source(spec, tensors)
        load_compressed_weights_packed_state(target, tensors, manifest)
        for key, expected_weight in zip(("w1", "w2", "w3"), expected):
            torch.testing.assert_close(
                offline._dequantize(key, dtype=torch.float32),
                expected_weight,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                target.experts._dequantize(key, dtype=torch.float32),
                expected_weight,
                rtol=0.0,
                atol=0.0,
            )

    def test_learned_rotation_is_orthogonal_and_nonregressing(self) -> None:
        torch.manual_seed(131)
        weights = (
            torch.randn(3, 11, 16) * torch.linspace(0.1, 2.0, 16),
            torch.randn(3, 11, 16) * torch.linspace(2.0, 0.1, 16),
        )
        result = learn_groupwise_expert_rotation(
            weights,
            group_size=4,
            optimization_steps=16,
            learning_rate=0.02,
            row_chunk_size=24,
            checkpoint_interval=4,
            device="cpu",
            max_workspace_gib=0.01,
        )
        identity = torch.eye(4)
        torch.testing.assert_close(
            result.rotation.T @ result.rotation,
            identity,
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertLessEqual(
            result.optimized_relative_error,
            result.initial_relative_error,
        )

    def test_learned_rotation_packed_state_roundtrips_with_explicit_gate(
        self,
    ) -> None:
        class DenseExperts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 3
                self.w1 = nn.Parameter(
                    torch.randn(3, 12, 16),
                    requires_grad=False,
                )
                self.w2 = nn.Parameter(
                    torch.randn(3, 16, 12),
                    requires_grad=False,
                )
                self.w3 = nn.Parameter(
                    torch.randn(3, 12, 16),
                    requires_grad=False,
                )

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.experts = DenseExperts()

        torch.manual_seed(137)
        source = Root()
        report = quantize_compressed_weights_modules(
            source,
            group_sizes=4,
            quant_format="int8",
            learn_expert_rotations=True,
            rotation_optimization_steps=8,
            rotation_learning_rate=0.02,
            rotation_row_chunk_size=32,
            rotation_checkpoint_interval=4,
            rotation_device="cpu",
            rotation_max_workspace_gib=0.01,
        )
        self.assertEqual(report.grouped_expert_modules, 1)
        self.assertIsInstance(source.experts, CompressedGroupedExperts)
        self.assertTrue(
            torch.equal(
                source.experts.expert_rotation("w1"),
                source.experts.expert_rotation("w3"),
            )
        )
        tensors, manifest = export_compressed_weights_packed_state(source)
        self.assertEqual(manifest["schema_version"], 5)
        self.assertEqual(
            set(manifest["modules"]["experts"]["rotations"]),
            {"w1", "w2", "w3"},
        )
        with self.assertRaises(ValueError):
            validate_learned_rotation_selection(manifest, "off")
        validate_learned_rotation_selection(manifest, "learned")
        offline_source = load_grouped_expert_source(
            manifest["modules"]["experts"],
            tensors,
        )
        for key in ("w1", "w2", "w3"):
            torch.testing.assert_close(
                offline_source.expert_rotation(key),
                source.experts.expert_rotation(key),
            )

        target = Root()
        prepare_compressed_weights_modules_from_manifest(target, manifest)
        load_compressed_weights_packed_state(
            target,
            tensors,
            manifest,
            strict=True,
        )
        for key in ("w1", "w2", "w3"):
            torch.testing.assert_close(
                getattr(target.experts, key),
                getattr(source.experts, key),
            )
            torch.testing.assert_close(
                target.experts.expert_rotation(key),
                source.experts.expert_rotation(key),
            )

    @unittest.skipIf(torch is None, "torch is required")
    def test_structured_2_4_grouped_expert_matches_pruned_reference(self) -> None:
        class DenseExperts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 2
                self.w1 = nn.Parameter(torch.randn(2, 8, 8), requires_grad=False)
                self.w2 = nn.Parameter(torch.randn(2, 8, 8), requires_grad=False)
                self.w3 = nn.Parameter(torch.randn(2, 8, 8), requires_grad=False)

        torch.manual_seed(11)
        sparse = StructuredSparse24GroupedExperts(DenseExperts(), backend="reference")
        tokens = torch.randn(4, 8, requires_grad=True)
        scores = torch.ones(4, 1)
        indices = torch.tensor([[0], [1], [0], [1]])
        output = sparse.run_direct_routed(tokens, scores, indices)
        expected = torch.zeros_like(output)
        for token_id, expert_id in enumerate(indices[:, 0].tolist()):
            selected = tokens[token_id : token_id + 1]
            gate = sparse_2_4_linear(selected, sparse._state("w1", expert_id))
            up = sparse_2_4_linear(selected, sparse._state("w3", expert_id))
            expected[token_id] = sparse_2_4_linear(
                torch.nn.functional.silu(gate) * up,
                sparse._state("w2", expert_id),
            )[0]
        torch.testing.assert_close(output, expected)
        output.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(tokens.grad).all()))
        payload = sparse.state_dict()
        restored = StructuredSparse24GroupedExperts(
            DenseExperts(), backend="reference"
        )
        restored.load_state_dict(payload)
        torch.testing.assert_close(
            restored.run_direct_routed(
                tokens.detach(), scores, indices
            ),
            output.detach(),
        )

    def test_mixed_precision_plan_respects_budget_and_roundtrips(self) -> None:
        rows = [
            ExpertPrecisionEvidence(
                expert_id=0,
                weight_numel=100,
                routing_frequency=10.0,
                format_error={"gguf_iq3": 4.0, "int8": 0.1},
            ),
            ExpertPrecisionEvidence(
                expert_id=1,
                weight_numel=100,
                routing_frequency=1.0,
                format_error={"gguf_iq3": 4.0, "int8": 0.1},
            ),
        ]
        plan = allocate_expert_precision(
            rows,
            budget_bytes=138,
            allowed_formats=("gguf_iq3", "int8"),
        )
        self.assertEqual(plan.formats, ("int8", "gguf_iq3"))
        self.assertLessEqual(plan.estimated_bytes, plan.budget_bytes)
        with tempfile.TemporaryDirectory() as directory:
            path = plan.save(Path(directory) / "precision.json")
            self.assertEqual(ExpertPrecisionPlan.load(path), plan)

    @unittest.skipIf(torch is None, "torch is required")
    def test_structured_2_4_reference_has_exact_pattern_and_gradients(self) -> None:
        torch.manual_seed(7)
        weight = torch.randn(5, 8)
        state = prune_to_2_4(weight)
        validate_2_4(state)
        inputs = torch.randn(3, 8, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        actual = sparse_2_4_linear(inputs, state)
        expected = torch.nn.functional.linear(reference_inputs, state.dense())
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad)

    @classmethod
    def setUpClass(cls) -> None:
        register_builtin_components()

    def _lingbot_config(self, *, strategy: str = "compressed_weights") -> TrainingConfig:
        return TrainingConfig(
            model=ModelConfig(
                type="lingbot-video",
                path="./nonexistent/lingbot-video",
                params=ModelParams(
                    variant="tiny-video",
                    flow_shift=3.0,
                    strict_native_assets=False,
                    latent_channels=1,
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
            training=TrainingSection(seed=11, batch_size=1, max_steps=1),
            memory=MemoryConfig(
                frozen_weight_quantization="int8",
                frozen_weight_quantization_strategy=strategy,
            ),
        )

    def test_linear_wrapper_preserves_forward_shape_and_close_values(self) -> None:
        torch.manual_seed(7)
        base = nn.Linear(16, 12, bias=True)
        x = torch.randn(3, 5, 16)

        expected = base(x)
        quantized = CompressedLinear(base, group_sizes=16)
        observed = quantized(x)

        self.assertEqual(observed.shape, expected.shape)
        self.assertEqual(quantized.weight_int8.dtype, torch.int8)
        torch.testing.assert_close(observed, expected, rtol=0.04, atol=0.04)

    def test_quantization_workspace_bounds_rotation_chunk(self) -> None:
        import mirai.core.models.compressed_weights as compressed_weights

        weight = torch.randn(256, 1024, dtype=torch.bfloat16)
        observed_rows: list[int] = []
        # ``_quantize_weight`` looks up ``_rotate_last_dim`` in its defining module
        # (the ``quant`` submodule), so patch it there, not on the package facade.
        original = compressed_weights.quant._rotate_last_dim

        def record_rows(value, group_size, *, inverse, rotation=None):
            observed_rows.append(int(value.shape[0]))
            return original(
                value,
                group_size,
                inverse=inverse,
                rotation=rotation,
            )

        with patch.object(
            compressed_weights.quant,
            "_rotate_last_dim",
            side_effect=record_rows,
        ):
            quantized, scale, _ = _quantize_weight(
                weight,
                group_size=16,
                workspace_bytes=1024 * 1024,
            )

        self.assertEqual(tuple(quantized.shape), tuple(weight.shape))
        self.assertEqual(tuple(scale.shape), (256, 1))
        self.assertGreater(len(observed_rows), 1)
        self.assertLessEqual(max(observed_rows), 64)

    def test_lingbot_quantizer_replaces_linears_and_grouped_experts(self) -> None:
        cfg = self._lingbot_config()
        pipeline = LingBotVideoPipeline(cfg.model)

        report = quantize_compressed_weights_modules(pipeline.transformer, group_sizes=16)

        self.assertGreater(report.linear_modules, 0)
        self.assertGreater(report.grouped_expert_modules, 0)
        self.assertGreater(report.quantized_numel, 0)
        self.assertGreater(
            sum(1 for module in pipeline.modules() if isinstance(module, CompressedLinear)),
            0,
        )
        self.assertGreater(
            sum(1 for module in pipeline.modules() if isinstance(module, CompressedGroupedExperts)),
            0,
        )

    def test_packed_state_roundtrip_restores_quantized_graph(self) -> None:
        torch.manual_seed(13)
        source = nn.Sequential(nn.Linear(16, 12), nn.SiLU(), nn.Linear(12, 8, bias=False))
        target = nn.Sequential(nn.Linear(16, 12), nn.SiLU(), nn.Linear(12, 8, bias=False))
        target.load_state_dict(source.state_dict())
        quantize_compressed_weights_modules(source, group_sizes=16)
        quantize_compressed_weights_modules(target, group_sizes=16)

        x = torch.randn(2, 3, 16)
        expected = source(x)
        tensors, manifest = export_compressed_weights_packed_state(source)
        first_key = next(iter(tensors))
        extra_tensors = dict(tensors)
        extra_tensors["unexpected"] = tensors[first_key]
        with self.assertRaisesRegex(ValueError, "Unexpected compressed_weights packed tensors"):
            load_compressed_weights_packed_state(target, extra_tensors, manifest)

        report = load_compressed_weights_packed_state(target, tensors, manifest)
        observed = target(x)

        self.assertEqual(report.linear_modules, 2)
        self.assertEqual(report.grouped_expert_modules, 0)
        self.assertEqual(report.quantized_tensors, 2)
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    def test_lingbot_packed_state_roundtrip_includes_grouped_experts(self) -> None:
        cfg = self._lingbot_config()
        torch.manual_seed(19)
        source = LingBotVideoPipeline(cfg.model)
        torch.manual_seed(23)
        target = LingBotVideoPipeline(cfg.model)
        quantize_compressed_weights_modules(
            source.transformer,
            group_sizes=16,
            expert_weight_access="active_dequant",
        )
        quantize_compressed_weights_modules(
            target.transformer,
            group_sizes=16,
            expert_weight_access="active_dequant",
        )

        tensors, manifest = export_compressed_weights_packed_state(source.transformer)
        report = load_compressed_weights_packed_state(target.transformer, tensors, manifest)

        self.assertGreater(report.linear_modules, 0)
        self.assertGreater(report.grouped_expert_modules, 0)
        self.assertEqual(report.expert_weight_access, "active_dequant")
        self.assertGreater(report.quantized_numel, 0)

    def test_packed_experts_can_override_disk_access_with_resident_weights(self) -> None:
        cfg = self._lingbot_config()
        source = LingBotVideoPipeline(cfg.model)
        target = LingBotVideoPipeline(cfg.model)
        quantize_compressed_weights_modules(
            source.transformer,
            group_sizes=16,
            expert_weight_access="active_dequant",
        )
        quantize_compressed_weights_modules(
            target.transformer,
            group_sizes=16,
            expert_weight_access="active_dequant",
        )
        tensors, manifest = export_compressed_weights_packed_state(source.transformer)

        report = load_compressed_weights_packed_state(
            target.transformer,
            tensors,
            manifest,
            expert_weight_access_override="full_dequant",
        )

        experts = next(
            module
            for module in target.transformer.modules()
            if isinstance(module, CompressedGroupedExperts)
        )
        self.assertEqual(report.expert_weight_access, "full_dequant")
        self.assertFalse(experts.prefers_for_loop())
        self.assertTrue(hasattr(experts, "w1_int8"))

    def test_lingbot_pipeline_loads_from_packed_compressed_weights_state(self) -> None:
        try:
            import safetensors  # noqa: F401
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("safetensors not installed")
        cfg = self._lingbot_config()
        torch.manual_seed(31)
        source = LingBotVideoPipeline(cfg.model)
        source.enable_quantized_frozen_weights("int8", strategy="compressed_weights")
        latents = torch.randn(1, 1, 1, 1, 1)
        timesteps = torch.tensor([0.5], dtype=torch.float32)
        text = {"lingbot": torch.randn(1, 1, 16)}
        expected = source.forward(latents, timesteps, text)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lingbot_compressed_weights.safetensors"
            save_compressed_weights_packed_state(path, source.transformer)
            target = LingBotVideoPipeline(
                cfg.model,
                memory_config=MemoryConfig(
                    frozen_weight_quantization="int8",
                    frozen_weight_quantization_strategy="compressed_weights",
                    frozen_weight_packed_state_path=str(path),
                ),
            )
            observed = target.forward(latents, timesteps, text)

        report = target.get_quantized_frozen_weight_report()
        self.assertIsNotNone(report)
        self.assertGreater(report["linear_modules"], 0)
        self.assertGreater(report["grouped_expert_modules"], 0)
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    def test_lingbot_packed_state_requires_quantized_memory_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.safetensors"
            with self.assertRaisesRegex(ValueError, "packed state requires"):
                LingBotVideoPipeline(
                    self._lingbot_config().model,
                    memory_config=MemoryConfig(
                        frozen_weight_quantization="none",
                        frozen_weight_packed_state_path=str(path),
                    ),
                )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "NF4 packed-state roundtrip requires CUDA",
    )
    def test_lingbot_pipeline_loads_nf4_artifact_without_requantization(self) -> None:
        try:
            import bitsandbytes  # noqa: F401
            import safetensors  # noqa: F401
        except (ImportError, OSError):  # pragma: no cover - environment-specific
            self.skipTest("bitsandbytes and safetensors are required")
        cfg = self._lingbot_config(strategy="auto")
        torch.manual_seed(37)
        source = LingBotVideoPipeline(cfg.model)
        source.enable_quantized_frozen_weights("nf4", block_size=64)
        latents = torch.randn(1, 1, 1, 1, 1, device="cuda")
        timesteps = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        text = {"lingbot": torch.randn(1, 1, 16, device="cuda")}
        source.transformer.to("cuda")
        expected = source.forward(latents, timesteps, text)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lingbot_nf4.safetensors"
            save_compressed_weights_packed_state(path, source.transformer)
            with patch(
                "mirai.core.models.compressed_weights.execution.linear._nf4_quantize_2d",
                side_effect=AssertionError("packed NF4 load must not quantize linear weights"),
            ), patch(
                "mirai.core.models.compressed_weights.execution.experts._nf4_quantize_2d",
                side_effect=AssertionError("packed NF4 load must not quantize expert weights"),
            ):
                target = LingBotVideoPipeline(
                    cfg.model,
                    memory_config=MemoryConfig(
                        frozen_weight_quantization="nf4",
                        frozen_weight_quantization_strategy="auto",
                        frozen_weight_packed_state_path=str(path),
                        quantization_block_size=64,
                    ),
                )
            target.transformer.to("cuda")
            observed = target.forward(latents, timesteps, text)

        self.assertEqual(target.get_quantized_frozen_weight_report()["quant_format"], "nf4")
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    def test_packed_state_safetensors_artifact_roundtrip(self) -> None:
        try:
            import safetensors  # noqa: F401
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("safetensors not installed")
        torch.manual_seed(29)
        source = nn.Sequential(nn.Linear(16, 8), nn.SiLU(), nn.Linear(8, 4))
        target = nn.Sequential(nn.Linear(16, 8), nn.SiLU(), nn.Linear(8, 4))
        target.load_state_dict(source.state_dict())
        quantize_compressed_weights_modules(source, group_sizes=16)
        quantize_compressed_weights_modules(target, group_sizes=16)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "packed_compressed_weights.safetensors"
            saved = save_compressed_weights_packed_state(path, source, metadata={"source": "unit-test"})
            report = load_compressed_weights_packed_state_file(saved, target)

        self.assertEqual(report.linear_modules, 2)
        self.assertEqual(report.quantized_tensors, 2)
        x = torch.randn(1, 2, 16)
        torch.testing.assert_close(target(x), source(x), rtol=0.0, atol=0.0)

    def test_sharded_packed_state_roundtrip_loads_from_index(self) -> None:
        try:
            import safetensors  # noqa: F401
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("safetensors not installed")
        source = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1024))
        target = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1024))
        target.load_state_dict(source.state_dict())
        quantize_compressed_weights_modules(source, group_sizes=16)
        quantize_compressed_weights_modules(target, group_sizes=16)
        with tempfile.TemporaryDirectory() as tmp, active_moe_optimization_policy(
            MoEOptimizationPolicy(packed_shard_size_mb=1)
        ):
            path = save_compressed_weights_packed_state(
                Path(tmp) / "packed.safetensors",
                source,
                storage_alignment_bytes=4096,
            )
            self.assertTrue(path.name.endswith(".index.json"))
            report = load_compressed_weights_packed_state_file(path, target)
            shards = list(Path(tmp).glob("*.safetensors"))
            self.assertGreater(len(shards), 1)
            for shard in shards:
                self.assertEqual(
                    read_safetensors_storage_alignment(shard).file_size % 4096,
                    0,
                )

        self.assertEqual(report.linear_modules, 2)
        x = torch.randn(1, 1024)
        torch.testing.assert_close(target(x), source(x), rtol=0.0, atol=0.0)

    def test_lazy_packed_membership_does_not_materialize_tensor(self) -> None:
        import mirai.core.models.compressed_weights as compressed_weights

        mapping = LazyPackedTensorMapping.__new__(LazyPackedTensorMapping)
        mapping._weight_map = {"weight": "shard.safetensors"}
        mapping._safe_open = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("membership materialized a tensor")
        )
        mapping._root = Path(".")
        self.assertIn("weight", mapping)
        self.assertNotIn("missing", mapping)

    def test_lingbot_trainer_dry_run_with_compressed_weights(self) -> None:
        trainer = Trainer(self._lingbot_config(strategy="compressed_weights"))

        loss, _metrics = trainer.compute_loss(
            {
                "latents": torch.randn(1, 1, 1, 1, 1),
                "text_embeds": torch.randn(1, 16),
            }
        )
        report = trainer.pipeline.get_quantized_frozen_weight_report()

        self.assertTrue(trainer.pipeline.has_quantized_frozen_weights())
        self.assertIsNotNone(report)
        self.assertGreater(report["linear_modules"], 0)
        self.assertGreater(report["grouped_expert_modules"], 0)
        self.assertTrue(math.isfinite(float(loss.detach().cpu().item())))


@unittest.skipIf(torch is None, "torch not installed")
class CompressedWeightPackedPreloadTests(unittest.TestCase):
    def _saved_lazy_mapping(self, tmp: str):
        import mirai.core.models.compressed_weights as compressed_weights

        torch.manual_seed(31)
        source = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 16))
        quantize_compressed_weights_modules(source, group_sizes=16)
        path = save_compressed_weights_packed_state(
            Path(tmp) / "packed_compressed_weights.safetensors", source
        )
        return LazyPackedTensorMapping(path)

    def test_preload_off_is_identity(self) -> None:
        import mirai.core.models.compressed_weights as compressed_weights

        try:
            import safetensors  # noqa: F401
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("safetensors not installed")
        with tempfile.TemporaryDirectory() as tmp:
            base = self._saved_lazy_mapping(tmp)
            wrapped, info = materialize_packed_tensors(base, "off")
            self.assertIs(wrapped, base)
            self.assertEqual(info["effective"], "off")

    def test_preload_ram_serves_bit_identical_slices_from_memory(self) -> None:
        import mirai.core.models.compressed_weights as compressed_weights

        try:
            import safetensors  # noqa: F401
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("safetensors not installed")
        with tempfile.TemporaryDirectory() as tmp:
            base = self._saved_lazy_mapping(tmp)
            keys = list(base)
            self.assertTrue(keys)
            wrapped, info = materialize_packed_tensors(base, "ram")
            self.assertIsInstance(wrapped, PreloadedPackedTensorMapping)
            self.assertEqual(info["effective"], "ram")
            self.assertEqual(info["tensors"], len(keys))
            self.assertGreater(info["bytes"], 0)
            for key in keys:
                self.assertTrue(
                    torch.equal(wrapped.get_slice(key, 0), base.get_slice(key, 0)),
                    f"slice mismatch for {key}",
                )
                self.assertTrue(torch.equal(wrapped[key], base[key]))
            # Membership / iteration still delegate to the base mapping.
            self.assertEqual(set(wrapped), set(base))
            self.assertIn(keys[0], wrapped)

    def test_preload_fails_fast_when_below_memory_floor(self) -> None:
        """RAM is the only primary path: below the floor we FAIL FAST rather than
        silently auto-degrading to disk streaming."""
        import types
        import mirai.core.models.compressed_weights as compressed_weights
        from mirai.core.training.residency.memory_safety import (
            configure_memory_safety,
            current_memory_safety_policy,
        )

        try:
            import safetensors  # noqa: F401
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("safetensors not installed")
        try:
            import psutil  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("psutil required for headroom fail-fast test")
        previous = current_memory_safety_policy()
        try:
            configure_memory_safety(
                types.SimpleNamespace(
                    cuda_memory_fraction=previous.cuda_memory_fraction,
                    minimum_system_memory_gib=1.0e9,
                )
            )
            with tempfile.TemporaryDirectory() as tmp:
                base = self._saved_lazy_mapping(tmp)
                with self.assertRaises(RuntimeError) as ctx:
                    materialize_packed_tensors(base, "pinned")
                msg = str(ctx.exception)
                # Error must name the shortfall and both remedies.
                self.assertIn("host RAM", msg)
                self.assertIn("packed_state_preload='off'", msg)
                self.assertIn("minimum_system_memory_gib", msg)
        finally:
            configure_memory_safety(
                types.SimpleNamespace(
                    cuda_memory_fraction=previous.cuda_memory_fraction,
                    minimum_system_memory_gib=previous.minimum_system_memory_gib,
                )
            )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "pinned preload requires CUDA"
    )
    def test_preload_pinned_produces_page_locked_tensors(self) -> None:
        import mirai.core.models.compressed_weights as compressed_weights

        try:
            import safetensors  # noqa: F401
        except ModuleNotFoundError:  # pragma: no cover
            self.skipTest("safetensors not installed")
        with tempfile.TemporaryDirectory() as tmp:
            base = self._saved_lazy_mapping(tmp)
            wrapped, info = materialize_packed_tensors(base, "pinned")
            self.assertEqual(info["effective"], "pinned")
            for key in list(base):
                self.assertTrue(wrapped._cache[str(key)].is_pinned())
                self.assertTrue(
                    torch.equal(wrapped.get_slice(key, 0), base.get_slice(key, 0))
                )

    def _quantized_grouped_stacks(self, *, num_experts: int, out_f: int, in_f: int):
        torch.manual_seed(7)
        stacks: dict[str, "torch.Tensor"] = {}
        scales: dict[str, "torch.Tensor"] = {}
        groups: dict[str, int] = {}
        shapes: dict[str, tuple[int, int, int]] = {}
        dims = {"w1": (out_f, in_f), "w3": (out_f, in_f), "w2": (in_f, out_f)}
        for key, (o, i) in dims.items():
            weight = torch.randn(num_experts, o, i, dtype=torch.bfloat16) * 0.05
            q, s, g = _quantize_weight(weight, group_size=16)
            stacks[key] = q
            scales[key] = s.float()
            groups[key] = int(g)
            shapes[key] = (num_experts, o, i)
        return stacks, scales, groups, shapes

    def test_batched_gather_bit_identical_module_attribute(self) -> None:
        """Module-buffer int8 path: index_select gather == per-expert stack."""
        num_experts, out_f, in_f = 8, 32, 48

        class _Base(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = num_experts
                torch.manual_seed(3)
                self.w1 = torch.randn(num_experts, out_f, in_f, dtype=torch.bfloat16) * 0.05
                self.w2 = torch.randn(num_experts, in_f, out_f, dtype=torch.bfloat16) * 0.05
                self.w3 = torch.randn(num_experts, out_f, in_f, dtype=torch.bfloat16) * 0.05

        experts = CompressedGroupedExperts(
            _Base(),
            quant_format="int8",
            expert_weight_access="full_dequant",
        )
        device = torch.device("cpu")
        active = [0, 2, 3, 6]  # deliberately non-contiguous subset
        for key in ("w1", "w2", "w3"):
            with active_moe_optimization_policy(
                MoEOptimizationPolicy(moe_batched_gather=True)
            ):
                q_on, s_on = experts._quantized_expert_batch(key, active, device=device)
            with active_moe_optimization_policy(
                MoEOptimizationPolicy(moe_batched_gather=False)
            ):
                q_off, s_off = experts._quantized_expert_batch(key, active, device=device)
            self.assertTrue(torch.equal(q_on, q_off), f"{key} int8 gather mismatch")
            self.assertTrue(torch.equal(s_on, s_off), f"{key} scale gather mismatch")

    def test_batched_gather_coalesces_unique_device_cache_misses(self) -> None:
        class _Base(nn.Module):
            def __init__(self) -> None:
                self.num_experts = 8
                super().__init__()
                self.w1 = torch.randn(8, 32, 48, dtype=torch.bfloat16)
                self.w2 = torch.randn(8, 48, 32, dtype=torch.bfloat16)
                self.w3 = torch.randn(8, 32, 48, dtype=torch.bfloat16)

        experts = CompressedGroupedExperts(
            _Base(), quant_format="int8", expert_weight_access="full_dequant"
        )
        experts.bind_expert_device_cache(
            ExpertDeviceCache(1 << 30), namespace="test"
        )
        device = torch.device("cpu")
        with active_moe_optimization_policy(
            MoEOptimizationPolicy(moe_batched_gather=False)
        ):
            experts._quantized_expert_batch("w1", [0], device=device)
        telemetry_before = experts.expert_device_cache_snapshot()
        original = experts._quantized_expert_batch_gather
        with (
            patch.object(
                experts,
                "_quantized_expert_batch_gather",
                wraps=original,
            ) as gather,
            active_moe_optimization_policy(
                MoEOptimizationPolicy(moe_batched_gather=True)
            ),
        ):
            actual_q, actual_s = experts._quantized_expert_batch(
                "w1", [0, 2, 2, 3], device=device
            )
        gather.assert_called_once()
        self.assertEqual(gather.call_args.args[1], [2, 3])
        expected_q = torch.stack(
            [getattr(experts, "w1_int8")[index] for index in [0, 2, 2, 3]]
        )
        expected_s = torch.stack(
            [getattr(experts, "w1_scale")[index].float() for index in [0, 2, 2, 3]]
        )
        self.assertTrue(torch.equal(actual_q, expected_q))
        self.assertTrue(torch.equal(actual_s, expected_s))
        snapshot = experts.expert_device_cache_snapshot()
        self.assertEqual(snapshot["entries"], 3)
        self.assertEqual(
            snapshot["transfer_requested_rows"]
            - telemetry_before["transfer_requested_rows"],
            4,
        )
        self.assertEqual(
            snapshot["transfer_hit_rows"] - telemetry_before["transfer_hit_rows"],
            1,
        )
        self.assertEqual(
            snapshot["transfer_miss_rows"] - telemetry_before["transfer_miss_rows"],
            3,
        )
        self.assertEqual(
            snapshot["transfer_unique_rows"]
            - telemetry_before["transfer_unique_rows"],
            2,
        )
        self.assertEqual(
            snapshot["transfer_deduplicated_rows"]
            - telemetry_before["transfer_deduplicated_rows"],
            1,
        )
        self.assertEqual(
            snapshot["transfer_coalesced_requests"]
            - telemetry_before["transfer_coalesced_requests"],
            1,
        )
        self.assertGreater(
            snapshot["transfer_bytes"] - telemetry_before["transfer_bytes"], 0
        )

    def test_cache_aware_batched_gather_preserves_output_and_gradients(self) -> None:
        class _Base(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 4
                torch.manual_seed(17)
                self.w1 = torch.randn(4, 24, 16, dtype=torch.bfloat16) * 0.05
                self.w2 = torch.randn(4, 16, 24, dtype=torch.bfloat16) * 0.05
                self.w3 = torch.randn(4, 24, 16, dtype=torch.bfloat16) * 0.05

        experts = CompressedGroupedExperts(
            _Base(),
            quant_format="int8",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=4,
        )
        experts.bind_expert_device_cache(
            ExpertDeviceCache(1 << 30), namespace="gradient-test"
        )
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            indices = torch.tensor(
                [[0, 2], [2, 3], [0, 2], [3, 2]], device=device
            )

            def execute(enabled: bool, device=device, indices=indices):
                experts._expert_device_cache.clear()
                tokens = torch.randn(
                    4, 16, dtype=torch.bfloat16, device=device
                ).requires_grad_(True)
                scores = torch.softmax(
                    torch.randn(4, 2, device=device), dim=-1
                ).requires_grad_(True)
                with active_moe_optimization_policy(
                    MoEOptimizationPolicy(moe_batched_gather=enabled)
                ):
                    output = experts.run_direct_routed(tokens, scores, indices)
                    output.float().square().mean().backward()
                return output.detach(), tokens.detach(), tokens.grad, scores.grad

            torch.manual_seed(29)
            reference = execute(False)
            torch.manual_seed(29)
            candidate = execute(True)
            for expected, actual in zip(reference, candidate):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _bound_packed_experts(self, *, num_experts, out_f, in_f, pin=False):
        import mirai.core.models.compressed_weights as compressed_weights

        stacks, scales, groups, shapes = self._quantized_grouped_stacks(
            num_experts=num_experts, out_f=out_f, in_f=in_f
        )
        cache: dict[str, "torch.Tensor"] = {}
        tensor_names: dict[str, str] = {}
        for key in ("w1", "w2", "w3"):
            q = stacks[key].pin_memory() if pin else stacks[key]
            s = scales[key].pin_memory() if pin else scales[key]
            cache[f"{key}_int8"] = q
            cache[f"{key}_scale"] = s
            tensor_names[f"{key}_int8"] = f"{key}_int8"
            tensor_names[f"{key}_scale"] = f"{key}_scale"
        source = PreloadedPackedTensorMapping(dict(cache), cache)
        experts = CompressedGroupedExperts.from_empty(
            num_experts=num_experts,
            quant_format="int8",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=4,
        )
        experts.bind_packed_source(
            source=source, tensor_names=tensor_names, group_sizes=groups, shapes=shapes
        )
        return experts

    def test_batched_gather_packed_source_routing_and_fallback(self) -> None:
        """Preloaded packed source: public entrypoint == per-expert bytes, and a
        pageable cache defers to the per-expert loop (gather returns None)."""
        experts = self._bound_packed_experts(num_experts=8, out_f=32, in_f=48)
        self.assertIsNotNone(experts._packed_source)
        self.assertFalse(hasattr(experts, "w1_int8"))
        device = torch.device("cpu")
        for active in ([2, 3, 4, 5], [1, 2, 5]):  # contiguous + non-contiguous
            for key in ("w1", "w2", "w3"):
                # Pageable (unpinned) CPU cache -> gather declines -> None.
                self.assertIsNone(
                    experts._quantized_expert_batch_gather(key, active, device=device)
                )
                q_loop, s_loop = experts._quantized_expert_batch_per_expert(
                    key, active, device=device
                )
                # Public entrypoint (gather on) still yields identical bytes.
                with active_moe_optimization_policy(
                    MoEOptimizationPolicy(moe_batched_gather=True)
                ):
                    q_on, s_on = experts._quantized_expert_batch(key, active, device=device)
                self.assertTrue(torch.equal(q_on, q_loop), f"{key} int8 mismatch")
                self.assertTrue(torch.equal(s_on, s_loop), f"{key} scale mismatch")
                self.assertEqual(s_on.dtype, torch.float32)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "pinned contiguous-slice gather requires CUDA",
    )
    def test_batched_gather_pinned_contiguous_bit_identical(self) -> None:
        """Pinned cache + contiguous run -> slice-view gather == per-expert stack;
        non-contiguous run declines to the per-expert fallback (None)."""
        experts = self._bound_packed_experts(num_experts=8, out_f=32, in_f=48, pin=True)
        device = torch.device("cuda")
        for key in ("w1", "w2", "w3"):
            contiguous = [2, 3, 4, 5]
            q_g, s_g = experts._quantized_expert_batch_gather(
                key, contiguous, device=device
            )
            q_l, s_l = experts._quantized_expert_batch_per_expert(
                key, contiguous, device=device
            )
            self.assertTrue(torch.equal(q_g, q_l), f"{key} int8 mismatch")
            self.assertTrue(torch.equal(s_g, s_l), f"{key} scale mismatch")
            self.assertEqual(s_g.dtype, torch.float32)
            # Non-contiguous -> gather declines (measured slower than per-expert).
            self.assertIsNone(
                experts._quantized_expert_batch_gather(key, [1, 2, 5], device=device)
            )

    def test_batched_gather_falls_back_for_disk_source(self) -> None:
        """A source without cached_tensor (disk-backed) -> None (per-expert path)."""
        num_experts, out_f, in_f = 4, 32, 48
        stacks, scales, groups, shapes = self._quantized_grouped_stacks(
            num_experts=num_experts, out_f=out_f, in_f=in_f
        )

        class _DiskLike:
            def __init__(self) -> None:
                self._data = {}
                for key in ("w1", "w2", "w3"):
                    self._data[f"{key}_int8"] = stacks[key]
                    self._data[f"{key}_scale"] = scales[key]

            def get_slice(self, name: str, index: int):
                return self._data[str(name)][int(index)]

        experts = CompressedGroupedExperts.from_empty(
            num_experts=num_experts,
            quant_format="int8",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
        )
        experts.bind_packed_source(
            source=_DiskLike(),
            tensor_names={f"{k}_{s}": f"{k}_{s}" for k in ("w1", "w2", "w3") for s in ("int8", "scale")},
            group_sizes=groups,
            shapes=shapes,
        )
        active = [0, 2]
        self.assertIsNone(
            experts._quantized_expert_batch_gather("w1", active, device=torch.device("cpu"))
        )
        # Public entrypoint still succeeds via the per-expert fallback.
        q, s = experts._quantized_expert_batch("w1", active, device=torch.device("cpu"))
        self.assertEqual(tuple(q.shape), (2, out_f, in_f))
        del s

    def test_packed_state_preload_config_plumbing(self) -> None:
        from mirai.core.moe.runtime.specs import MoEOptimizationPolicy

        cfg = TrainingConfig.from_dict(
            {
                "memory": {
                    "frozen_weight_quantization": "int8",
                    "packed_state_preload": "off",
                    "packed_stream_cache_gib": 1.5,
                    "packed_stream_prefetch_depth": 4,
                }
            }
        )
        # Explicit opt-in to disk streaming is honored.
        self.assertEqual(cfg.memory.packed_state_preload, "off")
        policy = MoEOptimizationPolicy.from_memory_config(cfg.memory)
        self.assertEqual(policy.packed_state_preload, "off")
        self.assertEqual(policy.packed_stream_cache_gib, 1.5)
        self.assertEqual(policy.packed_stream_prefetch_depth, 4)
        # RAM-resident "pinned" is the default everywhere (never disk by default).
        self.assertEqual(MemoryConfig().packed_state_preload, "pinned")
        self.assertEqual(
            TrainingConfig.from_dict({"memory": {}}).memory.packed_state_preload,
            "pinned",
        )
        self.assertEqual(MoEOptimizationPolicy().packed_state_preload, "pinned")
        self.assertEqual(MemoryConfig().packed_stream_cache_gib, 0.0)
        self.assertEqual(MoEOptimizationPolicy().packed_stream_cache_gib, 0.0)
        self.assertEqual(MemoryConfig().packed_stream_backend, "staged")
        self.assertEqual(MoEOptimizationPolicy().packed_stream_backend, "staged")
        self.assertEqual(MemoryConfig().packed_stream_prefetch_depth, 0)
        self.assertEqual(MoEOptimizationPolicy().packed_stream_prefetch_depth, 0)
        gds_cfg = TrainingConfig.from_dict(
            {
                "memory": {
                    "packed_state_preload": "off",
                    "packed_stream_backend": "gds",
                }
            }
        )
        self.assertEqual(
            MoEOptimizationPolicy.from_memory_config(
                gds_cfg.memory
            ).packed_stream_backend,
            "gds",
        )
        with self.assertRaises(ValueError):
            MoEOptimizationPolicy(packed_state_preload="bogus")
        with self.assertRaisesRegex(ValueError, "packed_state_preload='off'"):
            MoEOptimizationPolicy(packed_stream_cache_gib=1.0)
        with self.assertRaisesRegex(ValueError, "must be 0"):
            MoEOptimizationPolicy(
                packed_state_preload="off",
                packed_stream_cache_gib=1.0,
                packed_stream_backend="gds",
            )
        preload_prefetch = MoEOptimizationPolicy(packed_stream_prefetch_depth=2)
        self.assertEqual(preload_prefetch.packed_state_preload, "pinned")
        self.assertEqual(preload_prefetch.packed_stream_prefetch_depth, 2)
        with self.assertRaisesRegex(ValueError, "between 0 and 16"):
            MoEOptimizationPolicy(
                packed_state_preload="off",
                packed_stream_prefetch_depth=17,
            )


_CUDA = bool(torch is not None and torch.cuda.is_available())


@unittest.skipIf(torch is None, "torch not installed")
class CompressedWeightRotatedInt8DispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_builtin_components()

    def test_rotated_int8_is_the_only_int8_execution_backend(self) -> None:
        self.assertEqual(normalize_moe_kernel_backend("rotated_int8"), "rotated_int8")
        backend = build_moe_kernel_backend("rotated_int8", direct_routed=True)
        self.assertIsInstance(backend, RotatedInt8KernelBackend)
        for removed in ("rotated_bmm", "bitsandbytes_int8", "bnb_int8", "fused_int8", "w8a8"):
            with self.subTest(removed=removed), self.assertRaises(ValueError):
                normalize_moe_kernel_backend(removed)

    def test_grouped_backend_has_no_generic_direct_routed_implementation(self) -> None:
        self.assertEqual(normalize_moe_kernel_backend("grouped_gemm"), "grouped")
        for direct_routed in (False, True):
            with self.subTest(direct_routed=direct_routed), self.assertRaises(ValueError):
                build_moe_kernel_backend("grouped", direct_routed=direct_routed)

    def _make_experts(self, *, chunk_size: int, num_experts: int, hidden: int, inter: int, seed: int = 0):
        torch.manual_seed(seed)

        class _Base(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = num_experts
                self.w1 = torch.randn(num_experts, inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
                self.w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.05
                self.w3 = torch.randn(num_experts, inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.05

        return CompressedGroupedExperts(
            _Base(),
            quant_format="int8",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=chunk_size,
        ).cuda()

    @unittest.skipUnless(_CUDA, "rotated INT8 dispatch requires CUDA")
    def test_rotated_int8_matches_dequantized_reference_with_adapter_gradients(self) -> None:
        num_experts, hidden, inter, tokens_n = 16, 96, 192, 197
        torch.manual_seed(99)
        base_tokens = torch.randn(tokens_n, hidden, device="cuda", dtype=torch.bfloat16)
        top_indices = torch.randint(0, num_experts, (tokens_n, 2), device="cuda")
        top_scores = torch.rand(tokens_n, 2, device="cuda", dtype=torch.bfloat16) + 0.1

        for chunk_size in (2, 4, 8):
            with self.subTest(chunk_size=chunk_size):
                ref = self._make_experts(
                    chunk_size=1,
                    num_experts=num_experts,
                    hidden=hidden,
                    inter=inter,
                    seed=5,
                )
                rotated = self._make_experts(
                    chunk_size=chunk_size,
                    num_experts=num_experts,
                    hidden=hidden,
                    inter=inter,
                    seed=5,
                )
                for module in (ref, rotated):
                    torch.manual_seed(17)
                    for key in ("w1", "w2", "w3"):
                        adapter = module.attach_expert_lora(
                            tensor_name=key,
                            adapter_name="parity",
                            rank=4,
                            alpha=4.0,
                        )
                        with torch.no_grad():
                            adapter.lora_b.normal_(std=0.01)

                t_ref = base_tokens.clone().requires_grad_(True)
                t_rotated = base_tokens.clone().requires_grad_(True)
                out_ref = ref.run_direct_routed(t_ref, top_scores, top_indices)
                out_rotated = rotated.run_direct_routed_rotated_int8(
                    t_rotated, top_scores, top_indices
                )

                self.assertEqual(tuple(out_rotated.shape), (tokens_n, hidden))
                self.assertTrue(bool(torch.isfinite(out_rotated).all()))
                torch.testing.assert_close(out_rotated, out_ref, rtol=0.04, atol=0.02)

                loss_ref = out_ref.float().square().mean()
                loss_rotated = out_rotated.float().square().mean()
                torch.testing.assert_close(loss_rotated, loss_ref, rtol=0.05, atol=1e-5)
                loss_ref.backward()
                loss_rotated.backward()
                torch.testing.assert_close(t_rotated.grad, t_ref.grad, rtol=0.08, atol=2e-4)
                for key in ("w1", "w2", "w3"):
                    for parameter_name in ("lora_a", "lora_b"):
                        ref_grad = getattr(ref.expert_lora[key], parameter_name).grad
                        rotated_grad = getattr(
                            rotated.expert_lora[key], parameter_name
                        ).grad
                        self.assertIsNotNone(ref_grad)
                        self.assertIsNotNone(rotated_grad)
                        torch.testing.assert_close(
                            rotated_grad, ref_grad, rtol=0.1, atol=2e-4
                        )

    @unittest.skipUnless(_CUDA, "learned rotation dispatch requires CUDA")
    def test_learned_rotation_dispatch_matches_reference_and_input_gradient(
        self,
    ) -> None:
        torch.manual_seed(149)
        num_experts, hidden, inter, token_count = 4, 16, 12, 23
        w1 = torch.randn(num_experts, inter, hidden) / math.sqrt(hidden)
        w2 = torch.randn(num_experts, hidden, inter) / math.sqrt(inter)
        w3 = torch.randn(num_experts, inter, hidden) / math.sqrt(hidden)
        shared = learn_groupwise_expert_rotation(
            (w1, w3),
            group_size=4,
            optimization_steps=8,
            learning_rate=0.02,
            row_chunk_size=32,
            checkpoint_interval=4,
            device="cuda",
            max_workspace_gib=0.01,
        )
        down = learn_groupwise_expert_rotation(
            (w2,),
            group_size=4,
            optimization_steps=8,
            learning_rate=0.02,
            row_chunk_size=32,
            checkpoint_interval=4,
            device="cuda",
            max_workspace_gib=0.01,
        )
        experts = CompressedGroupedExperts.from_empty(
            num_experts=num_experts,
            group_sizes=4,
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
            quant_format="int8",
        )
        experts.load_dense_weight("w1", w1, rotation=shared.rotation)
        experts.load_dense_weight("w3", w3, rotation=shared.rotation)
        experts.load_dense_weight("w2", w2, rotation=down.rotation)
        experts = experts.cuda()

        tokens = torch.randn(
            token_count,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        scores = torch.rand(
            token_count,
            2,
            device="cuda",
            dtype=torch.bfloat16,
        )
        indices = torch.randint(
            0,
            num_experts,
            (token_count, 2),
            device="cuda",
        )
        reference_tokens = tokens.clone().requires_grad_(True)
        rotated_tokens = tokens.clone().requires_grad_(True)
        reference = experts.run_direct_routed(
            reference_tokens,
            scores,
            indices,
        )
        rotated = experts.run_direct_routed_rotated_int8(
            rotated_tokens,
            scores,
            indices,
        )
        torch.testing.assert_close(rotated, reference, rtol=0.04, atol=0.02)
        reference.float().square().mean().backward()
        rotated.float().square().mean().backward()
        torch.testing.assert_close(
            rotated_tokens.grad,
            reference_tokens.grad,
            rtol=0.08,
            atol=2e-4,
        )


if __name__ == "__main__":
    unittest.main()
