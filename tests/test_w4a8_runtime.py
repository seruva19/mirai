import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from mirai.core.models.compressed_weights import (
    CompressedGroupedExperts,
    export_compressed_weights_packed_state,
    load_compressed_weights_packed_state,
    load_compressed_weights_packed_state_file,
    prepare_compressed_weights_modules_from_manifest,
    quantize_compressed_weights_modules,
    save_compressed_weights_packed_state,
    w4a8_linear_reference,
)
from mirai.core.models.quantization import (
    QUANT_REGISTRY,
    expert_quantization_formats,
    is_quantized_linear,
    quantize_linear,
)


class _Experts(nn.Module):
    def __init__(self, *, experts: int = 3, hidden: int = 16, intermediate: int = 32):
        super().__init__()
        self.num_experts = experts
        self.w1 = nn.Parameter(torch.randn(experts, intermediate, hidden) * 0.08)
        self.w2 = nn.Parameter(torch.randn(experts, hidden, intermediate) * 0.08)
        self.w3 = nn.Parameter(torch.randn(experts, intermediate, hidden) * 0.08)


class _Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(16, 16)
        self.experts = _Experts()


def _make_compressed() -> CompressedGroupedExperts:
    torch.manual_seed(7)
    return CompressedGroupedExperts(
        _Experts(),
        quant_format="w4a8",
        expert_weight_access="active_dequant",
    )


def _attach_lora(module: CompressedGroupedExperts) -> None:
    for key in ("w1", "w2", "w3"):
        adapter = module.attach_expert_lora(
            tensor_name=key, adapter_name="test", rank=2, alpha=2.0
        )
        with torch.no_grad():
            adapter.lora_b.normal_(std=0.02)


def _manual_routed(module, tokens, scores, indices, *, a8: bool):
    result = tokens.new_zeros(tokens.shape)
    for expert in range(module.num_experts):
        assignments = torch.nonzero(indices == expert, as_tuple=False)
        if assignments.numel() == 0:
            continue
        rows, slots = assignments[:, 0], assignments[:, 1]
        x = tokens.index_select(0, rows)

        def linear(key, value, expert_idx=expert):
            if a8:
                output = w4a8_linear_reference(
                    value, module._w4a8_weight(key, expert_idx, device=value.device)
                )
            else:
                output = F.linear(
                    value,
                    module._dequantize_expert(
                        key, expert_idx, dtype=value.dtype, device=value.device
                    ),
                )
            if key in module.expert_lora:
                output = output + module.expert_lora[key](value, expert_idx=expert_idx)
            return output

        hidden = F.silu(linear("w1", x)) * linear("w3", x)
        expert_output = linear("w2", hidden)
        result = result.index_add(
            0, rows, expert_output * scores[rows, slots].unsqueeze(1)
        )
    return result


def _run_and_grad(module, tokens, scores, indices, runner):
    output = runner(module, tokens, scores, indices)
    loss = output.float().square().mean()
    loss.backward()
    adapter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
    }
    return output.detach(), loss.detach(), tokens.grad.detach(), scores.grad.detach(), adapter_grads


class W4A8RuntimeTests(unittest.TestCase):
    def test_model_registry_advertises_expert_only_w4a8(self):
        self.assertTrue(QUANT_REGISTRY.has("w4a8"))
        self.assertIn("w4a8", expert_quantization_formats())
        dense = nn.Linear(16, 16)
        self.assertFalse(is_quantized_linear(dense))
        with self.assertRaisesRegex(ValueError, "expert-only"):
            quantize_linear("w4a8", dense)
        self.assertFalse(is_quantized_linear(dense))

    def test_active_w4a8_stack_matches_per_expert_decode(self):
        module = CompressedGroupedExperts(
            _Experts(), quant_format="w4a8", expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
        )
        for key in ("w1", "w2", "w3"):
            stacked = module._dequant_expert_stack(
                key, (0, 2), dtype=torch.float32, device=torch.device("cpu")
            )
            expected = torch.stack(
                [
                    module._dequantize_expert(
                        key, expert, dtype=torch.float32, device=torch.device("cpu")
                    )
                    for expert in (0, 2)
                ]
            )
            torch.testing.assert_close(stacked, expected, rtol=1e-5, atol=3e-8)

    def test_reference_routing_preserves_input_route_and_lora_gradients(self):
        module = _make_compressed()
        _attach_lora(module)
        reference = copy.deepcopy(module)
        tokens = torch.randn(9, 16)
        scores = torch.softmax(torch.randn(9, 2), dim=-1).detach()
        indices = torch.tensor([[0, 1], [1, 2], [2, 0]] * 3)
        actual = _run_and_grad(
            module,
            tokens.clone().requires_grad_(),
            scores.clone().requires_grad_(),
            indices,
            lambda owner, x, s, i: owner.run_direct_routed(x, s, i),
        )
        expected = _run_and_grad(
            reference,
            tokens.clone().requires_grad_(),
            scores.clone().requires_grad_(),
            indices,
            lambda owner, x, s, i: _manual_routed(owner, x, s, i, a8=False),
        )
        for actual_value, expected_value in zip(actual[:4], expected[:4]):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        for name in actual[4]:
            torch.testing.assert_close(actual[4][name], expected[4][name], rtol=0, atol=0)
        self.assertTrue(
            all(name.startswith("expert_lora.") for name, _ in module.named_parameters())
        )
        self.assertFalse(any(buffer.dtype == torch.bfloat16 for buffer in module.buffers()))

    def test_accelerated_runner_matches_independent_a8_reference_on_cpu(self):
        module = CompressedGroupedExperts(
            _Experts(), quant_format="w4a8", expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
        )
        _attach_lora(module)
        reference = copy.deepcopy(module)
        tokens = torch.randn(7, 16)
        scores = torch.rand(7, 2)
        indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1], [0, 1]])
        expected = _run_and_grad(
            reference, tokens.clone().requires_grad_(), scores.clone().requires_grad_(),
            indices, lambda owner, x, s, i: _manual_routed(owner, x, s, i, a8=True),
        )
        with mock.patch(
            "mirai.core.models.compressed_weights.execution.experts.w4a8_linear",
            side_effect=lambda x, weight, backend: w4a8_linear_reference(x, weight),
        ):
            actual = _run_and_grad(
                module, tokens.clone().requires_grad_(), scores.clone().requires_grad_(),
                indices, lambda owner, x, s, i: owner.run_direct_routed_w4a8(x, s, i),
            )
        for actual_value, expected_value in zip(actual[:4], expected[:4]):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        for name in actual[4]:
            torch.testing.assert_close(actual[4][name], expected[4][name], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "W4A8 Triton routed parity requires CUDA")
    def test_triton_routed_matches_independent_a8_reference_and_gradients(self):
        module = CompressedGroupedExperts(
            _Experts(), quant_format="w4a8", expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
        )
        _attach_lora(module)
        module = module.cuda()
        reference = copy.deepcopy(module)
        tokens = torch.randn(11, 16, device="cuda")
        scores = torch.rand(11, 2, device="cuda")
        indices = torch.randint(0, 3, (11, 2), device="cuda")
        expected = _run_and_grad(
            reference, tokens.clone().requires_grad_(), scores.clone().requires_grad_(),
            indices, lambda owner, x, s, i: _manual_routed(owner, x, s, i, a8=True),
        )
        actual = _run_and_grad(
            module, tokens.clone().requires_grad_(), scores.clone().requires_grad_(),
            indices, lambda owner, x, s, i: owner.run_direct_routed_w4a8(x, s, i),
        )
        for actual_value, expected_value in zip(actual[:4], expected[:4]):
            torch.testing.assert_close(actual_value, expected_value, rtol=2e-3, atol=2e-3)
        for name in actual[4]:
            torch.testing.assert_close(
                actual[4][name], expected[4][name], rtol=2e-3, atol=2e-3
            )

    @unittest.skipUnless(torch.cuda.is_available(), "W4A8 chunked routing requires CUDA")
    def test_cuda_default_chunked_dispatch_matches_decoded_reference(self):
        module = CompressedGroupedExperts(
            _Experts(experts=8, hidden=16, intermediate=64),
            quant_format="w4a8",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=4,
        )
        _attach_lora(module)
        module = module.cuda()
        reference = copy.deepcopy(module)
        tokens = torch.randn(29, 16, device="cuda")
        scores = torch.rand(29, 2, device="cuda")
        indices = torch.randint(0, 8, (29, 2), device="cuda")
        expected = _run_and_grad(
            reference, tokens.clone().requires_grad_(), scores.clone().requires_grad_(),
            indices, lambda owner, x, s, i: _manual_routed(owner, x, s, i, a8=False),
        )
        actual = _run_and_grad(
            module, tokens.clone().requires_grad_(), scores.clone().requires_grad_(),
            indices, lambda owner, x, s, i: owner.run_direct_routed(x, s, i),
        )
        for actual_value, expected_value in zip(actual[:4], expected[:4]):
            torch.testing.assert_close(actual_value, expected_value, rtol=2e-3, atol=2e-3)
        for name in actual[4]:
            torch.testing.assert_close(
                actual[4][name], expected[4][name], rtol=2e-3, atol=2e-3
            )

    def test_accelerated_runner_coalesces_logical_alias_routes(self):
        module = CompressedGroupedExperts(
            _Experts(), quant_format="w4a8", expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
        )
        module.configure_logical_expert_aliases([0, 0, 1, 2])
        tokens = torch.randn(4, 16)
        scores = torch.tensor([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5]])
        logical = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
        mapped_scores, mapped_indices = module._coalesce_logical_routes(scores, logical)
        with mock.patch(
            "mirai.core.models.compressed_weights.execution.experts.w4a8_linear",
            side_effect=lambda x, weight, backend: w4a8_linear_reference(x, weight),
        ):
            actual = module.run_direct_routed_w4a8(tokens, scores, logical)
            reference = _manual_routed(
                module, tokens, mapped_scores, mapped_indices, a8=True
            )
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_runtime_paths_do_not_save_expanded_frozen_weights(self):
        module = _make_compressed()
        tokens = torch.randn(4, 16, requires_grad=True)
        scores = torch.rand(4, 1, requires_grad=True)
        indices = torch.tensor([[0], [1], [2], [0]])
        saved_shapes = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda value: saved_shapes.append(tuple(value.shape)) or value,
            lambda value: value,
        ):
            module.run_direct_routed(tokens, scores, indices).sum().backward()
        dense_shapes = {tuple(module.expert_weight_shape(key)[1:]) for key in ("w1", "w2", "w3")}
        self.assertTrue(dense_shapes.isdisjoint(saved_shapes))

    def test_packed_round_trip_is_strict_and_preserves_reference(self):
        source = _make_compressed()
        tensors, manifest = export_compressed_weights_packed_state(source)
        self.assertEqual(manifest["schema_version"], 6)
        self.assertEqual(manifest["modules"][""]["quant_format"], "w4a8")
        target = CompressedGroupedExperts.from_empty(
            num_experts=3,
            quant_format="w4a8",
            expert_weight_access="active_dequant",
        )
        load_compressed_weights_packed_state(target, tensors, manifest)
        for key in ("w1", "w2", "w3"):
            torch.testing.assert_close(getattr(target, key), getattr(source, key), rtol=0, atol=0)
        broken = dict(tensors)
        broken.pop(next(name for name in broken if name.endswith("w1_w4a8_packed")))
        with self.assertRaisesRegex(KeyError, "missing W4A8 tensors"):
            load_compressed_weights_packed_state(
                CompressedGroupedExperts.from_empty(
                    num_experts=3,
                    quant_format="w4a8",
                    expert_weight_access="active_dequant",
                ),
                broken,
                manifest,
            )
        wrong_manifest = copy.deepcopy(manifest)
        wrong_manifest["modules"][""]["w4a8_meta"]["w1"]["shape"][0] = 4
        with self.assertRaisesRegex(ValueError, "shape"):
            load_compressed_weights_packed_state(
                CompressedGroupedExperts.from_empty(
                    num_experts=3,
                    quant_format="w4a8",
                    expert_weight_access="active_dequant",
                ),
                tensors,
                wrong_manifest,
            )

    def test_manifest_prepare_and_expert_only_quantization(self):
        source = nn.Module()
        source.experts = _make_compressed()
        tensors, manifest = export_compressed_weights_packed_state(source)
        host = nn.Module()
        host.experts = _Experts()
        report = prepare_compressed_weights_modules_from_manifest(host, manifest)
        self.assertEqual(report.grouped_expert_modules, 1)
        load_compressed_weights_packed_state(host, tensors, manifest)
        self.assertIsInstance(host.experts, CompressedGroupedExperts)
        with self.assertRaisesRegex(ValueError, "expert-only"):
            quantize_compressed_weights_modules(_Host(), quant_format="w4a8")
        host = _Host()
        report = quantize_compressed_weights_modules(
            host,
            quant_format="w4a8",
            replace_linear=False,
            expert_weight_access="active_dequant",
        )
        self.assertEqual(report.linear_modules, 0)
        self.assertEqual(report.grouped_expert_modules, 1)
        self.assertIsInstance(host.dense, nn.Linear)

    def test_safetensors_ram_and_pinned_round_trip(self):
        source = nn.Module()
        source.experts = _make_compressed()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_compressed_weights_packed_state(Path(tmp) / "w4a8.safetensors", source)
            for preload in ("ram", "pinned"):
                target = nn.Module()
                target.experts = _Experts()
                manifest = __import__(
                    "mirai.core.models.compressed_weights", fromlist=["read_compressed_weights_packed_state_manifest"]
                ).read_compressed_weights_packed_state_manifest(path)
                prepare_compressed_weights_modules_from_manifest(target, manifest)
                load_compressed_weights_packed_state_file(
                    path, target, packed_state_preload=preload
                )
                for key in ("w1", "w2", "w3"):
                    self.assertEqual(getattr(target.experts, f"{key}_w4a8_packed").dtype, torch.uint8)
                    self.assertEqual(getattr(target.experts, f"{key}_w4a8_group_scales").dtype, torch.uint8)
                    self.assertEqual(getattr(target.experts, f"{key}_w4a8_row_scales").dtype, torch.float32)

    def test_accelerated_path_fails_explicitly_without_cuda(self):
        module = CompressedGroupedExperts(
            _Experts(),
            quant_format="w4a8",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
        )
        tokens = torch.randn(2, 16)
        scores = torch.ones(2, 1)
        indices = torch.zeros(2, 1, dtype=torch.long)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            module.run_direct_routed_w4a8(tokens, scores, indices)

    def test_provider_layout_and_access_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "active_dequant or chunked_dequant"):
            CompressedGroupedExperts(
                _Experts(), quant_format="w4a8", expert_weight_access="full_dequant"
            )


if __name__ == "__main__":
    unittest.main()
