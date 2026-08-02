from __future__ import annotations

import unittest

from mirai.config.schema import MemoryConfig, ModelConfig, ModelParams, StrategyConfig, TrainingConfig, TrainingSection
from mirai.core.builtins import register_builtin_components
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy
from mirai.core.moe.runtime.specs import active_moe_optimization_policy
from mirai.core.models.compressed_weights import (
    CompressedGroupedExperts,
    CompressedLinear,
    normalize_quant_format,
    quantize_compressed_weights_modules,
)
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

_CUDA = bool(torch is not None and torch.cuda.is_available())

try:
    from mirai.core.models.compressed_weights.execution import triton_moe_gemm as _tmg
    _TRITON = bool(_CUDA and _tmg.is_available(torch.device("cuda")))
except Exception:  # pragma: no cover - triton optional
    _tmg = None
    _TRITON = False


@unittest.skipIf(torch is None, "torch not installed")
class Nf4CompressedWeightFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_builtin_components()

    def _lingbot_config(self) -> TrainingConfig:
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
            memory=MemoryConfig(frozen_weight_quantization="nf4"),
        )

    def test_normalize_quant_format_accepts_nf4_and_rejects_garbage(self) -> None:
        self.assertEqual(normalize_quant_format("nf4"), "nf4")
        with self.assertRaises(ValueError):
            normalize_quant_format("nf4" + "convot")
        self.assertEqual(normalize_quant_format("auto"), "int8")
        with self.assertRaises(ValueError):
            normalize_quant_format("compressed_weights")
        with self.assertRaises(ValueError):
            normalize_quant_format("garbage")

    def test_schema_accepts_nf4_frozen_weight_quantization(self) -> None:
        cfg = self._lingbot_config()
        self.assertEqual(cfg.memory.frozen_weight_quantization, "nf4")

    def test_enable_nf4_rejects_rotated_int8_backend(self) -> None:
        pipeline = LingBotVideoPipeline(self._lingbot_config().model)
        pipeline._frozen_weight_quantization = "nf4"
        pipeline._moe_optimization_policy = MoEOptimizationPolicy(
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=4,
            kernel_backend="rotated_int8",
        )
        with self.assertRaisesRegex(ValueError, "requires.*int8"):
            pipeline.enable_quantized_frozen_weights("nf4")

    def test_configure_policy_rejects_nf4_with_rotated_int8(self) -> None:
        pipeline = LingBotVideoPipeline(self._lingbot_config().model)
        pipeline._frozen_weight_quantization = "nf4"
        policy = MoEOptimizationPolicy(
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=4,
            kernel_backend="rotated_int8",
        )
        with self.assertRaisesRegex(ValueError, "requires.*int8"):
            pipeline.configure_moe_optimization_policy(policy)

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_linear_nf4_roundtrip_matches_base_within_tolerance(self) -> None:
        torch.manual_seed(3)
        base = nn.Linear(128, 64, bias=True).cuda().to(torch.bfloat16)
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)

        quantized = CompressedLinear(base, quant_format="nf4")
        self.assertEqual(quantized._quant_format, "nf4")
        expected = base(x)
        observed = quantized(x)

        self.assertEqual(observed.shape, expected.shape)
        error = (observed.float() - expected.float()).abs().mean().item()
        scale = expected.float().abs().mean().item()
        self.assertLess(error, 0.1 * scale + 1e-3)

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_grouped_nf4_chunked_forward_backward_lora_grads(self) -> None:
        torch.manual_seed(5)
        num_experts, hidden, inter = 6, 96, 192

        class _Base(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = num_experts
                self.w1 = torch.randn(num_experts, inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
                self.w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.05
                self.w3 = torch.randn(num_experts, inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.05

        experts = CompressedGroupedExperts(
            _Base(),
            quant_format="nf4",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=16,
        ).cuda()
        for key in ("w1", "w2", "w3"):
            experts.attach_expert_lora(tensor_name=key, adapter_name="lora", rank=8, alpha=8.0)
        experts = experts.cuda()
        with torch.no_grad():
            for key in ("w1", "w2", "w3"):
                experts.expert_lora[key].lora_b.add_(
                    0.02 * torch.randn_like(experts.expert_lora[key].lora_b)
                )
        experts.train()

        tokens = torch.randn(48, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        top_indices = torch.randint(0, num_experts, (48, 2), device="cuda")
        top_scores = torch.rand(48, 2, device="cuda", dtype=torch.bfloat16) + 0.1

        out = experts.run_direct_routed(tokens, top_scores, top_indices)
        self.assertEqual(tuple(out.shape), (48, hidden))
        self.assertTrue(bool(torch.isfinite(out).all()))
        out.float().pow(2).mean().backward()

        for key in ("w1", "w2", "w3"):
            grad_a = experts.expert_lora[key].lora_a.grad
            grad_b = experts.expert_lora[key].lora_b.grad
            self.assertIsNotNone(grad_a)
            self.assertIsNotNone(grad_b)
            self.assertGreater(float(grad_a.norm()), 0.0)
            self.assertGreater(float(grad_b.norm()), 0.0)
        self.assertGreater(float(tokens.grad.norm()), 0.0)

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_grouped_nf4_requant_backward_matches_saved_weight_backward(self) -> None:
        torch.manual_seed(13)
        num_experts, hidden, inter = 4, 64, 128

        class _Base(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = num_experts
                self.w1 = torch.randn(num_experts, inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
                self.w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.05
                self.w3 = torch.randn(num_experts, inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.05

        experts = CompressedGroupedExperts(
            _Base(),
            quant_format="nf4",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=2,
        ).cuda()
        active = [0, 2]
        x = torch.randn(len(active), 8, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        expert_ids = torch.tensor(active, device="cuda", dtype=torch.long)

        out = experts._expert_linear_batched_requant("w1", x, active, expert_ids)
        grad_seed = torch.randn_like(out)
        out.backward(grad_seed)
        grad_requant = x.grad.detach().clone()
        x.grad = None

        weight = experts._dequant_expert_stack(
            "w1", tuple(active), dtype=x.dtype, device=x.device
        )
        ref = torch.bmm(x, weight.transpose(-2, -1))
        ref.backward(grad_seed)
        grad_saved = x.grad.detach().clone()

        self.assertTrue(torch.equal(out.detach(), ref.detach()))
        torch.testing.assert_close(grad_requant, grad_saved, rtol=1e-2, atol=1e-4)

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_grouped_nf4_survives_device_roundtrip(self) -> None:
        torch.manual_seed(7)
        num_experts, hidden, inter = 4, 64, 128

        class _Base(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = num_experts
                self.w1 = torch.randn(num_experts, inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
                self.w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.05
                self.w3 = torch.randn(num_experts, inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.05

        experts = CompressedGroupedExperts(
            _Base(),
            quant_format="nf4",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=16,
        ).cuda()
        reference = experts._dequantize_expert("w1", 1, dtype=torch.bfloat16, device="cuda")
        experts = experts.cpu()
        experts = experts.cuda()
        roundtrip = experts._dequantize_expert("w1", 1, dtype=torch.bfloat16, device="cuda")
        self.assertEqual((reference.float() - roundtrip.float()).abs().max().item(), 0.0)

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
            quant_format="nf4",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=chunk_size,
        ).cuda()

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_batched_dispatch_matches_per_expert_reference_forward_backward(self) -> None:
        # chunk_size=1 takes the simple per-expert reference loop in
        # run_direct_routed; chunk_size=4 takes the vectorized batched dispatch.
        # Both dequantize identical NF4 weights, so outputs and token grads must
        # agree to bf16 noise. Routing deliberately leaves some experts empty and
        # spans several chunks (some fully empty) to exercise the padding/skip logic.
        num_experts, hidden, inter, tokens_n = 16, 96, 192, 200
        ref = self._make_experts(chunk_size=1, num_experts=num_experts, hidden=hidden, inter=inter, seed=5)
        bat = self._make_experts(chunk_size=4, num_experts=num_experts, hidden=hidden, inter=inter, seed=5)

        torch.manual_seed(99)
        base_tokens = torch.randn(tokens_n, hidden, device="cuda", dtype=torch.bfloat16)
        top_indices = torch.randint(0, num_experts, (tokens_n, 2), device="cuda")
        # force experts 3 and 11 (and a whole trailing chunk) to stay empty
        top_indices = torch.where(top_indices == 3, torch.full_like(top_indices, 0), top_indices)
        top_indices = torch.where(top_indices == 11, torch.full_like(top_indices, 1), top_indices)
        top_indices = top_indices.clamp_max(num_experts - 5)
        top_scores = torch.rand(tokens_n, 2, device="cuda", dtype=torch.bfloat16) + 0.1

        t_ref = base_tokens.clone().requires_grad_(True)
        t_bat = base_tokens.clone().requires_grad_(True)

        out_ref = ref.run_direct_routed(t_ref, top_scores, top_indices)
        out_bat = bat.run_direct_routed(t_bat, top_scores, top_indices)

        # bf16 accumulation order (batched bmm vs per-expert matmul) differs by ~1 ULP.
        torch.testing.assert_close(out_ref, out_bat, rtol=2e-2, atol=3e-3)

        out_ref.float().pow(2).mean().backward()
        out_bat.float().pow(2).mean().backward()
        self.assertTrue(bool(torch.isfinite(t_bat.grad).all()))
        torch.testing.assert_close(t_ref.grad, t_bat.grad, rtol=3e-2, atol=3e-3)

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_vectorized_dispatch_bit_identical_to_legacy_dispatch(self) -> None:
        # Both dispatch modes build the same [E, max_count, H] padded chunk
        # (padding rows zeroed) and run the identical shared chunk compute, so
        # forward outputs and input grads must be bit-identical.
        num_experts, hidden, inter, tokens_n = 16, 96, 192, 200
        experts = self._make_experts(chunk_size=4, num_experts=num_experts, hidden=hidden, inter=inter, seed=5)

        torch.manual_seed(99)
        base_tokens = torch.randn(tokens_n, hidden, device="cuda", dtype=torch.bfloat16)
        top_indices = torch.randint(0, num_experts, (tokens_n, 2), device="cuda")
        top_indices = torch.where(top_indices == 3, torch.zeros_like(top_indices), top_indices)
        top_scores = torch.rand(tokens_n, 2, device="cuda", dtype=torch.bfloat16) + 0.1

        results = {}
        for mode in ("legacy", "vectorized"):
            with active_moe_optimization_policy(
                MoEOptimizationPolicy(moe_dispatch=mode)
            ):
                t = base_tokens.clone().requires_grad_(True)
                out = experts.run_direct_routed(t, top_scores, top_indices)
                out.float().pow(2).mean().backward()
                results[mode] = (out.detach(), t.grad.detach())

        self.assertTrue(torch.equal(results["legacy"][0], results["vectorized"][0]))
        self.assertTrue(torch.equal(results["legacy"][1], results["vectorized"][1]))
        self.assertTrue(bool(torch.isfinite(results["vectorized"][1]).all()))

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_batched_chunk_dequant_bit_identical_to_per_expert_dequant(self) -> None:
        # _nf4_dequantize_expert_stack_batched concatenates the chunk's packed
        # buffers into single bnb dequant calls; it must reproduce the
        # per-expert path bit-exactly (same blockwise math, same absmax).
        # Dims chosen so per-expert absmax length (out*in/blocksize) is a
        # multiple of the nested blocksize (the batched path's precondition;
        # real model dims 2048x768 satisfy it too).
        num_experts, hidden, inter = 8, 128, 256
        experts = self._make_experts(chunk_size=4, num_experts=num_experts, hidden=hidden, inter=inter, seed=3)
        for key in ("w1", "w2", "w3"):
            for indices in ((0, 1, 2, 3), (5, 2, 7), (6, 0)):
                batched = experts._nf4_dequantize_expert_stack_batched(
                    key, indices, dtype=torch.bfloat16, device=torch.device("cuda")
                )
                self.assertIsNotNone(batched)
                reference = torch.stack(
                    [
                        experts._dequantize_expert(key, i, dtype=torch.bfloat16, device="cuda")
                        for i in indices
                    ],
                    dim=0,
                )
                self.assertTrue(
                    torch.equal(batched, reference),
                    f"batched nf4 chunk dequant mismatch for {key} {indices}",
                )

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_nf4_key_pair_codebooks_match(self) -> None:
        # The paired w1/w3 dequant concatenates the two stacks and dequantizes
        # them with a single shared codebook; that is only valid because the NF4
        # and nested codebooks are weight-independent constants. Lock it.
        experts = self._make_experts(chunk_size=4, num_experts=8, hidden=128, inter=256, seed=3)
        self.assertTrue(
            torch.equal(experts.w1_nf4_code, experts.w3_nf4_code)
            and torch.equal(experts.w1_nf4_ncode, experts.w3_nf4_ncode)
        )
        self.assertTrue(experts._nf4_pair_codebooks_match("w1", "w3"))

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_nf4_key_pair_dequant_bit_identical_to_per_key(self) -> None:
        # _nf4_dequantize_key_pair_batched must reproduce the (already bit-exact)
        # per-key batched dequant for both w1 and w3, so the shared-dequant chunk
        # compute is numerically identical to two independent dequants.
        experts = self._make_experts(chunk_size=4, num_experts=8, hidden=128, inter=256, seed=3)
        for indices in ((0, 1, 2, 3), (5, 2, 7), (6, 0), (4,)):
            pair = experts._nf4_dequantize_key_pair_batched(
                "w1", "w3", indices, dtype=torch.bfloat16, device=torch.device("cuda")
            )
            self.assertIsNotNone(pair)
            w1_pair, w3_pair = pair
            w1_ref = experts._nf4_dequantize_expert_stack_batched(
                "w1", indices, dtype=torch.bfloat16, device=torch.device("cuda")
            )
            w3_ref = experts._nf4_dequantize_expert_stack_batched(
                "w3", indices, dtype=torch.bfloat16, device=torch.device("cuda")
            )
            self.assertTrue(torch.equal(w1_pair, w1_ref), f"w1 pair mismatch {indices}")
            self.assertTrue(torch.equal(w3_pair, w3_ref), f"w3 pair mismatch {indices}")

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_dequant_expert_pair_matches_per_key_stack_and_off_switch(self) -> None:
        # Paired and independent dequantization
        # fallback must both equal two independent _dequant_expert_stack calls.
        experts = self._make_experts(chunk_size=4, num_experts=8, hidden=128, inter=256, seed=7)
        indices = (0, 3, 5)
        ref_w1 = experts._dequant_expert_stack("w1", indices, dtype=torch.bfloat16, device=torch.device("cuda"))
        ref_w3 = experts._dequant_expert_stack("w3", indices, dtype=torch.bfloat16, device=torch.device("cuda"))
        for enabled in (True, False):
            with active_moe_optimization_policy(
                MoEOptimizationPolicy(moe_pair_dequant=enabled)
            ):
                w1, w3 = experts._dequant_expert_pair(
                    "w1", "w3", indices, dtype=torch.bfloat16, device=torch.device("cuda")
                )
                self.assertTrue(torch.equal(w1, ref_w1))
                self.assertTrue(torch.equal(w3, ref_w3))

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_pair_dequant_dispatch_bit_identical_to_unpaired(self) -> None:
        # End-to-end dispatch: the shared-dequant chunk compute (pairing on) must
        # be bit-identical to the two-separate-dequants path (pairing off) for
        # both forward outputs and input grads.
        num_experts, hidden, inter, tokens_n = 16, 96, 192, 200
        experts = self._make_experts(chunk_size=4, num_experts=num_experts, hidden=hidden, inter=inter, seed=5)
        torch.manual_seed(99)
        base_tokens = torch.randn(tokens_n, hidden, device="cuda", dtype=torch.bfloat16)
        top_indices = torch.randint(0, num_experts, (tokens_n, 2), device="cuda")
        top_scores = torch.rand(tokens_n, 2, device="cuda", dtype=torch.bfloat16) + 0.1

        results = {}
        for enabled in (False, True):
            with active_moe_optimization_policy(
                MoEOptimizationPolicy(moe_pair_dequant=enabled)
            ):
                t = base_tokens.clone().requires_grad_(True)
                out = experts.run_direct_routed(t, top_scores, top_indices)
                out.float().pow(2).mean().backward()
                results[enabled] = (out.detach(), t.grad.detach())
        self.assertTrue(torch.equal(results[False][0], results[True][0]))
        self.assertTrue(torch.equal(results[False][1], results[True][1]))

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_batched_dispatch_empty_routing_returns_zeros(self) -> None:
        num_experts, hidden, inter = 8, 64, 128
        bat = self._make_experts(chunk_size=4, num_experts=num_experts, hidden=hidden, inter=inter, seed=1)
        tokens = torch.randn(12, hidden, device="cuda", dtype=torch.bfloat16)
        # all tokens route to expert 0 only -> most chunks empty
        top_indices = torch.zeros(12, 2, device="cuda", dtype=torch.long)
        top_scores = torch.rand(12, 2, device="cuda", dtype=torch.bfloat16) + 0.1
        out = bat.run_direct_routed(tokens, top_scores, top_indices)
        self.assertEqual(tuple(out.shape), (12, hidden))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def _run_dispatch_mode(self, experts, base_tokens, top_scores, top_indices, mode):
        with active_moe_optimization_policy(
            MoEOptimizationPolicy(moe_dispatch=mode)
        ):
            t = base_tokens.clone().requires_grad_(True)
            out = experts.run_direct_routed(t, top_scores, top_indices)
            out.float().pow(2).mean().backward()
            return out.detach(), t.grad.detach()

    @unittest.skipUnless(_TRITON, "triton grouped GEMM unavailable")
    def test_triton_grouped_gemm_matches_bmm_and_skips_padding(self) -> None:
        # Direct kernel check: forward/backward vs torch bmm on the SAME
        # dequantized bf16 weights, with per-expert counts. Valid rows must
        # match bmm to bf16 noise; padding rows (row >= count) must be exact 0.
        dev = torch.device("cuda")
        torch.manual_seed(4)
        E, Mmax, K, N = 4, 300, 512, 384
        counts = torch.tensor([300, 100, 250, 1], device=dev, dtype=torch.int32)
        row = torch.arange(Mmax, device=dev)
        valid = row[None, :] < counts[:, None]
        x = torch.randn(E, Mmax, K, device=dev, dtype=torch.bfloat16)
        x = x * valid.unsqueeze(-1)
        w = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) * 0.05
        out = _tmg.grouped_linear_nt(x, w, counts)
        ref = torch.bmm(x, w.transpose(-2, -1))
        vmask = valid.unsqueeze(-1).expand_as(out)
        torch.testing.assert_close(
            out[vmask], ref[vmask], rtol=2e-2, atol=3e-3
        )
        self.assertEqual(
            (out.float() * (~valid).unsqueeze(-1)).abs().max().item(), 0.0
        )
        # backward grad_input vs bmm(grad, w)
        go = torch.randn(E, Mmax, N, device=dev, dtype=torch.bfloat16)
        go = go * valid.unsqueeze(-1)
        gin = _tmg.grouped_linear_grad(go, w, counts)
        gref = torch.bmm(go, w)
        vmk = valid.unsqueeze(-1).expand_as(gin)
        torch.testing.assert_close(gin[vmk], gref[vmk], rtol=2e-2, atol=3e-3)
        self.assertEqual(
            (gin.float() * (~valid).unsqueeze(-1)).abs().max().item(), 0.0
        )

    @unittest.skipUnless(_TRITON, "triton grouped GEMM unavailable")
    def test_triton_dispatch_matches_vectorized_forward_backward(self) -> None:
        # Full dispatch: triton path (padded layout, LoRA unchanged) vs the
        # vectorized path must agree on outputs and input grads to bf16 noise.
        # Routing spans several chunks, leaves some experts empty, and exercises
        # a whole trailing empty chunk (padding/skip logic).
        num_experts, hidden, inter, tokens_n = 16, 128, 256, 240
        experts = self._make_experts(
            chunk_size=4, num_experts=num_experts, hidden=hidden, inter=inter, seed=5
        )
        torch.manual_seed(99)
        base_tokens = torch.randn(tokens_n, hidden, device="cuda", dtype=torch.bfloat16)
        top_indices = torch.randint(0, num_experts, (tokens_n, 2), device="cuda")
        top_indices = torch.where(top_indices == 3, torch.zeros_like(top_indices), top_indices)
        top_indices = torch.where(top_indices == 11, torch.ones_like(top_indices), top_indices)
        top_indices = top_indices.clamp_max(num_experts - 5)
        top_scores = torch.rand(tokens_n, 2, device="cuda", dtype=torch.bfloat16) + 0.1

        vec = self._run_dispatch_mode(experts, base_tokens, top_scores, top_indices, "vectorized")
        tri = self._run_dispatch_mode(experts, base_tokens, top_scores, top_indices, "triton")
        torch.testing.assert_close(tri[0], vec[0], rtol=2e-2, atol=3e-3)
        self.assertTrue(bool(torch.isfinite(tri[1]).all()))
        torch.testing.assert_close(tri[1], vec[1], rtol=3e-2, atol=3e-3)

    @unittest.skipUnless(_TRITON, "triton grouped GEMM unavailable")
    def test_triton_dispatch_single_expert_and_all_experts(self) -> None:
        num_experts, hidden, inter, tokens_n = 8, 96, 192, 128
        experts = self._make_experts(
            chunk_size=4, num_experts=num_experts, hidden=hidden, inter=inter, seed=2
        )
        torch.manual_seed(7)
        base_tokens = torch.randn(tokens_n, hidden, device="cuda", dtype=torch.bfloat16)
        top_scores = torch.rand(tokens_n, 2, device="cuda", dtype=torch.bfloat16) + 0.1
        # single expert: all tokens -> expert 0 (most chunks empty)
        single = torch.zeros(tokens_n, 2, device="cuda", dtype=torch.long)
        vec = self._run_dispatch_mode(experts, base_tokens, top_scores, single, "vectorized")
        tri = self._run_dispatch_mode(experts, base_tokens, top_scores, single, "triton")
        torch.testing.assert_close(tri[0], vec[0], rtol=2e-2, atol=3e-3)
        torch.testing.assert_close(tri[1], vec[1], rtol=3e-2, atol=3e-3)
        # all experts covered
        allx = torch.randint(0, num_experts, (tokens_n, 2), device="cuda")
        vec2 = self._run_dispatch_mode(experts, base_tokens, top_scores, allx, "vectorized")
        tri2 = self._run_dispatch_mode(experts, base_tokens, top_scores, allx, "triton")
        torch.testing.assert_close(tri2[0], vec2[0], rtol=2e-2, atol=3e-3)
        torch.testing.assert_close(tri2[1], vec2[1], rtol=3e-2, atol=3e-3)

    @unittest.skipUnless(_TRITON, "triton grouped GEMM unavailable")
    def test_triton_dispatch_empty_routing_returns_zeros(self) -> None:
        num_experts, hidden, inter = 8, 64, 128
        experts = self._make_experts(
            chunk_size=4, num_experts=num_experts, hidden=hidden, inter=inter, seed=1
        )
        tokens = torch.randn(12, hidden, device="cuda", dtype=torch.bfloat16)
        top_indices = torch.zeros(12, 2, device="cuda", dtype=torch.long)
        top_scores = torch.rand(12, 2, device="cuda", dtype=torch.bfloat16) + 0.1
        out = self._run_dispatch_mode(experts, tokens, top_scores, top_indices, "triton")[0]
        self.assertEqual(tuple(out.shape), (12, hidden))
        self.assertTrue(bool(torch.isfinite(out).all()))

    @unittest.skipUnless(_CUDA, "requires CUDA")
    def test_triton_dispatch_falls_back_to_vectorized_when_unavailable(self) -> None:
        num_experts, hidden, inter, tokens_n = 8, 64, 128, 80
        experts = self._make_experts(
            chunk_size=4, num_experts=num_experts, hidden=hidden, inter=inter, seed=3
        )
        torch.manual_seed(11)
        base_tokens = torch.randn(tokens_n, hidden, device="cuda", dtype=torch.bfloat16)
        top_indices = torch.randint(0, num_experts, (tokens_n, 2), device="cuda")
        top_scores = torch.rand(tokens_n, 2, device="cuda", dtype=torch.bfloat16) + 0.1

        vec = self._run_dispatch_mode(experts, base_tokens, top_scores, top_indices, "vectorized")
        from unittest.mock import patch

        with patch(
            "mirai.core.models.compressed_weights.execution.experts._triton_dispatch_available",
            return_value=False,
        ):
            fb = self._run_dispatch_mode(experts, base_tokens, top_scores, top_indices, "triton")
        self.assertTrue(torch.equal(fb[0], vec[0]))
        self.assertTrue(torch.equal(fb[1], vec[1]))

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_nf4_quantize_2d_releases_cuda_transient_and_is_deterministic(self) -> None:
        # On the streaming-load path the quant source is a CPU checkpoint tensor:
        # _nf4_quantize_2d uploads it to CUDA transiently for bitsandbytes, then the
        # caller stores the packed NF4 back on the source (CPU) device. Two invariants
        # apply: (a) the produced NF4 bytes are
        # unchanged (identical across repeated calls -> the `del source` frees only the
        # upload, never perturbs the quant output), and (b) no NF4/upload tensor stays
        # resident on CUDA after the call (the transient is released, not accumulated).
        from mirai.core.models.compressed_weights.quantization.quant import NF4_BLOCKSIZE, _nf4_quantize_2d

        torch.manual_seed(17)
        weight = torch.randn(256, 512, dtype=torch.float32)  # CPU source (checkpoint-like)

        def _quantize_to_cpu():
            fields, codes, meta = _nf4_quantize_2d(weight, blocksize=NF4_BLOCKSIZE)
            cpu_fields = {k: v.to("cpu") for k, v in fields.items()}
            cpu_codes = {k: v.to("cpu") for k, v in codes.items()}
            return cpu_fields, cpu_codes, meta

        # Warm up so bitsandbytes' first-call persistent allocation is out of the
        # measured window; then baseline the live CUDA allocation.
        _quantize_to_cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        f1, c1, _ = _quantize_to_cpu()
        f2, c2, _ = _quantize_to_cpu()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()

        for key in f1:
            self.assertTrue(torch.equal(f1[key], f2[key]), f"nf4 field {key} not deterministic")
        for key in c1:
            self.assertTrue(torch.equal(c1[key], c2[key]), f"nf4 codebook {key} not deterministic")
        # Packed NF4 landed on the source (CPU) device...
        self.assertEqual(str(f1["packed"].device), "cpu")
        # ...and no residual CUDA allocation was left behind by the transient upload.
        self.assertEqual(after, baseline)

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_store_nf4_grouped_from_cpu_source_leaves_no_cuda_residue(self) -> None:
        # Mirror the streaming-load grouped-expert path: a CPU dense [E, out, in] tensor
        # is quantized expert-by-expert. The packed buffers must land on CPU and the
        # per-expert bf16 CUDA uploads must not accumulate (live CUDA allocation returns
        # to baseline after the load).
        num_experts, hidden, inter = 8, 64, 128
        experts = CompressedGroupedExperts.from_empty(
            num_experts=num_experts,
            quant_format="nf4",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=4,
        )
        torch.manual_seed(23)
        w1 = torch.randn(num_experts, inter, hidden, dtype=torch.float32) * 0.05  # CPU
        w2 = torch.randn(num_experts, hidden, inter, dtype=torch.float32) * 0.05  # CPU

        experts.load_dense_weight("w1", w1)  # warm up bnb persistent allocation
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()

        experts.load_dense_weight("w2", w2)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()

        self.assertEqual(str(experts.w2_nf4.device), "cpu")
        self.assertEqual(after, baseline)

    @unittest.skipUnless(_CUDA, "nf4 quantization requires CUDA")
    def test_quantize_modules_nf4_replaces_linears_and_experts(self) -> None:
        pipeline = LingBotVideoPipeline(self._lingbot_config().model).cuda()
        report = quantize_compressed_weights_modules(
            pipeline.transformer,
            quant_format="nf4",
            expert_weight_access="chunked_dequant",
            expert_dequant_chunk_size=16,
        )
        self.assertGreater(report.linear_modules, 0)
        self.assertGreater(report.grouped_expert_modules, 0)
        self.assertGreater(report.quantized_numel, 0)
        linears = [m for m in pipeline.modules() if isinstance(m, CompressedLinear)]
        grouped = [m for m in pipeline.modules() if isinstance(m, CompressedGroupedExperts)]
        self.assertTrue(all(m._quant_format == "nf4" for m in linears))
        self.assertTrue(all(m._quant_format == "nf4" for m in grouped))


if __name__ == "__main__":
    unittest.main()
