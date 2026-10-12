import torch
import pytest
import os

from mirai.vendors.agnes_video.dit import AgnesDiT, AgnesDiTConfig, Attention, BaseLinear, ExpertMLP, ModalityDispatcher, MultiModalityRMSNorm, NativeMoELinear, SparseMoEMLP, swiglu7


def small_config():
    return AgnesDiTConfig(num_layers=2, hidden_size=32, head_dim=8, num_query_groups=2,
        video_in_channels=4, audio_in_channels=2, text_in_channels=6,
        mm_layers=(0,), sparse_moe_layers=(1,), gelu7_layers=(), params_dtype=torch.bfloat16)


def test_interleaved_swiglu_matches_scalar_reference():
    x = torch.tensor([[-9., -12., 12., 20., 1., 2.]])
    expected = torch.tensor([[-9 * torch.sigmoid(torch.tensor(-9 * 1.702)) * -6,
        7 * torch.sigmoid(torch.tensor(7 * 1.702)) * 8,
        torch.sigmoid(torch.tensor(1.702)) * 3]])
    torch.testing.assert_close(swiglu7(x), expected)


def test_top_one_experts_and_router_aux_gradients():
    torch.manual_seed(7)
    moe = SparseMoEMLP(small_config())
    x = torch.randn(7, 32, requires_grad=True)
    out = moe(x)
    normalized = moe.pre_norm(x).to(torch.bfloat16)
    chosen = moe.router(normalized.float()).argmax(-1)
    expected = torch.stack([moe.experts[int(expert)](normalized[index:index+1])[0] for index, expert in enumerate(chosen)])
    torch.testing.assert_close(out, expected)
    out.float().square().sum().backward(retain_graph=True)
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert moe.router.weight.grad.abs().max() < 1e-6
    assert any(expert.down_proj.weight.grad is not None for expert in moe.experts)
    moe.zero_grad()
    moe._last_aux_loss.backward()
    assert moe.router.weight.grad.abs().sum() > 0


def test_native_linear_reference_gradient_and_frozen_saved_tensors():
    torch.manual_seed(11)
    linear = BaseLinear(5, 3)
    linear.weight.requires_grad_(False)
    x = torch.randn(4, 5, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        out = linear(x)
    grad = torch.randn_like(out)
    out.backward(grad)
    torch.testing.assert_close(out, torch.nn.functional.linear(x.to(torch.bfloat16), linear.weight).float())
    torch.testing.assert_close(x.grad, grad.float() @ linear.weight.float())
    assert all(tensor.data_ptr() != linear.weight.data_ptr() for tensor in saved)


def test_expert_projection_module_hooks_are_called():
    expert = ExpertMLP(small_config())
    calls = []
    handles = [module.register_forward_hook(lambda *args: calls.append(True)) for module in (expert.up_gate_proj, expert.down_proj)]
    expert(torch.randn(2, 32))
    for handle in handles:
        handle.remove()
    assert len(calls) == 2


def test_embedding_and_head_module_hooks_are_called():
    model = AgnesDiT(small_config())
    calls = []
    modules = (model.adapter.video_embedder, model.adapter.audio_embedder,
        model.adapter.text_embedder, model.final_linear_video, model.final_linear_audio)
    handles = [module.register_forward_hook(lambda *args: calls.append(True)) for module in modules]
    model(torch.randn(3, 6), torch.tensor([[0., 0., 0., 1., 1., 1., 1., 1., 1.]]).repeat(3, 1), torch.tensor([0, 1, 2]), torch.tensor([0, 3]))
    for handle in handles:
        handle.remove()
    assert len(calls) == 5


def test_packed_samples_are_isolated_and_checkpoint_backward():
    torch.manual_seed(8)
    model = AgnesDiT(small_config()).eval()
    tokens = torch.randn(8, 6)
    modality = torch.tensor([0, 0, 1, 2, 0, 0, 1, 2])
    coords = torch.tensor([[0., 0., 0., 1., 1., 1., 1., 1., 1.]]).repeat(8, 1)
    packed = model(tokens, coords, modality, torch.tensor([0, 4, 8]))
    separate = torch.cat([model(tokens[a:a+4], coords[a:a+4], modality[a:a+4], torch.tensor([0, 4])) for a in (0, 4)])
    torch.testing.assert_close(packed, separate)
    assert torch.count_nonzero(packed[modality == 2]) == 0
    model.train()
    model.block.gradient_checkpointing = True
    tokens.requires_grad_()
    model(tokens, coords, modality, torch.tensor([0, 4, 8])).float().square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()


def test_attention_registry_sdpa_fallback_preserves_gqa_packing_and_gradients():
    torch.manual_seed(106)
    attention = Attention(small_config(), 1)
    values = [torch.randn(7, heads, 8, dtype=torch.float64, requires_grad=True)
              for heads in (4, 2, 2)]
    references = [value.detach().clone().requires_grad_(True) for value in values]
    offsets = torch.tensor([0, 2, 7], dtype=torch.int32)
    actual = attention._packed_attention(*values, offsets)
    q, k, v = references
    k, v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
    expected = torch.cat([torch.nn.functional.scaled_dot_product_attention(
        q[start:end].transpose(0, 1).unsqueeze(0),
        k[start:end].transpose(0, 1).unsqueeze(0),
        v[start:end].transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        for start, end in ((0, 2), (2, 7))])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual.square().sum(), values)
    reference_gradients = torch.autograd.grad(expected.square().sum(), references)
    for gradient, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["flash", "cudnn", "flash3", "flash4"])
def test_explicit_unavailable_attention_backend_never_falls_back(backend):
    attention = Attention(small_config(), 1)
    attention.backend = backend
    values = [torch.randn(3, heads, 8) for heads in (4, 2, 2)]
    with pytest.raises(RuntimeError, match="CUDA|available"):
        attention._packed_attention(*values, torch.tensor([0, 3]))


def test_pipeline_binds_attention_policy_and_rejects_unknown_backend():
    from types import SimpleNamespace
    from mirai.core.models.agnes_video.pipeline import AgnesVideoPipeline
    config = SimpleNamespace(params=SimpleNamespace(family_params={}), attention_backend="flash3")
    pipeline = AgnesVideoPipeline(config, transformer=AgnesDiT(small_config()))
    assert all(layer.attention.backend == "flash3" for layer in pipeline.transformer.block.layers)
    config.attention_backend = "unknown"
    with pytest.raises(ValueError, match="attention backend"):
        AgnesVideoPipeline(config, transformer=AgnesDiT(small_config()))


def test_bf16_rmsnorm_shared_fp32_gradient_paths_match_reference():
    torch.manual_seed(107)
    norm = MultiModalityRMSNorm(32)
    value = torch.randn(3, 32, dtype=torch.bfloat16, requires_grad=True)
    reference = value.detach().clone().requires_grad_(True)
    gradient = torch.randn_like(value)
    floating = reference.float()
    expected = (floating * torch.rsqrt((floating ** 2).mean(-1, keepdim=True) + 1e-6)).to(reference.dtype)
    actual = norm(value)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)


def test_explicit_flex_preserves_packed_gqa_outputs_and_gradients():
    torch.manual_seed(108)
    attention = Attention(small_config(), 1)
    values = [torch.randn(7, heads, 8, requires_grad=True) for heads in (4, 2, 2)]
    references = [value.detach().clone().requires_grad_(True) for value in values]
    offsets = torch.tensor([0, 2, 7], dtype=torch.int32)
    attention.backend = "flex"
    actual = attention._packed_attention(*values, offsets)
    attention.backend = "auto"
    expected = attention._packed_attention(*references, offsets)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    gradients = torch.autograd.grad(actual.square().sum(), values)
    reference_gradients = torch.autograd.grad(expected.square().sum(), references)
    for gradient, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, reference, rtol=1e-5, atol=1e-5)


def test_frozen_modality_forward_does_not_retain_evicted_weight_storage():
    import gc
    from torch.multiprocessing.reductions import StorageWeakRef
    torch.manual_seed(109)
    linear = NativeMoELinear(5, 3, 3).requires_grad_(False)
    x = torch.randn(6, 5, requires_grad=True)
    dispatcher = ModalityDispatcher(torch.tensor([0, 0, 1, 1, 2, 2]))
    original = linear.weight.detach().clone()
    storage = StorageWeakRef(linear.weight.untyped_storage())
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
        out = linear(x, dispatcher)
    assert not saved  # Frozen base backward needs the live owner and input shape only.
    linear.weight.data = original.clone()
    gc.collect()
    assert storage.expired()
    gradient = torch.randn_like(out)
    out.backward(gradient)
    expected = torch.cat([gradient[start:start+2].float() @ original[index*3:(index+1)*3].float()
                          for index, start in enumerate((0, 2, 4))])
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)


def test_unfrozen_modality_native_weight_gradient_preserves_selected_rows():
    torch.manual_seed(110)
    linear = NativeMoELinear(5, 3, 3)
    x = torch.randn(4, 5, requires_grad=True)
    dispatcher = ModalityDispatcher(torch.tensor([0, 0, 2, 2]))
    gradient = torch.randn(4, 3)
    output = linear(x, dispatcher)
    output.backward(gradient)
    expected = torch.zeros_like(linear.weight)
    expected[:3] = gradient[:2].float().t() @ x[:2].detach().bfloat16().float()
    expected[6:] = gradient[2:].float().t() @ x[2:].detach().bfloat16().float()
    torch.testing.assert_close(linear.weight.grad, expected, rtol=0, atol=0)
    assert torch.count_nonzero(linear.weight.grad[3:6]) == 0


@pytest.mark.skipif(os.environ.get("MIRAI_AGNES_CUDA_PROBE") != "1" or not torch.cuda.is_available(),
                   reason="CUDA residency evidence requires explicit controlled GPU-worker opt-in")
def test_cuda_checkpoint_forward_releases_frozen_modality_storage_before_backward():
    import gc
    from torch.multiprocessing.reductions import StorageWeakRef
    from torch.utils.checkpoint import checkpoint
    linear = NativeMoELinear(64, 32, 3).cuda().requires_grad_(False)
    x = torch.randn(9, 64, device="cuda", requires_grad=True)
    dispatcher = ModalityDispatcher(torch.tensor([0]*3 + [1]*3 + [2]*3, device="cuda"))
    cpu_weight = linear.weight.detach().cpu().clone()
    storage = StorageWeakRef(linear.weight.untyped_storage())
    output = checkpoint(lambda value: linear(value, dispatcher), x, use_reentrant=False)
    linear.weight.data = cpu_weight
    torch.cuda.synchronize()
    gc.collect()
    assert storage.expired(), "Autograd retained an evicted CUDA modality-weight view."
    gradient = torch.randn_like(output)
    output.backward(gradient)
    expected = torch.cat([gradient[index*3:(index+1)*3].float() @ cpu_weight[index*32:(index+1)*32].cuda().float()
                          for index in range(3)])
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)


def test_cold_backend_backward_probe_is_outside_checkpoint_and_preserves_rng(monkeypatch):
    from contextlib import contextmanager
    import mirai.core.models.attention_backends as backends
    import mirai.vendors.agnes_video.dit as native
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    capturing = [False]
    probed = set()

    @contextmanager
    def capture_region():
        capturing[0] = True
        try:
            yield
        finally:
            capturing[0] = False

    def checked_checkpoint(function, *args, **kwargs):
        return torch_checkpoint(function, *args, **kwargs,
                                context_fn=lambda: (capture_region(), capture_region()))

    def cold_status(name, *, device):
        assert not capturing[0], "Capability probing entered checkpoint capture."
        if name not in probed:
            probed.add(name)
            probe = torch.randn(8, requires_grad=True)
            probe.square().sin().sum().backward()
        return backends.AttentionBackendStatus(name, False, "mock unavailable", None)

    monkeypatch.setattr(native, "checkpoint", checked_checkpoint)
    monkeypatch.setattr(backends, "varlen_attention_backward_status", cold_status)
    torch.manual_seed(111)
    model = AgnesDiT(small_config())
    model.gradient_checkpointing = True
    tokens = torch.randn(7, 6, requires_grad=True)
    coordinates = torch.tensor([[0., 0., 0., 1., 1., 1., 1., 1., 1.]]).repeat(7, 1)
    modalities = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    before = torch.get_rng_state().clone()
    output = model(tokens, coordinates, modalities, torch.tensor([0, 7]))
    output.float().square().sum().backward()
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert probed == {"flash4", "flash3"}
    cold_gradient = tokens.grad.clone()
    model.zero_grad()
    tokens.grad = None
    warm_output = model(tokens, coordinates, modalities, torch.tensor([0, 7]))
    warm_output.float().square().sum().backward()
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(warm_output, output, rtol=0, atol=0)
    torch.testing.assert_close(tokens.grad, cold_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["bf16", "fp16", "fp32"])
def test_globally_bf16_checkpoint_heads_and_norms_rematerialize_without_frozen_saved_storage(precision):
    import copy
    from types import SimpleNamespace
    from mirai.core.models.agnes_video.pipeline import AgnesVideoPipeline
    torch.manual_seed(115)
    native = AgnesDiT(small_config()).to(torch.bfloat16)
    pipeline = AgnesVideoPipeline(SimpleNamespace(dtype=precision, params=SimpleNamespace(family_params={})), transformer=native)
    pipeline.set_adapter_config(SimpleNamespace(type="lora", target_preset="attn_mlp", rank=2,
        alpha=2, train_router=False, init="kaiming"))
    reference = copy.deepcopy(pipeline)
    def ordinary_projection(x, module):
        # FP32 mode converts the frozen BF16 matrix for the explicit FP32 dense
        # computation; the source loop otherwise has a mixed-dtype mismatch.
        weight = module.weight if torch.is_autocast_enabled(x.device.type) else module.weight.to(x.dtype)
        bias = module.bias
        if bias is not None and not torch.is_autocast_enabled(x.device.type):
            bias = bias.to(x.dtype)
        return torch.nn.functional.linear(x, weight, bias)
    for head in (reference.transformer.final_linear_video, reference.transformer.final_linear_audio):
        head.forward = lambda x, head=head: ordinary_projection(x, head)
    for layer in reference.transformer.block.layers:
        if isinstance(layer.mlp, SparseMoEMLP):
            router = layer.mlp.router
            router.forward = lambda x, router=router: ordinary_projection(x, router)
    for norm in reference.transformer.modules():
        if not isinstance(norm, MultiModalityRMSNorm):
            continue
        def raw_norm(x, modality_dispatcher=None, norm=norm):
            floating = x.float()
            normalized = floating * torch.rsqrt((floating ** 2).mean(-1, keepdim=True) + norm.eps)
            if norm.num_modality == 1:
                return (normalized * (norm.weight + 1)).to(x.dtype)
            chunks = normalized.split(modality_dispatcher.group_size_cpu, 0)
            return torch.cat([chunk * (weight + 1) for chunk, weight in zip(chunks, norm.weight.chunk(3))]).to(x.dtype)
        norm.forward = raw_norm
    tokens = torch.randn(7, 6, requires_grad=True)
    source_tokens = tokens.detach().clone().requires_grad_(True)
    coords = torch.tensor([[0., 0., 0., 1., 1., 1., 1., 1., 1.]]).repeat(7, 1)
    modalities = torch.tensor([0, 0, 0, 1, 1, 2, 2])

    def run(model, value):
        frozen = {parameter.untyped_storage().data_ptr(): name for name, parameter in model.transformer.named_parameters()
                  if not parameter.requires_grad and parameter.dtype == torch.bfloat16}
        retained = set()
        def capture(tensor):
            if tensor.dtype == torch.bfloat16 and tensor.untyped_storage().data_ptr() in frozen:
                retained.add(frozen[tensor.untyped_storage().data_ptr()])
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(capture, lambda tensor: tensor):
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
            with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
                output = model.transformer(value, coords, modalities, torch.tensor([0, 7]))
            output.float().square().sum().backward()
        return output, retained

    actual, retained = run(pipeline, tokens)
    expected, reference_retained = run(reference, source_tokens)
    assert not retained
    if precision == "bf16":
        assert {"final_linear_video.weight", "final_linear_audio.weight"}.issubset(reference_retained)
        assert "block.layers.1.mlp.router.weight" in reference_retained
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(tokens.grad, source_tokens.grad, rtol=0, atol=0)
    for (name, parameter), (reference_name, reference_parameter) in zip(
            pipeline.get_named_trainable_parameters(), reference.get_named_trainable_parameters()):
        assert name == reference_name
        assert (parameter.grad is None) == (reference_parameter.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)


def test_frozen_bf16_norm_affine_scales_are_not_saved_and_keep_exact_rounding():
    torch.manual_seed(116)
    norm = MultiModalityRMSNorm(32, num_modality=3).to(torch.bfloat16).requires_grad_(False)
    norm.weight.data.normal_(std=.1)
    value = torch.randn(6, 32, dtype=torch.bfloat16, requires_grad=True)
    reference = value.detach().clone().requires_grad_(True)
    dispatcher = ModalityDispatcher(torch.tensor([0, 0, 1, 1, 2, 2]))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        actual = norm(value, dispatcher)
    assert not any(tensor.dtype == torch.bfloat16 and tensor.ndim == 1 for tensor in saved)
    floating = reference.float()
    normalized = floating * torch.rsqrt((floating ** 2).mean(-1, keepdim=True) + norm.eps)
    expected = torch.cat([chunk * (weight + 1) for chunk, weight in zip(
        normalized.split([2, 2, 2]), norm.weight.chunk(3))]).to(reference.dtype)
    gradient = torch.randn_like(value)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)


def test_frozen_dense_and_norm_canonical_relocation_releases_old_storage_and_preserves_gradients():
    import gc
    from torch.multiprocessing.reductions import StorageWeakRef
    from mirai.vendors.agnes_video.dit import FrozenDenseLinear
    torch.manual_seed(117)
    norm = MultiModalityRMSNorm(32, num_modality=3).to(torch.bfloat16).requires_grad_(False)
    norm.weight.data.normal_(std=.1)
    dense = FrozenDenseLinear(32, 16, dtype=torch.bfloat16).requires_grad_(False)
    norm_copy, weight_copy, bias_copy = [tensor.detach().clone() for tensor in (norm.weight, dense.weight, dense.bias)]
    old_norm = StorageWeakRef(norm.weight.untyped_storage())
    old_dense = StorageWeakRef(dense.weight.untyped_storage())
    old_bias = StorageWeakRef(dense.bias.untyped_storage())
    value = torch.randn(6, 32, dtype=torch.bfloat16, requires_grad=True)
    reference = value.detach().clone().requires_grad_(True)
    dispatcher = ModalityDispatcher(torch.tensor([0, 0, 1, 1, 2, 2]))
    actual = dense(norm(value, dispatcher))
    norm.weight.data = norm_copy.clone()
    dense.weight.data = weight_copy.clone()
    dense.bias.data = bias_copy.clone()
    gc.collect()
    assert old_norm.expired() and old_dense.expired() and old_bias.expired()
    floating = reference.float()
    normalized = floating * torch.rsqrt((floating ** 2).mean(-1, keepdim=True) + norm.eps)
    affine = torch.cat([chunk * (weight + 1) for chunk, weight in zip(
        normalized.split([2, 2, 2]), norm_copy.chunk(3))]).to(reference.dtype)
    expected = torch.nn.functional.linear(affine, weight_copy, bias_copy)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)


def test_native_vae_runs_eager_when_compilation_is_unavailable(monkeypatch):
    import importlib
    from mirai.vendors.agnes_video import vae
    def unavailable(*args, **kwargs):
        raise AssertionError("Default native VAE execution must not invoke torch.compile.")
    monkeypatch.setattr(torch, "compile", unavailable)
    module = importlib.reload(vae)
    convolution = module.CausalConv3d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        convolution.weight.fill_(1)
    result = convolution(torch.ones(1, 1, 3, 3, 3))
    torch.testing.assert_close(result[0, 0, :, 1, 1], torch.tensor([9., 18., 27.]))


def test_released_inference_auto_uses_plain_attention_per_sequence_without_expanding_gqa(monkeypatch):
    from mirai.vendors.agnes_video import dit
    from mirai.core.models import attention_backends

    attention = dit.Attention(AgnesDiTConfig(hidden_size=32, head_dim=8, num_query_groups=2), 1).eval()
    attention.released_inference_attention = True
    calls = []

    def plain(q, k, v):
        calls.append((q.shape, k.shape, v.shape))
        return q + 1

    monkeypatch.setattr(dit, "_released_inference_attention_function", lambda device: ("flash3", plain))
    monkeypatch.setattr(attention_backends, "dispatch_varlen_attention",
                        lambda *args, **kwargs: pytest.fail("released inference must use plain attention"))
    query = torch.randn(7, 4, 8)
    key, value = torch.randn(7, 2, 8), torch.randn(7, 2, 8)
    with torch.no_grad():
        output = attention._packed_attention(query, key, value, torch.tensor([0, 3, 7]))
    assert calls == [(torch.Size([1, 3, 4, 8]), torch.Size([1, 3, 2, 8]), torch.Size([1, 3, 2, 8])),
                     (torch.Size([1, 4, 4, 8]), torch.Size([1, 4, 2, 8]), torch.Size([1, 4, 2, 8]))]
    torch.testing.assert_close(output, query + 1)
    assert attention._backend_resolutions[("released_auto", "cpu", False)] == "flash3"


def test_released_inference_sdpa_fallback_matches_source_gqa_expansion(monkeypatch):
    from mirai.vendors.agnes_video import dit

    attention = dit.Attention(AgnesDiTConfig(hidden_size=32, head_dim=8, num_query_groups=2), 1).eval()
    attention.released_inference_attention = True
    monkeypatch.setattr(dit, "_released_inference_attention_function", lambda device: ("auto", None))
    query = torch.randn(7, 4, 8)
    key, value = torch.randn(7, 2, 8), torch.randn(7, 2, 8)
    expected = []
    for start, end in ((0, 3), (3, 7)):
        q, k, v = [tensor[start:end].transpose(0, 1).unsqueeze(0).contiguous()
                   for tensor in (query, key, value)]
        expected.append(torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1),
            dropout_p=0.0, is_causal=False).transpose(1, 2).squeeze(0))
    with torch.no_grad():
        output = attention._packed_attention(query, key, value, torch.tensor([0, 3, 7]))
    torch.testing.assert_close(output, torch.cat(expected), rtol=0, atol=0)


def test_explicit_attention_backend_keeps_generic_varlen_dispatch(monkeypatch):
    from mirai.vendors.agnes_video import dit
    from mirai.core.models import attention_backends

    attention = dit.Attention(AgnesDiTConfig(hidden_size=32, head_dim=8, num_query_groups=2), 1).eval()
    attention.released_inference_attention = True
    attention.backend = "flash3"
    monkeypatch.setattr(attention, "prepare_backend", lambda **kwargs: "flash3")
    monkeypatch.setattr(dit, "_released_inference_attention_function",
                        lambda device: pytest.fail("explicit backend must retain its generic policy"))
    calls = []
    def varlen(q, k, v, **kwargs):
        calls.append(kwargs["backend"])
        return q
    monkeypatch.setattr(attention_backends, "dispatch_varlen_attention", varlen)
    query = torch.randn(7, 4, 8)
    with torch.no_grad():
        output = attention._packed_attention(query, torch.randn(7, 2, 8), torch.randn(7, 2, 8),
                                             torch.tensor([0, 3, 7]))
    assert calls == ["flash3"]
    assert torch.equal(output, query)


@pytest.mark.parametrize("major,fa3,fa2,expected", [(9, True, True, "flash3"),
    (8, True, True, "flash2"), (9, False, True, "flash2"), (9, False, False, "auto")])
def test_released_attention_resolver_uses_source_hardware_and_import_order(monkeypatch, major, fa3, fa2, expected):
    import sys
    from types import ModuleType
    from mirai.vendors.agnes_video import dit

    function = lambda *args: None
    fa3_module = ModuleType("flash_attn_interface")
    fa3_module.flash_attn_func = function
    fa2_module = ModuleType("flash_attn.flash_attn_interface")
    fa2_module.flash_attn_func = function
    package = ModuleType("flash_attn")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "flash_attn_interface", fa3_module if fa3 else None)
    monkeypatch.setitem(sys.modules, "flash_attn", package)
    monkeypatch.setitem(sys.modules, "flash_attn.flash_attn_interface", fa2_module if fa2 else None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (major, 0))
    backend, resolved_function = dit._released_inference_attention_function(torch.device("cuda:0"))
    assert backend == expected
    assert resolved_function is (None if expected == "auto" else function)


def test_released_inference_flag_does_not_change_training_auto_policy(monkeypatch):
    from types import SimpleNamespace
    from mirai.vendors.agnes_video import dit
    from mirai.core.models import attention_backends

    attention = dit.Attention(AgnesDiTConfig(hidden_size=32, head_dim=8, num_query_groups=2), 1)
    attention.released_inference_attention = True
    monkeypatch.setattr(dit, "_released_inference_attention_function",
                        lambda device: pytest.fail("training retains generic backend selection"))
    monkeypatch.setattr(attention_backends, "attention_backend_status",
                        lambda *args, **kwargs: SimpleNamespace(available=True))
    assert attention.prepare_backend(device=torch.device("cpu"), needs_backward=False) == "flash4"


def test_inference_bmm_preserves_imbalanced_top_one_dispatch_and_empty_expert(monkeypatch):
    module = SparseMoEMLP(small_config()).eval().requires_grad_(False)
    module.released_inference_grouped = True
    inputs = torch.linspace(-1, 1, 5 * 32).reshape(5, 32).bfloat16()
    indices = torch.tensor([[2], [0], [2], [2], [0]])
    gates = torch.ones(5, 1, dtype=torch.bfloat16)
    expected = torch.stack([module.experts[int(expert)](value.unsqueeze(0))[0]
                            for expert, value in zip(indices[:, 0], inputs)])
    calls = []
    original = torch.bmm
    def capture(left, right):
        calls.append((tuple(left.shape), tuple(right.shape)))
        return original(left, right)
    monkeypatch.setattr(torch, "bmm", capture)
    with torch.no_grad():
        assert module._can_use_inference_bmm()
        actual = module._inference_bmm(inputs, indices, gates)
    assert calls[0][0] == (3, 3, 32)
    assert calls[1][0][:2] == (3, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    module.train()
    with torch.no_grad():
        assert not module._can_use_inference_bmm()
    module.eval()
    assert not module._can_use_inference_bmm()  # Grad-enabled execution retains the loop.
    module.top_k = 2
    with torch.no_grad():
        assert not module._can_use_inference_bmm()


def test_inference_bmm_keeps_active_lora_and_identical_base_weights_at_scale_zero(monkeypatch):
    from mirai.core.models.agnes_video.lora import AgnesLoRALinear

    torch.manual_seed(121)
    module = SparseMoEMLP(small_config()).eval().requires_grad_(False)
    module.released_inference_grouped = True
    for expert in module.experts:
        for name in ("up_gate_proj", "down_proj"):
            wrapped = AgnesLoRALinear(getattr(expert, name), rank=2, alpha=2)
            with torch.no_grad():
                wrapped.lora_b.fill_(0.125)
            setattr(expert, name, wrapped)
    module.eval()
    inputs = torch.randn(5, 32).bfloat16()
    indices = torch.tensor([[2], [0], [2], [2], [0]])
    gates = torch.ones(5, 1, dtype=torch.bfloat16)
    weights = []
    original = torch.bmm
    def capture(left, right):
        weights.append(right.clone())
        return original(left, right)
    monkeypatch.setattr(torch, "bmm", capture)
    results = []
    with torch.no_grad():
        for scale in (0, 1, 0):
            for expert in module.experts:
                expert.up_gate_proj.set_lora_scale(scale)
                expert.down_proj.set_lora_scale(scale)
            results.append(module._inference_bmm(inputs, indices, gates))
    assert not torch.equal(results[0], results[1])
    torch.testing.assert_close(results[0], results[2], rtol=0, atol=0)
    for index in (0, 1):
        torch.testing.assert_close(weights[index], weights[index+2], rtol=0, atol=0)
        torch.testing.assert_close(weights[index], weights[index+4], rtol=0, atol=0)


def test_grouped_lora_projection_reuses_existing_unmerged_delta_math():
    from mirai.core.models.agnes_video.lora import AgnesLoRALinear
    from mirai.vendors.agnes_video.dit import ExpertLinear

    torch.manual_seed(122)
    wrapped = AgnesLoRALinear(ExpertLinear(32, 64, bias=False, dtype=torch.bfloat16), rank=3, alpha=5).eval()
    with torch.no_grad():
        wrapped.lora_b.normal_(std=0.1)
    inputs = torch.randn(7, 32).bfloat16()
    for scale in (0, 0.5, 1):
        wrapped.set_lora_scale(scale)
        with torch.no_grad():
            expected = wrapped(inputs)
            actual = wrapped.apply_inference_delta(inputs, wrapped._base_forward(inputs))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adapter_cpu_autocast_keeps_source_input_dtype():
    from mirai.vendors.agnes_video.dit import Adapter

    adapter = Adapter(small_config()).eval()
    tokens = torch.randn(3, 6)
    coords = torch.tensor([[0., 0., 0., 1., 1., 1., 1., 1., 1.]]).repeat(3, 1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, _ = adapter(tokens, coords, torch.tensor([0, 1, 2]))
    assert output.dtype == tokens.dtype


@pytest.mark.skipif(os.environ.get("MIRAI_AGNES_CUDA_PROBE") != "1" or not torch.cuda.is_available(),
                   reason="CUDA adapter contracts require explicit controlled GPU-worker opt-in")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_adapter_output_uses_released_autocast_dtype(dtype):
    from mirai.vendors.agnes_video.dit import Adapter

    adapter = Adapter(small_config()).eval().cuda()
    tokens = torch.randn(3, 6, device="cuda")
    coords = torch.tensor([[0., 0., 0., 1., 1., 1., 1., 1., 1.]], device="cuda").repeat(3, 1)
    modalities = torch.tensor([0, 1, 2], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        output, _ = adapter(tokens, coords, modalities)
        expected = torch.cat([embedder(tokens[index:index+1, :embedder.in_features])
                              for index, embedder in enumerate((adapter.video_embedder,
                                  adapter.audio_embedder, adapter.text_embedder))])
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
