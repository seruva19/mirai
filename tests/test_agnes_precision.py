"""Agnes selected AMP precision and native mixed-dtype boundaries."""

from types import SimpleNamespace
import os

import pytest
import torch

from mirai.core.models.agnes_video.pipeline import AgnesVideoPipeline
from mirai.vendors.agnes_video.dit import AgnesDiT, AgnesDiTConfig


def make_pipeline(dtype, seed=112, native_config=None):
    torch.manual_seed(seed)
    native = AgnesDiT(native_config or AgnesDiTConfig(num_layers=2, hidden_size=16, head_dim=8,
        num_query_groups=1, video_in_channels=8, audio_in_channels=4, text_in_channels=12,
        mm_layers=(0,), sparse_moe_layers=(1,), gelu7_layers=(0,)))
    model = AgnesVideoPipeline(SimpleNamespace(dtype=dtype, params=SimpleNamespace(family_params={})), transformer=native)
    model.set_adapter_config(SimpleNamespace(type="lora", target_preset="attn_mlp", rank=2,
        alpha=2, train_router=False, init="kaiming"))
    return model


@pytest.mark.parametrize("name,initial,output", [("bf16", torch.bfloat16, torch.bfloat16),
    ("fp16", torch.float16, torch.float32), ("fp32", torch.float32, torch.float32)])
def test_selected_precision_controls_amp_and_preserves_native_residual_promotion(name, initial, output):
    model = make_pipeline(name)
    storage = {key: (value.dtype, value.data_ptr()) for key, value in model.transformer.named_parameters()}
    seen, projected = [], []
    model.transformer.block.layers[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].dtype))
    model.transformer.block.layers[1].register_forward_pre_hook(lambda module, args: seen.append(args[0].dtype))
    for expert in model.transformer.block.layers[1].mlp.experts:
        expert.up_gate_proj.register_forward_hook(lambda module, args, result: projected.append(result.dtype))
    result = model(torch.randn(1, 2, 1, 2, 2), torch.tensor([.5]),
                   {"t5": torch.randn(1, 2, 12)}, audio_latents=torch.randn(1, 3, 4))
    assert seen[0] == initial
    assert seen[1] == output  # FP16 + native BF16 branch promotes to FP32.
    assert all(value.dtype == output and torch.isfinite(value).all() for value in result)
    assert projected and all(value == (torch.float16 if name == "fp16" else torch.bfloat16) for value in projected)
    sum(value.float().square().mean() for value in result).backward()
    parameters = model.get_trainable_parameters()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert storage == {key: (value.dtype, value.data_ptr()) for key, value in model.transformer.named_parameters()}


def test_fp32_family_disables_an_enclosing_amp_context_and_placement_hook_updates_policy():
    model = make_pipeline("fp32")
    observed = []
    model.transformer.adapter.video_embedder.register_forward_hook(lambda module, args, result: observed.append(result.dtype))
    inputs = (torch.randn(1, 2, 1, 2, 2), torch.tensor([.5]), {"t5": torch.randn(1, 2, 12)})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(*inputs, audio_latents=torch.randn(1, 3, 4))
    assert observed == [torch.float32]
    model.set_compute_autocast_dtype(torch.float16)
    assert model.runtime_config.params_dtype == torch.float16
    assert model.transformer.config.params_dtype == torch.float16
    with pytest.raises(ValueError, match="compute dtype"):
        model.set_compute_autocast_dtype(torch.float64)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_frozen_lora_linear_amp_backward_matches_reference(dtype):
    from mirai.core.models.agnes_video.lora import AgnesLoRALinear
    torch.manual_seed(113)
    base = torch.nn.Linear(6, 4, bias=False).requires_grad_(False)
    wrapped = AgnesLoRALinear(base, rank=2, alpha=2)
    x = torch.randn(3, 6, requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=dtype):
        actual = wrapped(x)
        expected = torch.nn.functional.linear(reference, base.weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_adapter_lineage_allows_same_base_compute_policy_changes_but_not_wrong_base(dtype):
    source, target = make_pipeline("bf16"), make_pipeline(dtype)
    target.load_adapter_state(source.state_dict())
    assert source._adapter_base_fingerprint == target._adapter_base_fingerprint
    assert "params_dtype" not in source._adapter_lineage["architecture"]
    wrong = make_pipeline(dtype, seed=113)
    assert wrong._adapter_base_fingerprint != source._adapter_base_fingerprint
    before = {key: value.clone() for key, value in wrong.state_dict().items()}
    with pytest.raises(ValueError, match="base checkpoint"):
        wrong.load_adapter_state(source.state_dict())
    for key, value in wrong.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


controlled_cuda = pytest.mark.skipif(
    os.environ.get("MIRAI_AGNES_CUDA_PROBE") != "1" or not torch.cuda.is_available(),
    reason="CUDA precision contracts require explicit controlled GPU-worker opt-in",
)


@controlled_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_frozen_expert_amp_output_and_input_gradient_match_reference(dtype):
    from mirai.vendors.agnes_video.dit import ExpertLinear
    torch.manual_seed(114)
    linear = ExpertLinear(64, 128, bias=False, device="cuda", dtype=torch.float32).requires_grad_(False)
    value = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference = value.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=dtype):
        actual = linear(value)
        expected = torch.nn.functional.linear(reference, linear.weight)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
    assert linear.weight.grad is None


@controlled_cuda
@pytest.mark.parametrize("name,output_dtype", [("bf16", torch.bfloat16),
    ("fp16", torch.float32), ("fp32", torch.float32)])
def test_cuda_joint_precision_standard_checkpoint_forward_backward_is_finite(name, output_dtype):
    config = AgnesDiTConfig(num_layers=2, hidden_size=256, head_dim=64,
        num_query_groups=2, video_in_channels=8, audio_in_channels=4, text_in_channels=12,
        mm_layers=(0,), sparse_moe_layers=(1,), gelu7_layers=(0,))
    model = make_pipeline(name, native_config=config)
    model.set_gradient_checkpointing("standard")
    model.place_offloaded_modules(device="cuda", strategy="disabled")
    assert model.transformer.gradient_checkpointing
    frozen = {parameter.data_ptr() for parameter in model.transformer.parameters()
              if not parameter.requires_grad and parameter.dtype == torch.bfloat16}
    saved = []

    def capture(tensor):
        if tensor.dtype == torch.bfloat16:
            saved.append(tensor.data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(capture, lambda tensor: tensor):
        output = model(torch.randn(1, 2, 1, 2, 2, device="cuda"), torch.tensor([.5], device="cuda"),
            {"t5": torch.randn(1, 3, 12, device="cuda")}, audio_latents=torch.randn(1, 5, 4, device="cuda"))
        assert all(value.dtype == output_dtype and torch.isfinite(value).all() for value in output)
        sum(value.float().square().mean() for value in output).backward()
    assert not frozen.intersection(saved)
    gradients = [parameter.grad for parameter in model.get_trainable_parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
