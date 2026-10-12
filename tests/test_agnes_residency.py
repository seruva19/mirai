"""Agnes RAM block residency, checkpointing, and adapter ownership contracts."""

from types import SimpleNamespace

import pytest
import torch

from mirai.core.models.agnes_video.pipeline import AgnesVideoPipeline
from mirai.vendors.agnes_video.dit import AgnesDiT, AgnesDiTConfig


def make_pipeline(adapter=True):
    config = AgnesDiTConfig(num_layers=2, hidden_size=16, head_dim=8,
        num_query_groups=1, video_in_channels=8, audio_in_channels=4,
        text_in_channels=12, mm_layers=(0,), sparse_moe_layers=(1,), gelu7_layers=(0,))
    result = AgnesVideoPipeline(SimpleNamespace(params=SimpleNamespace(family_params={})),
                                transformer=AgnesDiT(config))
    if adapter:
        result.set_adapter_config(SimpleNamespace(type="lora", target_preset="attn_mlp",
            rank=2, alpha=2, train_router=False, init="kaiming"))
    return result


def arm(model, **overrides):
    request = dict(strategy="block_swap", blocks_to_swap=2, mode="sync")
    request.update(overrides)
    model.set_weight_residency_strategy(**request)


def test_default_disabled_and_strict_training_preconditions():
    model = make_pipeline(False)
    assert model.get_block_swap_state() == {"enabled": False}
    assert model.get_memory_feature_capabilities().block_swap
    with pytest.raises(ValueError, match="configured adapter"):
        arm(model)
    with pytest.raises(ValueError, match="RAM"):
        arm(model, strategy="stream_disk")
    model = make_pipeline()
    arm(model)
    with pytest.raises(ValueError, match="standard gradient checkpointing"):
        model.place_offloaded_modules(device="cpu", strategy="block_swap")


def test_cpu_ram_swap_matches_resident_outputs_and_adapter_gradients():
    torch.manual_seed(104)
    resident, swapped = make_pipeline(), make_pipeline()
    swapped.transformer.load_state_dict(resident.transformer.state_dict(), strict=True)
    resident.set_gradient_checkpointing("standard")
    swapped.set_gradient_checkpointing("standard")
    arm(swapped)
    swapped.place_offloaded_modules(device="cpu", strategy="block_swap")
    manager = swapped._block_swap_manager
    trainable_ids = {id(parameter) for parameter in swapped.get_trainable_parameters()}
    assert manager._units
    assert all(not slot.owner._parameters[slot.name].requires_grad
        for unit in manager._units.values() for slot in unit._slots if slot.kind == "parameter")
    assert {id(parameter) for parameter in swapped.get_trainable_parameters()} == trainable_ids
    video, audio, text = torch.randn(1, 2, 1, 2, 2), torch.randn(1, 3, 4), torch.randn(1, 2, 12)
    outputs = []
    for model in (resident, swapped):
        frozen = {parameter.data_ptr() for parameter in model.transformer.parameters()
                  if not parameter.requires_grad and parameter.dtype == torch.bfloat16}
        saved = []
        def capture(tensor, saved=saved):
            if tensor.dtype == torch.bfloat16:
                saved.append(tensor.data_ptr())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(capture, lambda tensor: tensor):
            output = model(video, torch.tensor([0.5]), {"t5": text}, audio_latents=audio)
            sum(value.float().square().mean() for value in output).backward()
        assert not frozen.intersection(saved)
        outputs.append(output)
    for first, second in zip(*outputs):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for (name, first), (other_name, second) in zip(resident.get_named_trainable_parameters(), swapped.get_named_trainable_parameters()):
        assert name == other_name
        assert (first.grad is None) == (second.grad is None)
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)
    state = swapped.get_block_swap_state()
    assert state["enabled"] and state["blocks_to_swap"] == 2
    assert any(event["kind"] == "swap_in" for event in state["events"])
    swapped.finish_backward_offloads()
    assert all(unit.device == "cpu" for unit in manager._units.values())


def test_rebinding_and_disabling_remove_stale_hooks():
    model = make_pipeline()
    model.set_gradient_checkpointing("standard")
    arm(model)
    model.place_offloaded_modules(device="cpu", strategy="block_swap")
    first_manager = model._block_swap_manager
    model.place_offloaded_modules(device="cpu", strategy="block_swap")
    assert len(model._block_hook_handles) == 4
    arm(model, strategy="disabled", blocks_to_swap=0)
    assert model.get_block_swap_state() == {"enabled": False}
    assert not model._block_hook_handles and not first_manager._backward_hooks
    model.place_offloaded_modules(device="cpu", strategy="disabled")


def test_inference_swap_requires_eval_but_not_adapter_or_checkpoint():
    model = make_pipeline(False)
    arm(model, execution_mode="inference")
    with pytest.raises(ValueError, match="eval"):
        model.place_offloaded_modules(device="cpu", strategy="block_swap")
    model.eval()
    model.place_offloaded_modules(device="cpu", strategy="block_swap")
    with torch.no_grad():
        output = model(torch.randn(1, 2, 1, 2, 2), torch.tensor([0.5]),
                       {"t5": torch.randn(1, 2, 12)}, audio_latents=torch.randn(1, 3, 4))
    assert all(torch.isfinite(value).all() for value in output)
    model.flush_runtime_offloads()
    with pytest.raises(ValueError, match="phase_aware"):
        arm(model, execution_mode="inference", block_residency_planner="phase_aware")
