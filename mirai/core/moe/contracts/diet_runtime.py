"""Behavioral contract for original-ID preserving DIET execution."""

from __future__ import annotations

import copy
import types

import pytest
import torch

from mirai.core.moe.storage.deletion import build_expert_deletion_plan
from mirai.core.models.compressed_weights.execution.experts import CompressedGroupedExperts
from mirai.vendors.lingbot_video.transformer_lingbot_video import (
    LingBotVideoSparseMoeBlock,
)


def _block(device: torch.device) -> LingBotVideoSparseMoeBlock:
    bulk_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    block = LingBotVideoSparseMoeBlock(
        hidden_size=16,
        intermediate_size=32,
        num_experts=8,
        top_k=2,
        moe_intermediate_size=32,
        score_func="softmax",
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
        n_shared_experts=0,
    ).to(device=device, dtype=bulk_dtype)
    block.router.to(dtype=torch.float32)
    with torch.no_grad():
        for parameter in block.parameters():
            if parameter.is_floating_point():
                parameter.uniform_(-0.1, 0.1)
    return block


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")]
    + ([torch.device("cuda")] if torch.cuda.is_available() else []),
)
def test_diet_native_original_id_parity_and_gradients(device: torch.device) -> None:
    torch.manual_seed(7)
    reference = _block(device)
    compact = copy.deepcopy(reference)
    keep = (0, 1, 4, 5)
    plan = build_expert_deletion_plan(
        8,
        keep,
        top_k=2,
        n_group=2,
        topk_group=1,
        group_score_topk=2,
        norm_topk_prob=True,
        route_scale=1.0,
    )
    deleted = tuple(sorted(set(range(8)) - set(keep)))
    with torch.no_grad():
        reference.router.e_score_correction_bias[list(deleted)] = -1.0e4

    def independent_mask(router, scores):
        mask = torch.zeros(8, dtype=torch.bool, device=scores.device)
        mask[list(keep)] = True
        return scores.masked_fill(~mask.unsqueeze(0), 0.0)

    reference.router._mask_deleted_experts = types.MethodType(
        independent_mask, reference.router
    )
    compact.router.configure_expert_deletion(plan)
    for name in ("w1", "w2", "w3"):
        source = getattr(compact.experts, name)
        setattr(
            compact.experts,
            name,
            torch.nn.Parameter(source[list(keep)].detach().clone()),
        )
    compact.experts.num_experts = len(keep)
    compact.experts.configure_expert_deletion(plan)

    input_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    left = torch.randn(2, 3, 16, device=device, dtype=input_dtype, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    expected = reference(left)
    actual = compact(right)
    tolerance = (2e-2, 2e-3) if device.type == "cuda" else (1e-5, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(
        right.grad, left.grad, rtol=tolerance[0], atol=tolerance[1]
    )
    torch.testing.assert_close(
        compact.router.weight.grad,
        reference.router.weight.grad,
        rtol=tolerance[0],
        atol=tolerance[1],
    )
    torch.testing.assert_close(
        compact.experts.w1.grad,
        reference.experts.w1.grad[list(keep)],
        rtol=tolerance[0],
        atol=tolerance[1],
    )
    for name in ("w2", "w3"):
        torch.testing.assert_close(
            getattr(compact.experts, name).grad,
            getattr(reference.experts, name).grad[list(keep)],
            rtol=tolerance[0],
            atol=tolerance[1],
        )


def test_diet_plan_rejects_insufficient_selected_group_capacity() -> None:
    with pytest.raises(ValueError, match="selected router-group"):
        build_expert_deletion_plan(
            8,
            (0, 1, 4, 5),
            top_k=3,
            n_group=2,
            topk_group=1,
            group_score_topk=2,
            norm_topk_prob=True,
            route_scale=1.0,
        )


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")]
    + ([torch.device("cuda")] if torch.cuda.is_available() else []),
)
def test_diet_compressed_expert_lora_output_and_gradient_parity(
    device: torch.device,
) -> None:
    torch.manual_seed(19)

    class Base(torch.nn.Module):
        def __init__(self, weights=None):
            super().__init__()
            self.num_experts = 8 if weights is None else 4
            source = weights or tuple(
                torch.randn(8, *shape, device=device) * 0.04
                for shape in ((6, 4), (4, 6), (6, 4))
            )
            for name, value in zip(("w1", "w2", "w3"), source):
                setattr(self, name, torch.nn.Parameter(value))

    keep = (0, 1, 4, 5)
    full_base = Base()
    compact_base = Base(tuple(getattr(full_base, key).detach()[list(keep)].clone() for key in ("w1", "w2", "w3")))
    full = CompressedGroupedExperts(
        full_base, quant_format="int8", expert_weight_access="active_dequant"
    )
    compact = CompressedGroupedExperts(
        compact_base, quant_format="int8", expert_weight_access="active_dequant"
    )
    for module in (full, compact):
        for key in ("w1", "w2", "w3"):
            module.attach_expert_lora(
                tensor_name=key, adapter_name="diet", rank=2, alpha=2.0
            )
    with torch.no_grad():
        for key in ("w1", "w2", "w3"):
            full.expert_lora[key].lora_b.normal_(0, 0.02)
            compact.expert_lora[key].lora_a.copy_(full.expert_lora[key].lora_a[list(keep)])
            compact.expert_lora[key].lora_b.copy_(full.expert_lora[key].lora_b[list(keep)])
    plan = build_expert_deletion_plan(
        8, keep, top_k=2, n_group=2, topk_group=1, group_score_topk=2,
        norm_topk_prob=True, route_scale=1.0,
    )
    compact.configure_expert_deletion(plan)
    full = full.to(device)
    compact = compact.to(device)
    logical = torch.tensor([[0, 4], [1, 5], [4, 0], [5, 1]], device=device)
    scores = torch.rand(4, 2, device=device) + 0.1
    left = torch.randn(4, 4, device=device, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    expected = full.run_direct_routed(left, scores, logical)
    actual = compact.run_direct_routed(right, scores, logical)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(right.grad, left.grad, rtol=3e-4, atol=3e-5)
    for key in ("w1", "w2", "w3"):
        for factor in ("lora_a", "lora_b"):
            compact_grad = getattr(compact.expert_lora[key], factor).grad
            full_grad = getattr(full.expert_lora[key], factor).grad[list(keep)]
            torch.testing.assert_close(compact_grad, full_grad, rtol=3e-4, atol=3e-5)


def test_diet_host_fallback_remaps_compressed_routes_once() -> None:
    torch.manual_seed(29)
    host = _block(torch.device("cpu"))
    keep = (0, 1, 4, 5)
    for key in ("w1", "w2", "w3"):
        setattr(
            host.experts,
            key,
            torch.nn.Parameter(getattr(host.experts, key)[list(keep)].detach().clone()),
        )
    host.experts.num_experts = 4
    compressed = CompressedGroupedExperts(
        host.experts, quant_format="int8", expert_weight_access="active_dequant"
    )
    plan = build_expert_deletion_plan(
        8, keep, top_k=2, n_group=2, topk_group=1, group_score_topk=2,
        norm_topk_prob=True, route_scale=1.0,
    )
    compressed.configure_expert_deletion(plan)
    for key in ("w1", "w2", "w3"):
        compressed.attach_expert_lora(
            tensor_name=key, adapter_name="host", rank=2, alpha=2.0
        )
        with torch.no_grad():
            compressed.expert_lora[key].lora_b.normal_(0, 0.02)
    direct = copy.deepcopy(compressed)
    host.experts = compressed
    logical = torch.tensor([[0, 4], [1, 5], [4, 0], [5, 1]])
    scores = torch.rand(4, 2) + 0.1
    left = torch.randn(4, 16, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    expected = direct.run_direct_routed(left, scores, logical)
    actual = host._run_selected_experts(right, scores, logical, drop_slots=False)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(right.grad, left.grad, rtol=3e-4, atol=3e-5)
    for key in ("w1", "w2", "w3"):
        for factor in ("lora_a", "lora_b"):
            torch.testing.assert_close(
                getattr(compressed.expert_lora[key], factor).grad,
                getattr(direct.expert_lora[key], factor).grad,
                rtol=3e-4,
                atol=3e-5,
            )
