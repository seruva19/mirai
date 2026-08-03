"""Behavioral contracts for attention-routed Mixture-of-Depths."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams, TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.models.attention_backends import attention_backend_status
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.routing.depth import (
    MixtureOfDepthsSpec,
    attention_with_received_scores,
    select_depth_tokens,
)
from mirai.core.training.policies.mixture_of_depths import (
    validate_mixture_of_depths_config,
)
from mirai.core.training.training_policy import TrainingPolicySet


def _cpu_varlen_attention(
    query,
    key,
    value,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    **_kwargs,
):
    outputs = []
    q_offsets = cu_seqlens_q.detach().cpu().tolist()
    k_offsets = cu_seqlens_k.detach().cpu().tolist()
    for q_start, q_end, k_start, k_end in zip(
        q_offsets[:-1],
        q_offsets[1:],
        k_offsets[:-1],
        k_offsets[1:],
        strict=True,
    ):
        q = query[q_start:q_end].transpose(0, 1)
        k = key[k_start:k_end].transpose(0, 1)
        v = value[k_start:k_end].transpose(0, 1)
        outputs.append(F.scaled_dot_product_attention(q, k, v).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _cuda_varlen_attention_available() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    return any(
        attention_backend_status(name, device=device, varlen=True).available
        for name in ("flash4", "flash3")
    )


def _manual_attention(query, key, value, mask=None):
    q = query.transpose(1, 2).float()
    k = key.transpose(1, 2).float()
    v = value.transpose(1, 2).float()
    logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.matmul(probabilities, v).transpose(1, 2).to(value.dtype)
    valid = (
        mask[:, 0, 0]
        if mask is not None
        else torch.ones(query.shape[:2], dtype=torch.bool, device=query.device)
    )
    received = (
        probabilities * valid[:, None, :, None].to(probabilities.dtype)
    ).sum(dim=(1, 2))
    received = received / (
        valid.sum(dim=1, keepdim=True).clamp_min(1) * query.shape[2]
    )
    received = received.masked_fill(~valid, 0.0)
    return output, received


def _enabled_config(**options) -> TrainingConfig:
    config = TrainingConfig()
    config.model.type = "lingbot-video"
    config.training.policy_options = {
        "mixture_of_depths": {"enabled": True, **options}
    }
    return config


def _pipeline(*, checkpointing: str = "off") -> LingBotVideoPipeline:
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
                num_layers=2,
                attention_heads=2,
                patch_size=1,
                moe_balance_mode="off",
            ),
        )
    )
    pipeline.set_adapter_config(
        AdapterConfig(
            type="lora",
            target_preset="attn_routed_experts",
            rank=2,
            alpha=2.0,
        )
    )
    pipeline.configure_mixture_of_depths(
        MixtureOfDepthsSpec(
            capacity_fraction=0.5,
            first_layer=1,
            layer_stride=2,
            attention_query_chunk_size=3,
        )
    )
    pipeline.set_gradient_checkpointing(checkpointing)
    pipeline.train()
    return pipeline


def test_received_attention_matches_equation_four_output_and_gradients() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(2, 5, 2, 3, generator=generator, requires_grad=True)
    key = torch.randn(2, 5, 2, 3, generator=generator, requires_grad=True)
    value = torch.randn(2, 5, 2, 3, generator=generator, requires_grad=True)
    mask = torch.tensor(
        [
            [[[[True, True, True, True, False]]]],
            [[[[True, True, True, True, True]]]],
        ]
    ).reshape(2, 1, 1, 5)
    output, received = attention_with_received_scores(
        query,
        key,
        value,
        attention_mask=mask,
        query_chunk_size=2,
    )
    expected_output, expected_received = _manual_attention(
        query, key, value, mask
    )
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(received, expected_received)

    gradients = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_gradients = torch.autograd.grad(
        expected_output.square().sum(), (query, key, value)
    )
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_packed_received_attention_is_sample_isolated() -> None:
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(1, 7, 2, 3, generator=generator)
    key = torch.randn(1, 7, 2, 3, generator=generator)
    value = torch.randn(1, 7, 2, 3, generator=generator)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    output, scores = attention_with_received_scores(
        query,
        key,
        value,
        cu_seqlens=cu,
        query_chunk_size=2,
    )
    expected = [
        attention_with_received_scores(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            query_chunk_size=2,
        )
        for start, end in ((0, 3), (3, 7))
    ]
    torch.testing.assert_close(output, torch.cat([item[0] for item in expected], 1))
    torch.testing.assert_close(scores, torch.cat([item[1] for item in expected], 1))


def test_capacity_selection_is_exact_per_sample_and_keeps_context() -> None:
    scores = torch.tensor([[0.2, 0.9, 0.9, 0.1, 0.8, 0.4, 0.7, 0.0]])
    eligible = torch.tensor([[True, True, True, False, True, True, True, False]])
    valid = torch.tensor([[True, True, True, True, True, True, True, False]])
    selection = select_depth_tokens(
        scores,
        eligible_mask=eligible,
        valid_mask=valid,
        cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
        capacity_fraction=0.5,
    )
    assert selection.selected_visual_tokens == (1, 1)
    assert selection.processed_tokens == (2, 1)
    # Stable ties choose token 1, while non-eligible valid token 3 is retained.
    assert selection.flat_indices.tolist() == [1, 3, 4]
    assert selection.cu_seqlens.tolist() == [0, 2, 3]


def test_policy_is_default_off_validated_and_checkpointed() -> None:
    register_builtin_components()
    assert "mixture_of_depths" not in TrainingPolicySet.from_config(
        TrainingConfig()
    ).active_names
    policies = TrainingPolicySet.from_config(
        _enabled_config(
            capacity_fraction=0.25,
            first_layer=1,
            layer_stride=2,
            attention_query_chunk_size=16,
        )
    )
    assert "mixture_of_depths" in policies.active_names
    assert policies.state_dict()["state"] == {}
    metadata = policies.checkpoint_metadata()["policies"]["mixture_of_depths"]
    assert metadata["capacity_fraction"] == 0.25
    invalid = _enabled_config(first_layer=0)
    assert any("first_layer" in item for item in validate_mixture_of_depths_config(invalid))


@pytest.mark.parametrize("checkpointing", ["off", "standard", "aggressive"])
def test_lingbot_forward_backward_and_capacity_diagnostics(checkpointing: str) -> None:
    pipeline = _pipeline(checkpointing=checkpointing)
    prediction = pipeline.forward(
        torch.randn(1, 2, 2, 4, 4),
        torch.tensor([0.5]),
        {"lingbot": torch.randn(1, 3, 16)},
    )
    loss = prediction.float().square().mean()
    loss.backward()
    assert bool(torch.isfinite(loss))
    trainable = [parameter for parameter in pipeline.parameters() if parameter.requires_grad]
    assert trainable and all(parameter.grad is not None for parameter in trainable)
    diagnostics = pipeline.get_training_diagnostics()
    assert diagnostics["mixture_of_depths_capacity_fraction"] == 0.5
    assert diagnostics["mixture_of_depths_routed_layer_count"] == 1.0
    assert diagnostics["mixture_of_depths_selected_visual_tokens"] == 16.0


def test_lingbot_packed_batch_keeps_per_sample_capacity(monkeypatch) -> None:
    monkeypatch.setattr(
        "mirai.vendors.lingbot_video.transformer_lingbot_video."
        "dispatch_varlen_attention",
        _cpu_varlen_attention,
    )
    pipeline = _pipeline()
    prediction = pipeline.forward(
        torch.randn(2, 2, 2, 4, 4),
        torch.tensor([0.3, 0.7]),
        {"lingbot": torch.randn(2, 3, 16)},
    )
    loss = prediction.float().square().mean()
    loss.backward()
    assert prediction.shape == (2, 2, 2, 4, 4)
    diagnostics = pipeline.get_training_diagnostics()
    assert diagnostics["mixture_of_depths_selected_visual_tokens"] == 32.0
    selection = pipeline.transformer.blocks[1]._mirai_last_depth_selection
    assert selection.selected_visual_tokens == (16, 16)
    assert selection.cu_seqlens.tolist() == [0, 19, 38]


def test_lingbot_evaluation_reuses_the_deterministic_depth_route() -> None:
    pipeline = _pipeline()
    pipeline.eval()
    latents = torch.randn(1, 2, 2, 4, 4)
    timestep = torch.tensor([0.5])
    conditioning = {"lingbot": torch.randn(1, 3, 16)}
    with torch.no_grad():
        first = pipeline.forward(latents, timestep, conditioning)
        first_selection = pipeline.transformer.blocks[1]._mirai_last_depth_selection
        second = pipeline.forward(latents, timestep, conditioning)
        second_selection = pipeline.transformer.blocks[1]._mirai_last_depth_selection
    torch.testing.assert_close(first, second)
    assert first_selection.flat_indices.tolist() == second_selection.flat_indices.tolist()


@pytest.mark.skipif(
    not _cuda_varlen_attention_available(),
    reason="CUDA with FlashAttention 3 or 4 varlen support is required",
)
def test_lingbot_cuda_bf16_packed_update() -> None:
    pipeline = _pipeline(checkpointing="aggressive").to(
        device="cuda", dtype=torch.bfloat16
    )
    trainable = [parameter for parameter in pipeline.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    before = [parameter.detach().clone() for parameter in trainable]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = pipeline.forward(
            torch.randn(2, 2, 2, 4, 4, device="cuda", dtype=torch.bfloat16),
            torch.tensor([0.3, 0.7], device="cuda"),
            {
                "lingbot": torch.randn(
                    2, 3, 16, device="cuda", dtype=torch.bfloat16
                )
            },
        )
    loss = prediction.float().square().mean()
    loss.backward()
    optimizer.step()
    assert bool(torch.isfinite(loss))
    assert all(parameter.grad is not None for parameter in trainable)
    assert any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, trainable, strict=True)
    )
    selection = pipeline.transformer.blocks[1]._mirai_last_depth_selection
    assert selection.selected_visual_tokens == (16, 16)


def test_spec_rejects_adjacent_or_first_block_routing() -> None:
    with pytest.raises(ValueError, match="first_layer"):
        MixtureOfDepthsSpec(first_layer=0).validate()
    with pytest.raises(ValueError, match="layer_stride"):
        MixtureOfDepthsSpec(layer_stride=1).validate()


__all__ = []
