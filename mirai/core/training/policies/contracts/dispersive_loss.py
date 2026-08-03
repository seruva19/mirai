"""Behavioral contracts for parameter-free Dispersive Loss."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams, TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.models.lingbot_video.dispersive_loss import (
    LingBotDispersiveLossRuntime,
)
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.training.policies.dispersive_loss import (
    DispersiveLossController,
    DispersiveLossSpec,
    dispersive_l2_loss,
    dispersive_l2_loss_reference,
    validate_dispersive_loss_config,
)
from mirai.core.training.runtime.gpu_lease import (
    acquire_gpu_lease,
    resolve_lease_lock_path,
)
from mirai.core.training.training_policy import TrainingPolicySet


def _cpu_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """CPU oracle for the orthogonal packed-attention dependency."""

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


def _enabled_config(**options) -> TrainingConfig:
    config = TrainingConfig()
    config.model.type = "lingbot-video"
    config.training.batch_size = 2
    config.training.policy_options = {
        "dispersive_loss": {
            "enabled": True,
            "weight": 0.5,
            "temperature": 0.5,
            "layer_fraction": 0.25,
            "chunk_features": 7,
            **options,
        }
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
    controller = DispersiveLossController(
        DispersiveLossSpec(
            weight=0.5,
            temperature=0.5,
            layer_fraction=0.5,
            chunk_features=19,
        )
    )
    pipeline.configure_dispersive_loss(controller)
    pipeline.set_gradient_checkpointing(checkpointing)
    pipeline.train()
    return pipeline


def _forward(
    pipeline: LingBotVideoPipeline,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    torch.manual_seed(41)
    return pipeline.forward(
        torch.randn(2, 2, 2, 4, 4, device=device, dtype=dtype),
        torch.tensor([0.25, 0.75], device=device, dtype=dtype),
        {"lingbot": torch.randn(2, 3, 16, device=device, dtype=dtype)},
    )


def test_formula_matches_official_full_pair_matrix() -> None:
    values = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 1.5], [2.5, 4.0]],
            [[-1.0, 0.5], [2.0, 2.0]],
        ],
        requires_grad=True,
    )
    temperature = 0.7
    flat = values.reshape(3, -1)
    pairs = torch.nn.functional.pdist(flat).square() / float(flat.shape[1])
    official_full = torch.cat((pairs, pairs, torch.zeros(3)))
    expected = torch.log(torch.exp(-official_full / temperature).mean())
    actual = dispersive_l2_loss(values, temperature=temperature, chunk_features=3)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_chunked_output_and_gradients_match_autograd_reference() -> None:
    torch.manual_seed(7)
    actual_input = torch.randn(4, 5, 6, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    actual = dispersive_l2_loss(
        actual_input,
        temperature=0.8,
        chunk_features=11,
    )
    reference = dispersive_l2_loss_reference(
        reference_input,
        temperature=0.8,
    )
    actual.backward()
    reference.backward()
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        rtol=2e-5,
        atol=2e-6,
    )


def test_spec_resolves_human_block_fraction() -> None:
    assert DispersiveLossSpec(layer_fraction=0.25).resolve_layer_index(12) == 2
    assert DispersiveLossSpec(layer_fraction=1.0).resolve_layer_index(12) == 11
    with pytest.raises(ValueError, match="layer_fraction"):
        DispersiveLossSpec(layer_fraction=0.0).validate()


def test_lingbot_packed_layout_regularizes_video_tokens_only() -> None:
    controller = DispersiveLossController(
        DispersiveLossSpec(layer_fraction=1.0, chunk_features=4)
    )
    runtime = LingBotDispersiveLossRuntime(controller)
    runtime.bind_depth(1)
    runtime.begin_forward(
        batch_size=2,
        video_tokens=3,
        text_lengths=(2, 1),
        packed_batch=True,
    )
    first_video = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    first_text = torch.full((1, 2, 2), 1000.0)
    second_video = torch.arange(6, 12, dtype=torch.float32).reshape(1, 3, 2)
    second_text = torch.full((1, 1, 2), -1000.0)
    packed = torch.cat(
        (first_video, first_text, second_video, second_text),
        dim=1,
    ).requires_grad_(True)
    actual = runtime.loss_for_hidden_states(0, packed) / controller.spec.weight
    expected = dispersive_l2_loss_reference(
        torch.cat((first_video, second_video), dim=0),
        temperature=controller.spec.temperature,
    )
    torch.testing.assert_close(actual, expected)


def test_default_is_absent_and_enabled_policy_is_stateless() -> None:
    register_builtin_components()
    assert "dispersive_loss" not in TrainingPolicySet.from_config(
        TrainingConfig()
    ).active_names
    enabled = TrainingPolicySet.from_config(_enabled_config())
    assert "dispersive_loss" in enabled.active_names
    assert enabled.state_dict()["state"] == {}
    metadata = enabled.checkpoint_metadata()["policies"]["dispersive_loss"]
    assert metadata["weight"] == 0.5
    assert metadata["layer_index"] is None


def test_config_rejects_degenerate_batch_and_unknown_options() -> None:
    register_builtin_components()
    batch_one = _enabled_config()
    batch_one.training.batch_size = 1
    assert any(
        "batch_size" in error
        for error in validate_dispersive_loss_config(batch_one)
    )
    unknown = _enabled_config(mystery=True)
    assert any(
        "unknown option 'mystery'" in error
        for error in validate_dispersive_loss_config(unknown)
    )


@pytest.mark.parametrize("checkpointing", ["off", "standard", "aggressive"])
def test_lingbot_forward_surfaces_trainable_loss(
    checkpointing: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mirai.vendors.lingbot_video.transformer_lingbot_video."
        "dispatch_varlen_attention",
        _cpu_varlen_attention,
    )
    pipeline = _pipeline(checkpointing=checkpointing)
    prediction = _forward(pipeline)
    auxiliary = pipeline.get_training_auxiliary_losses()
    assert set(auxiliary) >= {"dispersive_loss"}
    loss = prediction.float().square().mean() + auxiliary["dispersive_loss"]
    assert bool(torch.isfinite(loss)) and loss.requires_grad
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in pipeline.named_parameters()
        if "lora_" in name and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    diagnostics = pipeline.get_training_diagnostics()
    assert diagnostics["dispersive_loss_layer_index"] == 0
    assert math.isfinite(float(diagnostics["dispersive_loss_raw"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_pipeline_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise capture, chunked loss, backward, and diagnostics under lease."""

    monkeypatch.setattr(
        "mirai.vendors.lingbot_video.transformer_lingbot_video."
        "dispatch_varlen_attention",
        _cpu_varlen_attention,
    )
    with acquire_gpu_lease(
        lock_path=resolve_lease_lock_path(Path.cwd()),
        timeout_seconds=0.0,
    ):
        device = torch.device("cuda:0")
        pipeline = _pipeline(checkpointing="aggressive").to(
            device=device,
            dtype=torch.bfloat16,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = _forward(
                pipeline,
                device=device,
                dtype=torch.bfloat16,
            )
        auxiliary = pipeline.get_training_auxiliary_losses()
        loss = prediction.float().square().mean() + auxiliary["dispersive_loss"]
        loss.backward()
        assert bool(torch.isfinite(loss))
        assert any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            for name, parameter in pipeline.named_parameters()
            if "lora_" in name
        )
        diagnostics = pipeline.get_training_diagnostics()
        assert diagnostics["dispersive_loss_layer_index"] == 0
        assert math.isfinite(float(diagnostics["dispersive_loss_raw"]))
