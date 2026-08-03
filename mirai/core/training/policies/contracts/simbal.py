"""Behavioral contracts for similarity-preserving router balancing."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams, TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.adaptation.simbal import (
    SimBalController,
    SimBalSpec,
    simbal_router_loss,
)
from mirai.core.training.policies.simbal import validate_simbal_config
from mirai.core.training.runtime.gpu_lease import (
    acquire_gpu_lease,
    resolve_lease_lock_path,
)
from mirai.core.training.training_policy import TrainingPolicySet


def _config(**options) -> TrainingConfig:
    config = TrainingConfig()
    config.model.type = "lingbot-video"
    config.adapter.target_preset = "attn_router_routed_experts"
    config.adapter.train_router = True
    config.training.policy_options = {
        "simbal": {"enabled": True, "weight": 0.1, **options}
    }
    return config


def _pipeline(*, router_target: bool = True) -> LingBotVideoPipeline:
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
    pipeline.configure_simbal(
        SimBalController(SimBalSpec(weight=0.1)),
    )
    pipeline.set_adapter_config(
        AdapterConfig(
            type="lora",
            target_preset=(
                "attn_router_routed_experts" if router_target else "attn_only"
            ),
            rank=2,
            alpha=2.0,
            train_router=True,
        )
    )
    pipeline.train()
    return pipeline


def test_formula_matches_paper_appendix() -> None:
    weight = torch.tensor(
        [[0.5, -1.0, 0.25, 0.0], [1.5, 0.5, -0.5, 1.0]],
        requires_grad=True,
    )
    expected = torch.norm(
        weight @ weight.transpose(0, 1) - torch.eye(2),
        p=1,
    )
    actual = simbal_router_loss(weight)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    actual.backward()
    assert weight.grad is not None and bool(torch.isfinite(weight.grad).all())


def test_orthonormal_router_has_zero_loss_and_invalid_shape_fails() -> None:
    weight = torch.eye(3, 5)
    assert float(simbal_router_loss(weight)) == 0.0
    with pytest.raises(ValueError, match="hidden_size >= num_experts"):
        simbal_router_loss(torch.randn(5, 3))


def test_controller_averages_layers_and_applies_coefficient() -> None:
    controller = SimBalController(SimBalSpec(weight=0.25))
    weights = {"b": torch.eye(2, 3) * 2.0, "a": torch.eye(2, 3)}
    expected = torch.stack([simbal_router_loss(value) for value in weights.values()]).mean()
    actual = controller.loss(weights)
    torch.testing.assert_close(actual, expected * 0.25)
    diagnostics = controller.diagnostics()
    assert diagnostics["moe_simbal_router_count"] == 2
    assert math.isfinite(float(diagnostics["moe_simbal_raw_mean"]))


def test_default_absent_and_enabled_policy_is_stateless() -> None:
    register_builtin_components()
    assert "simbal" not in TrainingPolicySet.from_config(
        TrainingConfig()
    ).active_names
    policies = TrainingPolicySet.from_config(_config())
    assert "simbal" in policies.active_names
    assert policies.state_dict()["state"] == {}
    assert policies.checkpoint_metadata()["policies"]["simbal"] == {
        "weight": 0.1,
        "norm": "entrywise_l1",
    }


def test_config_rejects_unknown_weight_and_frozen_router() -> None:
    register_builtin_components()
    assert any("unknown option" in error for error in validate_simbal_config(_config(extra=1)))
    assert any("finite" in error for error in validate_simbal_config(_config(weight=float("nan"))))
    frozen = _config()
    frozen.adapter.train_router = False
    assert any("train_router=false" in error for error in validate_simbal_config(frozen))


def test_lingbot_effective_router_weight_receives_adapter_gradients() -> None:
    pipeline = _pipeline()
    router = pipeline._moe_router_modules[0]
    original = router.parametrizations.weight.original
    loss = pipeline._simbal_runtime.auxiliary_losses()["moe_simbal"]
    assert loss.requires_grad and bool(torch.isfinite(loss))
    loss.backward()
    assert original.grad is None
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for binding in pipeline._router_adapter_bindings()
        for parameter in binding.adapter.parameters()
    )


def test_lingbot_forward_surfaces_loss_and_diagnostics() -> None:
    pipeline = _pipeline()
    torch.manual_seed(17)
    prediction = pipeline.forward(
        torch.randn(1, 2, 2, 4, 4),
        torch.tensor([0.5]),
        {"lingbot": torch.randn(1, 3, 16)},
    )
    auxiliary = pipeline.get_training_auxiliary_losses()
    assert "moe_simbal" in auxiliary
    (prediction.float().square().mean() + auxiliary["moe_simbal"]).backward()
    diagnostics = pipeline.get_training_diagnostics()
    assert diagnostics["moe_simbal_router_count"] == 2


def test_router_free_preset_fails_binding() -> None:
    with pytest.raises(ValueError, match="router adapter on every"):
        _pipeline(router_target=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_effective_router_backward() -> None:
    with acquire_gpu_lease(
        lock_path=resolve_lease_lock_path(Path.cwd()),
        timeout_seconds=0.0,
    ):
        pipeline = _pipeline().to(device="cuda:0", dtype=torch.bfloat16)
        loss = pipeline._simbal_runtime.auxiliary_losses()["moe_simbal"]
        loss.backward()
        assert loss.dtype == torch.float32 and bool(torch.isfinite(loss))
        assert any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            for binding in pipeline._router_adapter_bindings()
            for parameter in binding.adapter.parameters()
        )
