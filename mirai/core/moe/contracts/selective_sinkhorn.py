"""Behavioral contracts for Selective Sinkhorn Routing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams, TrainingConfig
from mirai.core.builtins import register_builtin_components
from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
from mirai.core.moe.routing.selective_sinkhorn import (
    SelectiveSinkhornController,
    SelectiveSinkhornSpec,
    selective_sinkhorn_transport,
    transport_topk_routes,
)
from mirai.core.training.policies.selective_sinkhorn import (
    validate_selective_sinkhorn_config,
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


def _naive_transport(
    logits: torch.Tensor,
    *,
    cost_mode: str,
    entropy_regularization: float,
    iterations: int,
) -> torch.Tensor:
    cost = logits if cost_mode == "linear" else logits.softmax(dim=-1)
    kernel = torch.exp(cost / float(entropy_regularization))
    tokens, experts = logits.shape
    u = torch.ones(tokens, dtype=logits.dtype, device=logits.device)
    target_columns = torch.full(
        (experts,),
        float(tokens) / float(experts),
        dtype=logits.dtype,
        device=logits.device,
    )
    for _ in range(iterations):
        v = target_columns / (kernel.transpose(0, 1) @ u)
        u = torch.ones_like(u) / (kernel @ v)
    return u[:, None] * kernel * v[None, :]


def _enabled_config(**options) -> TrainingConfig:
    config = TrainingConfig()
    config.model.type = "lingbot-video"
    config.model.params.moe_balance_mode = "off"
    config.model.params.moe_router_z_loss_weight = 0.0
    config.training.policy_options = {
        "selective_sinkhorn": {
            "enabled": True,
            "probability": 0.001,
            "cost_mode": "softmax",
            "entropy_regularization": 0.05,
            "max_iterations": 100,
            "tolerance": 1e-4,
            "noise_scale": 0.0,
            **options,
        }
    }
    return config


def _pipeline(*, checkpointing: str = "off") -> tuple[
    LingBotVideoPipeline, SelectiveSinkhornController
]:
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
    controller = SelectiveSinkhornController(
        SelectiveSinkhornSpec(
            probability=1.0,
            cost_mode="softmax",
            entropy_regularization=0.05,
            max_iterations=100,
            tolerance=1e-4,
            seed=37,
        )
    )
    pipeline.configure_training_policy("selective_sinkhorn", controller)
    controller.bind_batch(global_batch_index=11, training=True)
    pipeline.set_gradient_checkpointing(checkpointing)
    pipeline.train()
    return pipeline, controller


@pytest.mark.parametrize("cost_mode", ["linear", "softmax"])
def test_log_domain_solver_matches_paper_scaling_reference(cost_mode: str) -> None:
    logits = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [0.7, 0.4, -0.1],
            [-0.4, 0.2, 0.6],
            [0.9, -0.7, 0.0],
        ],
        dtype=torch.float32,
    )
    actual = selective_sinkhorn_transport(
        logits,
        cost_mode=cost_mode,
        entropy_regularization=0.5,
        max_iterations=40,
        tolerance=1e-12,
    ).plan
    expected = _naive_transport(
        logits,
        cost_mode=cost_mode,
        entropy_regularization=0.5,
        iterations=40,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_transport_has_paper_marginals_and_topk_transport_weights() -> None:
    result = selective_sinkhorn_transport(
        torch.randn(17, 5, generator=torch.Generator().manual_seed(3)),
        cost_mode="softmax",
        entropy_regularization=0.05,
        max_iterations=100,
        tolerance=1e-4,
    )
    torch.testing.assert_close(
        result.plan.sum(dim=1), torch.ones(17), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        result.plan.sum(dim=0),
        torch.full((5,), 17.0 / 5.0),
        rtol=1e-4,
        atol=1e-4,
    )
    routes = transport_topk_routes(result, top_k=2, route_scale=1.7)
    expected_values, expected_indices = torch.topk(
        result.plan, k=2, dim=-1, sorted=False
    )
    expected_values = (
        expected_values / expected_values.sum(dim=-1, keepdim=True) * 1.7
    )
    assert torch.equal(routes.top_indices, expected_indices)
    torch.testing.assert_close(routes.top_weights, expected_values)


def test_controller_is_replayable_masks_padding_and_detaches_ot_router_path() -> None:
    controller = SelectiveSinkhornController(
        SelectiveSinkhornSpec(
            probability=1.0,
            noise_scale=0.3,
            seed=19,
        )
    )
    logits = torch.randn(8, 4, requires_grad=True)
    native_indices = logits.sigmoid().topk(2, dim=-1).indices
    native_weights = logits.sigmoid().gather(1, native_indices)
    valid = torch.tensor([True, True, True, True, True, True, False, False])
    controller.bind_batch(global_batch_index=7, training=True)
    first = controller.select(
        "blocks.0.router",
        logits,
        native_indices,
        native_weights,
        valid_token_mask=valid,
        route_scale=1.0,
        training=True,
    )
    controller.bind_batch(global_batch_index=7, training=True)
    second = controller.select(
        "blocks.0.router",
        logits,
        native_indices,
        native_weights,
        valid_token_mask=valid,
        route_scale=1.0,
        training=True,
    )
    assert first is not None and second is not None
    assert torch.equal(first.top_indices, second.top_indices)
    torch.testing.assert_close(first.top_weights, second.top_weights, rtol=0, atol=0)
    assert torch.equal(first.top_indices[~valid], native_indices[~valid])
    torch.testing.assert_close(first.top_weights[~valid], native_weights[~valid])
    first.top_weights.sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[valid]) == 0
    assert torch.count_nonzero(logits.grad[~valid]) > 0


def test_eval_and_disabled_policy_preserve_native_routing() -> None:
    register_builtin_components()
    assert "selective_sinkhorn" not in TrainingPolicySet.from_config(
        TrainingConfig()
    ).active_names
    pipeline, controller = _pipeline()
    router = pipeline._moe_router_modules[0]
    tokens = torch.randn(12, 16)
    pipeline.eval()
    controller.bind_batch(global_batch_index=5, training=False)
    with_extension = router(tokens)[:2]
    router.set_route_selection_extension(
        layer_name="blocks.0.ffn.router", selector=None
    )
    reference = router(tokens)[:2]
    assert torch.equal(with_extension[0], reference[0])
    torch.testing.assert_close(with_extension[1], reference[1], rtol=0, atol=0)


def test_policy_is_stateless_and_records_exact_configuration() -> None:
    register_builtin_components()
    policies = TrainingPolicySet.from_config(_enabled_config(seed=71))
    assert "selective_sinkhorn" in policies.active_names
    assert policies.state_dict()["state"] == {}
    metadata = policies.checkpoint_metadata()["policies"]["selective_sinkhorn"]
    assert metadata["seed"] == 71
    assert metadata["cost_mode"] == "softmax"
    assert metadata["branch_rng"].startswith("blake2b")


def test_config_rejects_unknown_and_non_paper_balancing_combinations() -> None:
    unknown = _enabled_config(mystery=1)
    assert any("unknown option 'mystery'" in error for error in validate_selective_sinkhorn_config(unknown))
    aux = _enabled_config()
    aux.model.params.moe_balance_mode = "aux_loss"
    assert any("moe_balance_mode='off'" in error for error in validate_selective_sinkhorn_config(aux))
    conflict = _enabled_config()
    conflict.training.policy_options["simbal"] = {"enabled": True}
    assert any("simbal" in error for error in validate_selective_sinkhorn_config(conflict))


@pytest.mark.parametrize("checkpointing", ["off", "standard", "aggressive"])
def test_lingbot_forward_backward_all_checkpoint_modes(
    checkpointing: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mirai.vendors.lingbot_video.transformer_lingbot_video."
        "dispatch_varlen_attention",
        _cpu_varlen_attention,
    )
    pipeline, controller = _pipeline(checkpointing=checkpointing)
    torch.manual_seed(41)
    prediction = pipeline.forward(
        torch.randn(1, 2, 2, 4, 4),
        torch.tensor([0.5]),
        {"lingbot": torch.randn(1, 3, 16)},
    )
    loss = prediction.float().square().mean()
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in pipeline.named_parameters()
        if "lora_" in name
    )
    diagnostics = controller.diagnostics()
    assert diagnostics["moe_selective_sinkhorn_applications"] > 0
    assert diagnostics["moe_selective_sinkhorn_last_iterations"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_pipeline_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mirai.vendors.lingbot_video.transformer_lingbot_video."
        "dispatch_varlen_attention",
        _cpu_varlen_attention,
    )
    with acquire_gpu_lease(
        lock_path=resolve_lease_lock_path(Path.cwd()), timeout_seconds=0.0
    ):
        pipeline, controller = _pipeline(checkpointing="aggressive")
        pipeline = pipeline.to(device="cuda:0", dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = pipeline.forward(
                torch.randn(1, 2, 2, 4, 4, device="cuda", dtype=torch.bfloat16),
                torch.tensor([0.5], device="cuda", dtype=torch.bfloat16),
                {
                    "lingbot": torch.randn(
                        1, 3, 16, device="cuda", dtype=torch.bfloat16
                    )
                },
            )
        loss = prediction.float().square().mean()
        loss.backward()
        assert bool(torch.isfinite(loss))
        assert controller.diagnostics()["moe_selective_sinkhorn_applications"] > 0
