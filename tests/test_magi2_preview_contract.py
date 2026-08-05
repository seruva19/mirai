from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.nn.utils import parametrize

from mirai.config.loader import load_config
from mirai.core.models.magi2_preview.grouped_moe import (
    _CONSUMED_POLICY_FIELDS,
    Magi2GroupedMoEBackend,
    Magi2GroupedMoEPlan,
    Magi2GroupedMoEPolicyError,
    resolve_magi2_moe_execution,
    validate_grouped_moe_backend_support,
)
from mirai.core.moe.runtime.gemm import grouped_mm_op
from mirai.core.models.magi2_preview.pipeline import LowRankWeight
from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy


def test_magi2_provider_is_native_sparse_moe() -> None:
    provider = get_model_family_provider("magi2-preview")
    assert provider is not None
    assert provider.is_native_model()
    assert provider.is_sparse_moe_model()
    assert provider.supports_batched_cfg_inference()
    assert provider.require_pipeline_type().__name__ == "Magi2PreviewPipeline"
    assert provider.validate_family_params({"audio_tokens": 1}) == []
    assert provider.validate_family_params({"unknown": True})


def test_magi2_offload_example_loads() -> None:
    config = load_config("configs/magi2_preview/train_offload.toml")
    assert config.model.type == "magi2-preview"
    assert config.training.blocks_to_swap == 40
    assert config.memory.weight_residency_strategy == "block_swap"
    assert config.adapter.target_preset == "attn_router"


def test_magi2_low_rank_weight_preserves_default_and_gradients() -> None:
    base = torch.randn(2, 3, 5)
    adapter = LowRankWeight(tuple(base.shape), rank=2, alpha=2.0)
    reference = adapter(base)
    assert torch.equal(reference, base)
    adapter.lora_b.data.normal_()
    loss = adapter(base).square().mean()
    loss.backward()
    assert adapter.lora_a.grad is not None
    assert adapter.lora_b.grad is not None
    assert torch.isfinite(adapter.lora_a.grad).all()
    assert torch.isfinite(adapter.lora_b.grad).all()


def _build_reduced_moe(
    *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.nn.Module, LowRankWeight]:
    """Reduced-shape vendored MoE layer with a router LoRA, as attn_router uses."""
    from mirai.vendors.magi2_preview.model.magi2_preview import (
        CoreMultiHeadMoE,
        CoreMultiHeadMoEConfig,
    )

    torch.manual_seed(0)
    module = CoreMultiHeadMoE(
        CoreMultiHeadMoEConfig(
            hidden_size=16,
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=12,
            num_layers=1,
            params_dtype=dtype,
            score_func="sigmoid",
            route_norm=True,
            route_scale=4.9,
        )
    )
    with torch.no_grad():
        for tensor in (module.gate, module.W_gate, module.W_up, module.W_down):
            tensor.normal_(std=0.1)
        module.router.expert_bias.normal_(std=0.05)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    parametrize.register_parametrization(
        module, "gate", LowRankWeight(tuple(module.gate.shape), rank=2, alpha=2.0)
    )
    module.parametrizations["gate"].original.requires_grad_(False)
    adapter = module.parametrizations["gate"][0]
    with torch.no_grad():
        adapter.lora_b.normal_(std=0.1)
    module.to(device=device)
    return module, adapter


def _run_reduced_moe(
    *, backend: str | None, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    module, adapter = _build_reduced_moe(device=device, dtype=dtype)
    if backend is not None:
        module._mirai_moe_kernel_backend = Magi2GroupedMoEBackend(
            Magi2GroupedMoEPlan(forward_backend=backend, dx_backend=backend)
        )
    torch.manual_seed(7)
    hidden = torch.randn(5, 16, device=device, dtype=dtype, requires_grad=True)
    output = module._forward_impl(hidden)
    loss = output.float().square().mean()
    loss.backward()
    return (
        output.detach(),
        loss.detach(),
        hidden.grad.detach().clone(),
        adapter.lora_a.grad.detach().clone(),
        adapter.lora_b.grad.detach().clone(),
    )


def test_magi2_grouped_moe_matches_reference_loop_on_cpu() -> None:
    device = torch.device("cpu")
    reference = _run_reduced_moe(backend=None, device=device, dtype=torch.float32)
    grouped = _run_reduced_moe(backend="bmm", device=device, dtype=torch.float32)
    for expected, actual in zip(reference, grouped):
        assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-6)
    assert reference[3].abs().max() > 0.0
    assert reference[4].abs().max() > 0.0


def test_magi2_grouped_moe_matches_reference_loop_in_bf16_on_cpu() -> None:
    """The fp32 clamp/activation ladder must hold for BF16 expert weights."""
    device = torch.device("cpu")
    reference = _run_reduced_moe(backend=None, device=device, dtype=torch.bfloat16)
    grouped = _run_reduced_moe(backend="bmm", device=device, dtype=torch.bfloat16)
    for expected, actual in zip(reference, grouped):
        assert torch.allclose(
            expected.float(), actual.float(), rtol=2e-2, atol=2e-3
        )
    assert reference[3].abs().max() > 0.0
    assert reference[4].abs().max() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_magi2_grouped_moe_matches_reference_loop_on_cuda() -> None:
    device = torch.device("cuda")
    reference = _run_reduced_moe(backend=None, device=device, dtype=torch.bfloat16)
    grouped = _run_reduced_moe(backend="auto", device=device, dtype=torch.bfloat16)
    for expected, actual in zip(reference, grouped):
        assert torch.allclose(
            expected.float(), actual.float(), rtol=2e-2, atol=2e-3
        )


def test_magi2_grouped_moe_rejects_trainable_expert_weights() -> None:
    module, _adapter = _build_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32
    )
    module.W_gate.requires_grad_(True)
    module._mirai_moe_kernel_backend = Magi2GroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    )
    with pytest.raises(RuntimeError, match="frozen expert weights"):
        module._forward_impl(torch.randn(4, 16))


def test_magi2_moe_policy_defaults_keep_the_reference_path() -> None:
    assert resolve_magi2_moe_execution(MoEOptimizationPolicy()) is None
    assert resolve_magi2_moe_execution(MoEOptimizationPolicy(kernel_backend="torch")) is None
    plan = resolve_magi2_moe_execution(
        MoEOptimizationPolicy(kernel_backend="grouped", moe_gemm_backend="bmm")
    )
    assert plan == Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    role_plan = resolve_magi2_moe_execution(
        MoEOptimizationPolicy(
            kernel_backend="grouped",
            moe_gemm_backend="bmm",
            moe_gemm_backend_dx="torch_grouped",
        )
    )
    assert role_plan.dx_backend == "torch_grouped"


def test_magi2_moe_policy_rejects_unsupported_fields() -> None:
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(MoEOptimizationPolicy(kernel_backend="megablocks"))
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(
            MoEOptimizationPolicy(
                kernel_backend="grouped", expert_weight_access="full_dequant"
            )
        )
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(
            MoEOptimizationPolicy(kernel_backend="grouped", moe_dispatch="triton")
        )
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(
            MoEOptimizationPolicy(
                kernel_backend="grouped", moe_gemm_backend_dw="torch_grouped"
            )
        )
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(
            MoEOptimizationPolicy(kernel_backend="grouped", moe_gemm_backend="persistent")
        )


# One non-default value per policy field MAGI-2 does not consume. Fields with a
# cross-field constraint carry the co-required keys so the policy itself builds.
_NON_DEFAULT_POLICY_VALUES: dict[str, dict[str, object]] = {
    "expert_weight_access": {"expert_weight_access": "full_dequant"},
    "expert_dequant_chunk_size": {"expert_dequant_chunk_size": 8},
    "expert_device_cache_gib": {"expert_device_cache_gib": 1.0},
    "device_residency_budget_gib": {"device_residency_budget_gib": 1.0},
    "quantize_experts_on_load": {"quantize_experts_on_load": True},
    "router_quantization": {"router_quantization": "int8_per_channel"},
    "router_quantization_calibration_path": {
        "router_quantization": "int8_per_channel",
        "router_quantization_calibration_path": "calibration.pt",
    },
    "packed_state_preload": {"packed_state_preload": "ram"},
    "packed_stream_cache_gib": {
        "packed_state_preload": "off",
        "packed_stream_cache_gib": 1.0,
    },
    "packed_stream_backend": {
        "packed_state_preload": "off",
        "packed_stream_backend": "gds",
    },
    "packed_stream_prefetch_depth": {"packed_stream_prefetch_depth": 2},
    "moe_dispatch": {"moe_dispatch": "legacy"},
    "moe_dispatch_preprocess": {"moe_dispatch_preprocess": "device"},
    "moe_gemm_backend_dw": {"moe_gemm_backend_dw": "bmm"},
    "moe_batched_dequant": {"moe_batched_dequant": False},
    "moe_pair_dequant": {"moe_pair_dequant": False},
    "moe_batched_gather": {"moe_batched_gather": True},
    "packed_shard_size_mb": {"packed_shard_size_mb": 512},
    "int8_workspace_mb": {"int8_workspace_mb": 64},
}


def test_magi2_policy_rejection_covers_every_unconsumed_field() -> None:
    declared = {field.name for field in dataclasses.fields(MoEOptimizationPolicy)}
    assert set(_NON_DEFAULT_POLICY_VALUES) | set(_CONSUMED_POLICY_FIELDS) == declared


@pytest.mark.parametrize("field_name", sorted(_NON_DEFAULT_POLICY_VALUES))
def test_magi2_moe_policy_rejects_every_unconsumed_field(field_name: str) -> None:
    policy = MoEOptimizationPolicy(
        kernel_backend="grouped", **_NON_DEFAULT_POLICY_VALUES[field_name]
    )
    with pytest.raises(Magi2GroupedMoEPolicyError):
        resolve_magi2_moe_execution(policy)


def test_magi2_grouped_backend_support_is_validated_before_execution() -> None:
    cpu = torch.device("cpu")
    validate_grouped_moe_backend_support(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm"), device=cpu
    )
    plan = Magi2GroupedMoEPlan(
        forward_backend="torch_grouped", dx_backend="torch_grouped"
    )
    if grouped_mm_op() is None:
        with pytest.raises(Magi2GroupedMoEPolicyError, match="torch build"):
            validate_grouped_moe_backend_support(plan, device=cpu)
    else:
        # The device architecture gate needs the execution device, which weight
        # residency assigns after the policy is configured.
        validate_grouped_moe_backend_support(plan, device=cpu)
    if torch.cuda.is_available():
        from mirai.core.moe.runtime.gemm import probe_backend

        probe = probe_backend("torch_grouped", device=torch.device("cuda"))
        if probe.available:
            validate_grouped_moe_backend_support(
                plan, device=torch.device("cuda")
            )
        else:
            with pytest.raises(Magi2GroupedMoEPolicyError):
                validate_grouped_moe_backend_support(
                    plan, device=torch.device("cuda")
                )


def test_magi2_provider_declares_moe_kernel_backend_capability() -> None:
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    capabilities = pipeline.get_memory_feature_capabilities()
    assert capabilities.moe_kernel_backend
    assert not capabilities.expert_weight_access_policy


def test_magi2_pipeline_attaches_and_detaches_the_grouped_seam() -> None:
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

    module, _adapter = _build_reduced_moe(
        device=torch.device("cpu"), dtype=torch.float32
    )
    container = torch.nn.Module()
    container.moe_mlp = module
    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = container

    pipeline.configure_moe_optimization_policy(MoEOptimizationPolicy())
    assert module._mirai_moe_kernel_backend is None

    pipeline.configure_moe_optimization_policy(
        MoEOptimizationPolicy(kernel_backend="grouped", moe_gemm_backend="bmm")
    )
    assert isinstance(module._mirai_moe_kernel_backend, Magi2GroupedMoEBackend)
    assert module._mirai_moe_kernel_backend.plan.forward_backend == "bmm"

    pipeline.configure_moe_optimization_policy(MoEOptimizationPolicy())
    assert module._mirai_moe_kernel_backend is None
