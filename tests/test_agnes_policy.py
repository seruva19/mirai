"""Fail-closed runtime-policy contracts for Agnes Video."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from mirai.core.models.agnes_video.pipeline import AgnesVideoPipeline
from mirai.core.moe.runtime.specs import MoEOptimizationPolicy


_NON_DEFAULT_POLICY_VALUES = {
    "expert_weight_access": {"expert_weight_access": "disabled"},
    "expert_dequant_chunk_size": {"expert_dequant_chunk_size": 8},
    "expert_device_cache_gib": {"expert_device_cache_gib": 1.0},
    "expert_device_cache_policy": {"expert_device_cache_policy": "routing_frequency"},
    "expert_transfer_profile_path": {"expert_transfer_profile_path": "profile.json"},
    "device_residency_budget_gib": {"device_residency_budget_gib": 1.0},
    "quantize_experts_on_load": {"quantize_experts_on_load": True},
    "router_quantization": {"router_quantization": "int8_per_channel"},
    "router_quantization_calibration_path": {
        "router_quantization": "int8_per_channel",
        "router_quantization_calibration_path": "calibration.pt",
    },
    "kernel_backend": {"kernel_backend": "grouped"},
    "packed_state_preload": {"packed_state_preload": "ram"},
    "packed_stream_cache_gib": {"packed_state_preload": "off", "packed_stream_cache_gib": 1.0},
    "packed_stream_backend": {"packed_state_preload": "off", "packed_stream_backend": "gds"},
    "packed_stream_prefetch_depth": {"packed_stream_prefetch_depth": 1},
    "moe_dispatch": {"moe_dispatch": "legacy"},
    "moe_dispatch_preprocess": {"moe_dispatch_preprocess": "device"},
    "moe_gemm_backend": {"moe_gemm_backend": "bmm"},
    "moe_gemm_backend_forward": {"moe_gemm_backend_forward": "bmm"},
    "moe_gemm_backend_dx": {"moe_gemm_backend_dx": "bmm"},
    "moe_gemm_backend_dw": {"moe_gemm_backend_dw": "bmm"},
    "moe_routed_gemm": {"moe_routed_gemm": "triton"},
    "moe_routed_gemm_tuning": {"moe_routed_gemm_tuning": "online"},
    "moe_routed_gemm_cache_path": {"moe_routed_gemm_cache_path": "tuning.json"},
    "moe_routed_gemm_architecture": {"moe_routed_gemm_architecture": "indexed"},
    "moe_batched_dequant": {"moe_batched_dequant": False},
    "moe_pair_dequant": {"moe_pair_dequant": False},
    "moe_batched_gather": {"moe_batched_gather": True},
    "moe_expert_autograd": {"moe_expert_autograd": "segmented_recompute"},
    "moe_activation_backend": {"moe_activation_backend": "triton"},
    "packed_shard_size_mb": {"packed_shard_size_mb": 1024},
    "int8_workspace_mb": {"int8_workspace_mb": 64},
}


def _pipeline() -> AgnesVideoPipeline:
    return AgnesVideoPipeline.__new__(AgnesVideoPipeline)


def _config(**params):
    model_params = {
        "moe_aux_loss_type": "model_native",
        "moe_aux_loss_weight": 0.01,
        "moe_router_z_loss_weight": 0.0,
        "moe_router_similarity_loss_weight": 0.0,
        "moe_balance_scope": "microbatch",
    }
    model_params.update(params)
    return SimpleNamespace(
        model=SimpleNamespace(
            attention_backend="auto",
            params=SimpleNamespace(**model_params),
        ),
        adapter=SimpleNamespace(
            timestep_rank_schedule="none", timestep_band_min=0.0,
            timestep_band_max=1.0,
        ),
        strategy=SimpleNamespace(type="text_to_video"),
        memory=SimpleNamespace(),
    )


def test_default_policy_is_the_only_supported_native_execution_policy():
    _pipeline().configure_moe_optimization_policy(MoEOptimizationPolicy())


def test_policy_rejection_inventory_covers_every_typed_field():
    assert set(_NON_DEFAULT_POLICY_VALUES) == {
        field.name for field in dataclasses.fields(MoEOptimizationPolicy)
    }


@pytest.mark.parametrize("field_name", sorted(_NON_DEFAULT_POLICY_VALUES))
def test_every_nondefault_unconsumed_policy_field_fails(field_name):
    policy = MoEOptimizationPolicy(**_NON_DEFAULT_POLICY_VALUES[field_name])
    with pytest.raises(ValueError, match=r"memory\."):
        _pipeline().configure_moe_optimization_policy(policy)


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"moe_aux_loss_type": "sequence"}, "moe_aux_loss_type"),
        ({"moe_router_z_loss_weight": 0.25}, "moe_router_z_loss_weight"),
        ({"moe_router_similarity_loss_weight": 0.5}, "moe_router_similarity_loss_weight"),
        ({"moe_balance_scope": "global_batch"}, "moe_balance_scope"),
    ],
)
def test_pipeline_validation_rejects_accepted_but_unimplemented_model_policy(params, field):
    errors = _pipeline().validate_config(_config(**params))
    assert any(field in error for error in errors)


def test_native_switch_microbatch_policy_and_weight_are_explicitly_supported():
    config = _config(moe_aux_loss_type="switch", moe_aux_loss_weight=0.25)
    assert _pipeline().validate_config(config) == []


@pytest.mark.parametrize(
    ("memory", "field"),
    [
        ({"moe_dispatch": "legacy"}, "moe_dispatch"),
        ({"moe_dispatch_preprocess": "device"}, "moe_dispatch_preprocess"),
        ({"moe_gemm_backend": "bmm"}, "moe_gemm_backend"),
    ],
)
def test_pipeline_validation_catches_policy_fields_ignored_by_requests_runtime_behavior(memory, field):
    config = _config()
    config.memory = SimpleNamespace(**memory)
    errors = _pipeline().validate_config(config)
    assert any(f"memory.{field}" in error for error in errors)
