from __future__ import annotations

from copy import deepcopy

import pytest

from mirai.config.loader import load_config
from mirai.core.models.lingbot_video.pipeline import (
    LINGBOT_EXPERT_MLP_EXECUTION_SPEC,
)
from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.runtime.kernels import build_moe_kernel_backend
from mirai.core.moe.runtime.specs import CANONICAL_PACKED_EXPERT_MLP_SPEC
from mirai.core.training.runtime.contract import validate_training_runtime_config


def _w4a8_config():
    config = load_config("configs/lingbot_video/train_bf16.toml")
    config.memory.frozen_weight_quantization = "w4a8"
    config.memory.frozen_weight_quantization_strategy = "compressed_weights"
    config.memory.expert_weight_access = "chunked_dequant"
    config.memory.expert_dequant_chunk_size = 64
    config.memory.moe_kernel_backend = "torch"
    return config


def test_w4a8_reference_policy_is_accepted_for_canonical_provider() -> None:
    config = _w4a8_config()

    validate_training_runtime_config(config)

    provider = get_model_family_provider(config.model.type)
    assert provider is not None
    assert provider.expert_mlp_execution_spec == CANONICAL_PACKED_EXPERT_MLP_SPEC
    assert LINGBOT_EXPERT_MLP_EXECUTION_SPEC == CANONICAL_PACKED_EXPERT_MLP_SPEC


def test_published_w4a8_config_loads_with_explicit_accelerated_policy() -> None:
    config = load_config("configs/lingbot_video/train_w4a8.toml")

    assert config.memory.frozen_weight_quantization == "w4a8"
    assert config.memory.frozen_weight_quantization_strategy == "compressed_weights"
    assert config.memory.expert_weight_access == "chunked_dequant"
    assert config.memory.expert_dequant_chunk_size > 0
    assert config.memory.moe_kernel_backend == "w4a8_int8"

    config.memory.moe_kernel_backend = "torch"
    validate_training_runtime_config(config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: setattr(config.memory, "expert_weight_access", "active_dequant"),
            "expert_weight_access='chunked_dequant'",
        ),
        (
            lambda config: setattr(config.memory, "expert_dequant_chunk_size", 0),
            "expert_dequant_chunk_size > 0",
        ),
        (
            lambda config: setattr(config.memory, "moe_kernel_backend", "rotated_int8"),
            "supports only memory.moe_kernel_backend",
        ),
        (
            lambda config: (
                setattr(config.memory, "frozen_weight_packed_state_path", "weights.index.json"),
                setattr(config.memory, "packed_state_preload", "off"),
            ),
            "Packed W4A8 artifacts require memory.packed_state_preload",
        ),
        (
            lambda config: setattr(config.memory, "weight_residency_strategy", "stream_disk"),
            "does not support disk-backed packed streaming",
        ),
        (
            lambda config: setattr(config.model.params, "expert_quantization_rotation", "learned"),
            "does not support learned expert rotations",
        ),
        (
            lambda config: setattr(config.memory, "expert_precision_plan_path", "plan.json"),
            "does not support mixed expert precision plans",
        ),
        (
            lambda config: setattr(config.adapter, "type", "selected_expert"),
            "requires unquantized dense expert weights",
        ),
    ],
)
def test_w4a8_rejects_unsupported_training_combinations(mutate, message: str) -> None:
    config = deepcopy(_w4a8_config())
    mutate(config)

    with pytest.raises(ValueError, match=message):
        validate_training_runtime_config(config)


def test_existing_provider_default_does_not_enable_w4a8() -> None:
    config = load_config("configs/lingbot_video/train_bf16.toml")

    validate_training_runtime_config(config)

    assert config.memory.frozen_weight_quantization == "none"
    assert config.memory.moe_kernel_backend == "auto"


def test_w4a8_int8_backend_rejects_other_weight_formats() -> None:
    config = load_config("configs/lingbot_video/train_bf16.toml")
    config.memory.moe_kernel_backend = "w4a8_int8"

    with pytest.raises(ValueError, match="requires.*frozen_weight_quantization='w4a8'"):
        validate_training_runtime_config(config)


def test_w4a8_rejects_provider_without_canonical_expert_layout() -> None:
    import mirai.core.models.magi2_preview.pipeline  # noqa: F401

    config = _w4a8_config()
    config.model.type = "magi2-preview"

    with pytest.raises(ValueError, match="canonical routed expert layout"):
        validate_training_runtime_config(config)


def test_w4a8_kernel_registry_requires_direct_routing_and_calls_exact_entrypoint() -> None:
    with pytest.raises(ValueError, match="direct-routed"):
        build_moe_kernel_backend("w4a8_int8", direct_routed=False)

    calls = []

    class _Experts:
        def run_direct_routed_w4a8(self, tokens, scores, indices):
            calls.append((tokens, scores, indices))
            return "result"

    backend = build_moe_kernel_backend("w4a8_int8", direct_routed=True)
    assert backend is not None
    assert backend.execute_direct(_Experts(), "tokens", "scores", "indices") == "result"
    assert calls == [("tokens", "scores", "indices")]

    with pytest.raises(RuntimeError, match="run_direct_routed_w4a8"):
        backend.execute_direct(object(), "tokens", "scores", "indices")
