from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from mirai.config.schema import TrainingConfig  # noqa: E402
from mirai.core.training.optim.optimizer import build_optimizer  # noqa: E402
from mirai.core.training.optim.selected_expert_adamw import (  # noqa: E402
    SelectedExpertAdamW,
)
from mirai.core.training.optim.stochastic_rounding import (  # noqa: E402
    StochasticRoundingAdamW,
    stochastic_round_bfloat16,
)
from mirai.core.training.runtime.contract import (  # noqa: E402
    validate_training_runtime_config,
)


def test_bfloat16_rounding_is_adjacent_and_unbiased() -> None:
    torch.manual_seed(7)
    bf16_ulp_at_one = 1.0 / 128.0
    source_value = 1.0 + 0.25 * bf16_ulp_at_one
    source = torch.full((131_072,), source_value, dtype=torch.float32)

    rounded = stochastic_round_bfloat16(source).float()

    assert set(rounded.unique().tolist()) == {1.0, 1.0 + bf16_ulp_at_one}
    assert abs(float(rounded.mean()) - source_value) < 5.0e-5


def test_adamw_updates_bfloat16_without_persistent_fp32_master() -> None:
    torch.manual_seed(11)
    parameter = torch.nn.Parameter(torch.ones(16_384, dtype=torch.bfloat16))
    parameter.grad = torch.ones_like(parameter)
    optimizer = StochasticRoundingAdamW([parameter], lr=1.0e-3)

    optimizer.step()

    assert bool((parameter != 1.0).any())
    assert abs(float(parameter.detach().float().mean()) - 0.999) < 1.0e-4
    state = optimizer.state[parameter]
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}


def test_bfloat16_second_moment_does_not_stall_below_one_ulp() -> None:
    torch.manual_seed(13)
    parameter = torch.nn.Parameter(torch.zeros(65_536, dtype=torch.bfloat16))
    optimizer = StochasticRoundingAdamW(
        [parameter],
        lr=0.0,
        betas=(0.9, 0.999),
    )
    state = optimizer.state[parameter]
    state["step"] = torch.tensor(0, dtype=torch.int64)
    state["exp_avg"] = torch.zeros_like(parameter)
    state["exp_avg_sq"] = torch.ones_like(parameter)
    parameter.grad = torch.full_like(parameter, 2.0**0.5)

    optimizer.step()

    observed = state["exp_avg_sq"].float()
    squared_gradient = float(parameter.grad[0].float().square())
    expected_mean = 0.999 + 0.001 * squared_gradient
    assert bool((observed != 1.0).any())
    assert abs(float(observed.mean()) - expected_mean) < 2.5e-4


def test_global_rng_checkpoint_restores_exact_stochastic_updates() -> None:
    torch.manual_seed(19)
    parameter = torch.nn.Parameter(torch.ones(8_192, dtype=torch.bfloat16))
    optimizer = StochasticRoundingAdamW([parameter], lr=7.5e-4)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    saved_parameter = parameter.detach().clone()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_rng = torch.get_rng_state()

    parameter.grad = torch.full_like(parameter, 0.75)
    optimizer.step()
    expected_parameter = parameter.detach().clone()
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    restored_parameter = torch.nn.Parameter(saved_parameter)
    restored = StochasticRoundingAdamW([restored_parameter], lr=7.5e-4)
    restored.load_state_dict(saved_optimizer)
    torch.set_rng_state(saved_rng)
    restored_parameter.grad = torch.full_like(restored_parameter, 0.75)
    restored.step()

    assert torch.equal(restored_parameter, expected_parameter)
    restored_state = restored.state_dict()["state"][0]
    expected_state = expected_optimizer["state"][0]
    assert torch.equal(restored_state["exp_avg"], expected_state["exp_avg"])
    assert torch.equal(
        restored_state["exp_avg_sq"],
        expected_state["exp_avg_sq"],
    )
    assert torch.equal(restored_state["step"], expected_state["step"])


def test_optimizer_builder_preserves_default_and_selects_opt_in_path() -> None:
    default_parameter = torch.nn.Parameter(torch.ones(4))
    default = build_optimizer(
        params=[default_parameter],
        optimizer_type="adamw",
        lr=1.0e-3,
        weight_decay=0.0,
        allow_fallback=True,
    ).optimizer
    assert type(default) is torch.optim.AdamW

    stochastic_parameter = torch.nn.Parameter(
        torch.ones(4, dtype=torch.bfloat16)
    )
    stochastic = build_optimizer(
        params=[stochastic_parameter],
        optimizer_type="adamw",
        lr=1.0e-3,
        weight_decay=0.0,
        allow_fallback=True,
        stochastic_rounding=True,
    ).optimizer
    assert isinstance(stochastic, StochasticRoundingAdamW)


def test_selected_expert_stochastic_update_keeps_compact_state() -> None:
    torch.manual_seed(23)
    parameter = torch.nn.Parameter(
        torch.ones((4, 4_096), dtype=torch.bfloat16)
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer = SelectedExpertAdamW(
        [parameter],
        expert_ids=(1, 3),
        lr=1.0e-3,
        stochastic_rounding=True,
    )

    optimizer.step()

    assert torch.equal(parameter[0], torch.ones_like(parameter[0]))
    assert torch.equal(parameter[2], torch.ones_like(parameter[2]))
    assert bool((parameter[1] != 1.0).any())
    assert bool((parameter[3] != 1.0).any())
    state = optimizer.state[parameter]
    assert tuple(state["exp_avg"].shape) == (2, 4_096)
    assert state["exp_avg"].dtype == torch.bfloat16


def test_runtime_contract_rejects_incompatible_optimizer_paths() -> None:
    config = TrainingConfig()
    config.optimizer.stochastic_rounding = True
    config.optimizer.type = "lion"
    with pytest.raises(ValueError, match="requires optimizer.type"):
        validate_training_runtime_config(config)

    config.optimizer.type = "adamw"
    config.training.optimizer_cpu_offload = True
    with pytest.raises(ValueError, match="optimizer_cpu_offload=false"):
        validate_training_runtime_config(config)
