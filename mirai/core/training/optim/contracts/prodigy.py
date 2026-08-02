"""Deterministic contracts for the Prodigy optimizer state and update."""

from __future__ import annotations

import copy

import torch

from mirai.core.training.optim.prodigy import Prodigy


def _optimizer(parameter: torch.nn.Parameter) -> Prodigy:
    return Prodigy(
        [parameter],
        lr=1.0,
        betas=(0.9, 0.999),
        beta3=0.99,
        eps=1e-8,
        d0=1e-3,
        d_coef=1.0,
        growth_rate=1.5,
        decouple=True,
        use_bias_correction=False,
        safeguard_warmup=False,
    )


def _apply(optimizer: Prodigy, parameter: torch.nn.Parameter, gradient) -> None:
    parameter.grad = torch.as_tensor(gradient, dtype=parameter.dtype).clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_first_step_matches_algorithm_three_moments_and_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    gradient = torch.tensor([0.25, -0.5], dtype=torch.float64)
    optimizer = _optimizer(parameter)
    before = parameter.detach().clone()

    _apply(optimizer, parameter, gradient)

    d = 1e-3
    exp_avg = d * (1.0 - 0.9) * gradient
    exp_avg_sq = d * d * (1.0 - 0.999) * gradient.square()
    expected = before - d * exp_avg / (exp_avg_sq.sqrt() + d * 1e-8)
    torch.testing.assert_close(parameter, expected, rtol=0.0, atol=1e-15)
    state = optimizer.state[parameter]
    torch.testing.assert_close(state["exp_avg"], exp_avg, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state["exp_avg_sq"], exp_avg_sq, rtol=0.0, atol=0.0)
    assert optimizer.param_groups[0]["d"] == d
    assert optimizer.param_groups[0]["k"] == 1


def test_state_dict_resume_matches_uninterrupted_updates() -> None:
    gradients = ([0.25, -0.5], [-0.75, 0.125], [0.5, 0.25])
    uninterrupted_parameter = torch.nn.Parameter(
        torch.tensor([1.0, -2.0], dtype=torch.float64)
    )
    uninterrupted = _optimizer(uninterrupted_parameter)
    for gradient in gradients:
        _apply(uninterrupted, uninterrupted_parameter, gradient)

    resumed_parameter = torch.nn.Parameter(
        torch.tensor([1.0, -2.0], dtype=torch.float64)
    )
    first_session = _optimizer(resumed_parameter)
    _apply(first_session, resumed_parameter, gradients[0])
    saved_parameter = resumed_parameter.detach().clone()
    saved_state = copy.deepcopy(first_session.state_dict())

    restored_parameter = torch.nn.Parameter(saved_parameter)
    restored = _optimizer(restored_parameter)
    restored.load_state_dict(saved_state)
    for gradient in gradients[1:]:
        _apply(restored, restored_parameter, gradient)

    torch.testing.assert_close(
        restored_parameter,
        uninterrupted_parameter,
        rtol=0.0,
        atol=0.0,
    )
    restored_group = restored.param_groups[0]
    uninterrupted_group = uninterrupted.param_groups[0]
    for key in ("d", "d_max", "d_hat", "d_numerator", "d_denom", "k"):
        assert restored_group[key] == uninterrupted_group[key]
    restored_state = restored.state[restored_parameter]
    uninterrupted_state = uninterrupted.state[uninterrupted_parameter]
    assert restored_state.keys() == uninterrupted_state.keys()
    for key, value in restored_state.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, uninterrupted_state[key], rtol=0.0, atol=0.0)
        else:
            assert value == uninterrupted_state[key]
