from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mirai.core.moe.adaptation.balance import MoEBalancePolicy  # noqa: E402
from mirai.core.moe.adaptation.phi_balance import (  # noqa: E402
    PhiBalanceController,
    load_phi_balance_checkpoint_state,
    phi_balance_checkpoint_state,
)
from mirai.core.moe.runtime.touch_guard import enforce_expert_touch_guard  # noqa: E402


def test_balance_modes_resolve_without_hidden_pressure() -> None:
    auxiliary = MoEBalancePolicy.resolve(
        "aux_loss", aux_loss_weight=0.1, bias_update_rate=0.2
    )
    bias_only = MoEBalancePolicy.resolve(
        "bias_only", aux_loss_weight=0.1, bias_update_rate=0.2
    )
    disabled = MoEBalancePolicy.resolve(
        "off", aux_loss_weight=0.1, bias_update_rate=0.2
    )
    assert (auxiliary.aux_loss_weight, auxiliary.bias_update_rate) == (0.1, 0.2)
    assert (bias_only.aux_loss_weight, bias_only.bias_update_rate) == (0.0, 0.2)
    assert (disabled.aux_loss_weight, disabled.bias_update_rate) == (0.0, 0.0)


def test_phi_balance_has_gradients_and_exact_state_replay() -> None:
    controller = PhiBalanceController(ema_rate=0.25, potential="negative_entropy")
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.3, 0.7]], requires_grad=True
    )
    loss = controller.loss("layer", probabilities)
    loss.backward()
    assert probabilities.grad is not None
    state = phi_balance_checkpoint_state(controller)
    restored = PhiBalanceController(ema_rate=0.25, potential="negative_entropy")
    load_phi_balance_checkpoint_state(restored, state)
    restored_state = phi_balance_checkpoint_state(restored)
    assert state.keys() == restored_state.keys()
    for key in state:
        torch.testing.assert_close(state[key], restored_state[key])


def test_expert_touch_guard_is_inert_or_fails_before_backward() -> None:
    diagnostics = {
        "moe_routing": {
            "layers_detail": [{"active_expert_fraction": 0.75}]
        }
    }
    disabled = enforce_expert_touch_guard(
        diagnostics, mode="off", max_fraction=0.5
    )
    assert disabled.observed_fraction is None
    with pytest.raises(RuntimeError, match="before backward"):
        enforce_expert_touch_guard(
            diagnostics, mode="error", max_fraction=0.5
        )
