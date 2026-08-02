from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mirai.core.moe.routing.dynamic_topk import BudgetedDynamicTopK  # noqa: E402


def test_dynamic_topk_honors_bounds_budget_and_mass() -> None:
    scores = torch.tensor(
        [[0.90, 0.08, 0.02], [0.40, 0.35, 0.25], [0.70, 0.20, 0.10]]
    )
    indices = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    output = BudgetedDynamicTopK(min_k=1, average_k=2.0).regularize(
        "layer", indices, scores, training=True
    )

    active = (output != 0).sum(dim=-1)
    assert int(active.min()) >= 1
    assert int(active.max()) <= 3
    assert int(active.sum()) == 6
    torch.testing.assert_close(output.sum(dim=-1), scores.sum(dim=-1))
    assert int(active[1]) >= int(active[0])


def test_dynamic_topk_disabled_call_is_identity() -> None:
    scores = torch.randn(5, 2)
    output = BudgetedDynamicTopK(min_k=1, average_k=1.5).regularize(
        "layer", torch.zeros(5, 2, dtype=torch.long), scores, training=False
    )
    assert output is scores


def test_dynamic_topk_preserves_retained_score_gradients() -> None:
    scores = torch.tensor([[0.6, 0.3, 0.1]], requires_grad=True)
    output = BudgetedDynamicTopK(min_k=1, average_k=2.0).regularize(
        "layer", torch.tensor([[0, 1, 2]]), scores, training=True
    )
    output.square().sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_dynamic_topk_rejects_router_budget_mismatch() -> None:
    with pytest.raises(ValueError, match="router max_k"):
        BudgetedDynamicTopK(min_k=2, average_k=3.0).regularize(
            "layer",
            torch.zeros(2, 2, dtype=torch.long),
            torch.ones(2, 2),
            training=True,
        )
