"""Contracts for deterministic routing-aware batch grouping."""

from __future__ import annotations

import pytest

from mirai.core.training.data.routing_batches import RoutingBatchItem, plan_routing_microbatches


def test_regrouping_keeps_samples_and_batch_sizes() -> None:
    first = RoutingBatchItem(("t2v", 16), frozenset({0, 1}))
    second = RoutingBatchItem(("t2v", 16), frozenset({2, 3}))
    original = ((first, second), (first, second))
    assert plan_routing_microbatches(original, cache_experts=2) == ((0, 2), (1, 3))
    assert plan_routing_microbatches(original, cache_experts=4) == ((0, 1), (2, 3))


def test_unknown_routes_and_incompatible_batches_do_not_move() -> None:
    first = RoutingBatchItem("shape_a", frozenset({1}))
    second = RoutingBatchItem("shape_a", frozenset({2}))
    unknown = RoutingBatchItem("shape_a", None)
    incompatible = RoutingBatchItem("shape_b", frozenset({1}))
    original = ((first, second), (first, second), (unknown,), (first,), (incompatible,), (second,))
    assert plan_routing_microbatches(original, cache_experts=1) == ((0, 2), (1, 3), (4,), (5,), (6,), (7,))
    with pytest.raises(ValueError, match="non-negative"):
        plan_routing_microbatches(original, cache_experts=-1)
