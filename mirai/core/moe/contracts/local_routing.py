"""Contracts for local-routing cache evidence."""

from mirai.core.moe.monitoring.local_routing import (
    SCHEMA,
    build_local_routing_cache_evidence,
    segment_cache_best_hit_rates,
)


def test_constant_routes_reach_full_oracle_hit_rate() -> None:
    result = segment_cache_best_hit_rates(
        [[1], [1], [1], [1]], num_experts=4, segment_length=2, cache_sizes=[1, 2]
    )
    assert result == {1: 1.0, 2: 1.0}


def test_larger_cache_never_reduces_sch() -> None:
    result = segment_cache_best_hit_rates(
        [[0, 1], [2, 3], [0, 2], [1, 3]],
        num_experts=4,
        segment_length=2,
        cache_sizes=[1, 2, 4],
    )
    assert 0.0 <= result[1] <= result[2] <= result[4] == 1.0


def test_evidence_is_layered_and_lineage_bound() -> None:
    evidence = build_local_routing_cache_evidence(
        {"num_experts": 3, "layers": {"block.0": [[[0], [0], [1]]] }},
        segment_lengths=[1, 2],
        cache_sizes=[1, 2],
        dataset_fingerprint="data-sha",
        model_fingerprint="model-sha",
    )
    assert evidence["schema"] == SCHEMA
    assert evidence["oracle"] is True
    assert evidence["model_fingerprint"] == "model-sha"
    assert set(evidence["layers"]["block.0"]) == {"1", "2"}


def test_invalid_duplicate_route_fails_closed() -> None:
    try:
        segment_cache_best_hit_rates(
            [[1, 1]], num_experts=2, segment_length=1, cache_sizes=[1]
        )
    except ValueError as exc:
        assert "same expert twice" in str(exc)
    else:
        raise AssertionError("duplicate routes must fail")
