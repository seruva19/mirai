"""Contracts for deterministic routing-aware batch grouping."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from mirai.config.schema import StrategyConfig
from mirai.core.training.data.routing_batches import ROUTING_TIMESTEPS_KEY, RoutingBatchItem, RoutingBatchProfile, plan_routing_microbatches
from mirai.core.training.lifecycle.training_step_pre import _plan_routing_window
from mirai.core.training.objectives.sampling import NoiseGenerator, UniformTimestepSampler
from mirai.core.training.strategies.text_to_video import TextToVideoStrategy


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


def test_profile_window_carries_each_samples_timestep_without_advancing_sampler(tmp_path) -> None:
    path = tmp_path / "routing.json"
    path.write_text(json.dumps({
        "schema_version": 1, "cache_experts": 1, "timestep_bins": 2,
        "samples": {"a": [[1], [1]], "b": [[2], [2]], "edge": [None, [3]]},
    }), encoding="utf-8")
    profile = RoutingBatchProfile.load(str(path))
    assert profile.experts("edge", 0.49) is None
    assert profile.experts("edge", 0.5) == profile.experts("edge", 1.0) == frozenset({3})
    sampler = UniformTimestepSampler(seed=4)
    session = SimpleNamespace(
        trainer=SimpleNamespace(timestep_sampler=sampler),
        compute_device=torch.device("cpu"), compute_dtype=torch.bfloat16,
    )
    a = {"sample_id": "a", "latent": torch.zeros(1, 1, 1)}
    b = {"sample_id": "b", "latent": torch.ones(1, 1, 1)}
    planned = _plan_routing_window(session, profile, [[a, b], [a, b]])
    assert [[row["sample_id"] for row in records] for records, _ in planned] == [["a", "a"], ["b", "b"]]
    original_draws = torch.stack([sampler.sample(2, like=torch.empty(())) for _ in range(2)])
    for (_, actual), expected in zip(planned, original_draws.T, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    live_sampler = UniformTimestepSampler(seed=4)
    strategy = TextToVideoStrategy(StrategyConfig(type="text_to_video", params={}))
    pipeline = SimpleNamespace(
        apply_noise=lambda latents, noise, times: latents + noise,
        prepare_model_timesteps=lambda times, **_: times,
    )
    noise = NoiseGenerator(seed=5)
    for records, planned_times in planned:
        inputs = strategy.prepare_inputs({
            "latents": torch.stack([record["latent"] for record in records]),
            "text_embeds": torch.zeros(2, 1, 1),
            ROUTING_TIMESTEPS_KEY: planned_times,
        }, pipeline, live_sampler, noise)
        torch.testing.assert_close(inputs.timestep, planned_times, rtol=0, atol=0)
    torch.testing.assert_close(live_sampler.sample(2, like=torch.empty(())), sampler.sample(2, like=torch.empty(())))
