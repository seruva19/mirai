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


def test_calibration_records_layer_and_head_ids_and_restores_backends(monkeypatch) -> None:
    from mirai.core.models.magi2_preview import routing_profile
    from mirai.core.training.calibration.routing_batches import collect_routing_batch_profile

    backend = SimpleNamespace(execute=lambda module, *args: args[0])
    layers = [SimpleNamespace(num_heads=2, num_experts=3, _mirai_moe_kernel_backend=backend) for _ in range(2)]
    monkeypatch.setattr(routing_profile, "iter_magi2_moe_layers", lambda _: enumerate(layers))
    events = []

    def forward(latents, times, _text):
        expert = int(times[0] >= 0.5)
        for layer in layers:
            selected = torch.tensor([expert, expert + 1]).view(2, 1, 1)
            assert layer._mirai_moe_kernel_backend.execute(layer, latents, None, selected) is latents

    pipeline = SimpleNamespace(
        observe_routing_profile=lambda: routing_profile.observe_magi2_expert_ids(None),
        forward=forward,
    )
    trainer = SimpleNamespace(
        pipeline=pipeline, objective=None,
        strategy=SimpleNamespace(prepare_inputs=lambda batch, _pipeline, timestep, _noise, **kwargs: SimpleNamespace(
            noisy_latents=batch["latents"], timestep=timestep.sample(1, batch["latents"]),
            text_embeds=batch["text_embeds"], extra_forward_kwargs={},
        )),
        begin_validation=lambda: events.append("begin"), end_validation=lambda _: events.append("end"),
    )
    session = SimpleNamespace(
        trainer=trainer, compute_device=torch.device("cpu"), compute_dtype=torch.float32,
        train_records=[{"sample_id": "clip", "latent": torch.empty(1), "text_embed": torch.empty(1, 1)}],
        config=SimpleNamespace(strategy=SimpleNamespace(type="text_to_video"), training=SimpleNamespace(objective="flow_matching", seed=4)),
    )
    profile = collect_routing_batch_profile(session, max_samples=1, cache_experts=4, timestep_bins=2, trials=2)
    assert profile["samples"] == {"clip": [[0, 4, 6, 10], [1, 5, 7, 11]]}
    assert all(layer._mirai_moe_kernel_backend is backend for layer in layers)
    assert events == ["begin", "end"]


def test_routing_plan_reduces_packed_shard_read_bytes(tmp_path) -> None:
    from safetensors.torch import save_file

    from mirai.core.models.compressed_weights.packed.packed_stream import StreamingPackedTensorMapping

    path = tmp_path / "experts.safetensors"
    stored = torch.arange(4 * 256, dtype=torch.int32).view(4, 256)
    save_file({"experts": stored}, path)
    a = RoutingBatchItem("shape", frozenset({0, 1}))
    b = RoutingBatchItem("shape", frozenset({2, 3}))
    items = (a, b, a, b)
    planned = plan_routing_microbatches((items[:2], items[2:]), cache_experts=2)

    def read(order):
        with StreamingPackedTensorMapping(path, cache_capacity_bytes=2048, cache_expert_slices=True) as reader:
            for batch in order:
                ids = sorted(set().union(*(items[index].experts for index in batch)))
                actual = reader.get_slices_to_device("experts", ids, device="cpu")
                torch.testing.assert_close(actual, stored[ids], rtol=0, atol=0)
            return reader.snapshot()["read_bytes"]

    assert read(((0, 1), (2, 3))) == 6144
    assert read(planned) == 4096
