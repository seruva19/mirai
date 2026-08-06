"""Behavioral contract for cross-timestep expert-branch feature caching.

The uncached grouped expert execution is the reference path. These probes pin
that the cache is inert when off, numerically transparent when armed but never
reusing, observable through its counters when reusing, and invalidated by a
routed-expert change.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.nn.utils import parametrize  # noqa: E402

from mirai.config.schema import ConfigError, TrainingConfig  # noqa: E402
from mirai.core.models.magi2_preview.grouped_moe import (  # noqa: E402
    Magi2GroupedMoEBackend,
    Magi2GroupedMoEPlan,
)
from mirai.core.models.magi2_preview.pipeline import LowRankWeight  # noqa: E402
from mirai.core.moe.runtime.expert_feature_cache import (  # noqa: E402
    CachedExpertExecution,
    ExpertFeatureCache,
    ExpertFeatureCacheError,
    ExpertFeatureCachePolicy,
    ExpertFeatureCacheTelemetry,
    normalize_expert_feature_cache_mode,
)

HIDDEN_SIZE = 16
D_EXPERT = 12


def _build_moe(*, dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    """Reduced-shape vendored MAGI-2 MoE layer with a router LoRA."""
    from mirai.vendors.magi2_preview.model.magi2_preview import (
        CoreMultiHeadMoE,
        CoreMultiHeadMoEConfig,
    )

    torch.manual_seed(0)
    module = CoreMultiHeadMoE(
        CoreMultiHeadMoEConfig(
            hidden_size=HIDDEN_SIZE,
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=D_EXPERT,
            num_layers=1,
            params_dtype=dtype,
            score_func="sigmoid",
            route_norm=True,
            route_scale=4.9,
        )
    )
    with torch.no_grad():
        for tensor in (module.gate, module.W_gate, module.W_up, module.W_down):
            tensor.normal_(std=0.1)
        module.router.expert_bias.normal_(std=0.05)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    parametrize.register_parametrization(
        module, "gate", LowRankWeight(tuple(module.gate.shape), rank=2, alpha=2.0)
    )
    module.parametrizations["gate"].original.requires_grad_(False)
    with torch.no_grad():
        module.parametrizations["gate"][0].lora_b.normal_(std=0.1)
    return module


def _backend() -> Magi2GroupedMoEBackend:
    return Magi2GroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    )


def _trajectory(steps: int, *, drift: float, tokens: int = 5) -> list[torch.Tensor]:
    """A denoising-like sequence of layer inputs with a controlled step size."""
    torch.manual_seed(7)
    base = torch.randn(tokens, HIDDEN_SIZE)
    direction = torch.randn(tokens, HIDDEN_SIZE)
    direction = direction / direction.norm() * base.norm()
    return [base + direction * (drift * index) for index in range(steps)]


def _run(module: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return module._forward_impl(hidden)


# -- policy validation ------------------------------------------------------


def test_default_policy_is_off_and_inert() -> None:
    policy = ExpertFeatureCachePolicy()
    assert policy.mode == "off"
    assert policy.enabled is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "aggressive"},
        {"mode": "branch", "drift_threshold": 1.5},
        {"mode": "branch", "drift_threshold": -0.1},
        {"mode": "branch", "max_reuse_span": -1},
        {"mode": "branch", "slots": 0},
    ],
)
def test_policy_rejects_out_of_contract_values(kwargs: dict) -> None:
    with pytest.raises(ExpertFeatureCacheError):
        ExpertFeatureCachePolicy(**kwargs)


def test_mode_normalization_is_case_and_space_tolerant() -> None:
    assert normalize_expert_feature_cache_mode("  Branch ") == "branch"
    assert normalize_expert_feature_cache_mode("") == "off"


def _config_payload(**inference: object) -> dict:
    return {
        "model": {"type": "magi2-preview", "path": "models/magi2"},
        "memory": {"moe_kernel_backend": "grouped"},
        "inference": dict(inference),
    }


def test_config_defaults_leave_the_cache_off() -> None:
    cfg = TrainingConfig.from_dict(
        {"model": {"type": "magi2-preview", "path": "models/magi2"}}
    )
    assert cfg.inference.expert_feature_cache == "off"
    assert cfg.inference.expert_feature_cache_drift_threshold == 0.05
    assert cfg.inference.expert_feature_cache_max_reuse_span == 2
    assert cfg.inference.expert_feature_cache_slots == 2


def test_config_accepts_the_documented_surface() -> None:
    cfg = TrainingConfig.from_dict(
        _config_payload(
            expert_feature_cache="branch",
            expert_feature_cache_drift_threshold=0.2,
            expert_feature_cache_max_reuse_span=3,
            expert_feature_cache_slots=1,
        )
    )
    assert cfg.inference.expert_feature_cache == "branch"
    assert cfg.inference.expert_feature_cache_drift_threshold == 0.2
    assert cfg.inference.expert_feature_cache_max_reuse_span == 3
    assert cfg.inference.expert_feature_cache_slots == 1


def test_config_rejects_an_unknown_mode() -> None:
    with pytest.raises(ConfigError):
        TrainingConfig.from_dict(_config_payload(expert_feature_cache="on"))


def test_config_rejects_an_unknown_cache_key() -> None:
    with pytest.raises(ConfigError):
        TrainingConfig.from_dict(_config_payload(expert_feature_cache_ttl=3))


def test_config_requires_the_grouped_expert_execution_seam() -> None:
    payload = _config_payload(expert_feature_cache="branch")
    payload["memory"]["moe_kernel_backend"] = "auto"
    with pytest.raises(ConfigError, match="moe_kernel_backend='grouped'"):
        TrainingConfig.from_dict(payload)


# -- disabled isolation -----------------------------------------------------


def test_disabled_cache_allocates_no_state_and_leaves_output_unchanged() -> None:
    module = _build_moe()
    backend = _backend()
    cache = ExpertFeatureCache(ExpertFeatureCachePolicy(mode="off"))
    module._mirai_moe_kernel_backend = backend
    reference = [_run(module, hidden) for hidden in _trajectory(4, drift=0.001)]

    module._mirai_moe_kernel_backend = CachedExpertExecution(
        inner=backend, cache=cache
    )
    cached = [_run(module, hidden) for hidden in _trajectory(4, drift=0.001)]

    for expected, actual in zip(reference, cached):
        assert torch.equal(expected, actual)
    assert cache.resident_entries == 0
    assert cache.telemetry.snapshot()["visits"] == 0


def test_disabled_cache_never_decomposes_branches() -> None:
    """A disabled cache delegates without building any branch plan."""

    class _SpyBackend:
        def __init__(self) -> None:
            self.executed = 0

        def execute(self, module, x_heads, topk_probs, topk_indices):
            self.executed += 1
            return torch.zeros_like(x_heads)

        def plan_branches(self, module, x_heads, topk_probs, topk_indices):
            raise AssertionError("a disabled cache must not decompose branches")

    spy = _SpyBackend()
    cache = ExpertFeatureCache(ExpertFeatureCachePolicy(mode="off"))
    out = cache.execute(
        spy,
        object(),
        torch.zeros(2, 2, 4),
        torch.zeros(2, 2, 2),
        torch.zeros(2, 2, 2, dtype=torch.long),
    )
    assert spy.executed == 1
    assert out.shape == (2, 2, 4)
    assert cache.resident_entries == 0


# -- reference parity -------------------------------------------------------


def test_armed_cache_without_reuse_is_bit_identical_to_the_uncached_path() -> None:
    """``max_reuse_span=0`` keeps the cache armed but always recomputes."""
    trajectory = _trajectory(5, drift=0.0005)

    module = _build_moe()
    module._mirai_moe_kernel_backend = _backend()
    reference = [_run(module, hidden) for hidden in trajectory]

    module = _build_moe()
    cache = ExpertFeatureCache(
        ExpertFeatureCachePolicy(
            mode="branch", drift_threshold=1.0, max_reuse_span=0, slots=2
        )
    )
    module._mirai_moe_kernel_backend = CachedExpertExecution(
        inner=_backend(), cache=cache
    )
    armed = [_run(module, hidden) for hidden in trajectory]

    for expected, actual in zip(reference, armed):
        assert torch.equal(expected, actual)
    snapshot = cache.telemetry.snapshot()
    assert snapshot["visits"] == len(trajectory)
    assert snapshot["reuse_visits"] == 0
    assert snapshot["reused_branches"] == 0
    assert snapshot["full_recomputes"] == len(trajectory)


def test_zero_drift_threshold_only_reuses_an_unchanged_input() -> None:
    module = _build_moe()
    cache = ExpertFeatureCache(
        ExpertFeatureCachePolicy(
            mode="branch", drift_threshold=0.0, max_reuse_span=8, slots=1
        )
    )
    module._mirai_moe_kernel_backend = CachedExpertExecution(
        inner=_backend(), cache=cache
    )
    hidden = _trajectory(1, drift=0.0)[0]
    first = _run(module, hidden)
    second = _run(module, hidden)
    assert torch.equal(first, second)
    snapshot = cache.telemetry.snapshot()
    assert snapshot["reuse_visits"] == 1
    assert snapshot["recomputed_branches"] == snapshot["reused_branches"] == 20


# -- reuse behavior ---------------------------------------------------------


def _reuse_run(
    *, drift: float, steps: int = 6, span: int = 8, threshold: float = 0.5
):
    module = _build_moe()
    reference_module = _build_moe()
    reference_module._mirai_moe_kernel_backend = _backend()
    cache = ExpertFeatureCache(
        ExpertFeatureCachePolicy(
            mode="branch",
            drift_threshold=threshold,
            max_reuse_span=span,
            slots=1,
        )
    )
    module._mirai_moe_kernel_backend = CachedExpertExecution(
        inner=_backend(), cache=cache
    )
    trajectory = _trajectory(steps, drift=drift)
    reference = [_run(reference_module, hidden) for hidden in trajectory]
    cached = [_run(module, hidden) for hidden in trajectory]
    return reference, cached, cache


def test_aggressive_reuse_stays_inside_a_documented_tolerance_envelope() -> None:
    """Small-step reuse perturbs the layer output, bounded relative to it.

    The bound is a property of this fixture, not a general quality claim: it
    pins that reuse degrades gracefully rather than diverging.
    """
    reference, cached, cache = _reuse_run(drift=0.002)
    snapshot = cache.telemetry.snapshot()
    assert snapshot["reuse_visits"] > 0
    assert snapshot["reused_branches"] > 0
    assert snapshot["branch_reuse_ratio"] > 0.0
    for expected, actual in zip(reference, cached):
        denominator = expected.abs().max().clamp(min=1e-6)
        assert ((actual - expected).abs().max() / denominator) < 0.25


def test_first_visit_of_every_layer_is_always_a_full_recompute() -> None:
    reference, cached, _cache = _reuse_run(drift=0.002, steps=1)
    assert torch.equal(reference[0], cached[0])


def test_reuse_span_forces_a_periodic_full_recompute() -> None:
    _reference, _cached, cache = _reuse_run(drift=0.0001, steps=7, span=2)
    snapshot = cache.telemetry.snapshot()
    assert snapshot["span_invalidations"] > 0
    # One warm-up miss plus one forced recompute per exhausted span.
    assert snapshot["full_recomputes"] == 1 + snapshot["span_invalidations"]


def test_input_drift_invalidates_the_whole_entry() -> None:
    _reference, _cached, cache = _reuse_run(drift=0.9, steps=4, threshold=0.05)
    snapshot = cache.telemetry.snapshot()
    assert snapshot["drift_invalidations"] == 3
    assert snapshot["reuse_visits"] == 0


def test_routing_drift_recomputes_exactly_the_reassigned_branches() -> None:
    """A router shift moves top-k assignments while the input barely moves."""
    module = _build_moe()
    cache = ExpertFeatureCache(
        ExpertFeatureCachePolicy(
            mode="branch", drift_threshold=0.5, max_reuse_span=8, slots=1
        )
    )
    module._mirai_moe_kernel_backend = CachedExpertExecution(
        inner=_backend(), cache=cache
    )
    torch.manual_seed(3)
    first = torch.randn(6, HIDDEN_SIZE)
    _run(module, first)
    with torch.no_grad():
        before = module._route(
            first.view(-1, module.num_heads, module.d_head)
        )[1].reshape(-1).clone()

    with torch.no_grad():
        module.router.expert_bias.mul_(-30.0)
    second = first * 1.001
    _run(module, second)
    with torch.no_grad():
        after = module._route(
            second.view(-1, module.num_heads, module.d_head)
        )[1].reshape(-1).clone()

    changed = int((before != after).sum().item())
    assert 0 < changed < int(before.numel()), "fixture must move some slots only"
    snapshot = cache.telemetry.snapshot()
    assert snapshot["reuse_visits"] == 1
    assert snapshot["drift_invalidations"] == 0
    # The first visit recomputes every slot; the second recomputes exactly the
    # reassigned ones.
    assert snapshot["recomputed_branches"] == int(before.numel()) + changed
    assert snapshot["reused_branches"] == int(before.numel()) - changed


def test_partial_branch_recompute_is_exact() -> None:
    """A masked recompute reproduces a full recompute on the masked slots.

    This is the property that makes reuse the only source of difference from
    the uncached path: nothing about a slot's expert matmul depends on which
    other slots were computed alongside it.
    """
    module = _build_moe()
    backend = _backend()
    torch.manual_seed(11)
    hidden = torch.randn(6, HIDDEN_SIZE)
    with torch.no_grad():
        x_heads = hidden.view(-1, module.num_heads, module.d_head)
        topk_probs, topk_indices = module._route(x_heads)
        plan = backend.plan_branches(module, x_heads, topk_probs, topk_indices)
        full = plan.compute_branch_features(None)
        torch.manual_seed(2)
        mask = torch.rand(int(plan.expert_ids.numel())) < 0.5
        assert bool(mask.any()) and not bool(mask.all())
        selected = plan.compute_branch_features(mask)
        complement = plan.compute_branch_features(~mask)
        # Exact up to the row-count-dependent blocking of the underlying GEMM:
        # a masked group holds fewer rows than the same group of a full pass.
        assert torch.allclose(selected[mask], full[mask], rtol=0.0, atol=1e-6)
        assert torch.allclose(complement[~mask], full[~mask], rtol=0.0, atol=1e-6)
        # And the two-step decomposition reproduces the uncached execution.
        assert torch.equal(
            plan.combine_branch_features(full),
            backend.execute(module, x_heads, topk_probs, topk_indices),
        )


# -- state isolation and telemetry -----------------------------------------


def test_training_never_engages_the_cache() -> None:
    module = _build_moe()
    cache = ExpertFeatureCache(
        ExpertFeatureCachePolicy(mode="branch", drift_threshold=1.0, max_reuse_span=8)
    )
    module._mirai_moe_kernel_backend = CachedExpertExecution(
        inner=_backend(), cache=cache
    )
    hidden = torch.randn(4, HIDDEN_SIZE, requires_grad=True)
    output = module._forward_impl(hidden)
    output.float().square().mean().backward()
    assert hidden.grad is not None
    assert cache.resident_entries == 0
    assert cache.telemetry.snapshot()["visits"] == 0


def test_reset_drops_every_entry_and_counter() -> None:
    _reference, _cached, cache = _reuse_run(drift=0.002)
    assert cache.resident_entries > 0
    cache.reset()
    assert cache.resident_entries == 0
    assert cache.telemetry.snapshot()["visits"] == 0


def test_slots_keep_interleaved_trajectories_apart() -> None:
    """Two alternating inputs, as sequential CFG produces, both find a slot."""
    module = _build_moe()
    cache = ExpertFeatureCache(
        ExpertFeatureCachePolicy(
            mode="branch", drift_threshold=0.05, max_reuse_span=8, slots=2
        )
    )
    module._mirai_moe_kernel_backend = CachedExpertExecution(
        inner=_backend(), cache=cache
    )
    torch.manual_seed(5)
    conditional = _trajectory(4, drift=0.001, tokens=4)
    unconditional = [tensor + 5.0 for tensor in conditional]
    for cond, uncond in zip(conditional, unconditional):
        _run(module, cond)
        _run(module, uncond)
    snapshot = cache.telemetry.snapshot()
    assert snapshot["visits"] == 8
    assert snapshot["reuse_visits"] == 6
    # One warm-up miss per trajectory: the second trajectory's first visit is
    # too far from the first entry and claims the free slot instead of it.
    assert snapshot["full_recomputes"] == 2
    assert snapshot["drift_invalidations"] == 1


def test_telemetry_snapshot_shape_is_report_ready() -> None:
    _reference, _cached, cache = _reuse_run(drift=0.002)
    snapshot = cache.telemetry.snapshot()
    for key in (
        "visits",
        "full_recomputes",
        "reuse_visits",
        "reused_branches",
        "recomputed_branches",
        "signature_invalidations",
        "drift_invalidations",
        "span_invalidations",
        "branch_reuse_ratio",
        "layers",
    ):
        assert key in snapshot
    assert isinstance(snapshot["layers"], dict)
    assert len(snapshot["layers"]) == 1
    (layer_stats,) = snapshot["layers"].values()
    assert layer_stats["visits"] == snapshot["visits"]

    empty = ExpertFeatureCacheTelemetry().snapshot()
    assert empty["visits"] == 0
    assert empty["branch_reuse_ratio"] == 0.0
    assert empty["layers"] == {}


def test_cached_execution_requires_a_branch_capable_backend() -> None:
    class _CombinedOnlyBackend:
        def execute(self, module, x_heads, topk_probs, topk_indices):
            raise AssertionError("not reached")

    with pytest.raises(ExpertFeatureCacheError):
        CachedExpertExecution(
            inner=_CombinedOnlyBackend(),
            cache=ExpertFeatureCache(ExpertFeatureCachePolicy(mode="branch")),
        )


def test_a_family_without_branch_decomposition_fails_explicitly() -> None:
    """The generic pipeline hook is a no-op when off and an error when armed."""
    from mirai.core.models.base import BasePipeline

    configure = BasePipeline.configure_expert_feature_cache
    holder = object()
    configure(holder, ExpertFeatureCache(ExpertFeatureCachePolicy(mode="off")))
    with pytest.raises(ValueError, match="expert_feature_cache"):
        configure(
            holder, ExpertFeatureCache(ExpertFeatureCachePolicy(mode="branch"))
        )
