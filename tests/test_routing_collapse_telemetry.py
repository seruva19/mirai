"""Behavioral contract for minority-side routing-collapse telemetry.

Covers the estimator definitions against crafted routing distributions, the
cross-step collapse/rebound/plateau state machine, its bounded-memory promise,
and the MAGI-2 monitoring tap that makes an otherwise silent aux-loss-free
router observable.
"""

from __future__ import annotations

import types

import pytest
import torch

from mirai.core.moe.monitoring.collapse import (
    COLLAPSE_HEALTH_RATIO,
    COLLAPSE_PLATEAU_WINDOW,
    RouterLoadSignature,
    RoutingCollapseTracker,
    collapse_metric_keys,
    router_expert_counts,
    router_keys,
    router_load_signatures,
    selection_load_signatures,
)


def _signature(counts: list[float]) -> RouterLoadSignature:
    return router_load_signatures(torch.tensor([counts], dtype=torch.float32))[0]


# --- Estimator definitions -------------------------------------------------


def test_uniform_load_is_balanced_and_not_collapsed() -> None:
    """A uniform router sits exactly at the normalized reference value 1.0."""
    signature = _signature([8.0, 8.0, 8.0, 8.0])
    assert signature.minority_share == pytest.approx(0.25)
    assert signature.normalized_minority_share == pytest.approx(1.0)
    assert signature.top1_share == pytest.approx(0.25)
    assert signature.dead_expert_fraction == 0.0
    assert signature.underused_expert_fraction == 0.0
    assert not signature.is_collapsed()


def test_single_expert_monopoly_is_collapse_with_dead_experts() -> None:
    """Everything on one expert: minority zero, the rest dead, flag raised."""
    signature = _signature([32.0, 0.0, 0.0, 0.0])
    assert signature.minority_share == 0.0
    assert signature.normalized_minority_share == 0.0
    assert signature.top1_share == pytest.approx(1.0)
    assert signature.dead_expert_fraction == pytest.approx(0.75)
    assert signature.underused_expert_fraction == pytest.approx(0.75)
    assert signature.is_collapsed()


def test_paper_two_expert_minority_baseline_separates_deadlock_from_skew() -> None:
    """The source study's 10%-of-tokens baseline against a 50% uniform share.

    Deep deadlock is reported at 7.33% and a skewed-but-live layer at 26.80%;
    expressed against uniform those are 0.1466 and 0.5360, so the ratio
    baseline separates them exactly where the study does.
    """
    deadlocked = _signature([7.33, 92.67])
    skewed = _signature([26.80, 73.20])
    healthy = _signature([49.80, 50.20])
    assert deadlocked.normalized_minority_share == pytest.approx(0.1466, abs=1e-4)
    assert deadlocked.is_collapsed()
    assert skewed.normalized_minority_share == pytest.approx(0.5360, abs=1e-4)
    assert not skewed.is_collapsed()
    assert healthy.normalized_minority_share == pytest.approx(0.996, abs=1e-3)
    assert not healthy.is_collapsed()


def test_wide_router_collapse_is_invisible_to_top1_monopoly() -> None:
    """Why the minority side is a separate estimator, not a restatement.

    Two of 256 experts carry every routed slot. The dominant share is 0.5 --
    far under the 0.90 single-expert deadlock threshold the health module uses
    -- while 254 experts are dead.
    """
    counts = [0.0] * 256
    counts[3] = 500.0
    counts[17] = 500.0
    signature = _signature(counts)
    assert signature.top1_share == pytest.approx(0.5)
    assert signature.top1_share < 0.90
    assert signature.dead_expert_fraction == pytest.approx(254 / 256)
    assert signature.is_collapsed()


def test_underused_generalizes_the_minority_baseline_beyond_dead_experts() -> None:
    """Experts alive but under the baseline count as underused, not dead."""
    # Uniform share 0.25; the baseline floor is 0.2 * 0.25 = 0.05 of the slots.
    signature = _signature([1.0, 4.0, 95.0, 100.0])
    assert signature.dead_expert_fraction == 0.0
    assert signature.underused_expert_fraction == pytest.approx(0.5)
    assert signature.is_collapsed()


def test_health_ratio_is_the_configured_boundary() -> None:
    """The flag follows the configured ratio, not a hardcoded absolute share."""
    signature = _signature([1.0, 3.0, 4.0, 4.0])  # normalized minority = 0.3333
    assert not signature.is_collapsed(COLLAPSE_HEALTH_RATIO)
    assert signature.is_collapsed(0.5)


def test_counts_reject_ids_outside_the_declared_expert_range() -> None:
    with pytest.raises(ValueError, match="routed expert ids"):
        router_expert_counts(torch.tensor([[0, 4]]), num_experts=4)
    with pytest.raises(ValueError, match="at least two experts"):
        router_expert_counts(torch.tensor([[0, 0]]), num_experts=1)


def test_empty_selection_reports_zero_load_without_dividing_by_zero() -> None:
    signature = router_load_signatures(torch.zeros(1, 4))[0]
    assert signature.routed_slots == 0
    assert signature.minority_share == 0.0
    assert signature.top1_share == 0.0
    assert signature.dead_expert_fraction == 1.0


def test_per_head_selections_are_reduced_independently() -> None:
    """A multi-head router is several routing surfaces, not one average."""
    healthy_head = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
    collapsed_head = torch.zeros_like(healthy_head)
    selection = torch.stack([healthy_head, collapsed_head], dim=0)
    signatures = selection_load_signatures(selection, num_experts=4)
    assert len(signatures) == 2
    assert not signatures[0].is_collapsed()
    assert signatures[1].is_collapsed()
    assert signatures[1].dead_expert_fraction == pytest.approx(0.75)


def test_router_keys_carry_layer_prefix_and_head_index() -> None:
    assert router_keys("blocks.4.moe", 3) == (
        "blocks.4.moe#head0",
        "blocks.4.moe#head1",
        "blocks.4.moe#head2",
    )


# --- Cross-step collapse state ---------------------------------------------


def test_collapse_streak_accumulates_and_resets_on_recovery() -> None:
    tracker = RoutingCollapseTracker()
    collapsed = {"L0": _signature([32.0, 0.0, 0.0, 0.0])}
    healthy = {"L0": _signature([8.0, 8.0, 8.0, 8.0])}
    assert tracker.update(collapsed)["moe_max_collapse_duration"] == 1.0
    assert tracker.update(collapsed)["moe_max_collapse_duration"] == 2.0
    recovered = tracker.update(healthy)
    assert recovered["moe_max_collapse_duration"] == 0.0
    assert recovered["moe_collapsed_router_fraction"] == 0.0


def test_rebound_requires_a_sustained_collapse_before_recovery() -> None:
    """The study's single observed self-recovery, distinguished from jitter."""
    collapsed = {"L0": _signature([32.0, 0.0, 0.0, 0.0])}
    healthy = {"L0": _signature([8.0, 8.0, 8.0, 8.0])}

    sustained = RoutingCollapseTracker()
    sustained.update(collapsed)
    sustained.update(collapsed)
    assert sustained.update(healthy)["moe_collapse_rebound_count"] == 1.0

    jitter = RoutingCollapseTracker()
    jitter.update(collapsed)
    assert jitter.update(healthy)["moe_collapse_rebound_count"] == 0.0


def test_plateau_flags_a_collapsed_router_that_stopped_moving() -> None:
    """The study's deadlocked layer that no balance pressure moved."""
    frozen = {"L0": _signature([1.0, 199.0])}
    tracker = RoutingCollapseTracker()
    for _step in range(COLLAPSE_PLATEAU_WINDOW - 1):
        assert tracker.update(frozen)["moe_collapse_plateau_fraction"] == 0.0
    assert tracker.update(frozen)["moe_collapse_plateau_fraction"] == 1.0


def test_moving_collapsed_router_is_not_a_plateau() -> None:
    tracker = RoutingCollapseTracker()
    metrics = {}
    for step in range(COLLAPSE_PLATEAU_WINDOW):
        minority = 1.0 + 2.0 * step
        metrics = tracker.update({"L0": _signature([minority, 400.0 - minority])})
    assert metrics["moe_collapsed_router_fraction"] == 1.0
    assert metrics["moe_collapse_plateau_fraction"] == 0.0


def test_unobserved_routers_keep_their_state() -> None:
    """A partial observation must not read as recovery for the missing routers."""
    tracker = RoutingCollapseTracker()
    collapsed = _signature([32.0, 0.0, 0.0, 0.0])
    tracker.update({"L0": collapsed, "L1": collapsed})
    tracker.update({"L0": collapsed})
    metrics = tracker.update({"L1": collapsed})
    # L0 was not observed on the third step, so its streak of two survives
    # untouched while L1 advances to two of its own.
    assert metrics["moe_max_collapse_duration"] == 2.0
    assert metrics["moe_collapsed_router_count"] == 1.0


def test_worst_router_is_reported_next_to_the_mean() -> None:
    tracker = RoutingCollapseTracker()
    metrics = tracker.update(
        {
            "L0": _signature([8.0, 8.0, 8.0, 8.0]),
            "L1": _signature([32.0, 0.0, 0.0, 0.0]),
        }
    )
    assert metrics["moe_worst_normalized_minority_share"] == 0.0
    assert metrics["moe_normalized_minority_share"] == pytest.approx(0.5)
    assert metrics["moe_collapsed_router_fraction"] == pytest.approx(0.5)
    assert metrics["moe_dead_expert_fraction"] == pytest.approx(0.375)


def test_empty_observation_emits_nothing() -> None:
    assert RoutingCollapseTracker().update({}) == {}


def test_non_signature_input_fails_closed() -> None:
    with pytest.raises(TypeError, match="RouterLoadSignature"):
        RoutingCollapseTracker().update({"L0": 0.5})


def test_emitted_keys_match_the_declared_metric_surface() -> None:
    metrics = RoutingCollapseTracker().update({"L0": _signature([8.0, 8.0])})
    assert set(metrics) == set(collapse_metric_keys())


# --- Bounded memory --------------------------------------------------------


@pytest.mark.parametrize("tokens", [16, 4096, 262144])
def test_state_and_metric_size_are_independent_of_token_count(tokens: int) -> None:
    """Aggregate cost is a function of routers and steps, never of tokens."""
    torch.manual_seed(0)
    selection = torch.randint(0, 8, (2, tokens, 2))
    signatures = selection_load_signatures(selection, num_experts=8)
    keyed = dict(zip(router_keys("blocks.0.moe", len(signatures)), signatures))
    tracker = RoutingCollapseTracker()
    for _step in range(3):
        metrics = tracker.update(keyed)
    assert tracker.tracked_routers == 2
    assert tracker.state_size() == 2 + 2 * 3
    assert len(metrics) == len(collapse_metric_keys())


def test_plateau_history_is_capped_by_the_window() -> None:
    tracker = RoutingCollapseTracker()
    signature = _signature([1.0, 199.0])
    for _step in range(COLLAPSE_PLATEAU_WINDOW * 5):
        tracker.update({"L0": signature})
    assert tracker.state_size() == 1 + COLLAPSE_PLATEAU_WINDOW


# --- MAGI-2 wiring ---------------------------------------------------------


_D_HEAD = 8


def _reduced_moe_layer(*, num_heads: int = 2, num_experts: int = 4):
    from mirai.vendors.magi2_preview.model.magi2_preview import (
        CoreMultiHeadMoE,
        CoreMultiHeadMoEConfig,
    )

    torch.manual_seed(0)
    module = CoreMultiHeadMoE(
        CoreMultiHeadMoEConfig(
            hidden_size=num_heads * _D_HEAD,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=2,
            expert_intermediate_size=12,
            num_layers=1,
            params_dtype=torch.float32,
            score_func="sigmoid",
            route_norm=True,
            route_scale=4.9,
        )
    )
    with torch.no_grad():
        for tensor in (module.gate, module.W_gate, module.W_up, module.W_down):
            tensor.normal_(std=0.1)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    container = torch.nn.Module()
    container.moe_mlp = module
    return container, module


def _bare_pipeline(container):
    from mirai.core.models.magi2_preview.pipeline import Magi2PreviewPipeline

    pipeline = Magi2PreviewPipeline.__new__(Magi2PreviewPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = container
    pipeline.runtime_config = types.SimpleNamespace(
        arch_config=types.SimpleNamespace(
            moe_config=types.SimpleNamespace(num_experts=4, top_k=2)
        )
    )
    return pipeline


def test_disabled_telemetry_attaches_nothing_and_reports_no_metrics() -> None:
    """Default off: no tap object, no seam change, no diagnostics, no claim."""
    from mirai.core.models.magi2_preview.routing_collapse import (
        MAGI2_COLLAPSE_TAP_ATTR,
    )

    container, module = _reduced_moe_layer()
    pipeline = _bare_pipeline(container)
    pipeline._routing_collapse = None
    assert not hasattr(module, MAGI2_COLLAPSE_TAP_ATTR)
    assert module._mirai_moe_kernel_backend is None
    assert pipeline.get_training_diagnostics() == {}
    assert not pipeline.get_sparse_moe_capabilities().emits_router_metrics


def test_enabled_tap_produces_per_layer_per_head_aggregates() -> None:
    from mirai.core.models.magi2_preview.routing_collapse import (
        MAGI2_COLLAPSE_TAP_ATTR,
        Magi2RoutingCollapseTap,
    )

    container, module = _reduced_moe_layer(num_heads=3)
    pipeline = _bare_pipeline(container)
    pipeline._enable_routing_collapse_telemetry()
    assert isinstance(getattr(module, MAGI2_COLLAPSE_TAP_ATTR), Magi2RoutingCollapseTap)
    assert module._mirai_moe_kernel_backend is getattr(
        module, MAGI2_COLLAPSE_TAP_ATTR
    )
    assert pipeline.get_sparse_moe_capabilities().emits_router_metrics

    state = pipeline._routing_collapse
    module._forward_impl(torch.randn(9, 3 * _D_HEAD))
    assert state.observer.router_order == (
        "moe_mlp#head0",
        "moe_mlp#head1",
        "moe_mlp#head2",
    )
    metrics = state.collect()
    assert set(collapse_metric_keys()).issubset(metrics)
    # The depth-banded deadlock counters of the health module ride the same
    # observation rather than requiring a second traversal.
    assert "moe_deadlocked_layer_count" in metrics
    assert state.collapse_tracker.tracked_routers == 3
    assert 0.0 <= metrics["moe_normalized_minority_share"] <= 1.0


def test_tap_does_not_change_execution() -> None:
    """Telemetry is observation only: the tapped forward is bit-identical."""
    container, module = _reduced_moe_layer()
    hidden = torch.randn(7, 2 * _D_HEAD)
    reference = module._forward_impl(hidden)

    pipeline = _bare_pipeline(container)
    pipeline._enable_routing_collapse_telemetry()
    tapped = module._forward_impl(hidden)
    assert torch.equal(reference, tapped)


def test_tap_wraps_and_restores_a_configured_execution_backend() -> None:
    from mirai.core.models.magi2_preview.grouped_moe import (
        Magi2GroupedMoEBackend,
        Magi2GroupedMoEPlan,
        attach_grouped_moe_backend,
    )
    from mirai.core.models.magi2_preview.routing_collapse import (
        MAGI2_COLLAPSE_TAP_ATTR,
        attach_routing_collapse_tap,
        detach_routing_collapse_tap,
    )

    container, module = _reduced_moe_layer()
    backend = Magi2GroupedMoEBackend(
        Magi2GroupedMoEPlan(forward_backend="bmm", dx_backend="bmm")
    )
    attach_grouped_moe_backend(container, backend)

    pipeline = _bare_pipeline(container)
    pipeline._enable_routing_collapse_telemetry()
    tap = getattr(module, MAGI2_COLLAPSE_TAP_ATTR)
    assert tap.inner is backend

    hidden = torch.randn(6, 2 * _D_HEAD)
    tapped = module._forward_impl(hidden)
    grouped = backend.execute(module, *_route_reference(module, hidden))
    assert torch.allclose(tapped, grouped.reshape(tapped.shape))
    assert pipeline._routing_collapse.observer.take()

    detach_routing_collapse_tap(container)
    assert module._mirai_moe_kernel_backend is backend
    assert not hasattr(module, MAGI2_COLLAPSE_TAP_ATTR)

    # A second attach over an existing tap rebinds rather than nesting taps.
    observer = pipeline._routing_collapse.observer
    attach_routing_collapse_tap(container, observer)
    attach_routing_collapse_tap(container, observer)
    assert getattr(module, MAGI2_COLLAPSE_TAP_ATTR).inner is backend


def _route_reference(module, hidden):
    x_heads = hidden.view(-1, module.num_heads, module.d_head)
    topk_probs, topk_indices = module._route(x_heads)
    return x_heads, topk_probs, topk_indices


def test_policy_reattach_keeps_exactly_one_tap_over_the_new_backend() -> None:
    from mirai.core.models.magi2_preview.routing_collapse import (
        MAGI2_COLLAPSE_TAP_ATTR,
        Magi2RoutingCollapseTap,
    )
    from mirai.core.moe.runtime.specs import MoEOptimizationPolicy

    container, module = _reduced_moe_layer()
    pipeline = _bare_pipeline(container)
    pipeline._expert_stores = {}
    pipeline._moe_optimization_policy = MoEOptimizationPolicy()
    pipeline._enable_routing_collapse_telemetry()

    pipeline.configure_moe_optimization_policy(
        MoEOptimizationPolicy(kernel_backend="grouped", moe_gemm_backend="bmm")
    )
    tap = module._mirai_moe_kernel_backend
    assert isinstance(tap, Magi2RoutingCollapseTap)
    assert not isinstance(tap.inner, Magi2RoutingCollapseTap)
    assert tap.inner is not None

    pipeline.configure_moe_optimization_policy(MoEOptimizationPolicy())
    tap = module._mirai_moe_kernel_backend
    assert isinstance(tap, Magi2RoutingCollapseTap)
    assert tap.inner is None
    assert getattr(module, MAGI2_COLLAPSE_TAP_ATTR) is tap


def test_silenced_observer_records_nothing() -> None:
    """Eval forwards pay no reduction and advance no cross-step state."""
    container, module = _reduced_moe_layer()
    pipeline = _bare_pipeline(container)
    pipeline._enable_routing_collapse_telemetry()
    state = pipeline._routing_collapse

    state.observer.set_enabled(False)
    module._forward_impl(torch.randn(5, 2 * _D_HEAD))
    assert state.observer.take() == {}
    assert state.collect() == {}
    assert state.collapse_tracker.tracked_routers == 0
