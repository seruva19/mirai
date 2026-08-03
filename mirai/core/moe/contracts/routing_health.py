"""MoE routing-health diagnostic contracts.

Unit coverage for the three opt-in alarms (homogenization cosine proxy,
per-depth single-expert deadlock duration, bf16 router-update underflow),
config plumbing, metric surfacing, and end-to-end gating (default off -> absent,
on -> present) on the LingBot pipeline.
"""

from __future__ import annotations

# Colocated behavioral contract for router diagnostics and capacity reports.

import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mirai.config.schema import TrainingConfig
from mirai.config.schema import all_config_keys
from mirai.core.moe.monitoring.health import (
    DEADLOCK_DEPTH_BANDS,
    DEADLOCK_MONOPOLY_THRESHOLD,
    DeadlockTracker,
    expert_output_cossim,
    router_underflow_fraction,
)
from mirai.core.moe.monitoring.fisher import (
    fisher_rao_distance_to_uniform,
    summarize_fisher_specialization,
)
from mirai.core.moe.monitoring.preemptive import (
    AttentionQKState,
    LowRankProjectionState,
    PreemptiveAttentionMonitor,
    attention_delta2_singular_values,
    per_token_router_entropy,
    router_conditioning_ratio,
    router_weight_similarity,
    singular_spectrum_effective_rank,
    snapshot_low_rank_projection,
)
from mirai.core.moe.monitoring.agreement import (
    RoutingAgreementAccumulator,
    RoutingSelection,
    RoutingSelectionCapture,
    RoutingSelectionTarget,
    build_routing_mode_agreement_evidence,
    compare_routing_capture_pairs,
    compare_routing_selections,
    compare_router_selections,
    selection_margin,
    topk_overlap_fraction,
    topk_set_agreement,
    validate_routing_mode_agreement_evidence,
)
from mirai.core.moe.monitoring.capacity import MoECapacitySpec, estimate_moe_capacity
from mirai.core.moe.monitoring.report import build_router_health_report
from mirai.core.training.observability.metrics import build_step_metrics
from mirai.core.training.calibration.routing_agreement import (
    run_routing_mode_agreement_session,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_DEPTH_BAND_KEYS = tuple(
    f"{prefix}_depth_{band}"
    for prefix in (
        "moe_deadlocked_layer_count",
        "moe_max_deadlock_duration",
    )
    for band in DEADLOCK_DEPTH_BANDS
)
_HEALTH_KEYS = (
    "moe_expert_output_cossim",
    "moe_fisher_specialization_index",
    "moe_fisher_specialization_fraction",
    "moe_fisher_specialization_min_layer",
    "moe_fisher_specialization_max_layer",
    "moe_fisher_specialization_layer_count",
    "moe_router_weight_similarity",
    "moe_router_conditioning_ratio",
    "moe_router_per_token_entropy",
    "moe_router_per_token_entropy_fraction",
    "moe_router_mechanism_layer_count",
    "moe_router_conditioning_layer_count",
    "moe_max_deadlock_duration",
    "moe_deadlocked_layer_count",
    *_DEPTH_BAND_KEYS,
)
_MARGIN_KEYS = (
    "moe_selection_margin_p05",
    "moe_selection_margin_min",
)
_ATTENTION_MONITOR_KEYS = (
    "moe_attention_qk_delta2_effective_rank",
    "moe_attention_qk_delta2_effective_rank_min",
    "moe_attention_qk_delta2_effective_rank_max",
    "moe_attention_qk_delta2_spectral_entropy",
    "moe_attention_qk_delta2_head_count",
    "moe_attention_qk_delta2_layer_count",
)


@unittest.skipIf(torch is None, "torch not installed")
class RoutingSelectionAgreementTests(unittest.TestCase):
    def test_pr2_overlap_uses_intersection_over_k(self) -> None:
        """Protects PR2's overlap/k metric from Jaccard substitution."""
        reference = torch.tensor([[0, 1], [0, 1]])
        candidate = torch.tensor([[0, 1], [0, 2]])
        self.assertEqual(
            topk_overlap_fraction(reference, candidate, num_experts=3),
            0.75,
        )
        self.assertAlmostEqual(
            topk_set_agreement(reference, candidate, num_experts=3),
            2.0 / 3.0,
        )

    def test_variable_cardinality_reports_churn_without_false_topk_overlap(self) -> None:
        """Protects active-set comparison when training changes route count."""
        reference = RoutingSelection.from_tensors(
            torch.tensor([[0, 1], [1, 2]]),
            num_experts=3,
        )
        candidate = RoutingSelection.from_tensors(
            torch.tensor([[0, 1], [1, 2]]),
            active_mask=torch.tensor([[True, False], [True, True]]),
            num_experts=3,
        )
        report = compare_routing_selections(reference, candidate)
        self.assertEqual(report.changed_token_fraction, 0.5)
        self.assertEqual(report.equal_cardinality_token_fraction, 0.5)
        self.assertEqual(report.matched_cardinality_overlap, 1.0)
        self.assertEqual(report.to_dict()["mean_jaccard"], 0.75)

    def test_capture_pairs_join_same_target_and_invocation(self) -> None:
        """Protects router-hook capture before family runtime state is cleared."""

        class _Router(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.last_top_indices = None
                self.last_route_active_mask = None

            def forward(self, indices):
                self.last_top_indices = indices
                self.last_route_active_mask = torch.ones_like(
                    indices, dtype=torch.bool
                )
                return indices

        router = _Router()
        targets = {
            "layer.router": RoutingSelectionTarget(
                name="layer.router",
                router=router,
                num_experts=3,
            )
        }
        with RoutingSelectionCapture(targets) as capture:
            router(torch.tensor([[0, 1], [1, 2]]))
        reference = capture.snapshots()
        with RoutingSelectionCapture(targets) as capture:
            router(torch.tensor([[0, 1], [0, 2]]))
        candidate = capture.snapshots()
        accumulators = {"layer.router": RoutingAgreementAccumulator()}
        compare_routing_capture_pairs(reference, candidate, accumulators)
        report = accumulators["layer.router"].report()
        self.assertEqual(report["router_invocations"], 1)
        self.assertEqual(report["changed_token_fraction"], 0.5)
        self.assertEqual(
            report["matched_cardinality_deviation_histogram"],
            {"0": 1, "1": 1},
        )

    def test_evidence_requires_lineage_and_declares_pairing(self) -> None:
        """Protects versioned evidence and its exact comparison semantics."""
        accumulator = RoutingAgreementAccumulator()
        accumulator.add(
            compare_routing_selections(
                RoutingSelection.from_tensors(
                    torch.tensor([[0, 1]]),
                    num_experts=3,
                ),
                RoutingSelection.from_tensors(
                    torch.tensor([[0, 2]]),
                    num_experts=3,
                ),
            )
        )
        evidence = build_routing_mode_agreement_evidence(
            {"layer.router": accumulator},
            calibration_steps=1,
            dataset_snapshot_id="dataset",
            model_snapshot_id="model",
            config_snapshot_id="config",
        )
        validate_routing_mode_agreement_evidence(evidence)
        self.assertEqual(
            evidence["comparison"]["pairing"],
            "same_batch_noise_timestep_token_layer_and_router_invocation",
        )
        with self.assertRaisesRegex(ValueError, "lineage"):
            build_routing_mode_agreement_evidence(
                {"layer.router": accumulator},
                calibration_steps=1,
                dataset_snapshot_id="",
                model_snapshot_id="model",
                config_snapshot_id="config",
            )

    def test_session_pairs_rng_and_restores_isolated_state(self) -> None:
        """Protects same-input pairing and non-perturbation of session state."""

        class _Router(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.last_top_indices = None
                self.last_route_active_mask = None

            def forward(self, _value):
                indices = (
                    torch.tensor([[0, 1], [1, 2]])
                    if self.training
                    else torch.tensor([[0, 1], [0, 2]])
                )
                self.last_top_indices = indices
                self.last_route_active_mask = torch.ones_like(
                    indices, dtype=torch.bool
                )
                return indices

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = _Router()

        class _Pipeline:
            def __init__(self) -> None:
                self.model = _Model()

            def get_training_model(self):
                return self.model

        class _Policies:
            def __init__(self) -> None:
                self.count = 0

            def state_dict(self):
                return {"count": self.count}

            def load_state_dict(self, state):
                self.count = int(state["count"])

        class _Trainer:
            def __init__(self) -> None:
                self.pipeline = _Pipeline()
                self.training_policies = _Policies()
                self.sampling = 0
                self.random_by_mode = {True: [], False: []}

            def _sampling_state_dict(self):
                return {"sampling": self.sampling}

            def _load_sampling_state_dict(self, state):
                self.sampling = int(state["sampling"])

            def compute_loss(self, _batch, *, training):
                self.random_by_mode[bool(training)].append(
                    float(torch.rand(()).item())
                )
                self.sampling += 1
                self.training_policies.count += 1
                self.pipeline.model.router(torch.zeros(()))
                return torch.zeros(()), {}

        class _Provider:
            def supports_routing_mode_agreement_evidence(self, _config):
                return True

            def build_routing_mode_agreement_targets(self, pipeline):
                return {
                    "block.router": RoutingSelectionTarget(
                        name="block.router",
                        router=pipeline.model.router,
                        num_experts=3,
                    )
                }

        trainer = _Trainer()
        session_rng = random.Random(17)
        initial_session_rng = session_rng.getstate()
        initial_torch_rng = torch.get_rng_state().clone()
        session = SimpleNamespace(
            config=SimpleNamespace(
                model=SimpleNamespace(
                    type="fake",
                    params=SimpleNamespace(
                        moe_routing_agreement_evidence="report"
                    ),
                )
            ),
            trainer=trainer,
            manifest=SimpleNamespace(
                dataset_snapshot_id="dataset",
                model_snapshot_id="model",
                config_snapshot_id="config",
            ),
            rng=session_rng,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "agreement.json"
            with (
                patch(
                    "mirai.core.training.calibration.routing_agreement."
                    "get_model_family_provider",
                    return_value=_Provider(),
                ),
                patch(
                    "mirai.core.training.calibration.routing_agreement."
                    "resolve_step_sampling_context",
                    return_value=object(),
                ),
                patch(
                    "mirai.core.training.calibration.routing_agreement."
                    "_build_training_batch_factory",
                    return_value=lambda _step: {},
                ),
            ):
                report = run_routing_mode_agreement_session(
                    session,
                    output_path=output,
                    calibration_steps=2,
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(
            trainer.random_by_mode[True],
            trainer.random_by_mode[False],
        )
        self.assertEqual(trainer.sampling, 0)
        self.assertEqual(trainer.training_policies.count, 0)
        self.assertEqual(session_rng.getstate(), initial_session_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), initial_torch_rng))
        self.assertTrue(trainer.pipeline.model.training)
        self.assertEqual(report.overall["changed_token_fraction"], 0.5)
        validate_routing_mode_agreement_evidence(payload)

    def test_identical_scores_report_no_selection_churn(self) -> None:
        """Protects identity agreement and zero changed-token fraction."""
        scores = torch.tensor([[4.0, 3.0, 2.0], [1.0, 3.0, 2.0]])
        report = compare_router_selections(
            scores, scores, top_k=2, num_experts=3
        )
        self.assertEqual(report.agreement, 1.0)
        self.assertEqual(report.changed_token_fraction, 0.0)

    def test_one_changed_token_matches_exact_set_metrics(self) -> None:
        """Protects exact changed-token fraction and token-wise Jaccard mean."""
        reference = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
        candidate = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 2.0, 3.0, 1.0]])
        report = compare_router_selections(
            reference, candidate, top_k=2, num_experts=4
        )
        self.assertEqual(report.changed_token_fraction, 0.5)
        self.assertAlmostEqual(report.agreement, 2.0 / 3.0)

    def test_different_expert_counts_require_shared_numbering(self) -> None:
        """Protects rejection of comparisons without shared expert numbering."""
        with self.assertRaisesRegex(ValueError, "shared expert numbering"):
            compare_router_selections(
                torch.randn(2, 4),
                torch.randn(2, 5),
                top_k=2,
                num_experts=4,
            )

    def test_top_k_must_leave_an_unselected_expert(self) -> None:
        """Protects the valid top-k interval [1, num_experts-1]."""
        scores = torch.randn(2, 4)
        for top_k in (0, 4):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    compare_router_selections(
                        scores,
                        scores,
                        top_k=top_k,
                        num_experts=4,
                    )

    def test_margins_are_measured_from_reference_scores(self) -> None:
        """Protects reference-side fragility regardless of candidate margins."""
        reference = torch.tensor([[4.0, 3.0, 2.0, 0.0]])
        candidate = torch.tensor([[400.0, 300.0, 0.0, -100.0]])
        report = compare_router_selections(
            reference, candidate, top_k=2, num_experts=4
        )
        self.assertEqual(report.margin_p05, 1.0)
        self.assertEqual(report.margin_min, 1.0)

    def test_permuted_slots_preserve_agreement(self) -> None:
        """Protects set semantics when unsorted top-k permutes selected slots."""
        reference = torch.tensor([[0, 2, 1], [3, 1, 2]])
        candidate = torch.tensor([[1, 0, 2], [2, 3, 1]])
        self.assertEqual(
            topk_set_agreement(reference, candidate, num_experts=4),
            topk_set_agreement(reference.flip(1), candidate.flip(1), num_experts=4),
        )

    def test_self_agreement_is_exactly_one(self) -> None:
        """Protects the identity case as an exact stability baseline."""
        indices = torch.tensor([[0, 2], [1, 3]])
        self.assertEqual(topk_set_agreement(indices, indices, num_experts=4), 1.0)

    def test_disjoint_selections_are_exactly_zero(self) -> None:
        """Protects the lower bound for complete route churn."""
        reference = torch.tensor([[0, 1], [0, 1]])
        candidate = torch.tensor([[2, 3], [2, 3]])
        self.assertEqual(
            topk_set_agreement(reference, candidate, num_experts=4), 0.0
        )

    def test_duplicate_ids_have_set_semantics(self) -> None:
        """Protects against repeated slots inflating intersections or unions."""
        candidate = torch.tensor([[0, 1, 2]])
        repeated = torch.tensor([[0, 0, 1]])
        deduplicated_with_padding = torch.tensor([[0, 1, 1]])
        self.assertEqual(
            topk_set_agreement(repeated, candidate, num_experts=3),
            topk_set_agreement(
                deduplicated_with_padding, candidate, num_experts=3
            ),
        )

    def test_partial_overlap_matches_hand_computed_jaccard(self) -> None:
        """Protects the exact intersection-over-union definition."""
        reference = torch.tensor([[0, 1, 2]])
        candidate = torch.tensor([[1, 2, 3]])
        self.assertEqual(
            topk_set_agreement(reference, candidate, num_experts=4), 0.5
        )

    def test_selection_margin_detects_ties_and_strict_order(self) -> None:
        """Protects the top-k boundary gap for tied and separated scores."""
        margins = selection_margin(
            torch.tensor([[4.0, 3.0, 3.0, 1.0], [4.0, 3.0, 2.0, 1.0]]),
            top_k=2,
        )
        self.assertEqual(float(margins[0]), 0.0)
        self.assertGreater(float(margins[1]), 0.0)


class CapacityAndReportTests(unittest.TestCase):
    def test_capacity_estimator_bounds_unique_experts_and_residency(self) -> None:
        estimate = estimate_moe_capacity(
            MoECapacitySpec(
                num_layers=2,
                num_experts=8,
                experts_per_token=2,
                hidden_size=16,
                expert_intermediate_size=8,
                tokens_per_step=4,
                weight_bits=4,
                adapter_rank=2,
                resident_experts_per_layer=2,
            )
        )
        self.assertLessEqual(estimate.expected_unique_experts_per_layer, 8.0)
        self.assertEqual(estimate.maximum_unique_experts_per_layer, 8)
        self.assertGreater(estimate.total_expert_weight_bytes, 0)
        self.assertGreater(estimate.maximum_streamed_weight_bytes_per_step, 0)

    def test_router_report_preserves_coverage_and_missing_metrics(self) -> None:
        report = build_router_health_report(
            [
                {"moe_routing_entropy": 0.5, "moe_top1_monopoly": 0.8},
                {"moe_routing_entropy": 0.7},
            ]
        )
        entropy = report["metrics"]["moe_routing_entropy"]
        self.assertEqual(entropy["coverage"], 1.0)
        self.assertAlmostEqual(entropy["mean"], 0.6)
        self.assertIn("moe_router_underflow_fraction", report["missing_metrics"])

    def test_router_report_summarizes_depth_bands(self) -> None:
        report = build_router_health_report(
            [
                {
                    "moe_deadlocked_layer_count_depth_q1": 2.0,
                    "moe_max_deadlock_duration_depth_q1": 3.0,
                },
                {
                    "moe_deadlocked_layer_count_depth_q1": 1.0,
                    "moe_max_deadlock_duration_depth_q1": 4.0,
                },
            ]
        )
        count = report["metrics"]["moe_deadlocked_layer_count_depth_q1"]
        duration = report["metrics"]["moe_max_deadlock_duration_depth_q1"]
        self.assertEqual(count["coverage"], 1.0)
        self.assertEqual(count["mean"], 1.5)
        self.assertEqual(duration["max"], 4.0)


@unittest.skipIf(torch is None, "torch not installed")
class ExpertOutputCossimTests(unittest.TestCase):
    def test_homogenized_experts_saturate_to_one(self) -> None:
        base = torch.rand(256, 1)
        # Every expert responds identically to the same tokens -> collapse.
        columns = base.repeat(1, 6)
        value = expert_output_cossim([columns])
        self.assertIsNotNone(value)
        self.assertGreater(value, 0.99)

    def test_diverse_experts_stay_low(self) -> None:
        torch.manual_seed(0)
        # Independent per-expert responses -> mean-centered columns ~orthogonal.
        columns = torch.randn(512, 4)
        value = expert_output_cossim([columns])
        self.assertIsNotNone(value)
        self.assertLess(abs(value), 0.2)

    def test_anti_correlated_experts_go_negative(self) -> None:
        col = torch.rand(128, 1)
        columns = torch.cat([col, -col], dim=1)
        value = expert_output_cossim([columns])
        self.assertIsNotNone(value)
        self.assertLess(value, -0.99)

    def test_layer_mean_over_multiple_layers(self) -> None:
        homo = torch.rand(128, 1).repeat(1, 4)
        value = expert_output_cossim([homo, homo])
        self.assertGreater(value, 0.99)

    def test_none_when_not_measurable(self) -> None:
        self.assertIsNone(expert_output_cossim([]))
        self.assertIsNone(expert_output_cossim([torch.rand(1, 4)]))  # one token
        self.assertIsNone(expert_output_cossim([torch.rand(10, 1)]))  # one expert
        self.assertIsNone(expert_output_cossim([None]))

    def test_sampling_bounds_cost_and_is_finite(self) -> None:
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(7)
        value = expert_output_cossim(
            [torch.randn(4096, 8)], sample_slots=128, generator=gen
        )
        self.assertIsNotNone(value)
        self.assertTrue(torch.isfinite(torch.tensor(value)))


@unittest.skipIf(torch is None, "torch not installed")
class FisherSpecializationTests(unittest.TestCase):
    def test_uniform_and_vertex_match_equation_four_bounds(self) -> None:
        uniform_index, uniform_fraction = fisher_rao_distance_to_uniform(
            torch.full((7, 4), 0.25)
        )
        self.assertEqual(float(uniform_index.item()), 0.0)
        self.assertEqual(float(uniform_fraction.item()), 0.0)

        vertex_index, vertex_fraction = fisher_rao_distance_to_uniform(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(5, 1)
        )
        expected = 2.0 * torch.acos(torch.tensor(0.5))
        torch.testing.assert_close(vertex_index, expected)
        torch.testing.assert_close(vertex_fraction, torch.tensor(1.0))

    def test_empirical_marginal_is_invariant_to_rows_and_expert_labels(self) -> None:
        probabilities = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float32
        )
        reference = fisher_rao_distance_to_uniform(probabilities)
        repeated = fisher_rao_distance_to_uniform(
            probabilities.repeat_interleave(3, dim=0)[:, [2, 0, 1]]
        )
        torch.testing.assert_close(reference[0], repeated[0])
        torch.testing.assert_close(reference[1], repeated[1])

    def test_layer_summary_reports_mean_and_extrema(self) -> None:
        summary = summarize_fisher_specialization(
            [
                torch.full((3, 4), 0.25),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            ]
        )
        self.assertIsNotNone(summary)
        metrics = summary.to_metrics()
        self.assertEqual(metrics["moe_fisher_specialization_layer_count"], 2.0)
        self.assertEqual(metrics["moe_fisher_specialization_min_layer"], 0.0)
        self.assertAlmostEqual(
            metrics["moe_fisher_specialization_fraction"], 0.5, places=6
        )
        self.assertEqual(summarize_fisher_specialization([]), None)

    def test_invalid_probability_populations_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fisher_rao_distance_to_uniform(torch.tensor([[1.1, -0.1]]))
        with self.assertRaisesRegex(ValueError, "positive mass"):
            fisher_rao_distance_to_uniform(torch.zeros(2, 3))
        with self.assertRaisesRegex(ValueError, "at least two experts"):
            fisher_rao_distance_to_uniform(torch.ones(2, 1))


class DeadlockTrackerTests(unittest.TestCase):
    def test_streak_accumulates_and_reports_max_and_count(self) -> None:
        tracker = DeadlockTracker()
        first = tracker.update({"L0": 0.95, "L1": 0.40})
        self.assertEqual(first["moe_max_deadlock_duration"], 1.0)
        self.assertEqual(first["moe_deadlocked_layer_count"], 1.0)
        second = tracker.update({"L0": 0.97, "L1": 0.40})
        self.assertEqual(second["moe_max_deadlock_duration"], 2.0)
        self.assertEqual(second["moe_deadlocked_layer_count"], 1.0)

    def test_recovery_resets_streak(self) -> None:
        tracker = DeadlockTracker()
        tracker.update({"L0": 0.95})
        tracker.update({"L0": 0.95})
        recovered = tracker.update({"L0": 0.20})
        self.assertEqual(recovered["moe_max_deadlock_duration"], 0.0)
        self.assertEqual(recovered["moe_deadlocked_layer_count"], 0.0)

    def test_independent_layers_tracked_separately(self) -> None:
        tracker = DeadlockTracker()
        tracker.update({"L0": 0.95, "L1": 0.10})
        out = tracker.update({"L0": 0.10, "L1": 0.95})
        # L0 reset to 0, L1 at 1 -> max 1, one deadlocked layer this step.
        self.assertEqual(out["moe_max_deadlock_duration"], 1.0)
        self.assertEqual(out["moe_deadlocked_layer_count"], 1.0)

    def test_threshold_is_inclusive(self) -> None:
        tracker = DeadlockTracker()
        out = tracker.update({"L0": DEADLOCK_MONOPOLY_THRESHOLD})
        self.assertEqual(out["moe_deadlocked_layer_count"], 1.0)

    def test_depth_quartiles_follow_explicit_model_order(self) -> None:
        tracker = DeadlockTracker()
        layer_order = tuple(f"block.{index}" for index in range(8))
        monopolies = {
            layer: (0.95 if index in {0, 1, 6, 7} else 0.10)
            for index, layer in reversed(tuple(enumerate(layer_order)))
        }
        out = tracker.update(monopolies, layer_order=layer_order)
        self.assertEqual(out["moe_deadlocked_layer_count_depth_q1"], 2.0)
        self.assertEqual(out["moe_deadlocked_layer_count_depth_q2"], 0.0)
        self.assertEqual(out["moe_deadlocked_layer_count_depth_q3"], 0.0)
        self.assertEqual(out["moe_deadlocked_layer_count_depth_q4"], 2.0)

        next_out = tracker.update(
            {
                layer: (0.95 if index == 0 else 0.10)
                for index, layer in enumerate(layer_order)
            },
            layer_order=layer_order,
        )
        self.assertEqual(next_out["moe_max_deadlock_duration_depth_q1"], 2.0)
        self.assertEqual(next_out["moe_max_deadlock_duration_depth_q4"], 0.0)
        self.assertEqual(next_out["moe_max_deadlock_duration"], 2.0)

    def test_depth_quartiles_reject_incomplete_or_duplicate_order(self) -> None:
        tracker = DeadlockTracker()
        with self.assertRaisesRegex(ValueError, "missing observed layers"):
            tracker.update({"L0": 0.95, "L1": 0.95}, layer_order=["L0"])
        with self.assertRaisesRegex(ValueError, "unique layer keys"):
            tracker.update({"L0": 0.95}, layer_order=["L0", "L0"])

    def test_empty_update_returns_empty(self) -> None:
        self.assertEqual(DeadlockTracker().update({}), {})


@unittest.skipIf(torch is None, "torch not installed")
class MechanismDrivenRouterTests(unittest.TestCase):
    def test_similarity_matches_explicit_ordered_pairs(self) -> None:
        torch.manual_seed(31)
        weight = torch.randn(7, 11)
        unit = torch.nn.functional.normalize(weight.float(), dim=1)
        explicit = (
            (unit @ unit.t()).sum() - float(weight.shape[0])
        ) / float(weight.shape[0] * (weight.shape[0] - 1))
        self.assertTrue(
            torch.allclose(router_weight_similarity(weight), explicit, atol=1e-6)
        )

    def test_conditioning_ratio_and_similarity_bound(self) -> None:
        common = torch.tensor([3.0, -2.0, 1.0, 4.0])
        deviations = torch.tensor(
            [[0.1, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0]]
        )
        weight = common + deviations
        ratio = router_conditioning_ratio(weight)
        self.assertIsNotNone(ratio)
        similarity = router_weight_similarity(weight)
        bound = 1.0 - 3.0 / 2.0 * ratio.square()
        self.assertGreaterEqual(float(similarity), float(bound) - 1e-6)
        self.assertIsNone(
            router_conditioning_ratio(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        )

    def test_full_softmax_entropy_precedes_unchanged_top1_counts(self) -> None:
        baseline = torch.softmax(
            torch.tensor([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0]]), dim=-1
        )
        sharpened = torch.softmax(
            torch.tensor([[8.0, 0.9, 0.0], [0.9, 8.0, 0.0]]), dim=-1
        )
        baseline_entropy, baseline_fraction = per_token_router_entropy(baseline)
        sharpened_entropy, sharpened_fraction = per_token_router_entropy(sharpened)
        self.assertTrue(torch.equal(baseline.argmax(-1), sharpened.argmax(-1)))
        self.assertLess(float(sharpened_entropy), float(baseline_entropy))
        self.assertLess(float(sharpened_fraction), float(baseline_fraction))


@unittest.skipIf(torch is None, "torch not installed")
class MechanismDrivenAttentionTests(unittest.TestCase):
    @staticmethod
    def _projection(
        base: torch.Tensor,
        factor_a: torch.Tensor,
        factor_b: torch.Tensor,
        *,
        scale: float,
        heads: int = 2,
    ) -> LowRankProjectionState:
        return LowRankProjectionState(
            base_weight=base,
            factor_a=factor_a,
            factor_b=factor_b,
            scale=scale,
            num_heads=heads,
        )

    def test_qr_core_matches_materialized_delta2_spectrum(self) -> None:
        torch.manual_seed(41)
        base_q = torch.randn(4, 5)
        base_k = torch.randn(4, 5)
        previous_q = self._projection(
            base_q, torch.randn(2, 5), torch.randn(4, 2), scale=0.3
        )
        previous_k = self._projection(
            base_k, torch.randn(2, 5), torch.randn(4, 2), scale=0.4
        )
        current_q = self._projection(
            base_q, torch.randn(2, 5), torch.randn(4, 2), scale=0.2
        )
        current_k = self._projection(
            base_k, torch.randn(2, 5), torch.randn(4, 2), scale=0.5
        )
        q_snapshot = snapshot_low_rank_projection(previous_q)
        k_snapshot = snapshot_low_rank_projection(previous_k)
        for head in range(2):
            start = head * 2
            stop = start + 2
            q_prev = (
                base_q + previous_q.scale * previous_q.factor_b @ previous_q.factor_a
            )[start:stop].t()
            k_prev = (
                base_k + previous_k.scale * previous_k.factor_b @ previous_k.factor_a
            )[start:stop].t()
            q_now = (
                base_q + current_q.scale * current_q.factor_b @ current_q.factor_a
            )[start:stop].t()
            k_now = (
                base_k + current_k.scale * current_k.factor_b @ current_k.factor_a
            )[start:stop].t()
            materialized = (q_now - q_prev) @ k_prev.t() + q_prev @ (
                k_now - k_prev
            ).t()
            expected = torch.linalg.svdvals(materialized)
            expected = expected[expected > torch.finfo(expected.dtype).eps * 5]
            actual = attention_delta2_singular_values(
                current_q,
                q_snapshot,
                current_k,
                k_snapshot,
                head=head,
            )
            self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

    def test_coherent_fault_has_lower_effective_rank(self) -> None:
        torch.manual_seed(43)
        base_q = torch.eye(4)
        base_k = torch.eye(4)
        previous_q = self._projection(
            base_q, torch.eye(4), torch.zeros(4, 4), scale=1.0
        )
        previous_k = self._projection(
            base_k, torch.eye(4), torch.zeros(4, 4), scale=1.0
        )
        q_snapshot = snapshot_low_rank_projection(previous_q)
        k_snapshot = snapshot_low_rank_projection(previous_k)
        diverse_q = self._projection(
            base_q, torch.eye(4), torch.eye(4), scale=0.1
        )
        coherent_b = torch.ones(4, 1) @ torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        coherent_q = self._projection(
            base_q, torch.eye(4), coherent_b, scale=0.1
        )
        diverse_values = attention_delta2_singular_values(
            diverse_q, q_snapshot, previous_k, k_snapshot, head=0
        )
        coherent_values = attention_delta2_singular_values(
            coherent_q, q_snapshot, previous_k, k_snapshot, head=0
        )
        _diverse_entropy, diverse_rank = singular_spectrum_effective_rank(
            diverse_values
        )
        _coherent_entropy, coherent_rank = singular_spectrum_effective_rank(
            coherent_values
        )
        self.assertLess(float(coherent_rank), float(diverse_rank))

    def test_monitor_state_round_trip_preserves_update_window(self) -> None:
        torch.manual_seed(47)
        base_q = torch.randn(4, 5)
        base_k = torch.randn(4, 5)
        factor_a_q = torch.randn(2, 5)
        factor_a_k = torch.randn(2, 5)
        factor_b_q = torch.zeros(4, 2)
        factor_b_k = torch.zeros(4, 2)
        target = AttentionQKState(
            name="block.0.attn",
            query=self._projection(base_q, factor_a_q, factor_b_q, scale=0.5),
            key=self._projection(base_k, factor_a_k, factor_b_k, scale=0.5),
        )
        monitor = PreemptiveAttentionMonitor()
        self.assertEqual(monitor.observe((target,)), {})
        factor_b_q.add_(torch.randn_like(factor_b_q) * 0.01)
        first = monitor.observe((target,))
        self.assertIn("moe_attention_qk_delta2_effective_rank", first)

        restored = PreemptiveAttentionMonitor()
        restored.load_state_dict(monitor.state_dict())
        self.assertEqual(restored.observe((target,)), first)
        factor_b_k.add_(torch.randn_like(factor_b_k) * 0.01)
        second = restored.observe((target,))
        self.assertIn("moe_attention_qk_delta2_spectral_entropy", second)


@unittest.skipIf(torch is None, "torch not installed")
class RouterUnderflowTests(unittest.TestCase):
    @staticmethod
    def _param(value: float, grad: float | None):
        p = torch.tensor([[float(value)]])
        if grad is not None:
            p.grad = torch.tensor([[float(grad)]])
        return p

    def test_tiny_update_is_flagged(self) -> None:
        # w=1.0 -> bf16 ULP = 2**-7; half-ULP ~= 3.9e-3. u = 1e-3*1e-6 << that.
        named = [("blocks.0.router.weight", self._param(1.0, 1e-6))]
        frac = router_underflow_fraction(named, lr=1e-3)
        self.assertEqual(frac, 1.0)

    def test_healthy_update_is_not_flagged(self) -> None:
        named = [("blocks.0.router.weight", self._param(1.0, 1.0))]
        frac = router_underflow_fraction(named, lr=0.1)
        self.assertEqual(frac, 0.0)

    def test_frozen_router_zero_grad_reports_zero(self) -> None:
        # train_router=False path: router param present, no grad -> 0.0, no crash.
        named = [("blocks.0.router.weight", self._param(1.0, None))]
        self.assertEqual(router_underflow_fraction(named, lr=1e-3), 0.0)

    def test_no_router_params_reports_none(self) -> None:
        named = [("blocks.0.experts.w1", self._param(1.0, 1e-9))]
        self.assertIsNone(router_underflow_fraction(named, lr=1e-3))

    def test_partial_fraction(self) -> None:
        weight = torch.tensor([1.0, 1.0])
        weight.grad = torch.tensor([1e-9, 1.0])  # one truncated, one healthy
        frac = router_underflow_fraction(
            [("blocks.0.router.weight", weight)], lr=0.1
        )
        self.assertAlmostEqual(frac, 0.5)

    def test_uses_offloaded_cpu_grad_when_present(self) -> None:
        p = torch.tensor([[1.0]])
        p._mirai_cpu_grad = torch.tensor([[1e-9]])
        frac = router_underflow_fraction([("blocks.0.router.weight", p)], lr=1e-3)
        self.assertEqual(frac, 1.0)


class ConfigPlumbingTests(unittest.TestCase):
    def test_default_is_off(self) -> None:
        self.assertFalse(TrainingConfig().model.params.moe_routing_health)

    def test_from_dict_round_trip(self) -> None:
        cfg = TrainingConfig.from_dict(
            {
                "model": {"type": "lingbot-video", "params": {"moe_routing_health": True}},
                "dataset": {"path": "./x", "cache_path": "./x/c.pt"},
            }
        )
        self.assertTrue(cfg.model.params.moe_routing_health)

    def test_key_registered_in_all_config_keys(self) -> None:
        self.assertIn("moe_routing_health", all_config_keys()["model.params"])

    def test_health_gate_activates_preemptive_monitoring_policy(self) -> None:
        from mirai.core.training.training_policy import TrainingPolicySet

        cfg = TrainingConfig.from_dict(
            {
                "model": {
                    "type": "lingbot-video",
                    "params": {"moe_routing_health": True},
                },
                "dataset": {"path": "./x", "cache_path": "./x/c.pt"},
            }
        )
        policies = TrainingPolicySet.from_config(cfg)
        self.assertIn("preemptive_monitoring", policies.active_names)

    def test_selection_margin_config_is_typed_registered_and_default_off(self) -> None:
        """Protects the explicit default-off config gate and its coercion path."""
        self.assertFalse(TrainingConfig().model.params.moe_selection_margin)
        cfg = TrainingConfig.from_dict(
            {
                "model": {
                    "type": "lingbot-video",
                    "params": {"moe_selection_margin": True},
                },
                "dataset": {"path": "./x", "cache_path": "./x/c.pt"},
            }
        )
        self.assertTrue(cfg.model.params.moe_selection_margin)
        self.assertIn(
            "moe_selection_margin", all_config_keys()["model.params"]
        )


class MetricSurfacingTests(unittest.TestCase):
    def test_health_keys_surface_when_present_in_diagnostics(self) -> None:
        depth_metrics = {
            key: float(index)
            for index, key in enumerate(_DEPTH_BAND_KEYS, start=1)
        }
        metrics = build_step_metrics(
            config=TrainingConfig(),
            last_metrics={
                "loss": 0.1,
                "diagnostics": {
                    "moe_expert_output_cossim": 0.42,
                    "moe_max_deadlock_duration": 3.0,
                    "moe_deadlocked_layer_count": 2.0,
                    **depth_metrics,
                },
            },
            lr=1e-3,
            grad_norm=0.2,
            skipped_steps=0,
            vram_used_mb=0.0,
        )
        self.assertEqual(metrics["moe_expert_output_cossim"], 0.42)
        self.assertEqual(metrics["moe_max_deadlock_duration"], 3.0)
        self.assertEqual(metrics["moe_deadlocked_layer_count"], 2.0)
        for key, value in depth_metrics.items():
            self.assertEqual(metrics[key], value)

    def test_health_keys_absent_when_diagnostics_empty(self) -> None:
        metrics = build_step_metrics(
            config=TrainingConfig(),
            last_metrics={"loss": 0.1, "diagnostics": {}},
            lr=1e-3,
            grad_norm=0.2,
            skipped_steps=0,
            vram_used_mb=0.0,
        )
        for key in _HEALTH_KEYS:
            self.assertNotIn(key, metrics)


@unittest.skipIf(torch is None, "torch not installed")
class PipelineGatingTests(unittest.TestCase):
    """Default off -> health metrics absent; on -> present, byte-identical when off."""

    def _pipeline(self, *, routing_health: bool, selection_margin: bool = False):
        from mirai.config.schema import AdapterConfig, ModelConfig, ModelParams
        from mirai.core.builtins import register_builtin_components
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

        register_builtin_components()
        mc = ModelConfig(
            type="lingbot-video",
            path="./models/lingbot_video",
            params=ModelParams(
                variant="tiny-video", latent_channels=2, num_experts=8,
                experts_per_token=2, shared_experts=1, hidden_size=16, num_layers=2,
                attention_heads=2, patch_size=1, moe_routing_health=routing_health,
                moe_selection_margin=selection_margin,
            ),
        )
        p = LingBotVideoPipeline(mc)
        p.set_adapter_config(
            AdapterConfig(type="lora", target_preset="attn_routed_experts", rank=4, alpha=4.0)
        )
        p.train()
        return p

    def _forward(self, p):
        torch.manual_seed(0)
        p.forward(
            torch.randn(1, 2, 4, 8, 8), torch.rand(1), {"lingbot": torch.randn(1, 3, 16)}
        )
        return p.get_training_diagnostics()

    def test_disabled_by_default_no_health_keys(self) -> None:
        diag = self._forward(self._pipeline(routing_health=False))
        # Stability metrics still present (unconditional), health metrics absent.
        self.assertIn("moe_routing_entropy", diag)
        for key in _HEALTH_KEYS:
            self.assertNotIn(key, diag)

    def test_disabled_selection_margin_adds_no_diagnostic_keys(self) -> None:
        """Protects the diagnostics surface when the optional gate is false."""
        diag = self._forward(
            self._pipeline(routing_health=False, selection_margin=False)
        )
        for key in _MARGIN_KEYS:
            self.assertNotIn(key, diag)

    def test_enabled_selection_margin_surfaces_scalar_summary(self) -> None:
        """Protects pipeline wiring from router scores through diagnostics."""
        pipeline = self._pipeline(routing_health=False, selection_margin=True)
        first = self._forward(pipeline)
        self.assertIn("moe_selection_margin_p05", first)
        second = self._forward(pipeline)
        for key in _MARGIN_KEYS:
            self.assertIn(key, second)
            self.assertIsInstance(second[key], float)

    def test_int8_representable_router_weight_has_exact_agreement(self) -> None:
        """Protects no-op quantization from reporting selection churn."""
        from mirai.vendors.lingbot_video.transformer_lingbot_video import (
            LingBotVideoRouter,
        )

        pipeline = self._pipeline(routing_health=False)
        router = next(
            module
            for module in pipeline.transformer.modules()
            if isinstance(module, LingBotVideoRouter)
        )
        integers = torch.arange(
            router.weight.numel(),
            dtype=torch.int32,
        ).remainder(255).sub(127)
        integers = integers.reshape_as(router.weight)
        integers[:, 0] = 127
        with torch.no_grad():
            router.weight.copy_(integers.float() * 0.25)
        tokens = torch.randn(17, router.weight.shape[1])
        reference_scores = torch.nn.functional.linear(tokens, router.weight.detach())
        router.weight.requires_grad_(False)
        router.enable_int8_weight()
        candidate_scores = torch.nn.functional.linear(
            tokens,
            router._execution_weight(device=tokens.device, dtype=tokens.dtype),
        )
        report = compare_router_selections(
            reference_scores,
            candidate_scores,
            top_k=router.top_k,
            num_experts=router.num_experts,
        )
        self.assertEqual(report.agreement, 1.0)
        self.assertEqual(report.changed_token_fraction, 0.0)

    def test_eval_forward_skips_training_telemetry(self) -> None:
        """Inference forwards must not pay for training telemetry (hundreds of
        GPU->CPU syncs per forward that the denoise loop never consumes):
        eval-mode diagnostics/aux losses are empty, train-mode keeps them."""
        p = self._pipeline(routing_health=True)
        diag_train = self._forward(p)
        self.assertIn("moe_routing_entropy", diag_train)
        p.eval()
        torch.manual_seed(0)
        p.forward(
            torch.randn(1, 2, 4, 8, 8), torch.rand(1), {"lingbot": torch.randn(1, 3, 16)}
        )
        self.assertEqual(p.get_training_diagnostics(), {})
        self.assertEqual(p.get_training_auxiliary_losses(), {})

    def test_enabled_surfaces_health_keys(self) -> None:
        diag = self._forward(self._pipeline(routing_health=True))
        for key in _HEALTH_KEYS:
            self.assertIn(key, diag)
        self.assertTrue(torch.isfinite(torch.tensor(diag["moe_expert_output_cossim"])))

    def test_attention_monitor_surfaces_after_effective_qk_update(self) -> None:
        from mirai.core.models.adapters.lora import LoRALinear

        pipeline = self._pipeline(routing_health=True)
        monitor = PreemptiveAttentionMonitor()
        pipeline.configure_preemptive_monitoring(monitor)
        first = self._forward(pipeline)
        for key in _ATTENTION_MONITOR_KEYS:
            self.assertNotIn(key, first)
        with torch.no_grad():
            for name, module in pipeline.transformer.named_modules():
                if isinstance(module, LoRALinear) and (
                    name.endswith("attn.to_q") or name.endswith("attn.to_k")
                ):
                    module.lora_b.add_(torch.randn_like(module.lora_b) * 0.01)
        second = self._forward(pipeline)
        for key in _ATTENTION_MONITOR_KEYS:
            self.assertIn(key, second)
            self.assertTrue(math.isfinite(float(second[key])))

    def test_second_forward_surfaces_logit_drift_and_checkpoint_restores_reference(self) -> None:
        torch.manual_seed(11)
        pipeline = self._pipeline(routing_health=True)
        self._forward(pipeline)
        diagnostics = self._forward(pipeline)
        self.assertIn("moe_router_logit_drift", diagnostics)
        context = diagnostics["moe_router_logit_drift_context"]
        self.assertEqual(
            context["latent_resolution"],
            {"frames": 4, "height": 8, "width": 8},
        )
        self.assertEqual(set(context["timestep"]), {"min", "max", "mean"})
        self.assertEqual(len(context["layers"]), 2)
        for layer in context["layers"].values():
            self.assertIn("most_drifted_expert", layer)
            self.assertIn("max_expert_normalized_abs", layer)
        state = pipeline.state_dict()

        torch.manual_seed(11)
        restored = self._pipeline(routing_health=True)
        restored.load_adapter_state(state)
        restored_diagnostics = self._forward(restored)
        self.assertIn("moe_router_logit_drift", restored_diagnostics)
        self.assertAlmostEqual(
            restored_diagnostics["moe_router_logit_drift"],
            diagnostics["moe_router_logit_drift"],
        )

    def test_validation_forward_does_not_capture_logit_drift_reference(self) -> None:
        pipeline = self._pipeline(routing_health=True)
        pipeline.transformer.eval()
        self._forward(pipeline)
        self.assertFalse(pipeline._router_drift_tracker.captured)

        pipeline.transformer.train()
        diagnostics = self._forward(pipeline)
        self.assertNotIn("moe_router_logit_drift", diagnostics)
        self.assertTrue(pipeline._router_drift_tracker.captured)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
