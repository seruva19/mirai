"""Contracts for domain-labelled reference-router evidence."""

from __future__ import annotations

import json

import pytest

from mirai.core.moe.monitoring.domain_testbed import (
    DOMAIN_ROUTING_TESTBED_SCHEMA,
    build_domain_routing_testbed_evidence,
    evaluate_domain_routing_layer,
)


REFERENCE = {"art": [0, 1], "motion": [2, 3]}


def test_exact_reference_routes_report_perfect_specialization() -> None:
    report = evaluate_domain_routing_layer(
        [
            {"domain": "art", "selected_experts": [0, 1]},
            {"domain": "motion", "selected_experts": [2, 3]},
        ],
        reference=REFERENCE,
        num_experts=4,
    )
    assert report.top1_reference_accuracy == 1.0
    assert report.selected_reference_precision == 1.0
    assert report.reference_expert_coverage == 1.0
    assert report.reference_regret == 0.0
    assert report.expert_domain_purity == 1.0
    assert report.normalized_mutual_information == pytest.approx(1.0)


def test_collapsed_routes_expose_regret_coverage_and_information_loss() -> None:
    report = evaluate_domain_routing_layer(
        [
            {"domain": "art", "selected_experts": [0]},
            {"domain": "motion", "selected_experts": [0]},
        ],
        reference=REFERENCE,
        num_experts=4,
    )
    assert report.top1_reference_accuracy == 0.5
    assert report.selected_reference_precision == 0.5
    assert report.reference_expert_coverage == 0.25
    assert report.reference_regret == 0.5
    assert report.expert_domain_purity == 0.5
    assert report.normalized_mutual_information == 0.0


def test_evidence_is_lineage_bound_and_deterministic() -> None:
    observations = {
        "num_experts": 4,
        "layers": {
            "block.0": [
                {"domain": "art", "selected_experts": [0]},
                {"domain": "motion", "selected_experts": [2]},
            ]
        },
    }
    reference = {"domains": REFERENCE}
    first = build_domain_routing_testbed_evidence(
        observations,
        reference,
        dataset_fingerprint="dataset-sha256",
        model_fingerprint="model-sha256",
    )
    second = build_domain_routing_testbed_evidence(
        json.loads(json.dumps(observations)),
        json.loads(json.dumps(reference)),
        dataset_fingerprint="dataset-sha256",
        model_fingerprint="model-sha256",
    )
    assert first == second
    assert first["schema"] == DOMAIN_ROUTING_TESTBED_SCHEMA
    assert len(first["observation_fingerprint"]) == 64
    assert len(first["reference_fingerprint"]) == 64


@pytest.mark.parametrize(
    ("records", "reference", "message"),
    [
        ([], REFERENCE, "requires observations"),
        ([{"domain": "unknown", "selected_experts": [0]}], REFERENCE, "unassigned"),
        ([{"domain": "art", "selected_experts": [0, 0]}], REFERENCE, "unique"),
        ([{"domain": "art", "selected_experts": [4]}], REFERENCE, "outside"),
    ],
)
def test_malformed_observations_fail_closed(records, reference, message) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_domain_routing_layer(records, reference=reference, num_experts=4)
