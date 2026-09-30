from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from mirai.core.moe.calibration.diet import (
    DietCalibrationEvidence,
    DietFamilyPostprocess,
    DietGroupTopology,
    DietLayerCapture,
    DietLineage,
    DietODLConfig,
    DietPairedCapture,
    derive_diet_signatures,
    load_diet_evidence,
    propose_regression_layer_counts,
    replay_deletion_responses,
    save_diet_evidence,
    select_odl_survivors,
    validate_published_diet_mask,
)


def _topology() -> DietGroupTopology:
    return DietGroupTopology(8, 2, 2, 1, 1, True, 1.7)


def _capture(offset: float = 0.0, *, postprocess: bool = False) -> DietLayerCapture:
    scores = torch.tensor([[0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.4, 0.3],
                           [0.4, 0.8, 0.7, 0.1, 0.9, 0.2, 0.6, 0.3]])
    outputs = torch.arange(2 * 8 * 3, dtype=torch.float64).reshape(2, 8, 3) / 10 + offset
    state = DietFamilyPostprocess(torch.ones(2, 3), torch.tensor([1.0, 2.0, 3.0]), 1e-5,
                                  torch.tensor([[0.5], [0.25]])) if postprocess else None
    return DietLayerCapture(outputs, scores, torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                            _topology(), state)


def _reference(capture: DietLayerCapture, removed: int | None) -> torch.Tensor:
    scores = torch.as_tensor(capture.router_scores, dtype=torch.float64)
    outputs = torch.as_tensor(capture.expert_outputs, dtype=torch.float64)
    bias = torch.as_tensor(capture.choice_bias, dtype=torch.float64)
    result = []
    for row in range(scores.shape[0]):
        candidates = [expert for expert in range(8) if expert != removed]
        group_best = []
        for group in range(2):
            members = [expert for expert in candidates if expert // 4 == group]
            group_best.append(max((float(scores[row, e] + bias[e]), e) for e in members))
        selected_group = max((value, group) for group, (value, _) in enumerate(group_best))[1]
        allowed = [expert for expert in candidates if expert // 4 == selected_group]
        selected = sorted(allowed, key=lambda e: float(scores[row, e] + bias[e]), reverse=True)[:2]
        gates = scores[row, selected]
        gates = gates / gates.sum() * 1.7
        result.append((outputs[row, selected] * gates[:, None]).sum(0))
    return torch.stack(result)


def test_deletion_replay_matches_independent_grouped_router_reference() -> None:
    capture = _capture()
    actual = replay_deletion_responses(capture)
    baseline = _reference(capture, None)
    expected = torch.stack([_reference(capture, expert) - baseline for expert in range(8)])
    torch.testing.assert_close(actual, expected)


def test_postprocess_and_cfg_pairing_match_formula() -> None:
    conditional = _capture(0.0, postprocess=True)
    unconditional = _capture(0.3, postprocess=True)
    paired = DietPairedCapture(conditional, unconditional, torch.tensor([4, 5]), torch.tensor([9, 10]))
    evidence = derive_diet_signatures(paired, guidance_scale=2.25)
    expected = (2.25 * replay_deletion_responses(conditional)
                + (1.0 - 2.25) * replay_deletion_responses(unconditional)).mean(1)
    torch.testing.assert_close(evidence.signatures, expected)


def test_odl_known_clusters_is_deterministic_and_group_feasible() -> None:
    signatures = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99],
                               [-1.0, 0.0], [-0.99, 0.01], [0.0, -1.0], [0.01, -0.99]])
    config = DietODLConfig(seed=31, random_restarts=2, swap_passes=3, anneal_steps=30)
    first = select_odl_survivors(signatures, 4, _topology(), config=config)
    second = select_odl_survivors(signatures, 4, _topology(), config=config)
    assert first == second
    assert len(first) == 4
    group_counts = [sum(expert // 4 == group for expert in first) for group in range(2)]
    assert min(count for count in group_counts if count >= 1) >= 2


def test_evidence_roundtrip_and_strict_lineage() -> None:
    evidence = DietCalibrationEvidence(torch.eye(8), 2, _topology(), "paired_cfg", 2.25)
    lineage = DietLineage("data", "model", "config", "manifest", "a" * 64, "capture")
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "diet.safetensors"
        save_diet_evidence(path, {"blocks.0.experts": evidence}, lineage=lineage)
        loaded, actual_lineage = load_diet_evidence(path, expected_lineage=lineage)
        torch.testing.assert_close(loaded["blocks.0.experts"].signatures, evidence.signatures)
        assert actual_lineage.to_dict()["packed_state_sha256"] == "a" * 64
        with pytest.raises(ValueError, match="lineage"):
            load_diet_evidence(path, expected_lineage=DietLineage("other", "model", "config", "manifest", "a" * 64, "capture"))


def test_published_mask_requires_declared_author_and_artifact_fingerprint() -> None:
    with pytest.raises(ValueError, match="lineage"):
        validate_published_diet_mask({"layer": [1]}, declared_source_lineage={"source_revision": "x"})
    assert validate_published_diet_mask(
        {"layer": [2, 1]},
        declared_source_lineage={"publisher": "authors", "source_repository": "https://example.invalid/diet",
                                 "source_revision": "commit", "artifact_sha256": "f" * 64},
    ) == {"layer": (1, 2)}


def test_regression_budget_proposal_honors_integer_budget_and_bounds() -> None:
    proposal = propose_regression_layer_counts(
        torch.tensor([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]),
        torch.tensor([[1.0], [2.0], [3.0]]),
        total_budget=4,
        lower_bounds=(1, 1),
        upper_bounds=(3, 3),
        baseline_scores=(0.0,),
        per_dimension_floors=(10.0,),
        ridge_penalty=1e-6,
    )
    assert proposal.layer_counts == (3, 1)
    assert sum(proposal.layer_counts) == 4
