"""Behavioral contracts for activation-aware shared-basis factorization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from mirai.core.models.compressed_weights.execution.experts import (
    CompressedGroupedExperts,
)
from mirai.core.models.compressed_weights.factorization.shared_basis import (
    factorize_dense_experts,
)
from mirai.core.moe.calibration.whitening import ActivationCovarianceAccumulator
from mirai.core.moe.calibration.whitening import load_expert_whitening_evidence
from mirai.core.moe.calibration.whitening import save_expert_whitening_evidence


def test_streamed_covariance_matches_concatenated_outer_products() -> None:
    accumulator = ActivationCovarianceAccumulator({"w1": 3, "w2": 2, "w3": 3})
    first = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    second = torch.tensor([[0.25, -2.0, 1.0]])
    down = torch.tensor([[2.0, -1.0], [0.5, 4.0], [-3.0, 2.0]])
    accumulator.record(("w1", "w3"), first)
    accumulator.record(("w1", "w3"), second)
    accumulator.record("w2", down)
    evidence = accumulator.evidence()

    routed = torch.cat((first, second), dim=0)
    expected_up = routed.T @ routed
    expected_down = down.T @ down
    assert torch.equal(evidence.projections["w1"].covariance, expected_up)
    assert torch.equal(evidence.projections["w3"].covariance, expected_up)
    assert torch.equal(evidence.projections["w2"].covariance, expected_down)
    assert evidence.projections["w1"].sample_count == 3
    assert evidence.projections["w2"].sample_count == 3


def test_whitened_svd_prefers_low_variance_weight_axis_with_high_activation_energy() -> None:
    weights = torch.tensor([[[1.0, 0.0], [0.0, 0.9]]])
    covariance = torch.diag(torch.tensor([0.01, 100.0]))
    vanilla = factorize_dense_experts(
        weights,
        rank=1,
        axis="right",
        factor_dtype="float32",
    )
    whitened = factorize_dense_experts(
        weights,
        rank=1,
        axis="right",
        factor_dtype="float32",
        input_covariance=covariance,
        whitening_regularization=0.0,
    )

    vanilla_residual = weights[0] - vanilla.reconstruct(
        0, dtype=torch.float32, device=torch.device("cpu")
    )
    whitened_residual = weights[0] - whitened.reconstruct(
        0, dtype=torch.float32, device=torch.device("cpu")
    )
    vanilla_energy = torch.sum((vanilla_residual @ covariance) * vanilla_residual)
    whitened_energy = torch.sum((whitened_residual @ covariance) * whitened_residual)
    assert whitened_energy < vanilla_energy
    assert whitened.whitened_relative_error is not None


def test_whitened_left_basis_reconstructs_declared_shape() -> None:
    generator = torch.Generator().manual_seed(17)
    weights = torch.randn(3, 5, 4, generator=generator)
    samples = torch.randn(32, 4, generator=generator)
    factors = factorize_dense_experts(
        weights,
        rank=2,
        axis="left",
        factor_dtype="float32",
        input_covariance=samples.T @ samples,
    )
    assert factors.basis.shape == (5, 2)
    assert factors.coefficients.shape == (3, 2, 4)
    assert factors.reconstruct(
        1, dtype=torch.float32, device=torch.device("cpu")
    ).shape == (5, 4)
    assert factors.whitened_relative_error is not None


def test_whitening_evidence_roundtrip_enforces_packed_lineage() -> None:
    accumulator = ActivationCovarianceAccumulator({"w1": 2, "w2": 3, "w3": 2})
    accumulator.record(("w1", "w3"), torch.eye(2))
    accumulator.record("w2", torch.eye(3))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "whitening.safetensors"
        save_expert_whitening_evidence(
            path,
            {"layers.0.experts": accumulator.evidence()},
            dataset_snapshot_id="dataset",
            model_snapshot_id="model",
            config_snapshot_id="config",
            packed_artifact_fingerprint="sha256:" + ("a" * 64),
        )
        loaded, lineage = load_expert_whitening_evidence(
            path,
            expected_packed_artifact_fingerprint="sha256:" + ("a" * 64),
        )
        assert set(loaded) == {"layers.0.experts"}
        assert lineage["dataset_snapshot_id"] == "dataset"
        with pytest.raises(ValueError, match="packed_artifact_fingerprint mismatch"):
            load_expert_whitening_evidence(
                path,
                expected_packed_artifact_fingerprint="sha256:" + ("b" * 64),
            )


def test_compressed_reference_execution_captures_actual_projection_inputs() -> None:
    class _DenseExperts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 2
            self.w1 = torch.nn.Parameter(torch.randn(2, 3, 2))
            self.w2 = torch.nn.Parameter(torch.randn(2, 2, 3))
            self.w3 = torch.nn.Parameter(torch.randn(2, 3, 2))

    host = CompressedGroupedExperts(
        _DenseExperts(),
        quant_format="int8",
        expert_weight_access="full_dequant",
    )
    accumulator = ActivationCovarianceAccumulator({"w1": 2, "w2": 3, "w3": 2})
    host.set_whitening_calibration_observer(accumulator)
    try:
        output = host.run_for_loop(
            torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]]),
            torch.tensor([2, 1]),
        )
    finally:
        host.clear_whitening_calibration_observer()
    evidence = accumulator.evidence()
    assert output.shape == (3, 2)
    assert evidence.projections["w1"].sample_count == 3
    assert evidence.projections["w2"].sample_count == 3
