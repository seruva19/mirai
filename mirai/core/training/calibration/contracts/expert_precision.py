"""Behavioral contracts for routed imatrix precision planning."""

from __future__ import annotations

from pathlib import Path
import tempfile

from mirai.core.models.compressed_weights.execution.mixed_precision import (
    MixedPrecisionGroupedExperts,
)
from mirai.core.models.compressed_weights.quantization.sensitivity import (
    measure_projection_format,
)
from mirai.core.moe.calibration.imatrix import ExpertImportanceAccumulator
from mirai.core.moe.calibration.imatrix import ExpertImportanceCalibrationTarget
from mirai.core.moe.calibration.alphaq import alpha_importance_weights
from mirai.core.moe.calibration.alphaq import pl_alpha_hill
from mirai.core.moe.calibration.precision import ExpertPrecisionPlan
from mirai.core.moe.calibration.precision import TensorPrecisionEvidence
from mirai.core.moe.calibration.precision import TensorPrecisionPlan
from mirai.core.moe.calibration.precision import allocate_tensor_precision
from mirai.core.moe.calibration.precision import load_precision_plan
from mirai.core.training.calibration.expert_precision import _router_norm_evidence
from mirai.core.moe.calibration.precision import RouterNormExpertEvidence
from mirai.core.moe.calibration.precision import rank_router_norm_experts
from mirai.core.moe.calibration.precision import router_norm_precision_floors

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


def test_importance_accumulator_separates_experts_and_projection_inputs() -> None:
    accumulator = ExpertImportanceAccumulator(
        num_experts=2,
        input_dims={"w1": 3, "w2": 2, "w3": 3},
    )
    accumulator.record(0, ("w1", "w3"), torch.tensor([[1.0, 2.0, 3.0]]))
    accumulator.record(1, ("w1", "w3"), torch.tensor([[2.0, 0.0, 1.0]]))
    accumulator.record(0, "w2", torch.tensor([[3.0, 4.0]]))
    accumulator.record(1, "w2", torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    evidence = accumulator.evidence()
    torch.testing.assert_close(
        evidence.mean_squares("w1", 0),
        torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        evidence.mean_squares("w2", 1),
        torch.tensor([5.0, 10.0], dtype=torch.float64),
    )
    assert torch.equal(
        evidence.input_sum_squares["w1"],
        evidence.input_sum_squares["w3"],
    )


def test_alphaq_importance_matches_paper_equation_and_orders_heavy_tail() -> None:
    weights, gamma = alpha_importance_weights((2.0, 3.0, 4.0), gamma=2.0)
    assert gamma == 2.0
    assert weights == (2.25, 1.0, 0.5625)
    assert weights[0] > weights[1] > weights[2]


def test_alphaq_data_free_gamma_matches_paper_default() -> None:
    weights, gamma = alpha_importance_weights((2.0, 3.0, 4.0), gamma=0.0)
    expected = 2.0 * (4.0 - 2.0) / (2.0 / 3.0)
    assert abs(gamma - expected) < 1e-12
    assert weights[0] > weights[1] > weights[2]


def test_alphaq_fixed_aspect_hill_estimator_is_deterministic() -> None:
    singular_values = torch.logspace(0.0, 2.0, 32)
    weight = torch.diag(singular_values)
    first = pl_alpha_hill(weight, block_size=32, histogram_bins=8)
    second = pl_alpha_hill(weight, block_size=32, histogram_bins=8)
    assert first == second
    assert first > 1.0


def test_tensor_allocator_protects_sensitive_projection_under_exact_budget() -> None:
    rows = []
    for projection, low_error in (("w1", 1.0), ("w2", 20.0), ("w3", 1.0)):
        rows.append(
            TensorPrecisionEvidence(
                module_name="blocks.0.experts",
                expert_id=0,
                projection=projection,
                weight_numel=100,
                format_error={"gguf_iq3": low_error, "int8": 0.0},
                format_bytes={"gguf_iq3": 38, "int8": 100},
            )
        )
    plan = allocate_tensor_precision(
        rows,
        budget_bytes=176,
        allowed_formats=("gguf_iq3", "int8"),
        dataset_snapshot_id="dataset",
        model_snapshot_id="model",
        config_snapshot_id="config",
        source_weight_fingerprint="weights",
    )
    formats = plan.formats_for_module("blocks.0.experts")
    assert formats["w2"] == ("int8",)
    assert formats["w1"] == ("gguf_iq3",)
    assert formats["w3"] == ("gguf_iq3",)
    assert plan.estimated_bytes == 176


def test_router_norm_ranking_matches_norm_change_and_variance_promotion() -> None:
    evidence = [
        RouterNormExpertEvidence("moe.experts", 0, 2.0, 1.0, 1.0),
        RouterNormExpertEvidence("moe.experts", 1, 3.0, 1.0, 1.0),
        RouterNormExpertEvidence("moe.experts", 2, 4.0, 10.0, 1.0),
    ]
    assert rank_router_norm_experts(evidence) == {"moe.experts": (2, 0, 1)}
    assert router_norm_precision_floors(
        evidence,
        protected_fraction=1.0 / 3.0,
        minimum_format="int8",
    ) == {("moe.experts", 2): "int8"}
    crossing = [
        RouterNormExpertEvidence("moe.experts", 0, 1.0, 1.0),
        RouterNormExpertEvidence("moe.experts", 1, 2.0, 5.0),
        RouterNormExpertEvidence("moe.experts", 2, 3.0, 10.0),
    ]
    assert rank_router_norm_experts(crossing) == {"moe.experts": (1, 2, 0)}


def test_router_norm_evidence_uses_final_norm_and_maximum_row_variance() -> None:
    class Host:
        def set_importance_calibration_observer(self, _observer) -> None:
            return None

        def clear_importance_calibration_observer(self) -> None:
            return None

    w1 = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 2.0]],
            [[0.0, 4.0], [1.0, 3.0]],
        ]
    )
    target = ExpertImportanceCalibrationTarget(
        name="moe.experts",
        host=Host(),
        weights={"w1": w1, "w2": w1, "w3": w1},
        router_weight=torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
    ).validate()
    evidence = _router_norm_evidence({"moe.experts": target})
    assert [row.final_router_norm for row in evidence] == [5.0, 2.0]
    assert [row.max_intra_neuron_variance for row in evidence] == [1.0, 4.0]


def test_tensor_allocator_enforces_router_norm_precision_floor() -> None:
    rows = [
        TensorPrecisionEvidence(
            module_name="moe.experts",
            expert_id=expert_id,
            projection=projection,
            weight_numel=64,
            format_error={"gguf_iq2": 0.0, "int8": 0.0},
            format_bytes={"gguf_iq2": 19, "int8": 64},
        )
        for expert_id in range(2)
        for projection in ("w1", "w2", "w3")
    ]
    plan = allocate_tensor_precision(
        rows,
        budget_bytes=3 * 64 + 3 * 19,
        allowed_formats=("gguf_iq2", "int8"),
        dataset_snapshot_id="dataset",
        model_snapshot_id="model",
        config_snapshot_id="config",
        source_weight_fingerprint="weights-and-router",
        minimum_expert_formats={("moe.experts", 0): "int8"},
    )
    formats = plan.formats_for_module("moe.experts")
    assert all(values[0] == "int8" for values in formats.values())
    assert all(values[1] == "gguf_iq2" for values in formats.values())


def test_empty_router_norm_floors_preserve_allocator_exactly() -> None:
    rows = [
        TensorPrecisionEvidence(
            module_name="moe.experts",
            expert_id=expert_id,
            projection=projection,
            weight_numel=8,
            format_error={"gguf_iq3": 1.0, "int8": 0.0},
            format_bytes={"gguf_iq3": 3, "int8": 8},
        )
        for expert_id in range(2)
        for projection in ("w1", "w2", "w3")
    ]
    kwargs = dict(
        budget_bytes=33,
        allowed_formats=("gguf_iq3", "int8"),
        dataset_snapshot_id="dataset",
        model_snapshot_id="model",
        config_snapshot_id="config",
        source_weight_fingerprint="weights",
    )
    assert allocate_tensor_precision(rows, **kwargs) == allocate_tensor_precision(
        rows,
        minimum_expert_formats={},
        **kwargs,
    )


def test_projection_measurement_uses_real_runtime_storage_and_weighted_error() -> None:
    weight = torch.tensor(
        [
            [0.1251, -0.2753, 0.8757, -1.1259],
            [0.0317, 0.4921, -0.7423, 1.2531],
        ],
        dtype=torch.float32,
    )
    importance = torch.tensor([1.0, 3.0, 7.0, 11.0])
    bf16 = measure_projection_format(
        weight,
        importance,
        quant_format="bf16",
        projection="w1",
    )
    int8 = measure_projection_format(
        weight,
        importance,
        quant_format="int8",
        projection="w1",
    )
    assert bf16.stored_bytes == weight.numel() * 2
    assert int8.stored_bytes > 0
    assert bf16.weighted_mse >= 0.0
    assert int8.weighted_mse >= 0.0
    assert bf16.weighted_mse != int8.weighted_mse


def test_precision_plan_schema_dispatch_preserves_v1_and_roundtrips_v2() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        v1 = ExpertPrecisionPlan(
            schema_version=1,
            formats=("int8", "gguf_iq3"),
            estimated_bytes=10,
            weighted_error=1.0,
            budget_bytes=10,
        )
        assert load_precision_plan(v1.save(root / "v1.json")) == v1
        rows = [
            TensorPrecisionEvidence(
                module_name="moe.experts",
                expert_id=expert_id,
                projection=projection,
                weight_numel=8,
                format_error={"int8": 0.1},
                format_bytes={"int8": 8},
            )
            for expert_id in range(2)
            for projection in ("w1", "w2", "w3")
        ]
        v2 = allocate_tensor_precision(
            rows,
            budget_bytes=48,
            allowed_formats=("int8",),
            dataset_snapshot_id="dataset",
            model_snapshot_id="model",
            config_snapshot_id="config",
            source_weight_fingerprint="weights",
        )
        loaded = load_precision_plan(v2.save(root / "v2.json"))
        assert isinstance(loaded, TensorPrecisionPlan)
        assert loaded == v2


def test_tensor_mixed_precision_runtime_matches_its_materialized_reference() -> None:
    class DenseExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 2
            self.w1 = nn.Parameter(torch.randn(2, 6, 4), requires_grad=False)
            self.w2 = nn.Parameter(torch.randn(2, 4, 6), requires_grad=False)
            self.w3 = nn.Parameter(torch.randn(2, 6, 4), requires_grad=False)

    torch.manual_seed(31)
    mixed = MixedPrecisionGroupedExperts(
        DenseExperts(),
        formats={
            "w1": ("int8", "bf16"),
            "w2": ("bf16", "int8"),
            "w3": ("int8", "int8"),
        },
    )
    inputs = torch.randn(5, 4, requires_grad=True)
    indices = torch.tensor([[0], [1], [0], [1], [1]])
    scores = torch.ones(5, 1)
    actual = mixed(inputs, scores, indices)
    expected = torch.empty_like(actual)
    for token_id, expert_id in enumerate(indices[:, 0].tolist()):
        hosts = mixed.projection_hosts[expert_id]
        x = inputs[token_id : token_id + 1]
        w1 = hosts["w1"].materialize(dtype=x.dtype, device=x.device)
        w3 = hosts["w3"].materialize(dtype=x.dtype, device=x.device)
        hidden = torch.nn.functional.silu(x @ w1.T) * (x @ w3.T)
        w2 = hosts["w2"].materialize(dtype=x.dtype, device=x.device)
        expected[token_id] = (hidden @ w2.T)[0]
    torch.testing.assert_close(actual, expected)
    restored = MixedPrecisionGroupedExperts(
        DenseExperts(),
        formats={
            "w1": ("int8", "bf16"),
            "w2": ("bf16", "int8"),
            "w3": ("int8", "int8"),
        },
    )
    restored.load_state_dict(mixed.state_dict())
    torch.testing.assert_close(
        restored(inputs.detach(), scores, indices),
        actual.detach(),
    )
    actual.square().mean().backward()
    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad).all().item())
