from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from mirai.core.models.compressed_weights.artifact_source import (  # noqa: E402
    load_grouped_expert_source,
)
from mirai.core.models.compressed_weights.execution.experts import (  # noqa: E402
    CompressedGroupedExperts,
)
from mirai.core.models.compressed_weights.packed.packed_state import (  # noqa: E402
    _packed_state_inventory,
)
from mirai.core.models.compressed_weights import (  # noqa: E402
    packed_artifact_fingerprint,
    read_compressed_weights_packed_state_manifest,
    save_compressed_weights_packed_tensors,
)
from mirai.core.moe.calibration.prototypes import (  # noqa: E402
    MeanExpertOutputAccumulator,
    PrototypeCalibrationEvidence,
    select_hierarchical_output_merge,
)
from mirai.core.moe.storage.consolidation import (  # noqa: E402
    consolidate_packed_state,
)
from scripts.tools.consolidate_experts import _load_plan  # noqa: E402
from scripts.tools.consolidate_experts import _validate_plan_modules  # noqa: E402
from scripts.tools import consolidate_experts as consolidate_tool  # noqa: E402


def _dense_experts(
    values: tuple[float, ...],
    *,
    quant_format: str = "int8",
    width: int = 4,
) -> CompressedGroupedExperts:
    module = CompressedGroupedExperts.from_empty(
        num_experts=len(values),
        group_sizes=4,
        expert_weight_access="active_dequant",
        quant_format=quant_format,
    )
    for projection, multiplier in (("w1", 1.0), ("w2", 0.5), ("w3", 0.25)):
        dense = torch.stack(
            [
                torch.full((width, width), float(value) * multiplier)
                for value in values
            ]
        )
        module.load_dense_weight(projection, dense)
    return module


def test_all_experts_share_the_same_output_calibration_population() -> None:
    module = _dense_experts((1.0, 1.1, 3.0, 3.1))
    observer = MeanExpertOutputAccumulator(4, max_tokens_per_observation=2)
    module.set_prototype_calibration_observer(observer)
    tokens = torch.tensor(
        [
            [1.0, 0.5, 0.25, 0.75],
            [0.25, 1.0, 0.5, 0.75],
            [0.5, 0.25, 1.0, 0.75],
        ]
    )
    module.run_direct_routed(
        tokens,
        torch.ones((3, 1)),
        torch.tensor([[0], [1], [0]]),
    )
    evidence = observer.evidence()
    assert evidence.distance_metric == "mean_output_euclidean"
    assert observer.output_tokens_per_expert == 2
    assert evidence.selected_count.tolist() == [2, 1, 0, 0]
    assert evidence.distance_matrix[0, 1] < evidence.distance_matrix[0, 2]


def test_average_linkage_and_frequency_weights_are_deterministic() -> None:
    evidence = PrototypeCalibrationEvidence(
        contribution_sum=torch.ones(4),
        selected_count=torch.tensor([9, 1, 2, 8]),
        distance_matrix=torch.tensor(
            [
                [0.0, 0.1, 4.0, 4.1],
                [0.1, 0.0, 4.1, 4.0],
                [4.0, 4.1, 0.0, 0.1],
                [4.1, 4.0, 0.1, 0.0],
            ],
            dtype=torch.float64,
        ),
        distance_metric="mean_output_euclidean",
    )
    plan = select_hierarchical_output_merge(evidence, 0.5)
    assert plan.consolidation.logical_to_prototype == (0, 0, 2, 2)
    assert plan.merge_weights == pytest.approx((0.9, 0.1, 0.2, 0.8))


def test_hierarchical_transform_merges_then_reencodes_without_mutation() -> None:
    root = nn.Module()
    root.experts = _dense_experts((1.0, 2.0, 5.0, 7.0))
    inventory, manifest = _packed_state_inventory(root)
    tensors = dict(inventory)
    source_manifest = copy.deepcopy(manifest)
    source_tensors = {key: value.clone() for key, value in tensors.items()}

    merged_tensors, merged_manifest = consolidate_packed_state(
        tensors,
        manifest,
        {
            "experts": {
                "logical_to_prototype": [0, 0, 2, 2],
                "merge_weights": [0.75, 0.25, 0.2, 0.8],
            }
        },
    )

    assert manifest == source_manifest
    assert all(torch.equal(tensors[key], source_tensors[key]) for key in tensors)
    merged_spec = merged_manifest["modules"]["experts"]
    assert merged_spec["consolidation_method"] == "hierarchical_output"
    assert merged_spec["logical_to_physical"] == [0, 0, 1, 1]
    assert merged_spec["num_experts"] == 2
    decoded = load_grouped_expert_source(merged_spec, merged_tensors)
    w1 = torch.stack(
        [
            decoded._dequantize_expert(
                "w1", expert, dtype=torch.float32, device=torch.device("cpu")
            )
            for expert in range(2)
        ]
    )
    expected = torch.stack(
        [
            torch.full((4, 4), 1.25),
            torch.full((4, 4), 6.6),
        ]
    )
    assert torch.allclose(w1, expected, atol=0.06, rtol=0.01)


@pytest.mark.parametrize(
    "quant_format",
    ("fp8", "gguf_iq4", "gguf_iq3", "gguf_iq2", "mxfp8_e4m3", "mxfp4", "nvfp4"),
)
def test_hierarchical_transform_reencodes_portable_packed_formats(
    quant_format: str,
) -> None:
    root = nn.Module()
    root.experts = _dense_experts(
        (1.0, 2.0, 5.0, 7.0),
        quant_format=quant_format,
        width=16,
    )
    source_w1 = torch.stack(
        [
            root.experts._dequantize_expert(
                "w1", expert, dtype=torch.float32, device=torch.device("cpu")
            )
            for expert in range(4)
        ]
    )
    inventory, manifest = _packed_state_inventory(root)
    merged_tensors, merged_manifest = consolidate_packed_state(
        dict(inventory),
        manifest,
        {
            "experts": {
                "logical_to_prototype": [0, 0, 2, 2],
                "merge_weights": [0.75, 0.25, 0.2, 0.8],
            }
        },
    )
    merged_spec = merged_manifest["modules"]["experts"]
    assert merged_spec["quant_format"] == quant_format
    decoded = load_grouped_expert_source(merged_spec, merged_tensors)
    merged_w1 = torch.stack(
        [
            decoded._dequantize_expert(
                "w1", expert, dtype=torch.float32, device=torch.device("cpu")
            )
            for expert in range(2)
        ]
    )
    expected = torch.stack(
        [
            (0.75 * source_w1[0]) + (0.25 * source_w1[1]),
            (0.2 * source_w1[2]) + (0.8 * source_w1[3]),
        ]
    )
    assert bool(torch.isfinite(merged_w1).all().item())
    assert torch.allclose(merged_w1, expected, atol=0.75, rtol=0.2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NF4 requires CUDA")
def test_hierarchical_transform_reencodes_nf4_packed_experts() -> None:
    root = nn.Module()
    root.experts = _dense_experts(
        (1.0, 2.0, 5.0, 7.0),
        quant_format="nf4",
        width=64,
    )
    source_w1 = torch.stack(
        [
            root.experts._dequantize_expert(
                "w1", expert, dtype=torch.float32, device=torch.device("cpu")
            )
            for expert in range(4)
        ]
    )
    inventory, manifest = _packed_state_inventory(root)
    merged_tensors, merged_manifest = consolidate_packed_state(
        dict(inventory),
        manifest,
        {
            "experts": {
                "logical_to_prototype": [0, 0, 2, 2],
                "merge_weights": [0.75, 0.25, 0.2, 0.8],
            }
        },
    )
    merged_spec = merged_manifest["modules"]["experts"]
    assert merged_spec["quant_format"] == "nf4"
    decoded = load_grouped_expert_source(merged_spec, merged_tensors)
    merged_w1 = torch.stack(
        [
            decoded._dequantize_expert(
                "w1", expert, dtype=torch.float32, device=torch.device("cpu")
            )
            for expert in range(2)
        ]
    )
    expected = torch.stack(
        [
            (0.75 * source_w1[0]) + (0.25 * source_w1[1]),
            (0.2 * source_w1[2]) + (0.8 * source_w1[3]),
        ]
    )
    assert bool(torch.isfinite(merged_w1).all().item())
    assert torch.allclose(merged_w1, expected, atol=0.5, rtol=0.2)


def test_reviewed_plan_requires_complete_source_lineage(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "format": "mirai.moe.expert_consolidation_plan",
                "schema_version": 2,
                "method": "hierarchical_output",
                "strategy": "average_linkage",
                "reduction_ratio": 0.5,
                "lineage": {
                    "dataset_snapshot_id": "dataset",
                    "model_snapshot_id": "model",
                    "config_snapshot_id": "config",
                },
                "modules": {
                    "experts": {
                        "logical_to_prototype": [0, 0],
                        "merge_weights": [0.75, 0.25],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lineage is incomplete"):
        _load_plan(plan_path, expected_method="hierarchical_output")


def test_reviewed_plan_preserves_policy_and_source_lineage(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    source_fingerprint = "sha256:" + ("a" * 64)
    plan_path.write_text(
        json.dumps(
            {
                "format": "mirai.moe.expert_consolidation_plan",
                "schema_version": 2,
                "method": "hierarchical_output",
                "strategy": "average_linkage",
                "reduction_ratio": 0.5,
                "lineage": {
                    "dataset_snapshot_id": "dataset",
                    "model_snapshot_id": "model",
                    "config_snapshot_id": "config",
                    "packed_artifact_fingerprint": source_fingerprint,
                },
                "modules": {
                    "experts": {
                        "logical_to_prototype": [0, 0],
                        "merge_weights": [0.75, 0.25],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = _load_plan(plan_path, expected_method="hierarchical_output")
    assert loaded.strategy == "average_linkage"
    assert loaded.reduction_ratio == 0.5
    assert loaded.lineage["packed_artifact_fingerprint"] == source_fingerprint
    assert loaded.modules["experts"]["logical_to_prototype"] == [0, 0]


def test_reviewed_plan_must_cover_exact_grouped_module_inventory() -> None:
    manifest = {
        "modules": {
            "blocks.0.experts": {"kind": "grouped_experts"},
            "blocks.1.experts": {"kind": "grouped_experts"},
            "blocks.0.router": {"kind": "linear"},
        }
    }
    with pytest.raises(ValueError, match="do not exactly match"):
        _validate_plan_modules(
            manifest,
            {"blocks.0.experts": {"logical_to_prototype": [0]}},
        )
    _validate_plan_modules(
        manifest,
        {
            "blocks.0.experts": {"logical_to_prototype": [0]},
            "blocks.1.experts": {"logical_to_prototype": [0]},
        },
    )


def test_artifact_transform_binds_plan_to_exact_packed_source(
    tmp_path,
    monkeypatch,
) -> None:
    root = nn.Module()
    root.experts = _dense_experts((1.0, 2.0, 5.0, 7.0))
    inventory, manifest = _packed_state_inventory(root)
    source_path = tmp_path / "source.safetensors"
    save_compressed_weights_packed_tensors(
        source_path,
        dict(inventory),
        manifest,
    )
    source_fingerprint = packed_artifact_fingerprint(source_path)
    lineage = {
        "dataset_snapshot_id": "dataset",
        "model_snapshot_id": "model",
        "config_snapshot_id": "config",
        "packed_artifact_fingerprint": source_fingerprint,
    }
    monkeypatch.setattr(
        consolidate_tool,
        "expected_prototype_calibration_lineage",
        lambda _config, _config_path: dict(lineage),
    )
    config = SimpleNamespace(
        model=SimpleNamespace(
            type="test_sparse_moe",
            params=SimpleNamespace(
                expert_consolidation="hierarchical_output",
                variant="contract",
            ),
        ),
        memory=SimpleNamespace(
            frozen_weight_packed_state_path=str(source_path),
        ),
    )
    payload = {
        "format": "mirai.moe.expert_consolidation_plan",
        "schema_version": 2,
        "method": "hierarchical_output",
        "strategy": "average_linkage",
        "reduction_ratio": 0.5,
        "lineage": lineage,
        "modules": {
            "experts": {
                "logical_to_prototype": [0, 0, 2, 2],
                "merge_weights": [0.75, 0.25, 0.2, 0.8],
            }
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "consolidated.safetensors"
    report = consolidate_tool.consolidate_packed_base(
        config,
        config_path=tmp_path / "config.toml",
        packed_state=source_path,
        plan_file=plan_path,
        output=output_path,
    )
    assert report["source_artifact_fingerprint"] == source_fingerprint
    output_manifest = read_compressed_weights_packed_state_manifest(output_path)
    assert output_manifest["expert_consolidation"]["lineage"] == lineage

    payload["lineage"] = {
        **lineage,
        "packed_artifact_fingerprint": "sha256:" + ("b" * 64),
    }
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="packed_artifact_fingerprint mismatch"):
        consolidate_tool.consolidate_packed_base(
            config,
            config_path=tmp_path / "config.toml",
            packed_state=source_path,
            plan_file=plan_path,
            output=tmp_path / "must-not-exist.safetensors",
        )
    assert not (tmp_path / "must-not-exist.safetensors").exists()
