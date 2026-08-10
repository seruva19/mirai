from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from mirai.core.moe.calibration.pruning import (  # noqa: E402
    ExpertPruningEvidence,
    ExpertPruningRoutedOutputObserver,
    ExpertPruningSaliencyAccumulator,
    aimer_expert_scores,
    load_expert_pruning_evidence,
    normalize_calibrated_expert_pruning_criterion,
    prune_packed_state,
    save_expert_pruning_evidence,
    select_pruned_experts,
)
from mirai.core.moe.calibration.flexmoe import (  # noqa: E402
    FlexMoEActionController,
    FlexMoEActionPlan,
    FlexMoEChannelSaliencyAccumulator,
    FlexMoERankingEvidence,
    FlexMoETaylorGradientObserver,
    action_entropy,
    apply_channel_permutation,
    channel_taylor_saliency,
    clean_action_probabilities,
    global_prune_budget,
    hardened_retention_ratios,
    load_action_plans,
    load_sensitive_cost,
    load_ranking_evidence,
    prefix_masks,
    rank_channels,
    save_action_plans,
    save_ranking_evidence,
    straight_through_gumbel_actions,
)
from mirai.core.models.compressed_weights.artifact_source import (  # noqa: E402
    load_grouped_expert_source,
)
from mirai.core.models.compressed_weights.execution.experts import (  # noqa: E402
    CompressedGroupedExperts,
)
from mirai.core.models.compressed_weights.flexmoe_nested import (  # noqa: E402
    FLEXMOE_NESTED_PROVIDER_NAME,
    FlexMoENestedPhysicalWeightProvider,
    transform_packed_state_flexmoe_nested,
)
from mirai.core.models.compressed_weights.packed.packed_state import (  # noqa: E402
    export_compressed_weights_packed_state,
    load_compressed_weights_packed_state,
    prepare_compressed_weights_modules_from_manifest,
)
from mirai.core.models.compressed_weights.factorization.prototype_projection import (  # noqa: E402
    CompressedExpertProjectionSource,
)
from mirai.core.moe.calibration.quantization import (  # noqa: E402
    ExpertAffinityAccumulator,
)
from mirai.core.moe.calibration.whitening import (  # noqa: E402
    ExpertWhiteningEvidence,
    ProjectionCovarianceEvidence,
)
from mirai.core.moe.storage.consolidation import consolidate_packed_state  # noqa: E402
from mirai.core.moe.storage.upcycling import (  # noqa: E402
    DropUpcyclingSpec,
    drop_upcycle_packed_state,
    validate_drop_upcycling_selection,
)
from mirai.core.models.compressed_weights.factorization.shared_basis_artifact import (  # noqa: E402
    factorize_packed_state_shared_basis,
)
from mirai.core.models.compressed_weights.factorization.mixture_basis import (  # noqa: E402
    MixtureBasisPhysicalWeightProvider,
    factorize_mixture_basis_experts,
)
from mirai.core.models.compressed_weights.factorization.mixture_basis_artifact import (  # noqa: E402
    factorize_packed_state_mixture_basis,
)
from mirai.core.models.compressed_weights.quantization.structured_sparse_provider import (  # noqa: E402
    PackedSparse24,
    Sparse24PhysicalWeightProvider,
    pack_sparse24,
    validate_packed_sparse24,
)
from mirai.core.models.compressed_weights.quantization.stun_artifact import (  # noqa: E402
    transform_packed_state_stun_sparse24,
)
from mirai.core.models.compressed_weights.packed.packed_graph import (  # noqa: E402
    assign_packed_state_tensor,
)
from mirai.core.moe.calibration.stun import (  # noqa: E402
    apply_stun_plan,
    cluster_router_experts,
    select_stun_representatives,
)
from mirai.core.moe.storage.physical_weights import (  # noqa: E402
    PhysicalWeightProviderContext,
)


def _packed_fixture():
    tensors = {
        "experts.w1": torch.arange(16).reshape(4, 2, 2),
        "experts.s1": torch.ones(4, 2),
        "blocks.0.router.weight": torch.arange(12).reshape(4, 3),
    }
    manifest = {
        "format": "mirai.compressed_weights.packed_state",
        "schema_version": 1,
        "modules": {
            "blocks.0.experts": {
                "kind": "grouped_experts",
                "num_experts": 4,
                "tensors": {
                    "w1_int8": "experts.w1",
                    "w1_scale": "experts.s1",
                },
                "shapes": {"w1": [4, 2, 2], "w1_scale": [4, 2]},
            }
        },
        "residual_tensors": {"blocks.0.router.weight": "blocks.0.router.weight"},
        "summary": {"quantized_numel": 24},
    }
    return tensors, manifest


def _aimer_packed_fixture():
    weights = torch.tensor(
        [
            [1, 1, 1, 1],
            [4, 0, 0, 0],
            [2, 2, 0, 0],
            [3, 1, 1, 0],
        ],
        dtype=torch.int8,
    )
    tensors = {
        "experts.w1": weights.reshape(4, 1, 4).clone(),
        "experts.s1": torch.ones(4, 1, 1),
        "experts.w2": weights.reshape(4, 1, 4).clone(),
        "experts.s2": torch.ones(4, 1, 1),
        "experts.w3": weights.reshape(4, 1, 4).clone(),
        "experts.s3": torch.ones(4, 1, 1),
        "experts.r1": torch.eye(4),
        "experts.r2": torch.eye(4),
        "experts.r3": torch.eye(4),
        "blocks.0.router.weight": torch.arange(12).reshape(4, 3),
    }
    manifest = {
        "format": "mirai.compressed_weights.packed_state",
        "schema_version": 5,
        "modules": {
            "blocks.0.experts": {
                "kind": "grouped_experts",
                "num_experts": 4,
                "quant_format": "int8",
                "tensors": {
                    "w1_int8": "experts.w1",
                    "w1_scale": "experts.s1",
                    "w2_int8": "experts.w2",
                    "w2_scale": "experts.s2",
                    "w3_int8": "experts.w3",
                    "w3_scale": "experts.s3",
                    "w1_rotation": "experts.r1",
                    "w2_rotation": "experts.r2",
                    "w3_rotation": "experts.r3",
                },
                "rotations": {
                    "w1": "experts.r1",
                    "w2": "experts.r2",
                    "w3": "experts.r3",
                },
                "group_sizes": {"w1": 4, "w2": 4, "w3": 4},
                "shapes": {
                    "w1": [4, 1, 4],
                    "w2": [4, 1, 4],
                    "w3": [4, 1, 4],
                },
            }
        },
        "residual_tensors": {"blocks.0.router.weight": "blocks.0.router.weight"},
        "summary": {"quantized_numel": 48},
    }
    return tensors, manifest


def test_manifest_authorizes_only_sibling_router_expert_axis_resize() -> None:
    class Router(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 4
            self.weight = nn.Parameter(torch.zeros(4, 3))
            self.register_buffer("correction_bias", torch.zeros(4))

    root = nn.Module()
    root.router = Router()
    assign_packed_state_tensor(
        root,
        "router.weight",
        torch.ones(2, 3),
        expert_axis_size=2,
    )
    assign_packed_state_tensor(
        root,
        "router.correction_bias",
        torch.ones(2),
        expert_axis_size=2,
    )
    assert root.router.num_experts == 2
    assert tuple(root.router.weight.shape) == (2, 3)
    assert tuple(root.router.correction_bias.shape) == (2,)

    with pytest.raises(ValueError, match="expected"):
        assign_packed_state_tensor(
            root,
            "router.weight",
            torch.ones(2, 4),
            expert_axis_size=2,
        )


def test_prototype_consolidation_preserves_source_and_logical_router_space() -> None:
    tensors, manifest = _packed_fixture()
    source_manifest = copy.deepcopy(manifest)
    source_weight = tensors["experts.w1"].clone()
    output, converted = consolidate_packed_state(
        tensors, manifest, {"blocks.0.experts": (0, 0, 2, 2)}
    )

    assert manifest == source_manifest
    assert torch.equal(tensors["experts.w1"], source_weight)
    spec = converted["modules"]["blocks.0.experts"]
    assert converted["schema_version"] == 3
    assert spec["num_experts"] == 2
    assert spec["logical_num_experts"] == 4
    assert spec["logical_to_physical"] == [0, 0, 1, 1]
    assert output["experts.w1"].shape[0] == 2
    assert output["blocks.0.router.weight"].shape[0] == 4


def _upcycling_fixture(
    *,
    quant_format: str = "int8",
    hidden_size: int = 4,
    intermediate_size: int = 4,
    group_sizes: tuple[int, ...] = (4,),
):
    class Router(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 2
            self.weight = nn.Parameter(
                torch.linspace(
                    -0.4,
                    0.5,
                    steps=2 * hidden_size,
                    dtype=torch.float32,
                ).reshape(2, hidden_size)
            )
            self.register_buffer(
                "e_score_correction_bias",
                torch.tensor([0.5, -0.5], dtype=torch.float32),
            )

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_experts = 2
            self.router = Router()
            self.experts = CompressedGroupedExperts.from_empty(
                num_experts=2,
                group_sizes=group_sizes,
                expert_weight_access="active_dequant",
                quant_format=quant_format,
            )
            generator = torch.Generator(device="cpu").manual_seed(17)
            self.experts.load_dense_weight(
                "w1",
                torch.randn(2, intermediate_size, hidden_size, generator=generator),
            )
            self.experts.load_dense_weight(
                "w2",
                torch.randn(2, hidden_size, intermediate_size, generator=generator),
            )
            self.experts.load_dense_weight(
                "w3",
                torch.randn(2, intermediate_size, hidden_size, generator=generator),
            )

    root = nn.Module()
    root.blocks = nn.ModuleList([Block()])
    tensors, manifest = export_compressed_weights_packed_state(root)
    return root, tensors, manifest


def test_drop_upcycling_expands_physical_experts_and_router_deterministically() -> None:
    _root, tensors, manifest = _upcycling_fixture()
    policy = DropUpcyclingSpec(
        copies_per_expert=1,
        reinitialization_ratio=0.5,
        seed=123,
    )
    output, converted = drop_upcycle_packed_state(tensors, manifest, policy)
    repeated, repeated_manifest = drop_upcycle_packed_state(tensors, manifest, policy)
    assert converted == repeated_manifest
    assert set(output) == set(repeated)
    for key in output:
        assert torch.equal(output[key], repeated[key])

    module_spec = converted["modules"]["blocks.0.experts"]
    assert module_spec["num_experts"] == 4
    assert module_spec["expert_upcycling"]["source_num_experts"] == 2
    assert module_spec["expert_upcycling"]["expanded_num_experts"] == 4
    tensor_names = manifest["modules"]["blocks.0.experts"]["tensors"]
    for key in ("w1_int8", "w1_scale", "w2_int8", "w2_scale", "w3_int8", "w3_scale"):
        tensor_key = tensor_names[key]
        assert torch.equal(output[tensor_key][0], tensors[tensor_key][0])
        assert torch.equal(output[tensor_key][2], tensors[tensor_key][1])

    router_key = manifest["residual_tensors"]["blocks.0.router.weight"]
    bias_key = manifest["residual_tensors"]["blocks.0.router.e_score_correction_bias"]
    assert torch.equal(output[router_key][0], tensors[router_key][0])
    assert torch.equal(output[router_key][2], tensors[router_key][1])
    assert not torch.equal(output[router_key][1], output[router_key][0])
    assert not torch.equal(output[router_key][3], output[router_key][2])
    torch.testing.assert_close(output[bias_key], torch.tensor([0.5, 0.0, -0.5, 0.0]))

    converted["expert_upcycling"]["lineage"] = {
        "source_artifact_fingerprint": "sha256:" + ("a" * 64)
    }
    validate_drop_upcycling_selection(
        converted,
        mode="drop",
        copies_per_expert=1,
        reinitialization_ratio=0.5,
        seed=123,
    )
    with pytest.raises(ValueError, match="policy"):
        validate_drop_upcycling_selection(
            converted,
            mode="drop",
            copies_per_expert=2,
            reinitialization_ratio=0.5,
            seed=123,
        )

    restored, _unused, _unused_manifest = _upcycling_fixture()
    prepare_compressed_weights_modules_from_manifest(restored, converted)
    load_compressed_weights_packed_state(restored, output, converted)
    assert restored.blocks[0].router.num_experts == 4
    assert restored.blocks[0].experts.num_experts == 4
    assert tuple(restored.blocks[0].router.weight.shape) == (4, 4)


@pytest.mark.parametrize(
    "quant_format",
    (
        "int8",
        "nf4",
        "gguf_iq4",
        "gguf_iq3",
        "gguf_iq2",
        "mxfp8_e4m3",
        "mxfp4",
        "nvfp4",
        "fp8",
    ),
)
def test_drop_upcycling_roundtrips_every_regular_packed_format(
    quant_format: str,
) -> None:
    _root, tensors, manifest = _upcycling_fixture(
        quant_format=quant_format,
        hidden_size=32,
        intermediate_size=8,
        group_sizes=(16, 4),
    )
    output, converted = drop_upcycle_packed_state(
        tensors,
        manifest,
        DropUpcyclingSpec(
            copies_per_expert=1,
            reinitialization_ratio=0.5,
            seed=321,
        ),
    )
    source_spec = manifest["modules"]["blocks.0.experts"]
    converted_spec = converted["modules"]["blocks.0.experts"]
    assert converted_spec["num_experts"] == 4
    changed_expert_payload = False
    for tensor_key in source_spec["tensors"].values():
        source_tensor = tensors[tensor_key]
        if source_tensor.ndim < 1 or int(source_tensor.shape[0]) != 2:
            continue
        assert torch.equal(output[tensor_key][0], source_tensor[0])
        assert torch.equal(output[tensor_key][2], source_tensor[1])
        changed_expert_payload = changed_expert_payload or not torch.equal(
            output[tensor_key][1], source_tensor[0]
        )
    assert changed_expert_payload

    restored, _unused, _unused_manifest = _upcycling_fixture(
        quant_format=quant_format,
        hidden_size=32,
        intermediate_size=8,
        group_sizes=(16, 4),
    )
    prepare_compressed_weights_modules_from_manifest(restored, converted)
    load_compressed_weights_packed_state(restored, output, converted)
    assert restored.blocks[0].experts.num_experts == 4
    assert restored.blocks[0].router.num_experts == 4
    assert tuple(restored.blocks[0].router.weight.shape) == (4, 32)


def test_structured_pruning_enforces_topk_and_rewrites_router_rows() -> None:
    with pytest.raises(ValueError, match="hard floor"):
        select_pruned_experts(
            {"blocks.0": torch.tensor([10.0, 0.0, 0.0, 0.0])},
            score_threshold=0.0,
            top_k=2,
        )
    tensors, manifest = _packed_fixture()
    output, converted = prune_packed_state(tensors, manifest, {"blocks.0.experts": (1, 3)})
    assert converted["modules"]["blocks.0.experts"]["num_experts"] == 2
    assert output["experts.w1"].shape[0] == 2
    assert output["blocks.0.router.weight"].shape[0] == 2
    assert tensors["experts.w1"].shape[0] == 4


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        ("frequency", [2.0, 2.0]),
        ("reap", [1.25, 5.5]),
        ("man", [3.5, 8.5]),
        ("msan", [12.5, 84.5]),
    ],
)
def test_pruning_criteria_match_paper_formulas(
    criterion: str,
    expected: list[float],
) -> None:
    accumulator = ExpertPruningSaliencyAccumulator(2, criterion=criterion)
    accumulator.record(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0.5, 0.25, 1.0, 0.5]),
        torch.tensor(
            [
                [3.0, 0.0],
                [0.0, 4.0],
                [3.0, 4.0],
                [0.0, 12.0],
            ]
        ),
    )
    torch.testing.assert_close(
        accumulator.evidence().scores(),
        torch.tensor(expected, dtype=torch.float64),
    )


def test_aimer_matches_weight_only_formula_and_keeps_low_scores() -> None:
    tensors, manifest = _aimer_packed_fixture()
    spec = manifest["modules"]["blocks.0.experts"]
    source = CompressedExpertProjectionSource(load_grouped_expert_source(spec, tensors))
    result = aimer_expert_scores(
        source,
        max_block_elements=4,
        device="cpu",
    )
    flattened = torch.cat(
        [
            tensors["experts.w1"].reshape(4, -1).double(),
            tensors["experts.w2"].reshape(4, -1).double(),
            tensors["experts.w3"].reshape(4, -1).double(),
        ],
        dim=1,
    )
    expected = flattened.abs().sum(dim=1) / (
        math.sqrt(flattened.shape[1]) * torch.linalg.vector_norm(flattened, ord=2, dim=1)
    )
    torch.testing.assert_close(result.scores, expected)
    assert result.elements_per_expert == 12
    keep = select_pruned_experts(
        {"blocks.0.experts": result.scores},
        keep_fraction=0.5,
        top_k=2,
        keep_largest=False,
    )
    assert keep == {"blocks.0.experts": (1, 2)}
    with pytest.raises(ValueError, match="calibration-free"):
        normalize_calibrated_expert_pruning_criterion("aimer")


def test_pruning_observer_joins_sorted_outputs_to_original_routes() -> None:
    accumulator = ExpertPruningSaliencyAccumulator(2, criterion="reap")
    observer = ExpertPruningRoutedOutputObserver(accumulator)
    observer.bind_routes(
        torch.tensor([[1, 0], [0, 1]]),
        torch.tensor([[0.25, 0.5], [1.0, 0.75]]),
    )
    observer.capture_sorted(
        torch.tensor([[3.0, 4.0], [0.0, 2.0], [4.0, 0.0], [0.0, 6.0]]),
        torch.tensor([1, 2, 0, 3]),
        num_tokens=2,
        top_k=2,
    )
    # Expert 0: (0.5*5 + 1.0*2)/2 = 2.25.
    # Expert 1: (0.25*4 + 0.75*6)/2 = 2.75.
    torch.testing.assert_close(
        accumulator.evidence().scores(),
        torch.tensor([2.25, 2.75], dtype=torch.float64),
    )


def test_pruning_observer_runs_through_compressed_dispatch_lifecycle() -> None:
    from mirai.core.models.compressed_weights import CompressedGroupedExperts

    module = CompressedGroupedExperts.from_empty(
        num_experts=2,
        group_sizes=4,
        expert_weight_access="active_dequant",
        expert_dequant_chunk_size=1,
        quant_format="int8",
    )
    torch.manual_seed(29)
    for key, shape in {
        "w1": (2, 8, 4),
        "w2": (2, 4, 8),
        "w3": (2, 8, 4),
    }.items():
        module.load_dense_weight(key, torch.randn(shape))
    accumulator = ExpertPruningSaliencyAccumulator(2, criterion="reap")
    module.set_routed_output_observer(ExpertPruningRoutedOutputObserver(accumulator))
    module.train()
    output = module.run_direct_routed(
        torch.randn(2, 4),
        torch.tensor([[0.75, 0.25], [0.4, 0.6]]),
        torch.tensor([[0, 1], [1, 0]]),
    )
    evidence = accumulator.evidence()
    assert output.shape == (2, 4)
    assert evidence.selected_count.tolist() == [2, 2]
    assert bool(torch.isfinite(evidence.scores()).all())
    assert bool(torch.all(evidence.scores() > 0))


def test_pruning_evidence_roundtrip_preserves_criterion_and_lineage() -> None:
    accumulator = ExpertPruningSaliencyAccumulator(2, criterion="man")
    accumulator.record(
        torch.tensor([0, 1]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "pruning.safetensors"
        save_expert_pruning_evidence(
            path,
            {"blocks.0.experts": accumulator.evidence()},
            dataset_snapshot_id="dataset",
            model_snapshot_id="model",
            config_snapshot_id="config",
        )
        loaded, lineage = load_expert_pruning_evidence(
            path,
            expected_criterion="man",
        )
    assert lineage == {
        "dataset_snapshot_id": "dataset",
        "model_snapshot_id": "model",
        "config_snapshot_id": "config",
    }
    torch.testing.assert_close(
        loaded["blocks.0.experts"].scores(),
        torch.tensor([5.0, 2.0], dtype=torch.float64),
    )


def test_pruning_tool_consumes_safe_evidence_and_reemits_lineage() -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from mirai.config.schema import TrainingConfig
    from mirai.core.models.compressed_weights.packed.packed_state import (
        COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY,
    )
    from scripts.tools.prune_experts import prune_packed_base

    tensors, manifest = _packed_fixture()
    config = TrainingConfig()
    config.model.params.expert_pruning = "prune"
    config.model.params.expert_pruning_criterion = "man"
    config.model.params.experts_per_token = 2
    evidence = ExpertPruningEvidence(
        criterion="man",
        score_sum=torch.tensor([1.0, 4.0, 2.0, 3.0], dtype=torch.float64),
        selected_count=torch.ones(4, dtype=torch.int64),
    ).validate()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        packed_path = root / "packed.safetensors"
        evidence_path = root / "evidence.safetensors"
        output_path = root / "pruned.safetensors"
        save_file(
            tensors,
            str(packed_path),
            metadata={COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY: json.dumps(manifest)},
        )
        save_expert_pruning_evidence(
            evidence_path,
            {"blocks.0.experts": evidence},
            dataset_snapshot_id="dataset",
            model_snapshot_id="model",
            config_snapshot_id="config",
        )
        report = prune_packed_base(
            config,
            packed_state=packed_path,
            calibration_file=evidence_path,
            output=output_path,
            keep_fraction=0.5,
        )
        output_tensors = load_file(str(output_path), device="cpu")
        with safe_open(str(output_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    assert report["criterion"] == "man"
    assert report["source_lineage"]["calibration"]["dataset_snapshot_id"] == "dataset"
    assert output_tensors["experts.w1"].shape[0] == 2
    assert output_tensors["blocks.0.router.weight"].shape[0] == 2
    assert "expert_pruning_calibration_lineage" in metadata


def test_aimer_pruning_needs_no_calibration_and_persists_weight_lineage() -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from mirai.config.schema import TrainingConfig
    from mirai.core.models.compressed_weights.packed.packed_state import (
        COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY,
    )
    from scripts.tools.prune_experts import prune_packed_base

    tensors, manifest = _aimer_packed_fixture()
    config = TrainingConfig()
    config.model.params.expert_pruning = "prune"
    config.model.params.expert_pruning_criterion = "aimer"
    config.model.params.experts_per_token = 2
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        packed_path = root / "packed.safetensors"
        output_path = root / "pruned.safetensors"
        save_file(
            tensors,
            str(packed_path),
            metadata={COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY: json.dumps(manifest)},
        )
        report = prune_packed_base(
            config,
            packed_state=packed_path,
            calibration_file=None,
            output=output_path,
            keep_fraction=0.5,
            max_block_elements=4,
            metric_device="cpu",
        )
        output_tensors = load_file(str(output_path), device="cpu")
        with safe_open(str(output_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        output_manifest = json.loads(metadata[COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY])
    assert report["criterion"] == "aimer"
    assert report["source_lineage"]["scoring"] == "weight_only"
    assert output_tensors["experts.w1"].shape[0] == 2
    assert output_tensors["blocks.0.router.weight"].shape[0] == 2
    assert torch.equal(output_tensors["experts.r1"], torch.eye(4))
    assert "expert_pruning_calibration_lineage" not in metadata
    transform = output_manifest["expert_pruning_transform"]
    assert transform["score_direction"] == "larger_is_more_removable"
    assert transform["kept_experts"]["blocks.0.experts"] == [1, 2]


def test_affinity_calibration_coalesces_routes_and_balances_samples() -> None:
    accumulator = ExpertAffinityAccumulator(3)
    accumulator.record(
        torch.tensor([[0, 0], [1, 2]]),
        torch.tensor([[0.25, 0.75], [0.6, 0.4]]),
    )
    evidence = accumulator.evidence()
    assert evidence.selected_count.tolist() == [1, 1, 1]
    weights = evidence.affinity_reconstruction_weights()
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0, dtype=weights.dtype))
    assert evidence.balanced_sample_indices(1) == (0,)


def test_shared_basis_transform_writes_provider_schema_without_source_mutation() -> None:
    tensors = {}
    tensor_map = {}
    shapes = {}
    group_sizes = {}
    for key in ("w1", "w2", "w3"):
        quantized_name = f"experts.{key}_int8"
        scale_name = f"experts.{key}_scale"
        tensors[quantized_name] = torch.arange(48).reshape(3, 4, 4).to(torch.int8)
        tensors[scale_name] = torch.ones(3, 4, 1)
        tensor_map[f"{key}_int8"] = quantized_name
        tensor_map[f"{key}_scale"] = scale_name
        shapes[key] = [3, 4, 4]
        group_sizes[key] = 4
    manifest = {
        "format": "mirai.compressed_weights.packed_state",
        "schema_version": 1,
        "modules": {
            "blocks.0.experts": {
                "kind": "grouped_experts",
                "quant_format": "int8",
                "num_experts": 3,
                "tensors": tensor_map,
                "shapes": shapes,
                "group_sizes": group_sizes,
            }
        },
    }
    source_keys = set(tensors)
    output, converted, report = factorize_packed_state_shared_basis(
        tensors, manifest, rank=1, device="cpu", factor_dtype="float32"
    )
    spec = converted["modules"]["blocks.0.experts"]
    assert manifest["schema_version"] == 1
    assert set(tensors) == source_keys
    assert converted["schema_version"] == 4
    assert spec["physical_weight_provider"]["name"] == "shared_basis"
    assert not source_keys.intersection(output)
    assert report["grouped_modules"] == 1
    assert report["factor_bytes"] > 0


def test_shared_basis_transform_consumes_lineage_bound_whitening_evidence() -> None:
    tensors = {}
    tensor_map = {}
    shapes = {}
    group_sizes = {}
    for key in ("w1", "w2", "w3"):
        quantized_name = f"experts.{key}_int8"
        scale_name = f"experts.{key}_scale"
        tensors[quantized_name] = torch.arange(48).reshape(3, 4, 4).to(torch.int8)
        tensors[scale_name] = torch.ones(3, 4, 1)
        tensor_map[f"{key}_int8"] = quantized_name
        tensor_map[f"{key}_scale"] = scale_name
        shapes[key] = [3, 4, 4]
        group_sizes[key] = 4
    manifest = {
        "format": "mirai.compressed_weights.packed_state",
        "schema_version": 1,
        "modules": {
            "blocks.0.experts": {
                "kind": "grouped_experts",
                "quant_format": "int8",
                "num_experts": 3,
                "tensors": tensor_map,
                "shapes": shapes,
                "group_sizes": group_sizes,
            }
        },
    }
    covariance = ProjectionCovarianceEvidence(
        covariance=torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])),
        sample_count=8,
    )
    evidence = ExpertWhiteningEvidence({"w1": covariance, "w2": covariance, "w3": covariance})
    lineage = {
        "dataset_snapshot_id": "dataset",
        "model_snapshot_id": "model",
        "config_snapshot_id": "config",
        "packed_artifact_fingerprint": "sha256:" + ("a" * 64),
    }
    _output, converted, report = factorize_packed_state_shared_basis(
        tensors,
        manifest,
        rank=1,
        device="cpu",
        factor_dtype="float32",
        whitening_evidence={"blocks.0.experts": evidence},
        whitening_lineage=lineage,
    )
    provider = converted["modules"]["blocks.0.experts"]["physical_weight_provider"]
    assert provider["basis_estimator"] == "whitened_population"
    assert converted["shared_basis_whitening"]["packed_artifact_fingerprint"] == (
        "sha256:" + ("a" * 64)
    )
    assert set(report["modules"]["blocks.0.experts"]["whitened_relative_error"]) == {
        "w1",
        "w2",
        "w3",
    }


def test_mixture_basis_optimizes_softmax_basis_reconstruction() -> None:
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(6, 8, 8, generator=generator)
    factors = factorize_mixture_basis_experts(
        source,
        rank=2,
        basis_count=2,
        activation="tanh",
        optimization_steps=4,
        learning_rate=0.03,
        expert_batch_size=2,
        row_chunk_size=4,
        checkpoint_interval=1,
        factor_dtype="float32",
        device="cpu",
        max_covariance_gib=0.01,
        max_optimizer_gib=0.01,
    )
    assert factors.optimization_steps == 4
    assert factors.optimized_relative_frobenius_error <= (factors.initial_relative_frobenius_error)
    assert math.isfinite(factors.stored_relative_frobenius_error)
    torch.testing.assert_close(
        factors.coefficients.sum(dim=-1),
        torch.ones(6),
        atol=1e-6,
        rtol=1e-6,
    )


def test_mixture_basis_artifact_preserves_packed_down_projection() -> None:
    tensors = {}
    tensor_map = {}
    shapes = {}
    group_sizes = {}
    generator = torch.Generator().manual_seed(23)
    for key in ("w1", "w2", "w3"):
        quantized_name = f"experts.{key}_int8"
        scale_name = f"experts.{key}_scale"
        tensors[quantized_name] = torch.randint(
            -16,
            17,
            (8, 8, 8),
            generator=generator,
            dtype=torch.int8,
        )
        tensors[scale_name] = torch.full((8, 8, 1), 0.05)
        tensor_map[f"{key}_int8"] = quantized_name
        tensor_map[f"{key}_scale"] = scale_name
        shapes[key] = [8, 8, 8]
        group_sizes[key] = 0
    manifest = {
        "format": "mirai.compressed_weights.packed_state",
        "schema_version": 1,
        "modules": {
            "blocks.0.experts": {
                "kind": "grouped_experts",
                "quant_format": "int8",
                "num_experts": 8,
                "tensors": tensor_map,
                "shapes": shapes,
                "group_sizes": group_sizes,
            }
        },
    }
    output, converted, report = factorize_packed_state_mixture_basis(
        tensors,
        manifest,
        rank=2,
        basis_count=2,
        activation="silu",
        optimization_steps=2,
        learning_rate=0.03,
        expert_batch_size=4,
        row_chunk_size=4,
        checkpoint_interval=1,
        factor_dtype="float16",
        device="cpu",
        source_artifact_fingerprint="sha256:" + ("b" * 64),
        max_covariance_gib=0.01,
        max_optimizer_gib=0.01,
    )
    spec = converted["modules"]["blocks.0.experts"]
    provider_spec = spec["physical_weight_provider"]
    assert provider_spec["name"] == "mixture_basis"
    assert set(provider_spec["projections"]) == {"w1", "w3"}
    assert provider_spec["down_projection"]["quant_format"] == "int8"
    assert "experts.w2_int8" in output
    assert "experts.w2_scale" in output
    assert "experts.w1_int8" not in output
    assert "experts.w3_int8" not in output
    assert report["byte_ratio"] < 1.0
    assert converted["mixture_basis_transform"]["source_artifact_fingerprint"] == "sha256:" + (
        "b" * 64
    )

    provider = MixtureBasisPhysicalWeightProvider(
        PhysicalWeightProviderContext(
            module_name="blocks.0.experts",
            num_experts=8,
            shapes=shapes,
            spec=provider_spec,
            tensors=output,
        )
    )
    expected_w2 = tensors["experts.w2_int8"][3].float() * tensors["experts.w2_scale"][3]
    torch.testing.assert_close(
        provider.materialize_expert(
            "w2",
            3,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
        expected_w2,
    )
    assert provider.materialize_expert(
        "w1",
        3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).shape == (8, 8)


def test_mixture_basis_artifact_preserves_blockwise_fp8_down_projection() -> None:
    torch.manual_seed(29)
    root = nn.Module()
    root.experts = CompressedGroupedExperts.from_empty(
        num_experts=4,
        group_sizes=128,
        expert_weight_access="active_dequant",
        quant_format="fp8",
    )
    for key in ("w1", "w2", "w3"):
        root.experts.load_dense_weight(key, torch.randn(4, 8, 8))
    tensors, manifest = export_compressed_weights_packed_state(root)
    expected_w2 = root.experts._dequantize_expert(
        "w2",
        2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    output, converted, _report = factorize_packed_state_mixture_basis(
        tensors,
        manifest,
        rank=2,
        basis_count=2,
        activation="silu",
        optimization_steps=2,
        learning_rate=0.03,
        expert_batch_size=2,
        row_chunk_size=4,
        checkpoint_interval=1,
        factor_dtype="float16",
        device="cpu",
        source_artifact_fingerprint="sha256:" + ("c" * 64),
        max_covariance_gib=0.01,
        max_optimizer_gib=0.01,
    )
    provider_spec = converted["modules"]["experts"]["physical_weight_provider"]
    down_projection = provider_spec["down_projection"]
    assert down_projection["quant_format"] == "fp8"
    assert set(down_projection["tensors"]) == {"codes", "scales"}
    provider = MixtureBasisPhysicalWeightProvider(
        PhysicalWeightProviderContext(
            module_name="experts",
            num_experts=4,
            shapes={key: [4, 8, 8] for key in ("w1", "w2", "w3")},
            spec=provider_spec,
            tensors=output,
        )
    )
    torch.testing.assert_close(
        provider.materialize_expert(
            "w2",
            2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
        expected_w2,
        rtol=0.0,
        atol=0.0,
    )


def _stun_packed_fixture():
    tensors = {}
    tensor_map = {}
    shapes = {}
    group_sizes = {}
    for offset, key in enumerate(("w1", "w2", "w3"), start=1):
        quantized_name = f"experts.{key}_int8"
        scale_name = f"experts.{key}_scale"
        base = torch.arange(1, 65, dtype=torch.int8).reshape(4, 4, 4)
        tensors[quantized_name] = (base + offset).contiguous()
        tensors[scale_name] = torch.ones(4, 4, 1)
        tensor_map[f"{key}_int8"] = quantized_name
        tensor_map[f"{key}_scale"] = scale_name
        shapes[key] = [4, 4, 4]
        group_sizes[key] = 4
    tensors["blocks.0.router.weight"] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0],
            [10.1, 10.0, 10.0, 10.0],
        ],
        dtype=torch.float32,
    )
    tensors["blocks.0.router.bias"] = torch.arange(4, dtype=torch.float32)
    manifest = {
        "format": "mirai.compressed_weights.packed_state",
        "schema_version": 1,
        "quant_formats": ["int8"],
        "modules": {
            "blocks.0.experts": {
                "kind": "grouped_experts",
                "quant_format": "int8",
                "num_experts": 4,
                "expert_weight_access": "active_dequant",
                "expert_dequant_chunk_size": 1,
                "tensors": tensor_map,
                "shapes": shapes,
                "group_sizes": group_sizes,
            }
        },
        "residual_tensors": {
            "blocks.0.router.weight": "blocks.0.router.weight",
            "blocks.0.router.bias": "blocks.0.router.bias",
        },
    }
    return tensors, manifest


def test_stun_router_clustering_and_centroid_selection_match_source_formulas() -> None:
    router = torch.tensor([[0.0, 0.0], [0.1, 0.0], [9.0, 9.0], [9.1, 9.0]])
    clusters = cluster_router_experts(router, target_experts=2)
    assert clusters == ((0, 1), (2, 3))
    weights = {
        key: torch.stack([torch.full((2, 4), value + offset) for value in (0.0, 2.0, 10.0, 14.0)])
        for offset, key in enumerate(("w1", "w2", "w3"))
    }
    plan = select_stun_representatives(
        clusters,
        weights,
        reconstruct_below=2,
    )
    assert [cluster.representative for cluster in plan.clusters] == [0, 2]
    assert not any(cluster.reconstruct for cluster in plan.clusters)
    torch.testing.assert_close(
        apply_stun_plan(torch.arange(4), plan),
        torch.tensor([0, 2]),
    )

    reconstructed = select_stun_representatives(
        (tuple(range(4)),),
        weights,
        reconstruct_below=3,
    )
    assert reconstructed.clusters[0].reconstruct
    torch.testing.assert_close(
        apply_stun_plan(torch.arange(4, dtype=torch.float32), reconstructed),
        torch.tensor([1.5]),
    )


def test_compact_sparse24_roundtrip_retains_exact_structure() -> None:
    weight = torch.tensor(
        [
            [1.0, -4.0, 3.0, 2.0, 8.0, 5.0, -7.0, 6.0],
            [9.0, 1.0, 2.0, -8.0, 3.0, -6.0, 4.0, 5.0],
        ]
    )
    packed = pack_sparse24(weight, quant_group_size=1)
    validate_packed_sparse24(packed)
    dense = packed.dense(dtype=torch.float32, device=torch.device("cpu"))
    assert dense.shape == weight.shape
    assert torch.count_nonzero(dense.reshape(2, 2, 4), dim=-1).tolist() == [
        [2, 2],
        [2, 2],
    ]
    expected_mask = torch.zeros_like(weight, dtype=torch.bool).reshape(2, 2, 4)
    expected_mask.scatter_(
        -1,
        weight.reshape(2, 2, 4).abs().topk(2, dim=-1).indices,
        True,
    )
    assert torch.equal(dense != 0, expected_mask.reshape_as(weight))
    dense_reference = weight * expected_mask.reshape_as(weight)
    torch.testing.assert_close(dense, dense_reference, rtol=0.01, atol=0.04)

    larger = torch.linspace(-4.0, 4.0, 2 * 4 * 128).reshape(2, 4, 128)
    compact = pack_sparse24(larger, quant_group_size=32)
    compact_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (compact.values, compact.scales, compact.positions)
    )
    dense_bf16_bytes = larger.numel() * torch.tensor([], dtype=torch.bfloat16).element_size()
    assert compact_bytes < dense_bf16_bytes


def test_stun_sparse24_transform_enforces_stage_order_and_provider_roundtrip() -> None:
    tensors, manifest = _stun_packed_fixture()
    source_manifest = copy.deepcopy(manifest)
    source_tensors = {key: value.clone() for key, value in tensors.items()}
    output, converted, report = transform_packed_state_stun_sparse24(
        tensors,
        manifest,
        target_experts=2,
        device=torch.device("cpu"),
        reconstruct_below=3,
        quant_group_size=32,
    )
    assert manifest == source_manifest
    assert all(torch.equal(tensors[key], source_tensors[key]) for key in tensors)
    spec = converted["modules"]["blocks.0.experts"]
    provider_spec = spec["physical_weight_provider"]
    assert converted["schema_version"] == 4
    assert spec["num_experts"] == 2
    assert provider_spec["structured_stage"] == "stun_router_similarity"
    assert provider_spec["second_stage"] == "semi_structured_2_4_adaptation"
    assert report["modules"]["blocks.0.experts"]["clusters"] == [[0, 1], [2, 3]]
    assert output["blocks.0.router.weight"].shape == (2, 4)
    assert output["blocks.0.router.bias"].shape == (2,)
    assert not any(
        name in output
        for name in source_manifest["modules"]["blocks.0.experts"]["tensors"].values()
    )

    provider = Sparse24PhysicalWeightProvider(
        PhysicalWeightProviderContext(
            module_name="blocks.0.experts",
            num_experts=2,
            shapes={key: tuple(value) for key, value in spec["shapes"].items()},
            spec=provider_spec,
            tensors=output,
        )
    )
    materialized = provider.materialize_expert(
        "w1",
        0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert materialized.shape == (4, 4)
    positions_name = provider_spec["projections"]["w1"]["positions"]
    assert output[positions_name].shape == (2, 4, 1, 2)
    assert report["packed_expert_bytes"] > 0
    assert math.isfinite(report["byte_ratio"])

    from mirai.core.models.compressed_weights import (
        load_compressed_weights_packed_tensors,
        read_compressed_weights_packed_state_manifest,
        save_compressed_weights_packed_tensors,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "stun_sparse24.safetensors"
        written = save_compressed_weights_packed_tensors(
            path,
            output,
            converted,
            metadata={"expert_weight_compression": "stun_sparse24"},
        )
        restored_manifest = read_compressed_weights_packed_state_manifest(written)
        restored_tensors = load_compressed_weights_packed_tensors(written)
    assert (
        restored_manifest["structured_expert_compression"]
        == (converted["structured_expert_compression"])
    )
    assert set(restored_tensors) == set(output)
    assert all(torch.equal(restored_tensors[name], output[name]) for name in output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_stun_sparse24_provider_cuda_output_and_input_gradient_parity() -> None:
    import torch.nn.functional as F

    from mirai.core.models.compressed_weights import CompressedGroupedExperts

    tensors, manifest = _stun_packed_fixture()
    output, converted, _report = transform_packed_state_stun_sparse24(
        tensors,
        manifest,
        target_experts=2,
        device=torch.device("cuda"),
        reconstruct_below=3,
        quant_group_size=32,
    )
    spec = converted["modules"]["blocks.0.experts"]
    provider = Sparse24PhysicalWeightProvider(
        PhysicalWeightProviderContext(
            module_name="blocks.0.experts",
            num_experts=2,
            shapes={key: tuple(value) for key, value in spec["shapes"].items()},
            spec=spec["physical_weight_provider"],
            tensors=output,
        )
    )
    module = CompressedGroupedExperts.from_empty(
        num_experts=2,
        quant_format="int8",
        expert_weight_access="active_dequant",
        expert_dequant_chunk_size=1,
    ).cuda()
    module.bind_physical_weight_provider(provider)
    indices = torch.tensor([[0, 1], [1, 0], [0, 1]], device="cuda")
    scores = torch.tensor(
        [[0.75, 0.25], [0.6, 0.4], [0.9, 0.1]],
        device="cuda",
    )
    actual_input = torch.randn(3, 4, device="cuda", requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    actual = module.run_direct_routed(actual_input, scores, indices)

    reference = torch.zeros_like(reference_input)
    for token_index in range(indices.shape[0]):
        for route_index in range(indices.shape[1]):
            expert = int(indices[token_index, route_index].item())
            token = reference_input[token_index]
            w1 = provider.materialize_expert("w1", expert, dtype=token.dtype, device=token.device)
            w2 = provider.materialize_expert("w2", expert, dtype=token.dtype, device=token.device)
            w3 = provider.materialize_expert("w3", expert, dtype=token.dtype, device=token.device)
            hidden = F.silu(token @ w1.transpose(0, 1))
            hidden = hidden * (token @ w3.transpose(0, 1))
            routed = hidden @ w2.transpose(0, 1)
            reference[token_index] = (
                reference[token_index] + routed * scores[token_index, route_index]
            )
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    actual.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        rtol=1e-5,
        atol=1e-5,
    )


def _flexmoe_weights(*, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(31)
    return {
        "w1": torch.randn(2, 4, 3, generator=generator, requires_grad=requires_grad),
        "w2": torch.randn(2, 3, 4, generator=generator, requires_grad=requires_grad),
        "w3": torch.randn(2, 4, 3, generator=generator, requires_grad=requires_grad),
    }


def test_flexmoe_equation2_saliency_and_batch_average() -> None:
    weights = _flexmoe_weights()
    gradients = {name: torch.full_like(value, 0.5) for name, value in weights.items()}
    actual = channel_taylor_saliency(weights, gradients)
    expected = (
        (weights["w1"] * gradients["w1"]).square().sum(dim=2)
        + (weights["w2"] * gradients["w2"]).square().sum(dim=1)
        + (weights["w3"] * gradients["w3"]).square().sum(dim=2)
    )
    torch.testing.assert_close(actual, expected.float(), rtol=0.0, atol=0.0)

    accumulator = FlexMoEChannelSaliencyAccumulator()
    accumulator.update(weights, gradients)
    doubled = {name: value * 2.0 for name, value in gradients.items()}
    accumulator.update(weights, doubled)
    torch.testing.assert_close(
        accumulator.mean(),
        (expected.double() + expected.double() * 4.0) / 2.0,
        rtol=1e-12,
        atol=1e-12,
    )


def test_flexmoe_equation3_reordering_preserves_full_expert_function() -> None:
    weights = _flexmoe_weights()
    saliency = torch.tensor([[1.0, 4.0, 4.0, 2.0], [3.0, 1.0, 2.0, 0.0]])
    order = rank_channels(saliency)
    assert order.tolist() == [[1, 2, 3, 0], [0, 2, 1, 3]]
    reordered = apply_channel_permutation(weights, order)
    inputs = torch.randn(2, 5, 3)
    for expert in range(2):
        native = torch.nn.functional.silu(inputs[expert] @ weights["w1"][expert].transpose(0, 1))
        native = native * (inputs[expert] @ weights["w3"][expert].transpose(0, 1))
        native = native @ weights["w2"][expert].transpose(0, 1)
        ranked = torch.nn.functional.silu(inputs[expert] @ reordered["w1"][expert].transpose(0, 1))
        ranked = ranked * (inputs[expert] @ reordered["w3"][expert].transpose(0, 1))
        ranked = ranked @ reordered["w2"][expert].transpose(0, 1)
        torch.testing.assert_close(ranked, native, rtol=1e-6, atol=1e-6)


def test_flexmoe_straight_through_actions_replay_and_backpropagate() -> None:
    logits = torch.tensor(
        [[0.0, 0.5, 1.0, 2.0], [1.5, 0.2, -0.3, 0.0]],
        requires_grad=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(77)
    initial_state = generator.get_state()
    soft, hard, sampled = straight_through_gumbel_actions(
        logits,
        temperature=0.7,
        generator=generator,
    )
    replay = torch.Generator(device="cpu")
    replay.set_state(initial_state)
    replay_soft, replay_hard, replay_sampled = straight_through_gumbel_actions(
        logits,
        temperature=0.7,
        generator=replay,
    )
    torch.testing.assert_close(soft, replay_soft, rtol=0.0, atol=0.0)
    assert torch.equal(hard, replay_hard)
    torch.testing.assert_close(sampled, replay_sampled, rtol=0.0, atol=0.0)
    assert torch.equal(sampled.detach(), hard)
    assert torch.equal(hard.sum(dim=-1), torch.ones(2))
    sampled.mul(torch.tensor([0.1, 0.4, 0.7, 1.0])).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad).item() > 0


def test_flexmoe_action_controller_maps_ranked_prefixes_and_resumes() -> None:
    controller = FlexMoEActionController(
        num_experts=2,
        action_ratios=(0.25, 0.5, 0.75, 1.0),
        thickest_logit_margin=100.0,
    )
    order = torch.tensor([[3, 2, 1, 0], [1, 3, 0, 2]])
    generator = torch.Generator(device="cpu").manual_seed(123)
    initial_rng = generator.get_state()
    masks, soft, hard = controller.sampled_original_channel_masks(
        order,
        temperature=1.0,
        generator=generator,
    )
    assert torch.equal(hard[:, -1], torch.ones(2))
    assert torch.equal(masks.detach(), torch.ones(2, 4))
    masks.mul(torch.arange(4, dtype=torch.float32)).sum().backward()
    assert controller.logits.grad is not None
    assert torch.isfinite(controller.logits.grad).all()

    state = copy.deepcopy(controller.state_dict())
    restored = FlexMoEActionController(
        num_experts=2,
        action_ratios=(0.25, 0.5, 0.75, 1.0),
        thickest_logit_margin=1.0,
    )
    restored.load_state_dict(state)
    replay = torch.Generator(device="cpu")
    replay.set_state(initial_rng)
    replay_masks, replay_soft, replay_hard = restored.sampled_original_channel_masks(
        order,
        temperature=1.0,
        generator=replay,
    )
    torch.testing.assert_close(replay_masks, masks.detach(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(replay_soft, soft.detach(), rtol=0.0, atol=0.0)
    assert torch.equal(replay_hard, hard)
    regularization, terms = restored.regularization(
        torch.tensor([0.75, 0.25]),
        cost_weight=2.0,
        entropy_weight=0.5,
    )
    torch.testing.assert_close(
        regularization,
        2.0 * terms["cost"] - 0.5 * terms["entropy"],
    )


def test_flexmoe_sampled_mask_backpropagates_through_grouped_expert() -> None:
    generator = torch.Generator(device="cpu").manual_seed(51)
    module = CompressedGroupedExperts.from_empty(
        num_experts=2,
        group_sizes=(4,),
        expert_weight_access="chunked_dequant",
        expert_dequant_chunk_size=2,
    )
    module.load_dense_weight("w1", torch.randn(2, 4, 4, generator=generator))
    module.load_dense_weight("w2", torch.randn(2, 4, 4, generator=generator))
    module.load_dense_weight("w3", torch.randn(2, 4, 4, generator=generator))
    controller = FlexMoEActionController(
        num_experts=2,
        action_ratios=(0.5, 1.0),
        thickest_logit_margin=1.0,
    )
    masks, _soft, _hard = controller.sampled_original_channel_masks(
        torch.tensor([[3, 2, 1, 0], [1, 3, 0, 2]]),
        temperature=1.0,
        generator=torch.Generator(device="cpu").manual_seed(82),
    )
    module.set_flexmoe_channel_mask(masks)
    inputs = torch.randn(4, 4, generator=generator, requires_grad=True)
    indices = torch.tensor([[0], [1], [0], [1]])
    scores = torch.ones(4, 1)
    output = module.run_direct_routed(inputs, scores, indices)
    output.square().mean().backward()
    assert controller.logits.grad is not None
    assert torch.isfinite(controller.logits.grad).all()
    assert torch.count_nonzero(controller.logits.grad).item() > 0
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    module.set_flexmoe_channel_mask(None)
    assert not module.flexmoe_channel_mask_active


def test_flexmoe_grouped_host_taylor_capture_matches_trainable_reference() -> None:
    generator = torch.Generator(device="cpu").manual_seed(63)
    weights = {
        "w1": torch.randn(2, 4, 4, generator=generator),
        "w2": torch.randn(2, 4, 4, generator=generator),
        "w3": torch.randn(2, 4, 4, generator=generator),
    }
    module = CompressedGroupedExperts.from_empty(
        num_experts=2,
        group_sizes=(4,),
        expert_weight_access="chunked_dequant",
        expert_dequant_chunk_size=2,
    )
    for name, value in weights.items():
        module.load_dense_weight(name, value)
    observer = FlexMoETaylorGradientObserver(num_experts=2, intermediate_size=4)
    observer.begin_batch(device=torch.device("cpu"))
    module.set_flexmoe_taylor_observer(observer)
    inputs = torch.randn(5, 4, generator=generator, requires_grad=True)
    indices = torch.tensor([[0], [1], [0], [1], [1]])
    scores = torch.tensor([[0.7], [0.9], [0.4], [0.6], [0.8]])
    actual = module.run_direct_routed(inputs, scores, indices)
    actual.sin().sum().backward()
    observer.finish_batch()
    module.set_flexmoe_taylor_observer(None)

    reference_weights = {
        name: torch.stack(
            [
                module._dequantize_expert(
                    name,
                    expert,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                )
                for expert in range(module.num_experts)
            ]
        ).requires_grad_(True)
        for name in weights
    }
    reference_outputs = []
    for token, expert, score in zip(
        inputs.detach(),
        indices[:, 0],
        scores[:, 0],
        strict=True,
    ):
        index = int(expert.item())
        gate = token @ reference_weights["w1"][index].transpose(0, 1)
        up = token @ reference_weights["w3"][index].transpose(0, 1)
        hidden = torch.nn.functional.silu(gate) * up
        reference_outputs.append(
            hidden @ reference_weights["w2"][index].transpose(0, 1) * score
        )
    torch.stack(reference_outputs).sin().sum().backward()
    expected = channel_taylor_saliency(
        {name: value.detach() for name, value in reference_weights.items()},
        {name: value.grad for name, value in reference_weights.items()},
    )
    torch.testing.assert_close(
        observer.evidence().saliency,
        expected.to(torch.float64),
        # Production groups every expert's tokens into one GEMM while the
        # independent oracle evaluates one GEMV per token.  Their FP32
        # reduction order is intentionally different.
        rtol=5e-6,
        atol=2e-6,
    )

    module.set_flexmoe_channel_mask(torch.ones(2, 4))
    with pytest.raises(ValueError, match="separate stages"):
        module.set_flexmoe_taylor_observer(observer)
    module.set_flexmoe_channel_mask(None)


def test_flexmoe_equations9_to12_and_prefix_nesting() -> None:
    ratios = (0.1, 0.4, 0.7, 1.0)
    logits = torch.tensor([[0.0, 0.0, 0.0, 5.0], [5.0, 0.0, 0.0, 0.0]])
    probabilities = clean_action_probabilities(logits)
    load = torch.tensor([0.75, 0.25])
    cost = load_sensitive_cost(probabilities, load, ratios)
    expected = (load * probabilities.matmul(torch.tensor(ratios, dtype=torch.float32))).sum()
    torch.testing.assert_close(cost, expected, rtol=0.0, atol=0.0)
    assert action_entropy(probabilities).item() > 0.0
    hardened = hardened_retention_ratios(logits, ratios)
    torch.testing.assert_close(hardened, torch.tensor([1.0, 0.1]))
    torch.testing.assert_close(global_prune_budget(hardened), torch.tensor(0.45))
    masks = prefix_masks(torch.tensor(ratios), intermediate=10)
    assert masks.sum(dim=1).tolist() == [1.0, 4.0, 7.0, 10.0]
    assert all(torch.all(masks[index] <= masks[index + 1]) for index in range(3))


def test_flexmoe_ranking_and_action_artifacts_round_trip_exact_lineage() -> None:
    ranking = FlexMoERankingEvidence(
        saliency=torch.tensor(
            [[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]],
            dtype=torch.float64,
        ),
        calibration_batches=3,
    )
    plan = FlexMoEActionPlan(
        action_ratios=(0.1, 0.4, 0.7, 1.0),
        logits=torch.tensor(
            [[0.0, 0.0, 4.0, 0.0], [0.0, 4.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        expert_load=torch.tensor([0.75, 0.25], dtype=torch.float64),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        ranking_path = Path(temp_dir) / "ranking.safetensors"
        action_path = Path(temp_dir) / "actions.safetensors"
        save_ranking_evidence(
            ranking_path,
            {"blocks.0.experts": ranking},
            dataset_snapshot_id="dataset-a",
            model_snapshot_id="model-a",
            config_snapshot_id="config-a",
        )
        loaded_ranking, ranking_lineage = load_ranking_evidence(ranking_path)
        save_action_plans(
            action_path,
            {"blocks.0.experts": plan},
            dataset_snapshot_id="dataset-a",
            model_snapshot_id="model-a",
            config_snapshot_id="config-b",
            ranking_snapshot_id="ranking-a",
        )
        loaded_actions, action_lineage = load_action_plans(action_path)

    restored_ranking = loaded_ranking["blocks.0.experts"]
    torch.testing.assert_close(
        restored_ranking.saliency,
        ranking.saliency,
        rtol=0.0,
        atol=0.0,
    )
    assert restored_ranking.calibration_batches == 3
    assert ranking_lineage == {
        "dataset_snapshot_id": "dataset-a",
        "model_snapshot_id": "model-a",
        "config_snapshot_id": "config-a",
    }
    restored_plan = loaded_actions["blocks.0.experts"]
    torch.testing.assert_close(restored_plan.logits, plan.logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        restored_plan.expert_load,
        plan.expert_load,
        rtol=0.0,
        atol=0.0,
    )
    assert restored_plan.action_ratios == plan.action_ratios
    assert action_lineage == {
        "dataset_snapshot_id": "dataset-a",
        "model_snapshot_id": "model-a",
        "config_snapshot_id": "config-b",
        "ranking_snapshot_id": "ranking-a",
    }


def _flexmoe_packed_fixture():
    experts, intermediate, hidden = 4, 32, 16
    generator = torch.Generator(device="cpu").manual_seed(91)
    tensors = {}
    tensor_map = {}
    shapes = {}
    group_sizes = {}
    for key in ("w1", "w2", "w3"):
        shape = (experts, hidden, intermediate) if key == "w2" else (experts, intermediate, hidden)
        quantized_name = f"experts.{key}_int8"
        scale_name = f"experts.{key}_scale"
        tensors[quantized_name] = torch.randint(
            -100,
            101,
            shape,
            dtype=torch.int8,
            generator=generator,
        )
        tensors[scale_name] = torch.full(
            (experts, int(shape[1]), 1),
            0.01,
            dtype=torch.float32,
        )
        tensor_map[f"{key}_int8"] = quantized_name
        tensor_map[f"{key}_scale"] = scale_name
        shapes[key] = list(shape)
        group_sizes[key] = hidden
    manifest = {
        "format": "mirai.compressed_weights.packed_state",
        "schema_version": 1,
        "quant_formats": ["int8"],
        "modules": {
            "blocks.0.experts": {
                "kind": "grouped_experts",
                "quant_format": "int8",
                "num_experts": experts,
                "expert_weight_access": "chunked_dequant",
                "expert_dequant_chunk_size": 4,
                "tensors": tensor_map,
                "shapes": shapes,
                "group_sizes": group_sizes,
            }
        },
        "residual_tensors": {},
    }
    ranking = FlexMoERankingEvidence(
        saliency=torch.stack(
            [
                torch.roll(torch.arange(intermediate, dtype=torch.float64), expert)
                for expert in range(experts)
            ]
        ),
        calibration_batches=2,
    )
    plan = FlexMoEActionPlan(
        action_ratios=(0.25, 0.5, 0.75, 1.0),
        logits=torch.tensor(
            [
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
            ]
        ),
        expert_load=torch.tensor([0.4, 0.3, 0.2, 0.1]),
    )
    return tensors, manifest, ranking, plan


def test_flexmoe_ragged_transform_physically_removes_channel_tails() -> None:
    tensors, manifest, ranking, plan = _flexmoe_packed_fixture()
    source_tensors = {name: value.clone() for name, value in tensors.items()}
    source_manifest = copy.deepcopy(manifest)
    output, converted, report = transform_packed_state_flexmoe_nested(
        tensors,
        manifest,
        ranking_by_module={"blocks.0.experts": ranking},
        actions_by_module={"blocks.0.experts": plan},
        ranking_lineage={
            "dataset_snapshot_id": "dataset-a",
            "model_snapshot_id": "model-a",
            "config_snapshot_id": "ranking-config",
        },
        action_lineage={
            "dataset_snapshot_id": "dataset-a",
            "model_snapshot_id": "model-a",
            "config_snapshot_id": "ranking-config",
            "ranking_snapshot_id": "ranking-a",
        },
        device=torch.device("cpu"),
    )
    assert manifest == source_manifest
    assert all(torch.equal(tensors[name], value) for name, value in source_tensors.items())
    spec = converted["modules"]["blocks.0.experts"]
    provider_spec = spec["physical_weight_provider"]
    assert converted["schema_version"] == 4
    assert provider_spec["name"] == FLEXMOE_NESTED_PROVIDER_NAME
    assert not any(
        name in output
        for name in source_manifest["modules"]["blocks.0.experts"]["tensors"].values()
    )
    provider = FlexMoENestedPhysicalWeightProvider(
        PhysicalWeightProviderContext(
            module_name="blocks.0.experts",
            num_experts=4,
            shapes={key: tuple(value) for key, value in spec["shapes"].items()},
            spec=provider_spec,
            tensors=output,
        )
    )
    assert [provider.retained_intermediate_width(index) for index in range(4)] == [
        8,
        16,
        24,
        24,
    ]
    assert provider.materialize_expert(
        "w1",
        0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).shape == (8, 16)
    assert provider.materialize_expert(
        "w2",
        1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ).shape == (16, 16)
    assert report["packed_expert_bytes"] < report["source_expert_bytes"]
    assert 0.0 < report["byte_ratio"] < 1.0
    assert report["global_prune_budget"] == pytest.approx(0.4375)
    assert converted["flexmoe_nested_transform"]["global_prune_budget"] == (
        pytest.approx(0.4375)
    )


def test_flexmoe_transform_rejects_cross_config_action_lineage() -> None:
    tensors, manifest, ranking, plan = _flexmoe_packed_fixture()
    with pytest.raises(ValueError, match="lineage disagree"):
        transform_packed_state_flexmoe_nested(
            tensors,
            manifest,
            ranking_by_module={"blocks.0.experts": ranking},
            actions_by_module={"blocks.0.experts": plan},
            ranking_lineage={
                "dataset_snapshot_id": "dataset-a",
                "model_snapshot_id": "model-a",
                "config_snapshot_id": "ranking-config",
            },
            action_lineage={
                "dataset_snapshot_id": "dataset-a",
                "model_snapshot_id": "model-a",
                "config_snapshot_id": "different-config",
                "ranking_snapshot_id": "ranking-a",
            },
            device=torch.device("cpu"),
        )


def test_flexmoe_transform_preserves_source_tensor_still_referenced_elsewhere() -> None:
    tensors, manifest, ranking, plan = _flexmoe_packed_fixture()
    shared_name = next(iter(manifest["modules"]["blocks.0.experts"]["tensors"].values()))
    manifest["residual_tensors"] = {"shared_reference": shared_name}
    output, _converted, _report = transform_packed_state_flexmoe_nested(
        tensors,
        manifest,
        ranking_by_module={"blocks.0.experts": ranking},
        actions_by_module={"blocks.0.experts": plan},
        ranking_lineage={
            "dataset_snapshot_id": "dataset-a",
            "model_snapshot_id": "model-a",
            "config_snapshot_id": "ranking-config",
        },
        action_lineage={
            "dataset_snapshot_id": "dataset-a",
            "model_snapshot_id": "model-a",
            "config_snapshot_id": "ranking-config",
            "ranking_snapshot_id": "ranking-a",
        },
        device=torch.device("cpu"),
    )
    assert shared_name in output


def test_flexmoe_ragged_provider_matches_explicit_routed_reference_and_gradient() -> None:
    tensors, manifest, ranking, plan = _flexmoe_packed_fixture()
    output, converted, _report = transform_packed_state_flexmoe_nested(
        tensors,
        manifest,
        ranking_by_module={"blocks.0.experts": ranking},
        actions_by_module={"blocks.0.experts": plan},
        ranking_lineage={
            "dataset_snapshot_id": "dataset-a",
            "model_snapshot_id": "model-a",
            "config_snapshot_id": "ranking-config",
        },
        action_lineage={
            "dataset_snapshot_id": "dataset-a",
            "model_snapshot_id": "model-a",
            "config_snapshot_id": "ranking-config",
            "ranking_snapshot_id": "ranking-a",
        },
        device=torch.device("cpu"),
    )
    spec = converted["modules"]["blocks.0.experts"]
    provider = FlexMoENestedPhysicalWeightProvider(
        PhysicalWeightProviderContext(
            module_name="blocks.0.experts",
            num_experts=4,
            shapes={key: tuple(value) for key, value in spec["shapes"].items()},
            spec=spec["physical_weight_provider"],
            tensors=output,
        )
    )
    module = CompressedGroupedExperts.from_empty(
        num_experts=4,
        quant_format="int8",
        expert_weight_access="chunked_dequant",
        expert_dequant_chunk_size=4,
    )
    module.bind_physical_weight_provider(provider)
    assert module._expert_chunk_groups(4) == ((0,), (1,), (2, 3))
    indices = torch.tensor([[0, 1], [2, 3], [1, 2]])
    scores = torch.tensor([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
    actual_input = torch.randn(3, 16, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    actual = module.run_direct_routed(actual_input, scores, indices)
    reference = torch.zeros_like(reference_input)
    for token_index in range(indices.shape[0]):
        for route_index in range(indices.shape[1]):
            expert = int(indices[token_index, route_index].item())
            token = reference_input[token_index]
            w1 = provider.materialize_expert("w1", expert, dtype=token.dtype, device=token.device)
            w2 = provider.materialize_expert("w2", expert, dtype=token.dtype, device=token.device)
            w3 = provider.materialize_expert("w3", expert, dtype=token.dtype, device=token.device)
            hidden = torch.nn.functional.silu(token @ w1.transpose(0, 1))
            hidden = hidden * (token @ w3.transpose(0, 1))
            routed = hidden @ w2.transpose(0, 1)
            reference[token_index] = (
                reference[token_index] + routed * scores[token_index, route_index]
            )
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=1e-5)
    actual.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        # Grouped execution and the route-by-route oracle use different FP32
        # accumulation orders in both the expert GEMMs and route reduction.
        rtol=1e-4,
        atol=3e-5,
    )


def test_flexmoe_ragged_provider_matches_expert_choice_reference() -> None:
    tensors, manifest, ranking, plan = _flexmoe_packed_fixture()
    output, converted, _report = transform_packed_state_flexmoe_nested(
        tensors,
        manifest,
        ranking_by_module={"blocks.0.experts": ranking},
        actions_by_module={"blocks.0.experts": plan},
        ranking_lineage={
            "dataset_snapshot_id": "dataset-a",
            "model_snapshot_id": "model-a",
            "config_snapshot_id": "ranking-config",
        },
        action_lineage={
            "dataset_snapshot_id": "dataset-a",
            "model_snapshot_id": "model-a",
            "config_snapshot_id": "ranking-config",
            "ranking_snapshot_id": "ranking-a",
        },
        device=torch.device("cpu"),
    )
    spec = converted["modules"]["blocks.0.experts"]
    provider = FlexMoENestedPhysicalWeightProvider(
        PhysicalWeightProviderContext(
            module_name="blocks.0.experts",
            num_experts=4,
            shapes={key: tuple(value) for key, value in spec["shapes"].items()},
            spec=spec["physical_weight_provider"],
            tensors=output,
        )
    )
    module = CompressedGroupedExperts.from_empty(
        num_experts=4,
        quant_format="int8",
        expert_weight_access="chunked_dequant",
        expert_dequant_chunk_size=4,
    )
    module.bind_physical_weight_provider(provider)
    token_indices = torch.tensor([[[0], [1], [2], [0]]])
    route_scores = torch.tensor([[[0.8], [0.7], [0.6], [0.5]]])
    actual_input = torch.randn(3, 16, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    actual = module.run_expert_choice_routed(
        actual_input,
        route_scores,
        token_indices,
        tokens_per_sample=3,
    )
    reference = torch.zeros_like(reference_input)
    for expert in range(4):
        token_index = int(token_indices[0, expert, 0].item())
        token = reference_input[token_index]
        w1 = provider.materialize_expert(
            "w1", expert, dtype=token.dtype, device=token.device
        )
        w2 = provider.materialize_expert(
            "w2", expert, dtype=token.dtype, device=token.device
        )
        w3 = provider.materialize_expert(
            "w3", expert, dtype=token.dtype, device=token.device
        )
        hidden = torch.nn.functional.silu(token @ w1.transpose(0, 1))
        hidden = hidden * (token @ w3.transpose(0, 1))
        reference[token_index] = reference[token_index] + (
            hidden @ w2.transpose(0, 1) * route_scores[0, expert, 0]
        )
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    actual.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        rtol=1e-5,
        atol=1e-5,
    )


def test_flexmoe_ragged_provider_reads_only_requested_payload_ranges() -> None:
    tensors, manifest, ranking, plan = _flexmoe_packed_fixture()
    output, converted, _report = transform_packed_state_flexmoe_nested(
        tensors,
        manifest,
        ranking_by_module={"blocks.0.experts": ranking},
        actions_by_module={"blocks.0.experts": plan},
        ranking_lineage={
            "dataset_snapshot_id": "dataset-a",
            "model_snapshot_id": "model-a",
            "config_snapshot_id": "ranking-config",
        },
        action_lineage={
            "dataset_snapshot_id": "dataset-a",
            "model_snapshot_id": "model-a",
            "config_snapshot_id": "ranking-config",
            "ranking_snapshot_id": "ranking-a",
        },
        device=torch.device("cpu"),
    )

    class RangeOnlyMapping(dict):
        def __init__(self, values):
            super().__init__(values)
            self.ranges = []

        def __getitem__(self, name):
            if str(name).endswith((".codes", ".scales")):
                raise AssertionError("full ragged payload materialization is forbidden")
            return super().__getitem__(name)

        def tensor_shape_dtype(self, name):
            value = dict.__getitem__(self, name)
            return tuple(value.shape), str(value.dtype)

        def get_range(self, name, start, end):
            self.ranges.append((str(name), int(start), int(end)))
            return dict.__getitem__(self, name)[int(start) : int(end)]

    lazy = RangeOnlyMapping(output)
    spec = converted["modules"]["blocks.0.experts"]
    provider = FlexMoENestedPhysicalWeightProvider(
        PhysicalWeightProviderContext(
            module_name="blocks.0.experts",
            num_experts=4,
            shapes={key: tuple(value) for key, value in spec["shapes"].items()},
            spec=spec["physical_weight_provider"],
            tensors=lazy,
        )
    )
    value = provider.materialize_expert(
        "w1",
        0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert value.shape == (8, 16)
    assert len(lazy.ranges) == 2
    assert lazy.ranges[0][0].endswith(".w1.codes")
    assert lazy.ranges[1][0].endswith(".w1.scales")
