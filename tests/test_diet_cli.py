import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from mirai.config.schema import TrainingConfig
from mirai.core.lineage import sha256_file
from mirai.core.models.compressed_weights.packed.packed_state import (
    COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY,
    read_compressed_weights_packed_state_manifest,
)
from mirai.core.moe.calibration.diet import (
    DietCalibrationEvidence, DietGroupTopology, DietLineage, save_diet_evidence,
)
from scripts.tools.prune_experts import prune_packed_base


def test_diet_cli_roundtrip_preserves_router_and_rejects_wrong_source(tmp_path):
    source = tmp_path / "base.safetensors"
    evidence_path = tmp_path / "evidence.safetensors"
    output = tmp_path / "pruned.safetensors"
    router = torch.arange(12).reshape(4, 3)
    tensors = {"w": torch.arange(16).reshape(4, 2, 2), "router": router}
    manifest = {
        "format": "mirai.compressed_weights.packed_state", "schema_version": 1,
        "modules": {"blocks.0.experts": {
            "kind": "grouped_experts", "num_experts": 4,
            "tensors": {"w1_int8": "w"}, "shapes": {"w1": [4, 2, 2]},
        }}, "residual_tensors": {"blocks.0.router.weight": "router"},
    }
    save_file(tensors, str(source), metadata={
        COMPRESSED_WEIGHT_PACKED_MANIFEST_METADATA_KEY: json.dumps(manifest),
    })
    fingerprint = sha256_file(source)
    topology = DietGroupTopology(4, 1, 2, 1, 1, True, 1.0)
    evidence = DietCalibrationEvidence(torch.eye(4), 2, topology, "paired_cfg", 3.0)
    save_diet_evidence(evidence_path, {"blocks.0.experts": evidence}, lineage=
        DietLineage("dataset", "model", "config", "requests", fingerprint, "capture"))
    config = TrainingConfig()
    config.model.params.expert_pruning = "prune"
    config.model.params.expert_pruning_criterion = "diet"
    config.model.params.experts_per_token = 1
    report = prune_packed_base(config, packed_state=source, calibration_file=evidence_path,
                              output=output, keep_fraction=0.5)
    result = load_file(str(output))
    assert report["criterion"] == "diet"
    assert result["w"].shape[0] == 2
    assert torch.equal(result["router"], router)
    assert sha256_file(source) == fingerprint
    plan = read_compressed_weights_packed_state_manifest(output)["modules"]["blocks.0.experts"]["expert_deletion"]
    assert plan["logical_num_experts"] == 4
    assert plan["n_group"] == 2
    save_diet_evidence(evidence_path, {"blocks.0.experts": evidence}, lineage=
        DietLineage("dataset", "model", "config", "requests", "0" * 64, "capture"))
    with pytest.raises(ValueError, match="different packed source"):
        prune_packed_base(config, packed_state=source, calibration_file=evidence_path,
                          output=tmp_path / "bad.safetensors", keep_fraction=0.5)
    config.model.params.expert_pruning = "off"
    with pytest.raises(ValueError, match="opt-in"):
        prune_packed_base(config, packed_state=source, calibration_file=evidence_path,
                          output=tmp_path / "disabled.safetensors", keep_fraction=0.5)


def test_diet_packed_restore_and_reexport_preserve_deletion():
    import dataclasses
    from mirai.core.models.compressed_weights.execution.experts import CompressedGroupedExperts
    from mirai.core.models.compressed_weights.packed.packed_state import (
        export_compressed_weights_packed_state, load_compressed_weights_packed_state,
        prepare_compressed_weights_modules_from_manifest,
    )
    from mirai.core.moe.calibration.pruning import compact_packed_state_for_diet

    def make_root():
        root = torch.nn.Module()
        block = torch.nn.Module()
        block.router = torch.nn.Linear(4, 8, bias=False)
        block.experts = CompressedGroupedExperts.from_empty(
            num_experts=8, group_sizes=4, expert_weight_access="active_dequant",
            quant_format="int8",
        )
        for key, shape in {"w1": (8, 6, 4), "w2": (8, 4, 6), "w3": (8, 6, 4)}.items():
            block.experts.load_dense_weight(key, torch.randn(shape))
        root.blocks = torch.nn.ModuleList([block])
        return root

    source = make_root()
    tensors, manifest = export_compressed_weights_packed_state(source)
    keep = (0, 1, 4, 5)
    topology = DietGroupTopology(8, 2, 2, 1, 2, True, 1.0)
    compact, plan = compact_packed_state_for_diet(
        tensors, manifest, {"blocks.0.experts": keep},
        topology_by_module={"blocks.0.experts": dataclasses.asdict(topology)},
    )
    restored = make_root()
    prepare_compressed_weights_modules_from_manifest(restored, plan)
    load_compressed_weights_packed_state(restored, compact, plan)
    assert restored.blocks[0].experts.num_experts == 4
    assert restored.blocks[0].experts.expert_deletion_plan().kept_logical_ids == keep
    assert torch.equal(restored.blocks[0].router.weight, source.blocks[0].router.weight)
    exported, reexported = export_compressed_weights_packed_state(restored)
    assert reexported["modules"]["blocks.0.experts"]["expert_deletion"] == plan["modules"]["blocks.0.experts"]["expert_deletion"]
    for key, value in compact.items():
        assert torch.equal(exported[key], value)


def test_native_diet_capture_matches_observer_contract():
    from types import SimpleNamespace
    from mirai.core.training.calibration.diet import _DietObserver
    from mirai.vendors.lingbot_video.transformer_lingbot_video import LingBotVideoSparseMoeBlock

    block = LingBotVideoSparseMoeBlock(
        hidden_size=4, intermediate_size=8, num_experts=8, top_k=2,
        moe_intermediate_size=6, score_func="sigmoid", norm_topk_prob=True,
        n_group=2, topk_group=1, routed_scaling_factor=1.0, n_shared_experts=1,
    ).eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.uniform_(-0.1, 0.1)
    target = SimpleNamespace(name="experts", num_experts=8, top_k=2, n_group=2,
                             topk_group=1, group_score_topk=2,
                             norm_topk_prob=True, route_scale=1.0)
    observer = _DietObserver(target, sample_budget=6, seed=3)
    observer.guidance_scale = 3.0
    block.set_diet_capture_observer(observer)
    gate = torch.ones(6, 4)
    gate[:, 0] = 0
    block._diet_parent_postprocess = {
        "rms_weight": torch.arange(1, 5).float(), "epsilon": 1e-5,
        "residual_gate": gate,
    }
    with torch.no_grad():
        block(torch.randn(2, 3, 4))
        assert observer.pending_capture.postprocess.shared_output.shape == (6, 4)
        assert observer.pending_capture.postprocess.rms_epsilon == 1e-5
        block(torch.randn(2, 3, 4))
    evidence = observer.evidence()
    assert evidence.sample_count == 6
    assert torch.equal(evidence.signatures[:, 0], torch.zeros(8))
    assert bool(torch.isfinite(evidence.signatures).all())
    assert bool(evidence.signatures[:, 1:].abs().sum() > 0)
