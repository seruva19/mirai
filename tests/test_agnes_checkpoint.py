from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mirai.core.models.agnes_video.checkpoints import (
    AGNES_CHECKPOINT_NAME,
    AGNES_SOURCE_REVISION,
    assemble_checkpoint_parts,
    checkpoint_parts,
    convert_agnes_checkpoint,
    load_agnes_pt_checkpoint,
    load_agnes_safetensors,
    iter_agnes_safetensors,
    validate_checkpoint_parts,
)


def _state() -> dict[str, torch.Tensor]:
    return {
        "module.blocks.0.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "module.router.bias": torch.tensor([1.5, -2.0]),
    }


def _split_checkpoint(root: Path, payload: object, *, count: int = 3) -> tuple[Path, ...]:
    archive = root / AGNES_CHECKPOINT_NAME
    torch.save(payload, archive)
    data = archive.read_bytes()
    archive.unlink()
    cut = max(1, len(data) // count)
    parts = []
    for number, start in enumerate(range(0, len(data), cut)):
        path = root / f"{AGNES_CHECKPOINT_NAME}.part-{number:03d}"
        path.write_bytes(data[start : start + cut])
        parts.append(path)
    return tuple(parts)


def test_split_checkpoint_loads_nested_model_and_normalizes_module_prefix(tmp_path: Path) -> None:
    parts = _split_checkpoint(tmp_path, {"model": _state()})

    assert checkpoint_parts(tmp_path) == parts
    loaded = load_agnes_pt_checkpoint(tmp_path, verify_released_parts=False)

    assert set(loaded) == {"blocks.0.weight", "router.bias"}
    torch.testing.assert_close(loaded["blocks.0.weight"], _state()["module.blocks.0.weight"])


def test_direct_pt_loads_state_dict_with_restricted_unpickler(tmp_path: Path) -> None:
    source = tmp_path / "weights.pt"
    torch.save({"state_dict": _state()}, source)

    loaded = load_agnes_pt_checkpoint(source)

    torch.testing.assert_close(loaded["router.bias"], _state()["module.router.bias"])


def test_missing_and_out_of_order_parts_fail_before_assembly(tmp_path: Path) -> None:
    part0 = tmp_path / f"{AGNES_CHECKPOINT_NAME}.part-000"
    part2 = tmp_path / f"{AGNES_CHECKPOINT_NAME}.part-002"
    part0.write_bytes(b"zero")
    part2.write_bytes(b"two")

    with pytest.raises(ValueError, match="missing: 001"):
        checkpoint_parts(tmp_path)
    with pytest.raises(ValueError, match="out of order"):
        validate_checkpoint_parts((part2, part0))


def test_part_size_and_hash_validation_detect_corruption(tmp_path: Path) -> None:
    part = tmp_path / f"{AGNES_CHECKPOINT_NAME}.part-000"
    part.write_bytes(b"payload")

    with pytest.raises(ValueError, match="has size 7, expected 8"):
        validate_checkpoint_parts((part,), expected_sizes=(8,))
    with pytest.raises(ValueError, match="SHA-256"):
        validate_checkpoint_parts((part,), expected_sha256=("0" * 64,))


def test_assembly_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    parts = _split_checkpoint(tmp_path, {"model": _state()}, count=2)
    output = tmp_path / "assembled.pt"

    assert assemble_checkpoint_parts(parts, output) == output
    assert load_agnes_pt_checkpoint(output)["router.bias"].shape == (2,)
    with pytest.raises(FileExistsError):
        assemble_checkpoint_parts(parts, output)


def test_mixed_module_namespace_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "mixed.pt"
    torch.save(
        {"model": {"module.layer.weight": torch.ones(1), "layer.bias": torch.zeros(1)}},
        source,
    )

    with pytest.raises(ValueError, match="mixes 'module.'"):
        load_agnes_pt_checkpoint(source)


def test_conversion_roundtrip_shards_and_records_lineage(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    torch.save({"model": _state()}, source)
    output = tmp_path / "converted"

    index_path = convert_agnes_checkpoint(source, output, max_shard_size=16)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    restored = load_agnes_safetensors(output)

    assert index["metadata"]["source_revision"] == AGNES_SOURCE_REVISION
    assert index["metadata"]["format_version"] == 1
    assert len(set(index["weight_map"].values())) == 2
    assert set(restored) == {"blocks.0.weight", "router.bias"}
    torch.testing.assert_close(restored["blocks.0.weight"], _state()["module.blocks.0.weight"])
    with pytest.raises(FileExistsError):
        convert_agnes_checkpoint(source, output, max_shard_size=16)


def test_loader_rejects_modified_lineage(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    torch.save({"model": _state()}, source)
    index_path = convert_agnes_checkpoint(source, tmp_path / "converted")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["metadata"]["source_revision"] = "different"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="lineage"):
        load_agnes_safetensors(index_path)


def test_safetensors_index_rejects_shard_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    torch.save({"model": _state()}, source)
    index_path = convert_agnes_checkpoint(source, tmp_path / "converted")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    first = next(iter(index["weight_map"]))
    index["weight_map"][first] = "../outside.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe shard path"):
        list(iter_agnes_safetensors(index_path))


def test_safetensors_shard_must_own_exact_index_keys(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    source = tmp_path / "source.pt"
    torch.save({"model": _state()}, source)
    index_path = convert_agnes_checkpoint(
        source, tmp_path / "converted", max_shard_size=16
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    save_file({"wrong.key": torch.ones(1)}, str(index_path.parent / shard_names[0]))

    with pytest.raises(ValueError, match="does not match its weight_map"):
        list(iter_agnes_safetensors(index_path))


def test_safetensors_iterator_yields_complete_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    torch.save({"model": _state()}, source)
    index_path = convert_agnes_checkpoint(
        source, tmp_path / "converted", max_shard_size=16
    )

    restored = dict(iter_agnes_safetensors(index_path))

    assert set(restored) == {"blocks.0.weight", "router.bias"}
