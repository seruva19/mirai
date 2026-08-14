from __future__ import annotations

import json
from pathlib import Path

import pytest

from mirai.core.moe.runtime.routed_gemm_autotune import (
    RoutedGemmKernelConfig,
    RoutedGemmShapeKey,
)
from mirai.core.persistence.routed_gemm_autotune import (
    RoutedGemmEnvironmentFingerprint,
    RoutedGemmTuningArtifact,
    RoutedGemmTuningEntry,
    load_routed_gemm_tuning_cache,
    migrate_routed_gemm_tuning_payload,
    routed_gemm_cache_lock,
    save_routed_gemm_tuning_cache,
)


def _environment(**updates: object) -> RoutedGemmEnvironmentFingerprint:
    values: dict[str, object] = {
        "backend": "cuda",
        "compute_capability": (9, 0),
        "sm_count": 120,
        "shared_memory_bytes": 232448,
        "cuda_runtime_class": "12.8",
        "cuda_driver_class": "12.8",
        "torch_version": "2.7.1",
        "triton_version": "3.3.1",
        "kernel_abi_fingerprint": "sha256:test",
        "compiler_target": "sm90",
    }
    values.update(updates)
    return RoutedGemmEnvironmentFingerprint(**values)  # type: ignore[arg-type]


def _key() -> RoutedGemmShapeKey:
    return RoutedGemmShapeKey(
        backend="cuda", compute_capability=(9, 0), sm_count=120,
        shared_memory_bytes=232448, implementation="indexed_sm80", role="forward",
        fusion="gather", input_dtype="bfloat16", output_dtype="bfloat16",
        accumulation_dtype="float32", k_size=64, n_size=128, group_count=4,
        routed_rows=12, top_k=2, nonempty_groups=3, max_group_rows=6,
        mean_group_rows_milli=3000, coefficient_of_variation_milli=816,
        routing_histogram=(1, 0, 2, 1, 0), stride_class="contiguous",
        alignment_bytes=16, segmented=False,
    )


def _artifact() -> RoutedGemmTuningArtifact:
    entry = RoutedGemmTuningEntry(
        shape_key=_key(), implementation="indexed_sm80",
        config=RoutedGemmKernelConfig(64, 32, 4, 2), measured_us=12.5,
        statistic="median", samples=20, parity_tolerance=1e-3,
        created_at="2026-08-13T12:00:00+00:00",
    )
    return RoutedGemmTuningArtifact(_environment(), (entry,))


def test_atomic_round_trip_and_exact_lookup(tmp_path: Path) -> None:
    path = tmp_path / "tuning.json"
    save_routed_gemm_tuning_cache(path, _artifact())
    loaded = load_routed_gemm_tuning_cache(path, _environment())
    assert loaded.status == "hit"
    assert loaded.artifact is not None
    assert loaded.artifact.entry_for(_key()) == _artifact().entries[0]
    assert not list(tmp_path.glob("*.tmp"))


def test_environment_mismatch_is_a_miss_without_partial_reuse(tmp_path: Path) -> None:
    path = tmp_path / "tuning.json"
    save_routed_gemm_tuning_cache(path, _artifact())
    loaded = load_routed_gemm_tuning_cache(path, _environment(triton_version="3.4.0"))
    assert loaded.status == "incompatible"
    assert loaded.artifact is None
    assert path.exists()


@pytest.mark.parametrize("field", ["schema_version", "kernel_contract_version"])
def test_contract_mismatch_is_incompatible_not_corrupt(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "tuning.json"
    payload = _artifact().to_dict()
    payload[field] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_routed_gemm_tuning_cache(path, _environment())
    assert loaded.status == "incompatible"
    assert loaded.artifact is None
    assert loaded.quarantined_path is None
    assert path.exists()


@pytest.mark.parametrize("content", ["{", "[]", "{\"schema_version\": 1}"])
def test_malformed_artifact_is_quarantined(tmp_path: Path, content: str) -> None:
    path = tmp_path / "tuning.json"
    path.write_text(content, encoding="utf-8")
    loaded = load_routed_gemm_tuning_cache(path, _environment())
    assert loaded.status == "corrupt"
    assert loaded.artifact is None
    assert loaded.quarantined_path is not None
    assert loaded.quarantined_path.exists()
    assert not path.exists()


def test_unknown_fields_and_entry_key_mismatch_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tuning.json"
    payload = _artifact().to_dict()
    payload["unknown"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_routed_gemm_tuning_cache(path, _environment()).status == "corrupt"

    payload = _artifact().to_dict()
    entries = payload["entries"]
    assert isinstance(entries, dict)
    entry = entries.pop(next(iter(entries)))
    entries["wrong"] = entry
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_routed_gemm_tuning_cache(path, _environment()).status == "corrupt"


def test_lock_contention_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "tuning.json"
    with routed_gemm_cache_lock(path):
        with pytest.raises(RuntimeError, match="already held"):
            with routed_gemm_cache_lock(path):
                pass


def test_current_schema_migration_is_validation_only() -> None:
    payload = _artifact().to_dict()
    assert migrate_routed_gemm_tuning_payload(payload) == payload
    payload["schema_version"] = 0
    with pytest.raises(ValueError, match="no routed GEMM tuning cache migration"):
        migrate_routed_gemm_tuning_payload(payload)
