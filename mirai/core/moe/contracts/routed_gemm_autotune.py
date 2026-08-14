from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import pytest

from mirai.core.moe.runtime.routed_gemm_autotune import (
    RoutedGemmAutotuneMap,
    RoutedGemmAutotuner,
    RoutedGemmBenchmarkResult,
    RoutedGemmKernelConfig,
    RoutedGemmShapeKey,
    routing_distribution_statistics,
    select_tuning_winner,
)
from mirai.core.persistence.routed_gemm_autotune import (
    RoutedGemmEnvironmentFingerprint,
)


def _key(**updates: object) -> RoutedGemmShapeKey:
    values: dict[str, object] = {
        "backend": "cuda",
        "compute_capability": (9, 0),
        "sm_count": 120,
        "shared_memory_bytes": 232448,
        "implementation": "indexed_sm80",
        "role": "forward",
        "fusion": "gather",
        "input_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "k_size": 64,
        "n_size": 128,
        "group_count": 4,
        "routed_rows": 12,
        "top_k": 2,
        "nonempty_groups": 3,
        "max_group_rows": 6,
        "mean_group_rows_milli": 3000,
        "coefficient_of_variation_milli": 816,
        "routing_histogram": (1, 0, 2, 1, 0),
        "stride_class": "contiguous",
        "alignment_bytes": 16,
        "segmented": False,
    }
    values.update(updates)
    return RoutedGemmShapeKey(**values)  # type: ignore[arg-type]


def test_shape_key_round_trip_is_canonical_and_semantic() -> None:
    key = _key()
    assert RoutedGemmShapeKey.from_dict(key.to_dict()) == key
    assert "provider" not in key.canonical()
    assert _key(role="dx").canonical() != key.canonical()
    assert _key(routing_histogram=(0, 1, 2, 1, 0)).canonical() != key.canonical()


def test_distribution_statistics_use_fixed_buckets() -> None:
    assert routing_distribution_statistics([0, 2, 4, 6]) == {
        "nonempty_groups": 3,
        "max_group_rows": 6,
        "mean_group_rows_milli": 3000,
        "coefficient_of_variation_milli": 745,
        "routing_histogram": (1, 0, 1, 2, 0),
    }


def test_winner_selection_prunes_resets_and_breaks_ties_deterministically() -> None:
    first = RoutedGemmKernelConfig(64, 32, 4, 2)
    second = RoutedGemmKernelConfig(128, 32, 4, 2)
    rejected = RoutedGemmKernelConfig(256, 32, 4, 2)
    resets: list[None] = []
    winner = select_tuning_winner(
        _key(),
        [rejected, second, first],
        lambda config: RoutedGemmBenchmarkResult(config, 10.0, 5),
        predicates=(lambda _key_value, config: config.block_n <= 128,),
        reset_output=lambda: resets.append(None),
    )
    assert winner.config == first
    assert len(resets) == 2


def test_off_mode_has_no_state_and_warmup_only_miss_is_explicit() -> None:
    key = _key()
    disabled = RoutedGemmAutotuneMap("off")
    assert disabled.resolve(key, lambda: "choice") == "choice"
    assert disabled.get(key) is None
    assert disabled.snapshot() == {}
    with pytest.raises(LookupError, match="warmup-only"):
        RoutedGemmAutotuneMap("warmup_only").resolve(key, lambda: "unused")


def test_online_resolution_is_single_flight() -> None:
    cache = RoutedGemmAutotuneMap("online")
    gate = Lock()
    calls = 0

    def factory() -> str:
        nonlocal calls
        with gate:
            calls += 1
        time.sleep(0.02)
        return "winner"

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: cache.resolve(_key(), factory), range(8)))
    assert results == ["winner"] * 8
    assert calls == 1


def _environment() -> RoutedGemmEnvironmentFingerprint:
    return RoutedGemmEnvironmentFingerprint(
        backend="cuda", compute_capability=(9, 0), sm_count=120,
        shared_memory_bytes=232448, cuda_runtime_class="12.8",
        cuda_driver_class="12.8", torch_version="2.7.1", triton_version="3.3.1",
        kernel_abi_fingerprint="sha256:test", compiler_target="sm90",
    )


def test_runtime_facade_off_isolated_and_online_round_trips(tmp_path) -> None:
    conservative = RoutedGemmKernelConfig(32, 32, 2, 1)
    disabled = RoutedGemmAutotuner(mode="off", conservative_config=conservative)
    assert disabled.resolve_config(_key()) == conservative

    path = tmp_path / "routed.json"
    tuner = RoutedGemmAutotuner(
        mode="online", environment_fingerprint=_environment(), cache_path=path
    )
    verified: list[RoutedGemmKernelConfig] = []
    selected = tuner.resolve_config(
        _key(),
        benchmark=lambda config: RoutedGemmBenchmarkResult(
            config, float(config.block_n), 3
        ),
        verify=lambda result: verified.append(result.config),
    )
    assert selected.block_n == 64
    assert verified == [selected]
    assert path.exists()
    warm = RoutedGemmAutotuner(
        mode="warmup_only", environment_fingerprint=_environment(), cache_path=path
    )
    assert warm.resolve_config(_key()) == selected


def test_warmup_only_writer_is_explicit(tmp_path) -> None:
    path = tmp_path / "routed.json"
    tuner = RoutedGemmAutotuner(
        mode="warmup_only", environment_fingerprint=_environment(), cache_path=path
    )
    with pytest.raises(LookupError, match="warmup-only"):
        tuner.resolve_config(_key())
    selected = tuner.resolve_config(
        _key(),
        warmup_write=True,
        benchmark=lambda config: RoutedGemmBenchmarkResult(config, 1.0, 2),
        verify=lambda _result: None,
    )
    assert selected == min(
        {
            RoutedGemmKernelConfig(64, 32, 4, 2, False),
            RoutedGemmKernelConfig(64, 32, 4, 2, True),
        }
    )
    assert path.exists()
