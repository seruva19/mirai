# SPDX-License-Identifier: Apache-2.0
"""Typed shape selection for routed grouped-GEMM configurations.

This module owns semantic tuning identity and deterministic winner selection.
It deliberately has no Triton, CUDA, filesystem, or model-provider dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from threading import Condition, RLock
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence, TypeVar

if TYPE_CHECKING:
    from mirai.core.persistence.routed_gemm_autotune import (
        RoutedGemmEnvironmentFingerprint,
    )


KERNEL_CONTRACT_VERSION = 1
BENCHMARK_PROTOCOL_VERSION = 1


class RoutedGemmTuningMode(str, Enum):
    OFF = "off"
    ONLINE = "online"
    WARMUP_ONLY = "warmup_only"


def normalize_routed_gemm_tuning_mode(value: str | None) -> str:
    text = str(value or "off").strip().lower()
    aliases = {"disabled": "off", "none": "off", "warmup": "warmup_only"}
    text = aliases.get(text, text)
    try:
        return RoutedGemmTuningMode(text).value
    except ValueError as exc:
        raise ValueError(
            "routed GEMM tuning mode must be one of: off, online, warmup_only"
        ) from exc


@dataclass(frozen=True, order=True)
class RoutedGemmKernelConfig:
    """Auditable launch configuration for one implementation."""

    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    persistent: bool = False

    def __post_init__(self) -> None:
        if self.block_n <= 0 or self.block_k <= 0:
            raise ValueError("routed GEMM block sizes must be positive")
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError("routed GEMM num_warps must be one of: 1, 2, 4, 8")
        if self.num_stages <= 0:
            raise ValueError("routed GEMM num_stages must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RoutedGemmKernelConfig":
        _require_exact_fields(
            value,
            {"block_n", "block_k", "num_warps", "num_stages", "persistent"},
            "kernel config",
        )
        return cls(
            block_n=_strict_int(value["block_n"], "block_n"),
            block_k=_strict_int(value["block_k"], "block_k"),
            num_warps=_strict_int(value["num_warps"], "num_warps"),
            num_stages=_strict_int(value["num_stages"], "num_stages"),
            persistent=_strict_bool(value["persistent"], "persistent"),
        )


@dataclass(frozen=True)
class RoutedGemmShapeKey:
    """Canonical semantic identity for one routed grouped-GEMM shape."""

    backend: str
    compute_capability: tuple[int, int]
    sm_count: int
    shared_memory_bytes: int
    implementation: str
    role: str
    fusion: str
    input_dtype: str
    output_dtype: str
    accumulation_dtype: str
    k_size: int
    n_size: int
    group_count: int
    routed_rows: int
    top_k: int
    nonempty_groups: int
    max_group_rows: int
    mean_group_rows_milli: int
    coefficient_of_variation_milli: int
    routing_histogram: tuple[int, ...]
    stride_class: str
    alignment_bytes: int
    segmented: bool

    def __post_init__(self) -> None:
        text_fields = (
            "backend",
            "implementation",
            "role",
            "fusion",
            "input_dtype",
            "output_dtype",
            "accumulation_dtype",
            "stride_class",
        )
        for name in text_fields:
            value = str(getattr(self, name)).strip().lower()
            if not value:
                raise ValueError(f"routed GEMM shape key {name} must be non-empty")
            object.__setattr__(self, name, value)
        capability = tuple(self.compute_capability)
        if len(capability) != 2 or any(type(part) is not int or part < 0 for part in capability):
            raise ValueError("compute_capability must contain two non-negative integers")
        object.__setattr__(self, "compute_capability", capability)
        object.__setattr__(self, "routing_histogram", tuple(self.routing_histogram))
        nonnegative = (
            "sm_count",
            "shared_memory_bytes",
            "k_size",
            "n_size",
            "group_count",
            "routed_rows",
            "nonempty_groups",
            "max_group_rows",
            "mean_group_rows_milli",
            "coefficient_of_variation_milli",
            "alignment_bytes",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in nonnegative):
            raise ValueError("routed GEMM shape dimensions and statistics must be non-negative integers")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("routed GEMM top_k must be positive")
        if self.nonempty_groups > self.group_count:
            raise ValueError("nonempty_groups cannot exceed group_count")
        if any(type(value) is not int or value < 0 for value in self.routing_histogram):
            raise ValueError("routing_histogram must contain non-negative integers")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["compute_capability"] = list(self.compute_capability)
        value["routing_histogram"] = list(self.routing_histogram)
        return value

    def canonical(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RoutedGemmShapeKey":
        fields = set(cls.__dataclass_fields__)
        _require_exact_fields(value, fields, "shape key")
        kwargs = dict(value)
        capability = kwargs["compute_capability"]
        histogram = kwargs["routing_histogram"]
        if not isinstance(capability, list) or not isinstance(histogram, list):
            raise ValueError("shape key capability and histogram must be JSON arrays")
        kwargs["compute_capability"] = tuple(
            _strict_int(item, "compute_capability") for item in capability
        )
        kwargs["routing_histogram"] = tuple(
            _strict_int(item, "routing_histogram") for item in histogram
        )
        for name in (
            "sm_count", "shared_memory_bytes", "k_size", "n_size", "group_count",
            "routed_rows", "top_k", "nonempty_groups", "max_group_rows",
            "mean_group_rows_milli", "coefficient_of_variation_milli", "alignment_bytes",
        ):
            kwargs[name] = _strict_int(kwargs[name], name)
        kwargs["segmented"] = _strict_bool(kwargs["segmented"], "segmented")
        return cls(**kwargs)  # type: ignore[arg-type]


def routing_distribution_statistics(counts: Iterable[int]) -> dict[str, object]:
    """Return stable integer statistics without retaining raw routing vectors."""

    values = tuple(int(value) for value in counts)
    if any(value < 0 for value in values):
        raise ValueError("routing counts must be non-negative")
    nonempty = [value for value in values if value]
    total = sum(values)
    mean = total / len(values) if values else 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
    cv = math.sqrt(variance) / mean if mean else 0.0
    # Fixed ratio-to-mean buckets: empty, <=.5x, <=1x, <=2x, and >2x.
    histogram = [0, 0, 0, 0, 0]
    for value in values:
        if value == 0:
            bucket = 0
        elif value <= mean * 0.5:
            bucket = 1
        elif value <= mean:
            bucket = 2
        elif value <= mean * 2.0:
            bucket = 3
        else:
            bucket = 4
        histogram[bucket] += 1
    return {
        "nonempty_groups": len(nonempty),
        "max_group_rows": max(values, default=0),
        "mean_group_rows_milli": round(mean * 1000),
        "coefficient_of_variation_milli": round(cv * 1000),
        "routing_histogram": tuple(histogram),
    }


@dataclass(frozen=True)
class RoutedGemmBenchmarkResult:
    config: RoutedGemmKernelConfig
    measured_us: float
    samples: int
    statistic: str = "median"

    def __post_init__(self) -> None:
        if not math.isfinite(self.measured_us) or self.measured_us < 0:
            raise ValueError("measured_us must be finite and non-negative")
        if self.samples <= 0:
            raise ValueError("benchmark samples must be positive")
        if not str(self.statistic).strip():
            raise ValueError("benchmark statistic must be non-empty")


CandidatePredicate = Callable[[RoutedGemmShapeKey, RoutedGemmKernelConfig], bool]
Benchmark = Callable[[RoutedGemmKernelConfig], RoutedGemmBenchmarkResult]


def select_tuning_winner(
    key: RoutedGemmShapeKey,
    candidates: Iterable[RoutedGemmKernelConfig],
    benchmark: Benchmark,
    *,
    predicates: Iterable[CandidatePredicate] = (),
    reset_output: Callable[[], None] | None = None,
) -> RoutedGemmBenchmarkResult:
    """Benchmark safe candidates and choose deterministically on equal timing."""

    safe = sorted(
        {
            candidate
            for candidate in candidates
            if all(predicate(key, candidate) for predicate in predicates)
        }
    )
    if not safe:
        raise RuntimeError("no safe routed GEMM tuning candidates")
    results: list[RoutedGemmBenchmarkResult] = []
    for candidate in safe:
        if reset_output is not None:
            reset_output()
        result = benchmark(candidate)
        if result.config != candidate:
            raise ValueError("benchmark result configuration does not match candidate")
        results.append(result)
    return min(results, key=lambda result: (result.measured_us, result.config))


_T = TypeVar("_T")


class RoutedGemmAutotuneMap:
    """Thread-safe exact-key map with single-flight cache-miss resolution."""

    def __init__(self, mode: str = "off") -> None:
        self.mode = normalize_routed_gemm_tuning_mode(mode)
        self._entries: dict[str, object] = {}
        self._resolving: set[str] = set()
        self._condition = Condition(RLock())

    def get(self, key: RoutedGemmShapeKey) -> object | None:
        if self.mode == "off":
            return None
        with self._condition:
            return self._entries.get(key.canonical())

    def publish(self, key: RoutedGemmShapeKey, value: _T) -> _T:
        if self.mode == "off":
            return value
        with self._condition:
            self._entries[key.canonical()] = value
            self._condition.notify_all()
        return value

    def resolve(self, key: RoutedGemmShapeKey, factory: Callable[[], _T]) -> _T:
        """Return one value, invoking ``factory`` once for concurrent misses."""

        if self.mode == "off":
            return factory()
        canonical = key.canonical()
        with self._condition:
            while canonical in self._resolving:
                self._condition.wait()
            existing = self._entries.get(canonical)
            if existing is not None:
                return existing  # type: ignore[return-value]
            if self.mode == "warmup_only":
                raise LookupError(
                    "routed GEMM warmup-only cache has no compatible entry for shape"
                )
            self._resolving.add(canonical)
        try:
            value = factory()
        except BaseException:
            with self._condition:
                self._resolving.discard(canonical)
                self._condition.notify_all()
            raise
        with self._condition:
            self._entries[canonical] = value
            self._resolving.discard(canonical)
            self._condition.notify_all()
        return value

    def snapshot(self) -> dict[str, object]:
        if self.mode == "off":
            return {}
        with self._condition:
            return dict(self._entries)


_MIRAI_CANDIDATES = (
    RoutedGemmKernelConfig(64, 32, 4, 2, False),
    RoutedGemmKernelConfig(64, 64, 4, 3, False),
    RoutedGemmKernelConfig(128, 32, 4, 3, False),
    RoutedGemmKernelConfig(128, 64, 8, 3, False),
)


def routed_gemm_candidate_registry(
    *, implementation: str, role: str
) -> tuple[RoutedGemmKernelConfig, ...]:
    """Return the finite Mirai launch space for a supported kernel role."""

    normalized_implementation = str(implementation).strip().lower()
    normalized_role = str(role).strip().lower()
    if normalized_implementation not in {"indexed_sm80", "regular_tma_sm90"}:
        raise ValueError(f"unsupported routed GEMM implementation {implementation!r}")
    if normalized_role not in {"forward", "dx", "dw", "weighted"}:
        raise ValueError(f"unsupported routed GEMM tuning role {role!r}")
    candidates = _MIRAI_CANDIDATES
    if normalized_implementation == "regular_tma_sm90":
        candidates = tuple(config for config in candidates if config.block_k >= 32)
    if normalized_role == "weighted":
        candidates = tuple(config for config in candidates if config.block_n <= 128)
    return candidates


def build_routed_gemm_environment_fingerprint(
    tensor: object,
    *,
    kernel_abi_fingerprint: str,
    compiler_target: str = "",
) -> RoutedGemmEnvironmentFingerprint:
    """Build the persistence identity at the eager CUDA execution boundary."""

    import torch

    from mirai.core.persistence.routed_gemm_autotune import (
        RoutedGemmEnvironmentFingerprint,
    )

    device = getattr(tensor, "device", None)
    if device is None or device.type != "cuda":
        raise ValueError("routed GEMM autotuning requires a CUDA tensor")
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    capability = torch.cuda.get_device_capability(index)
    triton_version = _module_version("triton")
    runtime = str(torch.version.cuda or "unknown")
    target = str(compiler_target).strip() or f"sm{capability[0]}{capability[1]}"
    driver = _cuda_driver_class(torch)
    return RoutedGemmEnvironmentFingerprint(
        backend="cuda",
        compute_capability=tuple(capability),
        sm_count=int(properties.multi_processor_count),
        shared_memory_bytes=int(
            getattr(properties, "shared_memory_per_block_optin", 0)
            or properties.shared_memory_per_block
        ),
        cuda_runtime_class=runtime,
        cuda_driver_class=driver,
        torch_version=str(torch.__version__),
        triton_version=triton_version,
        kernel_abi_fingerprint=str(kernel_abi_fingerprint),
        compiler_target=target,
    )


def build_routed_gemm_shape_key(
    activation: object,
    weight: object,
    counts: Sequence[int] | object,
    *,
    implementation: str,
    role: str,
    fusion: str,
    top_k: int,
    output_dtype: object | None = None,
    accumulation_dtype: str = "float32",
    segmented: bool = False,
) -> RoutedGemmShapeKey:
    """Construct a provider-agnostic exact shape key from CUDA operands."""

    import torch

    activation_device = getattr(activation, "device", None)
    weight_device = getattr(weight, "device", None)
    if activation_device is None or activation_device.type != "cuda":
        raise ValueError("routed GEMM shape keys require CUDA activation")
    if weight_device != activation_device:
        raise ValueError("routed GEMM activation and weight must share a CUDA device")
    if getattr(activation, "ndim", None) != 2 or getattr(weight, "ndim", None) != 3:
        raise ValueError("routed GEMM shape key expects rank-2 activation and rank-3 weight")
    if isinstance(counts, torch.Tensor):
        if counts.ndim != 1:
            raise ValueError("routed GEMM group counts must be rank-1")
        count_values = tuple(int(value) for value in counts.detach().to("cpu").tolist())
    else:
        count_values = tuple(int(value) for value in counts)  # type: ignore[arg-type]
    if len(count_values) != int(weight.shape[0]):
        raise ValueError("routed GEMM group counts must match the weight group axis")
    stats = routing_distribution_statistics(count_values)
    properties = torch.cuda.get_device_properties(activation_device)
    capability = torch.cuda.get_device_capability(activation_device)
    strides = tuple(int(value) for value in activation.stride()) + tuple(
        int(value) for value in weight.stride()
    )
    contiguous = bool(activation.is_contiguous() and weight.is_contiguous())
    pointers = (int(activation.data_ptr()), int(weight.data_ptr()))
    alignment = _common_power_of_two_alignment(pointers, maximum=128)
    return RoutedGemmShapeKey(
        backend="cuda",
        compute_capability=tuple(capability),
        sm_count=int(properties.multi_processor_count),
        shared_memory_bytes=int(
            getattr(properties, "shared_memory_per_block_optin", 0)
            or properties.shared_memory_per_block
        ),
        implementation=implementation,
        role=role,
        fusion=fusion,
        input_dtype=str(activation.dtype).removeprefix("torch."),
        output_dtype=str(output_dtype or activation.dtype).removeprefix("torch."),
        accumulation_dtype=accumulation_dtype,
        k_size=int(weight.shape[1]),
        n_size=int(weight.shape[2]),
        group_count=int(weight.shape[0]),
        routed_rows=sum(count_values),
        top_k=int(top_k),
        stride_class="contiguous" if contiguous else "strided:" + ",".join(map(str, strides)),
        alignment_bytes=alignment,
        segmented=bool(segmented),
        **stats,
    )


Verifier = Callable[[RoutedGemmBenchmarkResult], None]


class RoutedGemmAutotuner:
    """Runtime facade for deterministic, verified, optionally persistent tuning."""

    def __init__(
        self,
        *,
        mode: str = "off",
        environment_fingerprint: RoutedGemmEnvironmentFingerprint | None = None,
        cache_path: str | Path | None = None,
        conservative_config: RoutedGemmKernelConfig | None = None,
    ) -> None:
        self.mode = normalize_routed_gemm_tuning_mode(mode)
        self.environment_fingerprint = environment_fingerprint
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.conservative_config = conservative_config or RoutedGemmKernelConfig(
            64, 32, 4, 2, False
        )
        self._map = RoutedGemmAutotuneMap(self.mode)
        self._loaded = False
        self._load_lock = RLock()
        self._persist_lock = RLock()
        self.last_cache_status = "disabled" if self.mode == "off" else "not_loaded"

    def load(self) -> str:
        """Load compatible entries once; malformed state never reaches execution."""

        if self.mode == "off":
            return "disabled"
        with self._load_lock:
            if self._loaded:
                return self.last_cache_status
            self._loaded = True
            if self.cache_path is None:
                self.last_cache_status = "miss"
                return self.last_cache_status
            if self.environment_fingerprint is None:
                raise ValueError("persistent routed GEMM tuning requires an environment fingerprint")
            from mirai.core.persistence.routed_gemm_autotune import (
                load_routed_gemm_tuning_cache,
            )

            loaded = load_routed_gemm_tuning_cache(
                self.cache_path, self.environment_fingerprint  # type: ignore[arg-type]
            )
            self.last_cache_status = loaded.status
            if loaded.artifact is not None:
                for entry in loaded.artifact.entries:
                    self._map.publish(entry.shape_key, entry)
            return self.last_cache_status

    def resolve_config(
        self,
        key: RoutedGemmShapeKey,
        *,
        benchmark: Benchmark | None = None,
        verify: Verifier | None = None,
        predicates: Iterable[CandidatePredicate] = (),
        reset_output: Callable[[], None] | None = None,
        samples_parity_tolerance: float = 0.0,
        warmup_write: bool = False,
    ) -> RoutedGemmKernelConfig:
        """Resolve a launch config; tuning and writes occur only under policy."""

        if self.mode == "off":
            return self.conservative_config
        self.load()
        existing = self._map.get(key)
        if existing is not None:
            return existing.config  # type: ignore[no-any-return]
        if self.mode == "warmup_only" and not warmup_write:
            raise LookupError(
                "routed GEMM warmup-only cache has no compatible entry for shape"
            )
        if benchmark is None or verify is None:
            raise ValueError("an uncached routed GEMM tuning shape requires benchmark and verify callbacks")
        from mirai.core.persistence.routed_gemm_autotune import RoutedGemmTuningEntry

        def tune_entry() -> object:
            winner = select_tuning_winner(
                key,
                routed_gemm_candidate_registry(
                    implementation=key.implementation, role=key.role
                ),
                benchmark,
                predicates=predicates,
                reset_output=reset_output,
            )
            verify(winner)
            return RoutedGemmTuningEntry.create(
                shape_key=key,
                config=winner.config,
                measured_us=winner.measured_us,
                samples=winner.samples,
                parity_tolerance=samples_parity_tolerance,
                statistic=winner.statistic,
            )

        if self.mode == "online":
            entry = self._map.resolve(key, tune_entry)
        else:
            entry = self._map.publish(key, tune_entry())
        if self.cache_path is not None:
            self.persist()
        return entry.config  # type: ignore[no-any-return]

    def persist(self) -> None:
        if self.mode == "off":
            return
        if self.cache_path is None:
            return
        if self.environment_fingerprint is None:
            raise ValueError("persistent routed GEMM tuning requires an environment fingerprint")
        from mirai.core.persistence.routed_gemm_autotune import (
            RoutedGemmTuningArtifact,
            RoutedGemmTuningEntry,
            save_routed_gemm_tuning_cache,
        )

        entries = tuple(
            value
            for value in self._map.snapshot().values()
            if isinstance(value, RoutedGemmTuningEntry)
        )
        with self._persist_lock:
            save_routed_gemm_tuning_cache(
                self.cache_path,
                RoutedGemmTuningArtifact(
                    environment_fingerprint=self.environment_fingerprint,
                    entries=entries,
                ),
            )


def _module_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _cuda_driver_class(torch_module: object) -> str:
    cuda = getattr(torch_module, "cuda")
    driver_version = getattr(cuda, "driver_version", None)
    if callable(driver_version):
        return str(driver_version())
    getter = getattr(getattr(torch_module, "_C", object()), "_cuda_getDriverVersion", None)
    return str(getter()) if callable(getter) else "unknown"


def _common_power_of_two_alignment(values: Iterable[int], *, maximum: int) -> int:
    alignment = 1
    while alignment < maximum and all(value % (alignment * 2) == 0 for value in values):
        alignment *= 2
    return alignment


def _require_exact_fields(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise ValueError(f"invalid {label} fields: missing={missing}, unknown={unknown}")


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value
