# SPDX-License-Identifier: Apache-2.0
"""Versioned persistence for routed grouped-GEMM tuning results."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterator, Mapping
from uuid import uuid4

from mirai.core.moe.runtime.routed_gemm_autotune import (
    BENCHMARK_PROTOCOL_VERSION,
    KERNEL_CONTRACT_VERSION,
    RoutedGemmKernelConfig,
    RoutedGemmShapeKey,
)


ROUTED_GEMM_AUTOTUNE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RoutedGemmEnvironmentFingerprint:
    """Exact software and architecture identity, intentionally GPU-UUID-free."""

    backend: str
    compute_capability: tuple[int, int]
    sm_count: int
    shared_memory_bytes: int
    cuda_runtime_class: str
    cuda_driver_class: str
    torch_version: str
    triton_version: str
    kernel_abi_fingerprint: str
    compiler_target: str

    def __post_init__(self) -> None:
        for name in (
            "backend", "cuda_runtime_class", "cuda_driver_class", "torch_version",
            "triton_version", "kernel_abi_fingerprint", "compiler_target",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"environment fingerprint {name} must be non-empty")
            object.__setattr__(self, name, value)
        capability = tuple(self.compute_capability)
        if len(capability) != 2 or any(type(part) is not int or part < 0 for part in capability):
            raise ValueError("compute_capability must contain two non-negative integers")
        if type(self.sm_count) is not int or self.sm_count <= 0:
            raise ValueError("environment fingerprint sm_count must be positive")
        if type(self.shared_memory_bytes) is not int or self.shared_memory_bytes < 0:
            raise ValueError("shared_memory_bytes must be non-negative")
        object.__setattr__(self, "compute_capability", capability)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["compute_capability"] = list(self.compute_capability)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RoutedGemmEnvironmentFingerprint":
        _exact(value, set(cls.__dataclass_fields__), "environment_fingerprint")
        kwargs = dict(value)
        capability = kwargs["compute_capability"]
        if not isinstance(capability, list):
            raise ValueError("compute_capability must be a JSON array")
        kwargs["compute_capability"] = tuple(_integer(item, "compute_capability") for item in capability)
        kwargs["sm_count"] = _integer(kwargs["sm_count"], "sm_count")
        kwargs["shared_memory_bytes"] = _integer(
            kwargs["shared_memory_bytes"], "shared_memory_bytes"
        )
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RoutedGemmTuningEntry:
    shape_key: RoutedGemmShapeKey
    implementation: str
    config: RoutedGemmKernelConfig
    measured_us: float
    statistic: str
    samples: int
    parity_tolerance: float
    created_at: str
    benchmark_protocol_version: int = BENCHMARK_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if str(self.implementation).strip() != self.shape_key.implementation:
            raise ValueError("entry implementation must match its semantic shape key")
        if not _finite_nonnegative(self.measured_us):
            raise ValueError("entry measured_us must be finite and non-negative")
        if not _finite_nonnegative(self.parity_tolerance):
            raise ValueError("entry parity_tolerance must be finite and non-negative")
        if type(self.samples) is not int or self.samples <= 0:
            raise ValueError("entry samples must be positive")
        if type(self.benchmark_protocol_version) is not int or self.benchmark_protocol_version <= 0:
            raise ValueError("entry benchmark_protocol_version must be positive")
        if not str(self.statistic).strip():
            raise ValueError("entry statistic must be non-empty")
        try:
            parsed = datetime.fromisoformat(str(self.created_at).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("entry created_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("entry created_at must include a timezone")

    @classmethod
    def create(
        cls,
        *,
        shape_key: RoutedGemmShapeKey,
        config: RoutedGemmKernelConfig,
        measured_us: float,
        samples: int,
        parity_tolerance: float,
        statistic: str = "median",
    ) -> "RoutedGemmTuningEntry":
        return cls(
            shape_key=shape_key,
            implementation=shape_key.implementation,
            config=config,
            measured_us=measured_us,
            statistic=statistic,
            samples=samples,
            parity_tolerance=parity_tolerance,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "shape_key": self.shape_key.to_dict(),
            "implementation": self.implementation,
            "config": self.config.to_dict(),
            "measured_us": self.measured_us,
            "statistic": self.statistic,
            "samples": self.samples,
            "parity_tolerance": self.parity_tolerance,
            "created_at": self.created_at,
            "benchmark_protocol_version": self.benchmark_protocol_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RoutedGemmTuningEntry":
        _exact(value, set(cls.__dataclass_fields__), "tuning entry")
        shape = value["shape_key"]
        config = value["config"]
        if not isinstance(shape, dict) or not isinstance(config, dict):
            raise ValueError("entry shape_key and config must be JSON objects")
        return cls(
            shape_key=RoutedGemmShapeKey.from_dict(shape),
            implementation=_text(value["implementation"], "implementation"),
            config=RoutedGemmKernelConfig.from_dict(config),
            measured_us=_number(value["measured_us"], "measured_us"),
            statistic=_text(value["statistic"], "statistic"),
            samples=_integer(value["samples"], "samples"),
            parity_tolerance=_number(value["parity_tolerance"], "parity_tolerance"),
            created_at=_text(value["created_at"], "created_at"),
            benchmark_protocol_version=_integer(
                value["benchmark_protocol_version"], "benchmark_protocol_version"
            ),
        )


@dataclass(frozen=True)
class RoutedGemmTuningArtifact:
    environment_fingerprint: RoutedGemmEnvironmentFingerprint
    entries: tuple[RoutedGemmTuningEntry, ...] = ()
    schema_version: int = ROUTED_GEMM_AUTOTUNE_SCHEMA_VERSION
    kernel_contract_version: int = KERNEL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROUTED_GEMM_AUTOTUNE_SCHEMA_VERSION:
            raise ValueError(f"unsupported routed GEMM tuning schema_version {self.schema_version}")
        if self.kernel_contract_version != KERNEL_CONTRACT_VERSION:
            raise ValueError(
                f"incompatible routed GEMM kernel_contract_version {self.kernel_contract_version}"
            )
        entries = tuple(self.entries)
        canonical = [entry.shape_key.canonical() for entry in entries]
        if len(canonical) != len(set(canonical)):
            raise ValueError("routed GEMM tuning artifact contains duplicate shape keys")
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kernel_contract_version": self.kernel_contract_version,
            "environment_fingerprint": self.environment_fingerprint.to_dict(),
            "entries": {
                entry.shape_key.canonical(): entry.to_dict()
                for entry in sorted(self.entries, key=lambda item: item.shape_key.canonical())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RoutedGemmTuningArtifact":
        _exact(
            value,
            {"schema_version", "kernel_contract_version", "environment_fingerprint", "entries"},
            "tuning artifact",
        )
        environment = value["environment_fingerprint"]
        entries = value["entries"]
        if not isinstance(environment, dict) or not isinstance(entries, dict):
            raise ValueError("environment_fingerprint and entries must be JSON objects")
        parsed: list[RoutedGemmTuningEntry] = []
        for canonical, payload in entries.items():
            if not isinstance(canonical, str) or not isinstance(payload, dict):
                raise ValueError("artifact entries must map string keys to JSON objects")
            entry = RoutedGemmTuningEntry.from_dict(payload)
            if canonical != entry.shape_key.canonical():
                raise ValueError("artifact entry key does not match its semantic shape key")
            parsed.append(entry)
        return cls(
            schema_version=_integer(value["schema_version"], "schema_version"),
            kernel_contract_version=_integer(
                value["kernel_contract_version"], "kernel_contract_version"
            ),
            environment_fingerprint=RoutedGemmEnvironmentFingerprint.from_dict(environment),
            entries=tuple(parsed),
        )

    def entry_for(self, key: RoutedGemmShapeKey) -> RoutedGemmTuningEntry | None:
        canonical = key.canonical()
        return next(
            (entry for entry in self.entries if entry.shape_key.canonical() == canonical),
            None,
        )


@dataclass(frozen=True)
class RoutedGemmCacheLoad:
    artifact: RoutedGemmTuningArtifact | None
    status: str
    reason: str
    quarantined_path: Path | None = None


@contextmanager
def routed_gemm_cache_lock(path: str | Path) -> Iterator[None]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".lock")
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except FileExistsError as exc:
        raise RuntimeError(f"routed GEMM tuning cache lock is already held for {target}") from exc
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def save_routed_gemm_tuning_cache(
    path: str | Path, artifact: RoutedGemmTuningArtifact
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    content = json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n"
    with routed_gemm_cache_lock(target):
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def load_routed_gemm_tuning_cache(
    path: str | Path,
    expected_environment: RoutedGemmEnvironmentFingerprint,
    *,
    quarantine_corrupt: bool = True,
) -> RoutedGemmCacheLoad:
    target = Path(path)
    if not target.exists():
        return RoutedGemmCacheLoad(None, "miss", "cache file does not exist")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("tuning cache root must be a JSON object")
        _exact(
            value,
            {"schema_version", "kernel_contract_version", "environment_fingerprint", "entries"},
            "tuning artifact",
        )
        if value["schema_version"] != ROUTED_GEMM_AUTOTUNE_SCHEMA_VERSION:
            return RoutedGemmCacheLoad(
                None,
                "incompatible",
                f"unsupported routed GEMM tuning schema_version {value['schema_version']!r}",
            )
        if value["kernel_contract_version"] != KERNEL_CONTRACT_VERSION:
            return RoutedGemmCacheLoad(
                None,
                "incompatible",
                "routed GEMM tuning cache kernel contract does not match exactly",
            )
        artifact = RoutedGemmTuningArtifact.from_dict(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        quarantined = _quarantine(target) if quarantine_corrupt else None
        return RoutedGemmCacheLoad(
            None,
            "corrupt",
            f"invalid routed GEMM tuning cache: {exc}",
            quarantined,
        )
    if artifact.environment_fingerprint != expected_environment:
        return RoutedGemmCacheLoad(
            None,
            "incompatible",
            "routed GEMM tuning cache environment fingerprint does not match exactly",
        )
    return RoutedGemmCacheLoad(artifact, "hit", "compatible tuning cache")


def migrate_routed_gemm_tuning_payload(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the current schema; no implicit cross-version migration exists."""

    version = value.get("schema_version")
    if version != ROUTED_GEMM_AUTOTUNE_SCHEMA_VERSION:
        raise ValueError(
            f"no routed GEMM tuning cache migration from schema_version {version!r}"
        )
    return RoutedGemmTuningArtifact.from_dict(value).to_dict()


def _quarantine(path: Path) -> Path | None:
    destination = path.with_name(
        f"{path.name}.corrupt.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    try:
        os.replace(path, destination)
    except OSError:
        return None
    return destination


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"invalid {label} fields: missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a number")
    return float(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_nonnegative(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number >= 0 and number != float("inf") and number == number
