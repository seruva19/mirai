"""Versioned hardware calibration for routed-expert transfer scheduling."""

from __future__ import annotations

import json
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "mirai.expert_transfer_profile.v1"


def _finite_positive(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0.")
    return result


@dataclass(frozen=True)
class ExpertTransferProfile:
    """Measured transfer characteristics and bounded runtime recommendations."""

    gpu_name: str
    compute_capability: str
    expert_format: str
    expert_bytes: int
    h2d_gib_per_second: float
    routed_compute_gib_per_second: float
    recommended_device_cache_gib: float
    recommended_prefetch_depth: int
    benchmark_fingerprint: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"Unsupported expert transfer profile schema: {self.schema!r}.")
        for name in ("gpu_name", "compute_capability", "expert_format", "benchmark_fingerprint"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if int(self.expert_bytes) <= 0:
            raise ValueError("expert_bytes must be > 0.")
        _finite_positive("h2d_gib_per_second", self.h2d_gib_per_second)
        _finite_positive("routed_compute_gib_per_second", self.routed_compute_gib_per_second)
        cache = float(self.recommended_device_cache_gib)
        if not math.isfinite(cache) or cache < 0.0:
            raise ValueError("recommended_device_cache_gib must be finite and >= 0.")
        depth = int(self.recommended_prefetch_depth)
        if depth < 0 or depth > 16:
            raise ValueError("recommended_prefetch_depth must be between 0 and 16.")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertTransferProfile":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown expert transfer profile fields: {sorted(unknown)}.")
        missing = allowed - set(payload)
        if missing:
            raise ValueError(f"Missing expert transfer profile fields: {sorted(missing)}.")
        return cls(**{key: payload[key] for key in allowed})


def recommend_transfer_policy(
    *, h2d_gib_per_second: float, routed_compute_gib_per_second: float,
    expert_bytes: int, working_set_experts: int, cache_budget_gib: float,
) -> tuple[float, int]:
    """Return cache allocation and lookahead sufficient to hide one transfer."""

    h2d = _finite_positive("h2d_gib_per_second", h2d_gib_per_second)
    compute = _finite_positive("routed_compute_gib_per_second", routed_compute_gib_per_second)
    size, count, budget = int(expert_bytes), int(working_set_experts), float(cache_budget_gib)
    if size <= 0 or count <= 0:
        raise ValueError("expert_bytes and working_set_experts must be > 0.")
    if not math.isfinite(budget) or budget < 0.0:
        raise ValueError("cache_budget_gib must be finite and >= 0.")
    cache_gib = min(budget, size * count / float(1024**3))
    depth = min(4, max(1, math.ceil(compute / h2d)))
    return cache_gib, depth


def build_expert_transfer_profile(
    *, gpu_name: str, compute_capability: str, expert_format: str,
    expert_bytes: int, working_set_experts: int, cache_budget_gib: float,
    h2d_gib_per_second: float, routed_compute_gib_per_second: float,
    benchmark_protocol: Mapping[str, Any],
) -> ExpertTransferProfile:
    """Build a lineage-bound profile from an explicitly measured protocol."""

    canonical = json.dumps(
        dict(benchmark_protocol), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    cache_gib, depth = recommend_transfer_policy(
        h2d_gib_per_second=h2d_gib_per_second,
        routed_compute_gib_per_second=routed_compute_gib_per_second,
        expert_bytes=expert_bytes,
        working_set_experts=working_set_experts,
        cache_budget_gib=cache_budget_gib,
    )
    # Mirai's global device cache currently owns immutable INT8 operands only.
    # Other formats can still consume the calibrated transfer lookahead, but may
    # not turn a throughput profile into an unsupported cache configuration.
    if str(expert_format).strip().lower() != "int8":
        cache_gib = 0.0
    return ExpertTransferProfile(
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        expert_format=expert_format,
        expert_bytes=expert_bytes,
        h2d_gib_per_second=h2d_gib_per_second,
        routed_compute_gib_per_second=routed_compute_gib_per_second,
        recommended_device_cache_gib=cache_gib,
        recommended_prefetch_depth=depth,
        benchmark_fingerprint=fingerprint,
    )


def load_expert_transfer_profile(path: str | Path) -> ExpertTransferProfile:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load expert transfer profile {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expert transfer profile root must be an object.")
    return ExpertTransferProfile.from_dict(payload)


def validate_expert_transfer_profile_identity(
    profile: ExpertTransferProfile,
    *,
    gpu_name: str,
    compute_capability: str,
    expert_format: str,
) -> None:
    """Fail when calibration lineage differs from the execution environment."""

    observed = {
        "gpu_name": str(gpu_name).strip(),
        "compute_capability": str(compute_capability).strip(),
        "expert_format": str(expert_format).strip().lower(),
    }
    expected = {
        "gpu_name": str(profile.gpu_name).strip(),
        "compute_capability": str(profile.compute_capability).strip(),
        "expert_format": str(profile.expert_format).strip().lower(),
    }
    mismatches = [
        f"{name}: profile={expected[name]!r}, runtime={observed[name]!r}"
        for name in expected
        if expected[name] != observed[name]
    ]
    if mismatches:
        raise ValueError(
            "Expert transfer profile does not match runtime identity ("
            + "; ".join(mismatches)
            + "). Regenerate the profile on the target GPU and expert format."
        )


def save_expert_transfer_profile(path: str | Path, profile: ExpertTransferProfile) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


__all__ = ["ExpertTransferProfile", "SCHEMA", "build_expert_transfer_profile", "load_expert_transfer_profile", "recommend_transfer_policy", "save_expert_transfer_profile", "validate_expert_transfer_profile_identity"]
