"""Resolve opt-in device-dependent memory defaults from a data table.

The resolver uses compute capability and available device memory only when
``memory.hardware_policy`` selects tiered planning. Explicit config values take
precedence, and unavailable CUDA metadata fails the selected policy explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # 3.11+ ships tomllib; older interpreters use the tomli backport, as
    # config/loader.py already does. Keep the two in step.
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - exercised on py3.10
    import tomli as tomllib  # type: ignore[no-redef]

_TIERS_PATH = Path(__file__).resolve().parent / "defaults" / "hardware_tiers.toml"
HARDWARE_MEMORY_POLICIES = ("disabled", "tiered")


@dataclass(frozen=True)
class HardwareMemoryPlan:
    tier_name: str
    total_vram_gib: float
    device_residency_budget_gib: float
    expert_device_cache_gib: float
    expert_dequant_chunk_size: int


def normalize_hardware_memory_policy(value: str | None) -> str:
    normalized = str(value or "disabled").strip().lower()
    normalized = {"": "disabled", "none": "disabled", "off": "disabled"}.get(
        normalized, normalized
    )
    if normalized not in HARDWARE_MEMORY_POLICIES:
        raise ValueError(
            "memory.hardware_policy must be one of: "
            + ", ".join(HARDWARE_MEMORY_POLICIES)
            + "."
        )
    return normalized


def _load_tiers() -> list[dict[str, Any]]:
    try:
        with _TIERS_PATH.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return []
    tiers = data.get("tier")
    return [t for t in tiers if isinstance(t, dict)] if isinstance(tiers, list) else []


def detect_device_profile() -> tuple[tuple[int, int], float] | None:
    """Return ((major, minor), total_vram_gib) for the active CUDA device."""

    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - torch is a base dependency
        return None
    try:
        if not torch.cuda.is_available():
            return None
        index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        total = float(torch.cuda.get_device_properties(index).total_memory) / 1024**3
    except (RuntimeError, AssertionError):  # pragma: no cover - driver hiccups
        return None
    return (int(major), int(minor)), total


def _tier_matches(
    tier: dict[str, Any], capability: tuple[int, int], vram_gib: float
) -> bool:
    low = tier.get("compute_capability_min")
    high = tier.get("compute_capability_max")
    if isinstance(low, list) and tuple(capability) < (int(low[0]), int(low[1])):
        return False
    if isinstance(high, list) and tuple(capability) > (int(high[0]), int(high[1])):
        return False
    vram_min = tier.get("total_vram_gib_min")
    vram_max = tier.get("total_vram_gib_max")
    if vram_min is not None and vram_gib <= float(vram_min):
        return False
    if vram_max is not None and vram_gib > float(vram_max):
        return False
    return True


def resolve_tier(
    *, profile: tuple[tuple[int, int], float] | None = None
) -> dict[str, Any] | None:
    """First tier in the table matching this device, or None."""

    resolved = profile if profile is not None else detect_device_profile()
    if resolved is None:
        return None
    capability, vram_gib = resolved
    for tier in _load_tiers():
        if _tier_matches(tier, capability, vram_gib):
            return tier
    return None


def resolve_expert_dequant_chunk_size(
    configured: int,
    *,
    profile: tuple[tuple[int, int], float] | None = None,
) -> int:
    """Tier default when the config left the chunk size unset, else the config.

    ``0`` is the existing "unset" sentinel: chunked_dequant already rejects it
    (experts.py), so resolving it here changes no working configuration. Any
    explicit value is returned untouched -- the table is a default, not a policy.
    """

    if int(configured) > 0:
        return int(configured)
    tier = resolve_tier(profile=profile)
    if tier is None:
        return int(configured)
    value = tier.get("expert_dequant_chunk_size")
    return int(value) if isinstance(value, int) and value > 0 else int(configured)


def resolve_hardware_memory_plan(
    memory: Any,
    *,
    profile: tuple[tuple[int, int], float] | None = None,
) -> HardwareMemoryPlan | None:
    """Resolve an opt-in, table-owned single-device memory plan."""

    policy = normalize_hardware_memory_policy(
        getattr(memory, "hardware_policy", "disabled")
    )
    if policy == "disabled":
        return None
    resolved_profile = profile if profile is not None else detect_device_profile()
    if resolved_profile is None:
        raise RuntimeError(
            "memory.hardware_policy='tiered' requires an available CUDA device."
        )
    _capability, total_vram_gib = resolved_profile
    tier = resolve_tier(profile=resolved_profile)
    if tier is None:
        raise ValueError(
            "memory.hardware_policy='tiered' found no matching hardware tier."
        )
    residency_fraction = _positive_fraction(
        tier, "device_residency_fraction"
    )
    effective_fraction = min(
        residency_fraction,
        float(getattr(memory, "cuda_memory_fraction", 1.0)),
    )
    configured_residency = float(
        getattr(memory, "device_residency_budget_gib", 0.0)
    )
    residency_gib = (
        configured_residency
        if configured_residency > 0.0
        else float(total_vram_gib) * effective_fraction
    )
    configured_cache = float(
        getattr(memory, "expert_device_cache_gib", 0.0)
    )
    quantization = str(
        getattr(memory, "frozen_weight_quantization", "none")
    ).strip().lower()
    cache_gib = configured_cache
    if cache_gib <= 0.0 and quantization == "int8":
        cache_gib = float(total_vram_gib) * _positive_fraction(
            tier, "expert_device_cache_fraction"
        )
        cache_gib = min(
            cache_gib,
            float(tier.get("expert_device_cache_max_gib", cache_gib)),
        )
    chunk = resolve_expert_dequant_chunk_size(
        int(getattr(memory, "expert_dequant_chunk_size", 0)),
        profile=resolved_profile,
    )
    return HardwareMemoryPlan(
        tier_name=str(tier["name"]),
        total_vram_gib=float(total_vram_gib),
        device_residency_budget_gib=float(residency_gib),
        expert_device_cache_gib=float(cache_gib),
        expert_dequant_chunk_size=int(chunk),
    )


def apply_hardware_memory_plan(
    memory: Any,
    *,
    profile: tuple[tuple[int, int], float] | None = None,
) -> HardwareMemoryPlan | None:
    plan = resolve_hardware_memory_plan(memory, profile=profile)
    if plan is None:
        return None
    memory.device_residency_budget_gib = plan.device_residency_budget_gib
    memory.expert_device_cache_gib = plan.expert_device_cache_gib
    memory.expert_dequant_chunk_size = plan.expert_dequant_chunk_size
    return plan


def _positive_fraction(tier: dict[str, Any], key: str) -> float:
    value = float(tier.get(key, 0.0))
    if not 0.0 < value <= 1.0:
        raise ValueError(f"Hardware tier {key} must be in (0, 1].")
    return value
