"""Process-wide memory safety limits for one training runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemorySafetyPolicy:
    cuda_memory_fraction: float = 1.0
    minimum_system_memory_gib: float = 0.0
    # Absolute ceiling on page-locked (pinned) HOST memory EACH pinning
    # subsystem (block-swap residency and compressed_weights packed-state preload) may
    # allocate. The ceiling prevents pinned allocation from scaling with all
    # available host memory.
    max_pinned_host_gib: float = 12.0


_ACTIVE_POLICY = MemorySafetyPolicy()


def configure_memory_safety(memory_config: Any) -> MemorySafetyPolicy:
    global _ACTIVE_POLICY
    policy = MemorySafetyPolicy(
        cuda_memory_fraction=float(
            getattr(memory_config, "cuda_memory_fraction", 1.0)
        ),
        minimum_system_memory_gib=float(
            getattr(memory_config, "minimum_system_memory_gib", 0.0)
        ),
        max_pinned_host_gib=float(
            getattr(memory_config, "max_pinned_host_gib", 12.0)
        ),
    )
    if not 0.0 < policy.cuda_memory_fraction <= 1.0:
        raise ValueError("memory.cuda_memory_fraction must be in (0, 1].")
    if policy.minimum_system_memory_gib < 0.0:
        raise ValueError("memory.minimum_system_memory_gib must be >= 0.")
    if policy.max_pinned_host_gib <= 0.0:
        raise ValueError("memory.max_pinned_host_gib must be > 0.")
    _ACTIVE_POLICY = policy
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        return policy
    # A fraction of 1.0 leaves the CUDA caching allocator unmodified.
    if torch.cuda.is_available() and policy.cuda_memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(policy.cuda_memory_fraction)
    return policy


def current_memory_safety_policy() -> MemorySafetyPolicy:
    return _ACTIVE_POLICY


def assert_system_memory_headroom(*, context: str) -> None:
    minimum_gib = float(_ACTIVE_POLICY.minimum_system_memory_gib)
    if minimum_gib <= 0.0:
        return
    try:
        import psutil
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "memory.minimum_system_memory_gib requires psutil."
        ) from exc
    available_gib = float(psutil.virtual_memory().available) / (1024.0**3)
    if available_gib < minimum_gib:
        raise MemoryError(
            f"System memory safety limit reached during {context}: "
            f"{available_gib:.2f} GiB available, {minimum_gib:.2f} GiB required. "
            "Reduce expert_dequant_chunk_size, disable async block swap, or lower "
            "the training bucket before retrying."
        )
