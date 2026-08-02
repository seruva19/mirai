"""Pinned-memory budgeting helpers."""

from __future__ import annotations

import os


def total_system_ram_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size
        except Exception:
            pass
    return 16 * 1024 * 1024 * 1024


def estimate_pinned_memory_bytes(
    *,
    blocks_to_swap: int,
    num_workers: int,
    prefetch_factor: int,
    batch_size: int,
    bytes_per_swapped_block: int = 64 * 1024 * 1024,
    bytes_per_prefetch_batch: int = 8 * 1024 * 1024,
) -> int:
    swap_bytes = max(0, int(blocks_to_swap)) * int(bytes_per_swapped_block)
    loader_batches = max(0, int(num_workers)) * max(1, int(prefetch_factor))
    loader_bytes = loader_batches * max(1, int(batch_size)) * int(bytes_per_prefetch_batch)
    return int(swap_bytes + loader_bytes)


def pinned_memory_budget_warning(
    *,
    blocks_to_swap: int,
    num_workers: int,
    prefetch_factor: int,
    batch_size: int,
    budget_fraction: float,
    total_ram_bytes_override: int | None = None,
) -> str | None:
    total = int(total_ram_bytes_override or total_system_ram_bytes())
    if total <= 0:
        return None
    estimated = estimate_pinned_memory_bytes(
        blocks_to_swap=blocks_to_swap,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        batch_size=batch_size,
    )
    frac = float(estimated) / float(total)
    if frac <= float(budget_fraction):
        return None
    return (
        "Estimated pinned memory exceeds budget: "
        f"estimated={estimated} bytes ({frac:.1%}) > "
        f"budget={float(budget_fraction):.1%} of RAM ({total} bytes)."
    )
