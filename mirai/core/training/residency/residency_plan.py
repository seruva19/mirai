"""Phase-aware block-residency planning (pure logic over BlockSwapManager).

The planner decides, for a fixed ``(total_blocks, blocks_to_swap)`` split:

* which blocks stay VRAM-resident vs stream (``resident_blocks`` / ``swap_blocks``),
* which swap blocks to keep resident across the two phase turns so they are not
  evicted and immediately re-fetched (``forward_to_backward_pins`` /
  ``backward_to_forward_pins``), and
* the async prefetch depth (how many swap blocks may be in flight at once).

It is deliberately free of torch/CUDA so it can be unit-tested exhaustively.
Residency placement never changes the compute graph, so any plan preserves the
loss/gradient contract bit-for-bit; only *where* frozen weights live and *when*
they cross PCIe changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

BLOCK_RESIDENCY_PLANNERS = ("uniform", "phase_aware")
BLOCK_RESIDENCY_PRIORITIES = ("index", "routing_hot")
MIN_BLOCK_SWAP_PREFETCH_DEPTH = 1
MAX_BLOCK_SWAP_PREFETCH_DEPTH = 4


def normalize_block_residency_planner(value: str | None) -> str:
    text = str(value if value is not None else "uniform").strip().lower()
    aliases = {"": "uniform", "none": "uniform", "off": "uniform"}
    normalized = aliases.get(text, text)
    if normalized not in BLOCK_RESIDENCY_PLANNERS:
        raise ValueError(
            "memory.block_residency_planner must be one of: "
            + ", ".join(BLOCK_RESIDENCY_PLANNERS)
            + "."
        )
    return normalized


def normalize_block_residency_priority(value: str | None) -> str:
    text = str(value if value is not None else "index").strip().lower()
    aliases = {"": "index", "none": "index"}
    normalized = aliases.get(text, text)
    if normalized not in BLOCK_RESIDENCY_PRIORITIES:
        raise ValueError(
            "memory.block_residency_priority must be one of: "
            + ", ".join(BLOCK_RESIDENCY_PRIORITIES)
            + "."
        )
    return normalized


def normalize_block_swap_prefetch_depth(value: int | None) -> int:
    depth = int(value) if value is not None else MIN_BLOCK_SWAP_PREFETCH_DEPTH
    if depth < MIN_BLOCK_SWAP_PREFETCH_DEPTH or depth > MAX_BLOCK_SWAP_PREFETCH_DEPTH:
        raise ValueError(
            "memory.block_swap_prefetch_depth must be in "
            f"[{MIN_BLOCK_SWAP_PREFETCH_DEPTH}, {MAX_BLOCK_SWAP_PREFETCH_DEPTH}]."
        )
    return depth


def block_scores_from_router_loads(
    loads: Mapping[str, Sequence[float]],
) -> dict[int, float] | None:
    """Derive per-block hot-mass scores from accumulated router assignment counts.

    ``loads`` maps a router module name (``...blocks.<i>.ffn.router``) to its
    per-expert assignment-count vector. The score is the routing concentration
    (Herfindahl index of the normalized counts): a block whose tokens pile onto
    a few experts scores higher than one with flat routing. Returns ``None`` when
    no block index can be parsed / no mass is present, so the caller falls back
    to index-ordered residency. Pure: no torch, counts are plain numbers.
    """
    import re

    scores: dict[int, float] = {}
    pattern = re.compile(r"blocks\.(\d+)\.")
    for name, counts in loads.items():
        match = pattern.search(str(name))
        if match is None:
            continue
        values = [max(0.0, float(v)) for v in counts]
        total = sum(values)
        if total <= 0.0:
            continue
        herfindahl = sum((v / total) ** 2 for v in values)
        idx = int(match.group(1))
        scores[idx] = max(scores.get(idx, 0.0), herfindahl)
    return scores or None


@dataclass(frozen=True)
class ResidencyPlan:
    """Immutable residency plan consumed by BlockSwapManager."""

    total_blocks: int
    resident_blocks: tuple[int, ...]
    swap_blocks: tuple[int, ...]
    planner: str
    priority: str
    prefetch_depth: int
    forward_to_backward_pins: tuple[int, ...]
    backward_to_forward_pins: tuple[int, ...]

    @property
    def resident_set(self) -> frozenset[int]:
        return frozenset(self.resident_blocks)

    @property
    def swap_set(self) -> frozenset[int]:
        return frozenset(self.swap_blocks)

    @property
    def forward_pin_set(self) -> frozenset[int]:
        return frozenset(self.forward_to_backward_pins)

    @property
    def backward_pin_set(self) -> frozenset[int]:
        return frozenset(self.backward_to_forward_pins)


def _select_resident(
    *,
    total_blocks: int,
    gpu_count: int,
    priority: str,
    block_scores: Mapping[int, float] | Sequence[float] | None,
) -> tuple[int, ...]:
    if gpu_count <= 0:
        return ()
    if gpu_count >= total_blocks:
        return tuple(range(total_blocks))
    if priority == "routing_hot" and block_scores is not None:
        scores = _coerce_scores(block_scores, total_blocks)
        if scores is not None:
            # Highest hot-mass first; ties resolved by ascending block index so
            # the choice is deterministic and reproducible.
            ranked = sorted(
                range(total_blocks),
                key=lambda idx: (-scores[idx], idx),
            )
            return tuple(sorted(ranked[:gpu_count]))
    # The index policy keeps the first ``gpu_count`` blocks resident.
    return tuple(range(gpu_count))


def _coerce_scores(
    block_scores: Mapping[int, float] | Sequence[float],
    total_blocks: int,
) -> list[float] | None:
    scores = [0.0] * total_blocks
    saw_any = False
    if isinstance(block_scores, Mapping):
        for key, value in block_scores.items():
            idx = int(key)
            if 0 <= idx < total_blocks:
                scores[idx] = float(value)
                saw_any = True
    else:
        for idx, value in enumerate(block_scores):
            if idx >= total_blocks:
                break
            scores[idx] = float(value)
            saw_any = True
    return scores if saw_any else None


def build_residency_plan(
    *,
    total_blocks: int,
    blocks_to_swap: int,
    planner: str = "uniform",
    prefetch_depth: int = 1,
    priority: str = "index",
    block_scores: Mapping[int, float] | Sequence[float] | None = None,
) -> ResidencyPlan:
    total = max(0, int(total_blocks))
    swap_count = max(0, min(int(blocks_to_swap), total))
    gpu_count = total - swap_count
    planner = normalize_block_residency_planner(planner)
    priority = normalize_block_residency_priority(priority)
    depth = normalize_block_swap_prefetch_depth(prefetch_depth)

    resident = _select_resident(
        total_blocks=total,
        gpu_count=gpu_count,
        priority=priority,
        block_scores=block_scores,
    )
    resident_set = set(resident)
    # Swap blocks execute in ascending index order in the forward pass and in
    # descending order in the backward pass, regardless of which blocks were
    # chosen resident. Phase pins are keyed on that execution order.
    swap_blocks = tuple(idx for idx in range(total) if idx not in resident_set)

    if planner == "phase_aware" and swap_blocks:
        window = min(depth, len(swap_blocks))
        # Last `window` swap blocks (executed last in forward) are needed first
        # in backward -> keep resident through the fwd->bwd turn.
        forward_pins = tuple(swap_blocks[-window:])
        # First `window` swap blocks (executed last in backward) are needed
        # first in the next forward -> keep resident through the bwd->fwd turn.
        backward_pins = tuple(swap_blocks[:window])
    else:
        forward_pins = ()
        backward_pins = ()

    return ResidencyPlan(
        total_blocks=total,
        resident_blocks=resident,
        swap_blocks=swap_blocks,
        planner=planner,
        priority=priority,
        prefetch_depth=depth,
        forward_to_backward_pins=forward_pins,
        backward_to_forward_pins=backward_pins,
    )
