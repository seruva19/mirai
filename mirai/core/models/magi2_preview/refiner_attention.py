"""Native range-union attention for the MAGI-2 refiner's local-attention layers.

The vendored refiner dispatches its window attention through
``torch.ops.magi2.flex_flash_attn_func``, an operator only MagiCompiler
registers, whose implementation is MagiAttention's ``flex_flash_attn_func`` CUDA
extension. Neither package is installable from PyPI. The eager path vendored
beside the operator
(``mirai/vendors/magi2_preview/model/magi2_refiner.py::_custom_flex_flash_attn_func``)
still needs FlashAttention-2 and re-derives the result by merging per-range
partial softmaxes through their log-sum-exps, which double-counts any key
reachable from more than one range. This module supplies the third path: the
same semantics expressed once as a PyTorch FlexAttention mask.

Semantics reproduced here, read off SandAI's operator contract and its eager
reference:

* ``q_ranges`` and ``k_ranges`` are ``[N, 2]`` half-open ``[start, end)`` index
  ranges, paired row by row. Row ``i`` states that every query in
  ``q_ranges[i]`` may attend to every key in ``k_ranges[i]``.
* A query position covered by several rows attends to the UNION of those key
  ranges under ONE softmax. Union rather than sum is the whole content of the
  operator: the vendored reference reaches it only when the rows a query is
  covered by have disjoint key ranges, which is a property of the ranges the
  released refiner produces and not an enforced invariant of the operator.
* ``attn_type_map`` selects a per-range masking mode. Only the FULL mode (``0``)
  is implemented here; the released refiner profile emits an all-zero map.
* The softmax scale is ``1/sqrt(head_dim)``, the operator's default, and there
  is no softcap and no attention sink on this path.
* A query position covered by no row, or covered only by rows whose key range is
  empty, attends to nothing. Its output row is zero — matching the vendored
  reference, which leaves such rows at their ``zeros_like`` initialization — and
  its log-sum-exp is ``-inf``, the true log-sum-exp of an empty set.

``auto_range_merge`` and ``sparse_load`` are tile-scheduling and load-sparsity
knobs of the CUDA kernel. They select how the kernel visits the ranges, not
which key each query attends to, so this path ignores them.

Mask construction. Splitting the query axis at every ``q_ranges`` boundary
yields elementary segments, each of which every row either fully covers or is
disjoint from, so a segment's allowed key set is fixed. A boundary-event sweep
tracks the active key ranges and stores their merged union as a short interval
list per segment. The released profile normally needs two intervals: the local
video window and the dense conditioning tail. This compact representation is
important at 1080p: expanding every range across every query segment creates
billions of intermediate indices even though the resulting union is sparse.

Tensor semantics: ``query`` is ``[tokens, heads_q, head_dim]`` and ``key`` /
``value`` are ``[key_tokens, heads_kv, head_dim]``, unbatched, as the vendored
operator receives them after its own squeeze.

Attribution: the range-pairing semantics and the eager reference this path is
verified against come from SandAI MAGI-2-preview, Apache-2.0
(https://github.com/SandAI-org/MAGI-2-preview), whose refiner calls
MagiAttention's ``flex_flash_attn_func``
(https://github.com/SandAI-org/MagiAttention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from mirai.core.models.attention_backends import (
    attention_backend_status,
    flex_attention_function,
)


# Selectable refiner attention paths. ``auto`` yields to a registered
# ``torch.ops.magi2`` operator when MagiCompiler published one and takes this
# native path otherwise; ``vendor_eager`` always keeps the vendored dispatch.
MAGI2_REFINER_ATTENTION_BACKENDS: tuple[str, ...] = (
    "auto",
    "native_flex",
    "vendor_eager",
)
MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT = "auto"

# The one ``attn_type_map`` value this path implements: every pair is a full
# rectangle with no causal or bidirectional-window trimming.
MAGI2_REFINER_ATTENTION_TYPE_FULL = 0

# Operator this path replaces when it is selected.
MAGI2_REFINER_FLEX_OP = "flex_flash_attn_func"

# Operator the non-local refiner layers still dispatch through.
MAGI2_REFINER_DENSE_OP = "flash_attn_func"


class Magi2RefinerAttentionUnsupported(ValueError):
    """Raised when the native refiner attention path cannot serve a request."""


def normalize_refiner_attention_backend(value: Any) -> str:
    """Coerce a configured refiner attention backend name to its canonical form."""
    name = str(value if value is not None else "").strip().lower()
    if not name:
        return MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT
    if name not in MAGI2_REFINER_ATTENTION_BACKENDS:
        raise Magi2RefinerAttentionUnsupported(
            "MAGI-2 family_params.refiner_attention_backend must be one of: "
            + ", ".join(MAGI2_REFINER_ATTENTION_BACKENDS)
            + f"; got {value!r}."
        )
    return name


def validate_refiner_flex_support(device: torch.device | None = None) -> None:
    """Reject the native path before the refiner runs when torch cannot serve it."""
    probe = torch.device("cpu") if device is None else torch.device(device)
    status = attention_backend_status("flex", device=probe, varlen=True)
    if not status.available:
        raise Magi2RefinerAttentionUnsupported(
            "family_params.refiner_attention_backend selected PyTorch "
            f"FlexAttention for the MAGI-2 refiner, which is unavailable: {status.reason}"
        )


# --------------------------------------------------------------------------- #
# Range -> mask
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Magi2RangeUnionMask:
    """Per-query-segment key membership of a paired range list.

    ``interval_starts`` / ``interval_ends`` carry the disjoint key-range union
    for each elementary query segment plus a final empty row. ``segment_of``
    maps a query position to its row, so membership checks only the compact
    interval slots instead of indexing a dense ``segments * key_tokens`` table.
    """

    interval_starts: torch.Tensor
    interval_ends: torch.Tensor
    segment_of: torch.Tensor
    segment_has_keys: torch.Tensor
    boundaries: torch.Tensor
    key_tokens: int

    @property
    def allowed(self) -> torch.Tensor:
        """Materialize membership for diagnostics and small contract tests."""
        keys = torch.arange(
            int(self.key_tokens), device=self.interval_starts.device
        ).view(1, 1, -1)
        return (
            (self.interval_starts.unsqueeze(-1) <= keys)
            & (keys < self.interval_ends.unsqueeze(-1))
        ).any(dim=1)

    @property
    def query_has_keys(self) -> torch.Tensor:
        """Per query position: whether any key is reachable from it at all."""
        return self.segment_has_keys[self.segment_of]

    def mask_mod(self) -> Callable[..., torch.Tensor]:
        interval_starts = self.interval_starts
        interval_ends = self.interval_ends
        segment_of = self.segment_of
        slots = int(interval_starts.shape[1])

        def range_union_mask(
            batch: torch.Tensor,
            head: torch.Tensor,
            query_index: torch.Tensor,
            key_index: torch.Tensor,
        ) -> torch.Tensor:
            segment = segment_of[query_index]
            allowed = torch.zeros_like(query_index, dtype=torch.bool)
            # A fixed Python loop is unrolled while FlexAttention traces the
            # score modifier; released masks keep this slot count very small.
            for slot in range(slots):
                starts = interval_starts[:, slot]
                ends = interval_ends[:, slot]
                allowed = allowed | (
                    (starts[segment] <= key_index) & (key_index < ends[segment])
                )
            return allowed

        return range_union_mask

    def block_mask(self, *, query_tokens: int, block_size: int = 128) -> Any:
        """Build FlexAttention's sparse block metadata without a dense Q*K mask."""
        from torch.nn.attention.flex_attention import BlockMask

        query_tokens = int(query_tokens)
        key_tokens = int(self.key_tokens)
        block_size = int(block_size)
        query_blocks = math.ceil(query_tokens / block_size)
        key_blocks_by_query: list[set[int]] = [set() for _ in range(query_blocks)]
        boundaries = [int(value) for value in self.boundaries.detach().cpu().tolist()]
        starts = self.interval_starts.detach().cpu().tolist()
        ends = self.interval_ends.detach().cpu().tolist()
        for segment, (query_start, query_end) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True)
        ):
            if query_start >= query_end:
                continue
            first_query_block = query_start // block_size
            last_query_block = (query_end - 1) // block_size
            key_blocks: set[int] = set()
            for key_start, key_end in zip(
                starts[segment], ends[segment], strict=True
            ):
                if key_start >= key_end:
                    continue
                key_blocks.update(
                    range(key_start // block_size, (key_end - 1) // block_size + 1)
                )
            for query_block in range(first_query_block, last_query_block + 1):
                key_blocks_by_query[query_block].update(key_blocks)

        # BlockMask.from_kv_blocks uses the padded width as the total key-block
        # domain while transposing the sparse rows, so retain that full block
        # width even though only ``counts`` entries in each row are meaningful.
        max_key_blocks = max(math.ceil(key_tokens / block_size), 1)
        counts = torch.tensor(
            [len(indices) for indices in key_blocks_by_query],
            dtype=torch.int32,
            device=self.segment_of.device,
        ).view(1, 1, query_blocks)
        indices = torch.zeros(
            (1, 1, query_blocks, max_key_blocks),
            dtype=torch.int32,
            device=self.segment_of.device,
        )
        for query_block, key_blocks in enumerate(key_blocks_by_query):
            ordered = sorted(key_blocks)
            if ordered:
                indices[0, 0, query_block, : len(ordered)] = torch.tensor(
                    ordered, dtype=torch.int32, device=self.segment_of.device
                )
        return BlockMask.from_kv_blocks(
            counts,
            indices,
            BLOCK_SIZE=block_size,
            mask_mod=self.mask_mod(),
            seq_lengths=(query_tokens, key_tokens),
        )


def _validated_ranges(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    *,
    query_tokens: int,
    key_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the paired ranges as int64 rows after boundary validation."""
    if q_ranges.ndim != 2 or int(q_ranges.shape[1]) != 2:
        raise Magi2RefinerAttentionUnsupported(
            f"MAGI-2 refiner q_ranges must be [N, 2]; got {tuple(q_ranges.shape)}."
        )
    if tuple(k_ranges.shape) != tuple(q_ranges.shape):
        raise Magi2RefinerAttentionUnsupported(
            "MAGI-2 refiner q_ranges and k_ranges are paired row by row and must "
            f"share shape; got {tuple(q_ranges.shape)} and {tuple(k_ranges.shape)}."
        )
    q = q_ranges.to(dtype=torch.int64)
    k = k_ranges.to(dtype=torch.int64)
    if bool((q[:, 0] > q[:, 1]).any()) or bool((k[:, 0] > k[:, 1]).any()):
        raise Magi2RefinerAttentionUnsupported(
            "MAGI-2 refiner ranges are half-open [start, end) and require "
            "start <= end."
        )
    if bool((q[:, 0] < 0).any()) or bool((q[:, 1] > int(query_tokens)).any()):
        raise Magi2RefinerAttentionUnsupported(
            f"MAGI-2 refiner q_ranges must lie within [0, {int(query_tokens)}]."
        )
    if bool((k[:, 0] < 0).any()) or bool((k[:, 1] > int(key_tokens)).any()):
        raise Magi2RefinerAttentionUnsupported(
            f"MAGI-2 refiner k_ranges must lie within [0, {int(key_tokens)}]."
        )
    return q, k


def build_range_union_mask(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    *,
    query_tokens: int,
    key_tokens: int,
) -> Magi2RangeUnionMask:
    """Build the per-segment key membership table of a paired range list.

    Empty ranges contribute nothing: a zero-width query range covers no segment
    and a zero-width key range adds and removes the same boundary.
    """
    device = q_ranges.device
    q, k = _validated_ranges(
        q_ranges, k_ranges, query_tokens=int(query_tokens), key_tokens=int(key_tokens)
    )
    boundaries = torch.unique(q.reshape(-1), sorted=True)
    segments = max(int(boundaries.numel()) - 1, 0)

    # Build boundary events on CPU. Range metadata is small relative to video
    # activations, and the event sweep avoids materializing one entry per
    # (range, covered-segment) pair on CUDA.
    boundary_values = [int(value) for value in boundaries.detach().cpu().tolist()]
    boundary_index = {value: index for index, value in enumerate(boundary_values)}
    starts: list[list[tuple[int, int]]] = [[] for _ in boundary_values]
    ends: list[list[tuple[int, int]]] = [[] for _ in boundary_values]
    for q_pair, k_pair in zip(
        q.detach().cpu().tolist(), k.detach().cpu().tolist(), strict=True
    ):
        q_start, q_end = (int(q_pair[0]), int(q_pair[1]))
        k_start, k_end = (int(k_pair[0]), int(k_pair[1]))
        if q_start == q_end or k_start == k_end:
            continue
        interval = (k_start, k_end)
        starts[boundary_index[q_start]].append(interval)
        ends[boundary_index[q_end]].append(interval)

    active: dict[tuple[int, int], int] = {}
    segment_intervals: list[list[tuple[int, int]]] = []
    for index in range(segments):
        for interval in ends[index]:
            remaining = active[interval] - 1
            if remaining:
                active[interval] = remaining
            else:
                del active[interval]
        for interval in starts[index]:
            active[interval] = active.get(interval, 0) + 1

        merged: list[tuple[int, int]] = []
        for k_start, k_end in sorted(active):
            if merged and k_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], k_end))
            else:
                merged.append((k_start, k_end))
        segment_intervals.append(merged)

    slots = max((len(intervals) for intervals in segment_intervals), default=0)
    # Keep one always-false slot so mask_mod has a uniform, non-empty axis even
    # when every query range is empty.
    slots = max(slots, 1)
    padded_starts = [[int(key_tokens)] * slots for _ in range(segments + 1)]
    padded_ends = [[int(key_tokens)] * slots for _ in range(segments + 1)]
    for segment, intervals in enumerate(segment_intervals):
        for slot, (k_start, k_end) in enumerate(intervals):
            padded_starts[segment][slot] = k_start
            padded_ends[segment][slot] = k_end
    interval_starts = torch.tensor(
        padded_starts, dtype=torch.int64, device=device
    )
    interval_ends = torch.tensor(padded_ends, dtype=torch.int64, device=device)

    positions = torch.arange(int(query_tokens), device=device)
    # ``right=True`` yields the index i with boundaries[i-1] <= pos < boundaries[i],
    # so i - 1 is the segment. Positions below the first or at/after the last
    # boundary fall outside every range and take the all-false row.
    segment_of = torch.bucketize(positions, boundaries, right=True) - 1
    outside = (segment_of < 0) | (segment_of >= segments)
    segment_of = torch.where(
        outside, torch.full_like(segment_of, segments), segment_of
    )
    return Magi2RangeUnionMask(
        interval_starts=interval_starts,
        interval_ends=interval_ends,
        segment_of=segment_of,
        segment_has_keys=(interval_starts < interval_ends).any(dim=1),
        boundaries=boundaries,
        key_tokens=int(key_tokens),
    )


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
def _validate_attn_type_map(
    attn_type_map: torch.Tensor | None, *, ranges: int
) -> None:
    if attn_type_map is None:
        return
    if int(attn_type_map.numel()) == 0:
        return
    if int(attn_type_map.numel()) != int(ranges):
        raise Magi2RefinerAttentionUnsupported(
            "MAGI-2 refiner attn_type_map carries one entry per range; got "
            f"{int(attn_type_map.numel())} for {int(ranges)} ranges."
        )
    if bool((attn_type_map != MAGI2_REFINER_ATTENTION_TYPE_FULL).any()):
        observed = sorted({int(value) for value in attn_type_map.flatten().tolist()})
        raise Magi2RefinerAttentionUnsupported(
            "The native MAGI-2 refiner attention path implements the full "
            f"attention type ({MAGI2_REFINER_ATTENTION_TYPE_FULL}) only; the "
            f"request carries types {observed}. Set "
            "family_params.refiner_attention_backend='vendor_eager' to keep the "
            "vendored dispatch for a profile that needs another type."
        )


def flex_range_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    *,
    compile_kernel: bool | None = None,
    compile_block_mask: bool | None = None,
    range_mask: Magi2RangeUnionMask | None = None,
    block_mask: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlexAttention equivalent of MagiAttention's ``flex_flash_attn_func``.

    Returns ``(out, lse)`` shaped as the vendored operator returns them:
    ``out`` is ``[tokens, heads_q, head_dim]`` in the query dtype and ``lse`` is
    ``[tokens, heads_q]`` float32 natural-log softmax denominators. The refiner
    discards ``lse``; it is produced by FlexAttention itself rather than
    reconstructed, so it stays a real log-sum-exp.
    """
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise Magi2RefinerAttentionUnsupported(
            "MAGI-2 refiner attention expects unbatched [tokens, heads, head_dim] "
            "query, key, and value tensors; got "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
        )
    if tuple(key.shape[:2]) != tuple(value.shape[:2]):
        raise Magi2RefinerAttentionUnsupported(
            "MAGI-2 refiner attention key and value tensors must share their "
            "token and head axes."
        )
    heads_q = int(query.shape[1])
    heads_kv = int(key.shape[1])
    if heads_kv == 0 or heads_q % heads_kv:
        raise Magi2RefinerAttentionUnsupported(
            f"MAGI-2 refiner attention needs {heads_q} query heads to be a whole "
            f"multiple of {heads_kv} key/value heads."
        )
    if int(query.shape[2]) != int(key.shape[2]):
        raise Magi2RefinerAttentionUnsupported(
            "MAGI-2 refiner attention query and key head dimensions must match."
        )

    query_tokens = int(query.shape[0])
    key_tokens = int(key.shape[0])
    _validate_attn_type_map(attn_type_map, ranges=int(q_ranges.shape[0]))
    mask = range_mask
    if mask is None:
        mask = build_range_union_mask(
            q_ranges, k_ranges, query_tokens=query_tokens, key_tokens=key_tokens
        )
    elif int(mask.segment_of.numel()) != query_tokens or int(mask.key_tokens) != key_tokens:
        raise Magi2RefinerAttentionUnsupported(
            "Cached MAGI-2 range mask geometry does not match the attention input."
        )
    if max_seqlen_q is not None and int(max_seqlen_q) > 0:
        longest = int((q_ranges[:, 1] - q_ranges[:, 0]).max()) if q_ranges.numel() else 0
        if longest > int(max_seqlen_q):
            raise Magi2RefinerAttentionUnsupported(
                "MAGI-2 refiner attention was given max_seqlen_q="
                f"{int(max_seqlen_q)} but its longest query range spans {longest} "
                "tokens, so the declared scheduling envelope does not describe "
                "the ranges."
            )

    device = query.device
    if compile_kernel is None:
        compile_kernel = device.type == "cuda"
    if compile_block_mask is None:
        compile_block_mask = device.type == "cuda"

    # Construct sparse block metadata directly without evaluating a dense Q*K
    # boolean matrix.
    if block_mask is None:
        block_mask = mask.block_mask(query_tokens=query_tokens)
    flex = flex_attention_function(compiled=bool(compile_kernel))
    attended, log_sum_exp = flex(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        block_mask=block_mask,
        enable_gqa=heads_q != heads_kv,
        return_lse=True,
    )
    out = attended.squeeze(0).transpose(0, 1)
    lse = log_sum_exp.squeeze(0).transpose(0, 1).float()

    empty = ~mask.query_has_keys
    if bool(empty.any()):
        # A softmax over no keys is undefined; the vendored reference leaves
        # these rows at zero, and their true log-sum-exp is -inf.
        out = torch.where(empty.reshape(-1, 1, 1), torch.zeros_like(out), out)
        lse = torch.where(
            empty.reshape(-1, 1), torch.full_like(lse, float("-inf")), lse
        )
    return out.to(query.dtype), lse


@torch.compiler.disable
def _resolve_cached_range_mask(
    backend: "Magi2RefinerFlexAttentionBackend",
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    *,
    query_tokens: int,
    key_tokens: int,
) -> tuple[Magi2RangeUnionMask, Any]:
    """Reuse immutable geometry across refiner layers and denoise steps."""
    if (
        backend._cached_q_source is q_ranges
        and backend._cached_k_source is k_ranges
        and backend._cached_query_tokens == int(query_tokens)
        and backend._cached_key_tokens == int(key_tokens)
    ):
        assert backend._cached_range_mask is not None
        assert backend._cached_block_mask is not None
        return backend._cached_range_mask, backend._cached_block_mask
    q_cpu = q_ranges.detach().to(device="cpu", dtype=torch.int64)
    k_cpu = k_ranges.detach().to(device="cpu", dtype=torch.int64)
    cache_matches = (
        backend._cached_q_ranges is not None
        and backend._cached_k_ranges is not None
        and backend._cached_query_tokens == int(query_tokens)
        and backend._cached_key_tokens == int(key_tokens)
        and torch.equal(backend._cached_q_ranges, q_cpu)
        and torch.equal(backend._cached_k_ranges, k_cpu)
    )
    if cache_matches:
        assert backend._cached_range_mask is not None
        assert backend._cached_block_mask is not None
        return backend._cached_range_mask, backend._cached_block_mask

    mask = build_range_union_mask(
        q_cpu.to(device=q_ranges.device),
        k_cpu.to(device=k_ranges.device),
        query_tokens=int(query_tokens),
        key_tokens=int(key_tokens),
    )
    block_mask = mask.block_mask(query_tokens=int(query_tokens))
    object.__setattr__(backend, "_cached_q_ranges", q_cpu.clone())
    object.__setattr__(backend, "_cached_k_ranges", k_cpu.clone())
    object.__setattr__(backend, "_cached_q_source", q_ranges)
    object.__setattr__(backend, "_cached_k_source", k_ranges)
    object.__setattr__(backend, "_cached_query_tokens", int(query_tokens))
    object.__setattr__(backend, "_cached_key_tokens", int(key_tokens))
    object.__setattr__(backend, "_cached_range_mask", mask)
    object.__setattr__(backend, "_cached_block_mask", block_mask)
    return mask, block_mask


@dataclass
class Magi2RefinerFlexAttentionBackend:
    """Execution seam bound to every vendored refiner ``Attention`` module.

    ``compile_kernel`` defaults to the device decision: CUDA execution compiles,
    because only the compiled lowering is fused, while CPU execution stays
    eager so reference verification runs without an inductor toolchain.
    ``compile_block_mask`` remains an accepted compatibility field; compact
    block metadata is now constructed directly and has no dense mask to compile.
    """

    compile_kernel: bool | None = None
    compile_block_mask: bool | None = None
    _cached_q_ranges: torch.Tensor | None = field(default=None, init=False, repr=False)
    _cached_k_ranges: torch.Tensor | None = field(default=None, init=False, repr=False)
    _cached_q_source: torch.Tensor | None = field(default=None, init=False, repr=False)
    _cached_k_source: torch.Tensor | None = field(default=None, init=False, repr=False)
    _cached_query_tokens: int = field(default=-1, init=False, repr=False)
    _cached_key_tokens: int = field(default=-1, init=False, repr=False)
    _cached_range_mask: Magi2RangeUnionMask | None = field(
        default=None, init=False, repr=False
    )
    _cached_block_mask: Any | None = field(default=None, init=False, repr=False)

    def execute(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor | None,
        max_seqlen_q: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        range_mask, block_mask = _resolve_cached_range_mask(
            self,
            q_ranges,
            k_ranges,
            query_tokens=int(query.shape[0]),
            key_tokens=int(key.shape[0]),
        )
        return flex_range_attention(
            query,
            key,
            value,
            q_ranges,
            k_ranges,
            attn_type_map,
            max_seqlen_q,
            compile_kernel=self.compile_kernel,
            compile_block_mask=self.compile_block_mask,
            range_mask=range_mask,
            block_mask=block_mask,
        )


def _magi_attention_hopper_kernel_available() -> bool:
    """Whether the authors' single-GPU flex kernel can serve ``auto``.

    The kernel is Hopper-specific. Import reachability is checked lazily by the
    compatibility layer, including its functional-module fallback for installs
    that omit the distributed communication extension.
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability()
    except (AssertionError, RuntimeError):
        return False
    if major < 9:
        return False
    from mirai.vendors.magi2_preview.common.magi_compiler_compat import (
        magi_attention_flex_flash_attn_func,
    )

    return magi_attention_flex_flash_attn_func() is not None


def resolve_magi2_refiner_attention(
    backend: Any = MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT,
) -> Magi2RefinerFlexAttentionBackend | None:
    """Select the refiner attention path, or ``None`` to keep the vendored one.

    Precedence under ``auto``: a registered ``torch.ops.magi2`` operator wins,
    because a real MagiCompiler install owns its graph boundary and its dispatch
    semantics. Without one, the authors' single-GPU Hopper kernel wins when it
    imports; only then does Mirai use its portable FlexAttention path.
    ``native_flex`` selects the portable path unconditionally and
    ``vendor_eager`` never binds it.
    """
    from mirai.vendors.magi2_preview.common.magi_compiler_compat import (
        missing_magi2_custom_ops,
    )

    name = normalize_refiner_attention_backend(backend)
    if name == "vendor_eager":
        return None
    if name == "auto" and not missing_magi2_custom_ops((MAGI2_REFINER_FLEX_OP,)):
        return None
    if name == "auto" and _magi_attention_hopper_kernel_available():
        return None
    return Magi2RefinerFlexAttentionBackend()


def attach_refiner_attention_backend(
    transformer: Any, backend: Magi2RefinerFlexAttentionBackend | None
) -> int:
    """Bind (or clear) the attention execution seam on every refiner module."""
    from mirai.vendors.magi2_preview.model.magi2_refiner import Attention

    attached = 0
    for module in transformer.modules():
        if isinstance(module, Attention):
            module._mirai_refiner_attention_backend = backend
            attached += 1
    if backend is not None and attached == 0:
        raise Magi2RefinerAttentionUnsupported(
            "family_params.refiner_attention_backend selected a native refiner "
            "attention path but matched no MAGI-2 refiner attention layer."
        )
    return attached


def refiner_required_magi2_ops(
    arch_config: Any, backend: Magi2RefinerFlexAttentionBackend | None
) -> tuple[str, ...]:
    """Operators the configured refiner still dispatches through ``torch.ops.magi2``.

    Which operator a layer reaches is decided by ``local_attn_layers``: a local
    layer runs the flex range operator and every other layer runs the dense one.
    A bound native backend serves the flex operator itself, so it drops out of
    the precondition.
    """
    layers = int(getattr(arch_config, "num_layers", 0))
    local = {
        int(index)
        for index in getattr(arch_config, "local_attn_layers", ())
        if 0 <= int(index) < layers
    }
    required: list[str] = []
    if local and backend is None:
        required.append(MAGI2_REFINER_FLEX_OP)
    if len(local) < layers:
        required.append(MAGI2_REFINER_DENSE_OP)
    return tuple(required)


__all__ = [
    "MAGI2_REFINER_ATTENTION_BACKENDS",
    "MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT",
    "MAGI2_REFINER_ATTENTION_TYPE_FULL",
    "MAGI2_REFINER_DENSE_OP",
    "MAGI2_REFINER_FLEX_OP",
    "Magi2RangeUnionMask",
    "Magi2RefinerAttentionUnsupported",
    "Magi2RefinerFlexAttentionBackend",
    "attach_refiner_attention_backend",
    "build_range_union_mask",
    "flex_range_attention",
    "normalize_refiner_attention_backend",
    "refiner_required_magi2_ops",
    "resolve_magi2_refiner_attention",
    "validate_refiner_flex_support",
]
