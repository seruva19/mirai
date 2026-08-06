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

Mask construction. The union is expressed as a lookup rather than as a loop.
Splitting the query axis at every ``q_ranges`` boundary yields elementary
segments, each of which every row either fully covers or is disjoint from, so a
segment's allowed key set is fixed. Marking ``+1`` at ``k_start`` and ``-1`` at
``k_end`` per covering row and taking the running sum gives, per segment, a
boolean row over the key axis that is true exactly on the union — overlapping
ranges raise the count above one and change nothing else. ``mask_mod`` is then
two tensor indexings: segment of the query position, then key position in that
segment's row. The construction costs ``O(segments * key_tokens)`` and segments
are bounded by ``2 * N - 1``.

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

from dataclasses import dataclass
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

    ``allowed`` carries one boolean row per elementary query segment plus a
    final all-false row, which is the segment index assigned to any query
    position no ``q_range`` covers. ``segment_of`` maps a query position to its
    row, so membership is one lookup per ``(query, key)`` pair.
    """

    allowed: torch.Tensor
    segment_of: torch.Tensor
    segment_has_keys: torch.Tensor

    @property
    def query_has_keys(self) -> torch.Tensor:
        """Per query position: whether any key is reachable from it at all."""
        return self.segment_has_keys[self.segment_of]

    def mask_mod(self) -> Callable[..., torch.Tensor]:
        allowed = self.allowed
        segment_of = self.segment_of

        def range_union_mask(
            batch: torch.Tensor,
            head: torch.Tensor,
            query_index: torch.Tensor,
            key_index: torch.Tensor,
        ) -> torch.Tensor:
            return allowed[segment_of[query_index], key_index]

        return range_union_mask


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
    boundaries = torch.unique(q.reshape(-1))
    segments = max(int(boundaries.numel()) - 1, 0)

    # One extra row, kept all-false, is the segment a query position outside
    # every q_range is mapped to.
    allowed = torch.zeros(
        (segments + 1, int(key_tokens)), dtype=torch.bool, device=device
    )
    if segments > 0:
        segment_start = boundaries[:-1]
        segment_end = boundaries[1:]
        # Segments are elementary, so a row either covers one entirely or not
        # at all; containment of the segment bounds is therefore exact.
        covers = (q[:, 0].unsqueeze(1) <= segment_start.unsqueeze(0)) & (
            segment_end.unsqueeze(0) <= q[:, 1].unsqueeze(1)
        )
        rows, covered_segments = covers.nonzero(as_tuple=True)
        if int(rows.numel()) > 0:
            # Running sum of range openings minus closings: positive exactly on
            # the union, so a key reachable from several rows is counted once.
            crossings = torch.zeros(
                (segments, int(key_tokens) + 1), dtype=torch.int32, device=device
            )
            unit = torch.ones(int(rows.numel()), dtype=torch.int32, device=device)
            crossings.index_put_(
                (covered_segments, k[rows, 0]), unit, accumulate=True
            )
            crossings.index_put_(
                (covered_segments, k[rows, 1]), -unit, accumulate=True
            )
            allowed[:segments] = crossings.cumsum(dim=1)[:, : int(key_tokens)] > 0

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
        allowed=allowed,
        segment_of=segment_of,
        segment_has_keys=allowed.any(dim=1),
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
    mask = build_range_union_mask(
        q_ranges, k_ranges, query_tokens=query_tokens, key_tokens=key_tokens
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

    from torch.nn.attention.flex_attention import create_block_mask

    block_mask = create_block_mask(
        mask.mask_mod(),
        None,
        None,
        query_tokens,
        key_tokens,
        device=str(device),
        _compile=bool(compile_block_mask),
    )
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


@dataclass(frozen=True)
class Magi2RefinerFlexAttentionBackend:
    """Execution seam bound to every vendored refiner ``Attention`` module.

    ``compile_kernel`` and ``compile_block_mask`` default to the device
    decision: CUDA execution compiles, because only the compiled lowering is
    fused, while CPU execution stays eager so reference verification runs
    without an inductor toolchain.
    """

    compile_kernel: bool | None = None
    compile_block_mask: bool | None = None

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
        )


def resolve_magi2_refiner_attention(
    backend: Any = MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT,
) -> Magi2RefinerFlexAttentionBackend | None:
    """Select the refiner attention path, or ``None`` to keep the vendored one.

    Precedence under ``auto``: a registered ``torch.ops.magi2`` operator wins,
    because a real MagiCompiler install owns its graph boundary and its dispatch
    semantics; with the namespace empty this native path is selected.
    ``native_flex`` selects it unconditionally and ``vendor_eager`` never does.
    """
    from mirai.vendors.magi2_preview.common.magi_compiler_compat import (
        missing_magi2_custom_ops,
    )

    name = normalize_refiner_attention_backend(backend)
    if name == "vendor_eager":
        return None
    if name == "auto" and not missing_magi2_custom_ops((MAGI2_REFINER_FLEX_OP,)):
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
