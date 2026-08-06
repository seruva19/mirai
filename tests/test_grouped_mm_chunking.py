# SPDX-License-Identifier: Apache-2.0
"""Group-count segmentation for the ``torch_grouped`` GEMM primitive.

``torch grouped_mm`` refuses more than ``GROUPED_MM_MAX_GROUPS`` groups per
call, which a packed MoE layer that flattens (head, expert) onto one group axis
exceeds. These probes pin the splitting semantics (contiguous group ranges,
row ranges derived from the sorted layout, offsets rebased to segment-local
zero) and the numerical identity of the chunked and unchunked results.
"""

from __future__ import annotations

import pytest
import torch

from mirai.core.models.magi2_preview import grouped_moe
from mirai.core.moe.runtime.gemm import (
    GROUPED_MM_MAX_GROUPS,
    grouped_mm_segments,
    grouped_mm_stride_violation,
    grouped_mm_operand,
    run_grouped_mm,
)


def _reference_grouped_mm(lhs: torch.Tensor, weight: torch.Tensor, *, offs) -> torch.Tensor:
    """Jagged-offset grouped matmul as a plain per-group loop.

    Mirrors the ``torch grouped_mm`` contract (``offs`` are cumulative exclusive
    row ends, one per group) without its group-count cap, so it is a valid
    reference for any group count.
    """
    boundaries = [int(value) for value in offs.detach().to("cpu", torch.int64).tolist()]
    assert len(boundaries) == int(weight.shape[0])
    result = lhs.new_empty((int(lhs.shape[0]), int(weight.shape[-1])))
    start = 0
    for group, stop in enumerate(boundaries):
        if stop > start:
            result[start:stop] = lhs[start:stop] @ weight[group]
        start = stop
    return result


class _RecordingOp:
    """Reference executor that records the group count of every call."""

    def __init__(self) -> None:
        self.group_counts: list[int] = []
        self.offset_tensors: list[torch.Tensor] = []

    def __call__(self, lhs: torch.Tensor, weight: torch.Tensor, *, offs) -> torch.Tensor:
        self.group_counts.append(int(weight.shape[0]))
        self.offset_tensors.append(offs.detach().clone())
        return _reference_grouped_mm(lhs, weight, offs=offs)


def _problem(groups: int, *, k: int = 8, n: int = 8, seed: int = 0):
    """Random sorted-layout problem: per-group counts, offsets, rows, weights."""
    generator = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 3, (groups,), generator=generator)
    offsets = counts.cumsum(0).to(torch.int32)
    rows = int(offsets[-1])
    lhs = torch.randn(rows, k, generator=generator, dtype=torch.float32)
    weight = torch.randn(groups, k, n, generator=generator, dtype=torch.float32)
    return lhs, weight, offsets


# ---------------------------------------------------------------------------
# Pure splitting semantics
# ---------------------------------------------------------------------------


def test_group_limit_matches_the_documented_primitive_cap() -> None:
    assert GROUPED_MM_MAX_GROUPS == 1024


def test_production_group_count_splits_into_whole_segments() -> None:
    # 12 heads x 256 experts flattened onto one group axis.
    groups = 12 * 256
    boundaries = list(range(1, groups + 1))
    segments = grouped_mm_segments(boundaries)
    assert [segment.group_count for segment in segments] == [1024, 1024, 1024]
    assert [(segment.group_start, segment.group_stop) for segment in segments] == [
        (0, 1024),
        (1024, 2048),
        (2048, 3072),
    ]
    assert [(segment.row_start, segment.row_stop) for segment in segments] == [
        (0, 1024),
        (1024, 2048),
        (2048, 3072),
    ]


def test_group_count_at_the_limit_stays_one_segment() -> None:
    boundaries = list(range(1, GROUPED_MM_MAX_GROUPS + 1))
    segments = grouped_mm_segments(boundaries)
    assert len(segments) == 1
    assert segments[0].group_start == 0
    assert segments[0].group_stop == GROUPED_MM_MAX_GROUPS
    assert segments[0].row_start == 0


def test_one_group_over_the_limit_splits_into_two_segments() -> None:
    boundaries = list(range(1, GROUPED_MM_MAX_GROUPS + 2))
    segments = grouped_mm_segments(boundaries)
    assert [segment.group_count for segment in segments] == [GROUPED_MM_MAX_GROUPS, 1]
    assert segments[1].row_start == GROUPED_MM_MAX_GROUPS
    assert segments[1].row_stop == GROUPED_MM_MAX_GROUPS + 1


def test_empty_groups_spanning_a_split_point_keep_row_ranges_contiguous() -> None:
    # Every group around the split boundary is empty, so the second segment
    # starts and ends on the same row as the first one finished.
    boundaries = [0] * 4 + [3] * 8
    segments = grouped_mm_segments(boundaries, max_groups=4)
    assert [(s.group_start, s.group_stop) for s in segments] == [(0, 4), (4, 8), (8, 12)]
    assert [(s.row_start, s.row_stop) for s in segments] == [(0, 0), (0, 3), (3, 3)]
    assert [s.row_count for s in segments] == [0, 3, 0]
    # Row ranges tile the whole problem without gap or overlap.
    assert segments[0].row_start == 0
    assert segments[-1].row_stop == boundaries[-1]
    for previous, following in zip(segments, segments[1:]):
        assert previous.row_stop == following.row_start


def test_segments_reject_decreasing_offsets() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        grouped_mm_segments([0, 5, 3])


# ---------------------------------------------------------------------------
# Chunked execution
# ---------------------------------------------------------------------------


def test_group_count_within_the_limit_takes_the_unchunked_fast_path() -> None:
    lhs, weight, offsets = _problem(GROUPED_MM_MAX_GROUPS, seed=1)
    op = _RecordingOp()
    result = run_grouped_mm(op, lhs, weight, offsets)
    assert op.group_counts == [GROUPED_MM_MAX_GROUPS]
    # The offsets tensor is handed through untouched: no rebasing, no copy.
    assert torch.equal(op.offset_tensors[0], offsets)
    assert torch.equal(result, _reference_grouped_mm(lhs, weight, offs=offsets))


def test_production_group_count_matches_the_unchunked_reference_exactly() -> None:
    lhs, weight, offsets = _problem(12 * 256, seed=2)
    op = _RecordingOp()
    chunked = run_grouped_mm(op, lhs, weight, offsets)
    assert op.group_counts == [1024, 1024, 1024]
    assert all(count <= GROUPED_MM_MAX_GROUPS for count in op.group_counts)
    reference = _reference_grouped_mm(lhs, weight, offs=offsets)
    assert chunked.shape == reference.shape
    assert torch.equal(chunked, reference)


def test_every_segment_receives_offsets_rebased_to_segment_local_zero() -> None:
    lhs, weight, offsets = _problem(12 * 256, seed=3)
    op = _RecordingOp()
    run_grouped_mm(op, lhs, weight, offsets)
    segments = grouped_mm_segments(
        [int(value) for value in offsets.tolist()], max_groups=GROUPED_MM_MAX_GROUPS
    )
    assert len(op.offset_tensors) == len(segments)
    for segment, seen in zip(segments, op.offset_tensors):
        expected = offsets[segment.group_start : segment.group_stop] - segment.row_start
        assert torch.equal(seen, expected)
        assert int(seen[0]) >= 0
        assert int(seen[-1]) == segment.row_count
        assert seen.dtype == offsets.dtype


def test_transposed_weight_view_dx_role_chunks_and_matches_reference() -> None:
    lhs, weight, offsets = _problem(12 * 256, k=8, n=8, seed=4)
    # dX consumes dY [rows, n] against weight^T [groups, n, k], a non-contiguous
    # view whose stride pattern must survive group-axis slicing.
    grad_output = torch.randn(int(lhs.shape[0]), int(weight.shape[-1]))
    weight_t = weight.transpose(-2, -1)
    assert not weight_t.is_contiguous()
    op = _RecordingOp()
    chunked = run_grouped_mm(op, grad_output, weight_t, offsets)
    assert op.group_counts == [1024, 1024, 1024]
    assert torch.equal(chunked, _reference_grouped_mm(grad_output, weight_t, offs=offsets))


def test_group_axis_slicing_preserves_the_stride_precondition() -> None:
    # The alignment predicate reads per-operand strides, and slicing the leading
    # group axis changes neither the innermost strides nor the element size, so
    # a segment's verdict is identical to the whole call's -- for the stored
    # weight and for the transposed dX view alike.
    weight = torch.randn(3072, 16, 24, dtype=torch.bfloat16)
    for operand in (weight, weight.transpose(-2, -1)):
        whole = grouped_mm_operand(operand, label="weight")
        segment = grouped_mm_operand(operand[1024:2048], label="weight")
        assert whole.strides == segment.strides
        assert whole.element_size == segment.element_size
        assert grouped_mm_stride_violation(whole) == grouped_mm_stride_violation(segment)
        assert grouped_mm_stride_violation(segment) is None


def test_empty_trailing_segment_is_skipped_without_changing_the_result() -> None:
    groups = GROUPED_MM_MAX_GROUPS + 8
    counts = [1] * GROUPED_MM_MAX_GROUPS + [0] * 8
    offsets = torch.tensor(counts, dtype=torch.int32).cumsum(0).to(torch.int32)
    generator = torch.Generator().manual_seed(5)
    lhs = torch.randn(int(offsets[-1]), 8, generator=generator)
    weight = torch.randn(groups, 8, 8, generator=generator)
    op = _RecordingOp()
    chunked = run_grouped_mm(op, lhs, weight, offsets)
    # The all-empty second segment never reaches the op.
    assert op.group_counts == [GROUPED_MM_MAX_GROUPS]
    assert torch.equal(chunked, _reference_grouped_mm(lhs, weight, offs=offsets))


def test_offsets_must_carry_one_entry_per_group() -> None:
    lhs, weight, offsets = _problem(GROUPED_MM_MAX_GROUPS + 4, seed=6)
    with pytest.raises(ValueError, match="one cumulative row end per group"):
        run_grouped_mm(_RecordingOp(), lhs, weight, offsets[:-1])


# ---------------------------------------------------------------------------
# MAGI-2 grouped MoE call sites
# ---------------------------------------------------------------------------


def test_magi2_forward_and_dx_route_torch_grouped_through_the_chunker(monkeypatch) -> None:
    groups = 12 * 256
    lhs, weight, offsets = _problem(groups, k=8, n=8, seed=7)
    op = _RecordingOp()
    monkeypatch.setattr(grouped_moe, "grouped_mm_op", lambda: op)

    forward = grouped_moe._grouped_linear(lhs, weight, offsets, "torch_grouped")
    assert op.group_counts == [1024, 1024, 1024]
    assert torch.equal(
        forward, grouped_moe._grouped_linear(lhs, weight, offsets, "bmm")
    )

    op.group_counts.clear()
    grad_output = torch.randn(int(lhs.shape[0]), int(weight.shape[-1]))
    grad_x = grouped_moe._grouped_linear_dx(grad_output, weight, offsets, "torch_grouped")
    assert op.group_counts == [1024, 1024, 1024]
    assert torch.equal(
        grad_x, grouped_moe._grouped_linear_dx(grad_output, weight, offsets, "bmm")
    )
