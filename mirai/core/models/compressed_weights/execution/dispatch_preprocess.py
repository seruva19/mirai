# SPDX-License-Identifier: Apache-2.0
"""On-device MoE dispatch preprocessing (counts / offsets / stable order).

Single owner for the count/offset/stable-order/inverse-order preprocessing that
every direct-routed dispatch path shares. The body is fully device-resident: it
performs no host sync (no ``.cpu()`` / ``.tolist()`` / ``.item()``); ``top_k`` and
``total_slots`` are read from ``.shape`` (Python ints, no D2H). Callers that need
one scalar for a padded allocation take it explicitly (``plan.counts.max()``),
keeping the structurally-irreducible sync visible at the call site rather than
buried here.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from torch import Tensor
import torch


@dataclass(frozen=True)
class DispatchPlan:
    """Device-resident routing layout shared by every dispatch runner.

    ``offsets`` is the persistent-kernel contract: contiguous int32 inclusive
    cumsum with ``numel() == num_experts`` and ``offsets[-1] == total_slots``.
    ``starts`` is the exclusive-prefix segment start used by the padded gather.
    """

    counts: Tensor  # [E] int64 tokens routed to each expert
    starts: Tensor  # [E] int64 exclusive-prefix segment starts
    offsets: Tensor  # [E] int32 inclusive cumsum == persistent m_offsets
    sort_positions: Tensor  # [S] int64 stable argsort(flat_experts)
    sorted_token_indices: Tensor  # [S] int64 == sort_positions // top_k
    sorted_scores: Tensor  # [S] gate weights in sorted order
    inverse_order: Tensor  # [S] int64 inverse permutation of sort_positions
    total_slots: int
    top_k: int
    max_count_bound: int


def sonic_dispatch_preprocess_available() -> bool:
    """Whether the optional SonicMoE routing-metadata package is installed."""
    return importlib.util.find_spec("sonicmoe") is not None


def _build_sonic_dispatch_plan(
    top_indices: Tensor,
    top_scores: Tensor,
    num_experts: int,
) -> DispatchPlan:
    if top_indices.device.type != "cuda":
        raise RuntimeError("SonicMoE dispatch preprocessing requires CUDA tensors.")
    capability = torch.cuda.get_device_capability(top_indices.device)
    if capability[0] not in {9, 10, 12}:
        raise RuntimeError(
            "SonicMoE dispatch preprocessing requires SM90, SM100, or SM120; "
            f"found SM{capability[0]}{capability[1]}."
        )
    try:
        from sonicmoe.functional.triton_kernels import (
            general_routing_router_metadata_triton,
        )
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "memory.moe_dispatch_preprocess='sonic' requires an importable "
            "sonic-moe installation."
        ) from exc

    top_k = int(top_indices.shape[1])
    tokens = int(top_indices.shape[0])
    total_slots = tokens * top_k
    if total_slots >= 2**31:
        raise ValueError("SonicMoE dispatch plan exceeds the int32 slot range.")
    if int(num_experts) > 0xFFFF:
        raise ValueError("SonicMoE metadata supports at most 65535 experts.")
    device = top_indices.device
    token_indices = torch.arange(
        tokens,
        device=device,
        dtype=torch.int32,
    ).repeat_interleave(top_k)
    expert_indices = top_indices.reshape(-1).to(dtype=torch.int32)
    counts_i32 = torch.empty(num_experts, device=device, dtype=torch.int32)
    offsets_with_zero = torch.empty(
        num_experts + 1,
        device=device,
        dtype=torch.int32,
    )
    sorted_token_indices_i32 = torch.empty(
        total_slots,
        device=device,
        dtype=torch.int32,
    )
    sort_positions_i32 = torch.empty_like(sorted_token_indices_i32)
    inverse_order_i32 = torch.empty_like(sorted_token_indices_i32)
    token_offsets = torch.empty(tokens + 1, device=device, dtype=torch.int32)
    general_routing_router_metadata_triton(
        token_indices,
        expert_indices,
        tokens,
        int(num_experts),
        counts_i32,
        offsets_with_zero,
        sorted_token_indices_i32,
        sort_positions_i32,
        inverse_order_i32,
        token_offsets,
    )
    sort_positions = sort_positions_i32.to(torch.int64)
    counts = counts_i32.to(torch.int64)
    starts = offsets_with_zero[:-1].to(torch.int64)
    return DispatchPlan(
        counts=counts,
        starts=starts,
        offsets=offsets_with_zero[1:].contiguous(),
        sort_positions=sort_positions,
        sorted_token_indices=sorted_token_indices_i32.to(torch.int64),
        sorted_scores=top_scores.reshape(-1).index_select(0, sort_positions),
        inverse_order=inverse_order_i32.to(torch.int64),
        total_slots=total_slots,
        top_k=top_k,
        max_count_bound=tokens,
    )


def build_dispatch_plan(
    top_indices: Tensor,
    top_scores: Tensor,
    num_experts: int,
    *,
    backend: str = "device",
) -> DispatchPlan:
    """Build the shared dispatch plan with zero host syncs.

    ``top_indices`` / ``top_scores`` are ``[tokens, top_k]``. All derived tensors
    stay on ``top_indices.device``; the stable sort preserves token order within
    an expert segment so downstream reductions keep their exact semantics.
    """
    selected_backend = str(backend).strip().lower()
    if selected_backend == "sonic":
        return _build_sonic_dispatch_plan(top_indices, top_scores, num_experts)
    if selected_backend != "device":
        raise ValueError("dispatch-plan backend must be 'device' or 'sonic'.")

    top_k = int(top_indices.shape[1])
    flat_experts = top_indices.reshape(-1)
    total_slots = int(flat_experts.shape[0])
    assert total_slots < 2**31, "dispatch plan slot count exceeds int32 range"

    sort_positions = torch.argsort(flat_experts, stable=True)
    sorted_token_indices = torch.div(sort_positions, top_k, rounding_mode="floor")
    sorted_scores = top_scores.reshape(-1).index_select(0, sort_positions)

    counts = torch.bincount(flat_experts, minlength=int(num_experts))
    cumsum = torch.cumsum(counts, dim=0)
    offsets = cumsum.to(torch.int32).contiguous()
    starts = cumsum - counts

    inverse_order = torch.empty_like(sort_positions)
    inverse_order.scatter_(
        0,
        sort_positions,
        torch.arange(total_slots, device=sort_positions.device, dtype=sort_positions.dtype),
    )

    return DispatchPlan(
        counts=counts,
        starts=starts,
        offsets=offsets,
        sort_positions=sort_positions,
        sorted_token_indices=sorted_token_indices,
        sorted_scores=sorted_scores,
        inverse_order=inverse_order,
        total_slots=total_slots,
        top_k=top_k,
        max_count_bound=int(top_indices.shape[0]),
    )


def build_expert_choice_dispatch_plan(
    expert_token_indices: Tensor,
    expert_token_scores: Tensor,
    *,
    tokens_per_sample: int,
    num_physical_experts: int,
    logical_to_physical: tuple[int, ...] = (),
) -> DispatchPlan:
    """Build an expert-contiguous plan from ``[batch, expert, capacity]`` routes.

    The transformation is device-resident. Logical-to-physical alias metadata is
    static checkpoint state, so materializing its map on the route device is not
    route-dependent host preprocessing.
    """
    if expert_token_indices.shape != expert_token_scores.shape:
        raise ValueError("Expert-choice indices and scores must have matching shapes.")
    if expert_token_indices.ndim != 3:
        raise ValueError("Expert-choice routes must have shape [batch, expert, capacity].")
    batch, logical_experts, capacity = expert_token_indices.shape
    if expert_token_indices.device != expert_token_scores.device:
        raise ValueError("Expert-choice indices and scores must share a device.")
    if expert_token_indices.dtype.is_floating_point:
        raise TypeError("Expert-choice token indices must use an integer dtype.")
    if int(logical_experts) <= 0 or int(capacity) <= 0:
        raise ValueError("Expert-choice routes require experts and positive capacity.")
    if int(tokens_per_sample) <= 0:
        raise ValueError("tokens_per_sample must be > 0.")
    if int(num_physical_experts) <= 0:
        raise ValueError("num_physical_experts must be > 0.")
    if logical_to_physical and len(logical_to_physical) != int(logical_experts):
        raise ValueError("Logical expert alias map does not match Expert-Choice routes.")
    if logical_to_physical and any(
        value < 0 or value >= int(num_physical_experts)
        for value in logical_to_physical
    ):
        raise ValueError("Logical expert alias map contains an invalid physical id.")

    batch_offsets = (
        torch.arange(batch, device=expert_token_indices.device, dtype=torch.long)
        * int(tokens_per_sample)
    ).view(batch, 1, 1)
    token_indices = expert_token_indices.to(torch.long) + batch_offsets
    logical_ids = torch.arange(
        logical_experts, device=expert_token_indices.device, dtype=torch.long
    ).view(1, logical_experts, 1).expand(batch, logical_experts, capacity)
    if logical_to_physical:
        mapping = torch.tensor(
            logical_to_physical,
            device=expert_token_indices.device,
            dtype=torch.long,
        )
        physical_ids = mapping.index_select(0, logical_ids.reshape(-1))
        alias_counts = [
            logical_to_physical.count(index)
            for index in range(num_physical_experts)
        ]
        max_count_bound = int(batch * capacity * max(alias_counts, default=1))
    else:
        if int(logical_experts) != int(num_physical_experts):
            raise ValueError("Logical and physical expert counts differ without an alias map.")
        physical_ids = logical_ids.reshape(-1)
        max_count_bound = int(batch * capacity)

    flat_tokens = token_indices.reshape(-1)
    flat_scores = expert_token_scores.reshape(-1)
    total_slots = int(flat_tokens.shape[0])
    if total_slots >= 2**31:
        raise ValueError("Expert-choice dispatch plan exceeds the int32 slot range.")
    sort_positions = torch.argsort(physical_ids, stable=True)
    sorted_token_indices = flat_tokens.index_select(0, sort_positions)
    sorted_scores = flat_scores.index_select(0, sort_positions)
    counts = torch.bincount(physical_ids, minlength=int(num_physical_experts))
    cumsum = torch.cumsum(counts, dim=0)
    offsets = cumsum.to(torch.int32).contiguous()
    starts = cumsum - counts
    inverse_order = torch.empty_like(sort_positions)
    inverse_order.scatter_(
        0,
        sort_positions,
        torch.arange(
            total_slots,
            device=sort_positions.device,
            dtype=sort_positions.dtype,
        ),
    )
    return DispatchPlan(
        counts=counts,
        starts=starts,
        offsets=offsets,
        sort_positions=sort_positions,
        sorted_token_indices=sorted_token_indices,
        sorted_scores=sorted_scores,
        inverse_order=inverse_order,
        total_slots=total_slots,
        top_k=0,
        max_count_bound=max_count_bound,
    )
