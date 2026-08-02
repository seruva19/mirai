"""Compressed-weight execution adapter for canonical Expert-Choice routes."""

from __future__ import annotations

from typing import Any

import torch

from mirai.core.moe.runtime.specs import resolve_moe_dispatch_mode
from mirai.core.moe.runtime.gemm import resolve_moe_gemm_backend

from .dispatch_preprocess import build_expert_choice_dispatch_plan
from .linear import _persistent_dispatch_available, _triton_dispatch_available


def run_compressed_expert_choice(
    owner: Any,
    tokens: torch.Tensor,
    expert_token_scores: torch.Tensor,
    expert_token_indices: torch.Tensor,
    *,
    tokens_per_sample: int,
) -> torch.Tensor:
    """Execute Expert-Choice through one prebuilt device-resident route plan."""
    if owner.expert_weight_access not in {"active_dequant", "chunked_dequant"}:
        raise RuntimeError(
            "Expert-Choice compressed dispatch requires active_dequant or "
            "chunked_dequant expert access."
        )
    if int(tokens.shape[0]) != int(expert_token_indices.shape[0]) * int(
        tokens_per_sample
    ):
        raise ValueError("Expert-Choice batch/token dimensions do not match tokens.")
    if owner._prototype_calibration_observer is not None:
        raise RuntimeError("Prototype calibration does not support Expert-Choice dispatch.")
    if owner._whitening_calibration_observer is not None:
        raise RuntimeError(
            "Expert whitening calibration requires token-choice dispatch."
        )
    aliases = owner.logical_to_physical_map()
    plan = build_expert_choice_dispatch_plan(
        expert_token_indices,
        expert_token_scores,
        tokens_per_sample=int(tokens_per_sample),
        num_physical_experts=owner.num_experts,
        logical_to_physical=aliases,
    )
    observer = owner.active_routed_output_observer()
    if observer is not None:
        batch, logical_experts, capacity = expert_token_indices.shape
        expert_ids = torch.arange(
            logical_experts,
            device=expert_token_indices.device,
            dtype=torch.long,
        ).view(1, logical_experts, 1).expand(batch, logical_experts, capacity)
        if aliases:
            mapping = torch.as_tensor(
                aliases,
                device=expert_token_indices.device,
                dtype=torch.long,
            )
            expert_ids = mapping.index_select(0, expert_ids.reshape(-1)).reshape_as(
                expert_ids
            )
        observer.begin_routes(
            num_tokens=int(batch * logical_experts),
            top_k=int(capacity),
            device=tokens.device,
        )
        bind_routes = getattr(observer, "bind_routes", None)
        if callable(bind_routes):
            bind_routes(
                expert_ids.reshape(batch * logical_experts, capacity),
                expert_token_scores.reshape(batch * logical_experts, capacity),
            )

    def _execute_plan() -> torch.Tensor:
        output = torch.zeros(
            (tokens.shape[0], tokens.shape[1]),
            dtype=torch.float32,
            device=tokens.device,
        )
        chunk_size = max(1, int(owner.expert_dequant_chunk_size))
        forward_backend = resolve_moe_gemm_backend("forward", device=tokens.device)
        mode = resolve_moe_dispatch_mode() if forward_backend == "auto" else ""
        ragged_unsupported = forward_backend not in {"auto", "bmm"}
        if forward_backend == "auto" and mode == "triton":
            ragged_unsupported = _triton_dispatch_available(tokens.device)
        if forward_backend == "auto" and mode == "triton_persistent":
            ragged_unsupported = _persistent_dispatch_available(tokens.device)
        if owner.has_ragged_intermediate_widths() and ragged_unsupported:
            raise RuntimeError(
                "Ragged FlexMoE Expert-Choice requires vectorized BMM dispatch."
            )
        if forward_backend == "persistent" or (
            forward_backend == "auto"
            and mode == "triton_persistent"
            and _persistent_dispatch_available(tokens.device)
        ):
            from .persistent import run_persistent_dispatch_plan

            return run_persistent_dispatch_plan(
                owner, tokens, plan, chunk_size=chunk_size, output=output
            )
        if forward_backend == "torch_grouped":
            from .torch_grouped import run_torch_grouped_dispatch_plan

            return run_torch_grouped_dispatch_plan(owner, tokens, plan, output=output)

        if (
            forward_backend == "auto"
            and mode == "triton"
            and _triton_dispatch_available(tokens.device)
        ):

            def chunk_fn(padded, active, expert_ids, counts):
                return owner._run_expert_chunk_triton(
                    padded,
                    active,
                    expert_ids,
                    counts,
                    device=tokens.device,
                )

        else:

            def chunk_fn(padded, active, expert_ids, counts):
                del counts
                return owner._run_expert_chunk(
                    padded,
                    active,
                    expert_ids,
                    device=tokens.device,
                )

        return owner._run_direct_routed_device_chunks(
            plan,
            tokens,
            chunk_size=chunk_size,
            output=output,
            chunk_fn=chunk_fn,
        )

    try:
        result = _execute_plan()
    except BaseException:
        if observer is not None:
            observer.abort_capture()
        raise
    if observer is not None:
        observer.end_routes()
    return result
