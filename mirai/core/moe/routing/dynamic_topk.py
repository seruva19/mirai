"""Compute-budgeted dynamic route cardinality for token-choice MoE.

The policy adapts the number of active routes to token uncertainty while keeping
the router's fixed-width candidate and dispatch tensors unchanged.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class BudgetedDynamicTopK:
    """Allocate a bounded batch route budget to the least-confident tokens."""

    def __init__(self, *, min_k: int, average_k: float) -> None:
        if int(min_k) < 1:
            raise ValueError("Dynamic top-k min_k must be positive.")
        if float(average_k) < float(min_k):
            raise ValueError("Dynamic top-k average_k must be >= min_k.")
        self.min_k = int(min_k)
        self.average_k = float(average_k)

    def regularize(
        self,
        layer_name: str,
        top_indices: Any,
        top_scores: Any,
        *,
        training: bool,
    ) -> Any:
        _ = layer_name, top_indices
        if not training:
            return top_scores
        if not torch.is_tensor(top_scores) or top_scores.ndim != 2:
            raise ValueError("Dynamic top-k scores must have shape [tokens, max_k].")
        tokens, max_k = (int(value) for value in top_scores.shape)
        if self.min_k > max_k or self.average_k > float(max_k):
            raise ValueError(
                "Dynamic top-k requires min_k <= average_k <= router max_k."
            )
        if tokens == 0 or self.min_k == max_k:
            return top_scores

        detached = top_scores.detach().float().abs()
        order = detached.argsort(dim=-1, descending=True, stable=True)
        ranked = detached.gather(1, order)
        probability = ranked / ranked.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(ranked.dtype).tiny
        )
        uncertainty = 1.0 - probability[:, 0]

        active_ranked = torch.zeros_like(ranked, dtype=torch.bool)
        active_ranked[:, : self.min_k] = True
        extra_slots = round(tokens * (self.average_k - float(self.min_k)))
        extra_slots = min(extra_slots, tokens * (max_k - self.min_k))
        if extra_slots:
            rank = torch.arange(
                max_k - self.min_k,
                device=top_scores.device,
                dtype=uncertainty.dtype,
            )
            # Earlier extra ranks win before later ranks; uncertainty decides
            # which tokens receive a route within each rank.
            priority = uncertainty[:, None] - rank[None, :] * 2.0
            chosen = priority.reshape(-1).topk(
                k=extra_slots, largest=True, sorted=False
            ).indices
            extra_active = active_ranked[:, self.min_k :].clone().reshape(-1)
            extra_active.scatter_(0, chosen, True)
            active_ranked[:, self.min_k :] = extra_active.reshape(
                tokens, max_k - self.min_k
            )

        active = torch.zeros_like(active_ranked)
        active.scatter_(1, order, active_ranked)
        original_mass = top_scores.sum(dim=-1, keepdim=True)
        retained = top_scores * active.to(dtype=top_scores.dtype)
        retained_mass = retained.sum(dim=-1, keepdim=True)
        scale = torch.where(
            retained_mass != 0,
            original_mass
            / retained_mass.clamp_min(torch.finfo(top_scores.dtype).tiny),
            torch.ones_like(retained_mass),
        )
        return retained * scale


__all__ = ["BudgetedDynamicTopK"]
