"""Model-agnostic auxiliary objectives for sparse token-choice routers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TokenChoiceRouterState:
    top_indices: Any
    logits: Any
    probabilities: Any
    num_experts: int


def token_choice_auxiliary_losses(
    router_states: Iterable[TokenChoiceRouterState],
    *,
    load_balance_weight: float,
    z_loss_weight: float,
) -> dict[str, Any]:
    """Aggregate Switch balance and ST-MoE z-loss across routed layers."""
    balance_terms: list[Any] = []
    z_terms: list[Any] = []
    for state in router_states:
        if state.top_indices is None or state.logits is None or state.probabilities is None:
            continue
        experts = int(state.num_experts)
        probabilities = state.probabilities.float()
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-20)
        selected = F.one_hot(
            state.top_indices.reshape(-1), num_classes=experts
        ).float()
        expert_fraction = selected.mean(dim=0)
        mean_probability = probabilities.mean(dim=0)
        balance_terms.append(experts * torch.sum(expert_fraction * mean_probability))
        z_terms.append(
            torch.logsumexp(state.logits.float(), dim=-1).square().mean()
        )
    losses: dict[str, Any] = {}
    if balance_terms and float(load_balance_weight) != 0.0:
        losses["moe_load_balance"] = torch.stack(balance_terms).mean() * float(
            load_balance_weight
        )
    if z_terms and float(z_loss_weight) != 0.0:
        losses["moe_router_z"] = torch.stack(z_terms).mean() * float(z_loss_weight)
    return losses


def router_similarity_loss(
    top_indices: Any,
    logits: Any,
    *,
    num_experts: int,
) -> Any:
    """Expert Race router-similarity objective (Yuan et al., 2025, eqs. 9-11).

    The hard co-selection matrix supplies detached utilization weights while
    the probability correlation matrix carries router gradients. Diagonal and
    off-diagonal weights are normalized independently exactly as defined in the
    paper, so individual balance cannot hide collapsed expert combinations.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("router_similarity_loss requires torch.")
    experts = int(num_experts)
    if experts < 2:
        raise ValueError("router similarity requires at least two experts.")
    if logits.ndim != 2 or int(logits.shape[-1]) != experts:
        raise ValueError("router logits must have shape [tokens, num_experts].")
    if top_indices.ndim != 2 or int(top_indices.shape[0]) != int(logits.shape[0]):
        raise ValueError("top_indices must have shape [tokens, selected_experts].")
    if int(top_indices.shape[1]) < 1:
        raise ValueError("router similarity requires at least one selected expert.")
    if int(top_indices.min().item()) < 0 or int(top_indices.max().item()) >= experts:
        raise ValueError("top_indices contains an out-of-range expert id.")

    tokens = int(logits.shape[0])
    if tokens == 0:
        raise ValueError("router similarity requires at least one token.")
    probabilities = torch.softmax(logits.float(), dim=-1)
    indicator = F.one_hot(top_indices, num_classes=experts).amax(dim=1).float()
    selection_correlation = indicator.transpose(0, 1) @ indicator
    probability_correlation = probabilities.transpose(0, 1) @ probabilities

    diagonal_mask = torch.eye(
        experts, dtype=torch.bool, device=probability_correlation.device
    )
    diagonal = torch.where(
        diagonal_mask, selection_correlation, torch.zeros_like(selection_correlation)
    )
    off_diagonal = torch.where(
        diagonal_mask, torch.zeros_like(selection_correlation), selection_correlation
    )
    diagonal_weights = (
        diagonal
        / diagonal.sum().clamp_min(1.0)
        * float(experts)
    )
    off_diagonal_weights = (
        off_diagonal
        / off_diagonal.sum().clamp_min(1.0)
        * float(experts * experts - experts)
    )
    weights = (diagonal_weights + off_diagonal_weights).detach()
    return (weights * probability_correlation).sum() / float(tokens)


def expert_combination_usage(
    top_indices: Any,
    *,
    num_experts: int,
    minimum_fraction: float = 0.05,
) -> dict[str, Any]:
    """Summarize unordered expert-pair activation without quadratic token state."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("expert_combination_usage requires torch.")
    experts = int(num_experts)
    threshold = float(minimum_fraction)
    if experts < 2 or not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid expert count or combination threshold.")
    if top_indices.ndim != 2 or int(top_indices.shape[1]) < 2:
        return {
            "active_pairs": 0,
            "possible_pairs": experts * (experts - 1) // 2,
            "usage_ratio": 0.0,
        }
    pairs = torch.combinations(torch.arange(experts, device=top_indices.device), r=2)
    selected = F.one_hot(top_indices, num_classes=experts).amax(dim=1).bool()
    pair_fraction = (
        selected[:, pairs[:, 0]] & selected[:, pairs[:, 1]]
    ).float().mean(dim=0)
    active = int((pair_fraction >= threshold).sum().item())
    possible = int(pairs.shape[0])
    return {
        "active_pairs": active,
        "possible_pairs": possible,
        "usage_ratio": float(active / possible) if possible else 0.0,
    }
