"""Bounded expert-output orthogonality and router-variance objectives.

Reference: https://arxiv.org/abs/2505.22323, Section 3.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def deterministic_token_sample(values: Any, *, max_tokens: int) -> Any:
    """Select evenly spaced tokens without RNG or replacement."""
    if not torch.is_tensor(values) or values.ndim < 1:
        raise ValueError("specialization sampling requires a token-leading tensor.")
    limit = int(max_tokens)
    if limit <= 0:
        raise ValueError("specialization max_tokens must be positive.")
    tokens = int(values.shape[0])
    if tokens <= limit:
        return values
    indices = torch.div(
        torch.arange(limit, device=values.device) * tokens,
        limit,
        rounding_mode="floor",
    )
    return values.index_select(0, indices)


def active_expert_orthogonality_loss(
    selected_outputs: Any,
    *,
    max_tokens: int,
    epsilon: float = 1e-12,
) -> Any:
    """Mean squared projection between different active expert outputs.

    ``selected_outputs`` has shape ``[tokens, selected_experts, hidden]``.
    Only selected experts participate, matching the paper's indicator mask.
    """
    if not torch.is_tensor(selected_outputs) or selected_outputs.ndim != 3:
        raise ValueError("selected expert outputs must have shape [tokens, top_k, hidden].")
    if int(selected_outputs.shape[1]) < 2:
        raise ValueError("expert orthogonality requires top_k >= 2.")
    sampled = deterministic_token_sample(selected_outputs, max_tokens=max_tokens).float()
    gram = torch.matmul(sampled, sampled.transpose(-1, -2))
    norms = sampled.square().sum(dim=-1).clamp_min(float(epsilon))
    projection_norm_sq = gram.square() / norms.unsqueeze(-2)
    top_k = int(sampled.shape[1])
    off_diagonal = ~torch.eye(top_k, dtype=torch.bool, device=sampled.device)
    return projection_norm_sq[:, off_diagonal].mean()


def coactivated_intermediate_cosine_loss(
    selected_intermediates: Any,
    *,
    max_tokens: int,
    epsilon: float = 1e-12,
) -> Any:
    """Mean squared cosine for co-activated experts on identical tokens."""
    if not torch.is_tensor(selected_intermediates) or selected_intermediates.ndim != 3:
        raise ValueError(
            "expert intermediates must have shape [tokens, top_k, intermediate]."
        )
    if int(selected_intermediates.shape[1]) < 2:
        raise ValueError("intermediate specialization requires top_k >= 2.")
    sampled = deterministic_token_sample(
        selected_intermediates, max_tokens=max_tokens
    ).float()
    normalized = sampled / sampled.norm(dim=-1, keepdim=True).clamp_min(
        float(epsilon)
    )
    cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
    top_k = int(sampled.shape[1])
    off_diagonal = ~torch.eye(top_k, dtype=torch.bool, device=sampled.device)
    return cosine[:, off_diagonal].square().mean()


def router_score_variance_loss(scores: Any, *, max_tokens: int) -> Any:
    """Negative mean per-expert token variance from the paper's variance objective."""
    if not torch.is_tensor(scores) or scores.ndim != 2 or int(scores.shape[1]) < 2:
        raise ValueError("router scores must have shape [tokens, experts].")
    sampled = deterministic_token_sample(scores, max_tokens=max_tokens).float()
    centered = sampled - sampled.mean(dim=0, keepdim=True)
    return -centered.square().mean()


def cross_layer_topk_coupling_loss(
    layer_scores: list[Any],
    *,
    top_k: int,
    max_tokens: int,
    epsilon: float = 1e-20,
) -> Any:
    """Negative adjacent-layer joint top-k routing probability.

    Implements Eq. 7 of arXiv:2602.14159. Each input is ``[tokens, experts]``;
    scores are normalized onto the expert simplex before the active source and
    target top-k masses are multiplied. The same deterministic token rows are
    used in every layer so video resolution cannot grow retained work.
    """
    if len(layer_scores) < 2:
        raise ValueError("cross-layer coupling requires at least two MoE layers.")
    k = int(top_k)
    if k <= 0:
        raise ValueError("cross-layer coupling top_k must be positive.")
    expected_shape: tuple[int, int] | None = None
    top_masses: list[Any] = []
    for scores in layer_scores:
        if not torch.is_tensor(scores) or scores.ndim != 2:
            raise ValueError("cross-layer scores must have shape [tokens, experts].")
        shape = (int(scores.shape[0]), int(scores.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                "cross-layer coupling requires identical token/expert shapes."
            )
        if k > shape[1]:
            raise ValueError(
                f"cross-layer coupling top_k={k} exceeds experts={shape[1]}."
            )
        sampled = deterministic_token_sample(scores, max_tokens=max_tokens).float()
        probabilities = sampled / sampled.sum(dim=-1, keepdim=True).clamp_min(
            float(epsilon)
        )
        top_masses.append(torch.topk(probabilities, k=k, dim=-1).values.sum(dim=-1))
    pair_terms = [
        source_mass * target_mass
        for source_mass, target_mass in zip(top_masses, top_masses[1:])
    ]
    return -torch.stack(pair_terms).mean()


__all__ = [
    "active_expert_orthogonality_loss",
    "coactivated_intermediate_cosine_loss",
    "deterministic_token_sample",
    "cross_layer_topk_coupling_loss",
    "router_score_variance_loss",
]
