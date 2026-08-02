"""Similarity-preserving router balancing (SIMBAL).

SIMBAL softly orthogonalizes the expert rows of a router by minimizing the
entrywise L1 distance between its Gram matrix and identity.  This is the exact
per-router objective from Omi, Sen, and Farhadi, "Load Balancing Mixture of
Experts with Similarity Preserving Routers" (arXiv:2506.14038, Eq. 4 and
Appendix A.3).  It depends on router weights only, not tokens or batch size.

Paper: https://arxiv.org/abs/2506.14038
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch


def simbal_router_loss(router_weight: torch.Tensor) -> torch.Tensor:
    """Return ``||W W^T - I||_1`` for ``W=[experts, hidden]`` in FP32."""

    if router_weight.ndim != 2:
        raise ValueError("SIMBAL router weight must be [experts, hidden].")
    experts, hidden = (int(value) for value in router_weight.shape)
    if experts <= 0 or hidden <= 0:
        raise ValueError("SIMBAL router weight cannot be empty.")
    if hidden < experts:
        raise ValueError(
            "SIMBAL requires hidden_size >= num_experts so router rows can be "
            "orthonormal."
        )
    weight = router_weight.float()
    gram = weight @ weight.transpose(0, 1)
    identity = torch.eye(experts, device=weight.device, dtype=weight.dtype)
    return torch.linalg.vector_norm(gram - identity, ord=1)


@dataclass(frozen=True)
class SimBalSpec:
    """Model-independent coefficient for the paper's fixed L1 objective."""

    weight: float = 0.1

    def validate(self) -> "SimBalSpec":
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError("SIMBAL weight must be finite and > 0.")
        return self


class SimBalController:
    """Aggregate the paper-defined per-router terms without owning model modules."""

    def __init__(self, spec: SimBalSpec) -> None:
        self.spec = spec.validate()
        self.last_mean_loss: torch.Tensor | None = None
        self.last_max_loss: torch.Tensor | None = None
        self.last_router_count = 0

    def loss(self, router_weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not router_weights:
            raise ValueError("SIMBAL requires at least one trainable router weight.")
        terms = [
            simbal_router_loss(weight)
            for _, weight in sorted(router_weights.items())
        ]
        stacked = torch.stack(terms)
        mean = stacked.mean()
        self.last_mean_loss = mean.detach()
        self.last_max_loss = stacked.detach().max()
        self.last_router_count = len(terms)
        return mean * float(self.spec.weight)

    def diagnostics(self) -> dict[str, float | int]:
        output: dict[str, float | int] = {}
        if self.last_router_count:
            output["moe_simbal_router_count"] = int(self.last_router_count)
        if self.last_mean_loss is not None:
            output["moe_simbal_raw_mean"] = float(
                self.last_mean_loss.float().cpu().item()
            )
        if self.last_max_loss is not None:
            output["moe_simbal_raw_max"] = float(
                self.last_max_loss.float().cpu().item()
            )
        return output


__all__ = [
    "SimBalController",
    "SimBalSpec",
    "simbal_router_loss",
]
