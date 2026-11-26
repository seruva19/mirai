"""Deterministic head-local router candidates for packed expert streaming."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mirai.core.moe.routing.subset import RouterSubsetPolicy


@dataclass(frozen=True)
class Magi2ExpertWorkingSet:
    fraction: float = 1.0
    hold_steps: int = 4
    anneal_steps: int = 0

    def __post_init__(self) -> None:
        if type(self.fraction) not in (int, float) or not 0 < self.fraction <= 1:
            raise ValueError("MAGI-2 expert_working_set_fraction must be in (0, 1].")
        if type(self.hold_steps) is not int or self.hold_steps < 1:
            raise ValueError("MAGI-2 expert_working_set_hold_steps must be a positive integer.")
        if type(self.anneal_steps) is not int or self.anneal_steps < 0:
            raise ValueError("MAGI-2 expert_working_set_anneal_steps must be a nonnegative integer.")

    def candidates(
        self, *, step: int, seed: int, layer: int, heads: int, experts: int, top_k: int,
    ) -> torch.Tensor | None:
        window = step // self.hold_steps
        policy = RouterSubsetPolicy(fraction=self.fraction, anneal_steps=self.anneal_steps)
        count = policy.subset_size(
            num_experts=experts, top_k=top_k, step=window * self.hold_steps,
        )
        if count == experts:
            return None
        generator = torch.Generator(device="cpu")
        generator.manual_seed((seed * 1_000_003 + layer * 9_176 + window * 61) % (1 << 64))
        return torch.stack([
            torch.randperm(experts, generator=generator)[:count].sort().values
            for _ in range(heads)
        ])
