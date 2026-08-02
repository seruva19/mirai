"""Route-to-sample gates for domain-selective expert adapters."""

from __future__ import annotations

from typing import Any

import torch


class RoutedAdapterGate:
    """Resolve a sample×expert mask onto arbitrary routed-token layouts."""

    def __init__(
        self,
        sample_expert_mask: Any,
        *,
        tokens_per_sample: int | None = None,
    ) -> None:
        if sample_expert_mask.ndim != 2:
            raise ValueError("sample_expert_mask must have shape [batch, experts].")
        if tokens_per_sample is not None and int(tokens_per_sample) <= 0:
            raise ValueError("tokens_per_sample must be > 0.")
        if int(sample_expert_mask.shape[0]) <= 0 or int(sample_expert_mask.shape[1]) <= 0:
            raise ValueError("sample_expert_mask dimensions must be positive.")
        self.sample_expert_mask = sample_expert_mask.bool()
        self.tokens_per_sample = (
            None if tokens_per_sample is None else int(tokens_per_sample)
        )

    def bind_tokens_per_sample(self, tokens_per_sample: int) -> None:
        if int(tokens_per_sample) <= 0:
            raise ValueError("tokens_per_sample must be > 0.")
        self.tokens_per_sample = int(tokens_per_sample)

    def _require_tokens_per_sample(self) -> int:
        if self.tokens_per_sample is None:
            raise RuntimeError("Route gate requires tokens_per_sample binding.")
        return int(self.tokens_per_sample)

    def resolve(self, token_indices: Any, expert_indices: Any) -> Any:
        if token_indices.shape != expert_indices.shape:
            raise ValueError("Route token and expert indices must have matching shapes.")
        if token_indices.dtype.is_floating_point or expert_indices.dtype.is_floating_point:
            raise TypeError("Route token and expert indices must use integer dtypes.")
        tokens = token_indices.to(
            device=self.sample_expert_mask.device, dtype=torch.long
        )
        experts = expert_indices.to(device=self.sample_expert_mask.device, dtype=tokens.dtype)
        sample_indices = tokens.div(
            self._require_tokens_per_sample(), rounding_mode="floor"
        )
        if bool((tokens < 0).any()) or bool((experts < 0).any()):
            raise ValueError("Route token and expert indices must be non-negative.")
        if bool((sample_indices >= self.sample_expert_mask.shape[0]).any()):
            raise ValueError("Route token index exceeds the configured batch.")
        if bool((experts >= self.sample_expert_mask.shape[1]).any()):
            raise ValueError("Route expert index exceeds the configured expert count.")
        return self.sample_expert_mask[sample_indices, experts]

    def resolve_sorted(self, token_indices: Any, counts: Any) -> Any:
        """Resolve a trusted device-resident sorted dispatch plan.

        ``counts`` comes from the validated ``DispatchPlan`` contract. Keep this
        path free of route-dependent host reads; ``resolve`` remains the checked
        boundary for arbitrary layouts.
        """
        if counts.ndim != 1:
            raise ValueError("counts must have shape [experts].")
        if counts.dtype not in (torch.int32, torch.int64):
            raise TypeError("counts must use an integer dtype.")
        if int(counts.shape[0]) != int(self.sample_expert_mask.shape[1]):
            raise ValueError("counts must contain one entry per configured expert.")
        expert_indices = torch.repeat_interleave(
            torch.arange(
                int(counts.shape[0]),
                device=counts.device,
                dtype=torch.long,
            ),
            counts.to(dtype=torch.long),
            output_size=int(token_indices.numel()),
        )
        tokens = token_indices.to(
            device=self.sample_expert_mask.device, dtype=torch.long
        )
        experts = expert_indices.to(device=self.sample_expert_mask.device)
        sample_indices = tokens.div(
            self._require_tokens_per_sample(), rounding_mode="floor"
        )
        return self.sample_expert_mask[sample_indices, experts]


__all__ = ["RoutedAdapterGate"]
