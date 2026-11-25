"""Bounded timestep prediction reuse for one denoising run."""

from __future__ import annotations

import math

import torch


class DiffusionPredictionCache:
    """Reuse the latest prediction when inputs stay close to its anchor."""

    def __init__(self, *, threshold: float, max_reuse: int) -> None:
        if not math.isfinite(threshold) or threshold <= 0 or type(max_reuse) is not int or max_reuse < 1:
            raise ValueError("Diffusion caching requires a finite positive threshold and reuse limit.")
        self.threshold = threshold
        self.max_reuse = max_reuse
        self._inputs: tuple[torch.Tensor, ...] = ()
        self._prediction: tuple[torch.Tensor, ...] | None = None
        self._span = 0
        self.reused = 0
        self.computed = 0
        self.invalidated = 0

    def lookup(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...] | None:
        if self._prediction is None:
            return None
        compatible = len(inputs) == len(self._inputs) and all(
            (new.shape, new.dtype, new.device) == (old.shape, old.dtype, old.device)
            for new, old in zip(inputs, self._inputs)
        )
        if compatible and self._span < self.max_reuse:
            distances = []
            for new, old in zip(inputs, self._inputs, strict=True):
                if new.numel():
                    anchor = old.float()
                    distances.append((new.float() - anchor).abs().mean() / anchor.abs().mean().clamp_min(1e-6))
            if distances and bool(torch.stack(distances).amax() <= self.threshold):
                self._span += 1
                self.reused += 1
                return tuple(value.clone() for value in self._prediction)
        self.invalidated += 1
        self._inputs = ()
        self._prediction = None
        return None

    def store(self, inputs: tuple[torch.Tensor, ...], prediction: tuple[torch.Tensor, ...]) -> None:
        self._inputs = tuple(value.detach().clone() for value in inputs)
        self._prediction = tuple(value.detach().clone() for value in prediction)
        self._span = 0
        self.computed += 1

    def snapshot(self) -> dict[str, int]:
        return {"reused": self.reused, "computed": self.computed, "invalidated": self.invalidated}
