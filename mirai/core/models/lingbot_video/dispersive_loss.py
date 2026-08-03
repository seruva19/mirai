"""LingBot binding for model-agnostic Dispersive Loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from mirai.core.training.policies.dispersive_loss import (
    DispersiveLossController,
)


@dataclass
class LingBotDispersiveLossRuntime:
    """Extract fixed-size video tokens from LingBot's packed joint sequence."""

    controller: DispersiveLossController
    batch_size: int = 0
    video_tokens: int = 0
    text_lengths: tuple[int, ...] = ()
    packed_batch: bool = False

    def bind_depth(self, depth: int) -> int:
        return self.controller.bind_depth(depth)

    def is_layer_enabled(self, layer_index: int) -> bool:
        return self.controller.is_layer_enabled(layer_index)

    def begin_forward(
        self,
        *,
        batch_size: int,
        video_tokens: int,
        text_lengths: Sequence[int],
        packed_batch: bool,
    ) -> None:
        lengths = tuple(int(value) for value in text_lengths)
        if int(batch_size) < 2:
            raise ValueError(
                "Dispersive Loss requires a physical batch of at least 2."
            )
        if int(video_tokens) <= 0:
            raise ValueError("Dispersive Loss requires non-empty video tokens.")
        if len(lengths) != int(batch_size) or any(value < 0 for value in lengths):
            raise ValueError("LingBot text lengths do not match the batch.")
        self.batch_size = int(batch_size)
        self.video_tokens = int(video_tokens)
        self.text_lengths = lengths
        self.packed_batch = bool(packed_batch)

    def _video_representations(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.batch_size < 2 or self.video_tokens <= 0:
            raise RuntimeError("Dispersive Loss forward context is not initialized.")
        if hidden_states.ndim != 3:
            raise ValueError("LingBot hidden states must be [batch, tokens, hidden].")
        if not self.packed_batch:
            if int(hidden_states.shape[0]) != self.batch_size:
                raise ValueError("LingBot unpacked hidden-state batch is inconsistent.")
            return hidden_states[:, : self.video_tokens, :]
        if int(hidden_states.shape[0]) != 1:
            raise ValueError("LingBot packed hidden states must use a singleton batch axis.")
        cursor = 0
        samples: list[torch.Tensor] = []
        for text_length in self.text_lengths:
            end = cursor + self.video_tokens
            samples.append(hidden_states[:, cursor:end, :])
            cursor = end + int(text_length)
        if cursor > int(hidden_states.shape[1]):
            raise ValueError("LingBot packed hidden states are shorter than their layout.")
        return torch.cat(samples, dim=0)

    def loss_for_hidden_states(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.is_layer_enabled(layer_index):
            raise ValueError("Dispersive Loss was requested for an unselected layer.")
        return self.controller.loss(self._video_representations(hidden_states))

    def auxiliary_losses(self, model: Any) -> dict[str, Any]:
        terms = tuple(getattr(model, "_mirai_dispersive_loss_terms", ()) or ())
        if not terms:
            return {}
        return {"dispersive_loss": torch.stack(tuple(terms)).mean()}

    def diagnostics(self) -> dict[str, float | int]:
        return self.controller.diagnostics()


def configure_lingbot_dispersive_loss(
    pipeline: Any,
    policy: DispersiveLossController,
) -> None:
    if not isinstance(policy, DispersiveLossController):
        raise TypeError("dispersive_loss requires DispersiveLossController.")
    runtime = LingBotDispersiveLossRuntime(policy)
    runtime.bind_depth(len(pipeline.transformer.blocks))
    setter = getattr(pipeline.transformer, "set_dispersive_loss_runtime", None)
    if not callable(setter):
        raise ValueError("LingBot transformer does not expose representation capture.")
    setter(runtime)
    pipeline._dispersive_loss_runtime = runtime


__all__ = [
    "configure_lingbot_dispersive_loss",
    "LingBotDispersiveLossRuntime",
]
