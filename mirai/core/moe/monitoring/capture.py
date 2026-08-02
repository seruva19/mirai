"""Bounded capture of already-computed routed expert tensors."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from mirai.core.moe.adaptation.specialization_loss import deterministic_token_sample


def sampled_outputs_from_sorted_dispatch(
    expert_output: Any,
    sorted_positions: Any,
    *,
    num_tokens: int,
    top_k: int,
    max_tokens: int,
) -> Any:
    if expert_output.ndim != 2 or sorted_positions.ndim != 1:
        raise ValueError("sorted dispatch capture requires rank-2 outputs and positions.")
    if int(expert_output.shape[0]) != int(sorted_positions.numel()):
        raise ValueError("sorted dispatch output and position counts do not match.")
    if int(top_k) < 2 or int(num_tokens) <= 0:
        raise ValueError("routed capture requires num_tokens > 0 and top_k >= 2.")
    inverse = torch.full(
        (int(num_tokens) * int(top_k),),
        -1,
        device=sorted_positions.device,
        dtype=torch.long,
    )
    inverse[sorted_positions.long()] = torch.arange(
        sorted_positions.numel(), device=sorted_positions.device
    )
    token_ids = deterministic_token_sample(
        torch.arange(int(num_tokens), device=sorted_positions.device),
        max_tokens=max_tokens,
    )
    flat_positions = token_ids[:, None] * int(top_k) + torch.arange(
        int(top_k), device=sorted_positions.device
    )[None, :]
    output_positions = inverse[flat_positions]
    complete = (output_positions >= 0).all(dim=1)
    if not bool(complete.any()):
        raise ValueError("routed capture found no sampled token with all top-k routes.")
    return expert_output[output_positions[complete]].reshape(
        -1, int(top_k), int(expert_output.shape[-1])
    )


class RoutedExpertTensorCapture:
    """Forward-scoped bounded sink for already-computed routed expert tensors."""

    def __init__(self, *, max_tokens: int, loss_fn: Any) -> None:
        if int(max_tokens) <= 0:
            raise ValueError("specialization max_tokens must be positive.")
        self.max_tokens = int(max_tokens)
        self._loss_fn = loss_fn
        self._losses: list[Any] = []
        self.abort_capture()

    @property
    def is_enabled(self) -> bool:
        return True

    @contextmanager
    def suspended(self):
        try:
            yield
        finally:
            self.take_losses()

    def begin_routes(self, *, num_tokens: int, top_k: int, device: Any) -> None:
        if self._capture_shape is not None:
            raise RuntimeError("routed-tensor capture is already active.")
        if int(num_tokens) <= 0 or int(top_k) < 2:
            raise ValueError("routed capture requires num_tokens > 0 and top_k >= 2.")
        token_ids = deterministic_token_sample(
            torch.arange(int(num_tokens), device=device),
            max_tokens=self.max_tokens,
        )
        self._target_positions = (
            token_ids[:, None] * int(top_k)
            + torch.arange(int(top_k), device=device)[None, :]
        ).reshape(-1)
        self._capture_shape = (int(num_tokens), int(top_k))

    def capture_routes(self, routed_tensor: Any, route_positions: Any) -> None:
        if self._capture_shape is None or self._target_positions is None:
            raise RuntimeError("routed-tensor capture was not started.")
        if routed_tensor.ndim != 2 or route_positions.ndim != 1:
            raise ValueError("routed tensors require rank-2 values and positions.")
        if int(routed_tensor.shape[0]) != int(route_positions.numel()):
            raise ValueError("routed tensor and position counts differ.")
        matches = route_positions.long()[:, None] == self._target_positions[None, :]
        rows = torch.nonzero(matches.any(dim=1), as_tuple=False).reshape(-1)
        if rows.numel() > 0:
            self._route_outputs.append(routed_tensor.index_select(0, rows))
            self._route_positions.append(route_positions.long().index_select(0, rows))

    def begin_sorted(
        self,
        sorted_positions: Any,
        *,
        num_tokens: int,
        top_k: int,
        device: Any,
    ) -> None:
        if not torch.is_tensor(sorted_positions) or sorted_positions.ndim != 1:
            raise ValueError("sorted route positions must be rank one.")
        self.begin_routes(num_tokens=num_tokens, top_k=top_k, device=device)
        self._sorted_positions = sorted_positions

    def capture_sorted_chunk(self, routed_tensor: Any) -> None:
        if self._sorted_positions is None:
            raise RuntimeError("sorted routed-tensor capture was not started.")
        count = int(routed_tensor.shape[0])
        positions = self._sorted_positions[
            self._sorted_cursor : self._sorted_cursor + count
        ]
        if int(positions.numel()) != count:
            raise ValueError("sorted routed-tensor chunks exceed the route plan.")
        self._sorted_cursor += count
        self.capture_routes(routed_tensor, positions)

    def end_sorted(self) -> None:
        if self._sorted_positions is None:
            raise RuntimeError("sorted routed-tensor capture was not started.")
        if self._sorted_cursor != int(self._sorted_positions.numel()):
            self.abort_capture()
            raise ValueError("sorted routed-tensor chunks do not cover the route plan.")
        self._sorted_positions = None
        self._sorted_cursor = 0
        self.end_routes()

    def abort_capture(self) -> None:
        self._route_outputs: list[Any] = []
        self._route_positions: list[Any] = []
        self._capture_shape: tuple[int, int] | None = None
        self._target_positions: Any | None = None
        self._sorted_positions: Any | None = None
        self._sorted_cursor = 0

    def end_routes(self) -> None:
        if self._capture_shape is None:
            raise RuntimeError("routed-tensor capture was not started.")
        num_tokens, top_k = self._capture_shape
        try:
            if not self._route_outputs:
                raise ValueError("routed capture found no sampled routes.")
            selected = sampled_outputs_from_sorted_dispatch(
                torch.cat(self._route_outputs, dim=0),
                torch.cat(self._route_positions, dim=0),
                num_tokens=num_tokens,
                top_k=top_k,
                max_tokens=self.max_tokens,
            )
            self._losses.append(self._loss_fn(selected, max_tokens=self.max_tokens))
        finally:
            self.abort_capture()

    def capture_sorted(
        self,
        routed_tensor: Any,
        sorted_positions: Any,
        *,
        num_tokens: int,
        top_k: int,
    ) -> None:
        selected = sampled_outputs_from_sorted_dispatch(
            routed_tensor,
            sorted_positions,
            num_tokens=num_tokens,
            top_k=top_k,
            max_tokens=self.max_tokens,
        )
        self._losses.append(self._loss_fn(selected, max_tokens=self.max_tokens))

    def take_losses(self) -> list[Any]:
        losses, self._losses = self._losses, []
        return losses


__all__ = ["RoutedExpertTensorCapture", "sampled_outputs_from_sorted_dispatch"]
