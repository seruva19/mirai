"""Model-agnostic Mixture-of-Depths routing primitives.

The token-capacity rule follows Raposo et al. (2024), while received-attention
importance follows A-MoD Equation 4.  The exact attention path computes the
ordinary attention output and column-mean routing scores together, so enabling
the policy does not materialize an additional ``S x S`` attention matrix.

Sources:
- https://arxiv.org/abs/2404.02258
- https://arxiv.org/abs/2412.20875
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - config-only environments.
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MixtureOfDepthsSpec:
    """Static-capacity A-MoD schedule shared by model-family providers."""

    capacity_fraction: float = 0.5
    first_layer: int = 1
    layer_stride: int = 2
    attention_query_chunk_size: int = 128

    def validate(self) -> "MixtureOfDepthsSpec":
        if not 0.0 < float(self.capacity_fraction) < 1.0:
            raise ValueError("capacity_fraction must be strictly between 0 and 1.")
        if int(self.first_layer) < 1:
            raise ValueError(
                "first_layer must be >= 1 because A-MoD consumes the previous "
                "block's attention map."
            )
        if int(self.layer_stride) < 2:
            raise ValueError(
                "layer_stride must be >= 2 so every routed block follows a dense "
                "attention block."
            )
        if int(self.attention_query_chunk_size) < 1:
            raise ValueError("attention_query_chunk_size must be >= 1.")
        return self

    def routed_layers(self, num_layers: int) -> tuple[int, ...]:
        self.validate()
        return tuple(
            range(int(self.first_layer), int(num_layers), int(self.layer_stride))
        )


@dataclass(frozen=True)
class DepthTokenSelection:
    """Flat gathered-token layout with one fixed-capacity segment per sample."""

    flat_indices: Any
    cu_seqlens: Any
    selected_visual_tokens: tuple[int, ...]
    processed_tokens: tuple[int, ...]


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Mixture-of-Depths execution requires torch.")


def _normalize_key_mask(attention_mask: Any, *, batch: int, tokens: int) -> Any:
    if attention_mask is None or not bool(getattr(attention_mask, "dtype", None) == torch.bool):
        return None
    mask = attention_mask
    while int(mask.ndim) > 2 and int(mask.shape[1]) == 1:
        mask = mask.squeeze(1)
    if int(mask.ndim) == 1:
        mask = mask.unsqueeze(0)
    if tuple(mask.shape) != (batch, tokens):
        try:
            mask = mask.expand(batch, tokens)
        except RuntimeError as exc:
            raise ValueError(
                "Boolean attention mask cannot be reduced to a per-token mask."
            ) from exc
    return mask


def _attention_segment_with_received_scores(
    query: Any,
    key: Any,
    value: Any,
    *,
    attention_mask: Any | None,
    query_chunk_size: int,
) -> tuple[Any, Any]:
    batch, tokens, heads, head_dim = query.shape
    q = query.transpose(1, 2).float()
    k = key.transpose(1, 2).float()
    v = value.transpose(1, 2).float()
    key_mask = _normalize_key_mask(
        attention_mask,
        batch=int(batch),
        tokens=int(tokens),
    )
    received = q.new_zeros((batch, tokens))
    outputs = []
    scale = 1.0 / math.sqrt(float(head_dim))
    for start in range(0, int(tokens), int(query_chunk_size)):
        end = min(int(tokens), start + int(query_chunk_size))
        logits = torch.matmul(q[:, :, start:end], k.transpose(-1, -2)) * scale
        if attention_mask is not None:
            mask = attention_mask
            if mask.dtype == torch.bool:
                logits = logits.masked_fill(~mask, float("-inf"))
            else:
                logits = logits + mask.to(device=logits.device, dtype=logits.dtype)
        probabilities = torch.softmax(logits, dim=-1)
        outputs.append(torch.matmul(probabilities, v))
        if key_mask is None:
            received = received + probabilities.sum(dim=(1, 2))
        else:
            query_valid = key_mask[:, start:end]
            received = received + (
                probabilities
                * query_valid[:, None, :, None].to(probabilities.dtype)
            ).sum(dim=(1, 2))
    output = torch.cat(outputs, dim=2).transpose(1, 2).to(value.dtype)
    if key_mask is None:
        denominator = received.new_full((batch, 1), float(heads * tokens))
    else:
        denominator = (
            key_mask.sum(dim=1, keepdim=True).clamp_min(1).to(received.dtype)
            * float(heads)
        )
        received = received.masked_fill(~key_mask, 0.0)
    return output, received / denominator


def attention_with_received_scores(
    query: Any,
    key: Any,
    value: Any,
    *,
    attention_mask: Any | None = None,
    cu_seqlens: Any | None = None,
    query_chunk_size: int = 128,
) -> tuple[Any, Any]:
    """Return exact attention output and A-MoD received-attention scores.

    Inputs and output use ``(B, S, H, D)``.  Packed input is represented as
    ``B=1`` plus cumulative sequence lengths; each sample remains isolated.
    Softmax and accumulation run in FP32, and only a bounded query stripe is
    materialized at once.
    """

    _require_torch()
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes.")
    if int(query.ndim) != 4:
        raise ValueError("attention tensors must be shaped (B,S,H,D).")
    if int(query_chunk_size) < 1:
        raise ValueError("query_chunk_size must be >= 1.")
    if cu_seqlens is None:
        return _attention_segment_with_received_scores(
            query,
            key,
            value,
            attention_mask=attention_mask,
            query_chunk_size=int(query_chunk_size),
        )
    if int(query.shape[0]) != 1:
        raise ValueError("Packed attention scoring requires batch dimension 1.")
    boundaries = [int(item) for item in cu_seqlens.detach().cpu().tolist()]
    if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != int(query.shape[1]):
        raise ValueError("cu_seqlens must cover the complete packed token axis.")
    outputs = []
    scores = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            raise ValueError("Packed attention segments must be non-empty.")
        segment_output, segment_scores = _attention_segment_with_received_scores(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            attention_mask=None,
            query_chunk_size=int(query_chunk_size),
        )
        outputs.append(segment_output)
        scores.append(segment_scores)
    return torch.cat(outputs, dim=1), torch.cat(scores, dim=1)


def select_depth_tokens(
    scores: Any,
    *,
    eligible_mask: Any,
    valid_mask: Any,
    cu_seqlens: Any,
    capacity_fraction: float,
) -> DepthTokenSelection:
    """Select exact per-sample visual capacity and retain all context tokens."""

    _require_torch()
    flat_scores = scores.reshape(-1).float()
    eligible = eligible_mask.reshape(-1).bool()
    valid = valid_mask.reshape(-1).bool()
    if flat_scores.shape != eligible.shape or eligible.shape != valid.shape:
        raise ValueError("scores, eligible_mask, and valid_mask must align.")
    if bool((eligible & ~valid).any()):
        raise ValueError("Eligible depth-routing tokens must also be valid tokens.")
    boundaries = [int(item) for item in cu_seqlens.detach().cpu().tolist()]
    if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != int(flat_scores.numel()):
        raise ValueError("cu_seqlens must cover the complete token axis.")
    gathered = []
    selected_counts = []
    processed_counts = []
    output_boundaries = [0]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        local_eligible = torch.nonzero(eligible[start:end], as_tuple=False).flatten()
        if int(local_eligible.numel()) < 1:
            raise ValueError("Every sample must expose at least one eligible visual token.")
        capacity = max(
            1,
            int(math.floor(int(local_eligible.numel()) * float(capacity_fraction))),
        )
        ranked = torch.argsort(
            flat_scores[start:end].index_select(0, local_eligible),
            descending=True,
            stable=True,
        )
        selected = local_eligible.index_select(0, ranked[:capacity])
        context = torch.nonzero(
            valid[start:end] & ~eligible[start:end],
            as_tuple=False,
        ).flatten()
        local = torch.cat((selected, context)).sort().values + int(start)
        gathered.append(local)
        selected_counts.append(int(capacity))
        processed_counts.append(int(local.numel()))
        output_boundaries.append(output_boundaries[-1] + int(local.numel()))
    indices = torch.cat(gathered)
    return DepthTokenSelection(
        flat_indices=indices,
        cu_seqlens=torch.tensor(
            output_boundaries,
            device=cu_seqlens.device,
            dtype=torch.int32,
        ),
        selected_visual_tokens=tuple(selected_counts),
        processed_tokens=tuple(processed_counts),
    )


__all__ = [
    "attention_with_received_scores",
    "DepthTokenSelection",
    "MixtureOfDepthsSpec",
    "select_depth_tokens",
]
