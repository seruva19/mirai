"""Batched quantized-expert gather strategy with explicit fallback semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..quantization.quant import _is_contiguous_run

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BatchedExpertGatherStrategy:
    """Gather expert stacks when the source can do so without slower staging."""

    enabled: bool = False

    def gather(
        self,
        *,
        key: str,
        expert_indices: Sequence[int],
        device: Any,
        packed_source: Any | None,
        packed_tensor_names: Mapping[str, str],
        weight: Any | None,
        scale: Any | None,
    ) -> tuple[Any, Any] | None:
        if not self.enabled:
            return None
        if packed_source is not None and weight is None:
            cached_fn = getattr(packed_source, "cached_tensor", None)
            if not callable(cached_fn) or not _is_contiguous_run(expert_indices):
                return None
            q_cpu = cached_fn(packed_tensor_names[f"{key}_int8"])
            s_cpu = cached_fn(packed_tensor_names[f"{key}_scale"])
            if q_cpu is None or s_cpu is None:
                return None
            if not (
                bool(getattr(q_cpu, "is_pinned", lambda: False)())
                and bool(getattr(s_cpu, "is_pinned", lambda: False)())
            ):
                return None
            start = int(expert_indices[0])
            stop = start + len(expert_indices)
            return (
                q_cpu[start:stop].to(device=device),
                s_cpu[start:stop].to(device=device, dtype=torch.float32),
            )
        if weight is None or scale is None:
            return None
        index = torch.as_tensor(
            tuple(int(value) for value in expert_indices),
            device=weight.device,
            dtype=torch.long,
        )
        return (
            weight.index_select(0, index).to(device=device),
            scale.index_select(0, index).to(device=device, dtype=torch.float32),
        )
