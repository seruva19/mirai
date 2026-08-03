"""Reference and artifact contract for exact 2:4 frozen-expert sparsity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.moe.runtime.specs import CANONICAL_PACKED_EXPERT_MLP_SPEC

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass(frozen=True)
class StructuredSparse24:
    """Dense values plus a packed four-bit mask for a validated 2:4 tensor."""

    values: Any
    mask: Any
    original_shape: tuple[int, ...]

    def dense(self) -> Any:
        return self.values * self.mask.to(dtype=self.values.dtype)


def prune_to_2_4(weight: Any) -> StructuredSparse24:
    """Keep the two largest magnitudes in every contiguous group of four."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("2:4 sparsity requires torch.")
    if weight.ndim < 2 or int(weight.shape[-1]) % 4:
        raise ValueError("2:4 sparsity requires the input dimension to be divisible by four.")
    grouped = weight.detach().reshape(*weight.shape[:-1], -1, 4)
    selected = grouped.abs().topk(k=2, dim=-1, sorted=False).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, selected, True)
    mask = mask.reshape_as(weight)
    values = weight.detach() * mask.to(dtype=weight.dtype)
    return StructuredSparse24(
        values=values.contiguous(),
        mask=mask.contiguous(),
        original_shape=tuple(int(value) for value in weight.shape),
    )


def validate_2_4(state: StructuredSparse24) -> None:
    if tuple(state.values.shape) != state.original_shape:
        raise ValueError("2:4 values do not match original_shape.")
    if tuple(state.mask.shape) != state.original_shape:
        raise ValueError("2:4 mask does not match original_shape.")
    grouped = state.mask.reshape(*state.mask.shape[:-1], -1, 4)
    if not bool((grouped.sum(dim=-1) == 2).all().item()):
        raise ValueError("2:4 mask must retain exactly two values per group of four.")
    if bool((state.values.masked_select(~state.mask) != 0).any().item()):
        raise ValueError("2:4 values outside the structural mask must be zero.")


def sparse_2_4_linear(
    inputs: Any,
    state: StructuredSparse24,
    bias: Any = None,
    *,
    backend: str = "reference",
) -> Any:
    """Execute the reference path or PyTorch's semi-structured CUDA backend."""

    validate_2_4(state)
    mode = str(backend).strip().lower()
    if mode not in {"reference", "cuda", "auto"}:
        raise ValueError("2:4 backend must be reference, cuda, or auto.")
    use_cuda = mode == "cuda" or (
        mode == "auto"
        and inputs.device.type == "cuda"
        and inputs.dtype in {torch.float16, torch.bfloat16}
    )
    dense = state.dense().to(device=inputs.device, dtype=inputs.dtype)
    if not use_cuda:
        return F.linear(inputs, dense, bias)
    if inputs.device.type != "cuda":
        raise ValueError("2:4 CUDA execution requires CUDA inputs.")
    try:
        sparse = torch.sparse.to_sparse_semi_structured(dense)
    except (AttributeError, RuntimeError) as exc:
        if mode == "auto":
            return F.linear(inputs, dense, bias)
        raise RuntimeError(
            "The selected CUDA device/runtime cannot execute semi-structured 2:4 weights."
        ) from exc
    return F.linear(inputs, sparse, bias)


class StructuredSparse24GroupedExperts(nn.Module):
    """Frozen grouped experts with a provider-reachable 2:4 execution path."""

    def __init__(self, base: Any, *, backend: str = "auto") -> None:
        super().__init__()
        execution_spec = getattr(
            base,
            "mirai_expert_mlp_spec",
            CANONICAL_PACKED_EXPERT_MLP_SPEC,
        )
        if execution_spec != CANONICAL_PACKED_EXPERT_MLP_SPEC:
            raise ValueError(
                "Structured 2:4 grouped experts currently support only the "
                "canonical gated-product execution layout."
            )
        self.num_experts = int(getattr(base, "num_experts"))
        self.backend = str(backend).strip().lower()
        self._sparse_cache: dict[tuple[str, int, str, Any], Any] = {}
        parametrizations = getattr(base, "parametrizations", None)
        if parametrizations is not None and any(
            hasattr(parametrizations, key) for key in ("w1", "w2", "w3")
        ):
            raise ValueError(
                "2:4 expert execution requires an adapter preset that "
                "does not target grouped expert tensors."
            )
        for key in ("w1", "w2", "w3"):
            state = prune_to_2_4(getattr(base, key))
            self.register_buffer(f"{key}_values", state.values, persistent=True)
            self.register_buffer(f"{key}_mask", state.mask, persistent=True)

    def _state(self, key: str, expert_id: int) -> StructuredSparse24:
        values = getattr(self, f"{key}_values")[expert_id]
        mask = getattr(self, f"{key}_mask")[expert_id]
        return StructuredSparse24(values, mask, tuple(values.shape))

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self._sparse_cache.clear()
        super()._load_from_state_dict(*args, **kwargs)

    def _linear(self, inputs: Any, key: str, expert_id: int) -> Any:
        state = self._state(key, expert_id)
        use_cuda = (
            self.backend in {"auto", "cuda"}
            and inputs.device.type == "cuda"
            and inputs.dtype in {torch.float16, torch.bfloat16}
        )
        if not use_cuda:
            return sparse_2_4_linear(inputs, state, backend="reference")
        cache_key = (key, int(expert_id), str(inputs.device), inputs.dtype)
        sparse = self._sparse_cache.get(cache_key)
        if sparse is None:
            dense = state.dense().to(device=inputs.device, dtype=inputs.dtype)
            try:
                sparse = torch.sparse.to_sparse_semi_structured(dense)
            except (AttributeError, RuntimeError):
                if self.backend == "cuda":
                    raise
                return F.linear(inputs, dense)
            self._sparse_cache[cache_key] = sparse
        return F.linear(inputs, sparse)

    def run_direct_routed(self, tokens: Any, top_scores: Any, top_indices: Any) -> Any:
        original_shape = tuple(tokens.shape)
        flat = tokens.reshape(-1, original_shape[-1])
        scores = top_scores.reshape(flat.shape[0], -1)
        indices = top_indices.reshape(flat.shape[0], -1)
        output = torch.zeros_like(flat)
        for expert_id in range(self.num_experts):
            positions = torch.nonzero(indices == expert_id, as_tuple=False)
            if positions.numel() == 0:
                continue
            token_ids, slots = positions[:, 0], positions[:, 1]
            selected = flat.index_select(0, token_ids)
            gate = self._linear(selected, "w1", expert_id)
            up = self._linear(selected, "w3", expert_id)
            routed = self._linear(F.silu(gate) * up, "w2", expert_id)
            output.index_add_(
                0,
                token_ids,
                routed * scores[token_ids, slots].unsqueeze(-1),
            )
        return output.reshape(*original_shape)

    forward = run_direct_routed
