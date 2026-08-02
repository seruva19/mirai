"""Activation-space execution for LoRA-parametrized fused expert tensors.

The reference tensor parametrization forms ``B @ A`` in weight space.  This
executor uses associativity to evaluate ``(x @ A.T) @ B.T`` instead, so the
rank-3 ``[experts, out, in]`` adapter delta is never materialized.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class ExpertTensorLoRAExecutor:
    """Explicit host extension for native grouped-expert modules."""

    def __init__(self) -> None:
        self._routed_adapter_gate: Any | None = None

    def set_routed_adapter_gate(self, gate: Any | None) -> None:
        """Install a non-owning sample/expert route gate for the current batch."""
        self._routed_adapter_gate = gate

    def set_routed_adapter_tokens_per_sample(self, value: int) -> None:
        gate = self._routed_adapter_gate
        if gate is not None:
            gate.bind_tokens_per_sample(int(value))

    def resolve_route_gate(
        self,
        route_token_indices: torch.Tensor | None,
        counts: torch.Tensor,
    ) -> torch.Tensor | None:
        gate = self._routed_adapter_gate
        if gate is None:
            return None
        if route_token_indices is None:
            raise RuntimeError(
                "Routed native expert LoRA requires sorted token identity."
            )
        return gate.resolve_sorted(route_token_indices, counts)

    @staticmethod
    def _parts(
        owner: Any, key: str, like: torch.Tensor
    ) -> tuple[Any, Any, Any, float]:
        parametrizations = getattr(owner, "parametrizations", None)
        if parametrizations is None or not hasattr(parametrizations, key):
            raise RuntimeError(f"Expert tensor {key!r} is not parametrized for LoRA.")
        chain = getattr(parametrizations, key)
        if len(chain) != 1 or not hasattr(chain[0], "activation_factors"):
            raise RuntimeError(
                f"Expert tensor {key!r} does not expose the activation-space LoRA contract."
            )
        original = chain.original.to(device=like.device, dtype=like.dtype)
        a, b, scale = chain[0].activation_factors(like)
        return original, a, b, float(scale)

    def grouped_linear(
        self,
        owner: Any,
        key: str,
        tokens: torch.Tensor,
        *,
        offsets: torch.Tensor,
        route_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original, a, b, scale = self._parts(owner, key, tokens)
        grouped_mm = getattr(torch, "_grouped_mm", None)
        if not callable(grouped_mm):
            raise RuntimeError("Activation-space grouped expert LoRA is unavailable.")
        base = grouped_mm(tokens, original.transpose(-2, -1), offs=offsets)
        projected = grouped_mm(tokens, a.transpose(-2, -1), offs=offsets)
        b_transposed = b.transpose(-2, -1)
        # torch._grouped_mm on SM90 requires the non-unit matrix strides to be
        # 16-byte aligned. For low LoRA ranks the transposed view carries
        # ``rank * element_size`` as its output stride (rank=2 BF16 -> 4 bytes).
        # Materializing the small adapter factor keeps the grouped fast path
        # without ever forming the full expert delta ``B @ A``.
        if int(b_transposed.stride(-1)) * int(b_transposed.element_size()) % 16:
            b_transposed = b_transposed.contiguous()
        delta = grouped_mm(projected, b_transposed, offs=offsets)
        if route_gate is not None:
            if tuple(route_gate.shape) != tuple(delta.shape[:-1]):
                raise ValueError("route_gate must match every grouped route.")
            delta = delta * route_gate.to(
                device=delta.device, dtype=delta.dtype
            ).unsqueeze(-1)
        return base + delta * scale

    def linear(
        self,
        owner: Any,
        key: str,
        tokens: torch.Tensor,
        expert_index: int,
        *,
        route_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original, a, b, scale = self._parts(owner, key, tokens)
        index = int(expert_index)
        base = F.linear(tokens, original[index])
        delta = F.linear(F.linear(tokens, a[index]), b[index])
        if route_gate is not None:
            if tuple(route_gate.shape) != tuple(delta.shape[:-1]):
                raise ValueError("route_gate must match every expert route.")
            delta = delta * route_gate.to(
                device=delta.device, dtype=delta.dtype
            ).unsqueeze(-1)
        return base + delta * scale

    def run_for_loop(
        self,
        owner: Any,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        *,
        route_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        splits = torch.split(tokens, counts.tolist(), dim=0)
        resolved_gate = self.resolve_route_gate(route_token_indices, counts)
        gate_splits = (
            (None,) * len(splits)
            if resolved_gate is None
            else torch.split(resolved_gate, counts.tolist(), dim=0)
        )
        outputs: list[torch.Tensor] = []
        for expert_index, (expert_tokens, route_gate) in enumerate(
            zip(splits, gate_splits)
        ):
            if expert_tokens.numel() == 0:
                continue
            hidden = F.silu(
                self.linear(
                    owner,
                    "w1",
                    expert_tokens,
                    expert_index,
                    route_gate=route_gate,
                )
            )
            up = self.linear(
                owner,
                "w3",
                expert_tokens,
                expert_index,
                route_gate=route_gate,
            )
            outputs.append(
                self.linear(
                    owner,
                    "w2",
                    hidden * up,
                    expert_index,
                    route_gate=route_gate,
                )
            )
        if not outputs:
            return tokens.new_zeros(tokens.shape)
        return torch.cat(outputs, dim=0)


def install_expert_tensor_lora_executor(module: Any) -> None:
    """Install the executor through the native host's explicit extension seam."""

    setter = getattr(module, "set_linear_extension", None)
    if not callable(setter):
        raise TypeError(
            f"{type(module).__name__} does not implement set_linear_extension()."
        )
    setter(ExpertTensorLoRAExecutor())
