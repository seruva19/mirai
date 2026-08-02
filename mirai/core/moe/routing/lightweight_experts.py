"""Model-agnostic routing contract for lightweight expert slots.

The null-aware balance objective and physical-only gate normalization follow
AdaMoE, Sections 3.3--3.5 and Equation 6:
https://arxiv.org/abs/2406.13233

Copy and learned constant-mixture experts follow MMOE, Equations 4--5:
https://arxiv.org/abs/2607.24665
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


LIGHTWEIGHT_EXPERT_STATE_PREFIX = "lightweight_experts."
LIGHTWEIGHT_EXPERT_STATE_VERSION = 1


@dataclass(frozen=True)
class LightweightExpertRoutingDecision:
    """One logical route split into physical and lightweight contributions."""

    physical_indices: Any
    physical_scores: Any
    physical_active_mask: Any
    logical_indices: Any
    logical_selected_scores: Any
    logical_output_scores: Any
    logical_probabilities: Any
    load_balance_loss: Any
    z_loss: Any


def _null_aware_balance(
    probabilities: Any,
    selected_indices: Any,
    *,
    physical_experts: int,
    zero_experts: int,
    copy_experts: int = 0,
    constant_experts: int = 0,
    batch_size: int,
    tokens_per_sample: int,
    mode: str,
) -> Any:
    """AdaMoE Equation 6 with indistinguishable null-expert frequencies."""

    physical_experts = int(physical_experts)
    zero_experts = int(zero_experts)
    copy_experts = int(copy_experts)
    constant_experts = int(constant_experts)
    total_experts = (
        physical_experts + zero_experts + copy_experts + constant_experts
    )
    if probabilities.ndim != 2 or int(probabilities.shape[-1]) != total_experts:
        raise ValueError(
            "Null-aware balance probabilities must be [tokens, logical_experts]."
        )
    if selected_indices.ndim != 2:
        raise ValueError("Null-aware balance selections must be [tokens, top_k].")
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "disabled":
        return probabilities.float().sum() * 0.0
    if normalized_mode not in {"global", "sequence"}:
        raise ValueError(
            "Null-aware balance mode must be 'disabled', 'global', or 'sequence'."
        )

    probs = probabilities.float()
    if normalized_mode == "global":
        probability_mean = probs.mean(dim=0, keepdim=True)
        counts = probs.new_zeros((1, total_experts))
        counts.scatter_add_(
            1,
            selected_indices.reshape(1, -1),
            probs.new_ones((1, int(selected_indices.numel()))),
        )
        frequency = counts / float(max(1, int(probabilities.shape[0])))
    else:
        if (
            int(batch_size) <= 0
            or int(tokens_per_sample) <= 0
            or int(batch_size) * int(tokens_per_sample)
            != int(probabilities.shape[0])
        ):
            raise ValueError(
                "Sequence-wise null-aware balance requires valid batch and token "
                "shape metadata."
            )
        top_k = int(selected_indices.shape[-1])
        probability_mean = probs.reshape(
            int(batch_size), int(tokens_per_sample), total_experts
        ).mean(dim=1)
        counts = probs.new_zeros((int(batch_size), total_experts))
        counts.scatter_add_(
            1,
            selected_indices.reshape(
                int(batch_size), int(tokens_per_sample) * top_k
            ),
            probs.new_ones(
                (int(batch_size), int(tokens_per_sample) * top_k)
            ),
        )
        frequency = counts / float(int(tokens_per_sample))

    zero_start = physical_experts
    zero_end = zero_start + zero_experts
    physical_frequency = frequency[:, :zero_start]
    zero_frequency = frequency[:, zero_start:zero_end]
    ordinary_lightweight_frequency = frequency[:, zero_end:]
    physical_term = (
        physical_frequency * probability_mean[:, :zero_start]
    ).sum(dim=-1)
    if zero_experts:
        mean_zero_frequency = zero_frequency.mean(dim=-1, keepdim=True)
        zero_term = (
            mean_zero_frequency * probability_mean[:, zero_start:zero_end]
        ).sum(dim=-1)
    else:
        zero_term = physical_term * 0.0
    ordinary_lightweight_term = (
        ordinary_lightweight_frequency * probability_mean[:, zero_end:]
    ).sum(dim=-1)
    return (
        float(total_experts)
        * (physical_term + zero_term + ordinary_lightweight_term)
    ).mean()


class LightweightExpertPool(nn.Module):
    """Logical zero, copy, and learned constant-mixture expert routes."""

    def __init__(
        self,
        *,
        physical_experts: int,
        hidden_size: int,
        zero_experts: int,
        copy_experts: int = 0,
        constant_experts: int = 0,
        top_k: int,
        balance_mode: str,
        initial_router_rows: Any,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("LightweightExpertPool requires torch.")
        super().__init__()
        if int(physical_experts) <= 0:
            raise ValueError("physical_experts must be > 0.")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be > 0.")
        if min(
            int(zero_experts),
            int(copy_experts),
            int(constant_experts),
        ) < 0:
            raise ValueError("Lightweight expert counts must be >= 0.")
        lightweight_experts = (
            int(zero_experts)
            + int(copy_experts)
            + int(constant_experts)
        )
        if lightweight_experts <= 0:
            raise ValueError("At least one lightweight expert is required.")
        total_experts = int(physical_experts) + lightweight_experts
        if not 1 <= int(top_k) <= total_experts:
            raise ValueError(
                "top_k must be in [1, physical_experts + lightweight_experts]."
            )
        if tuple(initial_router_rows.shape) != (
            lightweight_experts,
            int(hidden_size),
        ):
            raise ValueError(
                "initial_router_rows must be [lightweight_experts, hidden_size]."
            )
        normalized_balance = str(balance_mode).strip().lower()
        if normalized_balance not in {"disabled", "global", "sequence"}:
            raise ValueError(
                "balance_mode must be 'disabled', 'global', or 'sequence'."
            )
        self.physical_experts = int(physical_experts)
        self.hidden_size = int(hidden_size)
        self.zero_experts = int(zero_experts)
        self.copy_experts = int(copy_experts)
        self.constant_experts = int(constant_experts)
        self.top_k = int(top_k)
        self.balance_mode = normalized_balance
        self.router_weight = nn.Parameter(
            initial_router_rows.detach().clone().to(dtype=torch.float32)
        )
        self.register_buffer(
            "correction_bias",
            torch.zeros(
                lightweight_experts,
                device=initial_router_rows.device,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        if self.constant_experts:
            self.constant_vectors = nn.Parameter(
                torch.zeros(
                    self.constant_experts,
                    self.hidden_size,
                    device=initial_router_rows.device,
                    dtype=torch.float32,
                )
            )
            self.constant_gates = nn.Parameter(
                torch.zeros(
                    self.constant_experts,
                    2,
                    self.hidden_size,
                    device=initial_router_rows.device,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("constant_vectors", None)
            self.register_parameter("constant_gates", None)

    @property
    def logical_experts(self) -> int:
        return int(
            self.physical_experts
            + self.zero_experts
            + self.copy_experts
            + self.constant_experts
        )

    @property
    def zero_start(self) -> int:
        return int(self.physical_experts)

    @property
    def copy_start(self) -> int:
        return int(self.zero_start + self.zero_experts)

    @property
    def constant_start(self) -> int:
        return int(self.copy_start + self.copy_experts)

    @classmethod
    def from_physical_router(
        cls,
        physical_router_weight: Any,
        *,
        zero_experts: int,
        copy_experts: int = 0,
        constant_experts: int = 0,
        top_k: int,
        balance_mode: str,
    ) -> "LightweightExpertPool":
        """Derive added router rows by deterministic cycling over pretrained rows."""

        if torch is None:  # pragma: no cover
            raise RuntimeError("LightweightExpertPool requires torch.")
        if physical_router_weight.ndim != 2:
            raise ValueError("physical_router_weight must be rank 2.")
        physical_experts, hidden_size = (
            int(physical_router_weight.shape[0]),
            int(physical_router_weight.shape[1]),
        )
        lightweight_experts = (
            int(zero_experts)
            + int(copy_experts)
            + int(constant_experts)
        )
        ids = torch.arange(
            lightweight_experts,
            device=physical_router_weight.device,
            dtype=torch.long,
        ).remainder(physical_experts)
        initial_rows = physical_router_weight.detach().index_select(0, ids)
        return cls(
            physical_experts=physical_experts,
            hidden_size=hidden_size,
            zero_experts=int(zero_experts),
            copy_experts=int(copy_experts),
            constant_experts=int(constant_experts),
            top_k=int(top_k),
            balance_mode=balance_mode,
            initial_router_rows=initial_rows,
        )

    def append_logits(self, tokens: Any, physical_logits: Any) -> Any:
        if tokens.ndim != 2 or int(tokens.shape[-1]) != self.hidden_size:
            raise ValueError(
                "Lightweight-expert router tokens must be [tokens, hidden_size]."
            )
        if (
            physical_logits.ndim != 2
            or int(physical_logits.shape[0]) != int(tokens.shape[0])
            or int(physical_logits.shape[-1]) != self.physical_experts
        ):
            raise ValueError(
                "Physical router logits must be [tokens, physical_experts]."
            )
        lightweight_logits = F.linear(
            tokens.float(),
            self.router_weight.to(device=tokens.device, dtype=torch.float32),
        )
        return torch.cat((physical_logits, lightweight_logits), dim=-1)

    def route(
        self,
        logical_logits: Any,
        *,
        physical_correction_bias: Any,
        score_func: str,
        route_scale: float,
        physical_choice_transform: Callable[[Any], Any] | None,
        batch_size: int,
        tokens_per_sample: int,
        training: bool,
    ) -> LightweightExpertRoutingDecision:
        """Select logical slots while returning only valid physical dispatch ids."""

        if (
            logical_logits.ndim != 2
            or int(logical_logits.shape[-1]) != self.logical_experts
        ):
            raise ValueError(
                "Logical router logits must be [tokens, logical_experts]."
            )
        score_name = str(score_func).strip().lower()
        if score_name == "softmax":
            probabilities = F.softmax(logical_logits, dim=-1)
        elif score_name == "sigmoid":
            probabilities = logical_logits.sigmoid()
        else:
            raise ValueError(f"Unsupported router score function '{score_func}'.")
        physical_bias = physical_correction_bias.to(
            device=logical_logits.device, dtype=probabilities.dtype
        )
        lightweight_bias = self.correction_bias.to(
            device=logical_logits.device, dtype=probabilities.dtype
        )
        choice_scores = probabilities + torch.cat(
            (physical_bias, lightweight_bias), dim=0
        ).unsqueeze(0)
        if physical_choice_transform is not None:
            transformed_physical = physical_choice_transform(
                choice_scores[:, : self.physical_experts]
            )
            choice_scores = torch.cat(
                (
                    transformed_physical,
                    choice_scores[:, self.physical_experts :],
                ),
                dim=-1,
            )
        logical_indices = torch.topk(
            choice_scores, k=self.top_k, dim=-1, sorted=False
        )[1]
        selected_scores = probabilities.gather(1, logical_indices)
        physical_active = logical_indices < self.physical_experts
        zero_active = (
            (logical_indices >= self.zero_start)
            & (logical_indices < self.copy_start)
        )
        output_active = ~zero_active
        output_scores = selected_scores * output_active.to(selected_scores.dtype)
        output_scores = output_scores / output_scores.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-20)
        output_scores = output_scores * float(route_scale)
        physical_scores = (
            output_scores * physical_active.to(output_scores.dtype)
        )
        physical_indices = logical_indices.clamp_max(self.physical_experts - 1)
        if training:
            balance = _null_aware_balance(
                probabilities,
                logical_indices,
                physical_experts=self.physical_experts,
                zero_experts=self.zero_experts,
                copy_experts=self.copy_experts,
                constant_experts=self.constant_experts,
                batch_size=int(batch_size),
                tokens_per_sample=int(tokens_per_sample),
                mode=self.balance_mode,
            )
        else:
            balance = probabilities.float().sum() * 0.0
        z_loss = torch.logsumexp(logical_logits.float(), dim=-1).square().mean()
        return LightweightExpertRoutingDecision(
            physical_indices=physical_indices,
            physical_scores=physical_scores,
            physical_active_mask=physical_active,
            logical_indices=logical_indices,
            logical_selected_scores=selected_scores,
            logical_output_scores=output_scores,
            logical_probabilities=probabilities,
            load_balance_loss=balance,
            z_loss=z_loss,
        )

    def topology(self) -> dict[str, int | str]:
        return {
            "kind": "zero_copy_constant",
            "physical_experts": int(self.physical_experts),
            "zero_experts": int(self.zero_experts),
            "copy_experts": int(self.copy_experts),
            "constant_experts": int(self.constant_experts),
            "hidden_size": int(self.hidden_size),
            "top_k": int(self.top_k),
            "balance_mode": str(self.balance_mode),
        }

    def output_contribution(
        self,
        tokens: Any,
        logical_indices: Any,
        logical_output_scores: Any,
    ) -> Any:
        """Return selected copy and constant-expert output contributions."""

        if (
            logical_indices.ndim != 2
            or logical_output_scores.shape != logical_indices.shape
            or int(logical_indices.shape[0]) != int(tokens.shape[0])
        ):
            raise ValueError(
                "Lightweight selections and scores must be [tokens, top_k]."
            )
        output = torch.zeros_like(tokens)
        copy_mask = (
            (logical_indices >= self.copy_start)
            & (logical_indices < self.constant_start)
        )
        copy_weight = (
            logical_output_scores
            * copy_mask.to(logical_output_scores.dtype)
        ).sum(dim=-1, keepdim=True)
        output = output + tokens * copy_weight.to(tokens.dtype)
        if self.constant_experts:
            constant_ids = torch.arange(
                self.constant_experts,
                device=logical_indices.device,
                dtype=logical_indices.dtype,
            ) + self.constant_start
            selected_constants = (
                logical_indices.unsqueeze(-1)
                == constant_ids.reshape(1, 1, -1)
            )
            outer_weights = (
                logical_output_scores.unsqueeze(-1)
                * selected_constants.to(logical_output_scores.dtype)
            ).sum(dim=1)
            mixture = torch.einsum(
                "nd,cmd->ncm",
                tokens.float(),
                self.constant_gates.to(
                    device=tokens.device, dtype=torch.float32
                ),
            ).softmax(dim=-1)
            constants = self.constant_vectors.to(
                device=tokens.device, dtype=torch.float32
            )
            expert_outputs = (
                mixture[..., :1] * tokens.float().unsqueeze(1)
                + mixture[..., 1:] * constants.unsqueeze(0)
            )
            output = output + torch.einsum(
                "nc,ncd->nd",
                outer_weights.float(),
                expert_outputs,
            ).to(tokens.dtype)
        return output


def export_lightweight_expert_state(root: Any) -> dict[str, Any]:
    """Export every enabled pool with an exact, versioned topology manifest."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Lightweight expert persistence requires torch.")
    pools = {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, LightweightExpertPool)
    }
    if not pools:
        return {}
    state: dict[str, Any] = {
        f"{LIGHTWEIGHT_EXPERT_STATE_PREFIX}schema_version": (
            LIGHTWEIGHT_EXPERT_STATE_VERSION
        ),
        f"{LIGHTWEIGHT_EXPERT_STATE_PREFIX}topology": {
            name: module.topology() for name, module in sorted(pools.items())
        },
    }
    for name, module in sorted(pools.items()):
        for key, value in module.state_dict().items():
            state[
                f"{LIGHTWEIGHT_EXPERT_STATE_PREFIX}modules.{name}.{key}"
            ] = value.detach().cpu().clone()
    return state


def load_lightweight_expert_state(root: Any, state: dict[str, Any]) -> None:
    """Load enabled pools and fail closed on topology or tensor mismatches."""

    prefix = LIGHTWEIGHT_EXPERT_STATE_PREFIX
    supplied = {str(key): value for key, value in state.items() if str(key).startswith(prefix)}
    pools = {
        name: module
        for name, module in root.named_modules()
        if isinstance(module, LightweightExpertPool)
    }
    if not pools:
        if supplied:
            raise ValueError(
                "Adapter contains lightweight-expert state, but the configured "
                "model has no lightweight expert pool."
            )
        return
    if not supplied:
        raise ValueError(
            "The configured lightweight expert pool requires adapter state."
        )
    version = supplied.get(f"{prefix}schema_version")
    if int(version) != LIGHTWEIGHT_EXPERT_STATE_VERSION:
        raise ValueError(
            f"Unsupported lightweight-expert state version {version!r}."
        )
    expected_topology = {
        name: module.topology() for name, module in sorted(pools.items())
    }
    if supplied.get(f"{prefix}topology") != expected_topology:
        raise ValueError("Lightweight-expert topology does not match the model.")
    consumed = {f"{prefix}schema_version", f"{prefix}topology"}
    for name, module in sorted(pools.items()):
        module_prefix = f"{prefix}modules.{name}."
        module_state = {
            key[len(module_prefix) :]: value
            for key, value in supplied.items()
            if key.startswith(module_prefix)
        }
        expected_keys = set(module.state_dict())
        if set(module_state) != expected_keys:
            raise ValueError(
                f"Lightweight-expert state for '{name}' has keys "
                f"{sorted(module_state)}, expected {sorted(expected_keys)}."
            )
        module.load_state_dict(module_state, strict=True)
        consumed.update(module_prefix + key for key in module_state)
    unknown = sorted(set(supplied) - consumed)
    if unknown:
        raise ValueError(f"Unknown lightweight-expert state keys: {unknown}.")


__all__ = [
    "LIGHTWEIGHT_EXPERT_STATE_PREFIX",
    "LIGHTWEIGHT_EXPERT_STATE_VERSION",
    "LightweightExpertRoutingDecision",
    "LightweightExpertPool",
    "export_lightweight_expert_state",
    "load_lightweight_expert_state",
]
