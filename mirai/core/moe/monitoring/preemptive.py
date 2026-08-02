"""Mechanism-driven router and attention-update monitoring.

The router metrics implement Equations 7 and 8 of
https://arxiv.org/abs/2606.28116.  The attention monitor implements the
first-order QK-product increment from Equations 2--4 and computes its nonzero
singular spectrum from a QR core.  It reports spectral shape only; a low rank
is not an alarm without a separately established healthy trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RouterMechanismSummary:
    """Layer-aggregated router weight and decision-side indicators."""

    weight_similarity: float
    conditioning_ratio: float | None
    per_token_entropy: float
    per_token_entropy_fraction: float
    layer_count: int
    conditioning_layer_count: int

    def to_metrics(self) -> dict[str, float]:
        metrics = {
            "moe_router_weight_similarity": float(self.weight_similarity),
            "moe_router_per_token_entropy": float(self.per_token_entropy),
            "moe_router_per_token_entropy_fraction": float(
                self.per_token_entropy_fraction
            ),
            "moe_router_mechanism_layer_count": float(self.layer_count),
        }
        if self.conditioning_ratio is not None:
            metrics["moe_router_conditioning_ratio"] = float(
                self.conditioning_ratio
            )
            metrics["moe_router_conditioning_layer_count"] = float(
                self.conditioning_layer_count
            )
        return metrics


def _matrix(value: Any, *, label: str) -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Mechanism-driven monitoring requires torch.")
    if not torch.is_tensor(value):
        raise TypeError(f"{label} must be a torch.Tensor.")
    matrix = value.detach().float()
    if int(matrix.ndim) != 2 or min(int(dim) for dim in matrix.shape) <= 0:
        raise ValueError(f"{label} must be a non-empty rank-2 tensor.")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f"{label} must be finite.")
    return matrix


def router_weight_similarity(weight: Any) -> Any:
    """Return Equation 7 for router rows ``[experts, input_features]``."""

    matrix = _matrix(weight, label="router weight")
    experts = int(matrix.shape[0])
    if experts < 2:
        raise ValueError("Router similarity requires at least two experts.")
    norms = matrix.norm(dim=1)
    if bool(torch.any(norms <= 0).item()):
        raise ValueError("Router similarity requires every expert row to be nonzero.")
    mean_unit = (matrix / norms[:, None]).mean(dim=0)
    return (experts * mean_unit.square().sum() - 1.0) / (experts - 1)


def router_conditioning_ratio(weight: Any) -> Any | None:
    """Return ``max_i ||w_i-w_bar|| / ||w_bar||`` when it is defined."""

    matrix = _matrix(weight, label="router weight")
    if int(matrix.shape[0]) < 2:
        raise ValueError("Router conditioning requires at least two experts.")
    if bool(torch.any(matrix.norm(dim=1) <= 0).item()):
        return None
    mean = matrix.mean(dim=0)
    mean_norm = mean.norm()
    if float(mean_norm.item()) == 0.0:
        return None
    return (matrix - mean).norm(dim=1).max() / mean_norm


def per_token_router_entropy(probabilities: Any) -> tuple[Any, Any]:
    """Return mean full-softmax entropy and its fraction of ``log(E)``."""

    matrix = _matrix(probabilities, label="router probabilities")
    experts = int(matrix.shape[1])
    if experts < 2:
        raise ValueError("Router entropy requires at least two experts.")
    if bool(torch.any(matrix < 0).item()):
        raise ValueError("Router probabilities must be non-negative.")
    mass = matrix.sum(dim=1, keepdim=True)
    if bool(torch.any(mass <= 0).item()):
        raise ValueError("Every router probability row must have positive mass.")
    normalized = matrix / mass
    entropy = -(normalized * normalized.clamp_min(1e-30).log()).sum(dim=1).mean()
    return entropy, entropy / math.log(experts)


def summarize_router_mechanisms(
    layers: Iterable[tuple[Any, Any]],
) -> RouterMechanismSummary | None:
    """Aggregate router weight metrics and per-token entropy over layers."""

    similarities: list[float] = []
    conditioning: list[float] = []
    entropies: list[float] = []
    fractions: list[float] = []
    for weight, probabilities in layers:
        similarity = router_weight_similarity(weight)
        ratio = router_conditioning_ratio(weight)
        entropy, fraction = per_token_router_entropy(probabilities)
        similarities.append(float(similarity.cpu().item()))
        if ratio is not None:
            conditioning.append(float(ratio.cpu().item()))
        entropies.append(float(entropy.cpu().item()))
        fractions.append(float(fraction.cpu().item()))
    if not similarities:
        return None
    return RouterMechanismSummary(
        weight_similarity=sum(similarities) / len(similarities),
        conditioning_ratio=(
            sum(conditioning) / len(conditioning) if conditioning else None
        ),
        per_token_entropy=sum(entropies) / len(entropies),
        per_token_entropy_fraction=sum(fractions) / len(fractions),
        layer_count=len(similarities),
        conditioning_layer_count=len(conditioning),
    )


@dataclass(frozen=True)
class LowRankProjectionState:
    """Provider-owned static low-rank projection state.

    ``base_weight`` and ``factor_b @ factor_a`` use PyTorch linear orientation
    ``[output, input]``.  Dynamic, input-dependent adapters are intentionally
    outside this contract because they do not define one Q/K weight matrix.
    """

    base_weight: Any
    factor_a: Any
    factor_b: Any
    scale: float
    num_heads: int


@dataclass(frozen=True)
class AttentionQKState:
    name: str
    query: LowRankProjectionState
    key: LowRankProjectionState


@dataclass(frozen=True)
class _ProjectionSnapshot:
    factor_a: Any
    factor_b: Any
    scale: float

    def to_state(self) -> dict[str, Any]:
        return {
            "factor_a": self.factor_a.detach().cpu().clone(),
            "factor_b": self.factor_b.detach().cpu().clone(),
            "scale": float(self.scale),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "_ProjectionSnapshot":
        factor_a = state.get("factor_a")
        factor_b = state.get("factor_b")
        if torch is None or not torch.is_tensor(factor_a) or not torch.is_tensor(
            factor_b
        ):
            raise ValueError("Attention monitor factor snapshots must be tensors.")
        return cls(
            factor_a=factor_a.detach().cpu().clone(),
            factor_b=factor_b.detach().cpu().clone(),
            scale=float(state.get("scale", 0.0)),
        )


@dataclass(frozen=True)
class _AttentionSnapshot:
    query: _ProjectionSnapshot
    key: _ProjectionSnapshot

    def to_state(self) -> dict[str, Any]:
        return {"query": self.query.to_state(), "key": self.key.to_state()}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "_AttentionSnapshot":
        query = state.get("query")
        key = state.get("key")
        if not isinstance(query, Mapping) or not isinstance(key, Mapping):
            raise ValueError("Attention monitor snapshot requires query and key state.")
        return cls(
            query=_ProjectionSnapshot.from_state(query),
            key=_ProjectionSnapshot.from_state(key),
        )


@dataclass(frozen=True)
class AttentionSpectrumSummary:
    mean_effective_rank: float
    minimum_effective_rank: float
    maximum_effective_rank: float
    mean_spectral_entropy: float
    head_count: int
    layer_count: int

    def to_metrics(self) -> dict[str, float]:
        return {
            "moe_attention_qk_delta2_effective_rank": float(
                self.mean_effective_rank
            ),
            "moe_attention_qk_delta2_effective_rank_min": float(
                self.minimum_effective_rank
            ),
            "moe_attention_qk_delta2_effective_rank_max": float(
                self.maximum_effective_rank
            ),
            "moe_attention_qk_delta2_spectral_entropy": float(
                self.mean_spectral_entropy
            ),
            "moe_attention_qk_delta2_head_count": float(self.head_count),
            "moe_attention_qk_delta2_layer_count": float(self.layer_count),
        }


def singular_spectrum_effective_rank(singular_values: Any) -> tuple[Any, Any]:
    """Return alpha=2 spectral entropy and effective rank."""

    if torch is None or not torch.is_tensor(singular_values):
        raise TypeError("singular_values must be a torch.Tensor.")
    values = singular_values.detach().float().reshape(-1)
    if not bool(torch.isfinite(values).all().item()) or bool(
        torch.any(values < 0).item()
    ):
        raise ValueError("singular values must be finite and non-negative.")
    energy = values.square()
    total = energy.sum()
    if float(total.item()) <= 0.0:
        raise ValueError("spectral entropy requires a nonzero update.")
    probabilities = energy[energy > 0] / total
    entropy = -(probabilities * probabilities.log()).sum()
    return entropy, entropy.exp()


def _snapshot(projection: LowRankProjectionState) -> _ProjectionSnapshot:
    return _ProjectionSnapshot(
        factor_a=projection.factor_a.detach().cpu().clone(),
        factor_b=projection.factor_b.detach().cpu().clone(),
        scale=float(projection.scale),
    )


def snapshot_low_rank_projection(
    projection: LowRankProjectionState,
) -> _ProjectionSnapshot:
    """Capture only the small trainable factors for the next update window."""

    _validate_projection(projection, label="projection")
    return _snapshot(projection)


def _same(current: LowRankProjectionState, previous: _ProjectionSnapshot) -> bool:
    return (
        float(current.scale) == float(previous.scale)
        and torch.equal(current.factor_a.detach().cpu(), previous.factor_a)
        and torch.equal(current.factor_b.detach().cpu(), previous.factor_b)
    )


def _validate_projection(projection: LowRankProjectionState, *, label: str) -> None:
    base = _matrix(projection.base_weight, label=f"{label} base weight")
    factor_a = _matrix(projection.factor_a, label=f"{label} factor A")
    factor_b = _matrix(projection.factor_b, label=f"{label} factor B")
    if int(factor_a.shape[0]) != int(factor_b.shape[1]):
        raise ValueError(f"{label} low-rank factors have mismatched rank.")
    if int(factor_a.shape[1]) != int(base.shape[1]) or int(
        factor_b.shape[0]
    ) != int(base.shape[0]):
        raise ValueError(f"{label} factors do not match the base weight.")
    if int(projection.num_heads) <= 0 or int(base.shape[0]) % int(
        projection.num_heads
    ):
        raise ValueError(f"{label} output width must divide evenly into heads.")
    if not math.isfinite(float(projection.scale)):
        raise ValueError(f"{label} scale must be finite.")


def _previous_times(
    current: LowRankProjectionState,
    previous: _ProjectionSnapshot,
    *,
    head: int,
    right: Any,
) -> Any:
    """Compute ``W_previous.T @ right`` without materializing ``W_previous``."""

    device = current.base_weight.device
    base = current.base_weight.detach()
    heads = int(current.num_heads)
    head_dim = int(base.shape[0]) // heads
    start = int(head) * head_dim
    stop = start + head_dim
    right_local = right.to(device=device, dtype=base.dtype)
    result = base[start:stop].transpose(0, 1).matmul(right_local).float()
    previous_a = previous.factor_a.to(device=device, dtype=torch.float32)
    previous_b = previous.factor_b[start:stop].to(
        device=device, dtype=torch.float32
    )
    result = result + previous_a.transpose(0, 1).matmul(
        (previous_b.transpose(0, 1).matmul(right.float())) * float(previous.scale)
    )
    return result


def _delta_factors(
    current: LowRankProjectionState,
    previous: _ProjectionSnapshot,
    *,
    head: int,
) -> tuple[Any, Any]:
    device = current.base_weight.device
    output = int(current.base_weight.shape[0])
    head_dim = output // int(current.num_heads)
    start = int(head) * head_dim
    stop = start + head_dim
    current_a = current.factor_a.detach().to(device=device, dtype=torch.float32)
    current_b = current.factor_b.detach()[start:stop].to(
        device=device, dtype=torch.float32
    )
    previous_a = previous.factor_a.to(device=device, dtype=torch.float32)
    previous_b = previous.factor_b[start:stop].to(
        device=device, dtype=torch.float32
    )
    left = torch.cat((current_a.transpose(0, 1), previous_a.transpose(0, 1)), dim=1)
    right = torch.cat(
        (
            current_b * float(current.scale),
            previous_b * -float(previous.scale),
        ),
        dim=1,
    )
    return left, right


def attention_delta2_singular_values(
    current_query: LowRankProjectionState,
    previous_query: _ProjectionSnapshot,
    current_key: LowRankProjectionState,
    previous_key: _ProjectionSnapshot,
    *,
    head: int,
) -> Any:
    """Compute one head's exact nonzero ``Delta2`` spectrum from a QR core."""

    _validate_projection(current_query, label="query")
    _validate_projection(current_key, label="key")
    if int(current_query.num_heads) != int(current_key.num_heads):
        raise ValueError("Query and key projections must have the same head count.")
    if tuple(current_query.base_weight.shape) != tuple(current_key.base_weight.shape):
        raise ValueError("Query and key base weights must have the same shape.")
    if not 0 <= int(head) < int(current_query.num_heads):
        raise ValueError("Attention head index is out of range.")
    query_left, query_right = _delta_factors(
        current_query, previous_query, head=int(head)
    )
    key_left, key_right = _delta_factors(
        current_key, previous_key, head=int(head)
    )
    left = torch.cat(
        (
            query_left,
            _previous_times(
                current_query,
                previous_query,
                head=int(head),
                right=key_right,
            ),
        ),
        dim=1,
    )
    right = torch.cat(
        (
            _previous_times(
                current_key,
                previous_key,
                head=int(head),
                right=query_right,
            ),
            key_left,
        ),
        dim=1,
    )
    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right, mode="reduced")
    del q_left, q_right
    values = torch.linalg.svdvals(r_left @ r_right.transpose(0, 1))
    tolerance = torch.finfo(values.dtype).eps * max(left.shape) * values.max()
    return values[values > tolerance]


class PreemptiveAttentionMonitor:
    """Stateful one-applied-update window over provider-bound Q/K factors."""

    schema_version = 1

    def __init__(self) -> None:
        self._snapshots: dict[str, _AttentionSnapshot] = {}
        self._last_summary: AttentionSpectrumSummary | None = None

    def observe(self, states: Iterable[AttentionQKState]) -> dict[str, float]:
        current_states = tuple(states)
        observed_names = [str(item.name) for item in current_states]
        if len(set(observed_names)) != len(observed_names):
            raise ValueError("Attention monitor target names must be unique.")
        ranks: list[float] = []
        entropies: list[float] = []
        changed_layers = 0
        for state in current_states:
            name = str(state.name)
            _validate_projection(state.query, label=f"{name} query")
            _validate_projection(state.key, label=f"{name} key")
            previous = self._snapshots.get(name)
            current_snapshot = _AttentionSnapshot(
                query=_snapshot(state.query),
                key=_snapshot(state.key),
            )
            if previous is None:
                self._snapshots[name] = current_snapshot
                continue
            changed = not (
                _same(state.query, previous.query)
                and _same(state.key, previous.key)
            )
            if not changed:
                continue
            changed_layers += 1
            for head in range(int(state.query.num_heads)):
                values = attention_delta2_singular_values(
                    state.query,
                    previous.query,
                    state.key,
                    previous.key,
                    head=head,
                )
                if int(values.numel()) == 0:
                    continue
                entropy, effective_rank = singular_spectrum_effective_rank(values)
                entropies.append(float(entropy.detach().cpu().item()))
                ranks.append(float(effective_rank.detach().cpu().item()))
            self._snapshots[name] = current_snapshot
        if ranks:
            self._last_summary = AttentionSpectrumSummary(
                mean_effective_rank=sum(ranks) / len(ranks),
                minimum_effective_rank=min(ranks),
                maximum_effective_rank=max(ranks),
                mean_spectral_entropy=sum(entropies) / len(entropies),
                head_count=len(ranks),
                layer_count=changed_layers,
            )
        return self._last_summary.to_metrics() if self._last_summary is not None else {}

    def state_dict(self) -> dict[str, Any]:
        summary = None
        if self._last_summary is not None:
            summary = {
                "mean_effective_rank": self._last_summary.mean_effective_rank,
                "minimum_effective_rank": self._last_summary.minimum_effective_rank,
                "maximum_effective_rank": self._last_summary.maximum_effective_rank,
                "mean_spectral_entropy": self._last_summary.mean_spectral_entropy,
                "head_count": self._last_summary.head_count,
                "layer_count": self._last_summary.layer_count,
            }
        return {
            "schema_version": self.schema_version,
            "snapshots": {
                name: snapshot.to_state()
                for name, snapshot in sorted(self._snapshots.items())
            },
            "last_summary": summary,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", 0)) != self.schema_version:
            raise ValueError("Unsupported preemptive attention monitor schema.")
        raw_snapshots = state.get("snapshots", {})
        if not isinstance(raw_snapshots, Mapping):
            raise ValueError("Attention monitor snapshots must be a mapping.")
        self._snapshots = {
            str(name): _AttentionSnapshot.from_state(snapshot)
            for name, snapshot in raw_snapshots.items()
            if isinstance(snapshot, Mapping)
        }
        if len(self._snapshots) != len(raw_snapshots):
            raise ValueError("Every attention monitor snapshot must be a mapping.")
        raw_summary = state.get("last_summary")
        if raw_summary is None:
            self._last_summary = None
        elif isinstance(raw_summary, Mapping):
            self._last_summary = AttentionSpectrumSummary(
                mean_effective_rank=float(raw_summary["mean_effective_rank"]),
                minimum_effective_rank=float(raw_summary["minimum_effective_rank"]),
                maximum_effective_rank=float(raw_summary["maximum_effective_rank"]),
                mean_spectral_entropy=float(raw_summary["mean_spectral_entropy"]),
                head_count=int(raw_summary["head_count"]),
                layer_count=int(raw_summary["layer_count"]),
            )
        else:
            raise ValueError("Attention monitor last_summary must be a mapping or null.")


__all__ = [
    "AttentionQKState",
    "AttentionSpectrumSummary",
    "LowRankProjectionState",
    "PreemptiveAttentionMonitor",
    "RouterMechanismSummary",
    "attention_delta2_singular_values",
    "per_token_router_entropy",
    "router_conditioning_ratio",
    "router_weight_similarity",
    "singular_spectrum_effective_rank",
    "snapshot_low_rank_projection",
    "summarize_router_mechanisms",
]
