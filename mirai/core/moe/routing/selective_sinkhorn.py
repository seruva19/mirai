"""Selective entropy-regularized optimal-transport routing.

The transport constraints and training-only selection rule follow
https://arxiv.org/abs/2511.08972v2 (Equations 3-6 and Algorithm 2).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SELECTIVE_SINKHORN_COST_MODES = frozenset({"linear", "softmax"})


@dataclass(frozen=True)
class SelectiveSinkhornSpec:
    probability: float = 0.001
    cost_mode: str = "softmax"
    entropy_regularization: float = 0.05
    max_iterations: int = 100
    tolerance: float = 1e-4
    noise_scale: float = 0.0
    seed: int = 0

    def validate(self) -> None:
        probability = float(self.probability)
        entropy = float(self.entropy_regularization)
        tolerance = float(self.tolerance)
        noise = float(self.noise_scale)
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("probability must be finite and in (0, 1]")
        if str(self.cost_mode).strip().lower() not in SELECTIVE_SINKHORN_COST_MODES:
            raise ValueError("cost_mode must be 'linear' or 'softmax'")
        if not math.isfinite(entropy) or entropy <= 0.0:
            raise ValueError("entropy_regularization must be finite and > 0")
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and > 0")
        if not math.isfinite(noise) or noise < 0.0:
            raise ValueError("noise_scale must be finite and >= 0")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class SinkhornTransportResult:
    plan: Any
    iterations: int
    converged: bool
    row_error: float
    column_error: float


@dataclass(frozen=True)
class SelectiveSinkhornRoutes:
    top_indices: Any
    top_weights: Any
    transport: SinkhornTransportResult


def _cost_matrix(logits: Any, *, mode: str) -> Any:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "linear":
        return logits
    if normalized_mode == "softmax":
        return logits.softmax(dim=-1)
    raise ValueError("cost_mode must be 'linear' or 'softmax'")


def selective_sinkhorn_transport(
    logits: Any,
    *,
    cost_mode: str,
    entropy_regularization: float,
    max_iterations: int,
    tolerance: float,
    noise_scale: float = 0.0,
    generator: Any | None = None,
) -> SinkhornTransportResult:
    """Solve the paper's maximum-cost OT problem in the log domain."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Selective Sinkhorn routing requires torch.")
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("logits must have shape [tokens, experts]")
    tokens, experts = (int(value) for value in logits.shape)
    if tokens <= 0 or experts <= 0:
        raise ValueError("logits must contain at least one token and expert")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite")
    spec = SelectiveSinkhornSpec(
        probability=1.0,
        cost_mode=cost_mode,
        entropy_regularization=entropy_regularization,
        max_iterations=max_iterations,
        tolerance=tolerance,
        noise_scale=noise_scale,
    )
    spec.validate()
    cost = _cost_matrix(logits.detach().float(), mode=spec.cost_mode)
    if float(spec.noise_scale) > 0.0:
        noise = torch.randn(
            cost.shape,
            device=cost.device,
            dtype=cost.dtype,
            generator=generator,
        )
        cost = cost + float(spec.noise_scale) * noise

    log_kernel = cost / float(spec.entropy_regularization)
    log_row_marginal = torch.zeros(
        tokens, device=cost.device, dtype=torch.float32
    )
    log_column_marginal = torch.full(
        (experts,),
        math.log(float(tokens) / float(experts)),
        device=cost.device,
        dtype=torch.float32,
    )
    log_u = torch.zeros_like(log_row_marginal)
    log_v = torch.zeros_like(log_column_marginal)
    converged = False
    row_error = float("inf")
    column_error = float("inf")
    iterations = 0
    plan = torch.empty_like(log_kernel)
    for iteration in range(1, int(spec.max_iterations) + 1):
        log_v = log_column_marginal - torch.logsumexp(
            log_kernel + log_u[:, None], dim=0
        )
        log_u = log_row_marginal - torch.logsumexp(
            log_kernel + log_v[None, :], dim=1
        )
        plan = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
        row_residual = plan.sum(dim=1) - 1.0
        column_residual = plan.sum(dim=0) - float(tokens) / float(experts)
        row_error = float(torch.linalg.vector_norm(row_residual).item())
        column_error = float(torch.linalg.vector_norm(column_residual).item())
        iterations = iteration
        if row_error < float(spec.tolerance) and column_error < float(
            spec.tolerance
        ):
            converged = True
            break
    if not bool(torch.isfinite(plan).all().item()):
        raise RuntimeError("Selective Sinkhorn produced a non-finite transport plan.")
    return SinkhornTransportResult(
        plan=plan,
        iterations=iterations,
        converged=converged,
        row_error=row_error,
        column_error=column_error,
    )


def transport_topk_routes(
    transport: SinkhornTransportResult,
    *,
    top_k: int,
    route_scale: float,
) -> SelectiveSinkhornRoutes:
    """Select and renormalize the largest transport entries per token."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Selective Sinkhorn routing requires torch.")
    plan = transport.plan
    experts = int(plan.shape[1])
    if int(top_k) <= 0 or int(top_k) > experts:
        raise ValueError("top_k must be in [1, experts]")
    weights, indices = torch.topk(
        plan, k=int(top_k), dim=-1, largest=True, sorted=False
    )
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).tiny
    )
    weights = weights * float(route_scale)
    return SelectiveSinkhornRoutes(
        top_indices=indices,
        top_weights=weights,
        transport=transport,
    )


def _deterministic_u64(*, seed: int, batch_index: int, layer_name: str, tag: str) -> int:
    payload = f"{int(seed)}:{int(batch_index)}:{layer_name}:{tag}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class SelectiveSinkhornController:
    """Choose sparse training-time OT routes without consuming global RNG state."""

    def __init__(self, spec: SelectiveSinkhornSpec) -> None:
        spec.validate()
        self.spec = spec
        self._global_batch_index = 0
        self._training = False
        self._opportunities = 0
        self._applications = 0
        self._converged = 0
        self._last_iterations = 0
        self._last_row_error = 0.0
        self._last_column_error = 0.0

    def bind_batch(self, *, global_batch_index: int, training: bool) -> None:
        if int(global_batch_index) < 0:
            raise ValueError("global_batch_index must be non-negative")
        self._global_batch_index = int(global_batch_index)
        self._training = bool(training)

    def should_apply(self, layer_name: str) -> bool:
        name = str(layer_name).strip()
        if not self._training or not name:
            return False
        sample = _deterministic_u64(
            seed=int(self.spec.seed),
            batch_index=self._global_batch_index,
            layer_name=name,
            tag="branch",
        )
        return (float(sample) / float(1 << 64)) < float(self.spec.probability)

    def select(
        self,
        layer_name: str,
        score_logits: Any,
        native_top_indices: Any,
        native_top_weights: Any,
        *,
        valid_token_mask: Any | None,
        route_scale: float,
        training: bool,
    ) -> SelectiveSinkhornRoutes | None:
        if not self._training or not bool(training):
            return None
        self._opportunities += 1
        if not self.should_apply(layer_name):
            return None
        if not torch.is_tensor(score_logits) or score_logits.ndim != 2:
            raise ValueError("score_logits must have shape [tokens, experts]")
        if tuple(native_top_indices.shape) != tuple(native_top_weights.shape):
            raise ValueError("native route indices and weights must have equal shape")
        if tuple(native_top_indices.shape[:-1]) != tuple(score_logits.shape[:-1]):
            raise ValueError("native routes must share the token axis with logits")
        if valid_token_mask is None:
            valid = torch.ones(
                int(score_logits.shape[0]),
                device=score_logits.device,
                dtype=torch.bool,
            )
        else:
            valid = torch.as_tensor(
                valid_token_mask, device=score_logits.device, dtype=torch.bool
            ).reshape(-1)
            if tuple(valid.shape) != (int(score_logits.shape[0]),):
                raise ValueError("valid_token_mask must have shape [tokens]")
        if not bool(valid.any().item()):
            return None
        generator = None
        if float(self.spec.noise_scale) > 0.0:
            generator = torch.Generator(device=score_logits.device)
            generator.manual_seed(
                _deterministic_u64(
                    seed=int(self.spec.seed),
                    batch_index=self._global_batch_index,
                    layer_name=str(layer_name),
                    tag="noise",
                )
                % (1 << 63)
            )
        transport = selective_sinkhorn_transport(
            score_logits[valid],
            cost_mode=self.spec.cost_mode,
            entropy_regularization=self.spec.entropy_regularization,
            max_iterations=self.spec.max_iterations,
            tolerance=self.spec.tolerance,
            noise_scale=self.spec.noise_scale,
            generator=generator,
        )
        selected = transport_topk_routes(
            transport,
            top_k=int(native_top_indices.shape[-1]),
            route_scale=float(route_scale),
        )
        indices = native_top_indices.clone()
        weights = native_top_weights.clone()
        indices[valid] = selected.top_indices.to(dtype=indices.dtype)
        weights[valid] = selected.top_weights.to(dtype=weights.dtype)
        self._applications += 1
        self._converged += int(transport.converged)
        self._last_iterations = int(transport.iterations)
        self._last_row_error = float(transport.row_error)
        self._last_column_error = float(transport.column_error)
        return SelectiveSinkhornRoutes(
            top_indices=indices,
            top_weights=weights,
            transport=transport,
        )

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "moe_selective_sinkhorn_opportunities": int(self._opportunities),
            "moe_selective_sinkhorn_applications": int(self._applications),
            "moe_selective_sinkhorn_converged": int(self._converged),
            "moe_selective_sinkhorn_last_iterations": int(self._last_iterations),
            "moe_selective_sinkhorn_last_row_error": float(self._last_row_error),
            "moe_selective_sinkhorn_last_column_error": float(
                self._last_column_error
            ),
        }


__all__ = [
    "SELECTIVE_SINKHORN_COST_MODES",
    "SelectiveSinkhornController",
    "SelectiveSinkhornRoutes",
    "SelectiveSinkhornSpec",
    "SinkhornTransportResult",
    "selective_sinkhorn_transport",
    "transport_topk_routes",
]
