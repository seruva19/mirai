"""Stochastic per-step expert-subset routing primitives (Mirai variant of EMoE).

EMoE (arXiv 2509.21892) samples ``k_train`` experts per token from a larger
router-top-``k_ideal`` pool. Mirai instead bounds the unique expert working set
at layer-and-step granularity.

The Mirai variant is therefore per-STEP and per-LAYER: sample one expert subset
of size ``ceil(fraction * num_experts)`` from the layer's router-top-ranked
experts, weighted by routing mass (hot-biased), mask the router scores outside
the subset so tokens re-route within it, and anneal ``fraction -> 1.0`` over
``anneal_steps``. Exactly ``size`` unique experts are touched per layer per step,
so the working set shrinks by ``fraction``.

All primitives here are pure and torch-only. A subset is a no-op whenever the
effective fraction reaches 1.0 or the subset spans every expert.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def validate_subset_params(
    *,
    fraction: float,
    pool_factor: float,
    anneal_steps: int,
    kl_weight: float,
) -> tuple[float, float, int, float]:
    """Validate + normalize the ``expert_subset_*`` model params.

    ``fraction`` in (0, 1]; ``pool_factor`` >= 1; ``anneal_steps`` >= 0;
    ``kl_weight`` >= 0. Returns the normalized tuple.
    """
    f = float(fraction)
    pf = float(pool_factor)
    steps = int(anneal_steps)
    klw = float(kl_weight)
    if not (0.0 < f <= 1.0):
        raise ValueError("model.params.expert_subset_fraction must be in (0, 1].")
    if pf < 1.0:
        raise ValueError("model.params.expert_subset_pool_factor must be >= 1.")
    if steps < 0:
        raise ValueError("model.params.expert_subset_anneal_steps must be >= 0.")
    if klw < 0.0:
        raise ValueError(
            "model.params.expert_subset_router_kl_weight must be >= 0."
        )
    return f, pf, steps, klw


class RouterSubsetPolicy:
    """Immutable per-step expert-subset policy.

    ``from_model_params(...)`` returns ``None`` for the do-nothing default
    (``fraction >= 1.0``) so routers keep the byte-identical legacy fast path.
    """

    __slots__ = ("fraction", "pool_factor", "anneal_steps", "kl_weight")

    def __init__(
        self,
        *,
        fraction: float,
        pool_factor: float = 2.0,
        anneal_steps: int = 0,
        kl_weight: float = 0.0,
    ) -> None:
        f, pf, steps, klw = validate_subset_params(
            fraction=fraction,
            pool_factor=pool_factor,
            anneal_steps=anneal_steps,
            kl_weight=0.0 if kl_weight is None else kl_weight,
        )
        self.fraction = f
        self.pool_factor = pf
        self.anneal_steps = steps
        self.kl_weight = klw

    def is_noop(self) -> bool:
        return self.fraction >= 1.0

    @classmethod
    def from_model_params(cls, params: Any) -> "RouterSubsetPolicy | None":
        fraction = float(getattr(params, "expert_subset_fraction", 1.0))
        policy = cls(
            fraction=fraction,
            pool_factor=float(getattr(params, "expert_subset_pool_factor", 2.0)),
            anneal_steps=int(getattr(params, "expert_subset_anneal_steps", 0)),
            kl_weight=float(
                getattr(params, "expert_subset_router_kl_weight", 0.0)
            ),
        )
        return None if policy.is_noop() else policy

    def effective_fraction(
        self,
        step: int,
        *,
        initial_fraction: float | None = None,
    ) -> float:
        """Annealed fraction at ``step``: ``fraction`` -> 1.0 over anneal_steps."""
        initial = (
            self.fraction
            if initial_fraction is None
            else float(initial_fraction)
        )
        if not 0.0 < initial <= 1.0:
            raise ValueError("Expert-subset initial_fraction must be in (0, 1].")
        if self.anneal_steps <= 0:
            return initial
        progress = min(1.0, max(0.0, float(int(step)) / float(self.anneal_steps)))
        return initial + (1.0 - initial) * progress

    def subset_size(
        self,
        *,
        num_experts: int,
        top_k: int,
        step: int,
        initial_fraction: float | None = None,
    ) -> int:
        """Number of experts in the subset at ``step``.

        Clamped to ``[top_k, num_experts]`` -- a subset smaller than ``top_k``
        could not supply ``top_k`` distinct experts per token. A size that
        reaches ``num_experts`` means "inactive" (spans every expert).
        """
        n = int(num_experts)
        k = max(1, int(top_k))
        frac = self.effective_fraction(
            step,
            initial_fraction=initial_fraction,
        )
        size = int(math.ceil(frac * n))
        return max(k, min(n, size))

    def is_active(self, *, num_experts: int, top_k: int, step: int) -> bool:
        return self.subset_size(
            num_experts=num_experts, top_k=top_k, step=step
        ) < int(num_experts)


def routing_mass_from_selection(
    scores: Any,
    top_indices: Any,
    *,
    num_experts: int,
) -> Any:
    """Per-expert routing mass ``[num_experts]`` from the natural top-k selection.

    ``mass[e]`` = sum over tokens of the router score gathered at the token's
    top-k slots that landed on expert ``e``. Detached; this is only a sampling
    weight ("router-top-ranked experts weighted by routing mass").
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("routing_mass_from_selection requires torch.")
    idx = top_indices.detach().reshape(-1).to(torch.long)
    gathered = scores.detach().gather(1, top_indices.detach().to(torch.long))
    mass = scores.new_zeros(int(num_experts), dtype=torch.float32)
    mass.scatter_add_(0, idx, gathered.reshape(-1).float())
    return mass


def select_expert_subset(
    routing_mass: Any,
    *,
    size: int,
    pool_factor: float,
    generator: Any | None = None,
) -> Any:
    """Hot-biased sample of ``size`` distinct experts from the top pool.

    The candidate pool is the top ``ceil(pool_factor * size)`` experts by mass;
    ``size`` are then drawn WITHOUT replacement, weighted by mass (hot experts
    more likely, warm tail still reachable). ``pool_factor == 1`` degenerates to
    the deterministic hottest-``size``. Returns a sorted ``LongTensor`` of ids.
    Deterministic given ``generator``.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("select_expert_subset requires torch.")
    mass = routing_mass.detach().float().reshape(-1)
    num_experts = int(mass.numel())
    if num_experts == 0:
        raise ValueError("routing_mass must contain at least one expert.")
    finite = torch.isfinite(mass)
    if not bool(finite.all()):
        invalid_count = int((~finite).sum().item())
        raise ValueError(
            "routing_mass must be finite; "
            f"received {invalid_count} non-finite value(s)."
        )
    size = max(1, min(int(size), num_experts))
    pool_size = int(math.ceil(float(pool_factor) * size))
    pool_size = max(size, min(num_experts, pool_size))
    # Top pool by mass (ties broken by index via topk's stable-ish order).
    pool_ids = torch.topk(mass, pool_size, sorted=False)[1]
    pool_mass = mass.index_select(0, pool_ids)
    # Strictly-positive weights so cold pool members remain reachable and
    # multinomial never sees an all-zero row.
    weights = pool_mass.clamp_min(0.0) + 1e-9
    if size >= pool_size:
        chosen = pool_ids
    else:
        picks = torch.multinomial(
            weights, size, replacement=False, generator=generator
        )
        chosen = pool_ids.index_select(0, picks)
    return torch.sort(chosen)[0].to(torch.long)


def subset_mask_from_ids(num_experts: int, subset_ids: Any, *, device: Any = None) -> Any:
    """Boolean ``[num_experts]`` mask: True for experts inside the subset."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("subset_mask_from_ids requires torch.")
    ids = subset_ids.detach().to(torch.long).reshape(-1)
    dev = device if device is not None else ids.device
    mask = torch.zeros(int(num_experts), dtype=torch.bool, device=dev)
    mask[ids.to(dev)] = True
    return mask


def subset_reverse_kl(logits: Any, subset_ids: Any) -> Any:
    """Reverse-KL router-consistency term ``KL(q || p)`` mean over tokens.

    ``p = softmax(logits)`` (full, detached); ``q = softmax(logits masked to the
    subset)``. Reverse (not forward) KL per EMoE: forward-KL gradients blow up as
    ``q -> 0``; reverse-KL sharpens without excessive concentration, preserving
    stable top-k rankings. Differentiable through ``logits`` (the router). Only
    meaningful when the router is trainable.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("subset_reverse_kl requires torch.")
    import torch.nn.functional as F

    full = logits.float()
    log_p = F.log_softmax(full, dim=-1).detach()
    mask = subset_mask_from_ids(int(full.shape[-1]), subset_ids, device=full.device)
    neg_inf = torch.finfo(full.dtype).min
    masked = full.masked_fill(~mask.unsqueeze(0), neg_inf)
    log_q = F.log_softmax(masked, dim=-1)
    q = log_q.exp()
    # Only subset entries contribute (q == 0 elsewhere); nan_to_num guards the
    # -inf * 0 slots from the masked columns.
    kl = (q * (log_q - log_p)).nan_to_num(0.0, 0.0, 0.0).sum(dim=-1)
    return kl.mean()
