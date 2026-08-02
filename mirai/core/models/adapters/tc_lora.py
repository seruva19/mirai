"""TC-LoRA: hypernetwork-generated, timestep-conditioned LoRA gate modulation.

TC-LoRA (arXiv 2510.09561, NVIDIA Research Taiwan, NeurIPS-W 2025 "Temporally
Modulated Conditional LoRA") replaces a static adapter with a hypernetwork that
generates the adapter's effect on the frozen backbone conditioned on the
diffusion timestep (and, in the paper, the guidance/condition). Mirai adopts the
**gate-modulation** variant of this idea:

- A tiny shared hypernetwork maps the post-shift sigma to a per-rank GATE vector
  ``g(sigma) in R^r`` applied MULTIPLICATIVELY on the low-rank intermediate
  (``x @ A^T``), exactly at the seam where the T-LoRA rank schedule / timestep
  band already multiply their rank mask. It strictly generalizes T-LoRA's binary
  rank-prefix mask to a learned, smooth, timestep-conditioned gate.

For stacked routed-expert LoRA (``lora_a`` is ``[E, rank, in]``), full per-step
A/B generation scales with every expert matrix. The shared gate emits ``rank``
scalars and uses the adapter mask-distribution path.

Init-to-identity (hard requirement): the output projection is zero-initialized
and the gate is ``1.0 + delta``, so at init ``g == 1.0`` EXACTLY. The modulated
LoRA is then bit-identical to the static LoRA, so the feature composes with
orthogonal init, timestep bands, and rank schedules without perturbing defaults.

Composition (multiplicative on the rank axis)::

    final_mask = band_mask * schedule_mask * tc_gate

All three factors multiply the same low-rank intermediate. Because the gate is a
multiplicative factor it can NEVER resurrect a rank column (T-LoRA prefix) or a
sample row (timestep band) that was already driven to exactly zero -- see
``combine_gate_with_mask``. The precedence is therefore commutative and safe.

This module is model-agnostic and imports only torch. The owning pipeline is
responsible for attachment, device placement, optimizer registration and
checkpoint I/O. Conditioning is extensible: ``forward`` accepts an optional
``cond`` vector (guidance scale, class embedding, ...) concatenated to the sigma
embedding via the ``cond_dim`` slot.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


# Sinusoidal sigma-embedding width. Small and fixed: the hypernetwork's capacity
# knob is ``hidden_dim``, not the embedding width.
SIGMA_EMBED_DIM = 16
# Sigma lives in [0, 1]; spread it across the sinusoidal frequency band the same
# way diffusion timestep embeddings spread t in [0, 1000].
_SIGMA_EMBED_SCALE = 1000.0


def sigma_embedding(sigmas: Any, dim: int = SIGMA_EMBED_DIM) -> Any:
    """Sinusoidal embedding ``[B, dim]`` of post-shift sigma ``[B]`` in [0, 1].

    Deterministic and parameter-free so the learned capacity sits entirely in the
    hypernetwork projections.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("sigma_embedding requires torch.")
    sig = sigmas.reshape(-1).float()
    half = int(dim) // 2
    if half < 1:
        return sig.reshape(-1, 1)
    exponents = torch.arange(half, device=sig.device, dtype=torch.float32)
    freqs = torch.exp(-math.log(10000.0) * exponents / float(half))
    args = sig.reshape(-1, 1) * freqs.reshape(1, -1) * _SIGMA_EMBED_SCALE
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if emb.shape[-1] < int(dim):  # odd dim -> pad the final column with zeros
        pad = torch.zeros(
            emb.shape[0], int(dim) - emb.shape[-1], device=sig.device, dtype=emb.dtype
        )
        emb = torch.cat([emb, pad], dim=-1)
    return emb


def combine_gate_with_mask(mask: Any, gate: Any) -> Any:
    """Multiplicative composition of a rank ``mask`` with the tc ``gate``.

    ``mask`` is the ``band_mask * schedule_mask`` product (``[B, rank]`` or
    ``[rank]``) produced by the timestep-axis substrate; ``gate`` is the
    hypernetwork's ``[B, rank]`` (or ``[rank]``) output. The result is their
    elementwise product, so any exactly-zero entry in ``mask`` (a masked rank or
    an out-of-band sample) stays exactly zero regardless of the gate.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("combine_gate_with_mask requires torch.")
    return mask * gate.to(device=mask.device, dtype=mask.dtype)


def gate_summary(gate: Any) -> dict[str, float]:
    """Detached mean/std of the gate (collapse / saturation telemetry)."""
    if torch is None:  # pragma: no cover
        return {}
    g = gate.detach().float()
    mean = float(g.mean().cpu().item())
    std = float(g.std(unbiased=False).cpu().item()) if g.numel() > 1 else 0.0
    return {"tc_gate_mean": mean, "tc_gate_std": std}


class TimestepGateHypernet(nn.Module):
    """Tiny hypernetwork: post-shift sigma -> per-rank LoRA gate ``g(sigma)``.

    ``forward`` returns ``[B, rank]`` initialized to EXACTLY 1.0 (identity gate).
    The gate is ``1.0 + out_proj(silu(in_proj(embed(sigma))))`` with ``out_proj``
    zero-initialized; the parameters are ordinary trainable ``nn.Parameter`` so
    the pipeline registers them with the optimizer like any other adapter param.
    """

    def __init__(
        self,
        *,
        rank: int,
        hidden_dim: int,
        embed_dim: int = SIGMA_EMBED_DIM,
        cond_dim: int = 0,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("TimestepGateHypernet requires torch.")
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("TC-LoRA gate rank must be > 0.")
        if int(hidden_dim) <= 0:
            raise ValueError("TC-LoRA gate hidden_dim must be > 0.")
        if int(cond_dim) < 0:
            raise ValueError("TC-LoRA gate cond_dim must be >= 0.")
        self.rank = int(rank)
        self.embed_dim = int(embed_dim)
        self.cond_dim = int(cond_dim)
        self.in_proj = nn.Linear(self.embed_dim + self.cond_dim, int(hidden_dim))
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(int(hidden_dim), self.rank)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.in_proj.bias)
        # Zero output projection -> delta == 0 -> gate == 1.0 EXACTLY at init.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, sigmas: Any, cond: Any | None = None) -> Any:
        emb = sigma_embedding(sigmas, self.embed_dim).to(
            device=self.in_proj.weight.device, dtype=self.in_proj.weight.dtype
        )
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError(
                    "TC-LoRA gate was built with cond_dim > 0 but no cond vector "
                    "was provided."
                )
            cond_t = cond.to(device=emb.device, dtype=emb.dtype).reshape(
                emb.shape[0], self.cond_dim
            )
            emb = torch.cat([emb, cond_t], dim=-1)
        elif cond is not None:
            raise ValueError(
                "TC-LoRA gate was built with cond_dim == 0; pass cond_dim > 0 to "
                "condition on an extra vector."
            )
        delta = self.out_proj(self.act(self.in_proj(emb)))
        return 1.0 + delta
