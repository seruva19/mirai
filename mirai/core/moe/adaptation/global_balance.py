"""Global-batch MoE load-balance accumulation (single owner).

Qwen-style *global-batch load balancing* for the token-fraction load-balance
signal. The auxiliary load-balance objective normally estimates each expert's
load fraction ``f_i`` from a single micro-batch. Under gradient accumulation the
optimizer only steps once per ``gradient_accumulation`` micro-batches, so the
per-micro-batch estimate sees a fraction of the tokens a true global batch would
see. This module accumulates the per-expert dispatch counts across the whole
accumulation window and exposes the *accumulated* load fraction, so the balance
signal matches what one large global batch would compute.

Scope key (``model.params.moe_balance_scope``):

* ``microbatch`` (default) — compute the estimate independently per micro-batch;
  the accumulator is not consulted.
* ``global_batch`` — the token-fraction load ``f_i`` fed into the aux term (and,
  conceptually, the online bias) is taken from the counts accumulated across the
  accumulation window.

Invariant: at ``gradient_accumulation == 1`` the accumulated counts equal the
single micro-batch's counts, so ``global_batch`` reduces *exactly* to
``microbatch`` (see tests). The accumulator is reset at every optimizer-step
boundary so each window starts fresh.

Composition with ``moe_balance_mode``:

* ``aux_loss`` — the accumulated fraction reweights the (detached) load term of
  the token-fraction aux loss; the differentiable mean-probability factor stays
  per-micro-batch so gradients still flow from the current micro-batch.
* ``bias_only`` — the online bias update already sums per-expert counts across
  the window (one signed update per optimizer step), i.e. it is already a
  global-batch statistic; the scope key leaves that path unchanged.
* ``off`` — inert (no balance pressure at all).

With the sequence-wise auxiliary form, ``microbatch`` preserves the native
per-sequence frequency exactly. Selecting ``global_batch`` intentionally
replaces that local frequency with the accumulation-window frequency while
retaining the current micro-batch's differentiable mean-probability term. This
makes the scope explicit and effective regardless of the provider's native
auxiliary-loss form.

The module owns one single-GPU accumulation window.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


MOE_BALANCE_SCOPES = ("microbatch", "global_batch")


def normalize_moe_balance_scope(value: str | None) -> str:
    """Normalise/validate ``model.params.moe_balance_scope``."""
    text = str(value if value is not None else "microbatch").strip().lower()
    aliases = {"": "microbatch", "micro": "microbatch", "global": "global_batch"}
    normalized = aliases.get(text, text)
    if normalized not in MOE_BALANCE_SCOPES:
        raise ValueError(
            "model.params.moe_balance_scope must be one of: "
            + ", ".join(MOE_BALANCE_SCOPES)
            + "."
        )
    return normalized


def dispatch_counts(indices: Any, num_experts: int) -> Any:
    """Per-expert selection counts for one micro-batch (detached float vector).

    ``indices`` is the router's selected top-k expert index tensor; every
    selection contributes one count. Matches the ``global`` aux-loss form's
    ``bincount(selected)`` exactly so the accumulated single-window fraction is
    bit-identical to the per-micro-batch fraction at accumulation == 1.
    """
    flat = indices.detach().reshape(-1)
    return torch.bincount(flat, minlength=int(num_experts)).float()


class GlobalBatchLoadAccumulator:
    """Per-layer accumulated expert dispatch counts over an accumulation window.

    Keyed by a stable per-layer key (the router module's qualified name). Holds
    detached count vectors only — never part of any autograd graph.
    """

    def __init__(self) -> None:
        self._counts: dict[str, Any] = {}

    def reset(self) -> None:
        """Clear the window (call at each optimizer-step boundary)."""
        self._counts = {}

    def accumulate(self, key: str, counts: Any) -> None:
        """Add one micro-batch's per-expert counts to the window for ``key``."""
        value = counts.detach().float()
        previous = self._counts.get(key)
        self._counts[key] = value.clone() if previous is None else previous + value

    def has(self, key: str) -> bool:
        return key in self._counts

    def fraction(self, key: str) -> Any:
        """Accumulated load fraction ``f_i`` for ``key`` (or ``None`` if unseen).

        Normalises by the summed count with a ``max(1, .)`` floor so it matches
        the per-micro-batch ``bincount / max(1, numel)`` form.
        """
        counts = self._counts.get(key)
        if counts is None:
            return None
        total = counts.sum()
        return counts / total.clamp_min(1.0)


__all__ = [
    "MOE_BALANCE_SCOPES",
    "GlobalBatchLoadAccumulator",
    "dispatch_counts",
    "normalize_moe_balance_scope",
]
