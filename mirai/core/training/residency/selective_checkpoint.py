"""Operator-selective activation checkpoint policy."""

from __future__ import annotations

from functools import partial


_SAVE_OP_NAMES = frozenset(
    {
        "aten.mm.default",
        "aten.bmm.default",
        "aten.addmm.default",
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
    }
)


def make_selective_checkpoint_context_fn():
    """Cache expensive matrix/attention ops and recompute elementwise work."""
    try:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )
    except ImportError as exc:  # pragma: no cover - torch-version guard
        raise RuntimeError(
            "gradient_checkpointing='selective' requires "
            "torch.utils.checkpoint.create_selective_checkpoint_contexts."
        ) from exc

    def policy_fn(ctx, op, *args, **kwargs):
        _ = ctx, args, kwargs
        name = str(op)
        if name in _SAVE_OP_NAMES:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return partial(create_selective_checkpoint_contexts, policy_fn)
