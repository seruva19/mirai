"""Structured dropout over LoRA factor parameters.

The LoRA Dropout formulation masks input-feature columns of ``A`` and
output-feature rows of ``B`` independently.  It deliberately does not mask the
shared rank axis: doing so would reduce the effective adapter rank instead of
inducing structured sparsity in the reconstructed update.

Reference: https://arxiv.org/abs/2404.09610, Section 3.2, Equation 2.
"""

from __future__ import annotations

import math
from typing import Any


def validate_lora_parameter_dropout(probability: float) -> float:
    """Return a finite dropout probability in the supported interval."""

    value = float(probability)
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError("adapter.lora_parameter_dropout must be in [0, 1).")
    return value


def apply_lora_parameter_dropout(
    lora_a: Any,
    lora_b: Any,
    *,
    probability: float,
    training: bool,
) -> tuple[Any, Any]:
    """Return factors with paper-defined structured Bernoulli masks.

    ``lora_a`` has shape ``[..., rank, in_features]`` and ``lora_b`` has shape
    ``[..., out_features, rank]``.  One mask value is sampled per input column
    of ``A`` and per output row of ``B`` for every leading adapter instance.
    The paper uses unscaled Bernoulli masks, so this function does not apply
    inverted-dropout compensation.

    Disabled and evaluation paths return the original tensor objects and do not
    consume RNG state.
    """

    value = validate_lora_parameter_dropout(probability)
    if not bool(training) or value == 0.0:
        return lora_a, lora_b
    if lora_a.dim() < 2 or lora_b.dim() < 2:
        raise ValueError("LoRA factors must have at least two dimensions.")
    if tuple(lora_a.shape[:-2]) != tuple(lora_b.shape[:-2]):
        raise ValueError("LoRA factors must share the same leading dimensions.")
    if int(lora_a.shape[-2]) != int(lora_b.shape[-1]):
        raise ValueError("LoRA factors must share the same rank dimension.")

    prefix = tuple(int(dim) for dim in lora_a.shape[:-2])
    input_mask = lora_a.new_empty((*prefix, 1, int(lora_a.shape[-1])))
    output_mask = lora_b.new_empty((*prefix, int(lora_b.shape[-2]), 1))
    input_mask.bernoulli_(1.0 - value)
    output_mask.bernoulli_(1.0 - value)
    return lora_a * input_mask, lora_b * output_mask
