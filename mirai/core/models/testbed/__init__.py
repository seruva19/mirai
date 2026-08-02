"""Executable CPU contract fixture for sparse-MoE trainer semantics.

Importing this package registers the explicitly selected ``sparse_moe_test``
model family and its ``ModelFamilyProvider``.
"""

from __future__ import annotations

from mirai.core.models.testbed.sparse_moe import (
    SparseMoETestPipeline,
    SparseMoETransformerBlock,
    TinySparseMoEDenoiser,
    VALID_VARIANTS,
)

__all__ = [
    "SparseMoETestPipeline",
    "SparseMoETransformerBlock",
    "TinySparseMoEDenoiser",
    "VALID_VARIANTS",
]
