"""Vendored woct0rdho persistent sorted-contiguous grouped GEMM (Apache-2.0).

Source: https://github.com/woct0rdho/transformers-qwen3-moe-fused,
``qwen3_moe_fused/grouped_gemm`` at the 2026-07-13 snapshot. Import surface:
``grouped_gemm`` (autograd Function), ``grouped_gemm_forward``, and
``grouped_gemm_backward_dw``. Mirai's provider-neutral pre-run warm-up is
derived from upstream pull request 21. See the adjacent Apache-2.0 LICENSE.
"""

from .interface import grouped_gemm  # noqa: F401
from .forward import grouped_gemm_forward  # noqa: F401
from .backward_dw import grouped_gemm_backward_dw  # noqa: F401
