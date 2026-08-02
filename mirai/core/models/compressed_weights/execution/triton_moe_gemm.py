"""Triton grouped-GEMM for the sparse-MoE expert dispatch (opt-in).

This module provides a *padded-layout, count-aware* grouped GEMM used by the
configured ``triton`` path in :mod:`compressed_weights`. The dispatch keeps the
exact same ``[E, max_count, H]`` padded batch layout as the vectorized path (so
the LoRA adapter path and the ``index_add`` scatter are byte-for-byte
unchanged), and only the *frozen* expert linear ``bmm(x, W^T)`` is replaced.

The win over ``torch.bmm`` (a single cuBLAS batched-GEMM) comes from
*count-aware tile skipping*: within a chunk the experts are padded to the chunk
maximum token count, and this kernel skips the ``m``-tiles that fall entirely in
the padding region of each expert. On balanced routing this is roughly a wash;
on imbalanced routing (some experts far below the chunk max) it removes the
padding compute.

The kernels operate on dequantized bf16 weights (dequant stays in
``compressed_weights``). Backward re-runs the transposed grouped GEMM against
re-dequantized weights (no bf16 weight stack is saved), preserving the
no-saved-bf16-stack invariant. Weights are frozen; LoRA adapters keep their own
autograd path.

Windows note: triton-windows falls through to whatever C compiler is on PATH for
its host stubs; if none of MSVC/env-MSVC is set it can pick an incompatible gcc.
:func:`ensure_cc_env` points ``CC`` at the bundled TinyCC when unset.
"""

from __future__ import annotations

import functools
import os

try:  # torch is always present in the trainer; guard for import-time safety.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


_MIN_CC_MAJOR = 8  # SM80+ (Ampere) required for the bf16 tensor-core path.


def ensure_cc_env() -> None:
    """On Windows, default ``CC`` to triton's bundled TinyCC when unset.

    triton-windows' ``get_cc()`` order falls through to ``gcc``/``clang`` on
    PATH when neither a ``CC`` env var, an env-configured MSVC, nor a bundled
    TinyCC in the *system* site-packages is found. When triton is installed into
    the *user* site-packages the bundled ``tcc.exe`` is present but not on the
    search path get_cc() uses, so it wrongly picks gcc (which cannot handle the
    ``\\\\?\\`` long-path prefix or MSVC ``.lib`` link inputs). Pointing ``CC``
    at the bundled tcc.exe makes the host-stub compile self-contained.
    """
    if os.name != "nt" or os.environ.get("CC"):
        return
    try:
        import triton  # noqa: F401
    except Exception:
        return
    try:
        base = os.path.dirname(os.path.abspath(triton.__file__))
    except Exception:
        return
    tcc = os.path.join(base, "runtime", "tcc", "tcc.exe")
    if os.path.exists(tcc):
        os.environ["CC"] = tcc


@functools.lru_cache(maxsize=1)
def _import_triton():
    ensure_cc_env()
    import triton
    import triton.language as tl

    return triton, tl


def autotune_mode() -> str:
    """Return the deterministic default autotune search space."""

    return "conservative"


def _smem_bytes(bm: int, bn: int, bk: int, stages: int, *, elt: int = 2) -> int:
    """bf16 double-buffered shared-memory estimate for one config.

    ``(stages-1)`` copies of both operand tiles are kept resident by the triton
    pipeliner. Mirrors ``exceeds_smem_capacity`` in woct0rdho's autotuning.py.
    """
    x_size = bm * bk * elt
    w_size = bn * bk * elt
    if stages <= 1:
        return max(x_size, w_size)
    return (stages - 1) * (x_size + w_size)


def build_autotune_configs(triton, *, mode: str | None = None, smem_limit: int | None = None):
    """Return the ``triton.Config`` list for the resolved autotune ``mode``.

    Module-level (not closed over ``_kernels``) so the pure config-generation
    logic is unit-testable without compiling a kernel. ``smem_limit`` defaults to
    a conservative 100 KB SM86 ceiling when the device cannot be queried.
    """
    mode = mode or autotune_mode()
    if mode != "extended":
        cfgs = []
        for bm, bn in ((64, 64), (128, 64), (64, 128)):
            for stages in (2, 3):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 32},
                        num_stages=stages,
                        num_warps=4,
                    )
                )
        return cfgs
    if smem_limit is None:
        try:
            props = triton.runtime.driver.active.utils.get_device_properties(
                torch.cuda.current_device()
            )
            smem_limit = int(props["max_shared_mem"])
        except Exception:
            smem_limit = 100 * 1024
    cfgs = []
    # SM86-safe tile range: cap BM*BN at 128*256 (larger is almost always useless
    # per their prune) and BK at 128; sweep stages 2..4, warps {4,8}. Configs over
    # the shared-memory ceiling are dropped.
    for bm in (32, 64, 128, 256):
        for bn in (32, 64, 128, 256):
            if bm * bn > 128 * 256:
                continue
            for bk in (32, 64, 128):
                for stages in (2, 3, 4):
                    if _smem_bytes(bm, bn, bk, stages) > smem_limit:
                        continue
                    for warps in (4, 8):
                        cfgs.append(
                            triton.Config(
                                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                                num_stages=stages,
                                num_warps=warps,
                            )
                        )
    if not cfgs:  # pragma: no cover - defensive: fall back to the safe list
        cfgs = [
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
                num_stages=2,
                num_warps=4,
            )
        ]
    return cfgs


@functools.lru_cache(maxsize=None)
def _kernels():
    """Build (and cache) the two grouped-GEMM kernels lazily.

    Kept out of module scope so importing this module never triggers a triton
    import / compile; callers gate on :func:`is_available` first.
    """
    triton, tl = _import_triton()

    def _configs():
        return build_autotune_configs(triton)

    @triton.autotune(configs=_configs(), key=["N", "K"])
    @triton.jit
    def grouped_nt_kernel(
        x_ptr,
        w_ptr,
        counts_ptr,
        out_ptr,
        M,
        N,
        K,
        s_xe,
        s_xm,
        s_xk,
        s_we,
        s_wn,
        s_wk,
        s_oe,
        s_om,
        s_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # out[e, m, n] = sum_k x[e, m, k] * w[e, n, k]   (x @ w^T per expert)
        pid_e = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        count = tl.load(counts_ptr + pid_e)
        if pid_m * BLOCK_M >= count:
            return  # whole tile is padding for this expert -> skip
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        m_mask = rm < count
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        x_base = x_ptr + pid_e * s_xe
        w_base = w_ptr + pid_e * s_we
        for k0 in range(0, K, BLOCK_K):
            k = k0 + rk
            k_mask = k < K
            a = tl.load(
                x_base + rm[:, None] * s_xm + k[None, :] * s_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                w_base + rn[None, :] * s_wn + k[:, None] * s_wk,
                mask=(rn[None, :] < N) & k_mask[:, None],
                other=0.0,
            )
            acc += tl.dot(a, b)
        out_base = out_ptr + pid_e * s_oe
        tl.store(
            out_base + rm[:, None] * s_om + rn[None, :] * s_on,
            acc.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & (rn[None, :] < N),
        )

    @triton.autotune(configs=_configs(), key=["N", "K"])
    @triton.jit
    def grouped_grad_kernel(
        g_ptr,
        w_ptr,
        counts_ptr,
        out_ptr,
        M,
        N,
        K,
        s_ge,
        s_gm,
        s_gn,
        s_we,
        s_wn,
        s_wk,
        s_oe,
        s_om,
        s_ok,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # grad_in[e, m, k] = sum_n grad_out[e, m, n] * w[e, n, k]   (g @ w)
        # Here BLOCK_N tiles the output feature (K of the weight), BLOCK_K the
        # contraction over N.
        pid_e = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        count = tl.load(counts_ptr + pid_e)
        if pid_m * BLOCK_M >= count:
            return
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rk = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # over K (weight cols)
        rn = tl.arange(0, BLOCK_K)  # contraction over N
        m_mask = rm < count
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        g_base = g_ptr + pid_e * s_ge
        w_base = w_ptr + pid_e * s_we
        for n0 in range(0, N, BLOCK_K):
            n = n0 + rn
            n_mask = n < N
            a = tl.load(
                g_base + rm[:, None] * s_gm + n[None, :] * s_gn,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                w_base + n[:, None] * s_wn + rk[None, :] * s_wk,
                mask=n_mask[:, None] & (rk[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(a, b)
        out_base = out_ptr + pid_e * s_oe
        tl.store(
            out_base + rm[:, None] * s_om + rk[None, :] * s_ok,
            acc.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & (rk[None, :] < K),
        )

    return grouped_nt_kernel, grouped_grad_kernel


@functools.lru_cache(maxsize=None)
def _probe(cc_major: int, cc_minor: int) -> bool:
    """Compile+run a tiny grouped GEMM once per device capability."""
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        dev = torch.device("cuda")
        x = torch.randn(2, 4, 8, device=dev, dtype=torch.bfloat16)
        w = torch.randn(2, 6, 8, device=dev, dtype=torch.bfloat16)
        counts = torch.tensor([4, 2], device=dev, dtype=torch.int32)
        out = grouped_linear_nt(x, w, counts)
        ref = torch.bmm(x.float(), w.float().transpose(-2, -1))
        # Only the valid (non-padding) rows must match.
        ok = torch.allclose(
            out[0].float(), ref[0], rtol=1e-2, atol=1e-2
        ) and torch.allclose(out[1, :2].float(), ref[1, :2], rtol=1e-2, atol=1e-2)
        return bool(ok)
    except Exception:
        return False


def is_available(device: "torch.device | None" = None) -> bool:
    """True if triton imports, compiles, and the device is SM80+.

    Cached per device capability; a compile/run failure returns False so the
    caller silently falls back to the vectorized path.
    """
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        idx = None
        if device is not None and device.type == "cuda":
            idx = device.index
        major, minor = torch.cuda.get_device_capability(idx)
    except Exception:
        return False
    if major < _MIN_CC_MAJOR:
        return False
    try:
        _import_triton()
    except Exception:
        return False
    return _probe(int(major), int(minor))


def grouped_linear_nt(
    x: "torch.Tensor", weight: "torch.Tensor", counts: "torch.Tensor"
) -> "torch.Tensor":
    """out[e] = x[e] @ weight[e]^T with per-expert row counts (padding skipped).

    x: ``[E, M, K]``, weight: ``[E, N, K]``, counts: ``[E]`` (int). Returns
    ``[E, M, N]`` zero-initialized so skipped padding rows are exact zeros.
    """
    triton, _ = _import_triton()
    grouped_nt_kernel, _grad = _kernels()
    E, M, K = x.shape
    N = weight.shape[1]
    x = x.contiguous()
    weight = weight.contiguous()
    counts = counts.to(device=x.device, dtype=torch.int32)
    out = torch.zeros((E, M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (  # noqa: E731
        E,
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    grouped_nt_kernel[grid](
        x,
        weight,
        counts,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


def grouped_linear_grad(
    grad_out: "torch.Tensor", weight: "torch.Tensor", counts: "torch.Tensor"
) -> "torch.Tensor":
    """grad_in[e] = grad_out[e] @ weight[e] with per-expert row counts.

    grad_out: ``[E, M, N]``, weight: ``[E, N, K]``, counts: ``[E]``. Returns
    ``[E, M, K]`` zero-initialized (padding rows exact zero).
    """
    triton, _ = _import_triton()
    _nt, grouped_grad_kernel = _kernels()
    E, M, N = grad_out.shape
    K = weight.shape[2]
    grad_out = grad_out.contiguous()
    weight = weight.contiguous()
    counts = counts.to(device=grad_out.device, dtype=torch.int32)
    out = torch.zeros((E, M, K), device=grad_out.device, dtype=grad_out.dtype)
    grid = lambda meta: (  # noqa: E731
        E,
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(K, meta["BLOCK_N"]),
    )
    grouped_grad_kernel[grid](
        grad_out,
        weight,
        counts,
        out,
        M,
        N,
        K,
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out
