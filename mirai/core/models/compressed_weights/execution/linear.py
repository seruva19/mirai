"""Autograd Functions, dispatch resolution, and ``CompressedLinear`` (pure move)."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from mirai.core.moe.runtime.specs import resolve_moe_dispatch_mode

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


from ..quantization.quant import (
    NF4_BLOCKSIZE,
    _Nf4Meta,
    _dequantize_weight,
    _nf4_dequantize,
    _nf4_quantize_2d,
    _quantize_weight,
    best_group_size,
    normalize_quant_format,
)
from ..quantization.microscaling_quant import (
    MICROSCALING_FORMATS,
    MicroscalingMeta,
    dequantize_microscaling,
    quantize_microscaling,
    validate_microscaling_payload,
)
from ..quantization.blockwise_fp8 import (
    BLOCKWISE_FP8_FORMATS,
    BlockwiseFP8Meta,
    blockwise_fp8_linear,
    dequantize_blockwise_fp8_weight,
    quantize_blockwise_fp8_weight,
    validate_blockwise_fp8_payload,
)

logger = logging.getLogger(__name__)


if torch is not None:
    class _FrozenCompressedLinear(torch.autograd.Function):
        """Dense frozen projection that re-materializes its weight in backward."""

        @staticmethod
        def forward(ctx, x, owner):
            weight = owner._materialize_weight(dtype=x.dtype, device=x.device)
            output = F.linear(x, weight)
            ctx.owner = owner
            ctx.input_dtype = x.dtype
            return output

        @staticmethod
        def backward(ctx, grad_output):
            weight = ctx.owner._materialize_weight(
                dtype=ctx.input_dtype,
                device=grad_output.device,
            )
            grad_input = grad_output.to(dtype=ctx.input_dtype) @ weight
            return grad_input, None


    class _FrozenDequantExpertLinear(torch.autograd.Function):
        """Per-expert frozen projection without retaining its dense weight."""

        @staticmethod
        def forward(ctx, x, weight, owner, key, expert_idx):
            output = x @ weight.transpose(-2, -1)
            ctx.owner = owner
            ctx.key = str(key)
            ctx.expert_idx = int(expert_idx)
            ctx.input_dtype = x.dtype
            return output

        @staticmethod
        def backward(ctx, grad_output):
            weight = ctx.owner._dequantize_expert(
                ctx.key,
                ctx.expert_idx,
                dtype=ctx.input_dtype,
                device=grad_output.device,
            )
            grad_input = grad_output.to(dtype=ctx.input_dtype) @ weight
            return grad_input, None, None, None, None


    class _FrozenDequantBatchedLinear(torch.autograd.Function):
        """Batched linear against frozen quantized experts, re-dequantized in backward.

        A plain ``torch.bmm`` against a dequantized weight stack makes autograd
        keep the full bf16 stack alive until the block's backward completes;
        re-dequantizing from the module's quantized buffers in backward keeps
        only per-chunk transients on the GPU.
        """

        @staticmethod
        def forward(ctx, x, owner, key, expert_indices):
            weight = owner._dequant_expert_stack(
                key, expert_indices, dtype=x.dtype, device=x.device
            )
            output = torch.bmm(x, weight.transpose(-2, -1))
            ctx.owner = owner
            ctx.key = key
            ctx.expert_indices = expert_indices
            ctx.weight_dtype = x.dtype
            return output

        @staticmethod
        def backward(ctx, grad_output):
            weight = ctx.owner._dequant_expert_stack(
                ctx.key,
                ctx.expert_indices,
                dtype=ctx.weight_dtype,
                device=grad_output.device,
            )
            grad_input = torch.bmm(grad_output.to(dtype=ctx.weight_dtype), weight)
            return grad_input, None, None, None


    class _FrozenDequantMegaBlocksLinear(torch.autograd.Function):
        """MegaBlocks grouped linear without retaining its floating weight stack."""

        @staticmethod
        def forward(ctx, x, owner, key, counts, grouped_gemm_ops):
            weight = owner._dequantize(str(key), dtype=x.dtype, device=x.device)
            batch_sizes = counts.detach().to(device="cpu", dtype=torch.int64)
            output = grouped_gemm_ops.gmm(x, weight, batch_sizes, trans_b=True)
            ctx.owner = owner
            ctx.key = str(key)
            ctx.counts = tuple(int(value) for value in batch_sizes.tolist())
            ctx.input_dtype = x.dtype
            ctx.input_shape = tuple(int(dim) for dim in x.shape)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            weight = ctx.owner._dequantize(
                ctx.key,
                dtype=ctx.input_dtype,
                device=grad_output.device,
            )
            grad_input = torch.empty(
                ctx.input_shape,
                dtype=ctx.input_dtype,
                device=grad_output.device,
            )
            offset = 0
            for expert_idx, count in enumerate(ctx.counts):
                end = offset + count
                if count:
                    grad_input[offset:end] = (
                        grad_output[offset:end].to(dtype=ctx.input_dtype)
                        @ weight[expert_idx]
                    )
                offset = end
            return grad_input, None, None, None, None

    class _FrozenDequantBatchedLinearPair(torch.autograd.Function):
        """Two frozen linears (same input ``x``) sharing one paired dequant.

        ``w1`` (gate) and ``w3`` (up) have identical shapes, so their quantized
        chunk stacks are dequantized together in a single batched call (see
        ``_dequant_expert_pair``) instead of one call per key. As with the
        single-key variant the dequantized bf16 weights are never saved on the
        ctx; they are re-dequantized in backward so nothing outlives the chunk's
        immediate use in autograd.
        """

        @staticmethod
        def forward(ctx, x, owner, key_a, key_b, expert_indices):
            weight_a, weight_b = owner._dequant_expert_pair(
                key_a, key_b, expert_indices, dtype=x.dtype, device=x.device
            )
            out_a = torch.bmm(x, weight_a.transpose(-2, -1))
            out_b = torch.bmm(x, weight_b.transpose(-2, -1))
            ctx.owner = owner
            ctx.key_a = key_a
            ctx.key_b = key_b
            ctx.expert_indices = expert_indices
            ctx.weight_dtype = x.dtype
            return out_a, out_b

        @staticmethod
        def backward(ctx, grad_out_a, grad_out_b):
            weight_a, weight_b = ctx.owner._dequant_expert_pair(
                ctx.key_a,
                ctx.key_b,
                ctx.expert_indices,
                dtype=ctx.weight_dtype,
                device=grad_out_a.device,
            )
            grad_input = torch.bmm(
                grad_out_a.to(dtype=ctx.weight_dtype), weight_a
            ) + torch.bmm(grad_out_b.to(dtype=ctx.weight_dtype), weight_b)
            return grad_input, None, None, None, None

    class _FrozenTritonGroupedLinear(torch.autograd.Function):
        """Frozen batched linear via the triton count-aware grouped GEMM.

        Same contract as ``_FrozenDequantBatchedLinear`` (dequant-then-linear
        against frozen experts, re-dequantized in backward) but the ``bmm`` is
        a padding-skipping grouped GEMM keyed on per-expert token ``counts``.
        Padding rows (``row >= counts[e]``) are exact zeros in the output.
        """

        @staticmethod
        def forward(ctx, x, owner, key, expert_indices, counts):
            from . import triton_moe_gemm as _tmg

            weight = owner._dequant_expert_stack(
                key, expert_indices, dtype=x.dtype, device=x.device
            )
            output = _tmg.grouped_linear_nt(x, weight, counts)
            ctx.owner = owner
            ctx.key = key
            ctx.expert_indices = expert_indices
            ctx.weight_dtype = x.dtype
            ctx.save_for_backward(counts)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            from . import triton_moe_gemm as _tmg

            (counts,) = ctx.saved_tensors
            weight = ctx.owner._dequant_expert_stack(
                ctx.key,
                ctx.expert_indices,
                dtype=ctx.weight_dtype,
                device=grad_output.device,
            )
            grad_input = _tmg.grouped_linear_grad(
                grad_output.to(dtype=ctx.weight_dtype), weight, counts
            )
            return grad_input, None, None, None, None

    class _FrozenTritonGroupedLinearPair(torch.autograd.Function):
        """Two frozen grouped linears (w1/w3) sharing one paired dequant.

        The triton analogue of ``_FrozenDequantBatchedLinearPair``: the paired
        bf16 dequant is shared, the two grouped GEMMs run count-aware, and the
        backward accumulates both grad-input contributions after re-dequant.
        """

        @staticmethod
        def forward(ctx, x, owner, key_a, key_b, expert_indices, counts):
            from . import triton_moe_gemm as _tmg

            weight_a, weight_b = owner._dequant_expert_pair(
                key_a, key_b, expert_indices, dtype=x.dtype, device=x.device
            )
            out_a = _tmg.grouped_linear_nt(x, weight_a, counts)
            out_b = _tmg.grouped_linear_nt(x, weight_b, counts)
            ctx.owner = owner
            ctx.key_a = key_a
            ctx.key_b = key_b
            ctx.expert_indices = expert_indices
            ctx.weight_dtype = x.dtype
            ctx.save_for_backward(counts)
            return out_a, out_b

        @staticmethod
        def backward(ctx, grad_out_a, grad_out_b):
            from . import triton_moe_gemm as _tmg

            (counts,) = ctx.saved_tensors
            weight_a, weight_b = ctx.owner._dequant_expert_pair(
                ctx.key_a,
                ctx.key_b,
                ctx.expert_indices,
                dtype=ctx.weight_dtype,
                device=grad_out_a.device,
            )
            grad_input = _tmg.grouped_linear_grad(
                grad_out_a.to(dtype=ctx.weight_dtype), weight_a, counts
            ) + _tmg.grouped_linear_grad(
                grad_out_b.to(dtype=ctx.weight_dtype), weight_b, counts
            )
            return grad_input, None, None, None, None, None

# One-shot log guards so the triton availability decision is logged once.
# ``resolve_moe_dispatch_mode`` (imported from moe.runtime.specs) selects the
# direct-routed dispatch implementation selected by the typed runtime policy:
# ``vectorized`` builds per-chunk padded batches; ``legacy`` uses the per-expert
# loop; ``triton`` swaps the frozen
# expert bmm for a count-aware grouped GEMM and silently falls back to vectorized
# when triton is unavailable or the device is < SM80 (see below).
_TRITON_DISPATCH_STATE: dict[str, bool] = {}


def _triton_dispatch_available(device: "torch.device") -> bool:
    """Resolve (once, cached) whether the triton grouped GEMM can be used.

    Logs a single line on first resolution. Any import/compile failure or a
    device below SM80 returns False so the caller uses the vectorized path.
    """
    key = "logged"
    try:
        from . import triton_moe_gemm as _tmg

        available = _tmg.is_available(device)
    except Exception as exc:  # pragma: no cover - defensive
        available = False
        if not _TRITON_DISPATCH_STATE.get(key):
            logger.warning("MoE triton dispatch unavailable (%s); using vectorized.", exc)
            _TRITON_DISPATCH_STATE[key] = True
        return False
    if not _TRITON_DISPATCH_STATE.get(key):
        if available:
            logger.info("MoE triton grouped-GEMM dispatch active.")
        else:
            logger.warning(
                "MoE triton dispatch requested but unavailable "
                "(no triton / compile failed / device < SM80); using vectorized."
            )
        _TRITON_DISPATCH_STATE[key] = True
    return available


_PERSISTENT_DISPATCH_STATE: dict[str, bool] = {}
_PERSISTENT_PROBE_CACHE: dict[tuple[int, int], bool] = {}


def _persistent_probe(cc_major: int, cc_minor: int) -> bool:
    """Compile + run the vendored persistent grouped GEMM once per capability.

    Runs a tiny sorted-contiguous forward and checks it against a per-expert bmm
    reference (bf16 tolerance). Any import/compile/run failure returns False so
    the caller falls back to the vectorized path. Cached per (major, minor).
    """
    key = (int(cc_major), int(cc_minor))
    cached = _PERSISTENT_PROBE_CACHE.get(key)
    if cached is not None:
        return cached
    ok = False
    try:
        from mirai.vendors.qwen3_moe_fused import grouped_gemm_forward

        dev = torch.device("cuda")
        counts = torch.tensor([2, 3, 1], device=dev, dtype=torch.int64)
        total = int(counts.sum())
        w = torch.randn(3, 4, 8, device=dev, dtype=torch.bfloat16)
        xs = torch.randn(total, 8, device=dev, dtype=torch.bfloat16)
        m_offsets = torch.cumsum(counts, dim=0).to(torch.int32)
        out = grouped_gemm_forward(xs, w, m_offsets)
        ref = torch.empty_like(out)
        start = 0
        for e in range(3):
            end = start + int(counts[e])
            ref[start:end] = xs[start:end].float().to(out.dtype) @ w[e].T
            start = end
        ok = bool(torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2))
    except Exception:  # pragma: no cover - defensive (triton compile flakiness)
        ok = False
    _PERSISTENT_PROBE_CACHE[key] = ok
    return ok


def _persistent_dispatch_available(device: "torch.device") -> bool:
    """Resolve (once, cached) whether the vendored persistent GEMM can be used.

    Requires torch CUDA, device SM80+, an importable triton, and a successful
    tiny compile/run of the vendored kernel. Logs a single line on first
    resolution and falls back to vectorized on any failure."""
    logged = "logged"
    if torch is None or not torch.cuda.is_available():
        return False
    # The dispatch device itself must be CUDA: on a CPU tensor (e.g. a dry-run)
    # the vendored kernel's ``assert x.is_cuda`` would fire, so fall back to the
    # device-agnostic vectorized path instead of probing the current GPU.
    if device is None or getattr(device, "type", None) != "cuda":
        return False
    try:
        idx = device.index if device is not None and device.type == "cuda" else None
        major, minor = torch.cuda.get_device_capability(idx)
    except Exception:
        major, minor = 0, 0
    if int(major) < 8:
        available = False
    else:
        from mirai.core.models.compressed_weights.execution.triton_moe_gemm import _import_triton

        try:
            _import_triton()
            available = _persistent_probe(int(major), int(minor))
        except Exception:  # pragma: no cover - defensive
            available = False
    if not _PERSISTENT_DISPATCH_STATE.get(logged):
        if available:
            logger.info("MoE persistent sorted-contiguous grouped-GEMM dispatch active.")
        else:
            logger.warning(
                "MoE triton_persistent dispatch requested but unavailable "
                "(no triton / compile failed / device < SM80); using vectorized."
            )
        _PERSISTENT_DISPATCH_STATE[logged] = True
    return available


class CompressedLinear(nn.Module):
    """Frozen linear layer stored as int8 with optional grouped Hadamard rotation."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        group_sizes: str | int | Iterable[int] | None = None,
        quant_format: str = "int8",
        nf4_blocksize: int = NF4_BLOCKSIZE,
    ):
        super().__init__()
        if torch is None:  # pragma: no cover
            raise RuntimeError("CompressedLinear requires torch.")
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self._quant_format = normalize_quant_format(quant_format)
        self._nf4_blocksize = int(nf4_blocksize)
        self._nf4_meta: _Nf4Meta | None = None
        self._microscaling_meta: MicroscalingMeta | None = None
        self._blockwise_fp8_meta: BlockwiseFP8Meta | None = None
        if (
            self._quant_format == "nf4"
            or self._quant_format in MICROSCALING_FORMATS
            or self._quant_format in BLOCKWISE_FP8_FORMATS
        ):
            self.quantization_group_size = 0
            self.load_dense_weight(source=base.weight)
        else:
            group_size = best_group_size(self.in_features, group_sizes)
            quantized, scale, resolved_group = _quantize_weight(base.weight, group_size=group_size)
            self.quantization_group_size = int(resolved_group)
            self.register_buffer("weight_int8", quantized)
            self.register_buffer("weight_scale", scale.float())
        if base.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", base.bias.detach().float().contiguous())

    @classmethod
    def from_empty(
        cls,
        *,
        in_features: int,
        out_features: int,
        group_size: int,
        has_bias: bool,
        quant_format: str = "int8",
        nf4_blocksize: int = NF4_BLOCKSIZE,
    ) -> CompressedLinear:
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.in_features = int(in_features)
        module.out_features = int(out_features)
        module.quantization_group_size = int(group_size)
        module._quant_format = normalize_quant_format(quant_format)
        module._nf4_blocksize = int(nf4_blocksize)
        module._nf4_meta = None
        module._microscaling_meta = None
        module._blockwise_fp8_meta = None
        if module._quant_format not in {
            "nf4",
            *MICROSCALING_FORMATS,
            *BLOCKWISE_FP8_FORMATS,
        }:
            module.register_buffer("weight_int8", torch.empty((int(out_features), int(in_features)), dtype=torch.int8))
            module.register_buffer("weight_scale", torch.empty((int(out_features), 1), dtype=torch.float32))
        if bool(has_bias):
            module.register_buffer("bias", torch.empty((int(out_features),), dtype=torch.float32))
        else:
            module.bias = None
        return module

    def _set_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    def _store_nf4_weight(
        self,
        fields: Mapping[str, torch.Tensor],
        codes: Mapping[str, torch.Tensor],
        meta: _Nf4Meta,
        *,
        device: torch.device,
    ) -> None:
        self._nf4_meta = meta
        self._set_buffer("weight_nf4", fields["packed"].to(device=device).contiguous())
        self._set_buffer("weight_nf4_absmax", fields["absmax"].to(device=device).contiguous())
        self._set_buffer("weight_nf4_nabsmax", fields["nested_absmax"].to(device=device).contiguous())
        self._set_buffer("weight_nf4_offset", fields["offset"].to(device=device).contiguous())
        self._set_buffer("weight_nf4_code", codes["code"].to(device=device).contiguous())
        self._set_buffer("weight_nf4_ncode", codes["nested_code"].to(device=device).contiguous())

    def _nf4_weight(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        fields = {
            "packed": self.weight_nf4,
            "absmax": self.weight_nf4_absmax,
            "nested_absmax": self.weight_nf4_nabsmax,
            "offset": self.weight_nf4_offset,
        }
        codes = {"code": self.weight_nf4_code, "nested_code": self.weight_nf4_ncode}
        return _nf4_dequantize(
            fields,
            codes,
            self._nf4_meta,
            shape=(self.out_features, self.in_features),
            dtype=dtype,
            device=device,
        )

    def frozen_quantized_numel(self) -> int:
        if (
            self._quant_format == "nf4"
            or self._quant_format in MICROSCALING_FORMATS
            or self._quant_format in BLOCKWISE_FP8_FORMATS
        ):
            return int(self.in_features) * int(self.out_features)
        return int(self.weight_int8.numel())

    def load_packed_state(
        self,
        *,
        weight_int8: torch.Tensor,
        weight_scale: torch.Tensor,
        group_size: int,
        bias: torch.Tensor | None = None,
    ) -> None:
        if tuple(weight_int8.shape) != (self.out_features, self.in_features):
            raise ValueError(
                "compressed_weights packed linear weight shape mismatch: "
                f"expected {(self.out_features, self.in_features)}, got {tuple(weight_int8.shape)}."
            )
        if tuple(weight_scale.shape) != (self.out_features, 1):
            raise ValueError(
                "compressed_weights packed linear scale shape mismatch: "
                f"expected {(self.out_features, 1)}, got {tuple(weight_scale.shape)}."
            )
        self._buffers["weight_int8"] = weight_int8.detach().to(torch.int8).clone(
            memory_format=torch.contiguous_format
        )
        self._buffers["weight_scale"] = weight_scale.detach().float().clone(
            memory_format=torch.contiguous_format
        )
        self.quantization_group_size = int(group_size)
        if bias is None:
            self._buffers.pop("bias", None)
            self.__dict__["bias"] = None
        else:
            if tuple(bias.shape) != (self.out_features,):
                raise ValueError(
                    "compressed_weights packed linear bias shape mismatch: "
                    f"expected {(self.out_features,)}, got {tuple(bias.shape)}."
                )
            self.__dict__.pop("bias", None)
            self._buffers["bias"] = bias.detach().float().clone(
                memory_format=torch.contiguous_format
            )

    def load_nf4_packed_state(
        self,
        *,
        packed: torch.Tensor,
        absmax: torch.Tensor,
        nested_absmax: torch.Tensor,
        offset: torch.Tensor,
        code: torch.Tensor,
        nested_code: torch.Tensor,
        meta: _Nf4Meta,
        bias: torch.Tensor | None = None,
    ) -> None:
        """Restore already-quantized NF4 storage without touching dense weights."""
        if self._quant_format != "nf4":
            raise ValueError("NF4 packed state requires an NF4 CompressedLinear wrapper.")
        if packed.numel() * 2 < self.in_features * self.out_features:
            raise ValueError("NF4 packed linear weight is too small for the declared shape.")
        fields = {
            "packed": packed.detach(),
            "absmax": absmax.detach(),
            "nested_absmax": nested_absmax.detach(),
            "offset": offset.detach(),
        }
        codes = {"code": code.detach(), "nested_code": nested_code.detach()}
        self._store_nf4_weight(fields, codes, meta, device=packed.device)
        if bias is None:
            self._buffers.pop("bias", None)
            self.__dict__["bias"] = None
        else:
            if tuple(bias.shape) != (self.out_features,):
                raise ValueError(
                    "NF4 packed linear bias shape mismatch: "
                    f"expected {(self.out_features,)}, got {tuple(bias.shape)}."
                )
            self.__dict__.pop("bias", None)
            self._buffers["bias"] = bias.detach().float().clone(
                memory_format=torch.contiguous_format
            )

    def load_microscaling_packed_state(
        self,
        *,
        packed: torch.Tensor,
        scales: torch.Tensor,
        global_scale: torch.Tensor,
        meta: MicroscalingMeta,
        bias: torch.Tensor | None = None,
    ) -> None:
        if self._quant_format not in MICROSCALING_FORMATS:
            raise ValueError(
                "Microscaling packed state requires a registered microscaling "
                "linear wrapper."
            )
        if meta.format != self._quant_format:
            raise ValueError("Microscaling packed-state format does not match the wrapper.")
        if meta.shape != (self.out_features, self.in_features):
            raise ValueError("Microscaling packed linear shape does not match the wrapper.")
        validate_microscaling_payload(
            packed,
            scales,
            global_scale,
            meta,
        )
        self._microscaling_meta = meta
        self._set_buffer("weight_mx", packed.detach().contiguous())
        self._set_buffer("weight_mx_scale", scales.detach().contiguous())
        self._set_buffer("weight_mx_global", global_scale.detach().float().contiguous())
        if bias is None:
            self._buffers.pop("bias", None)
            self.__dict__["bias"] = None
        else:
            if tuple(bias.shape) != (self.out_features,):
                raise ValueError("Microscaling packed linear bias shape mismatch.")
            self.__dict__.pop("bias", None)
            self._buffers["bias"] = bias.detach().float().contiguous()

    def load_blockwise_fp8_packed_state(
        self,
        *,
        codes: torch.Tensor,
        scales: torch.Tensor,
        meta: BlockwiseFP8Meta,
        bias: torch.Tensor | None = None,
    ) -> None:
        if self._quant_format not in BLOCKWISE_FP8_FORMATS:
            raise ValueError("Blockwise FP8 state requires an FP8 linear wrapper.")
        if meta.shape != (self.out_features, self.in_features):
            raise ValueError("Blockwise FP8 linear shape does not match the wrapper.")
        validate_blockwise_fp8_payload(codes, scales, meta)
        self._blockwise_fp8_meta = meta
        self._set_buffer("weight_fp8", codes.detach().contiguous())
        self._set_buffer("weight_fp8_scale", scales.detach().float().contiguous())
        if bias is None:
            self._buffers.pop("bias", None)
            self.__dict__["bias"] = None
        else:
            if tuple(bias.shape) != (self.out_features,):
                raise ValueError("Blockwise FP8 linear bias shape mismatch.")
            self.__dict__.pop("bias", None)
            self._buffers["bias"] = bias.detach().float().contiguous()

    def load_dense_weight(self, *, source: torch.Tensor) -> None:
        if source.ndim != 2:
            raise ValueError(
                f"compressed_weights linear expects 2D weights, got {tuple(source.shape)}."
            )
        if self._quant_format == "nf4":
            fields, codes, meta = _nf4_quantize_2d(
                source, blocksize=self._nf4_blocksize
            )
            self._store_nf4_weight(fields, codes, meta, device=source.device)
            return
        if self._quant_format in MICROSCALING_FORMATS:
            packed, scales, global_scale, meta = quantize_microscaling(
                self._quant_format, source
            )
            self._microscaling_meta = meta
            self._set_buffer("weight_mx", packed)
            self._set_buffer("weight_mx_scale", scales)
            self._set_buffer("weight_mx_global", global_scale)
            return
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            codes, scales, meta = quantize_blockwise_fp8_weight(source)
            self._blockwise_fp8_meta = meta
            self._set_buffer("weight_fp8", codes)
            self._set_buffer("weight_fp8_scale", scales)
            return
        quantized, scale, resolved_group = _quantize_weight(
            source.detach().to(torch.float32),
            group_size=self.quantization_group_size,
        )
        self._buffers["weight_int8"] = quantized.to(torch.int8).contiguous()
        self._buffers["weight_scale"] = scale.float().contiguous()
        self.quantization_group_size = int(resolved_group)

    def load_dense_bias(self, *, source: torch.Tensor) -> None:
        if source.ndim != 1:
            raise ValueError(
                f"compressed_weights linear bias must be 1D, got {tuple(source.shape)}."
            )
        if tuple(source.shape) != (self.out_features,):
            raise ValueError(
                "compressed_weights linear bias shape mismatch: "
                f"expected {(self.out_features,)}, got {tuple(source.shape)}."
            )
        self._buffers["bias"] = source.detach().float().contiguous()

    @property
    def weight(self) -> torch.Tensor:
        device = self._stored_weight_device()
        dtype = (
            self.weight_scale.dtype
            if self._quant_format == "int8" and self.weight_scale.is_floating_point()
            else torch.get_default_dtype()
        )
        return self._materialize_weight(dtype=dtype, device=device)

    def _stored_weight_device(self) -> torch.device:
        if self._quant_format == "nf4":
            return self.weight_nf4.device
        if self._quant_format in MICROSCALING_FORMATS:
            return self.weight_mx.device
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            return self.weight_fp8.device
        return self.weight_int8.device

    def _materialize_weight(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self._quant_format == "nf4":
            return self._nf4_weight(dtype=dtype, device=device)
        if self._quant_format in MICROSCALING_FORMATS:
            return dequantize_microscaling(
                self.weight_mx,
                self.weight_mx_scale,
                self.weight_mx_global,
                self._microscaling_meta,
                dtype=dtype,
                device=device,
            )
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            return dequantize_blockwise_fp8_weight(
                self.weight_fp8,
                self.weight_fp8_scale,
                self._blockwise_fp8_meta,
                dtype=dtype,
                device=device,
            )
        return _dequantize_weight(
            self.weight_int8,
            self.weight_scale,
            group_size=self.quantization_group_size,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            output = blockwise_fp8_linear(
                x,
                self.weight_fp8,
                self.weight_fp8_scale,
                self._blockwise_fp8_meta,
            )
            bias = (
                self.bias.to(device=x.device, dtype=x.dtype)
                if self.bias is not None
                else None
            )
            return output if bias is None else output + bias
        output = _FrozenCompressedLinear.apply(x, self)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return output if bias is None else output + bias

    def extra_repr(self) -> str:
        if self._quant_format == "nf4":
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"dtype=nf4, blocksize={self._nf4_blocksize}"
            )
        if self._quant_format in MICROSCALING_FORMATS:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"dtype={self._quant_format}, block_size={self._microscaling_meta.block_size}"
            )
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                "dtype=fp8_e4m3, weight_block=128x128, activation_block=128"
            )
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"dtype=int8, group_size={self.quantization_group_size}"
        )
