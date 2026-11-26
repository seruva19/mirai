"""Memory-efficient trainable attention for MAGI-2 packed varlen self-attention.

The vendored reference path
(``mirai/vendors/magi2_preview/model/magi2_preview.py::_torch_varlen_attention_with_sink``,
derived from SandAI's Apache-2.0 MAGI-2 preview release) is exact but
materializes a dense ``[heads, Lq, Lk]`` additive mask per sample, so its
activation cost grows with the square of the packed token count. The vendored
FlashAttention-3 sink kernel avoids that but has no backward, so training falls
back to the reference path. This module supplies the third path: PyTorch
FlexAttention, which has a backward and lowers to a fused kernel when compiled.

Sink semantics reproduced here, read off the reference implementation:

* ``sinks`` is a ``[sink_token_num, num_heads_q]`` float32 parameter. Per sample
  and per head it contributes ``sum_j exp(sink[j, head])`` to the softmax
  denominator and contributes nothing to the numerator, because the reference
  appends zero-valued keys and values and adds the sink logit through the
  additive mask. The sink logit is therefore used unscaled: the ``head_dim``
  softmax scale applies to the query-key product only.
* The sinks belong to each sample's own softmax; packed samples never share a
  denominator.
* ``softcap > 0`` caps the query-key logits with ``softcap * tanh(score /
  softcap)`` before the softmax while the sink logits stay uncapped. The
  released preview configuration disables it with ``attn_softcap = -1.0``.
* q/k RMS normalization and rotary embedding happen in ``Attention.forward``
  before this call, so the kernel sees post-RoPE tensors.
* Queries and keys are cast to the value dtype before attention and the output
  is cast back to the query dtype, matching the reference.

Rather than adding a virtual key position per sink, this path computes sinkless
attention with its log-sum-exp and rescales the result by
``sigmoid(lse - logsumexp(sinks))``. That is algebraically the same softmax --
the numerator is unchanged because sink values are zero and the denominator
gains exactly ``sum_j exp(sink_j)`` -- while keeping ``score_mod`` free of
captured parameters and keeping the sink gradient in plain autograd.

Tensor semantics: ``q`` is ``[1, tokens, heads_q, head_dim]`` and ``k``/``v``
are ``[1, tokens, heads_kv, head_dim]``, packed across samples and delimited by
``cu_seqlens``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from mirai.core.models.attention_backends import (
    attention_backend_status,
    flash4_varlen_with_lse,
    flex_attention_function,
    flex_document_block_mask,
)


class Magi2FlexAttentionUnsupported(ValueError):
    """Raised when the flex attention path cannot serve a MAGI-2 request."""


def validate_flex_attention_support(device: torch.device | None = None) -> None:
    """Reject the flex path before model execution when torch cannot run it."""
    probe = torch.device("cpu") if device is None else torch.device(device)
    status = attention_backend_status("flex", device=probe, varlen=True)
    if not status.available:
        raise Magi2FlexAttentionUnsupported(
            "model.attention_backend='flex' selected PyTorch FlexAttention for "
            f"MAGI-2 attention, which is unavailable: {status.reason}"
        )


def _sink_log_denominator(sink: torch.Tensor, *, heads: int) -> torch.Tensor:
    """Collapse the sink parameter to one log-denominator contribution per head."""
    if sink.ndim != 2:
        raise Magi2FlexAttentionUnsupported(
            "MAGI-2 attention sinks must be a [sink_token_num, heads] tensor; "
            f"got shape {tuple(sink.shape)}."
        )
    if int(sink.shape[1]) != int(heads):
        raise Magi2FlexAttentionUnsupported(
            f"MAGI-2 attention sinks carry {int(sink.shape[1])} heads but the "
            f"query tensor carries {int(heads)}."
        )
    return torch.logsumexp(sink.float(), dim=0)


@lru_cache(maxsize=16)
def _document_block_mask(
    query_offsets: tuple[int, ...],
    key_offsets: tuple[int, ...],
    query_tokens: int,
    key_tokens: int,
    device: torch.device,
    compile_mask: bool,
    stream: torch.cuda.Stream | None,
) -> Any:
    for offsets, tokens in ((query_offsets, query_tokens), (key_offsets, key_tokens)):
        if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != tokens:
            raise ValueError(
                "Packed cumulative sequence lengths must start at 0 and end at the token count."
            )
        if any(left > right for left, right in zip(offsets, offsets[1:])):
            raise ValueError("Packed cumulative sequence lengths must be nondecreasing.")
    if len(query_offsets) == len(key_offsets) == 2:
        return None

    # Match the producing stream on cache hits, and share across grad modes.
    with torch.inference_mode(False):
        return flex_document_block_mask(
            torch.tensor(query_offsets, dtype=torch.int64, device=device),
            torch.tensor(key_offsets, dtype=torch.int64, device=device),
            query_tokens=query_tokens,
            key_tokens=key_tokens,
            device=device,
            compile_mask=compile_mask,
        )


def flex_varlen_attention_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softcap: float,
    sink: torch.Tensor | None,
    compile_kernel: bool | None = None,
    compile_block_mask: bool | None = None,
) -> torch.Tensor:
    """FlexAttention equivalent of ``_torch_varlen_attention_with_sink``."""
    query_dtype = q.dtype
    q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise Magi2FlexAttentionUnsupported(
            "MAGI-2 flex attention expects packed [tokens, heads, head_dim] "
            "query, key, and value tensors."
        )
    if k.shape[:2] != v.shape[:2] or k.shape[2] != v.shape[2]:
        raise Magi2FlexAttentionUnsupported(
            "MAGI-2 flex attention key and value tensors must share shape."
        )
    sink_logits = (
        _sink_log_denominator(sink, heads=int(q.shape[1]))
        if sink is not None and int(sink.numel()) > 0 else None
    )
    device = q.device
    if compile_kernel is None:
        compile_kernel = device.type == "cuda"
    if compile_block_mask is None:
        compile_block_mask = device.type == "cuda"

    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("Packed cumulative sequence lengths must be one-dimensional.")
    query_offsets = tuple(cu_seqlens_q.to(dtype=torch.int64).tolist())
    key_offsets = (
        query_offsets
        if cu_seqlens_q is cu_seqlens_k
        else tuple(cu_seqlens_k.to(dtype=torch.int64).tolist())
    )
    block_mask = _document_block_mask(
        query_offsets,
        key_offsets,
        int(q.shape[0]),
        int(k.shape[0]),
        device,
        bool(compile_block_mask),
        torch.cuda.current_stream(device) if device.type == "cuda" else None,
    )

    score_mod = None
    if float(softcap) > 0:
        cap = float(softcap)

        def score_mod(
            score: torch.Tensor,
            batch: torch.Tensor,
            head: torch.Tensor,
            query_index: torch.Tensor,
            key_index: torch.Tensor,
        ) -> torch.Tensor:
            return cap * torch.tanh(score / cap)

    query = q.to(v.dtype).transpose(0, 1).unsqueeze(0)
    key = k.to(v.dtype).transpose(0, 1).unsqueeze(0)
    value = v.transpose(0, 1).unsqueeze(0)

    flex = flex_attention_function(compiled=bool(compile_kernel))
    attended, log_sum_exp = flex(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=int(query.shape[1]) != int(key.shape[1]),
        return_lse=True,
    )

    if sink_logits is not None:
        scale = torch.sigmoid(log_sum_exp.float() - sink_logits.view(1, -1, 1))
        attended = attended.float() * scale.unsqueeze(-1)

    return attended.squeeze(0).transpose(0, 1).to(query_dtype)


@dataclass(frozen=True)
class Magi2FlexAttentionBackend:
    """Execution seam bound to every vendored MAGI-2 ``Attention`` module.

    ``compile_kernel`` and ``compile_block_mask`` default to the device
    decision: CUDA execution compiles, because only the compiled lowering is
    fused, while CPU execution stays eager so reference verification runs
    without an inductor toolchain.
    """

    compile_kernel: bool | None = None
    compile_block_mask: bool | None = None

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softcap: float,
        sink: torch.Tensor | None,
    ) -> torch.Tensor:
        return flex_varlen_attention_with_sink(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            softcap=softcap,
            sink=sink,
            compile_kernel=self.compile_kernel,
            compile_block_mask=self.compile_block_mask,
        )


class Magi2Flash4AttentionBackend:
    """Apply the MAGI-2 sink denominator to packed FlashAttention-4 output."""

    def execute(self, q, k, v, *, cu_seqlens_q, cu_seqlens_k, softcap, sink):
        if torch.is_grad_enabled() and any(
            tensor is not None and tensor.requires_grad for tensor in (q, k, v, sink)
        ):
            raise Magi2FlexAttentionUnsupported("MAGI-2 FlashAttention-4 is inference-only.")
        if q.ndim == 4 and k.ndim == 4 and v.ndim == 4:
            q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
        if any(tensor.ndim != 3 for tensor in (q, k, v)):
            raise Magi2FlexAttentionUnsupported("MAGI-2 FlashAttention-4 requires packed Q/K/V.")
        sink_logits = (
            _sink_log_denominator(sink, heads=q.shape[1])
            if sink is not None and sink.numel() else None
        )
        output, lse = flash4_varlen_with_lse(
            q.to(v.dtype), k.to(v.dtype), v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, softcap=softcap,
        )
        if sink_logits is not None:
            output = output.float() * torch.sigmoid(lse.float() - sink_logits).unsqueeze(-1)
        return output.to(q.dtype)


def resolve_magi2_flex_attention(model_config: Any) -> Magi2FlexAttentionBackend | Magi2Flash4AttentionBackend | None:
    """Return the flex backend when ``model.attention_backend`` selects it.

    FlashAttention-4 applies the same sink denominator in inference.
    Other values keep the vendored dispatch.
    """
    requested = str(getattr(model_config, "attention_backend", "auto")).strip().lower()
    if requested == "flash4":
        return Magi2Flash4AttentionBackend()
    if requested != "flex":
        return None
    return Magi2FlexAttentionBackend()


def attach_flex_attention_backend(
    transformer: Any, backend: Magi2FlexAttentionBackend | Magi2Flash4AttentionBackend | None
) -> int:
    """Bind (or clear) the attention execution seam on every vendored module."""
    from mirai.vendors.magi2_preview.model.magi2_preview import Attention

    attached = 0
    for module in transformer.modules():
        if isinstance(module, Attention):
            module._mirai_attention_backend = backend
            attached += 1
    if backend is not None and attached == 0:
        raise Magi2FlexAttentionUnsupported(
            "The configured attention backend matched no MAGI-2 attention layer "
            "in this model."
        )
    return attached


__all__ = [
    "Magi2FlexAttentionBackend",
    "Magi2FlexAttentionUnsupported",
    "attach_flex_attention_backend",
    "flex_varlen_attention_with_sink",
    "resolve_magi2_flex_attention",
    "validate_flex_attention_support",
]
