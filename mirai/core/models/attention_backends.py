"""Model-agnostic attention backend registry and dispatch.

FlashAttention-3 and FlashAttention-4 interfaces follow the official
Dao-AILab implementation. FA4's Hopper/Blackwell implementation accompanies
arXiv:2603.05451; performance claims from that paper require independent
measurement for each Mirai workload.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import importlib
from typing import Any, Callable

import torch
import torch.nn.functional as F

try:  # torch>=2.3
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - unsupported torch build
    SDPBackend = None
    sdpa_kernel = None


@dataclass(frozen=True)
class AttentionBackendSpec:
    name: str
    engine: str
    minimum_cuda_capability: tuple[int, int] | None = None
    modules: tuple[str, ...] = ()
    function: str = ""
    varlen_function: str = ""


@dataclass(frozen=True)
class AttentionBackendStatus:
    name: str
    available: bool
    reason: str
    cuda_capability: tuple[int, int] | None


ATTENTION_BACKEND_SPECS = {
    "auto": AttentionBackendSpec("auto", "sdpa"),
    "cudnn": AttentionBackendSpec("cudnn", "sdpa"),
    "flash": AttentionBackendSpec(
        "flash",
        "sdpa",
        minimum_cuda_capability=(8, 0),
    ),
    "flash3": AttentionBackendSpec(
        "flash3",
        "external",
        minimum_cuda_capability=(9, 0),
        modules=("flash_attn_interface", "flash_attn_3.flash_attn_interface"),
        function="flash_attn_func",
        varlen_function="flash_attn_varlen_func",
    ),
    "flash4": AttentionBackendSpec(
        "flash4",
        "external",
        minimum_cuda_capability=(9, 0),
        modules=("flash_attn.cute",),
        function="flash_attn_func",
        varlen_function="flash_attn_varlen_func",
    ),
    "flex": AttentionBackendSpec(
        "flex",
        "flex",
        modules=("torch.nn.attention.flex_attention",),
        function="flex_attention",
        varlen_function="flex_attention",
    ),
}
ALLOWED_ATTENTION_BACKENDS = frozenset(ATTENTION_BACKEND_SPECS)


def normalize_attention_backend(value: str) -> str:
    name = str(value).strip().lower()
    if name not in ATTENTION_BACKEND_SPECS:
        raise ValueError(
            "attention backend must be one of: "
            + ", ".join(ATTENTION_BACKEND_SPECS)
            + f"; got {value!r}."
        )
    return name


def _cuda_capability(device: torch.device) -> tuple[int, int] | None:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    return tuple(int(v) for v in torch.cuda.get_device_capability(resolved))


@lru_cache(maxsize=None)
def _flex_attention_api() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return ``(flex_attention, create_block_mask)`` from the installed torch.

    FlexAttention is imported at its execution seam because the module pulls in
    the higher-order-op and compile machinery, which no other backend needs.
    """
    module = importlib.import_module("torch.nn.attention.flex_attention")
    flex = getattr(module, "flex_attention", None)
    create_block_mask = getattr(module, "create_block_mask", None)
    if not callable(flex) or not callable(create_block_mask):
        raise RuntimeError(
            "flex backend is unavailable (torch.nn.attention.flex_attention is "
            "incomplete in this torch build)."
        )
    return flex, create_block_mask


@lru_cache(maxsize=None)
def _load_external_function(name: str, *, varlen: bool) -> Callable[..., Any]:
    spec = ATTENTION_BACKEND_SPECS[name]
    symbol = spec.varlen_function if varlen else spec.function
    failures: list[str] = []
    for module_name in spec.modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}")
            continue
        function = getattr(module, symbol, None)
        if callable(function):
            return function
        failures.append(f"{module_name}: missing {symbol}")
    detail = "; ".join(failures) or "no module candidates"
    raise RuntimeError(f"{name} backend is unavailable ({detail}).")


def attention_backend_status(
    name: str,
    *,
    device: torch.device,
    varlen: bool = False,
) -> AttentionBackendStatus:
    resolved = normalize_attention_backend(name)
    spec = ATTENTION_BACKEND_SPECS[resolved]
    capability = _cuda_capability(torch.device(device))
    if resolved == "auto" and not varlen:
        return AttentionBackendStatus(resolved, True, "PyTorch automatic SDPA", capability)
    if resolved == "auto":
        candidates = tuple(
            attention_backend_status(name, device=device, varlen=True)
            for name in ("flash4", "flash3")
        )
        if any(status.available for status in candidates):
            return AttentionBackendStatus(
                resolved,
                True,
                "automatic packed FA4-to-FA3 selection",
                capability,
            )
        return AttentionBackendStatus(
            resolved,
            True,
            "PyTorch SDPA packed reference fallback ("
            + "; ".join(f"{status.name}: {status.reason}" for status in candidates)
            + ")",
            capability,
        )
    if spec.engine == "flex":
        # FlexAttention has an eager reference lowering, so it is functional on
        # any device; only a fused CUDA execution needs a compiled kernel.
        try:
            _flex_attention_api()
        except Exception as exc:
            return AttentionBackendStatus(
                resolved,
                False,
                f"flex backend is unavailable ({type(exc).__name__}).",
                capability,
            )
        return AttentionBackendStatus(
            resolved, True, "PyTorch FlexAttention", capability
        )
    if capability is None:
        return AttentionBackendStatus(resolved, False, "CUDA device required", None)
    minimum = spec.minimum_cuda_capability
    if minimum is not None and capability < minimum:
        return AttentionBackendStatus(
            resolved,
            False,
            f"requires compute capability {minimum[0]}.{minimum[1]} or newer",
            capability,
        )
    if spec.engine == "sdpa":
        if sdpa_kernel is None or SDPBackend is None:
            return AttentionBackendStatus(
                resolved,
                False,
                "torch.nn.attention backend selection is unavailable",
                capability,
            )
        if resolved == "flash" and not torch.backends.cuda.is_flash_attention_available():
            return AttentionBackendStatus(
                resolved,
                False,
                "PyTorch Flash Attention is not compiled in",
                capability,
            )
        if resolved == "cudnn" and not torch.backends.cudnn.is_available():
            return AttentionBackendStatus(
                resolved,
                False,
                "cuDNN is unavailable",
                capability,
            )
        return AttentionBackendStatus(resolved, True, "PyTorch SDPA backend", capability)
    try:
        _load_external_function(resolved, varlen=varlen)
    except RuntimeError as exc:
        return AttentionBackendStatus(resolved, False, str(exc), capability)
    return AttentionBackendStatus(resolved, True, "external kernel available", capability)


@lru_cache(maxsize=None)
def _varlen_backward_probe(name: str, device_index: int) -> tuple[bool, str]:
    """Execute one minimal packed backward and report whether it succeeded.

    A FlashAttention distribution can ship forward kernels only; its varlen
    entry point then imports and runs, and only the backward call raises. The
    presence of the symbol is therefore not evidence that the backend can train,
    and the sole deterministic evidence is running a backward once. The result
    is cached per backend and device, so the probe costs one tiny kernel launch.
    """
    try:
        function = _load_external_function(name, varlen=True)
    except RuntimeError as exc:
        return False, str(exc)
    device = torch.device("cuda", device_index)
    tokens, heads, head_dim = 8, 1, 64
    tensors = [
        torch.randn(
            tokens,
            heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    offsets = torch.tensor([0, tokens], device=device, dtype=torch.int32)
    try:
        result = function(
            q=tensors[0],
            k=tensors[1],
            v=tensors[2],
            cu_seqlens_q=offsets,
            cu_seqlens_k=offsets,
            max_seqlen_q=tokens,
            max_seqlen_k=tokens,
            causal=False,
        )
        output = result[0] if isinstance(result, tuple) else result
        output.float().square().sum().backward()
    except Exception as exc:  # kernel, build, and dispatch failures alike
        return False, f"{name} varlen backward failed ({type(exc).__name__}: {exc})"
    if any(tensor.grad is None for tensor in tensors):
        return False, f"{name} varlen backward produced no input gradients"
    return True, "external kernel supports varlen forward and backward"


def varlen_attention_backward_status(
    name: str,
    *,
    device: torch.device,
) -> AttentionBackendStatus:
    """Availability of ``name`` for packed attention that must also backpropagate.

    Extends :func:`attention_backend_status` with the one property the registry
    cannot infer from imports. External kernels are probed by execution; engines
    with a PyTorch autograd definition are reported as their forward status.
    """
    status = attention_backend_status(name, device=device, varlen=True)
    if not status.available:
        return status
    if ATTENTION_BACKEND_SPECS[status.name].engine != "external":
        return status
    resolved = torch.device(device)
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    supported, reason = _varlen_backward_probe(status.name, int(index))
    return AttentionBackendStatus(
        status.name, supported, reason, status.cuda_capability
    )


def probe_attention_backends(
    *,
    device: torch.device,
    varlen: bool = False,
) -> tuple[AttentionBackendStatus, ...]:
    """Return deterministic availability evidence for every configured backend."""
    return tuple(
        attention_backend_status(name, device=device, varlen=varlen)
        for name in ATTENTION_BACKEND_SPECS
    )


def _require_backend(
    name: str,
    *,
    device: torch.device,
    varlen: bool = False,
) -> str:
    resolved = normalize_attention_backend(name)
    status = attention_backend_status(resolved, device=device, varlen=varlen)
    if not status.available:
        raise RuntimeError(f"attention backend {resolved!r} is unavailable: {status.reason}.")
    return resolved


def _sdpa_selection(name: str) -> list[Any] | None:
    if name == "auto":
        return None
    if sdpa_kernel is None or SDPBackend is None:
        raise RuntimeError("explicit SDPA selection requires torch.nn.attention.")
    if name == "cudnn":
        return [SDPBackend.CUDNN_ATTENTION]
    if name == "flash":
        return [SDPBackend.FLASH_ATTENTION]
    raise ValueError(f"{name!r} is not a PyTorch SDPA backend.")


@lru_cache(maxsize=None)
def _compiled_flex_attention() -> Callable[..., Any]:
    flex, _ = _flex_attention_api()
    return torch.compile(flex)


def flex_attention_function(*, compiled: bool = False) -> Callable[..., Any]:
    """Return the FlexAttention entry point, eager or through ``torch.compile``.

    Only the compiled lowering is fused; the eager one is a reference
    implementation that materializes the score matrix, so callers select it
    when an inductor toolchain is unavailable or shapes are small.
    """
    if compiled:
        return _compiled_flex_attention()
    flex, _ = _flex_attention_api()
    return flex


def flex_document_ids(cu_seqlens: torch.Tensor, *, total: int) -> torch.Tensor:
    """Map every packed token position to the index of the sample owning it."""
    offsets = cu_seqlens.to(device=cu_seqlens.device, dtype=torch.int64)
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("Packed cumulative sequence lengths must hold at least one sample.")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(total):
        raise ValueError(
            "Packed cumulative sequence lengths must start at 0 and end at the "
            f"token count; got {int(offsets[0])}..{int(offsets[-1])} for {int(total)} tokens."
        )
    positions = torch.arange(int(total), device=offsets.device)
    # right=True places a position exactly on a boundary in the sample that
    # starts there, so token cu_seqlens[i] belongs to sample i, not i-1.
    return torch.bucketize(positions, offsets[1:], right=True)


def flex_document_block_mask(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    query_tokens: int,
    key_tokens: int,
    device: torch.device,
    compile_mask: bool,
) -> Any:
    """Build the varlen document ``BlockMask`` that isolates packed samples.

    ``compile_mask`` selects the compiled mask construction. The eager
    construction evaluates the mask over the full ``[Q, KV]`` grid, so it is
    only appropriate for the small shapes used by reference verification.
    """
    _, create_block_mask = _flex_attention_api()
    query_documents = flex_document_ids(cu_seqlens_q, total=int(query_tokens)).to(device)
    key_documents = flex_document_ids(cu_seqlens_k, total=int(key_tokens)).to(device)

    def document_mask(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        return query_documents[query_index] == key_documents[key_index]

    return create_block_mask(
        document_mask,
        None,
        None,
        int(query_tokens),
        int(key_tokens),
        device=str(device),
        _compile=bool(compile_mask),
    )


def dispatch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Execute attention for BSHD tensors through one explicit backend contract."""
    name = normalize_attention_backend(backend)
    if attn_mask is not None and name != "auto":
        raise ValueError(
            "Explicit attention backends require maskless input; use 'auto' for "
            "masked attention."
        )
    if name in {"flash3", "flash4"}:
        _require_backend(name, device=query.device)
        function = _load_external_function(name, varlen=False)
        result = function(query, key, value, causal=False)
        return result[0] if isinstance(result, tuple) else result

    if name == "flex":
        _require_backend(name, device=query.device)
        flex, _ = _flex_attention_api()
        output = flex(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            enable_gqa=query.shape[2] != key.shape[2],
        )
        return output.transpose(1, 2)

    if name != "auto":
        _require_backend(name, device=query.device)
    mask = attn_mask
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    selection = _sdpa_selection(name)
    context = sdpa_kernel(selection) if selection is not None else nullcontext()
    with context:
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=mask,
            dropout_p=0.0,
        )
    return output.transpose(1, 2)


def dispatch_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    backend: str,
) -> torch.Tensor:
    """Execute packed THD attention through FA3/FA4 or an SDPA reference path."""
    name = normalize_attention_backend(backend)
    if name == "flex":
        _require_backend(name, device=query.device, varlen=True)
        flex, _ = _flex_attention_api()
        block_mask = flex_document_block_mask(
            cu_seqlens_q,
            cu_seqlens_k,
            query_tokens=int(query.shape[0]),
            key_tokens=int(key.shape[0]),
            device=query.device,
            compile_mask=query.device.type == "cuda",
        )
        output = flex(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            block_mask=block_mask,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return output.squeeze(0).transpose(0, 1)
    candidates = ("flash4", "flash3") if name == "auto" else (name,)
    if any(candidate not in {"flash3", "flash4"} for candidate in candidates):
        raise ValueError(
            "Packed variable-length attention requires flash3, flash4, or flex."
        )
    failures: list[str] = []
    for candidate in candidates:
        status = attention_backend_status(
            candidate,
            device=query.device,
            varlen=True,
        )
        if not status.available:
            failures.append(f"{candidate}: {status.reason}")
            continue
        function = _load_external_function(candidate, varlen=True)
        result = function(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            causal=False,
        )
        return result[0] if isinstance(result, tuple) else result
    if name == "auto":
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError(
                "Packed attention expects THD query, key, and value tensors."
            )
        if key.shape != value.shape:
            raise ValueError("Packed attention key and value tensors must have equal shapes.")
        if query.shape[1:] != key.shape[1:]:
            raise ValueError("Packed attention tensors must share head dimensions.")
        if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
            raise ValueError("Packed attention cumulative sequence lengths must be 1D.")
        if cu_seqlens_q.numel() != cu_seqlens_k.numel():
            raise ValueError("Packed query and key batches must have equal size.")
        q_offsets = tuple(int(value) for value in cu_seqlens_q.tolist())
        k_offsets = tuple(int(value) for value in cu_seqlens_k.tolist())
        if (
            not q_offsets
            or q_offsets[0] != 0
            or k_offsets[0] != 0
            or q_offsets[-1] != query.shape[0]
            or k_offsets[-1] != key.shape[0]
        ):
            raise ValueError("Packed attention cumulative sequence lengths are inconsistent.")
        outputs: list[torch.Tensor] = []
        for index in range(len(q_offsets) - 1):
            q_start, q_end = q_offsets[index : index + 2]
            k_start, k_end = k_offsets[index : index + 2]
            if q_end < q_start or k_end <= k_start:
                raise ValueError(
                    "Packed attention sequence lengths must be non-negative and "
                    "keys non-empty."
                )
            if q_end == q_start:
                continue
            outputs.append(
                dispatch_attention(
                    query[q_start:q_end].unsqueeze(0),
                    key[k_start:k_end].unsqueeze(0),
                    value[k_start:k_end].unsqueeze(0),
                    backend="auto",
                ).squeeze(0)
            )
        return torch.cat(outputs, dim=0) if outputs else query.clone()
    raise RuntimeError(
        "No packed variable-length attention backend is available ("
        + "; ".join(failures)
        + ")."
    )


__all__ = [
    "ALLOWED_ATTENTION_BACKENDS",
    "ATTENTION_BACKEND_SPECS",
    "AttentionBackendSpec",
    "AttentionBackendStatus",
    "attention_backend_status",
    "dispatch_attention",
    "dispatch_varlen_attention",
    "flex_attention_function",
    "flex_document_block_mask",
    "flex_document_ids",
    "normalize_attention_backend",
    "probe_attention_backends",
    "varlen_attention_backward_status",
]
