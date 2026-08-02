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
            False,
            "; ".join(f"{status.name}: {status.reason}" for status in candidates),
            capability,
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
    """Execute packed THD attention through FA3/FA4 with explicit capability checks."""
    name = normalize_attention_backend(backend)
    candidates = ("flash4", "flash3") if name == "auto" else (name,)
    if any(candidate not in {"flash3", "flash4"} for candidate in candidates):
        raise ValueError("Packed variable-length attention requires flash3 or flash4.")
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
    "normalize_attention_backend",
    "probe_attention_backends",
]
