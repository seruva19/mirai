"""Gradient validation and policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

from mirai.core.training.residency.activation_compression import training_cuda_copy_streams
from mirai.core.training.residency.zero_bitpack import pack_zero_bitmap, unpack_zero_bitmap

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for gradient checks: {exc}")


def has_non_finite_gradients(params) -> bool:
    for param in params:
        grad = _effective_gradient(param)
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            return True
    return False


def _effective_gradient(param):
    cpu_grad = getattr(param, "_mirai_cpu_grad", None)
    if isinstance(cpu_grad, torch.Tensor):
        return cpu_grad
    return getattr(param, "grad", None)


def clip_grad_norm_with_offload(params, *, max_norm: float) -> float:
    params_list = list(params)
    offloaded = [
        _effective_gradient(param)
        for param in params_list
        if _effective_gradient(param) is not None
    ]
    if not offloaded:
        return 0.0
    norms = [grad.detach().float().norm(2) for grad in offloaded]
    total_norm = torch.stack([norm.to(device="cpu") for norm in norms]).norm(2)
    clip_coefficient = float(max_norm) / (float(total_norm.item()) + 1e-6)
    if clip_coefficient < 1.0:
        for grad in offloaded:
            grad.mul_(clip_coefficient)
    return float(total_norm.item())


def _accumulate_cpu_gradient(param, cpu_grad):
    existing = getattr(param, "_mirai_cpu_grad", None)
    if isinstance(existing, torch.Tensor):
        existing.add_(cpu_grad)
    else:
        param._mirai_cpu_grad = cpu_grad


@dataclass(frozen=True)
class _GradientPayload:
    values: torch.Tensor
    shape: torch.Size
    mask: torch.Tensor | None = None

    def fp32(self):
        if self.mask is None:
            output = torch.empty_like(self.values, dtype=torch.float32, pin_memory=False)
            return output.copy_(self.values)
        return unpack_zero_bitmap(self.values, self.mask, self.shape, dtype=torch.float32)


def _pack_sparse_gradient(grad):
    if grad.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64} or grad.nbytes < 65536:
        return _GradientPayload(grad, grad.shape)
    packed = pack_zero_bitmap(grad)
    if packed is None:
        return _GradientPayload(grad, grad.shape)
    values, mask = packed
    return _GradientPayload(values, grad.shape, mask)


class PendingGradientOffload:
    """Keep at most one microbatch's pinned gradient transfers in flight."""

    def __init__(self, *, bitpack: bool = False):
        self._pending = []
        self._last_event = {}
        self._bitpack = bitpack

    def submit(self, param, grad) -> bool:
        payload = _pack_sparse_gradient(grad) if self._bitpack else _GradientPayload(grad, grad.shape)
        try:
            host = torch.empty_like(payload.values, device="cpu", pin_memory=True)
            mask = None if payload.mask is None else torch.empty_like(payload.mask, device="cpu", pin_memory=True)
        except RuntimeError:
            return False
        d2h, _ = training_cuda_copy_streams(grad.device)
        d2h.wait_stream(torch.cuda.current_stream(grad.device))
        with torch.cuda.stream(d2h):
            host.copy_(payload.values, non_blocking=True)
            payload.values.record_stream(d2h)
            if mask is not None:
                mask.copy_(payload.mask, non_blocking=True)
                payload.mask.record_stream(d2h)
            self._last_event[grad.device] = d2h.record_event()
        self._pending.append((param, _GradientPayload(host, payload.shape, mask)))
        return True

    def finish(self):
        for event in self._last_event.values():
            event.synchronize()
        for param, payload in self._pending:
            _accumulate_cpu_gradient(param, payload.fp32())
        self._pending.clear()
        self._last_event.clear()


def offload_gradients_to_cpu(params, *, pending=None) -> int:
    moved = 0
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        queued = pending is not None and grad.is_cuda and grad.layout == torch.strided and pending.submit(param, grad.detach())
        if not queued:
            cpu_grad = grad.detach().to(device="cpu", dtype=torch.float32).clone()
            _accumulate_cpu_gradient(param, cpu_grad)
        # Always release the autograd-owned buffer. In particular, exposing the
        # CPU accumulator as ``param.grad`` would make the next backward update
        # that same storage before it is added here, doubling prior microbatches.
        param.grad = None
        moved += 1
    return moved


def restore_gradients_from_cpu(params) -> int:
    """Restore accumulated CPU gradients for validation, clipping, and update."""
    restored = 0
    for param in params:
        cpu_grad = getattr(param, "_mirai_cpu_grad", None)
        if not isinstance(cpu_grad, torch.Tensor):
            continue
        param.grad = cpu_grad.to(device=param.device, dtype=param.dtype)
        delattr(param, "_mirai_cpu_grad")
        restored += 1
    return restored


def clear_offloaded_gradients(params) -> int:
    cleared = 0
    for param in params:
        if hasattr(param, "_mirai_cpu_grad"):
            delattr(param, "_mirai_cpu_grad")
            cleared += 1
    return cleared


def release_gradient_memory(params) -> int:
    """Drop gradients to free their memory.

    Used after an applied optimizer step, before an auxiliary GPU operation
    such as preview sampling. The grads are stale at that point (they are
    re-zeroed and recomputed on the next backward), so they can be discarded
    outright. Unlike :func:`offload_gradients_to_cpu`, this is device-safe on
    CUDA — PyTorch forbids assigning a CPU grad to a CUDA parameter, so we
    clear ``param.grad`` instead of relocating it.
    """
    released = 0
    for param in params:
        if hasattr(param, "_mirai_cpu_grad"):
            delattr(param, "_mirai_cpu_grad")
            released += 1
        if getattr(param, "grad", None) is not None:
            param.grad = None
            released += 1
    return released


def resolve_non_finite_policy(
    *,
    policy: str,
    consecutive_skips: int,
    max_consecutive_skipped_steps: int,
) -> tuple[str, int]:
    key = policy.strip().lower()
    if key == "abort":
        return "abort", consecutive_skips
    if key != "skip_step":
        raise ValueError(f"Unsupported non_finite_grad_policy '{policy}'.")
    next_skips = consecutive_skips + 1
    if next_skips > max_consecutive_skipped_steps:
        raise RuntimeError(
            "Exceeded max_consecutive_skipped_steps due to non-finite gradients "
            f"({next_skips} > {max_consecutive_skipped_steps})."
        )
    return "skip_step", next_skips
