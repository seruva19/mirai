"""Fused CUDA SwiGLU7 activation kernels for MAGI-2 expert execution."""

from __future__ import annotations

from typing import Any


def _runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - runtime gate
        raise RuntimeError(
            "MAGI-2 Triton SwiGLU7 execution requires the installed Triton runtime."
        ) from exc
    return torch, triton, tl


try:
    _torch, _triton, _tl = _runtime()
except RuntimeError:  # pragma: no cover - import remains usable for policy validation
    _torch = _triton = _tl = None


if _triton is not None:

    @_triton.jit
    def _swiglu7_forward_kernel(
        gate_ptr,
        up_ptr,
        output_ptr,
        elements,
        BLOCK: _tl.constexpr,
    ):
        offsets = _tl.program_id(0) * BLOCK + _tl.arange(0, BLOCK)
        mask = offsets < elements
        gate = _tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(_tl.float32)
        up = _tl.load(up_ptr + offsets, mask=mask, other=0.0).to(_tl.float32)
        gate = _tl.minimum(gate, 7.0)
        up = _tl.maximum(_tl.minimum(up, 7.0), -7.0)
        sigmoid = 1.0 / (1.0 + _tl.exp(-1.702 * gate))
        _tl.store(output_ptr + offsets, gate * sigmoid * (up + 1.0), mask=mask)


    @_triton.jit
    def _swiglu7_backward_kernel(
        gate_ptr,
        up_ptr,
        grad_output_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        elements,
        BLOCK: _tl.constexpr,
    ):
        offsets = _tl.program_id(0) * BLOCK + _tl.arange(0, BLOCK)
        mask = offsets < elements
        raw_gate = _tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(_tl.float32)
        raw_up = _tl.load(up_ptr + offsets, mask=mask, other=0.0).to(_tl.float32)
        grad_output = _tl.load(
            grad_output_ptr + offsets, mask=mask, other=0.0
        ).to(_tl.float32)
        gate = _tl.minimum(raw_gate, 7.0)
        up = _tl.maximum(_tl.minimum(raw_up, 7.0), -7.0)
        sigmoid = 1.0 / (1.0 + _tl.exp(-1.702 * gate))
        activated_gate = gate * sigmoid
        gate_derivative = sigmoid + 1.702 * gate * sigmoid * (1.0 - sigmoid)
        gate_derivative = _tl.where(raw_gate <= 7.0, gate_derivative, 0.0)
        up_derivative = (raw_up >= -7.0) & (raw_up <= 7.0)
        _tl.store(
            grad_gate_ptr + offsets,
            grad_output * gate_derivative * (up + 1.0),
            mask=mask,
        )
        _tl.store(
            grad_up_ptr + offsets,
            grad_output * activated_gate * up_derivative,
            mask=mask,
        )


def _validate(gate: Any, up: Any) -> None:
    if _torch is None or _triton is None:
        _runtime()
        raise AssertionError("unreachable")
    if gate.shape != up.shape:
        raise ValueError("MAGI-2 SwiGLU7 gate/up shapes must match.")
    if not gate.is_cuda or not up.is_cuda:
        raise ValueError("MAGI-2 Triton SwiGLU7 requires CUDA tensors.")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("MAGI-2 Triton SwiGLU7 requires contiguous tensors.")


def triton_swiglu7(gate: Any, up: Any, *, output_dtype: Any) -> Any:
    _validate(gate, up)
    output = _torch.empty(gate.shape, dtype=output_dtype, device=gate.device)
    elements = int(gate.numel())
    _swiglu7_forward_kernel[(_triton.cdiv(elements, 256),)](
        gate,
        up,
        output,
        elements=elements,
        BLOCK=256,
        num_warps=4,
    )
    return output


def triton_swiglu7_backward(gate: Any, up: Any, grad_output: Any) -> tuple[Any, Any]:
    _validate(gate, up)
    if grad_output.shape != gate.shape or not grad_output.is_contiguous():
        raise ValueError(
            "MAGI-2 Triton SwiGLU7 gradient must be contiguous and match gate/up."
        )
    grad_gate = _torch.empty_like(gate)
    grad_up = _torch.empty_like(up)
    elements = int(gate.numel())
    _swiglu7_backward_kernel[(_triton.cdiv(elements, 256),)](
        gate,
        up,
        grad_output,
        grad_gate,
        grad_up,
        elements=elements,
        BLOCK=256,
        num_warps=4,
    )
    return grad_gate, grad_up


__all__ = ["triton_swiglu7", "triton_swiglu7_backward"]
