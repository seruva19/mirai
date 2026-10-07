from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from mirai.core.models.compressed_weights.quantization.quant import _rotate_last_dim  # noqa: E402
import mirai.core.models.compressed_weights.quantization.w4a8 as w4a8_module  # noqa: E402
from mirai.core.models.compressed_weights.quantization.w4a8 import (  # noqa: E402
    W4A8Weight,
    dequantize_w4a8,
    quantize_w4a8,
    unpack_w4a8_to_int8,
    w4a8_linear,
    w4a8_linear_reference,
)


def _independent_unpack(value: W4A8Weight) -> torch.Tensor:
    raw = value.packed.long()
    codes = torch.stack((raw & 15, (raw >> 4) & 15), dim=-1).flatten(-2)
    n, k = value.metadata.shape[-2:]
    levels = value.codebook[codes].reshape(*value.metadata.shape[:-1], k // 16, 16)
    scales = value.group_scales.view(torch.float8_e4m3fn).float()
    return (levels * scales.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8).reshape(value.metadata.shape)


def _independent_output(x: torch.Tensor, value: W4A8Weight) -> torch.Tensor:
    rg = value.metadata.rotation_group_size
    flat = x.reshape(-1, x.shape[-1]).float()
    if rg:
        flat = _rotate_last_dim(flat, rg, inverse=False)
    x_rot = flat.reshape(x.shape)
    x_scale = (x_rot.abs().amax(-1) / 127).clamp_min(1e-8)
    x_q = (x_rot / x_scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    w_q = _independent_unpack(value)
    if w_q.ndim == 2:
        accum = (x_q.reshape(-1, x.shape[-1]).double() @ w_q.t().double()).to(torch.int32).float()
        accum = accum.reshape(*x.shape[:-1], w_q.shape[-2])
    else:
        accum = torch.stack([
            (x_q[e].reshape(-1, x.shape[-1]).double() @ w_q[e].t().double()).to(torch.int32).float()
            for e in range(x.shape[0])
        ]).reshape(*x.shape[:-1], w_q.shape[-2])
    return (accum * x_scale.unsqueeze(-1) * value.row_scales.squeeze(-1).unsqueeze(-2)).to(x.dtype)


@pytest.mark.parametrize("shape", [(5, 32), (2, 5, 32)])
@pytest.mark.parametrize("rotation_group_size", [0, 4, 16])
def test_storage_roundtrip_shapes_and_original_basis(shape, rotation_group_size):
    torch.manual_seed(10)
    source = torch.randn(shape)
    value = quantize_w4a8(source, rotation_group_size=rotation_group_size)
    assert value.packed.shape == (*shape[:-1], shape[-1] // 2)
    assert value.group_scales.shape == (*shape[:-1], shape[-1] // 16)
    assert value.row_scales.shape == (*shape[:-1], 1)
    torch.testing.assert_close(unpack_w4a8_to_int8(value), _independent_unpack(value))
    reconstructed = dequantize_w4a8(value)
    assert reconstructed.shape == source.shape
    assert (reconstructed - source).square().mean().sqrt() < source.square().mean().sqrt() * 0.2


@pytest.mark.parametrize("expert_batched", [False, True])
def test_reference_matches_independent_a8_w4_integer_arithmetic(expert_batched):
    torch.manual_seed(11)
    weight = torch.randn(2, 7, 32) if expert_batched else torch.randn(7, 32)
    x = torch.randn(2, 3, 32) if expert_batched else torch.randn(2, 3, 32)
    value = quantize_w4a8(weight, rotation_group_size=16)
    actual = w4a8_linear_reference(x, value)
    expected = _independent_output(x, value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_backward_is_ste_against_decoded_original_basis_and_saves_no_dense_weight():
    torch.manual_seed(12)
    value = quantize_w4a8(torch.randn(6, 32), rotation_group_size=16)
    x = torch.randn(4, 32, requires_grad=True)
    grad_output = torch.randn(4, 6)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        output = w4a8_linear_reference(x, value)
    output.backward(grad_output)
    expected = grad_output.float() @ dequantize_w4a8(value)
    torch.testing.assert_close(x.grad, expected, rtol=2e-6, atol=2e-6)
    assert all(t is not x for t in saved)
    assert not any(tuple(t.shape) == tuple(value.metadata.shape) and t.is_floating_point() for t in saved)


def test_expert_batched_bfloat16_backward_shape_and_dtype():
    value = quantize_w4a8(torch.randn(2, 5, 48), rotation_group_size=16)
    x = torch.randn(2, 3, 48, dtype=torch.bfloat16, requires_grad=True)
    w4a8_linear_reference(x, value).float().sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.dtype == torch.bfloat16


def test_quantizing_trainable_source_detaches_all_storage():
    source = torch.nn.Parameter(torch.randn(2, 5, 32))
    value = quantize_w4a8(source, rotation_group_size=16)
    value.validate()
    assert all(
        not tensor.requires_grad
        for tensor in (
            value.packed,
            value.group_scales,
            value.row_scales,
            value.codebook,
        )
    )
    assert source.grad is None
    assert dequantize_w4a8(value, _numeric_validation=False).grad_fn is None


def test_metadata_and_tensor_fields_fail_closed():
    value = quantize_w4a8(torch.randn(3, 32))
    with pytest.raises(ValueError, match="version/scheme"):
        replace(value, metadata=replace(value.metadata, version=2)).validate()
    with pytest.raises(ValueError, match="scale format"):
        replace(value, metadata=replace(value.metadata, scale_format="float32")).validate()
    with pytest.raises(ValueError, match="packed"):
        replace(value, packed=value.packed.to(torch.int8)).validate()
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(value, codebook=value.codebook.flip(0)).validate()
    with pytest.raises(ValueError, match="positive"):
        replace(value, row_scales=torch.zeros_like(value.row_scales)).validate()
    with pytest.raises(ValueError, match="frozen"):
        replace(value, row_scales=value.row_scales.clone().requires_grad_()).validate()
    strided = value.packed.repeat_interleave(2, dim=-1)[..., ::2]
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        replace(value, packed=strided).validate()


@pytest.mark.parametrize("source", [torch.zeros(3, 32), torch.tensor([1e-20, 1e20]).repeat(3, 16)])
def test_zero_and_large_dynamic_range_have_valid_finite_storage(source):
    value = quantize_w4a8(source)
    value.validate()
    assert torch.isfinite(dequantize_w4a8(value)).all()


def test_invalid_geometry_and_explicit_acceleration_errors():
    with pytest.raises(ValueError, match="divisible by 16"):
        quantize_w4a8(torch.randn(3, 24))
    with pytest.raises(ValueError, match="zero or a power of four"):
        quantize_w4a8(torch.randn(3, 32), rotation_group_size=12)
    with pytest.raises(ValueError, match="power of four"):
        quantize_w4a8(torch.randn(3, 32), rotation_group_size=2)
    value = quantize_w4a8(torch.randn(3, 32))
    with pytest.raises(ValueError, match="unknown W4A8 backend"):
        w4a8_linear(torch.randn(2, 32), value, backend="auto")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        w4a8_linear(torch.randn(2, 32), value, backend="triton")


def test_accelerated_forward_does_not_expand_complete_weight(monkeypatch):
    value = quantize_w4a8(torch.randn(3, 32))
    x = torch.randn(2, 32)
    seen = {}

    def packed_linear(a, packed, group_scales, row_scales, codebook, x_scales, out_dtype):
        seen["shapes"] = (a.shape, packed.shape, group_scales.shape)
        return torch.zeros(a.shape[0], packed.shape[0], dtype=out_dtype)

    monkeypatch.setattr(w4a8_module, "_triton_packed_linear", packed_linear)
    monkeypatch.setattr(
        w4a8_module,
        "unpack_w4a8_to_int8",
        lambda *args, **kwargs: pytest.fail("accelerated forward expanded the full weight"),
    )
    result = w4a8_linear(x, value, backend="triton")
    assert result.shape == (2, 3)
    assert seen["shapes"] == ((2, 32), (3, 16), (3, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_matches_reference_on_cuda():
    # Executed only by Mirai's configured remote GPU validation contract.
    torch.manual_seed(13)
    value = quantize_w4a8(torch.randn(33, 48, device="cuda"), rotation_group_size=16)
    x = torch.randn(17, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected = w4a8_linear_reference(x, value)
    actual = w4a8_linear(x, value, backend="triton")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    assert x.grad is not None and x.grad.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_dynamic_routed_row_counts_share_one_kernel_contract():
    torch.manual_seed(14)
    value = quantize_w4a8(torch.randn(33, 48, device="cuda"), rotation_group_size=16)
    for rows in (17, 19, 23):
        x = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            w4a8_linear(x, value, backend="triton"),
            w4a8_linear_reference(x, value),
            rtol=0,
            atol=0,
        )
