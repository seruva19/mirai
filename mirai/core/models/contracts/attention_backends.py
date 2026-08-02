from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import mirai.core.models.attention_backends as attention
from mirai.core.models.attention_backends import AttentionBackendStatus


def test_attention_backend_registry_and_cpu_probe_are_explicit() -> None:
    assert tuple(attention.ATTENTION_BACKEND_SPECS) == (
        "auto",
        "cudnn",
        "flash",
        "flash3",
        "flash4",
    )
    statuses = {
        status.name: status
        for status in attention.probe_attention_backends(device=torch.device("cpu"))
    }
    assert statuses["auto"].available
    for name in ("cudnn", "flash", "flash3", "flash4"):
        assert not statuses[name].available
        assert statuses[name].reason == "CUDA device required"


def test_auto_attention_matches_sdpa_outputs_and_gradients() -> None:
    torch.manual_seed(211)
    values = [
        torch.randn(2, 7, 3, 8, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]
    references = [value.detach().clone().requires_grad_(True) for value in values]

    observed = attention.dispatch_attention(*values, backend="auto")
    expected = F.scaled_dot_product_attention(
        references[0].transpose(1, 2),
        references[1].transpose(1, 2),
        references[2].transpose(1, 2),
        dropout_p=0.0,
    ).transpose(1, 2)
    observed.square().mean().backward()
    expected.square().mean().backward()

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    for value, reference in zip(values, references):
        torch.testing.assert_close(value.grad, reference.grad, rtol=0.0, atol=0.0)


def test_explicit_backend_never_silently_ignores_a_mask() -> None:
    value = torch.randn(1, 4, 2, 8)
    with pytest.raises(ValueError, match="maskless"):
        attention.dispatch_attention(
            value,
            value,
            value,
            attn_mask=torch.ones(4, 4, dtype=torch.bool),
            backend="flash4",
        )


def test_external_fixed_and_varlen_dispatch_use_registered_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = torch.randn(1, 4, 2, 8)
    fixed_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def fixed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **_: object) -> torch.Tensor:
        fixed_calls.append((q, k, v))
        return q + k + v

    monkeypatch.setattr(attention, "_require_backend", lambda *args, **kwargs: "flash4")
    monkeypatch.setattr(
        attention,
        "_load_external_function",
        lambda name, *, varlen: fixed,
    )
    assert torch.equal(
        attention.dispatch_attention(value, value, value, backend="flash4"),
        value * 3,
    )
    assert len(fixed_calls) == 1

    packed = torch.randn(6, 2, 8)
    cu = torch.tensor([0, 2, 6], dtype=torch.int32)

    def available(name: str, **_: object) -> AttentionBackendStatus:
        return AttentionBackendStatus(name, True, "test", (9, 0))

    monkeypatch.setattr(attention, "attention_backend_status", available)
    assert torch.equal(
        attention.dispatch_varlen_attention(
            packed,
            packed,
            packed,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=4,
            max_seqlen_k=4,
            backend="auto",
        ),
        packed * 3,
    )


def test_unavailable_explicit_backend_fails_with_probe_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = torch.randn(1, 4, 2, 8)
    monkeypatch.setattr(
        attention,
        "attention_backend_status",
        lambda name, **kwargs: AttentionBackendStatus(
            name,
            False,
            "missing test kernel",
            (9, 0),
        ),
    )
    with pytest.raises(RuntimeError, match="missing test kernel"):
        attention.dispatch_attention(value, value, value, backend="flash3")


@pytest.mark.parametrize("backend", ("cudnn", "flash"))
def test_available_pytorch_cuda_backend_matches_auto_gradients(backend: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused attention parity.")
    device = torch.device("cuda:0")
    status = attention.attention_backend_status(backend, device=device)
    if not status.available:
        pytest.skip(status.reason)
    torch.manual_seed(223)
    values = [
        torch.randn(
            2,
            64,
            4,
            32,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    references = [value.detach().clone().requires_grad_(True) for value in values]

    observed = attention.dispatch_attention(*values, backend=backend)
    expected = attention.dispatch_attention(*references, backend="auto")
    observed.float().square().mean().backward()
    expected.float().square().mean().backward()

    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
    for value, reference in zip(values, references):
        torch.testing.assert_close(
            value.grad,
            reference.grad,
            rtol=3e-2,
            atol=3e-3,
        )
