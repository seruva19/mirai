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
        "sage",
        "sage2",
        "flex",
    )
    statuses = {
        status.name: status
        for status in attention.probe_attention_backends(device=torch.device("cpu"))
    }
    assert statuses["auto"].available
    for name in ("cudnn", "flash", "flash3", "flash4", "sage", "sage2"):
        assert not statuses[name].available
        assert statuses[name].reason == "CUDA device required"
    # FlexAttention carries an eager lowering, so it is device-portable.
    assert statuses["flex"].available
    assert statuses["flex"].reason == "PyTorch FlexAttention"


@pytest.mark.parametrize("capability, architecture", [((8, 9), "SM89"), ((9, 0), "SM90"), ((12, 0), "SM89")])
def test_sage2_probe_rejects_missing_cuda_extension(monkeypatch, capability, architecture) -> None:
    def dispatcher():
        pytest.fail("A backend status probe must not execute attention.")

    monkeypatch.setitem(dispatcher.__globals__, f"{architecture}_ENABLED", False)
    monkeypatch.setattr(attention, "_cuda_capability", lambda _device: capability)
    monkeypatch.setattr(attention, "_load_external_function", lambda _name, *, varlen: dispatcher)
    status = attention.attention_backend_status("sage2", device=torch.device("cuda"), varlen=True)
    assert not status.available
    assert architecture in status.reason


@pytest.mark.parametrize("backend", ["sage", "sage2"])
def test_sage_packed_dispatch_preserves_document_boundaries(monkeypatch, backend) -> None:
    monkeypatch.setattr(attention, "_cuda_capability", lambda _device: (8, 9))
    calls = []

    def dense(q, k, v, *, tensor_layout, is_causal):
        assert tensor_layout == "NHD" and not is_causal
        calls.append((q.shape[1], k.shape[1]))
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=True
        ).transpose(1, 2)

    def packed(q, k, v, cq, ck, mq, mk, *, is_causal):
        assert (mq, mk, is_causal) == (3, 4, False)
        return torch.cat([
            dense(q[qs:qe][None], k[ks:ke][None], v[ks:ke][None],
                  tensor_layout="NHD", is_causal=False)[0]
            for qs, qe, ks, ke in zip(cq.tolist(), cq.tolist()[1:], ck.tolist(), ck.tolist()[1:])
        ])

    monkeypatch.setattr(attention, "_load_external_function", lambda _name, *, varlen: packed if varlen else dense)
    torch.manual_seed(47)
    q = torch.randn(5, 2, 16, dtype=torch.float16)
    k, v = [torch.randn(7, 1, 16, dtype=torch.float16) for _ in range(2)]
    cq, ck = torch.tensor([0, 2, 5]), torch.tensor([0, 4, 7])
    kwargs = dict(cu_seqlens_q=cq, cu_seqlens_k=ck, max_seqlen_q=3, max_seqlen_k=4, backend=backend)
    observed = attention.dispatch_varlen_attention(q, k, v, **kwargs)
    assert calls == [(2, 4), (3, 3)]
    expected = torch.cat([
        dense(q[:2][None], k[:4][None], v[:4][None], tensor_layout="NHD", is_causal=False)[0],
        dense(q[2:][None], k[4:][None], v[4:][None], tensor_layout="NHD", is_causal=False)[0],
    ])
    torch.testing.assert_close(observed, expected)
    assert not attention.varlen_attention_backward_status(backend, device=q.device).available
    with pytest.raises(ValueError, match="inference-only"):
        attention.dispatch_varlen_attention(q.requires_grad_(), k, v, **kwargs)
    with pytest.raises(ValueError, match="nonempty"):
        attention.dispatch_varlen_attention(q.detach(), k, v, **{**kwargs, "max_seqlen_q": 2})


@pytest.mark.parametrize("backend", ["sage", "sage2"])
def test_sage_cuda_matches_sdpa_on_synthetic_packed_inputs(backend) -> None:
    device = torch.device("cuda")
    status = attention.attention_backend_status(backend, device=device, varlen=True)
    if not status.available:
        pytest.skip(status.reason)
    torch.manual_seed(61)
    q, k, v = [torch.randn(48, 2, 64, device=device, dtype=torch.bfloat16) for _ in range(3)]
    offsets = torch.tensor([0, 19, 48], device=device, dtype=torch.int32)
    observed = attention.dispatch_varlen_attention(
        q, k, v, cu_seqlens_q=offsets, cu_seqlens_k=offsets,
        max_seqlen_q=29, max_seqlen_k=29, backend=backend,
    )
    expected = torch.cat([
        F.scaled_dot_product_attention(
            q[start:stop].transpose(0, 1)[None],
            k[start:stop].transpose(0, 1)[None],
            v[start:stop].transpose(0, 1)[None],
        )[0].transpose(0, 1)
        for start, stop in ((0, 19), (19, 48))
    ])
    assert torch.isfinite(observed).all()
    assert F.cosine_similarity(observed.float().flatten(), expected.float().flatten(), dim=0) > 0.99


def test_flex_dispatch_matches_sdpa_outputs_and_gradients() -> None:
    torch.manual_seed(311)
    values = [
        torch.randn(1, 6, 3, 8, dtype=torch.float32, requires_grad=True)
        for _ in range(3)
    ]
    references = [value.detach().clone().requires_grad_(True) for value in values]

    observed = attention.dispatch_attention(*values, backend="flex")
    expected = attention.dispatch_attention(*references, backend="auto")
    observed.square().mean().backward()
    expected.square().mean().backward()

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
    for value, reference in zip(values, references):
        torch.testing.assert_close(value.grad, reference.grad, rtol=1e-5, atol=1e-5)


def test_flex_varlen_dispatch_isolates_packed_samples_with_gradients() -> None:
    torch.manual_seed(313)
    values = [
        torch.randn(7, 3, 8, dtype=torch.float32, requires_grad=True) for _ in range(3)
    ]
    references = [value.detach().clone().requires_grad_(True) for value in values]
    cu = torch.tensor([0, 2, 7], dtype=torch.int32)

    observed = attention.dispatch_varlen_attention(
        *values,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=5,
        max_seqlen_k=5,
        backend="flex",
    )
    expected = torch.cat(
        [
            attention.dispatch_attention(
                references[0][start:end].unsqueeze(0),
                references[1][start:end].unsqueeze(0),
                references[2][start:end].unsqueeze(0),
                backend="auto",
            ).squeeze(0)
            for start, end in ((0, 2), (2, 7))
        ],
        dim=0,
    )
    observed.square().mean().backward()
    expected.square().mean().backward()

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
    for value, reference in zip(values, references):
        torch.testing.assert_close(value.grad, reference.grad, rtol=1e-5, atol=1e-5)


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


def test_auto_varlen_falls_back_to_sdpa_with_output_and_gradient_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(227)
    values = [
        torch.randn(7, 3, 8, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    ]
    references = [value.detach().clone().requires_grad_(True) for value in values]
    cu = torch.tensor([0, 2, 7], dtype=torch.int32)

    original_status = attention.attention_backend_status

    def no_external(name: str, **kwargs: object) -> AttentionBackendStatus:
        if name in {"flash3", "flash4"}:
            return AttentionBackendStatus(name, False, "missing test kernel", None)
        return original_status(name, **kwargs)

    monkeypatch.setattr(attention, "attention_backend_status", no_external)
    observed = attention.dispatch_varlen_attention(
        *values,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=5,
        max_seqlen_k=5,
        backend="auto",
    )
    expected = torch.cat(
        [
            attention.dispatch_attention(
                references[0][start:end].unsqueeze(0),
                references[1][start:end].unsqueeze(0),
                references[2][start:end].unsqueeze(0),
                backend="auto",
            ).squeeze(0)
            for start, end in ((0, 2), (2, 7))
        ],
        dim=0,
    )
    observed.square().mean().backward()
    expected.square().mean().backward()

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    for value, reference in zip(values, references):
        torch.testing.assert_close(value.grad, reference.grad, rtol=0.0, atol=0.0)


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
