"""Behavioral contract for the trainable MAGI-2 FlexAttention path.

The vendored reference ``_torch_varlen_attention_with_sink`` defines the
semantics; every probe here compares the flex path against it, or against a
closed-form softmax written from those semantics.
"""

from __future__ import annotations

import math

import pytest
import torch

from mirai.core.models.attention_backends import (
    attention_backend_status,
    flex_document_ids,
    normalize_attention_backend,
)
from mirai.core.models.magi2_preview.flex_attention import (
    Magi2FlexAttentionBackend,
    Magi2FlexAttentionUnsupported,
    attach_flex_attention_backend,
    flex_varlen_attention_with_sink,
    resolve_magi2_flex_attention,
    validate_flex_attention_support,
)
from mirai.vendors.magi2_preview.model.magi2_preview import (
    VarlenHandler,
    _torch_varlen_attention_with_sink,
    flash_attn_with_sink,
)


CU_SEQLENS = torch.tensor([0, 5, 12], dtype=torch.int32)


def _packed_inputs(
    *,
    heads_q: int = 4,
    heads_kv: int | None = None,
    head_dim: int = 16,
    tokens: int = 12,
    sink_tokens: int = 1,
    dtype: torch.dtype = torch.float32,
    seed: int = 17,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    heads_kv = heads_q if heads_kv is None else heads_kv
    query = torch.randn(1, tokens, heads_q, head_dim, dtype=dtype, requires_grad=True)
    key = torch.randn(1, tokens, heads_kv, head_dim, dtype=dtype, requires_grad=True)
    value = torch.randn(1, tokens, heads_kv, head_dim, dtype=dtype, requires_grad=True)
    sink = torch.randn(sink_tokens, heads_q, dtype=torch.float32, requires_grad=True)
    return query, key, value, sink


def _cloned(tensors: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


@pytest.mark.parametrize("sink_tokens", (1, 2))
@pytest.mark.parametrize("softcap", (-1.0, 8.0))
def test_flex_matches_reference_outputs_and_every_gradient(
    softcap: float, sink_tokens: int
) -> None:
    inputs = _packed_inputs(sink_tokens=sink_tokens)
    references = _cloned(inputs)

    observed = flex_varlen_attention_with_sink(
        *inputs[:3],
        cu_seqlens_q=CU_SEQLENS,
        cu_seqlens_k=CU_SEQLENS,
        softcap=softcap,
        sink=inputs[3],
    )
    expected = _torch_varlen_attention_with_sink(
        *references[:3],
        cu_seqlens_q=CU_SEQLENS,
        cu_seqlens_k=CU_SEQLENS,
        softcap=softcap,
        sink=references[3],
    )
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)

    observed.square().mean().backward()
    expected.square().mean().backward()
    for name, value, reference in zip("qkvs", inputs, references):
        assert value.grad is not None, f"{name} received no gradient"
        torch.testing.assert_close(value.grad, reference.grad, rtol=1e-5, atol=1e-5)


def test_flex_matches_reference_in_bfloat16() -> None:
    inputs = _packed_inputs(dtype=torch.bfloat16, seed=41)
    references = _cloned(inputs)

    observed = flex_varlen_attention_with_sink(
        *inputs[:3],
        cu_seqlens_q=CU_SEQLENS,
        cu_seqlens_k=CU_SEQLENS,
        softcap=-1.0,
        sink=inputs[3],
    )
    expected = _torch_varlen_attention_with_sink(
        *references[:3],
        cu_seqlens_q=CU_SEQLENS,
        cu_seqlens_k=CU_SEQLENS,
        softcap=-1.0,
        sink=references[3],
    )
    assert observed.dtype is torch.bfloat16
    torch.testing.assert_close(observed, expected, rtol=3e-2, atol=3e-2)

    observed.float().square().mean().backward()
    expected.float().square().mean().backward()
    for name, value, reference in zip("qkvs", inputs, references):
        assert value.grad is not None, f"{name} received no gradient"
        torch.testing.assert_close(value.grad, reference.grad, rtol=5e-2, atol=5e-2)


def test_flex_matches_reference_for_grouped_query_attention() -> None:
    inputs = _packed_inputs(heads_q=4, heads_kv=2, seed=53)
    references = _cloned(inputs)

    observed = flex_varlen_attention_with_sink(
        *inputs[:3],
        cu_seqlens_q=CU_SEQLENS,
        cu_seqlens_k=CU_SEQLENS,
        softcap=-1.0,
        sink=inputs[3],
    )
    expected = _torch_varlen_attention_with_sink(
        *references[:3],
        cu_seqlens_q=CU_SEQLENS,
        cu_seqlens_k=CU_SEQLENS,
        softcap=-1.0,
        sink=references[3],
    )
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)

    observed.square().mean().backward()
    expected.square().mean().backward()
    for value, reference in zip(inputs, references):
        torch.testing.assert_close(value.grad, reference.grad, rtol=1e-5, atol=1e-5)


def test_sink_enters_the_denominator_unscaled_with_zero_value_contribution() -> None:
    """One query against two keys, checked against a hand-written softmax."""
    head_dim = 4
    scale = head_dim**-0.5
    query = torch.zeros(1, 3, 1, head_dim, dtype=torch.float64)
    key = torch.zeros(1, 3, 1, head_dim, dtype=torch.float64)
    value = torch.zeros(1, 3, 1, head_dim, dtype=torch.float64)
    # A single sample of three tokens; only the first query is inspected.
    query[0, 0, 0, 0] = 1.0
    key[0, 0, 0, 0] = 2.0
    key[0, 1, 0, 0] = -1.0
    key[0, 2, 0, 0] = 0.5
    value[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    value[0, 1, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    value[0, 2, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    sink = torch.tensor([[0.75]], dtype=torch.float32)
    cu = torch.tensor([0, 3], dtype=torch.int32)

    observed = flex_varlen_attention_with_sink(
        query,
        key,
        value,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        softcap=-1.0,
        sink=sink,
    )

    logits = [2.0 * scale, -1.0 * scale, 0.5 * scale]
    weights = [math.exp(logit) for logit in logits]
    # The sink logit is not multiplied by the head-dim scale and carries no value.
    denominator = sum(weights) + math.exp(0.75)
    expected = torch.tensor(
        [weights[0], weights[1], weights[2], 0.0], dtype=torch.float64
    ) / denominator
    # FlexAttention accumulates in float32 even for a float64 request, so the
    # agreement bound is float32 rather than the input dtype.
    torch.testing.assert_close(observed[0, 0], expected, rtol=1e-6, atol=1e-6)


def test_absent_sink_reduces_to_plain_varlen_attention() -> None:
    inputs = _packed_inputs(seed=61)
    references = _cloned(inputs)
    empty = torch.zeros(0, 4, dtype=torch.float32)

    observed = flex_varlen_attention_with_sink(
        *inputs[:3],
        cu_seqlens_q=CU_SEQLENS,
        cu_seqlens_k=CU_SEQLENS,
        softcap=-1.0,
        sink=empty,
    )
    expected = _torch_varlen_attention_with_sink(
        *references[:3],
        cu_seqlens_q=CU_SEQLENS,
        cu_seqlens_k=CU_SEQLENS,
        softcap=-1.0,
        sink=None,
    )
    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_packed_samples_never_share_attention_mass() -> None:
    """A token of sample A must not read values that only sample B carries."""
    head_dim = 8
    tokens = 12
    cu = CU_SEQLENS
    query = torch.zeros(1, tokens, 1, head_dim, dtype=torch.float64)
    key = torch.zeros(1, tokens, 1, head_dim, dtype=torch.float64)
    value = torch.zeros(1, tokens, 1, head_dim, dtype=torch.float64)
    # Every key in the second sample is a far better match than any key in the
    # first, so leaked attention would dominate the first sample's output.
    key[0, :5, 0, 0] = 1.0
    key[0, 5:, 0, 0] = 40.0
    query[0, :, 0, 0] = 1.0
    value[0, :5, 0, 0] = 1.0
    value[0, 5:, 0, 1] = 1.0
    sink = torch.zeros(1, 1, dtype=torch.float32)

    observed = flex_varlen_attention_with_sink(
        query,
        key,
        value,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        softcap=-1.0,
        sink=sink,
    )
    assert observed[:5, 0, 1].abs().max().item() == 0.0
    assert observed[:5, 0, 0].min().item() > 0.0
    assert observed[5:, 0, 0].abs().max().item() == 0.0
    assert observed[5:, 0, 1].min().item() > 0.0


def test_document_ids_reject_inconsistent_cumulative_lengths() -> None:
    assert flex_document_ids(CU_SEQLENS, total=12).tolist() == [0] * 5 + [1] * 7
    with pytest.raises(ValueError, match="must start at 0"):
        flex_document_ids(torch.tensor([0, 5, 12], dtype=torch.int32), total=13)


def test_attached_backend_is_used_in_grad_mode_and_absence_is_the_reference() -> None:
    inputs = _packed_inputs(seed=73)
    references = _cloned(inputs)
    handler = VarlenHandler(
        cu_seqlens_q=CU_SEQLENS,
        cu_seqlens_k=CU_SEQLENS,
        max_seqlen_q=7,
        max_seqlen_k=7,
    )
    split_sizes = torch.tensor([12], dtype=torch.int32)

    assert torch.is_grad_enabled()
    routed = flash_attn_with_sink(
        *inputs[:3],
        handler,
        split_sizes,
        -1.0,
        inputs[3],
        Magi2FlexAttentionBackend(),
    )
    unrouted = flash_attn_with_sink(
        *references[:3],
        handler,
        split_sizes,
        -1.0,
        references[3],
    )
    torch.testing.assert_close(routed, unrouted, rtol=1e-5, atol=1e-5)

    routed.square().mean().backward()
    unrouted.square().mean().backward()
    for value, reference in zip(inputs, references):
        torch.testing.assert_close(value.grad, reference.grad, rtol=1e-5, atol=1e-5)


def test_attachment_covers_every_attention_module_and_clears_back_to_default() -> None:
    module = _attention_stub()
    transformer = torch.nn.Module()
    transformer.block = module

    backend = Magi2FlexAttentionBackend()
    assert attach_flex_attention_backend(transformer, backend) == 1
    assert module._mirai_attention_backend is backend
    assert attach_flex_attention_backend(transformer, None) == 1
    assert module._mirai_attention_backend is None


def test_attachment_without_any_attention_layer_fails_explicitly() -> None:
    with pytest.raises(Magi2FlexAttentionUnsupported, match="matched no MAGI-2"):
        attach_flex_attention_backend(torch.nn.Module(), Magi2FlexAttentionBackend())


def test_config_surface_selects_flex_only_when_requested() -> None:
    assert normalize_attention_backend("flex") == "flex"

    class _ModelConfig:
        def __init__(self, backend: str) -> None:
            self.attention_backend = backend

    assert resolve_magi2_flex_attention(_ModelConfig("auto")) is None
    assert resolve_magi2_flex_attention(_ModelConfig("flash3")) is None
    assert isinstance(
        resolve_magi2_flex_attention(_ModelConfig("Flex")), Magi2FlexAttentionBackend
    )


def test_unavailable_flex_backend_is_reported_rather_than_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mirai.core.models.magi2_preview.flex_attention as flex_module
    from mirai.core.models.attention_backends import AttentionBackendStatus

    monkeypatch.setattr(
        flex_module,
        "attention_backend_status",
        lambda name, **kwargs: AttentionBackendStatus(
            name, False, "missing test flex lowering", None
        ),
    )
    with pytest.raises(Magi2FlexAttentionUnsupported, match="missing test flex lowering"):
        validate_flex_attention_support()


def test_flex_is_registered_and_available_on_the_local_device() -> None:
    status = attention_backend_status("flex", device=torch.device("cpu"), varlen=True)
    assert status.available
    assert status.reason == "PyTorch FlexAttention"


def test_mismatched_sink_head_count_fails_explicitly() -> None:
    query, key, value, _ = _packed_inputs(seed=83)
    with pytest.raises(Magi2FlexAttentionUnsupported, match="heads"):
        flex_varlen_attention_with_sink(
            query,
            key,
            value,
            cu_seqlens_q=CU_SEQLENS,
            cu_seqlens_k=CU_SEQLENS,
            softcap=-1.0,
            sink=torch.randn(1, 3),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_cuda_flex_matches_reference_outputs_and_gradients() -> None:
    device = torch.device("cuda:0")
    inputs = [
        tensor.detach().to(device).requires_grad_(True)
        for tensor in _packed_inputs(head_dim=64, dtype=torch.bfloat16, seed=97)[:3]
    ]
    sink = torch.randn(1, 4, device=device, dtype=torch.float32, requires_grad=True)
    inputs.append(sink)
    references = _cloned(tuple(inputs))
    cu = CU_SEQLENS.to(device)

    observed = flex_varlen_attention_with_sink(
        *inputs[:3],
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        softcap=-1.0,
        sink=inputs[3],
        compile_kernel=False,
        compile_block_mask=False,
    )
    expected = _torch_varlen_attention_with_sink(
        *references[:3],
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        softcap=-1.0,
        sink=references[3],
    )
    torch.testing.assert_close(observed, expected, rtol=3e-2, atol=3e-2)

    observed.float().square().mean().backward()
    expected.float().square().mean().backward()
    for value, reference in zip(inputs, references):
        torch.testing.assert_close(value.grad, reference.grad, rtol=5e-2, atol=5e-2)


def _attention_stub() -> torch.nn.Module:
    """Build a vendored ``Attention`` module without its heavy weight tensors."""
    from mirai.vendors.magi2_preview.model.magi2_preview import Attention

    module = Attention.__new__(Attention)
    torch.nn.Module.__init__(module)
    module._mirai_attention_backend = None
    return module
