"""Behavioral contract for the native MAGI-2 refiner range-union attention.

The vendored operator the refiner dispatches through
(``torch.ops.magi2.flex_flash_attn_func``, MagiAttention's CUDA extension) is
absent from every environment that does not ship the MAGI-2 release, and the
eager path vendored beside it imports FlashAttention-2. Both references are
reproduced here in exact torch so the contract runs on CPU:

* ``_dense_range_attention`` is a straightforward masked softmax over the union
  of every key range paired with a query position. It is the definition the
  native path claims to implement.
* ``_vendored_eager_reference`` is the vendored code itself
  (``_split_q_range_with_no_overlap`` and ``_flash_attn_with_correction``),
  executed with a torch stand-in installed for the FlashAttention-2 kernel it
  calls. The range splitting and the log-sum-exp merge under test are therefore
  SandAI's own, not a restatement of them.

Attribution: SandAI MAGI-2-preview, Apache-2.0
(https://github.com/SandAI-org/MAGI-2-preview).
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from mirai.core.models.magi2_preview.refiner_attention import (
    MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT,
    Magi2RefinerAttentionUnsupported,
    Magi2RefinerFlexAttentionBackend,
    attach_refiner_attention_backend,
    build_range_union_mask,
    flex_range_attention,
    normalize_refiner_attention_backend,
    refiner_required_magi2_ops,
    resolve_magi2_refiner_attention,
)
from mirai.core.models.magi2_preview.refiner import (
    attach_refiner_attention_chunking,
    attach_refiner_mlp_chunking,
)


HEAD_DIM = 16


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #
def _expand_kv(tensor: torch.Tensor, heads_q: int) -> torch.Tensor:
    return tensor.repeat_interleave(heads_q // int(tensor.shape[1]), dim=1).float()


def _allowed_pairs(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    *,
    query_tokens: int,
    key_tokens: int,
) -> torch.Tensor:
    """Dense ``[Q, K]`` union of every paired rectangle, built one row at a time."""
    allowed = torch.zeros(query_tokens, key_tokens, dtype=torch.bool)
    for (q_start, q_end), (k_start, k_end) in zip(
        q_ranges.tolist(), k_ranges.tolist()
    ):
        allowed[q_start:q_end, k_start:k_end] = True
    return allowed


def _dense_range_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One softmax per query over the union of its allowed key ranges."""
    heads_q = int(query.shape[1])
    allowed = _allowed_pairs(
        q_ranges,
        k_ranges,
        query_tokens=int(query.shape[0]),
        key_tokens=int(key.shape[0]),
    )
    keys = _expand_kv(key, heads_q)
    values = _expand_kv(value, heads_q)
    scores = torch.einsum("qhd,khd->hqk", query.float(), keys) / (HEAD_DIM**0.5)
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
    log_sum_exp = torch.logsumexp(scores, dim=-1)
    weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    attended = torch.einsum("hqk,khd->qhd", weights, values)
    return attended.to(query.dtype), log_sum_exp.transpose(0, 1)


def _torch_flash_attn_func(query, key, value, return_attn_probs=False, **_kwargs):
    """Exact torch stand-in for ``flash_attn.flash_attn_interface.flash_attn_func``.

    Shapes follow the FlashAttention-2 interface the vendored reference calls
    it through: ``[batch, tokens, heads, head_dim]`` in, attended output plus
    ``[batch, heads, tokens]`` natural-log softmax denominators out.
    """
    heads_q = int(query.shape[2])
    scale = float(query.shape[3]) ** -0.5
    keys = key.repeat_interleave(heads_q // int(key.shape[2]), dim=2).float()
    values = value.repeat_interleave(heads_q // int(value.shape[2]), dim=2).float()
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), keys) * scale
    log_sum_exp = torch.logsumexp(scores, dim=-1)
    attended = torch.einsum(
        "bhqk,bkhd->bqhd", torch.softmax(scores, dim=-1), values
    ).to(query.dtype)
    if return_attn_probs:
        return attended, log_sum_exp, None
    return attended


@pytest.fixture(scope="module")
def vendored_refiner_module():
    """Import the vendored refiner with a torch stand-in for FlashAttention-2."""
    if "flash_attn" not in sys.modules:
        package = types.ModuleType("flash_attn")
        package.__path__ = []
        interface = types.ModuleType("flash_attn.flash_attn_interface")
        interface.flash_attn_func = _torch_flash_attn_func
        layers = types.ModuleType("flash_attn.layers")
        layers.__path__ = []
        rotary = types.ModuleType("flash_attn.layers.rotary")
        rotary.apply_rotary_emb = lambda *args, **kwargs: args[0]
        package.flash_attn_interface = interface
        package.layers = layers
        layers.rotary = rotary
        sys.modules.update(
            {
                "flash_attn": package,
                "flash_attn.flash_attn_interface": interface,
                "flash_attn.layers": layers,
                "flash_attn.layers.rotary": rotary,
            }
        )
    import mirai.vendors.magi2_preview.model.magi2_refiner as module

    module.flash_attn_interface = None  # keeps the FA3 probe off this path
    return module


def _vendored_eager_reference(
    module, query, key, value, q_ranges, k_ranges
) -> tuple[torch.Tensor, torch.Tensor]:
    return module._custom_flex_flash_attn_func(
        query, key, value, q_ranges, k_ranges
    )


def _inputs(
    *,
    query_tokens: int,
    key_tokens: int,
    heads_q: int,
    heads_kv: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape_q = (query_tokens, heads_q, HEAD_DIM)
    shape_kv = (key_tokens, heads_kv, HEAD_DIM)
    return (
        torch.randn(shape_q, generator=generator).to(torch.bfloat16),
        torch.randn(shape_kv, generator=generator).to(torch.bfloat16),
        torch.randn(shape_kv, generator=generator).to(torch.bfloat16),
    )


def _ranges(pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.int32).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# Mask construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "q_pairs,k_pairs",
    [
        ([(0, 32)], [(0, 32)]),
        ([(0, 8), (8, 16), (16, 32)], [(0, 12), (4, 20), (12, 32)]),
        # Overlapping query ranges: positions 8..16 are covered twice.
        ([(0, 16), (8, 32)], [(0, 10), (20, 32)]),
        # Overlapping key ranges over the same query span.
        ([(0, 32), (0, 32)], [(0, 20), (10, 32)]),
        # Ranges in descending order, and a duplicated pair.
        ([(16, 32), (0, 16), (0, 16)], [(0, 16), (16, 32), (16, 32)]),
    ],
)
def test_segment_lookup_reproduces_the_dense_union(q_pairs, k_pairs) -> None:
    q_ranges, k_ranges = _ranges(q_pairs), _ranges(k_pairs)
    mask = build_range_union_mask(
        q_ranges, k_ranges, query_tokens=32, key_tokens=32
    )
    observed = mask.allowed[mask.segment_of]
    expected = _allowed_pairs(
        q_ranges, k_ranges, query_tokens=32, key_tokens=32
    )
    assert torch.equal(observed, expected)
    assert torch.equal(mask.query_has_keys, expected.any(dim=1))


def test_uncovered_query_positions_map_to_the_empty_segment() -> None:
    mask = build_range_union_mask(
        _ranges([(8, 16)]), _ranges([(0, 32)]), query_tokens=32, key_tokens=32
    )
    outside = torch.ones(32, dtype=torch.bool)
    outside[8:16] = False
    assert not bool(mask.query_has_keys[outside].any())
    assert bool(mask.query_has_keys[8:16].all())


def test_repeated_ranges_stay_compact() -> None:
    """Range count must not expand into range-by-covered-segment pairs."""
    q_pairs: list[tuple[int, int]] = []
    k_pairs: list[tuple[int, int]] = []
    for segment in range(64):
        start = segment * 128
        end = start + 128
        # Duplicate metadata rows exercise compact range coverage without
        # materializing a range-by-covered-segment index table.
        for _ in range(128):
            q_pairs.extend(((start, end), (start, end)))
            k_pairs.extend(((max(0, start - 128), min(8192, end + 128)), (8000, 8192)))
    mask = build_range_union_mask(
        _ranges(q_pairs), _ranges(k_pairs), query_tokens=8192, key_tokens=8192
    )
    assert mask.interval_starts.shape == (65, 2)
    assert mask.interval_ends.shape == (65, 2)
    block_mask = mask.block_mask(query_tokens=8192)
    assert block_mask.shape == (1, 1, 8192, 8192)


# --------------------------------------------------------------------------- #
# Attention semantics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "heads_q,heads_kv,q_pairs,k_pairs",
    [
        (4, 4, [(0, 32)], [(0, 32)]),
        (4, 4, [(0, 8), (8, 16), (16, 24), (24, 32)], [(0, 16), (0, 24), (8, 32), (16, 32)]),
        (8, 2, [(0, 16), (16, 32)], [(0, 24), (8, 32)]),
        (8, 1, [(0, 32)], [(4, 28)]),
    ],
)
def test_native_path_matches_the_dense_union_softmax(
    heads_q, heads_kv, q_pairs, k_pairs
) -> None:
    query, key, value = _inputs(
        query_tokens=32, key_tokens=32, heads_q=heads_q, heads_kv=heads_kv
    )
    q_ranges, k_ranges = _ranges(q_pairs), _ranges(k_pairs)
    observed, observed_lse = flex_range_attention(
        query,
        key,
        value,
        q_ranges,
        k_ranges,
        torch.zeros(q_ranges.shape[0], dtype=torch.int32),
        32,
    )
    expected, expected_lse = _dense_range_attention(
        query, key, value, q_ranges, k_ranges
    )
    assert observed.shape == query.shape
    assert observed.dtype == query.dtype
    assert observed_lse.shape == (32, heads_q)
    assert observed_lse.dtype == torch.float32
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(observed_lse, expected_lse, rtol=1e-3, atol=1e-3)


def test_disjoint_ranges_match_the_vendored_eager_reference(
    vendored_refiner_module,
) -> None:
    """The regime the released refiner produces: no query sees a key twice."""
    query, key, value = _inputs(
        query_tokens=32, key_tokens=32, heads_q=4, heads_kv=2, seed=3
    )
    # Frame-local windows over the video span plus one dense row onto the tail,
    # the shape refiner_data_proxy emits: every query segment reaches each of
    # its keys through exactly one range.
    q_ranges = _ranges([(0, 8), (8, 16), (16, 24), (24, 32), (0, 24)])
    k_ranges = _ranges([(0, 16), (0, 24), (8, 24), (16, 32), (24, 32)])
    observed, _ = flex_range_attention(query, key, value, q_ranges, k_ranges, None, 32)
    expected, _ = _vendored_eager_reference(
        vendored_refiner_module, query, key, value, q_ranges, k_ranges
    )
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)


def test_overlapping_ranges_take_the_union_where_the_vendored_reference_double_counts(
    vendored_refiner_module,
) -> None:
    """Documented divergence.

    Two rows covering one query segment with overlapping key ranges describe a
    single key set. The vendored reference attends to each range separately and
    merges the partial results by their log-sum-exps, so every key in the
    intersection enters the denominator twice and is weighted twice. The native
    path masks the union once, which is the operator's stated semantics.
    """
    query, key, value = _inputs(
        query_tokens=32, key_tokens=32, heads_q=4, heads_kv=4, seed=7
    )
    q_ranges = _ranges([(0, 32), (0, 32)])
    k_ranges = _ranges([(0, 24), (8, 32)])

    observed, _ = flex_range_attention(query, key, value, q_ranges, k_ranges, None, 32)
    expected, _ = _dense_range_attention(query, key, value, q_ranges, k_ranges)
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)

    vendored, _ = _vendored_eager_reference(
        vendored_refiner_module, query, key, value, q_ranges, k_ranges
    )
    assert not torch.allclose(
        vendored.float(), expected.float(), rtol=2e-2, atol=2e-2
    )


def test_empty_key_range_yields_a_zero_row_and_an_infinite_negative_lse() -> None:
    query, key, value = _inputs(
        query_tokens=32, key_tokens=32, heads_q=4, heads_kv=2, seed=11
    )
    q_ranges = _ranges([(0, 16), (16, 32)])
    k_ranges = _ranges([(8, 8), (0, 32)])
    observed, observed_lse = flex_range_attention(
        query, key, value, q_ranges, k_ranges, None, 16
    )
    assert torch.equal(observed[:16], torch.zeros_like(observed[:16]))
    assert bool(torch.isneginf(observed_lse[:16]).all())
    assert bool(torch.isfinite(observed_lse[16:]).all())
    assert not torch.equal(observed[16:], torch.zeros_like(observed[16:]))


def test_query_positions_outside_every_range_yield_zero_rows() -> None:
    query, key, value = _inputs(
        query_tokens=32, key_tokens=32, heads_q=4, heads_kv=4, seed=13
    )
    observed, observed_lse = flex_range_attention(
        query, key, value, _ranges([(8, 24)]), _ranges([(0, 32)]), None, 16
    )
    assert torch.equal(observed[:8], torch.zeros_like(observed[:8]))
    assert torch.equal(observed[24:], torch.zeros_like(observed[24:]))
    assert bool(torch.isneginf(observed_lse[:8]).all())
    assert bool(torch.isfinite(observed_lse[8:24]).all())


def test_asymmetric_key_length_is_accepted() -> None:
    query, key, value = _inputs(
        query_tokens=24, key_tokens=40, heads_q=4, heads_kv=2, seed=17
    )
    q_ranges, k_ranges = _ranges([(0, 12), (12, 24)]), _ranges([(0, 30), (10, 40)])
    observed, _ = flex_range_attention(query, key, value, q_ranges, k_ranges, None, 12)
    expected, _ = _dense_range_attention(query, key, value, q_ranges, k_ranges)
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)


def test_backend_reuses_block_metadata_for_identical_range_content() -> None:
    query, key, value = _inputs(
        query_tokens=32, key_tokens=32, heads_q=4, heads_kv=2, seed=19
    )
    q_ranges = _ranges([(0, 16), (16, 32)])
    k_ranges = _ranges([(0, 24), (8, 32)])
    backend = Magi2RefinerFlexAttentionBackend()
    first, _ = backend.execute(
        query,
        key,
        value,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=None,
        max_seqlen_q=16,
    )
    cached_mask = backend._cached_range_mask
    cached_blocks = backend._cached_block_mask
    second, _ = backend.execute(
        query,
        key,
        value,
        q_ranges=q_ranges.clone(),
        k_ranges=k_ranges.clone(),
        attn_type_map=None,
        max_seqlen_q=16,
    )
    assert backend._cached_range_mask is cached_mask
    assert backend._cached_block_mask is cached_blocks
    assert torch.equal(first, second)


def test_refiner_mlp_token_chunking_preserves_modality_outputs(
    vendored_refiner_module,
) -> None:
    config = vendored_refiner_module.MLPConfig(
        hidden_size=16,
        intermediate_size=16,
        activation_type=vendored_refiner_module.MLPActivationType.SWIGLU7,
        params_dtype=torch.bfloat16,
        num_modality=3,
        gated_act=True,
    )
    mlp = vendored_refiner_module.MLP(config).eval()
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.copy_(
                torch.linspace(
                    -0.05,
                    0.05,
                    parameter.numel(),
                    dtype=parameter.dtype,
                ).reshape_as(parameter)
            )
    mapping = torch.tensor([0] * 5 + [1] * 3 + [2] * 4)
    dispatcher = vendored_refiner_module.ModalityDispatcher(mapping, 3)
    value = torch.randn(12, 16, generator=torch.Generator().manual_seed(31)).to(
        torch.bfloat16
    )
    expected = mlp(value, dispatcher)
    assert attach_refiner_mlp_chunking(mlp, chunk_tokens=4) == 1
    observed = mlp(value, dispatcher)
    assert observed.dtype == torch.float32
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_refiner_attention_chunking_preserves_projection_and_rotary_outputs(
    vendored_refiner_module,
) -> None:
    config = vendored_refiner_module.AttentionConfig(
        hidden_size=16,
        num_heads_q=2,
        num_heads_kv=2,
        head_dim=8,
        params_dtype=torch.bfloat16,
        checkpoint_qk_layernorm_rope=False,
        num_modality=3,
        num_layers=1,
        use_local_attn=True,
        enable_attn_gating=True,
    )
    attention = vendored_refiner_module.Attention(config).eval()
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.copy_(
                torch.linspace(
                    -0.04,
                    0.04,
                    parameter.numel(),
                    dtype=parameter.dtype,
                ).reshape_as(parameter)
            )

    class EchoBackend:
        @staticmethod
        def execute(query, key, value, **_kwargs):
            del key, value
            return query, torch.zeros(
                query.shape[:2], dtype=torch.float32, device=query.device
            )

    attention._mirai_refiner_attention_backend = EchoBackend()
    mapping = torch.tensor([2, 0, 1, 0, 2, 0, 1, 2, 0])
    dispatcher = vendored_refiner_module.ModalityDispatcher(mapping, 3)
    hidden = torch.randn(9, 16, generator=torch.Generator().manual_seed(37)).to(
        torch.bfloat16
    )[dispatcher.permute_mapping]
    rope = torch.randn(9, 8, generator=torch.Generator().manual_seed(41))
    handler = types.SimpleNamespace(
        q_ranges=_ranges([(0, 9)]),
        k_ranges=_ranges([(0, 9)]),
        attn_type_map=torch.zeros(1, dtype=torch.int32),
        max_seqlen_q=9,
        auto_range_merge=False,
        sparse_load=False,
    )
    args = (
        hidden,
        rope,
        dispatcher.permute_mapping,
        dispatcher.inv_permute_mapping,
        None,
        handler,
        dispatcher,
        [],
    )
    expected = attention(*args)
    assert attach_refiner_attention_chunking(attention, chunk_tokens=3) == 1
    observed = attention(*args)
    assert observed.dtype == torch.bfloat16
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------- #
# Explicit rejections
# --------------------------------------------------------------------------- #
def test_non_full_attention_type_is_rejected() -> None:
    query, key, value = _inputs(
        query_tokens=32, key_tokens=32, heads_q=4, heads_kv=4
    )
    with pytest.raises(Magi2RefinerAttentionUnsupported, match="full attention type"):
        flex_range_attention(
            query,
            key,
            value,
            _ranges([(0, 32)]),
            _ranges([(0, 32)]),
            torch.ones(1, dtype=torch.int32),
            32,
        )


def test_max_seqlen_q_shorter_than_the_longest_range_is_rejected() -> None:
    query, key, value = _inputs(
        query_tokens=32, key_tokens=32, heads_q=4, heads_kv=4
    )
    with pytest.raises(Magi2RefinerAttentionUnsupported, match="max_seqlen_q"):
        flex_range_attention(
            query, key, value, _ranges([(0, 32)]), _ranges([(0, 32)]), None, 16
        )


@pytest.mark.parametrize(
    "q_pairs,k_pairs,message",
    [
        ([(0, 32)], [(0, 32), (0, 16)], "share shape"),
        ([(16, 8)], [(0, 32)], "start <= end"),
        ([(0, 64)], [(0, 32)], "q_ranges must lie"),
        ([(0, 32)], [(0, 64)], "k_ranges must lie"),
    ],
)
def test_inconsistent_ranges_are_rejected(q_pairs, k_pairs, message) -> None:
    with pytest.raises(Magi2RefinerAttentionUnsupported, match=message):
        build_range_union_mask(
            _ranges(q_pairs), _ranges(k_pairs), query_tokens=32, key_tokens=32
        )


def test_head_ratio_that_is_not_a_whole_multiple_is_rejected() -> None:
    query, key, value = _inputs(
        query_tokens=32, key_tokens=32, heads_q=6, heads_kv=4
    )
    with pytest.raises(Magi2RefinerAttentionUnsupported, match="whole"):
        flex_range_attention(
            query, key, value, _ranges([(0, 32)]), _ranges([(0, 32)]), None, 32
        )


# --------------------------------------------------------------------------- #
# Selection, precedence, and attachment
# --------------------------------------------------------------------------- #
def test_backend_names_normalize_and_reject_unknown_values() -> None:
    assert normalize_refiner_attention_backend(None) == "auto"
    assert normalize_refiner_attention_backend("") == "auto"
    assert normalize_refiner_attention_backend(" Native_Flex ") == "native_flex"
    assert (
        normalize_refiner_attention_backend(MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT)
        == "auto"
    )
    with pytest.raises(Magi2RefinerAttentionUnsupported, match="must be one of"):
        normalize_refiner_attention_backend("flash3")


def test_precedence_prefers_a_registered_operator_under_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mirai.core.models.magi2_preview.refiner_attention as refiner_attention
    import mirai.vendors.magi2_preview.common.magi_compiler_compat as compat

    monkeypatch.setattr(
        refiner_attention, "_magi_attention_hopper_kernel_available", lambda: False
    )
    monkeypatch.setattr(compat, "missing_magi2_custom_ops", lambda names: ())
    assert resolve_magi2_refiner_attention("auto") is None
    # An explicit selection is not a preference; it overrides the operator.
    assert isinstance(
        resolve_magi2_refiner_attention("native_flex"),
        Magi2RefinerFlexAttentionBackend,
    )

    monkeypatch.setattr(
        compat, "missing_magi2_custom_ops", lambda names: tuple(names)
    )
    assert isinstance(
        resolve_magi2_refiner_attention("auto"), Magi2RefinerFlexAttentionBackend
    )
    assert resolve_magi2_refiner_attention("vendor_eager") is None


def test_auto_prefers_the_authors_single_gpu_hopper_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mirai.core.models.magi2_preview.refiner_attention as refiner_attention
    import mirai.vendors.magi2_preview.common.magi_compiler_compat as compat

    monkeypatch.setattr(
        compat, "missing_magi2_custom_ops", lambda names: tuple(names)
    )
    monkeypatch.setattr(
        refiner_attention, "_magi_attention_hopper_kernel_available", lambda: True
    )
    assert resolve_magi2_refiner_attention("auto") is None


def test_attachment_covers_every_refiner_attention_module(
    vendored_refiner_module,
) -> None:
    module = vendored_refiner_module.Attention.__new__(
        vendored_refiner_module.Attention
    )
    torch.nn.Module.__init__(module)
    module._mirai_refiner_attention_backend = None
    transformer = torch.nn.Module()
    transformer.block = module

    backend = Magi2RefinerFlexAttentionBackend()
    assert attach_refiner_attention_backend(transformer, backend) == 1
    assert module._mirai_refiner_attention_backend is backend
    assert attach_refiner_attention_backend(transformer, None) == 1
    assert module._mirai_refiner_attention_backend is None


def test_attachment_without_any_refiner_attention_layer_fails_explicitly() -> None:
    with pytest.raises(Magi2RefinerAttentionUnsupported, match="matched no MAGI-2"):
        attach_refiner_attention_backend(
            torch.nn.Module(), Magi2RefinerFlexAttentionBackend()
        )


def test_bound_backend_removes_the_flex_operator_from_the_precondition() -> None:
    class _Arch:
        num_layers = 4
        local_attn_layers = (0, 1, 2, 3)

    all_local = _Arch()
    assert refiner_required_magi2_ops(all_local, None) == ("flex_flash_attn_func",)
    assert refiner_required_magi2_ops(
        all_local, Magi2RefinerFlexAttentionBackend()
    ) == ()

    class _Mixed(_Arch):
        local_attn_layers = (0, 1)

    mixed = _Mixed()
    assert refiner_required_magi2_ops(mixed, None) == (
        "flex_flash_attn_func",
        "flash_attn_func",
    )
    assert refiner_required_magi2_ops(
        mixed, Magi2RefinerFlexAttentionBackend()
    ) == ("flash_attn_func",)


def test_vendored_dispatch_uses_the_bound_backend(vendored_refiner_module) -> None:
    """The seam, not the operator namespace, decides which path runs."""
    query, key, value = _inputs(
        query_tokens=32, key_tokens=32, heads_q=4, heads_kv=2, seed=23
    )
    q_ranges, k_ranges = _ranges([(0, 16), (16, 32)]), _ranges([(0, 24), (8, 32)])
    observed = vendored_refiner_module.flex_flash_attn_with_cp(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        q_ranges,
        k_ranges,
        torch.zeros(2, dtype=torch.int32),
        16,
        False,
        False,
        [],
        Magi2RefinerFlexAttentionBackend(),
    )
    expected, _ = _dense_range_attention(query, key, value, q_ranges, k_ranges)
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------- #
# Refine-stage probe: which path it selects and what it then requires
# --------------------------------------------------------------------------- #
def _refine_probe():
    from mirai.core.models.magi2_preview.contracts import native_refine_step

    return native_refine_step


def test_refine_probe_selects_the_native_path_and_requires_no_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no MagiCompiler the probe binds the native backend and needs no op.

    The reduced architecture the probe builds is all-local, so once the native
    backend is bound the flex operator is served by it and the dense operator is
    never reached. The probe must therefore require an empty operator set rather
    than the full tuple, which is what previously drove it into MagiAttention.
    """
    import mirai.vendors.magi2_preview.common.magi_compiler_compat as compat

    probe = _refine_probe()
    monkeypatch.delenv(probe.MAGI2_REFINER_ATTENTION_BACKEND_ENV, raising=False)
    monkeypatch.setattr(compat, "missing_magi2_custom_ops", lambda names: tuple(names))

    requested = probe._requested_attention_backend()
    assert requested == MAGI2_REFINER_ATTENTION_BACKEND_DEFAULT
    backend = resolve_magi2_refiner_attention(requested)
    assert isinstance(backend, Magi2RefinerFlexAttentionBackend)

    arch = probe._reduced_refiner_config()
    assert set(arch.local_attn_layers) == set(range(arch.num_layers))
    assert refiner_required_magi2_ops(arch, backend) == ()
    # Without a bound backend the same architecture does reach the operator, so
    # the empty requirement is a consequence of the binding, not of the config.
    assert refiner_required_magi2_ops(arch, None) == ("flex_flash_attn_func",)


@pytest.mark.parametrize(
    "value, expected_native",
    [("native_flex", True), ("vendor_eager", False), (" Auto ", True)],
)
def test_refine_probe_environment_override_selects_the_path(
    monkeypatch: pytest.MonkeyPatch, value: str, expected_native: bool
) -> None:
    """The probe-scoped override forces either path on one machine."""
    import mirai.vendors.magi2_preview.common.magi_compiler_compat as compat

    probe = _refine_probe()
    monkeypatch.setattr(compat, "missing_magi2_custom_ops", lambda names: tuple(names))
    monkeypatch.setenv(probe.MAGI2_REFINER_ATTENTION_BACKEND_ENV, value)

    requested = probe._requested_attention_backend()
    assert requested == value.strip().lower()
    backend = resolve_magi2_refiner_attention(requested)
    assert isinstance(backend, Magi2RefinerFlexAttentionBackend) is expected_native


def test_refine_probe_rejects_an_unknown_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _refine_probe()
    monkeypatch.setenv(probe.MAGI2_REFINER_ATTENTION_BACKEND_ENV, "magi_attention")
    with pytest.raises(Magi2RefinerAttentionUnsupported, match="must be one of"):
        probe._requested_attention_backend()
