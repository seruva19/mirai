"""Segment materialization for MAGI-2 expert weights stored in shared bases."""

from __future__ import annotations

import torch

from mirai.core.models.compressed_weights.factorization.shared_basis import factorize_dense_experts

from .quantized_experts import (
    MAGI2_NF4_STORE_ATTR, MAGI2_ROUTED_EXPERT_TENSOR_NAMES, Magi2Nf4ExpertStore, iter_magi2_moe_layers,
)


class Magi2SharedBasisExpertStore(Magi2Nf4ExpertStore):
    """Keep frozen projection factors and reconstruct only requested groups."""

    quant_format = "shared_basis"

    def __init__(self, *, num_groups: int) -> None:
        super().__init__(num_groups=num_groups)
        self._axes: dict[str, str] = {}

    @torch.no_grad()
    def factorize_projection(
        self, key: str, source: torch.Tensor, *, rank: int, factor_dtype: str = "bfloat16",
    ) -> None:
        if key not in MAGI2_ROUTED_EXPERT_TENSOR_NAMES or source.ndim != 3 or source.shape[0] != self.num_groups:
            raise ValueError("MAGI-2 shared-basis projection has incompatible shape or key.")
        # MAGI-2 multiplies row activations by [in, out] matrices.
        factors = factorize_dense_experts(
            source.transpose(-1, -2), rank=rank, factor_dtype=factor_dtype,
            expert_weights=torch.ones(self.num_groups, device=source.device),
        )
        self._set_buffer(f"{key}_basis", factors.basis.detach())
        self._set_buffer(f"{key}_coefficients", factors.coefficients.detach())
        self._axes[key] = factors.axis
        self._shapes[key] = tuple(source.shape)
        self._dtypes[key] = source.dtype

    @torch.no_grad()
    def materialize_segment(self, key, group_start, group_stop, *, dtype, device, group_ids=None):
        self._require_loaded(key)
        count = self.num_groups if group_ids is None else group_ids.numel()
        if not 0 <= group_start < group_stop <= count:
            raise ValueError("MAGI-2 shared-basis segment is out of bounds.")
        coefficients = getattr(self, f"{key}_coefficients")
        if group_ids is None:
            selected = coefficients[group_start:group_stop]
        else:
            selected = coefficients.index_select(0, group_ids[group_start:group_stop].to(coefficients.device))
        basis = getattr(self, f"{key}_basis").to(device=device, dtype=dtype)
        selected = selected.to(device=device, dtype=dtype)
        dense = selected @ basis if self._axes[key] == "right" else basis @ selected
        return dense.transpose(-1, -2).contiguous()


def factorize_magi2_shared_basis_experts(
    transformer: torch.nn.Module, *, rank: int, factor_dtype: str = "bfloat16",
) -> dict[str, Magi2SharedBasisExpertStore]:
    """Replace frozen native expert projections with shared-basis stores."""
    layers = dict(iter_magi2_moe_layers(transformer))
    if not layers:
        raise ValueError("MAGI-2 shared-basis factorization matched no expert layers.")
    stores = {}
    for name, layer in layers.items():
        weights = {key: layer._parameters.get(key) for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES}
        if any(weight is None or weight.requires_grad or weight.is_meta for weight in weights.values()):
            raise ValueError("MAGI-2 shared-basis factorization requires materialized frozen projections.")
        groups = weights["W_gate"].shape[0]
        store = Magi2SharedBasisExpertStore(num_groups=groups)
        for key, weight in weights.items():
            store.factorize_projection(key, weight, rank=rank, factor_dtype=factor_dtype)
        stores[name] = store
    for name, layer in layers.items():
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            del layer._parameters[key]
        layer.add_module(MAGI2_NF4_STORE_ATTR, stores[name])
    return stores
