"""Segment materialization for MAGI-2 expert weights stored in shared bases."""

from __future__ import annotations

import hashlib
import json

import torch

from mirai.core.models.compressed_weights.factorization.shared_basis import factorize_shared_basis_blocks

from .quantized_experts import (
    MAGI2_NF4_STORE_ATTR, MAGI2_ROUTED_EXPERT_TENSOR_NAMES, Magi2Nf4ExpertStore, iter_magi2_moe_layers,
)

MAGI2_BASIS_SOURCE_METADATA = "magi2_basis_source_sha256"


def fingerprint_magi2_basis_checkpoint(checkpoint_dir: str) -> str:
    """Identify the complete name-to-shard content mapping independently of its path."""
    from mirai.core.lineage import sha256_file
    from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import read_safetensors_weight_map

    weight_map = read_safetensors_weight_map(checkpoint_dir)
    if not weight_map:
        raise ValueError("MAGI-2 basis checkpoint has no source tensors.")
    shard_digests = {path: sha256_file(path) for path in sorted(set(weight_map.values()))}
    inventory = [(key, shard_digests[path]) for key, path in sorted(weight_map.items())]
    return hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode("utf-8")).hexdigest()


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
        self.factorize_projection_blocks(
            key, lambda start, stop: source[start:stop],
            shape=tuple(source.shape), source_dtype=source.dtype, rank=rank, factor_dtype=factor_dtype,
        )

    def factorize_projection_blocks(self, key, load_block, *, shape, source_dtype, rank, factor_dtype):
        # MAGI-2 multiplies row activations by [in, out] matrices.
        groups, inputs, outputs = shape

        def load_transposed(start, stop):
            value = load_block(start, stop)
            if not value.is_floating_point():
                raise ValueError(f"MAGI-2 shared-basis projection {key!r} requires floating-point weights.")
            return value.to(dtype=source_dtype).transpose(-1, -2)

        factors = factorize_shared_basis_blocks(
            load_transposed, shape=(groups, outputs, inputs), rank=rank, factor_dtype=factor_dtype,
        )
        self._set_buffer(f"{key}_basis", factors.basis.detach())
        self._set_buffer(f"{key}_coefficients", factors.coefficients.detach())
        self._axes[key] = factors.axis
        self._shapes[key] = shape
        self._dtypes[key] = source_dtype

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

    def expected_buffers(self) -> set[str]:
        return {f"{key}_{part}" for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES for part in ("basis", "coefficients")}

    def packed_state_descriptor(self):
        if not self.is_fully_loaded():
            raise ValueError("MAGI-2 shared-basis store is incomplete.")
        return {
            "num_groups": self.num_groups,
            "expert_shapes": self._shapes,
            "expert_dtypes": {key: str(dtype).removeprefix("torch.") for key, dtype in self._dtypes.items()},
            "axes": self._axes,
            "buffers": sorted(self.expected_buffers()),
        }

    def restore_packed_state(self, *, descriptor, buffers, expected_rank=None):
        shapes = descriptor["expert_shapes"]
        dtypes = descriptor["expert_dtypes"]
        axes = descriptor["axes"]
        keys = set(MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
        if (
            descriptor["num_groups"] != self.num_groups or set(buffers) != self.expected_buffers()
            or set(shapes) != keys or set(dtypes) != keys or set(axes) != keys
        ):
            raise ValueError("MAGI-2 shared-basis artifact has incompatible projection metadata.")
        allowed = {torch.float32, torch.float16, torch.bfloat16}
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            groups, inputs, outputs = shapes[key]
            axis = axes[key]
            basis = buffers[f"{key}_basis"]
            coefficients = buffers[f"{key}_coefficients"]
            dtype = getattr(torch, str(dtypes[key]), None)
            if basis.ndim != 2 or axis not in {"left", "right"}:
                raise ValueError(f"Invalid shared basis for {key!r}.")
            rank = basis.shape[0 if axis == "right" else 1]
            basis_shape = (rank, inputs) if axis == "right" else (outputs, rank)
            coeff_shape = (groups, outputs, rank) if axis == "right" else (groups, rank, inputs)
            if (
                groups != self.num_groups or not 0 < rank <= min(inputs, outputs)
                or (expected_rank is not None and rank != expected_rank)
                or basis.shape != basis_shape or coefficients.shape != coeff_shape
                or dtype not in allowed or basis.dtype not in allowed or coefficients.dtype != basis.dtype
                or not torch.isfinite(basis).all() or not torch.isfinite(coefficients).all()
            ):
                raise ValueError(f"Invalid shared-basis factors for {key!r}.")
        for name, value in buffers.items():
            self._set_buffer(name, value)
        self._axes = dict(axes)
        self._shapes = {key: tuple(shape) for key, shape in shapes.items()}
        self._dtypes = {key: getattr(torch, dtypes[key]) for key in keys}


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


def stream_factorize_magi2_shared_basis_experts(
    transformer: torch.nn.Module, *, checkpoint_dir: str, rank: int, factor_dtype: str = "bfloat16",
) -> dict[str, Magi2SharedBasisExpertStore]:
    """Build all stores from source shards before replacing any expert parameters."""
    from safetensors import safe_open

    from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import read_safetensors_weight_map

    layers = dict(iter_magi2_moe_layers(transformer))
    if not layers:
        raise ValueError("MAGI-2 shared-basis factorization matched no expert layers.")
    weight_map = read_safetensors_weight_map(checkpoint_dir)
    stores = {}
    for name, layer in layers.items():
        store = Magi2SharedBasisExpertStore(num_groups=layer.local_flatten_num_experts)
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            full_key = f"{name}.{key}"
            if full_key not in weight_map:
                raise ValueError(f"MAGI-2 checkpoint is missing shared-basis projection {full_key!r}.")
            target = layer._parameters[key]
            with safe_open(weight_map[full_key], framework="pt", device="cpu") as shard:
                source = shard.get_slice(full_key)
                if tuple(source.get_shape()) != target.shape:
                    raise ValueError(f"MAGI-2 shared-basis projection {full_key!r} has incompatible shape or dtype.")
                store.factorize_projection_blocks(
                    key, lambda start, stop, shard_slice=source: shard_slice[start:stop],
                    shape=tuple(target.shape), source_dtype=target.dtype, rank=rank, factor_dtype=factor_dtype,
                )
        stores[name] = store
    for name, layer in layers.items():
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            del layer._parameters[key]
        layer.add_module(MAGI2_NF4_STORE_ATTR, stores[name])
    return stores
