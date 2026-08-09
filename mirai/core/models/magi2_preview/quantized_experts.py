"""Frozen NF4 storage for MAGI-2 multi-head sparse-MoE routed experts.

The vendored ``CoreMultiHeadMoE`` (``mirai/vendors/magi2_preview/model/
magi2_preview.py``, SandAI, Apache-2.0) owns three packed BF16 parameters per
MoE layer: ``W_gate``/``W_up`` of shape ``[groups, d_head, d_expert]`` and
``W_down`` of shape ``[groups, d_expert, d_head]``, where ``groups`` is the
flattened ``head * num_experts + expert`` axis. Only those three tensors are
quantized here; the router projection, the expert bias, the shared experts, the
hyper-connection state, and every norm keep their released dtype.

Storage layout mirrors the shared grouped-expert NF4 contract in
``mirai/core/models/compressed_weights``: one NF4 quantization per group,
stacked along the group axis, with the two codebooks shared across groups.
Keeping the group axis addressable is what lets a contiguous group range be
dequantized on its own, which is the unit the grouped execution seam consumes.

Frozen weights never enter autograd: every dequantization runs under
``torch.no_grad`` and produces a transient buffer that its caller drops. The
backward pass re-materializes the same segment from the same packed payload
rather than retaining a dense copy.
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch import nn

from mirai.core.models.compressed_weights.quantization.quant import (
    NF4_BLOCKSIZE,
    _Nf4Meta,
    _nf4_dequantize,
    _nf4_quantize_2d,
    nf4_dequantize_stack,
    nf4_stack_dequant_supported,
)
from mirai.core.moe.runtime.specs import normalize_expert_weight_access_policy


# The three routed tensors of one vendored MoE layer, with the expert-MLP role
# each one plays. ``W_gate``/``W_up`` feed the swiglu7 gated product and
# ``W_down`` projects back onto ``d_head``.
MAGI2_ROUTED_EXPERT_TENSORS: tuple[tuple[str, str], ...] = (
    ("W_gate", "gate"),
    ("W_up", "up"),
    ("W_down", "down"),
)
MAGI2_ROUTED_EXPERT_TENSOR_NAMES: tuple[str, ...] = tuple(
    name for name, _role in MAGI2_ROUTED_EXPERT_TENSORS
)

# Child-module name the store is attached under on a vendored MoE layer.
MAGI2_NF4_STORE_ATTR = "mirai_nf4_experts"

# Access policies the quantized MAGI-2 expert path implements. ``active_dequant``
# and ``fused_kernel`` describe per-routed-expert operations that the flattened
# head-major group axis does not expose.
MAGI2_EXPERT_ACCESS_POLICIES = ("full_dequant", "chunked_dequant")


class Magi2QuantizedExpertError(ValueError):
    """Raised when quantized MAGI-2 expert storage cannot be built or served."""


class Magi2Nf4ExpertStore(nn.Module):
    """NF4 payload for the three routed expert tensors of one MoE layer.

    Buffers are non-persistent: they are derived from the released checkpoint at
    load time and are not part of the adapter state this family saves. They are
    registered as buffers rather than kept outside the module so that block
    residency moves them exactly as it moved the BF16 parameters they replace.
    """

    def __init__(
        self,
        *,
        num_groups: int,
        blocksize: int = NF4_BLOCKSIZE,
        expert_weight_access: str = "full_dequant",
        expert_dequant_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        if self.num_groups <= 0:
            raise Magi2QuantizedExpertError(
                "MAGI-2 NF4 expert storage requires a positive group count."
            )
        self._blocksize = int(blocksize)
        self._meta: _Nf4Meta | None = None
        self._shapes: dict[str, tuple[int, int, int]] = {}
        self._dtypes: dict[str, torch.dtype] = {}
        self.expert_weight_access = "full_dequant"
        self.expert_dequant_chunk_size = 0
        self.set_expert_weight_access_policy(
            expert_weight_access=expert_weight_access,
            expert_dequant_chunk_size=expert_dequant_chunk_size,
        )

    # -- policy ------------------------------------------------------------
    def set_expert_weight_access_policy(
        self,
        *,
        expert_weight_access: str,
        expert_dequant_chunk_size: int = 0,
    ) -> None:
        """Bind the dequantization granularity this store serves segments at."""
        access = normalize_expert_weight_access_policy(expert_weight_access)
        if access in {"auto", "disabled"}:
            access = "full_dequant"
        if access not in MAGI2_EXPERT_ACCESS_POLICIES:
            raise Magi2QuantizedExpertError(
                "MAGI-2 quantized experts implement memory.expert_weight_access="
                + " or ".join(f"'{value}'" for value in MAGI2_EXPERT_ACCESS_POLICIES)
                + f"; got '{access}'. The flattened head-major group axis carries "
                "one weight slice per (head, expert) pair, so there is no "
                "per-routed-expert operand for 'active_dequant' or 'fused_kernel' "
                "to address."
            )
        chunk = int(expert_dequant_chunk_size)
        if access == "chunked_dequant" and chunk <= 0:
            raise Magi2QuantizedExpertError(
                "memory.expert_weight_access='chunked_dequant' requires "
                "memory.expert_dequant_chunk_size > 0."
            )
        self.expert_weight_access = access
        self.expert_dequant_chunk_size = max(0, chunk)

    def segment_group_span(self) -> int:
        """Groups dequantized per segment under the configured access policy."""
        if self.expert_weight_access == "chunked_dequant":
            return max(1, min(self.num_groups, int(self.expert_dequant_chunk_size)))
        return self.num_groups

    @property
    def blocksize(self) -> int:
        """NF4 block size encoded by this store."""
        return self._blocksize

    def packed_state_descriptor(self) -> dict[str, object]:
        """Version-independent metadata needed to restore this exact payload."""
        if not self.is_fully_loaded() or self._meta is None:
            raise Magi2QuantizedExpertError(
                "MAGI-2 NF4 store must be complete before it can be persisted."
            )
        meta = self._meta
        return {
            "num_groups": self.num_groups,
            "blocksize": self._blocksize,
            "expert_shapes": {
                key: list(self.expert_weight_shape(key))
                for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES
            },
            "expert_dtypes": {
                key: str(self.expert_weight_dtype(key)).removeprefix("torch.")
                for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES
            },
            "nf4_meta": {
                "blocksize": meta.blocksize,
                "nested_blocksize": meta.nested_blocksize,
                "nested_dtype": str(meta.nested_dtype).removeprefix("torch."),
                "weight_dtype": str(meta.weight_dtype).removeprefix("torch."),
            },
            "buffers": sorted(name for name, _value in self.named_buffers(recurse=False)),
        }

    def restore_packed_state(
        self,
        *,
        descriptor: dict[str, object],
        buffers: dict[str, torch.Tensor],
    ) -> None:
        """Restore a validated persisted payload without dense reconstruction."""
        if int(descriptor.get("num_groups", -1)) != self.num_groups:
            raise Magi2QuantizedExpertError("Packed MAGI-2 store has incompatible groups.")
        if int(descriptor.get("blocksize", -1)) != self._blocksize:
            raise Magi2QuantizedExpertError(
                "Packed MAGI-2 store has incompatible NF4 blocksize."
            )

        def parse_dtype(value: object) -> torch.dtype:
            dtype = getattr(torch, str(value), None)
            if not isinstance(dtype, torch.dtype):
                raise Magi2QuantizedExpertError(
                    f"Packed MAGI-2 store declares unknown dtype '{value}'."
                )
            return dtype

        shapes = descriptor.get("expert_shapes")
        dtypes = descriptor.get("expert_dtypes")
        if not isinstance(shapes, dict) or not isinstance(dtypes, dict):
            raise Magi2QuantizedExpertError(
                "Packed MAGI-2 store has incomplete tensor metadata."
            )
        expected_names = set(MAGI2_ROUTED_EXPERT_TENSOR_NAMES)
        if set(shapes) != expected_names or set(dtypes) != expected_names:
            raise Magi2QuantizedExpertError(
                "Packed MAGI-2 store has incomplete expert metadata."
            )
        raw_meta = descriptor.get("nf4_meta")
        if not isinstance(raw_meta, dict):
            raise Magi2QuantizedExpertError("Packed MAGI-2 store has no NF4 metadata.")
        self._meta = _Nf4Meta(
            blocksize=int(raw_meta["blocksize"]),
            nested_blocksize=int(raw_meta["nested_blocksize"]),
            nested_dtype=parse_dtype(raw_meta["nested_dtype"]),
            weight_dtype=parse_dtype(raw_meta["weight_dtype"]),
        )
        self._shapes = {
            key: tuple(int(dim) for dim in shapes[key]) for key in expected_names
        }
        self._dtypes = {key: parse_dtype(dtypes[key]) for key in expected_names}
        expected_buffers = set(descriptor.get("buffers", []))
        if expected_buffers != set(buffers):
            raise Magi2QuantizedExpertError(
                "Packed MAGI-2 store buffer inventory does not match its descriptor."
            )
        for name, value in buffers.items():
            self._set_buffer(name, value)

    # -- storage -----------------------------------------------------------
    def _set_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value, persistent=False)

    def is_loaded(self, key: str) -> bool:
        return key in self._shapes

    def is_fully_loaded(self) -> bool:
        return all(key in self._shapes for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES)

    def expert_weight_shape(self, key: str) -> tuple[int, int, int]:
        self._require_loaded(key)
        return self._shapes[key]

    def expert_weight_dtype(self, key: str) -> torch.dtype:
        self._require_loaded(key)
        return self._dtypes[key]

    def _require_loaded(self, key: str) -> None:
        if key not in self._shapes:
            raise Magi2QuantizedExpertError(
                f"MAGI-2 NF4 expert tensor '{key}' has not been quantized."
            )

    def layout_probe(self, key: str) -> torch.Tensor:
        """Zero-storage tensor carrying the layout a dequantized segment has.

        The grouped-GEMM stride precondition is decided from real sizes,
        strides, and element size; a meta tensor supplies all three without
        allocating the dense weight the store exists to avoid.
        """
        groups, out_features, in_features = self.expert_weight_shape(key)
        return torch.empty(
            (groups, out_features, in_features),
            dtype=self.expert_weight_dtype(key),
            device="meta",
        )

    def payload_bytes(self) -> int:
        return int(
            sum(
                int(buffer.numel()) * int(buffer.element_size())
                for buffer in self.buffers()
            )
        )

    def quantized_numel(self) -> int:
        return int(
            sum(
                int(groups) * int(out_features) * int(in_features)
                for groups, out_features, in_features in self._shapes.values()
            )
        )

    def quantize_dense(self, key: str, source: torch.Tensor) -> None:
        """Quantize one packed ``[groups, out, in]`` expert tensor to NF4.

        The payload is stored on ``source``'s device, so a host-resident
        streaming load keeps the packed result in host RAM while the transient
        quantization workspace stays on the CUDA device bitsandbytes requires.
        """
        if key not in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            raise Magi2QuantizedExpertError(
                f"Unknown MAGI-2 routed expert tensor '{key}'."
            )
        if source.ndim != 3:
            raise Magi2QuantizedExpertError(
                f"MAGI-2 routed expert tensor {key} must be 3-D, got "
                f"{tuple(source.shape)}."
            )
        groups, out_features, in_features = (int(dim) for dim in source.shape)
        if groups != self.num_groups:
            raise Magi2QuantizedExpertError(
                f"MAGI-2 routed expert tensor {key} carries {groups} groups, "
                f"expected {self.num_groups}."
            )
        storage_device = source.device
        packed_stack: list[torch.Tensor] = []
        absmax_stack: list[torch.Tensor] = []
        nested_stack: list[torch.Tensor] = []
        offset_stack: list[torch.Tensor] = []
        shared_codes: dict[str, torch.Tensor] | None = None
        with torch.no_grad():
            for index in range(groups):
                fields, codes, meta = _nf4_quantize_2d(
                    source[index], blocksize=self._blocksize
                )
                if shared_codes is None:
                    shared_codes = codes
                    if self._meta is not None and self._meta != meta:
                        raise Magi2QuantizedExpertError(
                            "MAGI-2 routed expert tensors were quantized with "
                            "inconsistent NF4 metadata."
                        )
                    self._meta = meta
                packed_stack.append(fields["packed"])
                absmax_stack.append(fields["absmax"])
                nested_stack.append(fields["nested_absmax"])
                offset_stack.append(fields["offset"])
            if shared_codes is None:  # pragma: no cover - groups > 0 is validated
                raise Magi2QuantizedExpertError("MAGI-2 expert tensor is empty.")
            self._set_buffer(
                f"{key}_nf4", torch.stack(packed_stack, dim=0).contiguous().to(storage_device)
            )
            self._set_buffer(
                f"{key}_nf4_absmax",
                torch.stack(absmax_stack, dim=0).contiguous().to(storage_device),
            )
            self._set_buffer(
                f"{key}_nf4_nabsmax",
                torch.stack(nested_stack, dim=0).contiguous().to(storage_device),
            )
            self._set_buffer(
                f"{key}_nf4_offset",
                torch.stack(offset_stack, dim=0).contiguous().to(storage_device),
            )
            self._set_buffer(
                f"{key}_nf4_code", shared_codes["code"].contiguous().to(storage_device)
            )
            self._set_buffer(
                f"{key}_nf4_ncode",
                shared_codes["nested_code"].contiguous().to(storage_device),
            )
        self._shapes[key] = (groups, out_features, in_features)
        self._dtypes[key] = self._meta.weight_dtype if self._meta is not None else source.dtype

    # -- execution ---------------------------------------------------------
    def materialize_segment(
        self,
        key: str,
        group_start: int,
        group_stop: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Dequantize groups ``[group_start, group_stop)`` into a dense buffer.

        The result is a freshly allocated contiguous ``[groups, out, in]``
        tensor produced under ``no_grad``; it is never registered, saved on an
        autograd context, or cached, so it lives only as long as the matmul that
        consumes it.
        """
        self._require_loaded(key)
        start = int(group_start)
        stop = int(group_stop)
        if not 0 <= start < stop <= self.num_groups:
            raise Magi2QuantizedExpertError(
                f"MAGI-2 expert segment [{start}, {stop}) lies outside the "
                f"{self.num_groups}-group axis of '{key}'."
            )
        meta = self._meta
        if meta is None:  # pragma: no cover - set by quantize_dense
            raise Magi2QuantizedExpertError("MAGI-2 NF4 metadata is missing.")
        target = torch.device(device)
        # ``rows``/``cols`` are the two dimensions of one group's weight slice as
        # the vendored matmul consumes it (``x @ W[group]``); the shared NF4
        # helpers name the same pair out/in.
        _groups, rows, cols = self._shapes[key]
        out_features, in_features = rows, cols
        count = stop - start
        with torch.no_grad():
            packed = getattr(self, f"{key}_nf4")[start:stop].to(device=target)
            absmax = getattr(self, f"{key}_nf4_absmax")[start:stop].to(device=target)
            nested = getattr(self, f"{key}_nf4_nabsmax")[start:stop].to(device=target)
            offsets = getattr(self, f"{key}_nf4_offset")[start:stop].to(device=target)
            code = getattr(self, f"{key}_nf4_code").to(device=target)
            nested_code = getattr(self, f"{key}_nf4_ncode").to(device=target)
            if count > 1 and target.type == "cuda" and nf4_stack_dequant_supported(
                elements=int(out_features) * int(in_features), meta=meta
            ):
                stacked = nf4_dequantize_stack(
                    packed=packed,
                    absmax_q=absmax,
                    nested_absmax=nested,
                    offsets=offsets,
                    code=code,
                    nested_code=nested_code,
                    num_selected=count,
                    out_features=int(out_features),
                    in_features=int(in_features),
                    dtype=dtype,
                    meta=meta,
                )
                if stacked is not None:
                    return stacked
            codes = {"code": code, "nested_code": nested_code}
            return torch.stack(
                [
                    _nf4_dequantize(
                        {
                            "packed": packed[index],
                            "absmax": absmax[index],
                            "nested_absmax": nested[index],
                            "offset": offsets[index],
                        },
                        codes,
                        meta,
                        shape=(int(out_features), int(in_features)),
                        dtype=dtype,
                        device=target,
                    )
                    for index in range(count)
                ],
                dim=0,
            )

    def extra_repr(self) -> str:
        return (
            f"groups={self.num_groups}, format=nf4, "
            f"blocksize={self._blocksize}, access={self.expert_weight_access}, "
            f"chunk={self.expert_dequant_chunk_size}"
        )


def iter_magi2_moe_layers(transformer: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield ``(module_name, module)`` for every vendored multi-head MoE layer."""
    from mirai.vendors.magi2_preview.model.magi2_preview import CoreMultiHeadMoE

    for name, module in transformer.named_modules():
        if isinstance(module, CoreMultiHeadMoE):
            yield name, module


def magi2_expert_store(module: nn.Module) -> Magi2Nf4ExpertStore | None:
    """The NF4 store bound to a vendored MoE layer, if its experts are packed."""
    store = getattr(module, MAGI2_NF4_STORE_ATTR, None)
    return store if isinstance(store, Magi2Nf4ExpertStore) else None


def install_magi2_nf4_expert_stores(
    transformer: nn.Module,
    *,
    blocksize: int = NF4_BLOCKSIZE,
) -> dict[str, Magi2Nf4ExpertStore]:
    """Replace every MoE layer's dense expert parameters with an NF4 store.

    The dense parameters are deleted before any checkpoint tensor is read, so
    the released BF16 expert stack never has to exist in host memory. A layer
    whose experts are already packed is left untouched.
    """
    stores: dict[str, Magi2Nf4ExpertStore] = {}
    for name, module in list(iter_magi2_moe_layers(transformer)):
        existing = magi2_expert_store(module)
        if existing is not None:
            stores[name] = existing
            continue
        store = Magi2Nf4ExpertStore(
            num_groups=int(module.local_flatten_num_experts), blocksize=blocksize
        )
        for tensor_name in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            if tensor_name in module._parameters:
                del module._parameters[tensor_name]
        module.add_module(MAGI2_NF4_STORE_ATTR, store)
        stores[name] = store
    if not stores:
        raise Magi2QuantizedExpertError(
            "MAGI-2 NF4 expert quantization matched no multi-head MoE layer."
        )
    return stores


def quantize_magi2_experts_in_place(
    transformer: nn.Module,
    *,
    blocksize: int = NF4_BLOCKSIZE,
) -> dict[str, Magi2Nf4ExpertStore]:
    """Quantize already-materialized dense expert parameters, layer by layer.

    Each layer's three tensors are converted and released before the next layer
    is read, so the transient cost is one packed tensor rather than a second
    copy of the expert stack. This path still requires the dense checkpoint to
    have been loaded; ``memory.quantize_experts_on_load`` avoids that.
    """
    from mirai.vendors.magi2_preview.model.magi2_preview import CoreMultiHeadMoE

    stores: dict[str, Magi2Nf4ExpertStore] = {}
    for name, module in list(iter_magi2_moe_layers(transformer)):
        if not isinstance(module, CoreMultiHeadMoE):  # pragma: no cover - defensive
            continue
        if magi2_expert_store(module) is not None:
            continue
        store = Magi2Nf4ExpertStore(
            num_groups=int(module.local_flatten_num_experts), blocksize=blocksize
        )
        for tensor_name in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            parameter = module._parameters.get(tensor_name)
            if parameter is None:
                raise Magi2QuantizedExpertError(
                    f"MAGI-2 MoE layer '{name}' has no dense '{tensor_name}' to quantize."
                )
            if bool(parameter.requires_grad):
                raise Magi2QuantizedExpertError(
                    "MAGI-2 routed experts must be frozen before they are packed; "
                    f"'{name}.{tensor_name}' requires a gradient."
                )
            store.quantize_dense(tensor_name, parameter.data)
            del module._parameters[tensor_name]
        module.add_module(MAGI2_NF4_STORE_ATTR, store)
        stores[name] = store
    if not stores:
        raise Magi2QuantizedExpertError(
            "MAGI-2 NF4 expert quantization matched no multi-head MoE layer."
        )
    return stores


def stream_quantize_magi2_experts(
    transformer: nn.Module,
    *,
    checkpoint_dir: str,
    blocksize: int = NF4_BLOCKSIZE,
) -> dict[str, Magi2Nf4ExpertStore]:
    """Read the released expert shards one tensor at a time and pack them.

    Every routed tensor is read from its safetensors shard, quantized, and
    dropped before the next one is read, so the peak host cost is one dense
    expert tensor rather than the whole expert stack. Missing tensors fail
    explicitly: an incomplete packed layer is a lineage mismatch, not a
    degraded mode.
    """
    from safetensors import safe_open

    from mirai.vendors.magi2_preview.infra.checkpoint.magi2_checkpointing import (
        read_safetensors_weight_map,
    )

    weight_map = read_safetensors_weight_map(str(checkpoint_dir))
    stores = install_magi2_nf4_expert_stores(transformer, blocksize=blocksize)
    for module_name, store in stores.items():
        for tensor_name in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            if store.is_loaded(tensor_name):
                continue
            key = f"{module_name}.{tensor_name}" if module_name else tensor_name
            shard = weight_map.get(key)
            if shard is None:
                raise Magi2QuantizedExpertError(
                    f"MAGI-2 checkpoint under '{checkpoint_dir}' has no tensor "
                    f"'{key}'; quantized expert loading requires the released "
                    "routed expert stack."
                )
            with safe_open(shard, framework="pt") as handle:
                dense = handle.get_tensor(key)
            store.quantize_dense(tensor_name, dense)
            del dense
    return stores
