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

from math import prod
from typing import Callable, Iterator, Sequence

import torch
from torch import nn

from mirai.core.models.compressed_weights.quantization.blockwise_fp8 import (
    BlockwiseFP8Meta,
    dequantize_blockwise_fp8_weight,
    quantize_blockwise_fp8_weight,
    validate_blockwise_fp8_payload,
)
from mirai.core.models.compressed_weights.quantization.gguf_quant import (
    dequantize_gguf,
    gguf_num_superblocks,
    quantize_gguf,
    validate_gguf_blocks,
)
from mirai.core.models.compressed_weights.quantization.microscaling_quant import (
    MicroscalingMeta,
    dequantize_microscaling,
    quantize_microscaling,
    validate_microscaling_payload,
)

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
MAGI2_EXPERT_ACCESS_POLICIES = ("full_dequant", "chunked_dequant", "active_group_dequant")
MAGI2_FUSED_GATE_UP = "W_gate_up"
MAGI2_NF4_GROUP_FIELDS = ("nf4", "nf4_absmax", "nf4_nabsmax", "nf4_offset")
MAGI2_NF4_CODE_FIELDS = ("nf4_code", "nf4_ncode")
MAGI2_PACKED_EXPERT_FORMATS = ("nf4", "gguf_iq1", "gguf_iq2", "gguf_iq3", "fp8", "nvfp4")


class Magi2QuantizedExpertError(ValueError):
    """Raised when quantized MAGI-2 expert storage cannot be built or served."""


class Magi2Nf4ExpertStore(nn.Module):
    """NF4 payload for the three routed expert tensors of one MoE layer.

    Buffers are non-persistent: they are derived from the released checkpoint at
    load time and are not part of the adapter state this family saves. They are
    registered as buffers rather than kept outside the module so that block
    residency moves them exactly as it moved the BF16 parameters they replace.
    """

    quant_format = "nf4"

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
        if self.expert_weight_access != "full_dequant" and self.expert_dequant_chunk_size:
            return max(1, min(self.num_groups, int(self.expert_dequant_chunk_size)))
        return self.num_groups

    @property
    def blocksize(self) -> int:
        """NF4 block size encoded by this store."""
        return self._blocksize

    def packed_state_descriptor(self, *, keys=MAGI2_ROUTED_EXPERT_TENSOR_NAMES) -> dict[str, object]:
        """Version-independent metadata needed to restore this exact payload."""
        if not all(self.is_loaded(key) for key in keys) or self._meta is None:
            raise Magi2QuantizedExpertError(
                "MAGI-2 NF4 store must be complete before it can be persisted."
            )
        meta = self._meta
        return {
            "num_groups": self.num_groups,
            "blocksize": self._blocksize,
            "expert_shapes": {
                key: list(self.expert_weight_shape(key))
                for key in keys
            },
            "expert_dtypes": {
                key: str(self.expert_weight_dtype(key)).removeprefix("torch.")
                for key in keys
            },
            "nf4_meta": {
                "blocksize": meta.blocksize,
                "nested_blocksize": meta.nested_blocksize,
                "nested_dtype": str(meta.nested_dtype).removeprefix("torch."),
                "weight_dtype": str(meta.weight_dtype).removeprefix("torch."),
            },
            "buffers": sorted(getattr(self, "_packed_range_reader", None) or self._buffers),
        }

    def fused_gate_up_buffers(self) -> dict[str, torch.Tensor]:
        """Pack two independently scaled NF4 projections as interleaved lanes."""
        if not self.is_fully_loaded():
            raise Magi2QuantizedExpertError("Fused gate/up requires a complete NF4 store.")
        if self.expert_weight_shape("W_gate") != self.expert_weight_shape("W_up"):
            raise Magi2QuantizedExpertError("Fused gate/up requires equal projection shapes.")
        if self.expert_weight_dtype("W_gate") != self.expert_weight_dtype("W_up"):
            raise Magi2QuantizedExpertError("Fused gate/up requires equal projection dtypes.")

        result = {}
        for suffix in MAGI2_NF4_GROUP_FIELDS:
            gate = getattr(self, f"W_gate_{suffix}")
            up = getattr(self, f"W_up_{suffix}")
            result[f"{MAGI2_FUSED_GATE_UP}_{suffix}"] = torch.stack((gate, up), dim=1)
        for suffix in MAGI2_NF4_CODE_FIELDS:
            gate = getattr(self, f"W_gate_{suffix}")
            up = getattr(self, f"W_up_{suffix}")
            if not torch.equal(gate, up):
                raise Magi2QuantizedExpertError("Fused gate/up requires identical NF4 codebooks.")
            result[f"{MAGI2_FUSED_GATE_UP}_{suffix}"] = gate
        for suffix in (*MAGI2_NF4_GROUP_FIELDS, *MAGI2_NF4_CODE_FIELDS):
            result[f"W_down_{suffix}"] = getattr(self, f"W_down_{suffix}")
        return result

    @property
    def has_fused_gate_up(self) -> bool:
        key = f"{MAGI2_FUSED_GATE_UP}_nf4"
        reader = getattr(self, "_packed_range_reader", None)
        return key in self._buffers or (reader is not None and key in reader)

    def restore_packed_state(
        self,
        *,
        descriptor: dict[str, object],
        buffers: dict[str, torch.Tensor],
        keys=MAGI2_ROUTED_EXPERT_TENSOR_NAMES,
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
        expected_names = set(keys)
        if set(shapes) != expected_names or set(dtypes) != expected_names:
            raise Magi2QuantizedExpertError(
                "Packed MAGI-2 store has incomplete expert metadata."
            )
        raw_meta = descriptor.get("nf4_meta")
        if not isinstance(raw_meta, dict):
            raise Magi2QuantizedExpertError("Packed MAGI-2 store has no NF4 metadata.")
        meta = _Nf4Meta(
            blocksize=int(raw_meta["blocksize"]),
            nested_blocksize=int(raw_meta["nested_blocksize"]),
            nested_dtype=parse_dtype(raw_meta["nested_dtype"]),
            weight_dtype=parse_dtype(raw_meta["weight_dtype"]),
        )
        if meta.blocksize != self._blocksize or meta.nested_blocksize <= 0:
            raise Magi2QuantizedExpertError("Packed MAGI-2 store has invalid NF4 scale block sizes.")
        restored_shapes = {
            key: tuple(int(dim) for dim in shapes[key]) for key in expected_names
        }
        restored_dtypes = {key: parse_dtype(dtypes[key]) for key in expected_names}
        for shape in restored_shapes.values():
            if len(shape) != 3 or shape[0] != self.num_groups or min(shape) <= 0:
                raise Magi2QuantizedExpertError("Packed MAGI-2 store has invalid NF4 expert shapes.")
        expected_buffers = set(descriptor.get("buffers", []))
        if expected_buffers != set(buffers):
            raise Magi2QuantizedExpertError(
                "Packed MAGI-2 store buffer inventory does not match its descriptor."
            )
        fused = descriptor.get("fused_gate_up", False)
        if fused:
            if restored_shapes["W_gate"] != restored_shapes["W_up"] or restored_dtypes["W_gate"] != restored_dtypes["W_up"]:
                raise Magi2QuantizedExpertError("Packed MAGI-2 fused projections have incompatible metadata.")
        for key, shape in restored_shapes.items():
            if fused and key == "W_up":
                continue
            fused_gate = fused and key == "W_gate"
            prefix = MAGI2_FUSED_GATE_UP if fused_gate else key
            leading_shape = (self.num_groups, 2) if fused_gate else (self.num_groups,)
            elements = prod(shape[1:])
            absmax = (elements + meta.blocksize - 1) // meta.blocksize
            nested = (absmax + meta.nested_blocksize - 1) // meta.nested_blocksize
            for suffix, count in zip(MAGI2_NF4_GROUP_FIELDS, ((elements + 1) // 2, absmax, nested, 1), strict=True):
                value = buffers[f"{prefix}_{suffix}"]
                valid_dtype = (
                    value.dtype == torch.uint8
                    if suffix in {"nf4", "nf4_absmax"} else value.is_floating_point()
                )
                if (
                    not valid_dtype or value.shape[:len(leading_shape)] != leading_shape
                    or value.numel() != count * prod(leading_shape)
                ):
                    raise Magi2QuantizedExpertError(f"Invalid packed MAGI-2 NF4 field '{prefix}_{suffix}'.")
            for suffix, count in zip(MAGI2_NF4_CODE_FIELDS, (16, 256), strict=True):
                value = buffers[f"{prefix}_{suffix}"]
                if value.shape != (count,) or not value.is_floating_point():
                    raise Magi2QuantizedExpertError(f"Invalid packed MAGI-2 NF4 codebook '{prefix}_{suffix}'.")
        self._meta = meta
        self._shapes = restored_shapes
        self._dtypes = restored_dtypes
        for name in set(self._buffers) - expected_buffers:
            delattr(self, name)
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
        if key == MAGI2_FUSED_GATE_UP and self.has_fused_gate_up:
            groups, rows, columns = self._shapes["W_gate"]
            return groups, rows, columns * 2
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
        reader = getattr(self, "_packed_range_reader", None)
        if reader is not None:
            return sum(reader.tensor_nbytes(key) for key in reader)
        return int(
            sum(
                int(buffer.numel()) * int(buffer.element_size())
                for buffer in self.buffers()
            )
        )

    def enable_packed_range_reads(self, shard_path) -> None:
        from mirai.core.models.compressed_weights.packed.packed_stream import StreamingPackedTensorMapping

        self.bind_packed_range_reader(StreamingPackedTensorMapping(shard_path, backend="staged"))

    def bind_packed_range_reader(self, reader) -> None:
        self._packed_range_reader = reader
        for name in tuple(self._buffers):
            if not name.endswith(("_nf4_code", "_nf4_ncode")):
                delattr(self, name)

    def quantized_numel(self) -> int:
        return int(
            sum(
                int(groups) * int(out_features) * int(in_features)
                for groups, out_features, in_features in self._shapes.values()
            )
        )

    def quantize_dense(
        self, key: str, source: torch.Tensor, *, group_ids: Sequence[int] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
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
        indices = range(source.shape[0]) if group_ids is None else group_ids
        groups, (out_features, in_features) = len(indices), source.shape[1:]
        if groups != self.num_groups:
            raise Magi2QuantizedExpertError(
                f"MAGI-2 routed expert tensor {key} carries {groups} groups, "
                f"expected {self.num_groups}."
            )
        storage_device = source.device
        field_names = {
            "packed": "nf4", "absmax": "nf4_absmax",
            "nested_absmax": "nf4_nabsmax", "offset": "nf4_offset",
        }
        payload: dict[str, torch.Tensor] = {}
        shared_codes: dict[str, torch.Tensor] | None = None
        with torch.no_grad():
            for row, index in enumerate(indices):
                group = source[index]
                if transform is not None:
                    group = transform(group)
                fields, codes, meta = _nf4_quantize_2d(
                    group, blocksize=self._blocksize
                )
                if shared_codes is None:
                    shared_codes = codes
                    if self._meta is not None and self._meta != meta:
                        raise Magi2QuantizedExpertError(
                            "MAGI-2 routed expert tensors were quantized with "
                            "inconsistent NF4 metadata."
                        )
                    self._meta = meta
                    payload = {
                        field: fields[field].new_empty((groups, *fields[field].shape), device=storage_device)
                        for field in field_names
                    }
                for field, output in payload.items():
                    output[row].copy_(fields[field])
            if shared_codes is None:  # pragma: no cover - groups > 0 is validated
                raise Magi2QuantizedExpertError("MAGI-2 expert tensor is empty.")
            for field, suffix in field_names.items():
                self._set_buffer(f"{key}_{suffix}", payload[field])
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
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dequantize a range in the native or selected group order.

        The result is a freshly allocated contiguous ``[groups, out, in]``
        tensor produced under ``no_grad``; it is never registered, saved on an
        autograd context, or cached, so it lives only as long as the matmul that
        consumes it.
        """
        both = key == MAGI2_FUSED_GATE_UP and self.has_fused_gate_up
        logical_key = "W_gate" if both else key
        self._require_loaded(logical_key)
        start = int(group_start)
        stop = int(group_stop)
        group_count = self.num_groups if group_ids is None else group_ids.numel()
        if not 0 <= start < stop <= group_count:
            raise Magi2QuantizedExpertError(
                f"MAGI-2 expert segment [{start}, {stop}) lies outside the "
                f"{group_count}-group axis of '{key}'."
            )
        meta = self._meta
        if meta is None:  # pragma: no cover - set by quantize_dense
            raise Magi2QuantizedExpertError("MAGI-2 NF4 metadata is missing.")
        target = torch.device(device)
        # ``rows``/``cols`` are the two dimensions of one group's weight slice as
        # the vendored matmul consumes it (``x @ W[group]``); the shared NF4
        # helpers name the same pair out/in.
        _groups, rows, cols = self._shapes[logical_key]
        out_features, in_features = rows, cols
        count = stop - start
        lane = None
        storage_key = key
        if self.has_fused_gate_up and key in {"W_gate", "W_up"}:
            storage_key = MAGI2_FUSED_GATE_UP
            lane = 0 if key == "W_gate" else 1
        if both:
            count *= 2

        def projection_layout(values: torch.Tensor) -> torch.Tensor:
            if not both:
                return values
            return values.reshape(stop - start, 2, rows, cols).transpose(1, 2).reshape(
                stop - start, rows, 2 * cols
            )

        with torch.no_grad():
            indices = None if group_ids is None else group_ids[start:stop]
            reader = getattr(self, "_packed_range_reader", None)
            ids = range(start, stop) if indices is None else (indices.tolist() if reader is not None else ())

            def select(suffix: str) -> torch.Tensor:
                name = f"{storage_key}_nf4{suffix}"
                if reader is not None:
                    field = reader.get_slices_to_device(name, ids, device=target)
                    if lane is not None:
                        field = field[:, lane]
                    if both:
                        field = field.flatten(0, 1)
                    return field.contiguous()
                field = getattr(self, name)
                if lane is not None:
                    field = field[:, lane]
                if indices is None:
                    field = field[start:stop]
                else:
                    field = field.index_select(0, indices.to(device=field.device))
                if both:
                    field = field.flatten(0, 1)
                return field.to(device=target).contiguous()

            packed = select("")
            absmax = select("_absmax")
            nested = select("_nabsmax")
            offsets = select("_offset")
            code = getattr(self, f"{storage_key}_nf4_code").to(device=target)
            nested_code = getattr(self, f"{storage_key}_nf4_ncode").to(device=target)
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
                    return projection_layout(stacked)
            codes = {"code": code, "nested_code": nested_code}
            return projection_layout(torch.stack(
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
            ))

    def extra_repr(self) -> str:
        return (
            f"groups={self.num_groups}, format={self.quant_format}, "
            f"blocksize={self._blocksize}, access={self.expert_weight_access}, "
            f"chunk={self.expert_dequant_chunk_size}"
        )


class Magi2PackedExpertStore(Magi2Nf4ExpertStore):
    """Addressable expert groups backed by a shared non-NF4 codec."""

    def __init__(self, *, num_groups: int, quant_format: str, blocksize: int = NF4_BLOCKSIZE) -> None:
        super().__init__(num_groups=num_groups, blocksize=blocksize)
        if quant_format not in MAGI2_PACKED_EXPERT_FORMATS[1:]:
            raise Magi2QuantizedExpertError(f"Unsupported MAGI-2 packed format '{quant_format}'.")
        self.quant_format = quant_format
        self.expected_shapes: dict[str, tuple[int, ...]] = {}

    def _buffer_names(self, key: str) -> tuple[str, ...]:
        suffixes = ("codes",)
        if self.quant_format == "fp8":
            suffixes += ("scales",)
        elif self.quant_format == "nvfp4":
            suffixes += ("scales", "global_scale")
        return tuple(f"{key}_{suffix}" for suffix in suffixes)

    def expected_buffers(self, *, keys=MAGI2_ROUTED_EXPERT_TENSOR_NAMES) -> set[str]:
        return {name for key in keys for name in self._buffer_names(key)}

    def _nvfp4_meta(self, shape):
        return MicroscalingMeta("nvfp4", 16, shape, (-prod(shape)) % 16)

    @torch.no_grad()
    def quantize_dense(
        self, key: str, source: torch.Tensor, *, group_ids: Sequence[int] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        indices = range(source.shape[0]) if source.ndim == 3 and group_ids is None else group_ids
        if key not in MAGI2_ROUTED_EXPERT_TENSOR_NAMES or source.ndim != 3 or indices is None or len(indices) != self.num_groups:
            raise Magi2QuantizedExpertError("Invalid dense MAGI-2 packed expert input.")
        payload = None
        for row, index in enumerate(indices):
            group = source[index]
            if transform is not None:
                group = transform(group)
            if self.quant_format == "fp8":
                values = quantize_blockwise_fp8_weight(group)[:2]
            elif self.quant_format == "nvfp4":
                values = quantize_microscaling(self.quant_format, group)[:3]
            else:
                values = (quantize_gguf(self.quant_format, group),)
            if payload is None:
                payload = tuple(value.new_empty((self.num_groups, *value.shape)) for value in values)
            for output, value in zip(payload, values, strict=True):
                output[row].copy_(value)
        if payload is None:
            raise Magi2QuantizedExpertError("MAGI-2 expert tensor is empty.")
        for name, value in zip(self._buffer_names(key), payload, strict=True):
            self._set_buffer(name, value)
        self._shapes[key] = (self.num_groups, *source.shape[1:])
        self._dtypes[key] = source.dtype

    def packed_state_descriptor(self, *, keys=MAGI2_ROUTED_EXPERT_TENSOR_NAMES) -> dict[str, object]:
        if not all(self.is_loaded(key) for key in keys):
            raise Magi2QuantizedExpertError("MAGI-2 packed store is incomplete.")
        return {
            "num_groups": self.num_groups,
            "expert_shapes": self._shapes,
            "expert_dtypes": {key: str(dtype).removeprefix("torch.") for key, dtype in self._dtypes.items()},
            "buffers": sorted(self.expected_buffers(keys=keys)),
        }

    def restore_packed_state(self, *, descriptor, buffers, keys=MAGI2_ROUTED_EXPERT_TENSOR_NAMES) -> None:
        shapes, dtypes = descriptor.get("expert_shapes"), descriptor.get("expert_dtypes")
        key_set = set(keys)
        if (
            descriptor.get("num_groups") != self.num_groups
            or set(buffers) != self.expected_buffers(keys=keys)
            or not isinstance(shapes, dict) or set(shapes) != key_set
            or not isinstance(dtypes, dict) or set(dtypes) != key_set
        ):
            raise Magi2QuantizedExpertError("Invalid MAGI-2 packed expert metadata or inventory.")
        restored_shapes = {}
        restored_dtypes = {}
        for key in keys:
            shape = tuple(shapes[key])
            dtype = getattr(torch, str(dtypes[key]), None)
            if (
                len(shape) != 3 or shape[0] != self.num_groups
                or any(type(dim) is not int or dim < 1 for dim in shape)
                or dtype not in {torch.float16, torch.bfloat16, torch.float32}
            ):
                raise Magi2QuantizedExpertError(f"Invalid packed shape or dtype for '{key}'.")
            group_shape = shape[1:]
            if key in self.expected_shapes and shape != self.expected_shapes[key]:
                raise Magi2QuantizedExpertError(f"Packed shape for '{key}' does not match the model.")
            fields = tuple(buffers[name] for name in self._buffer_names(key))
            if any(field.ndim < 1 or field.shape[0] != self.num_groups for field in fields):
                raise Magi2QuantizedExpertError(f"Invalid packed group axis for '{key}'.")
            if self.quant_format == "fp8":
                validate_blockwise_fp8_payload(*fields, BlockwiseFP8Meta(group_shape))
            elif self.quant_format == "nvfp4":
                validate_microscaling_payload(*fields, self._nvfp4_meta(group_shape))
            else:
                blocks = fields[0]
                if blocks.ndim != 3 or blocks.shape[1] != gguf_num_superblocks(prod(group_shape)):
                    raise Magi2QuantizedExpertError(f"Invalid GGUF block count for '{key}'.")
                validate_gguf_blocks(self.quant_format, blocks.flatten(0, 1))
            restored_shapes[key] = shape
            restored_dtypes[key] = dtype
        for name, value in buffers.items():
            self._set_buffer(name, value)
        self._shapes = restored_shapes
        self._dtypes = restored_dtypes

    @torch.no_grad()
    def materialize_segment(self, key, group_start, group_stop, *, dtype, device, group_ids=None):
        self._require_loaded(key)
        count = self.num_groups if group_ids is None else group_ids.numel()
        if not 0 <= group_start < group_stop <= count:
            raise Magi2QuantizedExpertError("MAGI-2 packed expert segment is out of bounds.")
        fields = []
        reader = getattr(self, "_packed_range_reader", None)
        if reader is not None:
            ids = range(group_start, group_stop) if group_ids is None else group_ids[group_start:group_stop].tolist()
        for name in self._buffer_names(key):
            if reader is not None:
                fields.append(reader.get_slices_to_device(name, ids, device=device))
                continue
            value = getattr(self, name)
            if group_ids is None:
                value = value[group_start:group_stop]
            else:
                value = value.index_select(0, group_ids[group_start:group_stop].to(value.device))
            fields.append(value.to(device=device))
        shape = self._shapes[key][1:]
        if self.quant_format == "fp8":
            return dequantize_blockwise_fp8_weight(*fields, BlockwiseFP8Meta(shape), dtype=dtype, device=device)
        if self.quant_format == "nvfp4":
            meta = self._nvfp4_meta(shape)
            return torch.stack([
                dequantize_microscaling(*group, meta, dtype=dtype, device=device)
                for group in zip(*fields, strict=True)
            ])
        return torch.stack([
            dequantize_gguf(self.quant_format, group, shape=shape, dtype=dtype, device=device)
            for group in fields[0]
        ])



def _new_expert_store(*, num_groups: int, blocksize: int, quant_format: str) -> Magi2Nf4ExpertStore:
    if quant_format == "nf4":
        return Magi2Nf4ExpertStore(num_groups=num_groups, blocksize=blocksize)
    return Magi2PackedExpertStore(num_groups=num_groups, blocksize=blocksize, quant_format=quant_format)


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
    quant_format: str = "nf4",
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
            if existing.quant_format != quant_format:
                raise Magi2QuantizedExpertError("MAGI-2 layer already carries a different packed format.")
            stores[name] = existing
            continue
        store = _new_expert_store(
            num_groups=int(module.local_flatten_num_experts), blocksize=blocksize, quant_format=quant_format
        )
        if isinstance(store, Magi2PackedExpertStore):
            store.expected_shapes = {
                key: tuple(module._parameters[key].shape)
                for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES
            }
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
    quant_format: str = "nf4",
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
        store = _new_expert_store(
            num_groups=int(module.local_flatten_num_experts), blocksize=blocksize, quant_format=quant_format
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
    quant_format: str = "nf4",
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
    stores = install_magi2_nf4_expert_stores(transformer, blocksize=blocksize, quant_format=quant_format)
    pending = []
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
            pending.append((store, tensor_name, key, shard))
    for store, tensor_name, key, shard in pending:
        with safe_open(shard, framework="pt") as handle:
            dense = handle.get_tensor(key)
        store.quantize_dense(tensor_name, dense)
        del dense
    return stores
