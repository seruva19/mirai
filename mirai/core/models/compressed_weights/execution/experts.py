"""Frozen quantized grouped-expert storage and routed execution."""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Mapping

from mirai.core.moe.storage.aliases import (
    PrototypeCalibrationObserver,
    coalesce_alias_routes,
    validate_logical_to_physical,
)
from mirai.core.moe.runtime.specs import (
    CANONICAL_PACKED_EXPERT_MLP_SPEC,
    ExpertMLPExecutionSpec,
    normalize_expert_weight_access_policy,
    resolve_moe_batched_dequant,
    resolve_moe_batched_gather,
    resolve_moe_dispatch_mode,
    resolve_moe_dispatch_preprocess,
    resolve_moe_pair_dequant,
)
from mirai.core.moe.runtime.gemm import resolve_moe_gemm_backend

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from .active_expert_lora import ActiveExpertLoRA
from .dispatch_preprocess import build_dispatch_plan
from ..quantization.quant import (
    NF4_BLOCKSIZE,
    _Nf4Meta,
    _dequantize_weight,
    _nf4_dequantize,
    _nf4_quantize_2d,
    _quantize_weight,
    _validate_rotation,
    best_group_size,
    nf4_dequantize_stack,
    nf4_stack_dequant_supported,
    normalize_quant_format,
)
from ..quantization.gguf_quant import (
    GGUF_FORMATS,
    _GgufMeta,
    dequantize_gguf,
    gguf_meta_for,
    quantize_gguf,
)
from ..quantization.microscaling_quant import (
    MICROSCALING_FORMATS,
    MicroscalingMeta,
    dequantize_microscaling,
    quantize_microscaling,
    validate_microscaling_payload,
)
from ..quantization.blockwise_fp8 import (
    BLOCKWISE_FP8_FORMATS,
    BlockwiseFP8Meta,
    blockwise_fp8_batched_linear,
    blockwise_fp8_linear,
    dequantize_blockwise_fp8_weight,
    quantize_blockwise_fp8_weight,
    validate_blockwise_fp8_payload,
)
from .expert_gather import BatchedExpertGatherStrategy
from .expert_device_cache import ExpertDeviceCache
from .routed_output_observer import RoutedOutputObserverHost
from .linear import (
    _FrozenDequantExpertLinear,
    _FrozenDequantBatchedLinear,
    _FrozenDequantBatchedLinearPair,
    _FrozenDequantMegaBlocksLinear,
    _FrozenTritonGroupedLinear,
    _FrozenTritonGroupedLinearPair,
    _persistent_dispatch_available,
    _triton_dispatch_available,
)

logger = logging.getLogger(__name__)


class CompressedGroupedExperts(RoutedOutputObserverHost, nn.Module):
    """Provider-described grouped experts stored in frozen packed formats."""

    def __init__(
        self,
        base: nn.Module,
        *,
        group_sizes: str | int | Iterable[int] | None = None,
        expert_weight_access: str = "full_dequant",
        expert_dequant_chunk_size: int = 0,
        quant_format: str = "int8",
        nf4_blocksize: int = NF4_BLOCKSIZE,
        execution_spec: ExpertMLPExecutionSpec | None = None,
    ):
        super().__init__()
        if torch is None:  # pragma: no cover
            raise RuntimeError("CompressedGroupedExperts requires torch.")
        # Direct construction predates provider-owned execution specs. Keep the
        # canonical fallback as a compatibility boundary; shipping preparation
        # requires and passes the model-owned spec explicitly.
        resolved_spec = (
            execution_spec
            or getattr(base, "mirai_expert_mlp_spec", None)
            or CANONICAL_PACKED_EXPERT_MLP_SPEC
        )
        self._init_empty(
            num_experts=int(getattr(base, "num_experts")),
            group_sizes=group_sizes,
            expert_weight_access=expert_weight_access,
            expert_dequant_chunk_size=expert_dequant_chunk_size,
            quant_format=quant_format,
            nf4_blocksize=nf4_blocksize,
            execution_spec=resolved_spec,
        )
        for key in resolved_spec.tensor_names:
            self.load_dense_weight(key, getattr(base, key))

    @classmethod
    def from_empty(
        cls,
        *,
        num_experts: int,
        group_sizes: str | int | Iterable[int] | None = None,
        expert_weight_access: str = "full_dequant",
        expert_dequant_chunk_size: int = 0,
        quant_format: str = "int8",
        nf4_blocksize: int = NF4_BLOCKSIZE,
        execution_spec: ExpertMLPExecutionSpec = CANONICAL_PACKED_EXPERT_MLP_SPEC,
    ) -> CompressedGroupedExperts:
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module._init_empty(
            num_experts=int(num_experts),
            group_sizes=group_sizes,
            expert_weight_access=expert_weight_access,
            expert_dequant_chunk_size=expert_dequant_chunk_size,
            quant_format=quant_format,
            nf4_blocksize=nf4_blocksize,
            execution_spec=execution_spec,
        )
        return module

    def _init_empty(
        self,
        *,
        num_experts: int,
        group_sizes: str | int | Iterable[int] | None,
        expert_weight_access: str,
        expert_dequant_chunk_size: int,
        quant_format: str = "int8",
        nf4_blocksize: int = NF4_BLOCKSIZE,
        execution_spec: ExpertMLPExecutionSpec,
    ) -> None:
        self.num_experts = int(num_experts)
        if not isinstance(execution_spec, ExpertMLPExecutionSpec):
            raise TypeError("execution_spec must use ExpertMLPExecutionSpec.")
        self.mirai_expert_mlp_spec = execution_spec
        self._group_sizes = group_sizes
        self._quant_format = normalize_quant_format(quant_format)
        self._nf4_blocksize = int(nf4_blocksize)
        self._nf4_meta: _Nf4Meta | None = None
        self._nf4_shapes: dict[str, tuple[int, ...]] = {}
        # GGUF sub-4-bit (gguf_iq4/gguf_iq3/gguf_iq2): one packed buffer per key holds
        # the canonical GGUF block bytes; _gguf_shapes records the dense [E,out,in].
        self._gguf_meta: _GgufMeta | None = None
        self._gguf_shapes: dict[str, tuple[int, ...]] = {}
        self._microscaling_meta: dict[str, MicroscalingMeta] = {}
        self._microscaling_shapes: dict[str, tuple[int, ...]] = {}
        self._blockwise_fp8_meta: dict[str, BlockwiseFP8Meta] = {}
        self._blockwise_fp8_shapes: dict[str, tuple[int, ...]] = {}
        # Cache of (key_a, key_b) -> bool: whether the two keys share bit-identical
        # NF4 codebooks so their packed stacks can be concatenated into one dequant.
        self._nf4_pair_codebook_cache: dict[tuple[str, str], bool] = {}
        self.expert_weight_access = normalize_expert_weight_access_policy(expert_weight_access)
        if self.expert_weight_access in {"auto", "disabled"}:
            self.expert_weight_access = "full_dequant"
        chunk = int(expert_dequant_chunk_size)
        if self.expert_weight_access in {"chunked_dequant", "fused_kernel"} and chunk <= 0:
            # Resolve an unset chunk from the hardware tier. Explicit values are
            # never overridden; an unmatched device still fails below.
            from mirai.config.hardware_tiers import resolve_expert_dequant_chunk_size

            chunk = int(resolve_expert_dequant_chunk_size(chunk))
        if self.expert_weight_access in {"chunked_dequant", "fused_kernel"} and chunk <= 0:
            raise ValueError(
                "chunked_dequant/fused_kernel requires expert_dequant_chunk_size > 0 "
                "(no hardware tier matched this device to supply a default)."
            )
        self.expert_dequant_chunk_size = max(0, chunk)
        self._loaded_dense_keys: set[str] = set()
        self.expert_lora = nn.ModuleDict()
        self._expert_device_cache = ExpertDeviceCache()
        self._expert_device_cache_namespace = ""
        self._routed_adapter_gate = None
        self._packed_source: Any | None = None
        self._packed_tensor_names: dict[str, str] = {}
        self._packed_shapes: dict[str, tuple[int, ...]] = {}
        self._expert_rotation_keys: set[str] = set()
        self._physical_weight_provider: Any | None = None
        self.logical_num_experts = int(num_experts)
        self.prototype_logical_ids: tuple[int, ...] = ()
        self._logical_to_physical: torch.Tensor | None = None
        self._prototype_calibration_observer: PrototypeCalibrationObserver | None = None
        self._whitening_calibration_observer: Any | None = None
        self._flexmoe_channel_mask: torch.Tensor | None = None
        self._flexmoe_taylor_observer: Any | None = None

    def set_flexmoe_taylor_observer(self, observer: Any | None) -> None:
        """Attach the exact parameter-gradient reconstruction calibration seam."""

        if observer is None:
            self._flexmoe_taylor_observer = None
            return
        if not self.expert_mlp_spec.uses_gated_product:
            raise ValueError("FlexMoE Taylor ranking requires a gated-product expert MLP.")
        if self._flexmoe_channel_mask is not None:
            raise ValueError(
                "FlexMoE Taylor ranking and action-mask learning are separate stages."
            )
        if self.expert_lora or self._routed_adapter_gate is not None:
            raise ValueError(
                "FlexMoE Taylor ranking requires frozen experts without active "
                "expert adapters."
            )
        if int(getattr(observer, "num_experts", -1)) != self.num_experts:
            raise ValueError("FlexMoE Taylor observer expert topology mismatch.")
        shape = self.expert_weight_shape(self.projection_for_role("gate"))
        if int(getattr(observer, "intermediate_size", -1)) != int(shape[1]):
            raise ValueError("FlexMoE Taylor observer channel topology mismatch.")
        if not callable(getattr(observer, "capture", None)):
            raise TypeError("FlexMoE Taylor observer must expose capture().")
        self._flexmoe_taylor_observer = observer

    def set_flexmoe_channel_mask(self, mask: torch.Tensor | None) -> None:
        """Attach one differentiable original-channel mask per expert."""

        if mask is None:
            self._flexmoe_channel_mask = None
            return
        if not self.expert_mlp_spec.uses_gated_product:
            raise ValueError("FlexMoE channel masks require a gated-product expert MLP.")
        if self._flexmoe_taylor_observer is not None:
            raise ValueError(
                "FlexMoE action-mask learning cannot run during Taylor ranking."
            )
        if self.has_ragged_intermediate_widths():
            raise RuntimeError(
                "FlexMoE action learning requires the full ranked source, not an "
                "already pruned ragged artifact."
            )
        value = torch.as_tensor(mask)
        shape = self.expert_weight_shape(self.projection_for_role("gate"))
        expected = (self.num_experts, int(shape[1]))
        if tuple(value.shape) != expected or not value.is_floating_point():
            raise ValueError(
                f"FlexMoE channel mask must be floating {expected}, got "
                f"{tuple(value.shape)}."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("FlexMoE channel mask contains non-finite values.")
        if bool(((value.detach() < 0.0) | (value.detach() > 1.0)).any().item()):
            raise ValueError("FlexMoE channel mask values must lie in [0, 1].")
        self._flexmoe_channel_mask = value

    @property
    def flexmoe_channel_mask_active(self) -> bool:
        return self._flexmoe_channel_mask is not None

    @property
    def flexmoe_taylor_observer_active(self) -> bool:
        return self._flexmoe_taylor_observer is not None

    @property
    def expert_mlp_spec(self) -> ExpertMLPExecutionSpec:
        return self.mirai_expert_mlp_spec

    def projection_for_role(self, role: str) -> str:
        return self.expert_mlp_spec.tensor_for_role(role)

    def _activate_expert_hidden(self, value: torch.Tensor) -> torch.Tensor:
        activation = self.expert_mlp_spec.activation
        if activation == "silu":
            return F.silu(value)
        if activation == "gelu":
            return F.gelu(value)
        if activation == "relu":
            return F.relu(value)
        if activation == "identity":
            return value
        raise RuntimeError(f"Unsupported expert activation {activation!r}.")

    def _combine_expert_inputs(
        self,
        primary: torch.Tensor,
        secondary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        activated = self._activate_expert_hidden(primary)
        if self.expert_mlp_spec.combiner == "activated":
            if secondary is not None:
                raise RuntimeError("Activated expert MLP does not accept an up projection.")
            return activated
        if self.expert_mlp_spec.combiner == "gated_product":
            if secondary is None:
                raise RuntimeError("Gated expert MLP requires an up projection.")
            return activated * secondary
        raise RuntimeError(
            f"Unsupported expert combiner {self.expert_mlp_spec.combiner!r}."
        )

    def _apply_flexmoe_channel_mask(
        self,
        hidden: torch.Tensor,
        expert_index: int,
    ) -> torch.Tensor:
        mask = self._flexmoe_channel_mask
        if mask is None:
            return hidden
        return hidden * mask[int(expert_index)].to(
            device=hidden.device,
            dtype=hidden.dtype,
        )

    def _expert_chunk_groups(self, chunk_size: int) -> tuple[tuple[int, ...], ...]:
        """Return static execution groups, bucketed by physical expert width."""

        size = max(1, int(chunk_size))
        if not self.has_ragged_intermediate_widths():
            return tuple(
                tuple(range(start, min(self.num_experts, start + size)))
                for start in range(0, self.num_experts, size)
            )
        provider = self._physical_weight_provider
        width_getter = getattr(provider, "retained_intermediate_width", None)
        if not callable(width_getter):
            raise RuntimeError(
                "Ragged expert providers must expose retained_intermediate_width()."
            )
        buckets: dict[int, list[int]] = {}
        for expert in range(self.num_experts):
            width = int(width_getter(expert))
            buckets.setdefault(width, []).append(expert)
        groups: list[tuple[int, ...]] = []
        for width in sorted(buckets):
            experts = buckets[width]
            groups.extend(
                tuple(experts[start : start + size])
                for start in range(0, len(experts), size)
            )
        return tuple(groups)

    def set_expert_rotation(
        self,
        key: str,
        rotation: torch.Tensor | None,
        *,
        group_size: int,
    ) -> None:
        """Attach an optional learned groupwise orthogonal INT8 rotation."""

        if key not in self.expert_mlp_spec.tensor_names:
            raise ValueError(f"Unknown grouped expert rotation key {key!r}.")
        buffer_name = f"{key}_rotation"
        if rotation is None:
            self._buffers.pop(buffer_name, None)
            self._expert_rotation_keys.discard(key)
            return
        if self._quant_format != "int8":
            raise ValueError("Learned expert rotations require INT8 storage.")
        validated = _validate_rotation(
            rotation,
            int(group_size),
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).contiguous()
        self._set_buffer(buffer_name, validated)
        self._expert_rotation_keys.add(key)

    def expert_rotation(self, key: str) -> torch.Tensor | None:
        if key not in self._expert_rotation_keys:
            return None
        return getattr(self, f"{key}_rotation")

    def configure_logical_expert_aliases(
        self,
        logical_to_physical: Iterable[int],
        *,
        prototype_logical_ids: Iterable[int] | None = None,
    ) -> None:
        """Attach a validated logical-router to physical-expert map."""
        if self._prototype_calibration_observer is not None:
            raise RuntimeError(
                "Clear the prototype calibration observer before configuring "
                "logical expert aliases."
            )
        values = validate_logical_to_physical(
            tuple(int(value) for value in logical_to_physical),
            physical_experts=self.num_experts,
        )
        if prototype_logical_ids is None:
            prototypes = tuple(
                next(
                    logical_id
                    for logical_id, mapped_id in enumerate(values)
                    if mapped_id == physical_id
                )
                for physical_id in range(self.num_experts)
            )
        else:
            prototypes = tuple(int(value) for value in prototype_logical_ids)
        if len(prototypes) != self.num_experts or len(set(prototypes)) != len(prototypes):
            raise ValueError(
                "prototype_logical_ids must contain one unique id per physical expert."
            )
        if any(value < 0 or value >= len(values) for value in prototypes):
            raise ValueError("prototype_logical_ids contains an out-of-range logical id.")
        if any(
            values[logical_id] != physical_id for physical_id, logical_id in enumerate(prototypes)
        ):
            raise ValueError("Each prototype logical id must map to its corresponding physical id.")
        self.logical_num_experts = len(values)
        self.prototype_logical_ids = prototypes
        self._logical_to_physical = torch.tensor(values, dtype=torch.long)

    def has_logical_expert_aliases(self) -> bool:
        return self._logical_to_physical is not None

    def logical_to_physical_map(self) -> tuple[int, ...]:
        if self._logical_to_physical is None:
            return ()
        return tuple(int(value) for value in self._logical_to_physical.tolist())

    def set_prototype_calibration_observer(
        self,
        observer: PrototypeCalibrationObserver,
    ) -> None:
        """Attach an explicit observer to an unconsolidated reference module."""
        if self.has_logical_expert_aliases():
            raise RuntimeError("Prototype calibration requires an unconsolidated expert module.")
        self._prototype_calibration_observer = observer

    def clear_prototype_calibration_observer(self) -> None:
        self._prototype_calibration_observer = None

    def _record_all_expert_output_prototypes(self, tokens: torch.Tensor) -> None:
        """Evaluate every physical expert on the same tokens for HC-SMoE evidence."""
        observer = self._prototype_calibration_observer
        record = getattr(observer, "record_all_expert_output", None)
        if not callable(record):
            return
        if self.has_logical_expert_aliases():
            raise RuntimeError("All-expert output calibration requires unconsolidated experts.")
        select_inputs = getattr(observer, "shared_calibration_inputs", None)
        calibration_tokens = select_inputs(tokens) if callable(select_inputs) else tokens
        if (
            not torch.is_tensor(calibration_tokens)
            or calibration_tokens.ndim != 2
            or int(calibration_tokens.shape[1]) != int(tokens.shape[1])
            or int(calibration_tokens.shape[0]) < 1
        ):
            raise ValueError(
                "Prototype observer shared inputs must preserve a non-empty "
                "[tokens, hidden] tensor."
            )
        input_role = "gate" if self.expert_mlp_spec.uses_gated_product else "input"
        input_key = self.projection_for_role(input_role)
        down_key = self.projection_for_role("down")
        up_key = (
            self.projection_for_role("up")
            if self.expert_mlp_spec.uses_gated_product
            else None
        )
        for expert_idx in range(self.num_experts):
            input_weight = self._dequantize_expert(
                input_key,
                expert_idx,
                dtype=calibration_tokens.dtype,
                device=calibration_tokens.device,
            )
            down_weight = self._dequantize_expert(
                down_key,
                expert_idx,
                dtype=calibration_tokens.dtype,
                device=calibration_tokens.device,
            )
            primary = F.linear(calibration_tokens, input_weight)
            secondary = None
            if up_key is not None:
                up_weight = self._dequantize_expert(
                    up_key,
                    expert_idx,
                    dtype=calibration_tokens.dtype,
                    device=calibration_tokens.device,
                )
                secondary = F.linear(calibration_tokens, up_weight)
            hidden = self._combine_expert_inputs(primary, secondary)
            expert_output = F.linear(hidden, down_weight)
            record(expert_idx, expert_output)
            del input_weight, down_weight, primary, secondary, hidden, expert_output

    def set_whitening_calibration_observer(self, observer: Any) -> None:
        """Attach a streaming observer for the actual expert-linear inputs."""
        if observer is None or not callable(getattr(observer, "record", None)):
            raise TypeError("Whitening calibration observer must expose record().")
        if self._whitening_calibration_observer is not None:
            raise RuntimeError("A whitening calibration observer is already attached.")
        self._whitening_calibration_observer = observer

    def clear_whitening_calibration_observer(self) -> None:
        self._whitening_calibration_observer = None

    def _record_whitening_inputs(
        self,
        projections: str | tuple[str, ...],
        inputs: torch.Tensor,
    ) -> None:
        observer = self._whitening_calibration_observer
        if observer is not None:
            observer.record(projections, inputs)

    def _coalesce_logical_routes(
        self,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_logical_expert_aliases():
            return top_scores, top_indices
        if self._logical_to_physical is None:  # pragma: no cover - narrowed above
            raise RuntimeError("Logical expert alias state is inconsistent.")
        return coalesce_alias_routes(
            top_scores,
            top_indices,
            self._logical_to_physical,
        )

    def attach_expert_lora(
        self,
        *,
        tensor_name: str,
        adapter_name: str,
        rank: int,
        alpha: float,
        init: str = "kaiming",
        use_rslora: bool = False,
    ) -> nn.Module:
        key = str(tensor_name)
        if key not in self.expert_mlp_spec.tensor_names:
            raise ValueError(f"Unknown grouped expert LoRA tensor '{key}'.")
        if key in self.expert_lora:
            raise ValueError(f"Grouped expert LoRA is already attached to '{key}'.")
        shape = self.expert_weight_shape(key)
        adapter = ActiveExpertLoRA(
            adapter_name=str(adapter_name),
            num_experts=int(shape[0]),
            out_features=int(shape[1]),
            in_features=int(shape[2]),
            rank=int(rank),
            alpha=float(alpha),
            init=str(init),
            use_rslora=bool(use_rslora),
        )
        self.expert_lora[key] = adapter
        return adapter

    def set_routed_adapter_gate(self, gate: Any | None) -> None:
        self._routed_adapter_gate = gate

    def routed_adapter_gate(self) -> Any | None:
        return self._routed_adapter_gate

    def set_routed_adapter_tokens_per_sample(self, value: int) -> None:
        gate = self._routed_adapter_gate
        if gate is not None:
            gate.bind_tokens_per_sample(int(value))

    def bind_packed_source(
        self,
        *,
        source: Any,
        tensor_names: Mapping[str, str],
        group_sizes: Mapping[str, int],
        shapes: Mapping[str, Iterable[int]],
        rotations: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if self.expert_weight_access not in {"active_dequant", "chunked_dequant", "fused_kernel"}:
            raise ValueError(
                "Disk-backed grouped experts require active_dequant or chunked_dequant."
            )
        self._packed_source = source
        self._packed_tensor_names = {str(k): str(v) for k, v in tensor_names.items()}
        self._packed_shapes = {
            str(key): tuple(int(dim) for dim in shape) for key, shape in shapes.items()
        }
        for key in self.expert_mlp_spec.tensor_names:
            shape = self._packed_shapes.get(key)
            if shape is None or len(shape) != 3 or int(shape[0]) != self.num_experts:
                raise ValueError(f"Invalid disk-backed grouped expert shape for '{key}': {shape}.")
            setattr(self, f"{key}_group_size", int(group_sizes[key]))
            self.set_expert_rotation(
                key,
                None if rotations is None else rotations.get(key),
                group_size=int(group_sizes[key]),
            )
            self._loaded_dense_keys.add(key)

    def bind_nf4_packed_source(
        self,
        *,
        source: Any,
        tensor_names: Mapping[str, str],
        shapes: Mapping[str, Iterable[int]],
        meta: _Nf4Meta,
        codebooks: Mapping[str, torch.Tensor],
    ) -> None:
        """Bind expert-indexed NF4 payloads without materializing them in RAM."""
        if self._quant_format != "nf4":
            raise ValueError("NF4 disk-backed state requires an NF4 grouped-expert wrapper.")
        if self.expert_weight_access not in {"active_dequant", "chunked_dequant", "fused_kernel"}:
            raise ValueError(
                "Disk-backed grouped experts require active_dequant or chunked_dequant."
            )
        self._packed_source = source
        self._packed_tensor_names = {str(k): str(v) for k, v in tensor_names.items()}
        self._packed_shapes = {
            str(key): tuple(int(dim) for dim in shape) for key, shape in shapes.items()
        }
        self._nf4_meta = meta
        self._nf4_blocksize = int(meta.blocksize)
        self._nf4_pair_codebook_cache.clear()
        for key in self.expert_mlp_spec.tensor_names:
            shape = self._packed_shapes.get(key)
            if shape is None or len(shape) != 3 or int(shape[0]) != self.num_experts:
                raise ValueError(
                    f"Invalid NF4 disk-backed grouped expert shape for '{key}': {shape}."
                )
            code_name = f"{key}_nf4_code"
            nested_code_name = f"{key}_nf4_ncode"
            if code_name not in codebooks or nested_code_name not in codebooks:
                raise ValueError(f"Missing NF4 codebooks for disk-backed expert '{key}'.")
            self._nf4_shapes[key] = shape
            self._set_buffer(code_name, codebooks[code_name].detach().contiguous())
            self._set_buffer(
                nested_code_name,
                codebooks[nested_code_name].detach().contiguous(),
            )
            self._loaded_dense_keys.add(key)

    def bind_blockwise_fp8_packed_source(
        self,
        *,
        source: Any,
        tensor_names: Mapping[str, str],
        shapes: Mapping[str, Iterable[int]],
        metadata: Mapping[str, BlockwiseFP8Meta],
    ) -> None:
        """Bind expert-indexed FP8 codes and scales without loading the pool."""
        if self._quant_format not in BLOCKWISE_FP8_FORMATS:
            raise ValueError("Blockwise FP8 disk state requires an FP8 wrapper.")
        if self.expert_weight_access not in {
            "active_dequant",
            "chunked_dequant",
            "fused_kernel",
        }:
            raise ValueError(
                "Disk-backed grouped experts require active_dequant or chunked_dequant."
            )
        self._packed_source = source
        self._packed_tensor_names = {str(k): str(v) for k, v in tensor_names.items()}
        self._packed_shapes = {
            str(key): tuple(int(dim) for dim in shape) for key, shape in shapes.items()
        }
        for key in self.expert_mlp_spec.tensor_names:
            shape = self._packed_shapes.get(key)
            meta = metadata.get(key)
            if (
                shape is None
                or len(shape) != 3
                or int(shape[0]) != self.num_experts
                or meta is None
                or meta.shape != shape[1:]
            ):
                raise ValueError(f"Invalid blockwise FP8 disk-backed metadata for '{key}'.")
            self._blockwise_fp8_shapes[key] = shape
            self._blockwise_fp8_meta[key] = meta
            self._loaded_dense_keys.add(key)

    def bind_physical_weight_provider(self, provider: Any) -> None:
        if self._physical_weight_provider is not None:
            raise RuntimeError("A physical expert-weight provider is already attached.")
        if int(getattr(provider, "num_experts", -1)) != self.num_experts:
            raise ValueError("Physical weight provider expert count does not match module.")
        for key in self.expert_mlp_spec.tensor_names:
            shape = tuple(int(value) for value in provider.expert_weight_shape(key))
            if len(shape) != 3 or shape[0] != self.num_experts:
                raise ValueError(
                    f"Physical weight provider returned invalid shape for {key!r}: {shape}."
                )
            self._loaded_dense_keys.add(key)
        if bool(getattr(provider, "ragged_intermediate_widths", False)) and (
            self.expert_weight_access == "full_dequant"
        ):
            raise ValueError(
                "Ragged expert providers require active_dequant, chunked_dequant, "
                "or fused_kernel access."
            )
        self._physical_weight_provider = provider

    def has_ragged_intermediate_widths(self) -> bool:
        return bool(
            self._physical_weight_provider is not None
            and getattr(
                self._physical_weight_provider,
                "ragged_intermediate_widths",
                False,
            )
        )

    def has_packed_weight(self, key: str) -> bool:
        return (
            self._physical_weight_provider is not None
            or hasattr(self, f"{key}_int8")
            or hasattr(self, f"{key}_nf4")
            or hasattr(self, f"{key}_gguf")
            or hasattr(self, f"{key}_mx")
            or hasattr(self, f"{key}_fp8")
            or str(key) in self._packed_shapes
        )

    def expert_weight_shape(self, key: str) -> tuple[int, ...]:
        if self._physical_weight_provider is not None:
            return tuple(
                int(value) for value in self._physical_weight_provider.expert_weight_shape(str(key))
            )
        if str(key) in self._gguf_shapes:
            return self._gguf_shapes[str(key)]
        if str(key) in self._nf4_shapes:
            return self._nf4_shapes[str(key)]
        if str(key) in self._microscaling_shapes:
            return self._microscaling_shapes[str(key)]
        if str(key) in self._blockwise_fp8_shapes:
            return self._blockwise_fp8_shapes[str(key)]
        if hasattr(self, f"{key}_int8"):
            return tuple(int(dim) for dim in getattr(self, f"{key}_int8").shape)
        if str(key) in self._packed_shapes:
            return self._packed_shapes[str(key)]
        raise AttributeError(f"Grouped expert weight '{key}' is not loaded.")

    def set_expert_weight_access_policy(
        self,
        *,
        expert_weight_access: str,
        expert_dequant_chunk_size: int = 0,
    ) -> None:
        access = normalize_expert_weight_access_policy(expert_weight_access)
        if access in {"auto", "disabled"}:
            access = "full_dequant"
        chunk = int(expert_dequant_chunk_size)
        if access in {"chunked_dequant", "fused_kernel"} and chunk <= 0:
            raise ValueError("chunked_dequant/fused_kernel requires expert_dequant_chunk_size > 0.")
        self.expert_weight_access = access
        self.expert_dequant_chunk_size = max(0, chunk)

    def configure_expert_device_cache(self, capacity_gib: float) -> None:
        capacity = float(capacity_gib)
        if capacity < 0.0:
            raise ValueError("expert device cache capacity must be >= 0 GiB.")
        if capacity > 0.0 and self._quant_format != "int8":
            raise ValueError("expert device cache supports only int8 compressed experts.")
        self._expert_device_cache = ExpertDeviceCache(int(capacity * (1024**3)))
        self._expert_device_cache_namespace = ""

    def bind_expert_device_cache(
        self,
        cache: ExpertDeviceCache,
        *,
        namespace: str,
    ) -> None:
        if not isinstance(cache, ExpertDeviceCache):
            raise TypeError("expert device cache must use ExpertDeviceCache.")
        if cache.enabled and self._quant_format != "int8":
            raise ValueError("expert device cache supports only int8 compressed experts.")
        self._expert_device_cache = cache
        self._expert_device_cache_namespace = str(namespace)

    def expert_device_cache_snapshot(self) -> Mapping[str, Any]:
        return self._expert_device_cache.snapshot()

    def prefers_for_loop(self) -> bool:
        return self.flexmoe_channel_mask_active or self._quant_format in BLOCKWISE_FP8_FORMATS or self.expert_weight_access in {
            "active_dequant",
            "chunked_dequant",
            "fused_kernel",
        }

    def grouped_linear(
        self,
        key: str,
        x: torch.Tensor,
        *,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if self.prefers_for_loop():
            raise RuntimeError(
                "Grouped compressed_weights execution requires resident full_dequant expert weights."
            )
        from mirai.core.moe.runtime.gemm import grouped_mm_op

        if grouped_mm_op() is None or x.device.type != "cuda":
            raise RuntimeError(
                "Grouped compressed_weights execution requires CUDA and a "
                "framework grouped_mm operation."
            )
        from .torch_grouped import _FrozenGroupedMMLinear

        output = _FrozenGroupedMMLinear.apply(x, self, str(key), offsets)
        adapter = self.expert_lora[key] if key in self.expert_lora else None
        if adapter is not None:
            output = output + adapter.grouped_forward(x, offsets=offsets)
        return output

    def batched_linear(self, key: str, x: torch.Tensor) -> torch.Tensor:
        if self.prefers_for_loop():
            raise RuntimeError(
                "Batched compressed_weights execution requires resident full_dequant expert weights."
            )
        output = _FrozenDequantBatchedLinear.apply(
            x,
            self,
            str(key),
            tuple(range(self.num_experts)),
        )
        adapter = self.expert_lora[key] if key in self.expert_lora else None
        if adapter is not None:
            output = output + adapter.batched_forward(x)
        return output

    def megablocks_linear(
        self,
        key: str,
        x: torch.Tensor,
        *,
        counts: torch.Tensor,
        grouped_gemm_ops: Any,
    ) -> torch.Tensor:
        if self.prefers_for_loop():
            raise RuntimeError(
                "MegaBlocks grouped execution requires resident full_dequant experts."
            )
        output = _FrozenDequantMegaBlocksLinear.apply(
            x,
            self,
            str(key),
            counts,
            grouped_gemm_ops,
        )
        adapter = self.expert_lora[key] if key in self.expert_lora else None
        if adapter is not None:
            output = output + adapter.megablocks_forward(
                x,
                counts=counts,
                grouped_gemm_ops=grouped_gemm_ops,
            )
        return output

    def run_megablocks_grouped(
        self,
        tokens: torch.Tensor,
        counts: torch.Tensor,
        *,
        grouped_gemm_ops: Any,
    ) -> torch.Tensor:
        if self._whitening_calibration_observer is not None:
            raise RuntimeError(
                "Expert whitening calibration requires the reference grouped "
                "execution path, not MegaBlocks."
            )
        if tokens.device.type != "cuda":
            raise RuntimeError("MegaBlocks grouped experts require CUDA tensors.")
        primary_role = "gate" if self.expert_mlp_spec.uses_gated_product else "input"
        primary = self.megablocks_linear(
            self.projection_for_role(primary_role),
            tokens,
            counts=counts,
            grouped_gemm_ops=grouped_gemm_ops,
        )
        secondary = None
        if self.expert_mlp_spec.uses_gated_product:
            secondary = self.megablocks_linear(
                self.projection_for_role("up"),
                tokens,
                counts=counts,
                grouped_gemm_ops=grouped_gemm_ops,
            )
        return self.megablocks_linear(
            self.projection_for_role("down"),
            self._capture_and_return_sorted_intermediate(
                self._combine_expert_inputs(primary, secondary)
            ),
            counts=counts,
            grouped_gemm_ops=grouped_gemm_ops,
        )

    def _capture_and_return_sorted_intermediate(self, hidden: torch.Tensor) -> torch.Tensor:
        self._capture_sorted_intermediate(hidden)
        return hidden

    def run_batched(self, tokens: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        if self._whitening_calibration_observer is not None:
            return self.run_for_loop(tokens, counts)
        if self.prefers_for_loop():
            return self.run_for_loop(tokens, counts)
        counts_i64 = counts.to(device=tokens.device, dtype=torch.int64)
        max_count = int(counts_i64.max().item()) if counts_i64.numel() else 0
        if max_count <= 0:
            return tokens.new_zeros(tokens.shape)
        starts = torch.cumsum(counts_i64, dim=0) - counts_i64
        local = torch.arange(max_count, device=tokens.device, dtype=torch.int64)
        source = starts[:, None] + local[None, :]
        valid = local[None, :] < counts_i64[:, None]
        padded = tokens.new_zeros((self.num_experts, max_count, tokens.shape[-1]))
        padded[valid] = tokens[source[valid]]
        primary_role = "gate" if self.expert_mlp_spec.uses_gated_product else "input"
        primary = self.batched_linear(self.projection_for_role(primary_role), padded)
        secondary = None
        if self.expert_mlp_spec.uses_gated_product:
            secondary = self.batched_linear(self.projection_for_role("up"), padded)
        hidden = self._combine_expert_inputs(primary, secondary)
        self._capture_sorted_intermediate(hidden[valid])
        output = self.batched_linear(self.projection_for_role("down"), hidden)
        return output[valid]

    def load_dense_weight(
        self,
        key: str,
        source: torch.Tensor,
        *,
        rotation: torch.Tensor | None = None,
    ) -> None:
        if key not in self.expert_mlp_spec.tensor_names:
            raise ValueError(f"Unknown compressed_weights grouped expert tensor '{key}'.")
        if source.ndim != 3:
            raise ValueError(
                f"compressed_weights grouped expert tensor {key} must be 3D, got {tuple(source.shape)}."
            )
        if self._quant_format == "nf4":
            if rotation is not None:
                raise ValueError("Learned expert rotations require INT8 storage.")
            self._store_nf4_grouped(key, source)
            self._loaded_dense_keys.add(key)
            self._packed_source = None
            return
        if self._quant_format in GGUF_FORMATS:
            if rotation is not None:
                raise ValueError("Learned expert rotations require INT8 storage.")
            self._store_gguf_grouped(key, source)
            self._loaded_dense_keys.add(key)
            self._packed_source = None
            return
        if self._quant_format in MICROSCALING_FORMATS:
            if rotation is not None:
                raise ValueError("Learned expert rotations require INT8 storage.")
            self._store_microscaling_grouped(key, source)
            self._loaded_dense_keys.add(key)
            self._packed_source = None
            return
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            if rotation is not None:
                raise ValueError("Learned expert rotations require INT8 storage.")
            self._store_blockwise_fp8_grouped(key, source)
            self._loaded_dense_keys.add(key)
            self._packed_source = None
            return
        group_size = best_group_size(int(source.shape[-1]), self._group_sizes)
        quantized, scale, resolved_group = _quantize_weight(
            source,
            group_size=group_size,
            rotation=rotation,
        )
        self._set_buffer(f"{key}_int8", quantized)
        self._set_buffer(f"{key}_scale", scale.float())
        setattr(self, f"{key}_group_size", int(resolved_group))
        self.set_expert_rotation(
            key,
            rotation,
            group_size=int(resolved_group),
        )
        self._loaded_dense_keys.add(key)
        self._packed_source = None

    def _store_nf4_grouped(self, key: str, source: torch.Tensor) -> None:
        num_experts, out_features, in_features = (int(dim) for dim in source.shape)
        if num_experts != int(self.num_experts):
            raise ValueError(
                f"compressed_weights grouped expert tensor {key} has {num_experts} experts, "
                f"expected {self.num_experts}."
            )
        packed_stack: list[torch.Tensor] = []
        absmax_stack: list[torch.Tensor] = []
        nabsmax_stack: list[torch.Tensor] = []
        offset_stack: list[torch.Tensor] = []
        shared_codes: dict[str, torch.Tensor] | None = None
        for expert_idx in range(num_experts):
            fields, codes, meta = _nf4_quantize_2d(
                source[expert_idx], blocksize=self._nf4_blocksize
            )
            if shared_codes is None:
                shared_codes = codes
                self._nf4_meta = meta
            packed_stack.append(fields["packed"].to(device=source.device))
            absmax_stack.append(fields["absmax"].to(device=source.device))
            nabsmax_stack.append(fields["nested_absmax"].to(device=source.device))
            offset_stack.append(fields["offset"].to(device=source.device))
        self._nf4_shapes[key] = (num_experts, out_features, in_features)
        self._set_buffer(f"{key}_nf4", torch.stack(packed_stack, dim=0).contiguous())
        self._set_buffer(f"{key}_nf4_absmax", torch.stack(absmax_stack, dim=0).contiguous())
        self._set_buffer(f"{key}_nf4_nabsmax", torch.stack(nabsmax_stack, dim=0).contiguous())
        self._set_buffer(f"{key}_nf4_offset", torch.stack(offset_stack, dim=0).contiguous())
        self._set_buffer(
            f"{key}_nf4_code", shared_codes["code"].to(device=source.device).contiguous()
        )
        self._set_buffer(
            f"{key}_nf4_ncode", shared_codes["nested_code"].to(device=source.device).contiguous()
        )

    def _nf4_dequantize_expert(
        self,
        key: str,
        expert_idx: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        index = int(expert_idx)
        packed, absmax, nested_absmax, offset = self._nf4_gather_buffers(
            key, (index,), torch.device(device)
        )
        fields = {
            "packed": packed[0],
            "absmax": absmax[0],
            "nested_absmax": nested_absmax[0],
            "offset": offset[0],
        }
        codes = {
            "code": getattr(self, f"{key}_nf4_code"),
            "nested_code": getattr(self, f"{key}_nf4_ncode"),
        }
        _, out_features, in_features = self._nf4_shapes[key]
        return _nf4_dequantize(
            fields,
            codes,
            self._nf4_meta,
            shape=(int(out_features), int(in_features)),
            dtype=dtype,
            device=device,
        )

    def _store_gguf_grouped(self, key: str, source: torch.Tensor) -> None:
        """Quantize a dense [E, out, in] tensor to stacked GGUF packed blocks."""
        num_experts, out_features, in_features = (int(dim) for dim in source.shape)
        if num_experts != int(self.num_experts):
            raise ValueError(
                f"gguf grouped expert tensor {key} has {num_experts} experts, "
                f"expected {self.num_experts}."
            )
        meta = gguf_meta_for(self._quant_format)
        if self._gguf_meta is not None and self._gguf_meta != meta:
            raise ValueError("gguf grouped expert tensors use inconsistent metadata.")
        self._gguf_meta = meta
        blocks = torch.stack(
            [quantize_gguf(self._quant_format, source[e]) for e in range(num_experts)],
            dim=0,
        ).contiguous()
        self._gguf_shapes[key] = (num_experts, out_features, in_features)
        self._set_buffer(f"{key}_gguf", blocks)

    def _store_microscaling_grouped(self, key: str, source: torch.Tensor) -> None:
        num_experts, out_features, in_features = (int(dim) for dim in source.shape)
        if num_experts != self.num_experts:
            raise ValueError(
                f"microscaling grouped expert tensor {key} has {num_experts} experts, "
                f"expected {self.num_experts}."
            )
        packed_stack = []
        scale_stack = []
        global_stack = []
        meta: MicroscalingMeta | None = None
        for expert_idx in range(num_experts):
            packed, scales, global_scale, current_meta = quantize_microscaling(
                self._quant_format, source[expert_idx]
            )
            if meta is not None and current_meta != meta:
                raise ValueError("microscaling experts use inconsistent metadata.")
            meta = current_meta
            packed_stack.append(packed)
            scale_stack.append(scales)
            global_stack.append(global_scale)
        if meta is None:
            raise ValueError("microscaling grouped expert tensor cannot be empty.")
        self._microscaling_meta[key] = meta
        self._microscaling_shapes[key] = (num_experts, out_features, in_features)
        self._set_buffer(f"{key}_mx", torch.stack(packed_stack).contiguous())
        self._set_buffer(f"{key}_mx_scale", torch.stack(scale_stack).contiguous())
        self._set_buffer(f"{key}_mx_global", torch.stack(global_stack).contiguous())

    def _store_blockwise_fp8_grouped(self, key: str, source: torch.Tensor) -> None:
        num_experts, out_features, in_features = (int(dim) for dim in source.shape)
        if num_experts != self.num_experts:
            raise ValueError(
                f"blockwise FP8 tensor {key} has {num_experts} experts, "
                f"expected {self.num_experts}."
            )
        code_stack: list[torch.Tensor] = []
        scale_stack: list[torch.Tensor] = []
        meta: BlockwiseFP8Meta | None = None
        for expert_idx in range(num_experts):
            codes, scales, current_meta = quantize_blockwise_fp8_weight(source[expert_idx])
            if meta is not None and current_meta != meta:
                raise ValueError("blockwise FP8 experts use inconsistent metadata.")
            meta = current_meta
            code_stack.append(codes)
            scale_stack.append(scales)
        if meta is None:
            raise ValueError("blockwise FP8 grouped expert tensor cannot be empty.")
        self._blockwise_fp8_meta[key] = meta
        self._blockwise_fp8_shapes[key] = (
            num_experts,
            out_features,
            in_features,
        )
        self._set_buffer(f"{key}_fp8", torch.stack(code_stack).contiguous())
        self._set_buffer(f"{key}_fp8_scale", torch.stack(scale_stack).float().contiguous())

    def _blockwise_fp8_payload(
        self,
        key: str,
        expert_indices: tuple[int, ...],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, BlockwiseFP8Meta]:
        if self._packed_source is not None and not hasattr(self, f"{key}_fp8"):
            get_slices = getattr(self._packed_source, "get_slices_to_device", None)
            if callable(get_slices):
                codes = get_slices(
                    self._packed_tensor_names[f"{key}_fp8"],
                    expert_indices,
                    device=device,
                )
                scales = get_slices(
                    self._packed_tensor_names[f"{key}_fp8_scale"],
                    expert_indices,
                    device=device,
                )
            else:
                codes = torch.stack(
                    [
                        self._packed_source.get_slice(
                            self._packed_tensor_names[f"{key}_fp8"], index
                        )
                        for index in expert_indices
                    ]
                ).to(device=device)
                scales = torch.stack(
                    [
                        self._packed_source.get_slice(
                            self._packed_tensor_names[f"{key}_fp8_scale"], index
                        )
                        for index in expert_indices
                    ]
                ).to(device=device)
        else:
            index = torch.as_tensor(
                expert_indices,
                dtype=torch.long,
                device=getattr(self, f"{key}_fp8").device,
            )
            codes = getattr(self, f"{key}_fp8").index_select(0, index).to(device)
            scales = (
                getattr(self, f"{key}_fp8_scale")
                .index_select(0, index)
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )
        return codes, scales, self._blockwise_fp8_meta[key]

    def _blockwise_fp8_linear(
        self,
        key: str,
        x: torch.Tensor,
        expert_indices: tuple[int, ...],
    ) -> torch.Tensor:
        codes, scales, meta = self._blockwise_fp8_payload(
            key,
            expert_indices,
            device=x.device,
        )
        if resolve_moe_gemm_backend("forward", device=x.device) == "deepgemm_fp8":
            from ..quantization.deepgemm_fp8 import (
                deepgemm_blockwise_fp8_batched_linear,
            )

            return deepgemm_blockwise_fp8_batched_linear(x, codes, scales, meta)
        if len(expert_indices) == 1 and x.ndim == 2:
            return blockwise_fp8_linear(x, codes[0], scales[0], meta)
        return blockwise_fp8_batched_linear(x, codes, scales, meta)

    def _microscaling_dequantize_expert(
        self,
        key: str,
        expert_idx: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        index = int(expert_idx)
        from mirai.core.moe.runtime.specs import get_active_moe_optimization_policy

        decoder = dequantize_microscaling
        if get_active_moe_optimization_policy().kernel_backend == "compiled_packed":
            from ..quantization.compiled_dequant import (
                compiled_dequantize_microscaling,
            )

            decoder = compiled_dequantize_microscaling
        return decoder(
            getattr(self, f"{key}_mx")[index],
            getattr(self, f"{key}_mx_scale")[index],
            getattr(self, f"{key}_mx_global")[index],
            self._microscaling_meta[key],
            dtype=dtype,
            device=device,
        )

    def _gguf_dequantize_expert(
        self,
        key: str,
        expert_idx: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        blocks = getattr(self, f"{key}_gguf")[int(expert_idx)]
        _, out_features, in_features = self._gguf_shapes[key]
        from mirai.core.moe.runtime.specs import get_active_moe_optimization_policy

        decoder = dequantize_gguf
        if get_active_moe_optimization_policy().kernel_backend == "compiled_packed":
            from ..quantization.compiled_dequant import compiled_dequantize_gguf

            decoder = compiled_dequantize_gguf
        return decoder(
            self._quant_format,
            blocks,
            shape=(int(out_features), int(in_features)),
            dtype=dtype,
            device=device,
        )

    def load_gguf_packed_weight(
        self,
        key: str,
        *,
        blocks: torch.Tensor,
        shape: Iterable[int],
        meta: _GgufMeta,
    ) -> None:
        """Restore one grouped-expert GGUF tensor from packed uint8 blocks."""
        if key not in self.expert_mlp_spec.tensor_names:
            raise ValueError(f"Unknown compressed_weights grouped expert tensor '{key}'.")
        if self._quant_format not in GGUF_FORMATS:
            raise ValueError("GGUF packed state requires a gguf grouped-expert wrapper.")
        resolved_shape = tuple(int(dim) for dim in shape)
        if len(resolved_shape) != 3 or resolved_shape[0] != self.num_experts:
            raise ValueError(
                f"gguf grouped expert tensor {key} has invalid dense shape {resolved_shape}."
            )
        if self._gguf_meta is not None and self._gguf_meta != meta:
            raise ValueError("gguf grouped expert tensors use inconsistent metadata.")
        self._gguf_meta = meta
        self._gguf_shapes[key] = resolved_shape
        self._set_buffer(f"{key}_gguf", blocks.detach().contiguous())
        self._loaded_dense_keys.add(key)
        self._packed_source = None

    def load_microscaling_packed_weight(
        self,
        key: str,
        *,
        packed: torch.Tensor,
        scales: torch.Tensor,
        global_scales: torch.Tensor,
        shape: Iterable[int],
        meta: MicroscalingMeta,
    ) -> None:
        if key not in self.expert_mlp_spec.tensor_names:
            raise ValueError(f"Unknown compressed_weights grouped expert tensor '{key}'.")
        if self._quant_format not in MICROSCALING_FORMATS:
            raise ValueError(
                "Microscaling packed state requires a registered microscaling expert wrapper."
            )
        resolved_shape = tuple(int(dim) for dim in shape)
        if len(resolved_shape) != 3 or resolved_shape[0] != self.num_experts:
            raise ValueError(
                f"microscaling grouped expert tensor {key} has invalid shape {resolved_shape}."
            )
        if meta.format != self._quant_format or meta.shape != resolved_shape[1:]:
            raise ValueError("Microscaling grouped expert metadata is inconsistent.")
        if packed.ndim != 3 or int(packed.shape[0]) != self.num_experts:
            raise ValueError("Microscaling packed expert payload has invalid expert axis.")
        if scales.ndim != 2 or int(scales.shape[0]) != self.num_experts:
            raise ValueError("Microscaling scale payload has invalid expert axis.")
        if tuple(global_scales.shape) != (self.num_experts,):
            raise ValueError("Microscaling global scales have invalid expert axis.")
        validate_microscaling_payload(
            packed,
            scales,
            global_scales,
            meta,
        )
        self._microscaling_meta[key] = meta
        self._microscaling_shapes[key] = resolved_shape
        self._set_buffer(f"{key}_mx", packed.detach().contiguous())
        self._set_buffer(f"{key}_mx_scale", scales.detach().contiguous())
        self._set_buffer(f"{key}_mx_global", global_scales.detach().float().contiguous())
        self._loaded_dense_keys.add(key)
        self._packed_source = None

    def load_blockwise_fp8_packed_weight(
        self,
        key: str,
        *,
        codes: torch.Tensor,
        scales: torch.Tensor,
        shape: Iterable[int],
        meta: BlockwiseFP8Meta,
    ) -> None:
        if key not in self.expert_mlp_spec.tensor_names:
            raise ValueError(f"Unknown compressed_weights grouped expert tensor '{key}'.")
        if self._quant_format not in BLOCKWISE_FP8_FORMATS:
            raise ValueError("Blockwise FP8 state requires an FP8 expert wrapper.")
        resolved_shape = tuple(int(dim) for dim in shape)
        if len(resolved_shape) != 3 or resolved_shape[0] != self.num_experts:
            raise ValueError(
                f"blockwise FP8 grouped expert tensor {key} has invalid shape {resolved_shape}."
            )
        if meta.shape != resolved_shape[1:]:
            raise ValueError("Blockwise FP8 grouped expert metadata is inconsistent.")
        validate_blockwise_fp8_payload(codes, scales, meta)
        self._blockwise_fp8_meta[key] = meta
        self._blockwise_fp8_shapes[key] = resolved_shape
        self._set_buffer(f"{key}_fp8", codes.detach().contiguous())
        self._set_buffer(f"{key}_fp8_scale", scales.detach().float().contiguous())
        self._loaded_dense_keys.add(key)
        self._packed_source = None

    def frozen_quantized_numel(self) -> int:
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            return int(sum(math.prod(shape) for shape in self._blockwise_fp8_shapes.values()))
        if self._quant_format in MICROSCALING_FORMATS:
            return int(sum(math.prod(shape) for shape in self._microscaling_shapes.values()))
        if self._quant_format in GGUF_FORMATS:
            return int(sum(math.prod(shape) for shape in self._gguf_shapes.values()))
        if self._quant_format == "nf4":
            return int(sum(math.prod(shape) for shape in self._nf4_shapes.values()))
        return int(
            sum(
                getattr(self, f"{key}_int8").numel()
                for key in self.expert_mlp_spec.tensor_names
                if hasattr(self, f"{key}_int8")
            )
        )

    def load_quantized_weight(
        self,
        key: str,
        *,
        weight_int8: torch.Tensor,
        weight_scale: torch.Tensor,
        group_size: int,
        rotation: torch.Tensor | None = None,
    ) -> None:
        if key not in self.expert_mlp_spec.tensor_names:
            raise ValueError(f"Unknown compressed_weights grouped expert tensor '{key}'.")
        if weight_int8.ndim != 3:
            raise ValueError(
                f"compressed_weights grouped expert tensor {key} must be 3D, got {tuple(weight_int8.shape)}."
            )
        if int(weight_int8.shape[0]) != int(self.num_experts):
            raise ValueError(
                f"compressed_weights grouped expert tensor {key} has {weight_int8.shape[0]} experts, "
                f"expected {self.num_experts}."
            )
        expected_scale_shape = (*tuple(weight_int8.shape[:-1]), 1)
        if tuple(weight_scale.shape) != expected_scale_shape:
            raise ValueError(
                f"compressed_weights grouped expert scale {key} has shape {tuple(weight_scale.shape)}, "
                f"expected {expected_scale_shape}."
            )
        self._set_buffer(
            f"{key}_int8",
            weight_int8.detach().to(torch.int8).clone(memory_format=torch.contiguous_format),
        )
        self._set_buffer(
            f"{key}_scale",
            weight_scale.detach().float().clone(memory_format=torch.contiguous_format),
        )
        setattr(self, f"{key}_group_size", int(group_size))
        self.set_expert_rotation(
            key,
            rotation,
            group_size=int(group_size),
        )
        self._loaded_dense_keys.add(key)

    def load_nf4_packed_weight(
        self,
        key: str,
        *,
        packed: torch.Tensor,
        absmax: torch.Tensor,
        nested_absmax: torch.Tensor,
        offset: torch.Tensor,
        code: torch.Tensor,
        nested_code: torch.Tensor,
        shape: Iterable[int],
        meta: _Nf4Meta,
    ) -> None:
        """Restore one grouped-expert NF4 tensor from packed buffers."""
        if key not in self.expert_mlp_spec.tensor_names:
            raise ValueError(f"Unknown compressed_weights grouped expert tensor '{key}'.")
        if self._quant_format != "nf4":
            raise ValueError("NF4 packed state requires an NF4 grouped-expert wrapper.")
        resolved_shape = tuple(int(dim) for dim in shape)
        if len(resolved_shape) != 3 or resolved_shape[0] != self.num_experts:
            raise ValueError(
                f"NF4 grouped expert tensor {key} has invalid dense shape {resolved_shape}."
            )
        for name, tensor in {
            "packed": packed,
            "absmax": absmax,
            "nested_absmax": nested_absmax,
            "offset": offset,
        }.items():
            if tensor.ndim == 0 or int(tensor.shape[0]) != self.num_experts:
                raise ValueError(
                    f"NF4 grouped expert {key} {name} must have expert axis "
                    f"{self.num_experts}, got {tuple(tensor.shape)}."
                )
        if self._nf4_meta is not None and self._nf4_meta != meta:
            raise ValueError("NF4 grouped expert tensors use inconsistent quantization metadata.")
        self._nf4_meta = meta
        self._nf4_blocksize = int(meta.blocksize)
        self._nf4_shapes[key] = resolved_shape
        self._set_buffer(f"{key}_nf4", packed.detach().contiguous())
        self._set_buffer(f"{key}_nf4_absmax", absmax.detach().contiguous())
        self._set_buffer(f"{key}_nf4_nabsmax", nested_absmax.detach().contiguous())
        self._set_buffer(f"{key}_nf4_offset", offset.detach().contiguous())
        self._set_buffer(f"{key}_nf4_code", code.detach().contiguous())
        self._set_buffer(f"{key}_nf4_ncode", nested_code.detach().contiguous())
        self._loaded_dense_keys.add(key)
        self._packed_source = None

    def _set_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    def is_fully_loaded(self) -> bool:
        return self._loaded_dense_keys.issuperset(self.expert_mlp_spec.tensor_names)

    @property
    def w1(self) -> torch.Tensor:
        return self._dequantize("w1")

    @property
    def w2(self) -> torch.Tensor:
        return self._dequantize("w2")

    @property
    def w3(self) -> torch.Tensor:
        return self._dequantize("w3")

    def _dequantize(
        self,
        key: str,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if self._physical_weight_provider is not None:
            if self.has_ragged_intermediate_widths():
                raise RuntimeError(
                    "Full dequantization is undefined for ragged expert widths; "
                    "materialize experts individually."
                )
            resolved_device = device if device is not None else torch.device("cpu")
            resolved_dtype = dtype if dtype is not None else torch.get_default_dtype()
            return torch.stack(
                [
                    self._physical_weight_provider.materialize_expert(
                        key,
                        expert_idx,
                        dtype=resolved_dtype,
                        device=resolved_device,
                    )
                    for expert_idx in range(self.num_experts)
                ],
                dim=0,
            )
        if self._quant_format in MICROSCALING_FORMATS:
            num_experts = int(self._microscaling_shapes[key][0])
            resolved_device = device or getattr(self, f"{key}_mx").device
            resolved_dtype = dtype or torch.get_default_dtype()
            return torch.stack(
                [
                    self._microscaling_dequantize_expert(
                        key, expert_idx, dtype=resolved_dtype, device=resolved_device
                    )
                    for expert_idx in range(num_experts)
                ],
                dim=0,
            )
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            resolved_device = device or (
                getattr(self, f"{key}_fp8").device
                if hasattr(self, f"{key}_fp8")
                else torch.device("cpu")
            )
            resolved_dtype = dtype or torch.get_default_dtype()
            codes, scales, meta = self._blockwise_fp8_payload(
                key,
                tuple(range(self.num_experts)),
                device=resolved_device,
            )
            return dequantize_blockwise_fp8_weight(
                codes,
                scales,
                meta,
                dtype=resolved_dtype,
                device=resolved_device,
            )
        if self._packed_source is not None and not hasattr(self, f"{key}_int8"):
            raise RuntimeError(
                "Full expert dequantization is unavailable for disk-backed packed experts."
            )
        if self._quant_format in GGUF_FORMATS:
            num_experts = int(self._gguf_shapes[key][0])
            resolved_device = device if device is not None else getattr(self, f"{key}_gguf").device
            resolved_dtype = dtype if dtype is not None else torch.get_default_dtype()
            return torch.stack(
                [
                    self._gguf_dequantize_expert(
                        key, expert_idx, dtype=resolved_dtype, device=resolved_device
                    )
                    for expert_idx in range(num_experts)
                ],
                dim=0,
            )
        if self._quant_format == "nf4":
            num_experts = int(self._nf4_shapes[key][0])
            resolved_device = (
                device
                if device is not None
                else (
                    getattr(self, f"{key}_nf4").device
                    if hasattr(self, f"{key}_nf4")
                    else torch.device("cpu")
                )
            )
            resolved_dtype = dtype if dtype is not None else torch.get_default_dtype()
            experts = [
                self._nf4_dequantize_expert(
                    key, expert_idx, dtype=resolved_dtype, device=resolved_device
                )
                for expert_idx in range(num_experts)
            ]
            return torch.stack(experts, dim=0)
        quantized = getattr(self, f"{key}_int8")
        scale = getattr(self, f"{key}_scale")
        resolved_dtype = (
            dtype
            if dtype is not None
            else scale.dtype
            if scale.is_floating_point()
            else torch.get_default_dtype()
        )
        return _dequantize_weight(
            quantized,
            scale,
            group_size=int(getattr(self, f"{key}_group_size")),
            rotation=self.expert_rotation(key),
            dtype=resolved_dtype,
            device=device if device is not None else quantized.device,
        )

    def _dequantize_expert(
        self, key: str, expert_idx: int, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if self._physical_weight_provider is not None:
            return self._physical_weight_provider.materialize_expert(
                key,
                int(expert_idx),
                dtype=dtype,
                device=device,
            )
        if self._quant_format in GGUF_FORMATS:
            return self._gguf_dequantize_expert(key, int(expert_idx), dtype=dtype, device=device)
        if self._quant_format in MICROSCALING_FORMATS:
            return self._microscaling_dequantize_expert(
                key, int(expert_idx), dtype=dtype, device=device
            )
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            codes, scales, meta = self._blockwise_fp8_payload(
                key,
                (int(expert_idx),),
                device=device,
            )
            return dequantize_blockwise_fp8_weight(
                codes[0],
                scales[0],
                meta,
                dtype=dtype,
                device=device,
            )
        if self._quant_format == "nf4":
            return self._nf4_dequantize_expert(key, int(expert_idx), dtype=dtype, device=device)
        if self._packed_source is not None and not hasattr(self, f"{key}_int8"):
            quantized = self._packed_source.get_slice(
                self._packed_tensor_names[f"{key}_int8"], int(expert_idx)
            )
            scale = self._packed_source.get_slice(
                self._packed_tensor_names[f"{key}_scale"], int(expert_idx)
            )
        else:
            quantized = getattr(self, f"{key}_int8")[int(expert_idx)]
            scale = getattr(self, f"{key}_scale")[int(expert_idx)]
        return _dequantize_weight(
            quantized,
            scale,
            group_size=int(getattr(self, f"{key}_group_size")),
            rotation=self.expert_rotation(key),
            dtype=dtype,
            device=device,
        )

    def run_for_loop(self, tokens: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        if self._prototype_calibration_observer is not None:
            raise RuntimeError(
                "Prototype calibration requires direct routed execution with route weights."
            )
        if not self.is_fully_loaded():
            missing = sorted(set(self.expert_mlp_spec.tensor_names) - self._loaded_dense_keys)
            raise RuntimeError(
                f"compressed_weights grouped experts are missing weights: {missing}."
            )
        count_list = [int(v) for v in counts.detach().cpu().tolist()]
        splits = torch.split(tokens, count_list, dim=0)
        outputs: list[torch.Tensor] = []
        primary_role = "gate" if self.expert_mlp_spec.uses_gated_product else "input"
        primary_key = self.projection_for_role(primary_role)
        up_key = (
            self.projection_for_role("up")
            if self.expert_mlp_spec.uses_gated_product
            else None
        )
        down_key = self.projection_for_role("down")
        input_keys = (primary_key, up_key) if up_key is not None else primary_key
        if self.expert_weight_access == "full_dequant":
            weights = {
                key: self._dequantize(key).to(device=tokens.device, dtype=tokens.dtype)
                for key in self.expert_mlp_spec.tensor_names
            }
            for expert_idx, expert_tokens in enumerate(splits):
                if expert_tokens.numel() == 0:
                    continue
                self._record_whitening_inputs(input_keys, expert_tokens)
                primary = self._expert_linear(
                    primary_key, expert_tokens, weights[primary_key][expert_idx], expert_idx
                )
                secondary = (
                    self._expert_linear(
                        up_key, expert_tokens, weights[up_key][expert_idx], expert_idx
                    )
                    if up_key is not None
                    else None
                )
                h = self._combine_expert_inputs(primary, secondary)
                h = self._apply_flexmoe_channel_mask(h, expert_idx)
                self._capture_sorted_intermediate(h)
                self._record_whitening_inputs(down_key, h)
                outputs.append(
                    self._expert_linear(
                        down_key, h, weights[down_key][expert_idx], expert_idx
                    )
                )
            if not outputs:
                return tokens.new_zeros(tokens.shape)
            return torch.cat(outputs, dim=0)

        chunk_size = (
            int(self.expert_dequant_chunk_size)
            if self.expert_weight_access in {"chunked_dequant", "fused_kernel"}
            else 1
        )
        if self.has_ragged_intermediate_widths():
            chunk_size = 1
        chunk_size = max(1, chunk_size)
        for chunk_start in range(0, len(splits), chunk_size):
            for offset, expert_tokens in enumerate(splits[chunk_start : chunk_start + chunk_size]):
                if expert_tokens.numel() == 0:
                    continue
                expert_idx = chunk_start + offset
                self._record_whitening_inputs(input_keys, expert_tokens)
                weights = {
                    key: self._dequantize_expert(
                        key, expert_idx, dtype=tokens.dtype, device=tokens.device
                    )
                    for key in self.expert_mlp_spec.tensor_names
                }
                primary = self._expert_linear(
                    primary_key, expert_tokens, weights[primary_key], expert_idx
                )
                secondary = (
                    self._expert_linear(up_key, expert_tokens, weights[up_key], expert_idx)
                    if up_key is not None
                    else None
                )
                h = self._combine_expert_inputs(primary, secondary)
                h = self._apply_flexmoe_channel_mask(h, expert_idx)
                self._capture_sorted_intermediate(h)
                self._record_whitening_inputs(down_key, h)
                outputs.append(
                    self._expert_linear(down_key, h, weights[down_key], expert_idx)
                )
        if not outputs:
            return tokens.new_zeros(tokens.shape)
        return torch.cat(outputs, dim=0)

    def run_direct_routed(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self.prefers_for_loop():
            raise RuntimeError(
                "Direct routed compressed_weights execution requires active_dequant, "
                "chunked_dequant, or fused_kernel expert access."
            )
        if top_scores.shape != top_indices.shape or top_indices.ndim != 2:
            raise ValueError(
                "Direct routed compressed_weights execution requires matching "
                "[tokens, top_k] score and index tensors."
            )
        if int(top_indices.shape[0]) != int(tokens.shape[0]):
            raise ValueError("Direct routed expert indices do not match token count.")
        top_scores, top_indices = self._coalesce_logical_routes(top_scores, top_indices)
        self._record_all_expert_output_prototypes(tokens)
        observer = self.active_routed_output_observer()
        intermediate_observer = self.active_routed_intermediate_observer()
        top_k = int(top_indices.shape[1])
        if observer is not None:
            observer.begin_routes(
                num_tokens=int(tokens.shape[0]),
                top_k=top_k,
                device=tokens.device,
            )
            bind_routes = getattr(observer, "bind_routes", None)
            if callable(bind_routes):
                bind_routes(top_indices, top_scores)
        if intermediate_observer is not None:
            intermediate_observer.begin_routes(
                num_tokens=int(tokens.shape[0]),
                top_k=top_k,
                device=tokens.device,
            )
        output = torch.zeros(
            (tokens.shape[0], tokens.shape[1]),
            dtype=torch.float32,
            device=tokens.device,
        )
        primary_role = "gate" if self.expert_mlp_spec.uses_gated_product else "input"
        primary_key = self.projection_for_role(primary_role)
        up_key = (
            self.projection_for_role("up")
            if self.expert_mlp_spec.uses_gated_product
            else None
        )
        down_key = self.projection_for_role("down")
        input_keys = (primary_key, up_key) if up_key is not None else primary_key
        chunk_size = (
            int(self.expert_dequant_chunk_size)
            if self.expert_weight_access in {"chunked_dequant", "fused_kernel"}
            else 1
        )
        if self.flexmoe_channel_mask_active:
            chunk_size = 1
        if self._flexmoe_taylor_observer is not None:
            chunk_size = 1
        if chunk_size > 1 and self._whitening_calibration_observer is None:
            if self._prototype_calibration_observer is not None:
                raise RuntimeError(
                    "Prototype calibration does not support batched expert dispatch."
                )
            try:
                result = self._run_direct_routed_batched(
                    tokens,
                    top_scores,
                    top_indices,
                    chunk_size=chunk_size,
                    output=output,
                )
            except BaseException:
                if observer is not None:
                    observer.abort_capture()
                if intermediate_observer is not None:
                    intermediate_observer.abort_capture()
                raise
            if observer is not None:
                observer.end_routes()
            if intermediate_observer is not None:
                intermediate_observer.end_routes()
            return result
        try:
            for expert_idx in range(self.num_experts):
                assignment_mask = top_indices == int(expert_idx)
                if self.has_logical_expert_aliases():
                    assignment_mask = assignment_mask & (top_scores != 0)
                assignments = torch.nonzero(assignment_mask, as_tuple=False)
                if assignments.numel() == 0:
                    continue
                token_indices = assignments[:, 0]
                route_slots = assignments[:, 1]
                expert_tokens = tokens.index_select(0, token_indices)
                self._record_whitening_inputs(input_keys, expert_tokens)
                projection_weights = {
                    key: self._dequantize_expert(
                        key, expert_idx, dtype=tokens.dtype, device=tokens.device
                    )
                    for key in self.expert_mlp_spec.tensor_names
                }
                primary = self._expert_linear(
                    primary_key,
                    expert_tokens,
                    projection_weights[primary_key],
                    expert_idx,
                )
                secondary = (
                    self._expert_linear(
                        up_key,
                        expert_tokens,
                        projection_weights[up_key],
                        expert_idx,
                    )
                    if up_key is not None
                    else None
                )
                hidden = self._combine_expert_inputs(primary, secondary)
                hidden = self._apply_flexmoe_channel_mask(hidden, expert_idx)
                self._capture_routed_intermediates(hidden, token_indices * top_k + route_slots)
                self._record_whitening_inputs(down_key, hidden)
                expert_output = self._expert_linear(
                    down_key,
                    hidden,
                    projection_weights[down_key],
                    expert_idx,
                )
                if self._flexmoe_taylor_observer is not None:
                    self._flexmoe_taylor_observer.capture(
                        expert_index=expert_idx,
                        inputs=expert_tokens,
                        w1=projection_weights[primary_key],
                        w2=projection_weights[down_key],
                        w3=projection_weights[up_key],
                        gate=primary,
                        up=secondary,
                        hidden=hidden,
                        output=expert_output,
                    )
                weights = top_scores[token_indices, route_slots].float().unsqueeze(1)
                if self._prototype_calibration_observer is not None:
                    self._prototype_calibration_observer.record(
                        expert_idx,
                        weights.squeeze(1),
                        expert_output,
                    )
                self._capture_routed_outputs(expert_output, token_indices * top_k + route_slots)
                output.index_add_(0, token_indices, expert_output.float() * weights)
            return output.to(dtype=tokens.dtype)
        finally:
            if observer is not None:
                observer.end_routes()
            if intermediate_observer is not None:
                intermediate_observer.end_routes()

    def run_direct_routed_rotated_int8(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Direct routing that rotates activations instead of un-rotating weights.

        Opt-in via memory.moe_kernel_backend='rotated_int8'. Consumes int8 weights
        through a cuBLAS bmm with the Hadamard rotation moved to activations and
        the scale applied as an epilogue; the expensive per-weight rotation
        matmul is deleted. Not bit-identical to the dequant default (reduction
        order and rounding placement differ), so it must stay explicitly
        selected and never be reached through 'auto'.
        """
        if self._whitening_calibration_observer is not None:
            raise RuntimeError("Expert whitening calibration requires unrotated reference inputs.")
        if self.expert_weight_access not in {"chunked_dequant", "fused_kernel"}:
            raise RuntimeError(
                "packed INT8 direct routing requires chunked_dequant or fused_kernel expert access."
            )
        if int(self.expert_dequant_chunk_size) <= 1:
            raise RuntimeError(
                "rotated_int8 direct routing requires expert_dequant_chunk_size > 1."
            )
        top_scores, top_indices = self._coalesce_logical_routes(top_scores, top_indices)
        output = torch.zeros(
            (tokens.shape[0], tokens.shape[1]),
            dtype=torch.float32,
            device=tokens.device,
        )
        return self._run_direct_routed_batched(
            tokens,
            top_scores,
            top_indices,
            chunk_size=int(self.expert_dequant_chunk_size),
            output=output,
            rotated=True,
        )

    def run_expert_choice_routed(
        self,
        tokens: torch.Tensor,
        expert_token_scores: torch.Tensor,
        expert_token_indices: torch.Tensor,
        *,
        tokens_per_sample: int,
    ) -> torch.Tensor:
        """Execute per-sample Expert-Choice routes through compressed experts.

        The route conversion and every promoted runner remain device-resident;
        no token-choice padding or route-dependent host list is introduced.
        """
        from .expert_choice import run_compressed_expert_choice

        return run_compressed_expert_choice(
            self,
            tokens,
            expert_token_scores,
            expert_token_indices,
            tokens_per_sample=tokens_per_sample,
        )

    def _run_direct_routed_batched(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        *,
        chunk_size: int,
        output: torch.Tensor,
        rotated: bool = False,
    ) -> torch.Tensor:
        if self._prototype_calibration_observer is not None:
            raise RuntimeError("Prototype calibration does not support batched expert dispatch.")
        # The gemm-backend registry sits BELOW moe_dispatch: "auto" (the default)
        # does not intercept, so the moe_dispatch selection below runs
        # unchanged. An explicit forward backend picks the primitive
        # directly and fails fast (in resolve) when it is unavailable on this
        # device. dX/dW roles are registered/validated but inherit the runner the
        # forward role selects (independent per-role wiring is a documented
        # backend contract in moe/runtime/gemm.py).
        forward_backend = resolve_moe_gemm_backend("forward", device=tokens.device)
        if forward_backend == "bmm":
            runner = self._run_direct_routed_batched_vectorized
        elif forward_backend == "persistent":
            runner = self._run_direct_routed_batched_persistent
        elif forward_backend == "torch_grouped":
            runner = self._run_direct_routed_batched_torch_grouped
        else:  # auto or a format-owned primitive -> dispatch selects token layout
            mode = resolve_moe_dispatch_mode()
            if mode == "legacy":
                runner = self._run_direct_routed_batched_legacy
            elif mode == "triton" and _triton_dispatch_available(tokens.device):
                runner = self._run_direct_routed_batched_triton
            elif mode == "triton_persistent" and _persistent_dispatch_available(tokens.device):
                runner = self._run_direct_routed_batched_persistent
            else:
                runner = self._run_direct_routed_batched_vectorized
        if self.has_logical_expert_aliases():
            if resolve_moe_dispatch_preprocess() in {"device", "sonic"}:
                raise RuntimeError(
                    "Prototype-consolidated experts do not support "
                    "device-resident dispatch preprocessing; use 'host'."
                )
            supported = {
                self._run_direct_routed_batched_vectorized,
                self._run_direct_routed_batched_legacy,
            }
            if runner not in supported:
                raise RuntimeError(
                    "Prototype-consolidated experts support only the "
                    "host vectorized or legacy dispatch backend."
                )
        if rotated and runner not in {
            self._run_direct_routed_batched_vectorized,
            self._run_direct_routed_batched_legacy,
        }:
            # The triton/persistent/torch_grouped runners ignore per-chunk
            # weight-format flags and always dequantize, so a silent fallback
            # Reject unsupported runners instead of bypassing the configured
            # rotated representation.
            raise RuntimeError(
                "rotated_int8 supports only the vectorized or legacy "
                "dispatch runner; set memory.moe_dispatch='vectorized'."
            )
        if self._routed_adapter_gate is not None and runner not in {
            self._run_direct_routed_batched_persistent,
            self._run_direct_routed_batched_torch_grouped,
        }:
            raise RuntimeError(
                "Mixed-domain compressed expert LoRA requires the persistent "
                "or torch_grouped sorted dispatch backend."
            )
        if self.has_ragged_intermediate_widths() and runner != (
            self._run_direct_routed_batched_vectorized
        ):
            raise RuntimeError(
                "Ragged FlexMoE experts require the vectorized BMM dispatch backend."
            )
        return runner(
            tokens,
            top_scores,
            top_indices,
            chunk_size=chunk_size,
            output=output,
            rotated=rotated,
        )

    def _run_expert_chunk(
        self,
        padded: torch.Tensor,
        active: list[int],
        expert_ids: torch.Tensor,
        *,
        device: torch.device,
        rotated: bool = False,
    ) -> torch.Tensor:
        """Shared per-chunk expert MLP: identical math for both dispatch modes."""
        if self._whitening_calibration_observer is not None:
            raise RuntimeError(
                "Expert whitening calibration must use the per-expert reference path."
            )
        prefetch = getattr(self._packed_source, "prefetch_expert_tensors", None)
        if callable(prefetch):
            prefetch(
                self._packed_tensor_names,
                quant_format=self._quant_format,
                keys=self.expert_mlp_spec.tensor_names,
                expert_indices=active,
                device=device,
            )
        if rotated:
            if self.expert_mlp_spec != CANONICAL_PACKED_EXPERT_MLP_SPEC:
                raise RuntimeError(
                    "rotated_int8 currently supports only the canonical "
                    "gated-product expert layout."
                )
            from ..quantization.rotated_int8 import rotate_activations, rotated_int8_linear

            # w1/w3 share one activation rotation; fp32/tf32 bmm retains parity.
            g1 = int(getattr(self, "w1_group_size"))
            r1 = self.expert_rotation("w1")
            r3 = self.expert_rotation("w3")
            if (r1 is None) != (r3 is None) or (r1 is not None and not torch.equal(r1, r3)):
                raise RuntimeError("Packed INT8 w1/w3 execution requires a shared expert rotation.")
            padded_rot = rotate_activations(
                padded,
                g1,
                rotation=r1,
            )
            q1, s1 = self._quantized_expert_batch("w1", active, device=device)
            q3, s3 = self._quantized_expert_batch("w3", active, device=device)
            gate = rotated_int8_linear(padded_rot, q1, s1, g1, x_already_rotated=True)
            up = rotated_int8_linear(padded_rot, q3, s3, g1, x_already_rotated=True)
            del q1, s1, q3, s3, padded_rot
            adapter_w1 = self.expert_lora["w1"] if "w1" in self.expert_lora else None
            if adapter_w1 is not None:
                gate = gate + adapter_w1.batched_subset_forward(padded, expert_indices=expert_ids)
            adapter_w3 = self.expert_lora["w3"] if "w3" in self.expert_lora else None
            if adapter_w3 is not None:
                up = up + adapter_w3.batched_subset_forward(padded, expert_indices=expert_ids)
        else:
            primary_role = "gate" if self.expert_mlp_spec.uses_gated_product else "input"
            primary_key = self.projection_for_role(primary_role)
            up_key = (
                self.projection_for_role("up")
                if self.expert_mlp_spec.uses_gated_product
                else None
            )
            if up_key is None:
                gate = self._expert_linear_batched_requant(
                    primary_key, padded, active, expert_ids
                )
                up = None
            else:
                gate, up = self._expert_linear_batched_requant_pair(
                    primary_key, up_key, padded, active, expert_ids
                )
        hidden = self._combine_expert_inputs(gate, up)
        del gate, up
        self._capture_chunk_intermediate(hidden)
        if rotated:
            from ..quantization.rotated_int8 import rotated_int8_linear

            # w2's input is silu(gate)*up, produced inside this chunk, so its
            # rotation cannot be hoisted above the loop like w1/w3. Cheaper
            # regardless: K here is moe_intermediate (768), not hidden (2048).
            g2 = int(getattr(self, "w2_group_size"))
            q2, s2 = self._quantized_expert_batch("w2", active, device=device)
            expert_output = rotated_int8_linear(
                hidden,
                q2,
                s2,
                g2,
                rotation=self.expert_rotation("w2"),
            )
            del q2, s2
            adapter_w2 = self.expert_lora["w2"] if "w2" in self.expert_lora else None
            if adapter_w2 is not None:
                expert_output = expert_output + adapter_w2.batched_subset_forward(
                    hidden, expert_indices=expert_ids
                )
        else:
            expert_output = self._expert_linear_batched_requant(
                self.projection_for_role("down"), hidden, active, expert_ids
            )
        return expert_output

    def _run_direct_routed_device_chunks(
        self,
        plan,
        tokens: torch.Tensor,
        *,
        chunk_size: int,
        output: torch.Tensor,
        chunk_fn,
    ) -> torch.Tensor:
        """DispatchPlan-driven padded gather/scatter (device or Sonic metadata).

        FULLY DEVICE-RESIDENT: no route-dependent Python active-expert assembly,
        no full-[E] counts ``.tolist()``, and no ``plan.counts.max().item()``.
        Every expert in every chunk is processed, zero-count ones as fully-masked
        pad rows (they add nothing to the reduction).

        The padded width is ``plan.max_count_bound``, a route-shape/checkpoint
        metadata bound computed without device readback. For token-choice it is
        ``num_tokens``; for Expert-Choice with logical aliases it includes the
        maximum static alias multiplicity. Rows ``>= count[e]`` are masked to
        zero and never index_add'd, so widening from exact ``counts.max()`` leaves
        every valid contribution unchanged.

        The result is NUMERICALLY EQUIVALENT to the host path, not bit-identical:
        including masked empty-expert batches and a uniform padded width changes
        the batched-GEMM batch/M dimensions and thus its tiling/rounding (an fp32
        rounding-order difference, same precedent as the sorted-contiguous
        persistent path which is checked within tolerance). This is why the device
        path stays opt-in through ``moe_dispatch_preprocess`` and the host
        default remains bit-identical.
        ``chunk_fn(padded, active, expert_ids, counts_active)`` runs the per-chunk
        expert MLP (bmm or triton grouped GEMM).
        """
        total_slots = plan.total_slots
        # Route-independent pad width from the tensor shape (no D2H sync).
        max_count = int(plan.max_count_bound)
        if max_count == 0:
            return output.to(dtype=tokens.dtype)
        row = torch.arange(max_count, device=tokens.device)
        for static_group in self._expert_chunk_groups(chunk_size):
            active = list(static_group)
            if not active:
                continue
            expert_ids = torch.tensor(active, device=tokens.device, dtype=torch.long)
            starts_active = plan.starts.index_select(0, expert_ids)
            counts_active = plan.counts.index_select(0, expert_ids)
            valid = row.unsqueeze(0) < counts_active.unsqueeze(1)
            slot = (starts_active.unsqueeze(1) + row.unsqueeze(0)).clamp_(
                max=max(total_slots - 1, 0)
            )
            flat_slot = slot.reshape(-1)
            token_matrix = plan.sorted_token_indices.index_select(0, flat_slot).reshape(
                len(active), max_count
            )
            score_matrix = plan.sorted_scores.index_select(0, flat_slot).reshape(
                len(active), max_count
            )
            route_matrix = plan.sort_positions.index_select(0, flat_slot).reshape(
                len(active), max_count
            )
            padded = tokens.index_select(0, token_matrix.reshape(-1)).reshape(
                len(active), max_count, tokens.shape[-1]
            )
            padded = padded * valid.unsqueeze(-1).to(padded.dtype)
            expert_output = self._run_with_routed_intermediate_capture(
                chunk_fn,
                padded,
                active,
                expert_ids,
                counts_active,
                capture_mask=valid,
                route_positions=route_matrix,
            )
            self._capture_routed_outputs(expert_output[valid], route_matrix[valid])
            output.index_add_(
                0,
                token_matrix[valid],
                expert_output[valid].float() * score_matrix[valid].float().unsqueeze(1),
            )
        return output.to(dtype=tokens.dtype)

    def _run_direct_routed_batched_vectorized(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        *,
        chunk_size: int,
        output: torch.Tensor,
        rotated: bool = False,
    ) -> torch.Tensor:
        preprocess = resolve_moe_dispatch_preprocess()
        if preprocess in {"device", "sonic"}:
            plan = build_dispatch_plan(
                top_indices,
                top_scores,
                self.num_experts,
                backend=preprocess,
            )
            return self._run_direct_routed_device_chunks(
                plan,
                tokens,
                chunk_size=chunk_size,
                output=output,
                chunk_fn=lambda padded, active, expert_ids, counts_active: self._run_expert_chunk(
                    padded,
                    active,
                    expert_ids,
                    device=tokens.device,
                    rotated=rotated,
                ),
            )
        top_k = int(top_indices.shape[1])
        flat_scores = top_scores.reshape(-1)
        if self.has_logical_expert_aliases():
            route_positions = torch.nonzero(flat_scores != 0, as_tuple=False).reshape(-1)
            flat_experts = top_indices.reshape(-1).index_select(0, route_positions)
            sorted_positions = torch.argsort(flat_experts, stable=True)
            sorted_route_positions = route_positions.index_select(0, sorted_positions)
            sorted_token_indices = torch.div(sorted_route_positions, top_k, rounding_mode="floor")
            sorted_scores = flat_scores.index_select(0, sorted_route_positions)
        else:
            flat_experts = top_indices.reshape(-1)
            sorted_positions = torch.argsort(flat_experts, stable=True)
            sorted_route_positions = sorted_positions
            sorted_token_indices = torch.div(sorted_positions, top_k, rounding_mode="floor")
            sorted_scores = flat_scores.index_select(0, sorted_positions)
        total_slots = int(flat_experts.numel())
        counts_gpu = torch.bincount(flat_experts, minlength=self.num_experts)
        # Exclusive-prefix segment starts stay on-GPU so per-chunk gather indices
        # never round-trip through the host. A single D2H copy of the counts
        # vector (below) is the only host sync per layer, used purely for Python
        # control flow (which experts are active, padded batch shape).
        starts_gpu = torch.cumsum(counts_gpu, dim=0) - counts_gpu
        counts = counts_gpu.detach().cpu().tolist()

        arange_cache: dict[int, torch.Tensor] = {}
        for static_group in self._expert_chunk_groups(chunk_size):
            active = [
                expert_idx
                for expert_idx in static_group
                if int(counts[expert_idx]) > 0
            ]
            if not active:
                continue
            max_count = max(int(counts[expert_idx]) for expert_idx in active)
            # Vectorized dispatch: build the [E, max_count] gather layout for the
            # whole chunk with a handful of tensor ops instead of a per-expert
            # index_select / F.pad / stack loop. Invalid (padding) slots gather a
            # clamped real slot and are then zeroed, so `padded` matches the
            # reference F.pad(zeros) layout while valid rows are unchanged.
            expert_ids = torch.tensor(active, device=tokens.device, dtype=torch.long)
            starts_active = starts_gpu.index_select(0, expert_ids)
            counts_active = counts_gpu.index_select(0, expert_ids)
            row = arange_cache.get(max_count)
            if row is None:
                row = torch.arange(max_count, device=tokens.device)
                arange_cache[max_count] = row
            valid = row.unsqueeze(0) < counts_active.unsqueeze(1)
            slot = (starts_active.unsqueeze(1) + row.unsqueeze(0)).clamp_(
                max=max(total_slots - 1, 0)
            )
            flat_slot = slot.reshape(-1)
            token_matrix = sorted_token_indices.index_select(0, flat_slot).reshape(
                len(active), max_count
            )
            score_matrix = sorted_scores.index_select(0, flat_slot).reshape(len(active), max_count)
            route_matrix = sorted_route_positions.index_select(0, flat_slot).reshape(
                len(active), max_count
            )
            padded = tokens.index_select(0, token_matrix.reshape(-1)).reshape(
                len(active), max_count, tokens.shape[-1]
            )
            padded = padded * valid.unsqueeze(-1).to(padded.dtype)
            expert_output = self._run_with_routed_intermediate_capture(
                self._run_expert_chunk,
                padded,
                active,
                expert_ids,
                device=tokens.device,
                rotated=rotated,
                capture_mask=valid,
                route_positions=route_matrix,
            )
            self._capture_routed_outputs(expert_output[valid], route_matrix[valid])
            output.index_add_(
                0,
                token_matrix[valid],
                expert_output[valid].float() * score_matrix[valid].float().unsqueeze(1),
            )
        return output.to(dtype=tokens.dtype)

    def _run_expert_chunk_triton(
        self,
        padded: torch.Tensor,
        active: list[int],
        expert_ids: torch.Tensor,
        counts: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-chunk expert MLP using the count-aware triton grouped GEMM.

        Mirrors ``_run_expert_chunk`` (non-fused branch) but the frozen w1/w3/w2
        linears run through the triton kernel. Quantized weights are dequantized
        to bf16 first (dequant unchanged), so this path is format-agnostic
        (nf4 / int8-module / int8-packed). LoRA stays on its existing padded
        ``batched_subset_forward`` path.
        """
        primary_role = "gate" if self.expert_mlp_spec.uses_gated_product else "input"
        primary_key = self.projection_for_role(primary_role)
        up_key = (
            self.projection_for_role("up")
            if self.expert_mlp_spec.uses_gated_product
            else None
        )
        if up_key is None:
            gate = self._expert_linear_batched_triton(
                primary_key, padded, active, expert_ids, counts
            )
            up = None
        else:
            gate, up = self._expert_linear_batched_triton_pair(
                primary_key, up_key, padded, active, expert_ids, counts
            )
        hidden = self._combine_expert_inputs(gate, up)
        del gate, up
        self._capture_chunk_intermediate(hidden)
        expert_output = self._expert_linear_batched_triton(
            self.projection_for_role("down"), hidden, active, expert_ids, counts
        )
        return expert_output

    def _run_direct_routed_batched_triton(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        *,
        chunk_size: int,
        output: torch.Tensor,
        rotated: bool = False,
    ) -> torch.Tensor:
        # Identical dispatch/scatter to the vectorized path; only the per-chunk
        # frozen linears differ (triton grouped GEMM vs bmm).
        preprocess = resolve_moe_dispatch_preprocess()
        if preprocess in {"device", "sonic"}:
            plan = build_dispatch_plan(
                top_indices,
                top_scores,
                self.num_experts,
                backend=preprocess,
            )
            return self._run_direct_routed_device_chunks(
                plan,
                tokens,
                chunk_size=chunk_size,
                output=output,
                chunk_fn=lambda padded, active, expert_ids, counts_active: (
                    self._run_expert_chunk_triton(
                        padded,
                        active,
                        expert_ids,
                        counts_active,
                        device=tokens.device,
                    )
                ),
            )
        top_k = int(top_indices.shape[1])
        flat_experts = top_indices.reshape(-1)
        total_slots = int(flat_experts.numel())
        sorted_positions = torch.argsort(flat_experts, stable=True)
        sorted_token_indices = torch.div(sorted_positions, top_k, rounding_mode="floor")
        sorted_scores = top_scores.reshape(-1).index_select(0, sorted_positions)
        counts_gpu = torch.bincount(flat_experts, minlength=self.num_experts)
        starts_gpu = torch.cumsum(counts_gpu, dim=0) - counts_gpu
        counts = counts_gpu.detach().cpu().tolist()

        arange_cache: dict[int, torch.Tensor] = {}
        for chunk_start in range(0, self.num_experts, int(chunk_size)):
            active = [
                expert_idx
                for expert_idx in range(
                    chunk_start, min(self.num_experts, chunk_start + int(chunk_size))
                )
                if int(counts[expert_idx]) > 0
            ]
            if not active:
                continue
            max_count = max(int(counts[expert_idx]) for expert_idx in active)
            expert_ids = torch.tensor(active, device=tokens.device, dtype=torch.long)
            starts_active = starts_gpu.index_select(0, expert_ids)
            counts_active = counts_gpu.index_select(0, expert_ids)
            row = arange_cache.get(max_count)
            if row is None:
                row = torch.arange(max_count, device=tokens.device)
                arange_cache[max_count] = row
            valid = row.unsqueeze(0) < counts_active.unsqueeze(1)
            slot = (starts_active.unsqueeze(1) + row.unsqueeze(0)).clamp_(
                max=max(total_slots - 1, 0)
            )
            flat_slot = slot.reshape(-1)
            token_matrix = sorted_token_indices.index_select(0, flat_slot).reshape(
                len(active), max_count
            )
            score_matrix = sorted_scores.index_select(0, flat_slot).reshape(len(active), max_count)
            route_matrix = sorted_positions.index_select(0, flat_slot).reshape(
                len(active), max_count
            )
            padded = tokens.index_select(0, token_matrix.reshape(-1)).reshape(
                len(active), max_count, tokens.shape[-1]
            )
            padded = padded * valid.unsqueeze(-1).to(padded.dtype)
            expert_output = self._run_with_routed_intermediate_capture(
                self._run_expert_chunk_triton,
                padded,
                active,
                expert_ids,
                counts_active,
                device=tokens.device,
                capture_mask=valid,
                route_positions=route_matrix,
            )
            self._capture_routed_outputs(expert_output[valid], route_matrix[valid])
            output.index_add_(
                0,
                token_matrix[valid],
                expert_output[valid].float() * score_matrix[valid].float().unsqueeze(1),
            )
        return output.to(dtype=tokens.dtype)

    def _run_direct_routed_batched_persistent(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        *,
        chunk_size: int,
        output: torch.Tensor,
        rotated: bool = False,
    ) -> torch.Tensor:
        from .persistent import run_persistent_dispatch

        return run_persistent_dispatch(
            self,
            tokens,
            top_scores,
            top_indices,
            chunk_size=int(chunk_size),
            output=output,
            routed_adapter_gate=self._routed_adapter_gate,
        )

    def _run_direct_routed_batched_torch_grouped(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        *,
        chunk_size: int,
        output: torch.Tensor,
        rotated: bool = False,
    ) -> torch.Tensor:
        # moe_gemm_backend="torch_grouped": public torch grouped_mm on the shared
        # sorted-contiguous plan.
        from .torch_grouped import run_torch_grouped_dispatch

        return run_torch_grouped_dispatch(
            self,
            tokens,
            top_scores,
            top_indices,
            chunk_size=int(chunk_size),
            output=output,
            routed_adapter_gate=self._routed_adapter_gate,
        )

    def _run_direct_routed_batched_legacy(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        *,
        chunk_size: int,
        output: torch.Tensor,
        rotated: bool = False,
    ) -> torch.Tensor:
        preprocess = resolve_moe_dispatch_preprocess()
        if preprocess in {"device", "sonic"}:
            plan = build_dispatch_plan(
                top_indices,
                top_scores,
                self.num_experts,
                backend=preprocess,
            )
            return self._run_direct_routed_device_chunks(
                plan,
                tokens,
                chunk_size=chunk_size,
                output=output,
                chunk_fn=lambda padded, active, expert_ids, counts_active: self._run_expert_chunk(
                    padded,
                    active,
                    expert_ids,
                    device=tokens.device,
                    rotated=rotated,
                ),
            )
        top_k = int(top_indices.shape[1])
        flat_scores = top_scores.reshape(-1)
        if self.has_logical_expert_aliases():
            route_positions = torch.nonzero(flat_scores != 0, as_tuple=False).reshape(-1)
            flat_experts = top_indices.reshape(-1).index_select(0, route_positions)
            sorted_positions = torch.argsort(flat_experts, stable=True)
            sorted_route_positions = route_positions.index_select(0, sorted_positions)
            sorted_token_indices = torch.div(sorted_route_positions, top_k, rounding_mode="floor")
            sorted_scores = flat_scores.index_select(0, sorted_route_positions)
        else:
            flat_experts = top_indices.reshape(-1)
            sorted_positions = torch.argsort(flat_experts, stable=True)
            sorted_route_positions = sorted_positions
            sorted_token_indices = torch.div(sorted_positions, top_k, rounding_mode="floor")
            sorted_scores = flat_scores.index_select(0, sorted_positions)
        counts = torch.bincount(flat_experts, minlength=self.num_experts).detach().cpu().tolist()
        starts = [0]
        for count in counts:
            starts.append(starts[-1] + int(count))

        for chunk_start in range(0, self.num_experts, int(chunk_size)):
            active = [
                expert_idx
                for expert_idx in range(
                    chunk_start, min(self.num_experts, chunk_start + int(chunk_size))
                )
                if int(counts[expert_idx]) > 0
            ]
            if not active:
                continue
            max_count = max(int(counts[expert_idx]) for expert_idx in active)
            token_rows = []
            index_rows = []
            score_rows = []
            route_rows = []
            valid_rows = []
            for expert_idx in active:
                start = starts[expert_idx]
                end = starts[expert_idx + 1]
                token_indices = sorted_token_indices[start:end]
                count = int(end - start)
                pad = int(max_count - count)
                token_rows.append(F.pad(tokens.index_select(0, token_indices), (0, 0, 0, pad)))
                index_rows.append(F.pad(token_indices, (0, pad)))
                score_rows.append(F.pad(sorted_scores[start:end], (0, pad)))
                route_rows.append(F.pad(sorted_route_positions[start:end], (0, pad)))
                valid_rows.append(torch.arange(max_count, device=tokens.device) < count)
            padded = torch.stack(token_rows, dim=0)
            expert_ids = torch.tensor(active, device=tokens.device, dtype=torch.long)
            valid = torch.stack(valid_rows, dim=0)
            route_matrix = torch.stack(route_rows, dim=0)
            expert_output = self._run_with_routed_intermediate_capture(
                self._run_expert_chunk,
                padded,
                active,
                expert_ids,
                device=tokens.device,
                rotated=rotated,
                capture_mask=valid,
                route_positions=route_matrix,
            )
            token_matrix = torch.stack(index_rows, dim=0)
            score_matrix = torch.stack(score_rows, dim=0)
            self._capture_routed_outputs(expert_output[valid], route_matrix[valid])
            output.index_add_(
                0,
                token_matrix[valid],
                expert_output[valid].float() * score_matrix[valid].float().unsqueeze(1),
            )
        return output.to(dtype=tokens.dtype)

    def _quantized_expert_batch(
        self,
        key: str,
        expert_indices: list[int],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # memory.moe_batched_gather opts into contiguous batched gather. The
        # per-expert path transfers pinned RAM-cache views independently.
        if self._expert_device_cache.enabled:
            return self._quantized_expert_batch_per_expert(key, expert_indices, device=device)
        if resolve_moe_batched_gather():
            batched = self._quantized_expert_batch_gather(key, expert_indices, device=device)
            if batched is not None:
                return batched
        return self._quantized_expert_batch_per_expert(key, expert_indices, device=device)

    def _quantized_expert_batch_per_expert(
        self,
        key: str,
        expert_indices: list[int],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not self._expert_device_cache.enabled
            and self._packed_source is not None
            and not hasattr(self, f"{key}_int8")
        ):
            stream_to_device = getattr(self._packed_source, "get_slices_to_device", None)
            if callable(stream_to_device):
                quantized = stream_to_device(
                    self._packed_tensor_names[f"{key}_int8"],
                    expert_indices,
                    device=device,
                )
                scale = stream_to_device(
                    self._packed_tensor_names[f"{key}_scale"],
                    expert_indices,
                    device=device,
                    dtype=torch.float32,
                )
                return quantized, scale
        quantized_parts = []
        scale_parts = []
        for expert_idx in expert_indices:
            cache_key = self._expert_device_cache.key(
                f"{self._expert_device_cache_namespace}:{key}",
                expert_idx,
                device,
            )
            cached = self._expert_device_cache.get(cache_key)
            if cached is not None:
                quantized_parts.append(cached[0])
                scale_parts.append(cached[1])
                continue
            if self._packed_source is not None and not hasattr(self, f"{key}_int8"):
                quantized = self._packed_source.get_slice(
                    self._packed_tensor_names[f"{key}_int8"], expert_idx
                ).to(device=device)
                scale = self._packed_source.get_slice(
                    self._packed_tensor_names[f"{key}_scale"], expert_idx
                ).to(device=device, dtype=torch.float32)
            else:
                quantized = getattr(self, f"{key}_int8")[expert_idx].to(device=device)
                scale = getattr(self, f"{key}_scale")[expert_idx].to(
                    device=device, dtype=torch.float32
                )
            self._expert_device_cache.put(cache_key, (quantized, scale))
            quantized_parts.append(quantized)
            scale_parts.append(scale)
        return torch.stack(quantized_parts, dim=0), torch.stack(scale_parts, dim=0)

    def _quantized_expert_batch_gather(
        self,
        key: str,
        expert_indices: list[int],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return BatchedExpertGatherStrategy(enabled=True).gather(
            key=key,
            expert_indices=expert_indices,
            device=device,
            packed_source=self._packed_source,
            packed_tensor_names=self._packed_tensor_names,
            weight=getattr(self, f"{key}_int8", None),
            scale=getattr(self, f"{key}_scale", None),
        )

    def _expert_linear(
        self,
        key: str,
        x: torch.Tensor,
        weight: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        output = (
            self._blockwise_fp8_linear(key, x, (int(expert_idx),))
            if self._quant_format in BLOCKWISE_FP8_FORMATS
            else _FrozenDequantExpertLinear.apply(
                x,
                weight,
                self,
                key,
                int(expert_idx),
            )
        )
        adapter = self.expert_lora[key] if key in self.expert_lora else None
        if adapter is not None:
            if self.has_ragged_intermediate_widths():
                raise RuntimeError(
                    "FlexMoE recovery adapters must be merged before ragged artifact export."
                )
            output = output + adapter(x, expert_idx=int(expert_idx))
        return output

    def _expert_linear_batched(
        self,
        key: str,
        x: torch.Tensor,
        weight: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        output = (
            self._blockwise_fp8_linear(
                key,
                x,
                tuple(int(v) for v in expert_indices.detach().cpu().tolist()),
            )
            if self._quant_format in BLOCKWISE_FP8_FORMATS
            else torch.bmm(x, weight.transpose(-2, -1))
        )
        adapter = self.expert_lora[key] if key in self.expert_lora else None
        if adapter is not None:
            output = output + adapter.batched_subset_forward(x, expert_indices=expert_indices)
        return output

    def _dequant_expert_stack(
        self,
        key: str,
        expert_indices: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self._physical_weight_provider is not None:
            return torch.stack(
                [
                    self._physical_weight_provider.materialize_expert(
                        key,
                        expert_idx,
                        dtype=dtype,
                        device=device,
                    )
                    for expert_idx in expert_indices
                ],
                dim=0,
            )
        # memory.moe_batched_dequant (default ON) is
        # the explicit selector for chunk-batched dequant versus per-expert execution.
        if len(expert_indices) > 1 and resolve_moe_batched_dequant():
            if self._quant_format == "nf4":
                batched = self._nf4_dequantize_expert_stack_batched(
                    key, expert_indices, dtype=dtype, device=device
                )
                if batched is not None:
                    return batched
            elif self._quant_format in GGUF_FORMATS:
                # Each GGUF expert dequantizes from its own packed super-blocks;
                # use the per-expert stack.
                pass
            elif self._quant_format in BLOCKWISE_FP8_FORMATS:
                pass
            else:
                quantized, scale = self._quantized_expert_batch(
                    key, list(expert_indices), device=device
                )
                return _dequantize_weight(
                    quantized,
                    scale,
                    group_size=int(getattr(self, f"{key}_group_size")),
                    rotation=self.expert_rotation(key),
                    dtype=dtype,
                    device=device,
                )
        return torch.stack(
            [
                self._dequantize_expert(key, idx, dtype=dtype, device=device)
                for idx in expert_indices
            ],
            dim=0,
        )

    def _nf4_batched_preconditions_ok(self, key: str, meta: "_Nf4Meta") -> bool:
        """Shape divisibility guards for the concatenated blockwise dequant."""
        if key not in self._nf4_shapes:
            return False
        _, out_features, in_features = self._nf4_shapes[key]
        return nf4_stack_dequant_supported(
            elements=int(out_features) * int(in_features), meta=meta
        )

    def _nf4_gather_buffers(
        self, key: str, expert_indices: tuple[int, ...], target: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._packed_source is not None and not hasattr(self, f"{key}_nf4"):
            get_slices_to_device = getattr(self._packed_source, "get_slices_to_device", None)
            if callable(get_slices_to_device):

                def load_batch(suffix: str) -> torch.Tensor:
                    return get_slices_to_device(
                        self._packed_tensor_names[f"{key}_{suffix}"],
                        expert_indices,
                        device=target,
                    )

                return (
                    load_batch("nf4"),
                    load_batch("nf4_absmax"),
                    load_batch("nf4_nabsmax"),
                    load_batch("nf4_offset"),
                )
            get_slice = getattr(self._packed_source, "get_slice", None)
            if not callable(get_slice):
                raise RuntimeError("Disk-backed NF4 source does not expose expert slice reads.")

            def load_slices(suffix: str) -> torch.Tensor:
                return torch.stack(
                    [
                        get_slice(
                            self._packed_tensor_names[f"{key}_{suffix}"],
                            expert_idx,
                        )
                        for expert_idx in expert_indices
                    ],
                    dim=0,
                ).to(device=target)

            return (
                load_slices("nf4"),
                load_slices("nf4_absmax"),
                load_slices("nf4_nabsmax"),
                load_slices("nf4_offset"),
            )
        index = torch.as_tensor(
            expert_indices,
            device=getattr(self, f"{key}_nf4").device,
            dtype=torch.long,
        )
        packed = getattr(self, f"{key}_nf4").index_select(0, index).to(device=target)
        absmax_q = getattr(self, f"{key}_nf4_absmax").index_select(0, index).to(device=target)
        nested_absmax = getattr(self, f"{key}_nf4_nabsmax").index_select(0, index).to(device=target)
        offsets = getattr(self, f"{key}_nf4_offset").index_select(0, index).to(device=target)
        return packed, absmax_q, nested_absmax, offsets

    def _nf4_dequantize_core(
        self,
        *,
        packed: torch.Tensor,
        absmax_q: torch.Tensor,
        nested_absmax: torch.Tensor,
        offsets: torch.Tensor,
        code: torch.Tensor,
        nested_code: torch.Tensor,
        num_selected: int,
        out_features: int,
        in_features: int,
        dtype: torch.dtype,
        meta: "_Nf4Meta",
    ) -> torch.Tensor | None:
        """Concatenated per-expert NF4 dequant, owned by the quantization seam."""
        return nf4_dequantize_stack(
            packed=packed,
            absmax_q=absmax_q,
            nested_absmax=nested_absmax,
            offsets=offsets,
            code=code,
            nested_code=nested_code,
            num_selected=int(num_selected),
            out_features=int(out_features),
            in_features=int(in_features),
            dtype=dtype,
            meta=meta,
        )

    def _nf4_dequantize_expert_stack_batched(
        self,
        key: str,
        expert_indices: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Dequantize a chunk of NF4 experts with two bnb calls instead of one
        pair per expert.

        bitsandbytes computes ``absmax = dequantize_blockwise(absmax_q, state2)
        + offset`` and then ``weight = dequantize_4bit(packed, absmax)``; both
        stages are blockwise with block sizes that divide each expert's segment,
        so concatenating the per-expert buffers along dim 0 yields bit-identical
        results while cutting kernel launches and QuantState rebuild overhead by
        the chunk size. Metadata is re-read from module buffers on every call so
        the path survives block swap, and any API mismatch falls back to the
        per-expert loop (returns None).
        """
        meta = self._nf4_meta
        if meta is None or key not in self._nf4_shapes:
            return None
        target = torch.device(device)
        if target.type != "cuda":
            return None
        if not self._nf4_batched_preconditions_ok(key, meta):
            return None
        num_selected = len(expert_indices)
        _, out_features, in_features = self._nf4_shapes[key]
        packed, absmax_q, nested_absmax, offsets = self._nf4_gather_buffers(
            key, expert_indices, target
        )
        code = getattr(self, f"{key}_nf4_code").to(device=target)
        nested_code = getattr(self, f"{key}_nf4_ncode").to(device=target)
        return self._nf4_dequantize_core(
            packed=packed,
            absmax_q=absmax_q,
            nested_absmax=nested_absmax,
            offsets=offsets,
            code=code,
            nested_code=nested_code,
            num_selected=num_selected,
            out_features=int(out_features),
            in_features=int(in_features),
            dtype=dtype,
            meta=meta,
        )

    def _nf4_pair_codebooks_match(self, key_a: str, key_b: str) -> bool:
        """Whether two keys share bit-identical NF4/nested codebooks (cached).

        The codebooks are quantization constants (weight-independent), so w1 and
        w3 always match in practice; verifying it once guards the concat-dequant
        against subsequent divergence without a per-call host sync."""
        pair = (key_a, key_b)
        cached = self._nf4_pair_codebook_cache.get(pair)
        if cached is not None:
            return cached
        ok = bool(
            torch.equal(getattr(self, f"{key_a}_nf4_code"), getattr(self, f"{key_b}_nf4_code"))
            and torch.equal(
                getattr(self, f"{key_a}_nf4_ncode"),
                getattr(self, f"{key_b}_nf4_ncode"),
            )
        )
        self._nf4_pair_codebook_cache[pair] = ok
        return ok

    def _nf4_dequantize_key_pair_batched(
        self,
        key_a: str,
        key_b: str,
        expert_indices: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Dequantize the same chunk of two identically-shaped NF4 keys (w1/w3)
        in a single pair of bnb calls by concatenating their packed stacks.

        Returns (weight_a, weight_b), each ``[num_selected, out, in]``, bit-
        identical to running ``_nf4_dequantize_expert_stack_batched`` per key.
        Returns None (caller falls back) if shapes differ, codebooks diverge, or
        a bnb precondition/API check fails."""
        meta = self._nf4_meta
        if meta is None or key_a not in self._nf4_shapes or key_b not in self._nf4_shapes:
            return None
        if self._nf4_shapes[key_a] != self._nf4_shapes[key_b]:
            return None
        target = torch.device(device)
        if target.type != "cuda":
            return None
        if not self._nf4_batched_preconditions_ok(key_a, meta):
            return None
        if not self._nf4_pair_codebooks_match(key_a, key_b):
            return None
        num_selected = len(expert_indices)
        _, out_features, in_features = self._nf4_shapes[key_a]
        pa, aa, na, oa = self._nf4_gather_buffers(key_a, expert_indices, target)
        pb, ab, nb, ob = self._nf4_gather_buffers(key_b, expert_indices, target)
        packed = torch.cat([pa, pb], dim=0)
        absmax_q = torch.cat([aa, ab], dim=0)
        nested_absmax = torch.cat([na, nb], dim=0)
        offsets = torch.cat([oa, ob], dim=0)
        code = getattr(self, f"{key_a}_nf4_code").to(device=target)
        nested_code = getattr(self, f"{key_a}_nf4_ncode").to(device=target)
        weight = self._nf4_dequantize_core(
            packed=packed,
            absmax_q=absmax_q,
            nested_absmax=nested_absmax,
            offsets=offsets,
            code=code,
            nested_code=nested_code,
            num_selected=2 * num_selected,
            out_features=int(out_features),
            in_features=int(in_features),
            dtype=dtype,
            meta=meta,
        )
        if weight is None:
            return None
        return weight[:num_selected], weight[num_selected:]

    def _int8_dequantize_key_pair_batched(
        self,
        key_a: str,
        key_b: str,
        expert_indices: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Dequantize two identically-shaped int8 (torch-backend) keys in a
        single ``_dequantize_weight`` over their concatenated stacks. Returns
        None for disk-backed / mismatched-shape / mismatched-group_size cases."""
        if self._packed_source is not None and not hasattr(self, f"{key_a}_int8"):
            return None
        if not hasattr(self, f"{key_a}_int8") or not hasattr(self, f"{key_b}_int8"):
            return None
        q_a = getattr(self, f"{key_a}_int8")
        q_b = getattr(self, f"{key_b}_int8")
        if tuple(q_a.shape[1:]) != tuple(q_b.shape[1:]):
            return None
        group_a = int(getattr(self, f"{key_a}_group_size"))
        group_b = int(getattr(self, f"{key_b}_group_size"))
        if group_a != group_b:
            return None
        rotation_a = self.expert_rotation(key_a)
        rotation_b = self.expert_rotation(key_b)
        if (rotation_a is None) != (rotation_b is None):
            return None
        if rotation_a is not None and not torch.equal(rotation_a, rotation_b):
            return None
        s_a = getattr(self, f"{key_a}_scale")
        s_b = getattr(self, f"{key_b}_scale")
        index_a = torch.as_tensor(expert_indices, device=q_a.device, dtype=torch.long)
        index_b = torch.as_tensor(expert_indices, device=q_b.device, dtype=torch.long)
        num_selected = len(expert_indices)
        quantized = torch.cat([q_a.index_select(0, index_a), q_b.index_select(0, index_b)], dim=0)
        scale = torch.cat([s_a.index_select(0, index_a), s_b.index_select(0, index_b)], dim=0)
        weight = _dequantize_weight(
            quantized,
            scale,
            group_size=group_a,
            rotation=rotation_a,
            dtype=dtype,
            device=device,
        )
        return weight[:num_selected], weight[num_selected:]

    def _dequant_expert_pair(
        self,
        key_a: str,
        key_b: str,
        expert_indices: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize two same-shaped keys (w1/w3) together when possible.

        memory.moe_pair_dequant=false, or disabling
        batched dequant and disables the paired path; the
        fallback dequantizes each key separately (identical result to the
        pre-pairing behaviour)."""
        if (
            len(expert_indices) >= 1
            and resolve_moe_pair_dequant()
            and resolve_moe_batched_dequant()
        ):
            if self._quant_format == "nf4":
                paired = self._nf4_dequantize_key_pair_batched(
                    key_a, key_b, expert_indices, dtype=dtype, device=device
                )
                if paired is not None:
                    return paired
            else:
                paired = self._int8_dequantize_key_pair_batched(
                    key_a, key_b, expert_indices, dtype=dtype, device=device
                )
                if paired is not None:
                    return paired
        weight_a = self._dequant_expert_stack(key_a, expert_indices, dtype=dtype, device=device)
        weight_b = self._dequant_expert_stack(key_b, expert_indices, dtype=dtype, device=device)
        return weight_a, weight_b

    def _expert_linear_batched_requant(
        self,
        key: str,
        x: torch.Tensor,
        active: list[int],
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        output = (
            self._blockwise_fp8_linear(key, x, tuple(active))
            if self._quant_format in BLOCKWISE_FP8_FORMATS
            else _FrozenDequantBatchedLinear.apply(x, self, key, tuple(active))
        )
        adapter = self.expert_lora[key] if key in self.expert_lora else None
        if adapter is not None:
            output = output + adapter.batched_subset_forward(x, expert_indices=expert_indices)
        return output

    def _expert_linear_batched_requant_pair(
        self,
        key_a: str,
        key_b: str,
        x: torch.Tensor,
        active: list[int],
        expert_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched linear for two same-input keys (w1/w3) sharing one dequant.

        Equivalent to two ``_expert_linear_batched_requant`` calls but the frozen
        weights are dequantized once for both keys (see
        ``_FrozenDequantBatchedLinearPair`` / ``_dequant_expert_pair``). LoRA
        adapters remain per-key on top of the shared frozen linear."""
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            out_a = self._blockwise_fp8_linear(key_a, x, tuple(active))
            out_b = self._blockwise_fp8_linear(key_b, x, tuple(active))
        else:
            out_a, out_b = _FrozenDequantBatchedLinearPair.apply(
                x, self, key_a, key_b, tuple(active)
            )
        adapter_a = self.expert_lora[key_a] if key_a in self.expert_lora else None
        if adapter_a is not None:
            out_a = out_a + adapter_a.batched_subset_forward(x, expert_indices=expert_indices)
        adapter_b = self.expert_lora[key_b] if key_b in self.expert_lora else None
        if adapter_b is not None:
            out_b = out_b + adapter_b.batched_subset_forward(x, expert_indices=expert_indices)
        return out_a, out_b

    def _expert_linear_batched_triton(
        self,
        key: str,
        x: torch.Tensor,
        active: list[int],
        expert_indices: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Triton grouped-GEMM analogue of ``_expert_linear_batched_requant``.

        Frozen linear via ``_FrozenTritonGroupedLinear`` (count-aware, padding
        skipped); LoRA adapter added on the identical padded path."""
        output = (
            self._blockwise_fp8_linear(key, x, tuple(active))
            if self._quant_format in BLOCKWISE_FP8_FORMATS
            else _FrozenTritonGroupedLinear.apply(x, self, key, tuple(active), counts)
        )
        adapter = self.expert_lora[key] if key in self.expert_lora else None
        if adapter is not None:
            output = output + adapter.batched_subset_forward(x, expert_indices=expert_indices)
        return output

    def _expert_linear_batched_triton_pair(
        self,
        key_a: str,
        key_b: str,
        x: torch.Tensor,
        active: list[int],
        expert_indices: torch.Tensor,
        counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Triton analogue of ``_expert_linear_batched_requant_pair`` (w1/w3)."""
        if self._quant_format in BLOCKWISE_FP8_FORMATS:
            out_a = self._blockwise_fp8_linear(key_a, x, tuple(active))
            out_b = self._blockwise_fp8_linear(key_b, x, tuple(active))
        else:
            out_a, out_b = _FrozenTritonGroupedLinearPair.apply(
                x, self, key_a, key_b, tuple(active), counts
            )
        adapter_a = self.expert_lora[key_a] if key_a in self.expert_lora else None
        if adapter_a is not None:
            out_a = out_a + adapter_a.batched_subset_forward(x, expert_indices=expert_indices)
        adapter_b = self.expert_lora[key_b] if key_b in self.expert_lora else None
        if adapter_b is not None:
            out_b = out_b + adapter_b.batched_subset_forward(x, expert_indices=expert_indices)
        return out_a, out_b

    def extra_repr(self) -> str:
        if self._quant_format in GGUF_FORMATS:
            storage = self._quant_format
        else:
            storage = self._quant_format
        return (
            f"num_experts={self.num_experts}, dtype={storage}, "
            f"expert_weight_access={self.expert_weight_access}"
        )
