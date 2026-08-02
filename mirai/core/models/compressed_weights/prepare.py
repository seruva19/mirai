"""Module preparation/replacement helpers for compressed_weights quantization (pure move)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping

from mirai.core.moe.runtime.specs import normalize_expert_weight_access_policy

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


from .quantization.quant import (
    CompressedWeightReport,
    NF4_BLOCKSIZE,
    best_group_size,
    normalize_quant_format,
)
from .execution.linear import CompressedLinear
from .execution.experts import CompressedGroupedExperts
from .execution.mixed_precision import MixedPrecisionGroupedExperts
from .quantization.structured_sparsity import StructuredSparse24GroupedExperts
from .quantization.learned_rotation import (
    expert_weight_fingerprint,
    learn_groupwise_expert_rotation,
)

logger = logging.getLogger(__name__)


def apply_structured_2_4_experts(root: nn.Module, *, backend: str = "auto") -> int:
    """Replace every native grouped-expert host and return the replacement count."""

    replaced = 0

    def visit(parent: nn.Module) -> None:
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            if isinstance(child, StructuredSparse24GroupedExperts):
                continue
            if is_dense_grouped_expert_module(child):
                setattr(
                    parent,
                    child_name,
                    StructuredSparse24GroupedExperts(child, backend=backend),
                )
                replaced += 1
                continue
            visit(child)

    visit(root)
    return replaced


def is_dense_grouped_expert_module(module: Any) -> bool:
    """Return whether ``module`` exposes Mirai's dense grouped-expert contract."""

    if isinstance(module, CompressedGroupedExperts):
        return False
    if not hasattr(module, "num_experts"):
        return False
    for key in ("w1", "w2", "w3"):
        if not hasattr(module, key):
            return False
        value = getattr(module, key)
        if not torch.is_tensor(value) or value.ndim != 3:
            return False
        if int(value.shape[0]) != int(getattr(module, "num_experts")):
            return False
    return True


def _quantize_grouped_expert_module(
    module: nn.Module,
    *,
    module_name: str,
    group_sizes: str | int | Iterable[int] | None,
    expert_weight_access: str,
    expert_dequant_chunk_size: int,
    quant_format: str = "int8",
    nf4_blocksize: int = NF4_BLOCKSIZE,
    learn_rotations: bool = False,
    rotation_optimization_steps: int = 200,
    rotation_learning_rate: float = 0.01,
    rotation_row_chunk_size: int = 4096,
    rotation_checkpoint_interval: int = 25,
    rotation_device: Any = "cpu",
    rotation_max_workspace_gib: float = 2.0,
) -> CompressedGroupedExperts:
    replacement = CompressedGroupedExperts.from_empty(
        num_experts=int(getattr(module, "num_experts")),
        group_sizes=group_sizes,
        expert_weight_access=expert_weight_access,
        expert_dequant_chunk_size=expert_dequant_chunk_size,
        quant_format=quant_format,
        nf4_blocksize=nf4_blocksize,
    )
    parametrizations = getattr(module, "parametrizations", None)
    adapters: list[tuple[str, Any]] = []
    reference_weights: dict[str, Any] = {}
    for key in ("w1", "w2", "w3"):
        source = getattr(module, key)
        if parametrizations is not None and hasattr(parametrizations, key):
            chain = getattr(parametrizations, key)
            source = chain.original
            if len(chain) != 1 or not hasattr(chain[0], "lora_a"):
                raise ValueError(
                    f"Unsupported expert parametrization chain for '{key}' during int8 conversion."
                )
            if bool(getattr(chain[0], "use_dora", False)):
                raise ValueError(
                    f"DoRA expert target '{key}' cannot migrate to packed "
                    "execution because its normalized direction requires the "
                    "complete dense expert tensor."
                )
            adapters.append((key, chain[0]))
        reference_weights[key] = source.detach()

    rotations: dict[str, Any] = {}
    rotation_report: dict[str, Any] | None = None
    if learn_rotations:
        if normalize_quant_format(quant_format) != "int8":
            raise ValueError("Learned expert rotations require INT8 quantization.")
        group_w1 = best_group_size(
            int(reference_weights["w1"].shape[-1]),
            group_sizes,
        )
        group_w3 = best_group_size(
            int(reference_weights["w3"].shape[-1]),
            group_sizes,
        )
        if group_w1 <= 0 or group_w1 != group_w3:
            raise ValueError(
                "Learned expert rotations require w1/w3 to share a positive "
                "quantization group size."
            )
        group_w2 = best_group_size(
            int(reference_weights["w2"].shape[-1]),
            group_sizes,
        )
        if group_w2 <= 0:
            raise ValueError(
                "Learned expert rotations require a positive w2 group size."
            )
        shared = learn_groupwise_expert_rotation(
            (reference_weights["w1"], reference_weights["w3"]),
            group_size=group_w1,
            optimization_steps=rotation_optimization_steps,
            learning_rate=rotation_learning_rate,
            row_chunk_size=rotation_row_chunk_size,
            checkpoint_interval=rotation_checkpoint_interval,
            device=rotation_device,
            max_workspace_gib=rotation_max_workspace_gib,
        )
        down = learn_groupwise_expert_rotation(
            (reference_weights["w2"],),
            group_size=group_w2,
            optimization_steps=rotation_optimization_steps,
            learning_rate=rotation_learning_rate,
            row_chunk_size=rotation_row_chunk_size,
            checkpoint_interval=rotation_checkpoint_interval,
            device=rotation_device,
            max_workspace_gib=rotation_max_workspace_gib,
        )
        rotations = {
            "w1": shared.rotation,
            "w3": shared.rotation,
            "w2": down.rotation,
        }
        rotation_report = {
            "module": str(module_name),
            "source_weight_fingerprint": expert_weight_fingerprint(
                {
                    f"{module_name}.{key}": value
                    for key, value in reference_weights.items()
                }
            ),
            "w1_w3": {
                "group_size": int(shared.group_size),
                "initial_relative_error": float(
                    shared.initial_relative_error
                ),
                "optimized_relative_error": float(
                    shared.optimized_relative_error
                ),
            },
            "w2": {
                "group_size": int(down.group_size),
                "initial_relative_error": float(
                    down.initial_relative_error
                ),
                "optimized_relative_error": float(
                    down.optimized_relative_error
                ),
            },
            "optimization_steps": int(rotation_optimization_steps),
            "learning_rate": float(rotation_learning_rate),
        }

    for key, source in reference_weights.items():
        replacement.load_dense_weight(
            key,
            source,
            rotation=rotations.get(key),
        )
    if rotation_report is not None:
        replacement._learned_rotation_report = rotation_report
    for key, source_adapter in adapters:
        adapter = replacement.attach_expert_lora(
            tensor_name=key,
            adapter_name=str(source_adapter.adapter_name),
            rank=int(source_adapter.rank),
            alpha=float(source_adapter.lora_alpha.detach().float().item()),
            init=str(getattr(source_adapter, "_lora_init", "kaiming")),
            use_rslora=bool(getattr(source_adapter, "use_rslora", False)),
        )
        with torch.no_grad():
            adapter.lora_a.copy_(source_adapter.lora_a)
            adapter.lora_b.copy_(source_adapter.lora_b)
        initialize_expert = getattr(
            adapter, "initialize_expert_from_quantized_base", None
        )
        if callable(initialize_expert):
            for expert_idx in range(int(replacement.num_experts)):
                initialize_expert(
                    expert_idx=expert_idx,
                    reference_weight=reference_weights[key][expert_idx],
                    quantized_weight=replacement._dequantize_expert(
                        key,
                        expert_idx,
                        dtype=torch.float32,
                        device=reference_weights[key].device,
                    ),
                )
        adapter.set_lora_scale(float(source_adapter._lora_scale))
        adapter.set_rank_dropout(float(source_adapter._rank_dropout))
        adapter.set_lora_parameter_dropout(
            float(source_adapter._lora_parameter_dropout)
        )
        adapter.set_rank_schedule_scale(float(source_adapter._rank_schedule_scale))
    return replacement


def prepare_compressed_weights_modules_for_checkpoint_load(
    root: nn.Module,
    *,
    group_sizes: str | int | Iterable[int] | None = None,
    expert_weight_access: str = "full_dequant",
    expert_dequant_chunk_size: int = 0,
    replace_linear: bool = True,
    replace_grouped_experts: bool = True,
    quant_format: str = "int8",
    nf4_blocksize: int = NF4_BLOCKSIZE,
) -> tuple[CompressedWeightReport, dict[str, Callable[[torch.Tensor], None]]]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("compressed_weights quantization requires torch.")
    quant_format = normalize_quant_format(quant_format)
    if not replace_linear and not replace_grouped_experts:
        return (
            CompressedWeightReport(
                linear_modules=0,
                grouped_expert_modules=0,
                quantized_tensors=0,
                quantized_numel=0,
            ),
            {},
        )

    linear_count = 0
    expert_count = 0
    tensor_count = 0
    quantized_numel = 0
    skipped: list[str] = []
    handlers: dict[str, Callable[[torch.Tensor], None]] = {}

    def visit(parent: nn.Module, prefix: str = "") -> None:
        nonlocal linear_count, expert_count, tensor_count, quantized_numel
        for child_name, child in list(parent.named_children()):
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, (CompressedLinear, CompressedGroupedExperts)):
                skipped.append(child_prefix)
                continue
            if isinstance(child, nn.Linear) and replace_linear:
                replacement = CompressedLinear.from_empty(
                    in_features=int(child.in_features),
                    out_features=int(child.out_features),
                    group_size=best_group_size(int(child.in_features), group_sizes),
                    has_bias=child.bias is not None,
                    quant_format=quant_format,
                    nf4_blocksize=nf4_blocksize,
                )
                setattr(parent, child_name, replacement)
                linear_count += 1
                tensor_count += 1
                quantized_numel += replacement.frozen_quantized_numel()
                handlers[f"{child_prefix}.weight"] = lambda source, module=replacement: module.load_dense_weight(
                    source=source
                )
                if child.bias is not None:
                    handlers[f"{child_prefix}.bias"] = lambda source, module=replacement: module.load_dense_bias(
                        source=source
                    )
                continue
            if is_dense_grouped_expert_module(child) and replace_grouped_experts:
                if quant_format != "int8":
                    replacement = CompressedGroupedExperts.from_empty(
                        num_experts=int(getattr(child, "num_experts")),
                        group_sizes=group_sizes,
                        expert_weight_access=expert_weight_access,
                        expert_dequant_chunk_size=expert_dequant_chunk_size,
                        quant_format=quant_format,
                        nf4_blocksize=nf4_blocksize,
                    )
                else:
                    replacement = CompressedGroupedExperts(
                        child,
                        group_sizes=group_sizes,
                        expert_weight_access=expert_weight_access,
                        expert_dequant_chunk_size=expert_dequant_chunk_size,
                        quant_format=quant_format,
                        nf4_blocksize=nf4_blocksize,
                    )
                setattr(parent, child_name, replacement)
                expert_count += 1
                tensor_count += 3
                quantized_numel += replacement.frozen_quantized_numel()
                for key in ("w1", "w2", "w3"):
                    handlers[f"{child_prefix}.{key}"] = lambda source, module=replacement, name=key: module.load_dense_weight(
                        name, source
                    )
                continue
            visit(child, child_prefix)

    visit(root)
    return (
        CompressedWeightReport(
            linear_modules=linear_count,
            grouped_expert_modules=expert_count,
            quantized_tensors=tensor_count,
            quantized_numel=quantized_numel,
            expert_weight_access=normalize_expert_weight_access_policy(expert_weight_access),
            expert_dequant_chunk_size=int(expert_dequant_chunk_size),
            skipped_modules=tuple(skipped),
        ),
        handlers,
    )


def quantize_compressed_weights_modules(
    root: nn.Module,
    *,
    group_sizes: str | int | Iterable[int] | None = None,
    expert_weight_access: str = "full_dequant",
    expert_dequant_chunk_size: int = 0,
    quant_format: str = "int8",
    nf4_blocksize: int = NF4_BLOCKSIZE,
    expert_formats: Iterable[str] | None = None,
    expert_tensor_formats: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
    learn_expert_rotations: bool = False,
    rotation_optimization_steps: int = 200,
    rotation_learning_rate: float = 0.01,
    rotation_row_chunk_size: int = 4096,
    rotation_checkpoint_interval: int = 25,
    rotation_device: Any = "cpu",
    rotation_max_workspace_gib: float = 2.0,
) -> CompressedWeightReport:
    if torch is None:  # pragma: no cover
        raise RuntimeError("compressed_weights quantization requires torch.")
    quant_format = normalize_quant_format(quant_format)
    if learn_expert_rotations and quant_format != "int8":
        raise ValueError("Learned expert rotations require INT8 quantization.")
    if learn_expert_rotations and (
        expert_formats is not None or expert_tensor_formats is not None
    ):
        raise ValueError(
            "Learned expert rotations cannot be combined with mixed expert "
            "precision plans."
        )

    linear_count = 0
    expert_count = 0
    tensor_count = 0
    quantized_numel = 0
    skipped: list[str] = []
    used_tensor_plans: set[str] = set()

    def visit(parent: nn.Module, prefix: str = "") -> None:
        nonlocal linear_count, expert_count, tensor_count, quantized_numel
        for child_name, child in list(parent.named_children()):
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(
                child,
                (CompressedLinear, CompressedGroupedExperts, MixedPrecisionGroupedExperts),
            ):
                skipped.append(child_prefix)
                continue
            if isinstance(child, nn.Linear):
                replacement = CompressedLinear(
                    child,
                    group_sizes=group_sizes,
                    quant_format=quant_format,
                    nf4_blocksize=nf4_blocksize,
                )
                setattr(parent, child_name, replacement)
                initialize_from_quantized_base = getattr(
                    parent, "initialize_from_quantized_base", None
                )
                if callable(initialize_from_quantized_base):
                    initialize_from_quantized_base(
                        reference_weight=child.weight.detach(),
                        quantized_weight=replacement.weight.detach(),
                    )
                linear_count += 1
                tensor_count += 1
                quantized_numel += replacement.frozen_quantized_numel()
                continue
            if is_dense_grouped_expert_module(child):
                tensor_formats = (
                    expert_tensor_formats.get(child_prefix)
                    if expert_tensor_formats is not None
                    else None
                )
                if expert_tensor_formats is not None and tensor_formats is None:
                    raise ValueError(
                        f"Tensor precision plan has no grouped expert module "
                        f"{child_prefix!r}."
                    )
                if tensor_formats is not None:
                    replacement = MixedPrecisionGroupedExperts(
                        child,
                        formats={
                            key: tuple(tensor_formats[key])
                            for key in ("w1", "w2", "w3")
                        },
                        group_sizes=group_sizes,
                    )
                    used_tensor_plans.add(child_prefix)
                elif expert_formats is None:
                    replacement = _quantize_grouped_expert_module(
                        child,
                        module_name=child_prefix,
                        group_sizes=group_sizes,
                        expert_weight_access=expert_weight_access,
                        expert_dequant_chunk_size=expert_dequant_chunk_size,
                        quant_format=quant_format,
                        nf4_blocksize=nf4_blocksize,
                        learn_rotations=learn_expert_rotations,
                        rotation_optimization_steps=rotation_optimization_steps,
                        rotation_learning_rate=rotation_learning_rate,
                        rotation_row_chunk_size=rotation_row_chunk_size,
                        rotation_checkpoint_interval=rotation_checkpoint_interval,
                        rotation_device=rotation_device,
                        rotation_max_workspace_gib=rotation_max_workspace_gib,
                    )
                else:
                    replacement = MixedPrecisionGroupedExperts(
                        child,
                        formats=tuple(expert_formats),
                        group_sizes=group_sizes,
                    )
                setattr(parent, child_name, replacement)
                expert_count += 1
                tensor_count += 3
                quantized_numel += replacement.frozen_quantized_numel()
                continue
            visit(child, child_prefix)

    visit(root)
    if expert_tensor_formats is not None:
        unused = set(expert_tensor_formats) - used_tensor_plans
        if unused:
            raise ValueError(
                "Tensor precision plan references unknown grouped expert modules: "
                + ", ".join(sorted(unused))
                + "."
            )
    return CompressedWeightReport(
        linear_modules=linear_count,
        grouped_expert_modules=expert_count,
        quantized_tensors=tensor_count,
        quantized_numel=quantized_numel,
        expert_weight_access=normalize_expert_weight_access_policy(expert_weight_access),
        expert_dequant_chunk_size=int(expert_dequant_chunk_size),
        skipped_modules=tuple(skipped),
    )


def combine_compressed_weights_reports(*reports: CompressedWeightReport | None) -> CompressedWeightReport | None:
    present = [report for report in reports if report is not None]
    if not present:
        return None
    return CompressedWeightReport(
        linear_modules=sum(report.linear_modules for report in present),
        grouped_expert_modules=sum(report.grouped_expert_modules for report in present),
        quantized_tensors=sum(report.quantized_tensors for report in present),
        quantized_numel=sum(report.quantized_numel for report in present),
        expert_weight_access=present[-1].expert_weight_access,
        expert_dequant_chunk_size=present[-1].expert_dequant_chunk_size,
        skipped_modules=tuple(
            item for report in present for item in report.skipped_modules
        ),
    )
