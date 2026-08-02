"""Offline expert expansion with Drop-Upcycling-style diversification.

The transform operates on Mirai packed grouped-expert artifacts.  Every source
expert is retained byte-for-byte and followed by a uniform number of physical
copies.  Each copy reinitializes one shared set of intermediate channels across
``w1``, ``w2``, and ``w3`` using the selected source-channel statistics.  The
interleaved layout preserves contiguous router groups when all experts receive
the same number of copies.

This is an adaptation of Drop-Upcycling's diversity initialization to splitting
an already routed expert.  It is not a dense-to-MoE conversion and makes no
claim about the paper's language-model pretraining results.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from mirai.core.models.compressed_weights.artifact_source import (
    load_grouped_expert_source,
)
from mirai.core.models.compressed_weights.execution.experts import (
    CompressedGroupedExperts,
)
from mirai.core.models.compressed_weights.factorization.prototype_projection import (
    CompressedExpertProjectionSource,
)
from mirai.core.models.compressed_weights.packed.packed_state import (
    _nf4_meta_from_spec,
)
from mirai.core.models.compressed_weights.quantization.blockwise_fp8 import (
    BLOCKWISE_FP8_FORMATS,
)
from mirai.core.models.compressed_weights.quantization.gguf_quant import GGUF_FORMATS
from mirai.core.models.compressed_weights.quantization.microscaling_quant import (
    MICROSCALING_FORMATS,
)
from mirai.core.models.compressed_weights.quantization.quant import (
    NF4_BLOCKSIZE,
    normalize_quant_format,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


DROP_UPCYCLING_ALGORITHM = "drop_upcycling_partial_reinit_v1"
DROP_UPCYCLING_ROUTER_STD = 0.02


@dataclass(frozen=True)
class DropUpcyclingSpec:
    """Validated data-free expert-splitting policy."""

    copies_per_expert: int
    reinitialization_ratio: float
    seed: int

    def validate(self) -> "DropUpcyclingSpec":
        if int(self.copies_per_expert) < 1:
            raise ValueError("Drop-Upcycling requires copies_per_expert >= 1.")
        ratio = float(self.reinitialization_ratio)
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError(
                "Drop-Upcycling reinitialization_ratio must be finite and in (0, 1]."
            )
        if int(self.seed) < 0:
            raise ValueError("Drop-Upcycling seed must be >= 0.")
        return self

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "algorithm": DROP_UPCYCLING_ALGORITHM,
            "copies_per_expert": int(self.copies_per_expert),
            "reinitialization_ratio": float(self.reinitialization_ratio),
            "seed": int(self.seed),
            "router_initialization": "uniform_std_0.02",
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def _module_seed(seed: int, module_name: str) -> int:
    digest = hashlib.sha256(
        f"{int(seed)}\0{str(module_name)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _expert_axis_names(quant_format: str, key: str) -> tuple[str, ...]:
    if quant_format == "nf4":
        return (
            f"{key}_nf4",
            f"{key}_nf4_absmax",
            f"{key}_nf4_nabsmax",
            f"{key}_nf4_offset",
        )
    if quant_format in GGUF_FORMATS:
        return (f"{key}_gguf",)
    if quant_format in MICROSCALING_FORMATS:
        return (f"{key}_mx", f"{key}_mx_scale", f"{key}_mx_global")
    if quant_format in BLOCKWISE_FP8_FORMATS:
        return (f"{key}_fp8", f"{key}_fp8_scale")
    return (f"{key}_int8", f"{key}_scale")


def _shared_names(quant_format: str, key: str) -> tuple[str, ...]:
    if quant_format == "nf4":
        return (f"{key}_nf4_code", f"{key}_nf4_ncode")
    return ()


def _encoder_for_projection(
    spec: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    key: str,
    dense: Any,
) -> CompressedGroupedExperts:
    quant_format = normalize_quant_format(str(spec.get("quant_format", "int8")))
    group_sizes = spec.get("group_sizes")
    if not isinstance(group_sizes, Mapping):
        raise ValueError("Drop-Upcycling source has no group_sizes map.")
    nf4_blocksize = NF4_BLOCKSIZE
    if quant_format == "nf4":
        nf4_blocksize = int(_nf4_meta_from_spec(spec).blocksize)
    encoded = CompressedGroupedExperts.from_empty(
        num_experts=1,
        group_sizes=(int(group_sizes[key]) if quant_format == "int8" else None),
        expert_weight_access="active_dequant",
        quant_format=quant_format,
        nf4_blocksize=nf4_blocksize,
    )
    rotation = None
    rotations = spec.get("rotations")
    if rotations is not None:
        if quant_format != "int8" or not isinstance(rotations, Mapping):
            raise ValueError("Drop-Upcycling source has invalid learned rotations.")
        rotation_key = str(rotations.get(key, ""))
        if not rotation_key or rotation_key not in tensors:
            raise ValueError("Drop-Upcycling source has incomplete learned rotations.")
        rotation = tensors[rotation_key]
    encoded.load_dense_weight(key, dense.unsqueeze(0), rotation=rotation)
    return encoded


def _partially_reinitialize(
    dense_by_key: Mapping[str, Any],
    *,
    ratio: float,
    generator: Any,
) -> tuple[dict[str, Any], Any]:
    intermediate = int(dense_by_key["w1"].shape[0])
    if int(dense_by_key["w3"].shape[0]) != intermediate:
        raise ValueError("Drop-Upcycling requires equal w1/w3 intermediate axes.")
    if int(dense_by_key["w2"].shape[1]) != intermediate:
        raise ValueError("Drop-Upcycling requires w2 columns to match w1/w3 rows.")
    count = int(math.floor(float(ratio) * intermediate))
    if count < 1:
        raise ValueError(
            "Drop-Upcycling ratio selects no intermediate channel for this module."
        )
    selected = torch.randperm(intermediate, generator=generator)[:count].sort().values
    output: dict[str, Any] = {}
    for key in ("w1", "w2", "w3"):
        source = dense_by_key[key].float()
        clone = source.clone()
        selected_values = (
            source.index_select(1, selected)
            if key == "w2"
            else source.index_select(0, selected)
        )
        mean = selected_values.mean()
        std = selected_values.std(unbiased=True)
        if not bool(torch.isfinite(mean).item()) or not bool(torch.isfinite(std).item()):
            raise ValueError(
                f"Drop-Upcycling found non-finite selected-channel statistics for {key}."
            )
        replacement = torch.normal(
            mean=float(mean.item()),
            std=float(std.item()),
            size=tuple(selected_values.shape),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        if key == "w2":
            clone[:, selected] = replacement
        else:
            clone[selected, :] = replacement
        output[key] = clone
    return output, selected


def _expand_residual_rows(
    tensors: dict[str, Any],
    residual_tensor_keys: tuple[str, ...],
    *,
    parent_prefix: str,
    old_num_experts: int,
    source_ids: Any,
    clone_positions: Any,
    hidden_size: int,
    generator: Any,
) -> tuple[str, ...]:
    router_matrices: list[str] = []
    bound = math.sqrt(3.0) * DROP_UPCYCLING_ROUTER_STD
    for tensor_key in residual_tensor_keys:
        shares_parent = (
            tensor_key.startswith(parent_prefix + ".") if parent_prefix else True
        )
        tensor = tensors.get(tensor_key)
        if (
            not shares_parent
            or tensor is None
            or tensor.ndim < 1
            or int(tensor.shape[0]) != int(old_num_experts)
        ):
            continue
        expanded = tensor.index_select(0, source_ids.to(tensor.device)).contiguous()
        if tensor.ndim == 2 and int(tensor.shape[1]) == int(hidden_size):
            random_rows = torch.empty(
                (int(clone_positions.numel()), int(hidden_size)),
                dtype=torch.float32,
                device="cpu",
            ).uniform_(-bound, bound, generator=generator)
            expanded[clone_positions.to(expanded.device)] = random_rows.to(
                device=expanded.device, dtype=expanded.dtype
            )
            router_matrices.append(tensor_key)
        elif tensor.ndim == 1:
            expanded[clone_positions.to(expanded.device)] = 0
        tensors[tensor_key] = expanded
    if not router_matrices:
        raise ValueError(
            "Drop-Upcycling requires a sibling [experts, hidden] router residual "
            f"for grouped module under {parent_prefix!r}."
        )
    return tuple(router_matrices)


def drop_upcycle_packed_state(
    tensors: Mapping[str, Any],
    manifest: Mapping[str, Any],
    spec: DropUpcyclingSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a new packed artifact with uniformly split physical experts."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Drop-Upcycling requires torch.")
    spec = spec.validate()
    if manifest.get("format") != "mirai.compressed_weights.packed_state":
        raise ValueError("Drop-Upcycling requires a Mirai packed-state artifact.")
    if manifest.get("expert_upcycling") is not None:
        raise ValueError("Drop-Upcycling does not accept an already upcycled artifact.")
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("Drop-Upcycling source manifest has no modules object.")
    grouped_names = tuple(
        str(name)
        for name, module_spec in modules.items()
        if isinstance(module_spec, Mapping)
        and str(module_spec.get("kind")) == "grouped_experts"
    )
    if not grouped_names:
        raise ValueError("Drop-Upcycling source has no grouped experts.")

    new_manifest = copy.deepcopy(dict(manifest))
    new_tensors = {str(key): value for key, value in tensors.items()}
    residual = new_manifest.get("residual_tensors") or {}
    if not isinstance(residual, Mapping):
        raise ValueError("Drop-Upcycling source has invalid residual_tensors metadata.")
    residual_tensor_keys = tuple(str(value) for value in residual.values())
    summary_delta = 0
    module_reports: dict[str, Any] = {}

    for module_name in grouped_names:
        module_spec = new_manifest["modules"][module_name]
        if "logical_to_physical" in module_spec or int(
            module_spec.get("logical_num_experts", module_spec.get("num_experts", 0))
        ) != int(module_spec.get("num_experts", 0)):
            raise ValueError("Drop-Upcycling requires an unconsolidated expert pool.")
        if module_spec.get("physical_weight_provider") is not None:
            raise ValueError(
                "Drop-Upcycling does not rewrite physical-weight provider artifacts."
            )
        old_num = int(module_spec.get("num_experts", 0))
        if old_num < 1:
            raise ValueError(f"Grouped module {module_name!r} has no experts.")
        copies = int(spec.copies_per_expert)
        source_ids = torch.arange(old_num, dtype=torch.long).repeat_interleave(
            copies + 1
        )
        expanded_num = int(source_ids.numel())
        clone_mask = (
            torch.arange(expanded_num, dtype=torch.long).remainder(copies + 1) != 0
        )
        clone_positions = torch.nonzero(clone_mask, as_tuple=False).flatten()
        source = load_grouped_expert_source(module_spec, new_tensors)
        projections = CompressedExpertProjectionSource(source)
        projection_specs = {
            item.name: item for item in projections.prototype_projection_specs()
        }
        hidden_size = int(projection_specs["w1"].shape[1])
        quant_format = normalize_quant_format(
            str(module_spec.get("quant_format", "int8"))
        )
        tensor_names = module_spec.get("tensors")
        if not isinstance(tensor_names, Mapping):
            raise ValueError(f"Grouped module {module_name!r} has no tensor map.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_module_seed(spec.seed, module_name))
        encoded_clones: dict[str, list[Any]] = {
            local_name: []
            for key in ("w1", "w2", "w3")
            for local_name in _expert_axis_names(quant_format, key)
        }
        mask_digest = hashlib.sha256()

        for source_expert in range(old_num):
            dense_source = {
                key: projections.load_prototype_projection_block(
                    key,
                    source_expert,
                    source_expert + 1,
                    device="cpu",
                    dtype=torch.float32,
                )[0]
                for key in ("w1", "w2", "w3")
            }
            for copy_index in range(copies):
                diversified, selected = _partially_reinitialize(
                    dense_source,
                    ratio=spec.reinitialization_ratio,
                    generator=generator,
                )
                mask_digest.update(int(source_expert).to_bytes(8, "little"))
                mask_digest.update(int(copy_index).to_bytes(8, "little"))
                mask_digest.update(selected.numpy().tobytes())
                for key in ("w1", "w2", "w3"):
                    encoded = _encoder_for_projection(
                        module_spec,
                        new_tensors,
                        key=key,
                        dense=diversified[key],
                    )
                    state = encoded.state_dict()
                    for local_name in _expert_axis_names(quant_format, key):
                        encoded_clones[local_name].append(
                            state[local_name][0].detach().cpu().contiguous()
                        )
                    for local_name in _shared_names(quant_format, key):
                        tensor_key = str(tensor_names.get(local_name, ""))
                        if not tensor_key or not torch.equal(
                            state[local_name].detach().cpu(),
                            new_tensors[tensor_key].detach().cpu(),
                        ):
                            raise ValueError(
                                f"Drop-Upcycling changed shared metadata {local_name!r}."
                            )

        for key in ("w1", "w2", "w3"):
            for local_name in _expert_axis_names(quant_format, key):
                tensor_key = str(tensor_names.get(local_name, ""))
                if not tensor_key or tensor_key not in new_tensors:
                    raise KeyError(
                        f"Drop-Upcycling source is missing tensor {local_name!r}."
                    )
                source_tensor = new_tensors[tensor_key]
                if source_tensor.ndim < 1 or int(source_tensor.shape[0]) != old_num:
                    raise ValueError(
                        f"Drop-Upcycling tensor {local_name!r} has no expert axis."
                    )
                expanded = source_tensor.index_select(
                    0, source_ids.to(source_tensor.device)
                ).contiguous()
                clone_values = torch.stack(encoded_clones[local_name], dim=0).to(
                    device=expanded.device, dtype=expanded.dtype
                )
                expanded[clone_positions.to(expanded.device)] = clone_values
                new_tensors[tensor_key] = expanded
            shape = [int(value) for value in module_spec["shapes"][key]]
            summary_delta += (expanded_num - old_num) * math.prod(shape[1:])
            shape[0] = expanded_num
            module_spec["shapes"][key] = shape

        parent = module_name.rsplit(".", 1)[0] if "." in module_name else ""
        router_keys = _expand_residual_rows(
            new_tensors,
            residual_tensor_keys,
            parent_prefix=parent,
            old_num_experts=old_num,
            source_ids=source_ids,
            clone_positions=clone_positions,
            hidden_size=hidden_size,
            generator=generator,
        )
        module_spec["num_experts"] = expanded_num
        module_spec["expert_upcycling"] = {
            "algorithm": DROP_UPCYCLING_ALGORITHM,
            "source_num_experts": old_num,
            "expanded_num_experts": expanded_num,
            "copies_per_expert": copies,
            "reinitialization_ratio": float(spec.reinitialization_ratio),
            "module_seed": _module_seed(spec.seed, module_name),
            "mask_fingerprint": "sha256:" + mask_digest.hexdigest(),
            "router_tensor_keys": list(router_keys),
        }
        module_reports[module_name] = copy.deepcopy(module_spec["expert_upcycling"])

    summary = new_manifest.get("summary")
    if isinstance(summary, dict) and "quantized_numel" in summary:
        summary["quantized_numel"] = int(summary["quantized_numel"]) + summary_delta
    new_manifest["expert_upcycling"] = {
        "schema_version": 1,
        **spec.to_dict(),
        "transform_fingerprint": spec.fingerprint(),
        "modules": module_reports,
    }
    return new_tensors, new_manifest


def validate_drop_upcycling_selection(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    copies_per_expert: int,
    reinitialization_ratio: float,
    seed: int,
) -> None:
    """Bind runtime opt-in to the exact transformed artifact policy."""

    normalized_mode = str(mode).strip().lower()
    metadata = manifest.get("expert_upcycling")
    if normalized_mode == "off":
        if metadata is not None:
            raise ValueError(
                "Packed artifact contains upcycled experts but "
                "model.params.expert_upcycling='off'."
            )
        return
    if normalized_mode != "drop":
        raise ValueError("expert_upcycling must be 'off' or 'drop'.")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "expert_upcycling='drop' requires a Drop-Upcycling packed artifact."
        )
    expected = DropUpcyclingSpec(
        copies_per_expert=int(copies_per_expert),
        reinitialization_ratio=float(reinitialization_ratio),
        seed=int(seed),
    ).validate()
    if str(metadata.get("algorithm")) != DROP_UPCYCLING_ALGORITHM:
        raise ValueError("Packed artifact has an unsupported upcycling algorithm.")
    if str(metadata.get("transform_fingerprint")) != expected.fingerprint():
        raise ValueError(
            "Packed expert-upcycling policy does not match the active config."
        )
    if not isinstance(metadata.get("lineage"), Mapping) or not str(
        metadata["lineage"].get("source_artifact_fingerprint", "")
    ).startswith("sha256:"):
        raise ValueError("Packed expert-upcycling artifact has incomplete lineage.")


__all__ = [
    "DROP_UPCYCLING_ALGORITHM",
    "DROP_UPCYCLING_ROUTER_STD",
    "DropUpcyclingSpec",
    "drop_upcycle_packed_state",
    "validate_drop_upcycling_selection",
]
