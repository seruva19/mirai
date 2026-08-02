"""Post-hoc global spectral rank allocation for trained LoRA adapters.

Implements PARA's QR-subspace decomposition and global ``gamma`` / ``epsilon``
selection policies from https://arxiv.org/abs/2604.27796. Grouped expert LoRA
is treated as one adapter matrix per expert. The persisted artifact is ragged;
standard padded factor tensors are reconstructed only when it is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

from mirai.core.models.adapters.lora_allocation import PARA_RANK_STATE_SUFFIX
from mirai.core.models.adapters.lora_allocation import RSLORA_STATE_SUFFIX
from mirai.core.models.adapters.sparse_expert_export import (
    expand_sparse_expert_state,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PARA_SCHEMA = "mirai.para_adapter"
PARA_SCHEMA_VERSION = 1
PARA_METADATA_KEY = "mirai_para_manifest"
PARA_TRANSFORM_KEY = "mirai_adapter_transform"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ParaUnit:
    module_name: str
    group_index: int
    input_features: int
    output_features: int
    original_rank: int
    input_dtype: Any
    output_dtype: Any
    left_vectors: Any
    singular_values: Any
    right_vectors_t: Any

    @property
    def identity(self) -> str:
        return f"{self.module_name}#{self.group_index:06d}"


@dataclass(frozen=True)
class ParaCompressionResult:
    tensors: dict[str, Any]
    manifest: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "policy": self.manifest["policy"],
            "initial_total_rank": self.manifest["initial_total_rank"],
            "retained_total_rank": self.manifest["retained_total_rank"],
            "retained_rank_fraction": self.manifest["retained_rank_fraction"],
            "retained_energy_fraction": self.manifest["retained_energy_fraction"],
            "modules": {
                item["name"]: {
                    "grouped": item["grouped"],
                    "original_rank": item["original_rank"],
                    "retained_ranks": item["retained_ranks"],
                    "runtime_rank": item["runtime_rank"],
                }
                for item in self.manifest["modules"]
            },
        }


def _require_tensor(value: Any, *, name: str) -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PARA compression requires torch.")
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _scalar_float(value: Any, *, default: float) -> float:
    if value is None:
        return float(default)
    if torch is not None and torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("LoRA alpha and scaling markers must be scalar tensors.")
        return float(value.detach().float().item())
    return float(value)


def _decompose_unit(
    *,
    module_name: str,
    group_index: int,
    lora_a: Any,
    lora_b: Any,
) -> ParaUnit:
    a = _require_tensor(lora_a, name=f"{module_name}.lora_a")
    b = _require_tensor(lora_b, name=f"{module_name}.lora_b")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("PARA decomposition units must be matrix LoRA factors.")
    rank = int(a.shape[0])
    if rank <= 0 or int(b.shape[1]) != rank:
        raise ValueError(f"LoRA factors for {module_name!r} disagree on rank.")
    input_features = int(a.shape[1])
    output_features = int(b.shape[0])
    if rank > min(input_features, output_features):
        raise ValueError(
            f"LoRA rank for {module_name!r} exceeds a matrix dimension; "
            "PARA's compact QR derivation requires rank <= min(in, out)."
        )
    compute_dtype = (
        torch.float64
        if a.dtype == torch.float64 or b.dtype == torch.float64
        else torch.float32
    )
    a_compute = a.detach().to(device="cpu", dtype=compute_dtype)
    b_compute = b.detach().to(device="cpu", dtype=compute_dtype)
    q_b, r_b = torch.linalg.qr(b_compute, mode="reduced")
    q_a, r_a = torch.linalg.qr(a_compute.transpose(0, 1), mode="reduced")
    u_small, singular_values, vh_small = torch.linalg.svd(
        r_b @ r_a.transpose(0, 1),
        full_matrices=False,
    )
    return ParaUnit(
        module_name=str(module_name),
        group_index=int(group_index),
        input_features=input_features,
        output_features=output_features,
        original_rank=rank,
        input_dtype=a.dtype,
        output_dtype=b.dtype,
        left_vectors=(q_b @ u_small).contiguous(),
        singular_values=singular_values.contiguous(),
        right_vectors_t=(vh_small @ q_a.transpose(0, 1)).contiguous(),
    )


def _discover_units(
    state: Mapping[str, Any],
) -> tuple[list[ParaUnit], dict[str, dict[str, Any]], dict[str, Any]]:
    expanded = expand_sparse_expert_state(dict(state))
    a_prefixes = {
        key[: -len(".lora_a")]
        for key in expanded
        if str(key).endswith(".lora_a")
    }
    b_prefixes = {
        key[: -len(".lora_b")]
        for key in expanded
        if str(key).endswith(".lora_b")
    }
    if a_prefixes != b_prefixes:
        missing_a = sorted(str(item) for item in b_prefixes - a_prefixes)
        missing_b = sorted(str(item) for item in a_prefixes - b_prefixes)
        raise ValueError(
            "PARA input contains orphan LoRA factors: "
            f"missing A for {missing_a}, missing B for {missing_b}."
        )
    prefixes = sorted(str(item) for item in a_prefixes)
    if not prefixes:
        raise ValueError("PARA found no native LoRA factor pairs.")
    units: list[ParaUnit] = []
    modules: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    for prefix in prefixes:
        a_key = f"{prefix}.lora_a"
        b_key = f"{prefix}.lora_b"
        if b_key not in expanded:
            raise ValueError(f"PARA input is missing {b_key!r}.")
        a = _require_tensor(expanded[a_key], name=a_key)
        b = _require_tensor(expanded[b_key], name=b_key)
        if a.ndim not in {2, 3} or b.ndim != a.ndim:
            raise ValueError(
                f"PARA supports matrix or grouped-matrix factors; got {a_key} "
                f"shape {tuple(a.shape)} and {b_key} shape {tuple(b.shape)}."
            )
        grouped = a.ndim == 3
        groups = int(a.shape[0]) if grouped else 1
        if grouped and int(b.shape[0]) != groups:
            raise ValueError(f"Grouped LoRA factors for {prefix!r} disagree.")
        original_rank = int(a.shape[-2])
        if int(b.shape[-1]) != original_rank:
            raise ValueError(f"LoRA factors for {prefix!r} disagree on rank.")
        alpha_key = f"{prefix}.lora_alpha"
        alpha = _scalar_float(expanded.get(alpha_key), default=float(original_rank))
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"LoRA alpha for {prefix!r} must be finite and positive.")
        scaling_key = f"{prefix}{RSLORA_STATE_SUFFIX}"
        raw_scaling_rule = _scalar_float(
            expanded.get(scaling_key),
            default=0.0,
        )
        if raw_scaling_rule not in {0.0, 1.0}:
            raise ValueError(
                f"rsLoRA marker for {prefix!r} must be exactly zero or one."
            )
        use_rslora = bool(raw_scaling_rule)
        denominator = math.sqrt(original_rank) if use_rslora else float(original_rank)
        modules[prefix] = {
            "name": prefix,
            "grouped": grouped,
            "groups": groups,
            "original_rank": original_rank,
            "input_features": int(a.shape[-1]),
            "output_features": int(b.shape[-2]),
            "input_dtype": str(a.dtype).removeprefix("torch."),
            "output_dtype": str(b.dtype).removeprefix("torch."),
            "original_alpha": alpha,
            "original_scale": alpha / denominator,
            "use_rslora": use_rslora,
        }
        for group_index in range(groups):
            units.append(
                _decompose_unit(
                    module_name=prefix,
                    group_index=group_index,
                    lora_a=a[group_index] if grouped else a,
                    lora_b=b[group_index] if grouped else b,
                )
            )
        consumed.update(
            {
                a_key,
                b_key,
                alpha_key,
                scaling_key,
                f"{prefix}{PARA_RANK_STATE_SUFFIX}",
            }
        )
    collisions = sorted(
        str(key) for key in expanded if str(key).startswith("para.unit_")
    )
    if collisions:
        raise ValueError(
            "PARA input reserves tensor names beginning with 'para.unit_'; "
            f"found {collisions[:3]}."
        )
    passthrough = {
        str(key): value
        for key, value in expanded.items()
        if str(key) not in consumed and torch.is_tensor(value)
    }
    return units, modules, passthrough


def _select_components(
    units: list[ParaUnit],
    *,
    policy: str,
    rank_preservation_ratio: float,
    energy_preservation_ratio: float,
) -> tuple[dict[str, set[int]], float, float]:
    entries: list[tuple[float, str, int]] = []
    total_energy = 0.0
    for unit in units:
        for component_index, raw_value in enumerate(unit.singular_values.tolist()):
            value = float(raw_value)
            entries.append((value, unit.identity, int(component_index)))
            total_energy += value * value
    entries.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not entries:
        raise ValueError("PARA found no singular components.")
    mode = str(policy).strip().lower()
    if mode == "rank":
        ratio = float(rank_preservation_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("PARA rank preservation ratio must be in (0, 1].")
        # Eq. 12 indexes a discrete global budget. Ceil is the conservative
        # extension when gamma * B_init is not integral: the artifact never
        # preserves less rank than the requested ratio.
        retain_count = max(
            1,
            min(len(entries), math.ceil(len(entries) * ratio)),
        )
    elif mode == "energy":
        ratio = float(energy_preservation_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("PARA energy preservation ratio must be in (0, 1].")
        target_energy = total_energy * ratio
        retained_energy = 0.0
        retain_count = 0
        for value, _identity, _component in entries:
            retained_energy += value * value
            retain_count += 1
            if retained_energy >= target_energy:
                break
    else:
        raise ValueError("PARA policy must be 'rank' or 'energy'.")
    selected: dict[str, set[int]] = {}
    retained_energy = 0.0
    for value, identity, component_index in entries[:retain_count]:
        selected.setdefault(identity, set()).add(component_index)
        retained_energy += value * value
    threshold = float(entries[retain_count - 1][0])
    retained_fraction = (
        1.0 if total_energy == 0.0 else retained_energy / total_energy
    )
    return selected, threshold, retained_fraction


def compress_lora_state_para(
    state: Mapping[str, Any],
    *,
    source_adapter_sha256: str,
    policy: str,
    rank_preservation_ratio: float = 0.25,
    energy_preservation_ratio: float = 0.99,
) -> ParaCompressionResult:
    """Build a ragged PARA artifact without materializing full update matrices."""
    source_sha256 = str(source_adapter_sha256).strip().lower()
    if _SHA256_PATTERN.fullmatch(source_sha256) is None:
        raise ValueError(
            "PARA requires source adapter lineage as a 64-character SHA-256."
        )
    units, module_specs, passthrough = _discover_units(state)
    selected, threshold, retained_energy_fraction = _select_components(
        units,
        policy=policy,
        rank_preservation_ratio=rank_preservation_ratio,
        energy_preservation_ratio=energy_preservation_ratio,
    )
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in passthrough.items()
    }
    module_ranks: dict[str, list[int]] = {
        name: [0] * int(spec["groups"])
        for name, spec in module_specs.items()
    }
    unit_records: list[dict[str, Any]] = []
    retained_total_rank = 0
    for unit_index, unit in enumerate(units):
        component_ids = sorted(selected.get(unit.identity, set()))
        rank = len(component_ids)
        retained_total_rank += rank
        module_ranks[unit.module_name][unit.group_index] = rank
        tensor_prefix = f"para.unit_{unit_index:06d}"
        if rank:
            component_tensor = torch.as_tensor(component_ids, dtype=torch.long)
            singular = unit.singular_values.index_select(0, component_tensor)
            root = torch.sqrt(singular)
            left = unit.left_vectors.index_select(1, component_tensor)
            right_t = unit.right_vectors_t.index_select(0, component_tensor)
            tensors[f"{tensor_prefix}.lora_b"] = (
                left * root.unsqueeze(0)
            ).to(dtype=unit.output_dtype).contiguous()
            tensors[f"{tensor_prefix}.lora_a"] = (
                root.unsqueeze(1) * right_t
            ).to(dtype=unit.input_dtype).contiguous()
        unit_records.append(
            {
                "module": unit.module_name,
                "group_index": unit.group_index,
                "rank": rank,
                "a_tensor": f"{tensor_prefix}.lora_a" if rank else "",
                "b_tensor": f"{tensor_prefix}.lora_b" if rank else "",
            }
        )
    module_records: list[dict[str, Any]] = []
    for name, spec in module_specs.items():
        retained_ranks = module_ranks[name]
        runtime_rank = max(1, max(retained_ranks))
        denominator = (
            math.sqrt(runtime_rank)
            if bool(spec["use_rslora"])
            else float(runtime_rank)
        )
        module_records.append(
            {
                **spec,
                "retained_ranks": retained_ranks,
                "runtime_rank": runtime_rank,
                "output_alpha": float(spec["original_scale"]) * denominator,
            }
        )
    initial_total_rank = sum(
        int(unit.singular_values.numel()) for unit in units
    )
    manifest = {
        "schema": PARA_SCHEMA,
        "schema_version": PARA_SCHEMA_VERSION,
        "source_adapter_sha256": source_sha256,
        "policy": str(policy).strip().lower(),
        "rank_preservation_ratio": float(rank_preservation_ratio),
        "energy_preservation_ratio": float(energy_preservation_ratio),
        "threshold": threshold,
        "tie_policy": "descending_value_then_unit_identity_then_component",
        "initial_total_rank": initial_total_rank,
        "retained_total_rank": retained_total_rank,
        "retained_rank_fraction": retained_total_rank / initial_total_rank,
        "retained_energy_fraction": float(retained_energy_fraction),
        "modules": module_records,
        "units": unit_records,
    }
    validate_para_manifest(manifest, tensors)
    return ParaCompressionResult(tensors=tensors, manifest=manifest)


def _dtype_from_name(name: str) -> Any:
    dtype = getattr(torch, str(name), None)
    if dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise ValueError(f"PARA artifact has unsupported dtype {name!r}.")
    return dtype


def validate_para_manifest(
    manifest: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> None:
    if manifest.get("schema") != PARA_SCHEMA:
        raise ValueError("Unsupported PARA adapter schema.")
    if int(manifest.get("schema_version", 0)) != PARA_SCHEMA_VERSION:
        raise ValueError("Unsupported PARA adapter schema version.")
    source_sha256 = str(manifest.get("source_adapter_sha256", "")).strip()
    if _SHA256_PATTERN.fullmatch(source_sha256) is None:
        raise ValueError("PARA artifact has invalid source-adapter lineage.")
    if manifest.get("policy") not in {"rank", "energy"}:
        raise ValueError("PARA artifact has an unsupported policy.")
    if (
        manifest.get("tie_policy")
        != "descending_value_then_unit_identity_then_component"
    ):
        raise ValueError("PARA artifact has an unsupported tie policy.")
    initial_total_rank = int(manifest.get("initial_total_rank", 0))
    retained_total_rank = int(manifest.get("retained_total_rank", -1))
    if initial_total_rank <= 0 or not (
        0 < retained_total_rank <= initial_total_rank
    ):
        raise ValueError("PARA artifact has invalid aggregate ranks.")
    threshold = float(manifest.get("threshold", float("nan")))
    rank_fraction = float(
        manifest.get("retained_rank_fraction", float("nan"))
    )
    energy_fraction = float(
        manifest.get("retained_energy_fraction", float("nan"))
    )
    if (
        not math.isfinite(threshold)
        or threshold < 0.0
        or not math.isfinite(rank_fraction)
        or not math.isfinite(energy_fraction)
        or not (0.0 < rank_fraction <= 1.0)
        or not (0.0 <= energy_fraction <= 1.0)
        or not math.isclose(
            rank_fraction,
            retained_total_rank / initial_total_rank,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("PARA artifact has invalid aggregate diagnostics.")
    modules = manifest.get("modules")
    units = manifest.get("units")
    if not isinstance(modules, list) or not modules:
        raise ValueError("PARA artifact requires module records.")
    if not isinstance(units, list) or not units:
        raise ValueError("PARA artifact requires decomposition-unit records.")
    module_names: set[str] = set()
    module_by_name: dict[str, Mapping[str, Any]] = {}
    for module in modules:
        if not isinstance(module, Mapping):
            raise ValueError("PARA module records must be mappings.")
        name = str(module.get("name", ""))
        groups = int(module.get("groups", 0))
        ranks = module.get("retained_ranks")
        runtime_rank = int(module.get("runtime_rank", 0))
        original_rank = int(module.get("original_rank", 0))
        input_features = int(module.get("input_features", 0))
        output_features = int(module.get("output_features", 0))
        grouped = bool(module.get("grouped", False))
        original_alpha = float(module.get("original_alpha", float("nan")))
        original_scale = float(module.get("original_scale", float("nan")))
        output_alpha = float(module.get("output_alpha", float("nan")))
        if (
            not name
            or name in module_names
            or groups <= 0
            or (not grouped and groups != 1)
            or original_rank <= 0
            or input_features <= 0
            or output_features <= 0
            or original_rank > min(input_features, output_features)
            or not isinstance(ranks, list)
            or len(ranks) != groups
            or any(
                int(rank) < 0 or int(rank) > original_rank
                for rank in ranks
            )
            or runtime_rank != max(1, max(int(rank) for rank in ranks))
            or not math.isfinite(original_alpha)
            or not math.isfinite(original_scale)
            or not math.isfinite(output_alpha)
            or min(original_alpha, original_scale, output_alpha) <= 0.0
        ):
            raise ValueError("PARA module topology is invalid.")
        _dtype_from_name(str(module.get("input_dtype", "")))
        _dtype_from_name(str(module.get("output_dtype", "")))
        module_names.add(name)
        module_by_name[name] = module
    seen_units: set[tuple[str, int]] = set()
    referenced_factor_keys: set[str] = set()
    observed_rank = 0
    for unit in units:
        if not isinstance(unit, Mapping):
            raise ValueError("PARA unit records must be mappings.")
        name = str(unit.get("module", ""))
        group_index = int(unit.get("group_index", -1))
        rank = int(unit.get("rank", -1))
        identity = (name, group_index)
        module = module_by_name.get(name)
        if (
            module is None
            or identity in seen_units
            or group_index < 0
            or group_index >= int(module["groups"])
            or rank < 0
            or rank > int(module["original_rank"])
            or rank != int(module["retained_ranks"][group_index])
        ):
            raise ValueError("PARA unit topology is invalid.")
        seen_units.add(identity)
        observed_rank += rank
        a_key = str(unit.get("a_tensor", ""))
        b_key = str(unit.get("b_tensor", ""))
        if rank == 0:
            if a_key or b_key:
                raise ValueError("Zero-rank PARA units must not store factor tensors.")
        else:
            if (
                not a_key.startswith("para.unit_")
                or not b_key.startswith("para.unit_")
                or a_key in referenced_factor_keys
                or b_key in referenced_factor_keys
                or a_key not in tensors
                or b_key not in tensors
            ):
                raise ValueError("PARA unit is missing unique factor tensors.")
            a = _require_tensor(tensors[a_key], name=a_key)
            b = _require_tensor(tensors[b_key], name=b_key)
            if tuple(a.shape) != (
                rank,
                int(module["input_features"]),
            ) or tuple(b.shape) != (
                int(module["output_features"]),
                rank,
            ):
                raise ValueError(
                    "PARA unit factor shape does not match its manifest."
                )
            if (
                a.dtype != _dtype_from_name(str(module["input_dtype"]))
                or b.dtype != _dtype_from_name(str(module["output_dtype"]))
            ):
                raise ValueError(
                    "PARA unit factor dtype does not match its manifest."
                )
            referenced_factor_keys.update({a_key, b_key})
    expected_units = sum(int(module["groups"]) for module in modules)
    if len(seen_units) != expected_units:
        raise ValueError("PARA artifact does not cover every module group.")
    if observed_rank != retained_total_rank:
        raise ValueError("PARA unit ranks disagree with aggregate retained rank.")
    if sum(
        int(module["groups"]) * int(module["original_rank"])
        for module in modules
    ) != initial_total_rank:
        raise ValueError("PARA module ranks disagree with aggregate initial rank.")
    stored_factor_keys = {
        str(key) for key in tensors if str(key).startswith("para.unit_")
    }
    if stored_factor_keys != referenced_factor_keys:
        raise ValueError("PARA artifact contains unreferenced factor tensors.")


def expand_para_adapter_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct standard padded LoRA state from a validated ragged artifact."""
    metadata = payload.get("_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("PARA safetensors artifact is missing metadata.")
    if str(metadata.get(PARA_TRANSFORM_KEY, "")).strip().lower() != "para":
        raise ValueError("PARA safetensors artifact has the wrong transform marker.")
    raw_manifest = metadata.get(PARA_METADATA_KEY)
    if not isinstance(raw_manifest, str):
        raise ValueError("PARA safetensors artifact is missing its manifest.")
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as exc:
        raise ValueError("PARA safetensors manifest is not valid JSON.") from exc
    tensors = {
        str(key): value
        for key, value in payload.items()
        if str(key) != "_metadata"
    }
    validate_para_manifest(manifest, tensors)
    state = {
        key: value
        for key, value in tensors.items()
        if not key.startswith("para.unit_")
    }
    module_map = {str(item["name"]): item for item in manifest["modules"]}
    unit_map = {
        (str(item["module"]), int(item["group_index"])): item
        for item in manifest["units"]
    }
    for name, module in module_map.items():
        groups = int(module["groups"])
        runtime_rank = int(module["runtime_rank"])
        input_features = int(module["input_features"])
        output_features = int(module["output_features"])
        input_dtype = _dtype_from_name(str(module["input_dtype"]))
        output_dtype = _dtype_from_name(str(module["output_dtype"]))
        grouped = bool(module["grouped"])
        a_shape = (
            (groups, runtime_rank, input_features)
            if grouped
            else (runtime_rank, input_features)
        )
        b_shape = (
            (groups, output_features, runtime_rank)
            if grouped
            else (output_features, runtime_rank)
        )
        a_padded = torch.zeros(a_shape, dtype=input_dtype)
        b_padded = torch.zeros(b_shape, dtype=output_dtype)
        for group_index in range(groups):
            unit = unit_map[(name, group_index)]
            rank = int(unit["rank"])
            if rank <= 0:
                continue
            a = _require_tensor(
                tensors[str(unit["a_tensor"])],
                name=str(unit["a_tensor"]),
            )
            b = _require_tensor(
                tensors[str(unit["b_tensor"])],
                name=str(unit["b_tensor"]),
            )
            if tuple(a.shape) != (rank, input_features) or tuple(b.shape) != (
                output_features,
                rank,
            ):
                raise ValueError("PARA unit factor shape does not match its manifest.")
            if grouped:
                a_padded[group_index, :rank].copy_(a.to(dtype=input_dtype))
                b_padded[group_index, :, :rank].copy_(b.to(dtype=output_dtype))
            else:
                a_padded[:rank].copy_(a.to(dtype=input_dtype))
                b_padded[:, :rank].copy_(b.to(dtype=output_dtype))
        state[f"{name}.lora_a"] = a_padded
        state[f"{name}.lora_b"] = b_padded
        state[f"{name}.lora_alpha"] = torch.tensor(
            [float(module["output_alpha"])],
            dtype=torch.float32,
        )
        state[f"{name}{PARA_RANK_STATE_SUFFIX}"] = torch.tensor(
            [runtime_rank],
            dtype=torch.int64,
        )
        if bool(module["use_rslora"]):
            state[f"{name}{RSLORA_STATE_SUFFIX}"] = torch.ones(
                1,
                dtype=torch.uint8,
            )
    return {
        "adapter_state": state,
        "metadata": {
            "transform": "para",
            "manifest": manifest,
        },
    }


def save_para_adapter_safetensors(
    path: str | Path,
    result: ParaCompressionResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist a validated ragged PARA adapter artifact."""
    from safetensors.torch import save_file

    output = Path(path)
    if output.suffix.lower() != ".safetensors":
        raise ValueError("PARA output must use the .safetensors extension.")
    if output.exists() and not bool(overwrite):
        raise ValueError(f"PARA output already exists: {output}.")
    validate_para_manifest(result.manifest, result.tensors)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in result.tensors.items()
            },
            str(temporary),
            metadata={
                PARA_TRANSFORM_KEY: "para",
                PARA_METADATA_KEY: json.dumps(
                    result.manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        )
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


__all__ = [
    "PARA_METADATA_KEY",
    "PARA_RANK_STATE_SUFFIX",
    "PARA_SCHEMA",
    "PARA_SCHEMA_VERSION",
    "PARA_TRANSFORM_KEY",
    "ParaCompressionResult",
    "compress_lora_state_para",
    "expand_para_adapter_payload",
    "save_para_adapter_safetensors",
    "validate_para_manifest",
]
