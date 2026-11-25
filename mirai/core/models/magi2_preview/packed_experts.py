"""Versioned safetensors persistence for MAGI-2 NF4 routed experts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from mirai.core.models.magi2_preview.mixed_experts import Magi2MixedExpertStore
from mirai.core.models.magi2_preview.quantized_experts import (
    MAGI2_NF4_STORE_ATTR,
    MAGI2_ROUTED_EXPERT_TENSOR_NAMES,
    MAGI2_FUSED_GATE_UP,
    MAGI2_NF4_GROUP_FIELDS,
    MAGI2_NF4_CODE_FIELDS,
    Magi2Nf4ExpertStore,
    Magi2PackedExpertStore,
    install_magi2_nf4_expert_stores,
    iter_magi2_moe_layers,
    magi2_expert_store,
)


MAGI2_PACKED_EXPERT_SCHEMA = "mirai.magi2_preview.nf4_experts"
MAGI2_PACKED_EXPERT_SCHEMA_VERSION = 3
_MULTIFORMAT_SCHEMA_VERSION = 4
_MIXED_SCHEMA_VERSION = 5
_MANIFEST_FILE = "manifest.json"


class Magi2PackedExpertStateError(ValueError):
    """Raised when a packed MAGI-2 expert artifact is incompatible."""


def save_magi2_nf4_packed_state(
    path: str | Path,
    transformer: torch.nn.Module,
    *,
    metadata: dict[str, str] | None = None,
    fuse_gate_up: bool = False,
) -> Path:
    """Write the already-quantized expert stores without dense reconstruction."""

    from safetensors.torch import save_file

    target = Path(path).expanduser()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"Packed MAGI-2 export directory must be empty: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    layers: dict[str, Any] = {}
    quant_format = None
    mixed = None
    mixed_lineage = None
    for layer_index, (module_name, module) in enumerate(
        iter_magi2_moe_layers(transformer)
    ):
        store = magi2_expert_store(module)
        if store is None or not store.is_fully_loaded():
            raise Magi2PackedExpertStateError(
                f"MAGI-2 layer '{module_name}' has no complete NF4 expert store."
            )
        if quant_format is None:
            quant_format = store.quant_format
            mixed = isinstance(store, Magi2MixedExpertStore)
            mixed_lineage = store.lineage if mixed else None
        if (
            quant_format != store.quant_format or mixed != isinstance(store, Magi2MixedExpertStore)
            or (fuse_gate_up and (mixed or quant_format != "nf4"))
        ):
            raise Magi2PackedExpertStateError("Incompatible gate/up or packed format across expert layers.")
        if mixed and store.lineage != mixed_lineage:
            raise Magi2PackedExpertStateError("Mixed MAGI-2 layers must share one precision plan.")
        descriptor = store.packed_state_descriptor()
        tensors: dict[str, torch.Tensor] = {}
        buffers = (
            store.fused_gate_up_buffers().items()
            if fuse_gate_up and not store.has_fused_gate_up
            else store.named_buffers(recurse=bool(mixed))
        )
        for buffer_name, value in buffers:
            tensors[buffer_name] = value.detach().cpu().contiguous()
        descriptor["buffers"] = sorted(tensors)
        descriptor["fused_gate_up"] = f"{MAGI2_FUSED_GATE_UP}_nf4" in tensors
        shard_name = f"layer_{layer_index:03d}.safetensors"
        save_file(tensors, str(target / shard_name))
        layers[module_name] = {"shard": shard_name, **descriptor}
    if not layers:
        raise Magi2PackedExpertStateError(
            "MAGI-2 packed expert export matched no routed-expert layers."
        )
    manifest = {
        "schema": MAGI2_PACKED_EXPERT_SCHEMA,
        "schema_version": (
            _MIXED_SCHEMA_VERSION if mixed else _MULTIFORMAT_SCHEMA_VERSION
            if quant_format != "nf4"
            else (MAGI2_PACKED_EXPERT_SCHEMA_VERSION if any(layer["fused_gate_up"] for layer in layers.values()) else 2)
        ),
        "quant_format": quant_format,
        "layers": layers,
        "metadata": dict(metadata or {}),
    }
    (target / _MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return target


def load_magi2_nf4_packed_state(
    path: str | Path,
    transformer: torch.nn.Module,
    *,
    blocksize: int,
    quant_format: str = "nf4",
    expected_metadata: dict[str, str] | None = None,
) -> dict[str, Magi2Nf4ExpertStore]:
    """Restore NF4 stores directly and reject topology or schema mismatches."""

    from safetensors import safe_open

    source = Path(path).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Packed MAGI-2 expert state not found: {source}")
    manifest_path = source / _MANIFEST_FILE
    if not manifest_path.is_file():
        raise Magi2PackedExpertStateError(
            "Packed MAGI-2 expert state has no versioned manifest."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    if (
        manifest.get("schema") != MAGI2_PACKED_EXPERT_SCHEMA
        or type(version) is not int
        or version not in (2, MAGI2_PACKED_EXPERT_SCHEMA_VERSION, _MULTIFORMAT_SCHEMA_VERSION, _MIXED_SCHEMA_VERSION)
    ):
        raise Magi2PackedExpertStateError(
            "Packed MAGI-2 expert state has an unsupported schema or version."
        )
    if manifest.get("quant_format") != quant_format or (
        version != _MIXED_SCHEMA_VERSION
        and (version == _MULTIFORMAT_SCHEMA_VERSION) != (quant_format != "nf4")
    ):
        raise Magi2PackedExpertStateError(
            f"Packed MAGI-2 expert state does not match requested format '{quant_format}'."
        )
    artifact_metadata = manifest.get("metadata")
    if not isinstance(artifact_metadata, dict):
        raise Magi2PackedExpertStateError(
            "Packed MAGI-2 expert state has invalid lineage metadata."
        )
    for key, value in (expected_metadata or {}).items():
        if str(artifact_metadata.get(key, "")) != str(value):
            raise Magi2PackedExpertStateError(
                f"Packed MAGI-2 expert state has incompatible {key}."
            )
    stores = install_magi2_nf4_expert_stores(transformer, blocksize=blocksize, quant_format=quant_format)
    mixed_layers = dict(iter_magi2_moe_layers(transformer)) if version == _MIXED_SCHEMA_VERSION else {}
    layer_specs = manifest.get("layers")
    if not isinstance(layer_specs, dict) or set(layer_specs) != set(stores):
        raise Magi2PackedExpertStateError(
            "Packed MAGI-2 expert layer inventory does not match the model."
        )
    if mixed_layers:
        lineage = next(iter(layer_specs.values())).get("lineage")
        if any(spec.get("lineage") != lineage for spec in layer_specs.values()):
            raise Magi2PackedExpertStateError("Mixed MAGI-2 layers declare different precision plans.")
    for module_name, store in stores.items():
        spec = layer_specs[module_name]
        fused = spec.get("fused_gate_up", False)
        if not isinstance(fused, bool) or (version != MAGI2_PACKED_EXPERT_SCHEMA_VERSION and fused):
            raise Magi2PackedExpertStateError("Packed MAGI-2 layer has an invalid gate/up layout.")
        shard_name = str(spec.get("shard", ""))
        if not shard_name or Path(shard_name).name != shard_name:
            raise Magi2PackedExpertStateError(
                f"Packed MAGI-2 layer '{module_name}' has an invalid shard path."
            )
        shard_path = source / shard_name
        if not shard_path.is_file():
            raise Magi2PackedExpertStateError(
                f"Packed MAGI-2 layer '{module_name}' is missing shard '{shard_name}'."
            )
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            expected_buffers = set(spec.get("buffers", []))
            expected_from_keys = {
                f"{key}_{suffix}"
                for key in (
                    (MAGI2_FUSED_GATE_UP, "W_down") if fused else MAGI2_ROUTED_EXPERT_TENSOR_NAMES
                )
                for suffix in (*MAGI2_NF4_GROUP_FIELDS, *MAGI2_NF4_CODE_FIELDS)
            }
            if isinstance(store, Magi2PackedExpertStore):
                expected_from_keys = store.expected_buffers()
            if not mixed_layers and expected_buffers != expected_from_keys:
                raise Magi2PackedExpertStateError(
                    f"Packed MAGI-2 layer '{module_name}' has incomplete buffers."
                )
            if set(handle.keys()) != expected_buffers:
                raise Magi2PackedExpertStateError(
                    f"Packed MAGI-2 layer '{module_name}' shard inventory is invalid."
                )
            restored_buffers: dict[str, torch.Tensor] = {}
            for buffer_name in sorted(expected_buffers):
                try:
                    value = handle.get_tensor(buffer_name)
                except KeyError as exc:
                    raise Magi2PackedExpertStateError(
                        f"Packed MAGI-2 layer '{module_name}' is missing '{buffer_name}'."
                    ) from exc
                restored_buffers[buffer_name] = value.clone().contiguous()
            if mixed_layers:
                layer = mixed_layers[module_name]
                gate = (store.num_groups, int(layer.d_head), int(layer.d_expert))
                down = (store.num_groups, int(layer.d_expert), int(layer.d_head))
                try:
                    packed = Magi2MixedExpertStore.from_packed_state(
                        descriptor=spec, buffers=restored_buffers, blocksize=blocksize,
                        base_format=quant_format, group_shapes={"W_gate": gate, "W_up": gate, "W_down": down},
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise Magi2PackedExpertStateError(f"Invalid mixed MAGI-2 layer '{module_name}': {exc}") from exc
                setattr(layer, MAGI2_NF4_STORE_ATTR, packed)
                stores[module_name] = packed
            else:
                store.restore_packed_state(descriptor=spec, buffers=restored_buffers)
    return stores


__all__ = [
    "MAGI2_PACKED_EXPERT_SCHEMA",
    "MAGI2_PACKED_EXPERT_SCHEMA_VERSION",
    "Magi2PackedExpertStateError",
    "load_magi2_nf4_packed_state",
    "save_magi2_nf4_packed_state",
]
