"""Versioned safetensors persistence for MAGI-2 NF4 routed experts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from mirai.core.models.magi2_preview.quantized_experts import (
    MAGI2_ROUTED_EXPERT_TENSOR_NAMES,
    Magi2Nf4ExpertStore,
    install_magi2_nf4_expert_stores,
    iter_magi2_moe_layers,
    magi2_expert_store,
)


MAGI2_PACKED_EXPERT_SCHEMA = "mirai.magi2_preview.nf4_experts"
MAGI2_PACKED_EXPERT_SCHEMA_VERSION = 2
_MANIFEST_FILE = "manifest.json"


class Magi2PackedExpertStateError(ValueError):
    """Raised when a packed MAGI-2 expert artifact is incompatible."""


def save_magi2_nf4_packed_state(
    path: str | Path,
    transformer: torch.nn.Module,
    *,
    metadata: dict[str, str] | None = None,
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
    for layer_index, (module_name, module) in enumerate(
        iter_magi2_moe_layers(transformer)
    ):
        store = magi2_expert_store(module)
        if store is None or not store.is_fully_loaded():
            raise Magi2PackedExpertStateError(
                f"MAGI-2 layer '{module_name}' has no complete NF4 expert store."
            )
        descriptor = store.packed_state_descriptor()
        tensors: dict[str, torch.Tensor] = {}
        for buffer_name, value in store.named_buffers(recurse=False):
            tensors[buffer_name] = value.detach().cpu().contiguous()
        shard_name = f"layer_{layer_index:03d}.safetensors"
        save_file(tensors, str(target / shard_name))
        layers[module_name] = {"shard": shard_name, **descriptor}
    if not layers:
        raise Magi2PackedExpertStateError(
            "MAGI-2 packed expert export matched no routed-expert layers."
        )
    manifest = {
        "schema": MAGI2_PACKED_EXPERT_SCHEMA,
        "schema_version": MAGI2_PACKED_EXPERT_SCHEMA_VERSION,
        "quant_format": "nf4",
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
    expected_metadata: dict[str, str] | None = None,
) -> dict[str, Magi2Nf4ExpertStore]:
    """Restore NF4 stores directly and reject topology or schema mismatches."""

    from safetensors import safe_open

    source = Path(path).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Packed MAGI-2 expert state not found: {source}")
    stores = install_magi2_nf4_expert_stores(transformer, blocksize=blocksize)
    manifest_path = source / _MANIFEST_FILE
    if not manifest_path.is_file():
        raise Magi2PackedExpertStateError(
            "Packed MAGI-2 expert state has no versioned manifest."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MAGI2_PACKED_EXPERT_SCHEMA or int(
        manifest.get("schema_version", -1)
    ) != MAGI2_PACKED_EXPERT_SCHEMA_VERSION:
        raise Magi2PackedExpertStateError(
            "Packed MAGI-2 expert state has an unsupported schema or version."
        )
    if manifest.get("quant_format") != "nf4":
        raise Magi2PackedExpertStateError(
            "Packed MAGI-2 expert state must declare quant_format='nf4'."
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
    layer_specs = manifest.get("layers")
    if not isinstance(layer_specs, dict) or set(layer_specs) != set(stores):
        raise Magi2PackedExpertStateError(
            "Packed MAGI-2 expert layer inventory does not match the model."
        )
    for module_name, store in stores.items():
        spec = layer_specs[module_name]
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
                for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES
                for suffix in (
                    "nf4",
                    "nf4_absmax",
                    "nf4_nabsmax",
                    "nf4_offset",
                    "nf4_code",
                    "nf4_ncode",
                )
            }
            if expected_buffers != expected_from_keys:
                raise Magi2PackedExpertStateError(
                    f"Packed MAGI-2 layer '{module_name}' has incomplete NF4 buffers."
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
            store.restore_packed_state(
                descriptor=spec, buffers=restored_buffers
            )
    return stores


__all__ = [
    "MAGI2_PACKED_EXPERT_SCHEMA",
    "MAGI2_PACKED_EXPERT_SCHEMA_VERSION",
    "Magi2PackedExpertStateError",
    "load_magi2_nf4_packed_state",
    "save_magi2_nf4_packed_state",
]
