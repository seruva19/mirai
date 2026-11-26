"""Versioned safetensors persistence for MAGI-2 NF4 routed experts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from mirai.core.models.compressed_weights.packed.packed_storage_alignment import (
    GDS_STORAGE_ALIGNMENT_BYTES, read_safetensors_storage_alignment, save_safetensors_with_storage_alignment,
)
from mirai.core.models.magi2_preview.basis_experts import Magi2MixtureBasisExpertStore, Magi2SharedBasisExpertStore
from mirai.core.models.magi2_preview.mixed_experts import Magi2MixedExpertStore
from mirai.core.models.magi2_preview.quantized_experts import (
    MAGI2_NF4_STORE_ATTR,
    MAGI2_ROUTED_EXPERT_TENSOR_NAMES,
    MAGI2_FUSED_GATE_UP,
    MAGI2_NF4_GROUP_FIELDS,
    MAGI2_NF4_CODE_FIELDS,
    MAGI2_PACKED_EXPERT_FORMATS,
    Magi2Nf4ExpertStore,
    Magi2PackedExpertStore,
    PackedExpertRangeView,
    _new_expert_store,
    iter_magi2_moe_layers,
    magi2_expert_store,
)


MAGI2_PACKED_EXPERT_SCHEMA = "mirai.magi2_preview.nf4_experts"
MAGI2_PACKED_EXPERT_SCHEMA_VERSION = 3
_MULTIFORMAT_SCHEMA_VERSION = 4
_MIXED_SCHEMA_VERSION = 5
_ROTATED_MIXED_SCHEMA_VERSION = 6
_SHARED_BASIS_SCHEMA_VERSION = 7
_MIXTURE_BASIS_SCHEMA_VERSION = 8
_MANIFEST_FILE = "manifest.json"
_PRUNING_DIGEST_KEY = "head_expert_pruning_digest"
_CONSOLIDATION_DIGEST_KEY = "head_expert_consolidation_digest"


class Magi2PackedExpertStateError(ValueError):
    """Raised when a packed MAGI-2 expert artifact is incompatible."""


def _validate_pruning_topology(transformer: torch.nn.Module, digest: bytes | None) -> None:
    if digest is not None and (not isinstance(digest, bytes) or len(digest) != 32):
        raise Magi2PackedExpertStateError("Packed MAGI-2 expert pruning digest must contain 32 bytes.")
    pruned = any(getattr(layer, "_mirai_experts_pruned", False) for _, layer in iter_magi2_moe_layers(transformer))
    if pruned != (digest is not None):
        raise Magi2PackedExpertStateError("Packed MAGI-2 expert pruning lineage requires a matching topology.")


def _head_layout(layer: torch.nn.Module) -> dict[str, int]:
    return {"num_heads": layer.local_num_heads, "experts_per_head": layer.num_experts}


def _consolidation_digest_hex(transformer: torch.nn.Module, digest: bytes | None) -> str | None:
    aliased = any(
        getattr(layer, "_mirai_expert_aliases", None) is not None for _, layer in iter_magi2_moe_layers(transformer)
    )
    if aliased != (digest is not None) or (
        digest is not None and (not isinstance(digest, bytes) or len(digest) != 32)
    ):
        raise Magi2PackedExpertStateError("Packed MAGI-2 consolidation requires a matching plan digest.")
    return None if digest is None else digest.hex()


def save_magi2_nf4_packed_state(
    path: str | Path,
    transformer: torch.nn.Module,
    *,
    metadata: dict[str, str] | None = None,
    fuse_gate_up: bool = False,
    pruning_digest: bytes | None = None,
    consolidation_digest: bytes | None = None,
    storage_alignment_bytes: int = 0,
) -> Path:
    """Write the already-quantized expert stores without dense reconstruction."""

    from safetensors.torch import save_file

    if type(storage_alignment_bytes) is not int or storage_alignment_bytes not in {0, GDS_STORAGE_ALIGNMENT_BYTES}:
        raise Magi2PackedExpertStateError("Packed MAGI-2 expert export requires zero or 4096-byte alignment.")
    consolidation_hex = _consolidation_digest_hex(transformer, consolidation_digest)
    _validate_pruning_topology(transformer, pruning_digest)
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
    rotation_group_size = 0
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
            rotation_group_size = store.rotation_group_size if mixed else 0
        if (
            quant_format != store.quant_format or mixed != isinstance(store, Magi2MixedExpertStore)
            or (fuse_gate_up and (mixed or quant_format != "nf4"))
        ):
            raise Magi2PackedExpertStateError("Incompatible gate/up or packed format across expert layers.")
        if mixed and (store.lineage != mixed_lineage or store.rotation_group_size != rotation_group_size):
            raise Magi2PackedExpertStateError("Mixed MAGI-2 layers must share one precision plan.")
        descriptor = store.packed_state_descriptor()
        tensors: dict[str, torch.Tensor] = {}
        reader = getattr(store, "_packed_range_reader", None)
        if reader is not None and fuse_gate_up and not store.has_fused_gate_up:
            raise Magi2PackedExpertStateError("Fuse gate/up before enabling MAGI-2 packed range reads.")
        buffers = (
            ((name, reader[name]) for name in sorted(reader))
            if reader is not None else
            store.fused_gate_up_buffers().items()
            if fuse_gate_up and not store.has_fused_gate_up
            else store.named_buffers(recurse=bool(mixed) or isinstance(store, Magi2MixtureBasisExpertStore))
        )
        for buffer_name, value in buffers:
            tensors[buffer_name] = value.detach().cpu().contiguous()
        descriptor["buffers"] = sorted(tensors)
        descriptor["fused_gate_up"] = f"{MAGI2_FUSED_GATE_UP}_nf4" in tensors
        shard_name = f"layer_{layer_index:03d}.safetensors"
        save_safetensors_with_storage_alignment(
            save_file, tensors, target / shard_name, metadata=None, alignment_bytes=storage_alignment_bytes,
        )
        layers[module_name] = {"shard": shard_name, "head_layout": _head_layout(module), **descriptor}
        aliases = getattr(module, "_mirai_expert_aliases", None)
        if aliases is not None:
            layers[module_name]["logical_to_physical"] = aliases
    if not layers:
        raise Magi2PackedExpertStateError(
            "MAGI-2 packed expert export matched no routed-expert layers."
        )
    manifest = {
        "schema": MAGI2_PACKED_EXPERT_SCHEMA,
        "schema_version": (
            _SHARED_BASIS_SCHEMA_VERSION if quant_format == "shared_basis" else
            _MIXTURE_BASIS_SCHEMA_VERSION if quant_format == "mixture_basis" else
            _ROTATED_MIXED_SCHEMA_VERSION if rotation_group_size else
            _MIXED_SCHEMA_VERSION if mixed else _MULTIFORMAT_SCHEMA_VERSION
            if quant_format != "nf4"
            else (MAGI2_PACKED_EXPERT_SCHEMA_VERSION if any(layer["fused_gate_up"] for layer in layers.values()) else 2)
        ),
        "quant_format": quant_format,
        "layers": layers,
        "metadata": dict(metadata or {}),
    }
    if pruning_digest is not None:
        manifest[_PRUNING_DIGEST_KEY] = pruning_digest.hex()
    if consolidation_hex is not None:
        manifest[_CONSOLIDATION_DIGEST_KEY] = consolidation_hex
    if storage_alignment_bytes:
        manifest["storage_alignment_bytes"] = storage_alignment_bytes
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
    expected_pruning_digest: bytes | None = None,
    expected_consolidation_digest: bytes | None = None,
    expected_shared_basis_rank: int | None = None,
    expected_mixture_basis: tuple[int, int, str] | None = None,
    retain_mapped_buffers: bool = False,
    range_read_backend: str | None = None,
    stream_cache_bytes: int = 0,
    stream_prefetch_depth: int = 0,
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
    alignment = manifest.get("storage_alignment_bytes", 0)
    if type(alignment) is not int or alignment not in {0, GDS_STORAGE_ALIGNMENT_BYTES}:
        raise Magi2PackedExpertStateError("Packed MAGI-2 expert state has invalid shard alignment.")
    if range_read_backend in {"gds", "direct_io"} and alignment != GDS_STORAGE_ALIGNMENT_BYTES:
        raise Magi2PackedExpertStateError(f"MAGI-2 {range_read_backend} requires a 4096-byte-aligned packed export.")
    if (
        manifest.get("schema") != MAGI2_PACKED_EXPERT_SCHEMA
        or type(version) is not int
        or version not in (2, MAGI2_PACKED_EXPERT_SCHEMA_VERSION, _MULTIFORMAT_SCHEMA_VERSION, _MIXED_SCHEMA_VERSION, _ROTATED_MIXED_SCHEMA_VERSION, _SHARED_BASIS_SCHEMA_VERSION, _MIXTURE_BASIS_SCHEMA_VERSION)
    ):
        raise Magi2PackedExpertStateError(
            "Packed MAGI-2 expert state has an unsupported schema or version."
        )
    _validate_pruning_topology(transformer, expected_pruning_digest)
    expected_hex = expected_pruning_digest.hex() if expected_pruning_digest is not None else None
    if manifest.get(_PRUNING_DIGEST_KEY) != expected_hex:
        raise Magi2PackedExpertStateError("Packed MAGI-2 expert state has incompatible pruning lineage.")
    consolidation_hex = _consolidation_digest_hex(transformer, expected_consolidation_digest)
    if manifest.get(_CONSOLIDATION_DIGEST_KEY) != consolidation_hex:
        raise Magi2PackedExpertStateError("Packed MAGI-2 expert state has incompatible consolidation lineage.")
    shared_basis = version == _SHARED_BASIS_SCHEMA_VERSION
    mixture_basis = version == _MIXTURE_BASIS_SCHEMA_VERSION
    if (
        manifest.get("quant_format") != quant_format
        or shared_basis != (quant_format == "shared_basis") or mixture_basis != (quant_format == "mixture_basis")
    ) or (
        version not in {_MIXED_SCHEMA_VERSION, _ROTATED_MIXED_SCHEMA_VERSION, _SHARED_BASIS_SCHEMA_VERSION, _MIXTURE_BASIS_SCHEMA_VERSION}
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
    layers = dict(iter_magi2_moe_layers(transformer))
    mixed = version in {_MIXED_SCHEMA_VERSION, _ROTATED_MIXED_SCHEMA_VERSION}
    layer_specs = manifest.get("layers")
    if (
        not isinstance(layer_specs, dict) or not layers or set(layer_specs) != set(layers)
        or any(not isinstance(spec, dict) for spec in layer_specs.values())
    ):
        raise Magi2PackedExpertStateError(
            "Packed MAGI-2 expert layer inventory does not match the model."
        )
    if mixed:
        lineage = next(iter(layer_specs.values())).get("lineage")
        if any(spec.get("lineage") != lineage for spec in layer_specs.values()):
            raise Magi2PackedExpertStateError("Mixed MAGI-2 layers declare different precision plans.")
        rotations = [spec.get("hadamard_input_group_size", 0) for spec in layer_specs.values()]
        if (
            any(type(size) is not int or size != rotations[0] for size in rotations)
            or bool(rotations[0]) != (version == _ROTATED_MIXED_SCHEMA_VERSION)
        ):
            raise Magi2PackedExpertStateError("Mixed MAGI-2 artifact has inconsistent rotation metadata.")
    stores = {}
    tensor_aliases = {}
    for module_name, layer in layers.items():
        spec = layer_specs[module_name]
        if "head_layout" in spec and spec["head_layout"] != _head_layout(layer):
            raise Magi2PackedExpertStateError(f"Packed MAGI-2 layer '{module_name}' has incompatible head topology.")
        aliases = getattr(layer, "_mirai_expert_aliases", None)
        if spec.get("logical_to_physical") != (list(aliases) if aliases is not None else None):
            raise Magi2PackedExpertStateError(f"Packed MAGI-2 layer '{module_name}' has incompatible expert aliases.")
        groups = int(getattr(layer, "_mirai_physical_expert_count", layer.local_flatten_num_experts))
        gate = (groups, int(layer.d_head), int(layer.d_expert))
        down = (groups, int(layer.d_expert), int(layer.d_head))
        shapes = {"W_gate": gate, "W_up": gate, "W_down": down}
        if spec.get("expert_shapes") != {key: list(shape) for key, shape in shapes.items()}:
            raise Magi2PackedExpertStateError(f"Packed MAGI-2 layer '{module_name}' has incompatible projection shapes.")
        if mixture_basis:
            down_format = spec.get("down_format")
            if down_format not in MAGI2_PACKED_EXPERT_FORMATS:
                raise Magi2PackedExpertStateError("Packed MAGI-2 mixture-basis down format is invalid.")
            if expected_mixture_basis is not None and down_format != expected_mixture_basis[2]:
                raise Magi2PackedExpertStateError("Packed MAGI-2 mixture-basis down format does not match the config.")
            store = Magi2MixtureBasisExpertStore(num_groups=groups, blocksize=blocksize, down_format=down_format)
        elif shared_basis:
            store = Magi2SharedBasisExpertStore(num_groups=groups)
        else:
            store = _new_expert_store(num_groups=groups, blocksize=blocksize, quant_format=quant_format)
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
        if alignment:
            read_safetensors_storage_alignment(shard_path, required_alignment=alignment)
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            expected_buffers = set(spec.get("buffers", []))
            expected_from_keys = {
                f"{key}_{suffix}"
                for key in (
                    (MAGI2_FUSED_GATE_UP, "W_down") if fused else MAGI2_ROUTED_EXPERT_TENSOR_NAMES
                )
                for suffix in (*MAGI2_NF4_GROUP_FIELDS, *MAGI2_NF4_CODE_FIELDS)
            }
            if isinstance(store, (Magi2PackedExpertStore, Magi2SharedBasisExpertStore)):
                expected_from_keys = store.expected_buffers()
            if not mixed and not mixture_basis and expected_buffers != expected_from_keys:
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
                restored_buffers[buffer_name] = value if retain_mapped_buffers or range_read_backend is not None else value.clone().contiguous()
            if mixed:
                try:
                    packed = Magi2MixedExpertStore.from_packed_state(
                        descriptor=spec, buffers=restored_buffers, blocksize=blocksize,
                        base_format=quant_format, group_shapes=shapes,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise Magi2PackedExpertStateError(f"Invalid mixed MAGI-2 layer '{module_name}': {exc}") from exc
                store = packed
            else:
                try:
                    if mixture_basis:
                        rank, count = expected_mixture_basis[:2] if expected_mixture_basis is not None else (None, None)
                        store.restore_mixture_state(spec, restored_buffers, expected_rank=rank, expected_count=count)
                    elif shared_basis:
                        store.restore_packed_state(
                            descriptor=spec, buffers=restored_buffers, expected_rank=expected_shared_basis_rank,
                        )
                    else:
                        store.restore_packed_state(descriptor=spec, buffers=restored_buffers)
                except (KeyError, TypeError, ValueError) as exc:
                    raise Magi2PackedExpertStateError(f"Invalid packed MAGI-2 layer '{module_name}': {exc}") from exc
        if range_read_backend is not None:
            tensor_aliases.update({f"{module_name}:{key}": (shard_name, key) for key in expected_buffers})
        stores[module_name] = store
    if range_read_backend is not None:
        from mirai.core.models.compressed_weights.packed.packed_stream import StreamingPackedTensorMapping

        reader = StreamingPackedTensorMapping(
            manifest_path, tensor_aliases=tensor_aliases, cache_capacity_bytes=stream_cache_bytes,
            prefetch_depth=stream_prefetch_depth, backend=range_read_backend, cache_expert_slices=True,
        )
        try:
            for module_name, store in stores.items():
                store.bind_packed_range_reader(PackedExpertRangeView(reader, f"{module_name}:"))
        except Exception:
            reader.close()
            raise
    for module_name, layer in layers.items():
        for key in MAGI2_ROUTED_EXPERT_TENSOR_NAMES:
            if key in layer._parameters:
                del layer._parameters[key]
        setattr(layer, MAGI2_NF4_STORE_ATTR, stores[module_name])
    return stores


__all__ = [
    "MAGI2_PACKED_EXPERT_SCHEMA",
    "MAGI2_PACKED_EXPERT_SCHEMA_VERSION",
    "Magi2PackedExpertStateError",
    "load_magi2_nf4_packed_state",
    "save_magi2_nf4_packed_state",
]
