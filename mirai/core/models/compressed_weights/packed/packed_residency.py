"""Packed expert residency policy across disk, pageable RAM, and pinned RAM."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirai.core.moe.runtime.specs import normalize_packed_state_preload
from mirai.core.moe.runtime.specs import normalize_packed_stream_backend
from mirai.core.models.compressed_weights.packed.packed_stream import (
    StreamingPackedTensorMapping,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _safetensors_dtype_nbytes(dtype: str) -> int:
    return _SAFETENSORS_DTYPE_BYTES.get(str(dtype).upper(), 4)


class LazyPackedTensorMapping(Mapping[str, Any]):
    """Disk-backed safetensors mapping; each access opens only its owning shard."""

    def __init__(self, path: Path) -> None:
        try:
            from safetensors import safe_open
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required to load packed states.") from exc
        self._safe_open = safe_open
        self._root = path.parent
        if path.name.endswith(".index.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_map = payload.get("weight_map", {})
            if not isinstance(raw_map, dict) or not raw_map:
                raise ValueError("Packed compressed_weights index has no weight_map.")
            self._weight_map = {str(key): str(value) for key, value in raw_map.items()}
        else:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                self._weight_map = {str(key): path.name for key in handle.keys()}

    def __getitem__(self, key: str) -> Any:
        shard_name = self._weight_map[str(key)]
        with self._safe_open(str(self._root / shard_name), framework="pt", device="cpu") as handle:
            return handle.get_tensor(str(key))

    def get_slice(self, key: str, index: int) -> Any:
        shard_name = self._weight_map[str(key)]
        with self._safe_open(str(self._root / shard_name), framework="pt", device="cpu") as handle:
            return handle.get_slice(str(key))[int(index)]

    def get_range(self, key: str, start: int, end: int) -> Any:
        """Read a contiguous leading-axis range without loading the tensor."""

        shard_name = self._weight_map[str(key)]
        with self._safe_open(str(self._root / shard_name), framework="pt", device="cpu") as handle:
            return handle.get_slice(str(key))[int(start) : int(end)]

    def tensor_shape_dtype(self, key: str) -> tuple[tuple[int, ...], str]:
        """Return safetensors metadata without materializing payload bytes."""

        shard_name = self._weight_map[str(key)]
        with self._safe_open(str(self._root / shard_name), framework="pt", device="cpu") as handle:
            handle_slice = handle.get_slice(str(key))
            return (
                tuple(int(value) for value in handle_slice.get_shape()),
                str(handle_slice.get_dtype()),
            )

    def tensor_nbytes(self, key: str) -> int:
        shard_name = self._weight_map[str(key)]
        with self._safe_open(str(self._root / shard_name), framework="pt", device="cpu") as handle:
            handle_slice = handle.get_slice(str(key))
            shape = handle_slice.get_shape()
            dtype = handle_slice.get_dtype()
        count = 1
        for dim in shape:
            count *= int(dim)
        return int(count) * _safetensors_dtype_nbytes(dtype)

    def __iter__(self) -> Iterator[str]:
        return iter(self._weight_map)

    def __len__(self) -> int:
        return len(self._weight_map)

    def __contains__(self, key: object) -> bool:
        return str(key) in self._weight_map


class PreloadedPackedTensorMapping(Mapping[str, Any]):
    """RAM cache over a lazy mapping, exposing whole tensors for batched gather."""

    def __init__(self, base: Mapping[str, Any], cache: Mapping[str, Any]) -> None:
        self._base = base
        self._cache = dict(cache)

    def __getitem__(self, key: str) -> Any:
        cached = self._cache.get(str(key))
        return cached if cached is not None else self._base[str(key)]

    def get_slice(self, key: str, index: int) -> Any:
        cached = self._cache.get(str(key))
        if cached is not None:
            return cached[int(index)]
        get_slice = getattr(self._base, "get_slice", None)
        if callable(get_slice):
            return get_slice(str(key), index)
        return self._base[str(key)][int(index)]

    def get_range(self, key: str, start: int, end: int) -> Any:
        cached = self._cache.get(str(key))
        if cached is not None:
            return cached[int(start) : int(end)]
        get_range = getattr(self._base, "get_range", None)
        if callable(get_range):
            return get_range(str(key), int(start), int(end))
        return self._base[str(key)][int(start) : int(end)]

    def tensor_shape_dtype(self, key: str) -> tuple[tuple[int, ...], str]:
        cached = self._cache.get(str(key))
        if cached is not None:
            return tuple(int(value) for value in cached.shape), str(cached.dtype)
        metadata = getattr(self._base, "tensor_shape_dtype", None)
        if callable(metadata):
            return metadata(str(key))
        value = self._base[str(key)]
        return tuple(int(item) for item in value.shape), str(value.dtype)

    def cached_tensor(self, key: str) -> Any | None:
        return self._cache.get(str(key))

    def __iter__(self) -> Iterator[str]:
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)

    def __contains__(self, key: object) -> bool:
        return key in self._base


@dataclass(frozen=True)
class PackedStateResidencyPolicy:
    """Resolve one explicit packed-state residency mode without hidden fallback."""

    mode: str = "pinned"
    stream_cache_gib: float = 0.0
    stream_backend: str = "staged"
    stream_prefetch_depth: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_packed_state_preload(self.mode))
        backend = normalize_packed_stream_backend(self.stream_backend)
        cache_gib = float(self.stream_cache_gib)
        prefetch_depth = int(self.stream_prefetch_depth)
        if cache_gib < 0.0:
            raise ValueError("packed stream cache capacity must be >= 0 GiB.")
        if cache_gib > 0.0 and self.mode != "off":
            raise ValueError("packed stream cache requires packed_state_preload='off'.")
        if backend != "staged" and self.mode != "off":
            raise ValueError("packed stream backend 'gds' requires packed_state_preload='off'.")
        if backend == "gds" and cache_gib > 0.0:
            raise ValueError("packed stream cache capacity must be 0 GiB with backend 'gds'.")
        if prefetch_depth < 0 or prefetch_depth > 16:
            raise ValueError("packed stream prefetch depth must be between 0 and 16.")
        object.__setattr__(self, "stream_cache_gib", cache_gib)
        object.__setattr__(self, "stream_backend", backend)
        object.__setattr__(self, "stream_prefetch_depth", prefetch_depth)

    def open(self, path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
        source: Mapping[str, Any]
        if self.mode == "off":
            source = StreamingPackedTensorMapping(
                path,
                cache_capacity_bytes=int(self.stream_cache_gib * (1024**3)),
                backend=self.stream_backend,
                prefetch_depth=self.stream_prefetch_depth,
            )
        else:
            source = LazyPackedTensorMapping(path)
        return self.materialize(source)

    def materialize(self, base: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, Any]]:
        info: dict[str, Any] = {
            "requested": self.mode,
            "effective": "off",
            "reason": "disabled",
            "bytes": 0,
            "tensors": 0,
        }
        if self.mode == "off":
            logger.warning(
                "compressed_weights packed_state_preload='off': DISK STREAMING mode engaged "
                "-- packed expert weights are read from disk on every forward/backward "
                "(sustained read I/O can cause contention and thermal throttling). "
                "Use expert_weight_access='chunked_dequant' with a measured "
                "expert_dequant_chunk_size to reduce storage operations; "
                "active_dequant minimizes H2D bytes but issues singleton reads. "
                "This is opt-in; the default is RAM-resident 'pinned'."
            )
            return base, info

        keys = list(base)
        nbytes_fn = getattr(base, "tensor_nbytes", None)
        try:
            if callable(nbytes_fn):
                total = int(sum(int(nbytes_fn(key)) for key in keys))
            else:
                total = int(sum(int(base[key].numel() * base[key].element_size()) for key in keys))
        except Exception as exc:
            raise RuntimeError(
                f"compressed_weights packed_state_preload={self.mode!r} could not measure "
                f"the packed state size ({type(exc).__name__}: {exc}). RAM preload "
                "cannot be verified. Free the source, or set "
                "memory.packed_state_preload='off' to stream from disk explicitly "
                "(opt-in; sustained read I/O)."
            ) from exc
        info["bytes"] = total
        info["tensors"] = len(keys)
        floor_bytes = _packed_preload_floor_bytes()
        try:
            import psutil

            available: int | None = int(psutil.virtual_memory().available)
        except ModuleNotFoundError:  # pragma: no cover
            available = None
        if available is not None and float(available) - float(total) < floor_bytes:
            info["effective"] = "insufficient_ram"
            info["reason"] = "insufficient_ram"
            info["available"] = available
            raise RuntimeError(
                f"compressed_weights packed_state_preload={self.mode!r} needs "
                f"{total / 1024**3:.2f} GiB of host RAM but only "
                f"{available / 1024**3:.2f} GiB is available above the "
                f"{floor_bytes / 1024**3:.2f} GiB system floor. RAM is the only "
                "primary path (it never auto-degrades to disk). Remedies: free host "
                "RAM (or lower memory.minimum_system_memory_gib), or set "
                "memory.packed_state_preload='off' to stream packed experts from disk "
                "explicitly -- opt-in, with sustained storage traffic."
            )

        want_pin = self.mode == "pinned" and torch is not None and torch.cuda.is_available()
        # Apply the configured host-pin ceiling independently to packed preload.
        # Over-budget tensors stay in pageable RAM and use the staged copy path.
        pin_cap_bytes = _packed_pin_budget_bytes() if want_pin else None
        pin_failed = False
        pin_capped = False
        pinned_bytes = 0
        cache: dict[str, Any] = {}
        for key in keys:
            tensor = base[key]
            if hasattr(tensor, "contiguous"):
                tensor = tensor.contiguous()
            if want_pin and not pin_failed and not pin_capped:
                nbytes = int(tensor.numel()) * int(tensor.element_size())
                if pin_cap_bytes is not None and pinned_bytes + nbytes > pin_cap_bytes:
                    pin_capped = True  # ceiling hit; this and later tensors stay pageable
                else:
                    try:
                        tensor = tensor.pin_memory()
                        pinned_bytes += nbytes
                    except (RuntimeError, MemoryError):  # pragma: no cover
                        pin_failed = True
            cache[str(key)] = tensor
        if self.mode == "pinned" and pinned_bytes > 0:
            effective = "pinned"
            reason = "budget_capped" if pin_capped else "ok"
        elif self.mode == "pinned":
            effective = "ram"
            reason = "pin_unavailable"
        else:
            effective = "ram"
            reason = "ok"
        info["effective"] = effective
        info["reason"] = reason
        info["pinned_bytes"] = pinned_bytes
        logger.info(
            "compressed_weights packed_state_preload=%s -> %s (%d tensors, %.2f GiB CPU RAM, "
            "%.2f GiB pinned).",
            self.mode,
            effective,
            len(cache),
            total / 1024**3,
            pinned_bytes / 1024**3,
        )
        mapping: Mapping[str, Any] = PreloadedPackedTensorMapping(base, cache)
        if self.stream_prefetch_depth > 0:
            from .packed_preload_ring import PrefetchedPreloadedPackedTensorMapping

            pin_budget = _packed_pin_budget_bytes()
            if pin_budget is not None:
                pin_budget = max(0, int(pin_budget) - int(pinned_bytes))
            mapping = PrefetchedPreloadedPackedTensorMapping(
                mapping,
                prefetch_depth=self.stream_prefetch_depth,
                pin_budget_bytes=pin_budget,
            )
            info["prefetch_depth"] = self.stream_prefetch_depth
            info["prefetch_pin_budget_bytes"] = pin_budget
        return mapping, info


def _packed_preload_floor_bytes() -> float:
    try:
        from mirai.core.training.residency.memory_safety import current_memory_safety_policy

        return float(current_memory_safety_policy().minimum_system_memory_gib) * (1024.0**3)
    except Exception:  # pragma: no cover
        return 0.0


def _packed_pin_budget_bytes() -> int | None:
    """Absolute ceiling (bytes) on page-locked packed-state RAM, or None.

    Shares memory.max_pinned_host_gib with the block-swap host-pin budget so
    each subsystem caps its pinning at the same value; over-budget tensors stay
    pageable (bit-identical, only slower H2D). Needs no psutil -- it is a fixed
    absolute cap -- so it stays fail-safe even when free-RAM cannot be read.
    """

    try:
        from mirai.core.training.residency.memory_safety import current_memory_safety_policy

        cap_gib = float(current_memory_safety_policy().max_pinned_host_gib)
    except Exception:  # pragma: no cover
        return None
    return int(cap_gib * (1024**3)) if cap_gib > 0.0 else None


def materialize_packed_tensors(
    base: Mapping[str, Any], mode: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    return PackedStateResidencyPolicy(mode).materialize(base)
