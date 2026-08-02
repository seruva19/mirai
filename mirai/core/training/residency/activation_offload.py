"""Model-agnostic, bounded saved-activation CPU residency.

The layer-aware scheduler adapts the budget/defer/prefetch and view-replay
behavior described by the TorchTitan graph offload pass:
https://docs.pytorch.org/devlogs/distributed/2026-06-23-cpu-offloading/

Mirai remains an eager runtime here. Providers declare repeated layer regions;
core uses those boundaries to delay D2H copies and start H2D restores before
the corresponding layer's backward. This is not a joint-graph transformation.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import Any, Callable

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class _OffloadedTensor:
    host: "torch.Tensor"
    device: "torch.device"
    nbytes: int
    owner: "SelectiveActivationOffload"
    staged: "torch.cuda.Event | None" = None


@dataclass
class _PooledBuffer:
    """A staging buffer plus the event guarding its last outstanding read.

    ``ready`` is recorded on the consumer stream when the buffer is returned
    while its host-to-device restore may still be queued. A later writer must
    wait on it before reusing the storage.
    """

    host: "torch.Tensor"
    ready: "torch.cuda.Event | None"


@dataclass
class _LayeredStorage:
    """One host/device transfer shared by saved views of the same storage."""

    host: "torch.Tensor | None"
    device: "torch.device"
    nbytes: int
    owner: "SelectiveActivationOffload"
    layer_index: int
    source: "torch.Tensor | None"
    staged: "torch.cuda.Event | None" = None
    restored: "torch.Tensor | None" = None
    restored_event: "torch.cuda.Event | None" = None
    handle_count: int = 0
    offload_started: bool = False
    restore_started: bool = False
    reservation_live: bool = True


@dataclass
class _LayeredTensor:
    storage: _LayeredStorage
    size: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


@dataclass(frozen=True)
class ActivationOffloadPolicy:
    """Validated configuration for saved-activation host residency."""

    enabled: bool = False
    min_bytes: int = 0
    max_bytes: int = 0
    pin_memory: bool = False
    defer_layers: int = 0
    prefetch_layers: int = 0
    view_replay: bool = False

    def __post_init__(self) -> None:
        if int(self.min_bytes) < 0 or int(self.max_bytes) < 0:
            raise ValueError("Activation offload byte limits must be non-negative.")
        if int(self.defer_layers) < 0:
            raise ValueError(
                "training.activation_cpu_offload_defer_layers must be >= 0."
            )
        if int(self.prefetch_layers) < 0:
            raise ValueError(
                "training.activation_cpu_offload_prefetch_layers must be >= 0."
            )
        if self.enabled and int(self.max_bytes) <= 0:
            raise ValueError(
                "Enabled activation CPU offload requires a positive host budget."
            )

    @property
    def layer_aware(self) -> bool:
        return bool(int(self.defer_layers) or int(self.prefetch_layers))

    @classmethod
    def from_training_config(cls, training: Any) -> "ActivationOffloadPolicy":
        return cls(
            enabled=bool(getattr(training, "activation_cpu_offload", False)),
            min_bytes=(
                int(getattr(training, "activation_cpu_offload_min_mib", 8))
                * 1024**2
            ),
            max_bytes=int(
                float(getattr(training, "activation_cpu_offload_max_gib", 0.0))
                * 1024**3
            ),
            pin_memory=bool(
                getattr(training, "activation_cpu_offload_pin_memory", False)
            ),
            defer_layers=int(
                getattr(training, "activation_cpu_offload_defer_layers", 0)
            ),
            prefetch_layers=int(
                getattr(training, "activation_cpu_offload_prefetch_layers", 0)
            ),
            view_replay=bool(
                getattr(training, "activation_cpu_offload_view_replay", False)
            ),
        )


@dataclass
class ActivationOffloadRegion:
    """One provider-owned repeated layer instrumented by the scheduler."""

    name: str
    owner: Any
    attribute: str = "forward"
    _eager_callable: Callable[..., Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("Activation offload region name must not be empty.")
        target = getattr(self.owner, self.attribute, None)
        if not callable(target):
            raise TypeError(
                f"Activation offload region '{self.name}' target "
                f"{type(self.owner).__name__}.{self.attribute} is not callable."
            )
        self._eager_callable = target

    @property
    def eager_callable(self) -> Callable[..., Any]:
        return self._eager_callable

    def install(
        self,
        *,
        index: int,
        session: "ActivationOffloadSession",
    ) -> None:
        eager = self._eager_callable

        @wraps(eager)
        def scheduled(*args: Any, **kwargs: Any) -> Any:
            active = session.active_offloader
            if active is None:
                return eager(*args, **kwargs)
            with active.layer(int(index)):
                output = eager(*args, **kwargs)
            active.complete_forward_layer(int(index))
            if active.prefetch_layers:
                _register_backward_prefetch_hooks(
                    output,
                    offloader=active,
                    layer_index=int(index),
                )
            return output

        setattr(self.owner, self.attribute, scheduled)

    def restore(self) -> None:
        setattr(self.owner, self.attribute, self._eager_callable)


def _register_backward_prefetch_hooks(
    value: Any,
    *,
    offloader: "SelectiveActivationOffload",
    layer_index: int,
) -> None:
    """Attach graph-local prefetch triggers without altering output structure."""

    if torch is not None and isinstance(value, torch.Tensor):
        if value.requires_grad:
            def prefetch(gradient: "torch.Tensor") -> "torch.Tensor":
                offloader.prefetch_before_backward(int(layer_index))
                return gradient

            value.register_hook(prefetch)
        return
    if isinstance(value, dict):
        for item in value.values():
            _register_backward_prefetch_hooks(
                item,
                offloader=offloader,
                layer_index=layer_index,
            )
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _register_backward_prefetch_hooks(
                item,
                offloader=offloader,
                layer_index=layer_index,
            )


class SelectiveActivationOffload:
    """Offload only large CUDA tensors within an explicit host-memory budget.

    The saved-tensor hook owns no persistent GPU copy. The reservation is
    released when autograd unpacks the tensor.

    Pinned staging copies run on a dedicated device-to-host stream so they
    overlap with compute. Ordering is carried by CUDA events rather than a
    host-side stream synchronize: the copy stream waits for the producing
    compute stream, and ``unpack`` makes the consumer stream wait for the copy
    event before reading the staged buffer back. ``record_stream`` keeps the
    caching allocator from handing the source storage to another operator while
    the copy is still in flight.

    Staging buffers are pooled by ``(dtype, shape)``. Activation shapes repeat
    across steps, so the pool removes a page-locked host allocation per saved
    tensor. Retained free buffers are bounded by the same host budget as live
    reservations.
    """

    def __init__(
        self,
        *,
        min_bytes: int,
        max_bytes: int,
        pin_memory: bool,
        defer_layers: int = 0,
        prefetch_layers: int = 0,
        view_replay: bool = False,
        layer_count: int = 0,
    ) -> None:
        if int(min_bytes) < 0 or int(max_bytes) < 0:
            raise ValueError("Activation offload byte limits must be non-negative.")
        if int(defer_layers) < 0 or int(prefetch_layers) < 0:
            raise ValueError("Activation offload layer distances must be non-negative.")
        self.min_bytes = int(min_bytes)
        self.max_bytes = int(max_bytes)
        self.pin_memory = bool(pin_memory)
        self.defer_layers = int(defer_layers)
        self.prefetch_layers = int(prefetch_layers)
        self.view_replay = bool(view_replay)
        self.layer_count = int(layer_count)
        self._reserved_bytes = 0
        self._pooled_bytes = 0
        self._free_buffers: dict[tuple, list[_PooledBuffer]] = {}
        self._copy_streams: dict[int, "torch.cuda.Stream"] = {}
        self._layer_stack: list[int] = []
        self._pending_by_layer: dict[int, list[_LayeredStorage]] = {}
        self._storages_by_layer: dict[int, list[_LayeredStorage]] = {}
        self._shared_storages: dict[tuple[Any, ...], _LayeredStorage] = {}
        self._unique_storage_id = 0
        self._offloaded_tensors = 0
        self._prefetched_tensors = 0
        self._view_handles = 0
        self._lock = Lock()

    @property
    def reserved_bytes(self) -> int:
        with self._lock:
            return int(self._reserved_bytes)

    @property
    def pooled_bytes(self) -> int:
        with self._lock:
            return int(self._pooled_bytes)

    def _reserve(self, nbytes: int, *, reusable_key: tuple | None = None) -> bool:
        with self._lock:
            reusable_bytes = 0
            if reusable_key is not None and self._free_buffers.get(reusable_key):
                reusable_bytes = int(nbytes)
            projected = (
                self._reserved_bytes
                + self._pooled_bytes
                + int(nbytes)
                - reusable_bytes
            )
            if self.max_bytes and projected > self.max_bytes:
                return False
            self._reserved_bytes += int(nbytes)
            return True

    def _release(self, nbytes: int) -> None:
        with self._lock:
            self._reserved_bytes = max(0, self._reserved_bytes - int(nbytes))

    def _copy_stream(self, device: "torch.device") -> "torch.cuda.Stream":
        index = int(getattr(device, "index", None) or 0)
        stream = self._copy_streams.get(index)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._copy_streams[index] = stream
        return stream

    @staticmethod
    def _buffer_key(tensor: "torch.Tensor") -> tuple:
        return (
            tensor.dtype,
            tuple(int(dim) for dim in tensor.shape),
            tuple(int(value) for value in tensor.stride()),
        )

    def _acquire_host_buffer(
        self,
        tensor: "torch.Tensor",
        *,
        writer_stream: "torch.cuda.Stream | None" = None,
    ) -> "torch.Tensor":
        key = self._buffer_key(tensor)
        pooled: _PooledBuffer | None = None
        with self._lock:
            available = self._free_buffers.get(key)
            if available:
                pooled = available.pop()
                self._pooled_bytes = max(
                    0,
                    self._pooled_bytes
                    - int(pooled.host.numel() * pooled.host.element_size()),
                )
        if pooled is None:
            return torch.empty_like(tensor, device="cpu", pin_memory=self.pin_memory)
        if pooled.ready is not None and writer_stream is not None:
            # A restore from this buffer may still be queued; order the next
            # write behind it instead of blocking the host.
            writer_stream.wait_event(pooled.ready)
        return pooled.host

    def _return_host_buffer(
        self,
        host: "torch.Tensor",
        *,
        ready: "torch.cuda.Event | None" = None,
        reservation_bytes: int = 0,
    ) -> None:
        nbytes = int(host.numel() * host.element_size())
        with self._lock:
            projected = (
                self._pooled_bytes
                + nbytes
                + max(0, self._reserved_bytes - int(reservation_bytes))
            )
            if self.max_bytes and projected > self.max_bytes:
                return
            key = (
                host.dtype,
                tuple(int(dim) for dim in host.shape),
                tuple(int(value) for value in host.stride()),
            )
            self._free_buffers.setdefault(key, []).append(
                _PooledBuffer(host=host, ready=ready)
            )
            self._pooled_bytes += nbytes

    @property
    def layer_aware(self) -> bool:
        return bool(self.layer_count and (self.defer_layers or self.prefetch_layers))

    @contextmanager
    def layer(self, index: int):
        self._layer_stack.append(int(index))
        try:
            yield
        finally:
            popped = self._layer_stack.pop()
            if popped != int(index):
                raise RuntimeError("Activation offload layer stack became inconsistent.")

    def _eligible_layer(self, layer_index: int) -> bool:
        if not self.layer_aware:
            return True
        return int(layer_index) + self.defer_layers < self.layer_count

    @staticmethod
    def _flat_storage_view(tensor: "torch.Tensor") -> "torch.Tensor":
        storage_bytes = int(tensor.untyped_storage().nbytes())
        element_bytes = int(tensor.element_size())
        if storage_bytes % element_bytes:
            raise ValueError("Tensor storage is not aligned to its element size.")
        return tensor.as_strided(
            (storage_bytes // element_bytes,),
            (1,),
            0,
        ).detach()

    def _storage_source_and_metadata(
        self,
        tensor: "torch.Tensor",
    ) -> tuple["torch.Tensor", tuple[int, ...], tuple[int, ...], int, tuple[Any, ...]]:
        if self.view_replay and getattr(tensor, "_base", None) is not None:
            source = self._flat_storage_view(tensor)
            storage = tensor.untyped_storage()
            key = (
                int(tensor.device.index or 0),
                int(storage.data_ptr()),
                int(storage.nbytes()),
                tensor.dtype,
            )
            self._view_handles += 1
            return (
                source,
                tuple(int(value) for value in tensor.shape),
                tuple(int(value) for value in tensor.stride()),
                int(tensor.storage_offset()),
                key,
            )
        self._unique_storage_id += 1
        source = tensor.detach()
        return (
            source,
            tuple(int(value) for value in source.shape),
            tuple(int(value) for value in source.stride()),
            0,
            ("tensor", self._unique_storage_id),
        )

    def _start_offload(self, storage: _LayeredStorage) -> None:
        if storage.offload_started or storage.source is None:
            return
        source = storage.source
        host: "torch.Tensor | None" = None
        try:
            if not self.pin_memory:
                host = self._acquire_host_buffer(source)
                host.copy_(source)
                storage.host = host
                storage.source = None
                storage.offload_started = True
            else:
                copy_stream = self._copy_stream(storage.device)
                copy_stream.wait_stream(torch.cuda.current_stream(storage.device))
                host = self._acquire_host_buffer(
                    source,
                    writer_stream=copy_stream,
                )
                with torch.cuda.stream(copy_stream):
                    host.copy_(source, non_blocking=True)
                source.record_stream(copy_stream)
                staged = torch.cuda.Event()
                staged.record(copy_stream)
                storage.host = host
                storage.staged = staged
                storage.source = None
                storage.offload_started = True
            self._offloaded_tensors += 1
        except Exception:
            if host is not None:
                self._return_host_buffer(
                    host,
                    reservation_bytes=storage.nbytes,
                )
            self._release_storage_reservation(storage)
            storage.host = None
            storage.source = source
            raise

    def _release_storage_reservation(self, storage: _LayeredStorage) -> None:
        if storage.reservation_live:
            self._release(storage.nbytes)
            storage.reservation_live = False

    def _retire_storage(self, storage: _LayeredStorage) -> None:
        self._release_storage_reservation(storage)
        for key, candidate in tuple(self._shared_storages.items()):
            if candidate is storage:
                del self._shared_storages[key]
        for mapping in (self._pending_by_layer, self._storages_by_layer):
            for layer_index, candidates in tuple(mapping.items()):
                remaining = [
                    candidate for candidate in candidates if candidate is not storage
                ]
                if remaining:
                    mapping[layer_index] = remaining
                else:
                    del mapping[layer_index]

    def _pack_layered(self, tensor: "torch.Tensor"):
        if self.layer_aware:
            if not self._layer_stack:
                return tensor
            layer_index = int(self._layer_stack[-1])
            if not self._eligible_layer(layer_index):
                return tensor
        else:
            layer_index = int(self._layer_stack[-1]) if self._layer_stack else 0
        if tensor.is_leaf:
            return tensor
        try:
            source, size, stride, offset, key = self._storage_source_and_metadata(
                tensor
            )
        except (RuntimeError, ValueError):
            return tensor
        nbytes = int(source.numel() * source.element_size())
        if nbytes < self.min_bytes:
            return tensor
        storage = self._shared_storages.get(key)
        if storage is None:
            if not self._reserve(
                nbytes,
                reusable_key=self._buffer_key(source),
            ):
                return tensor
            storage = _LayeredStorage(
                host=None,
                device=tensor.device,
                nbytes=nbytes,
                owner=self,
                layer_index=layer_index,
                source=source,
            )
            self._shared_storages[key] = storage
            self._storages_by_layer.setdefault(layer_index, []).append(storage)
            if self.defer_layers:
                self._pending_by_layer.setdefault(layer_index, []).append(storage)
            else:
                self._start_offload(storage)
        storage.handle_count += 1
        return _LayeredTensor(
            storage=storage,
            size=size,
            stride=stride,
            storage_offset=offset,
        )

    def complete_forward_layer(self, layer_index: int) -> None:
        if not self.defer_layers:
            return
        due = int(layer_index) - self.defer_layers
        for source_layer in sorted(
            layer for layer in self._pending_by_layer if layer <= due
        ):
            pending = self._pending_by_layer.pop(source_layer, ())
            for storage in pending:
                self._start_offload(storage)

    def _start_restore(self, storage: _LayeredStorage) -> "torch.Tensor":
        if storage.source is not None:
            self._release_storage_reservation(storage)
            return storage.source
        if storage.restored is not None:
            return storage.restored
        if storage.host is None:
            raise RuntimeError("Offloaded activation has no host storage.")
        host = storage.host
        if storage.staged is None:
            restored = torch.empty_strided(
                host.size(),
                host.stride(),
                dtype=host.dtype,
                device=storage.device,
            )
            restored.copy_(host)
            storage.restored = restored
            storage.restore_started = True
            self._return_host_buffer(
                host,
                reservation_bytes=storage.nbytes,
            )
            storage.host = None
            self._release_storage_reservation(storage)
            return restored

        copy_stream = self._copy_stream(storage.device)
        copy_stream.wait_event(storage.staged)
        with torch.cuda.stream(copy_stream):
            restored = torch.empty_strided(
                host.size(),
                host.stride(),
                dtype=host.dtype,
                device=storage.device,
            )
            restored.copy_(host, non_blocking=True)
        restored.record_stream(copy_stream)
        ready = torch.cuda.Event()
        ready.record(copy_stream)
        storage.restored = restored
        storage.restored_event = ready
        storage.restore_started = True
        self._return_host_buffer(
            host,
            ready=ready,
            reservation_bytes=storage.nbytes,
        )
        storage.host = None
        self._release_storage_reservation(storage)
        return restored

    def prefetch_before_backward(self, layer_index: int) -> None:
        if self.prefetch_layers <= 0:
            return
        target = int(layer_index) - self.prefetch_layers
        if target < 0:
            return
        for storage in self._storages_by_layer.get(target, ()):
            if storage.offload_started and not storage.restore_started:
                self._start_restore(storage)
                self._prefetched_tensors += 1

    @staticmethod
    def _unpack_layered(value: _LayeredTensor) -> "torch.Tensor":
        storage = value.storage
        owner = storage.owner
        restored = owner._start_restore(storage)
        if storage.restored_event is not None:
            consumer = torch.cuda.current_stream(storage.device)
            consumer.wait_event(storage.restored_event)
            restored.record_stream(consumer)
        output = restored.as_strided(
            value.size,
            value.stride,
            value.storage_offset,
        )
        storage.handle_count = max(0, storage.handle_count - 1)
        if storage.handle_count == 0:
            storage.restored = None
            storage.source = None
            owner._retire_storage(storage)
        return output

    def pack(self, tensor: "torch.Tensor"):
        if tensor.device.type != "cuda":
            return tensor
        if self.layer_aware or self.view_replay:
            return self._pack_layered(tensor)
        nbytes = int(tensor.numel() * tensor.element_size())
        if nbytes < self.min_bytes or not self._reserve(
            nbytes,
            reusable_key=self._buffer_key(tensor),
        ):
            return tensor
        host = None
        try:
            if not self.pin_memory:
                # Pageable staging: the copy is synchronous, so no stream
                # ordering is observable and no event is needed.
                host = self._acquire_host_buffer(tensor)
                host.copy_(tensor.detach())
                return _OffloadedTensor(host, tensor.device, nbytes, self)
            copy_stream = self._copy_stream(tensor.device)
            copy_stream.wait_stream(torch.cuda.current_stream(tensor.device))
            host = self._acquire_host_buffer(tensor, writer_stream=copy_stream)
            with torch.cuda.stream(copy_stream):
                host.copy_(tensor.detach(), non_blocking=True)
            # Autograd may drop the source as soon as pack returns; the caching
            # allocator must not reissue that storage while the copy stream
            # still reads it.
            tensor.record_stream(copy_stream)
            staged = torch.cuda.Event()
            staged.record(copy_stream)
            return _OffloadedTensor(host, tensor.device, nbytes, self, staged)
        except Exception:
            if host is not None:
                self._return_host_buffer(host, reservation_bytes=nbytes)
            self._release(nbytes)
            raise

    @staticmethod
    def unpack(value):
        if isinstance(value, _LayeredTensor):
            return SelectiveActivationOffload._unpack_layered(value)
        if not isinstance(value, _OffloadedTensor):
            return value
        owner = value.owner
        ready: "torch.cuda.Event | None" = None
        try:
            if value.staged is None:
                return value.host.to(device=value.device)
            consumer = torch.cuda.current_stream(value.device)
            consumer.wait_event(value.staged)
            restored = value.host.to(device=value.device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(consumer)
            return restored
        finally:
            owner._return_host_buffer(
                value.host,
                ready=ready,
                reservation_bytes=value.nbytes,
            )
            owner._release(value.nbytes)

    def context(self):
        graph = getattr(getattr(torch, "autograd", None), "graph", None)
        hooks = getattr(graph, "saved_tensors_hooks", None)
        if hooks is None:
            raise RuntimeError(
                "Selective activation offload requires "
                "torch.autograd.graph.saved_tensors_hooks."
            )
        return hooks(self.pack, self.unpack)

    def diagnostics(self) -> dict[str, int]:
        return {
            "reserved_bytes": self.reserved_bytes,
            "pooled_bytes": self.pooled_bytes,
            "offloaded_tensors": int(self._offloaded_tensors),
            "prefetched_tensors": int(self._prefetched_tensors),
            "view_handles": int(self._view_handles),
        }


@dataclass
class ActivationOffloadSession:
    """Installed provider boundaries plus per-forward offload owners."""

    pipeline: Any
    policy: ActivationOffloadPolicy
    enabled: bool = False
    regions: list[ActivationOffloadRegion] = field(default_factory=list)
    _active_var: ContextVar[SelectiveActivationOffload | None] = field(
        default_factory=lambda: ContextVar("mirai_activation_offload", default=None),
        init=False,
        repr=False,
    )
    _last: SelectiveActivationOffload | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def active_offloader(self) -> SelectiveActivationOffload | None:
        return self._active_var.get()

    @contextmanager
    def context(self):
        if not self.enabled or torch is None or not torch.is_grad_enabled():
            yield
            return
        offloader = SelectiveActivationOffload(
            min_bytes=self.policy.min_bytes,
            max_bytes=self.policy.max_bytes,
            pin_memory=self.policy.pin_memory,
            defer_layers=self.policy.defer_layers,
            prefetch_layers=self.policy.prefetch_layers,
            view_replay=self.policy.view_replay,
            layer_count=len(self.regions),
        )
        self._last = offloader
        token = self._active_var.set(offloader)
        try:
            with offloader.context():
                yield
        finally:
            self._active_var.reset(token)

    def diagnostics(self) -> dict[str, Any]:
        values = self._last.diagnostics() if self._last is not None else {}
        return {
            "enabled": bool(self.enabled),
            "defer_layers": int(self.policy.defer_layers),
            "prefetch_layers": int(self.policy.prefetch_layers),
            "view_replay": bool(self.policy.view_replay),
            "regions": [region.name for region in self.regions],
            **values,
        }

    def close(self) -> None:
        for region in reversed(self.regions):
            region.restore()
        self.regions.clear()
        self.enabled = False


def prepare_activation_offload(
    *,
    pipeline: Any,
    policy: ActivationOffloadPolicy,
) -> ActivationOffloadSession:
    """Install provider-declared layer boundaries when scheduling needs them."""

    session = ActivationOffloadSession(pipeline=pipeline, policy=policy)
    if not policy.enabled:
        return session
    if policy.layer_aware:
        getter = getattr(pipeline, "get_activation_offload_regions", None)
        if not callable(getter):
            raise ValueError(
                f"{type(pipeline).__name__} does not expose activation offload regions."
            )
        regions = list(getter())
        if not regions or not all(
            isinstance(region, ActivationOffloadRegion) for region in regions
        ):
            raise TypeError(
                "get_activation_offload_regions() must return "
                "ActivationOffloadRegion objects."
            )
        names = [region.name for region in regions]
        if len(names) != len(set(names)):
            raise ValueError("Activation offload region names must be unique.")
        try:
            for index, region in enumerate(regions):
                region.install(index=index, session=session)
                session.regions.append(region)
        except Exception:
            session.close()
            raise
    session.enabled = True
    return session


def activation_offload_context(
    *,
    enabled: bool,
    min_bytes: int = 0,
    max_bytes: int = 0,
    pin_memory: bool = False,
):
    if not enabled or torch is None or not torch.is_grad_enabled():
        return nullcontext()
    return SelectiveActivationOffload(
        min_bytes=int(min_bytes),
        max_bytes=int(max_bytes),
        pin_memory=bool(pin_memory),
    ).context()


__all__ = [
    "ActivationOffloadPolicy",
    "ActivationOffloadRegion",
    "ActivationOffloadSession",
    "SelectiveActivationOffload",
    "activation_offload_context",
    "prepare_activation_offload",
]
