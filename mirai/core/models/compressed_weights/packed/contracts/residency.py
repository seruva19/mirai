"""Behavioral contract for packed-state residency and streaming."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from mirai.config.schema import MemoryConfig
from mirai.core.models.compressed_weights.packed.packed_residency import (
    PackedStateResidencyPolicy,
    PreloadedPackedTensorMapping,
)
from mirai.core.models.compressed_weights.packed.packed_preload_ring import (
    PrefetchedPreloadedPackedTensorMapping,
)
from mirai.core.models.compressed_weights.packed.packed_stream_prefetch import PackedCudaTransfer
from mirai.core.training.residency.memory_safety import configure_memory_safety

_GIB = 1024**3


class PackedStateResidencyPolicyTests(unittest.TestCase):
    def test_nf4_stack_writes_into_supplied_dense_slot_view(self) -> None:
        import sys
        from types import SimpleNamespace

        from mirai.core.models.compressed_weights.quantization import quant

        slot = torch.empty(64, dtype=torch.uint8)
        out = slot[:16].view(torch.float32).view(2, 1, 2)

        def dequantize(_packed, state, *, out, **_kwargs):
            self.assertEqual(out.data_ptr(), slot.data_ptr())
            self.assertEqual(out.shape, state.shape)
            out.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            return out

        bnb = SimpleNamespace(
            QuantState=SimpleNamespace,
            dequantize_blockwise=mock.Mock(return_value=torch.ones(2)),
            dequantize_4bit=dequantize,
        )
        with mock.patch.dict(sys.modules, {"bitsandbytes.functional": bnb}), mock.patch.object(
            quant, "_import_bnb_functional", return_value=bnb,
        ):
            result = quant.nf4_dequantize_stack(
                packed=torch.zeros(2, dtype=torch.uint8), absmax_q=torch.zeros(2), nested_absmax=torch.ones(2),
                offsets=torch.zeros(2), code=torch.empty(0), nested_code=torch.empty(0),
                num_selected=2, out_features=1, in_features=2, dtype=torch.float32,
                meta=quant._Nf4Meta(64, 256, torch.float32, torch.float32), out=out,
            )
        self.assertEqual(result.data_ptr(), out.data_ptr())
        torch.testing.assert_close(result, torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]))

    def test_raw_cuda_arena_subdivides_one_allocation_into_aligned_slots(self) -> None:
        from contextlib import nullcontext
        import sys
        from types import SimpleNamespace

        from mirai.core.models.compressed_weights.packed.fixed_cuda_arena import FixedCudaArena

        cuda = SimpleNamespace(
            Device=mock.Mock(side_effect=lambda _: nullcontext()),
            ExternalStream=mock.Mock(side_effect=lambda _: nullcontext()),
            Memory=mock.Mock(return_value=SimpleNamespace(ptr=4099)),
            MemoryPointer=mock.Mock(),
        )
        cupy = SimpleNamespace(cuda=cuda, uint8=object(), ndarray=mock.Mock())
        stream = SimpleNamespace(device=torch.device("cuda:0"), cuda_stream=12)
        storage = torch.zeros(8192, dtype=torch.uint8)
        with mock.patch.dict(sys.modules, {"cupy": cupy}), mock.patch(
            "torch.cuda.current_stream", return_value=stream,
        ), mock.patch("torch.from_dlpack", return_value=storage):
            arena = FixedCudaArena(device="cuda:0", slots=2, slot_bytes=4096, alignment=4096)
        cuda.Memory.assert_called_once_with(8192 + 4095)
        cuda.MemoryPointer.assert_called_once_with(cuda.Memory.return_value, 4093)
        cuda.ExternalStream.assert_called_once_with(12)
        self.assertEqual(arena.slot(1).data_ptr() - arena.slot(0).data_ptr(), 4096)
        self.assertIs(arena.slot(0).untyped_storage(), arena.slot(1).untyped_storage())
        with self.assertRaises(IndexError):
            arena.slot(2)
        producer, consumer = mock.Mock(), mock.Mock()
        first_done, second_done, third_done = mock.Mock(), mock.Mock(), mock.Mock()
        with mock.patch("torch.cuda.Event", side_effect=[first_done, second_done, third_done]):
            with self.assertRaisesRegex(ValueError, "consumer failed"):
                with arena.lease(producer) as first:
                    with arena.lease(producer) as second:
                        self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    raise ValueError("consumer failed")
            with arena.lease(consumer) as reused:
                self.assertEqual(reused.data_ptr(), second.data_ptr())
                consumer.wait_event.assert_called_once_with(first_done)
        first_done.record.assert_called_once_with(producer)
        second_done.record.assert_called_once_with(producer)
        third_done.record.assert_called_once_with(consumer)

    def test_raw_staged_upload_reuses_slots_without_aliasing_results(self) -> None:
        from contextlib import nullcontext
        from itertools import cycle
        from types import SimpleNamespace

        from mirai.core.models.compressed_weights.packed.packed_stream import StreamingPackedTensorMapping

        reader = StreamingPackedTensorMapping.__new__(StreamingPackedTensorMapping)
        reader._closed = True
        reader._staging_arena = None
        reader._staging_arena_device = None
        reader._staging_pool = mock.Mock()

        def copy_into(sources, targets, *, stream):
            targets[0].copy_(sources[0])

        reader._staging_pool.copy_into.side_effect = copy_into
        storage = torch.zeros(2, 8, dtype=torch.uint8)
        arena = mock.Mock()
        arena.slot.side_effect = lambda index: storage[index]
        positions = cycle(range(2))
        arena.lease.side_effect = lambda _: nullcontext(arena.slot(next(positions)))
        device = torch.device("cpu")
        stream = SimpleNamespace(device=device)
        module = "mirai.core.models.compressed_weights.packed"
        with mock.patch(f"{module}.fixed_cuda_arena.FixedCudaArena", return_value=arena) as allocate, mock.patch(
            f"{module}.packed_stream._MAX_RANGE_BYTES", 8,
        ):
            first = reader._copy_through_cuda_arena([torch.arange(5, dtype=torch.int32)], device, None, stream)[0]
            second = reader._copy_through_cuda_arena([torch.tensor([23], dtype=torch.int32)], device, torch.float64, stream)[0]
        allocate.assert_called_once()
        self.assertEqual(arena.slot.call_args_list, [mock.call(0), mock.call(1), mock.call(0), mock.call(1)])
        torch.testing.assert_close(first, torch.arange(5, dtype=torch.int32))
        torch.testing.assert_close(second, torch.tensor([23], dtype=torch.float64))

    def test_raw_borrowed_fields_alias_aligned_slot_during_consumer(self) -> None:
        from contextlib import contextmanager
        from types import SimpleNamespace

        from mirai.core.models.compressed_weights.packed.packed_stream import StreamingPackedTensorMapping

        reader = StreamingPackedTensorMapping.__new__(StreamingPackedTensorMapping)
        reader._closed = False
        reader._raw_staging = True
        reader._backend = "staged"
        reader._prefetch = None
        reader._staging_lock = reader._stats_lock = mock.MagicMock()
        reader._h2d_ops = reader._h2d_bytes = 0
        reader._region = mock.Mock(return_value=SimpleNamespace(shape=(2,), nbytes=8))
        first = torch.tensor([7, 5], dtype=torch.int32)
        second = torch.tensor([2.5, 1.5], dtype=torch.float32)
        reader.get_slices_to_device = mock.Mock(side_effect=[first, second])
        slot = torch.empty(32, dtype=torch.uint8)
        released = mock.Mock()

        @contextmanager
        def lease(_stream):
            try:
                yield slot
            finally:
                released()

        reader._cuda_staging_arena = mock.Mock(return_value=SimpleNamespace(lease=lease))
        with mock.patch("torch.cuda.current_stream"):
            with reader.borrow_packed_fields(("weights", "scales"), (1, 0), device="cuda:0") as fields:
                released.assert_not_called()
                torch.testing.assert_close(fields[0], first)
                torch.testing.assert_close(fields[1], second)
                self.assertEqual(fields[0].data_ptr(), slot.data_ptr())
                self.assertEqual(fields[1].data_ptr(), slot.data_ptr() + 16)
        released.assert_called_once()
        self.assertEqual(reader._h2d_bytes, 16)
        self.assertEqual(reader.get_slices_to_device.call_args_list, [
            mock.call("weights", (1, 0), device="cpu"), mock.call("scales", (1, 0), device="cpu"),
        ])
        reader._closed = True

    def test_uring_fixed_window_reuses_one_buffer_across_chunks(self) -> None:
        import ctypes
        import threading
        from types import SimpleNamespace

        from mirai.core.models.compressed_weights.packed.packed_stream_io import IoUringPackedShardReader

        reader = IoUringPackedShardReader.__new__(IoUringPackedShardReader)
        reader._FIXED_BYTES = 4096
        reader._fixed = (ctypes.c_ubyte * 4096)()
        reader._ring_lock = threading.Lock()
        reader._handle, reader.fd = 11, 7
        reader.alignment = SimpleNamespace(file_size=16384, alignment_bytes=4096)
        reader._bridge = mock.Mock()
        source = (bytes(range(251)) * 66)[:16384]
        calls = []

        def read_fixed(handle, fd, length, offset):
            calls.append((handle, fd, length, offset))
            ctypes.memmove(reader._fixed, source[offset:offset + length], length)
            return length

        reader._bridge.mirai_uring_read_fixed.side_effect = read_fixed
        output = bytearray(8195)
        self.assertEqual(reader.readinto(memoryview(output), 4093), 12288)
        self.assertEqual(output, source[4093:12288])
        self.assertEqual(calls, [(11, 7, 4096, start) for start in (0, 4096, 8192)])

    def test_explicit_disk_mode_preserves_lazy_source_identity(self) -> None:
        source = {"experts": torch.arange(12).reshape(3, 4)}

        with self.assertLogs(
            "mirai.core.models.compressed_weights.packed.packed_residency", level="WARNING"
        ):
            mapping, info = PackedStateResidencyPolicy("off").materialize(source)

        self.assertIs(mapping, source)
        self.assertEqual(info["requested"], "off")
        self.assertEqual(info["effective"], "off")
        self.assertEqual(info["bytes"], 0)

    def test_ram_mode_materializes_exact_tensor_and_slice_views(self) -> None:
        tensor = torch.arange(12).reshape(3, 4)
        source = {"experts": tensor}

        mapping, info = PackedStateResidencyPolicy("ram").materialize(source)

        self.assertIsInstance(mapping, PreloadedPackedTensorMapping)
        self.assertEqual(info["effective"], "ram")
        self.assertEqual(info["bytes"], tensor.numel() * tensor.element_size())
        self.assertTrue(torch.equal(mapping["experts"], tensor))
        self.assertTrue(torch.equal(mapping.get_slice("experts", 1), tensor[1]))

    def test_invalid_mode_fails_at_policy_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "packed_state_preload"):
            PackedStateResidencyPolicy("automatic")

    def test_stream_cache_requires_explicit_disk_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "packed_state_preload='off'"):
            PackedStateResidencyPolicy("ram", stream_cache_gib=1.0)
        with self.assertRaisesRegex(ValueError, "capacity"):
            PackedStateResidencyPolicy("off", stream_cache_gib=-1.0)

    def test_gds_backend_is_explicit_disk_only_and_bypasses_host_cache(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires packed_state_preload='off'"):
            PackedStateResidencyPolicy("ram", stream_backend="gds")
        with self.assertRaisesRegex(ValueError, "0 GiB"):
            PackedStateResidencyPolicy(
                "off",
                stream_cache_gib=1.0,
                stream_backend="gds",
            )

    def test_gds_async_read_awaits_its_stream_and_trims_aligned_bytes(self) -> None:
        import contextlib
        import threading

        from mirai.core.models.compressed_weights.packed.packed_stream_gds import KvikioGdsReader, _GdsSlot

        reader = KvikioGdsReader.__new__(KvikioGdsReader)
        reader._closed = False
        reader._slots = [_GdsSlot()]
        reader._slot_bytes = 8192
        reader._slot_device = None
        reader._slot_condition = threading.Condition()
        reader._raw_arena = False
        reader._arena = None
        reader._compat_context = None
        reader._stats_lock = threading.Lock()
        for counter in ("_read_ops", "_read_bytes", "_requested_bytes", "_overread_bytes",
                        "_read_seconds", "_trimmed_ops", "_async_ops"):
            setattr(reader, counter, 0)
        handle = mock.Mock()
        reader._files = {"shard": handle}
        reader._file_sizes = {"shard": 8192}
        stream = mock.Mock(cuda_stream=9)
        empty = torch.empty
        pending = mock.Mock()
        fills = []
        pointers = []

        def submit(buffer, *, raw_stream, size, file_offset):
            self.assertEqual((raw_stream, size, file_offset), (9, 8192, 0))
            self.assertEqual(buffer.data_ptr() % 4096, 0)
            pointers.append(buffer.data_ptr())
            fills.append(len(fills) * 7)

            def complete():
                buffer.copy_((torch.arange(size, dtype=torch.int64) + fills[-1]).to(torch.uint8))
                return size

            pending.check_bytes_done.side_effect = complete
            return pending

        handle.raw_read_async.side_effect = submit
        ready = mock.Mock()
        with mock.patch("torch.cuda.Stream", return_value=stream), mock.patch(
            "torch.cuda.stream", side_effect=lambda _: contextlib.nullcontext()
        ), mock.patch("torch.empty", side_effect=lambda size, **_: empty(size, dtype=torch.uint8)) as allocate, mock.patch(
            "torch.cuda.Event", return_value=ready,
        ):
            result = reader.read_device("shard", offset=4093, nbytes=12, device="cuda:0")
            second = reader.read_device("shard", offset=4093, nbytes=12, device="cuda:0")
            allocate.assert_called_once()
        torch.testing.assert_close(result.tensor, torch.arange(4093, 4105).to(torch.uint8))
        torch.testing.assert_close(second.tensor, torch.arange(4100, 4112).to(torch.uint8))
        self.assertEqual(pointers[0], pointers[1])
        self.assertEqual(pending.check_bytes_done.call_count, 2)
        self.assertEqual(ready.record.call_args_list, [mock.call(stream), mock.call(stream)])
        stream.synchronize.assert_not_called()
        handle.read.assert_not_called()
        self.assertEqual(reader._async_ops, 2)
        reader.close()
        stream.synchronize.assert_called_once_with()
        handle.close.assert_called_once_with()

    def test_prefetch_ring_supports_disk_and_preloaded_ram(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 16"):
            PackedStateResidencyPolicy("off", stream_prefetch_depth=17)
        policy = PackedStateResidencyPolicy("off", stream_prefetch_depth=4)
        self.assertEqual(policy.stream_prefetch_depth, 4)
        mapping, info = PackedStateResidencyPolicy(
            "ram", stream_prefetch_depth=2
        ).materialize({"experts": torch.arange(12).reshape(3, 4)})
        self.assertIsInstance(mapping, PrefetchedPreloadedPackedTensorMapping)
        self.assertEqual(info["prefetch_depth"], 2)
        try:
            self.assertTrue(
                mapping.prefetch_slices_to_device(
                    "experts", [2, 0], device="cpu"
                )
            )
            torch.testing.assert_close(
                mapping.get_slices_to_device("experts", [2, 0], device="cpu"),
                torch.tensor([[8, 9, 10, 11], [0, 1, 2, 3]]),
            )
        finally:
            mapping.close()

    def test_cuda_prefetch_waits_for_its_request_only_at_consumption(self) -> None:
        mapping, _ = PackedStateResidencyPolicy(
            "ram", stream_prefetch_depth=2
        ).materialize({"experts": torch.arange(12).reshape(3, 4)})
        ready = mock.sentinel.ready
        tensor = mock.Mock(device=torch.device("cuda:0"))
        stream = mock.Mock()
        request = dict(device="cpu")
        try:
            with mock.patch.object(
                mapping, "_get_slices_to_device_now", return_value=PackedCudaTransfer(tensor, ready)
            ) as load, mock.patch("torch.cuda.current_stream", return_value=stream):
                self.assertTrue(mapping.prefetch_slices_to_device("experts", [2, 0], **request))
                stream.wait_event.assert_not_called()
                tensor.record_stream.assert_not_called()
                self.assertIs(mapping.get_slices_to_device("experts", [2, 0], **request), tensor)
                load.assert_called_once()
                stream.wait_event.assert_called_once_with(ready)
                tensor.record_stream.assert_called_once_with(stream)
        finally:
            mapping.close()


class PackedStatePinBudgetTests(unittest.TestCase):
    """Packed-state pinning must honor memory.max_pinned_host_gib."""

    def tearDown(self) -> None:
        configure_memory_safety(MemoryConfig())

    def _source(self, n: int, nbytes_each: int) -> dict[str, torch.Tensor]:
        return {
            f"t{i}": torch.zeros(nbytes_each, dtype=torch.uint8) for i in range(n)
        }

    def test_pinning_stops_at_the_budget_cap(self) -> None:
        # The small cap admits the first tensor and stops at the next pair
        # boundary, leaving the remaining tensors pageable.
        configure_memory_safety(
            MemoryConfig(minimum_system_memory_gib=0.0, max_pinned_host_gib=1e-6)
        )
        source = self._source(4, 512)
        total = sum(t.numel() for t in source.values())
        # pin_memory needs no real CUDA here: stub it to identity so the budget
        # accounting (not the allocator) is what we exercise, CPU-only.
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch.object(
            torch.Tensor, "pin_memory", lambda self: self
        ):
            mapping, info = PackedStateResidencyPolicy("pinned").materialize(source)
        self.assertEqual(info["effective"], "pinned")
        self.assertEqual(info["reason"], "budget_capped")
        self.assertGreater(info["pinned_bytes"], 0)
        self.assertLess(info["pinned_bytes"], total)
        # Numerics unchanged: every tensor still equals the source, pinned or not.
        for key, tensor in source.items():
            self.assertTrue(torch.equal(mapping[key], tensor))

    def test_generous_cap_pins_everything(self) -> None:
        configure_memory_safety(
            MemoryConfig(minimum_system_memory_gib=0.0, max_pinned_host_gib=8.0)
        )
        source = self._source(4, 512)
        total = sum(t.numel() for t in source.values())
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch.object(
            torch.Tensor, "pin_memory", lambda self: self
        ):
            _, info = PackedStateResidencyPolicy("pinned").materialize(source)
        self.assertEqual(info["effective"], "pinned")
        self.assertEqual(info["reason"], "ok")
        self.assertEqual(info["pinned_bytes"], total)


if __name__ == "__main__":
    unittest.main()
