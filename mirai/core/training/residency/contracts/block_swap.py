"""Behavioral contract for block residency transitions."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from mirai.config.schema import (
    MemoryConfig,
    ModelConfig,
    ModelParams,
    StrategyConfig,
    TrainingConfig,
    TrainingSection,
)
from mirai.core.builtins import register_builtin_components
from mirai.core.training.residency.block_swap import BlockSwapManager
from mirai.core.training.residency.selective_checkpoint import (
    make_selective_checkpoint_context_fn,
)
from mirai.core.training.residency.activation_offload import (
    ActivationOffloadPolicy,
    ActivationOffloadRegion,
    SelectiveActivationOffload,
    _LayeredTensor,
    prepare_activation_offload,
)
from mirai.core.training.residency.activation_compression import (
    LowRankActivationCompression,
)
from mirai.core.training.trainer import Trainer

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class BlockSwapManagerTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch not installed")
    def test_selective_checkpoint_preserves_output_and_gradient(self) -> None:
        import torch.utils.checkpoint as checkpoint

        weight = torch.randn(4, 4)
        reference_x = torch.randn(3, 4, requires_grad=True)
        actual_x = reference_x.detach().clone().requires_grad_(True)

        def operation(x):
            return torch.sin(x @ weight).square()

        reference = operation(reference_x)
        actual = checkpoint.checkpoint(
            operation,
            actual_x,
            use_reentrant=False,
            context_fn=make_selective_checkpoint_context_fn(),
        )
        torch.testing.assert_close(actual, reference)
        reference.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(actual_x.grad, reference_x.grad)

    def test_sync_mode_events(self) -> None:
        mgr = BlockSwapManager(total_blocks=4, blocks_to_swap=2, mode="sync")
        mgr.reset_step()
        for idx in range(4):
            mgr.before_block(idx)
            mgr.after_block(idx)
        kinds = [e.kind for e in mgr.events]
        self.assertEqual(kinds, ["swap_in", "swap_out", "swap_in", "swap_out"])

    def test_async_mode_emits_prefetch_events(self) -> None:
        mgr = BlockSwapManager(total_blocks=4, blocks_to_swap=2, mode="async")
        mgr.reset_step()
        for idx in range(4):
            mgr.before_block(idx)
            mgr.after_block(idx)
        kinds = [e.kind for e in mgr.events]
        self.assertIn("prefetch", kinds)
        self.assertIn("prefetch_done", kinds)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_disk_backing_preserves_block_output(self) -> None:
        class AdapterBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.frozen = torch.nn.Linear(8, 8, bias=False)
                self.frozen.weight.requires_grad_(False)
                self.adapter = torch.nn.Linear(8, 8, bias=False)

            def forward(self, inputs):
                return self.frozen(inputs) + self.adapter(inputs)

        block = AdapterBlock()
        inputs = torch.randn(3, 8)
        expected = block(inputs).square().mean()
        expected.backward()
        expected_gradient = block.adapter.weight.grad.detach().clone()
        block.adapter.weight.grad = None
        with tempfile.TemporaryDirectory() as tmp:
            manager = BlockSwapManager(
                total_blocks=1,
                blocks_to_swap=1,
                mode="sync",
                disk_offload_dir=str(Path(tmp) / "blocks"),
            )
            manager.bind([(0, block)], device=torch.device("cpu"))
            manager.before_block(0)
            actual = block(inputs).square().mean()
            actual.backward()
            manager.after_block(0)
            state = manager.snapshot()
            self.assertGreater(state["disk_backing_bytes"], 0)
            self.assertTrue((Path(tmp) / "blocks" / "block_0000.safetensors").is_file())
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(block.adapter.weight.grad, expected_gradient)

    @unittest.skipIf(
        torch is None or not torch.cuda.is_available(), "CUDA not available"
    )
    def test_async_prefetch_records_consumer_stream(self) -> None:
        device = torch.device("cuda")
        mgr = BlockSwapManager(total_blocks=2, blocks_to_swap=2, mode="async")
        blocks = [torch.nn.Linear(8, 8) for _ in range(2)]
        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad_(False)
        mgr.bind(list(enumerate(blocks)), device=device)
        recorded = []
        original = torch.Tensor.record_stream

        def spy(tensor, stream) -> None:
            recorded.append(stream)
            original(tensor, stream)

        torch.Tensor.record_stream = spy
        try:
            mgr.before_block(0)  # prefetches block 1 on the transfer stream
            mgr.after_block(0)
            mgr.before_block(1)  # consumes the prefetch: wait_event + record_stream
        finally:
            torch.Tensor.record_stream = original
        current = torch.cuda.current_stream(device)
        self.assertTrue(any(stream == current for stream in recorded))
        mgr.finish_backward()

    @unittest.skipIf(
        torch is None or not torch.cuda.is_available(), "CUDA not available"
    )
    def test_async_full_swap_matches_the_resident_reference(self) -> None:
        """Every block swapped, checkpointed, driven by module hooks.

        With no block held resident the first block is swapped, and its inputs
        carry no gradient, so its full backward hook releases it before its
        forward is recomputed. Async prefetch must still hand device weights to
        that recompute and reproduce the resident gradients exactly.
        """

        import torch.utils.checkpoint as checkpoint

        class Layer(torch.nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.frozen = torch.nn.Parameter(torch.randn(width, width) * 0.05)
                self.frozen.requires_grad_(False)
                self.adapter = torch.nn.Linear(width, width, bias=False)

            def forward(self, value):
                return torch.tanh(self.adapter(torch.matmul(value, self.frozen)))

        class Stack(torch.nn.Module):
            def __init__(self, width: int, depth: int) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [Layer(width) for _ in range(depth)]
                )

            def forward(self, value):
                for layer in self.layers:
                    value = checkpoint.checkpoint(
                        lambda item, owner=layer: owner(item),
                        value,
                        use_reentrant=False,
                    )
                return value

        device = torch.device("cuda")
        width, depth = 64, 4
        torch.manual_seed(5)
        inputs = [torch.randn(16, width, device=device) for _ in range(2)]

        torch.manual_seed(3)
        reference = Stack(width, depth).to(device)
        for value in inputs:
            reference(value).square().mean().backward()
        expected = [
            parameter.grad.detach().clone()
            for parameter in reference.parameters()
            if parameter.requires_grad
        ]

        torch.manual_seed(3)
        actual = Stack(width, depth)
        for parameter in actual.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(device)
        manager = BlockSwapManager(
            total_blocks=depth, blocks_to_swap=depth, mode="async"
        )
        units = list(enumerate(actual.layers))
        manager.bind(units, device=device)
        for index, layer in units:
            layer.register_forward_pre_hook(
                lambda _module, _args, idx=index: manager.before_block(idx)
            )
            layer.register_forward_hook(
                lambda _module, _args, output, idx=index: (
                    manager.after_block(idx),
                    output,
                )[1]
            )
        for value in inputs:
            actual(value).square().mean().backward()
            manager.finish_backward()
        observed = [
            parameter.grad.detach().clone()
            for parameter in actual.parameters()
            if parameter.requires_grad
        ]
        for observed_grad, expected_grad in zip(observed, expected, strict=True):
            torch.testing.assert_close(observed_grad, expected_grad)


class _RecordingStream:
    """Consumer stream stub that records the ordering primitives issued on it."""

    def __init__(self, log: list[str]) -> None:
        self._log = log

    def wait_event(self, event: object) -> None:
        self._log.append(f"wait_event:{event}")


class _StubUnit:
    """Residency unit stub with observable host/device transitions."""

    def __init__(self, index: int, log: list[str]) -> None:
        self.index = index
        self._log = log
        self._resident = False
        self.bytes = 0
        self.backing_store_bytes = 0

    @property
    def device(self) -> str:
        return "cuda" if self._resident else "cpu"

    def load(self, device: object, *, stream: object | None = None) -> object | None:
        self._resident = True
        if stream is None:
            self._log.append(f"sync_load:{self.index}")
            return None
        self._log.append(f"async_load:{self.index}")
        return f"event{self.index}"

    def record_stream(self, stream: object) -> None:
        self._log.append(f"record_stream:{self.index}")

    def offload(self) -> None:
        self._resident = False
        self._log.append(f"offload:{self.index}")


@unittest.skipIf(torch is None, "torch not installed")
class AsyncBlockSwapOrderingTests(unittest.TestCase):
    """Async prefetch must prove residency, never assume it.

    A pending transfer event is evidence that a copy was *issued*. It stops
    being evidence of residency as soon as the block is offloaded, which can
    happen before the block's forward: a module whose inputs do not require
    gradients has its full backward hook invoked when gradients with respect to
    its outputs exist, i.e. before its checkpointed forward is recomputed.
    """

    def _manager(self, log: list[str]) -> tuple[object, dict[int, _StubUnit]]:
        manager = BlockSwapManager(total_blocks=3, blocks_to_swap=3, mode="async")
        # A device that reports type "cuda" drives the async branch without
        # creating a CUDA context.
        manager._device = types.SimpleNamespace(type="cuda")
        manager._transfer_stream = object()
        units = {idx: _StubUnit(idx, log) for idx in range(3)}
        manager._units = dict(units)
        return manager, units

    def test_consumer_stream_waits_on_the_prefetch_before_the_block_runs(self) -> None:
        log: list[str] = []
        manager, _ = self._manager(log)
        with mock.patch(
            "torch.cuda.current_stream", return_value=_RecordingStream(log)
        ):
            manager.before_block(0)
            manager.after_block(0)
            manager.before_block(1)
        self.assertEqual(
            log,
            [
                "sync_load:0",
                "async_load:1",
                "offload:0",
                "wait_event:event1",
                "record_stream:1",
                "async_load:2",
            ],
        )

    def test_offload_between_prefetch_and_forward_forces_a_synchronous_load(
        self,
    ) -> None:
        log: list[str] = []
        manager, units = self._manager(log)
        with mock.patch(
            "torch.cuda.current_stream", return_value=_RecordingStream(log)
        ):
            manager.before_block(0)
            self.assertEqual(units[1].device, "cuda")
            # Stands in for the full backward hook of a block whose inputs carry
            # no gradient: it releases the block before its forward is reached.
            manager._units[1].offload()
            manager._discard_prefetch_event(1)
            manager.before_block(1)
        self.assertEqual(units[1].device, "cuda")
        self.assertIn("sync_load:1", log)
        self.assertNotIn("wait_event:event1", log)

    def test_a_released_block_never_leaves_a_consumable_transfer_record(self) -> None:
        log: list[str] = []
        manager, _ = self._manager(log)
        with mock.patch(
            "torch.cuda.current_stream", return_value=_RecordingStream(log)
        ):
            manager.before_block(0)
            self.assertIn(1, manager._prefetch_events)
            manager.after_block(1)
        self.assertNotIn(1, manager._prefetch_events)

    def test_residency_is_restored_even_if_a_stale_record_survives(self) -> None:
        log: list[str] = []
        manager, units = self._manager(log)
        with mock.patch(
            "torch.cuda.current_stream", return_value=_RecordingStream(log)
        ):
            manager.before_block(0)
            # Release the block without clearing its record, the exact state the
            # production failure reached.
            units[1].offload()
            manager.before_block(1)
        self.assertEqual(units[1].device, "cuda")
        self.assertIn("sync_load:1", log)


@unittest.skipIf(torch is None, "torch not installed")
class BlockSwapPipelineIntegrationTests(unittest.TestCase):
    def test_trainer_pipeline_records_block_swap_events(self) -> None:
        register_builtin_components()
        cfg = TrainingConfig(
            preset=None,
            model=ModelConfig(
                type="sparse_moe_test",
                path="./models/sparse_moe_test",
                dtype="bf16",
                attention_backend="auto",
                params=ModelParams(variant="tiny-video", flow_shift=3.0, vae_chunk_size=16),
            ),
            strategy=StrategyConfig(type="text_to_video", params={}),
            training=TrainingSection(
                seed=0,
                batch_size=2,
                blocks_to_swap=1,
                block_swap_mode="sync",
            ),
            memory=MemoryConfig(weight_residency_strategy="block_swap"),
        )
        trainer = Trainer(cfg)
        loss, _ = trainer.compute_loss(
            {
                "latents": torch.tensor([0.1, 0.2], dtype=torch.float32),
                "text_embeds": torch.tensor([0.3, 0.4], dtype=torch.float32),
            }
        )
        loss.backward()
        state = trainer.pipeline.get_block_swap_state()
        self.assertTrue(state["enabled"])
        self.assertTrue(any(event["kind"] == "swap_in" for event in state["events"]))


    @unittest.skipIf(torch is None, "torch not installed")
    def test_low_rank_activation_compression_reduces_saved_storage(self) -> None:
        compression = LowRankActivationCompression(rank=2, min_bytes=0, seed=3)
        left = torch.randn(16, 2)
        right = torch.randn(2, 12)
        inputs = (left @ right).requires_grad_(True)
        weight = (
            torch.randn(12, 2) @ torch.randn(2, 8)
        ).requires_grad_(True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        reference_weight = weight.detach().clone().requires_grad_(True)
        reference_output = (reference_inputs @ reference_weight).square().sum()
        reference_output.backward()
        with compression.context():
            output = (inputs @ weight).square().sum()
        output.backward()
        self.assertGreater(compression.compressed_tensors, 0)
        self.assertLess(compression.stored_bytes, compression.original_bytes)
        torch.testing.assert_close(output, reference_output, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            inputs.grad, reference_inputs.grad, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            weight.grad, reference_weight.grad, rtol=1e-3, atol=1e-3
        )


@unittest.skipIf(torch is None, "torch not installed")
class SelectiveActivationOffloadStagingTests(unittest.TestCase):
    """Staging-buffer reuse and budget invariants for saved-activation offload.

    The pool exists so a page-locked host allocation is not paid per saved
    tensor. These CPU-only probes cover acquisition, reuse, the retained-buffer
    bound, and reservation release; device-stream ordering is covered by the
    accelerator-gated round trip below.
    """

    def _offload(self, *, max_bytes: int = 0) -> SelectiveActivationOffload:
        return SelectiveActivationOffload(
            min_bytes=0, max_bytes=max_bytes, pin_memory=False
        )

    def test_returned_buffer_is_reused_for_the_same_shape(self) -> None:
        offload = self._offload()
        probe = torch.zeros(4, 8)
        first = offload._acquire_host_buffer(probe)
        offload._return_host_buffer(first)
        self.assertGreater(offload.pooled_bytes, 0)
        second = offload._acquire_host_buffer(probe)
        self.assertIs(second, first)
        self.assertEqual(offload.pooled_bytes, 0)

    def test_distinct_shapes_do_not_share_a_buffer(self) -> None:
        offload = self._offload()
        wide = offload._acquire_host_buffer(torch.zeros(4, 8))
        offload._return_host_buffer(wide)
        narrow = offload._acquire_host_buffer(torch.zeros(2, 2))
        self.assertIsNot(narrow, wide)

    def test_retained_buffers_respect_the_host_budget(self) -> None:
        probe = torch.zeros(4, 8)
        nbytes = int(probe.numel() * probe.element_size())
        offload = self._offload(max_bytes=nbytes)
        first = offload._acquire_host_buffer(probe)
        second = offload._acquire_host_buffer(probe)
        offload._return_host_buffer(first)
        offload._return_host_buffer(second)
        # The second return would exceed the budget and is dropped rather than
        # growing pinned host residency without bound.
        self.assertEqual(offload.pooled_bytes, nbytes)

    def test_live_reservations_and_free_pool_share_one_budget(self) -> None:
        probe = torch.zeros(4, 8)
        nbytes = int(probe.numel() * probe.element_size())
        offload = self._offload(max_bytes=nbytes)
        pooled = offload._acquire_host_buffer(probe)
        offload._return_host_buffer(pooled)
        different = torch.zeros(2, 16).transpose(0, 1)
        self.assertFalse(
            offload._reserve(
                nbytes,
                reusable_key=offload._buffer_key(different),
            )
        )
        self.assertTrue(
            offload._reserve(
                nbytes,
                reusable_key=offload._buffer_key(probe),
            )
        )

    def test_unpack_releases_the_reservation_and_pools_the_buffer(self) -> None:
        offload = self._offload()
        probe = torch.zeros(4, 8)
        nbytes = int(probe.numel() * probe.element_size())
        self.assertTrue(offload._reserve(nbytes))
        staged = offload._acquire_host_buffer(probe)
        staged.copy_(probe)
        from mirai.core.training.residency.activation_offload import _OffloadedTensor

        value = _OffloadedTensor(staged, torch.device("cpu"), nbytes, offload)
        restored = SelectiveActivationOffload.unpack(value)
        torch.testing.assert_close(restored, probe)
        self.assertEqual(offload.reserved_bytes, 0)
        self.assertEqual(offload.pooled_bytes, nbytes)

    def test_non_cuda_tensors_are_passed_through_untouched(self) -> None:
        offload = self._offload()
        probe = torch.zeros(4, 8)
        self.assertIs(offload.pack(probe), probe)
        self.assertEqual(offload.reserved_bytes, 0)

    def test_defer_launches_d2h_at_the_declared_layer_distance(self) -> None:
        offload = SelectiveActivationOffload(
            min_bytes=0,
            max_bytes=1 << 20,
            pin_memory=False,
            defer_layers=1,
            layer_count=3,
        )
        source = (
            torch.arange(16, dtype=torch.float32, requires_grad=True) * 2
        ).reshape(4, 4)
        with offload.layer(0):
            packed = offload._pack_layered(source)
        self.assertIsInstance(packed, _LayeredTensor)
        storage = packed.storage
        self.assertIsNotNone(storage.source)
        self.assertFalse(storage.offload_started)
        offload.complete_forward_layer(0)
        self.assertFalse(storage.offload_started)
        offload.complete_forward_layer(1)
        self.assertTrue(storage.offload_started)
        self.assertIsNone(storage.source)
        self.assertIsNotNone(storage.host)

    def test_prefetch_restores_the_target_before_its_backward(self) -> None:
        offload = SelectiveActivationOffload(
            min_bytes=0,
            max_bytes=1 << 20,
            pin_memory=False,
            defer_layers=1,
            prefetch_layers=1,
            layer_count=3,
        )
        source = (
            torch.arange(16, dtype=torch.float32, requires_grad=True) * 2
        ).reshape(4, 4)
        with offload.layer(0):
            packed = offload._pack_layered(source)
        offload.complete_forward_layer(1)
        self.assertFalse(packed.storage.restore_started)
        offload.prefetch_before_backward(1)
        self.assertTrue(packed.storage.restore_started)
        restored = SelectiveActivationOffload.unpack(packed)
        torch.testing.assert_close(restored, source)
        self.assertEqual(offload.reserved_bytes, 0)

    def test_view_replay_shares_base_storage_and_restores_metadata(self) -> None:
        offload = SelectiveActivationOffload(
            min_bytes=0,
            max_bytes=1 << 20,
            pin_memory=False,
            view_replay=True,
        )
        base = (
            torch.arange(36, dtype=torch.float32, requires_grad=True).reshape(6, 6)
            * 2
        )
        first = base[1:5, ::2]
        second = base.transpose(0, 1)[2:5, 1:5]
        packed_first = offload._pack_layered(first)
        packed_second = offload._pack_layered(second)
        self.assertIsInstance(packed_first, _LayeredTensor)
        self.assertIsInstance(packed_second, _LayeredTensor)
        self.assertIs(packed_first.storage, packed_second.storage)
        restored_first = SelectiveActivationOffload.unpack(packed_first)
        restored_second = SelectiveActivationOffload.unpack(packed_second)
        torch.testing.assert_close(restored_first, first)
        torch.testing.assert_close(restored_second, second)
        self.assertEqual(restored_first.stride(), first.stride())
        self.assertEqual(restored_second.stride(), second.stride())
        self.assertEqual(restored_first.storage_offset(), first.storage_offset())
        self.assertEqual(restored_second.storage_offset(), second.storage_offset())

    def test_disabled_policy_does_not_instrument_provider_regions(self) -> None:
        class Block(torch.nn.Module):
            def forward(self, value):
                return value + 1

        class Pipeline:
            def __init__(self) -> None:
                self.block = Block()

            def get_activation_offload_regions(self):
                return [ActivationOffloadRegion(name="block", owner=self.block)]

        pipeline = Pipeline()
        original = pipeline.block.forward
        session = prepare_activation_offload(
            pipeline=pipeline,
            policy=ActivationOffloadPolicy(enabled=False),
        )
        self.assertFalse(session.enabled)
        self.assertEqual(session.regions, [])
        self.assertIs(pipeline.block.forward.__func__, original.__func__)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "staged device-to-host offload requires an available accelerator",
    )
    def test_pinned_round_trip_preserves_values_without_a_host_sync(self) -> None:
        offload = SelectiveActivationOffload(
            min_bytes=0, max_bytes=1 << 30, pin_memory=True
        )
        source = torch.randn(64, 64, device="cuda")
        packed = offload.pack(source)
        self.assertIsNot(packed, source)
        restored = SelectiveActivationOffload.unpack(packed)
        torch.testing.assert_close(restored, source, rtol=0, atol=0)
        self.assertEqual(offload.reserved_bytes, 0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "staged device-to-host offload requires an available accelerator",
    )
    def test_backward_through_offloaded_activations_matches_reference(self) -> None:
        torch.manual_seed(0)
        inputs = torch.randn(32, 32, device="cuda", requires_grad=True)
        weight = torch.randn(32, 32, device="cuda", requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        reference_weight = weight.detach().clone().requires_grad_(True)
        (reference_inputs @ reference_weight).square().sum().backward()
        offload = SelectiveActivationOffload(
            min_bytes=0, max_bytes=1 << 30, pin_memory=True
        )
        with offload.context():
            output = (inputs @ weight).square().sum()
        output.backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0, atol=0)
        self.assertEqual(offload.reserved_bytes, 0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "layer-aware activation offload requires an available accelerator",
    )
    def test_layer_defer_prefetch_and_view_replay_match_cuda_reference(self) -> None:
        class Block(torch.nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(width, width, bias=False)

            def forward(self, value):
                projected = torch.sin(self.projection(value))
                return projected.reshape(value.shape)

        class Pipeline(torch.nn.Module):
            def __init__(self, width: int, depth: int) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [Block(width) for _ in range(depth)]
                )

            def forward(self, value):
                for block in self.blocks:
                    value = block(value)
                return value

            def get_activation_offload_regions(self):
                return [
                    ActivationOffloadRegion(name=f"blocks.{idx}", owner=block)
                    for idx, block in enumerate(self.blocks)
                ]

        torch.manual_seed(11)
        device = torch.device("cuda")
        reference = Pipeline(64, 4).to(device)
        actual = Pipeline(64, 4).to(device)
        actual.load_state_dict(reference.state_dict())
        reference_input = torch.randn(
            8, 64, device=device, requires_grad=True
        )
        actual_input = reference_input.detach().clone().requires_grad_(True)
        reference.forward(reference_input).square().mean().backward()

        session = prepare_activation_offload(
            pipeline=actual,
            policy=ActivationOffloadPolicy(
                enabled=True,
                min_bytes=0,
                max_bytes=1 << 30,
                pin_memory=True,
                defer_layers=1,
                prefetch_layers=1,
                view_replay=True,
            ),
        )
        with session.context():
            actual.forward(actual_input).square().mean().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_input.grad, reference_input.grad)
        for actual_parameter, reference_parameter in zip(
            actual.parameters(), reference.parameters()
        ):
            torch.testing.assert_close(
                actual_parameter.grad,
                reference_parameter.grad,
            )
        diagnostics = session.diagnostics()
        self.assertGreater(diagnostics["offloaded_tensors"], 0)
        self.assertGreater(diagnostics["prefetched_tensors"], 0)
        self.assertGreater(diagnostics["view_handles"], 0)
        self.assertEqual(diagnostics["reserved_bytes"], 0)
        session.close()

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "activation residency accounting requires an available accelerator",
    )
    def test_layered_offload_reduces_saved_activation_residency(self) -> None:
        class Block(torch.nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(width, width, bias=False)

            def forward(self, value):
                return torch.nn.functional.silu(self.projection(value))

        class Pipeline(torch.nn.Module):
            def __init__(self, width: int, depth: int) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [Block(width) for _ in range(depth)]
                )

            def forward(self, value):
                for block in self.blocks:
                    value = block(value)
                return value

            def get_activation_offload_regions(self):
                return [
                    ActivationOffloadRegion(name=f"blocks.{idx}", owner=block)
                    for idx, block in enumerate(self.blocks)
                ]

        device = torch.device("cuda")
        torch.manual_seed(17)
        reference = Pipeline(512, 5).to(device)
        actual = Pipeline(512, 5).to(device)
        actual.load_state_dict(reference.state_dict())
        reference_input = torch.randn(
            2048, 512, device=device, requires_grad=True
        )
        actual_input = reference_input.detach().clone().requires_grad_(True)

        torch.cuda.synchronize()
        reference_baseline = int(torch.cuda.memory_allocated(device))
        reference_output = reference(reference_input)
        torch.cuda.synchronize()
        reference_saved = (
            int(torch.cuda.memory_allocated(device)) - reference_baseline
        )
        reference_output.square().mean().backward()
        del reference_output
        torch.cuda.synchronize()

        session = prepare_activation_offload(
            pipeline=actual,
            policy=ActivationOffloadPolicy(
                enabled=True,
                min_bytes=0,
                max_bytes=1 << 30,
                pin_memory=True,
                defer_layers=1,
                prefetch_layers=1,
                view_replay=True,
            ),
        )
        actual_baseline = int(torch.cuda.memory_allocated(device))
        with session.context():
            actual_output = actual(actual_input)
        torch.cuda.synchronize()
        actual_saved = int(torch.cuda.memory_allocated(device)) - actual_baseline
        self.assertGreater(reference_saved, 0)
        self.assertLess(actual_saved, reference_saved)
        actual_output.square().mean().backward()
        torch.cuda.synchronize()
        self.assertEqual(session.diagnostics()["reserved_bytes"], 0)
        session.close()


if __name__ == "__main__":
    unittest.main()
