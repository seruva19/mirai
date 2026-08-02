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
from mirai.core.training.residency.memory_safety import configure_memory_safety

_GIB = 1024**3


class PackedStateResidencyPolicyTests(unittest.TestCase):
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
