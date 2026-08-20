from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mirai.core.moe.runtime.expert_transfer_profile import (
    ExpertTransferProfile, build_expert_transfer_profile, load_expert_transfer_profile,
    recommend_transfer_policy, save_expert_transfer_profile,
    validate_expert_transfer_profile_identity,
)


class ExpertTransferProfileTests(unittest.TestCase):
    def test_recommendation_is_bounded_and_round_trips(self) -> None:
        cache, depth = recommend_transfer_policy(
            h2d_gib_per_second=16.0, routed_compute_gib_per_second=64.0,
            expert_bytes=64 * 1024 * 1024, working_set_experts=32,
            cache_budget_gib=1.25,
        )
        self.assertEqual((cache, depth), (1.25, 4))
        profile = ExpertTransferProfile(
            gpu_name="test-gpu", compute_capability="8.0", expert_format="int8",
            expert_bytes=64 * 1024 * 1024, h2d_gib_per_second=16.0,
            routed_compute_gib_per_second=64.0, recommended_device_cache_gib=cache,
            recommended_prefetch_depth=depth, benchmark_fingerprint="test-v1",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            save_expert_transfer_profile(path, profile)
            self.assertEqual(load_expert_transfer_profile(path), profile)

    def test_invalid_and_unknown_data_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "working_set_experts"):
            recommend_transfer_policy(h2d_gib_per_second=1, routed_compute_gib_per_second=1, expert_bytes=1, working_set_experts=0, cache_budget_gib=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text('{"schema":"mirai.expert_transfer_profile.v1","extra":1}')
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_expert_transfer_profile(path)
            path.write_text('{"schema":"mirai.expert_transfer_profile.v1"}')
            with self.assertRaisesRegex(ValueError, "Missing"):
                load_expert_transfer_profile(path)

    def test_profile_fills_only_unset_runtime_values(self) -> None:
        from mirai.core.moe.runtime.specs import MoEOptimizationPolicy

        profile = ExpertTransferProfile(
            gpu_name="test-gpu", compute_capability="8.0", expert_format="int8",
            expert_bytes=1024, h2d_gib_per_second=12.0,
            routed_compute_gib_per_second=24.0, recommended_device_cache_gib=2.0,
            recommended_prefetch_depth=2, benchmark_fingerprint="test-v1",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            save_expert_transfer_profile(path, profile)
            resolved = MoEOptimizationPolicy.from_memory_config(SimpleNamespace(
                expert_transfer_profile_path=str(path), expert_device_cache_gib=0.0,
                packed_stream_prefetch_depth=0,
            ))
            self.assertEqual(resolved.expert_device_cache_gib, 2.0)
            self.assertEqual(resolved.packed_stream_prefetch_depth, 2)
            explicit = MoEOptimizationPolicy.from_memory_config(SimpleNamespace(
                expert_transfer_profile_path=str(path), expert_device_cache_gib=0.5,
                packed_stream_prefetch_depth=3,
            ))
            self.assertEqual(explicit.expert_device_cache_gib, 0.5)
            self.assertEqual(explicit.packed_stream_prefetch_depth, 3)

    def test_builder_fingerprints_protocol_and_derives_policy(self) -> None:
        profile = build_expert_transfer_profile(
            gpu_name="gpu", compute_capability="9.0", expert_format="bf16",
            expert_bytes=1024, working_set_experts=8, cache_budget_gib=1.0,
            h2d_gib_per_second=20.0, routed_compute_gib_per_second=40.0,
            benchmark_protocol={"iterations": 10, "shape": [8, 16]},
        )
        self.assertEqual(len(profile.benchmark_fingerprint), 64)
        self.assertEqual(profile.recommended_prefetch_depth, 2)
        self.assertEqual(profile.recommended_device_cache_gib, 0.0)

        int8_profile = build_expert_transfer_profile(
            gpu_name="gpu", compute_capability="9.0", expert_format="int8",
            expert_bytes=1024, working_set_experts=8, cache_budget_gib=1.0,
            h2d_gib_per_second=20.0, routed_compute_gib_per_second=40.0,
            benchmark_protocol={"iterations": 10, "shape": [8, 16]},
        )
        self.assertGreater(int8_profile.recommended_device_cache_gib, 0.0)

    def test_runtime_identity_mismatch_fails_explicitly(self) -> None:
        profile = ExpertTransferProfile(
            gpu_name="gpu-a", compute_capability="9.0", expert_format="int8",
            expert_bytes=1024, h2d_gib_per_second=10,
            routed_compute_gib_per_second=20, recommended_device_cache_gib=1,
            recommended_prefetch_depth=2, benchmark_fingerprint="test",
        )
        validate_expert_transfer_profile_identity(
            profile, gpu_name="gpu-a", compute_capability="9.0",
            expert_format="INT8",
        )
        with self.assertRaisesRegex(ValueError, "gpu_name.*gpu-b"):
            validate_expert_transfer_profile_identity(
                profile, gpu_name="gpu-b", compute_capability="9.0",
                expert_format="int8",
            )
        with self.assertRaisesRegex(ValueError, "expert_format"):
            validate_expert_transfer_profile_identity(
                profile, gpu_name="gpu-a", compute_capability="9.0",
                expert_format="bf16",
            )


if __name__ == "__main__":
    unittest.main()
