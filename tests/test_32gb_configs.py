"""The 32 GiB hardware-profile configs state a runnable, self-consistent profile.

The assertions below are the profiles' requirements, not restatements of the
files: every example must survive the real loader and runtime validators, the
MAGI-2 training profile must fit a 128 GiB host through load-time NF4 packing,
and the LingBot profile must keep the compressed base fully resident.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from mirai.config.hardware_tiers import (
    resolve_expert_dequant_chunk_size,
    resolve_tier,
)
from mirai.config.loader import load_config
from mirai.config.runtime_policy import (
    apply_runtime_policy,
    validate_cli_model_contract,
    validate_runtime_compatibility,
)
from mirai.core.builtins import register_builtin_components
from mirai.core.training.runtime.contract import validate_training_runtime_config
from mirai.vendors.magi2_preview.common.magi2_config import ModelConfig as Magi2ModelConfig

_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = _ROOT / "configs"

MAGI2_TRAIN = _CONFIGS / "magi2_preview" / "train_offload_32gb.toml"
MAGI2_INFER = _CONFIGS / "magi2_preview" / "inference_offload_32gb.toml"
LINGBOT_TRAIN = _CONFIGS / "lingbot_video" / "train_nf4_32gb.toml"
LINGBOT_INFER = _CONFIGS / "lingbot_video" / "inference_nf4_32gb.toml"

_TRAIN_CONFIGS = (MAGI2_TRAIN, LINGBOT_TRAIN)
_INFER_CONFIGS = (MAGI2_INFER, LINGBOT_INFER)
_ALL_CONFIGS = _TRAIN_CONFIGS + _INFER_CONFIGS

PROFILE_HOST_MEMORY_GIB = 128.0
MAGI2_PACKED_LOAD_PEAK_GIB = 68.0
# The profile's declared device: 32 GiB of VRAM.
PROFILE_DEVICE_MEMORY_GIB = 32.0


def _resolved(path: Path, *, entrypoint: str):
    register_builtin_components()
    config = load_config(path)
    validate_cli_model_contract(config, entrypoint=entrypoint)
    apply_runtime_policy(config, entrypoint=entrypoint)
    validate_runtime_compatibility(config, entrypoint=entrypoint)
    return config


class ThirtyTwoGibProfileConfigTests(unittest.TestCase):
    def test_every_profile_config_loads_and_validates(self) -> None:
        for path in _ALL_CONFIGS:
            with self.subTest(config=path.name):
                self.assertTrue(path.exists(), f"{path} is missing.")
                entrypoint = "train" if path in _TRAIN_CONFIGS else "infer"
                config = _resolved(path, entrypoint=entrypoint)
                self.assertIn(
                    config.model.type, {"magi2-preview", "lingbot-video"}
                )

    def test_training_profiles_pass_the_training_runtime_contract(self) -> None:
        for path in _TRAIN_CONFIGS:
            with self.subTest(config=path.name):
                config = _resolved(path, entrypoint="train")
                validate_training_runtime_config(config)

    def test_magi2_training_profile_packs_experts_with_bounded_prefetch(self) -> None:
        config = _resolved(MAGI2_TRAIN, entrypoint="train")
        self.assertEqual(config.memory.frozen_weight_quantization, "nf4")
        self.assertEqual(
            config.memory.frozen_weight_quantization_strategy,
            "compressed_weights",
        )
        self.assertTrue(config.memory.quantize_experts_on_load)
        self.assertEqual(config.memory.weight_residency_strategy, "block_swap")
        self.assertGreater(config.training.blocks_to_swap, 0)
        self.assertLess(
            config.training.blocks_to_swap,
            int(Magi2ModelConfig().num_layers),
        )
        self.assertEqual(config.training.block_swap_mode, "async")
        self.assertEqual(config.memory.block_swap_prefetch_depth, 1)

    def test_magi2_inference_profile_streams_every_block_synchronously(self) -> None:
        config = _resolved(MAGI2_INFER, entrypoint="infer")
        self.assertEqual(config.memory.weight_residency_strategy, "block_swap")
        self.assertGreaterEqual(
            config.training.blocks_to_swap,
            int(Magi2ModelConfig().num_layers),
        )
        self.assertEqual(config.training.block_swap_mode, "sync")

    def test_magi2_training_host_memory_keys_fit_a_128_gib_machine(self) -> None:
        memory = _resolved(MAGI2_TRAIN, entrypoint="train").memory
        self.assertGreater(memory.minimum_system_memory_gib, 0.0)
        self.assertLessEqual(
            MAGI2_PACKED_LOAD_PEAK_GIB
            + memory.minimum_system_memory_gib
            + memory.max_pinned_host_gib,
            PROFILE_HOST_MEMORY_GIB,
        )

    def test_magi2_training_shape_stays_inside_the_profile_budget(self) -> None:
        config = _resolved(MAGI2_TRAIN, entrypoint="train")
        self.assertEqual(config.training.batch_size, 1)
        self.assertEqual(config.training.gradient_checkpointing, "standard")
        self.assertEqual(config.adapter.type, "lora")
        self.assertEqual(config.adapter.target_preset, "attn_router")
        for frames in config.dataset.frame_buckets:
            # The native cache path trims video to 8n + 1 frames.
            self.assertEqual((int(frames) - 1) % 8, 0)
            self.assertLessEqual(int(frames), 17)
        self.assertEqual(config.dataset.bucket_resolutions, ["256x448"])

    def test_magi2_inference_never_co_resides_auxiliary_models(self) -> None:
        config = _resolved(MAGI2_INFER, entrypoint="infer")
        self.assertFalse(config.inference.keep_text_encoder_resident)
        self.assertFalse(config.inference.keep_vae_resident)
        self.assertEqual(config.inference.cfg_mode, "batched")

    def test_lingbot_profile_is_compressed_and_fully_resident(self) -> None:
        for path in (LINGBOT_TRAIN, LINGBOT_INFER):
            with self.subTest(config=path.name):
                entrypoint = "train" if path is LINGBOT_TRAIN else "infer"
                config = _resolved(path, entrypoint=entrypoint)
                self.assertEqual(config.memory.frozen_weight_quantization, "nf4")
                self.assertEqual(
                    config.memory.frozen_weight_quantization_strategy,
                    "compressed_weights",
                )
                self.assertTrue(config.memory.quantize_experts_on_load)
                self.assertEqual(
                    config.memory.weight_residency_strategy,
                    "disabled",
                    "the compressed base fits the device; swapping it would buy nothing",
                )
                self.assertEqual(config.training.blocks_to_swap, 0)
                self.assertEqual(config.training.block_swap_mode, "sync")
                self.assertLessEqual(config.memory.cuda_memory_fraction, 0.95)

    def test_lingbot_inference_releases_auxiliary_models_between_phases(self) -> None:
        config = _resolved(LINGBOT_INFER, entrypoint="infer")
        self.assertFalse(config.inference.keep_text_encoder_resident)
        self.assertFalse(config.inference.keep_vae_resident)

    def test_explicit_chunk_size_agrees_with_the_device_memory_tier(self) -> None:
        """The named profile states the value the tier table would choose."""

        tier_choice = resolve_expert_dequant_chunk_size(
            0, profile=((8, 9), PROFILE_DEVICE_MEMORY_GIB)
        )
        for path in (LINGBOT_TRAIN, LINGBOT_INFER):
            with self.subTest(config=path.name):
                entrypoint = "train" if path is LINGBOT_TRAIN else "infer"
                config = _resolved(path, entrypoint=entrypoint)
                self.assertEqual(
                    config.memory.expert_dequant_chunk_size, tier_choice
                )

    def test_profiles_do_not_delegate_to_the_tier_table(self) -> None:
        """A named hardware profile states its own keys.

        The tier table stops at compute capability 10.x, so a current 32 GiB
        consumer device has no matching tier and `hardware_policy = "tiered"`
        would fail closed instead of filling anything in.
        """

        self.assertIsNone(resolve_tier(profile=((12, 0), PROFILE_DEVICE_MEMORY_GIB)))
        for path in _ALL_CONFIGS:
            with self.subTest(config=path.name):
                entrypoint = "train" if path in _TRAIN_CONFIGS else "infer"
                config = _resolved(path, entrypoint=entrypoint)
                self.assertEqual(config.memory.hardware_policy, "disabled")


if __name__ == "__main__":
    unittest.main()
