from __future__ import annotations

import unittest

from mirai.config.runtime_policy import (
    has_available_sparse_moe_backend,
    is_supported_sparse_moe_model_type,
    runtime_policy_summary,
    validate_cli_model_contract,
    validate_native_backend_availability,
    validate_runtime_compatibility,
)
from mirai.config.schema import (
    ModelConfig,
    ModelParams,
    StrategyConfig,
    TrainingConfig,
    TrainingSection,
)
from mirai.core.models.providers import ModelFamilyProvider, register_model_family_provider


class RuntimePolicyTests(unittest.TestCase):
    def _cfg(self, model_type: str, *, variant: str = "scratch") -> TrainingConfig:
        return TrainingConfig(
            preset=None,
            model=ModelConfig(
                type=model_type,
                path="./models/moe",
                dtype="bf16",
                attention_backend="auto",
                params=ModelParams(
                    variant=variant,
                    flow_shift=3.0,
                    strict_native_assets=False,
                    latent_channels=1,
                    num_experts=4,
                    experts_per_token=2,
                    shared_experts=1,
                    hidden_size=16,
                    num_layers=1,
                    attention_heads=4,
                ),
            ),
            strategy=StrategyConfig(type="text_to_video", params={}),
            training=TrainingSection(seed=1, batch_size=1),
        )

    def test_sparse_moe_models_are_supported(self) -> None:
        self.assertTrue(is_supported_sparse_moe_model_type("lingbot-video"))
        self.assertTrue(is_supported_sparse_moe_model_type("sparse_moe_test"))

    def test_only_integrated_backends_are_available(self) -> None:
        self.assertTrue(has_available_sparse_moe_backend("lingbot-video"))
        self.assertTrue(has_available_sparse_moe_backend("sparse_moe_test"))

    def test_runtime_policy_summary_marks_lingbot_video_as_sparse_moe(self) -> None:
        cfg = self._cfg("lingbot-video", variant="tiny-video")
        cfg.model.params.denoiser_subfolder = "refiner"
        summary = runtime_policy_summary(cfg)

        self.assertEqual(summary["model_type"], "lingbot-video")
        self.assertEqual(summary["model_variant"], "tiny-video")
        self.assertEqual(summary["denoiser_subfolder"], "refiner")
        self.assertTrue(summary["registered_provider"])
        self.assertTrue(summary["native_model"])
        self.assertTrue(summary["sparse_moe_model"])
        self.assertTrue(summary["runtime_backend_available"])

    def test_runtime_compatibility_accepts_sparse_moe_models(self) -> None:
        validate_runtime_compatibility(
            self._cfg("lingbot-video", variant="tiny-video"),
            entrypoint="unit",
        )
        validate_runtime_compatibility(
            self._cfg("sparse_moe_test", variant="tiny-video"),
            entrypoint="unit",
        )

    def test_caption_formats_are_owned_by_the_model_provider(self) -> None:
        lingbot = self._cfg("lingbot-video", variant="tiny-video")
        lingbot.dataset.caption_format = "lingbot_json"
        validate_runtime_compatibility(lingbot, entrypoint="unit")

        testbed = self._cfg("sparse_moe_test", variant="tiny-video")
        testbed.dataset.caption_format = "lingbot_json"
        with self.assertRaisesRegex(ValueError, "does not support dataset caption format"):
            validate_runtime_compatibility(testbed, entrypoint="unit")

    def test_non_video_moe_name_is_not_supported(self) -> None:
        self.assertFalse(is_supported_sparse_moe_model_type("image-moe-model"))
        self.assertFalse(is_supported_sparse_moe_model_type("toy_moe_dit"))

    def test_cli_contract_rejects_unknown_model(self) -> None:
        cfg = self._cfg("unknown-model")
        with self.assertRaisesRegex(ValueError, "not a registered sparse-MoE"):
            validate_cli_model_contract(
                cfg,
                entrypoint="unit",
            )

    def test_cli_contract_rejects_registered_native_non_sparse_model(self) -> None:
        register_model_family_provider(
            "plain-native-test",
            ModelFamilyProvider(model_type="plain-native-test", native=True, sparse_moe=False),
        )
        with self.assertRaisesRegex(ValueError, "not a sparse-MoE diffusion model"):
            validate_cli_model_contract(
                self._cfg("plain-native-test"),
                entrypoint="unit",
            )

    def test_backend_availability_rejects_unknown_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a registered sparse-MoE"):
            validate_native_backend_availability(
                self._cfg("unknown-model"),
                entrypoint="unit",
            )


if __name__ == "__main__":
    unittest.main()
