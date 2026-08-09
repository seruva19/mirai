from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from mirai.config.loader import load_config
from mirai.config.schema import ConfigError, TrainingConfig
from mirai.core.training.runtime.contract import validate_training_runtime_config


class ConfigTests(unittest.TestCase):
    def test_explicit_default_family_uses_provider_owned_defaults(self) -> None:
        cfg = load_config(self._write('[model]\ntype = "lingbot-video"\n'))
        self.assertEqual(cfg.model.path, "./models/lingbot_video")
        self.assertEqual(cfg.model.params.variant, "lingbot-video-moe-30b-a3b")
        self.assertTrue(cfg.model.params.strict_native_assets)
        self.assertEqual(cfg.model.params.latent_channels, 16)
        self.assertEqual(cfg.model.params.hidden_size, 2048)
        self.assertEqual(cfg.model.params.num_layers, 48)
        self.assertEqual(cfg.model.params.attention_heads, 16)
        self.assertEqual(cfg.model.params.patch_size, 2)
        self.assertEqual(cfg.model.params.num_experts, 128)
        self.assertEqual(cfg.model.params.experts_per_token, 8)
        self.assertEqual(cfg.model.params.shared_experts, 1)

    def test_custom_provider_params_are_preserved_without_lingbot_defaults(self) -> None:
        cfg = TrainingConfig.from_dict(
            {
                "model": {
                    "type": "external_family",
                    "params": {"family_params": {"width": 24, "activation": "gelu"}},
                }
            }
        )
        self.assertEqual(
            cfg.model.params.family_params,
            {"width": 24, "activation": "gelu"},
        )
        self.assertEqual(cfg.model.path, "")
        self.assertEqual(cfg.model.params.variant, "")
        self.assertFalse(cfg.model.params.strict_native_assets)

    def test_invalid_model_dtype_is_rejected_in_schema(self) -> None:
        with self.assertRaisesRegex(ConfigError, "model.dtype must be one of"):
            TrainingConfig.from_dict({"model": {"dtype": "bf61"}})

    def test_runtime_rejects_invalid_memory_policy_enums_early(self) -> None:
        cases = (
            ("weight_residency_strategy", "somewhere"),
            ("expert_weight_access", "somehow"),
            ("packed_state_preload", "sometimes"),
        )
        for key, value in cases:
            with self.subTest(key=key):
                cfg = TrainingConfig.from_dict({"memory": {key: value}})
                with self.assertRaisesRegex(ValueError, key):
                    validate_training_runtime_config(cfg)

    def test_runtime_rejects_dynamic_topk_with_expert_choice(self) -> None:
        cfg = TrainingConfig.from_dict(
            {
                "model": {
                    "params": {
                        "moe_routing_mode": "expert_choice",
                        "moe_dynamic_topk_min": 1,
                        "moe_dynamic_topk_average": 1.5,
                    }
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "dynamic top-k"):
            validate_training_runtime_config(cfg)

    def _write(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        tmp.write(textwrap.dedent(content).strip() + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_unknown_key_rejected_with_hint(self) -> None:
        path = self._write(
            """
            [training]
            gradient_accumlation = 4
            """
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("Unknown config key 'training.gradient_accumlation'", str(ctx.exception))
        self.assertIn("gradient_accumulation", str(ctx.exception))

    def test_variant_must_be_model_params(self) -> None:
        path = self._write(
            """
            [model]
            type = "sparse_moe_test"
            variant = "tiny-video"
            """
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("model.params.variant", str(ctx.exception))

    def test_preset_merge_and_override(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [model.params]
            variant = "tiny-video"
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.model.params.variant, "tiny-video")
        self.assertEqual(cfg.training.batch_size, 1)

    def test_sample_pack_arrays_parse(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [logging]
            sample_prompts = ["a cat", "a dog"]
            sample_seeds = [7, 11]
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.logging.sample_prompts, ["a cat", "a dog"])
        self.assertEqual(cfg.logging.sample_seeds, [7, 11])

    def test_contrastive_flow_weight_parses(self) -> None:
        path = self._write(
            """
            [training]
            batch_size = 2
            contrastive_flow_weight = 0.05
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.training.contrastive_flow_weight, 0.05)

    def test_regional_compile_policy_parses(self) -> None:
        path = self._write(
            """
            [training]
            compile = true
            compile_scope = "regional"
            compile_mode = "reduce-overhead"
            compile_dynamic = true
            compile_token_buckets = [4096, 8192, 16384]
            """
        )
        cfg = load_config(path)
        self.assertTrue(cfg.training.compile)
        self.assertEqual(cfg.training.compile_scope, "regional")
        self.assertEqual(cfg.training.compile_mode, "reduce-overhead")
        self.assertIs(cfg.training.compile_dynamic, True)
        self.assertEqual(
            cfg.training.compile_token_buckets,
            [4096, 8192, 16384],
        )
        validate_training_runtime_config(cfg)

    def test_compile_dynamic_rejects_non_boolean(self) -> None:
        path = self._write(
            """
            [training]
            compile_dynamic = "auto"
            """
        )
        with self.assertRaisesRegex(ConfigError, "compile_dynamic"):
            load_config(path)

    def test_compile_token_buckets_reject_invalid_policy(self) -> None:
        path = self._write(
            """
            [training]
            compile = true
            compile_dynamic = false
            compile_token_buckets = [8192, 4096]
            """
        )
        cfg = load_config(path)
        with self.assertRaisesRegex(ValueError, "compile_token_buckets"):
            validate_training_runtime_config(cfg)

    def test_layer_aware_activation_offload_parses_and_validates(self) -> None:
        path = self._write(
            """
            [training]
            activation_cpu_offload = true
            activation_cpu_offload_max_gib = 4.0
            activation_cpu_offload_pin_memory = true
            activation_cpu_offload_defer_layers = 1
            activation_cpu_offload_prefetch_layers = 2
            activation_cpu_offload_view_replay = true
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.training.activation_cpu_offload_defer_layers, 1)
        self.assertEqual(cfg.training.activation_cpu_offload_prefetch_layers, 2)
        self.assertTrue(cfg.training.activation_cpu_offload_view_replay)
        validate_training_runtime_config(cfg)

    def test_activation_prefetch_requires_pinned_memory(self) -> None:
        path = self._write(
            """
            [training]
            activation_cpu_offload = true
            activation_cpu_offload_max_gib = 4.0
            activation_cpu_offload_prefetch_layers = 1
            """
        )
        cfg = load_config(path)
        with self.assertRaisesRegex(ValueError, "pin_memory"):
            validate_training_runtime_config(cfg)

    def test_activation_schedule_is_not_inert_while_offload_is_disabled(self) -> None:
        path = self._write(
            """
            [training]
            activation_cpu_offload_defer_layers = 1
            """
        )
        cfg = load_config(path)
        with self.assertRaisesRegex(ValueError, "require"):
            validate_training_runtime_config(cfg)

    def test_moe_token_chunking_parses_and_validates(self) -> None:
        path = self._write(
            """
            [training]
            moe_token_chunk_size = 2048
            """
        )
        cfg = load_config(path)

        self.assertEqual(cfg.training.moe_token_chunk_size, 2048)
        validate_training_runtime_config(cfg)

    def test_inference_moe_token_chunking_parses_and_validates(self) -> None:
        path = self._write(
            """
            [inference]
            moe_token_chunk_size = 8192
            """
        )
        cfg = load_config(path)

        self.assertEqual(cfg.inference.moe_token_chunk_size, 8192)

    def test_inference_sequential_component_staging_parses_and_validates(self) -> None:
        path = self._write(
            """
            [inference]
            stage_text_encoder_before_denoiser = true
            """
        )
        cfg = load_config(path)
        self.assertTrue(cfg.inference.stage_text_encoder_before_denoiser)

        incompatible = self._write(
            """
            [inference]
            stage_text_encoder_before_denoiser = true
            keep_text_encoder_resident = true
            """
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_config(incompatible)

        quantized = self._write(
            """
            [inference]
            stage_text_encoder_before_denoiser = true
            text_encoder_weight_quantization = "nf4"
            """
        )
        self.assertEqual(
            load_config(quantized).inference.text_encoder_weight_quantization,
            "nf4",
        )

        missing_staging = self._write(
            """
            [inference]
            text_encoder_weight_quantization = "nf4"
            """
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            load_config(missing_staging)

    def test_inference_moe_token_chunking_rejects_negative_size(self) -> None:
        path = self._write(
            """
            [inference]
            moe_token_chunk_size = -1
            """
        )
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            load_config(path)

    def test_inference_moe_token_chunking_rejects_branch_cache(self) -> None:
        path = self._write(
            """
            [inference]
            expert_feature_cache = "branch"
            moe_token_chunk_size = 8

            [memory]
            moe_kernel_backend = "grouped"
            """
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_config(path)

    def test_moe_token_chunking_rejects_parameter_dropout(self) -> None:
        path = self._write(
            """
            [training]
            moe_token_chunk_size = 64

            [adapter]
            lora_parameter_dropout = 0.2
            """
        )
        cfg = load_config(path)

        with self.assertRaisesRegex(ValueError, "masks must be shared"):
            validate_training_runtime_config(cfg)

    def test_routing_agreement_evidence_gate_parses(self) -> None:
        path = self._write(
            """
            [model.params]
            moe_routing_agreement_evidence = "report"
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.model.params.moe_routing_agreement_evidence, "report")

    def test_routing_agreement_evidence_gate_rejects_unknown_mode(self) -> None:
        path = self._write(
            """
            [model.params]
            moe_routing_agreement_evidence = "trace"
            """
        )
        with self.assertRaisesRegex(
            ConfigError,
            "moe_routing_agreement_evidence",
        ):
            load_config(path)

    def test_memory_feature_flags_parse(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [memory]
            frozen_weight_quantization = "fp8"
            frozen_weight_quantization_strategy = "auto"
            frozen_weight_packed_state_path = "./models/packed_int8.safetensors"
            weight_residency_strategy = "stream_disk"
            expert_weight_access = "chunked_dequant"
            expert_dequant_chunk_size = 4
            quantize_experts_on_load = true
            router_quantization = "disabled"
            moe_kernel_backend = "megablocks"
            moe_expert_autograd = "segmented_recompute"
            moe_activation_backend = "triton"
            cuda_memory_fraction = 0.85
            minimum_system_memory_gib = 12.0
            trainable_parameter_offload = true
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.memory.frozen_weight_quantization, "fp8")
        self.assertEqual(cfg.memory.frozen_weight_quantization_strategy, "auto")
        self.assertEqual(cfg.memory.frozen_weight_packed_state_path, "./models/packed_int8.safetensors")
        self.assertEqual(cfg.memory.weight_residency_strategy, "stream_disk")
        self.assertEqual(cfg.memory.expert_weight_access, "chunked_dequant")
        self.assertEqual(cfg.memory.expert_dequant_chunk_size, 4)
        self.assertTrue(cfg.memory.quantize_experts_on_load)
        self.assertEqual(cfg.memory.router_quantization, "disabled")
        self.assertEqual(cfg.memory.moe_kernel_backend, "megablocks")
        self.assertEqual(cfg.memory.moe_expert_autograd, "segmented_recompute")
        self.assertEqual(cfg.memory.moe_activation_backend, "triton")
        self.assertEqual(cfg.memory.cuda_memory_fraction, 0.85)
        self.assertEqual(cfg.memory.minimum_system_memory_gib, 12.0)
        self.assertTrue(cfg.memory.trainable_parameter_offload)

    def test_preview_cfg_fields_parse(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [logging]
            sample_cfg_scale = 6.5
            sample_negative_prompt = "blurry"
            sample_solver = "unipc"
            sample_resolution = "128x96"
            sample_frame_count = 33
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.logging.sample_cfg_scale, 6.5)
        self.assertEqual(cfg.logging.sample_negative_prompt, "blurry")
        self.assertEqual(cfg.logging.sample_solver, "unipc")
        self.assertEqual(cfg.logging.sample_resolution, "128x96")
        self.assertEqual(cfg.logging.sample_frame_count, 33)

    def test_wandb_fields_parse(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [logging]
            wandb = true
            wandb_project = "proj"
            wandb_run_name = "run-1"
            """
        )
        cfg = load_config(path)
        self.assertTrue(cfg.logging.wandb)
        self.assertEqual(cfg.logging.wandb_project, "proj")
        self.assertEqual(cfg.logging.wandb_run_name, "run-1")

    def test_adapter_dropout_and_schedule_parse(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [adapter]
            rank_dropout = 0.25
            lora_parameter_dropout = 0.15
            rank_schedule_start = 10
            rank_schedule_end = 100
            rank_schedule_min_scale = 0.4
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.adapter.rank_dropout, 0.25)
        self.assertEqual(cfg.adapter.lora_parameter_dropout, 0.15)
        self.assertEqual(cfg.adapter.rank_schedule_start, 10)
        self.assertEqual(cfg.adapter.rank_schedule_end, 100)
        self.assertEqual(cfg.adapter.rank_schedule_min_scale, 0.4)

    def test_esft_controls_parse_and_validate(self) -> None:
        cfg = load_config(
            self._write(
                """
                [adapter]
                type = "selected_expert"
                expert_selection = "esft_token"
                esft_selection_mass = 0.35
                esft_calibration_samples = 48

                [optimizer]
                type = "selected_expert_adamw"
                """
            )
        )
        self.assertEqual(cfg.adapter.expert_selection, "esft_token")
        self.assertEqual(cfg.adapter.esft_selection_mass, 0.35)
        self.assertEqual(cfg.adapter.esft_calibration_samples, 48)

        for value in ("0.0", "1.1", "nan"):
            with self.subTest(selection_mass=value):
                with self.assertRaisesRegex(ConfigError, "esft_selection_mass"):
                    load_config(
                        self._write(
                            "[adapter]\nesft_selection_mass = "
                            f"{value}\n"
                        )
                    )
        with self.assertRaisesRegex(ConfigError, "esft_calibration_samples"):
            load_config(
                self._write(
                    "[adapter]\nesft_calibration_samples = 0\n"
                )
            )
        with self.assertRaisesRegex(ConfigError, "expert_selection"):
            load_config(
                self._write(
                    '[adapter]\nexpert_selection = "esft_unknown"\n'
                )
            )

    def test_para_posthoc_transform_gate_parses_and_requires_lora(self) -> None:
        cfg = load_config(
            self._write(
                "[adapter]\nposthoc_rank_compression = \"para\"\n"
            )
        )
        self.assertEqual(cfg.adapter.posthoc_rank_compression, "para")

        with self.assertRaisesRegex(ConfigError, "posthoc_rank_compression"):
            load_config(
                self._write(
                    "[adapter]\nposthoc_rank_compression = \"unknown\"\n"
                )
            )
        with self.assertRaisesRegex(ConfigError, "requires.*lora"):
            load_config(
                self._write(
                    "[adapter]\ntype = \"sparse_delta\"\n"
                    "posthoc_rank_compression = \"para\"\n"
                )
            )

    def test_adapter_allocation_and_rslora_parse(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [adapter]
            rank_budget = 24
            adaptive_rank_plan_path = "./artifacts/rank-plan.json"
            use_rslora = true

            [adapter.rank_pattern]
            "blocks.0.*" = 4

            [adapter.alpha_pattern]
            "*.experts.w1" = 8.0
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.adapter.rank_pattern, {"blocks.0.*": 4})
        self.assertEqual(cfg.adapter.alpha_pattern, {"*.experts.w1": 8.0})
        self.assertEqual(cfg.adapter.rank_budget, 24)
        self.assertEqual(
            cfg.adapter.adaptive_rank_plan_path,
            "./artifacts/rank-plan.json",
        )
        self.assertTrue(cfg.adapter.use_rslora)

    def test_lora_fa_parses_and_rejects_dynamic_a_masks(self) -> None:
        cfg = load_config(self._write("[adapter]\nuse_lora_fa = true\n"))
        self.assertTrue(cfg.adapter.use_lora_fa)

        with self.assertRaisesRegex(ConfigError, "fixed A projection"):
            load_config(
                self._write(
                    "[adapter]\nuse_lora_fa = true\nrank_dropout = 0.1\n"
                )
            )
        with self.assertRaisesRegex(ConfigError, "fixed A projection"):
            load_config(
                self._write(
                    "[adapter]\nuse_lora_fa = true\n"
                    "lora_parameter_dropout = 0.1\n"
                )
            )

        for value in ("-0.1", "1.0", "nan"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ConfigError, "lora_parameter_dropout"
                ):
                    load_config(
                        self._write(
                            "[adapter]\nlora_parameter_dropout = "
                            f"{value}\n"
                        )
                    )

    def test_dora_parses_and_runtime_contract_rejects_unsupported_hosts(
        self,
    ) -> None:
        cfg = load_config(
            self._write(
                """
                [model]
                type = "lingbot-video"
                [adapter]
                use_dora = true
                """
            )
        )
        self.assertTrue(cfg.adapter.use_dora)
        validate_training_runtime_config(cfg)

        activation = load_config(
            self._write(
                """
                [model]
                type = "lingbot-video"
                [adapter]
                use_dora = true
                expert_tensor_lora_backend = "activation"
                """
            )
        )
        with self.assertRaisesRegex(ValueError, "weight_space"):
            validate_training_runtime_config(activation)

        packed = load_config(
            self._write(
                """
                [model]
                type = "lingbot-video"
                [adapter]
                use_dora = true
                [memory]
                frozen_weight_quantization = "nf4"
                """
            )
        )
        with self.assertRaisesRegex(ValueError, "unpacked frozen weights"):
            validate_training_runtime_config(packed)

        unsupported = load_config(
            self._write(
                """
                [model]
                type = "sparse_moe_test"
                [adapter]
                use_dora = true
                """
            )
        )
        with self.assertRaisesRegex(ValueError, "does not expose"):
            validate_training_runtime_config(unsupported)

    def test_eva_calibration_controls_parse_and_validate(self) -> None:
        cfg = load_config(
            self._write(
                "[adapter]\nlora_init = \"eva\"\n"
                "eva_calibration_steps = 12\n"
                "eva_samples_per_target = 64\n"
                "eva_convergence_threshold = 0.995\n"
            )
        )
        self.assertEqual(cfg.adapter.lora_init, "eva")
        self.assertEqual(cfg.adapter.eva_calibration_steps, 12)
        self.assertEqual(cfg.adapter.eva_samples_per_target, 64)
        self.assertEqual(cfg.adapter.eva_convergence_threshold, 0.995)

        with self.assertRaisesRegex(ConfigError, "eva_calibration_steps"):
            load_config(self._write("[adapter]\neva_calibration_steps = 1\n"))
        with self.assertRaisesRegex(ConfigError, "eva_samples_per_target"):
            load_config(self._write("[adapter]\neva_samples_per_target = 0\n"))
        with self.assertRaisesRegex(ConfigError, "eva_convergence_threshold"):
            load_config(
                self._write("[adapter]\neva_convergence_threshold = 1.1\n")
            )

    def test_gora_controls_parse_and_runtime_contract(self) -> None:
        cfg = load_config(
            self._write(
                """
                preset = "sparse_moe_test"
                [model]
                type = "sparse_moe_test"
                [adapter]
                lora_init = "gora"
                use_rslora = true
                gora_calibration_steps = 12
                gora_min_rank = 2
                gora_max_rank = 24
                gora_stable_gamma = 0.08
                """
            )
        )
        self.assertEqual(cfg.adapter.lora_init, "gora")
        self.assertTrue(cfg.adapter.use_rslora)
        self.assertEqual(cfg.adapter.gora_calibration_steps, 12)
        self.assertEqual(cfg.adapter.gora_min_rank, 2)
        self.assertEqual(cfg.adapter.gora_max_rank, 24)
        self.assertEqual(cfg.adapter.gora_stable_gamma, 0.08)
        validate_training_runtime_config(cfg)

        with self.assertRaisesRegex(ConfigError, "gora_calibration_steps"):
            load_config(
                self._write("[adapter]\ngora_calibration_steps = 0\n")
            )
        with self.assertRaisesRegex(ConfigError, "gora_max_rank"):
            load_config(
                self._write(
                    "[adapter]\ngora_min_rank = 8\ngora_max_rank = 4\n"
                )
            )
        without_rslora = load_config(
            self._write(
                '[adapter]\nlora_init = "gora"\nuse_rslora = false\n'
            )
        )
        with self.assertRaisesRegex(ValueError, "use_rslora=true"):
            validate_training_runtime_config(without_rslora)
        with_rank_pattern = load_config(
            self._write(
                """
                [adapter]
                lora_init = "gora"
                use_rslora = true
                [adapter.rank_pattern]
                "*.to_q" = 4
                """
            )
        )
        with self.assertRaisesRegex(ValueError, "owns rank allocation"):
            validate_training_runtime_config(with_rank_pattern)

    def test_adapter_allocation_rejects_boolean_rank_and_nonfinite_alpha(self) -> None:
        boolean_rank = self._write(
            """
            preset = "sparse_moe_test"
            [model]
            type = "sparse_moe_test"
            [adapter.rank_pattern]
            "blocks.*" = true
            """
        )
        with self.assertRaisesRegex(ConfigError, "rank_pattern"):
            load_config(boolean_rank)

        nonfinite_alpha = self._write(
            """
            preset = "sparse_moe_test"
            [model]
            type = "sparse_moe_test"
            [adapter.alpha_pattern]
            "blocks.*" = nan
            """
        )
        with self.assertRaisesRegex(ConfigError, "finite"):
            load_config(nonfinite_alpha)

    def test_adapter_lr_multipliers_parse(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [adapter.lr_multipliers]
            cross_attn = 1.0
            self_attn = 0.5
            ffn = 0.3
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.adapter.lr_multipliers["cross_attn"], 1.0)
        self.assertEqual(cfg.adapter.lr_multipliers["self_attn"], 0.5)
        self.assertEqual(cfg.adapter.lr_multipliers["ffn"], 0.3)

    def test_unsupported_adapter_type_is_rejected_by_runtime_contract(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [adapter]
            type = "lokr"
            """
        )
        cfg = load_config(path)
        with self.assertRaisesRegex(ValueError, "adapter.type must be 'lora'"):
            validate_training_runtime_config(cfg)

    def test_curriculum_parse(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [training.curriculum]
            enabled = true
            resolution_schedule = { "0" = "512x512", "100" = "768x768" }
            frame_schedule = { "0" = 16, "200" = 33 }
            """
        )
        cfg = load_config(path)
        self.assertTrue(bool(cfg.training.curriculum.get("enabled")))
        self.assertEqual(cfg.training.curriculum["resolution_schedule"]["100"], "768x768")

    def test_timestep_sampling_controls_parse(self) -> None:
        cfg = load_config(
            self._write(
                """
                [training]
                timestep_sampling = "logit_normal"
                timestep_sampling_mean = 0.25
                timestep_sampling_std = 1.5
                timestep_sampling_mode_scale = 1.1
                """
            )
        )
        self.assertEqual(cfg.training.timestep_sampling, "logit_normal")
        self.assertEqual(cfg.training.timestep_sampling_mean, 0.25)
        self.assertEqual(cfg.training.timestep_sampling_std, 1.5)
        self.assertEqual(cfg.training.timestep_sampling_mode_scale, 1.1)
        validate_training_runtime_config(cfg)

    def test_stochastic_rounding_control_parses(self) -> None:
        cfg = load_config(
            self._write(
                """
                [optimizer]
                type = "adamw"
                stochastic_rounding = true
                """
            )
        )
        self.assertTrue(cfg.optimizer.stochastic_rounding)
        validate_training_runtime_config(cfg)

    def test_lora_pro_contract_accepts_exact_path_and_rejects_masks(self) -> None:
        cfg = load_config(
            self._write(
                """
                [optimizer]
                type = "lora_pro_adamw"
                lora_pro_damping = 1e-7
                stochastic_rounding = true
                """
            )
        )
        self.assertEqual(cfg.optimizer.lora_pro_damping, 1e-7)
        validate_training_runtime_config(cfg)

        masked = load_config(
            self._write(
                """
                [optimizer]
                type = "lora_pro_adamw"
                [adapter]
                timestep_rank_schedule = "tlora"
                """
            )
        )
        with self.assertRaisesRegex(ValueError, "timestep rank masks"):
            validate_training_runtime_config(masked)

        decayed = load_config(
            self._write(
                """
                [optimizer]
                type = "lora_pro_adamw"
                weight_decay = 0.01
                """
            )
        )
        with self.assertRaisesRegex(ValueError, "weight_decay=0"):
            validate_training_runtime_config(decayed)

    def test_lora_muon_contract_and_split_weight_decay(self) -> None:
        cfg = load_config(
            self._write(
                """
                [optimizer]
                type = "lora_muon"
                weight_decay = 0.01
                weight_decay_filter = "none"
                muon_momentum = 0.9
                lora_muon_gauge_rebalance_interval = 8
                lora_muon_gauge_rebalance_alpha = 0.5
                stochastic_rounding = true
                """
            )
        )
        self.assertEqual(cfg.optimizer.muon_momentum, 0.9)
        self.assertEqual(cfg.optimizer.lora_muon_gauge_rebalance_interval, 8)
        self.assertEqual(cfg.optimizer.lora_muon_gauge_rebalance_alpha, 0.5)
        validate_training_runtime_config(cfg)

        filtered = load_config(
            self._write(
                """
                [optimizer]
                type = "lora_muon"
                weight_decay = 0.01
                """
            )
        )
        with self.assertRaisesRegex(ValueError, "split decay"):
            validate_training_runtime_config(filtered)

        masked = load_config(
            self._write(
                """
                [optimizer]
                type = "lora_muon"
                [adapter]
                lora_parameter_dropout = 0.1
                """
            )
        )
        with self.assertRaisesRegex(ValueError, "LoRA dropout"):
            validate_training_runtime_config(masked)

    def test_selected_expert_muon_config_and_runtime_contract(self) -> None:
        cfg = load_config(
            self._write(
                """
                [adapter]
                type = "selected_expert"

                [optimizer]
                type = "selected_expert_adamuon"
                selected_expert_ids = [0, 2]
                muon_momentum = 0.9
                muon_nesterov = false
                muon_ns_steps = 4
                muon_eps = 1e-7
                muon_rms_target = 0.15
                stochastic_rounding = true
                """
            )
        )
        self.assertEqual(cfg.optimizer.muon_momentum, 0.9)
        self.assertFalse(cfg.optimizer.muon_nesterov)
        self.assertEqual(cfg.optimizer.muon_ns_steps, 4)
        self.assertEqual(cfg.optimizer.muon_eps, 1e-7)
        self.assertEqual(cfg.optimizer.muon_rms_target, 0.15)
        validate_training_runtime_config(cfg)

        wrong_adapter = load_config(
            self._write(
                """
                [optimizer]
                type = "selected_expert_muon"
                selected_expert_ids = [0]
                """
            )
        )
        with self.assertRaisesRegex(ValueError, "adapter.type='selected_expert'"):
            validate_training_runtime_config(wrong_adapter)

    def test_online_dataset_flags_parse(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [dataset]
            online_tag_shuffle = true
            online_tag_shuffle_dropout = 0.3
            online_tag_shuffle_keep_first_n_tags = 2
            online_temporal_resampling = true
            """
        )
        cfg = load_config(path)
        self.assertTrue(cfg.dataset.online_tag_shuffle)
        self.assertEqual(cfg.dataset.online_tag_shuffle_dropout, 0.3)
        self.assertEqual(cfg.dataset.online_tag_shuffle_keep_first_n_tags, 2)
        self.assertTrue(cfg.dataset.online_temporal_resampling)

    def test_dataset_preprocess_flags_parse(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [model]
            type = "sparse_moe_test"

            [dataset]
            auto_preprocess_cache = true
            preprocess_raw_media_to_pt = true
            """
        )
        cfg = load_config(path)
        self.assertTrue(cfg.dataset.auto_preprocess_cache)
        self.assertTrue(cfg.dataset.preprocess_raw_media_to_pt)

    def test_moe_dataset_routing_policy_parses(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [dataset.moe_routing]
            specialization_mode = "soft_affinity"
            domain_metadata_key = "visual_domain"
            routing_prior_weight = 0.05
            router_warmup_steps = 100
            expert_affinity = { anime = [0, 1], realism = [2, 3] }
            """
        )
        cfg = load_config(path)
        policy = cfg.dataset.moe_routing
        self.assertEqual(policy.specialization_mode, "soft_affinity")
        self.assertEqual(policy.domain_metadata_key, "visual_domain")
        self.assertEqual(policy.expert_affinity["anime"], [0, 1])
        self.assertEqual(policy.routing_prior_weight, 0.05)
        self.assertEqual(policy.router_warmup_steps, 100)

    def test_lingbot_video_model_type_uses_moe_defaults(self) -> None:
        path = self._write(
            """
            preset = "lingbot_video"

            [model]
            type = "lingbot-video"
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.model.type, "lingbot-video")
        self.assertEqual(cfg.model.path, "./models/lingbot_video")
        self.assertEqual(cfg.model.params.variant, "lingbot-video-moe-30b-a3b")
        self.assertTrue(cfg.model.params.strict_native_assets)

    def test_lingbot_video_preset_targets_release_assets(self) -> None:
        cfg = load_config("mirai/config/presets/lingbot_video.toml")

        self.assertEqual(cfg.dataset.frame_buckets, [33])
        self.assertEqual(cfg.logging.sample_every_n_steps, 0)
        self.assertEqual(cfg.model.params.variant, "lingbot-video-moe-30b-a3b")
        self.assertTrue(cfg.model.params.strict_native_assets)

    def test_removed_moe_boundary_key_is_rejected(self) -> None:
        path = self._write(
            """
            preset = "lingbot_video"

            [model.params]
            moe_boundary = 0.5
            """
        )
        with self.assertRaisesRegex(ConfigError, "moe_boundary.*was removed"):
            load_config(path)

    def test_removed_allow_unassigned_domains_key_is_rejected(self) -> None:
        path = self._write(
            """
            preset = "sparse_moe_test"

            [dataset.moe_routing]
            allow_unassigned_domains = true
            """
        )
        with self.assertRaisesRegex(ConfigError, "allow_unassigned_domains.*was removed"):
            load_config(path)

    def test_native_moe_training_policy_parses_from_model_params(self) -> None:
        path = self._write(
            """
            preset = "lingbot_video"

            [model.params]
            moe_aux_loss_type = "sequence"
            moe_bias_update_rate = 0.001
            moe_bias_centering = false
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.model.params.moe_aux_loss_type, "sequence")
        self.assertEqual(cfg.model.params.moe_bias_update_rate, 0.001)
        self.assertFalse(cfg.model.params.moe_bias_centering)

    def test_expert_choice_routing_and_capacity_schedule_parse(self) -> None:
        path = self._write(
            """
            preset = "lingbot_video"

            [model.params]
            moe_routing_mode = "expert_choice"
            moe_expert_choice_capacity_factor = 1.25
            moe_expert_choice_coverage_alarm_threshold = 0.9
            moe_router_timestep_weight = 0.5

            [[model.params.moe_expert_choice_capacity_schedule]]
            start_step = 0
            end_step = 100
            first_layer = 0
            end_layer = 8
            capacity_factor = 1.5

            [[model.params.moe_expert_choice_capacity_schedule]]
            start_step = 100
            first_layer = 0
            end_layer = 8
            capacity_factor = 0.75
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.model.params.moe_routing_mode, "expert_choice")
        self.assertEqual(cfg.model.params.moe_expert_choice_capacity_factor, 1.25)
        self.assertEqual(
            cfg.model.params.moe_expert_choice_coverage_alarm_threshold,
            0.9,
        )
        self.assertEqual(cfg.model.params.moe_router_timestep_weight, 0.5)
        self.assertEqual(
            cfg.model.params.moe_expert_choice_capacity_schedule[1]["start_step"],
            100,
        )

    def test_decoupled_router_timestep_requires_expert_choice(self) -> None:
        path = self._write(
            """
            [model.params]
            moe_routing_mode = "token_choice"
            moe_router_timestep_weight = 0.5
            """
        )
        with self.assertRaisesRegex(ConfigError, "expert_choice"):
            load_config(path)

    def test_expert_choice_capacity_schedule_rejects_overlap(self) -> None:
        path = self._write(
            """
            preset = "lingbot_video"

            [model.params]
            moe_routing_mode = "expert_choice"

            [[model.params.moe_expert_choice_capacity_schedule]]
            start_step = 0
            end_step = 100
            first_layer = 0
            end_layer = 8
            capacity_factor = 1.5

            [[model.params.moe_expert_choice_capacity_schedule]]
            start_step = 50
            end_step = 150
            first_layer = 4
            end_layer = 12
            capacity_factor = 0.75
            """
        )
        with self.assertRaisesRegex(ConfigError, "must not overlap"):
            load_config(path)

    def test_expert_choice_capacity_schedule_rejects_later_increase(self) -> None:
        path = self._write(
            """
            preset = "lingbot_video"

            [model.params]
            moe_routing_mode = "expert_choice"

            [[model.params.moe_expert_choice_capacity_schedule]]
            start_step = 0
            end_step = 100
            first_layer = 0
            end_layer = 8
            capacity_factor = 1.0

            [[model.params.moe_expert_choice_capacity_schedule]]
            start_step = 100
            first_layer = 4
            end_layer = 12
            capacity_factor = 1.25
            """
        )
        with self.assertRaisesRegex(ConfigError, "monotone non-increasing"):
            load_config(path)

    def test_expert_choice_coverage_alarm_requires_expert_choice(self) -> None:
        path = self._write(
            """
            preset = "lingbot_video"

            [model.params]
            moe_expert_choice_coverage_alarm_threshold = 0.9
            """
        )
        with self.assertRaisesRegex(ConfigError, "requires moe_routing_mode"):
            load_config(path)

    def test_denoiser_subfolder_parses_from_model_params(self) -> None:
        path = self._write(
            """
            preset = "lingbot_video"

            [model.params]
            denoiser_subfolder = "refiner"
            """
        )
        cfg = load_config(path)
        self.assertEqual(cfg.model.params.denoiser_subfolder, "refiner")

    def test_training_policy_modules_parse_as_explicit_plugin_list(self) -> None:
        path = self._write(
            """
            [training]
            policy_modules = ["example.routing_policy", "example.loss_policy"]
            """
        )
        cfg = load_config(path)
        self.assertEqual(
            cfg.training.policy_modules,
            ["example.routing_policy", "example.loss_policy"],
        )

    def test_training_policy_modules_reject_non_list_value(self) -> None:
        path = self._write(
            """
            [training]
            policy_modules = "example.routing_policy"
            """
        )
        with self.assertRaisesRegex(ConfigError, "must be a list"):
            load_config(path)

    def test_training_policy_options_parse_as_namespaced_tables(self) -> None:
        path = self._write(
            """
            [training.policy_options.example_policy]
            enabled = true
            strength = 0.25
            """
        )
        cfg = load_config(path)
        self.assertEqual(
            cfg.training.policy_options,
            {"example_policy": {"enabled": True, "strength": 0.25}},
        )

    def test_training_policy_options_reject_scalar_namespace(self) -> None:
        path = self._write(
            """
            [training]
            policy_options = { example_policy = true }
            """
        )
        with self.assertRaisesRegex(ConfigError, "table of policy-name tables"):
            load_config(path)

    def test_control_characters_in_path_are_rejected(self) -> None:
        path = self._write(
            r"""
            [model]
            path = "model\nweights"
            """
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("model.path contains control characters", str(ctx.exception))

    def test_lightweight_expert_pool_defaults_are_disabled(self) -> None:
        cfg = load_config(self._write(""))
        self.assertEqual(cfg.model.params.moe_zero_experts, 0)
        self.assertEqual(cfg.model.params.moe_copy_experts, 0)
        self.assertEqual(cfg.model.params.moe_constant_experts, 0)
        self.assertEqual(cfg.model.params.moe_lightweight_top_k, 0)

    def test_lightweight_expert_pool_parses_explicit_logical_topk(self) -> None:
        cfg = load_config(
            self._write(
                """
                [model.params]
                num_experts = 4
                moe_zero_experts = 2
                moe_copy_experts = 1
                moe_constant_experts = 2
                moe_lightweight_top_k = 5
                """
            )
        )
        self.assertEqual(cfg.model.params.moe_zero_experts, 2)
        self.assertEqual(cfg.model.params.moe_copy_experts, 1)
        self.assertEqual(cfg.model.params.moe_constant_experts, 2)
        self.assertEqual(cfg.model.params.moe_lightweight_top_k, 5)

    def test_zero_expert_pool_requires_an_explicit_valid_topk(self) -> None:
        for top_k in (0, 7):
            with self.subTest(top_k=top_k):
                path = self._write(
                    f"""
                    [model.params]
                    num_experts = 4
                    moe_zero_experts = 2
                    moe_lightweight_top_k = {top_k}
                    """
                )
                with self.assertRaisesRegex(
                    ConfigError, "moe_lightweight_top_k"
                ):
                    load_config(path)

    def test_disabled_lightweight_expert_pool_rejects_nonzero_topk(self) -> None:
        path = self._write(
            """
            [model.params]
            moe_zero_experts = 0
            moe_lightweight_top_k = 1
            """
        )
        with self.assertRaisesRegex(ConfigError, "must be 0"):
            load_config(path)

    def test_each_lightweight_expert_count_must_be_nonnegative(self) -> None:
        for key in (
            "moe_zero_experts",
            "moe_copy_experts",
            "moe_constant_experts",
        ):
            with self.subTest(key=key):
                path = self._write(
                    f"""
                    [model.params]
                    {key} = -1
                    """
                )
                with self.assertRaisesRegex(ConfigError, key):
                    load_config(path)

    def test_zero_expert_pool_rejects_expert_choice(self) -> None:
        path = self._write(
            """
            [model.params]
            num_experts = 4
            moe_zero_experts = 1
            moe_lightweight_top_k = 2
            moe_routing_mode = "expert_choice"
            """
        )
        with self.assertRaisesRegex(ConfigError, "token_choice"):
            load_config(path)

    def test_shipped_presets_do_not_offload_params_away_from_paged_optimizer(
        self,
    ) -> None:
        """Every shipped preset must be startable, not just parseable.

        PagedAdamW8bit requires the trainable parameters to be resident on CUDA
        (see _build_paged_adamw_8bit). memory.trainable_parameter_offload moves
        them off it, so the pair is mutually exclusive — and with
        optimizer.allow_fallback disabled the mismatch is a hard failure at
        optimizer-build time, on every GPU. lingbot_video_offload shipped with
        exactly that pair: it parsed fine and could never start.
        """
        preset_dir = Path("mirai/config/presets")
        presets = sorted(preset_dir.glob("*.toml"))
        self.assertTrue(presets, "no presets found to check")
        for preset in presets:
            with self.subTest(preset=preset.name):
                cfg = load_config(str(preset))
                if cfg.optimizer.type != "paged_adamw_8bit":
                    continue
                if cfg.optimizer.allow_fallback:
                    continue
                self.assertFalse(
                    cfg.memory.trainable_parameter_offload,
                    f"{preset.name} pairs optimizer.type='paged_adamw_8bit' "
                    "(allow_fallback=false) with "
                    "memory.trainable_parameter_offload=true; PagedAdamW8bit "
                    "needs those params on CUDA, so this preset cannot start.",
                )

    def test_compression_provider_gates_parse_and_invalid_provider_fails(self) -> None:
        path = self._write(
            """
            [model.params]
            expert_pruning = "prune"
            expert_consolidation = "hierarchical_output"
            expert_weight_compression = "stun_sparse24"
            """
        )
        cfg = load_config(path)
        self.assertEqual(
            cfg.model.params.expert_consolidation,
            "hierarchical_output",
        )
        self.assertEqual(cfg.model.params.expert_weight_compression, "stun_sparse24")

        mixture = load_config(
            self._write(
                """
                [model.params]
                expert_weight_compression = "mixture_basis"
                """
            )
        )
        self.assertEqual(
            mixture.model.params.expert_weight_compression,
            "mixture_basis",
        )

        flexmoe = load_config(
            self._write(
                """
                [model.params]
                flexmoe_calibration = "nested"
                """
            )
        )
        self.assertEqual(flexmoe.model.params.flexmoe_calibration, "nested")
        self.assertEqual(flexmoe.model.params.expert_weight_compression, "off")

        with self.assertRaisesRegex(ConfigError, "complete source"):
            load_config(
                self._write(
                    """
                    [model.params]
                    flexmoe_calibration = "nested"
                    expert_weight_compression = "flexmoe_nested"
                    """
                )
            )

        with self.assertRaisesRegex(ConfigError, "token_choice"):
            load_config(
                self._write(
                    """
                    [model.params]
                    flexmoe_calibration = "nested"
                    moe_routing_mode = "expert_choice"
                    """
                )
            )

        invalid = self._write(
            """
            [model.params]
            expert_weight_compression = "not_a_provider"
            """
        )
        with self.assertRaisesRegex(ConfigError, "expert_weight_compression"):
            load_config(invalid)

    def test_drop_upcycling_requires_trainable_router_and_expert_lora(self) -> None:
        cfg = load_config(
            self._write(
                """
                [model.params]
                expert_upcycling = "drop"
                expert_upcycling_copies = 1
                expert_upcycling_reinit_ratio = 0.5
                expert_upcycling_seed = 123

                [adapter]
                type = "lora"
                target_preset = "attn_router_routed_experts"
                train_router = true

                [memory]
                frozen_weight_packed_state_path = "upcycled-base.safetensors"
                """
            )
        )
        self.assertEqual(cfg.model.params.expert_upcycling, "drop")
        self.assertEqual(cfg.model.params.expert_upcycling_copies, 1)
        self.assertEqual(cfg.model.params.expert_upcycling_seed, 123)

        with self.assertRaisesRegex(ConfigError, "train_router=true"):
            load_config(
                self._write(
                    """
                    [model.params]
                    expert_upcycling = "drop"
                    expert_upcycling_copies = 1

                    [adapter]
                    type = "lora"
                    target_preset = "attn_routed_experts"
                    train_router = false

                    [memory]
                    frozen_weight_packed_state_path = "upcycled-base.safetensors"
                    """
                )
            )

    def test_aimer_pruning_criterion_parses_and_unknown_criterion_fails(self) -> None:
        cfg = load_config(
            self._write(
                """
                [model.params]
                expert_pruning = "prune"
                expert_pruning_criterion = "aimer"
                """
            )
        )
        self.assertEqual(cfg.model.params.expert_pruning_criterion, "aimer")
        with self.assertRaisesRegex(ConfigError, "expert_pruning_criterion"):
            load_config(
                self._write(
                    """
                    [model.params]
                    expert_pruning_criterion = "unknown"
                    """
                )
            )

    def test_learned_expert_rotation_gate_parses_and_rejects_unknown(self) -> None:
        cfg = load_config(
            self._write(
                """
                [model.params]
                expert_quantization_rotation = "learned"
                """
            )
        )
        self.assertEqual(
            cfg.model.params.expert_quantization_rotation,
            "learned",
        )
        with self.assertRaisesRegex(
            ConfigError,
            "expert_quantization_rotation",
        ):
            load_config(
                self._write(
                    """
                    [model.params]
                    expert_quantization_rotation = "random"
                    """
                )
            )

    def test_post_compression_router_repair_config_contract(self) -> None:
        cfg = load_config(
            self._write(
                """
                [model.params]
                post_compression_router_repair = "router_kd"
                router_repair_artifact_path = "router-repair.safetensors"

                [memory]
                frozen_weight_packed_state_path = "compressed-base.safetensors"
                """
            )
        )
        self.assertEqual(
            cfg.model.params.post_compression_router_repair,
            "router_kd",
        )
        self.assertEqual(
            cfg.model.params.router_repair_artifact_path,
            "router-repair.safetensors",
        )

        with self.assertRaisesRegex(
            ConfigError,
            "router_repair_artifact_path requires",
        ):
            load_config(
                self._write(
                    """
                    [model.params]
                    router_repair_artifact_path = "router-repair.safetensors"
                    """
                )
            )

        with self.assertRaisesRegex(
            ConfigError,
            "post_compression_router_repair",
        ):
            load_config(
                self._write(
                    """
                    [model.params]
                    post_compression_router_repair = "unknown"
                    """
                )
            )

    def test_expert_factorization_calibration_config_contract(self) -> None:
        cfg = load_config(
            self._write(
                """
                [model.params]
                expert_weight_compression = "shared_basis"
                expert_factorization_calibration = "whitened"
                """
            )
        )
        self.assertEqual(
            cfg.model.params.expert_factorization_calibration,
            "whitened",
        )

        collection_cfg = load_config(
            self._write(
                """
                [model.params]
                expert_factorization_calibration = "whitened"
                """
            )
        )
        self.assertEqual(collection_cfg.model.params.expert_weight_compression, "off")

        with self.assertRaisesRegex(ConfigError, "evidence collection"):
            load_config(
                self._write(
                    """
                    [model.params]
                    expert_pruning = "prune"
                    expert_weight_compression = "stun_sparse24"
                    expert_factorization_calibration = "whitened"
                    """
                )
            )

        with self.assertRaisesRegex(
            ConfigError,
            "requires moe_routing_mode='token_choice'",
        ):
            load_config(
                self._write(
                    """
                    [model.params]
                    moe_routing_mode = "expert_choice"
                    expert_weight_compression = "shared_basis"
                    expert_factorization_calibration = "whitened"
                    """
                )
            )

        with self.assertRaisesRegex(
            ConfigError,
            "expert_factorization_calibration",
        ):
            load_config(
                self._write(
                    """
                    [model.params]
                    expert_weight_compression = "shared_basis"
                    expert_factorization_calibration = "unknown"
                    """
                )
            )

    def test_router_quantization_calibration_config_contract(self) -> None:
        collection = load_config(
            self._write(
                """
                [model.params]
                router_quantization_calibration = "eaquant"
                """
            )
        )
        self.assertEqual(
            collection.model.params.router_quantization_calibration,
            "eaquant",
        )
        self.assertEqual(collection.memory.router_quantization, "disabled")

        runtime = load_config(
            self._write(
                """
                [memory]
                router_quantization = "int8_per_channel"
                router_quantization_calibration_path = "router-scale.safetensors"
                """
            )
        )
        self.assertEqual(
            runtime.memory.router_quantization_calibration_path,
            "router-scale.safetensors",
        )

        with self.assertRaisesRegex(
            ConfigError,
            "router_quantization_calibration_path requires",
        ):
            load_config(
                self._write(
                    """
                    [memory]
                    router_quantization_calibration_path = "router-scale.safetensors"
                    """
                )
            )
        with self.assertRaisesRegex(
            ConfigError,
            "EAQuant evidence collection requires",
        ):
            load_config(
                self._write(
                    """
                    [model.params]
                    router_quantization_calibration = "eaquant"

                    [memory]
                    router_quantization = "int8_per_channel"
                    """
                )
            )
        with self.assertRaisesRegex(
            ConfigError,
            "router_quantization_calibration",
        ):
            load_config(
                self._write(
                    """
                    [model.params]
                    router_quantization_calibration = "unknown"
                    """
                )
            )

    def test_expert_precision_calibration_config_contract(self) -> None:
        collection = load_config(
            self._write(
                """
                [model.params]
                expert_precision_calibration = "imatrix"
                """
            )
        )
        self.assertEqual(
            collection.model.params.expert_precision_calibration,
            "imatrix",
        )
        runtime = load_config(
            self._write(
                """
                [memory]
                frozen_weight_quantization = "int8"
                expert_precision_plan_path = "expert-precision.json"
                """
            )
        )
        self.assertEqual(
            runtime.memory.expert_precision_plan_path,
            "expert-precision.json",
        )
        with self.assertRaisesRegex(
            ConfigError,
            "expert_precision_plan_path requires",
        ):
            load_config(
                self._write(
                    """
                    [memory]
                    expert_precision_plan_path = "expert-precision.json"
                    """
                )
            )
        with self.assertRaisesRegex(
            ConfigError,
            "Imatrix precision calibration requires",
        ):
            load_config(
                self._write(
                    """
                    [model.params]
                    expert_precision_calibration = "imatrix"

                    [memory]
                    frozen_weight_quantization = "int8"
                    """
                )
            )
        with self.assertRaisesRegex(ConfigError, "expert_precision_calibration"):
            load_config(
                self._write(
                    """
                    [model.params]
                    expert_precision_calibration = "unknown"
                    """
                )
            )

    def test_sonic_dispatch_preprocess_config_contract(self) -> None:
        cfg = load_config(
            self._write(
                """
                [memory]
                moe_dispatch_preprocess = "sonic"
                """
            )
        )
        self.assertEqual(cfg.memory.moe_dispatch_preprocess, "sonic")
        validate_training_runtime_config(cfg)

    def test_deepgemm_fp8_forward_config_contract(self) -> None:
        cfg = load_config(
            self._write(
                """
                [memory]
                frozen_weight_quantization = "fp8"
                moe_gemm_backend_forward = "deepgemm_fp8"
                """
            )
        )
        validate_training_runtime_config(cfg)

    def test_deepgemm_fp8_rejects_inherited_all_role_selection(self) -> None:
        cfg = load_config(
            self._write(
                """
                [memory]
                frozen_weight_quantization = "fp8"
                moe_gemm_backend = "deepgemm_fp8"
                """
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "forward-only frozen-expert backend",
        ):
            validate_training_runtime_config(cfg)

    def test_deepgemm_fp8_rejects_gradient_roles(self) -> None:
        for role in ("dx", "dw"):
            with self.subTest(role=role):
                cfg = load_config(
                    self._write(
                        f"""
                        [memory]
                        frozen_weight_quantization = "fp8"
                        moe_gemm_backend_{role} = "deepgemm_fp8"
                        """
                    )
                )
                with self.assertRaisesRegex(ValueError, "may only be selected"):
                    validate_training_runtime_config(cfg)

    def test_deepgemm_fp8_requires_fp8_quantization(self) -> None:
        cfg = load_config(
            self._write(
                """
                [memory]
                moe_gemm_backend_forward = "deepgemm_fp8"
                """
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires.*frozen_weight_quantization",
        ):
            validate_training_runtime_config(cfg)


if __name__ == "__main__":
    unittest.main()
