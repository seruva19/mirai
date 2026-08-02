from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from mirai.config.schema import ConfigError, TrainingConfig
from mirai.config.loader import load_config
from mirai.core.inference.cfg_batching import (
    build_batched_cfg_inputs,
    split_batched_cfg_output,
)


class CfgBatchingContractTests(unittest.TestCase):
    def test_variable_length_contexts_are_padded_with_prefix_masks(self) -> None:
        sample = torch.randn(1, 2, 3)
        timestep = torch.tensor([500.0])
        cond = torch.randn(1, 3, 4)
        uncond = torch.randn(1, 1, 4)
        batch = build_batched_cfg_inputs(
            sample=sample,
            timestep=timestep,
            conditional=cond,
            unconditional=uncond,
        )
        self.assertEqual(tuple(batch.sample.shape), (2, 2, 3))
        self.assertEqual(tuple(batch.timestep.shape), (2,))
        self.assertEqual(tuple(batch.text_embeds["t5"].shape), (2, 3, 4))
        self.assertEqual(
            batch.text_embeds["text_mask"].tolist(),
            [[True, False, False], [True, True, True]],
        )
        self.assertTrue(torch.equal(batch.text_embeds["t5"][1:2], cond))
        self.assertTrue(torch.equal(batch.text_embeds["t5"][0:1, :1], uncond))

    def test_output_split_is_unconditional_then_conditional(self) -> None:
        output = torch.tensor([[1.0], [3.0]])
        uncond, cond = split_batched_cfg_output(output)
        self.assertEqual(uncond.tolist(), [[1.0]])
        self.assertEqual(cond.tolist(), [[3.0]])
        with self.assertRaisesRegex(RuntimeError, "batch size 2"):
            split_batched_cfg_output(torch.ones(1, 2))

    def test_config_defaults_sequential_and_rejects_unknown(self) -> None:
        self.assertEqual(TrainingConfig.from_dict({}).inference.cfg_mode, "sequential")
        self.assertEqual(
            TrainingConfig.from_dict({"inference": {"cfg_mode": "batched"}})
            .inference.cfg_mode,
            "batched",
        )
        with self.assertRaises(ConfigError):
            TrainingConfig.from_dict({"inference": {"cfg_mode": "parallelish"}})

    def test_bf16_resident_inference_preset_is_explicit(self) -> None:
        cfg = load_config("mirai/config/presets/lingbot_video_inference_bf16.toml")
        self.assertEqual(cfg.model.dtype, "bf16")
        self.assertEqual(cfg.memory.frozen_weight_quantization, "none")
        self.assertEqual(cfg.memory.weight_residency_strategy, "disabled")
        self.assertEqual(cfg.inference.cfg_mode, "batched")
        self.assertTrue(cfg.inference.keep_text_encoder_resident)
        self.assertTrue(cfg.inference.keep_vae_resident)

    def test_variable_length_masked_forward_matches_sequential_cfg(self) -> None:
        from mirai.core.training.preview.preview import run_native_denoise_loop

        class MaskAwarePipeline:
            def __init__(self) -> None:
                self.anchor = torch.nn.Linear(1, 1, bias=False)
                self.model_config = SimpleNamespace(
                    params=SimpleNamespace(flow_shift=1.0)
                )

            def get_training_model(self):
                return self.anchor

            def load_text_encoder(self, *, device: str) -> None:
                _ = device

            def offload_text_encoder(self) -> None:
                return None

            def encode_prompt(self, prompt: str, *, device: str):
                length = 3 if prompt == "conditional" else 1
                return torch.arange(1, length + 1, dtype=torch.float32).reshape(
                    1, length, 1
                ).to(device)

            def preview_latent_geometry(self, **_kwargs):
                return (1, 1, 1, 1)

            def resolve_flow_shift_for_latent_shape(self, _latent_shape):
                return self.model_config.params.flow_shift

            def forward(self, sample, timestep, text_embeds):
                _ = timestep
                context = text_embeds["t5"]
                mask = text_embeds.get("text_mask")
                if mask is None:
                    value = context.mean(dim=(1, 2))
                else:
                    weighted = context * mask.unsqueeze(-1)
                    value = weighted.sum(dim=(1, 2)) / mask.sum(dim=1)
                return sample * 0.1 + value.reshape(-1, 1, 1, 1, 1)

        kwargs = dict(
            pipeline=MaskAwarePipeline(),
            prompt="conditional",
            negative_prompt="unconditional",
            cfg_scale=3.0,
            seed=5,
            step=0,
            denoise_steps=2,
            scheduler="euler",
            frame_count=1,
            height=8,
            width=8,
        )
        sequential, _ = run_native_denoise_loop(cfg_mode="sequential", **kwargs)
        batched, _ = run_native_denoise_loop(cfg_mode="batched", **kwargs)
        self.assertTrue(torch.allclose(sequential, batched, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
