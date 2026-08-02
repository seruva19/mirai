from __future__ import annotations

import unittest

from mirai.config.schema import (
    ModelConfig,
    ModelParams,
    StrategyConfig,
    TrainingConfig,
    TrainingSection,
)
from mirai.core.builtins import register_builtin_components
from mirai.core.training.preview.preview import _prompt_embed, run_cfg_denoise_loop
from mirai.core.training.trainer import Trainer

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class SamplerParityTests(unittest.TestCase):
    def setUp(self) -> None:
        register_builtin_components()

    def _trainer(self) -> Trainer:
        cfg = TrainingConfig(
            model=ModelConfig(
                type="sparse_moe_test",
                params=ModelParams(variant="tiny-video", flow_shift=3.0, vae_chunk_size=16),
            ),
            strategy=StrategyConfig(type="text_to_video"),
            training=TrainingSection(seed=0, batch_size=1),
        )
        return Trainer(cfg)

    def test_euler_cfg_denoise_matches_reference_loop(self) -> None:
        trainer = self._trainer()
        pipe = trainer.pipeline
        seed = 123
        step = 5
        got_latent, got_stats = run_cfg_denoise_loop(
            pipeline=pipe,
            prompt="a cat walking",
            negative_prompt="blurry",
            cfg_scale=7.5,
            seed=seed,
            step=step,
            denoise_steps=6,
            scheduler="euler",
        )

        g = torch.Generator(device="cpu")
        g.manual_seed(seed + step)
        latent = torch.randn((1,), generator=g, dtype=torch.float32)
        dt = 1.0 / 6.0
        cond = {"t5": torch.tensor([_prompt_embed("a cat walking")], dtype=torch.float32)}
        uncond = {"t5": torch.tensor([_prompt_embed("blurry")], dtype=torch.float32)}
        ref_stats: list[dict[str, float]] = []
        with torch.no_grad():
            for idx in range(6):
                t = torch.tensor([1.0 - (idx / 6.0)], dtype=torch.float32)
                v_cond = pipe.forward(latent, t, cond)
                v_uncond = pipe.forward(latent, t, uncond)
                v = v_uncond + 7.5 * (v_cond - v_uncond)
                latent = latent - (v * dt)
                ref_stats.append(
                    {
                        "latent_mean": float(latent.mean().item()),
                        "latent_std": float(latent.std(unbiased=False).item()),
                    }
                )

        self.assertTrue(torch.allclose(got_latent, latent, atol=1e-7, rtol=0.0))
        for got, ref in zip(got_stats, ref_stats, strict=True):
            self.assertAlmostEqual(float(got["latent_mean"]), float(ref["latent_mean"]), places=7)
            self.assertAlmostEqual(float(got["latent_std"]), float(ref["latent_std"]), places=7)


if __name__ == "__main__":
    unittest.main()
