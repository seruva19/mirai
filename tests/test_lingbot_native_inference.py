"""Behavioral tests for LingBot-Video native inference hooks.

Covers the deliverable contract: embedding parity between the cache-encoding
training path and inference-time `encode_prompt` (bit-exact through the shared
code object), chunked-vs-whole VAE decode equivalence against the vendored
reference, latent geometry sourced from VAE compression ratios, fail-fast asset
remediation, and infer.py native-mode resolution.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mirai.config.schema import ModelConfig, ModelParams
from mirai.core.models.providers import NativeCacheEncoderConfig

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _model_config(path: str, *, vae_chunk_size: int = 16) -> ModelConfig:
    return ModelConfig(
        type="lingbot-video",
        path=path,
        dtype="fp32",
        params=ModelParams(
            variant="tiny-video",
            flow_shift=3.0,
            strict_native_assets=False,
            latent_channels=1,
            num_experts=4,
            experts_per_token=2,
            shared_experts=1,
            hidden_size=16,
            num_layers=1,
            attention_heads=2,
            patch_size=1,
            vae_chunk_size=vae_chunk_size,
        ),
    )


# --- fake Qwen3-VL processor/encoder for deterministic parity testing --------
class _FakeBatch(dict):
    def to(self, *args, **kwargs):  # BatchEncoding-like: device move is a no-op
        return self


class _FakeProcessor:
    def __call__(self, text=None, images=None, videos=None, return_tensors=None, **kwargs):
        source = text[0] if isinstance(text, list) else text
        ids = [(ord(char) % 250) + 1 for char in source] or [1]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return _FakeBatch(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


class _FakeTextEncoder(torch.nn.Module if torch is not None else object):
    def __init__(self, vocab: int = 260, dim: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        hidden = self.proj(self.embed(input_ids))
        return SimpleNamespace(hidden_states=(hidden * 0.5, hidden), last_hidden_state=hidden)


def _build_fake_assets(**kwargs):
    # Fixed seed => two independent loads produce numerically identical weights,
    # emulating both paths loading the same on-disk checkpoint.
    torch.manual_seed(12345)
    return _FakeProcessor(), _FakeTextEncoder().eval()


def _tiny_vae():
    torch.manual_seed(7)
    from mirai.vendors.lingbot_video.autoencoder_kl_wan import AutoencoderKLWan

    vae = AutoencoderKLWan(
        base_dim=4,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[True],
        latents_mean=[0.1, -0.2, 0.3, -0.4],
        latents_std=[1.0, 1.1, 1.2, 1.3],
        in_channels=3,
        out_channels=3,
        scale_factor_temporal=4,
        scale_factor_spatial=2,
    )
    return vae.eval()


@unittest.skipIf(torch is None, "torch not installed")
class LingBotNativeEmbeddingParityTests(unittest.TestCase):
    def test_encode_prompt_matches_cache_path_bit_exact(self) -> None:
        from mirai.core.models.lingbot_video.cache import LingBotVideoNativeCacheEncoder
        from mirai.core.models.lingbot_video.inference import LingBotVideoNativeInference

        caption = "a red fox runs across a snowy field at dawn"
        with tempfile.TemporaryDirectory() as tmp, patch(
            "torch.cuda.is_available", return_value=False
        ), patch(
            "mirai.core.models.lingbot_video.text_encoder.load_qwen3vl_text_assets",
            side_effect=_build_fake_assets,
        ):
            cache_encoder = LingBotVideoNativeCacheEncoder(
                NativeCacheEncoderConfig(
                    enabled=True,
                    model_type="lingbot-video",
                    variant="tiny-video",
                    model_path=tmp,
                    dtype_name="fp32",
                    max_frames=9,
                )
            )
            embed_cache = cache_encoder.encode_text(caption)

            inference = LingBotVideoNativeInference(_model_config(tmp))
            embed_infer = inference.encode_prompt(caption, device="cpu")

        # Same shape/dtype/layout and bit-identical values: the inference path
        # produces exactly what the cached-latent training path stored.
        self.assertEqual(tuple(embed_cache.shape), tuple(embed_infer.shape))
        self.assertEqual(embed_cache.dtype, torch.float32)
        self.assertEqual(embed_infer.dtype, torch.float32)
        self.assertTrue(torch.equal(embed_cache, embed_infer))

    def test_both_paths_route_through_shared_embedding_code_object(self) -> None:
        from mirai.core.models.lingbot_video import text_encoder as te
        from mirai.core.models.lingbot_video.cache import LingBotVideoNativeCacheEncoder
        from mirai.core.models.lingbot_video.inference import LingBotVideoNativeInference

        # The single owner of the embedding math is encode_prompt_embedding, called
        # by LingBotVideoTextEncoder.encode, which both paths hold.
        self.assertIn(
            "encode_prompt_embedding",
            te.LingBotVideoTextEncoder.encode.__code__.co_names,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_encoder = LingBotVideoNativeCacheEncoder(
                NativeCacheEncoderConfig(
                    enabled=True,
                    model_type="lingbot-video",
                    variant="tiny-video",
                    model_path=tmp,
                    dtype_name="fp32",
                    max_frames=9,
                )
            )
            inference = LingBotVideoNativeInference(_model_config(tmp))
        self.assertIsInstance(cache_encoder._text_backend, te.LingBotVideoTextEncoder)
        self.assertIsInstance(inference.text_backend, te.LingBotVideoTextEncoder)

    def test_multimodal_prompt_keeps_vision_tokens_in_the_embedding(self) -> None:
        from PIL import Image

        from mirai.core.models.lingbot_video.inference import LingBotVideoNativeInference

        with tempfile.TemporaryDirectory() as tmp, patch(
            "mirai.core.models.lingbot_video.text_encoder.load_qwen3vl_text_assets",
            side_effect=_build_fake_assets,
        ):
            inference = LingBotVideoNativeInference(_model_config(tmp))
            text_only = inference.encode_prompt("move forward", device="cpu")
            multimodal = inference.encode_prompt(
                "move forward",
                device="cpu",
                image=Image.new("RGB", (32, 24), "red"),
            )
        self.assertGreater(int(multimodal.shape[0]), int(text_only.shape[0]))


@unittest.skipIf(torch is None, "torch not installed")
class LingBotNativeVaeDecodeTests(unittest.TestCase):
    def _inference_with_vae(self, tmp: str, vae, *, vae_chunk_size: int):
        from mirai.core.models.lingbot_video.inference import LingBotVideoNativeInference

        inference = LingBotVideoNativeInference(_model_config(tmp, vae_chunk_size=vae_chunk_size))
        with patch(
            "mirai.core.models.lingbot_video.inference.load_lingbot_native_vae",
            return_value=vae,
        ):
            inference.load_vae(device="cpu")
        return inference

    def test_chunked_decode_matches_whole_and_vendored_reference(self) -> None:
        from mirai.core.models.lingbot_video.vae import dit_latent_to_vae_latent

        vae = _tiny_vae()
        torch.manual_seed(3)
        dit_latent = torch.randn(4, 3, 8, 8)  # [C, T_lat, H, W] in DiT space

        with tempfile.TemporaryDirectory() as tmp:
            whole = self._inference_with_vae(tmp, vae, vae_chunk_size=64)
            chunked = self._inference_with_vae(tmp, vae, vae_chunk_size=1)
            frames_whole = whole.decode_latents_native([dit_latent])
            frames_chunked = chunked.decode_latents_native([dit_latent])

        # chunk_size only bounds host-offload grouping, never the values.
        self.assertTrue(torch.equal(frames_whole, frames_chunked))

        # Reference oracle: vendored AutoencoderKLWan.decode on the same VAE latent.
        vae_latent = dit_latent_to_vae_latent(dit_latent.unsqueeze(0), vae.config)
        with torch.no_grad():
            reference = vae.decode(vae_latent).sample
        reference_frames = reference[0].permute(1, 0, 2, 3).add(1.0).mul(0.5).clamp(0.0, 1.0)
        self.assertTrue(torch.allclose(frames_whole, reference_frames, atol=1e-6, rtol=0.0))

    def test_decode_output_shape_and_range_contract(self) -> None:
        vae = _tiny_vae()
        torch.manual_seed(11)
        dit_latent = torch.randn(4, 3, 8, 8)
        with tempfile.TemporaryDirectory() as tmp:
            inference = self._inference_with_vae(tmp, vae, vae_chunk_size=2)
            frames = inference.decode_latents_native([dit_latent])
        # [T, 3, H*, W*] with pixels in [0, 1]; T_lat=3 -> 1 + 2*2 = 5 frames.
        self.assertEqual(frames.ndim, 4)
        self.assertEqual(int(frames.shape[1]), 3)
        self.assertEqual(int(frames.shape[0]), 5)
        self.assertGreaterEqual(float(frames.min()), 0.0)
        self.assertLessEqual(float(frames.max()), 1.0)

    def test_single_frame_image_decode(self) -> None:
        vae = _tiny_vae()
        torch.manual_seed(5)
        dit_latent = torch.randn(4, 1, 8, 8)  # single latent frame (image)
        with tempfile.TemporaryDirectory() as tmp:
            inference = self._inference_with_vae(tmp, vae, vae_chunk_size=16)
            frames = inference.decode_latents_native([dit_latent])
        self.assertEqual(int(frames.shape[0]), 1)
        self.assertEqual(int(frames.shape[1]), 3)

    def test_decode_before_load_vae_fails(self) -> None:
        from mirai.core.models.lingbot_video.inference import LingBotVideoNativeInference

        with tempfile.TemporaryDirectory() as tmp:
            inference = LingBotVideoNativeInference(_model_config(tmp))
            with self.assertRaisesRegex(RuntimeError, "VAE is not loaded"):
                inference.decode_latents_native([torch.randn(4, 1, 4, 4)])


@unittest.skipIf(torch is None, "torch not installed")
class LingBotConditioningTests(unittest.TestCase):
    def test_ti2v_prepares_first_frame_latent_and_prompt_image(self) -> None:
        from PIL import Image

        from mirai.core.inference.conditioning import (
            IMAGE_TO_VIDEO,
            InferenceConditioningRequest,
        )
        from mirai.core.models.lingbot_video.inference import LingBotVideoNativeInference

        vae = _tiny_vae()
        with tempfile.TemporaryDirectory() as tmp:
            inference = LingBotVideoNativeInference(_model_config(tmp))
            with patch(
                "mirai.core.models.lingbot_video.inference.load_lingbot_native_vae",
                return_value=vae,
            ):
                prepared = inference.prepare_conditioning(
                    InferenceConditioningRequest(
                        task=IMAGE_TO_VIDEO,
                        input_image=Image.new("RGB", (40, 20), "blue"),
                        frame_count=5,
                        height=16,
                        width=16,
                    ),
                    device="cpu",
                    generator=torch.Generator(device="cpu").manual_seed(9),
                )
        self.assertEqual(prepared.task, IMAGE_TO_VIDEO)
        self.assertEqual(prepared.prompt_media.size, (16, 16))
        self.assertEqual(tuple(prepared.condition_latent.shape[:3]), (1, 4, 1))

    def test_v2v_uses_deterministic_posterior_mode(self) -> None:
        from mirai.core.inference.conditioning import (
            VIDEO_TO_VIDEO,
            InferenceConditioningRequest,
        )
        from mirai.core.models.lingbot_video.inference import LingBotVideoNativeInference

        vae = _tiny_vae()
        video = torch.rand(3, 5, 16, 16)
        request = InferenceConditioningRequest(
            task=VIDEO_TO_VIDEO,
            input_video=video,
            denoising_strength=0.5,
            frame_count=5,
            height=16,
            width=16,
        )
        with tempfile.TemporaryDirectory() as tmp:
            inference = LingBotVideoNativeInference(_model_config(tmp))
            with patch(
                "mirai.core.models.lingbot_video.inference.load_lingbot_native_vae",
                return_value=vae,
            ):
                first = inference.prepare_conditioning(
                    request,
                    device="cpu",
                    generator=torch.Generator(device="cpu").manual_seed(1),
                )
                second = inference.prepare_conditioning(
                    request,
                    device="cpu",
                    generator=torch.Generator(device="cpu").manual_seed(2),
                )
        self.assertTrue(torch.equal(first.source_latent, second.source_latent))


@unittest.skipIf(torch is None, "torch not installed")
class LingBotPreviewGeometryTests(unittest.TestCase):
    def test_geometry_follows_vae_compression_ratios(self) -> None:
        from mirai.core.models.native_video import VideoLatentLayout
        from mirai.vendors.lingbot_video.autoencoder_kl_wan import AutoencoderKLWan

        vae = AutoencoderKLWan(
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            temperal_downsample=[False, True, True],
            latents_mean=[0.0] * 16,
            latents_std=[1.0] * 16,
            scale_factor_temporal=4,
            scale_factor_spatial=8,
        )
        cfg = vae.config
        # Source the ratios from config, not hardcoded literals in the assertion.
        temporal = int(cfg.scale_factor_temporal)
        spatial = int(cfg.scale_factor_spatial)
        channels = int(cfg.z_dim)
        layout = VideoLatentLayout(
            latent_channels=channels,
            temporal_downsample=temporal,
            spatial_downsample=spatial,
            frame_count_modulus=4,
            frame_count_remainder=1,
            request_spatial_multiple=16,
        )
        # 33 -> (33-1)/4+1 = 9 latent frames; 512/8 = 64 latent pixels.
        self.assertEqual(
            layout.preview_geometry(frame_count=33, height=512, width=512),
            (channels, 9, 64, 64),
        )
        # single image: 1 -> 1 latent frame.
        self.assertEqual(
            layout.preview_geometry(frame_count=1, height=512, width=512),
            (channels, 1, 64, 64),
        )

    def test_pipeline_layout_matches_vendored_vae_ratios(self) -> None:
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline
        from mirai.vendors.lingbot_video.autoencoder_kl_wan import AutoencoderKLWan

        vae_defaults = AutoencoderKLWan(latents_mean=[0.0] * 16, latents_std=[1.0] * 16)
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = LingBotVideoPipeline(_model_config(tmp))
        layout = pipeline.get_video_latent_layout()
        # The pipeline's compression ratios are the vendored Wan VAE's declared
        # scale factors, not arbitrary constants.
        self.assertEqual(
            int(layout.temporal_downsample),
            int(vae_defaults.config.scale_factor_temporal),
        )
        self.assertEqual(
            int(layout.spatial_downsample),
            int(vae_defaults.config.scale_factor_spatial),
        )


@unittest.skipIf(torch is None, "torch not installed")
class LingBotNativeFailFastTests(unittest.TestCase):
    def test_missing_text_encoder_fails_with_hf_remediation(self) -> None:
        from mirai.core.models.lingbot_video.inference import LingBotVideoNativeInference

        with tempfile.TemporaryDirectory() as tmp:
            inference = LingBotVideoNativeInference(_model_config(tmp))
            with self.assertRaises(FileNotFoundError) as ctx:
                inference.load_text_encoder(device="cpu")
        message = str(ctx.exception)
        self.assertIn("text_encoder", message)
        self.assertIn("hf download", message)

    def test_missing_vae_fails_with_hf_remediation(self) -> None:
        from mirai.core.models.lingbot_video.inference import LingBotVideoNativeInference

        with tempfile.TemporaryDirectory() as tmp:
            inference = LingBotVideoNativeInference(_model_config(tmp))
            with self.assertRaises(FileNotFoundError) as ctx:
                inference.load_vae(device="cpu")
        message = str(ctx.exception)
        self.assertIn("vae", message.lower())
        self.assertIn("hf download", message)

    def test_missing_packed_state_names_export_script(self) -> None:
        from mirai.core.models.compressed_weights import read_compressed_weights_packed_state_manifest

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "transformer-compressed_weights.safetensors.index.json"
            with self.assertRaises(FileNotFoundError) as ctx:
                read_compressed_weights_packed_state_manifest(missing)
        self.assertIn("export_compressed_weights_packed_state.py", str(ctx.exception))


@unittest.skipIf(torch is None, "torch not installed")
class LingBotInferModeResolutionTests(unittest.TestCase):
    def test_resolve_inference_mode_native_for_pipeline_with_hooks(self) -> None:
        from mirai.core.inference.session import resolve_inference_mode

        class _StubNative:
            def has_native_inference(self) -> bool:
                return True

        class _StubUnsupported:
            def has_native_inference(self) -> bool:
                return False

            def supports_validation_inference(self) -> bool:
                return False

        self.assertEqual(resolve_inference_mode(_StubNative()), "native")
        with self.assertRaisesRegex(RuntimeError, "does not expose native inference"):
            resolve_inference_mode(_StubUnsupported())

    def test_real_lingbot_pipeline_resolves_native(self) -> None:
        from mirai.core.inference.session import resolve_inference_mode
        from mirai.core.models.lingbot_video.pipeline import LingBotVideoPipeline

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = LingBotVideoPipeline(_model_config(tmp))
        self.assertTrue(pipeline.has_native_inference())
        self.assertEqual(resolve_inference_mode(pipeline), "native")


if __name__ == "__main__":
    unittest.main()
