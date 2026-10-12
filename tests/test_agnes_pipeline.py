"""Joint Agnes adapter training and persistence contracts."""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from mirai.core.models.agnes_video.assets import AgnesAssets
from mirai.core.models.agnes_video.pipeline import AgnesVideoPipeline
from mirai.core.models.agnes_video.pipeline import AgnesVideoModelFamilyProvider
from mirai.core.models.agnes_video.pipeline import agnes_default_negative_prompt
from mirai.vendors.agnes_video.dit import AgnesDiT, AgnesDiTConfig


def pipeline():
    config = AgnesDiTConfig(num_layers=2, hidden_size=16, head_dim=8, num_query_groups=1,
                           video_in_channels=8, audio_in_channels=4, text_in_channels=12,
                           mm_layers=(0,), sparse_moe_layers=(1,), gelu7_layers=(0,))
    model = SimpleNamespace(params=SimpleNamespace(family_params={}))
    result = AgnesVideoPipeline(model, transformer=AgnesDiT(config))
    result.set_adapter_config(SimpleNamespace(type="lora", target_preset="attn_mlp", rank=2,
                                             alpha=2, train_router=False, init="kaiming"))
    return result


def test_family_options_fail_explicitly_before_runtime():
    provider = AgnesVideoModelFamilyProvider()
    for options in ({"fps": 0}, {"fps": float("nan")}, {"audio_guidance_scale": -1},
                    {"allow_missing_audio": "false"}, {"video_vae_path": 12}):
        assert provider.validate_family_params(options)
    assert not provider.validate_family_params({"fps": 30.0, "audio_guidance_scale": 0,
                                                "allow_missing_audio": False})


def test_generation_defaults_use_released_negative_prompt_and_preserve_override():
    defaults = AgnesVideoModelFamilyProvider().generation_defaults()
    released = agnes_default_negative_prompt()
    assert len(released) == 1686
    assert hashlib.sha256(released.encode("utf-8")).hexdigest() == (
        "7063dc76af5ef57dfb81bdcfdcdf1609b997b932e9d08d2910973b2883845a34"
    )
    assert defaults.resolve_negative_prompt(None) == released
    assert defaults.resolve_negative_prompt("custom negative") == "custom negative"
    assert defaults.resolve_negative_prompt("") == ""


def test_zero_token_prompt_fails_before_t5gemma_forward():
    class Tokens(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        def __call__(self, prompts, **_kwargs):
            assert prompts == [""]
            return Tokens(input_ids=torch.empty((1, 0), dtype=torch.long))

    class Encoder:
        def __call__(self, **_tokens):
            raise AssertionError("zero-token input reached T5Gemma")

    assets = AgnesAssets({})
    assets.tokenizer = Tokenizer()
    assets.text_encoder = Encoder()
    assets.load_text_encoder = lambda _device="cpu": assets.text_encoder
    with pytest.raises(ValueError, match="zero tokens.*released Agnes default"):
        assets.encode_prompt("")


def test_prompt_is_trimmed_after_t5gemma_encoding():
    class Tokens(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        def __call__(self, prompts, **kwargs):
            assert prompts == ["long prompt"]
            assert kwargs == {"return_tensors": "pt"}
            return Tokens(input_ids=torch.arange(700).unsqueeze(0))

    class Output:
        last_hidden_state = torch.arange(700).reshape(1, 700, 1).expand(-1, -1, 3584)

    class Encoder:
        def __call__(self, **tokens):
            assert tokens["input_ids"].shape == (1, 700)
            return Output()

    assets = AgnesAssets({})
    assets.tokenizer = Tokenizer()
    assets.text_encoder = Encoder()
    assets.load_text_encoder = lambda _device="cpu": assets.text_encoder
    hidden = assets.encode_prompt("long prompt")
    assert hidden.shape == (1, 640, 3584)
    assert hidden[0, -1, 0] == 639


def test_prompt_hidden_states_use_released_fp16_quantization():
    class Tokens(dict):
        def to(self, _device):
            return self

    assets = AgnesAssets({})
    assets.tokenizer = lambda _prompts, **_kwargs: Tokens(input_ids=torch.ones(1, 1, dtype=torch.long))
    source = torch.tensor([[[1.0003]]], dtype=torch.float32).expand(1, 1, 3584)
    assets.text_encoder = lambda **_tokens: SimpleNamespace(last_hidden_state=source)
    assets.load_text_encoder = lambda _device="cpu": assets.text_encoder
    hidden = assets.encode_prompt("prompt")
    assert hidden.dtype is torch.float16
    assert hidden[0, 0, 0] == source.half()[0, 0, 0]


def test_native_assets_inherit_pipeline_compute_dtype():
    model = pipeline()
    model.model_config.path = ""
    model.set_compute_autocast_dtype(torch.float32)
    assert model._native_assets().weight_dtype is torch.float32
    assert model._native_assets().text_weight_dtype is torch.float32


def test_inference_runtime_casts_frozen_base_and_buffers_without_casting_lora():
    model = pipeline().eval()
    training = pipeline()
    original_training = {name: value.dtype for name, value in training.transformer.state_dict().items()}
    trainable = {name: (value.dtype, value.detach().clone())
                 for name, value in model.transformer.named_parameters() if value.requires_grad}
    assert trainable and all(dtype == torch.float32 for dtype, _ in trainable.values())
    assert model.transformer.adapter.rope.bands.dtype == torch.float32
    alpha_buffers = {name: value.detach().clone()
                     for name, value in model.transformer.named_buffers()
                     if name.endswith("lora_alpha")}
    assert alpha_buffers
    for name, value in model.transformer.named_buffers():
        if name in alpha_buffers:
            value.fill_(1.003)
            alpha_buffers[name] = value.detach().clone()
    with torch.no_grad():
        model.transformer.final_norm_video.weight.fill_(0.00391)
    expected = model.transformer.final_norm_video.weight.detach().to(torch.bfloat16)
    model.prepare_inference_runtime()
    assert all(value.dtype == torch.bfloat16 for value in model.transformer.parameters()
               if not value.requires_grad and value.is_floating_point())
    assert all(value.dtype == torch.bfloat16 for name, value in model.transformer.named_buffers()
               if value.is_floating_point() and not name.endswith("lora_alpha"))
    for name, value in model.transformer.named_buffers():
        if name in alpha_buffers:
            assert value.dtype == torch.float32
            torch.testing.assert_close(value, alpha_buffers[name], rtol=0, atol=0)
    torch.testing.assert_close(model.transformer.final_norm_video.weight, expected, rtol=0, atol=0)
    for name, value in model.transformer.named_parameters():
        if name in trainable:
            dtype, before = trainable[name]
            assert value.dtype == dtype
            torch.testing.assert_close(value, before, rtol=0, atol=0)
    assert {name: value.dtype for name, value in training.transformer.state_dict().items()} == original_training
    assert training.transformer.adapter.rope.bands.dtype == torch.float32
    assert model.transformer.block.layers[1].mlp.released_inference_grouped
    assert not training.transformer.block.layers[1].mlp.released_inference_grouped
    with torch.no_grad():
        assert model.transformer.block.layers[1].mlp._can_use_inference_bmm()
        assert not training.transformer.block.layers[1].mlp._can_use_inference_bmm()
    assert not model.transformer.block.layers[1].mlp._can_use_inference_bmm()


def test_standalone_assets_preserve_fp32_cache_default():
    assets = AgnesAssets({})
    assert assets.weight_dtype is torch.float32
    assert assets.text_weight_dtype is torch.bfloat16


def test_video_decode_uses_released_compute_dtype():
    model = pipeline()
    model.set_compute_autocast_dtype(torch.bfloat16)
    model._decode_device = "cpu"
    seen = []

    class VAE:
        def decode(self, latent):
            seen.append(latent.dtype)
            return torch.zeros(1, 3, 1, 2, 2, dtype=latent.dtype)

    model._assets = SimpleNamespace(load_video_vae=lambda _device: VAE())
    decoded = model.decode_latents_native([torch.ones(48, 1, 1, 1)])
    assert seen == [torch.bfloat16]
    assert decoded.dtype is torch.float32


def test_joint_lora_gradients_and_adapter_roundtrip():
    torch.manual_seed(19)
    model = pipeline()
    video = torch.randn(2, 2, 1, 2, 2)
    audio = torch.randn(2, 3, 4)
    clean = torch.cat((video.flatten(1), audio.flatten(1)), 1)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.3, 0.7])
    output = model(model.apply_noise(clean, noise, sigma), sigma,
                   {"t5": torch.randn(2, 3, 12)}, video_shape=video.shape,
                   audio_shape=audio.shape, audio_lengths=torch.tensor([2, 3]))
    assert output.shape == clean.shape
    loss = (output - model.compute_target(noise, clean, sigma)).square().mean()
    loss.backward()
    grads = dict(model.get_named_trainable_parameters())
    assert all(p.grad is None for n, p in model.transformer.named_parameters() if n not in grads)
    assert any("linear_qkv" in n and p.grad is not None and p.grad.abs().sum() > 0 for n, p in grads.items())
    assert any("experts" in n and "lora_b" in n and p.grad is not None and p.grad.abs().sum() > 0 for n, p in grads.items())
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    torch.manual_seed(19)
    other = pipeline()
    other.load_adapter_state(state)
    for key, value in other.state_dict().items():
        torch.testing.assert_close(value, state[key])


def test_joint_input_corruption_uses_shared_sigma_and_padding_mask():
    model = pipeline()
    sigma = torch.tensor([0.25, 0.75])
    sampler = SimpleNamespace(sample=lambda count, like: sigma)
    generator = SimpleNamespace(sample=lambda value: torch.ones_like(value))
    batch = {"latents": torch.zeros(2, 2, 1, 2, 2), "audio_latents": torch.zeros(2, 3, 4),
             "audio_lengths": torch.tensor([1, 3]), "text_embeds": torch.ones(2, 2, 12)}
    inputs = model.prepare_training_inputs(batch, timestep_sampler=sampler, noise_generator=generator,
                                          objective=None, training=True)
    torch.testing.assert_close(inputs.noisy_latents, sigma[:, None].expand_as(inputs.noisy_latents))
    assert inputs.loss_mask.sum(1).tolist() == [12, 20]
    batch["loss_mask"] = torch.tensor([0.0, 1.0])
    masked = model.prepare_training_inputs(batch, timestep_sampler=sampler, noise_generator=generator,
                                          objective=None, training=True)
    assert masked.loss_mask.sum(1).tolist() == [0, 20]


def test_joint_i2v_pins_prefix_and_masks_video_without_corrupting_audio_or_batch():
    from mirai.config.schema import StrategyConfig
    from mirai.core.training.strategies.image_to_video import ImageToVideoStrategy

    model = pipeline()
    video = torch.full((2, 2, 2, 2, 2), 7.)
    audio = torch.full((2, 3, 4), 2.)
    sigma = torch.tensor([.25, .75])
    calls = []

    def noise(value):
        calls.append(tuple(value.shape))
        return torch.full_like(value, 1. if value.ndim == 5 else 3.)

    batch = {"latents": video, "audio_latents": audio, "audio_lengths": torch.tensor([1, 3]),
             "text_embeds": torch.ones(2, 2, 12)}
    strategy = ImageToVideoStrategy(StrategyConfig(type="image_to_video", params={"first_frame_conditioning_p": 1.}))
    inputs = strategy.prepare_inputs(batch, model, SimpleNamespace(sample=lambda count, like: sigma),
                                     SimpleNamespace(sample=noise))
    noisy_video = inputs.noisy_latents[:, :16].reshape_as(video)
    noisy_audio = inputs.noisy_latents[:, 16:].reshape_as(audio)
    torch.testing.assert_close(noisy_video[:, :, 0], video[:, :, 0], rtol=0, atol=0)
    torch.testing.assert_close(noisy_video[:, :, 1], torch.tensor([5.5, 2.5])[:, None, None, None].expand(2, 2, 2, 2))
    torch.testing.assert_close(noisy_audio, torch.tensor([2.25, 2.75])[:, None, None].expand_as(audio))
    assert inputs.loss_mask[:, :16].reshape_as(video)[:, :, 0].count_nonzero() == 0
    assert inputs.loss_mask.sum(1).tolist() == [12, 20]
    assert inputs.extra_forward_kwargs["condition_frame_indexes"] == [[0], [0]]
    assert calls == [tuple(video.shape), tuple(audio.shape)]
    assert torch.all(video == 7) and torch.all(audio == 2)
    assert "clip_embed" not in model.get_required_batch_keys(strategy_type="image_to_video")
    prediction = torch.zeros_like(inputs.clean_latents, requires_grad=True)
    target = model.compute_target(inputs.noise, inputs.clean_latents, inputs.timestep)
    strategy.compute_per_sample_loss(prediction, target, inputs, lambda pred, truth: (pred-truth).square()).sum().backward()
    assert prediction.grad[:, :16].reshape_as(video)[:, :, 0].count_nonzero() == 0
    assert prediction.grad[1, 16:].abs().sum() > 0
    output = model(inputs.noisy_latents, inputs.timestep, inputs.text_embeds, **inputs.extra_forward_kwargs)
    strategy.compute_per_sample_loss(output, target, inputs, lambda pred, truth: (pred-truth).square()).mean().backward()
    gradients = {name: parameter.grad for name, parameter in model.get_named_trainable_parameters()
                 if parameter.grad is not None}
    assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
    assert any("linear_qkv" in name and value.abs().sum() > 0 for name, value in gradients.items())
    assert any("experts" in name and "lora_b" in name and value.abs().sum() > 0 for name, value in gradients.items())


def test_joint_i2v_conditioning_probability_preserves_source_masks_and_eval_policy():
    from mirai.core.training.strategies.base import VideoTrainingConditioning

    model = pipeline()
    batch = {"latents": torch.ones(2, 2, 2, 2, 2), "audio_latents": torch.zeros(2, 3, 4),
             "audio_lengths": torch.tensor([3, 3]), "text_embeds": torch.ones(2, 2, 12),
             "loss_mask": torch.tensor([0., 1.])}
    sampler = SimpleNamespace(sample=lambda count, like: torch.full((count,), .5))
    noise = SimpleNamespace(sample=lambda value: torch.zeros_like(value))
    for probability, training, expected in ((0., True, [[], []]), (.5, False, [[0], [0]])):
        inputs = model.prepare_training_inputs(batch, timestep_sampler=sampler, noise_generator=noise,
                                               objective=None, training=training,
                                               conditioning=VideoTrainingConditioning(probability))
        assert inputs.extra_forward_kwargs["condition_frame_indexes"] == expected
        assert inputs.loss_mask[0].count_nonzero() == 0
        assert inputs.loss_mask[1, 16:].sum() == 12
        if probability == 0:
            assert torch.all(inputs.noisy_latents[:, :16] == .5)
        else:
            assert inputs.loss_mask[1, :16].sum() == 8
    torch.manual_seed(10)
    expected_selected = torch.rand(2) < .5
    torch.manual_seed(10)
    inputs = model.prepare_training_inputs(batch, timestep_sampler=sampler, noise_generator=noise,
                                           objective=None, training=True, conditioning=VideoTrainingConditioning(.5))
    assert inputs.extra_forward_kwargs["condition_frame_indexes"] == [[0] if value else [] for value in expected_selected.tolist()]


def test_joint_sampling_reproducible_and_both_modalities_evolve():
    model = pipeline().eval()
    model._assets = SimpleNamespace(audio_geometry=lambda: (30, 1, 4), offload_audio_vae=lambda: None)
    noise = torch.ones(2, 1, 2, 2)
    context = torch.randn(1, 2, 12)
    def sample():
        return model.sample_native_preview(noise=noise, context=context, context_null=context*0,
                                           denoise_steps=2, guidance_scale=2,
                                           generator=torch.Generator().manual_seed(21), solver_name="unipc")
    first, second = sample(), sample()
    torch.testing.assert_close(first.video, second.video)
    torch.testing.assert_close(first.audio, second.audio)
    assert first.audio.shape == (1, 4)
    assert not torch.equal(first.video, noise)
    assert torch.isfinite(first.video).all() and torch.isfinite(first.audio).all()


def test_adapter_backward_does_not_save_frozen_bf16_weights():
    model = pipeline()
    frozen_weights = {p.data_ptr() for p in model.transformer.parameters()
                      if not p.requires_grad and p.dtype == torch.bfloat16}
    seen = []
    def capture(value):
        if value.dtype == torch.bfloat16:
            seen.append(value.data_ptr())
        return value
    with torch.autograd.graph.saved_tensors_hooks(capture, lambda x: x):
        output = model(torch.randn(1, 2, 1, 2, 2), torch.tensor([0.5]),
                       {"t5": torch.randn(1, 2, 12)}, audio_latents=torch.randn(1, 3, 4))
        sum(value.float().square().mean() for value in output).backward()
    assert not frozen_weights.intersection(seen)


def test_native_conditioned_sampler_pins_image_and_freezes_supplied_audio():
    from mirai.core.inference.conditioning import InferenceConditioningRequest, PreparedInferenceConditioning
    model = pipeline().eval()
    model._assets = SimpleNamespace(audio_geometry=lambda: (30, 1, 4), offload_audio_vae=lambda: None)
    request = InferenceConditioningRequest(task="image_to_video", input_image="fixture.png",
                                          frame_count=5, height=32, width=32)
    fixed_audio = torch.randn(1, 4, 4)
    prepared = PreparedInferenceConditioning(task="image_to_video",
                                            condition_latent=torch.full((1, 2, 1, 2, 2), 7.0),
                                            forward_media={"audio_latents": fixed_audio, "freeze_audio": True})
    context = torch.randn(1, 2, 12)
    output = model.sample_native_conditioned_media(request=request, prepared=prepared,
                                                   noise=torch.randn(2, 2, 2, 2), context=context,
                                                   context_null=context*0, denoise_steps=2,
                                                   guidance_scale=2, generator=torch.Generator().manual_seed(8),
                                                   solver_name="unipc")
    torch.testing.assert_close(output.video[:, :1], prepared.condition_latent[0], rtol=0, atol=0)
    torch.testing.assert_close(output.audio, fixed_audio[0], rtol=0, atol=0)


def test_batched_cfg_matches_sequential_with_variable_prompt_lengths():
    model = pipeline().eval()
    model._assets = SimpleNamespace(audio_geometry=lambda: (30, 1, 4), offload_audio_vae=lambda: None)
    noise = torch.randn(2, 1, 2, 2)
    context, context_null = torch.randn(1, 5, 12), torch.randn(1, 2, 12)
    outputs = [model.sample_native_preview(noise=noise, context=context, context_null=context_null,
                                          denoise_steps=2, guidance_scale=5,
                                          generator=torch.Generator().manual_seed(21), solver_name="unipc",
                                          cfg_mode=mode) for mode in ("sequential", "batched")]
    torch.testing.assert_close(outputs[0].video, outputs[1].video, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0].audio, outputs[1].audio, rtol=0, atol=0)


def test_agnes_sampler_uses_released_grid_and_cfg1_fresh_noise():
    from mirai.core.models.agnes_video.sampler import (
        AgnesDDIMFlowSolver,
        AgnesFlowUniPCMultistepSolver,
    )

    guided = AgnesFlowUniPCMultistepSolver(shift=3.0)
    guided.set_timesteps(4, shift=3.0)
    base = torch.linspace(0.999, 0.0, 5)[:-1]
    expected = 3.0 * base / (1.0 + 2.0 * base)
    torch.testing.assert_close(guided.timesteps, expected, rtol=0, atol=1e-7)
    assert guided.timesteps[0] < 1.0

    # The source thresholds CFG using its integer timestep computed before the
    # sigma grid is cast to float32. Near 0.501, reconstructing it from the
    # normalized float32 sigma incorrectly selects the low-guidance branch.
    midpoint = float(torch.tensor(0.999, dtype=torch.float32).item()) / 2
    target = 0.501000005
    boundary_shift = target * (1 - midpoint) / (midpoint * (1 - target))
    boundary = AgnesFlowUniPCMultistepSolver(shift=boundary_shift)
    boundary.set_timesteps(32, shift=boundary_shift)
    assert boundary.source_timesteps[16].item() == 501
    assert int(float(boundary.timesteps[16]) * 1000) == 500
    boundary.set_begin_index(16)
    assert boundary.source_timesteps[0].item() == 501

    stochastic = AgnesDDIMFlowSolver(shift=3.0)
    stochastic.set_timesteps(4, shift=3.0)
    sample = torch.tensor([[0.25, -0.5]])
    velocity = torch.tensor([[0.1, 0.2]])
    actual_generator = torch.Generator().manual_seed(31)
    expected_generator = torch.Generator().manual_seed(31)
    fresh_noise = torch.randn(sample.shape, generator=expected_generator)
    current, following = stochastic._sigmas[:2]
    expected_sample = following * fresh_noise + (1 - following) * (
        sample - current * velocity
    )
    actual = stochastic.step(
        velocity, stochastic.timesteps[0], sample, generator=actual_generator
    ).prev_sample
    torch.testing.assert_close(actual, expected_sample, rtol=0, atol=0)


def test_agnes_61_frame_timeline_is_two_seconds_for_audio():
    from mirai.core.models.agnes_video.sampler import (
        agnes_audio_token_count,
        agnes_export_frame_count,
        agnes_timeline_seconds,
    )

    seconds = agnes_timeline_seconds(61, 30)
    assert seconds == 2.0
    assert agnes_export_frame_count(61) == 60
    assert agnes_export_frame_count(1) == 1
    assert round(seconds * 48_000 / 960) == 100
    assert round(seconds * 48_000) == 96_000
    assert agnes_audio_token_count(9, 30, 48_000, 960, conditioned=False) == 13
    assert agnes_audio_token_count(9, 30, 48_000, 960, conditioned=True) == 14
    boundary_fps = 4 * 48_000 / (960 * 1.49999)
    assert agnes_audio_token_count(5, boundary_fps, 48_000, 960, conditioned=False) == 1

    from mirai.core.models.native_video import JointMediaLatents

    assets = SimpleNamespace(
        offload_video_vae=lambda: None,
        decode_audio=lambda audio, device: torch.zeros(100_000),
        audio_geometry=lambda: (48_000, 960, 128),
    )
    decoder = SimpleNamespace(
        options={"fps": 30},
        _decode_device="cpu",
        _native_assets=lambda: assets,
        decode_latents_native=lambda latents: torch.zeros(61, 3, 2, 2),
    )
    decoded = AgnesVideoPipeline.decode_media_native(
        decoder,
        JointMediaLatents(
            video=torch.zeros(48, 16, 2, 2),
            audio=torch.zeros(100, 128),
            fps=30,
            sample_rate=48_000,
        ),
    )
    assert decoded.frames.shape[0] == 60
    assert decoded.audio.shape == (96_000,)
    assert decoded.sample_rate == 48_000


def test_five_second_request_preserves_source_geometry_and_export():
    from mirai.core.inference.conditioning import InferenceConditioningRequest, PreparedInferenceConditioning
    from mirai.core.models.native_video import JointMediaLatents

    model = pipeline().eval()
    layout = model.get_video_latent_layout()
    assert layout.preview_geometry(frame_count=151, height=256, width=448)[1:] == (38, 16, 28)
    model._assets = SimpleNamespace(audio_geometry=lambda: (48_000, 960, 4),
                                    offload_audio_vae=lambda: None)
    request = InferenceConditioningRequest(task="text_to_video", frame_count=151,
                                          height=256, width=448)
    audio = torch.ones(1, 250, 4)
    prepared = PreparedInferenceConditioning(task="text_to_video", forward_media={"audio_noise": audio})
    model._forward_native = lambda video, timestep, context, **kwargs: (
        torch.zeros_like(video), torch.ones_like(kwargs["audio_latents"])
    )
    output = model.sample_native_conditioned_media(
        request=request, prepared=prepared, noise=torch.zeros(2, 38, 2, 2),
        context=torch.zeros(1, 1, 12), context_null=torch.zeros(1, 1, 12),
        denoise_steps=2, guidance_scale=5, generator=torch.Generator().manual_seed(2),
        solver_name="unipc",
    )
    assert output.video.shape[1] == 38
    assert output.audio.shape == (250, 4)
    assert output.requested_frame_count == 151
    assert not torch.equal(output.audio, audio[0])
    output = JointMediaLatents.from_payload(output.to_payload())
    decoded_frames = torch.arange(149).float().reshape(149, 1, 1, 1)
    decoder = SimpleNamespace(
        _decode_device="cpu", options={},
        decode_latents_native=lambda _: decoded_frames,
        _native_assets=lambda: SimpleNamespace(
            offload_video_vae=lambda: None, audio_geometry=lambda: (48_000, 960, 4),
            decode_audio=lambda latent, device: torch.ones(239_000),
        ),
    )
    decoded = AgnesVideoPipeline.decode_media_native(decoder, output)
    assert decoded.frames.shape[0] == 150
    assert torch.equal(decoded.frames[-1], decoded_frames[-1])
    assert torch.equal(decoded.frames[-2], decoded_frames[-1])
    assert decoded.audio.shape == (240_000,)
    assert torch.all(decoded.audio[239_000:] == 0)
