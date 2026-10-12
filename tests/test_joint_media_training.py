from types import SimpleNamespace

import pytest
import torch

from mirai.core.dataset.cache import load_indexed_cache_records, save_indexed_cache_sidecar
from mirai.core.training.data.batches import build_batch_from_records
from mirai.core.training.strategies.base import TrainingInputs
from mirai.core.training.strategies.text_to_video import TextToVideoStrategy


def _record(length):
    return {
        "latent": torch.ones(2, 1, 2, 2),
        "text_embed": torch.ones(2, 3),
        "extra_tensors": {"audio_latents": torch.arange(length * 3).reshape(length, 3).float()},
        "extra_tensor_schema": {"audio_latents": {"collation": "pad_sequence", "lengths_key": "audio_lengths"}},
    }


def test_indexed_audio_roundtrip_and_variable_length_collation(tmp_path):
    records = [_record(2), _record(4)]
    path = tmp_path / "cache.pt"
    torch.save({"records": records}, path)
    save_indexed_cache_sidecar(path, records)
    indexed = load_indexed_cache_records(path, verify_checksum=False)
    assert indexed is not None
    assert "audio_latents" in indexed[0]
    torch.testing.assert_close(indexed[0]["audio_latents"], records[0]["extra_tensors"]["audio_latents"])
    batch = build_batch_from_records(indexed)
    assert batch["audio_latents"].shape == (2, 4, 3)
    assert batch["audio_lengths"].tolist() == [2, 4]
    assert torch.count_nonzero(batch["audio_latents"][0, 2:]) == 0


def test_extra_schema_mismatch_fails():
    records = [_record(2), _record(4)]
    records[1]["extra_tensor_schema"] = {}
    with pytest.raises(ValueError, match="inconsistent"):
        build_batch_from_records(records)


def test_pipeline_owned_training_preparation_dispatches():
    expected = TrainingInputs(None, None, None, None, {})
    seen = {}

    def prepare(batch, **kwargs):
        seen.update(kwargs)
        assert batch == {"joint": True}
        return expected

    pipeline = SimpleNamespace(prepare_training_inputs=prepare)
    strategy = TextToVideoStrategy(SimpleNamespace(params={}))
    actual = strategy.prepare_inputs({"joint": True}, pipeline, "sampler", "noise", objective="objective")
    assert actual is expected
    assert seen == {"timestep_sampler": "sampler", "noise_generator": "noise", "objective": "objective", "training": True}


def test_agnes_audio_encoder_preserves_token_layout_and_trims_padding():
    from mirai.core.models.agnes_video.assets import AgnesAssets

    assets = AgnesAssets({})
    observed = {}

    class Audio:
        def encode_audio(self, wave, *, sample_posterior):
            observed["shape"] = wave.shape
            observed["posterior"] = sample_posterior
            return torch.arange(20).reshape(1, 2, 10).float(), None

    assets._audio_geometry = (48000, 4, 2)
    assets.load_audio_vae = lambda device: Audio()
    encoded = assets.encode_audio(torch.ones(9))
    assert encoded.shape == (3, 2)
    assert observed == {"shape": (1, 1, 9), "posterior": False}
    torch.testing.assert_close(encoded[:, 0], torch.tensor([0., 1., 2.]))


def test_agnes_precomputed_cache_requires_joint_modalities(tmp_path):
    from mirai.core.models.agnes_video.cache import AgnesNativeCacheEncoder
    from mirai.core.models.providers import NativeCacheEncoderConfig

    encoder = AgnesNativeCacheEncoder(NativeCacheEncoderConfig(True, "agnes_video", "", "", "bf16", 17))
    path = tmp_path / "joint.pt"
    torch.save({"video_latents": torch.ones(48, 2, 2, 2), "audio_latents": torch.ones(5, 128)}, path)
    video, _ = encoder.encode_latent(path)
    extra, schema = encoder.encode_extra_tensors(path)
    assert video.shape == (48, 2, 2, 2)
    assert extra["audio_latents"].shape == (5, 128)
    assert schema["audio_latents"]["lengths_key"] == "audio_lengths"
    torch.save({"latent": video}, path)
    with pytest.raises(ValueError, match="requires video_latents and audio_latents"):
        encoder.encode_latent(path)


def test_agnes_audio_extraction_uses_video_selection_and_missing_audio_is_explicit(tmp_path, monkeypatch):
    import subprocess
    from mirai.core.models.agnes_video.cache import AgnesNativeCacheEncoder
    from mirai.core.models.providers import NativeCacheEncoderConfig

    path = tmp_path / "clip.mp4"
    encoder = AgnesNativeCacheEncoder(NativeCacheEncoderConfig(True, "agnes_video", "", "", "bf16", 17))
    encoder._selected = (path.resolve(), 2.5, 0.5)
    encoder.assets = SimpleNamespace(offload_video_vae=lambda: None, offload_text_encoder=lambda: None,
                                    load_audio_vae=lambda device: None, audio_geometry=lambda: (100, 4, 128),
                                    offload_audio_vae=lambda: None, encode_audio=lambda wave, device: wave[:, None])
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ValueError, match="requires audio"):
        encoder.encode_extra_tensors(path)
    assert calls[0][calls[0].index("-ss") + 1] == "2.5"
    assert calls[0][calls[0].index("-t") + 1] == "0.5"
    encoder.allow_missing_audio = True
    extra, _ = encoder.encode_extra_tensors(path)
    assert extra["audio_latents"].shape == (50, 1)
    assert torch.count_nonzero(extra["audio_latents"]) == 0


def test_agnes_external_asset_content_mutation_changes_lineage(tmp_path):
    from mirai.core.models.agnes_video.cache import AgnesNativeCacheEncoder
    from mirai.core.models.providers import NativeCacheEncoderConfig

    options = {}
    for name in ("text_encoder_path", "video_vae_path", "audio_vae_path"):
        path = tmp_path / name
        path.mkdir()
        (path / "weights.bin").write_bytes(b"initial")
        (path / "download_manifest.json").write_text("{}")
        options[name] = str(path)
    code = tmp_path / "code"
    (code / "omnivae").mkdir(parents=True)
    (code / "omnivae" / "__init__.py").write_text("")
    options["audio_vae_code_root"] = str(code)
    encoder = AgnesNativeCacheEncoder(NativeCacheEncoderConfig(True, "agnes-video", "", "", "bf16", 17, family_params=options))
    before = encoder.cache_asset_fingerprints()
    (tmp_path / "audio_vae_path" / "weights.bin").write_bytes(b"changed")
    after = encoder.cache_asset_fingerprints()
    assert before["audio_vae_path"] != after["audio_vae_path"]
    assert before["text_encoder_path"] == after["text_encoder_path"]


def test_loader_rejects_changed_native_assets(tmp_path, monkeypatch):
    import mirai.core.training.data.loader as loader
    from mirai.core.models.providers import NativeCacheEncoderConfig

    record = _record(2)
    path = tmp_path / "cache.pt"
    torch.save({"records": [record]}, path)
    save_indexed_cache_sidecar(path, [record], metadata_extra={"native_asset_fingerprints": {"audio": "old"}})
    monkeypatch.setattr(loader, "build_native_cache_encoder", lambda cfg: SimpleNamespace(cache_asset_fingerprints=lambda: {"audio": "new"}))
    cfg = NativeCacheEncoderConfig(True, "example", "", "", "bf16", 1)
    with pytest.raises(ValueError, match="native asset lineage mismatch"):
        loader.load_prepared_training_data(cache_path=path, model_type="example", strategy_type="text_to_video",
                                           val_every_n_steps=0, native_cache_config=cfg)


def test_cache_builder_captures_provider_extra_tensors(tmp_path, monkeypatch):
    import mirai.core.dataset.cache as cache
    from mirai.core.dataset.native_encode import NativeCacheStatus

    data = tmp_path / "data"
    data.mkdir()
    torch.save(torch.ones(1), data / "sample.pt")
    calls = []

    def video(path):
        calls.append(("video", path))
        return torch.ones(2, 1, 2, 2), None

    def audio(path):
        calls.append(("audio", path))
        return {"audio_latents": torch.ones(3, 128)}, {"audio_latents": {"collation": "pad_sequence", "lengths_key": "audio_lengths"}}

    native = SimpleNamespace(latent_channels=2, encode_latent=video, encode_extra_tensors=audio,
                             encode_text=lambda caption: torch.ones(2, 3), encode_clip=lambda path: None,
                             status=lambda: NativeCacheStatus(False, False, False, False, "fixture", "fixture", "none"))
    monkeypatch.setattr(cache, "build_native_cache_encoder", lambda config: native)
    payload = cache.build_cache(data, tmp_path / "cache.pt")
    assert calls == [("video", data / "sample.pt"), ("audio", data / "sample.pt")]
    assert payload["records"][0]["extra_tensors"]["audio_latents"].shape == (3, 128)
    indexed = cache.load_indexed_cache_records(tmp_path / "cache.pt")
    assert indexed is not None
    assert build_batch_from_records(indexed)["audio_lengths"].tolist() == [3]


def test_text_padding_mask_preserves_real_prompt_and_explicit_masks():
    first, second = _record(2), _record(3)
    first["text_embed"] = torch.ones(2, 3)
    first["text_mask"] = [1]
    second["text_embed"] = torch.ones(4, 3)
    second["text_mask"] = [1, 1, 1, 0]
    alone = build_batch_from_records([first])
    together = build_batch_from_records([first, second])
    assert together["text_mask"].tolist() == [[True, True, False, False], [True, True, True, False]]
    torch.testing.assert_close(alone["text_embeds"][0][alone["text_mask"][0]],
                               together["text_embeds"][0][together["text_mask"][0]])


def test_family_conditioning_dropout_runs_after_noise_and_keeps_input_unchanged():
    text = torch.ones(3, 2, 4)
    batch = {"latents": torch.zeros(3, 1), "text_embeds": text}
    expected = TrainingInputs(None, None, None, None, {"t5": text, "text_mask": torch.ones(3, 2, dtype=torch.bool)})
    draw = {}

    def prepare(batch, **kwargs):
        draw["noise"] = torch.rand(2)
        return expected

    strategy = TextToVideoStrategy(SimpleNamespace(params={"conditioning_dropout_p": .5, "empty_text_embed": -1.}))
    torch.manual_seed(77)
    reference_noise = torch.rand(2)
    reference_mask = torch.rand(3) < .5
    torch.manual_seed(77)
    actual = strategy.prepare_inputs(batch, SimpleNamespace(prepare_training_inputs=prepare), None, None)
    torch.testing.assert_close(draw["noise"], reference_noise)
    expected_text = text.clone()
    expected_text[reference_mask] = -1
    torch.testing.assert_close(actual.text_embeds["t5"], expected_text)
    assert torch.all(text == 1)
    assert torch.all(actual.text_embeds["text_mask"])


def test_generic_clip_hook_encodes_each_window_once_and_binds_recovery(tmp_path, monkeypatch):
    import mirai.core.dataset.cache as cache
    from mirai.core.dataset.native_encode import NativeCacheStatus

    data = tmp_path / "data"
    data.mkdir()
    torch.save(torch.ones(1), data / "sample.pt")
    calls = []
    offset = [0]

    def variant(path, *, clip_index, clip_count):
        calls.append(clip_index)
        value = clip_index + offset[0]
        return cache.EncodedMediaVariant(
            torch.full((2, 1, 2, 2), float(value)),
            extra_tensors={"audio_latents": torch.full((3, 128), float(value))},
            extra_tensor_schema={"audio_latents": {"collation": "pad_sequence", "lengths_key": "audio_lengths"}},
            metadata={"media_timeline": {"start_seconds": value, "duration_seconds": 1., "fps": 30.}},
        )

    def forbidden(path):
        raise AssertionError("Initial encode must be skipped for the variant hook.")

    native = SimpleNamespace(latent_channels=2, encode_latent=forbidden, encode_media_variant=variant,
                             encode_text=lambda caption: torch.ones(2, 3), encode_clip=lambda path: None,
                             status=lambda: NativeCacheStatus(False, False, False, False, "fixture", "fixture", "none"))
    monkeypatch.setattr(cache, "build_native_cache_encoder", lambda config: native)
    path = tmp_path / "cache.pt"
    first = cache.build_cache(data, path, clips_per_video=3)
    assert calls == [0, 1, 2]
    assert [record["metadata"]["media_timeline"]["start_seconds"] for record in first["records"]] == [0, 1, 2]
    assert len({record["fingerprints"]["latent"] for record in first["records"]}) == 3
    offset[0] = 10
    second = cache.build_cache(data, path, clips_per_video=3)
    assert second["records"][0]["latent"].mean() == 10
    assert second["records"][0]["fingerprints"]["latent"] != first["records"][0]["fingerprints"]["latent"]


def test_agnes_variant_windows_share_audio_video_start_and_mark_short_duplicates(tmp_path, monkeypatch):
    import json
    import subprocess
    from mirai.core.models.agnes_video.cache import AgnesNativeCacheEncoder
    from mirai.core.models.providers import NativeCacheEncoderConfig

    encoder = AgnesNativeCacheEncoder(NativeCacheEncoderConfig(True, "agnes-video", "", "", "bf16", 5,
                                                               family_params={"fps": 5}))
    duration = [5.]
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps({"streams": [{"duration": str(duration[0])}]}).encode()))

    def video(path):
        encoder._selected = (path.resolve(), encoder._selection_start, 1.)
        return torch.ones(48, 2, 2, 2), None

    def audio(path):
        assert encoder._selected[1] == encoder._selection_start
        return {"audio_latents": torch.ones(5, 128)}, {}

    encoder.encode_latent, encoder.encode_extra_tensors = video, audio
    path = tmp_path / "clip.mp4"
    results = [encoder.encode_media_variant(path, clip_index=index, clip_count=3) for index in range(3)]
    assert [result.metadata["media_timeline"]["start_seconds"] for result in results] == [0., 2., 4.]
    duration[0] = .5
    result = encoder.encode_media_variant(path, clip_index=1, clip_count=3)
    assert result.metadata["media_timeline"]["identical_short_clip"] is True
    with pytest.raises(ValueError, match="do not support temporal"):
        encoder.encode_media_variant(tmp_path / "joint.pt", clip_index=0, clip_count=2)


def test_agnes_ffmpeg_temporal_windows_extract_matching_audio(tmp_path):
    import shutil
    import subprocess
    from mirai.core.models.agnes_video.cache import AgnesNativeCacheEncoder
    from mirai.core.models.providers import NativeCacheEncoderConfig

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe unavailable")
    path = tmp_path / "timeline.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=32x32:rate=5:duration=3",
                    "-f", "lavfi", "-i", "aevalsrc=t/3:s=1000:d=3", "-c:v", "ffv1", "-c:a", "pcm_f32le", str(path)],
                   check=True, capture_output=True)
    encoder = AgnesNativeCacheEncoder(NativeCacheEncoderConfig(True, "agnes-video", "", "", "bf16", 5,
                                                               family_params={"fps": 5}))
    encoder.assets = SimpleNamespace(
        offload_text_encoder=lambda: None, offload_video_vae=lambda: None, offload_audio_vae=lambda: None,
        load_video_vae=lambda device: SimpleNamespace(encode=lambda video: video.mean().expand(1, 48, 2, 2, 2)),
        load_audio_vae=lambda device: None, audio_geometry=lambda: (1000, 200, 128),
        encode_audio=lambda waveform, device: waveform.reshape(5, 200).mean(1)[:, None].expand(5, 128),
    )
    windows = [encoder.encode_media_variant(path, clip_index=index, clip_count=3) for index in range(3)]
    starts = [window.metadata["media_timeline"]["start_seconds"] for window in windows]
    assert starts == pytest.approx([0., 1., 2.], abs=.01)
    for window, start in zip(windows, starts, strict=True):
        assert window.metadata["media_timeline"]["duration_seconds"] == 1.
        # The audio signal is a known linear clock, so its value proves which
        # exact source interval was extracted alongside the video window.
        assert window.extra_tensors["audio_latents"].mean().item() == pytest.approx((start + .4995) / 3, abs=.005)


def test_native_audio_strict_state_loading_rejects_missing_and_unexpected(tmp_path):
    from mirai.core.models.agnes_video.audio import load_dac_state

    reference = torch.nn.Linear(3, 2)
    path = tmp_path / "state_dict.pt"
    state = {"module.audio_vae." + name: value for name, value in reference.state_dict().items()}
    state["module.video_vae.ignored"] = torch.zeros(1)
    torch.save({"model_state_dict": state}, path)
    restored = torch.nn.Linear(3, 2, device="meta")
    load_dac_state(restored, path)
    torch.testing.assert_close(restored.weight, reference.weight)
    state.pop("module.audio_vae.bias")
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_dac_state(torch.nn.Linear(3, 2, device="meta"), path)
    state["module.audio_vae.bias"] = reference.bias.detach()
    state["module.audio_vae.unknown"] = torch.zeros(1)
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_dac_state(torch.nn.Linear(3, 2, device="meta"), path)


def test_native_audio_small_dac_matches_direct_posterior_and_decode(tmp_path):
    import sys
    from pathlib import Path
    import yaml
    from mirai.core.models.agnes_video.audio import NativeOmniAudio
    from mirai.core.models.agnes_video.assets import AgnesAssets

    before = {name for name in sys.modules if name == "diffusers" or name.startswith("diffusers.")}
    module = pytest.importorskip("omnivae.models.audio_vae_dac")
    after = {name for name in sys.modules if name == "diffusers" or name.startswith("diffusers.")}
    assert after == before
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(4)
        dac = module.DAC(encoder_dim=4, encoder_rates=[2], latent_dim=128,
                         decoder_dim=16, decoder_rates=[2], sample_rate=48000, continuous=True).eval()
        wrapper = NativeOmniAudio(dac)
        wave = torch.randn(1, 1, 257)
        with torch.no_grad():
            posterior = dac.encode(dac.preprocess(wave))[0]
            expected = posterior.mode()
            actual, _ = wrapper.encode_audio(wave, sample_posterior=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(wrapper.decode_audio(actual), dac.decode(expected), rtol=0, atol=0)
        assert actual.shape == (1, 128, 129)
        assert wrapper.decode_audio(actual).shape == (1, 1, 258)
        config = {"model": {"audio": {"encoder_dim": 4, "encoder_rates": [2], "latent_dim": 128,
                                     "decoder_dim": 16, "decoder_rates": [2], "sample_rate": 48000, "continuous": True}}}
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        torch.save({"state_dict": {"audio_vae." + name: tensor for name, tensor in dac.state_dict().items()}}, tmp_path / "state_dict.pt")
        assets = AgnesAssets({"audio_vae_code_root": str(Path(module.__file__).parents[3]), "audio_vae_path": str(tmp_path)})
        assert assets.audio_geometry() == (48000, 2, 128)
        tokens = assets.encode_audio(wave.flatten())
        torch.testing.assert_close(tokens, expected[0].transpose(0, 1), rtol=0, atol=0)
        torch.testing.assert_close(assets.decode_audio(tokens), dac.decode(expected), rtol=0, atol=0)
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("enabled,allow_tf32", [(True, True), (True, False), (False, True), (False, False)])
def test_native_audio_decode_scopes_deterministic_cudnn_flags(enabled, allow_tf32):
    from mirai.core.models.agnes_video.audio import NativeOmniAudio

    observed = []

    class Decoder(torch.nn.Module):
        def decode(self, latent):
            observed.append((torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                             torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32))
            return latent

    original = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = enabled
    torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        latent = torch.ones(1, 2, 3)
        assert NativeOmniAudio(Decoder()).decode_audio(latent) is latent
        assert observed == [(True, False, enabled, allow_tf32)]
        assert (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32) == (False, True, enabled, allow_tf32)
    finally:
        (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
         torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32) = original


def test_native_audio_primes_each_decode_layout_only_for_no_grad_eval():
    from mirai.core.models.agnes_video.audio import NativeOmniAudio

    class Decoder(torch.nn.Module):
        calls = 0

        def decode(self, latent):
            self.calls += 1
            return latent

    decoder = Decoder()
    model = NativeOmniAudio(decoder).eval()
    with torch.no_grad():
        latent = torch.ones(1, 2, 3)
        assert model.decode_audio(latent) is latent
        assert decoder.calls == 2
        model.decode_audio(latent)
        assert decoder.calls == 3
        model.decode_audio(torch.ones(1, 2, 4))
        assert decoder.calls == 5
    model.decode_audio(torch.ones(1, 2, 5, requires_grad=True))
    assert decoder.calls == 6


def test_nonfinite_extra_tensor_rejected_on_save_and_tampered_indexed_load(tmp_path):
    import mirai.core.dataset.cache as cache

    path = tmp_path / "cache.pt"
    record = _record(2)
    torch.save({"records": [record]}, path)
    cache.save_indexed_cache_sidecar(path, [record])
    tensor_path = cache.indexed_cache_tensor_path(path)
    tensors = cache.load_tensor_cache_safetensors(tensor_path)
    metadata = cache.load_indexed_cache_metadata(path)
    audio_key = metadata["records"][0]["extra_tensor_keys"]["audio_latents"]
    tensors[audio_key][0, 0] = float("nan")
    cache.save_tensor_cache_safetensors(tensor_path, tensors)
    with pytest.raises(ValueError, match="finite non-empty"):
        cache.load_indexed_cache_records(path)
    record["extra_tensors"]["audio_latents"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite non-empty"):
        cache.save_indexed_cache_sidecar(path, [record])
    torch.save({"records": [record]}, path)
    with pytest.raises(ValueError, match="finite non-empty"):
        cache.load_cache(path, verify_checksum=False)


def test_invalid_extra_tensor_schema_fails_before_indexed_write(tmp_path):
    record = _record(2)
    record["extra_tensor_schema"]["audio_latents"]["lengths_key"] = "text_mask"
    with pytest.raises(ValueError, match="lengths_key"):
        save_indexed_cache_sidecar(tmp_path / "bad.pt", [record])
    record["extra_tensor_schema"]["audio_latents"] = {"collation": "unrecognized"}
    with pytest.raises(ValueError, match="collation"):
        save_indexed_cache_sidecar(tmp_path / "bad.pt", [record])
