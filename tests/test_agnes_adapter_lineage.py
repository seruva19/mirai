"""Strict Agnes adapter content lineage and portable metadata contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mirai.core.models.agnes_video.pipeline import AgnesVideoPipeline
from mirai.core.training.adapters import load_adapter_payload, normalize_adapter_state, save_kohya_adapter_safetensors
from mirai.core.training.optim.ema import init_ema_state, update_ema_state
from mirai.vendors.agnes_video.dit import AgnesDiT, AgnesDiTConfig


def pipeline(seed=11, **adapter_options):
    torch.manual_seed(seed)
    native = AgnesDiT(AgnesDiTConfig(num_layers=2, hidden_size=16, head_dim=8,
        num_query_groups=1, video_in_channels=8, audio_in_channels=4, text_in_channels=12,
        mm_layers=(0,), sparse_moe_layers=(1,), gelu7_layers=(0,)))
    model = AgnesVideoPipeline(SimpleNamespace(params=SimpleNamespace(family_params={})), transformer=native)
    spec = dict(type="lora", target_preset="attn_mlp", rank=2, alpha=2,
                train_router=False, init="kaiming")
    spec.update(adapter_options)
    model.set_adapter_config(SimpleNamespace(**spec))
    return model


def test_same_shape_wrong_base_rejected_before_adapter_writes():
    first, other = pipeline(11), pipeline(12)
    saved = {key: value.clone() for key, value in other.state_dict().items()}
    with pytest.raises(ValueError, match="base checkpoint"):
        other.load_adapter_state(first.state_dict())
    for key, value in other.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [("family", "other-family"),
    ("source_revision", "different-revision"), ("schema_version", 99)])
def test_explicit_family_revision_and_schema_mismatches_fail(field, value):
    model = pipeline()
    state = model.state_dict()
    lineage = json.loads(bytes(state["__mirai_adapter_lineage__"].tolist()))
    lineage[field] = value
    state["__mirai_adapter_lineage__"] = torch.tensor(list(json.dumps(lineage).encode()), dtype=torch.uint8)
    with pytest.raises(ValueError, match="source revision"):
        model.load_adapter_state(state)


def test_missing_lineage_and_same_shape_changed_adapter_alpha_fail():
    model = pipeline()
    state = model.state_dict()
    del state["__mirai_adapter_lineage__"]
    with pytest.raises(ValueError, match="missing required"):
        model.load_adapter_state(state)
    with pytest.raises(ValueError, match="topology"):
        pipeline(alpha=8).load_adapter_state(model.state_dict())


def test_native_and_portable_roundtrip_preserve_lineage(tmp_path):
    model, other = pipeline(), pipeline()
    state = model.state_dict()
    other.load_adapter_state({"adapter_state": state})
    file = tmp_path / "adapter.safetensors"
    save_kohya_adapter_safetensors(file, adapter_state=state, rank=2, alpha=2,
                                  target_modules=["linear_qkv"], model_path="fixture")
    payload = load_adapter_payload(file)
    assert "mirai_adapter_lineage" in payload["_metadata"]
    other.validate_adapter_payload(payload)
    normalized = normalize_adapter_state(payload)
    other.load_adapter_state(normalized)
    assert set(other.state_dict()) == set(state)
    for key, value in other.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="base checkpoint"):
        pipeline(12).validate_adapter_payload(payload)


@pytest.mark.parametrize("portable", [False, True], ids=["raw", "safetensors"])
def test_inference_session_loads_versioned_lineage_artifacts(
    tmp_path, monkeypatch, portable
):
    from mirai.core.inference import session as session_module

    source = pipeline()
    state = source.state_dict()
    if portable:
        artifact = tmp_path / "adapter.safetensors"
        save_kohya_adapter_safetensors(
            artifact,
            adapter_state=state,
            rank=2,
            alpha=2,
            target_modules=["linear_qkv"],
            model_path="fixture",
        )
    else:
        artifact = tmp_path / "adapter.pt"
        torch.save(state, artifact)

    class SmallTrainer:
        weight_residency_strategy = "disabled"

        def __init__(self, _config):
            self.pipeline = pipeline()

    monkeypatch.setattr(session_module, "Trainer", SmallTrainer)
    monkeypatch.setattr(session_module, "register_builtin_components", lambda: None)
    config = Path(__file__).parents[1] / "configs" / "agnes_video" / "inference.toml"
    loaded = session_module.InferenceSession.from_config(
        str(config),
        adapter=str(artifact),
        place_on_device=False,
        device_fn=lambda: torch.device("cpu"),
        dtype_fn=lambda _config: torch.float32,
    )
    restored = loaded.pipeline.state_dict()
    assert set(restored) == set(state)
    for key, value in restored.items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


@pytest.mark.parametrize("options", [dict(timestep_rank_schedule="linear"),
    dict(timestep_band_min=.2), dict(timestep_band_max=.8)])
def test_unimplemented_timestep_controls_fail_before_wrapping(options):
    with pytest.raises(ValueError, match="timestep"):
        pipeline(**options)


def test_ema_preserves_static_lineage_through_portable_export(tmp_path):
    model = pipeline()
    live = model.state_dict()
    ema = init_ema_state(live)
    ema = update_ema_state(ema, live, decay=.9)
    torch.testing.assert_close(ema["__mirai_adapter_lineage__"], live["__mirai_adapter_lineage__"])
    file = tmp_path / "adapter_ema.safetensors"
    save_kohya_adapter_safetensors(file, adapter_state=ema, rank=2, alpha=2,
                                  target_modules=["linear_qkv"], model_path="fixture")
    payload = load_adapter_payload(file)
    restored = pipeline()
    restored.validate_adapter_payload(payload)
    restored.load_adapter_state(normalize_adapter_state(payload))
    changed = dict(live)
    changed["__mirai_adapter_lineage__"] = live["__mirai_adapter_lineage__"].clone()
    changed["__mirai_adapter_lineage__"][0] ^= 1
    with pytest.raises(ValueError, match="lineage changed"):
        update_ema_state(ema, changed, decay=.9)
    del changed["__mirai_adapter_lineage__"]
    with pytest.raises(ValueError, match="lineage changed"):
        update_ema_state(ema, changed, decay=.9)
