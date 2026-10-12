"""CPU validation of the Agnes full-release evidence envelope."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import gc
import weakref

import pytest
import torch

from mirai.core.models.agnes_video.checkpoints import AGNES_SOURCE_REVISION
from mirai.core.models.agnes_video.contracts.report import (
    RELEASE_ARCHITECTURE,
    SCHEMA_VERSION,
    validate_inference_report,
    validate_train_report,
)
from mirai.core.models.agnes_video.contracts._runtime import release_training_session
from mirai.core.models.agnes_video.contracts._runtime import ContractUnavailable, checkpoint_digest
from mirai.core.models.agnes_video.contracts._runtime import runtime_identity
from mirai.vendors.agnes_video.dit import AgnesDiTConfig


def _common(contract: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": contract,
        "status": "passed",
        "runtime": {
            "family": "agnes-video",
            "pipeline_class": "AgnesVideoPipeline",
            "transformer_class": "AgnesDiT",
            "source_revision": AGNES_SOURCE_REVISION,
            "device_type": "cuda",
            "cuda_available": True,
            "cuda_device_name": "contract-device",
            "architecture": dict(RELEASE_ARCHITECTURE),
            "checkpoint_constructed": True,
            "configured_checkpoint_source": "configured-checkpoint",
            "parameter_tensors": 100,
            "all_parameters_materialized": True,
        },
        "snapshot": {"source": "configured-checkpoint", "sha256": "a" * 64},
    }


def _train(tmp_path: Path) -> dict:
    adapter, checkpoint = tmp_path / "adapter.pt", tmp_path / "step1.pt"
    adapter.touch()
    checkpoint.touch()
    return {
        **_common("agnes_native_train_step"),
        "proof": {
            "joint_loss": 1.0,
            "video_loss": 0.5,
            "audio_loss": 0.5,
            "adapter_gradient_tensors": 2,
            "adapter_changed_tensors": 2,
            "video_target_gradient": True,
            "audio_target_gradient": True,
            "only_lora_trainable": True,
            "optimizer_step": True,
            "adapter_roundtrip_same_base": True,
            "source_mask_verified": True,
            "noise_mask_verified": True,
        },
        "resume_equivalence": {
            "steps": 2,
            "split_step": 1,
            "adapter_state": True,
            "optimizer_state": True,
            "scheduler_state": True,
            "python_rng": True,
            "torch_rng": True,
        },
        "artifacts": {"adapter": str(adapter), "checkpoint": str(checkpoint)},
    }


def _inference(tmp_path: Path) -> dict:
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"media")
    return {
        **_common("agnes_native_inference"),
        "seed": 42,
        "proof": {
            "native_inference": True,
            "text_encoder": "T5GemmaEncoderModel",
            "joint_denoising": True,
            "video_decoded": True,
            "audio_decoded": True,
            "audio_muxed": True,
            "adapter_scale_effect": True,
        },
        "adapter_scale": {
            "baseline": 0.0,
            "candidate": 1.0,
            "max_abs_latent_difference": 0.02,
            "video_max_abs_latent_difference": 0.01,
            "audio_max_abs_latent_difference": 0.02,
        },
        "paired_inputs": {
            "baseline": {
                "seed": 42,
                "initial_video_sha256": "b" * 64,
                "initial_audio_sha256": "c" * 64,
                "conditioned_context_sha256": "d" * 64,
                "unconditional_context_sha256": "e" * 64,
                "sigma_schedule_sha256": "f" * 64,
            },
            "candidate": {
                "seed": 42,
                "initial_video_sha256": "b" * 64,
                "initial_audio_sha256": "c" * 64,
                "conditioned_context_sha256": "d" * 64,
                "unconditional_context_sha256": "e" * 64,
                "sigma_schedule_sha256": "f" * 64,
            },
        },
        "media": {
            "path": str(media), "width": 448, "height": 256, "frames": 17,
            "audio_samples": 24000, "sample_rate": 48000,
            "video_duration_seconds": 0.57, "audio_duration_seconds": 0.5,
            "fps": 30.0,
        },
    }


def test_complete_train_and_inference_envelopes_are_accepted(tmp_path):
    assert validate_train_report(_train(tmp_path)) == []
    assert validate_inference_report(_inference(tmp_path)) == []


def test_tiny_or_injected_transformer_cannot_claim_full_release_proof(tmp_path):
    report = _train(tmp_path)
    report["runtime"]["architecture"].update(num_layers=2, hidden_size=16)
    report["runtime"]["checkpoint_constructed"] = False
    failures = validate_train_report(report)
    assert any("num_layers" in failure for failure in failures)
    assert any("hidden_size" in failure for failure in failures)
    assert any("injected transformer" in failure for failure in failures)


def test_train_report_rejects_missing_modality_and_resume_evidence(tmp_path):
    report = _train(tmp_path)
    report["proof"]["audio_target_gradient"] = False
    report["proof"]["audio_loss"] = float("nan")
    report["resume_equivalence"]["optimizer_state"] = False
    failures = validate_train_report(report)
    assert any("audio_target_gradient" in failure for failure in failures)
    assert any("audio_loss" in failure for failure in failures)
    assert any("optimizer_state" in failure for failure in failures)


def test_inference_report_rejects_dropped_audio_and_no_adapter_effect(tmp_path):
    report = _inference(tmp_path)
    report["proof"]["audio_muxed"] = False
    report["media"]["audio_samples"] = 0
    report["adapter_scale"]["max_abs_latent_difference"] = 0.0
    failures = validate_inference_report(report)
    assert any("audio_muxed" in failure for failure in failures)
    assert any("audio_samples" in failure for failure in failures)
    assert any("max_abs_latent_difference" in failure for failure in failures)


def test_inference_report_rejects_unpaired_inputs_and_modality_without_effect(tmp_path):
    report = _inference(tmp_path)
    report["paired_inputs"]["candidate"]["initial_audio_sha256"] = "1" * 64
    report["paired_inputs"]["candidate"]["sigma_schedule_sha256"] = "not-a-digest"
    report["adapter_scale"]["audio_max_abs_latent_difference"] = float("nan")
    failures = validate_inference_report(report)
    assert any("initial_audio_sha256 must match" in failure for failure in failures)
    assert any("candidate.sigma_schedule_sha256" in failure for failure in failures)
    assert any("audio_max_abs_latent_difference" in failure for failure in failures)


def test_inference_report_allows_one_modality_to_remain_unchanged(tmp_path):
    report = _inference(tmp_path)
    report["adapter_scale"]["audio_max_abs_latent_difference"] = 0.0
    report["adapter_scale"]["max_abs_latent_difference"] = 0.01
    assert validate_inference_report(report) == []


def test_incomplete_or_cpu_report_never_validates_as_passed(tmp_path):
    report = _inference(tmp_path)
    report["status"] = "incomplete"
    report["runtime"]["device_type"] = "cpu"
    report["runtime"]["cuda_available"] = False
    failures = validate_inference_report(report)
    assert any("status" in failure for failure in failures)
    assert any("CUDA" in failure for failure in failures)


def test_inference_report_rejects_unsynchronized_media(tmp_path):
    report = _inference(tmp_path)
    report["media"]["audio_duration_seconds"] = 1.5
    failures = validate_inference_report(report)
    assert any("durations" in failure for failure in failures)


def test_session_release_drops_heavy_runtime_references(monkeypatch):
    import torch

    class Heavy:
        pass

    heavy = Heavy()
    reference = weakref.ref(heavy)
    session = SimpleNamespace(
        trainer=heavy, optimizer=object(), optimizer_result=object(), scheduler=object(),
        params=[object()], named_params=[("p", object())], train_records=[{}],
        val_records=[{}], temporal_groups={"x": [{}]}, callbacks=[],
        build_ckpt_payload=(lambda retained=heavy: retained), live_control=None,
        event_bus=object(),
        close_callbacks=lambda: None,
    )
    del heavy
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    release_training_session(session)
    gc.collect()
    assert session.trainer is None
    assert session.params is None
    assert session.build_ckpt_payload is None
    assert reference() is None


def test_inference_input_probe_restores_sampler_without_pipeline_cycle():
    from mirai.core.models.agnes_video.contracts.native_inference import (
        _install_input_probe,
        _remove_input_probe,
    )

    class Pipeline(torch.nn.Module):
        def sample_native_preview(self, **_kwargs):
            return None

    pipeline = Pipeline()
    reference = weakref.ref(pipeline)
    _, handle, restore_state = _install_input_probe(pipeline, seed=42)
    assert "sample_native_preview" in pipeline.__dict__
    _remove_input_probe(pipeline, handle, restore_state)
    assert "sample_native_preview" not in pipeline.__dict__
    del handle, restore_state, pipeline
    gc.collect()
    assert reference() is None


def test_inference_input_probe_preserves_existing_instance_override():
    from mirai.core.models.agnes_video.contracts.native_inference import (
        _install_input_probe,
        _remove_input_probe,
    )

    pipeline = torch.nn.Module()
    override = lambda **_kwargs: None
    pipeline.sample_native_preview = override
    _, handle, restore_state = _install_input_probe(pipeline, seed=42)
    _remove_input_probe(pipeline, handle, restore_state)
    assert pipeline.sample_native_preview is override


def test_snapshot_digest_rejects_index_shard_path_escape(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        '{"metadata":{"format":"mirai.agnes.safetensors","format_version":1,'
        '"source_revision":"fc3971e378b53c766baa06898bfe84d7a56653ea"},'
        '"weight_map":{"weight":"../outside.safetensors"}}',
        encoding="utf-8",
    )
    try:
        checkpoint_digest(tmp_path)
    except ContractUnavailable as exc:
        assert "invalid" in str(exc)
    else:
        raise AssertionError("unsafe shard path was accepted")


@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_runtime_identity_treats_compute_precision_as_non_structural(monkeypatch, compute_dtype):
    transformer = torch.nn.Linear(1, 1)
    pipeline = SimpleNamespace(
        runtime_config=AgnesDiTConfig(params_dtype=compute_dtype),
        transformer=transformer,
        model_config=SimpleNamespace(path="configured-checkpoint"),
        _adapter_base_fingerprint="fingerprint",
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "contract-device")
    identity = runtime_identity(pipeline, torch.device("cuda"))
    assert identity["checkpoint_constructed"] is True
    assert identity["compute_dtype"] == str(compute_dtype)
    assert "params_dtype" not in identity["architecture"]


def test_runtime_identity_rejects_altered_released_topology(monkeypatch):
    pipeline = SimpleNamespace(
        runtime_config=replace(AgnesDiTConfig(), num_layers=39),
        transformer=torch.nn.Linear(1, 1),
        model_config=SimpleNamespace(path="configured-checkpoint"),
        _adapter_base_fingerprint="fingerprint",
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "contract-device")
    identity = runtime_identity(pipeline, torch.device("cuda"))
    assert identity["checkpoint_constructed"] is False


@pytest.mark.parametrize(
    ("module_name", "config_env", "report_env", "contract"),
    [
        ("native_train_step", "MIRAI_AGNES_TRAIN_CONFIG", "MIRAI_AGNES_TRAIN_REPORT",
         "agnes_native_train_step"),
        ("native_inference", "MIRAI_AGNES_INFERENCE_CONFIG", "MIRAI_AGNES_INFERENCE_REPORT",
         "agnes_native_inference"),
    ],
)
def test_executable_contract_without_private_resources_is_incomplete(
    monkeypatch, capsys, module_name, config_env, report_env, contract
):
    module = __import__(
        f"mirai.core.models.agnes_video.contracts.{module_name}", fromlist=["main"]
    )
    monkeypatch.delenv(config_env, raising=False)
    monkeypatch.delenv(report_env, raising=False)
    assert module.main([]) == 2
    report = __import__("json").loads(capsys.readouterr().out)
    assert report["contract"] == contract
    assert report["status"] == "incomplete"


@pytest.mark.parametrize(
    ("module_name", "config_env", "report_env"),
    [
        ("native_train_step", "MIRAI_AGNES_TRAIN_CONFIG", "MIRAI_AGNES_TRAIN_REPORT"),
        ("native_inference", "MIRAI_AGNES_INFERENCE_CONFIG", "MIRAI_AGNES_INFERENCE_REPORT"),
    ],
)
def test_executable_contract_uses_private_environment_paths(
    monkeypatch, tmp_path, module_name, config_env, report_env
):
    module = __import__(
        f"mirai.core.models.agnes_video.contracts.{module_name}", fromlist=["main"]
    )
    config = tmp_path / "private.toml"
    output = tmp_path / "report.json"
    monkeypatch.setenv(config_env, str(config))
    monkeypatch.setenv(report_env, str(output))
    calls = []
    monkeypatch.setattr(
        module, "run", lambda config_path, output_path, **kwargs:
        calls.append((config_path, output_path, kwargs)) or {"status": "passed"}
    )
    assert module.main([]) == 0
    assert calls[0][:2] == (str(config), str(output))
    assert output.is_file()


def test_agnes_contract_commands_are_executable_manifest_probes():
    manifest = __import__("json").loads(
        (Path(__file__).parents[1] / "agent" / "checks.json").read_text(encoding="utf-8")
    )
    model_family = next(check for check in manifest["checks"] if check["name"] == "model_family")
    assert "python -m mirai.core.models.agnes_video.contracts.native_train_step" in model_family["commands"]
    assert "python -m mirai.core.models.agnes_video.contracts.native_inference" in model_family["commands"]
