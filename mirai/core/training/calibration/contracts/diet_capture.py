"""CPU contract for the native paired-CFG DIET capture driver."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from mirai.core.moe.calibration.diet import load_diet_evidence
from mirai.core.training.calibration.diet import (
    DietCalibrationRequest,
    run_diet_calibration_session,
)


class _Host:
    def __init__(self) -> None:
        self.observer = None

    def get_diet_capture_observer(self):
        return self.observer

    def set_diet_capture_observer(self, observer) -> None:
        self.observer = observer


class _Session:
    def __init__(self, host: _Host) -> None:
        self.pipeline = SimpleNamespace(get_training_model=lambda: None)
        self.host = host
        self.cfg = SimpleNamespace(
            memory=SimpleNamespace(frozen_weight_packed_state_path=""),
            model=SimpleNamespace(
                type="fake",
                params=SimpleNamespace(
                    expert_pruning="prune", expert_pruning_criterion="diet"
                ),
            )
        )

    def generate(self, **kwargs):
        observer = self.host.observer
        tokens = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        for _step in range(int(kwargs["steps"])):
            for branch in (0.25, -0.25):
                indices = observer.sample_token_indices(tokens, layer_name="blocks.0.moe")
                expert_outputs = torch.stack(
                    [tokens.index_select(0, indices) + expert + branch for expert in range(4)],
                    dim=1,
                )
                observer.capture_layer(
                    tokens=tokens,
                    router_scores=torch.sigmoid(tokens[:, :1] + torch.arange(4)),
                    choice_bias=torch.zeros(4),
                    expert_outputs=expert_outputs,
                    sampled_token_indices=indices,
                    shared_output=None,
                )
        return {"status": "partial"}


def test_diet_session_captures_pairs_restores_observer_and_saves_lineage(tmp_path: Path) -> None:
    host = _Host()
    target = SimpleNamespace(
        name="blocks.0.moe", host=host, num_experts=4, top_k=2, n_group=2,
        topk_group=2, group_score_topk=1, norm_topk_prob=True, route_scale=1.0,
    )
    provider = SimpleNamespace(build_diet_calibration_targets=lambda pipeline: (target,))
    packed = tmp_path / "base.safetensors"
    packed.write_bytes(b"exact packed source")
    output = tmp_path / "diet.safetensors"
    request = DietCalibrationRequest("prompt", "negative", 7, 2, 5.0, 1, 8, 8)
    session = _Session(host)
    session.cfg.memory.frozen_weight_packed_state_path = str(packed)
    with mock.patch(
        "mirai.core.training.calibration.diet.get_model_family_provider",
        return_value=provider,
    ):
        report = run_diet_calibration_session(
            session, requests=[request], output_path=output,
            samples_per_layer=3, source_packed_state=packed,
            calibration_manifest_id="manifest", config_snapshot_id="config",
        )
    evidence, lineage = load_diet_evidence(output)
    assert host.observer is None
    assert report.modules == {"blocks.0.moe": 3}
    assert evidence["blocks.0.moe"].signatures.shape == (4, 4)
    assert lineage.calibration_manifest_id == "manifest"
    assert len(lineage.packed_state_sha256) == 64


@pytest.mark.parametrize("field,value", [
    ("cfg_scale", float("nan")), ("seed", -1), ("seed", 1.5),
    ("steps", 1.5), ("frames", -1), ("height", 0), ("width", 1.25),
])
def test_manifest_rejects_invalid_numbers(field: str, value: object) -> None:
    raw = {"prompt": "p", "negative_prompt": "n", "seed": 1, "steps": 2,
           "cfg_scale": 5.0, "frames": 1, "height": 8, "width": 8}
    raw[field] = value
    with pytest.raises(ValueError):
        DietCalibrationRequest.from_mapping(raw)


def _kwargs(tmp_path: Path, session: _Session, packed: Path) -> dict:
    session.cfg.memory.frozen_weight_packed_state_path = str(packed)
    return {"session": session, "requests": [DietCalibrationRequest(
        "prompt", "negative", 7, 2, 5.0, 1, 8, 8)],
        "output_path": tmp_path / "diet.safetensors", "samples_per_layer": 3,
        "source_packed_state": packed, "calibration_manifest_id": "manifest",
        "config_snapshot_id": "config"}


def test_session_rejects_mixed_guidance_source_mismatch_and_source_output(tmp_path: Path) -> None:
    session, packed = _Session(_Host()), tmp_path / "base.safetensors"
    packed.write_bytes(b"base")
    kwargs = _kwargs(tmp_path, session, packed)
    kwargs["requests"].append(DietCalibrationRequest("p", "n", 8, 2, 4.0, 1, 8, 8))
    with pytest.raises(ValueError, match="one CFG"):
        run_diet_calibration_session(**kwargs)
    kwargs = _kwargs(tmp_path, session, packed)
    session.cfg.memory.frozen_weight_packed_state_path = str(tmp_path / "other")
    with pytest.raises(ValueError, match="exactly match"):
        run_diet_calibration_session(**kwargs)
    kwargs = _kwargs(tmp_path, session, packed)
    kwargs.update(output_path=packed, overwrite=True)
    with pytest.raises(ValueError, match="cannot overwrite"):
        run_diet_calibration_session(**kwargs)


def test_session_restores_observer_rng_and_mode_on_forward_failure(tmp_path: Path) -> None:
    host = _Host()
    target = SimpleNamespace(name="blocks.0.moe", host=host, num_experts=4,
        top_k=2, n_group=2, topk_group=2, group_score_topk=1,
        norm_topk_prob=True, route_scale=1.0)
    provider = SimpleNamespace(build_diet_calibration_targets=lambda pipeline: {target.name: target})
    packed = tmp_path / "base.safetensors"
    packed.write_bytes(b"base")
    session = _Session(host)
    model = torch.nn.Linear(1, 1).train(True)
    session.pipeline.get_training_model = lambda: model
    session.generate = mock.Mock(side_effect=RuntimeError("forward failed"))
    rng = torch.get_rng_state().clone()
    with mock.patch("mirai.core.training.calibration.diet.get_model_family_provider", return_value=provider):
        with pytest.raises(RuntimeError, match="forward failed"):
            run_diet_calibration_session(**_kwargs(tmp_path, session, packed))
    assert host.observer is None and model.training
    assert torch.equal(torch.get_rng_state(), rng)


def test_session_distributes_budget_across_every_request(tmp_path: Path) -> None:
    host = _Host()
    target = SimpleNamespace(name="blocks.0.moe", host=host, num_experts=4,
        top_k=2, n_group=2, topk_group=2, group_score_topk=1,
        norm_topk_prob=True, route_scale=1.0)
    provider = SimpleNamespace(build_diet_calibration_targets=lambda pipeline: {target.name: target})
    packed = tmp_path / "base.safetensors"
    packed.write_bytes(b"base")
    session = _Session(host)
    session.generate = mock.Mock(wraps=session.generate)
    kwargs = _kwargs(tmp_path, session, packed)
    kwargs["samples_per_layer"] = 6
    kwargs["requests"] = [
        DietCalibrationRequest("first", "negative", 7, 2, 5.0, 1, 8, 8),
        DietCalibrationRequest("second", "negative", 8, 2, 5.0, 1, 8, 8),
    ]
    with mock.patch("mirai.core.training.calibration.diet.get_model_family_provider", return_value=provider):
        report = run_diet_calibration_session(**kwargs)
    assert session.generate.call_count == 2
    assert [call.kwargs["prompt"] for call in session.generate.call_args_list] == ["first", "second"]
    assert report.requests == 2
    assert report.modules == {"blocks.0.moe": 6}
