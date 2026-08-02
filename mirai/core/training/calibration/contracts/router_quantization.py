"""Behavioral contracts for EAQuant-style router scale calibration."""

# ruff: noqa: E402

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from mirai.core.moe.calibration.router_quantization import (
    RouterLinearCalibrationBatch,
)
from mirai.core.moe.calibration.router_quantization import (
    RouterQuantizationCalibrationArtifact,
)
from mirai.core.moe.calibration.router_quantization import (
    RouterQuantizationCalibrationTarget,
)
from mirai.core.moe.calibration.router_quantization import (
    apply_router_quantization_calibration,
)
from mirai.core.moe.calibration.router_quantization import (
    calibrate_symmetric_int8_router,
)
from mirai.core.moe.calibration.router_quantization import kl_top_divergence
from mirai.core.moe.calibration.router_quantization import (
    load_router_quantization_calibration,
)
from mirai.core.moe.calibration.router_quantization import (
    save_router_quantization_calibration,
)
from mirai.core.moe.calibration.router_quantization import source_router_tensors
from mirai.core.moe.calibration.router_repair import router_tensor_fingerprint
from mirai.vendors.lingbot_video.transformer_lingbot_video import (
    LingBotVideoRouter,
)


def _synthetic_problem():
    generator = torch.Generator().manual_seed(731)
    weight = torch.randn(4, 8, generator=generator)
    weight[:, 0] = torch.tensor([90.0, -100.0, 110.0, -95.0])
    features = torch.randn(96, 8, generator=generator)
    features[:, 0] = 0.0
    return weight, RouterLinearCalibrationBatch(features=features)


def test_kl_top_full_relaxation_matches_full_kl() -> None:
    generator = torch.Generator().manual_seed(19)
    reference = torch.randn(13, 5, generator=generator)
    candidate = reference + 0.15 * torch.randn(13, 5, generator=generator)
    observed = kl_top_divergence(
        reference,
        candidate,
        top_k=2,
        relaxation=1.0,
    )
    expected = F.kl_div(
        F.log_softmax(candidate, dim=1),
        F.softmax(reference, dim=1),
        reduction="batchmean",
    )
    assert torch.allclose(observed, expected, atol=1e-7, rtol=1e-6)


def test_calibration_reduces_dual_objective_from_absmax() -> None:
    weight, batch = _synthetic_problem()
    result = calibrate_symmetric_int8_router(
        weight,
        batch,
        top_k=2,
        relaxation=0.0,
        minimum_clipping_ratio=0.2,
        grid_size=17,
        coordinate_sweeps=2,
    )
    assert result.calibrated_objective < result.baseline_objective
    assert result.calibrated_logit_mse < result.baseline_logit_mse
    assert bool((torch.as_tensor(result.clipping_ratio) < 1.0).any())


class _TargetHost(nn.Module):
    def forward(self, features):
        return features


def _target(weight: torch.Tensor):
    host = _TargetHost()
    parameter = nn.Parameter(weight.clone(), requires_grad=False)
    installed = []

    def _read():
        return parameter

    def _capture(args, _kwargs):
        return RouterLinearCalibrationBatch(features=args[0])

    def _install(scale):
        installed.append(torch.as_tensor(scale).clone())

    target = RouterQuantizationCalibrationTarget(
        name="layer.router",
        observation_module=host,
        num_experts=int(weight.shape[0]),
        input_features=int(weight.shape[1]),
        top_k=2,
        read_weight=_read,
        capture_batch=_capture,
        install_int8_scale=_install,
    ).validate()
    return target, parameter, installed


def test_artifact_roundtrip_and_exact_source_lineage(tmp_path) -> None:
    weight, batch = _synthetic_problem()
    target, parameter, installed = _target(weight)
    targets = {target.name: target}
    result = calibrate_symmetric_int8_router(
        weight,
        batch,
        top_k=2,
        grid_size=9,
    )
    artifact = RouterQuantizationCalibrationArtifact(
        modules={target.name: result},
        topology={
            target.name: {
                "num_experts": 4,
                "input_features": 8,
                "top_k": 2,
            }
        },
        dataset_snapshot_id="dataset",
        model_snapshot_id="model",
        config_snapshot_id="config",
        source_router_fingerprint=router_tensor_fingerprint(
            source_router_tensors(targets)
        ),
        relaxation=0.0,
        minimum_clipping_ratio=0.35,
        grid_size=9,
        coordinate_sweeps=1,
    ).validate()
    path = tmp_path / "router-calibration.safetensors"
    save_router_quantization_calibration(path, artifact)
    loaded = load_router_quantization_calibration(path)
    apply_router_quantization_calibration(targets, loaded)
    assert len(installed) == 1
    assert torch.equal(installed[0], torch.as_tensor(result.scale))

    with torch.no_grad():
        parameter[0, 0].add_(1.0)
    with pytest.raises(ValueError, match="different source weights"):
        apply_router_quantization_calibration(targets, loaded)


def _router() -> LingBotVideoRouter:
    router = LingBotVideoRouter(
        hidden_size=8,
        num_experts=4,
        top_k=2,
        score_func="softmax",
        norm_topk_prob=True,
        n_group=None,
        topk_group=None,
        route_scale=1.0,
    )
    with torch.no_grad():
        router.weight.copy_(
            torch.arange(32, dtype=torch.float32).reshape(4, 8).sub_(16.0)
        )
    router.weight.requires_grad_(False)
    return router


def test_disabled_runtime_path_retains_absmax_and_calibrated_path_uses_artifact_scale() -> None:
    reference = _router()
    expected = reference.weight.detach().abs().amax(dim=1) / 127.0
    reference.enable_int8_weight()
    assert torch.equal(reference.weight_scale, expected)

    calibrated = _router()
    scale = expected * 0.75
    calibrated.enable_int8_weight(calibrated_scale=scale)
    assert torch.equal(calibrated.weight_scale, scale)
    with pytest.raises(ValueError, match="different calibration scales"):
        calibrated.enable_int8_weight(calibrated_scale=expected)
