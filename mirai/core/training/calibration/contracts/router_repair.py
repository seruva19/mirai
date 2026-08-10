"""Behavioral contracts for post-compression Router-KD."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mirai.core.models.providers import ModelFamilyProvider
from mirai.core.models.providers import register_model_family_provider
from mirai.core.moe.calibration.router_repair import (
    RouterRepairTarget,
    apply_router_repair_artifact,
    diffusion_router_kd_loss,
    load_router_repair_artifact,
    router_target_tensors,
    router_tensor_fingerprint,
    save_router_repair_artifact,
)
from mirai.core.training.calibration.router_repair import (
    fit_router_kd_session,
    load_router_kd_example,
    save_router_kd_example,
)
from mirai.core.training.strategies.base import TrainingInputs


class _TinyRouterStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(2, 2, bias=False)
        self.frozen = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.router.weight.zero_()
            self.frozen.weight.copy_(torch.eye(2))

    def forward(self, inputs: TrainingInputs) -> torch.Tensor:
        return self.frozen(inputs.noisy_latents) + self.router(inputs.noisy_latents)


class _TinyPipeline:
    def __init__(self, model: _TinyRouterStudent) -> None:
        self.model = model

    def get_training_model(self):
        return self.model

    def finish_backward_offloads(self) -> None:
        return None

    def prepare_optimizer_step(self) -> None:
        return None

    def finish_optimizer_step(self) -> None:
        return None


class _TinyTrainer:
    def __init__(self, model: _TinyRouterStudent) -> None:
        self.pipeline = _TinyPipeline(model)

    def predict_calibration_inputs(self, inputs: TrainingInputs):
        return self.pipeline.model(inputs)

    def predict_objective_calibration_inputs(
        self,
        inputs: TrainingInputs,
        *,
        training: bool = False,
    ):
        _ = training
        return self.pipeline.model(inputs)

    def evaluate_calibration_task_loss(self, *, batch, inputs, prediction):
        _ = batch
        target = 1.5 * inputs.clean_latents
        return SimpleNamespace(loss_pre_accum=(prediction - target).square().mean())


class _TinyRouterRepairProvider(ModelFamilyProvider):
    def __init__(self) -> None:
        super().__init__(
            model_type="router-kd-contract",
            post_compression_router_repair=True,
        )

    def build_router_repair_targets(self, pipeline):
        target = RouterRepairTarget(
            name="router.weight",
            parameter=pipeline.model.router.weight,
        ).validate()
        return {target.name: target}


try:
    register_model_family_provider(
        "router-kd-contract",
        _TinyRouterRepairProvider(),
    )
except KeyError:
    pass


def _inputs(values: list[list[float]]) -> TrainingInputs:
    tensor = torch.tensor(values, dtype=torch.float32)
    return TrainingInputs(
        noisy_latents=tensor,
        timestep=torch.zeros(tensor.shape[0]),
        noise=torch.zeros_like(tensor),
        clean_latents=tensor.clone(),
        text_embeds={},
    )


def _session(model: _TinyRouterStudent):
    return SimpleNamespace(
        config=SimpleNamespace(model=SimpleNamespace(type="router-kd-contract")),
        trainer=_TinyTrainer(model),
        compute_device=torch.device("cpu"),
        manifest=SimpleNamespace(
            dataset_snapshot_id="dataset:test",
            model_snapshot_id="model:student",
            config_snapshot_id="config:test",
        ),
    )


def _write_examples(root: Path) -> list[Path]:
    teacher_weight = torch.tensor([[0.5, -0.25], [0.1, 0.4]])
    paths: list[Path] = []
    for index, values in enumerate(
        (
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [-1.0, 0.5]],
            [[0.25, -0.5], [0.75, 0.25]],
        )
    ):
        inputs = _inputs(values)
        teacher = inputs.noisy_latents + inputs.noisy_latents @ teacher_weight.T
        path = save_router_kd_example(
            root / f"{index}.safetensors",
            inputs=inputs,
            teacher_prediction=teacher,
        )
        restored_inputs, restored_teacher = load_router_kd_example(path)
        torch.testing.assert_close(
            restored_inputs.noisy_latents,
            inputs.noisy_latents,
        )
        torch.testing.assert_close(restored_teacher, teacher)
        paths.append(path)
    return paths


def _write_task_examples(root: Path) -> list[Path]:
    paths = []
    for index, values in enumerate(
        (
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [-1.0, 0.5]],
            [[0.25, -0.5], [0.75, 0.25]],
        )
    ):
        inputs = _inputs(values)
        paths.append(
            save_router_kd_example(
                root / f"{index}.safetensors",
                inputs=inputs,
                teacher_prediction=1.5 * inputs.clean_latents,
            )
        )
    return paths


def test_diffusion_router_kd_uses_final_prediction_and_detaches_teacher() -> None:
    student = torch.tensor([[1.0, 3.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loss = diffusion_router_kd_loss(student, teacher)
    torch.testing.assert_close(loss, torch.tensor(2.5))
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([[1.0, 2.0]]))
    assert teacher.grad is None


def test_router_kd_improves_holdout_and_changes_only_router(tmp_path: Path) -> None:
    paths = _write_examples(tmp_path / "examples")
    model = _TinyRouterStudent()
    frozen_before = model.frozen.weight.detach().clone()
    report = fit_router_kd_session(
        _session(model),
        example_paths=paths,
        train_examples=2,
        learning_rate=0.1,
        gradient_accumulation=2,
        compressed_artifact_fingerprint="sha256:" + ("a" * 64),
        teacher_model_snapshot_id="model:teacher",
    )
    assert report.artifact.repaired_holdout_mse <= (
        report.artifact.baseline_holdout_mse
    )
    assert report.artifact.lineage.initial_router_fingerprint != (
        report.artifact.lineage.repaired_router_fingerprint
    )
    torch.testing.assert_close(model.frozen.weight, frozen_before, rtol=0, atol=0)

    artifact_path = save_router_repair_artifact(
        tmp_path / "repair.safetensors",
        report.artifact,
    )
    restored = load_router_repair_artifact(artifact_path)
    fresh = _TinyRouterStudent()
    targets = {
        "router.weight": RouterRepairTarget(
            name="router.weight",
            parameter=fresh.router.weight,
        )
    }
    assert router_tensor_fingerprint(router_target_tensors(targets)) == (
        restored.lineage.initial_router_fingerprint
    )
    apply_router_repair_artifact(
        targets,
        restored,
        compressed_artifact_fingerprint="sha256:" + ("a" * 64),
    )
    torch.testing.assert_close(
        fresh.router.weight,
        model.router.weight,
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="different compressed base"):
        apply_router_repair_artifact(
            targets,
            restored,
            compressed_artifact_fingerprint="sha256:" + ("b" * 64),
        )


def test_router_task_optimizes_native_loss_with_prediction_guard(tmp_path: Path) -> None:
    paths = _write_task_examples(tmp_path / "task-examples")
    model = _TinyRouterStudent()
    report = fit_router_kd_session(
        _session(model),
        example_paths=paths,
        train_examples=2,
        learning_rate=0.1,
        gradient_accumulation=2,
        compressed_artifact_fingerprint="sha256:" + ("d" * 64),
        teacher_model_snapshot_id="model:teacher",
        repair_objective="task",
    )
    assert report.repair_objective == "task"
    assert report.artifact.repaired_holdout_mse < (
        report.artifact.baseline_holdout_mse
    )


def test_zero_step_router_kd_is_exact_noop(tmp_path: Path) -> None:
    paths = _write_examples(tmp_path / "examples")
    model = _TinyRouterStudent()
    before = model.router.weight.detach().clone()
    report = fit_router_kd_session(
        _session(model),
        example_paths=[paths[-1]],
        train_examples=0,
        learning_rate=0.1,
        gradient_accumulation=1,
        compressed_artifact_fingerprint="sha256:" + ("c" * 64),
        teacher_model_snapshot_id="model:teacher",
    )
    torch.testing.assert_close(model.router.weight, before, rtol=0, atol=0)
    assert report.artifact.lineage.initial_router_fingerprint == (
        report.artifact.lineage.repaired_router_fingerprint
    )
    assert report.artifact.baseline_holdout_mse == (
        report.artifact.repaired_holdout_mse
    )
