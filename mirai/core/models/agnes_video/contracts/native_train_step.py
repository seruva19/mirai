"""Run the full released Agnes training and checkpoint-resume contract."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any

import torch

from mirai.core.builtins import register_builtin_components
from mirai.core.models.agnes_video.contracts._runtime import (
    ContractUnavailable,
    checkpoint_digest,
    cpu_tree,
    release_training_session,
    require_cuda,
    runtime_identity,
    tensor_tree_equal,
    unavailable_reason,
    write_report,
)
from mirai.core.models.agnes_video.contracts.report import (
    SCHEMA_VERSION,
    validate_train_report,
)
from mirai.core.persistence.checkpoints import save_checkpoint
from mirai.core.training.lifecycle.training_session import create_training_session
from mirai.core.training.lifecycle.training_step import run_training_step
from mirai.core.training.lifecycle.training_step_pre import (
    _build_training_batch_factory,
    resolve_step_sampling_context,
)
from mirai.core.training.lifecycle.session_state import (
    capture_torch_rng_state,
    restore_torch_rng_state,
)
from mirai.core.training.runtime.cli import load_runtime_config


def _modality_gradient_proof(session: Any) -> dict[str, Any]:
    trainer = session.trainer
    saved_trainer = cpu_tree(trainer.state_dict())
    saved_python_rng = session.rng.getstate()
    saved_torch_rng = capture_torch_rng_state()
    batch = _build_training_batch_factory(
        session=session,
        sampling_context=resolve_step_sampling_context(session),
    )(0)
    inputs = trainer.prepare_objective_calibration_inputs(batch, training=True)
    prediction = trainer.predict_objective_calibration_inputs(inputs, training=True)
    if not isinstance(prediction, torch.Tensor) or prediction.ndim != 2:
        raise RuntimeError("Agnes training must produce one packed joint prediction tensor.")
    target = trainer.pipeline.compute_target(inputs.noise, inputs.clean_latents, inputs.timestep)
    video_shape = tuple(inputs.extra_forward_kwargs["video_shape"])
    video_elements = 1
    for value in video_shape[1:]:
        video_elements *= int(value)
    mask = inputs.loss_mask.to(prediction)
    squared = (prediction.float() - target.float()).square() * mask.float()
    video_loss = squared[:, :video_elements].sum() / mask[:, :video_elements].sum().clamp_min(1)
    audio_loss = squared[:, video_elements:].sum() / mask[:, video_elements:].sum().clamp_min(1)
    named = list(trainer.pipeline.get_named_trainable_parameters())

    def has_grad(loss: torch.Tensor, *, retain_graph: bool) -> tuple[bool, int]:
        for _, parameter in named:
            parameter.grad = None
        loss.backward(retain_graph=retain_graph)
        gradients = [parameter.grad for _, parameter in named if parameter.grad is not None]
        return (
            any(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients),
            sum(1 for gradient in gradients if torch.isfinite(gradient).all() and gradient.abs().sum() > 0),
        )

    video_gradient, video_count = has_grad(video_loss, retain_graph=True)
    audio_gradient, audio_count = has_grad(audio_loss, retain_graph=False)
    source = batch.get("loss_mask")
    if source is None:
        source_mask_verified = True
    else:
        source_tensor = torch.as_tensor(source, device=mask.device)
        if source_tensor.ndim == 1 and source_tensor.shape[0] == mask.shape[0]:
            source_mask_verified = bool(torch.all(mask[source_tensor == 0] == 0))
        elif source_tensor.shape == mask.shape:
            source_mask_verified = bool(torch.all(mask[source_tensor == 0] == 0))
        else:
            video_source = source_tensor.expand(video_shape).flatten(1)
            source_mask_verified = bool(
                torch.all(mask[:, :video_elements][video_source == 0] == 0)
            )
    audio_shape = tuple(inputs.extra_forward_kwargs["audio_shape"])
    lengths = inputs.extra_forward_kwargs["audio_lengths"]
    valid_audio = torch.arange(audio_shape[1], device=mask.device)[None, :] < lengths[:, None]
    expected_audio = valid_audio[..., None].expand(audio_shape).flatten(1)
    noise_mask_verified = torch.equal(mask[:, video_elements:].bool(), expected_audio.bool()) if source is None else bool(
        torch.all(~mask[:, video_elements:].bool() | expected_audio.bool())
    )
    trainer.load_state_dict(saved_trainer)
    session.rng.setstate(saved_python_rng)
    restore_torch_rng_state(saved_torch_rng)
    for _, parameter in named:
        parameter.grad = None
    return {
        "joint_loss": float(video_loss.detach() + audio_loss.detach()),
        "video_loss": float(video_loss.detach()),
        "audio_loss": float(audio_loss.detach()),
        "video_target_gradient": video_gradient,
        "audio_target_gradient": audio_gradient,
        "adapter_gradient_tensors": min(video_count, audio_count),
        "source_mask_verified": source_mask_verified,
        "noise_mask_verified": noise_mask_verified,
    }


def _run_steps(session: Any, count: int) -> list[dict[str, Any]]:
    results = []
    for _ in range(count):
        result = run_training_step(session)
        if not result.advanced:
            raise RuntimeError("Agnes optimizer step did not advance.")
        results.append({"step": int(session.run_state.global_step), "grad_norm": result.grad_norm_value,
                        "metrics": result.step_metrics})
    return results


def run(config_path: str, output_path: str) -> dict[str, Any]:
    device = require_cuda()
    register_builtin_components()
    config, notes = load_runtime_config(config_path, entrypoint="train")
    if str(config.model.type) != "agnes-video":
        raise ContractUnavailable("The Agnes release contract requires model.type='agnes-video'.")
    source_sha = checkpoint_digest(config.model.path)
    with tempfile.TemporaryDirectory(prefix="mirai-agnes-contract-") as temporary:
        root = Path(temporary)
        continuous = create_training_session(config=config, config_path=config_path,
                                             runtime_policy_notes=list(notes), output_dir=root / "continuous")
        pipeline = continuous.trainer.pipeline
        runtime = runtime_identity(pipeline, device)
        trainable_names = [name for name, _ in pipeline.get_named_trainable_parameters()]
        only_lora = bool(trainable_names) and all("lora_" in name for name in trainable_names)
        proof = _modality_gradient_proof(continuous)
        initial_adapter = cpu_tree(pipeline.state_dict())
        continuous_steps = _run_steps(continuous, 2)
        continuous_state = cpu_tree(continuous.trainer.state_dict())
        continuous_optimizer = cpu_tree(continuous.optimizer.state_dict())
        continuous_scheduler = cpu_tree(continuous.scheduler.state_dict())
        continuous_python_rng = continuous.rng.getstate()
        continuous_torch_rng = cpu_tree(capture_torch_rng_state())
        final_adapter = cpu_tree(pipeline.state_dict())
        changed = sum(
            1 for key, value in final_adapter.items()
            if isinstance(value, torch.Tensor) and key in initial_adapter and not torch.equal(value, initial_adapter[key])
        )
        adapter_path = Path(output_path).with_suffix(".adapter.pt")
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_adapter, adapter_path)
        roundtrip = copy.deepcopy(final_adapter)
        pipeline.load_adapter_state(roundtrip)
        roundtrip_ok = tensor_tree_equal(cpu_tree(pipeline.state_dict()), final_adapter)
        del pipeline
        release_training_session(continuous)
        del continuous

        partial = create_training_session(config=config, config_path=config_path,
                                          runtime_policy_notes=list(notes), output_dir=root / "partial")
        _run_steps(partial, 1)
        checkpoint_path = Path(output_path).with_suffix(".step1.pt")
        save_checkpoint(checkpoint_path, partial.build_ckpt_payload(1))
        release_training_session(partial)
        del partial

        resumed = create_training_session(config=config, config_path=config_path,
                                          runtime_policy_notes=list(notes), output_dir=root / "resumed",
                                          resume_path=str(checkpoint_path))
        _run_steps(resumed, 1)
        resume = {
            "steps": 2,
            "split_step": 1,
            "adapter_state": tensor_tree_equal(cpu_tree(resumed.trainer.state_dict()), continuous_state),
            "optimizer_state": tensor_tree_equal(cpu_tree(resumed.optimizer.state_dict()), continuous_optimizer),
            "scheduler_state": tensor_tree_equal(cpu_tree(resumed.scheduler.state_dict()), continuous_scheduler),
            "python_rng": resumed.rng.getstate() == continuous_python_rng,
            "torch_rng": tensor_tree_equal(cpu_tree(capture_torch_rng_state()), continuous_torch_rng),
        }
        release_training_session(resumed)
        del resumed
    proof.update({
        "only_lora_trainable": only_lora,
        "optimizer_step": all(step["step"] > 0 and step["grad_norm"] > 0 for step in continuous_steps),
        "adapter_changed_tensors": changed,
        "adapter_roundtrip_same_base": roundtrip_ok,
    })
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": "agnes_native_train_step",
        "status": "passed",
        "runtime": runtime,
        "snapshot": {"source": str(config.model.path), "sha256": source_sha},
        "proof": proof,
        "resume_equivalence": resume,
        "artifacts": {"adapter": str(adapter_path), "checkpoint": str(checkpoint_path)},
    }
    failures = validate_train_report(report)
    if failures:
        report["status"] = "failed"
        report["failures"] = failures
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("MIRAI_AGNES_TRAIN_CONFIG"))
    parser.add_argument("--output", default=os.environ.get("MIRAI_AGNES_TRAIN_REPORT"))
    args = parser.parse_args(argv)
    if not args.config:
        report = {"schema_version": SCHEMA_VERSION, "contract": "agnes_native_train_step",
                  "status": "incomplete",
                  "failures": ["Set MIRAI_AGNES_TRAIN_CONFIG or pass --config."]}
    elif not args.output:
        report = {"schema_version": SCHEMA_VERSION, "contract": "agnes_native_train_step",
                  "status": "incomplete",
                  "failures": ["Set MIRAI_AGNES_TRAIN_REPORT or pass --output."]}
    else:
        try:
            report = run(args.config, args.output)
        except ContractUnavailable as exc:
            report = {"schema_version": SCHEMA_VERSION, "contract": "agnes_native_train_step",
                      "status": "incomplete", "failures": [str(exc)]}
        except BaseException as exc:
            unavailable = unavailable_reason(exc)
            report = {"schema_version": SCHEMA_VERSION, "contract": "agnes_native_train_step",
                      "status": "incomplete" if unavailable else "failed",
                      "failures": [unavailable or f"{type(exc).__name__}: {exc}"]}
            if unavailable is None:
                report["traceback"] = traceback.format_exc()
    if args.output:
        write_report(args.output, report)
    else:
        print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else (2 if report["status"] == "incomplete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
