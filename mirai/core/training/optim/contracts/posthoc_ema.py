"""Contracts for power-function and post-hoc EMA."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mirai.config.schema import TrainingConfig  # noqa: E402
from mirai.core.training.runtime.contract import (  # noqa: E402
    validate_training_runtime_config,
)
from mirai.core.training.lifecycle.posthoc_ema import (  # noqa: E402
    update_and_maybe_save_posthoc_ema,
)
from mirai.core.training.optim.posthoc_ema import (  # noqa: E402
    load_posthoc_ema_snapshot,
)
from mirai.core.training.optim.posthoc_ema import (  # noqa: E402
    build_posthoc_ema_snapshot,
    init_posthoc_ema_state,
    power_function_beta,
    reconstruct_posthoc_ema,
    update_posthoc_ema_state,
)


def _state(value: float) -> dict[str, object]:
    return {
        "adapter.weight": torch.tensor([value], dtype=torch.float32),
        "adapter.index": torch.tensor([3], dtype=torch.int64),
        "adapter_type": "lora",
    }


def test_power_function_update_matches_direct_recurrence() -> None:
    state = init_posthoc_ema_state(_state(0.0), profile_stds=(0.05, 0.1))
    expected = [0.0, 0.0]
    for step, value in enumerate((1.0, 3.0, 8.0), start=1):
        state = update_posthoc_ema_state(state, _state(value), next_step=step)
        for index, std in enumerate((0.05, 0.1)):
            beta = power_function_beta(std=std, next_step=step, step_delta=1)
            expected[index] = (expected[index] * beta) + (value * (1.0 - beta))
            assert state["profiles"][index]["adapter.weight"].item() == pytest.approx(
                expected[index]
            )
    assert state["step"] == 3
    assert state["static_state"]["adapter_type"] == "lora"


def test_resume_state_produces_exact_next_update() -> None:
    uninterrupted = init_posthoc_ema_state(_state(0.0), profile_stds=(0.05, 0.1))
    uninterrupted = update_posthoc_ema_state(
        uninterrupted, _state(2.0), next_step=1
    )
    snapshot = build_posthoc_ema_snapshot(uninterrupted)
    uninterrupted = update_posthoc_ema_state(
        uninterrupted, _state(5.0), next_step=2
    )
    resumed = update_posthoc_ema_state(snapshot, _state(5.0), next_step=2)
    for left, right in zip(
        uninterrupted["profiles"],
        resumed["profiles"],
        strict=True,
    ):
        torch.testing.assert_close(
            left["adapter.weight"],
            right["adapter.weight"],
            rtol=0.0,
            atol=0.0,
        )


def test_reconstruction_recovers_tracked_input_profile() -> None:
    state = init_posthoc_ema_state(_state(0.0), profile_stds=(0.05, 0.1))
    snapshots = []
    for step, value in enumerate((1.0, 2.0, 4.0, 7.0), start=1):
        state = update_posthoc_ema_state(state, _state(value), next_step=step)
        if step in {2, 4}:
            snapshots.append(build_posthoc_ema_snapshot(state))
    reconstructed = reconstruct_posthoc_ema(
        snapshots,
        output_std=0.05,
        output_step=4,
    )
    torch.testing.assert_close(
        reconstructed["adapter_state"]["adapter.weight"],
        snapshots[-1]["profiles"][0]["adapter.weight"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.equal(
        reconstructed["adapter_state"]["adapter.index"],
        torch.tensor([3]),
    )
    assert np.isfinite(reconstructed["posthoc_ema"]["coefficients"]).all()


def test_invalid_profile_or_topology_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="at least two"):
        init_posthoc_ema_state(_state(0.0), profile_stds=(0.05,))
    state = init_posthoc_ema_state(_state(0.0), profile_stds=(0.05, 0.1))
    with pytest.raises(ValueError, match="topology changed"):
        update_posthoc_ema_state(
            state,
            {"different": torch.tensor([1.0])},
            next_step=1,
        )


def test_config_requires_enabled_bounded_snapshot_policy() -> None:
    validate_training_runtime_config(TrainingConfig())
    enabled = TrainingConfig.from_dict(
        {
            "training": {
                "posthoc_ema_enabled": True,
                "posthoc_ema_profile_stds": [0.05, 0.1],
                "posthoc_ema_snapshot_every_n_steps": 10,
            }
        }
    )
    validate_training_runtime_config(enabled)
    with pytest.raises(ValueError, match="must be > 0"):
        validate_training_runtime_config(
            TrainingConfig.from_dict(
                {"training": {"posthoc_ema_enabled": True}}
            )
        )
    with pytest.raises(ValueError, match="at least two"):
        validate_training_runtime_config(
            TrainingConfig.from_dict(
                {
                    "training": {
                        "posthoc_ema_enabled": True,
                        "posthoc_ema_profile_stds": [0.05],
                        "posthoc_ema_snapshot_every_n_steps": 10,
                    }
                }
            )
        )


def test_lifecycle_writes_atomic_adapter_only_snapshot(tmp_path) -> None:
    live = _state(2.0)
    session = SimpleNamespace(
        config=SimpleNamespace(
            training=SimpleNamespace(
                posthoc_ema_enabled=True,
                posthoc_ema_snapshot_every_n_steps=1,
            )
        ),
        run_state=SimpleNamespace(
            global_step=1,
            posthoc_ema_state=init_posthoc_ema_state(
                _state(0.0),
                profile_stds=(0.05, 0.1),
            ),
        ),
        trainer=SimpleNamespace(
            pipeline=SimpleNamespace(state_dict=lambda: live)
        ),
        manifest=SimpleNamespace(
            dataset_snapshot_id="dataset",
            cache_snapshot_id="cache",
            model_snapshot_id="model",
            config_snapshot_id="config",
            manifest_sha256="manifest",
        ),
        ckpt_dir=tmp_path,
        log_on_this_rank=True,
    )
    path = update_and_maybe_save_posthoc_ema(session)
    assert path is not None and path.exists()
    assert path.with_name(path.name + ".sha256").exists()
    snapshot = load_posthoc_ema_snapshot(path)
    assert snapshot["step"] == 1
    assert snapshot["static_state"]["adapter_type"] == "lora"
