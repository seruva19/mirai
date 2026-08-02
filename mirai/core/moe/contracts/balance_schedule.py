"""Behavioral contracts for scheduled auxiliary balance-loss relaxation."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from mirai.config.schema import TrainingConfig  # noqa: E402
from mirai.core.models.lingbot_video.pipeline import (  # noqa: E402
    LingBotVideoPipeline,
)
from mirai.core.moe.adaptation.balance_schedule import (  # noqa: E402
    AuxiliaryBalanceLossSchedule,
)
from mirai.core.training.runtime.contract import (  # noqa: E402
    validate_training_runtime_config,
)


def _config(**overrides) -> TrainingConfig:
    params = {
        "variant": "tiny-video",
        "hidden_size": 12,
        "attention_heads": 2,
        "num_layers": 2,
        "num_experts": 4,
        "experts_per_token": 2,
        "shared_experts": 0,
        "latent_channels": 1,
    }
    params.update(overrides)
    return TrainingConfig.from_dict(
        {
            "model": {
                "type": "lingbot_video",
                "params": params,
            }
        }
    )


def test_default_constructs_no_schedule_owner() -> None:
    params = _config().model.params
    assert AuxiliaryBalanceLossSchedule.from_model_params(params) is None


def test_schedule_switches_at_exact_global_step() -> None:
    schedule = AuxiliaryBalanceLossSchedule(disable_step=90)
    assert schedule.weight(0.01, step=0) == 0.01
    assert schedule.weight(0.01, step=89) == 0.01
    assert schedule.weight(0.01, step=90) == 0.0
    assert schedule.weight(0.01, step=200) == 0.0
    assert schedule.is_exploration(step=89)
    assert not schedule.is_exploration(step=90)


def test_schedule_rejects_invalid_or_semantically_ambiguous_config() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        _config(moe_balance_loss_disable_step=-1)
    with pytest.raises(ValueError, match="requires moe_balance_mode='aux_loss'"):
        validate_training_runtime_config(
            _config(
                moe_balance_loss_disable_step=10,
                moe_balance_mode="off",
            )
        )
    with pytest.raises(ValueError, match="requires moe_aux_loss_weight > 0"):
        validate_training_runtime_config(
            _config(
                moe_balance_loss_disable_step=10,
                moe_aux_loss_weight=0.0,
            )
        )
    with pytest.raises(ValueError, match="requires moe_bias_update_rate=0"):
        validate_training_runtime_config(
            _config(
                moe_balance_loss_disable_step=10,
                moe_bias_update_rate=0.1,
            )
        )


def test_lingbot_pipeline_relaxes_only_auxiliary_balance_weight() -> None:
    pipeline = LingBotVideoPipeline.from_training_config(
        _config(
            moe_balance_loss_disable_step=10,
            moe_aux_loss_weight=0.025,
            moe_router_z_loss_weight=0.003,
        )
    )
    assert pipeline.supports_balance_loss_schedule_progress()
    assert pipeline._moe_aux_loss_weight == 0.025

    pipeline.set_balance_loss_schedule_progress(step=9)
    assert pipeline._moe_aux_loss_weight == 0.025
    assert pipeline._moe_router_z_loss_weight == 0.003
    assert pipeline._moe_bias_update_rate == 0.0

    pipeline.set_balance_loss_schedule_progress(step=10)
    assert pipeline._moe_aux_loss_weight == 0.0
    assert pipeline._moe_router_z_loss_weight == 0.003
    assert pipeline._moe_bias_update_rate == 0.0


def test_schedule_is_resume_exact_without_mutable_state() -> None:
    first = AuxiliaryBalanceLossSchedule(disable_step=10)
    resumed = AuxiliaryBalanceLossSchedule(disable_step=10)
    assert first.weight(0.02, step=37) == resumed.weight(0.02, step=37) == 0.0
