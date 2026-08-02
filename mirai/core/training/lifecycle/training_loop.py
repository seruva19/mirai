"""Training-loop execution for assembled training sessions."""

from __future__ import annotations

from mirai.core.training.lifecycle.artifacts import FinalRunArtifacts
from mirai.core.training.lifecycle.run_reporting import (
    emit_run_completed,
    emit_run_failed,
    emit_run_started,
    finalize_run_artifacts,
)
from mirai.core.training.calibration.expert_selection import (
    maybe_calibrate_expert_selection,
)
from mirai.core.training.calibration.lora_eva import maybe_initialize_lora_eva
from mirai.core.training.calibration.lora_ga import maybe_initialize_lora_ga
from mirai.core.training.control.live_control_actions import build_live_runtime_state_payload
from mirai.core.training.lifecycle.training_step import run_training_step
from mirai.core.training.lifecycle.training_session import TrainingSession


def run_training_loop(session: TrainingSession) -> FinalRunArtifacts:
    run_state = session.run_state
    if session.live_control is not None:
        session.live_control.publish_capabilities(
            global_step=int(run_state.global_step),
            active=True,
            runtime_state=build_live_runtime_state_payload(session),
        )
    emit_run_started(session)
    # Routing-guided expert selection (MoE-Sieve): profile routing and mask the
    # per-expert LoRA adapters before the first optimizer step. No-op unless
    # adapter.expert_selection == "routing_topk".
    maybe_calibrate_expert_selection(session)
    maybe_initialize_lora_ga(session)
    # EVA uses downstream activation PCs. Routing-guided expert selection runs
    # first so only experts that can contribute need to converge.
    maybe_initialize_lora_eva(session)
    try:
        while run_state.global_step < session.max_steps:
            _ = run_training_step(session)
            if run_state.stopped_early:
                break
        final_artifacts = finalize_run_artifacts(session)
        emit_run_completed(session, final_artifacts=final_artifacts)
        return final_artifacts
    except BaseException as exc:
        emit_run_failed(session, exc=exc)
        raise
    finally:
        if session.live_control is not None:
            session.live_control.close(
                global_step=int(run_state.global_step),
                runtime_state=build_live_runtime_state_payload(session),
            )
        session.close_callbacks()
