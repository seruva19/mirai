"""Collect clip and timestep route signatures for the microbatch planner."""

from __future__ import annotations

import torch

from mirai.core.training.data.batches import build_batch_from_records
from mirai.core.training.objectives.sampling import NoiseGenerator
from mirai.core.training.residency.device_placement import move_batch_to_device


class _CalibrationTimestep:
    def __init__(self, value: float):
        self.value = value

    def sample(self, batch_size, like):
        return like.new_full((batch_size,), self.value)


def collect_routing_batch_profile(session, *, max_samples, timestep_bins, trials, cache_experts):
    if min(max_samples, timestep_bins, trials) < 1 or cache_experts < 0:
        raise ValueError("Routing calibration requires positive sample, bin, and trial counts and a nonnegative cache size.")
    trainer = session.trainer
    pipeline = trainer.pipeline
    observe = getattr(pipeline, "observe_routing_profile", None)
    if observe is None or session.config.strategy.type != "text_to_video" or session.config.training.objective != "flow_matching":
        raise ValueError("The selected pipeline and strategy do not support routing profile calibration.")
    samples = {}
    noise = NoiseGenerator(seed=session.config.training.seed + 1)
    validation = trainer.begin_validation()
    try:
        with torch.no_grad():
            for record in session.train_records:
                sample_id = str(record.get("sample_id") or "")
                if not sample_id or sample_id in samples:
                    continue
                batch = move_batch_to_device(
                    build_batch_from_records([record], training=False), device=session.compute_device, dtype=session.compute_dtype,
                )
                bins = []
                for index in range(timestep_bins):
                    timestep = _CalibrationTimestep((index + 0.5) / timestep_bins)
                    routes = set()
                    for _ in range(trials):
                        inputs = trainer.strategy.prepare_inputs(
                            batch, pipeline, timestep, noise, training=False, objective=trainer.objective,
                        )
                        with observe() as experts:
                            pipeline.forward(inputs.noisy_latents, inputs.timestep, inputs.text_embeds, **inputs.extra_forward_kwargs)
                        routes.update(experts)
                    bins.append(sorted(routes))
                samples[sample_id] = bins
                if len(samples) == max_samples:
                    break
    finally:
        trainer.end_validation(validation)
    if not samples:
        raise ValueError("Routing calibration requires cache records with sample IDs.")
    return {"schema_version": 1, "cache_experts": cache_experts, "timestep_bins": timestep_bins, "samples": samples}
