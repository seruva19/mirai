"""Load-once, generate-many inference session (A4).

The :class:`InferenceSession` performs exactly what ``scripts/infer.py`` does
once per process today (register components, load the runtime config, build the
:class:`~mirai.core.training.trainer.Trainer`, load a checkpoint/adapter, place
the pipeline on the compute device) and then exposes ``generate()`` for a single
denoise + decode run. Building the model once and calling ``generate()`` many
times amortizes the fixed build/placement cost across prompts.
"""

from __future__ import annotations

from mirai.core.inference.session import (
    InferenceSession,
    decode_pipeline_video,
    resolve_inference_mode,
)

__all__ = [
    "InferenceSession",
    "decode_pipeline_video",
    "resolve_inference_mode",
]
