"""Power-function EMA tracking and post-hoc adapter reconstruction.

The implementation follows the response functions and correlation system from
EDM2 without depending on the reference repository at runtime.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mirai.core.persistence.checkpoints import load_checkpoint, save_checkpoint

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(f"Torch is required for post-hoc EMA: {exc}")


POSTHOC_EMA_SCHEMA_VERSION = 1
POSTHOC_EMA_SNAPSHOT_KIND = "mirai_power_function_ema_snapshot"


def _validate_std(value: float) -> float:
    std = float(value)
    if not math.isfinite(std) or not 0.0 < std < 0.289:
        raise ValueError(
            "Power-function EMA relative standard deviation must be finite "
            "and in (0, 0.289)."
        )
    return std


def normalize_profile_stds(values: Iterable[float]) -> tuple[float, ...]:
    stds = tuple(_validate_std(value) for value in values)
    if len(stds) < 2:
        raise ValueError("Post-hoc EMA requires at least two profile stds.")
    if len(set(stds)) != len(stds):
        raise ValueError("Post-hoc EMA profile stds must be distinct.")
    return stds


def power_function_exponent(std: float) -> float:
    """Convert relative response standard deviation to power exponent."""

    validated = _validate_std(std)
    inverse_variance = validated**-2
    roots = np.roots((1.0, 7.0, 16.0 - inverse_variance, 12.0 - inverse_variance))
    exponent = float(np.max(roots.real))
    if not math.isfinite(exponent) or exponent <= -1.0:
        raise ValueError("Power-function EMA exponent is not numerically valid.")
    return exponent


def power_function_beta(
    *,
    std: float,
    next_step: int,
    step_delta: int,
) -> float:
    """Return the exact recurrence coefficient for one training interval."""

    next_value = int(next_step)
    delta = int(step_delta)
    if next_value <= 0 or delta <= 0 or delta > next_value:
        raise ValueError(
            "Power-function EMA requires 0 < step_delta <= next_step."
        )
    base = 1.0 - (float(delta) / float(next_value))
    return float(base ** (power_function_exponent(std) + 1.0))


def _split_state(
    state: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    floating: dict[str, torch.Tensor] = {}
    static: dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            floating[name] = value.detach().cpu().float().clone()
        elif isinstance(value, torch.Tensor):
            static[name] = value.detach().cpu().clone()
        else:
            static[name] = value
    return floating, static


def init_posthoc_ema_state(
    live_state: dict[str, Any],
    *,
    profile_stds: Iterable[float],
) -> dict[str, Any]:
    stds = normalize_profile_stds(profile_stds)
    floating, static = _split_state(live_state)
    if not floating:
        raise ValueError("Post-hoc EMA found no floating adapter tensors.")
    return {
        "schema_version": POSTHOC_EMA_SCHEMA_VERSION,
        "step": 0,
        "stds": list(stds),
        "profiles": [
            {key: value.clone() for key, value in floating.items()} for _ in stds
        ],
        "static_state": static,
    }


def normalize_posthoc_ema_state(state: dict[str, Any]) -> dict[str, Any]:
    if int(state.get("schema_version", -1)) != POSTHOC_EMA_SCHEMA_VERSION:
        raise ValueError("Unsupported post-hoc EMA state schema.")
    step = int(state.get("step", -1))
    if step < 0:
        raise ValueError("Post-hoc EMA state step must be >= 0.")
    stds = normalize_profile_stds(state.get("stds", ()))
    profiles = state.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != len(stds):
        raise ValueError("Post-hoc EMA profile count does not match its stds.")
    normalized_profiles: list[dict[str, torch.Tensor]] = []
    expected_keys: tuple[str, ...] | None = None
    for profile in profiles:
        if not isinstance(profile, dict) or not profile:
            raise ValueError("Post-hoc EMA profiles must be non-empty mappings.")
        normalized = {
            str(key): value.detach().cpu().float().clone()
            for key, value in profile.items()
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
        }
        keys = tuple(sorted(normalized))
        if not keys or (expected_keys is not None and keys != expected_keys):
            raise ValueError("Post-hoc EMA profiles have inconsistent tensor keys.")
        expected_keys = keys
        normalized_profiles.append(normalized)
    static = state.get("static_state", {})
    if not isinstance(static, dict):
        raise ValueError("Post-hoc EMA static state must be a mapping.")
    _, normalized_static = _split_state(static)
    return {
        "schema_version": POSTHOC_EMA_SCHEMA_VERSION,
        "step": step,
        "stds": list(stds),
        "profiles": normalized_profiles,
        "static_state": normalized_static,
    }


@torch.no_grad()
def update_posthoc_ema_state(
    state: dict[str, Any],
    live_state: dict[str, Any],
    *,
    next_step: int,
) -> dict[str, Any]:
    normalized = normalize_posthoc_ema_state(state)
    current_step = int(normalized["step"])
    target_step = int(next_step)
    if target_step <= current_step:
        raise ValueError("Post-hoc EMA update step must increase monotonically.")
    floating, static = _split_state(live_state)
    profiles = normalized["profiles"]
    expected = set(profiles[0])
    if set(floating) != expected:
        raise ValueError("Post-hoc EMA adapter tensor topology changed during training.")
    for std, profile in zip(normalized["stds"], profiles, strict=True):
        beta = power_function_beta(
            std=float(std),
            next_step=target_step,
            step_delta=target_step - current_step,
        )
        for key, live in floating.items():
            previous = profile[key]
            if previous.shape != live.shape:
                raise ValueError(
                    f"Post-hoc EMA tensor shape changed for '{key}'."
                )
            previous.lerp_(live, 1.0 - beta)
    normalized["step"] = target_step
    normalized["static_state"] = static
    return normalized


def build_posthoc_ema_snapshot(
    state: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_posthoc_ema_state(state)
    return {
        "kind": POSTHOC_EMA_SNAPSHOT_KIND,
        "schema_version": POSTHOC_EMA_SCHEMA_VERSION,
        "step": int(normalized["step"]),
        "stds": list(normalized["stds"]),
        "profiles": normalized["profiles"],
        "static_state": normalized["static_state"],
        "metadata": dict(metadata or {}),
    }


def save_posthoc_ema_snapshot(
    path: str | Path,
    state: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    return save_checkpoint(
        path,
        build_posthoc_ema_snapshot(state, metadata=metadata),
    )


def load_posthoc_ema_snapshot(path: str | Path) -> dict[str, Any]:
    payload = load_checkpoint(path)
    if payload.get("kind") != POSTHOC_EMA_SNAPSHOT_KIND:
        raise ValueError(f"'{path}' is not a Mirai post-hoc EMA snapshot.")
    return normalize_posthoc_ema_state(payload)


def _profile_correlation(
    a_step: np.ndarray,
    a_std: np.ndarray,
    b_step: np.ndarray,
    b_std: np.ndarray,
) -> np.ndarray:
    a_exp = np.vectorize(power_function_exponent)(a_std)
    b_exp = np.vectorize(power_function_exponent)(b_std)
    ratio = a_step / b_step
    time_exp = np.where(a_step < b_step, b_exp, -a_exp)
    maximum = np.maximum(a_step, b_step)
    numerator = (a_exp + 1.0) * (b_exp + 1.0) * ratio**time_exp
    denominator = (a_exp + b_exp + 1.0) * maximum
    return numerator / denominator


def solve_posthoc_coefficients(
    *,
    input_steps: Iterable[int],
    input_stds: Iterable[float],
    output_step: int,
    output_std: float,
) -> np.ndarray:
    steps = np.asarray(tuple(int(value) for value in input_steps), dtype=np.float64)
    stds = np.asarray(tuple(_validate_std(value) for value in input_stds), dtype=np.float64)
    target_step = int(output_step)
    target_std = _validate_std(output_std)
    if steps.ndim != 1 or stds.ndim != 1 or steps.size != stds.size or not steps.size:
        raise ValueError("Post-hoc EMA reconstruction inputs must be aligned.")
    if np.any(steps <= 0) or target_step <= 0 or np.any(steps > target_step):
        raise ValueError("Post-hoc EMA reconstruction steps are outside the target.")
    a_steps = steps.reshape(-1, 1)
    a_stds = stds.reshape(-1, 1)
    gram = _profile_correlation(
        a_steps,
        a_stds,
        steps.reshape(1, -1),
        stds.reshape(1, -1),
    )
    target = _profile_correlation(
        a_steps,
        a_stds,
        np.asarray([[target_step]], dtype=np.float64),
        np.asarray([[target_std]], dtype=np.float64),
    )
    coefficients = np.linalg.solve(gram, target).reshape(-1)
    coefficients /= coefficients.sum()
    return coefficients


def reconstruct_posthoc_ema(
    snapshots: Iterable[dict[str, Any]],
    *,
    output_std: float,
    output_step: int | None = None,
) -> dict[str, Any]:
    normalized = [normalize_posthoc_ema_state(snapshot) for snapshot in snapshots]
    if not normalized:
        raise ValueError("Post-hoc EMA reconstruction requires snapshots.")
    target_step = (
        max(int(snapshot["step"]) for snapshot in normalized)
        if output_step is None
        else int(output_step)
    )
    eligible = [
        snapshot for snapshot in normalized if 0 < int(snapshot["step"]) <= target_step
    ]
    if not eligible or not any(int(item["step"]) == target_step for item in eligible):
        raise ValueError("Output step must match an available snapshot.")
    input_steps: list[int] = []
    input_stds: list[float] = []
    profiles: list[dict[str, torch.Tensor]] = []
    for snapshot in eligible:
        for std, profile in zip(
            snapshot["stds"],
            snapshot["profiles"],
            strict=True,
        ):
            input_steps.append(int(snapshot["step"]))
            input_stds.append(float(std))
            profiles.append(profile)
    coefficients = solve_posthoc_coefficients(
        input_steps=input_steps,
        input_stds=input_stds,
        output_step=target_step,
        output_std=output_std,
    )
    expected_keys = tuple(sorted(profiles[0]))
    if any(tuple(sorted(profile)) != expected_keys for profile in profiles[1:]):
        raise ValueError("Post-hoc EMA snapshots have inconsistent tensor topology.")
    target_snapshot = next(
        item for item in reversed(eligible) if int(item["step"]) == target_step
    )
    reconstructed = dict(target_snapshot["static_state"])
    for key in expected_keys:
        reference = profiles[0][key]
        accumulator = torch.zeros_like(reference, dtype=torch.float64, device="cpu")
        for coefficient, profile in zip(coefficients, profiles, strict=True):
            value = profile[key]
            if value.shape != reference.shape:
                raise ValueError(
                    f"Post-hoc EMA snapshots disagree on shape for '{key}'."
                )
            accumulator.add_(value.double(), alpha=float(coefficient))
        reconstructed[key] = accumulator.float()
    return {
        "adapter_state": reconstructed,
        "posthoc_ema": {
            "schema_version": POSTHOC_EMA_SCHEMA_VERSION,
            "output_step": target_step,
            "output_std": float(output_std),
            "input_profile_count": len(profiles),
            "coefficients": coefficients.tolist(),
        },
    }
