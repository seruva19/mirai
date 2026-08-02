"""Deterministic timestep and noise sampling for training objectives."""

from __future__ import annotations

import math
import random
from typing import Any, Protocol

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mirai.core.tensors import is_torch_tensor


TIMESTEP_SAMPLING_MODES = frozenset({"uniform", "logit_normal", "mode"})
MODE_SHIFT_MIN_SCALE = -1.0
MODE_SHIFT_MAX_SCALE = 2.0 / (math.pi - 2.0)


class TimestepSampler(Protocol):
    """Checkpointable source of normalized training timesteps."""

    def sample(self, batch_size: int, like=None) -> Any:
        ...

    def state_dict(self) -> dict:
        ...

    def load_state_dict(self, state: dict) -> None:
        ...


class UniformTimestepSampler:
    def __init__(self, seed: int, eps: float = 1e-5):
        self._rng = random.Random(seed)
        self._eps = eps
        self._torch_generator = None
        if torch is not None:
            self._torch_generator = torch.Generator()
            self._torch_generator.manual_seed(seed)

    def sample(self, batch_size: int, like=None):
        lo = self._eps
        hi = 1.0 - self._eps
        if is_torch_tensor(like):
            # Generate on CPU with the seeded generator, then move to target
            # device. torch.Generator only works reliably on CPU.
            t = torch.rand(
                (batch_size,),
                generator=self._torch_generator,
                dtype=like.dtype if like.dtype.is_floating_point else torch.float32,
            )
            t = t * (hi - lo) + lo
            return t.to(device=like.device)
        return [self._rng.uniform(lo, hi) for _ in range(batch_size)]

    def state_dict(self) -> dict:
        out = {"python_rng_state": self._rng.getstate()}
        if self._torch_generator is not None and torch is not None:
            out["torch_rng_state"] = self._torch_generator.get_state()
        return out

    def load_state_dict(self, state: dict) -> None:
        py_state = state.get("python_rng_state")
        if py_state is not None:
            self._rng.setstate(py_state)
        torch_state = state.get("torch_rng_state")
        if (
            torch_state is not None
            and self._torch_generator is not None
            and torch is not None
        ):
            self._torch_generator.set_state(torch_state)


class LogitNormalTimestepSampler(UniformTimestepSampler):
    """Sample ``sigmoid(N(mean, std))`` timesteps."""

    def __init__(
        self,
        seed: int,
        eps: float = 1e-5,
        *,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        if not math.isfinite(float(mean)):
            raise ValueError("Logit-normal timestep mean must be finite.")
        if not math.isfinite(float(std)) or float(std) <= 0.0:
            raise ValueError("Logit-normal timestep std must be finite and > 0.")
        super().__init__(seed=seed, eps=eps)
        self._mean = float(mean)
        self._std = float(std)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exponent = math.exp(value)
        return exponent / (1.0 + exponent)

    def sample(self, batch_size: int, like=None):
        lo = self._eps
        hi = 1.0 - self._eps
        if is_torch_tensor(like):
            dtype = like.dtype if like.dtype.is_floating_point else torch.float32
            values = torch.randn(
                (batch_size,),
                generator=self._torch_generator,
                dtype=dtype,
            )
            values = torch.sigmoid(values * self._std + self._mean)
            return values.clamp(min=lo, max=hi).to(device=like.device)
        return [
            min(
                hi,
                max(
                    lo,
                    self._sigmoid(self._rng.gauss(self._mean, self._std)),
                ),
            )
            for _ in range(batch_size)
        ]


class ModeShiftTimestepSampler(UniformTimestepSampler):
    """Sample the rectified-flow mode distribution from uniform variates."""

    def __init__(
        self,
        seed: int,
        eps: float = 1e-5,
        *,
        scale: float = 1.29,
    ):
        scale = float(scale)
        if (
            not math.isfinite(scale)
            or scale < MODE_SHIFT_MIN_SCALE
            or scale > MODE_SHIFT_MAX_SCALE
        ):
            raise ValueError(
                "Mode-shift timestep scale must be finite and in "
                f"[{MODE_SHIFT_MIN_SCALE}, {MODE_SHIFT_MAX_SCALE}]."
            )
        super().__init__(seed=seed, eps=eps)
        self._scale = scale

    def sample(self, batch_size: int, like=None):
        lo = self._eps
        hi = 1.0 - self._eps
        if is_torch_tensor(like):
            dtype = like.dtype if like.dtype.is_floating_point else torch.float32
            uniform = torch.rand(
                (batch_size,),
                generator=self._torch_generator,
                dtype=dtype,
            )
            cosine = torch.cos((math.pi / 2.0) * uniform)
            values = (
                1.0
                - uniform
                - self._scale * (cosine.square() - 1.0 + uniform)
            )
            return values.clamp(min=lo, max=hi).to(device=like.device)
        values = []
        for _ in range(batch_size):
            uniform = self._rng.random()
            cosine = math.cos((math.pi / 2.0) * uniform)
            value = (
                1.0
                - uniform
                - self._scale * (cosine * cosine - 1.0 + uniform)
            )
            values.append(min(hi, max(lo, value)))
        return values


def build_timestep_sampler(
    *,
    mode: str,
    seed: int,
    eps: float,
    mean: float = 0.0,
    std: float = 1.0,
    mode_scale: float = 1.29,
) -> TimestepSampler:
    """Construct a configured sampler without changing the uniform default path."""

    normalized = str(mode).strip().lower()
    if normalized == "uniform":
        return UniformTimestepSampler(seed=seed, eps=eps)
    if normalized == "logit_normal":
        return LogitNormalTimestepSampler(
            seed=seed,
            eps=eps,
            mean=mean,
            std=std,
        )
    if normalized == "mode":
        return ModeShiftTimestepSampler(
            seed=seed,
            eps=eps,
            scale=mode_scale,
        )
    raise ValueError(
        "Timestep sampling mode must be one of: "
        + ", ".join(sorted(TIMESTEP_SAMPLING_MODES))
        + "."
    )


class NoiseGenerator:
    def __init__(self, seed: int):
        self._rng = random.Random(seed)
        self._torch_generator = None
        if torch is not None:
            self._torch_generator = torch.Generator()
            self._torch_generator.manual_seed(seed)

    def sample(self, batch_size_or_like):
        if is_torch_tensor(batch_size_or_like):
            like = batch_size_or_like
            # Generate on CPU with the seeded generator, then move to target
            # device. torch.Generator only works reliably on CPU.
            noise = torch.randn(
                like.shape,
                generator=self._torch_generator,
                dtype=like.dtype if like.dtype.is_floating_point else torch.float32,
            )
            return noise.to(device=like.device)
        batch_size = int(batch_size_or_like)
        return [self._rng.gauss(0.0, 1.0) for _ in range(batch_size)]

    def state_dict(self) -> dict:
        out = {"python_rng_state": self._rng.getstate()}
        if self._torch_generator is not None and torch is not None:
            out["torch_rng_state"] = self._torch_generator.get_state()
        return out

    def load_state_dict(self, state: dict) -> None:
        py_state = state.get("python_rng_state")
        if py_state is not None:
            self._rng.setstate(py_state)
        torch_state = state.get("torch_rng_state")
        if (
            torch_state is not None
            and self._torch_generator is not None
            and torch is not None
        ):
            self._torch_generator.set_state(torch_state)
