"""Few-step solver for the LingBot-Video DMD student.

The schedule and transition geometry are adapted from LingBot-Video's
Apache-2.0 ``scheduling_dmd_student.py`` at commit
``dd5c231e793406c6a8893e9e4307008a9c2adfe4``:
https://github.com/Robbyant/lingbot-video/blob/dd5c231e793406c6a8893e9e4307008a9c2adfe4/lingbot_video/scheduling_dmd_student.py

The distilled checkpoint is trained on one fixed recipe: eight steps, flow
shift 3, DDIM ancestral noise above warped sigma 0.5, and an Euler landing
below it. This native implementation keeps that recipe explicit and has no
Diffusers dependency.
"""

from __future__ import annotations

import torch

from mirai.core.inference.solvers.flow import SolverOutput


def _ddim_ancestral_std(
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    *,
    eta: float,
) -> torch.Tensor:
    """Return DDIM ancestral noise deviation in linear-flow coordinates."""
    signal_t, noise_t = 1.0 - sigma, sigma
    signal_next, noise_next = 1.0 - sigma_next, sigma_next
    normalizer_t = 1.0 / torch.sqrt(signal_t**2 + noise_t**2)
    normalizer_next = 1.0 / torch.sqrt(signal_next**2 + noise_next**2)
    signal_ratio_sq = (
        (signal_t * normalizer_t)
        / (signal_next * normalizer_next + 1.0e-8)
    ) ** 2
    residual = (1.0 - signal_ratio_sq).clamp(min=0.0)
    std = (
        float(eta)
        * (noise_next * normalizer_next)
        / (noise_t * normalizer_t + 1.0e-8)
        * residual.sqrt()
        / (normalizer_next + 1.0e-8)
    )
    return torch.minimum(std, noise_next)


def _randn_tensor(
    shape: torch.Size,
    *,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw from ``generator``, including the common CPU-generator/CUDA case."""
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    generator_device = torch.device(generator.device)
    if generator_device.type == "cpu" and device.type != "cpu":
        return torch.randn(
            shape,
            generator=generator,
            device=generator_device,
            dtype=dtype,
        ).to(device=device)
    if generator_device.type != device.type:
        raise ValueError(
            f"Generator device {generator_device} is incompatible with sample "
            f"device {device}."
        )
    return torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )


class DMDStudentSolver:
    """Fixed eight-step LingBot DMD student sampler.

    The high-noise transitions are stochastic and therefore consume the
    ``torch.Generator`` supplied to :meth:`step`. Solver arithmetic and output
    remain FP32 even when model predictions and latents use a lower precision.
    """

    REQUIRED_STEPS = 8
    REQUIRED_SHIFT = 3.0
    HIGH_NOISE_THRESHOLD = 0.5
    DDIM_ETA = 1.0

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 3.0):
        if int(num_train_timesteps) != 1000:
            raise ValueError("The LingBot DMD student requires 1000 train timesteps.")
        if float(shift) != self.REQUIRED_SHIFT:
            raise ValueError("The LingBot DMD student requires flow shift 3.0.")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.timesteps: torch.Tensor = torch.tensor([])
        self._sigmas: torch.Tensor = torch.tensor([])
        self._step_index = 0

    @property
    def step_index(self) -> int:
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str = "cpu",
        shift: float | None = None,
    ) -> None:
        """Install the exact float32 schedule used to distill the student."""
        steps = int(num_inference_steps)
        effective_shift = self.shift if shift is None else float(shift)
        if steps != self.REQUIRED_STEPS:
            raise ValueError("The LingBot DMD student requires exactly 8 steps.")
        if effective_shift != self.REQUIRED_SHIFT:
            raise ValueError("The LingBot DMD student requires flow shift 3.0.")

        # Match the reference's float64 linspace/warp and float32 storage. Build
        # on CPU first so the grid is independent of accelerator math modes.
        grid = torch.linspace(
            1.0,
            1.0 / self.num_train_timesteps,
            steps + 1,
            dtype=torch.float64,
            device="cpu",
        )[:-1]
        grid = effective_shift * grid / (1.0 + (effective_shift - 1.0) * grid)
        if torch.abs(grid[0] - 1.0) < 1.0e-6:
            grid[0] -= 1.0e-6
        sigmas = torch.cat((grid, grid.new_zeros((1,)))).to(
            device=device,
            dtype=torch.float32,
        )
        self._sigmas = sigmas
        self.timesteps = sigmas[:-1]
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> SolverOutput:
        """Apply one DDIM-ancestral or Euler transition in FP32."""
        _ = timestep
        if self._sigmas.numel() == 0:
            raise RuntimeError("Call set_timesteps before step().")
        if self._step_index >= len(self.timesteps):
            raise RuntimeError("The DMD student schedule is exhausted.")

        velocity = model_output.float()
        latent = sample.float()
        sigma = self._sigmas[self._step_index].to(device=latent.device)
        sigma_next = self._sigmas[self._step_index + 1].to(device=latent.device)

        if float(sigma) >= self.HIGH_NOISE_THRESHOLD:
            std = _ddim_ancestral_std(
                sigma,
                sigma_next,
                eta=self.DDIM_ETA,
            )
            clean = latent - sigma * velocity
            noise_direction = latent + (1.0 - sigma) * velocity
            mean = (
                clean * (1.0 - sigma_next)
                + noise_direction * torch.sqrt(sigma_next**2 - std**2)
            )
            noise = _randn_tensor(
                velocity.shape,
                generator=generator,
                device=velocity.device,
                dtype=velocity.dtype,
            )
            prev_sample = mean + std * noise
        else:
            prev_sample = latent + velocity * (sigma_next - sigma)

        self._step_index += 1
        return SolverOutput(prev_sample=prev_sample)
