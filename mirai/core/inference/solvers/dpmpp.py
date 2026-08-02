"""DPM-Solver++(2M) for Mirai rectified-flow inference."""

from __future__ import annotations

import torch

from mirai.core.inference.solvers.flow import SolverOutput, shifted_timesteps


class FlowDPMSolverMultistep:
    """DPM-Solver++(2M) midpoint update for flow velocity models.

    Velocity is converted to data prediction with ``x0 = x - sigma * v``.
    First and final updates use the first-order formula, avoiding multistep
    extrapolation through the sigma-zero endpoint.
    """

    _LAMBDA_EPS = 1.0e-6

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.timesteps: torch.Tensor = torch.tensor([])
        self._sigmas: torch.Tensor = torch.tensor([])
        self._step_index = 0
        self._previous_x0: torch.Tensor | None = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str = "cpu",
        shift: float | None = None,
    ) -> None:
        effective_shift = self.shift if shift is None else float(shift)
        self.timesteps = shifted_timesteps(
            num_inference_steps,
            shift=effective_shift,
            device=device,
        )
        self._sigmas = torch.cat(
            (self.timesteps, self.timesteps.new_zeros((1,))), dim=0
        )
        self._step_index = 0
        self._previous_x0 = None

    def _lambda(self, sigma: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(1.0 - sigma, min=self._LAMBDA_EPS)
        return torch.log(alpha) - torch.log(
            torch.clamp(sigma, min=self._LAMBDA_EPS)
        )

    def set_begin_index(self, begin_index: int) -> None:
        """Drop the high-noise prefix and reset multistep state."""
        begin = int(begin_index)
        if not 0 <= begin <= len(self.timesteps):
            raise ValueError("begin_index is outside the inference schedule.")
        self.timesteps = self.timesteps[begin:]
        self._sigmas = torch.cat(
            (self.timesteps, self.timesteps.new_zeros((1,))), dim=0
        )
        self._step_index = 0
        self._previous_x0 = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
    ) -> SolverOutput:
        _ = timestep
        idx = self._step_index
        sigma_s = self._sigmas[idx]
        sigma_t = self._sigmas[idx + 1]
        alpha_t = 1.0 - sigma_t
        sample_f = sample.to(torch.float32)
        x0 = sample_f - sigma_s * model_output.to(torch.float32)
        h = self._lambda(sigma_t) - self._lambda(sigma_s)
        phi_1 = torch.expm1(-h)

        first_or_final = self._previous_x0 is None or idx + 1 == len(self.timesteps)
        data_prediction = x0
        if not first_or_final:
            sigma_prev = self._sigmas[idx - 1]
            h_prev = self._lambda(sigma_s) - self._lambda(sigma_prev)
            r = h_prev / h
            derivative = (x0 - self._previous_x0) / r
            data_prediction = x0 + 0.5 * derivative

        prev_sample = (
            (sigma_t / sigma_s) * sample_f - alpha_t * phi_1 * data_prediction
        )
        self._previous_x0 = x0
        self._step_index += 1
        return SolverOutput(prev_sample=prev_sample.to(sample.dtype))
