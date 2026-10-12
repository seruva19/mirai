"""Native Agnes sampling policy matching the released inference pipeline.

Behavior is derived from the Apache-2.0 Agnes release at
``Agnes-AI/Agnes-Video-2.0-MoE@fc3971e378b53c766baa06898bfe84d7a56653ea``:
``inference/pipeline/scheduler_unipc.py`` and ``video_generate.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from mirai.core.inference.solvers.flow import FlowUniPCMultistepSolver, SolverOutput


def agnes_timeline_seconds(frame_count: int, fps: float) -> float:
    """Return the physical timeline spanned by an Agnes 4n+1 frame tensor."""
    frames = int(frame_count)
    rate = float(fps)
    if frames < 1 or rate <= 0:
        raise ValueError("Agnes frame_count and fps must be positive.")
    return max(1.0 / rate, (frames - 1) / rate)


def agnes_export_frame_count(frame_count: int) -> int:
    """Drop the internal 4n+1 endpoint frame for physical media export."""
    frames = int(frame_count)
    if frames < 1:
        raise ValueError("Agnes frame_count must be positive.")
    return max(1, frames - 1)


def agnes_sample_count(frame_count: int, fps: float, sample_rate: int) -> int:
    return max(1, round(agnes_timeline_seconds(frame_count, fps) * int(sample_rate)))


def agnes_audio_token_count(
    frame_count: int, fps: float, sample_rate: int, hop: int, *, conditioned: bool
) -> int:
    samples = agnes_sample_count(frame_count, fps, sample_rate)
    if conditioned:
        return max(1, (samples + int(hop) - 1) // int(hop))
    return max(1, round(agnes_timeline_seconds(frame_count, fps) * int(sample_rate) / int(hop)))


def _agnes_schedule(
    num_steps: int, *, shift: float, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(num_steps) < 1:
        raise ValueError("Agnes inference requires at least one denoising step.")
    # The released scheduler derives sigma_max=0.999 from its 1000-step
    # training grid before applying the inference-time flow shift.
    sigma_max = torch.tensor(0.999, dtype=torch.float32).item()
    sigmas_np = np.linspace(sigma_max, 0.0, int(num_steps) + 1)[:-1]
    if float(shift) != 1.0:
        sigmas_np = (
            float(shift)
            * sigmas_np
            / (1.0 + (float(shift) - 1.0) * sigmas_np)
        )
    source_timesteps = torch.from_numpy((sigmas_np * 1000.0).astype(np.int64)).to(
        device=device
    )
    sigmas = torch.from_numpy(sigmas_np.astype(np.float32)).to(device=device)
    return sigmas, source_timesteps


class AgnesFlowUniPCMultistepSolver(FlowUniPCMultistepSolver):
    """Agnes' released Flow-UniPC recurrence with its checkpoint sigma grid."""

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str = "cpu",
        shift: float | None = None,
    ) -> None:
        effective_shift = self.shift if shift is None else float(shift)
        self.timesteps, self.source_timesteps = _agnes_schedule(
            num_inference_steps, shift=effective_shift, device=device
        )
        # The release deliberately keeps scalar scheduler arithmetic on CPU,
        # even when the samples and coefficient linear systems live on CUDA.
        sigmas = self.timesteps.cpu()
        self._sigmas = torch.cat([sigmas, sigmas.new_zeros((1,))])
        self._step_index = 0
        self._reset_multistep_state()

    def set_begin_index(self, begin_index: int) -> None:
        begin = int(begin_index)
        if not 0 <= begin <= len(self.timesteps):
            raise ValueError("begin_index is outside the inference schedule.")
        self.source_timesteps = self.source_timesteps[begin:]
        self.timesteps = self.timesteps[begin:]
        # Do not rebuild these from device-resident public timesteps: V2V
        # must retain the same CPU scalar arithmetic as a complete trajectory.
        self._sigmas = self._sigmas[begin:]
        self._step_index = 0
        self._reset_multistep_state()

    @staticmethod
    def _lambda(sigma: torch.Tensor) -> torch.Tensor:
        # In particular the final zero sigma must give +inf, not a clamped
        # approximation: expm1(-inf) then supplies the exact endpoint limit.
        return torch.log(1.0 - sigma) - torch.log(sigma)

    def _released_update(
        self,
        *,
        sample: torch.Tensor,
        order: int,
        corrector: bool,
        model_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        idx = self._step_index
        previous = idx - 1 if corrector else idx
        sigma_t = self._sigmas[idx if corrector else idx + 1]
        sigma_s0 = self._sigmas[previous]
        alpha_t = 1.0 - sigma_t
        lambda_s0 = self._lambda(sigma_s0)
        h = self._lambda(sigma_t) - lambda_s0
        m0 = self.model_outputs[-1]
        rks = []
        differences = []
        for i in range(1, order):
            rk = (self._lambda(self._sigmas[previous - i]) - lambda_s0) / h
            rks.append(rk)
            differences.append((self.model_outputs[-(i + 1)] - m0) / rk)
        rks.append(1.0)
        # These explicit copies match the release's CPU scalar -> device
        # boundary; computing the transcendental functions on CUDA is not
        # numerically interchangeable over a complete denoising trajectory.
        rk_tensor = torch.tensor(rks, device=sample.device)
        hh = -h
        phi = torch.expm1(hh)
        phi_k = phi / hh - 1
        rows = []
        rhs = []
        factorial = 1
        for i in range(1, order + 1):
            rows.append(torch.pow(rk_tensor, i - 1))
            rhs.append(phi_k * factorial / phi)
            factorial *= i + 1
            phi_k = phi_k / hh - 1 / factorial
        matrix = torch.stack(rows)
        coefficients = torch.tensor(rhs, device=sample.device)
        if corrector and order > 1:
            weights = torch.linalg.solve(matrix, coefficients).to(sample.dtype)
        elif corrector or order == 2:
            weights = torch.tensor([0.5], dtype=sample.dtype, device=sample.device)
        elif differences:
            weights = torch.linalg.solve(matrix[:-1, :-1], coefficients[:-1]).to(sample.dtype)
        else:
            weights = None
        result = sigma_t / sigma_s0 * sample - alpha_t * phi * m0
        residual = 0
        if differences:
            difference_tensor = torch.stack(differences, dim=1)
            residual = torch.einsum(
                "k,bkc...->bc...",
                weights[:-1] if corrector else weights,
                difference_tensor,
            )
        if corrector:
            residual = residual + weights[-1] * (model_t - m0)
        return (result - alpha_t * phi * residual).to(sample.dtype)

    def _uni_p_bh_update(self, *, m0, sample, order):
        return self._released_update(sample=sample, order=order, corrector=False)

    def _uni_c_bh_update(self, model_t, last_sample, this_sample, order):
        return self._released_update(
            sample=last_sample, order=order, corrector=True, model_t=model_t
        )

    def step(self, model_output, timestep, sample, *, generator=None) -> SolverOutput:
        del timestep, generator
        idx = self._step_index
        # Multiplication precedes promotion to the sample dtype in the release.
        # BF16 velocities therefore retain their BF16 rounding boundary here.
        converted = sample - self._sigmas[idx] * model_output
        if idx > 0 and self.last_sample is not None:
            sample = self._uni_c_bh_update(
                converted, self.last_sample, sample, self.this_order
            )
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = converted
        order = min(self.solver_order, len(self.timesteps) - idx) if self.lower_order_final else self.solver_order
        self.this_order = min(order, self.lower_order_nums + 1)
        self.last_sample = sample
        result = self._uni_p_bh_update(m0=converted, sample=sample, order=self.this_order)
        self.lower_order_nums = min(self.lower_order_nums + 1, self.solver_order)
        self._step_index += 1
        return SolverOutput(prev_sample=result)


class AgnesDDIMFlowSolver(AgnesFlowUniPCMultistepSolver):
    """Released Agnes cfg=1 update, which injects fresh noise each step."""

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> SolverOutput:
        del timestep
        idx = self._step_index
        if idx >= len(self.timesteps):
            raise IndexError("Agnes sampler has no remaining denoising steps.")
        current_sigma = self._sigmas[idx].to(device=sample.device, dtype=sample.dtype)
        next_sigma = self._sigmas[idx + 1].to(device=sample.device, dtype=sample.dtype)
        variance_noise = torch.randn(
            sample.shape,
            generator=generator,
            device=sample.device,
            dtype=sample.dtype,
        )
        clean = sample - current_sigma * model_output
        prev_sample = next_sigma * variance_noise + (1.0 - next_sigma) * clean
        self._step_index += 1
        return SolverOutput(prev_sample=prev_sample)


def build_agnes_solver(
    *, num_inference_steps: int, shift: float, device: str, stochastic: bool
):
    solver_type = AgnesDDIMFlowSolver if stochastic else AgnesFlowUniPCMultistepSolver
    solver = solver_type(shift=float(shift), solver_order=2)
    solver.set_timesteps(
        int(num_inference_steps), device=device, shift=float(shift)
    )
    return solver


__all__ = [
    "AgnesDDIMFlowSolver",
    "AgnesFlowUniPCMultistepSolver",
    "agnes_export_frame_count",
    "agnes_audio_token_count",
    "agnes_sample_count",
    "agnes_timeline_seconds",
    "build_agnes_solver",
]
