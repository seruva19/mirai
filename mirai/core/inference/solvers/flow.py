"""Mirai flow-matching ODE solvers for inference, with no diffusers dependency.

Provides Euler and UniPC solvers for rectified-flow schedules: sigma goes from
1.0 (pure noise) to 0.0 (clean).

The implementation is an original, diffusers-free realization of the
rectified-flow (flow-matching)
sampling convention used by the LingBot/Wan video ecosystem. The filename and
the shifted-sigma schedule (``shift * s / (1 + (shift - 1) * s)``) follow the
Wan-family ``fm_solvers`` sampler and the diffusers ``FlowMatchEulerDiscrete``
scheduler. ``FlowUniPCMultistepSolver`` reimplements the UniPC predictor-
corrector update (Zhao et al., 2023) in the data-prediction / ``bh2`` form used
by Wan's ``FlowUniPCMultistepScheduler``, adapted to the rectified-flow
convention (``alpha_t = 1 - sigma_t``, ``x0 = sample - sigma_t * v``). No
upstream code is copied and ``diffusers`` is never imported.
- conceptual source: https://github.com/Wan-Video/Wan2.1 (``wan/utils/fm_solvers.py``)
  and diffusers ``FlowMatchEulerDiscreteScheduler``
- implementation: native Mirai code; no upstream code copied
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SolverOutput:
    prev_sample: torch.Tensor


def shifted_timesteps(
    num_steps: int,
    *,
    shift: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate shifted sigma schedule for flow matching.

    Returns tensor of shape [num_steps] with values from ~1.0 → ~0.0.
    Shift > 1 places more schedule mass at high-noise timesteps.
    """
    # Linear spacing from 1 → 0 (exclusive of 0)
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:num_steps]
    # Apply shift
    if shift != 1.0:
        sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    return sigma


class EulerFlowSolver:
    """Euler ODE solver for flow matching models.

    The flow matching objective predicts velocity v = dx/dt.
    Euler integration: x_{t-dt} = x_t - v * dt
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.timesteps: torch.Tensor = torch.tensor([])
        self._step_index = 0

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str = "cpu",
        shift: float | None = None,
    ) -> None:
        effective_shift = shift if shift is not None else self.shift
        self.timesteps = shifted_timesteps(
            num_inference_steps,
            shift=effective_shift,
            device=device,
        )
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
    ) -> SolverOutput:
        """One Euler step: x_{t-dt} = x_t - v * dt."""
        idx = self._step_index
        sigmas = self.timesteps
        if idx + 1 < len(sigmas):
            dt = float(sigmas[idx].item()) - float(sigmas[idx + 1].item())
        else:
            dt = float(sigmas[idx].item())
        prev_sample = sample - model_output * dt
        self._step_index += 1
        return SolverOutput(prev_sample=prev_sample)

    def set_begin_index(self, begin_index: int) -> None:
        """Drop the high-noise prefix for source-conditioned inference."""
        begin = int(begin_index)
        if not 0 <= begin <= len(self.timesteps):
            raise ValueError("begin_index is outside the inference schedule.")
        self.timesteps = self.timesteps[begin:]
        self._step_index = 0


class FlowUniPCMultistepSolver:
    """UniPC multistep solver (``bh2`` variant) for rectified-flow models.

    Implements the UniPC predictor-corrector update (Zhao et al., 2023,
    "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of
    Diffusion Models") specialised to the flow-matching / rectified-flow
    convention used by Wan's ``FlowUniPCMultistepScheduler``: data prediction
    (``predict_x0``), the ``bh2`` update function, and ``lower_order_final``
    bootstrapping of the multistep history.

    Object contract mirrors :class:`EulerFlowSolver` so the native denoise loop
    needs no changes:

    - ``.timesteps`` is the shared shifted-sigma grid produced by
      :func:`shifted_timesteps` — bit-identical to the Euler grid for the same
      ``(steps, shift)``.
    - ``.step(model_output, timestep, sample) -> SolverOutput`` advances
      ``self._step_index`` and keeps a bounded ``self.model_outputs`` history of
      length ``solver_order`` with a ``lower_order_nums`` bootstrap.

    Flow-space conversion (``alpha_t = 1 - sigma_t``): the model predicts the
    velocity ``v`` and the data prediction is ``x0 = sample - sigma_t * v``.

    Multistep state (history, corrector sample, bootstrap counters) is reset on
    every :meth:`set_timesteps` call, so a fresh run never leaks state from a
    previous one. The first step degenerates to order 1, which is algebraically
    identical to a single Euler step for this rectified-flow schedule.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        solver_order: int = 2,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.solver_order = int(solver_order)
        # Data-prediction (predict_x0) + bh2 is the Wan Flow-UniPC configuration.
        self.predict_x0 = True
        self.solver_type = "bh2"
        self.lower_order_final = True
        self.timesteps: torch.Tensor = torch.tensor([])
        # Internal extended sigma grid: the shared timesteps plus a trailing 0.0
        # so the final step lands exactly on the x0 prediction.
        self._sigmas: torch.Tensor = torch.tensor([])
        self._step_index = 0
        self._reset_multistep_state()

    def _reset_multistep_state(self) -> None:
        """Clear all multistep history so a new run starts from scratch."""
        self.model_outputs: list[torch.Tensor | None] = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample: torch.Tensor | None = None
        # Order used by the *next* corrector call; set by the previous predictor.
        self.this_order = 0

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str = "cpu",
        shift: float | None = None,
    ) -> None:
        effective_shift = shift if shift is not None else self.shift
        # Bit-shared with EulerFlowSolver: identical shifted_timesteps() call.
        self.timesteps = shifted_timesteps(
            num_inference_steps,
            shift=effective_shift,
            device=device,
        )
        zero = torch.zeros(
            1, dtype=self.timesteps.dtype, device=self.timesteps.device
        )
        self._sigmas = torch.cat([self.timesteps, zero])
        self._step_index = 0
        self._reset_multistep_state()

    def set_begin_index(self, begin_index: int) -> None:
        """Drop the high-noise prefix and reset all multistep history."""
        begin = int(begin_index)
        if not 0 <= begin <= len(self.timesteps):
            raise ValueError("begin_index is outside the inference schedule.")
        self.timesteps = self.timesteps[begin:]
        zero = self.timesteps.new_zeros((1,))
        self._sigmas = torch.cat([self.timesteps, zero])
        self._step_index = 0
        self._reset_multistep_state()

    # -- UniPC math -------------------------------------------------------

    # Keeps the log-SNR finite at the grid endpoints. The shared sigma grid
    # includes sigma == 1.0 (alpha == 0) as its first point and sigma == 0.0 as
    # the trailing landing point; both make one of the logs diverge. Clamping
    # only inside the log-SNR keeps every predictor/corrector coefficient finite
    # while the raw sigmas still drive the ``sigma_t / sigma_s0`` ratio and the
    # shared grid, so the endpoints behave as their well-defined limits.
    _LAMBDA_EPS = 1.0e-6

    def _lambda(self, sigma: torch.Tensor) -> torch.Tensor:
        """Half-log-SNR ``lambda = log(alpha) - log(sigma)`` for flow sigmas."""
        eps = self._LAMBDA_EPS
        alpha = torch.clamp(1.0 - sigma, min=eps)
        sigma = torch.clamp(sigma, min=eps)
        return torch.log(alpha) - torch.log(sigma)

    def _bh_coefficients(
        self, hh: torch.Tensor, rks: list[torch.Tensor], order: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``(R, b, B_h)`` for the bh2 predictor/corrector linear system."""
        h_phi_1 = torch.expm1(hh)  # phi_1(hh) * hh = e^{hh} - 1
        B_h = torch.expm1(hh)  # bh2
        h_phi_k = h_phi_1 / hh - 1.0
        rows: list[torch.Tensor] = []
        b_vals: list[torch.Tensor] = []
        factorial_i = 1.0
        rks_t = torch.stack(rks)
        for i in range(1, order + 1):
            rows.append(torch.pow(rks_t, i - 1))
            b_vals.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / factorial_i
        R = torch.stack(rows)
        b = torch.stack(b_vals)
        return R, b, B_h

    def _uni_p_bh_update(
        self, m0: torch.Tensor, sample: torch.Tensor, order: int
    ) -> torch.Tensor:
        """UniPC predictor: advance ``sample`` from step_index to step_index+1."""
        idx = self._step_index
        sig = self._sigmas
        sigma_t, sigma_s0 = sig[idx + 1], sig[idx]
        alpha_t = 1.0 - sigma_t
        lambda_t = self._lambda(sigma_t)
        lambda_s0 = self._lambda(sigma_s0)
        h = lambda_t - lambda_s0

        rks: list[torch.Tensor] = []
        d1s: list[torch.Tensor] = []
        for i in range(1, order):
            mi = self.model_outputs[-(i + 1)]
            lambda_si = self._lambda(sig[idx - i])
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)
        rks.append(torch.ones_like(h))

        hh = -h  # predict_x0
        h_phi_1 = torch.expm1(hh)
        B_h = torch.expm1(hh)  # bh2

        # x_t_ is the order-1 (exponential-Euler) prediction; higher orders add a
        # correction built from the finite differences of past x0 predictions.
        x_t = (sigma_t / sigma_s0) * sample - alpha_t * h_phi_1 * m0
        if d1s:
            if order == 2:
                rhos_p = [torch.full_like(h, 0.5)]
            else:
                R, b, _ = self._bh_coefficients(hh, rks, order)
                rhos_p = list(torch.linalg.solve(R[:-1, :-1], b[:-1]))
            pred_res = sum(rho * d1 for rho, d1 in zip(rhos_p, d1s))
            x_t = x_t - alpha_t * B_h * pred_res
        return x_t

    def _uni_c_bh_update(
        self,
        model_t: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """UniPC corrector: refine ``this_sample`` using the fresh x0 pred."""
        idx = self._step_index
        sig = self._sigmas
        m0 = self.model_outputs[-1]
        sigma_t, sigma_s0 = sig[idx], sig[idx - 1]
        alpha_t = 1.0 - sigma_t
        lambda_t = self._lambda(sigma_t)
        lambda_s0 = self._lambda(sigma_s0)
        h = lambda_t - lambda_s0

        rks: list[torch.Tensor] = []
        d1s: list[torch.Tensor] = []
        for i in range(1, order):
            mi = self.model_outputs[-(i + 1)]
            lambda_si = self._lambda(sig[idx - 1 - i])
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)
        rks.append(torch.ones_like(h))

        hh = -h  # predict_x0
        h_phi_1 = torch.expm1(hh)
        B_h = torch.expm1(hh)  # bh2

        if order == 1:
            rhos_c = [torch.full_like(h, 0.5)]
        else:
            R, b, _ = self._bh_coefficients(hh, rks, order)
            rhos_c = list(torch.linalg.solve(R, b))

        x_t = (sigma_t / sigma_s0) * last_sample - alpha_t * h_phi_1 * m0
        corr_res = sum(rho * d1 for rho, d1 in zip(rhos_c[:-1], d1s))
        d1_t = model_t - m0
        x_t = x_t - alpha_t * B_h * (corr_res + rhos_c[-1] * d1_t)
        return x_t

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
    ) -> SolverOutput:
        """One UniPC predictor-corrector step; advances ``self._step_index``."""
        idx = self._step_index
        orig_dtype = sample.dtype
        # UniPC's finite differences are ill-conditioned in low precision; run
        # the multistep algebra in float32 and cast the result back, matching the
        # data-prediction history precision.
        sample_f = sample.to(torch.float32)
        model_output_f = model_output.to(torch.float32)

        sigma_s0 = self._sigmas[idx]
        # Flow-space data prediction: x0 = sample - sigma_t * v.
        m_convert = sample_f - sigma_s0 * model_output_f

        use_corrector = idx > 0 and self.last_sample is not None
        if use_corrector:
            sample_f = self._uni_c_bh_update(
                model_t=m_convert,
                last_sample=self.last_sample,
                this_sample=sample_f,
                order=self.this_order,
            )

        # Roll the bounded history and record the fresh x0 prediction.
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = m_convert

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - idx)
        else:
            this_order = self.solver_order
        this_order = min(this_order, self.lower_order_nums + 1)
        self.this_order = this_order

        self.last_sample = sample_f
        prev_sample = self._uni_p_bh_update(
            m0=m_convert, sample=sample_f, order=this_order
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self._step_index += 1
        return SolverOutput(prev_sample=prev_sample.to(orig_dtype))
