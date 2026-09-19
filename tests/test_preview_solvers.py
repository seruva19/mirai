from __future__ import annotations

import math
import unittest

from mirai.core.registry import MissingRegistrationError
from mirai.core.training.preview.preview_solvers import (
    PreviewSolverRegistry,
    PreviewSolverSpec,
    register_preview_solver,
    resolve_preview_solver,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class PreviewSolverRegistryTests(unittest.TestCase):
    def test_euler_builtin_is_registered(self) -> None:
        self.assertIn("euler", PreviewSolverRegistry.names())

    def test_resolve_unknown_solver_fails_fast(self) -> None:
        spec = PreviewSolverSpec(
            num_inference_steps=6, flow_shift=3.0, device="cpu"
        )
        with self.assertRaises(MissingRegistrationError) as ctx:
            resolve_preview_solver("does_not_exist", spec)
        # The error names the available solvers so a misspelled config key is
        # actionable rather than a silent fallback.
        self.assertIn("euler", str(ctx.exception))

    def test_resolution_is_case_insensitive(self) -> None:
        if torch is None:
            self.skipTest("torch not installed")
        spec = PreviewSolverSpec(
            num_inference_steps=4, flow_shift=1.0, device="cpu"
        )
        solver = resolve_preview_solver("EULER", spec)
        self.assertEqual(int(len(solver.timesteps)), 4)

    def test_duplicate_registration_rejected(self) -> None:
        with self.assertRaises(Exception):
            register_preview_solver("euler")(lambda spec: object())


@unittest.skipIf(torch is None, "torch not installed")
class EulerByteIdentityTests(unittest.TestCase):
    """The Euler builtin must reproduce the canonical solver construction:

        solver = EulerFlowSolver(num_train_timesteps=1000, shift=flow_shift)
        solver.set_timesteps(steps, device=str(device), shift=flow_shift)

    We assert the registry-built solver has an identical observable state
    (class, shift, num_train_timesteps, step index, and the full timestep grid)
    for several (steps, shift) combinations. This is a CPU, VAE-free check on
    solver state, not on generated video.
    """

    def _reference_solver(self, steps: int, shift: float):
        from mirai.core.inference.solvers.flow import EulerFlowSolver

        solver = EulerFlowSolver(num_train_timesteps=1000, shift=shift)
        solver.set_timesteps(steps, device="cpu", shift=shift)
        return solver

    def test_euler_matches_hardwired_construction(self) -> None:
        from mirai.core.inference.solvers.flow import EulerFlowSolver

        for steps, shift in ((6, 3.0), (4, 1.0), (10, 5.0), (1, 2.0)):
            with self.subTest(steps=steps, shift=shift):
                built = resolve_preview_solver(
                    "euler",
                    PreviewSolverSpec(
                        num_inference_steps=steps,
                        flow_shift=shift,
                        device="cpu",
                        num_train_timesteps=1000,
                    ),
                )
                reference = self._reference_solver(steps, shift)
                self.assertIsInstance(built, EulerFlowSolver)
                self.assertEqual(built.num_train_timesteps, reference.num_train_timesteps)
                self.assertEqual(float(built.shift), float(reference.shift))
                self.assertEqual(int(built._step_index), int(reference._step_index))
                self.assertEqual(
                    tuple(built.timesteps.shape), tuple(reference.timesteps.shape)
                )
                self.assertTrue(
                    torch.equal(built.timesteps, reference.timesteps),
                    f"timestep grid drift for steps={steps} shift={shift}",
                )

    def test_building_euler_does_not_touch_global_rng(self) -> None:
        # Solver construction/timestep setup is deterministic and must not
        # consume the process-global RNG stream that training relies on. This is
        # the CPU-checkable core of the preview RNG-isolation invariant.
        before = torch.random.get_rng_state()
        resolve_preview_solver(
            "euler",
            PreviewSolverSpec(num_inference_steps=8, flow_shift=3.0, device="cpu"),
        )
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))


@unittest.skipIf(torch is None, "torch not installed")
class UniPCSolverTests(unittest.TestCase):
    """Behavioural invariants for the ``unipc`` builtin (Flow-UniPC, bh2, order 2).

    These check registration/resolution, the shared-grid invariant against
    euler, the order-1 bootstrap equalling an euler step, RNG isolation, per-run
    history reset, and a synthetic ODE where the multistep solver is strictly
    more accurate than euler at the same low step count.
    """

    @staticmethod
    def _spec(steps: int, shift: float) -> PreviewSolverSpec:
        return PreviewSolverSpec(
            num_inference_steps=steps, flow_shift=shift, device="cpu"
        )

    def test_unipc_is_registered(self) -> None:
        self.assertIn("unipc", PreviewSolverRegistry.names())

    def test_unipc_resolves_case_insensitively(self) -> None:
        from mirai.core.inference.solvers.flow import FlowUniPCMultistepSolver

        solver = resolve_preview_solver("UniPC", self._spec(4, 1.0))
        self.assertIsInstance(solver, FlowUniPCMultistepSolver)
        self.assertEqual(int(len(solver.timesteps)), 4)

    def test_unipc_shares_euler_sigma_grid(self) -> None:
        # The whole point of building on shifted_timesteps: unipc must inherit
        # the exact euler sigma grid so switching solvers cannot silently move
        # the schedule. torch.equal is a bit-for-bit comparison.
        for steps, shift in ((4, 1.0), (6, 3.0), (10, 5.0), (20, 3.0), (2, 2.0)):
            with self.subTest(steps=steps, shift=shift):
                euler = resolve_preview_solver("euler", self._spec(steps, shift))
                unipc = resolve_preview_solver("unipc", self._spec(steps, shift))
                self.assertEqual(
                    tuple(unipc.timesteps.shape), tuple(euler.timesteps.shape)
                )
                self.assertTrue(
                    torch.equal(unipc.timesteps, euler.timesteps),
                    f"grid drift for steps={steps} shift={shift}",
                )

    def test_first_step_matches_euler(self) -> None:
        # The bootstrap step runs at order 1, which for this rectified-flow
        # schedule is algebraically a single euler step. It is computed through
        # the log-SNR / expm1 path, so tiny float error is expected but the two
        # must agree to a tight tolerance.
        torch.manual_seed(0)
        for steps, shift in ((8, 3.0), (4, 1.0), (12, 5.0)):
            with self.subTest(steps=steps, shift=shift):
                euler = resolve_preview_solver("euler", self._spec(steps, shift))
                unipc = resolve_preview_solver("unipc", self._spec(steps, shift))
                sample = torch.randn(2, 3, 4)
                velocity = torch.randn(2, 3, 4)
                euler_out = euler.step(
                    velocity, euler.timesteps[0], sample.clone()
                ).prev_sample
                unipc_out = unipc.step(
                    velocity, unipc.timesteps[0], sample.clone()
                ).prev_sample
                self.assertTrue(
                    torch.allclose(euler_out, unipc_out, atol=1e-4, rtol=1e-4),
                    "order-1 unipc bootstrap diverged from a single euler step",
                )

    def test_building_unipc_does_not_touch_global_rng(self) -> None:
        # Mirrors the euler RNG-isolation invariant: constructing the solver and
        # laying out its (deterministic) timestep grid must not consume the
        # process-global RNG stream that training draws from.
        before = torch.random.get_rng_state()
        resolve_preview_solver("unipc", self._spec(8, 3.0))
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))

    def test_history_resets_between_runs(self) -> None:
        # Multistep state (model_outputs history, corrector sample, bootstrap
        # counters) must reset on every set_timesteps call. If it leaked, a
        # reused solver would produce a different trajectory than a fresh one.
        def trajectory(solver: object) -> list[torch.Tensor]:
            solver.set_timesteps(4, device="cpu", shift=1.0)
            x = torch.arange(1.0, 5.0)
            out: list[torch.Tensor] = []
            for sig in solver.timesteps:
                x = solver.step(-x, sig, x).prev_sample
                out.append(x.clone())
            return out

        reused = resolve_preview_solver("unipc", self._spec(4, 1.0))
        first = trajectory(reused)
        second = trajectory(reused)  # same object, re-run
        fresh = trajectory(resolve_preview_solver("unipc", self._spec(4, 1.0)))
        for a, b, c in zip(first, second, fresh):
            self.assertTrue(torch.equal(a, b), "reused solver leaked state")
            self.assertTrue(torch.equal(a, c), "reused run differs from fresh run")

    def test_more_accurate_than_euler_on_linear_velocity_ode(self) -> None:
        # Integrate the linear-velocity ODE dx/dsigma = v = -x over the shared
        # sigma grid (sigma: 1 -> 0). Its exact trajectory is x(sigma) =
        # x(1) * exp(1 - sigma), so the clean-sample target is x(0) = x(1) * e.
        # Euler is first-order on this genuinely curved (exponential) path;
        # UniPC (order 2) must be strictly more accurate at the same N=4 steps.
        shift, steps = 1.0, 4
        noise = torch.linspace(-2.0, 3.0, 64)
        exact = noise * math.e

        def integrate(name: str) -> torch.Tensor:
            solver = resolve_preview_solver(name, self._spec(steps, shift))
            x = noise.clone()
            for sig in solver.timesteps:
                x = solver.step(-x, sig, x).prev_sample
            return x

        euler_err = float((integrate("euler") - exact).norm())
        unipc_err = float((integrate("unipc") - exact).norm())
        self.assertLess(
            unipc_err,
            euler_err,
            f"unipc ({unipc_err:.4f}) not more accurate than euler "
            f"({euler_err:.4f}) at N={steps}",
        )
        # Comfortable margin so the invariant is not a coin-flip near equality.
        self.assertLess(unipc_err, 0.9 * euler_err)


@unittest.skipIf(torch is None, "torch not installed")
class DPMSolverPlusPlusTests(unittest.TestCase):
    @staticmethod
    def _spec(steps: int, shift: float) -> PreviewSolverSpec:
        return PreviewSolverSpec(
            num_inference_steps=steps, flow_shift=shift, device="cpu"
        )

    def test_registered_and_shares_reference_grid(self) -> None:
        from mirai.core.inference.solvers.dpmpp import FlowDPMSolverMultistep

        dpmpp = resolve_preview_solver("dpmpp_2m", self._spec(8, 3.0))
        euler = resolve_preview_solver("euler", self._spec(8, 3.0))
        self.assertIsInstance(dpmpp, FlowDPMSolverMultistep)
        self.assertTrue(torch.equal(dpmpp.timesteps, euler.timesteps))

    def test_first_step_matches_euler_formula(self) -> None:
        sample = torch.linspace(-2.0, 2.0, 17)
        velocity = torch.linspace(1.0, -1.0, 17)
        dpmpp = resolve_preview_solver("dpmpp_2m", self._spec(6, 2.0))
        euler = resolve_preview_solver("euler", self._spec(6, 2.0))
        actual = dpmpp.step(velocity, dpmpp.timesteps[0], sample).prev_sample
        expected = euler.step(velocity, euler.timesteps[0], sample).prev_sample
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

    def test_second_step_matches_midpoint_2m_equation(self) -> None:
        solver = resolve_preview_solver("dpmpp_2m", self._spec(5, 1.0))
        sample0 = torch.tensor([0.5, -1.0, 2.0])
        velocity0 = torch.tensor([-0.2, 0.3, -0.4])
        sample1 = solver.step(
            velocity0, solver.timesteps[0], sample0
        ).prev_sample
        previous_x0 = sample0.float() - solver._sigmas[0] * velocity0.float()
        velocity1 = torch.tensor([0.1, -0.4, 0.25])
        current_x0 = sample1.float() - solver._sigmas[1] * velocity1.float()
        lambda_0 = solver._lambda(solver._sigmas[0])
        lambda_1 = solver._lambda(solver._sigmas[1])
        lambda_2 = solver._lambda(solver._sigmas[2])
        h_prev = lambda_1 - lambda_0
        h = lambda_2 - lambda_1
        derivative = (current_x0 - previous_x0) / (h_prev / h)
        expected = (
            (solver._sigmas[2] / solver._sigmas[1]) * sample1.float()
            - (1.0 - solver._sigmas[2])
            * torch.expm1(-h)
            * (current_x0 + 0.5 * derivative)
        )
        actual = solver.step(
            velocity1, solver.timesteps[1], sample1
        ).prev_sample
        self.assertTrue(torch.equal(actual, expected))

    def test_state_resets_and_does_not_touch_rng(self) -> None:
        before = torch.random.get_rng_state()
        solver = resolve_preview_solver("dpmpp_2m", self._spec(4, 1.0))
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        solver.step(torch.ones(2), solver.timesteps[0], torch.zeros(2))
        self.assertIsNotNone(solver._previous_x0)
        solver.set_timesteps(4, device="cpu", shift=1.0)
        self.assertIsNone(solver._previous_x0)
        self.assertEqual(solver._step_index, 0)

    def test_more_accurate_than_euler_on_linear_velocity_ode(self) -> None:
        noise = torch.linspace(-2.0, 3.0, 64)
        exact = noise * math.e

        def integrate(name: str) -> torch.Tensor:
            solver = resolve_preview_solver(name, self._spec(4, 1.0))
            value = noise.clone()
            for sigma in solver.timesteps:
                value = solver.step(-value, sigma, value).prev_sample
            return value

        euler_error = float((integrate("euler") - exact).norm())
        dpmpp_error = float((integrate("dpmpp_2m") - exact).norm())
        self.assertLess(dpmpp_error, euler_error)


@unittest.skipIf(torch is None, "torch not installed")
class DMDStudentSolverTests(unittest.TestCase):
    @staticmethod
    def _solver():
        return resolve_preview_solver(
            "dmd_student",
            PreviewSolverSpec(
                num_inference_steps=8,
                flow_shift=3.0,
                device="cpu",
            ),
        )

    def test_registered_with_exact_reference_schedule(self) -> None:
        from mirai.core.inference.solvers.dmd import DMDStudentSolver

        solver = self._solver()
        self.assertIsInstance(solver, DMDStudentSolver)
        expected = torch.tensor(
            [
                0.99999899,
                0.95459503,
                0.90011996,
                0.83355546,
                0.75037479,
                0.64346898,
                0.50099897,
                0.30167764,
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(solver.timesteps, expected))
        self.assertEqual(float(solver._sigmas[-1]), 0.0)

    def test_rejects_off_distribution_step_count_and_shift(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 8 steps"):
            resolve_preview_solver(
                "dmd_student",
                PreviewSolverSpec(
                    num_inference_steps=7,
                    flow_shift=3.0,
                    device="cpu",
                ),
            )
        with self.assertRaisesRegex(ValueError, "flow shift 3.0"):
            resolve_preview_solver(
                "dmd_student",
                PreviewSolverSpec(
                    num_inference_steps=8,
                    flow_shift=2.0,
                    device="cpu",
                ),
            )

    def test_transition_matches_reference_equations_and_branch_gate(self) -> None:
        solver = self._solver()
        generator = torch.Generator(device="cpu").manual_seed(19)
        reference_generator = torch.Generator(device="cpu").manual_seed(19)
        sample = torch.tensor([0.25, -1.0, 2.5], dtype=torch.float16)
        velocity = torch.tensor([-0.5, 0.75, 1.25], dtype=torch.float16)

        sigma, sigma_next = solver._sigmas[:2]
        signal, noise = 1.0 - sigma, sigma
        signal_next, noise_next = 1.0 - sigma_next, sigma_next
        normalizer = 1.0 / torch.sqrt(signal**2 + noise**2)
        normalizer_next = 1.0 / torch.sqrt(signal_next**2 + noise_next**2)
        ratio_sq = (
            (signal * normalizer)
            / (signal_next * normalizer_next + 1.0e-8)
        ) ** 2
        std = torch.minimum(
            (noise_next * normalizer_next)
            / (noise * normalizer + 1.0e-8)
            * (1.0 - ratio_sq).clamp(min=0.0).sqrt()
            / (normalizer_next + 1.0e-8),
            noise_next,
        )
        latent_f, velocity_f = sample.float(), velocity.float()
        clean = latent_f - sigma * velocity_f
        noise_direction = latent_f + (1.0 - sigma) * velocity_f
        expected = (
            clean * (1.0 - sigma_next)
            + noise_direction * torch.sqrt(sigma_next.square() - std.square())
            + std
            * torch.randn(
                velocity.shape,
                generator=reference_generator,
                dtype=torch.float32,
            )
        )
        actual = solver.step(
            velocity,
            solver.timesteps[0],
            sample,
            generator=generator,
        ).prev_sample
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

        # Advance through the remaining stochastic region. Sigma 0.30167764 is
        # below the 0.5 gate, so the final landing must not consume RNG.
        current = actual
        for index in range(1, 7):
            current = solver.step(
                velocity,
                solver.timesteps[index],
                current,
                generator=generator,
            ).prev_sample
        before_final = generator.get_state()
        sigma = solver._sigmas[7]
        expected_final = current.float() + velocity.float() * (0.0 - sigma)
        actual_final = solver.step(
            velocity,
            solver.timesteps[7],
            current,
            generator=generator,
        ).prev_sample
        self.assertTrue(torch.equal(actual_final, expected_final))
        self.assertTrue(torch.equal(before_final, generator.get_state()))

    def test_supplied_generator_is_reproducible_and_global_rng_isolated(self) -> None:
        def trajectory(seed: int) -> torch.Tensor:
            solver = self._solver()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            value = torch.linspace(-1.0, 1.0, 9)
            velocity = torch.linspace(0.5, -0.5, 9)
            for timestep in solver.timesteps:
                value = solver.step(
                    velocity,
                    timestep,
                    value,
                    generator=generator,
                ).prev_sample
            return value

        before = torch.random.get_rng_state()
        first = trajectory(123)
        after = torch.random.get_rng_state()
        second = trajectory(123)
        different = trajectory(124)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))


if __name__ == "__main__":
    unittest.main()
