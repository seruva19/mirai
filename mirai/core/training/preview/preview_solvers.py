"""Registry for preview flow-matching solvers.

Builtins register factories here, config selects ``logging.sample_solver``, and
resolution fails on unknown names. Euler is the numerical reference. Flow-UniPC
uses the same sigma grid with a second-order multistep update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mirai.core.registry import Registry


@dataclass(frozen=True)
class PreviewSolverSpec:
    """Immutable inputs required to construct a preview solver."""

    num_inference_steps: int
    flow_shift: float
    device: str
    num_train_timesteps: int = 1000


# A factory returns a solver with its inference timesteps configured.
PreviewSolverFactory = Callable[[PreviewSolverSpec], Any]
PreviewSolverRegistry: Registry[PreviewSolverFactory] = Registry("preview solver")


def register_preview_solver(name: str) -> Callable[[PreviewSolverFactory], PreviewSolverFactory]:
    """Register a preview solver factory under ``name`` (builtin or plugin)."""
    return PreviewSolverRegistry.decorator(name)


def resolve_preview_solver(name: str, spec: PreviewSolverSpec) -> Any:
    """Build the solver registered under ``name``; fail fast on unknown names.

    ``Registry.get`` raises ``MissingRegistrationError`` (a ``ValueError``) with
    the available names when ``name`` is not registered, so a misspelled
    ``logging.sample_solver`` never silently falls back to a default.
    """
    factory = PreviewSolverRegistry.get(name)
    return factory(spec)


@register_preview_solver("euler")
def _build_euler_solver(spec: PreviewSolverSpec) -> Any:
    """Construct the Euler flow-matching reference solver."""
    from mirai.core.inference.solvers.flow import EulerFlowSolver

    solver = EulerFlowSolver(
        num_train_timesteps=int(spec.num_train_timesteps),
        shift=float(spec.flow_shift),
    )
    solver.set_timesteps(
        int(spec.num_inference_steps),
        device=str(spec.device),
        shift=float(spec.flow_shift),
    )
    return solver


@register_preview_solver("unipc")
def _build_unipc_solver(spec: PreviewSolverSpec) -> Any:
    """Builtin: Wan-style Flow-UniPC multistep solver (bh2, solver_order 2).

    Mirrors the euler factory's wiring so the solver shares the exact shifted
    sigma grid (``shifted_timesteps``); only the per-step update differs. The
    multistep history is reset inside ``set_timesteps``, so each preview run is
    self-contained.
    """
    from mirai.core.inference.solvers.flow import FlowUniPCMultistepSolver

    solver = FlowUniPCMultistepSolver(
        num_train_timesteps=int(spec.num_train_timesteps),
        shift=float(spec.flow_shift),
        solver_order=2,
    )
    solver.set_timesteps(
        int(spec.num_inference_steps),
        device=str(spec.device),
        shift=float(spec.flow_shift),
    )
    return solver


@register_preview_solver("dpmpp_2m")
def _build_dpmpp_2m_solver(spec: PreviewSolverSpec) -> Any:
    """Builtin DPM-Solver++(2M) midpoint solver for rectified flow."""
    from mirai.core.inference.solvers.dpmpp import FlowDPMSolverMultistep

    solver = FlowDPMSolverMultistep(
        num_train_timesteps=int(spec.num_train_timesteps),
        shift=float(spec.flow_shift),
    )
    solver.set_timesteps(
        int(spec.num_inference_steps),
        device=str(spec.device),
        shift=float(spec.flow_shift),
    )
    return solver
