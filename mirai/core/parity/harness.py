"""Numerical parity harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import json

from mirai.core.tensors import is_torch_tensor, to_list

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class ParityResult:
    name: str
    max_abs: float
    max_rel: float
    mean_abs: float
    atol: float
    rtol: float
    passed: bool
    count: int


def _as_tensor(value: Any):
    if torch is None:
        raise RuntimeError("Torch is required for parity harness.")
    if is_torch_tensor(value):
        return value.detach().float().cpu()
    return torch.tensor(to_list(value), dtype=torch.float32)


def compare_outputs(
    *,
    name: str,
    candidate: Any,
    reference: Any,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> ParityResult:
    c = _as_tensor(candidate).reshape(-1)
    r = _as_tensor(reference).reshape(-1)
    if c.numel() != r.numel():
        raise ValueError(
            f"Shape mismatch for parity case '{name}': {tuple(c.shape)} vs {tuple(r.shape)}."
        )
    abs_diff = (c - r).abs()
    rel_diff = abs_diff / r.abs().clamp(min=1e-12)
    max_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
    max_rel = float(rel_diff.max().item()) if rel_diff.numel() else 0.0
    mean_abs = float(abs_diff.mean().item()) if abs_diff.numel() else 0.0
    passed = bool(torch.allclose(c, r, atol=atol, rtol=rtol))
    return ParityResult(
        name=name,
        max_abs=max_abs,
        max_rel=max_rel,
        mean_abs=mean_abs,
        atol=atol,
        rtol=rtol,
        passed=passed,
        count=int(c.numel()),
    )


def run_forward_parity_case(
    *,
    name: str,
    candidate_fn: Callable[[Any], Any],
    reference_fn: Callable[[Any], Any],
    inputs: Any,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> ParityResult:
    candidate = candidate_fn(inputs)
    reference = reference_fn(inputs)
    return compare_outputs(
        name=name,
        candidate=candidate,
        reference=reference,
        atol=atol,
        rtol=rtol,
    )


def write_parity_result(path: str | Path, result: ParityResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
