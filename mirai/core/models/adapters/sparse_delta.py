"""Fixed-support sparse high-rank weight deltas for linear projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SparseDeltaSpec:
    rows: int
    columns: int
    density: float
    selection: str
    seed: int


class SparseDeltaLinear(nn.Module):
    """Frozen linear plus a directly indexed, externally portable sparse delta.

    Only ``values`` is trainable.  The immutable flattened indices are part of
    the state dict, so resume and export cannot silently reconstruct a different
    support.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        density: float,
        selection: str = "magnitude",
        seed: int = 0,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("SparseDeltaLinear requires torch.")
        super().__init__()
        weight = getattr(base, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError("SparseDeltaLinear requires a dense two-dimensional weight.")
        value = float(density)
        if not 0.0 < value <= 1.0:
            raise ValueError("Sparse delta density must be in (0, 1].")
        mode = str(selection).strip().lower()
        if mode not in {"magnitude", "random"}:
            raise ValueError("Sparse delta selection must be magnitude or random.")
        self.base = base
        for parameter in base.parameters(recurse=True):
            parameter.requires_grad_(False)
        count = max(1, int(round(weight.numel() * value)))
        flat = weight.detach().reshape(-1)
        if mode == "magnitude":
            indices = flat.abs().topk(count, sorted=True).indices
        else:
            generator = torch.Generator(device=flat.device)
            generator.manual_seed(int(seed))
            indices = torch.randperm(flat.numel(), generator=generator, device=flat.device)[:count]
            indices = indices.sort().values
        self.register_buffer("indices", indices.to(dtype=torch.int64), persistent=True)
        self.values = nn.Parameter(torch.zeros(count, dtype=weight.dtype, device=weight.device))
        self.rows = int(weight.shape[0])
        self.columns = int(weight.shape[1])
        self.density = value
        self.selection = mode
        self.seed = int(seed)

    def dense_delta(self, *, device: Any, dtype: Any) -> Any:
        flat = torch.zeros(
            self.rows * self.columns,
            device=device,
            dtype=dtype,
        )
        return flat.scatter(
            0,
            self.indices.to(device=device),
            self.values.to(device=device, dtype=dtype),
        ).reshape(self.rows, self.columns)

    def forward(self, inputs: Any) -> Any:
        base_weight = self.base.weight.to(device=inputs.device, dtype=inputs.dtype)
        bias = getattr(self.base, "bias", None)
        if bias is not None:
            bias = bias.to(device=inputs.device, dtype=inputs.dtype)
        weight = base_weight + self.dense_delta(device=inputs.device, dtype=inputs.dtype)
        return F.linear(inputs, weight, bias)

    def spec(self) -> SparseDeltaSpec:
        return SparseDeltaSpec(
            rows=self.rows,
            columns=self.columns,
            density=self.density,
            selection=self.selection,
            seed=self.seed,
        )


def apply_sparse_delta_to_linear_modules(
    root: nn.Module,
    *,
    target_modules: list[str],
    density: float,
    selection: str,
    seed: int,
) -> tuple[str, ...]:
    """Replace matching dense linear modules and return stable qualified names."""

    matched: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, SparseDeltaLinear):
                continue
            selected = target_modules == ["*"] or any(
                name.endswith(target) for target in target_modules
            )
            if isinstance(child, nn.Linear) and selected:
                setattr(
                    parent,
                    child_name,
                    SparseDeltaLinear(
                        child,
                        density=density,
                        selection=selection,
                        seed=seed + len(matched),
                    ),
                )
                matched.append(name)
                continue
            visit(child, name)

    visit(root)
    return tuple(matched)
