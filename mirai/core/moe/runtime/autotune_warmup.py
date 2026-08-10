"""Pre-run autotuning for Mirai's persistent grouped-GEMM kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class GroupedGemmWarmupProblem:
    """One persistent grouped-GEMM autotune key and representative row count."""

    num_experts: int
    in_features: int
    out_features: int
    routed_rows: int

    def __post_init__(self) -> None:
        if min(self.num_experts, self.in_features, self.out_features, self.routed_rows) <= 0:
            raise ValueError("Grouped-GEMM warm-up dimensions must be positive.")


def grouped_gemm_warmup_problems(
    expert_tensor_specs: Iterable[object], *, routed_rows: int
) -> tuple[GroupedGemmWarmupProblem, ...]:
    """Derive unique forward keys from provider-owned expert tensor specs."""

    rows = int(routed_rows)
    if rows <= 0:
        raise ValueError("routed_rows must be positive.")
    problems: set[GroupedGemmWarmupProblem] = set()
    for spec in expert_tensor_specs:
        if not bool(getattr(spec, "routed", False)) or bool(getattr(spec, "router", False)):
            continue
        layout = tuple(getattr(spec, "layout", ()))
        shape = tuple(int(value) for value in getattr(spec, "shape", ()))
        if not {"expert", "out", "in"}.issubset(layout) or len(layout) != len(shape):
            continue
        dimensions = dict(zip(layout, shape))
        problems.add(
            GroupedGemmWarmupProblem(
                num_experts=dimensions["expert"],
                in_features=dimensions["in"],
                out_features=dimensions["out"],
                routed_rows=rows,
            )
        )
    return tuple(sorted(problems))


def warmup_persistent_grouped_gemm(
    problems: Iterable[GroupedGemmWarmupProblem],
    *,
    dtype=None,
    device=None,
    include_input_gradient: bool = True,
    empty_cache: bool = True,
) -> tuple[GroupedGemmWarmupProblem, ...]:
    """Populate Triton's cache before training activations consume device memory."""

    selected = tuple(dict.fromkeys(problems))
    if not selected:
        return ()

    import torch

    from mirai.vendors.qwen3_moe_fused import grouped_gemm_forward

    target = torch.device(device or "cuda")
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Persistent grouped-GEMM warm-up requires CUDA.")
    target_dtype = dtype or torch.bfloat16
    for problem in selected:
        counts = torch.full(
            (problem.num_experts,),
            problem.routed_rows // problem.num_experts,
            device=target,
            dtype=torch.int32,
        )
        counts[: problem.routed_rows % problem.num_experts] += 1
        offsets = counts.cumsum(0, dtype=torch.int32)
        inputs = torch.empty(
            (problem.routed_rows, problem.in_features), device=target, dtype=target_dtype
        )
        weights = torch.empty(
            (problem.num_experts, problem.out_features, problem.in_features),
            device=target,
            dtype=target_dtype,
        )
        grouped_gemm_forward(inputs, weights, offsets)
        if include_input_gradient:
            output_gradient = torch.empty(
                (problem.routed_rows, problem.out_features),
                device=target,
                dtype=target_dtype,
            )
            grouped_gemm_forward(
                output_gradient,
                weights,
                offsets,
                target_dtype,
                transpose_w=True,
            )
            del output_gradient
        del counts, offsets, inputs, weights
        torch.cuda.synchronize(target)
        if empty_cache:
            torch.cuda.empty_cache()
    return selected


__all__ = [
    "GroupedGemmWarmupProblem",
    "grouped_gemm_warmup_problems",
    "warmup_persistent_grouped_gemm",
]
