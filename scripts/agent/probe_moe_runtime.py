"""Bounded GPU behavioral probe for optional MoE memory and execution extensions."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from mirai.core.models.compressed_weights.execution.mixed_precision import (
    MixedPrecisionGroupedExperts,
)
from mirai.core.models.compressed_weights.quantization.structured_sparsity import (
    prune_to_2_4,
    sparse_2_4_linear,
)
from mirai.core.training.residency.activation_compression import (
    LowRankActivationCompression,
)
from mirai.core.training.residency.memory import (
    current_resource_telemetry,
    track_resource_peaks,
)
from mirai.core.training.runtime.gpu_lease import (
    acquire_gpu_lease,
    resolve_lease_lock_path,
)


class _DenseExperts(nn.Module):
    def __init__(self, *, experts: int, width: int, device: torch.device) -> None:
        super().__init__()
        self.num_experts = int(experts)
        shape = (experts, width, width)
        self.w1 = nn.Parameter(torch.randn(shape, device=device), requires_grad=False)
        self.w2 = nn.Parameter(torch.randn(shape, device=device), requires_grad=False)
        self.w3 = nn.Parameter(torch.randn(shape, device=device), requires_grad=False)


def warmup_routed_gemm(device: torch.device, cache_path: Path) -> dict[str, object]:
    from mirai.core.moe.runtime.routed_gemm_autotune import (
        RoutedGemmAutotuner,
        RoutedGemmBenchmarkResult,
        build_routed_gemm_environment_fingerprint,
        build_routed_gemm_shape_key,
    )
    from mirai.core.moe.runtime.routed_gemm_triton import _launch_routed_grouped_mm

    rows, groups, k_size, n_size = 64, 8, 128, 128
    activation = torch.randn(rows, k_size, device=device, dtype=torch.bfloat16)
    weight = torch.randn(groups, k_size, n_size, device=device, dtype=torch.bfloat16)
    counts = torch.full((groups,), rows // groups, device=device, dtype=torch.int32)
    boundaries = counts.cumsum(0)
    environment = build_routed_gemm_environment_fingerprint(
        activation, kernel_abi_fingerprint="routed-grouped-mm-v2"
    )
    key = build_routed_gemm_shape_key(
        activation, weight, counts, implementation="indexed_sm80", role="forward",
        fusion="grouped", top_k=1,
    )
    tuner = RoutedGemmAutotuner(
        mode="warmup_only", environment_fingerprint=environment, cache_path=cache_path
    )

    def benchmark(config):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        empty = torch.empty(0, device=device, dtype=torch.int64)
        output = _launch_routed_grouped_mm(
            activation, weight, boundaries, empty, empty, rows, config,
            gather=False, scatter=False,
        )
        end.record(); end.synchronize()
        del output
        return RoutedGemmBenchmarkResult(config, start.elapsed_time(end) * 1000.0, 1)

    def verify(winner):
        empty = torch.empty(0, device=device, dtype=torch.int64)
        output = _launch_routed_grouped_mm(
            activation, weight, boundaries, empty, empty, rows, winner.config,
            gather=False, scatter=False,
        )
        reference = torch.cat([
            activation[index * (rows // groups):(index + 1) * (rows // groups)] @ weight[index]
            for index in range(groups)
        ])
        torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)

    config = tuner.resolve_config(
        key, benchmark=benchmark, verify=verify, warmup_write=True,
        samples_parity_tolerance=0.02,
    )
    return {"routed_gemm_cache": str(cache_path), "routed_gemm_config": config.to_dict()}


def _path_size_bytes(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    return sum(
        int(item.stat().st_size)
        for item in path.rglob("*")
        if item.is_file()
    )


def _latency_samples_ms(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        output = operation()
        del output
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del output
    return samples


def _summarize_latency(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered) + 0.5) - 1))
    return {
        "samples": len(samples),
        "mean_ms": float(statistics.fmean(samples)),
        "median_ms": float(statistics.median(samples)),
        "p95_ms": float(ordered[p95_index]),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def _measure_operation(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = int(torch.cuda.memory_allocated(device))
    host_before = current_resource_telemetry()
    with track_resource_peaks(sample_interval_seconds=0.02) as host_tracker:
        samples = _latency_samples_ms(
            operation, warmup=warmup, iterations=iterations
        )
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    host_after = current_resource_telemetry()
    host_peaks = host_tracker.snapshot()
    return {
        "latency": _summarize_latency(samples),
        "peak_cuda_allocated_bytes": peak_allocated,
        "incremental_peak_cuda_allocated_bytes": max(
            0, peak_allocated - allocated_before
        ),
        "process_rss_before_bytes": round(
            float(host_before["process_rss_mb"]) * 1024 * 1024
        ),
        "process_rss_after_bytes": round(
            float(host_after["process_rss_mb"]) * 1024 * 1024
        ),
        "process_peak_rss_bytes": round(
            float(host_peaks["tracked_process_peak_rss_mb"]) * 1024 * 1024
        ),
    }


def benchmark_routed_gemm(
    device: torch.device,
    *,
    tokens: int,
    top_k: int,
    groups: int,
    k_size: int,
    n_size: int,
    warmup: int,
    iterations: int,
    cache_path: Path | None,
    distribution: str = "quadratic_descending",
) -> dict[str, object]:
    """Measure a deterministic routed projection and its reference."""

    from mirai.core.moe.runtime.routed_gemm import (
        RoutedFusionSpec,
        RoutedGroupLayout,
        routed_gemm_reference,
    )
    from mirai.core.moe.runtime.routed_gemm_autotune import (
        build_routed_gemm_environment_fingerprint,
        routing_distribution_statistics,
    )
    from mirai.core.moe.runtime.routed_gemm_triton import triton_routed_grouped_mm

    if min(tokens, top_k, groups, k_size, n_size, iterations) <= 0 or warmup < 0:
        raise ValueError("routed GEMM benchmark dimensions and iterations must be positive")
    generator = torch.Generator(device="cpu").manual_seed(31)
    if distribution == "balanced":
        assigned_experts = torch.arange(tokens * top_k, dtype=torch.int64) % groups
    elif distribution == "quadratic_descending":
        probabilities = torch.arange(groups, 0, -1, dtype=torch.float64).square()
        assigned_experts = torch.multinomial(
            probabilities,
            tokens * top_k,
            replacement=True,
            generator=generator,
        )
    else:
        raise ValueError("routed GEMM benchmark distribution must be balanced or quadratic_descending")
    assignment_rows = torch.argsort(assigned_experts, stable=True).to(
        device=device, dtype=torch.int64
    )
    counts_cpu = torch.bincount(assigned_experts, minlength=groups)
    boundaries = counts_cpu.cumsum(0).to(device=device, dtype=torch.int32)
    gather_rows = torch.div(assignment_rows, top_k, rounding_mode="floor")
    activation = torch.randn(
        tokens, k_size, device=device, dtype=torch.bfloat16
    )
    weight = torch.randn(
        groups, k_size, n_size, device=device, dtype=torch.bfloat16
    )
    layout = RoutedGroupLayout(
        boundaries=boundaries,
        assignment_rows=assignment_rows,
        token_count=tokens,
        top_k=top_k,
        group_count=groups,
    )

    def reference() -> torch.Tensor:
        return routed_gemm_reference(
            activation, weight, layout, RoutedFusionSpec(gather_tokens=True)
        )

    def candidate() -> torch.Tensor:
        return triton_routed_grouped_mm(
            activation,
            weight,
            boundaries,
            gather_rows=gather_rows,
            output_rows=tokens * top_k,
            role="forward",
            top_k=top_k,
        )

    expected = reference()
    actual = candidate()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    absolute_error = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-6)
    parity = {
        "passed": True,
        "rtol": 0.02,
        "atol": 0.02,
        "max_absolute_error": float(absolute_error.max().item()),
        "max_relative_error": float((absolute_error / denominator).max().item()),
    }
    del actual, expected, absolute_error, denominator

    environment = build_routed_gemm_environment_fingerprint(
        activation, kernel_abi_fingerprint="routed-grouped-mm-v2"
    )
    cache_bytes_before = _path_size_bytes(cache_path)
    reference_measurement = _measure_operation(
        reference,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    candidate_measurement = _measure_operation(
        candidate,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    return {
        "schema": "mirai.routed_gemm_benchmark.v1",
        "environment": environment.to_dict(),
        "shape": {
            "tokens": tokens,
            "top_k": top_k,
            "routed_rows": tokens * top_k,
            "groups": groups,
            "k_size": k_size,
            "n_size": n_size,
            "dtype": str(activation.dtype).removeprefix("torch."),
        },
        "routing": {
            "seed": 31,
            "distribution": distribution,
            "counts": [int(value) for value in counts_cpu.tolist()],
            **routing_distribution_statistics(counts_cpu.tolist()),
        },
        "protocol": {
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "synchronized_per_sample": True,
            "operation": "gathered_grouped_forward",
        },
        "reference": reference_measurement,
        "candidate": candidate_measurement,
        "parity": parity,
        "persistent_cache": {
            "path_configured": cache_path is not None,
            "bytes_before": cache_bytes_before,
            "bytes_after": _path_size_bytes(cache_path),
        },
    }


def benchmark_expert_h2d(
    device: torch.device,
    *,
    expert_bytes: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Measure the actual pinned-host transfer used by expert streaming."""

    if expert_bytes <= 0 or warmup < 0 or iterations <= 0:
        raise ValueError("expert H2D benchmark sizes and iterations must be positive")
    elements = (int(expert_bytes) + 1) // 2
    source = torch.empty(elements, dtype=torch.bfloat16, pin_memory=True)
    destination = torch.empty(elements, dtype=torch.bfloat16, device=device)
    moved_bytes = int(source.numel() * source.element_size())
    samples = _latency_samples_ms(
        lambda: destination.copy_(source, non_blocking=True),
        warmup=warmup,
        iterations=iterations,
    )
    latency = _summarize_latency(samples)
    gib_per_second = (moved_bytes / float(1024**3)) / (
        float(latency["median_ms"]) / 1000.0
    )
    return {
        "bytes_per_transfer": moved_bytes,
        "latency": latency,
        "gib_per_second": gib_per_second,
        "pinned_source": bool(source.is_pinned()),
    }


def emit_expert_transfer_profile(
    *,
    output: Path,
    device: torch.device,
    expert_format: str,
    expert_bytes: int,
    working_set_experts: int,
    cache_budget_gib: float,
    h2d: dict[str, object],
    routed_gemm: dict[str, object],
) -> dict[str, object]:
    """Derive and atomically persist a profile from measured probe results."""

    from mirai.core.moe.runtime.expert_transfer_profile import (
        build_expert_transfer_profile,
        save_expert_transfer_profile,
    )

    shape = routed_gemm["shape"]
    candidate = routed_gemm["candidate"]
    weight_bytes = int(shape["groups"]) * int(shape["k_size"]) * int(shape["n_size"]) * 2
    compute_seconds = float(candidate["latency"]["median_ms"]) / 1000.0
    compute_gib_per_second = (weight_bytes / float(1024**3)) / compute_seconds
    capability = torch.cuda.get_device_capability(device)
    protocol = {
        "schema": "mirai.expert_transfer_benchmark.v1",
        "h2d": h2d,
        "routed_gemm": routed_gemm,
        "device_index": device.index,
    }
    profile = build_expert_transfer_profile(
        gpu_name=torch.cuda.get_device_name(device),
        compute_capability=f"{capability[0]}.{capability[1]}",
        expert_format=expert_format,
        expert_bytes=expert_bytes,
        working_set_experts=working_set_experts,
        cache_budget_gib=cache_budget_gib,
        h2d_gib_per_second=float(h2d["gib_per_second"]),
        routed_compute_gib_per_second=compute_gib_per_second,
        benchmark_protocol=protocol,
    )
    save_expert_transfer_profile(output, profile)
    return {"path": str(output), "profile": profile.to_dict(), "protocol": protocol}


def run_probe(device: torch.device) -> dict[str, float | int | bool]:
    torch.manual_seed(19)

    weight = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
    sparse_state = prune_to_2_4(weight)
    inputs = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
    dense_output = sparse_2_4_linear(inputs, sparse_state, backend="reference")
    sparse_output = sparse_2_4_linear(inputs, sparse_state, backend="cuda")
    torch.testing.assert_close(sparse_output, dense_output, rtol=0.02, atol=0.02)

    mixed = MixedPrecisionGroupedExperts(
        _DenseExperts(experts=2, width=64, device=device),
        formats=("int8", "mxfp4"),
    )
    tokens = torch.randn(16, 64, device=device, requires_grad=True)
    scores = torch.softmax(torch.randn(16, 2, device=device), dim=-1).requires_grad_(True)
    indices = torch.tensor([[0, 1]] * 16, device=device)
    mixed_output = mixed.run_direct_routed(tokens, scores, indices)
    mixed_output.float().square().mean().backward()
    if tokens.grad is None or scores.grad is None:
        raise AssertionError("Mixed-precision routed execution lost trainable gradients.")

    compression = LowRankActivationCompression(rank=8, min_bytes=0, seed=23)
    low_rank = (
        torch.randn(1024, 8, device=device)
        @ torch.randn(8, 512, device=device)
    ).requires_grad_(True)
    projection = (
        torch.randn(512, 8, device=device)
        @ torch.randn(8, 128, device=device)
    ).requires_grad_(True)
    reference_low_rank = low_rank.detach().clone().requires_grad_(True)
    reference_projection = projection.detach().clone().requires_grad_(True)
    reference_loss = (reference_low_rank @ reference_projection).square().mean()
    reference_loss.backward()
    with compression.context():
        loss = (low_rank @ projection).square().mean()
    loss.backward()
    if compression.compressed_tensors <= 0:
        raise AssertionError("Activation compression did not claim an eligible tensor.")
    if compression.stored_bytes >= compression.original_bytes:
        raise AssertionError("Activation compression did not reduce saved storage.")
    torch.testing.assert_close(loss, reference_loss, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        low_rank.grad, reference_low_rank.grad, rtol=3e-2, atol=3e-3
    )
    torch.testing.assert_close(
        projection.grad, reference_projection.grad, rtol=3e-2, atol=3e-3
    )

    return {
        "structured_2_4_parity": True,
        "mixed_precision_gradients": True,
        "activation_compressed_tensors": int(compression.compressed_tensors),
        "activation_original_bytes": int(compression.original_bytes),
        "activation_stored_bytes": int(compression.stored_bytes),
        "activation_storage_ratio": float(
            compression.stored_bytes / compression.original_bytes
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--routed-gemm-warmup-cache", type=Path)
    parser.add_argument("--benchmark-routed-gemm", action="store_true")
    parser.add_argument("--benchmark-tokens", type=int, default=1024)
    parser.add_argument("--benchmark-top-k", type=int, default=2)
    parser.add_argument("--benchmark-groups", type=int, default=8)
    parser.add_argument("--benchmark-k", type=int, default=1024)
    parser.add_argument("--benchmark-n", type=int, default=2048)
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--expert-transfer-profile-out", type=Path)
    parser.add_argument("--expert-format", default="int8")
    parser.add_argument("--expert-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--expert-working-set", type=int, default=32)
    parser.add_argument("--expert-cache-budget-gib", type=float, default=2.0)
    parser.add_argument(
        "--benchmark-routing-distribution",
        choices=("balanced", "quadratic_descending"),
        default="quadratic_descending",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This probe requires an explicitly leased CUDA device.")
    with acquire_gpu_lease(
        lock_path=resolve_lease_lock_path(Path.cwd()),
        timeout_seconds=0.0,
    ):
        report = run_probe(device)
        if args.routed_gemm_warmup_cache is not None:
            report.update(warmup_routed_gemm(device, args.routed_gemm_warmup_cache))
        routed_gemm_report = None
        if args.benchmark_routed_gemm or args.expert_transfer_profile_out is not None:
            routed_gemm_report = benchmark_routed_gemm(
                device,
                tokens=args.benchmark_tokens,
                top_k=args.benchmark_top_k,
                groups=args.benchmark_groups,
                k_size=args.benchmark_k,
                n_size=args.benchmark_n,
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                cache_path=args.routed_gemm_warmup_cache,
                distribution=args.benchmark_routing_distribution,
            )
            report["routed_gemm_benchmark"] = routed_gemm_report
        if args.expert_transfer_profile_out is not None:
            h2d = benchmark_expert_h2d(
                device, expert_bytes=args.expert_bytes,
                warmup=args.benchmark_warmup, iterations=args.benchmark_iterations,
            )
            report["expert_transfer"] = emit_expert_transfer_profile(
                output=args.expert_transfer_profile_out,
                device=device,
                expert_format=args.expert_format,
                expert_bytes=args.expert_bytes,
                working_set_experts=args.expert_working_set,
                cache_budget_gib=args.expert_cache_budget_gib,
                h2d=h2d,
                routed_gemm=routed_gemm_report,
            )
        torch.cuda.synchronize(device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
