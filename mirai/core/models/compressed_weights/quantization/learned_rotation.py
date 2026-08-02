"""Data-free learned orthogonal rotations for grouped expert INT8 weights.

The implementation adapts the learned-rotation principle from SpinQuant to
Mirai's expert-weight storage boundary.  It does not reproduce SpinQuant's
LLM-wide activation/KV calibration: one groupwise orthogonal transform is
optimized for the shared w1/w3 input and one for w2, using only frozen expert
weights and the exact rowwise INT8 quantizer used by Mirai.

Reference:
    Liu et al., "SpinQuant: LLM quantization with learned rotations",
    arXiv:2405.16406, https://github.com/facebookresearch/SpinQuant
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .quant import _hadamard


LEARNED_EXPERT_ROTATION_NAME = "learned"
LEARNED_EXPERT_ROTATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LearnedRotationResult:
    rotation: Any
    group_size: int
    initial_relative_error: float
    optimized_relative_error: float
    optimization_steps: int
    learning_rate: float


def _validate_sources(weights: Sequence[Any], group_size: int) -> tuple[int, int]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Learned expert rotations require torch.")
    if not weights:
        raise ValueError("Learned rotation requires at least one weight tensor.")
    resolved_group = int(group_size)
    if resolved_group <= 0:
        raise ValueError("Learned rotation requires a positive group size.")
    rows = 0
    for weight in weights:
        if not torch.is_tensor(weight) or weight.ndim < 2:
            raise ValueError("Learned rotation weights must be tensors with ndim >= 2.")
        if int(weight.shape[-1]) % resolved_group != 0:
            raise ValueError(
                f"Learned rotation group size {resolved_group} does not divide "
                f"in_features={int(weight.shape[-1])}."
            )
        if not bool(torch.isfinite(weight.detach()).all().item()):
            raise ValueError("Learned rotation source weights must be finite.")
        rows += int(weight.numel() // int(weight.shape[-1]))
    if rows <= 0:
        raise ValueError("Learned rotation source weights are empty.")
    return resolved_group, rows


def expert_weight_fingerprint(weights: Mapping[str, Any]) -> str:
    """Fingerprint dense source weights before they are replaced by packed data."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Expert weight fingerprinting requires torch.")
    if not weights:
        raise ValueError("Expert weight fingerprint requires at least one tensor.")
    digest = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expert weight {name!r} is not a tensor.")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(int(dim) for dim in value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def validate_learned_rotation_selection(
    manifest: Mapping[str, Any],
    selection: str | None,
) -> None:
    """Require the explicit config gate to match rotation-bearing artifacts."""

    modules = manifest.get("modules")
    if not isinstance(modules, Mapping):
        raise ValueError("Packed-state manifest has no modules object.")
    present = any(
        isinstance(spec, Mapping) and bool(spec.get("rotations"))
        for spec in modules.values()
    )
    selected = str(selection or "off").strip().lower()
    if selected not in {"off", LEARNED_EXPERT_ROTATION_NAME}:
        raise ValueError(
            "Expert quantization rotation selection must be 'off' or 'learned'."
        )
    if present and selected != LEARNED_EXPERT_ROTATION_NAME:
        raise ValueError(
            "Packed artifact contains learned expert rotations; select "
            "model.params.expert_quantization_rotation='learned'."
        )
    if selected == LEARNED_EXPERT_ROTATION_NAME and not present:
        raise ValueError(
            "model.params.expert_quantization_rotation='learned' requires a "
            "rotation-bearing packed artifact."
        )


def _cayley_delta(generator: Any) -> Any:
    skew = generator - generator.transpose(-2, -1)
    identity = torch.eye(
        int(skew.shape[0]),
        dtype=skew.dtype,
        device=skew.device,
    )
    return torch.linalg.solve(identity - skew, identity + skew)


def _hard_fake_quantize(rotated: Any) -> Any:
    scale = (rotated.detach().abs().amax(dim=-1, keepdim=True) / 127.0).clamp(
        min=1e-30
    )
    return (rotated / scale).round().clamp(-127, 127) * scale


def _ste_fake_quantize(rotated: Any) -> Any:
    hard = _hard_fake_quantize(rotated)
    return rotated + (hard - rotated).detach()


def _group_rows(weight: Any, group_size: int) -> Any:
    in_features = int(weight.shape[-1])
    return (
        weight.detach()
        .reshape(-1, in_features)
        .reshape(-1, int(group_size))
    )


def _row_chunks(
    weights: Sequence[Any],
    *,
    group_size: int,
    row_chunk_size: int,
) -> list[Any]:
    chunks: list[Any] = []
    for weight in weights:
        rows = _group_rows(weight, group_size)
        for start in range(0, int(rows.shape[0]), int(row_chunk_size)):
            chunks.append(rows[start : start + int(row_chunk_size)])
    if not chunks:
        raise ValueError("Learned rotation produced no source row chunks.")
    return chunks


def _relative_error(
    chunks: Sequence[Any],
    rotation: Any,
    *,
    device: Any,
) -> float:
    squared_error = 0.0
    squared_reference = 0.0
    with torch.no_grad():
        for chunk in chunks:
            source = chunk.to(device=device, dtype=torch.float32)
            rotated = source @ rotation
            quantized = _hard_fake_quantize(rotated)
            squared_error += float(
                (quantized - rotated).square().sum().detach().cpu().item()
            )
            squared_reference += float(
                rotated.square().sum().detach().cpu().item()
            )
    return math.sqrt(squared_error / max(squared_reference, 1e-30))


def learn_groupwise_expert_rotation(
    weights: Sequence[Any],
    *,
    group_size: int,
    optimization_steps: int = 200,
    learning_rate: float = 0.01,
    row_chunk_size: int = 4096,
    checkpoint_interval: int = 25,
    device: Any = "cpu",
    max_workspace_gib: float = 2.0,
) -> LearnedRotationResult:
    """Optimize one shared groupwise rotation without materializing all rows.

    A skew-symmetric generator is mapped through a Cayley transform and applied
    after Mirai's existing Hadamard baseline.  Every candidate is orthogonal by
    construction.  Hard-quantization checkpoints retain the best matrix, so the
    stored result cannot regress from the fixed-Hadamard initialization on the
    exact source tensors.
    """

    resolved_group, _rows = _validate_sources(weights, group_size)
    steps = int(optimization_steps)
    chunk_size = int(row_chunk_size)
    checkpoint = int(checkpoint_interval)
    lr = float(learning_rate)
    if steps < 0:
        raise ValueError("optimization_steps must be non-negative.")
    if lr <= 0.0 or not math.isfinite(lr):
        raise ValueError("learning_rate must be finite and positive.")
    if chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive.")
    if checkpoint <= 0:
        raise ValueError("checkpoint_interval must be positive.")
    if float(max_workspace_gib) <= 0.0:
        raise ValueError("max_workspace_gib must be positive.")
    estimated_workspace = (
        chunk_size * resolved_group * 4 * 8
        + resolved_group * resolved_group * 4 * 10
    )
    workspace_limit = int(float(max_workspace_gib) * (1024**3))
    if estimated_workspace > workspace_limit:
        raise ValueError(
            "Learned rotation estimated workspace exceeds max_workspace_gib: "
            f"{estimated_workspace} > {workspace_limit} bytes."
        )

    target_device = torch.device(device)
    chunks = _row_chunks(
        weights,
        group_size=resolved_group,
        row_chunk_size=chunk_size,
    )
    hadamard = _hadamard(
        resolved_group,
        device=target_device,
        dtype=torch.float32,
    )
    generator = torch.zeros(
        (resolved_group, resolved_group),
        dtype=torch.float32,
        device=target_device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam((generator,), lr=lr)

    initial_error = _relative_error(chunks, hadamard, device=target_device)
    best_error = float(initial_error)
    best_rotation = hadamard.detach().to(device="cpu").clone()
    for step in range(steps):
        source = chunks[step % len(chunks)].to(
            device=target_device,
            dtype=torch.float32,
        )
        rotation = hadamard @ _cayley_delta(generator)
        rotated = source @ rotation
        quantized = _ste_fake_quantize(rotated)
        denominator = rotated.detach().square().mean().clamp(min=1e-30)
        loss = (quantized - rotated.detach()).square().mean() / denominator
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Learned rotation optimization produced non-finite loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_checkpoint = (
            step + 1 == steps or (step + 1) % checkpoint == 0
        )
        if should_checkpoint:
            with torch.no_grad():
                candidate = hadamard @ _cayley_delta(generator)
                candidate_error = _relative_error(
                    chunks,
                    candidate,
                    device=target_device,
                )
                if candidate_error < best_error:
                    best_error = float(candidate_error)
                    best_rotation = candidate.detach().to(device="cpu").clone()

    identity = torch.eye(resolved_group, dtype=torch.float32)
    gram_error = float(
        (best_rotation.transpose(0, 1) @ best_rotation - identity)
        .abs()
        .amax()
        .item()
    )
    if gram_error > 2e-4:
        raise RuntimeError(
            f"Learned rotation lost orthogonality (max error {gram_error:.6g})."
        )
    return LearnedRotationResult(
        rotation=best_rotation.contiguous(),
        group_size=resolved_group,
        initial_relative_error=float(initial_error),
        optimized_relative_error=float(best_error),
        optimization_steps=steps,
        learning_rate=lr,
    )


__all__ = [
    "LEARNED_EXPERT_ROTATION_NAME",
    "LEARNED_EXPERT_ROTATION_SCHEMA_VERSION",
    "LearnedRotationResult",
    "expert_weight_fingerprint",
    "learn_groupwise_expert_rotation",
    "validate_learned_rotation_selection",
]
