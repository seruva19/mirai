"""Shared-basis physical expert weights and offline dense factorization."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any, Mapping

from mirai.core.moe.storage.physical_weights import PhysicalWeightProviderContext
from mirai.core.moe.storage.physical_weights import register_physical_weight_provider

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SHARED_BASIS_PROVIDER_NAME = "shared_basis"
SHARED_BASIS_PROVIDER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SharedBasisFactors:
    """One projection represented by an orthonormal basis plus expert factors."""

    axis: str
    basis: Any
    coefficients: Any
    shape: tuple[int, int, int]
    rank: int
    relative_frobenius_error: float
    weighted_relative_frobenius_error: float | None = None
    whitened_relative_error: float | None = None

    def reconstruct(self, expert_index: int, *, dtype: Any, device: Any) -> Any:
        basis = self.basis.to(device=device, dtype=dtype)
        coefficients = self.coefficients[int(expert_index)].to(
            device=device, dtype=dtype
        )
        if self.axis == "right":
            return coefficients @ basis
        return basis @ coefficients


def _storage_dtype(name: str) -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Shared-basis factorization requires torch.")
    normalized = str(name).strip().lower()
    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if normalized not in choices:
        raise ValueError("factor_dtype must be bfloat16, float16, or float32.")
    return choices[normalized]


def factorize_dense_experts(
    weights: Any,
    *,
    rank: int,
    axis: str = "auto",
    reference_expert: int = 0,
    factor_dtype: str = "bfloat16",
    expert_weights: Any | None = None,
    input_covariance: Any | None = None,
    whitening_regularization: float = 1e-6,
) -> SharedBasisFactors:
    """Project experts onto a shared low-rank basis.

    ``auto`` shares a right basis for up/gate-shaped projections and a left
    basis for down-shaped projections by choosing the smaller factor payload.
    With ``input_covariance``, the factorization is a shared-basis
    adaptation of SVD-LLM's truncation-aware whitening
    (https://arxiv.org/abs/2403.07378): SVD is applied after multiplying each
    projection by the Cholesky factor of its routed-input covariance, and the
    input transform is removed from the stored right-side factor.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Shared-basis factorization requires torch.")
    dense = torch.as_tensor(weights).detach()
    if dense.ndim != 3:
        raise ValueError("Shared-basis weights must have shape [experts, out, in].")
    experts, out_features, in_features = (int(value) for value in dense.shape)
    if experts < 1:
        raise ValueError("Shared-basis factorization requires at least one expert.")
    resolved_rank = int(rank)
    if resolved_rank <= 0 or resolved_rank > min(out_features, in_features):
        raise ValueError(
            f"rank must be in [1, {min(out_features, in_features)}], got {rank}."
        )
    reference = int(reference_expert)
    if reference < 0 or reference >= experts:
        raise ValueError(f"reference_expert {reference} is outside [0, {experts}).")
    resolved_axis = str(axis).strip().lower()
    if resolved_axis == "auto":
        right_elements = resolved_rank * in_features + experts * out_features * resolved_rank
        left_elements = out_features * resolved_rank + experts * resolved_rank * in_features
        resolved_axis = "right" if right_elements <= left_elements else "left"
    if resolved_axis not in {"right", "left"}:
        raise ValueError("axis must be auto, right, or left.")
    if not bool(torch.isfinite(dense).all().item()):
        raise ValueError("Shared-basis source weights contain non-finite values.")

    compute = dense.to(dtype=torch.float32)
    normalized_weights = None
    if expert_weights is not None:
        normalized_weights = torch.as_tensor(
            expert_weights,
            device=compute.device,
            dtype=torch.float32,
        )
        if normalized_weights.ndim != 1 or int(normalized_weights.numel()) != experts:
            raise ValueError(
                f"expert_weights must have shape [{experts}], got "
                f"{tuple(normalized_weights.shape)}."
            )
        if not bool(torch.isfinite(normalized_weights).all().item()):
            raise ValueError("expert_weights contain non-finite values.")
        if bool((normalized_weights <= 0).any().item()):
            raise ValueError("expert_weights must provide positive coverage for every expert.")
        normalized_weights = normalized_weights / normalized_weights.mean()

    whitening_factor = None
    if input_covariance is not None:
        regularization = float(whitening_regularization)
        if not math.isfinite(regularization) or regularization < 0.0:
            raise ValueError("whitening_regularization must be finite and non-negative.")
        covariance = torch.as_tensor(
            input_covariance,
            device=compute.device,
            dtype=torch.float64,
        )
        if covariance.shape != (in_features, in_features):
            raise ValueError(
                "input_covariance must have shape "
                f"[{in_features}, {in_features}], got {tuple(covariance.shape)}."
            )
        if not bool(torch.isfinite(covariance).all().item()):
            raise ValueError("input_covariance contains non-finite values.")
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        diagonal_mean = covariance.diagonal().mean()
        if not bool((diagonal_mean > 0).item()):
            raise ValueError("input_covariance must have positive mean diagonal energy.")
        covariance = covariance / diagonal_mean
        identity = torch.eye(
            in_features,
            dtype=covariance.dtype,
            device=covariance.device,
        )
        stabilized = covariance + regularization * identity
        whitening_factor, info = torch.linalg.cholesky_ex(stabilized)
        if int(info.max().item()) != 0:
            minimum = torch.linalg.eigvalsh(stabilized).min()
            shift = max(
                torch.finfo(stabilized.dtype).eps,
                float((-minimum).clamp_min(0).item())
                + torch.finfo(stabilized.dtype).eps,
            )
            stabilized = stabilized + shift * identity
            whitening_factor = torch.linalg.cholesky(stabilized)
        whitening_factor = whitening_factor.to(dtype=torch.float32)
        inverse_whitening = torch.linalg.solve_triangular(
            whitening_factor,
            torch.eye(
                in_features,
                dtype=torch.float32,
                device=compute.device,
            ),
            upper=False,
        )
        whitened = torch.matmul(compute, whitening_factor)
        if normalized_weights is None:
            scale = torch.ones(
                experts,
                dtype=torch.float32,
                device=compute.device,
            )
        else:
            scale = normalized_weights.sqrt()
        if resolved_axis == "right":
            matrix = (
                whitened * scale[:, None, None]
            ).reshape(experts * out_features, in_features)
            u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
            weighted_coefficients = (
                u[:, :resolved_rank] * singular_values[:resolved_rank]
            ).reshape(experts, out_features, resolved_rank)
            coefficients_compute = weighted_coefficients / scale[:, None, None]
            basis_compute = torch.matmul(
                vh[:resolved_rank],
                inverse_whitening,
            )
        else:
            matrix = (
                whitened * scale[:, None, None]
            ).permute(1, 0, 2).reshape(out_features, experts * in_features)
            u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
            basis_compute = u[:, :resolved_rank].contiguous()
            weighted_coefficients = (
                singular_values[:resolved_rank, None] * vh[:resolved_rank]
            ).reshape(resolved_rank, experts, in_features).permute(1, 0, 2)
            coefficients_compute = torch.matmul(
                weighted_coefficients,
                inverse_whitening,
            ) / scale[:, None, None]
    elif normalized_weights is None:
        reference_weight = compute[reference]
        u, _singular_values, vh = torch.linalg.svd(
            reference_weight,
            full_matrices=False,
        )
        if resolved_axis == "right":
            basis_compute = vh[:resolved_rank].contiguous()
        else:
            basis_compute = u[:, :resolved_rank].contiguous()
    else:
        covariance_size = in_features if resolved_axis == "right" else out_features
        covariance = torch.zeros(
            (covariance_size, covariance_size),
            dtype=torch.float32,
            device=compute.device,
        )
        for expert_index in range(experts):
            weight = compute[expert_index]
            if resolved_axis == "right":
                covariance.addmm_(
                    weight.transpose(0, 1),
                    weight,
                    beta=1.0,
                    alpha=float(normalized_weights[expert_index].item()),
                )
            else:
                covariance.addmm_(
                    weight,
                    weight.transpose(0, 1),
                    beta=1.0,
                    alpha=float(normalized_weights[expert_index].item()),
                )
        _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        leading = eigenvectors[:, -resolved_rank:].flip(1).contiguous()
        basis_compute = (
            leading.transpose(0, 1).contiguous()
            if resolved_axis == "right"
            else leading
        )

    if input_covariance is None:
        if resolved_axis == "right":
            coefficients_compute = torch.matmul(
                compute,
                basis_compute.transpose(-2, -1),
            )
        else:
            coefficients_compute = torch.matmul(
                basis_compute.transpose(-2, -1).unsqueeze(0),
                compute,
            )

    storage_dtype = _storage_dtype(factor_dtype)
    basis = basis_compute.to(device="cpu", dtype=storage_dtype).contiguous()
    coefficients = coefficients_compute.to(
        device="cpu", dtype=storage_dtype
    ).contiguous()
    basis_for_error = basis.to(device=compute.device, dtype=torch.float32)
    coefficients_for_error = coefficients.to(device=compute.device, dtype=torch.float32)
    if resolved_axis == "right":
        reconstructed = torch.matmul(coefficients_for_error, basis_for_error)
    else:
        reconstructed = torch.matmul(
            basis_for_error.unsqueeze(0), coefficients_for_error
        )
    denominator = torch.linalg.vector_norm(compute).clamp_min(torch.finfo(torch.float32).tiny)
    error = float((torch.linalg.vector_norm(compute - reconstructed) / denominator).item())
    if not math.isfinite(error):
        raise RuntimeError("Shared-basis reconstruction error is non-finite.")
    weighted_error = None
    if normalized_weights is not None:
        residual_energy = (compute - reconstructed).square().flatten(1).sum(1)
        source_energy = compute.square().flatten(1).sum(1)
        weighted_denominator = torch.sum(normalized_weights * source_energy).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        weighted_error = float(
            torch.sqrt(
                torch.sum(normalized_weights * residual_energy) / weighted_denominator
            ).item()
        )
        if not math.isfinite(weighted_error):
            raise RuntimeError("Weighted shared-basis reconstruction error is non-finite.")
    whitened_error = None
    if whitening_factor is not None:
        source_whitened = torch.matmul(compute, whitening_factor)
        residual_whitened = torch.matmul(compute - reconstructed, whitening_factor)
        if normalized_weights is None:
            numerator = residual_whitened.square().sum()
            denominator = source_whitened.square().sum()
        else:
            numerator = torch.sum(
                normalized_weights
                * residual_whitened.square().flatten(1).sum(1)
            )
            denominator = torch.sum(
                normalized_weights
                * source_whitened.square().flatten(1).sum(1)
            )
        whitened_error = float(
            torch.sqrt(
                numerator
                / denominator.clamp_min(torch.finfo(torch.float32).tiny)
            ).item()
        )
        if not math.isfinite(whitened_error):
            raise RuntimeError("Whitened shared-basis reconstruction error is non-finite.")
    return SharedBasisFactors(
        axis=resolved_axis,
        basis=basis,
        coefficients=coefficients,
        shape=(experts, out_features, in_features),
        rank=resolved_rank,
        relative_frobenius_error=error,
        weighted_relative_frobenius_error=weighted_error,
        whitened_relative_error=whitened_error,
    )


class SharedBasisPhysicalWeightProvider:
    """Materialize expert matrices from schema-v1 shared-basis factors."""

    name = SHARED_BASIS_PROVIDER_NAME

    def __init__(self, context: PhysicalWeightProviderContext) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Shared-basis provider requires torch.")
        version = int(context.spec.get("schema_version", 0))
        if version != SHARED_BASIS_PROVIDER_SCHEMA_VERSION:
            raise ValueError(f"Unsupported shared-basis provider schema {version}.")
        projections = context.spec.get("projections")
        if not isinstance(projections, Mapping) or set(projections) != {"w1", "w2", "w3"}:
            raise ValueError("Shared-basis provider requires w1, w2, and w3 projections.")
        self.num_experts = int(context.num_experts)
        self._manifest_spec = copy.deepcopy(dict(context.spec))
        self._shapes = {str(key): tuple(value) for key, value in context.shapes.items()}
        self._specs: dict[str, dict[str, Any]] = {}
        self._tensors = context.tensors
        names: set[str] = set()
        for key, raw in projections.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"Shared-basis projection {key!r} must be an object.")
            axis = str(raw.get("axis", ""))
            if axis not in {"right", "left"}:
                raise ValueError(f"Shared-basis projection {key!r} has invalid axis {axis!r}.")
            basis_name = str(raw.get("basis", ""))
            coefficient_name = str(raw.get("coefficients", ""))
            if not basis_name or not coefficient_name:
                raise ValueError(f"Shared-basis projection {key!r} has missing tensor names.")
            if basis_name not in context.tensors or coefficient_name not in context.tensors:
                raise KeyError(f"Shared-basis tensors for projection {key!r} are missing.")
            shape = self._shapes.get(str(key))
            if shape is None or len(shape) != 3 or int(shape[0]) != self.num_experts:
                raise ValueError(f"Shared-basis projection {key!r} has invalid shape {shape}.")
            self._specs[str(key)] = dict(raw)
            names.update({basis_name, coefficient_name})
        self._names = frozenset(names)

    def expert_weight_shape(self, key: str) -> tuple[int, ...]:
        try:
            return tuple(int(value) for value in self._shapes[str(key)])
        except KeyError as exc:
            raise AttributeError(f"Unknown shared-basis projection {key!r}.") from exc

    def packed_tensor_names(self) -> frozenset[str]:
        return self._names

    def manifest_spec(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._manifest_spec)

    def packed_tensors(self) -> Mapping[str, Any]:
        return {name: self._tensors[name] for name in self._names}

    def _coefficient(self, name: str, expert_index: int) -> Any:
        get_slice = getattr(self._tensors, "get_slice", None)
        if callable(get_slice):
            return get_slice(name, int(expert_index))
        return self._tensors[name][int(expert_index)]

    def materialize_expert(
        self,
        key: str,
        expert_index: int,
        *,
        dtype: Any,
        device: Any,
    ) -> Any:
        projection = self._specs[str(key)]
        index = int(expert_index)
        if index < 0 or index >= self.num_experts:
            raise IndexError(f"Expert index {index} is outside [0, {self.num_experts}).")
        with torch.no_grad():
            basis = self._tensors[str(projection["basis"])].to(
                device=device, dtype=dtype
            )
            coefficients = self._coefficient(
                str(projection["coefficients"]), index
            ).to(device=device, dtype=dtype)
            if str(projection["axis"]) == "right":
                return (coefficients @ basis).detach()
            return (basis @ coefficients).detach()


@register_physical_weight_provider(SHARED_BASIS_PROVIDER_NAME)
def _build_shared_basis_provider(
    context: PhysicalWeightProviderContext,
) -> SharedBasisPhysicalWeightProvider:
    return SharedBasisPhysicalWeightProvider(context)


__all__ = [
    "SHARED_BASIS_PROVIDER_NAME",
    "SHARED_BASIS_PROVIDER_SCHEMA_VERSION",
    "SharedBasisFactors",
    "SharedBasisPhysicalWeightProvider",
    "factorize_dense_experts",
]
