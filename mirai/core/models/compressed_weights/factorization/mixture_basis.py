"""Optimized mixture-of-basis expert weights for packed MoE artifacts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping

from mirai.core.models.compressed_weights.packed.packed_contract import (
    _blockwise_fp8_meta_from_spec,
)
from mirai.core.models.compressed_weights.packed.packed_contract import (
    _microscaling_meta_from_spec,
)
from mirai.core.models.compressed_weights.packed.packed_contract import (
    _nf4_meta_from_spec,
)
from mirai.core.models.compressed_weights.quantization.blockwise_fp8 import (
    BLOCKWISE_FP8_FORMATS,
    dequantize_blockwise_fp8_weight,
)
from mirai.core.models.compressed_weights.quantization.gguf_quant import GGUF_FORMATS
from mirai.core.models.compressed_weights.quantization.gguf_quant import (
    dequantize_gguf,
)
from mirai.core.models.compressed_weights.quantization.microscaling_quant import (
    MICROSCALING_FORMATS,
)
from mirai.core.models.compressed_weights.quantization.microscaling_quant import (
    dequantize_microscaling,
)
from mirai.core.models.compressed_weights.quantization.quant import _dequantize_weight
from mirai.core.models.compressed_weights.quantization.quant import _nf4_dequantize
from mirai.core.models.compressed_weights.quantization.quant import (
    normalize_quant_format,
)
from mirai.core.moe.storage.physical_weights import PhysicalWeightProviderContext
from mirai.core.moe.storage.physical_weights import (
    register_physical_weight_provider,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


MIXTURE_BASIS_PROVIDER_NAME = "mixture_basis"
MIXTURE_BASIS_PROVIDER_SCHEMA_VERSION = 1
MIXTURE_BASIS_PROJECTIONS = frozenset({"w1", "w3"})
MIXTURE_BASIS_ACTIVATIONS = frozenset({"silu", "tanh", "gelu"})


def _storage_dtype(name: str) -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Mixture-basis factorization requires torch.")
    normalized = str(name).strip().lower()
    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if normalized not in choices:
        raise ValueError("factor_dtype must be bfloat16, float16, or float32.")
    return choices[normalized]


def _activate(value: Any, activation: str) -> Any:
    normalized = str(activation).strip().lower()
    if normalized == "silu":
        return torch.nn.functional.silu(value)
    if normalized == "tanh":
        return torch.tanh(value)
    if normalized == "gelu":
        return torch.nn.functional.gelu(value)
    raise ValueError(
        "mixture-basis activation must be silu, tanh, or gelu."
    )


def _relative_error(
    source: Any,
    transforms: Any,
    bases: Any,
    coefficients: Any,
    *,
    activation: str,
    expert_batch_size: int,
    row_chunk_size: int,
    device: Any,
) -> float:
    numerator = 0.0
    denominator = 0.0
    experts, out_features, _in_features = (int(value) for value in source.shape)
    for expert_start in range(0, experts, expert_batch_size):
        expert_end = min(experts, expert_start + expert_batch_size)
        mixed = torch.einsum(
            "ek,kri->eri",
            coefficients[expert_start:expert_end],
            bases,
        )
        activated = _activate(mixed, activation)
        for row_start in range(0, out_features, row_chunk_size):
            row_end = min(out_features, row_start + row_chunk_size)
            target = source[
                expert_start:expert_end, row_start:row_end
            ].to(device=device, dtype=torch.float32)
            reconstructed = torch.matmul(
                transforms[
                    expert_start:expert_end, row_start:row_end
                ],
                activated,
            )
            numerator += float((target - reconstructed).square().sum().item())
            denominator += float(target.square().sum().item())
    return math.sqrt(numerator / max(denominator, torch.finfo(torch.float32).tiny))


@dataclass(frozen=True)
class MixtureBasisFactors:
    """One projection represented by learned expert transforms and shared bases."""

    transforms: Any
    bases: Any
    coefficients: Any
    shape: tuple[int, int, int]
    rank: int
    basis_count: int
    activation: str
    optimization_steps: int
    initial_relative_frobenius_error: float
    optimized_relative_frobenius_error: float
    stored_relative_frobenius_error: float
    normalization_std: float
    mean_to_std_ratio: float

    def reconstruct(self, expert_index: int, *, dtype: Any, device: Any) -> Any:
        index = int(expert_index)
        transform = self.transforms[index].to(device=device, dtype=dtype)
        bases = self.bases.to(device=device, dtype=dtype)
        coefficients = self.coefficients[index].to(
            device=device, dtype=torch.float32
        )
        coefficients = coefficients / coefficients.sum().clamp_min(
            torch.finfo(coefficients.dtype).tiny
        )
        mixed = torch.einsum(
            "k,kri->ri",
            coefficients.to(dtype=dtype),
            bases,
        )
        return transform @ _activate(mixed, self.activation)


def factorize_mixture_basis_experts(
    weights: Any,
    *,
    rank: int,
    basis_count: int,
    activation: str = "silu",
    optimization_steps: int = 1000,
    learning_rate: float = 0.07,
    expert_batch_size: int = 8,
    row_chunk_size: int = 256,
    checkpoint_interval: int = 100,
    factor_dtype: str = "bfloat16",
    device: Any = "cpu",
    max_covariance_gib: float = 2.0,
    max_optimizer_gib: float = 24.0,
) -> MixtureBasisFactors:
    """Learn a data-free MoBE decomposition by reconstruction-error minimization.

    Implements the MoBE factorization from arXiv:2508.05257. Experts are
    initialized from groupwise right singular subspaces, then ``A``, shared
    bases, and softmax mixture logits are jointly optimized with Adam. The
    source stays in CPU memory; expert and row chunks bound target uploads and
    reconstruction activations.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("Mixture-basis factorization requires torch.")
    dense = torch.as_tensor(weights).detach().to(device="cpu", dtype=torch.float32)
    if dense.ndim != 3:
        raise ValueError(
            "Mixture-basis weights must have shape [experts, out, in]."
        )
    if not bool(torch.isfinite(dense).all().item()):
        raise ValueError("Mixture-basis source weights contain non-finite values.")
    experts, out_features, in_features = (int(value) for value in dense.shape)
    if experts < 1:
        raise ValueError("Mixture-basis factorization requires at least one expert.")
    resolved_rank = int(rank)
    if resolved_rank <= 0 or resolved_rank > min(out_features, in_features):
        raise ValueError(
            f"rank must be in [1, {min(out_features, in_features)}], got {rank}."
        )
    resolved_basis_count = int(basis_count)
    if resolved_basis_count <= 0 or resolved_basis_count > experts:
        raise ValueError(
            f"basis_count must be in [1, {experts}], got {basis_count}."
        )
    resolved_activation = str(activation).strip().lower()
    if resolved_activation not in MIXTURE_BASIS_ACTIVATIONS:
        raise ValueError(
            "mixture-basis activation must be silu, tanh, or gelu."
        )
    steps = int(optimization_steps)
    if steps <= 0:
        raise ValueError("optimization_steps must be positive.")
    lr = float(learning_rate)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("learning_rate must be finite and positive.")
    batch_size = int(expert_batch_size)
    row_chunk = int(row_chunk_size)
    checkpoint_every = int(checkpoint_interval)
    if batch_size <= 0 or row_chunk <= 0 or checkpoint_every <= 0:
        raise ValueError(
            "expert_batch_size, row_chunk_size, and checkpoint_interval "
            "must be positive."
        )
    resolved_device = torch.device(device)
    covariance_bytes = in_features * in_features * 4
    covariance_limit = float(max_covariance_gib) * (1024**3)
    if (
        not math.isfinite(covariance_limit)
        or covariance_limit <= 0
        or covariance_bytes > covariance_limit
    ):
        raise ValueError(
            "Mixture-basis initialization covariance exceeds "
            f"max_covariance_gib={max_covariance_gib}."
        )
    parameter_elements = (
        experts * out_features * resolved_rank
        + resolved_basis_count * resolved_rank * in_features
        + experts * resolved_basis_count
    )
    estimated_optimizer_bytes = parameter_elements * 16
    optimizer_limit = float(max_optimizer_gib) * (1024**3)
    if (
        not math.isfinite(optimizer_limit)
        or optimizer_limit <= 0
        or estimated_optimizer_bytes > optimizer_limit
    ):
        raise ValueError(
            "Estimated FP32 Adam working set exceeds "
            f"max_optimizer_gib={max_optimizer_gib}."
        )

    normalization_std = float(dense.std(unbiased=False).item())
    if not math.isfinite(normalization_std) or normalization_std <= 0.0:
        raise ValueError(
            "Mixture-basis factorization requires non-zero finite weight variance."
        )
    mean_to_std_ratio = abs(float(dense.mean().item())) / normalization_std
    normalized = dense / normalization_std

    bases = torch.empty(
        (resolved_basis_count, resolved_rank, in_features),
        device=resolved_device,
        dtype=torch.float32,
    )
    transforms = torch.empty(
        (experts, out_features, resolved_rank),
        device=resolved_device,
        dtype=torch.float32,
    )
    assignments = torch.div(
        torch.arange(experts, dtype=torch.int64) * resolved_basis_count,
        experts,
        rounding_mode="floor",
    )
    for basis_index in range(resolved_basis_count):
        members = torch.nonzero(
            assignments == basis_index, as_tuple=False
        ).flatten().tolist()
        covariance = torch.zeros(
            (in_features, in_features),
            device=resolved_device,
            dtype=torch.float32,
        )
        for expert_index in members:
            for row_start in range(0, out_features, row_chunk):
                row_end = min(out_features, row_start + row_chunk)
                block = normalized[
                    expert_index, row_start:row_end
                ].to(device=resolved_device)
                covariance.addmm_(
                    block.transpose(0, 1),
                    block,
                    beta=1.0,
                    alpha=1.0,
                )
        _eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        basis = (
            eigenvectors[:, -resolved_rank:]
            .flip(1)
            .transpose(0, 1)
            .contiguous()
        )
        bases[basis_index].copy_(basis)
        for expert_index in members:
            for row_start in range(0, out_features, row_chunk):
                row_end = min(out_features, row_start + row_chunk)
                block = normalized[
                    expert_index, row_start:row_end
                ].to(device=resolved_device)
                transforms[expert_index, row_start:row_end].copy_(
                    block @ basis.transpose(0, 1)
                )
        del covariance

    probability = 0.95
    margin = math.log(
        probability
        * max(1, resolved_basis_count - 1)
        / max(torch.finfo(torch.float32).eps, 1.0 - probability)
    )
    logits = torch.full(
        (experts, resolved_basis_count),
        -0.5 * margin,
        device=resolved_device,
        dtype=torch.float32,
    )
    logits[
        torch.arange(experts, device=resolved_device),
        assignments.to(device=resolved_device),
    ] = 0.5 * margin
    transforms.requires_grad_(True)
    bases.requires_grad_(True)
    logits.requires_grad_(True)

    with torch.no_grad():
        initial_error = _relative_error(
            normalized,
            transforms,
            bases,
            torch.softmax(logits, dim=-1),
            activation=resolved_activation,
            expert_batch_size=batch_size,
            row_chunk_size=row_chunk,
            device=resolved_device,
        )
    best_error = initial_error
    best_state = (
        transforms.detach().cpu().clone(),
        bases.detach().cpu().clone(),
        logits.detach().cpu().clone(),
    )
    optimizer = torch.optim.Adam(
        (transforms, bases, logits),
        lr=lr,
    )
    total_elements = int(normalized.numel())
    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for expert_start in range(0, experts, batch_size):
            expert_end = min(experts, expert_start + batch_size)
            for row_start in range(0, out_features, row_chunk):
                row_end = min(out_features, row_start + row_chunk)
                coefficients = torch.softmax(
                    logits[expert_start:expert_end],
                    dim=-1,
                )
                mixed = torch.einsum(
                    "ek,kri->eri",
                    coefficients,
                    bases,
                )
                activated = _activate(mixed, resolved_activation)
                reconstructed = torch.matmul(
                    transforms[
                        expert_start:expert_end, row_start:row_end
                    ],
                    activated,
                )
                target = normalized[
                    expert_start:expert_end, row_start:row_end
                ].to(device=resolved_device)
                loss = (target - reconstructed).square().sum() / total_elements
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(
                        "Mixture-basis reconstruction loss became non-finite."
                    )
                loss.backward()
        optimizer.step()
        should_checkpoint = (
            (step_index + 1) % checkpoint_every == 0
            or step_index + 1 == steps
        )
        if not should_checkpoint:
            continue
        with torch.no_grad():
            candidate_error = _relative_error(
                normalized,
                transforms,
                bases,
                torch.softmax(logits, dim=-1),
                activation=resolved_activation,
                expert_batch_size=batch_size,
                row_chunk_size=row_chunk,
                device=resolved_device,
            )
        if not math.isfinite(candidate_error):
            raise RuntimeError(
                "Mixture-basis reconstruction error became non-finite."
            )
        if candidate_error < best_error:
            best_error = candidate_error
            best_state = (
                transforms.detach().cpu().clone(),
                bases.detach().cpu().clone(),
                logits.detach().cpu().clone(),
            )

    best_transforms, best_bases, best_logits = best_state
    best_coefficients = torch.softmax(best_logits, dim=-1)
    storage_dtype = _storage_dtype(factor_dtype)
    stored_transforms = (
        best_transforms * normalization_std
    ).to(dtype=storage_dtype).contiguous()
    stored_bases = best_bases.to(dtype=storage_dtype).contiguous()
    stored_coefficients = best_coefficients.to(
        dtype=storage_dtype
    ).contiguous()
    with torch.no_grad():
        stored_error = _relative_error(
            dense,
            stored_transforms.to(device=resolved_device, dtype=torch.float32),
            stored_bases.to(device=resolved_device, dtype=torch.float32),
            (
                stored_coefficients.to(
                    device=resolved_device, dtype=torch.float32
                )
                / stored_coefficients.to(
                    device=resolved_device, dtype=torch.float32
                )
                .sum(dim=-1, keepdim=True)
                .clamp_min(torch.finfo(torch.float32).tiny)
            ),
            activation=resolved_activation,
            expert_batch_size=batch_size,
            row_chunk_size=row_chunk,
            device=resolved_device,
        )
    if not math.isfinite(stored_error):
        raise RuntimeError(
            "Stored mixture-basis reconstruction error is non-finite."
        )
    return MixtureBasisFactors(
        transforms=stored_transforms,
        bases=stored_bases,
        coefficients=stored_coefficients,
        shape=(experts, out_features, in_features),
        rank=resolved_rank,
        basis_count=resolved_basis_count,
        activation=resolved_activation,
        optimization_steps=steps,
        initial_relative_frobenius_error=float(initial_error),
        optimized_relative_frobenius_error=float(best_error),
        stored_relative_frobenius_error=float(stored_error),
        normalization_std=normalization_std,
        mean_to_std_ratio=mean_to_std_ratio,
    )


class MixtureBasisPhysicalWeightProvider:
    """Materialize optimized w1/w3 factors and an unchanged packed w2."""

    name = MIXTURE_BASIS_PROVIDER_NAME

    def __init__(self, context: PhysicalWeightProviderContext) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("Mixture-basis provider requires torch.")
        version = int(context.spec.get("schema_version", 0))
        if version != MIXTURE_BASIS_PROVIDER_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported mixture-basis provider schema {version}."
            )
        projections = context.spec.get("projections")
        if (
            not isinstance(projections, Mapping)
            or set(projections) != MIXTURE_BASIS_PROJECTIONS
        ):
            raise ValueError(
                "Mixture-basis provider requires w1 and w3 factor projections."
            )
        self.num_experts = int(context.num_experts)
        self._manifest_spec = copy.deepcopy(dict(context.spec))
        self._shapes = {
            str(key): tuple(int(value) for value in shape)
            for key, shape in context.shapes.items()
        }
        if set(self._shapes) != {"w1", "w2", "w3"}:
            raise ValueError(
                "Mixture-basis provider requires w1, w2, and w3 shapes."
            )
        self._tensors = context.tensors
        self._specs: dict[str, dict[str, Any]] = {}
        names: set[str] = set()
        for key, raw in projections.items():
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Mixture-basis projection {key!r} must be an object."
                )
            projection = dict(raw)
            activation = str(projection.get("activation", "")).strip().lower()
            if activation not in MIXTURE_BASIS_ACTIVATIONS:
                raise ValueError(
                    f"Mixture-basis projection {key!r} has invalid activation."
                )
            tensor_names = {
                str(projection.get("transforms", "")),
                str(projection.get("bases", "")),
                str(projection.get("coefficients", "")),
            }
            if "" in tensor_names or not tensor_names.issubset(context.tensors):
                raise KeyError(
                    f"Mixture-basis tensors for projection {key!r} are missing."
                )
            shape = self._shapes[str(key)]
            if len(shape) != 3 or shape[0] != self.num_experts:
                raise ValueError(
                    f"Mixture-basis projection {key!r} has invalid shape."
                )
            rank = int(projection.get("rank", 0))
            basis_count = int(projection.get("basis_count", 0))
            if rank <= 0 or rank > min(shape[1], shape[2]):
                raise ValueError(
                    f"Mixture-basis {key!r} has invalid rank."
                )
            if basis_count <= 0 or basis_count > self.num_experts:
                raise ValueError(
                    f"Mixture-basis {key!r} has invalid basis count."
                )
            self._specs[str(key)] = projection
            names.update(tensor_names)

        down = context.spec.get("down_projection")
        if not isinstance(down, Mapping):
            raise ValueError(
                "Mixture-basis provider requires an unchanged w2 projection."
            )
        self._down = copy.deepcopy(dict(down))
        self._down_format = normalize_quant_format(
            str(self._down.get("quant_format", "int8"))
        )
        down_tensors = self._down.get("tensors")
        if not isinstance(down_tensors, Mapping):
            raise ValueError("Mixture-basis w2 tensor map is missing.")
        self._down_tensors = {
            str(key): str(value) for key, value in down_tensors.items()
        }
        required = self._required_down_roles(self._down_format)
        if set(self._down_tensors) != required:
            raise ValueError(
                "Mixture-basis w2 tensor roles do not match its quantization format."
            )
        down_names = set(self._down_tensors.values())
        if "" in down_names or not down_names.issubset(context.tensors):
            raise KeyError("Mixture-basis unchanged w2 tensors are missing.")
        names.update(down_names)
        self._names = frozenset(names)

    @staticmethod
    def _required_down_roles(quant_format: str) -> set[str]:
        if quant_format == "nf4":
            return {
                "packed",
                "absmax",
                "nested_absmax",
                "offset",
                "code",
                "nested_code",
            }
        if quant_format in GGUF_FORMATS:
            return {"blocks"}
        if quant_format in MICROSCALING_FORMATS:
            return {"packed", "scales", "global_scale"}
        if quant_format in BLOCKWISE_FP8_FORMATS:
            return {"codes", "scales"}
        return {"quantized", "scale"}

    def expert_weight_shape(self, key: str) -> tuple[int, ...]:
        try:
            return self._shapes[str(key)]
        except KeyError as exc:
            raise AttributeError(
                f"Unknown mixture-basis projection {key!r}."
            ) from exc

    def packed_tensor_names(self) -> frozenset[str]:
        return self._names

    def manifest_spec(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._manifest_spec)

    def packed_tensors(self) -> Mapping[str, Any]:
        return {name: self._tensors[name] for name in self._names}

    def _slice(self, name: str, expert_index: int) -> Any:
        get_slice = getattr(self._tensors, "get_slice", None)
        if callable(get_slice):
            return get_slice(name, int(expert_index))
        return self._tensors[name][int(expert_index)]

    def _materialize_down(
        self,
        expert_index: int,
        *,
        dtype: Any,
        device: Any,
    ) -> Any:
        names = self._down_tensors
        shape = self._shapes["w2"][1:]
        if self._down_format == "nf4":
            fields = {
                key: self._slice(names[key], expert_index)
                for key in ("packed", "absmax", "nested_absmax", "offset")
            }
            codes = {
                "code": self._tensors[names["code"]],
                "nested_code": self._tensors[names["nested_code"]],
            }
            return _nf4_dequantize(
                fields,
                codes,
                _nf4_meta_from_spec(self._down),
                shape=shape,
                dtype=dtype,
                device=torch.device(device),
            )
        if self._down_format in GGUF_FORMATS:
            return dequantize_gguf(
                self._down_format,
                self._slice(names["blocks"], expert_index),
                shape=shape,
                dtype=dtype,
                device=torch.device(device),
            )
        if self._down_format in MICROSCALING_FORMATS:
            return dequantize_microscaling(
                self._slice(names["packed"], expert_index),
                self._slice(names["scales"], expert_index),
                self._slice(names["global_scale"], expert_index),
                _microscaling_meta_from_spec(self._down),
                dtype=dtype,
                device=torch.device(device),
            )
        if self._down_format in BLOCKWISE_FP8_FORMATS:
            return dequantize_blockwise_fp8_weight(
                self._slice(names["codes"], expert_index),
                self._slice(names["scales"], expert_index),
                _blockwise_fp8_meta_from_spec(self._down),
                dtype=dtype,
                device=torch.device(device),
            )
        return _dequantize_weight(
            self._slice(names["quantized"], expert_index),
            self._slice(names["scale"], expert_index),
            group_size=int(self._down.get("group_size", 0)),
            dtype=dtype,
            device=torch.device(device),
        )

    def materialize_expert(
        self,
        key: str,
        expert_index: int,
        *,
        dtype: Any,
        device: Any,
    ) -> Any:
        projection_key = str(key)
        index = int(expert_index)
        if index < 0 or index >= self.num_experts:
            raise IndexError(
                f"Expert index {index} is outside [0, {self.num_experts})."
            )
        if projection_key == "w2":
            return self._materialize_down(
                index,
                dtype=dtype,
                device=device,
            ).detach()
        projection = self._specs[projection_key]
        transforms = self._slice(
            str(projection["transforms"]),
            index,
        ).to(device=device, dtype=dtype)
        bases = self._tensors[str(projection["bases"])].to(
            device=device, dtype=dtype
        )
        coefficients = self._slice(
            str(projection["coefficients"]),
            index,
        ).to(device=device, dtype=torch.float32)
        rank = int(projection["rank"])
        basis_count = int(projection["basis_count"])
        shape = self._shapes[projection_key]
        if tuple(transforms.shape) != (shape[1], rank):
            raise ValueError(
                f"Mixture-basis {projection_key!r} transform slice has "
                "invalid shape."
            )
        if tuple(bases.shape) != (basis_count, rank, shape[2]):
            raise ValueError(
                f"Mixture-basis {projection_key!r} bases have invalid shape."
            )
        if tuple(coefficients.shape) != (basis_count,):
            raise ValueError(
                f"Mixture-basis {projection_key!r} coefficient slice has "
                "invalid shape."
            )
        if not bool(torch.isfinite(coefficients).all().item()):
            raise ValueError(
                f"Mixture-basis {projection_key!r} coefficients are non-finite."
            )
        if bool((coefficients < 0).any().item()):
            raise ValueError(
                f"Mixture-basis {projection_key!r} coefficients must be "
                "non-negative."
            )
        if not bool((coefficients.sum() > 0).item()):
            raise ValueError(
                f"Mixture-basis {projection_key!r} coefficients have zero mass."
            )
        coefficients = coefficients / coefficients.sum().clamp_min(
            torch.finfo(coefficients.dtype).tiny
        )
        mixed = torch.einsum(
            "k,kri->ri",
            coefficients.to(dtype=dtype),
            bases,
        )
        return (
            transforms
            @ _activate(mixed, str(projection["activation"]))
        ).detach()


@register_physical_weight_provider(MIXTURE_BASIS_PROVIDER_NAME)
def _build_mixture_basis_provider(
    context: PhysicalWeightProviderContext,
) -> MixtureBasisPhysicalWeightProvider:
    return MixtureBasisPhysicalWeightProvider(context)


__all__ = [
    "MIXTURE_BASIS_ACTIVATIONS",
    "MIXTURE_BASIS_PROJECTIONS",
    "MIXTURE_BASIS_PROVIDER_NAME",
    "MIXTURE_BASIS_PROVIDER_SCHEMA_VERSION",
    "MixtureBasisFactors",
    "MixtureBasisPhysicalWeightProvider",
    "factorize_mixture_basis_experts",
]
