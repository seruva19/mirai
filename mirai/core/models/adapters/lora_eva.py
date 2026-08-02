"""Fixed-rank EVA activation calibration for dense and routed LoRA hosts.

This module implements the ``rho=1`` case from EVA: streaming principal
components of downstream input activations initialize each LoRA A factor while
B remains zero. Rank redistribution is intentionally a separate shape-changing
feature because it also owns optimizer and checkpoint migration contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mirai.core.models.adapters.lora import LoRAExpertTensorParametrization
from mirai.core.models.adapters.lora import LoRALinear
from mirai.core.models.compressed_weights.execution.active_expert_lora import ActiveExpertLoRA

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EVATargetReport:
    name: str
    kind: str
    calibrated_ranks: int
    activation_samples: int
    updates: int
    minimum_component_similarity: float


@dataclass(frozen=True)
class EVAInitializationReport:
    calibration_steps: int
    targets: tuple[EVATargetReport, ...]

    @property
    def calibrated_ranks(self) -> int:
        return sum(item.calibrated_ranks for item in self.targets)

    @property
    def activation_samples(self) -> int:
        return sum(item.activation_samples for item in self.targets)


class IncrementalActivationSVD:
    """Bounded-memory sequential Karhunen-Loève update for activation rows."""

    def __init__(
        self,
        *,
        n_components: int,
        feature_dim: int,
        samples_per_update: int,
        convergence_threshold: float,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise RuntimeError("EVA calibration requires torch.")
        self.n_components = int(n_components)
        self.feature_dim = int(feature_dim)
        self.samples_per_update = int(samples_per_update)
        self.convergence_threshold = float(convergence_threshold)
        if self.n_components <= 0 or self.n_components > self.feature_dim:
            raise ValueError(
                "EVA rank must be positive and no larger than the input dimension."
            )
        if self.samples_per_update <= 0:
            raise ValueError("EVA samples_per_update must be positive.")
        if not 0.0 < self.convergence_threshold <= 1.0:
            raise ValueError("EVA convergence threshold must be in (0, 1].")
        self.components: Any | None = None
        self.singular_values: Any | None = None
        self.mean: Any | None = None
        self.converged_components = torch.zeros(self.n_components, dtype=torch.bool)
        self.component_similarity = torch.zeros(self.n_components, dtype=torch.float32)
        self.num_samples = 0
        self.updates = 0

    @property
    def converged(self) -> bool:
        return bool(self.converged_components.all().item())

    def partial_fit(self, activations: Any) -> None:
        # Match EVA's convergence-driven hook removal: once every retained
        # direction converges, freeze this target while slower targets finish.
        if self.converged:
            return
        if not isinstance(activations, torch.Tensor):
            raise TypeError("EVA activation observations must be tensors.")
        if activations.ndim < 2 or int(activations.shape[-1]) != self.feature_dim:
            raise ValueError(
                "EVA activation rows must have the calibrated target's input width."
            )
        rows = activations.detach().reshape(-1, self.feature_dim)
        if int(rows.shape[0]) == 0:
            return
        if int(rows.shape[0]) > self.samples_per_update:
            # Deterministic, order-preserving coverage of the complete token axis.
            indices = (
                torch.arange(self.samples_per_update, device=rows.device)
                * int(rows.shape[0])
                // self.samples_per_update
            )
            rows = rows.index_select(0, indices)
        rows = rows.to(dtype=torch.float32)
        if not bool(torch.isfinite(rows).all().item()):
            raise ValueError("EVA calibration received non-finite activations.")

        batch_size = int(rows.shape[0])
        batch_mean = rows.mean(dim=0)
        centered = rows - batch_mean
        previous = self.components
        if self.num_samples == 0:
            update_matrix = centered
            next_mean = batch_mean
        else:
            old_mean = self.mean.to(device=rows.device, dtype=torch.float32)
            old_components = self.components.to(device=rows.device, dtype=torch.float32)
            old_singular = self.singular_values.to(
                device=rows.device, dtype=torch.float32
            )
            total = self.num_samples + batch_size
            mean_correction = (
                (old_mean - batch_mean)
                * ((self.num_samples * batch_size / total) ** 0.5)
            ).unsqueeze(0)
            update_matrix = torch.cat(
                (
                    old_singular.unsqueeze(1) * old_components,
                    centered,
                    mean_correction,
                ),
                dim=0,
            )
            next_mean = (
                old_mean * self.num_samples + batch_mean * batch_size
            ) / total

        _, singular_values, right_vectors = torch.linalg.svd(
            update_matrix, full_matrices=False
        )
        keep = min(self.n_components, int(right_vectors.shape[0]))
        resolved = right_vectors[:keep]
        similarities = torch.zeros(
            self.n_components, device=rows.device, dtype=torch.float32
        )
        converged = torch.zeros(
            self.n_components, device=rows.device, dtype=torch.bool
        )
        if previous is not None and int(previous.shape[0]) >= self.n_components and keep >= self.n_components:
            old = previous.to(device=rows.device, dtype=torch.float32)
            similarities = torch.nn.functional.cosine_similarity(
                resolved[: self.n_components], old[: self.n_components], dim=1
            ).abs()
            converged = similarities >= self.convergence_threshold
        self.components = resolved.detach().to(device="cpu")
        self.singular_values = singular_values[:keep].detach().to(device="cpu")
        self.mean = next_mean.detach().to(device="cpu")
        self.converged_components = converged.detach().to(device="cpu")
        self.component_similarity = similarities.detach().to(device="cpu")
        self.num_samples += batch_size
        self.updates += 1

    def resolved_components(self) -> Any:
        if self.components is None or int(self.components.shape[0]) < self.n_components:
            raise RuntimeError("EVA target did not observe enough activation rows.")
        if not self.converged:
            raise RuntimeError("EVA target principal components did not converge.")
        return self.components[: self.n_components]


class RoutedExpertEVAObserver:
    """Translate explicit expert dispatch layouts into physical-expert rows."""

    def __init__(self, accumulators: dict[int, IncrementalActivationSVD]) -> None:
        self.accumulators = dict(accumulators)

    def observe_single(self, activations: Any, *, expert_idx: int) -> None:
        accumulator = self.accumulators.get(int(expert_idx))
        if accumulator is not None:
            accumulator.partial_fit(activations)

    def observe_batched(self, activations: Any, *, expert_indices: Any) -> None:
        indices = torch.as_tensor(expert_indices, dtype=torch.long).reshape(-1).tolist()
        if int(activations.shape[0]) != len(indices):
            raise ValueError("EVA batched expert observations have inconsistent shapes.")
        for position, expert_idx in enumerate(indices):
            self.observe_single(activations[position], expert_idx=int(expert_idx))

    def observe_segmented(
        self,
        activations: Any,
        *,
        expert_indices: Any,
        counts: Any,
    ) -> None:
        indices = torch.as_tensor(expert_indices, dtype=torch.long).reshape(-1).tolist()
        sizes = torch.as_tensor(counts, dtype=torch.long).reshape(-1).tolist()
        if len(indices) != len(sizes) or sum(int(item) for item in sizes) != int(
            activations.shape[0]
        ):
            raise ValueError("EVA segmented expert observations have inconsistent shapes.")
        start = 0
        for expert_idx, count in zip(indices, sizes):
            end = start + int(count)
            if end > start:
                self.observe_single(
                    activations[start:end], expert_idx=int(expert_idx)
                )
            start = end


class EVAActivationCalibration:
    """Install activation observers and finalize fixed-rank EVA factors."""

    def __init__(
        self,
        root: Any,
        *,
        samples_per_target: int,
        convergence_threshold: float,
        components_per_target: int | None = None,
    ) -> None:
        if nn is None:  # pragma: no cover
            raise RuntimeError("EVA calibration requires torch.")
        self.root = root
        self.samples_per_target = int(samples_per_target)
        self.convergence_threshold = float(convergence_threshold)
        self.components_per_target = (
            None if components_per_target is None else int(components_per_target)
        )
        if self.components_per_target is not None and self.components_per_target <= 0:
            raise ValueError("EVA components_per_target must be positive.")
        self._dense: dict[str, tuple[Any, IncrementalActivationSVD]] = {}
        self._experts: dict[
            str, tuple[Any, dict[int, IncrementalActivationSVD], RoutedExpertEVAObserver]
        ] = {}
        self._handles: list[Any] = []
        unsupported: list[str] = []
        for name, module in root.named_modules():
            if str(getattr(module, "_lora_init", "")).strip().lower() != "eva":
                continue
            if isinstance(module, LoRALinear):
                self._dense[name] = (
                    module,
                    self._new_accumulator(module.rank, module.in_features),
                )
            elif isinstance(module, ActiveExpertLoRA):
                if bool(getattr(module, "_expert_selection_active", False)):
                    expert_ids = module.active_expert_ids()
                else:
                    expert_ids = list(range(int(module.lora_a.shape[0])))
                accumulators = {
                    expert_idx: self._new_accumulator(
                        module.rank, int(module.lora_a.shape[-1])
                    )
                    for expert_idx in expert_ids
                }
                self._experts[name] = (
                    module,
                    accumulators,
                    RoutedExpertEVAObserver(accumulators),
                )
            elif isinstance(module, LoRAExpertTensorParametrization):
                unsupported.append(name)
        if unsupported:
            raise ValueError(
                "EVA requires explicit routed-expert activation hosts; unsupported "
                "weight-parametrization targets: " + ", ".join(sorted(unsupported))
            )
        if not self._dense and not self._experts:
            raise ValueError("EVA requested but no compatible LoRA targets were found.")

    def _new_accumulator(self, rank: int, feature_dim: int) -> IncrementalActivationSVD:
        return IncrementalActivationSVD(
            n_components=min(
                int(feature_dim),
                int(rank) if self.components_per_target is None else self.components_per_target,
            ),
            feature_dim=int(feature_dim),
            samples_per_update=self.samples_per_target,
            convergence_threshold=self.convergence_threshold,
        )

    def install(self) -> None:
        if self._handles:
            raise RuntimeError("EVA activation observers are already installed.")
        for _, (module, accumulator) in self._dense.items():
            def observe(_module: Any, inputs: tuple[Any, ...], *, target=accumulator) -> None:
                if not inputs:
                    raise ValueError("EVA LoRA target received no positional input.")
                target.partial_fit(inputs[0])

            self._handles.append(module.register_forward_pre_hook(observe))
        for _, (module, _, observer) in self._experts.items():
            module.set_activation_calibration_observer(observer)

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for module, _, _ in self._experts.values():
            module.set_activation_calibration_observer(None)

    @property
    def converged(self) -> bool:
        return all(acc.converged for acc in self._all_accumulators())

    def pending_targets(self) -> tuple[str, ...]:
        pending = [name for name, (_, acc) in self._dense.items() if not acc.converged]
        for name, (_, accumulators, _) in self._experts.items():
            pending.extend(
                f"{name}[expert={expert_idx}]"
                for expert_idx, accumulator in accumulators.items()
                if not accumulator.converged
            )
        return tuple(pending)

    def _all_accumulators(self) -> list[IncrementalActivationSVD]:
        result = [accumulator for _, accumulator in self._dense.values()]
        for _, accumulators, _ in self._experts.values():
            result.extend(accumulators.values())
        return result

    def explained_variance_spectra(self) -> dict[str, tuple[float, ...]]:
        """Return target-level component energy for offline rank allocation."""
        pending = self.pending_targets()
        if pending:
            raise RuntimeError(
                "EVA activation calibration did not converge for: "
                + ", ".join(pending[:8])
            )
        spectra: dict[str, tuple[float, ...]] = {}
        for name, (_, accumulator) in self._dense.items():
            values = accumulator.singular_values[: accumulator.n_components].square()
            spectra[name] = tuple(float(value) for value in values.tolist())
        for name, (_, accumulators, _) in self._experts.items():
            combined = None
            for accumulator in accumulators.values():
                values = accumulator.singular_values[: accumulator.n_components].square()
                combined = values if combined is None else combined + values
            if combined is None:
                raise RuntimeError(f"EVA expert target {name!r} has no physical experts.")
            spectra[name] = tuple(float(value) for value in combined.tolist())
        return spectra

    def initialize(self, *, calibration_steps: int) -> EVAInitializationReport:
        pending = self.pending_targets()
        if pending:
            preview = ", ".join(pending[:8])
            suffix = " ..." if len(pending) > 8 else ""
            raise RuntimeError(
                "EVA activation calibration did not converge for: " + preview + suffix
            )
        reports: list[EVATargetReport] = []
        with torch.no_grad():
            for name, (module, accumulator) in self._dense.items():
                module.lora_a.copy_(
                    accumulator.resolved_components().to(
                        device=module.lora_a.device, dtype=module.lora_a.dtype
                    )
                )
                module.lora_b.zero_()
                reports.append(self._report(name, "dense", accumulator, module.rank))
            for name, (module, accumulators, _) in self._experts.items():
                module.lora_b.zero_()
                for expert_idx, accumulator in accumulators.items():
                    module.lora_a[int(expert_idx)].copy_(
                        accumulator.resolved_components().to(
                            device=module.lora_a.device, dtype=module.lora_a.dtype
                        )
                    )
                    reports.append(
                        self._report(
                            f"{name}[expert={expert_idx}]",
                            "routed_expert",
                            accumulator,
                            module.rank,
                        )
                    )
        return EVAInitializationReport(
            calibration_steps=int(calibration_steps), targets=tuple(reports)
        )

    @staticmethod
    def _report(
        name: str,
        kind: str,
        accumulator: IncrementalActivationSVD,
        rank: int,
    ) -> EVATargetReport:
        return EVATargetReport(
            name=str(name),
            kind=str(kind),
            calibrated_ranks=int(rank),
            activation_samples=int(accumulator.num_samples),
            updates=int(accumulator.updates),
            minimum_component_similarity=float(
                accumulator.component_similarity.min().item()
            ),
        )


__all__ = [
    "EVAActivationCalibration",
    "EVAInitializationReport",
    "EVATargetReport",
    "IncrementalActivationSVD",
    "RoutedExpertEVAObserver",
]
