"""EAQuant-style calibration for frozen router quantization.

The objective follows Equation 10 of EAQuant and restricts the probability
alignment term to the reference top-m experts as defined by Equation 13:
https://arxiv.org/abs/2506.13329
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from mirai.core.moe.calibration.router_repair import router_tensor_fingerprint

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


ROUTER_QUANTIZATION_CALIBRATION_SCHEMA = "mirai.router_quantization_calibration"
ROUTER_QUANTIZATION_CALIBRATION_VERSION = 1
ROUTER_QUANTIZATION_CALIBRATION_METADATA_KEY = (
    "mirai_router_quantization_calibration"
)


@dataclass(frozen=True)
class RouterLinearCalibrationBatch:
    """Linear router features plus any unquantized additive logit branch."""

    features: Any
    additive_logits: Any | None = None

    def validate(
        self,
        *,
        input_features: int,
        num_experts: int,
    ) -> "RouterLinearCalibrationBatch":
        if torch is None:  # pragma: no cover
            raise RuntimeError("Router quantization calibration requires torch.")
        features = torch.as_tensor(self.features)
        if (
            features.ndim != 2
            or int(features.shape[0]) <= 0
            or int(features.shape[1]) != int(input_features)
            or not features.is_floating_point()
            or not bool(torch.isfinite(features).all().item())
        ):
            raise ValueError(
                "Router calibration features must be finite floating-point "
                "[tokens, input_features]."
            )
        if self.additive_logits is not None:
            additive = torch.as_tensor(self.additive_logits)
            if (
                additive.shape != (int(features.shape[0]), int(num_experts))
                or not additive.is_floating_point()
                or not bool(torch.isfinite(additive).all().item())
            ):
                raise ValueError(
                    "Router calibration additive logits must be finite "
                    "[tokens, num_experts]."
                )
        return self


@dataclass(frozen=True)
class RouterQuantizationCalibrationTarget:
    """Provider-owned view of one frozen linear router."""

    name: str
    observation_module: Any
    num_experts: int
    input_features: int
    top_k: int
    read_weight: Callable[[], Any]
    capture_batch: Callable[
        [tuple[Any, ...], Mapping[str, Any]], RouterLinearCalibrationBatch
    ]
    install_int8_scale: Callable[[Any], None]

    def validate(self) -> "RouterQuantizationCalibrationTarget":
        if not self.name:
            raise ValueError("Router quantization target name must be non-empty.")
        if (
            int(self.num_experts) < 2
            or int(self.input_features) <= 0
            or int(self.top_k) <= 0
            or int(self.top_k) >= int(self.num_experts)
        ):
            raise ValueError("Router quantization target topology is invalid.")
        if not callable(
            getattr(self.observation_module, "register_forward_pre_hook", None)
        ):
            raise TypeError("Router observation modules must support pre-hooks.")
        for label, callback in (
            ("read_weight", self.read_weight),
            ("capture_batch", self.capture_batch),
            ("install_int8_scale", self.install_int8_scale),
        ):
            if not callable(callback):
                raise TypeError(f"Router quantization target {label} must be callable.")
        weight = torch.as_tensor(self.read_weight())
        if (
            weight.shape != (int(self.num_experts), int(self.input_features))
            or not weight.is_floating_point()
            or not bool(torch.isfinite(weight).all().item())
        ):
            raise ValueError(
                "Router quantization target weight must be finite floating-point "
                "[num_experts, input_features]."
            )
        return self

    @property
    def estimated_input_bytes_per_token(self) -> int:
        return 4 * (int(self.input_features) + int(self.num_experts))


class RouterInputAccumulator:
    """Bounded deterministic sampling of provider-normalized router inputs."""

    def __init__(
        self,
        target: RouterQuantizationCalibrationTarget,
        *,
        max_tokens: int,
        max_tokens_per_observation: int,
    ) -> None:
        self.target = target.validate()
        self.max_tokens = int(max_tokens)
        self.max_tokens_per_observation = int(max_tokens_per_observation)
        if self.max_tokens <= 0 or self.max_tokens_per_observation <= 0:
            raise ValueError("Router calibration token limits must be positive.")
        self._features: list[Any] = []
        self._additive: list[Any] = []
        self._token_count = 0
        self._saw_additive = False
        self._handle: Any | None = None

    def _record(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        remaining = self.max_tokens - self._token_count
        if remaining <= 0:
            return
        batch = self.target.capture_batch(args, kwargs).validate(
            input_features=self.target.input_features,
            num_experts=self.target.num_experts,
        )
        features = torch.as_tensor(batch.features).detach()
        take = min(
            int(features.shape[0]),
            remaining,
            self.max_tokens_per_observation,
        )
        if take <= 0:
            return
        if take == int(features.shape[0]):
            indices = torch.arange(take, device=features.device)
        else:
            indices = torch.linspace(
                0,
                int(features.shape[0]) - 1,
                steps=take,
                device=features.device,
            ).round().to(torch.long)
        sampled_features = features.index_select(0, indices)
        self._features.append(
            sampled_features.to(device="cpu", dtype=torch.float32).contiguous()
        )
        if batch.additive_logits is None:
            sampled_additive = torch.zeros(
                (take, int(self.target.num_experts)),
                dtype=torch.float32,
            )
        else:
            self._saw_additive = True
            sampled_additive = (
                torch.as_tensor(batch.additive_logits)
                .detach()
                .index_select(0, indices)
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
        self._additive.append(sampled_additive)
        self._token_count += take

    def attach(self) -> None:
        if self._handle is not None:
            raise RuntimeError("Router input accumulator is already attached.")
        self._handle = self.target.observation_module.register_forward_pre_hook(
            self._record,
            with_kwargs=True,
        )

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def batch(self) -> RouterLinearCalibrationBatch:
        if not self._features:
            raise ValueError(
                f"Router target {self.target.name!r} observed no calibration inputs."
            )
        features = torch.cat(self._features, dim=0)
        additive = torch.cat(self._additive, dim=0) if self._saw_additive else None
        return RouterLinearCalibrationBatch(
            features=features,
            additive_logits=additive,
        ).validate(
            input_features=self.target.input_features,
            num_experts=self.target.num_experts,
        )


def kl_top_divergence(
    reference_logits: Any,
    candidate_logits: Any,
    *,
    top_k: int,
    relaxation: float = 0.0,
) -> Any:
    """Mean reference-to-candidate KL contribution on reference top-m experts."""
    reference = torch.as_tensor(reference_logits, dtype=torch.float32)
    candidate = torch.as_tensor(
        candidate_logits,
        dtype=torch.float32,
        device=reference.device,
    )
    if (
        reference.ndim != 2
        or reference.shape != candidate.shape
        or int(reference.shape[0]) <= 0
    ):
        raise ValueError("KL-Top logits must be aligned [tokens, experts].")
    experts = int(reference.shape[1])
    k = int(top_k)
    alpha = float(relaxation)
    if k <= 0 or k >= experts or not (0.0 <= alpha <= 1.0):
        raise ValueError("KL-Top requires 0 < top_k < experts and 0 <= alpha <= 1.")
    selected = k + int(alpha * (experts - k))
    indices = torch.topk(reference, k=selected, dim=1, sorted=False).indices
    reference_log_prob = F.log_softmax(reference, dim=1)
    candidate_log_prob = F.log_softmax(candidate, dim=1)
    probability = reference_log_prob.exp().gather(1, indices)
    return (
        probability
        * (
            reference_log_prob.gather(1, indices)
            - candidate_log_prob.gather(1, indices)
        )
    ).sum(dim=1).mean()


def router_alignment_objective(
    reference_logits: Any,
    candidate_logits: Any,
    *,
    top_k: int,
    relaxation: float = 0.0,
) -> tuple[Any, Any, Any]:
    """EAQuant logit-MSE plus KL-Top objective and its two components."""
    reference = torch.as_tensor(reference_logits, dtype=torch.float32)
    candidate = torch.as_tensor(
        candidate_logits,
        dtype=torch.float32,
        device=reference.device,
    )
    if reference.shape != candidate.shape:
        raise ValueError("Router alignment logits must have equal shapes.")
    mse = (reference - candidate).square().sum(dim=1).mean()
    divergence = kl_top_divergence(
        reference,
        candidate,
        top_k=top_k,
        relaxation=relaxation,
    )
    return mse + divergence, mse, divergence


@dataclass(frozen=True)
class RouterQuantizationCalibration:
    scale: Any
    clipping_ratio: Any
    token_count: int
    baseline_objective: float
    calibrated_objective: float
    baseline_logit_mse: float
    calibrated_logit_mse: float
    baseline_kl_top: float
    calibrated_kl_top: float

    def validate(self, *, num_experts: int) -> "RouterQuantizationCalibration":
        scale = torch.as_tensor(self.scale)
        ratio = torch.as_tensor(self.clipping_ratio)
        if scale.shape != (int(num_experts),) or ratio.shape != scale.shape:
            raise ValueError("Router calibration scales must match the expert axis.")
        if (
            not bool(torch.isfinite(scale).all().item())
            or not bool(torch.isfinite(ratio).all().item())
            or bool(torch.any(scale <= 0).item())
            or bool(torch.any(ratio <= 0).item())
            or bool(torch.any(ratio > 1).item())
        ):
            raise ValueError("Router calibration scales or ratios are invalid.")
        metrics = (
            self.baseline_objective,
            self.calibrated_objective,
            self.baseline_logit_mse,
            self.calibrated_logit_mse,
            self.baseline_kl_top,
            self.calibrated_kl_top,
        )
        if int(self.token_count) <= 0 or not all(
            float(value) == float(value) and abs(float(value)) != float("inf")
            for value in metrics
        ):
            raise ValueError("Router calibration metrics are invalid.")
        if float(self.calibrated_objective) > float(self.baseline_objective) + 1e-6:
            raise ValueError("Calibrated router objective regressed from absmax INT8.")
        return self


def _dequantized_rows(
    weight: Any,
    ratios: Any,
) -> tuple[Any, Any]:
    maximum = weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    scale = maximum[:, None] * ratios[None, :] / 127.0
    quantized = torch.round(weight[:, None, :] / scale[:, :, None]).clamp(
        -127,
        127,
    )
    return quantized * scale[:, :, None], scale


def calibrate_symmetric_int8_router(
    weight: Any,
    batch: RouterLinearCalibrationBatch,
    *,
    top_k: int,
    relaxation: float = 0.0,
    minimum_clipping_ratio: float = 0.35,
    grid_size: int = 101,
    coordinate_sweeps: int = 1,
) -> RouterQuantizationCalibration:
    """Coordinate-search per-output INT8 scales against the EAQuant objective."""
    source = torch.as_tensor(weight, dtype=torch.float32)
    if source.ndim != 2 or not bool(torch.isfinite(source).all().item()):
        raise ValueError("Router weight must be a finite rank-2 tensor.")
    batch.validate(
        input_features=int(source.shape[1]),
        num_experts=int(source.shape[0]),
    )
    minimum = float(minimum_clipping_ratio)
    points = int(grid_size)
    sweeps = int(coordinate_sweeps)
    if not (0.0 < minimum <= 1.0) or points < 2 or sweeps <= 0:
        raise ValueError("Router clipping search settings are invalid.")
    device = source.device
    features = torch.as_tensor(
        batch.features,
        dtype=torch.float32,
        device=device,
    )
    additive = (
        torch.zeros(
            (int(features.shape[0]), int(source.shape[0])),
            dtype=torch.float32,
            device=device,
        )
        if batch.additive_logits is None
        else torch.as_tensor(
            batch.additive_logits,
            dtype=torch.float32,
            device=device,
        )
    )
    ratios = torch.linspace(minimum, 1.0, steps=points, device=device)
    rows, candidate_scales = _dequantized_rows(source, ratios)
    reference_logits = F.linear(features, source) + additive
    current_weight = rows[:, -1, :].clone()
    current_logits = F.linear(features, current_weight) + additive
    baseline, baseline_mse, baseline_kl = router_alignment_objective(
        reference_logits,
        current_logits,
        top_k=int(top_k),
        relaxation=float(relaxation),
    )
    selected_ratio = torch.ones(int(source.shape[0]), device=device)
    selected_scale = candidate_scales[:, -1].clone()

    reference_log_prob = F.log_softmax(reference_logits, dim=1)
    selected_count = int(top_k) + int(
        float(relaxation) * (int(source.shape[0]) - int(top_k))
    )
    top_indices = torch.topk(
        reference_logits,
        k=selected_count,
        dim=1,
        sorted=False,
    ).indices
    top_probability = reference_log_prob.exp().gather(1, top_indices)
    reference_entropy_term = (
        top_probability * reference_log_prob.gather(1, top_indices)
    ).sum(dim=1)
    top_mass = top_probability.sum(dim=1)

    for _ in range(sweeps):
        for expert in range(int(source.shape[0])):
            candidate_row_logits = (
                features @ rows[expert].transpose(0, 1)
                + additive[:, expert : expert + 1]
            )
            current_row = current_logits[:, expert]
            current_error = (reference_logits - current_logits).square().sum(dim=1)
            row_error = (reference_logits[:, expert] - current_row).square()
            candidate_mse = (
                current_error[:, None]
                - row_error[:, None]
                + (
                    reference_logits[:, expert : expert + 1]
                    - candidate_row_logits
                ).square()
            ).mean(dim=0)

            without_row = current_logits.clone()
            without_row[:, expert] = -torch.inf
            other_logsumexp = torch.logsumexp(without_row, dim=1)
            candidate_logsumexp = torch.logaddexp(
                other_logsumexp[:, None],
                candidate_row_logits,
            )
            is_current_expert = top_indices == expert
            current_selected_logits = current_logits.gather(1, top_indices)
            other_weighted_logits = (
                top_probability
                * torch.where(
                    is_current_expert,
                    torch.zeros_like(current_selected_logits),
                    current_selected_logits,
                )
            ).sum(dim=1)
            expert_probability = (
                top_probability
                * is_current_expert.to(dtype=top_probability.dtype)
            ).sum(dim=1)
            candidate_kl = (
                reference_entropy_term[:, None]
                - other_weighted_logits[:, None]
                - expert_probability[:, None] * candidate_row_logits
                + top_mass[:, None] * candidate_logsumexp
            ).mean(dim=0)
            best = int(torch.argmin(candidate_mse + candidate_kl).item())
            current_weight[expert] = rows[expert, best]
            current_logits[:, expert] = candidate_row_logits[:, best]
            selected_ratio[expert] = ratios[best]
            selected_scale[expert] = candidate_scales[expert, best]

    calibrated, calibrated_mse, calibrated_kl = router_alignment_objective(
        reference_logits,
        current_logits,
        top_k=int(top_k),
        relaxation=float(relaxation),
    )
    result = RouterQuantizationCalibration(
        scale=selected_scale.detach().cpu(),
        clipping_ratio=selected_ratio.detach().cpu(),
        token_count=int(features.shape[0]),
        baseline_objective=float(baseline.detach().cpu().item()),
        calibrated_objective=float(calibrated.detach().cpu().item()),
        baseline_logit_mse=float(baseline_mse.detach().cpu().item()),
        calibrated_logit_mse=float(calibrated_mse.detach().cpu().item()),
        baseline_kl_top=float(baseline_kl.detach().cpu().item()),
        calibrated_kl_top=float(calibrated_kl.detach().cpu().item()),
    )
    return result.validate(num_experts=int(source.shape[0]))


@dataclass(frozen=True)
class RouterQuantizationCalibrationArtifact:
    modules: Mapping[str, RouterQuantizationCalibration]
    topology: Mapping[str, Mapping[str, int]]
    dataset_snapshot_id: str
    model_snapshot_id: str
    config_snapshot_id: str
    source_router_fingerprint: str
    relaxation: float
    minimum_clipping_ratio: float
    grid_size: int
    coordinate_sweeps: int

    def validate(self) -> "RouterQuantizationCalibrationArtifact":
        if not self.modules or set(self.modules) != set(self.topology):
            raise ValueError("Router calibration artifact inventory is inconsistent.")
        for key in (
            self.dataset_snapshot_id,
            self.model_snapshot_id,
            self.config_snapshot_id,
            self.source_router_fingerprint,
        ):
            if not str(key).strip():
                raise ValueError("Router calibration artifact lineage is incomplete.")
        if not (0.0 <= float(self.relaxation) <= 1.0):
            raise ValueError("Router calibration relaxation is invalid.")
        if (
            not (0.0 < float(self.minimum_clipping_ratio) <= 1.0)
            or int(self.grid_size) < 2
            or int(self.coordinate_sweeps) <= 0
        ):
            raise ValueError("Router calibration search metadata is invalid.")
        for name, result in self.modules.items():
            spec = self.topology[name]
            if (
                int(spec.get("num_experts", 0)) < 2
                or int(spec.get("input_features", 0)) <= 0
                or int(spec.get("top_k", 0)) <= 0
            ):
                raise ValueError(f"Router calibration topology {name!r} is invalid.")
            result.validate(num_experts=int(spec["num_experts"]))
        return self


def source_router_tensors(
    targets: Mapping[str, RouterQuantizationCalibrationTarget],
) -> dict[str, Any]:
    if not targets:
        raise ValueError("Router quantization calibration requires targets.")
    tensors: dict[str, Any] = {}
    for name, target in sorted(targets.items()):
        target.validate()
        if str(name) != target.name or name in tensors:
            raise ValueError("Router quantization target names must match and be unique.")
        tensors[str(name)] = (
            torch.as_tensor(target.read_weight()).detach().cpu().contiguous().clone()
        )
    return tensors


def save_router_quantization_calibration(
    path: str | Path,
    artifact: RouterQuantizationCalibrationArtifact,
) -> None:
    from safetensors.torch import save_file

    artifact.validate()
    tensors: dict[str, Any] = {}
    modules: list[dict[str, Any]] = []
    for index, name in enumerate(sorted(artifact.modules)):
        result = artifact.modules[name]
        prefix = f"module_{index:04d}"
        tensors[f"{prefix}.scale"] = torch.as_tensor(
            result.scale,
            dtype=torch.float32,
        ).contiguous()
        tensors[f"{prefix}.clipping_ratio"] = torch.as_tensor(
            result.clipping_ratio,
            dtype=torch.float32,
        ).contiguous()
        modules.append(
            {
                "name": name,
                "scale": f"{prefix}.scale",
                "clipping_ratio": f"{prefix}.clipping_ratio",
                "topology": dict(artifact.topology[name]),
                "metrics": {
                    key: getattr(result, key)
                    for key in (
                        "token_count",
                        "baseline_objective",
                        "calibrated_objective",
                        "baseline_logit_mse",
                        "calibrated_logit_mse",
                        "baseline_kl_top",
                        "calibrated_kl_top",
                    )
                },
            }
        )
    manifest = {
        "schema": ROUTER_QUANTIZATION_CALIBRATION_SCHEMA,
        "schema_version": ROUTER_QUANTIZATION_CALIBRATION_VERSION,
        "lineage": {
            "dataset_snapshot_id": artifact.dataset_snapshot_id,
            "model_snapshot_id": artifact.model_snapshot_id,
            "config_snapshot_id": artifact.config_snapshot_id,
            "source_router_fingerprint": artifact.source_router_fingerprint,
        },
        "objective": {
            "name": "logit_mse_plus_kl_top",
            "relaxation": float(artifact.relaxation),
        },
        "search": {
            "minimum_clipping_ratio": float(artifact.minimum_clipping_ratio),
            "grid_size": int(artifact.grid_size),
            "coordinate_sweeps": int(artifact.coordinate_sweeps),
        },
        "modules": modules,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            ROUTER_QUANTIZATION_CALIBRATION_METADATA_KEY: json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
            )
        },
    )


def load_router_quantization_calibration(
    path: str | Path,
) -> RouterQuantizationCalibrationArtifact:
    from safetensors import safe_open
    from safetensors.torch import load_file

    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        raw = (handle.metadata() or {}).get(
            ROUTER_QUANTIZATION_CALIBRATION_METADATA_KEY
        )
    if not raw:
        raise ValueError("Router quantization calibration metadata is missing.")
    manifest = json.loads(raw)
    if (
        manifest.get("schema") != ROUTER_QUANTIZATION_CALIBRATION_SCHEMA
        or int(manifest.get("schema_version", 0))
        != ROUTER_QUANTIZATION_CALIBRATION_VERSION
    ):
        raise ValueError("Unsupported router quantization calibration schema.")
    tensors = load_file(str(source), device="cpu")
    modules: dict[str, RouterQuantizationCalibration] = {}
    topology: dict[str, dict[str, int]] = {}
    for spec in manifest.get("modules", ()):
        name = str(spec.get("name", ""))
        if not name or name in modules:
            raise ValueError("Router calibration module names are invalid.")
        metrics = dict(spec.get("metrics", {}))
        modules[name] = RouterQuantizationCalibration(
            scale=tensors[str(spec["scale"])],
            clipping_ratio=tensors[str(spec["clipping_ratio"])],
            **metrics,
        )
        topology[name] = {
            key: int(value)
            for key, value in dict(spec.get("topology", {})).items()
        }
    lineage = dict(manifest.get("lineage", {}))
    objective = dict(manifest.get("objective", {}))
    search = dict(manifest.get("search", {}))
    return RouterQuantizationCalibrationArtifact(
        modules=modules,
        topology=topology,
        dataset_snapshot_id=str(lineage.get("dataset_snapshot_id", "")),
        model_snapshot_id=str(lineage.get("model_snapshot_id", "")),
        config_snapshot_id=str(lineage.get("config_snapshot_id", "")),
        source_router_fingerprint=str(
            lineage.get("source_router_fingerprint", "")
        ),
        relaxation=float(objective.get("relaxation", -1.0)),
        minimum_clipping_ratio=float(
            search.get("minimum_clipping_ratio", -1.0)
        ),
        grid_size=int(search.get("grid_size", 0)),
        coordinate_sweeps=int(search.get("coordinate_sweeps", 0)),
    ).validate()


def apply_router_quantization_calibration(
    targets: Mapping[str, RouterQuantizationCalibrationTarget],
    artifact: RouterQuantizationCalibrationArtifact,
) -> None:
    """Validate exact source lineage, then install calibrated INT8 scales."""
    artifact.validate()
    current = source_router_tensors(targets)
    if set(current) != set(artifact.modules):
        raise ValueError("Router calibration target inventory changed.")
    if router_tensor_fingerprint(current) != artifact.source_router_fingerprint:
        raise ValueError("Router calibration belongs to different source weights.")
    for name, target in targets.items():
        expected = {
            "num_experts": int(target.num_experts),
            "input_features": int(target.input_features),
            "top_k": int(target.top_k),
        }
        if dict(artifact.topology[name]) != expected:
            raise ValueError(f"Router calibration topology {name!r} changed.")
    for name, target in targets.items():
        target.install_int8_scale(artifact.modules[name].scale)


__all__ = [
    "ROUTER_QUANTIZATION_CALIBRATION_METADATA_KEY",
    "ROUTER_QUANTIZATION_CALIBRATION_SCHEMA",
    "ROUTER_QUANTIZATION_CALIBRATION_VERSION",
    "RouterInputAccumulator",
    "RouterLinearCalibrationBatch",
    "RouterQuantizationCalibration",
    "RouterQuantizationCalibrationArtifact",
    "RouterQuantizationCalibrationTarget",
    "apply_router_quantization_calibration",
    "calibrate_symmetric_int8_router",
    "kl_top_divergence",
    "load_router_quantization_calibration",
    "router_alignment_objective",
    "save_router_quantization_calibration",
    "source_router_tensors",
]
