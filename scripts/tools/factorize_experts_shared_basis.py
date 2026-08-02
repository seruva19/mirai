"""Create a schema-v4 shared-basis compressed packed-state artifact."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirai.core.builtins import register_builtin_components
from mirai.core.models.compressed_weights import load_compressed_weights_packed_tensors
from mirai.core.models.compressed_weights import packed_artifact_fingerprint
from mirai.core.models.compressed_weights import read_compressed_weights_packed_state_manifest
from mirai.core.models.compressed_weights import save_compressed_weights_packed_tensors
from mirai.core.models.compressed_weights.factorization.shared_basis_artifact import (
    factorize_packed_state_shared_basis,
)
from mirai.core.lineage import bind_snapshot_component
from mirai.core.lineage import normalize_snapshot_component
from mirai.core.lineage import snapshot_descriptor_for_path
from mirai.core.moe.calibration.prototypes import load_prototype_calibration_evidence
from mirai.core.moe.calibration.quantization import (
    load_quantization_calibration_evidence,
)
from mirai.core.moe.calibration.whitening import ExpertWhiteningEvidence
from mirai.core.moe.calibration.whitening import load_expert_whitening_evidence
from mirai.core.training.runtime.cli import load_runtime_config
from mirai.core.training.runtime.gpu_lease import acquire_gpu_lease
from mirai.core.training.runtime.gpu_lease import resolve_lease_lock_path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required: {exc}")


def _reference_experts(
    evidence_path: str | Path | None,
    manifest: dict[str, Any],
    config: Any,
    *,
    packed_state: str | Path,
) -> dict[str, int]:
    if not evidence_path:
        return {}
    dataset_snapshot = snapshot_descriptor_for_path(config.dataset.path)
    model_snapshot = bind_snapshot_component(
        snapshot_descriptor_for_path(config.model.path),
        component=normalize_snapshot_component(
            getattr(config.model.params, "denoiser_subfolder", "transformer")
            or "transformer"
        ),
        component_label="denoiser_subfolder",
    )
    evidence, _lineage = load_prototype_calibration_evidence(
        evidence_path,
        expected_dataset_snapshot_id=dataset_snapshot.snapshot_id,
        expected_model_snapshot_id=model_snapshot.snapshot_id,
        expected_packed_artifact_fingerprint=packed_artifact_fingerprint(
            packed_state
        ),
    )
    modules = manifest.get("modules", {})
    references: dict[str, int] = {}
    for module_name, item in evidence.items():
        logical = int(torch.as_tensor(item.selected_count).argmax().item())
        spec = modules.get(module_name, {})
        aliases = spec.get("logical_to_physical") if isinstance(spec, dict) else None
        references[module_name] = int(aliases[logical]) if aliases is not None else logical
    return references


def _affinity_weights(
    evidence_path: str | Path | None,
    manifest: dict[str, Any],
    config: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not evidence_path:
        return {}, {}
    dataset_snapshot = snapshot_descriptor_for_path(config.dataset.path)
    model_snapshot = bind_snapshot_component(
        snapshot_descriptor_for_path(config.model.path),
        component=normalize_snapshot_component(
            getattr(config.model.params, "denoiser_subfolder", "transformer")
            or "transformer"
        ),
        component_label="denoiser_subfolder",
    )
    evidence, lineage = load_quantization_calibration_evidence(
        evidence_path,
        expected_dataset_snapshot_id=dataset_snapshot.snapshot_id,
        expected_model_snapshot_id=model_snapshot.snapshot_id,
    )
    modules = manifest.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("Packed-state manifest must contain a modules object.")
    grouped = {
        str(name): spec
        for name, spec in modules.items()
        if isinstance(spec, dict) and str(spec.get("kind")) == "grouped_experts"
    }
    if set(evidence) != set(grouped):
        raise ValueError(
            "Quantization calibration modules do not exactly match packed grouped modules."
        )
    weights: dict[str, Any] = {}
    for name, item in evidence.items():
        expected_experts = int(grouped[name].get("num_experts", 0))
        if item.num_experts != expected_experts:
            raise ValueError(
                f"Quantization calibration module {name!r} has {item.num_experts} "
                f"experts; packed state expects {expected_experts}."
            )
        weights[name] = item.affinity_reconstruction_weights(
            require_full_coverage=True
        )
    return weights, lineage


def _whitening_evidence(
    evidence_path: str | Path | None,
    manifest: dict[str, Any],
    config: Any,
    *,
    packed_state: str | Path,
) -> tuple[dict[str, ExpertWhiteningEvidence], dict[str, str]]:
    gate = str(
        getattr(config.model.params, "expert_factorization_calibration", "off")
    ).strip().lower()
    if not evidence_path:
        if gate == "whitened":
            raise ValueError(
                "Whitened factorization requires --whitening-evidence."
            )
        return {}, {}
    if gate != "whitened":
        raise ValueError(
            "--whitening-evidence requires "
            "model.params.expert_factorization_calibration='whitened'."
        )
    dataset_snapshot = snapshot_descriptor_for_path(config.dataset.path)
    model_snapshot = bind_snapshot_component(
        snapshot_descriptor_for_path(config.model.path),
        component=normalize_snapshot_component(
            getattr(config.model.params, "denoiser_subfolder", "transformer")
            or "transformer"
        ),
        component_label="denoiser_subfolder",
    )
    evidence, lineage = load_expert_whitening_evidence(
        evidence_path,
        expected_dataset_snapshot_id=dataset_snapshot.snapshot_id,
        expected_model_snapshot_id=model_snapshot.snapshot_id,
        expected_packed_artifact_fingerprint=packed_artifact_fingerprint(
            packed_state
        ),
    )
    modules = manifest.get("modules")
    if not isinstance(modules, dict):
        raise ValueError("Packed-state manifest must contain a modules object.")
    grouped = {
        str(name)
        for name, spec in modules.items()
        if isinstance(spec, dict) and str(spec.get("kind")) == "grouped_experts"
    }
    if set(evidence) != grouped:
        raise ValueError(
            "Whitening evidence modules do not exactly match packed grouped modules."
        )
    return evidence, lineage


def factorize_expert_artifact(
    *,
    config_path: str | Path,
    packed_state: str | Path,
    output: str | Path,
    rank: int,
    axis: str,
    factor_dtype: str,
    device: str,
    calibration_evidence: str | Path | None = None,
    quantization_calibration_evidence: str | Path | None = None,
    whitening_evidence: str | Path | None = None,
    whitening_regularization: float = 1e-6,
    max_reconstruction_error: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config, _notes = load_runtime_config(
        config_path,
        entrypoint="shared-basis-factorization",
    )
    gate = str(config.model.params.expert_weight_compression).strip().lower()
    if gate != "shared_basis":
        raise ValueError(
            "Shared-basis factorization requires "
            "model.params.expert_weight_compression='shared_basis'."
        )
    source_path = Path(packed_state)
    output_path = Path(output)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Shared-basis output must differ from its source artifact.")
    if output_path.exists() and not overwrite:
        raise ValueError(f"Shared-basis output already exists: {output_path}.")
    resolved_device = str(device).strip().lower()
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Shared-basis factorization requested CUDA, but CUDA is unavailable.")
    if resolved_device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")

    manifest = read_compressed_weights_packed_state_manifest(source_path)
    tensors = load_compressed_weights_packed_tensors(source_path)
    references = _reference_experts(
        calibration_evidence,
        manifest,
        config,
        packed_state=source_path,
    )
    affinity_weights, calibration_lineage = _affinity_weights(
        quantization_calibration_evidence,
        manifest,
        config,
    )
    covariance_evidence, whitening_lineage = _whitening_evidence(
        whitening_evidence,
        manifest,
        config,
        packed_state=source_path,
    )
    lease = (
        acquire_gpu_lease(
            lock_path=str(resolve_lease_lock_path(ROOT)),
            timeout_seconds=float(os.environ.get("MIRAI_GPU_LEASE_TIMEOUT", "0")),
        )
        if resolved_device == "cuda"
        else nullcontext()
    )
    with lease:
        converted, converted_manifest, report = factorize_packed_state_shared_basis(
            tensors,
            manifest,
            rank=int(rank),
            device=torch.device(resolved_device),
            axis=axis,
            factor_dtype=factor_dtype,
            reference_experts=references,
            expert_weights=affinity_weights,
            calibration_lineage=calibration_lineage,
            whitening_evidence=covariance_evidence,
            whitening_lineage=whitening_lineage,
            whitening_regularization=float(whitening_regularization),
        )
        if max_reconstruction_error is not None:
            maximum = max(
                float(value)
                for module in report["modules"].values()
                for value in module["relative_frobenius_error"].values()
            )
            if maximum > float(max_reconstruction_error):
                raise ValueError(
                    f"Shared-basis reconstruction error {maximum:.6f} exceeds "
                    f"the configured maximum {float(max_reconstruction_error):.6f}."
                )
        written = save_compressed_weights_packed_tensors(
            output_path,
            converted,
            converted_manifest,
            metadata={
                "model_type": str(config.model.type),
                "model_variant": str(config.model.params.variant),
                "expert_weight_compression": "shared_basis",
            },
        )
    report.update(
        {
            "status": "ok",
            "output": str(written),
            "device": resolved_device,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--packed-state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--axis", choices=("auto", "right", "left"), default="auto")
    parser.add_argument(
        "--factor-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--calibration-evidence")
    parser.add_argument("--quantization-calibration-evidence")
    parser.add_argument("--whitening-evidence")
    parser.add_argument("--whitening-regularization", type=float, default=1e-6)
    parser.add_argument("--max-reconstruction-error", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    register_builtin_components()
    report = factorize_expert_artifact(
        config_path=args.config,
        packed_state=args.packed_state,
        output=args.output,
        rank=args.rank,
        axis=args.axis,
        factor_dtype=args.factor_dtype,
        device=args.device,
        calibration_evidence=args.calibration_evidence,
        quantization_calibration_evidence=args.quantization_calibration_evidence,
        whitening_evidence=args.whitening_evidence,
        whitening_regularization=args.whitening_regularization,
        max_reconstruction_error=args.max_reconstruction_error,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
