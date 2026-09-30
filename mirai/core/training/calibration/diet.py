"""Bounded provider-driven DIET capture over native paired-CFG inference."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from mirai.core.models.providers import get_model_family_provider
from mirai.core.moe.calibration.diet import (
    DietCalibrationEvidence,
    DietFamilyPostprocess,
    DietGroupTopology,
    DietLayerCapture,
    DietLineage,
    DietPairedCapture,
    derive_diet_signatures,
    save_diet_evidence,
)


@dataclasses.dataclass(frozen=True)
class DietCalibrationRequest:
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    cfg_scale: float
    frames: int
    height: int
    width: int
    scheduler: str = "euler"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DietCalibrationRequest":
        def exact_int(name: str) -> int:
            raw = value[name]
            parsed = int(raw)
            if isinstance(raw, bool) or float(raw) != float(parsed):
                raise ValueError(f"DIET manifest {name} must be an exact integer.")
            return parsed

        try:
            request = cls(
                prompt=str(value["prompt"]),
                negative_prompt=str(value["negative_prompt"]),
                seed=exact_int("seed"),
                steps=exact_int("steps"),
                cfg_scale=float(value["cfg_scale"]),
                frames=exact_int("frames"),
                height=exact_int("height"),
                width=exact_int("width"),
                scheduler=str(value.get("scheduler", "euler")),
            )
        except KeyError as exc:
            raise ValueError(f"DIET manifest entry is missing {exc.args[0]!r}.") from exc
        if (not request.prompt or request.seed < 0 or request.steps <= 0
                or not math.isfinite(request.cfg_scale) or request.cfg_scale <= 1.0):
            raise ValueError("DIET capture requires a prompt, positive steps, and cfg_scale > 1.")
        if min(request.frames, request.height, request.width) <= 0 or not request.scheduler:
            raise ValueError("DIET capture geometry and scheduler must be valid.")
        return request


@dataclasses.dataclass(frozen=True)
class DietCalibrationRunReport:
    output_path: str
    requests: int
    samples_per_layer: int
    modules: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"status": "ok", "output": self.output_path, "requests": self.requests,
                "samples_per_layer": self.samples_per_layer, "modules": self.modules}


def load_diet_calibration_manifest(path: str | Path) -> tuple[list[DietCalibrationRequest], str]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        payload = json.loads(raw)
        entries = payload.get("samples") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("DIET calibration manifest must contain a non-empty sample list.")
    if any(not isinstance(item, Mapping) for item in entries):
        raise ValueError("Every DIET calibration manifest entry must be an object.")
    requests = [DietCalibrationRequest.from_mapping(item) for item in entries]
    manifest_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return requests, manifest_id


def fingerprint_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _DietObserver:
    def __init__(self, target: Any, *, sample_budget: int, seed: int) -> None:
        self.target = target
        self.sample_budget = int(sample_budget)
        self.seed = int(seed)
        self.pending_indices: torch.Tensor | None = None
        self.pending_capture: DietLayerCapture | None = None
        self.pending_positions: torch.Tensor | None = None
        self.signature_sum: torch.Tensor | None = None
        self.sample_count = 0
        self.guidance_scale: float | None = None
        self.max_rows_per_pair: int | None = None
        self.pair_index = 0

    @property
    def is_enabled(self) -> bool:
        return self.sample_count < self.sample_budget or self.pending_capture is not None

    def sample_token_indices(self, tokens: Any, *, layer_name: str = "") -> torch.Tensor:
        del layer_name
        flat_rows = int(torch.as_tensor(tokens).reshape(-1, torch.as_tensor(tokens).shape[-1]).shape[0])
        if self.pending_indices is not None:
            return self.pending_indices.to(device=torch.as_tensor(tokens).device)
        remaining = self.sample_budget - self.sample_count
        pair_cap = flat_rows if self.max_rows_per_pair is None else self.max_rows_per_pair
        count = min(max(remaining, 0), flat_rows, max(int(pair_cap), 0))
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=torch.as_tensor(tokens).device)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.pair_index)
        indices = torch.randperm(flat_rows, generator=generator)[:count].sort().values
        self.pending_indices = indices
        return indices.to(device=torch.as_tensor(tokens).device)

    def capture_layer(self, **raw: Any) -> None:
        indices = torch.as_tensor(raw["sampled_token_indices"]).detach().cpu().to(torch.long)
        if indices.numel() == 0:
            return
        topology = DietGroupTopology(
            num_experts=int(raw.get("num_experts", self.target.num_experts)),
            top_k=int(raw.get("top_k", self.target.top_k)),
            num_groups=int(raw.get("n_group", self.target.n_group)),
            topk_group=int(raw.get("topk_group", self.target.topk_group)),
            group_score_topk=int(raw.get("group_score_topk", self.target.group_score_topk)),
            normalize_topk_prob=bool(raw.get("norm_topk_prob", self.target.norm_topk_prob)),
            route_scale=float(raw.get("route_scale", self.target.route_scale)),
        )
        shared = raw.get("shared_output")
        postprocess = None if shared is None and raw.get("rms_weight") is None and raw.get("residual") is None else DietFamilyPostprocess(
            shared_output=None if shared is None else torch.as_tensor(shared).detach().cpu(),
            rms_weight=None if raw.get("rms_weight") is None else torch.as_tensor(raw["rms_weight"]).detach().cpu(),
            rms_epsilon=float(raw.get("epsilon") or 1e-6),
            residual_gate=None if raw.get("residual") is None else torch.as_tensor(raw["residual"]).detach().cpu(),
        )
        raw_scores = torch.as_tensor(raw["router_scores"]).detach().cpu()
        sampled_scores = (
            raw_scores
            if int(raw_scores.shape[0]) == int(indices.numel())
            else raw_scores.index_select(0, indices)
        )
        capture = DietLayerCapture(
            expert_outputs=torch.as_tensor(raw["expert_outputs"]).detach().cpu(),
            router_scores=sampled_scores,
            choice_bias=torch.as_tensor(raw["choice_bias"]).detach().cpu(),
            topology=topology,
            postprocess=postprocess,
        ).validate()
        if self.pending_capture is None:
            self.pending_capture, self.pending_positions = capture, indices
            return
        if not torch.equal(indices, self.pending_positions):
            raise RuntimeError("DIET conditional and unconditional token rows are not aligned.")
        if self.guidance_scale is None:
            raise RuntimeError("DIET observer guidance scale was not set for the inference request.")
        paired = DietPairedCapture(
            conditional=self.pending_capture,
            unconditional=capture,
            pair_ids=torch.full((indices.numel(),), self.pair_index, dtype=torch.int64),
            token_positions=indices,
        )
        evidence = derive_diet_signatures(paired, guidance_scale=self.guidance_scale)
        contribution = torch.as_tensor(evidence.signatures, dtype=torch.float64) * evidence.sample_count
        self.signature_sum = contribution if self.signature_sum is None else self.signature_sum + contribution
        self.sample_count += evidence.sample_count
        self.pair_index += 1
        self.pending_capture = self.pending_positions = self.pending_indices = None

    def evidence(self) -> DietCalibrationEvidence:
        if self.pending_capture is not None:
            raise RuntimeError("DIET capture ended between paired CFG branches.")
        if self.signature_sum is None or self.sample_count <= 0 or self.guidance_scale is None:
            raise ValueError(f"DIET target {self.target.name!r} captured no paired rows.")
        topology = DietGroupTopology(
            self.target.num_experts, self.target.top_k, self.target.n_group,
            self.target.topk_group, self.target.group_score_topk,
            bool(self.target.norm_topk_prob), float(self.target.route_scale),
        )
        return DietCalibrationEvidence(
            signatures=(self.signature_sum / self.sample_count).to(torch.float32),
            sample_count=self.sample_count, topology=topology, cfg_mode="paired_cfg",
            guidance_scale=self.guidance_scale,
        ).validate()


def run_diet_calibration_session(
    session: Any, *, requests: Sequence[DietCalibrationRequest], output_path: str | Path,
    samples_per_layer: int, source_packed_state: str | Path,
    calibration_manifest_id: str, config_snapshot_id: str, overwrite: bool = False,
    seed: int = 0,
) -> DietCalibrationRunReport:
    config = session.cfg
    if str(getattr(session, "checkpoint", "")).strip() or str(
        getattr(session, "adapter", "")
    ).strip() or bool(getattr(session, "merge", False)):
        raise ValueError("DIET calibration requires an unmodified packed base session.")
    if str(config.model.params.expert_pruning).strip().lower() != "prune" or str(config.model.params.expert_pruning_criterion).strip().lower() != "diet":
        raise ValueError("DIET calibration requires expert_pruning='prune' and expert_pruning_criterion='diet'.")
    if (int(samples_per_layer) <= 0 or not requests or int(seed) < 0
            or int(samples_per_layer) < len(requests)):
        raise ValueError("DIET calibration requires requests and a positive sample budget.")
    for request in requests:
        if (not request.prompt or request.seed < 0 or request.steps <= 0
                or min(request.frames, request.height, request.width) <= 0
                or not request.scheduler):
            raise ValueError("DIET calibration request fields are invalid.")
    guidance_scales = {float(request.cfg_scale) for request in requests}
    if len(guidance_scales) != 1 or not all(
        math.isfinite(value) and value > 1.0 for value in guidance_scales
    ):
        raise ValueError("One DIET evidence artifact requires one CFG guidance scale.")
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise ValueError(f"DIET calibration output already exists: {output}.")
    source = Path(source_packed_state)
    if not source.is_file():
        raise ValueError(f"DIET source packed state does not exist: {source}.")
    if output.resolve() == source.resolve():
        raise ValueError("DIET evidence output cannot overwrite its packed source state.")
    configured_value = str(
        getattr(config.memory, "frozen_weight_packed_state_path", "")
    ).strip()
    if not configured_value or Path(configured_value).resolve() != source.resolve():
        raise ValueError(
            "DIET source packed state must exactly match "
            "memory.frozen_weight_packed_state_path loaded by the inference session."
        )
    source_fingerprint = fingerprint_file(source)
    provider = get_model_family_provider(str(config.model.type))
    build = None if provider is None else getattr(provider, "build_diet_calibration_targets", None)
    if not callable(build):
        raise ValueError(f"Model provider {config.model.type!r} does not support DIET calibration.")
    raw_targets = build(session.pipeline)
    targets = tuple(raw_targets.values()) if isinstance(raw_targets, Mapping) else tuple(raw_targets)
    if not targets or len({str(t.name) for t in targets}) != len(targets):
        raise ValueError("DIET provider targets must be non-empty and uniquely named.")
    for target in targets:
        validate = getattr(target, "validate", None)
        if callable(validate):
            validate()
    observers = {str(t.name): _DietObserver(t, sample_budget=0, seed=seed) for t in targets}
    previous = {str(t.name): t.host.get_diet_capture_observer() for t in targets}
    if any(value is not None for value in previous.values()):
        raise ValueError("DIET calibration cannot replace an active capture observer.")
    training_model = session.pipeline.get_training_model()
    was_training = None if training_model is None else bool(training_model.training)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        for target in targets:
            target.host.set_diet_capture_observer(observers[str(target.name)])
        with tempfile.TemporaryDirectory(prefix="mirai-diet-capture-") as temp_dir:
            for index, request in enumerate(requests):
                cumulative_budget = ((index + 1) * int(samples_per_layer)) // len(requests)
                prior_budget = (index * int(samples_per_layer)) // len(requests)
                request_quota = cumulative_budget - prior_budget
                for observer in observers.values():
                    observer.guidance_scale = request.cfg_scale
                    observer.sample_budget = cumulative_budget
                    observer.max_rows_per_pair = max(
                        1, math.ceil(request_quota / int(request.steps))
                    )
                session.generate(
                    prompt=request.prompt, negative_prompt=request.negative_prompt,
                    seed=request.seed, steps=request.steps, cfg_scale=request.cfg_scale,
                    frames=request.frames, height=request.height, width=request.width,
                    scheduler=request.scheduler, cfg_mode="sequential",
                    out_path=Path(temp_dir) / f"capture-{index:04d}.pt",
                    allow_latent_output_only=True,
                )
                short_request = {
                    name: observer.sample_count
                    for name, observer in observers.items()
                    if observer.sample_count < cumulative_budget
                }
                if short_request:
                    raise ValueError(
                        "DIET request did not reach its cumulative sample quota "
                        f"{cumulative_budget}: {short_request}."
                    )
    finally:
        for target in reversed(targets):
            target.host.set_diet_capture_observer(previous[str(target.name)])
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if training_model is not None and was_training is not None:
            training_model.train(was_training)
    short = {name: observer.sample_count for name, observer in observers.items()
             if observer.sample_count < int(samples_per_layer)}
    if short:
        raise ValueError(f"DIET calibration exhausted its manifest before reaching the sample budget: {short}.")
    evidence = {name: observer.evidence() for name, observer in observers.items()}
    if fingerprint_file(source) != source_fingerprint:
        raise RuntimeError("DIET packed source state changed during calibration.")
    capture_payload = {name: {"samples": item.sample_count, "signature": hashlib.sha256(torch.as_tensor(item.signatures).numpy().tobytes()).hexdigest()} for name, item in sorted(evidence.items())}
    capture_fingerprint = hashlib.sha256(json.dumps(capture_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    save_diet_evidence(output, evidence, lineage=DietLineage(
        dataset_snapshot_id=calibration_manifest_id,
        model_snapshot_id=source_fingerprint,
        config_snapshot_id=config_snapshot_id,
        calibration_manifest_id=calibration_manifest_id,
        packed_state_sha256=source_fingerprint,
        capture_fingerprint=capture_fingerprint,
    ))
    return DietCalibrationRunReport(str(output), len(requests), int(samples_per_layer), {name: item.sample_count for name, item in evidence.items()})


__all__ = ["DietCalibrationRequest", "DietCalibrationRunReport", "fingerprint_file",
           "load_diet_calibration_manifest", "run_diet_calibration_session"]
