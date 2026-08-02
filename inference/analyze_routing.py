"""Offline analyzer for inference routing traces.

Consumes a routing trace captured by the LingBot inference pipeline
(`model.params.inference_routing_telemetry=true`, written through
`scripts/infer.py --routing-trace-out`) and reports cross-step routing stability.

Metrics (per adjacent-step pair, matched on same branch + same layer):

  * hard churn %       -- fraction of tokens whose top-k *set* changed t -> t+1
  * Jaccard@k per token -- |set_t & set_{t+1}| / |set_t | set_{t+1}|, mean + p10
  * bucketed by diffusion-timestep quartile [0, .25, .5, .75, 1]

plus, per step:

  * unique-expert count per layer (single branch) -- coverage
  * cond-union-uncond union size per layer         -- the true PCIe working set

CFG parity: with cfg_scale > 1 the denoise loop issues two forwards per step,
cond first then uncond, so forward_idx parity tags the branch (even=cond,
odd=uncond) and step = forward_idx // 2. With cfg_scale <= 1 there is one
forward per step (branch 0, step = forward_idx).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Diffusion-timestep quartile bucket edges (normalized to [0, 1]).
_BUCKET_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)


def derive_step_branch(forward_idx: int, cfg_scale: float) -> tuple[int, int]:
    """Map a forward index to (step, branch) given the CFG scale.

    cfg_scale > 1 -> two forwards per step (cond=even parity, uncond=odd).
    cfg_scale <= 1 -> one forward per step, always branch 0.
    """
    if cfg_scale is not None and float(cfg_scale) > 1.0:
        return forward_idx // 2, forward_idx % 2
    return forward_idx, 0


def _as_index_array(indices: Any) -> np.ndarray:
    """Coerce a [N, K] index snapshot (torch / numpy / list) to a 2-D int array."""
    if hasattr(indices, "detach"):  # torch tensor
        indices = indices.detach().cpu().numpy()
    arr = np.asarray(indices)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D [N, K] index array, got shape {arr.shape}")
    return arr.astype(np.int64)


def _membership(idx: np.ndarray, num_experts: int) -> np.ndarray:
    """One-hot [N, num_experts] boolean set-membership for a [N, K] index array."""
    n, _k = idx.shape
    mem = np.zeros((n, num_experts), dtype=bool)
    rows = np.repeat(np.arange(n), idx.shape[1])
    mem[rows, idx.reshape(-1)] = True
    return mem


def _pair_metrics(a: np.ndarray, b: np.ndarray, num_experts: int) -> dict[str, float]:
    """Per-token churn / Jaccard between two [N, K] top-k snapshots."""
    mem_a = _membership(a, num_experts)
    mem_b = _membership(b, num_experts)
    inter = np.logical_and(mem_a, mem_b).sum(axis=1)
    union = np.logical_or(mem_a, mem_b).sum(axis=1)
    union_safe = np.maximum(union, 1)
    jaccard = inter / union_safe
    # A token "churned" iff its set changed, i.e. the union is larger than the
    # intersection (equivalently Jaccard < 1).
    changed = union > inter
    return {
        "hard_churn_pct": float(changed.mean() * 100.0),
        "jaccard_mean": float(jaccard.mean()),
        "jaccard_p10": float(np.percentile(jaccard, 10)),
        "num_tokens": int(a.shape[0]),
    }


def _normalize_timesteps(timesteps: list[float], steps: list[int]) -> dict[Any, float]:
    """Map each observed timestep value to [0, 1]; fall back to step fraction."""
    finite = [t for t in timesteps if t is not None and math.isfinite(t)]
    if finite:
        lo, hi = min(finite), max(finite)
        span = hi - lo
        out: dict[Any, float] = {}
        for t in timesteps:
            if t is None or not math.isfinite(t):
                out[t] = 0.5
            else:
                out[t] = (t - lo) / span if span > 0 else 0.5
        return out
    # No usable timesteps: signal the caller to normalize by step index instead.
    _ = steps
    return {}


def _bucket_label(idx: int) -> str:
    return f"{_BUCKET_EDGES[idx]:.2f}-{_BUCKET_EDGES[idx + 1]:.2f}"


def _bucket_of(norm_t: float) -> int:
    for i in range(4):
        hi = _BUCKET_EDGES[i + 1]
        if norm_t < hi or (i == 3 and norm_t <= 1.0 + 1e-9):
            return i
    return 3


def analyze_routing_trace(
    entries: list[dict[str, Any]],
    *,
    cfg_scale: float,
    num_experts: int | None = None,
) -> dict[str, Any]:
    """Compute routing-stability metrics from a list of trace entries.

    Each entry is a dict with keys: forward_idx, layer_idx, timestep,
    top_indices ([N, K] int array/tensor), and optionally num_experts.
    """
    if not entries:
        return {"num_entries": 0, "churn": {}, "coverage": {}, "working_set": {}}

    # Resolve expert cardinality (max declared, else max index + 1).
    declared = [int(e["num_experts"]) for e in entries if e.get("num_experts")]
    max_index = 0
    normed_entries: list[dict[str, Any]] = []
    for e in entries:
        idx = _as_index_array(e["top_indices"])
        max_index = max(max_index, int(idx.max()))
        step, branch = derive_step_branch(int(e["forward_idx"]), cfg_scale)
        normed_entries.append(
            {
                "step": step,
                "branch": branch,
                "layer": int(e["layer_idx"]),
                "timestep": (
                    float(e["timestep"]) if e.get("timestep") is not None else None
                ),
                "idx": idx,
            }
        )
    resolved_experts = int(
        num_experts if num_experts is not None
        else (max(declared) if declared else max_index + 1)
    )
    resolved_experts = max(resolved_experts, max_index + 1)

    timesteps = [e["timestep"] for e in normed_entries]
    steps_all = [e["step"] for e in normed_entries]
    norm_map = _normalize_timesteps(timesteps, steps_all)
    step_max = max(steps_all) if steps_all else 0

    def _norm_t(entry: dict[str, Any]) -> float:
        t = entry["timestep"]
        if norm_map:
            return norm_map.get(t, 0.5)
        return (entry["step"] / step_max) if step_max > 0 else 0.0

    # Index entries by (branch, layer) -> {step: entry}.
    grouped: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for e in normed_entries:
        grouped.setdefault((e["branch"], e["layer"]), {})[e["step"]] = e

    # --- Adjacent-step-pair churn / Jaccard ---
    pairs: list[dict[str, Any]] = []
    per_layer_pairs: dict[int, list[dict[str, Any]]] = {}
    for (branch, layer), by_step in grouped.items():
        ordered = sorted(by_step)
        for s_a, s_b in zip(ordered, ordered[1:]):
            if s_b != s_a + 1:
                continue  # only truly adjacent steps
            e_a, e_b = by_step[s_a], by_step[s_b]
            m = _pair_metrics(e_a["idx"], e_b["idx"], resolved_experts)
            m.update(
                branch=branch, layer=layer, step=s_a, norm_t=_norm_t(e_a)
            )
            pairs.append(m)
            per_layer_pairs.setdefault(layer, []).append(m)

    churn = {
        "overall": _agg_pairs(pairs),
        "by_timestep_bucket": _agg_by_bucket(pairs),
        "by_layer": {
            str(layer): _agg_pairs(ps) for layer, ps in sorted(per_layer_pairs.items())
        },
    }

    # --- Per-step unique-expert coverage per layer (branch 0) ---
    coverage_by_layer: dict[str, Any] = {}
    layers = sorted({e["layer"] for e in normed_entries})
    for layer in layers:
        uniques = [
            int(np.unique(by_step[s]["idx"]).size)
            for (br, lyr), by_step in grouped.items()
            if lyr == layer and br == 0
            for s in by_step
        ]
        if uniques:
            coverage_by_layer[str(layer)] = {
                "unique_experts_mean": float(np.mean(uniques)),
                "unique_experts_min": int(min(uniques)),
                "unique_experts_max": int(max(uniques)),
                "num_steps": len(uniques),
            }

    # --- cond-union-uncond size per layer (true PCIe working set) ---
    has_uncond = any(e["branch"] == 1 for e in normed_entries)
    working_set: dict[str, Any] = {"available": bool(has_uncond), "by_layer": {}}
    if has_uncond:
        for layer in layers:
            cond = grouped.get((0, layer), {})
            uncond = grouped.get((1, layer), {})
            union_sizes = []
            for s in sorted(set(cond) & set(uncond)):
                merged = np.concatenate(
                    [cond[s]["idx"].reshape(-1), uncond[s]["idx"].reshape(-1)]
                )
                union_sizes.append(int(np.unique(merged).size))
            if union_sizes:
                working_set["by_layer"][str(layer)] = {
                    "union_size_mean": float(np.mean(union_sizes)),
                    "union_size_min": int(min(union_sizes)),
                    "union_size_max": int(max(union_sizes)),
                    "num_steps": len(union_sizes),
                }

    return {
        "num_entries": len(entries),
        "num_experts": resolved_experts,
        "cfg_scale": float(cfg_scale),
        "layers": layers,
        "branches": sorted({e["branch"] for e in normed_entries}),
        "num_steps": step_max + 1,
        "churn": churn,
        "coverage": {"by_layer": coverage_by_layer},
        "working_set": working_set,
    }


def _agg_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {"num_pairs": 0}
    return {
        "num_pairs": len(pairs),
        "hard_churn_pct_mean": float(np.mean([p["hard_churn_pct"] for p in pairs])),
        "jaccard_mean": float(np.mean([p["jaccard_mean"] for p in pairs])),
        "jaccard_p10_mean": float(np.mean([p["jaccard_p10"] for p in pairs])),
    }


def _agg_by_bucket(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for p in pairs:
        label = _bucket_label(_bucket_of(p["norm_t"]))
        buckets.setdefault(label, []).append(p)
    return {label: _agg_pairs(ps) for label, ps in buckets.items()}


# --------------------------------------------------------------------------- #
# Trace loading (written by scripts/infer.py --routing-trace-out)
# --------------------------------------------------------------------------- #


def load_trace(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a routing trace (.pt or .npz) into (entries, metadata)."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        top = data["top_indices"]
        forward_idx = data["forward_idx"]
        layer_idx = data["layer_idx"]
        timestep = data["timestep"]
        num_experts = data["num_experts"] if "num_experts" in data else None
        metadata = (
            json.loads(str(data["metadata"])) if "metadata" in data else {}
        )
    else:
        import torch  # local import: analysis of .npz traces needs no torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        top = payload["top_indices"]
        if hasattr(top, "numpy"):
            top = top.numpy()
        forward_idx = np.asarray(payload["forward_idx"])
        layer_idx = np.asarray(payload["layer_idx"])
        timestep = np.asarray(payload["timestep"])
        num_experts = payload.get("num_experts")
        if num_experts is not None and hasattr(num_experts, "numpy"):
            num_experts = num_experts.numpy()
        metadata = dict(payload.get("metadata", {}))

    # Ragged traces (per-forward token counts differ, e.g. a stride filter or a
    # cfg<=1 run mixed with cfg>1) cannot be coerced into one ndarray — keep the
    # per-entry arrays as-is in that case.
    if not isinstance(top, (list, tuple)):
        top = np.asarray(top)
    else:
        as_np = [np.asarray(t) for t in top]
        top = (
            np.asarray(as_np)
            if len({tuple(t.shape) for t in as_np}) == 1
            else as_np
        )
    entries: list[dict[str, Any]] = []
    for i in range(len(forward_idx)):
        entries.append(
            {
                "forward_idx": int(forward_idx[i]),
                "layer_idx": int(layer_idx[i]),
                "timestep": float(timestep[i]),
                "num_experts": (
                    int(num_experts[i]) if num_experts is not None else None
                ),
                "top_indices": top[i],
            }
        )
    return entries, metadata


def format_report(report: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Dense human-readable text report."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("INFERENCE ROUTING STABILITY REPORT")
    lines.append("=" * 72)
    lines.append(
        f"entries={report.get('num_entries', 0)}  "
        f"experts={report.get('num_experts', '?')}  "
        f"cfg_scale={report.get('cfg_scale', '?')}  "
        f"steps={report.get('num_steps', '?')}  "
        f"layers={len(report.get('layers', []))}  "
        f"branches={report.get('branches', [])}"
    )
    if metadata:
        lines.append(f"metadata: {json.dumps(metadata, sort_keys=True)}")
    lines.append("")

    overall = report.get("churn", {}).get("overall", {})
    lines.append("-- Cross-step routing churn (adjacent steps, same branch+layer) --")
    if overall.get("num_pairs"):
        lines.append(
            f"  overall: pairs={overall['num_pairs']}  "
            f"hard_churn={overall['hard_churn_pct_mean']:.2f}%  "
            f"Jaccard@k mean={overall['jaccard_mean']:.4f}  "
            f"p10={overall['jaccard_p10_mean']:.4f}"
        )
    else:
        lines.append("  (no adjacent step pairs found)")

    buckets = report.get("churn", {}).get("by_timestep_bucket", {})
    if buckets:
        lines.append("")
        lines.append("  by diffusion-timestep quartile:")
        for label in sorted(buckets):
            b = buckets[label]
            if b.get("num_pairs"):
                lines.append(
                    f"    t[{label}]: pairs={b['num_pairs']:4d}  "
                    f"churn={b['hard_churn_pct_mean']:6.2f}%  "
                    f"Jaccard={b['jaccard_mean']:.4f}  p10={b['jaccard_p10_mean']:.4f}"
                )

    by_layer = report.get("churn", {}).get("by_layer", {})
    if by_layer:
        lines.append("")
        lines.append("  by layer:")
        for layer in sorted(by_layer, key=int):
            b = by_layer[layer]
            if b.get("num_pairs"):
                lines.append(
                    f"    layer {layer:>3}: churn={b['hard_churn_pct_mean']:6.2f}%  "
                    f"Jaccard={b['jaccard_mean']:.4f}  p10={b['jaccard_p10_mean']:.4f}"
                )

    cov = report.get("coverage", {}).get("by_layer", {})
    if cov:
        lines.append("")
        lines.append("-- Per-step unique-expert coverage (branch 0) --")
        for layer in sorted(cov, key=int):
            c = cov[layer]
            lines.append(
                f"    layer {layer:>3}: unique mean={c['unique_experts_mean']:.2f}  "
                f"min={c['unique_experts_min']}  max={c['unique_experts_max']}  "
                f"steps={c['num_steps']}"
            )

    ws = report.get("working_set", {})
    lines.append("")
    lines.append("-- cond-union-uncond working set (true PCIe stream) --")
    if ws.get("available") and ws.get("by_layer"):
        for layer in sorted(ws["by_layer"], key=int):
            w = ws["by_layer"][layer]
            lines.append(
                f"    layer {layer:>3}: union mean={w['union_size_mean']:.2f}  "
                f"min={w['union_size_min']}  max={w['union_size_max']}  "
                f"steps={w['num_steps']}"
            )
    else:
        lines.append("    (no uncond branch: cfg<=1 or single-branch trace)")
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="Path to routing trace (.pt or .npz)")
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="CFG scale used at generation (overrides trace metadata). "
        ">1 -> two forwards/step tagged cond/uncond by parity.",
    )
    parser.add_argument(
        "--json-out", default="", help="Write the report as JSON to this path"
    )
    args = parser.parse_args(argv)

    entries, metadata = load_trace(args.trace)
    cfg_scale = args.cfg_scale
    if cfg_scale is None:
        cfg_scale = float(metadata.get("cfg_scale", 1.0))
    report = analyze_routing_trace(entries, cfg_scale=cfg_scale)
    print(format_report(report, metadata))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"metadata": metadata, "report": report}, indent=2),
            encoding="utf-8",
        )
        print(f"\n[analyze_routing] wrote JSON report -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
