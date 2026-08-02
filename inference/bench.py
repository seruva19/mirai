"""Arm-based inference benchmark harness.

This model-agnostic measurement layer owns benchmark policy only and never
touches model code. Every run is a
fresh ``scripts/infer.py`` subprocess invoked with ``--timings-out`` (the
per-phase load/denoise/decode measurement seam), so timings and peak VRAM
are attributed to an isolated CUDA context per run.

Input: an *arm-matrix* TOML file. Globals plus a ``[[arm]]`` array::

    warmup_runs = 1            # discarded warmup runs per arm (default 1)
    runs_per_seed = 2          # measured runs per seed
    out = "outputs/bench"      # output directory (timings.jsonl lands here)
    gpu_index = 0              # target GPU for the precheck (default 0)
    precheck_foreign_vram_mib = 2048   # foreign-VRAM abort threshold (default 2048)

    [[arm]]
    name = "base"
    config = "path/to/config.toml"
    checkpoint = "path/to/ckpt.pt"   # or adapter = "path/to/adapter"
    prompt = "a red cube spinning"   # or prompt_file = "path/to/prompt.txt"
    negative_prompt_file = ""        # optional
    seeds = [0, 1, 2]
    steps = 20
    cfg_scale = 3.0
    frames = 33
    height = 480
    width = 832
    scheduler = "euler"

Output: one JSON line per run appended to ``<out>/timings.jsonl`` with keys
``arm, seed, run_idx, warmup, load_s, denoise_s, decode_s, wall_s,
peak_vram_mb, precheck_ok, config, git_rev``. Runtime variants belong in each
arm's referenced config. An arm whose GPU precheck
fails is *aborted* (fail-fast, not a warning) and records a single row with
``precheck_ok = false`` and null timings.

GPU precheck (fail-fast, per arm): ``torch.cuda.mem_get_info()`` plus
``nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid`` joined against
``--query-gpu=index,uuid`` (compute-apps carry no index field, only gpu_uuid).
Foreign VRAM on the target GPU above the threshold aborts the arm. Prechecks are
skipped when CUDA is unavailable, and the result records that no device precheck
was performed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - py310 fallback
    import tomli as tomllib  # type: ignore[no-redef]

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

DEFAULT_PRECHECK_FOREIGN_VRAM_MIB = 2048  # 2 GiB


# --------------------------------------------------------------------------- #
# CUDA and nvidia-smi access are isolated behind module-level query functions.
# --------------------------------------------------------------------------- #
def _cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _mem_get_info(gpu_index: int) -> tuple[int, int] | None:
    """(free, total) bytes for ``gpu_index``, or None if unavailable."""
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return tuple(torch.cuda.mem_get_info(gpu_index))  # type: ignore[return-value]
    except Exception:
        return None


def _run_nvidia_smi(args: list[str]) -> str | None:
    """Run ``nvidia-smi`` with ``args`` and return stdout, or None on failure.

    Both precheck queries use this command boundary.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _query_compute_apps() -> list[tuple[int, float, str]]:
    """[(pid, used_memory_mib, gpu_uuid), ...] for every compute process."""
    out = _run_nvidia_smi(
        [
            "--query-compute-apps=pid,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    rows: list[tuple[int, float, str]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            pid = int(parts[0])
            mem = float(parts[1])
        except ValueError:
            continue
        rows.append((pid, mem, parts[2]))
    return rows


def _query_gpu_uuids() -> dict[int, str]:
    """{index: uuid} — needed to attribute compute-apps (which lack an index)."""
    out = _run_nvidia_smi(["--query-gpu=index,uuid", "--format=csv,noheader"])
    if not out:
        return {}
    mapping: dict[int, str] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            mapping[int(parts[0])] = parts[1]
        except ValueError:
            continue
    return mapping


def gpu_precheck(
    gpu_index: int,
    threshold_mib: float,
    own_pid: int | None = None,
) -> dict[str, Any]:
    """Fail-fast foreign-VRAM precheck for one target GPU.

    Returns a detail dict with ``precheck_ok``. When CUDA is unavailable the
    precheck is skipped and reported as such. Foreign VRAM
    is the sum of compute-apps' ``used_memory`` on the target GPU's UUID,
    excluding our own process. If the UUID map is unavailable, every compute-app
    is counted (conservative). If nvidia-smi is entirely unavailable, we fall
    back to ``mem_get_info`` (total-free) as the foreign estimate.
    """
    if own_pid is None:
        own_pid = os.getpid()
    if not _cuda_available():
        return {
            "precheck_ok": True,
            "skipped": True,
            "reason": "cuda-unavailable",
            "gpu_index": gpu_index,
            "threshold_mib": threshold_mib,
            "foreign_vram_mib": 0.0,
        }

    mem = _mem_get_info(gpu_index)
    free_total = None if mem is None else {"free_bytes": mem[0], "total_bytes": mem[1]}

    uuids = _query_gpu_uuids()
    target_uuid = uuids.get(gpu_index)
    apps = _query_compute_apps()

    if not apps and _run_nvidia_smi(["--query-gpu=index", "--format=csv,noheader"]) is None:
        # nvidia-smi unusable: fall back to the coarse mem_get_info estimate.
        if mem is not None:
            foreign_mib = (mem[1] - mem[0]) / (1024 * 1024)
            return {
                "precheck_ok": foreign_mib <= threshold_mib,
                "skipped": False,
                "reason": "mem_get_info-fallback",
                "gpu_index": gpu_index,
                "threshold_mib": threshold_mib,
                "foreign_vram_mib": foreign_mib,
                "target_uuid": None,
                "mem_get_info": free_total,
            }

    foreign_mib = 0.0
    foreign_pids: list[int] = []
    for pid, used_mib, uuid in apps:
        if pid == own_pid:
            continue
        if target_uuid is not None and uuid != target_uuid:
            continue
        foreign_mib += used_mib
        foreign_pids.append(pid)

    return {
        "precheck_ok": foreign_mib <= threshold_mib,
        "skipped": False,
        "gpu_index": gpu_index,
        "threshold_mib": threshold_mib,
        "foreign_vram_mib": foreign_mib,
        "foreign_pids": foreign_pids,
        "target_uuid": target_uuid,
        "mem_get_info": free_total,
    }


# --------------------------------------------------------------------------- #
# Matrix parsing / provenance
# --------------------------------------------------------------------------- #
def _git_rev() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def load_matrix(matrix_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse an arm-matrix TOML into (globals, arms)."""
    with matrix_path.open("rb") as f:
        data = tomllib.load(f)
    arms = data.get("arm", [])
    if not isinstance(arms, list) or not arms:
        raise ValueError(f"{matrix_path}: matrix must define at least one [[arm]].")
    globals_ = {k: v for k, v in data.items() if k != "arm"}
    return globals_, arms


def _resolve_prompt(arm: dict[str, Any], base_dir: Path) -> str:
    if arm.get("prompt_file"):
        return Path(_resolve_path(str(arm["prompt_file"]), base_dir)).read_text(
            encoding="utf-8"
        ).strip()
    prompt = arm.get("prompt")
    if not prompt:
        raise ValueError(f"arm {arm.get('name')!r}: provide 'prompt' or 'prompt_file'.")
    return str(prompt)


def _resolve_negative(arm: dict[str, Any], base_dir: Path) -> str:
    if arm.get("negative_prompt_file"):
        return Path(_resolve_path(str(arm["negative_prompt_file"]), base_dir)).read_text(
            encoding="utf-8"
        ).strip()
    return str(arm.get("negative_prompt", ""))


def _resolve_path(value: str, base_dir: Path) -> str:
    """Resolve a matrix-relative path against the matrix file's directory."""
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


# --------------------------------------------------------------------------- #
# Run execution
# --------------------------------------------------------------------------- #
def _run_infer_subprocess(
    *,
    arm: dict[str, Any],
    prompt: str,
    negative_prompt: str,
    seed: int,
    out_path: Path,
    timings_path: Path,
    base_dir: Path,
) -> dict[str, Any]:
    """Invoke scripts/infer.py once; return {wall_s, timings, returncode, error}."""
    config_path = _resolve_path(str(arm["config"]), base_dir)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "infer.py"),
        "--config",
        config_path,
        "--prompt",
        prompt,
        "--negative-prompt",
        negative_prompt,
        "--seed",
        str(seed),
        "--steps",
        str(int(arm.get("steps", 20))),
        "--cfg-scale",
        str(float(arm.get("cfg_scale", 5.0))),
        "--frames",
        str(int(arm.get("frames", 17))),
        "--height",
        str(int(arm.get("height", 480))),
        "--width",
        str(int(arm.get("width", 832))),
        "--scheduler",
        str(arm.get("scheduler", "euler")),
        "--out",
        str(out_path),
        "--timings-out",
        str(timings_path),
    ]
    if arm.get("checkpoint"):
        cmd += ["--checkpoint", _resolve_path(str(arm["checkpoint"]), base_dir)]
    if arm.get("adapter"):
        cmd += ["--adapter", _resolve_path(str(arm["adapter"]), base_dir)]
    if arm.get("fps"):
        cmd += ["--fps", str(int(arm["fps"]))]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    wall_s = time.perf_counter() - t0

    timings: dict[str, Any] | None = None
    if timings_path.exists():
        try:
            timings = json.loads(timings_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            timings = None

    error = None
    if proc.returncode != 0:
        error = (proc.stderr or proc.stdout or "").strip()[-2000:]
    return {
        "wall_s": wall_s,
        "timings": timings,
        "returncode": proc.returncode,
        "error": error,
    }


def _run_row(
    *,
    arm_name: str,
    config: str,
    git_rev: str,
    seed: int | None,
    run_idx: int | None,
    warmup: bool,
    precheck_ok: bool,
    result: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timings = (result or {}).get("timings") or {}
    row: dict[str, Any] = {
        "arm": arm_name,
        "seed": seed,
        "run_idx": run_idx,
        "warmup": warmup,
        "load_s": timings.get("load_s"),
        "denoise_s": timings.get("denoise_s"),
        "decode_s": timings.get("decode_s"),
        "wall_s": (result or {}).get("wall_s"),
        "peak_vram_mb": timings.get("peak_vram_mb"),
        "precheck_ok": precheck_ok,
        "config": config,
        "git_rev": git_rev,
    }
    if result is not None and result.get("returncode") not in (None, 0):
        row["returncode"] = result.get("returncode")
        row["error"] = result.get("error")
    if extra:
        row.update(extra)
    return row


def run_matrix(
    matrix_path: str | Path,
    out_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run every arm in the matrix, append rows to ``<out>/timings.jsonl``.

    Returns the list of rows written (also useful for tests).
    """
    matrix_path = Path(matrix_path).resolve()
    base_dir = matrix_path.parent
    globals_, arms = load_matrix(matrix_path)

    warmup_runs = int(globals_.get("warmup_runs", 1))
    runs_per_seed = int(globals_.get("runs_per_seed", 1))
    gpu_index = int(globals_.get("gpu_index", 0))
    threshold_mib = float(
        globals_.get("precheck_foreign_vram_mib", DEFAULT_PRECHECK_FOREIGN_VRAM_MIB)
    )

    if out_dir is not None:
        out_root = Path(out_dir)
    elif globals_.get("out"):
        out_root = Path(_resolve_path(str(globals_["out"]), base_dir))
    else:
        out_root = ROOT / "outputs" / "bench"
    out_root.mkdir(parents=True, exist_ok=True)
    timings_jsonl = out_root / "timings.jsonl"

    git_rev = _git_rev()
    rows: list[dict[str, Any]] = []

    with timings_jsonl.open("a", encoding="utf-8") as sink:

        def _emit(row: dict[str, Any]) -> None:
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            rows.append(row)

        for arm in arms:
            arm_name = str(arm.get("name", "arm"))
            config = _resolve_path(str(arm["config"]), base_dir)
            precheck = gpu_precheck(gpu_index, threshold_mib)
            if not precheck["precheck_ok"]:
                _emit(
                    _run_row(
                        arm_name=arm_name,
                        config=config,
                        git_rev=git_rev,
                        seed=None,
                        run_idx=None,
                        warmup=False,
                        precheck_ok=False,
                        result=None,
                        extra={"precheck": precheck},
                    )
                )
                print(
                    f"[bench] arm {arm_name!r} ABORTED: foreign VRAM "
                    f"{precheck.get('foreign_vram_mib')} MiB > "
                    f"{threshold_mib} MiB on GPU {gpu_index}.",
                    file=sys.stderr,
                )
                continue

            prompt = _resolve_prompt(arm, base_dir)
            negative_prompt = _resolve_negative(arm, base_dir)
            seeds = [
                int(seed)
                for seed in (arm.get("seeds") or [int(arm.get("seed", 0))])
            ]
            arm_dir = out_root / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)

            for warmup_index in range(warmup_runs):
                warm_seed = seeds[0]
                out_path = arm_dir / f"warmup{warmup_index}.mp4"
                timings_path = arm_dir / f"warmup{warmup_index}.timings.json"
                result = _run_infer_subprocess(
                    arm=arm,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=warm_seed,
                    out_path=out_path,
                    timings_path=timings_path,
                    base_dir=base_dir,
                )
                _emit(
                    _run_row(
                        arm_name=arm_name,
                        config=config,
                        git_rev=git_rev,
                        seed=warm_seed,
                        run_idx=warmup_index,
                        warmup=True,
                        precheck_ok=True,
                        result=result,
                    )
                )

            for seed in seeds:
                for run_idx in range(runs_per_seed):
                    out_path = arm_dir / f"seed{seed}_run{run_idx}.mp4"
                    timings_path = arm_dir / f"seed{seed}_run{run_idx}.timings.json"
                    result = _run_infer_subprocess(
                        arm=arm,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        out_path=out_path,
                        timings_path=timings_path,
                        base_dir=base_dir,
                    )
                    _emit(
                        _run_row(
                            arm_name=arm_name,
                            config=config,
                            git_rev=git_rev,
                            seed=seed,
                            run_idx=run_idx,
                            warmup=False,
                            precheck_ok=True,
                            result=result,
                        )
                    )

    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", required=True, help="Path to the arm-matrix TOML.")
    p.add_argument(
        "--out",
        default="",
        help="Output directory override (else matrix 'out' or outputs/bench).",
    )
    args = p.parse_args()
    rows = run_matrix(args.matrix, out_dir=(args.out or None))
    aborted = sum(1 for r in rows if not r["precheck_ok"])
    measured = sum(1 for r in rows if r["precheck_ok"] and not r["warmup"])
    print(
        json.dumps(
            {
                "rows": len(rows),
                "measured_runs": measured,
                "aborted_arms": aborted,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
