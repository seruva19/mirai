"""Simple shared GPU lease lock."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Iterator
from contextlib import contextmanager
import json
import hashlib
import tempfile
import uuid


class GpuLeaseError(RuntimeError):
    """Raised when GPU lease cannot be acquired."""


_LEGACY_LEASE_FILENAME = ".mirai_gpu_lease.lock"
_GLOBAL_SCOPES = frozenset({"", "global", "legacy", "shared"})
_PER_GPU_SCOPES = frozenset({"per-gpu", "per_gpu", "pergpu", "gpu"})


def _sanitize_device_id(raw: str) -> str:
    """Keep filename-safe alphanumerics from one CUDA_VISIBLE_DEVICES entry."""

    return "".join(ch for ch in raw.strip() if ch.isalnum())


def gpu_lease_device_token(cuda_visible_devices: str | None) -> str:
    """Normalize a ``CUDA_VISIBLE_DEVICES`` value into a lease-filename token.

    * unset / empty / all-blank -> ``"gpuall"`` (identity of an any-GPU run)
    * ``"2"`` -> ``"gpu2"``
    * ``"2,3"`` / ``"3,2"`` -> ``"gpu2-3"`` (numerically sorted, joined)

    The token is deterministic (device order does not change it) so two runs
    that were handed the same physical device(s) land on the same lock file.
    """

    if cuda_visible_devices is None:
        return "gpuall"
    raw = cuda_visible_devices.strip()
    if raw == "":
        return "gpuall"
    ids = [tok for tok in (_sanitize_device_id(p) for p in raw.split(",")) if tok]
    if not ids:
        return "gpuall"
    if all(tok.isdigit() for tok in ids):
        ids = sorted(ids, key=int)
    else:
        ids = sorted(ids)
    return "gpu" + "-".join(ids)


def resolve_lease_lock_path(
    root: str | Path,
    *,
    env: "os._Environ[str] | dict[str, str] | None" = None,
) -> Path:
    """Resolve the effective GPU-lease lockfile path.

    Precedence (highest first):

    1. ``MIRAI_GPU_LEASE_PATH`` -- explicit override, used verbatim.
    2. ``MIRAI_GPU_LEASE_SCOPE`` -- ``global`` (default, legacy single lockfile)
       or ``per-gpu`` (GPU-aware default filename derived from
       ``CUDA_VISIBLE_DEVICES``).

    Scope semantics (see AGENTS.md / CONFIG_REFERENCE.md):

    * ``global`` uses ``ROOT/.mirai_gpu_lease.lock`` -- every
      run contends on one lockfile, so mutual exclusion is airtight regardless
      of which GPU a run uses.
    * ``per-gpu`` gives each pinned device set its own lockfile
      (``.mirai_gpu_lease.gpu2.lock``), so two runs pinned to *different* GPUs
      never serialize. An unpinned (any-GPU / ``gpuall``) run has no isolable
      identity, so it degrades to the shared global lockfile -- the safest
      available fallback (it still excludes other unpinned runs). ``per-gpu``
      therefore assumes concurrent runs pin ``CUDA_VISIBLE_DEVICES``; that is
      the intended parallel-training deployment.
    """

    environ = os.environ if env is None else env
    override = str(environ.get("MIRAI_GPU_LEASE_PATH", "") or "").strip()
    if override:
        return Path(override)
    scope = str(environ.get("MIRAI_GPU_LEASE_SCOPE", "") or "").strip().lower()
    root_path = Path(root)
    if scope in _GLOBAL_SCOPES:
        return root_path / _LEGACY_LEASE_FILENAME
    if scope in _PER_GPU_SCOPES:
        token = gpu_lease_device_token(environ.get("CUDA_VISIBLE_DEVICES"))
        if token == "gpuall":
            return root_path / _LEGACY_LEASE_FILENAME
        return root_path / f".mirai_gpu_lease.{token}.lock"
    raise GpuLeaseError(
        f"Unknown MIRAI_GPU_LEASE_SCOPE={scope!r}; expected 'global' or 'per-gpu'."
    )


def _epoch_path(lock_path: Path) -> Path:
    return lock_path.with_name(lock_path.name + ".epoch")


def _next_lease_epoch(lock_path: Path) -> int:
    path = _epoch_path(lock_path)
    current = 0
    if path.exists():
        try:
            current = int(path.read_text(encoding="utf-8").strip())
        except Exception:
            current = 0
    nxt = current + 1
    path.write_text(str(nxt) + "\n", encoding="utf-8")
    return nxt


def _pid_is_running(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).is_running())
    except ModuleNotFoundError:
        pass
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        return True
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                int(pid),
            )
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False


def _read_ticket_pid(ticket_path: Path) -> int | None:
    try:
        raw = ticket_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _reap_stale_queue_tickets(
    queue_dir: Path,
    *,
    stale_timeout_seconds: float,
) -> int:
    if stale_timeout_seconds <= 0.0 or not queue_dir.exists():
        return 0
    now = time.time()
    removed = 0
    for ticket in queue_dir.glob("*.ticket"):
        owner_pid = _read_ticket_pid(ticket)
        # Dead-owner tickets are always safe to reap immediately; this avoids
        # post-crash queue head stalls.
        if owner_pid is not None and not _pid_is_running(owner_pid):
            try:
                ticket.unlink()
                removed += 1
            except FileNotFoundError:
                continue
            continue
        try:
            age = now - float(ticket.stat().st_mtime)
        except FileNotFoundError:
            continue
        if age < stale_timeout_seconds:
            continue
        if owner_pid is not None and _pid_is_running(owner_pid):
            continue
        try:
            ticket.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def _read_owner_pid(lock_path: Path) -> int | None:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and "pid" in payload:
            return int(payload["pid"])
    except Exception:
        pass
    try:
        return int(raw)
    except Exception:
        return None


def read_active_lease_epoch(lock_path: str | Path) -> int | None:
    p = Path(lock_path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "epoch" in payload:
            return int(payload["epoch"])
    except Exception:
        return None
    return None


def reap_stale_lease(lock_path: str | Path, *, stale_timeout_seconds: float) -> bool:
    path = Path(lock_path)
    if stale_timeout_seconds <= 0.0 or not path.exists():
        return False
    age = time.time() - float(path.stat().st_mtime)
    if age < float(stale_timeout_seconds):
        return False
    owner_pid = _read_owner_pid(path)
    if owner_pid is not None and _pid_is_running(owner_pid):
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


@contextmanager
def acquire_gpu_lease(
    *,
    lock_path: str | Path,
    timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.2,
    stale_timeout_seconds: float = 30.0,
) -> Iterator[Path]:
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    queue_root_env = os.environ.get("MIRAI_GPU_LEASE_QUEUE_DIR", "").strip()
    queue_root = (
        Path(queue_root_env)
        if queue_root_env
        else Path(tempfile.gettempdir()) / "mirai_gpu_lease_queue"
    )
    lock_key = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    queue_dir = queue_root / lock_key
    queue_dir.mkdir(parents=True, exist_ok=True)
    ticket = queue_dir / f"{time.time_ns():020d}-{os.getpid()}-{uuid.uuid4().hex}.ticket"
    ticket.write_text(str(os.getpid()), encoding="utf-8")
    ticket_stale_seconds = max(5.0, float(stale_timeout_seconds or 0.0))

    def _is_head_ticket() -> bool:
        tickets = sorted(queue_dir.glob("*.ticket"))
        if not tickets:
            return False
        return tickets[0].name == ticket.name

    start = time.monotonic()
    fd: int | None = None
    try:
        while True:
            _reap_stale_queue_tickets(queue_dir, stale_timeout_seconds=ticket_stale_seconds)
            if _is_head_ticket():
                try:
                    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    epoch = _next_lease_epoch(path)
                    payload = {
                        "pid": int(os.getpid()),
                        "acquired_at_unix": float(time.time()),
                        "epoch": int(epoch),
                    }
                    os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
                    break
                except FileExistsError:
                    pass

            if stale_timeout_seconds > 0.0 and reap_stale_lease(
                path, stale_timeout_seconds=stale_timeout_seconds
            ):
                continue

            if timeout_seconds <= 0.0:
                raise GpuLeaseError(f"GPU lease is held by another process: {path}")
            if time.monotonic() - start >= timeout_seconds:
                raise GpuLeaseError(
                    f"Timed out waiting for GPU lease after {timeout_seconds:.1f}s: {path}"
                )
            time.sleep(max(0.01, float(poll_interval_seconds)))

        if ticket.exists():
            ticket.unlink()
        yield path
    finally:
        if ticket.exists():
            ticket.unlink()
        if fd is not None:
            os.close(fd)
        if fd is not None and path.exists():
            path.unlink()
