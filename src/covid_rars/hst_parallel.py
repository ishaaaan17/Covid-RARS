from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ContextManager, Iterator

import pandas as pd


@dataclass(frozen=True)
class ResourceSnapshot:
    logical_cpus: int
    cpu_affinity_count: int
    mem_available_bytes: int
    cgroup_headroom_bytes: int
    parent_rss_bytes: int
    dev_shm_available_bytes: int
    swap_used_bytes: int


@dataclass(frozen=True)
class GPULease:
    path: Path
    gpu_uuid: str
    run_id: str
    pid: int
    process_start_identity: str


_LOCAL_GPU_LEASES: set[str] = set()
_LOCAL_GPU_LEASES_LOCK = threading.Lock()


class _PilotCacheDataset:
    """Spawn-pickleable cache reader used only by the DataLoader resource pilot."""

    def __init__(self, paths: tuple[str, ...], required_items: int) -> None:
        self.paths = paths
        self.required_items = required_items

    def __len__(self) -> int:
        return self.required_items

    def __getitem__(self, index: int) -> object:
        import numpy as np
        import torch

        array = np.load(self.paths[index % len(self.paths)], allow_pickle=False).astype(
            np.float32, copy=False
        )
        if array.shape != (224, 224) or not np.isfinite(array).all():
            raise ValueError("Invalid HST cache tensor in DataLoader pilot")
        return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).repeat(3, 1, 1)


def _read_mem_available() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values = {}
        for line in meminfo.read_text(encoding="ascii").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0]) * 1024
        return int(values.get("MemAvailable", 0))
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        return 0


def _cgroup_headroom(default: int) -> int:
    for limit_path, usage_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
    ):
        try:
            raw_limit = limit_path.read_text(encoding="ascii").strip()
            if raw_limit == "max":
                return default
            return max(0, int(raw_limit) - int(usage_path.read_text(encoding="ascii").strip()))
        except (FileNotFoundError, OSError, ValueError):
            continue
    return default


def capture_resource_snapshot() -> ResourceSnapshot:
    logical = os.cpu_count() or 1
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else logical
    available = _read_mem_available()
    try:
        import psutil

        process = psutil.Process()
        parent_rss = int(process.memory_info().rss + sum(child.memory_info().rss for child in process.children(recursive=True)))
        swap_used = int(psutil.swap_memory().used)
    except ImportError:
        parent_rss = 0
        swap_used = 0
    try:
        shm_available = int(os.statvfs("/dev/shm").f_bavail * os.statvfs("/dev/shm").f_frsize)
    except (AttributeError, FileNotFoundError, OSError):
        shm_available = available
    return ResourceSnapshot(
        logical_cpus=int(logical),
        cpu_affinity_count=int(affinity),
        mem_available_bytes=int(available),
        cgroup_headroom_bytes=int(_cgroup_headroom(available)),
        parent_rss_bytes=parent_rss,
        dev_shm_available_bytes=shm_available,
        swap_used_bytes=swap_used,
    )


def choose_preprocess_workers(
    *,
    snapshot: ResourceSnapshot,
    estimated_worker_bytes: int,
    reserve_cpus: int,
    reserve_ram_bytes: int,
    candidates: tuple[int, ...],
) -> int:
    if estimated_worker_bytes <= 0 or reserve_cpus < 0 or reserve_ram_bytes < 0:
        raise ValueError("Invalid resource-sizing arguments")
    if not candidates or any(candidate <= 0 for candidate in candidates):
        raise ValueError("Worker candidates must be positive")
    cpu_limit = max(0, snapshot.cpu_affinity_count - reserve_cpus)
    live_headroom = min(snapshot.mem_available_bytes, snapshot.cgroup_headroom_bytes)
    memory_limit = max(0, (live_headroom - reserve_ram_bytes) // estimated_worker_bytes)
    feasible = [candidate for candidate in sorted(set(candidates)) if candidate <= cpu_limit and candidate <= memory_limit]
    if not feasible:
        raise RuntimeError("No preprocessing worker candidate preserves CPU and live-memory reserves")
    return int(max(feasible))


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    return environment


def _atomic_worker_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, 15)
        else:
            process.terminate()
        process.wait(timeout=10)
    except Exception:
        process.kill()
        process.wait(timeout=10)


def parallel_build_spectrograms(
    metadata: pd.DataFrame,
    *,
    workers: int,
    config: object,
    output_dir: Path,
) -> pd.DataFrame:
    if workers <= 0:
        raise ValueError("workers must be positive")
    worker_script = Path(__file__).resolve().parents[2] / "scripts" / "hst_preprocess_worker.py"
    if not worker_script.is_file():
        raise FileNotFoundError(worker_script)
    output_dir = Path(output_dir)
    invocation_id = uuid.uuid4().hex
    jobs_root = output_dir / "worker_jobs" / invocation_id
    jobs_root.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)  # type: ignore[arg-type]

    def run_one(index: int, row: dict[str, object]) -> dict[str, object]:
        request_payload = {
            "invocation_id": invocation_id,
            "index": index,
            "metadata": row,
            "config": config_payload,
            "output_dir": output_dir.as_posix(),
        }
        request_id = hashlib.sha256(
            json.dumps(
                request_payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        job_id = hashlib.sha256(
            f"{request_id}\0{row.get('recording_key')}".encode("utf-8")
        ).hexdigest()
        job_path = jobs_root / f"{job_id}.job.json"
        result_path = jobs_root / f"{job_id}.result.json"
        _atomic_worker_json(
            job_path,
            {**request_payload, "request_id": request_id},
        )
        command = [
            sys.executable,
            str(worker_script),
            "--job-json",
            str(job_path),
            "--result-json",
            str(result_path),
        ]
        for attempt in (1, 2):
            result_path.unlink(missing_ok=True)
            process = subprocess.Popen(
                command,
                cwd=str(worker_script.parents[1]),
                env=_worker_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
                if attempt == 2:
                    raise TimeoutError(f"HST preprocessing timed out twice for {row.get('recording_key')}")
                continue
            if process.returncode == 0 and result_path.is_file():
                response = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(response, dict) or response.get("request_id") != request_id:
                    raise RuntimeError(
                        f"HST preprocessing request identity mismatch for {row.get('recording_key')}"
                    )
                result = response.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("HST preprocessing worker returned a malformed result")
                result["worker_attempt"] = attempt
                return result
            if attempt == 2:
                raise RuntimeError(
                    f"HST preprocessing failed for {row.get('recording_key')}: "
                    f"stdout={stdout.decode(errors='replace')[-2000:]}; "
                    f"stderr={stderr.decode(errors='replace')[-4000:]}"
                )
        raise AssertionError("unreachable")

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, index, row): index
            for index, row in enumerate(metadata.to_dict(orient="records"))
        }
        for future in as_completed(futures):
            results.append(future.result())
    return pd.DataFrame(results).sort_values("recording_key").reset_index(drop=True)


def benchmark_preprocess_workers(
    metadata: pd.DataFrame,
    *,
    candidates: tuple[int, ...],
    sample_size: int,
    config: object,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    sample = metadata.sort_values("recording_key").head(sample_size)
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="hst-worker-benchmark-") as directory:
        for workers in candidates:
            before = capture_resource_snapshot()
            started = time.perf_counter()
            try:
                result = parallel_build_spectrograms(
                    sample,
                    workers=workers,
                    config=config,
                    output_dir=Path(directory) / f"workers-{workers}",
                )
                valid = True
                reason = ""
            except Exception as exc:
                result = pd.DataFrame()
                valid = False
                reason = f"{type(exc).__name__}:{exc}"
            elapsed = max(time.perf_counter() - started, 1e-9)
            after = capture_resource_snapshot()
            rows.append(
                {
                    "workers": workers,
                    "sample_size": len(sample),
                    "completed": len(result),
                    "seconds": elapsed,
                    "recordings_per_second": len(result) / elapsed,
                    "parent_rss_delta_bytes": after.parent_rss_bytes - before.parent_rss_bytes,
                    "mem_available_delta_bytes": after.mem_available_bytes - before.mem_available_bytes,
                    "swap_delta_bytes": after.swap_used_bytes - before.swap_used_bytes,
                    "dev_shm_delta_bytes": after.dev_shm_available_bytes - before.dev_shm_available_bytes,
                    "valid": valid,
                    "rejection_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def benchmark_dataloader_workers(
    cache_index: pd.DataFrame,
    *,
    candidates: tuple[int, ...],
    batches: int,
    batch_size: int,
) -> pd.DataFrame:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the DataLoader resource pilot") from exc
    if batches <= 0 or batch_size <= 0:
        raise ValueError("DataLoader pilot batches and batch_size must be positive")
    if not candidates or any(worker < 0 for worker in candidates):
        raise ValueError("DataLoader worker candidates must be non-negative")
    required = {"cache_path", "eligible"}
    missing = sorted(required - set(cache_index.columns))
    if missing:
        raise ValueError(f"DataLoader pilot cache index missing columns: {missing}")
    paths = [
        Path(value)
        for value in cache_index.loc[cache_index["eligible"].eq(True), "cache_path"].astype(str)
        if str(value).strip()
    ]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("DataLoader pilot requires existing eligible cache tensors")

    warmup_batches = 20
    required_items = (warmup_batches + batches) * batch_size

    rows: list[dict[str, object]] = []
    for workers in sorted(set(candidates)):
        before = capture_resource_snapshot()
        loader: object | None = None
        started = time.perf_counter()
        measured_started = started
        completed = 0
        valid = True
        error = ""
        try:
            options: dict[str, object] = {
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": workers,
                "pin_memory": bool(torch.cuda.is_available()),
            }
            if workers > 0:
                options.update(
                    {
                        "persistent_workers": True,
                        "prefetch_factor": 2,
                        "multiprocessing_context": "spawn",
                    }
                )
            loader = DataLoader(
                _PilotCacheDataset(tuple(path.as_posix() for path in paths), required_items),
                **options,
            )
            for index, batch in enumerate(loader):
                if torch.cuda.is_available():
                    batch = batch.to("cuda", non_blocking=True)
                    torch.cuda.synchronize()
                if index + 1 == warmup_batches:
                    measured_started = time.perf_counter()
                if index >= warmup_batches:
                    completed += 1
                if completed >= batches:
                    break
            if completed != batches:
                raise RuntimeError(f"completed only {completed}/{batches} measured batches")
        except Exception as exc:
            valid = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed = max(time.perf_counter() - measured_started, 1e-9)
        del loader
        import gc

        gc.collect()
        after = capture_resource_snapshot()
        rows.append(
            {
                "workers": workers,
                "batch_size": batch_size,
                "warmup_batches": warmup_batches,
                "measured_batches": completed,
                "seconds": elapsed,
                "batches_per_second": completed / elapsed,
                "rss_delta_bytes": after.parent_rss_bytes - before.parent_rss_bytes,
                "mem_available_delta_bytes": after.mem_available_bytes - before.mem_available_bytes,
                "swap_delta_bytes": after.swap_used_bytes - before.swap_used_bytes,
                "dev_shm_delta_bytes": after.dev_shm_available_bytes - before.dev_shm_available_bytes,
                "valid": valid,
                "rejection_reason": error,
            }
        )
    return pd.DataFrame(rows)


def select_dataloader_workers(benchmark: pd.DataFrame) -> int:
    forbidden = {column for column in benchmark if column.casefold() in {"auroc", "auprc", "f1", "accuracy"}}
    if forbidden:
        raise ValueError(f"Resource selection must not contain model metrics: {sorted(forbidden)}")
    required = {"workers", "batches_per_second", "valid"}
    missing = sorted(required - set(benchmark.columns))
    if missing:
        raise ValueError(f"DataLoader benchmark missing columns: {missing}")
    valid = benchmark.loc[benchmark["valid"].astype(bool)].copy()
    if valid.empty:
        raise RuntimeError("No valid DataLoader worker configuration")
    valid = valid.sort_values(["batches_per_second", "workers"], ascending=[False, True])
    return int(valid.iloc[0]["workers"])


def run_single_gpu_job_queue(jobs: list[object], *, device_count: int = 1) -> pd.DataFrame:
    if device_count != 1:
        raise ValueError("The fixed HST host contract permits exactly one GPU execution lane")
    rows: list[dict[str, object]] = []
    for index, job in enumerate(jobs):
        started = time.time()
        try:
            if callable(job):
                result = job()
            elif hasattr(job, "run") and callable(job.run):
                result = job.run()
            else:
                raise TypeError("GPU queue jobs must be callable or expose run()")
            status = "success"
            error = ""
        except Exception as exc:
            result = None
            status = "failed"
            error = f"{type(exc).__name__}:{exc}"
        rows.append(
            {
                "job_index": index,
                "started_unix_time": started,
                "ended_unix_time": time.time(),
                "concurrent_gpu_jobs": 1,
                "status": status,
                "result": result,
                "error": error,
            }
        )
        if status == "failed":
            break
    return pd.DataFrame(rows)


def build_deduplicated_job_plan(config: object, manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    config_payload = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    rows: list[dict[str, object]] = []
    for name, manifest in sorted(manifests.items()):
        required = [
            column
            for column in ("participant_key", "recording_key", "label_binary", "modality", "split", "tensor_sha256")
            if column in manifest
        ]
        payload = manifest[required].astype(str).sort_values(required).to_csv(index=False, lineterminator="\n")
        manifest_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        job_hash = hashlib.sha256(f"{config_hash}\0{manifest_hash}".encode("ascii")).hexdigest()
        rows.append(
            {
                "manifest_name": name,
                "manifest_sha256": manifest_hash,
                "configuration_sha256": config_hash,
                "job_sha256": job_hash,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["canonical_job"] = ~frame["job_sha256"].duplicated()
    return frame


def _process_start_identity() -> str:
    try:
        import psutil

        return f"{os.getpid()}:{psutil.Process().create_time():.6f}"
    except ImportError:
        return f"{os.getpid()}:unknown"


@contextlib.contextmanager
def acquire_gpu_execution_lease(
    lease_root: Path,
    *,
    gpu_uuid: str,
    run_id: str,
) -> Iterator[GPULease]:
    lease_root = Path(lease_root)
    lease_root.mkdir(parents=True, exist_ok=True)
    safe_uuid = hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()[:24]
    path = lease_root / f"{safe_uuid}.lock"
    local_key = str(path.resolve())
    with _LOCAL_GPU_LEASES_LOCK:
        if local_key in _LOCAL_GPU_LEASES:
            raise BlockingIOError(f"GPU {gpu_uuid} already has a live local lease")
        _LOCAL_GPU_LEASES.add(local_key)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise
        else:
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BlockingIOError(f"GPU {gpu_uuid} lease is busy") from exc
        locked = True
        lease = GPULease(path, gpu_uuid, run_id, os.getpid(), _process_start_identity())
        payload = {
            **asdict(lease),
            "path": path.as_posix(),
            "host": platform.node(),
            "heartbeat_unix_time": time.time(),
        }
        data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, data)
        os.fsync(descriptor)
        yield lease
    finally:
        if locked:
            if os.name == "posix":
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            else:
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                with contextlib.suppress(OSError):
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.close(descriptor)
        with _LOCAL_GPU_LEASES_LOCK:
            _LOCAL_GPU_LEASES.discard(local_key)
