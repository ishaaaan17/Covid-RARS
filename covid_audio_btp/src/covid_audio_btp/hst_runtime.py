from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import enum
import getpass
import hashlib
import json
import math
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


RUNTIME_SCHEMA_VERSION = 1
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_CPU_BUDGET_LOCK = threading.RLock()
_TERMINAL_STATUSES = frozenset({"success", "failed", "stopped"})
_RUNTIME_STATUSES = frozenset(
    {"pending", "running", "success", "failed", "stopped"}
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class RuntimeStateError(RuntimeError):
    """Raised when durable runtime state is malformed or internally inconsistent."""


class ProcessLiveness(str, enum.Enum):
    ALIVE = "alive"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class ProcessIdentity:
    host: str
    pid: int
    start_identity: str

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ProcessIdentity:
        try:
            host = str(record["host"])
            pid = int(record["pid"])
            start_identity = str(record["process_start_identity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeStateError("Lock record has an invalid process identity") from exc
        if not host or pid <= 0 or not start_identity:
            raise RuntimeStateError("Lock record has an invalid process identity")
        return cls(host=host, pid=pid, start_identity=start_identity)


def _canonicalize(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Canonical JSON mappings require string keys")
            normalized[key] = _canonicalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_canonicalize(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Canonical JSON cannot contain non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    normalized = _canonicalize(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _replace_atomic_file(source: Path, destination: Path) -> None:
    deadline = time.monotonic() + 1.0
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(0.005)


def atomic_write_json(path: Path, payload: object) -> Path:
    """Durably replace *path* with one complete canonical JSON document."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        serialized = json.dumps(
            _canonicalize(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(OSError):
            os.chmod(temporary, 0o600)
        _replace_atomic_file(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return path


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeStateError(f"Expected a JSON object in {path}")
    return value


def _linux_process_start_identity(
    pid: int,
) -> tuple[ProcessLiveness, str | None]:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        text = stat_path.read_text(encoding="ascii")
    except FileNotFoundError:
        return ProcessLiveness.DEAD, None
    except (PermissionError, OSError):
        return ProcessLiveness.UNKNOWN, None
    try:
        close_paren = text.rfind(")")
        fields_after_name = text[close_paren + 2 :].split()
        start_ticks = fields_after_name[19]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (FileNotFoundError, PermissionError, IndexError, OSError, ValueError):
        return ProcessLiveness.UNKNOWN, None
    return ProcessLiveness.ALIVE, f"linux:{boot_id}:{start_ticks}"


def _windows_process_start_identity(
    pid: int,
) -> tuple[ProcessLiveness, str | None]:
    if os.name != "nt":
        return ProcessLiveness.UNKNOWN, None

    class FILETIME(ctypes.Structure):
        _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
    ]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    process_query_limited_information = 0x1000
    handle = kernel32.OpenProcess(process_query_limited_information, 0, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error in {87, 1168}:
            return ProcessLiveness.DEAD, None
        return ProcessLiveness.UNKNOWN, None
    creation = FILETIME()
    exit_time = FILETIME()
    kernel_time = FILETIME()
    user_time = FILETIME()
    try:
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return ProcessLiveness.UNKNOWN, None
    finally:
        kernel32.CloseHandle(handle)
    ticks = (int(creation.high) << 32) | int(creation.low)
    return ProcessLiveness.ALIVE, f"windows-filetime:{ticks}"


def _query_process_start_identity(
    pid: int,
) -> tuple[ProcessLiveness, str | None]:
    try:
        if os.name == "nt":
            state, identity = _windows_process_start_identity(pid)
        else:
            state, identity = _linux_process_start_identity(pid)
    except Exception:
        state, identity = ProcessLiveness.UNKNOWN, None
    if state is not ProcessLiveness.UNKNOWN:
        return state, identity
    # An auxiliary process-existence query cannot recover the canonical native
    # start token, so it must not upgrade unknown into either alive or dead.
    return ProcessLiveness.UNKNOWN, None


def _native_process_identity_scheme(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if re.fullmatch(r"windows-filetime:[0-9]+", value):
        return "windows-filetime"
    if re.fullmatch(r"linux:[^:]+:[0-9]+", value):
        return "linux"
    return None


def capture_process_identity(
    pid: int | None = None, *, host: str | None = None
) -> ProcessIdentity:
    pid = os.getpid() if pid is None else int(pid)
    if pid <= 0:
        raise ValueError("pid must be positive")
    state, start_identity = _query_process_start_identity(pid)
    if state is ProcessLiveness.DEAD:
        raise ProcessLookupError(pid)
    if state is ProcessLiveness.UNKNOWN or start_identity is None:
        raise RuntimeStateError(f"Cannot determine process start identity for PID {pid}")
    return ProcessIdentity(
        host=host or socket.gethostname(),
        pid=pid,
        start_identity=start_identity,
    )


def process_identity_liveness(
    identity: ProcessIdentity, *, current_host: str | None = None
) -> ProcessLiveness:
    host = current_host or socket.gethostname()
    if identity.host.casefold() != host.casefold():
        return ProcessLiveness.UNKNOWN
    expected_scheme = _native_process_identity_scheme(identity.start_identity)
    if expected_scheme is None:
        return ProcessLiveness.UNKNOWN
    try:
        state, actual_start = _query_process_start_identity(identity.pid)
    except Exception:
        return ProcessLiveness.UNKNOWN
    if state is not ProcessLiveness.ALIVE:
        return state
    actual_scheme = _native_process_identity_scheme(actual_start)
    if actual_scheme is None or actual_scheme != expected_scheme:
        return ProcessLiveness.UNKNOWN
    if actual_start == identity.start_identity:
        return ProcessLiveness.ALIVE
    return ProcessLiveness.DEAD


def process_identity_is_alive(
    identity: ProcessIdentity, *, current_host: str | None = None
) -> bool:
    """Compatibility predicate; recovery code uses the tri-state API directly."""

    return (
        process_identity_liveness(identity, current_host=current_host)
        is ProcessLiveness.ALIVE
    )


def _identity_fields(identity: ProcessIdentity) -> dict[str, object]:
    if _native_process_identity_scheme(identity.start_identity) is None:
        raise ValueError("Runtime records require a canonical native process-start identity")
    return {
        "host": identity.host,
        "pid": identity.pid,
        "process_start_identity": identity.start_identity,
    }


def write_heartbeat_receipt(
    path: Path,
    *,
    identity: ProcessIdentity,
    run_id: str,
    status: str = "running",
    stage: str | None = None,
    sequence: int = 0,
    timestamp: float | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not run_id:
        raise ValueError("run_id is required")
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    receipt: dict[str, object] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "receipt_type": "heartbeat",
        "run_id": run_id,
        "status": status,
        "stage": stage,
        "sequence": int(sequence),
        "heartbeat_at_unix": time.time() if timestamp is None else float(timestamp),
        **_identity_fields(identity),
    }
    if details:
        receipt["details"] = dict(details)
    atomic_write_json(path, receipt)
    return receipt


def write_exit_receipt(
    path: Path,
    *,
    identity: ProcessIdentity,
    run_id: str,
    status: str,
    exit_code: int,
    error: str | None = None,
    traceback_text: str | None = None,
    timestamp: float | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not run_id:
        raise ValueError("run_id is required")
    if status not in _TERMINAL_STATUSES:
        raise ValueError(f"Exit status must be one of {sorted(_TERMINAL_STATUSES)}")
    if status == "success" and int(exit_code) != 0:
        raise ValueError("A successful exit receipt must have exit_code 0")
    receipt: dict[str, object] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "receipt_type": "exit",
        "run_id": run_id,
        "status": status,
        "exit_code": int(exit_code),
        "error": error,
        "traceback": traceback_text,
        "finished_at_unix": time.time() if timestamp is None else float(timestamp),
        **_identity_fields(identity),
    }
    if details:
        receipt["details"] = dict(details)
    atomic_write_json(path, receipt)
    return receipt


def _normalize_receipt_outputs(
    outputs: Mapping[str, str] | None,
    output_checksums: Mapping[str, str] | None,
    *,
    require_nonempty: bool,
    error_type: type[Exception] = ValueError,
) -> tuple[dict[str, str], dict[str, str]]:
    if outputs is not None and not isinstance(outputs, Mapping):
        raise error_type("Receipt outputs must be a mapping")
    if output_checksums is not None and not isinstance(output_checksums, Mapping):
        raise error_type("Receipt output checksums must be a mapping")
    normalized_outputs = dict(outputs or {})
    supplied_checksums = dict(output_checksums or {})
    if normalized_outputs.keys() != supplied_checksums.keys():
        raise error_type("outputs and output_checksums must have identical keys")
    if require_nonempty and not normalized_outputs:
        raise error_type("A successful receipt requires a checksummed output")

    normalized_checksums: dict[str, str] = {}
    for name, value in normalized_outputs.items():
        if not isinstance(name, str) or not name.strip():
            raise error_type("Receipt output names must be nonempty strings")
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise error_type("Receipt output paths must be nonempty strings")
        portable = value.replace("\\", "/")
        posix_path = PurePosixPath(portable)
        windows_path = PureWindowsPath(value)
        if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise error_type("Receipt output paths must be relative")
        if ".." in posix_path.parts:
            raise error_type("Receipt output path escapes the run root")
        normalized = posix_path.as_posix()
        if normalized in {"", "."}:
            raise error_type("Receipt output paths must identify an artifact")
        checksum = supplied_checksums[name]
        if not isinstance(checksum, str) or _SHA256_RE.fullmatch(checksum) is None:
            raise error_type("Receipt output checksum must be a valid SHA-256")
        normalized_outputs[name] = normalized
        normalized_checksums[name] = checksum.lower()
    return normalized_outputs, normalized_checksums


def _coerce_receipt_timestamp(
    value: object, *, field_name: str, error_type: type[Exception]
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise error_type(f"Receipt {field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise error_type(f"Receipt {field_name} must be finite")
    return result


def _validate_receipt_lifecycle(
    *,
    status: object,
    started_at: object,
    updated_at: object,
    finished_at: object,
    error_type: type[Exception] = RuntimeStateError,
) -> tuple[float, float, float | None]:
    if not isinstance(status, str) or status not in _RUNTIME_STATUSES:
        raise error_type(f"Receipt status must be one of {sorted(_RUNTIME_STATUSES)}")
    started = _coerce_receipt_timestamp(
        started_at, field_name="started_at_unix", error_type=error_type
    )
    updated = _coerce_receipt_timestamp(
        updated_at, field_name="updated_at_unix", error_type=error_type
    )
    if updated < started:
        raise error_type("Receipt updated timestamp is before started timestamp")
    if status in _TERMINAL_STATUSES:
        if finished_at is None:
            raise error_type("Terminal receipt requires a finished timestamp")
        finished = _coerce_receipt_timestamp(
            finished_at, field_name="finished_at_unix", error_type=error_type
        )
        if finished < started:
            raise error_type("Receipt finished timestamp is before started timestamp")
        return started, updated, finished
    if finished_at is not None:
        raise error_type("Nonterminal receipt finished timestamp must be None")
    return started, updated, None


def _validate_receipt_error(
    *, status: object, error: object, error_type: type[Exception]
) -> None:
    if error is not None and (not isinstance(error, str) or not error.strip()):
        raise error_type("Receipt error must be a nonempty string or None")
    if status == "success" and error is not None:
        raise error_type("A successful receipt cannot contain an error")


def _existing_receipt_started_at(
    path: Path,
    *,
    receipt_type: str,
    identity_fields: Mapping[str, object],
) -> float | None:
    path = Path(path)
    if not path.exists():
        return None
    existing = read_json(path)
    if existing.get("receipt_type") != receipt_type:
        raise RuntimeStateError("Cannot replace a different receipt type")
    for name, expected in identity_fields.items():
        if existing.get(name) != expected:
            raise RuntimeStateError(f"Cannot replace receipt with a different {name}")
    return _coerce_receipt_timestamp(
        existing.get("started_at_unix"),
        field_name="started_at_unix",
        error_type=RuntimeStateError,
    )


def _structured_runtime_receipt(
    *,
    receipt_type: str,
    status: str,
    fingerprint: str,
    outputs: Mapping[str, str] | None,
    output_checksums: Mapping[str, str] | None,
    error: str | None,
    started_at: float | None,
    finished_at: float | None,
    timestamp: float | None,
    prior_started_at: float | None,
    fields: Mapping[str, object],
) -> dict[str, object]:
    if status not in _RUNTIME_STATUSES:
        raise ValueError(f"Status must be one of {sorted(_RUNTIME_STATUSES)}")
    if not fingerprint:
        raise ValueError("fingerprint is required")
    _validate_receipt_error(status=status, error=error, error_type=ValueError)
    normalized_outputs, normalized_checksums = _normalize_receipt_outputs(
        outputs,
        output_checksums,
        require_nonempty=status == "success",
    )
    now = time.time() if timestamp is None else float(timestamp)
    if not math.isfinite(now):
        raise ValueError("timestamp must be finite")
    if prior_started_at is not None:
        if started_at is not None and float(started_at) != prior_started_at:
            raise ValueError("Receipt transition cannot change started_at")
        started = prior_started_at
    else:
        started = now if started_at is None else float(started_at)
    if status in _TERMINAL_STATUSES:
        finished: float | None = now if finished_at is None else float(finished_at)
    else:
        if finished_at is not None:
            raise ValueError("Nonterminal receipt finished_at must be None")
        finished = None
    started, now, finished = _validate_receipt_lifecycle(
        status=status,
        started_at=started,
        updated_at=now,
        finished_at=finished,
        error_type=ValueError,
    )
    receipt: dict[str, object] = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "receipt_type": receipt_type,
        "status": status,
        "fingerprint": fingerprint,
        "outputs": normalized_outputs,
        "output_checksums": normalized_checksums,
        "error": error,
        "started_at_unix": started,
        "updated_at_unix": now,
        "finished_at_unix": finished,
        **fields,
    }
    return receipt


def write_launch_receipt(
    path: Path,
    *,
    launch_id: str,
    status: str,
    fingerprint: str,
    run_id: str | None = None,
    outputs: Mapping[str, str] | None = None,
    output_checksums: Mapping[str, str] | None = None,
    error: str | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
    timestamp: float | None = None,
) -> dict[str, object]:
    if not launch_id:
        raise ValueError("launch_id is required")
    identity_fields = {"launch_id": launch_id}
    prior_started_at = _existing_receipt_started_at(
        path,
        receipt_type="launch",
        identity_fields=identity_fields,
    )
    receipt = _structured_runtime_receipt(
        receipt_type="launch",
        status=status,
        fingerprint=fingerprint,
        outputs=outputs,
        output_checksums=output_checksums,
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        timestamp=timestamp,
        prior_started_at=prior_started_at,
        fields={"launch_id": launch_id, "run_id": run_id},
    )
    atomic_write_json(path, receipt)
    return receipt


def write_stage_receipt(
    path: Path,
    *,
    run_id: str,
    stage: str,
    status: str,
    fingerprint: str,
    outputs: Mapping[str, str] | None = None,
    output_checksums: Mapping[str, str] | None = None,
    error: str | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
    timestamp: float | None = None,
    row_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    if not run_id:
        raise ValueError("run_id is required")
    if not stage:
        raise ValueError("stage is required")
    identity_fields = {"run_id": run_id, "stage": stage}
    prior_started_at = _existing_receipt_started_at(
        path,
        receipt_type="stage",
        identity_fields=identity_fields,
    )
    receipt = _structured_runtime_receipt(
        receipt_type="stage",
        status=status,
        fingerprint=fingerprint,
        outputs=outputs,
        output_checksums=output_checksums,
        error=error,
        started_at=started_at,
        finished_at=finished_at,
        timestamp=timestamp,
        prior_started_at=prior_started_at,
        fields={
            "run_id": run_id,
            "stage": stage,
            "row_counts": dict(row_counts or {}),
        },
    )
    atomic_write_json(path, receipt)
    return receipt


class HeartbeatEmitter:
    """Periodically publish an atomic heartbeat without owning worker lifetime."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        interval_seconds: float,
        identity: ProcessIdentity | None = None,
        status: str = "running",
        stage: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.path = Path(path)
        self.run_id = run_id
        self.interval_seconds = float(interval_seconds)
        self.identity = identity or capture_process_identity()
        self.status = status
        self.stage = stage
        self._clock = clock
        self._sequence = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._failure: BaseException | None = None

    def beat(self) -> dict[str, object]:
        with self._lock:
            self._sequence += 1
            return write_heartbeat_receipt(
                self.path,
                identity=self.identity,
                run_id=self.run_id,
                status=self.status,
                stage=self.stage,
                sequence=self._sequence,
                timestamp=self._clock(),
            )

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self.interval_seconds):
                self.beat()
        except BaseException as exc:  # pragma: no cover - surfaced by raise_if_failed
            self._failure = exc
            self._stop_event.set()

    def start(self) -> HeartbeatEmitter:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Heartbeat emitter is already running")
        self._stop_event.clear()
        self._failure = None
        self.beat()
        self._thread = threading.Thread(
            target=self._run,
            name=f"hst-heartbeat-{self.run_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("Heartbeat emitter did not stop")

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("Heartbeat emitter failed") from self._failure

    def __enter__(self) -> HeartbeatEmitter:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
        self.raise_if_failed()


class _KernelFileLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle: Any = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b", buffering=0)
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise BlockingIOError(f"Runtime lock is already held: {self.path}") from exc
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


class RuntimeLease:
    def __init__(
        self,
        *,
        kind: str,
        lock_path: Path,
        record_path: Path,
        run_id: str,
        stale_after_seconds: float,
        identity: ProcessIdentity | None,
        clock: Callable[[], float],
        process_probe: Callable[[ProcessIdentity], ProcessLiveness],
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        if not run_id:
            raise ValueError("run_id is required")
        self.kind = kind
        self.lock_path = Path(lock_path)
        self.record_path = Path(record_path)
        self.run_id = run_id
        self.stale_after_seconds = float(stale_after_seconds)
        self.identity = identity or capture_process_identity()
        self._clock = clock
        self._process_probe = process_probe
        self._metadata = dict(metadata or {})
        self._guard = _KernelFileLock(self.lock_path)
        self._token = uuid.uuid4().hex
        self._record: dict[str, object] | None = None
        self._owned = False
        self._mutex = threading.RLock()

    @property
    def token(self) -> str:
        return self._token

    def _load_previous_record(self) -> dict[str, Any] | None:
        if not self.record_path.exists():
            return None
        try:
            return read_json(self.record_path)
        except (OSError, json.JSONDecodeError, RuntimeStateError) as exc:
            raise RuntimeStateError(
                f"Cannot safely inspect existing lease record {self.record_path}"
            ) from exc

    def _stale_token_or_raise(
        self, previous: Mapping[str, object] | None, now: float
    ) -> str | None:
        if previous is None:
            return None
        owner = ProcessIdentity.from_record(previous)
        try:
            heartbeat = float(previous["heartbeat_at_unix"])
            previous_token = str(previous["token"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeStateError("Existing lease record is incomplete") from exc
        expired = now - heartbeat > self.stale_after_seconds
        if _native_process_identity_scheme(owner.start_identity) is None:
            owner_state = ProcessLiveness.UNKNOWN
        else:
            owner_state = self._process_probe(owner)
            if isinstance(owner_state, bool):
                owner_state = (
                    ProcessLiveness.ALIVE if owner_state else ProcessLiveness.UNKNOWN
                )
            else:
                try:
                    owner_state = ProcessLiveness(owner_state)
                except ValueError as exc:
                    raise RuntimeStateError(
                        "Process probe returned an invalid state"
                    ) from exc
        if owner_state is not ProcessLiveness.DEAD or not expired:
            if owner_state is ProcessLiveness.ALIVE:
                reason = "owner is alive"
            elif owner_state is ProcessLiveness.UNKNOWN:
                reason = "owner liveness is unknown"
            else:
                reason = "heartbeat is not stale"
            raise BlockingIOError(
                f"{self.kind} lease is not recoverable because {reason}: "
                f"{self.record_path}"
            )
        return previous_token

    def __enter__(self) -> RuntimeLease:
        with self._mutex:
            if self._owned:
                raise RuntimeError("Lease is already acquired")
            self._guard.acquire()
            try:
                now = float(self._clock())
                previous = self._load_previous_record()
                recovered_token = self._stale_token_or_raise(previous, now)
                record: dict[str, object] = {
                    "schema_version": RUNTIME_SCHEMA_VERSION,
                    "lease_kind": self.kind,
                    "token": self._token,
                    "run_id": self.run_id,
                    "acquired_at_unix": now,
                    "heartbeat_at_unix": now,
                    "heartbeat_sequence": 0,
                    **_identity_fields(self.identity),
                    **self._metadata,
                }
                if recovered_token is not None:
                    record["recovered_stale_token"] = recovered_token
                atomic_write_json(self.record_path, record)
                self._record = record
                self._owned = True
            except BaseException:
                self._guard.release()
                raise
        return self

    def heartbeat(
        self,
        *,
        stage: str | None = None,
        status: str = "running",
        details: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        with self._mutex:
            if not self._owned or self._record is None:
                raise RuntimeError("Cannot heartbeat an unowned lease")
            record = dict(self._record)
            record["heartbeat_at_unix"] = float(self._clock())
            record["heartbeat_sequence"] = int(record["heartbeat_sequence"]) + 1
            record["status"] = status
            record["stage"] = stage
            if details:
                record["details"] = dict(details)
            atomic_write_json(self.record_path, record)
            self._record = record
            return dict(record)

    def release(self) -> None:
        with self._mutex:
            if not self._owned:
                return
            try:
                try:
                    current = read_json(self.record_path)
                except (FileNotFoundError, OSError, json.JSONDecodeError, RuntimeStateError):
                    current = None
                if current is not None and current.get("token") == self._token:
                    with contextlib.suppress(FileNotFoundError):
                        self.record_path.unlink()
                    _fsync_directory(self.record_path.parent)
            finally:
                self._owned = False
                self._guard.release()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def _safe_gpu_key(gpu_uuid: str) -> str:
    if not gpu_uuid:
        raise ValueError("gpu_uuid is required")
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", gpu_uuid).strip("._-")
    readable = (readable or "gpu")[:48]
    suffix = hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{suffix}"


def gpu_token_paths(lease_root: Path, gpu_uuid: str) -> tuple[Path, Path]:
    root = Path(lease_root).resolve()
    key = _safe_gpu_key(gpu_uuid)
    return root / f"{key}.lock", root / f"{key}.json"


def default_gpu_lease_root(
    *,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    user_identity: str | None = None,
) -> Path:
    environment = os.environ if environment is None else environment
    platform_name = os.name if platform_name is None else platform_name
    if platform_name == "nt":
        local_app_data = environment.get("LOCALAPPDATA")
        if local_app_data:
            return (
                Path(local_app_data) / "covid_audio_btp" / "hst_gpu"
            ).resolve()
        program_data = environment.get("PROGRAMDATA") or environment.get(
            "ALLUSERSPROFILE"
        )
        if program_data:
            stable_identity = user_identity or _stable_user_identity(environment)
            user_key = hashlib.sha256(
                stable_identity.casefold().encode("utf-8")
            ).hexdigest()[:20]
            return (
                Path(program_data)
                / "covid_audio_btp"
                / "users"
                / user_key
                / "hst_gpu"
            ).resolve()
        return (
            Path.home() / "AppData" / "Local" / "covid_audio_btp" / "hst_gpu"
        ).resolve()
    xdg_runtime = environment.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        candidate = Path(xdg_runtime)
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate / "covid_audio_btp" / "hst_gpu"
    uid = os.getuid()
    return Path("/var/tmp") / f"covid_audio_btp-{uid}" / "hst_gpu"


def _stable_user_identity(environment: Mapping[str, str]) -> str:
    username = environment.get("USERNAME") or environment.get("USER")
    domain = environment.get("USERDOMAIN")
    if username and domain:
        return f"{domain}\\{username}"
    if username:
        return username
    return getpass.getuser()


def default_run_lock_root(
    *,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    user_identity: str | None = None,
) -> Path:
    gpu_root = default_gpu_lease_root(
        environment=environment,
        platform_name=platform_name,
        user_identity=user_identity,
    )
    return gpu_root.parent / "hst_runs"


def run_lock_paths(run_lock_root: Path, run_id: str) -> tuple[Path, Path]:
    if not run_id:
        raise ValueError("run_id is required")
    root = Path(run_lock_root).resolve()
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("._-")
    readable = (readable or "run")[:48]
    suffix = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    key = f"{readable}-{suffix}"
    return root / f"{key}.lock", root / f"{key}.json"


def acquire_gpu_token(
    *,
    gpu_uuid: str,
    run_id: str,
    lease_root: Path | None = None,
    stale_after_seconds: float = 180.0,
    identity: ProcessIdentity | None = None,
    clock: Callable[[], float] = time.time,
    process_probe: Callable[
        [ProcessIdentity], ProcessLiveness
    ] = process_identity_liveness,
) -> RuntimeLease:
    root = default_gpu_lease_root() if lease_root is None else Path(lease_root)
    lock_path, record_path = gpu_token_paths(root, gpu_uuid)
    return RuntimeLease(
        kind="gpu",
        lock_path=lock_path,
        record_path=record_path,
        run_id=run_id,
        stale_after_seconds=stale_after_seconds,
        identity=identity,
        clock=clock,
        process_probe=process_probe,
        metadata={"device_uuid": gpu_uuid},
    )


def acquire_gpu_execution_lease(
    lease_root: Path,
    *,
    gpu_uuid: str,
    run_id: str,
    stale_after_seconds: float = 180.0,
    identity: ProcessIdentity | None = None,
    clock: Callable[[], float] = time.time,
    process_probe: Callable[
        [ProcessIdentity], ProcessLiveness
    ] = process_identity_liveness,
) -> RuntimeLease:
    return acquire_gpu_token(
        lease_root=lease_root,
        gpu_uuid=gpu_uuid,
        run_id=run_id,
        stale_after_seconds=stale_after_seconds,
        identity=identity,
        clock=clock,
        process_probe=process_probe,
    )


def acquire_run_lock(
    runtime_dir: Path,
    *,
    run_id: str,
    config_hash: str,
    run_lock_root: Path | None = None,
    stale_after_seconds: float = 180.0,
    identity: ProcessIdentity | None = None,
    clock: Callable[[], float] = time.time,
    process_probe: Callable[
        [ProcessIdentity], ProcessLiveness
    ] = process_identity_liveness,
) -> RuntimeLease:
    runtime_dir = Path(runtime_dir).resolve()
    root = default_run_lock_root() if run_lock_root is None else Path(run_lock_root)
    lock_path, record_path = run_lock_paths(root, run_id)
    return RuntimeLease(
        kind="run",
        lock_path=lock_path,
        record_path=record_path,
        run_id=run_id,
        stale_after_seconds=stale_after_seconds,
        identity=identity,
        clock=clock,
        process_probe=process_probe,
        metadata={"config_hash": config_hash, "runtime_dir": runtime_dir.as_posix()},
    )


@dataclasses.dataclass(frozen=True)
class VerifiedFileHash:
    sha256: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int


def _file_stat_identity(stat: object) -> tuple[int, int, int, int]:
    return (
        int(getattr(stat, "st_dev")),
        int(getattr(stat, "st_ino")),
        int(getattr(stat, "st_size")),
        int(getattr(stat, "st_mtime_ns")),
    )


def verified_file_sha256(
    path: Path, chunk_size: int = 8 * 1024 * 1024
) -> VerifiedFileHash:
    """Hash one open file and verify its descriptor metadata stayed coherent.

    This detects ordinary concurrent writes and path replacement. It cannot
    protect against a malicious writer that changes bytes and then restores all
    observable identity, size, and timestamp metadata during the same read.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    before_identity = _file_stat_identity(before)
    after_identity = _file_stat_identity(after)
    if before_identity != after_identity:
        raise RuntimeStateError(f"File changed while hashing: {path}")
    path_identity = _file_stat_identity(path.stat())
    if path_identity != after_identity:
        raise RuntimeStateError(f"File path changed while hashing: {path}")
    return VerifiedFileHash(
        sha256=digest.hexdigest(),
        size_bytes=after_identity[2],
        mtime_ns=after_identity[3],
        device=after_identity[0],
        inode=after_identity[1],
    )


def stable_file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    return verified_file_sha256(path, chunk_size=chunk_size).sha256


def source_tree_hash(root: Path, relative_paths: Sequence[str | Path]) -> str:
    root = Path(root).resolve()
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for supplied in relative_paths:
        relative = Path(supplied)
        if relative.is_absolute():
            raise ValueError(f"Source path must be relative and cannot escape root: {supplied}")
        resolved = (root / relative).resolve()
        try:
            normalized = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Source path cannot escape root: {supplied}") from exc
        if normalized in seen:
            raise ValueError(f"Duplicate source path: {normalized}")
        seen.add(normalized)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        verified = verified_file_sha256(resolved)
        records.append(
            {
                "path": normalized,
                "size_bytes": verified.size_bytes,
                "sha256": verified.sha256,
            }
        )
    records.sort(key=lambda record: str(record["path"]))
    return canonical_json_sha256(
        {"schema_version": RUNTIME_SCHEMA_VERSION, "files": records}
    )


@dataclasses.dataclass(frozen=True)
class StageRequirements:
    required_upstream: frozenset[str] = frozenset()
    required_accepted: frozenset[str] = frozenset()
    require_input_hashes: bool = True
    require_manifests: bool = False
    require_checkpoints: bool = False
    require_pip_freeze: bool = False


_FULL_ACCEPTED_HASHES = frozenset(
    {"pilot_freeze", "data_contracts_freeze", "environment_lock"}
)

STAGE_REQUIREMENTS: dict[str, StageRequirements] = {
    "preflight": StageRequirements(),
    "data_contracts": StageRequirements(required_upstream=frozenset({"preflight"})),
    "checkpoint": StageRequirements(
        required_upstream=frozenset({"preflight"}),
        require_checkpoints=True,
    ),
    "preprocess_worker_pilot": StageRequirements(
        required_upstream=frozenset({"data_contracts"})
    ),
    "spectrogram_cache": StageRequirements(
        required_upstream=frozenset(
            {"data_contracts", "preprocess_worker_pilot"}
        )
    ),
    "manifests": StageRequirements(
        required_upstream=frozenset({"data_contracts", "spectrogram_cache"})
    ),
    "small_smoke": StageRequirements(
        required_upstream=frozenset(
            {"checkpoint", "manifests", "spectrogram_cache"}
        ),
        require_manifests=True,
        require_checkpoints=True,
        require_pip_freeze=True,
    ),
    "base_resource_pilot": StageRequirements(
        required_upstream=frozenset({"small_smoke"}),
        require_manifests=True,
        require_checkpoints=True,
        require_pip_freeze=True,
    ),
    "aligned_comparator": StageRequirements(
        required_upstream=frozenset({"manifests"}),
        required_accepted=_FULL_ACCEPTED_HASHES,
        require_manifests=True,
        require_pip_freeze=True,
    ),
    "internal_cv": StageRequirements(
        required_upstream=frozenset(
            {"base_resource_pilot", "aligned_comparator"}
        ),
        required_accepted=_FULL_ACCEPTED_HASHES,
        require_manifests=True,
        require_checkpoints=True,
        require_pip_freeze=True,
    ),
    "split_policy_contrast": StageRequirements(
        required_upstream=frozenset({"internal_cv"}),
        required_accepted=_FULL_ACCEPTED_HASHES,
        require_manifests=True,
        require_checkpoints=True,
        require_pip_freeze=True,
    ),
    "reverse_temporal": StageRequirements(
        required_upstream=frozenset({"internal_cv"}),
        required_accepted=_FULL_ACCEPTED_HASHES,
        require_manifests=True,
        require_checkpoints=True,
        require_pip_freeze=True,
    ),
    "external_transfer": StageRequirements(
        required_upstream=frozenset({"internal_cv"}),
        required_accepted=_FULL_ACCEPTED_HASHES,
        require_manifests=True,
        require_checkpoints=True,
        require_pip_freeze=True,
    ),
    "fusion": StageRequirements(
        required_upstream=frozenset(
            {
                "internal_cv",
                "split_policy_contrast",
                "reverse_temporal",
                "external_transfer",
                "aligned_comparator",
            }
        ),
        required_accepted=_FULL_ACCEPTED_HASHES,
        require_manifests=True,
        require_checkpoints=True,
        require_pip_freeze=True,
    ),
    "statistics": StageRequirements(
        required_upstream=frozenset({"fusion"}),
        required_accepted=_FULL_ACCEPTED_HASHES,
        require_manifests=True,
        require_checkpoints=True,
        require_pip_freeze=True,
    ),
    "gradcam": StageRequirements(
        required_upstream=frozenset({"internal_cv"}),
        required_accepted=_FULL_ACCEPTED_HASHES,
        require_manifests=True,
        require_checkpoints=True,
        require_pip_freeze=True,
    ),
    "evidence_pack": StageRequirements(
        required_upstream=frozenset({"statistics", "gradcam"}),
        required_accepted=_FULL_ACCEPTED_HASHES,
        require_manifests=True,
        require_checkpoints=True,
        require_pip_freeze=True,
    ),
}


def _nonempty_hash_mapping(
    value: Mapping[str, str] | None, *, field_name: str
) -> dict[str, str]:
    normalized = dict(value or {})
    empty_keys = [key for key, item in normalized.items() if not key or not item]
    if empty_keys:
        raise ValueError(f"{field_name} contains an empty key or hash")
    return normalized


def validate_stage_fingerprint_inputs(
    stage: str,
    *,
    input_hashes: Mapping[str, str] | None,
    configuration_hash: str,
    executable_source_hash: str,
    dependency_lock_hash: str,
    hst_commit: str,
    checkpoint_hashes: Mapping[str, str] | None,
    manifest_hashes: Mapping[str, str] | None,
    upstream_hashes: Mapping[str, str] | None,
    accepted_hashes: Mapping[str, str] | None,
    pip_freeze_hash: str | None,
) -> None:
    if stage not in STAGE_REQUIREMENTS:
        raise ValueError(f"Unknown HST stage: {stage}")
    base_fields = {
        "configuration hash": configuration_hash,
        "executable source hash": executable_source_hash,
        "dependency lock hash": dependency_lock_hash,
        "HST commit": hst_commit,
    }
    missing_base = [name for name, value in base_fields.items() if not value]
    if missing_base:
        raise ValueError(f"Missing required {', '.join(missing_base)}")

    inputs = _nonempty_hash_mapping(input_hashes, field_name="input hashes")
    checkpoints = _nonempty_hash_mapping(
        checkpoint_hashes, field_name="checkpoint hashes"
    )
    manifests = _nonempty_hash_mapping(manifest_hashes, field_name="manifest hashes")
    upstream = _nonempty_hash_mapping(upstream_hashes, field_name="upstream hashes")
    accepted = _nonempty_hash_mapping(accepted_hashes, field_name="accepted hashes")
    requirements = STAGE_REQUIREMENTS[stage]
    if requirements.require_input_hashes and not inputs:
        raise ValueError(f"Stage {stage} requires input hashes")
    missing_upstream = sorted(requirements.required_upstream - upstream.keys())
    if missing_upstream:
        raise ValueError(f"Missing required upstream hashes: {missing_upstream}")
    missing_accepted = sorted(requirements.required_accepted - accepted.keys())
    if missing_accepted:
        raise ValueError(f"Missing required accepted hashes: {missing_accepted}")
    if requirements.require_manifests and not manifests:
        raise ValueError(f"Stage {stage} requires manifest hashes")
    if requirements.require_checkpoints and not checkpoints:
        raise ValueError(f"Stage {stage} requires checkpoint hashes")
    if requirements.require_pip_freeze and not pip_freeze_hash:
        raise ValueError(f"Stage {stage} requires a pip freeze hash")


def stage_fingerprint(
    stage: str,
    *,
    input_hashes: Mapping[str, str] | None,
    configuration_hash: str,
    executable_source_hash: str,
    dependency_lock_hash: str,
    hst_commit: str,
    checkpoint_hashes: Mapping[str, str] | None = None,
    manifest_hashes: Mapping[str, str] | None = None,
    upstream_hashes: Mapping[str, str] | None = None,
    accepted_hashes: Mapping[str, str] | None = None,
    pip_freeze_hash: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> str:
    validate_stage_fingerprint_inputs(
        stage,
        input_hashes=input_hashes,
        configuration_hash=configuration_hash,
        executable_source_hash=executable_source_hash,
        dependency_lock_hash=dependency_lock_hash,
        hst_commit=hst_commit,
        checkpoint_hashes=checkpoint_hashes,
        manifest_hashes=manifest_hashes,
        upstream_hashes=upstream_hashes,
        accepted_hashes=accepted_hashes,
        pip_freeze_hash=pip_freeze_hash,
    )
    payload = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "stage": stage,
        "input_hashes": dict(input_hashes or {}),
        "configuration_hash": configuration_hash,
        "executable_source_hash": executable_source_hash,
        "dependency_lock_hash": dependency_lock_hash,
        "hst_commit": hst_commit,
        "checkpoint_hashes": dict(checkpoint_hashes or {}),
        "manifest_hashes": dict(manifest_hashes or {}),
        "upstream_hashes": dict(upstream_hashes or {}),
        "accepted_hashes": dict(accepted_hashes or {}),
        "pip_freeze_hash": pip_freeze_hash,
        "extra": dict(extra or {}),
    }
    return canonical_json_sha256(payload)


def _validated_stage_receipt_outputs(
    receipt: Mapping[str, object],
    *,
    run_root: Path,
    expected_stage: str | None = None,
) -> tuple[str, str, dict[str, str], dict[str, str]]:
    if receipt.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise RuntimeStateError("Stage receipt schema version is invalid")
    if receipt.get("receipt_type") != "stage":
        raise RuntimeStateError("Expected a stage receipt")
    if receipt.get("status") != "success":
        raise RuntimeStateError("Only successful stage receipts can be dependencies")
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeStateError("Stage receipt has no run ID")
    stage = receipt.get("stage")
    fingerprint = receipt.get("fingerprint")
    outputs = receipt.get("outputs")
    checksums = receipt.get("output_checksums")
    if not isinstance(stage, str) or not stage:
        raise RuntimeStateError("Stage receipt has no stage name")
    if expected_stage is not None and stage != expected_stage:
        raise RuntimeStateError("Completed stage name does not match")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeStateError("Stage receipt has no fingerprint")
    normalized_outputs, normalized_checksums = _normalize_receipt_outputs(
        outputs if isinstance(outputs, Mapping) else None,
        checksums if isinstance(checksums, Mapping) else None,
        require_nonempty=True,
        error_type=RuntimeStateError,
    )
    _validate_receipt_lifecycle(
        status=receipt.get("status"),
        started_at=receipt.get("started_at_unix"),
        updated_at=receipt.get("updated_at_unix"),
        finished_at=receipt.get("finished_at_unix"),
    )
    _validate_receipt_error(
        status=receipt.get("status"),
        error=receipt.get("error"),
        error_type=RuntimeStateError,
    )

    root = Path(run_root).resolve()
    verified_checksums: dict[str, str] = {}
    for name, relative_value in normalized_outputs.items():
        output_path = (root / Path(relative_value)).resolve()
        try:
            output_path.relative_to(root)
        except ValueError as exc:
            raise RuntimeStateError("Stage receipt output escapes run root") from exc
        if not output_path.is_file():
            raise RuntimeStateError(f"Stage receipt output is missing: {name}")
        actual_checksum = stable_file_sha256(output_path)
        if actual_checksum != normalized_checksums[name]:
            raise RuntimeStateError(f"Stage receipt output checksum mismatch: {name}")
        verified_checksums[name] = actual_checksum
    return stage, fingerprint, normalized_outputs, verified_checksums


def stage_receipt_dependency_hash(
    receipt: Mapping[str, object], *, run_root: Path
) -> str:
    stage, fingerprint, outputs, verified_checksums = _validated_stage_receipt_outputs(
        receipt,
        run_root=run_root,
    )
    return canonical_json_sha256(
        {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "stage": stage,
            "fingerprint": fingerprint,
            "outputs": outputs,
            "output_checksums": verified_checksums,
        }
    )


def downstream_stage_fingerprint(
    stage: str,
    *,
    run_root: Path,
    upstream_receipts: Mapping[str, Mapping[str, object]],
    input_hashes: Mapping[str, str] | None,
    configuration_hash: str,
    executable_source_hash: str,
    dependency_lock_hash: str,
    hst_commit: str,
    checkpoint_hashes: Mapping[str, str] | None = None,
    manifest_hashes: Mapping[str, str] | None = None,
    accepted_hashes: Mapping[str, str] | None = None,
    pip_freeze_hash: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> str:
    upstream_hashes: dict[str, str] = {}
    for name, receipt in upstream_receipts.items():
        if receipt.get("stage") != name:
            raise RuntimeStateError(
                f"Upstream receipt key {name!r} does not match its stage"
            )
        upstream_hashes[name] = stage_receipt_dependency_hash(
            receipt,
            run_root=run_root,
        )
    return stage_fingerprint(
        stage,
        input_hashes=input_hashes,
        configuration_hash=configuration_hash,
        executable_source_hash=executable_source_hash,
        dependency_lock_hash=dependency_lock_hash,
        hst_commit=hst_commit,
        checkpoint_hashes=checkpoint_hashes,
        manifest_hashes=manifest_hashes,
        upstream_hashes=upstream_hashes,
        accepted_hashes=accepted_hashes,
        pip_freeze_hash=pip_freeze_hash,
        extra=extra,
    )


def validate_completed_stage_reuse(
    receipt_path: Path,
    *,
    expected_fingerprint: str,
    run_root: Path,
    expected_stage: str | None = None,
) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    if receipt.get("fingerprint") != expected_fingerprint:
        raise RuntimeStateError("Completed stage fingerprint does not match")
    _validated_stage_receipt_outputs(
        receipt,
        run_root=run_root,
        expected_stage=expected_stage,
    )
    return receipt


def completed_stage_is_reusable(
    receipt_path: Path,
    *,
    expected_fingerprint: str,
    run_root: Path,
    expected_stage: str | None = None,
) -> bool:
    try:
        validate_completed_stage_reuse(
            receipt_path,
            expected_fingerprint=expected_fingerprint,
            run_root=run_root,
            expected_stage=expected_stage,
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeStateError):
        return False
    return True


def cpu_thread_environment(
    threads: int,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    threads = int(threads)
    if threads <= 0:
        raise ValueError("threads must be positive")
    result = dict(os.environ if base_environment is None else base_environment)
    for name in CPU_THREAD_ENV_VARS:
        result[name] = str(threads)
    return result


@contextlib.contextmanager
def cpu_thread_budget(
    threads: int,
    *,
    environment: MutableMapping[str, str] | None = None,
    limiter_factory: Callable[..., object] | None = None,
) -> Iterator[None]:
    threads = int(threads)
    if threads <= 0:
        raise ValueError("threads must be positive")
    environment = os.environ if environment is None else environment
    if limiter_factory is None:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError as exc:
            raise RuntimeError("threadpoolctl is required for CPU thread budgets") from exc
        limiter_factory = threadpool_limits

    missing = object()
    with _CPU_BUDGET_LOCK:
        previous: dict[str, object] = {
            name: environment.get(name, missing) for name in CPU_THREAD_ENV_VARS
        }
        try:
            for name in CPU_THREAD_ENV_VARS:
                environment[name] = str(threads)
            limiter = limiter_factory(limits=threads)
            with limiter:  # type: ignore[attr-defined]
                yield
        finally:
            for name, value in previous.items():
                if value is missing:
                    environment.pop(name, None)
                else:
                    environment[name] = str(value)
