from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


class _PreprocessWorkerProcess:
    job_directories: list[Path] = []
    mismatch_request_identity = False

    def __init__(self, command: list[str], **_kwargs: object) -> None:
        job_path = Path(command[command.index("--job-json") + 1])
        result_path = Path(command[command.index("--result-json") + 1])
        self.job_directories.append(job_path.parent)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        row = dict(job["metadata"])
        row["eligible"] = True
        request_id = str(job.get("request_id", ""))
        if self.mismatch_request_identity:
            request_id = "wrong-request"
        payload: dict[str, object]
        if "request_id" in job:
            payload = {"request_id": request_id, "result": row}
        else:
            payload = row
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        self.returncode = 0

    def communicate(self, timeout: int) -> tuple[bytes, bytes]:
        assert timeout == 600
        return b"", b""

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -1

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: int) -> int:
        return self.returncode


def _hold_gpu_lease(
    lease_root: str, ready: object, release: object, result: object
) -> None:
    from covid_audio_btp.hst_runtime import acquire_gpu_token

    try:
        with acquire_gpu_token(
            lease_root=Path(lease_root),
            gpu_uuid="GPU-cross-process",
            run_id="child-run",
        ):
            ready.set()
            if not release.wait(10):
                raise TimeoutError("parent did not release child lease")
        result.put("success")
    except BaseException as exc:
        result.put(f"{type(exc).__name__}: {exc}")


def _hold_run_lease(
    runtime_dir: str,
    stable_root: str,
    ready: object,
    release: object,
    result: object,
) -> None:
    from covid_audio_btp.hst_runtime import acquire_run_lock

    try:
        with acquire_run_lock(
            Path(runtime_dir),
            run_lock_root=Path(stable_root),
            run_id="shared-run",
            config_hash="config",
        ):
            ready.set()
            if not release.wait(10):
                raise TimeoutError("parent did not release child lease")
        result.put("success")
    except BaseException as exc:
        result.put(f"{type(exc).__name__}: {exc}")


def test_parallel_preprocessing_rejects_result_from_another_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd
    import covid_audio_btp.hst_parallel as parallel

    _PreprocessWorkerProcess.job_directories.clear()
    _PreprocessWorkerProcess.mismatch_request_identity = True
    monkeypatch.setattr(parallel.subprocess, "Popen", _PreprocessWorkerProcess)
    metadata = pd.DataFrame(
        [{"recording_key": "coswara::r1", "audio_path": "unused.wav"}]
    )

    with pytest.raises(RuntimeError, match="request identity mismatch"):
        parallel.parallel_build_spectrograms(
            metadata,
            workers=1,
            config={"representation_id": "paper_logmel_224"},
            output_dir=tmp_path / "cache",
        )


def test_parallel_preprocessing_uses_unique_invocation_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd
    import covid_audio_btp.hst_parallel as parallel

    _PreprocessWorkerProcess.job_directories.clear()
    _PreprocessWorkerProcess.mismatch_request_identity = False
    monkeypatch.setattr(parallel.subprocess, "Popen", _PreprocessWorkerProcess)
    metadata = pd.DataFrame(
        [{"recording_key": "coswara::r1", "audio_path": "unused.wav"}]
    )
    for _ in range(2):
        result = parallel.parallel_build_spectrograms(
            metadata,
            workers=1,
            config={"representation_id": "paper_logmel_224"},
            output_dir=tmp_path / "cache",
        )
        assert result.loc[0, "recording_key"] == "coswara::r1"

    assert len(_PreprocessWorkerProcess.job_directories) == 2
    assert len(set(_PreprocessWorkerProcess.job_directories)) == 2


def test_atomic_json_write_replaces_complete_document_and_cleans_temp_files(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import atomic_write_json, read_json

    path = tmp_path / "runtime" / "heartbeat.json"
    atomic_write_json(path, {"sequence": 1, "status": "running"})
    atomic_write_json(path, {"sequence": 2, "status": "success"})

    assert read_json(path) == {"sequence": 2, "status": "success"}
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_atomic_json_write_preserves_previous_document_if_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    path = tmp_path / "state.json"
    runtime.atomic_write_json(path, {"status": "running"})

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected replacement failure")

    monkeypatch.setattr(runtime.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        runtime.atomic_write_json(path, {"status": "success"})

    assert runtime.read_json(path) == {"status": "running"}
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_process_identity_uses_pid_and_start_identity() -> None:
    from covid_audio_btp.hst_runtime import (
        ProcessIdentity,
        ProcessLiveness,
        capture_process_identity,
        process_identity_liveness,
    )

    identity = capture_process_identity()
    assert identity.pid == os.getpid()
    assert process_identity_liveness(identity) is ProcessLiveness.ALIVE

    prefix, ticks = identity.start_identity.rsplit(":", 1)
    reused_pid = ProcessIdentity(
        host=identity.host,
        pid=identity.pid,
        start_identity=f"{prefix}:{int(ticks) + 1}",
    )
    assert process_identity_liveness(reused_pid) is ProcessLiveness.DEAD


def test_host_mismatch_and_query_failure_are_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    identity = runtime.capture_process_identity()
    foreign = runtime.ProcessIdentity(
        host="different-host",
        pid=identity.pid,
        start_identity=identity.start_identity,
    )
    assert (
        runtime.process_identity_liveness(foreign)
        is runtime.ProcessLiveness.UNKNOWN
    )

    monkeypatch.setattr(
        runtime,
        "_query_process_start_identity",
        lambda pid: (runtime.ProcessLiveness.UNKNOWN, None),
    )
    assert (
        runtime.process_identity_liveness(identity)
        is runtime.ProcessLiveness.UNKNOWN
    )


def test_current_process_query_failure_is_not_replaced_by_synthetic_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(runtime.os, "name", "posix")
    monkeypatch.setattr(
        runtime,
        "_linux_process_start_identity",
        lambda pid: (runtime.ProcessLiveness.UNKNOWN, None),
    )

    state, identity = runtime._query_process_start_identity(os.getpid())
    assert state is runtime.ProcessLiveness.UNKNOWN
    assert identity is None


def test_unexpected_process_query_exception_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    identity = runtime.capture_process_identity()

    def fail_query(pid: int) -> object:
        raise PermissionError("query denied")

    monkeypatch.setattr(runtime, "_query_process_start_identity", fail_query)
    assert (
        runtime.process_identity_liveness(identity)
        is runtime.ProcessLiveness.UNKNOWN
    )


def test_process_identity_always_uses_native_encoding_when_psutil_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    class FakePsutilError(Exception):
        pass

    fake_psutil = SimpleNamespace(
        AccessDenied=FakePsutilError,
        Error=FakePsutilError,
        NoSuchProcess=FakePsutilError,
        ZombieProcess=FakePsutilError,
        Process=lambda pid: SimpleNamespace(create_time=lambda: 123.456),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(runtime.os, "name", "nt")
    monkeypatch.setattr(
        runtime,
        "_windows_process_start_identity",
        lambda pid: (runtime.ProcessLiveness.ALIVE, "windows-filetime:987654"),
    )

    state, identity = runtime._query_process_start_identity(123)

    assert state is runtime.ProcessLiveness.ALIVE
    assert identity == "windows-filetime:987654"


def test_psutil_alive_cannot_replace_unknown_native_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    class FakePsutilError(Exception):
        pass

    fake_psutil = SimpleNamespace(
        AccessDenied=FakePsutilError,
        Error=FakePsutilError,
        NoSuchProcess=FakePsutilError,
        ZombieProcess=FakePsutilError,
        Process=lambda pid: SimpleNamespace(create_time=lambda: 123.456),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(runtime.os, "name", "nt")
    monkeypatch.setattr(
        runtime,
        "_windows_process_start_identity",
        lambda pid: (runtime.ProcessLiveness.UNKNOWN, None),
    )

    state, identity = runtime._query_process_start_identity(123)

    assert state is runtime.ProcessLiveness.UNKNOWN
    assert identity is None


def test_psutil_no_such_process_cannot_upgrade_unknown_native_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    class NoSuchProcess(Exception):
        pass

    class FakePsutilError(Exception):
        pass

    def missing_process(pid: int) -> object:
        raise NoSuchProcess(pid)

    fake_psutil = SimpleNamespace(
        AccessDenied=FakePsutilError,
        Error=FakePsutilError,
        NoSuchProcess=NoSuchProcess,
        ZombieProcess=NoSuchProcess,
        Process=missing_process,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(runtime.os, "name", "nt")
    monkeypatch.setattr(
        runtime,
        "_windows_process_start_identity",
        lambda pid: (runtime.ProcessLiveness.UNKNOWN, None),
    )

    state, identity = runtime._query_process_start_identity(123)

    assert state is runtime.ProcessLiveness.UNKNOWN
    assert identity is None


def test_legacy_process_identity_encoding_is_unknown_not_dead() -> None:
    import covid_audio_btp.hst_runtime as runtime

    current = runtime.capture_process_identity()
    legacy = runtime.ProcessIdentity(
        host=current.host,
        pid=current.pid,
        start_identity="psutil-create-time:123.456000000",
    )

    assert (
        runtime.process_identity_liveness(legacy)
        is runtime.ProcessLiveness.UNKNOWN
    )


def test_runtime_receipts_refuse_non_native_process_identity(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    legacy = runtime.ProcessIdentity(
        host="host",
        pid=123,
        start_identity="psutil-create-time:123.456000000",
    )
    with pytest.raises(ValueError, match="native process-start identity"):
        runtime.write_heartbeat_receipt(
            tmp_path / "heartbeat.json",
            identity=legacy,
            run_id="run",
        )


def test_native_process_query_exception_degrades_to_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    class FakePsutilError(Exception):
        pass

    fake_psutil = SimpleNamespace(
        AccessDenied=FakePsutilError,
        Error=FakePsutilError,
        NoSuchProcess=FakePsutilError,
        ZombieProcess=FakePsutilError,
        Process=lambda pid: SimpleNamespace(create_time=lambda: 123.456),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(runtime.os, "name", "nt")

    def fail_native_query(pid: int) -> object:
        raise PermissionError("native query denied")

    monkeypatch.setattr(runtime, "_windows_process_start_identity", fail_native_query)

    state, identity = runtime._query_process_start_identity(123)
    assert state is runtime.ProcessLiveness.UNKNOWN
    assert identity is None


def test_heartbeat_and_exit_receipts_preserve_failure_visibility(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import (
        capture_process_identity,
        read_json,
        write_exit_receipt,
        write_heartbeat_receipt,
    )

    identity = capture_process_identity()
    heartbeat_path = tmp_path / "heartbeat.json"
    exit_path = tmp_path / "exit.json"
    write_heartbeat_receipt(
        heartbeat_path,
        identity=identity,
        run_id="run-123",
        status="running",
        stage="internal_cv",
        sequence=7,
        timestamp=100.5,
    )
    write_exit_receipt(
        exit_path,
        identity=identity,
        run_id="run-123",
        status="failed",
        exit_code=1,
        error="CUDA out of memory",
        traceback_text="Traceback: test",
        timestamp=101.5,
    )

    heartbeat = read_json(heartbeat_path)
    exit_receipt = read_json(exit_path)
    assert heartbeat["pid"] == os.getpid()
    assert heartbeat["process_start_identity"] == identity.start_identity
    assert heartbeat["heartbeat_at_unix"] == 100.5
    assert heartbeat["sequence"] == 7
    assert exit_receipt["status"] == "failed"
    assert exit_receipt["error"] == "CUDA out of memory"
    assert exit_receipt["traceback"] == "Traceback: test"


def test_background_heartbeat_is_cpu_only_and_stops_cleanly(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import HeartbeatEmitter, read_json

    path = tmp_path / "heartbeat.json"
    emitter = HeartbeatEmitter(path, run_id="run-heartbeat", interval_seconds=0.01)
    emitter.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if path.exists() and read_json(path).get("sequence", 0) >= 2:
            break
        time.sleep(0.01)
    emitter.stop()
    emitter.raise_if_failed()

    sequence = read_json(path)["sequence"]
    time.sleep(0.03)
    assert sequence >= 2
    assert read_json(path)["sequence"] == sequence


def test_gpu_token_rejects_cross_run_overlap_without_importing_cuda(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import acquire_gpu_token

    with acquire_gpu_token(
        lease_root=tmp_path,
        gpu_uuid="GPU-test",
        run_id="run-a",
        stale_after_seconds=60,
    ) as first:
        first.heartbeat(stage="internal_cv")
        with pytest.raises(BlockingIOError):
            with acquire_gpu_token(
                lease_root=tmp_path,
                gpu_uuid="GPU-test",
                run_id="run-b",
                stale_after_seconds=60,
            ):
                pass


@pytest.mark.parametrize(
    ("owner_state", "heartbeat_age", "recoverable"),
    [
        ("dead", 121.0, True),
        ("dead", 30.0, False),
        ("alive", 121.0, False),
        ("unknown", 121.0, False),
    ],
)
def test_gpu_stale_recovery_requires_dead_owner_and_expired_heartbeat(
    tmp_path: Path,
    owner_state: str,
    heartbeat_age: float,
    recoverable: bool,
) -> None:
    from covid_audio_btp.hst_runtime import (
        ProcessLiveness,
        acquire_gpu_token,
        atomic_write_json,
        read_json,
    )

    with acquire_gpu_token(
        lease_root=tmp_path,
        gpu_uuid="GPU-stale",
        run_id="old-run",
        clock=lambda: 1_000.0,
    ) as old_lease:
        old_record = read_json(old_lease.record_path)

    old_record["heartbeat_at_unix"] = 2_000.0 - heartbeat_age
    old_record["pid"] = 987_654
    prefix, ticks = str(old_record["process_start_identity"]).rsplit(":", 1)
    old_record["process_start_identity"] = f"{prefix}:{int(ticks) + 1}"
    atomic_write_json(old_lease.record_path, old_record)

    operation = acquire_gpu_token(
        lease_root=tmp_path,
        gpu_uuid="GPU-stale",
        run_id="new-run",
        stale_after_seconds=120.0,
        clock=lambda: 2_000.0,
        process_probe=lambda identity: ProcessLiveness(owner_state),
    )
    if recoverable:
        with operation as lease:
            record = read_json(lease.record_path)
            assert record["run_id"] == "new-run"
            assert record["recovered_stale_token"] == old_record["token"]
    else:
        with pytest.raises(BlockingIOError):
            with operation:
                pass


def test_non_native_owner_identity_never_authorizes_stale_recovery(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import (
        ProcessLiveness,
        acquire_gpu_token,
        atomic_write_json,
        read_json,
    )

    with acquire_gpu_token(
        lease_root=tmp_path,
        gpu_uuid="GPU-legacy-owner",
        run_id="old-run",
        clock=lambda: 1_000.0,
    ) as old_lease:
        old_record = read_json(old_lease.record_path)

    old_record["heartbeat_at_unix"] = 1_000.0
    old_record["process_start_identity"] = "psutil-create-time:123.456000000"
    atomic_write_json(old_lease.record_path, old_record)

    with pytest.raises(BlockingIOError, match="unknown"):
        with acquire_gpu_token(
            lease_root=tmp_path,
            gpu_uuid="GPU-legacy-owner",
            run_id="new-run",
            stale_after_seconds=120.0,
            clock=lambda: 2_000.0,
            process_probe=lambda identity: ProcessLiveness.DEAD,
        ):
            pass


def test_gpu_token_release_removes_owner_record_and_allows_reuse(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import acquire_gpu_token

    with acquire_gpu_token(
        lease_root=tmp_path, gpu_uuid="GPU-reuse", run_id="run-a"
    ) as lease:
        record_path = lease.record_path
        assert record_path.exists()
    assert not record_path.exists()

    with acquire_gpu_token(
        lease_root=tmp_path, gpu_uuid="GPU-reuse", run_id="run-b"
    ) as second:
        assert second.run_id == "run-b"


def test_gpu_uuid_cannot_escape_lease_root(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import gpu_token_paths

    lock_path, record_path = gpu_token_paths(tmp_path, "../../GPU:unsafe/value")
    assert lock_path.parent == tmp_path.resolve()
    assert record_path.parent == tmp_path.resolve()
    assert ".." not in lock_path.name
    assert "/" not in lock_path.name
    assert "\\" not in lock_path.name


def test_run_lock_records_config_and_rejects_duplicate_live_owner(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import acquire_run_lock, read_json

    runtime_dir = tmp_path / "clone-a" / "runtime"
    stable_root = tmp_path / "stable-run-locks"
    with acquire_run_lock(
        runtime_dir,
        run_lock_root=stable_root,
        run_id="run-a",
        config_hash="config-abc",
        stale_after_seconds=60,
    ) as lease:
        record = read_json(lease.record_path)
        assert record["config_hash"] == "config-abc"
        assert record["host"] == lease.identity.host
        assert lease.lock_path.parent == stable_root.resolve()
        with pytest.raises(BlockingIOError):
            with acquire_run_lock(
                tmp_path / "clone-b" / "runtime",
                run_lock_root=stable_root,
                run_id="run-a",
                config_hash="config-abc",
                stale_after_seconds=60,
            ):
                pass


def test_source_tree_hash_is_order_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import source_tree_hash

    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 2\n", encoding="utf-8")
    first = source_tree_hash(tmp_path, ["b.py", "a.py"])
    second = source_tree_hash(tmp_path, ["a.py", "b.py"])
    (tmp_path / "b.py").write_text("B = 3\n", encoding="utf-8")
    changed = source_tree_hash(tmp_path, ["a.py", "b.py"])

    assert first == second
    assert changed != first
    with pytest.raises(ValueError, match="escape"):
        source_tree_hash(tmp_path, ["../outside.py"])


def test_stage_fingerprint_is_canonical_and_tracks_upstream_changes() -> None:
    from covid_audio_btp.hst_runtime import stage_fingerprint

    common = _internal_cv_fingerprint_inputs()
    first = stage_fingerprint("internal_cv", **common)
    reordered = dict(common)
    reordered["upstream_hashes"] = {
        "aligned_comparator": "comparator-stage",
        "base_resource_pilot": "pilot-stage",
    }
    second = stage_fingerprint("internal_cv", **reordered)
    changed = dict(common)
    changed["upstream_hashes"] = {
        "base_resource_pilot": "pilot-stage-changed",
        "aligned_comparator": "comparator-stage",
    }

    assert first == second
    assert stage_fingerprint("internal_cv", **changed) != first


def test_cpu_thread_environment_sets_all_native_thread_controls() -> None:
    from covid_audio_btp.hst_runtime import CPU_THREAD_ENV_VARS, cpu_thread_environment

    result = cpu_thread_environment(2, base_environment={"KEEP": "yes"})
    assert result["KEEP"] == "yes"
    assert {result[name] for name in CPU_THREAD_ENV_VARS} == {"2"}
    with pytest.raises(ValueError, match="positive"):
        cpu_thread_environment(0)


def test_cpu_thread_budget_applies_limiter_and_restores_environment() -> None:
    from covid_audio_btp.hst_runtime import CPU_THREAD_ENV_VARS, cpu_thread_budget

    environment = {"OMP_NUM_THREADS": "9", "KEEP": "yes"}
    calls: list[tuple[str, int]] = []

    class FakeLimiter:
        def __enter__(self) -> None:
            calls.append(("enter", 3))

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            calls.append(("exit", 3))

    def limiter_factory(*, limits: int) -> FakeLimiter:
        assert limits == 3
        return FakeLimiter()

    with pytest.raises(RuntimeError, match="inside budget"):
        with cpu_thread_budget(
            3,
            environment=environment,
            limiter_factory=limiter_factory,
        ):
            assert {environment[name] for name in CPU_THREAD_ENV_VARS} == {"3"}
            raise RuntimeError("inside budget")

    assert environment == {"OMP_NUM_THREADS": "9", "KEEP": "yes"}
    assert calls == [("enter", 3), ("exit", 3)]


def test_windows_gpu_root_is_stable_user_wide_and_never_pid_scoped(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import default_gpu_lease_root

    environment = {
        "LOCALAPPDATA": str(tmp_path / "local-app-data"),
        "PROGRAMDATA": str(tmp_path / "program-data"),
        "USERDOMAIN": "RESEARCH",
        "USERNAME": "covid",
    }
    first = default_gpu_lease_root(
        environment=environment,
        platform_name="nt",
        user_identity="RESEARCH\\covid",
    )
    second = default_gpu_lease_root(
        environment=environment,
        platform_name="nt",
        user_identity="RESEARCH\\covid",
    )

    expected = (tmp_path / "local-app-data" / "covid_audio_btp" / "hst_gpu").resolve()
    assert first == expected
    assert second == expected
    assert "/var/tmp" not in first.as_posix()
    assert str(os.getpid()) not in first.name


def test_windows_gpu_root_programdata_fallback_is_stable_per_user(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import default_gpu_lease_root

    environment = {"PROGRAMDATA": str(tmp_path / "program-data")}
    first = default_gpu_lease_root(
        environment=environment,
        platform_name="nt",
        user_identity="DOMAIN\\user-a",
    )
    repeated = default_gpu_lease_root(
        environment=environment,
        platform_name="nt",
        user_identity="DOMAIN\\user-a",
    )
    other_user = default_gpu_lease_root(
        environment=environment,
        platform_name="nt",
        user_identity="DOMAIN\\user-b",
    )

    assert first == repeated
    assert first != other_user
    assert first.is_relative_to((tmp_path / "program-data").resolve())
    assert "/var/tmp" not in first.as_posix()


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing violation behavior")
def test_atomic_json_retries_windows_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    path = tmp_path / "receipt.json"
    real_replace = runtime.os.replace
    attempts = 0

    def flaky_replace(source: object, destination: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", flaky_replace)
    runtime.atomic_write_json(path, {"status": "success"})

    assert attempts == 3
    assert runtime.read_json(path) == {"status": "success"}
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_gpu_lock_excludes_a_real_second_process(tmp_path: Path) -> None:
    import multiprocessing

    from covid_audio_btp.hst_runtime import acquire_gpu_token

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    result = context.Queue()
    process = context.Process(
        target=_hold_gpu_lease,
        args=(str(tmp_path), ready, release, result),
    )
    process.start()
    try:
        assert ready.wait(10), "child did not acquire GPU lease"
        with pytest.raises(BlockingIOError):
            with acquire_gpu_token(
                lease_root=tmp_path,
                gpu_uuid="GPU-cross-process",
                run_id="parent-run",
            ):
                pass
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)

    assert process.exitcode == 0
    assert result.get(timeout=2) == "success"


def test_run_lock_excludes_same_run_across_clone_directories(tmp_path: Path) -> None:
    import multiprocessing

    from covid_audio_btp.hst_runtime import acquire_run_lock

    stable_root = tmp_path / "stable-run-locks"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    result = context.Queue()
    process = context.Process(
        target=_hold_run_lease,
        args=(
            str(tmp_path / "clone-a" / "runtime"),
            str(stable_root),
            ready,
            release,
            result,
        ),
    )
    process.start()
    try:
        assert ready.wait(10), "child did not acquire run lease"
        with pytest.raises(BlockingIOError):
            with acquire_run_lock(
                tmp_path / "clone-b" / "runtime",
                run_lock_root=stable_root,
                run_id="shared-run",
                config_hash="config",
            ):
                pass
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)

    assert process.exitcode == 0
    assert result.get(timeout=2) == "success"


def test_launch_and_stage_receipts_are_atomic_complete_documents(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import (
        read_json,
        stable_file_sha256,
        write_launch_receipt,
        write_stage_receipt,
    )

    output = tmp_path / "run" / "metrics.csv"
    output.parent.mkdir(parents=True)
    output.write_text("metric,value\nauroc,0.9\n", encoding="utf-8")
    checksum = stable_file_sha256(output)
    launch_path = tmp_path / "launch.json"
    stage_path = tmp_path / "run" / "stages" / "internal_cv.json"

    write_launch_receipt(
        launch_path,
        launch_id="launch-1",
        status="success",
        fingerprint="launch-fingerprint",
        run_id="run-1",
        outputs={"metrics": "run/metrics.csv"},
        output_checksums={"metrics": checksum},
        started_at=10.0,
        finished_at=20.0,
    )
    write_stage_receipt(
        stage_path,
        run_id="run-1",
        stage="internal_cv",
        status="success",
        fingerprint="stage-fingerprint",
        outputs={"metrics": "metrics.csv"},
        output_checksums={"metrics": checksum},
        started_at=11.0,
        finished_at=19.0,
    )

    launch = read_json(launch_path)
    stage = read_json(stage_path)
    assert launch["receipt_type"] == "launch"
    assert launch["schema_version"] >= 1
    assert launch["fingerprint"] == "launch-fingerprint"
    assert launch["outputs"] == {"metrics": "run/metrics.csv"}
    assert launch["started_at_unix"] == 10.0
    assert launch["finished_at_unix"] == 20.0
    assert stage["receipt_type"] == "stage"
    assert stage["stage"] == "internal_cv"
    assert stage["output_checksums"] == {"metrics": checksum}
    assert stage["error"] is None


def test_releasing_lease_does_not_delete_runtime_receipts(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import (
        acquire_run_lock,
        write_launch_receipt,
        write_stage_receipt,
    )

    runtime_dir = tmp_path / "run" / "runtime"
    launch = runtime_dir / "launch.json"
    stage = runtime_dir / "stage.json"
    write_launch_receipt(
        launch,
        launch_id="launch",
        run_id="run",
        status="running",
        fingerprint="launch-fingerprint",
    )
    write_stage_receipt(
        stage,
        run_id="run",
        stage="preflight",
        status="running",
        fingerprint="stage-fingerprint",
    )
    with acquire_run_lock(
        runtime_dir,
        run_lock_root=tmp_path / "stable-run-locks",
        run_id="run",
        config_hash="config",
    ):
        pass

    assert launch.exists()
    assert stage.exists()


def _internal_cv_fingerprint_inputs() -> dict[str, object]:
    return {
        "input_hashes": {
            "coswara_source_contract": "coswara-source",
            "coughvid_source_contract": "coughvid-source",
        },
        "configuration_hash": "config",
        "executable_source_hash": "source",
        "dependency_lock_hash": "dependencies",
        "hst_commit": "hst-commit",
        "checkpoint_hashes": {"hst_base": "checkpoint"},
        "manifest_hashes": {"track_a": "manifest"},
        "upstream_hashes": {
            "base_resource_pilot": "pilot-stage",
            "aligned_comparator": "comparator-stage",
        },
        "accepted_hashes": {
            "pilot_freeze": "pilot",
            "data_contracts_freeze": "contracts",
            "environment_lock": "environment",
        },
        "pip_freeze_hash": "freeze",
    }


def test_raw_input_hashes_are_required_and_invalidate_stage_fingerprints() -> None:
    from covid_audio_btp.hst_runtime import STAGE_REQUIREMENTS, stage_fingerprint

    common = {
        "configuration_hash": "config",
        "executable_source_hash": "source",
        "dependency_lock_hash": "dependencies",
        "hst_commit": "hst-commit",
    }
    assert STAGE_REQUIREMENTS["preflight"].require_input_hashes
    assert STAGE_REQUIREMENTS["data_contracts"].require_input_hashes
    with pytest.raises(ValueError, match="input hashes"):
        stage_fingerprint("preflight", input_hashes={}, **common)

    first = stage_fingerprint(
        "preflight",
        input_hashes={"coswara_source_contract": "source-a"},
        **common,
    )
    second = stage_fingerprint(
        "preflight",
        input_hashes={"coswara_source_contract": "source-b"},
        **common,
    )
    assert first != second

    data_contract = stage_fingerprint(
        "data_contracts",
        input_hashes={"coswara_source_contract": "source-a"},
        upstream_hashes={"preflight": first},
        **common,
    )
    changed_data_contract = stage_fingerprint(
        "data_contracts",
        input_hashes={"coswara_source_contract": "source-b"},
        upstream_hashes={"preflight": second},
        **common,
    )
    assert data_contract != changed_data_contract

    downstream_inputs = _internal_cv_fingerprint_inputs()
    downstream = stage_fingerprint("internal_cv", **downstream_inputs)
    downstream_inputs["input_hashes"] = {
        "coswara_source_contract": "changed-source",
        "coughvid_source_contract": "coughvid-source",
    }
    assert stage_fingerprint("internal_cv", **downstream_inputs) != downstream


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("input_hashes", {}, "input hashes"),
        ("checkpoint_hashes", {}, "checkpoint"),
        ("manifest_hashes", {}, "manifest"),
        ("accepted_hashes", {}, "accepted"),
        ("upstream_hashes", {}, "upstream"),
        ("pip_freeze_hash", None, "pip freeze"),
    ],
)
def test_training_stage_fingerprint_rejects_omitted_required_inputs(
    field: str, replacement: object, message: str
) -> None:
    from covid_audio_btp.hst_runtime import stage_fingerprint

    inputs = _internal_cv_fingerprint_inputs()
    inputs[field] = replacement
    with pytest.raises(ValueError, match=message):
        stage_fingerprint("internal_cv", **inputs)


def test_aligned_comparator_requires_all_accepted_freeze_hashes() -> None:
    from covid_audio_btp.hst_runtime import stage_fingerprint

    with pytest.raises(ValueError, match="accepted"):
        stage_fingerprint(
            "aligned_comparator",
            input_hashes={"data_contract": "contract"},
            configuration_hash="config",
            executable_source_hash="source",
            dependency_lock_hash="dependencies",
            hst_commit="hst-commit",
            manifest_hashes={"track_a": "manifest"},
            upstream_hashes={"manifests": "manifest-stage"},
            accepted_hashes={},
            pip_freeze_hash="freeze",
        )


@pytest.mark.parametrize(
    ("output_path", "checksum"),
    [
        ("", "a" * 64),
        ("../escape.csv", "a" * 64),
        ("/absolute.csv", "a" * 64),
        ("metrics.csv", ""),
        ("metrics.csv", "not-a-sha256"),
        ("metrics.csv", "g" * 64),
    ],
)
def test_successful_receipt_rejects_invalid_output_paths_and_checksums(
    tmp_path: Path, output_path: str, checksum: str
) -> None:
    from covid_audio_btp.hst_runtime import write_stage_receipt

    with pytest.raises(ValueError, match="output|checksum|SHA-256|relative"):
        write_stage_receipt(
            tmp_path / "stage.json",
            run_id="run",
            stage="preflight",
            status="success",
            fingerprint="fingerprint",
            outputs={"artifact": output_path},
            output_checksums={"artifact": checksum},
        )


def test_successful_launch_receipt_uses_same_output_validation(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import write_launch_receipt

    with pytest.raises(ValueError, match="checksum|SHA-256"):
        write_launch_receipt(
            tmp_path / "launch.json",
            launch_id="launch",
            status="success",
            fingerprint="fingerprint",
            outputs={"artifact": "artifact.json"},
            output_checksums={"artifact": "placeholder"},
        )


def test_successful_receipt_rejects_error_payload(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import write_stage_receipt

    with pytest.raises(ValueError, match="error"):
        write_stage_receipt(
            tmp_path / "stage.json",
            run_id="run",
            stage="preflight",
            status="success",
            fingerprint="fingerprint",
            outputs={"artifact": "artifact.json"},
            output_checksums={"artifact": "a" * 64},
            error="contradictory failure",
        )


def test_receipt_lifecycle_timestamps_are_valid_and_preserved(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import write_stage_receipt

    path = tmp_path / "stage.json"
    running = write_stage_receipt(
        path,
        run_id="run",
        stage="preflight",
        status="running",
        fingerprint="fingerprint",
        started_at=10.0,
        timestamp=11.0,
    )
    assert running["finished_at_unix"] is None

    completed = write_stage_receipt(
        path,
        run_id="run",
        stage="preflight",
        status="success",
        fingerprint="fingerprint",
        outputs={"artifact": "artifact.json"},
        output_checksums={"artifact": "a" * 64},
        finished_at=20.0,
        timestamp=20.0,
    )
    assert completed["started_at_unix"] == 10.0
    assert completed["finished_at_unix"] == 20.0

    with pytest.raises(ValueError, match="finished"):
        write_stage_receipt(
            tmp_path / "running.json",
            run_id="run",
            stage="preflight",
            status="running",
            fingerprint="fingerprint",
            finished_at=12.0,
        )
    with pytest.raises(ValueError, match="before|started"):
        write_stage_receipt(
            tmp_path / "backward.json",
            run_id="run",
            stage="preflight",
            status="failed",
            fingerprint="fingerprint",
            started_at=20.0,
            finished_at=19.0,
        )
    with pytest.raises(ValueError, match="started"):
        write_stage_receipt(
            path,
            run_id="run",
            stage="preflight",
            status="success",
            fingerprint="fingerprint",
            outputs={"artifact": "artifact.json"},
            output_checksums={"artifact": "a" * 64},
            started_at=12.0,
            finished_at=20.0,
        )


def test_exit_receipt_rejects_empty_run_id(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import capture_process_identity, write_exit_receipt

    with pytest.raises(ValueError, match="run_id"):
        write_exit_receipt(
            tmp_path / "exit.json",
            identity=capture_process_identity(),
            run_id="",
            status="failed",
            exit_code=1,
        )


def test_successful_stage_receipt_requires_a_checksummed_output(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import write_stage_receipt

    with pytest.raises(ValueError, match="output"):
        write_stage_receipt(
            tmp_path / "stage.json",
            run_id="run",
            stage="preflight",
            status="success",
            fingerprint="fingerprint",
            outputs={},
            output_checksums={},
        )


def test_completed_stage_reuse_validates_fingerprint_and_output_checksums(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import (
        completed_stage_is_reusable,
        stable_file_sha256,
        validate_completed_stage_reuse,
        write_stage_receipt,
    )

    run_root = tmp_path / "run"
    output = run_root / "metrics" / "predictions.csv"
    output.parent.mkdir(parents=True)
    output.write_text("participant,probability\np1,0.8\n", encoding="utf-8")
    receipt_path = run_root / "runtime" / "internal_cv.json"
    write_stage_receipt(
        receipt_path,
        run_id="run",
        stage="internal_cv",
        status="success",
        fingerprint="expected",
        outputs={"predictions": "metrics/predictions.csv"},
        output_checksums={"predictions": stable_file_sha256(output)},
        started_at=1.0,
        finished_at=2.0,
    )

    validated = validate_completed_stage_reuse(
        receipt_path,
        expected_fingerprint="expected",
        run_root=run_root,
    )
    assert validated["status"] == "success"
    assert completed_stage_is_reusable(
        receipt_path,
        expected_fingerprint="expected",
        run_root=run_root,
    )

    output.write_text("participant,probability\np1,0.1\n", encoding="utf-8")
    assert not completed_stage_is_reusable(
        receipt_path,
        expected_fingerprint="expected",
        run_root=run_root,
    )
    with pytest.raises(Exception, match="checksum"):
        validate_completed_stage_reuse(
            receipt_path,
            expected_fingerprint="expected",
            run_root=run_root,
        )


def test_stage_dependency_hash_verifies_output_bytes_and_receipt_shape(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import (
        RuntimeStateError,
        stable_file_sha256,
        stage_receipt_dependency_hash,
        write_stage_receipt,
    )

    output = tmp_path / "outputs" / "pilot.json"
    output.parent.mkdir()
    output.write_text('{"pilot": 1}\n', encoding="utf-8")
    receipt = write_stage_receipt(
        tmp_path / "pilot-receipt.json",
        run_id="run",
        stage="base_resource_pilot",
        status="success",
        fingerprint="pilot-stage",
        outputs={"pilot": "outputs/pilot.json"},
        output_checksums={"pilot": stable_file_sha256(output)},
        started_at=1.0,
        finished_at=2.0,
    )
    first = stage_receipt_dependency_hash(receipt, run_root=tmp_path)
    assert len(first) == 64

    output.write_text('{"pilot": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeStateError, match="checksum"):
        stage_receipt_dependency_hash(receipt, run_root=tmp_path)

    malformed = dict(receipt)
    malformed["finished_at_unix"] = None
    with pytest.raises(RuntimeStateError, match="finished"):
        stage_receipt_dependency_hash(malformed, run_root=tmp_path)


def test_downstream_fingerprint_includes_verified_stage_receipt_outputs(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_runtime import (
        downstream_stage_fingerprint,
        stable_file_sha256,
        write_stage_receipt,
    )

    inputs = _internal_cv_fingerprint_inputs()
    pilot = tmp_path / "pilot.json"
    predictions = tmp_path / "predictions.csv"
    pilot.write_text('{"pilot": 1}\n', encoding="utf-8")
    predictions.write_text("participant,probability\np1,0.8\n", encoding="utf-8")
    upstream = {}
    for stage, path, name in (
        ("base_resource_pilot", pilot, "pilot"),
        ("aligned_comparator", predictions, "predictions"),
    ):
        upstream[stage] = write_stage_receipt(
            tmp_path / f"{stage}.json",
            run_id="run",
            stage=stage,
            status="success",
            fingerprint=f"{stage}-fingerprint",
            outputs={name: path.name},
            output_checksums={name: stable_file_sha256(path)},
            started_at=1.0,
            finished_at=2.0,
        )
    inputs.pop("upstream_hashes")
    first = downstream_stage_fingerprint(
        "internal_cv", run_root=tmp_path, upstream_receipts=upstream, **inputs
    )
    predictions.write_text("participant,probability\np1,0.2\n", encoding="utf-8")
    changed = {name: dict(receipt) for name, receipt in upstream.items()}
    changed["aligned_comparator"]["output_checksums"] = {
        "predictions": stable_file_sha256(predictions)
    }
    second = downstream_stage_fingerprint(
        "internal_cv", run_root=tmp_path, upstream_receipts=changed, **inputs
    )

    assert first != second


def test_verified_file_hash_detects_descriptor_metadata_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import covid_audio_btp.hst_runtime as runtime

    path = tmp_path / "source.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    real_fstat = runtime.os.fstat
    calls = 0

    def changing_fstat(file_descriptor: int) -> object:
        nonlocal calls
        calls += 1
        stat = real_fstat(file_descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_dev=stat.st_dev,
                st_ino=stat.st_ino,
                st_size=stat.st_size,
                st_mtime_ns=stat.st_mtime_ns + 1,
            )
        return stat

    monkeypatch.setattr(runtime.os, "fstat", changing_fstat)
    with pytest.raises(runtime.RuntimeStateError, match="changed while hashing"):
        runtime.stable_file_sha256(path)


def test_source_tree_uses_verified_file_hash_stat_result(tmp_path: Path) -> None:
    from covid_audio_btp.hst_runtime import (
        canonical_json_sha256,
        source_tree_hash,
        verified_file_sha256,
    )

    path = tmp_path / "source.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    verified = verified_file_sha256(path)
    expected = canonical_json_sha256(
        {
            "schema_version": 1,
            "files": [
                {
                    "path": "source.py",
                    "size_bytes": verified.size_bytes,
                    "sha256": verified.sha256,
                }
            ],
        }
    )
    assert source_tree_hash(tmp_path, ["source.py"]) == expected


def test_real_threadpoolctl_limit_and_environment_restoration() -> None:
    numpy = pytest.importorskip("numpy")
    threadpoolctl = pytest.importorskip("threadpoolctl")
    from covid_audio_btp.hst_runtime import CPU_THREAD_ENV_VARS, cpu_thread_budget

    numpy.dot(numpy.ones((32, 32)), numpy.ones((32, 32)))
    if not threadpoolctl.threadpool_info():
        pytest.skip("no native thread pool was loaded")
    before_environment = {name: os.environ.get(name) for name in CPU_THREAD_ENV_VARS}

    with cpu_thread_budget(1):
        pools = threadpoolctl.threadpool_info()
        assert pools
        assert all(int(pool["num_threads"]) <= 1 for pool in pools)
        assert {os.environ[name] for name in CPU_THREAD_ENV_VARS} == {"1"}

    assert {
        name: os.environ.get(name) for name in CPU_THREAD_ENV_VARS
    } == before_environment
