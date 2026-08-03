from __future__ import annotations

import json
import importlib.util
import os
import socket
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_audio_input_manifest_hash_changes_when_source_bytes_change(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_reliability import audio_input_manifest_sha256

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"first")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "recording_key,modality,audio_path\ncoswara::r1,cough,audio.wav\n",
        encoding="utf-8",
    )
    first = audio_input_manifest_sha256(
        metadata,
        project_root=tmp_path,
        modality="cough",
    )
    audio.write_bytes(b"second")
    second = audio_input_manifest_sha256(
        metadata,
        project_root=tmp_path,
        modality="cough",
    )

    assert len(first) == 64
    assert first != second


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "src" / "covid_audio_btp").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / "requirements-hst.txt").write_text("pandas==2.0\n", encoding="utf-8")
    (project / "src" / "covid_audio_btp" / "hst_demo.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (project / "scripts" / "72_run_hst_reliability.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )
    config = {
        "schema_version": 1,
        "source": {"commit": "7f94ad81e392da856c7aac6d364d036c28e26c32"},
        "checkpoints": {
            "hst_small_imagenet": {"sha256": "a" * 64},
            "hst_base_imagenet": {"sha256": "b" * 64},
        },
        "experiment": {"name": "test"},
    }
    config_path = project / "configs" / "hst_reliability.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return project, config_path


def test_smoke_preflight_builds_content_derived_run_without_cuda(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import run_preflight

    project, config = _project(tmp_path)
    result = run_preflight(
        config_path=config,
        project_root=project,
        mode="smoke",
        device="cpu",
        accepted_freezes_path=project / "reports" / "hst" / "accepted_freezes.json",
    )
    assert result["status"] == "ready"
    assert str(result["run_id"]).startswith("hst-")
    assert result["mode"] == "smoke"
    assert result["checks"]["configuration"] == "ok"


def test_full_preflight_fails_closed_without_accepted_freezes(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import run_preflight

    project, config = _project(tmp_path)
    result = run_preflight(
        config_path=config,
        project_root=project,
        mode="full",
        device="cuda",
        accepted_freezes_path=project / "reports" / "hst" / "accepted_freezes.json",
    )
    assert result["status"] == "blocked"
    assert any("accepted freeze" in error.casefold() for error in result["errors"])


def test_pilot_run_identity_binds_the_live_python_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_reliability as reliability

    project, config_path = _project(tmp_path)
    live_hash = "d" * 64
    monkeypatch.setattr(
        reliability,
        "capture_live_pip_freeze",
        lambda: (["package==1"], live_hash),
        raising=False,
    )

    config = reliability.load_controller_config(
        config_path=config_path,
        project_root=project,
        mode="pilot",
        device="cpu",
        accepted_freezes_path=project / "reports" / "hst" / "accepted_freezes.json",
    )

    assert config.pip_freeze_hash == live_hash


def test_full_controller_rejects_live_environment_different_from_accepted_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_reliability as reliability

    project, config_path = _project(tmp_path)
    accepted = project / "reports" / "hst" / "accepted_freezes.json"
    accepted.parent.mkdir(parents=True)
    accepted.write_text(
        json.dumps(
            {
                "approval_status": "manually_approved",
                "approved_by": "reviewer",
                "approved_at_utc": "2026-08-02T00:00:00+00:00",
                "accepted_hashes": {
                    "data_contracts_freeze": "a" * 64,
                    "pilot_freeze": "b" * 64,
                    "environment_lock": "c" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        reliability,
        "capture_live_pip_freeze",
        lambda: (["changed-package==2"], "d" * 64),
        raising=False,
    )

    with pytest.raises(ValueError, match="live Python environment"):
        reliability.load_controller_config(
            config_path=config_path,
            project_root=project,
            mode="full",
            device="cpu",
            accepted_freezes_path=accepted,
        )


def test_full_preflight_rejects_review_only_acceptance_candidate(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import run_preflight

    project, config = _project(tmp_path)
    candidate = project / "reports" / "hst" / "accepted_freezes.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        json.dumps(
            {
                "candidate_status": "requires_manual_review",
                "accepted_hashes": {
                    "data_contracts_freeze": "a" * 64,
                    "pilot_freeze": "b" * 64,
                    "environment_lock": "c" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_preflight(
        config_path=config,
        project_root=project,
        mode="full",
        device="cpu",
        accepted_freezes_path=candidate,
    )

    assert result["status"] == "blocked"
    assert any("manual approval" in error.casefold() for error in result["errors"])


def test_pilot_preflight_resolves_configured_sibling_hst_repository(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_reliability import run_preflight

    project, config_path = _project(tmp_path)
    workspace = project.parent
    hst_model = workspace / "HST" / "model" / "hst_model.py"
    hst_model.parent.mkdir(parents=True)
    hst_model.write_text("class HSTModel: pass\n", encoding="utf-8")
    checkpoint = project / ".cache" / "hst" / "checkpoints" / "hst_small_imagenet.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint.parent / "hst_base_imagenet.pth").write_bytes(b"base-checkpoint")
    metadata = project / "data" / "processed" / "metadata_with_quality.csv"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("participant_id,label_binary\np1,negative\n", encoding="utf-8")
    coughvid = project / "data" / "processed" / "coughvid_metadata_compare_is10_external.csv"
    coughvid.write_text("uuid,status_SSL\nc1,negative\n", encoding="utf-8")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["source"]["path"] = "../HST"
    config["paths"] = {
        "coswara_metadata": "data/processed/metadata_with_quality.csv",
        "checkpoint_directory": ".cache/hst/checkpoints",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = run_preflight(
        config_path=config_path,
        project_root=project,
        mode="pilot",
        device="cpu",
        accepted_freezes_path=project / "reports" / "hst" / "accepted_freezes.json",
    )

    assert result["status"] == "ready"
    assert result["checks"]["required_inputs"] == "ok"


def test_pilot_preflight_requires_external_metadata_for_shared_data_freeze(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_reliability import run_preflight

    project, config_path = _project(tmp_path)
    hst_model = project / "HST" / "model" / "hst_model.py"
    hst_model.parent.mkdir(parents=True)
    hst_model.write_text("class HSTModel: pass\n", encoding="utf-8")
    checkpoint_root = project / ".cache" / "hst" / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "hst_small_imagenet.pth").write_bytes(b"small")
    (checkpoint_root / "hst_base_imagenet.pth").write_bytes(b"base")
    metadata = project / "data" / "processed" / "metadata_with_quality.csv"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("participant_id,label_binary\np1,negative\n", encoding="utf-8")

    result = run_preflight(
        config_path=config_path,
        project_root=project,
        mode="pilot",
        device="cpu",
        accepted_freezes_path=project / "reports" / "hst" / "accepted_freezes.json",
    )

    assert result["status"] == "blocked"
    assert any("coughvid_metadata_compare_is10_external.csv" in error for error in result["errors"])


def test_pilot_preflight_requires_base_checkpoint_for_resource_probe(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_reliability import run_preflight

    project, config_path = _project(tmp_path)
    hst_model = project / "HST" / "model" / "hst_model.py"
    hst_model.parent.mkdir(parents=True)
    hst_model.write_text("class HSTModel: pass\n", encoding="utf-8")
    checkpoint = project / ".cache" / "hst" / "checkpoints" / "hst_small_imagenet.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"small")
    metadata = project / "data" / "processed" / "metadata_with_quality.csv"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("participant_id,label_binary\np1,negative\n", encoding="utf-8")

    result = run_preflight(
        config_path=config_path,
        project_root=project,
        mode="pilot",
        device="cpu",
        accepted_freezes_path=project / "reports" / "hst" / "accepted_freezes.json",
    )

    assert result["status"] == "blocked"
    assert any("hst_base_imagenet.pth" in error for error in result["errors"])


def test_controller_command_uses_module_script_and_all_frozen_arguments(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import _detached_command

    project, config = _project(tmp_path)
    command = _detached_command(
        project_root=project,
        config_path=config,
        accepted_freezes_path=project / "freezes.json",
        mode="full",
        device="cuda",
        through="evidence_pack",
        expected_run_id="hst-expected",
        launch_id="launch-123",
        resume=False,
        force_stage=("spectrogram_cache", "internal_cv"),
    )
    assert command[1].endswith("72_run_hst_reliability.py")
    launch_option = command.index("--launch-id")
    assert command[launch_option : launch_option + 2] == ["--launch-id", "launch-123"]
    assert "--expected-run-id" in command
    assert "hst-expected" in command
    assert "--no-resume" in command
    assert command.count("--force-stage") == 2
    assert command[command.index("--force-stage") + 1] == "spectrogram_cache"
    second_force = command.index("--force-stage", command.index("--force-stage") + 1)
    assert command[second_force + 1] == "internal_cv"


def test_cpu_and_cuda_controller_runs_use_distinct_output_roots(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import HSTPipeline, HSTPipelineConfig

    cpu = HSTPipeline(HSTPipelineConfig.smoke(tmp_path, device="cpu"))
    cuda = HSTPipeline(HSTPipelineConfig.smoke(tmp_path, device="cuda"))

    assert cpu.run_id != cuda.run_id
    assert cpu.run_root != cuda.run_root


def test_detached_parent_writes_intent_before_spawn_then_publishes_child_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_reliability as reliability

    project, config_path = _project(tmp_path)
    monkeypatch.setattr(
        reliability,
        "run_preflight",
        lambda **_kwargs: {"status": "ready", "run_id": "hst-run"},
    )

    class FakeProcess:
        pid = 4321

    def fake_popen(*_args: object, **_kwargs: object) -> FakeProcess:
        receipts = list((project / "reports" / "hst" / "launches").glob("*.json"))
        assert len(receipts) == 1
        intent = json.loads(receipts[0].read_text(encoding="utf-8"))
        assert intent["status"] == "initializing"
        assert intent["pid"] is None
        return FakeProcess()

    monkeypatch.setattr(reliability.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        reliability,
        "capture_process_identity",
        lambda pid=None: SimpleNamespace(
            pid=4321,
            host=socket.gethostname(),
            start_identity="test-start-identity",
        ),
    )

    launched = reliability.launch_detached_run(
        config_path=config_path,
        project_root=project,
        mode="smoke",
        device="cpu",
        through="preflight",
        accepted_freezes_path=project / "reports" / "hst" / "accepted_freezes.json",
    )

    persisted = json.loads(
        (
            project
            / "reports"
            / "hst"
            / "launches"
            / f"{launched['launch_id']}.json"
        ).read_text(encoding="utf-8")
    )
    assert launched["status"] == "launching"
    assert persisted["status"] == "launching"
    assert persisted["pid"] == 4321
    assert persisted["process_start_identity"] == "test-start-identity"


def test_detached_child_waits_for_matching_parent_identity(tmp_path: Path) -> None:
    import covid_audio_btp.hst_reliability as reliability

    project, _ = _project(tmp_path)
    launch_root = project / "reports" / "hst" / "launches"
    launch_root.mkdir(parents=True, exist_ok=True)
    path = launch_root / "launch-abc.json"
    path.write_text(
        json.dumps(
            {
                "launch_id": "launch-abc",
                "status": "launching",
                "pid": 4321,
                "host": socket.gethostname(),
                "process_start_identity": "test-start-identity",
            }
        ),
        encoding="utf-8",
    )

    receipt = reliability.wait_for_parent_launch_initialization(
        project_root=project,
        launch_id="launch-abc",
        child_identity=SimpleNamespace(
            pid=4321,
            host=socket.gethostname(),
            start_identity="test-start-identity",
        ),
        timeout_seconds=1.0,
    )

    assert receipt["pid"] == 4321


def test_read_run_status_rejects_path_escape_and_returns_receipt(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import read_run_status

    project, _ = _project(tmp_path)
    launch_root = project / "reports" / "hst" / "launches"
    launch_root.mkdir(parents=True)
    receipt = {"launch_id": "launch-abc", "status": "running", "stage": "internal_cv"}
    (launch_root / "launch-abc.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert read_run_status(project_root=project, status_id="launch-abc") == receipt
    with pytest.raises(ValueError, match="status_id"):
        read_run_status(project_root=project, status_id="../escape")


def test_child_status_update_preserves_immutable_launch_metadata(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import update_detached_run_status

    project = tmp_path / "project"
    path = project / "reports" / "hst" / "launches" / "launch-abc.json"
    path.parent.mkdir(parents=True)
    absolute_heartbeat = (
        project / "data" / "outputs" / "hst" / "run-1" / "runtime" / "heartbeat.json"
    )
    immutable = {
        "pid": 321,
        "command": ["python", "run.py"],
        "log_path": "logs/hst/launch-abc.log",
        "host": "worker-host",
        "process_start_identity": "linux:boot-id:99",
        "heartbeat_path": "data/outputs/hst/run-1/runtime/heartbeat.json",
        "launched_at": "2026-08-02T10:00:00+00:00",
        "launched_at_unix": 100.0,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "launch_id": "launch-abc",
                "run_id": "run-1",
                "status": "launching",
                "stage": "preflight",
                "error": None,
                "updated_at": "2026-08-02T10:00:00+00:00",
                "updated_at_unix": 100.0,
                **immutable,
            }
        ),
        encoding="utf-8",
    )

    updated = update_detached_run_status(
        path,
        launch_id="launch-abc",
        run_id="run-1",
        status="running",
        stage="internal_cv",
        heartbeat_path=absolute_heartbeat,
        timestamp=125.0,
    )

    assert {name: updated[name] for name in immutable} == immutable
    assert updated["status"] == "running"
    assert updated["stage"] == "internal_cv"
    assert updated["updated_at_unix"] == 125.0


@pytest.mark.parametrize("failure_mode", ["dead", "stale"])
def test_read_run_status_detects_dead_or_stale_detached_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    import covid_audio_btp.hst_reliability as reliability
    from covid_audio_btp.hst_runtime import ProcessLiveness

    project, _ = _project(tmp_path)
    launch_root = project / "reports" / "hst" / "launches"
    launch_root.mkdir(parents=True)
    heartbeat = project / "data" / "outputs" / "hst" / "run-1" / "runtime" / "heartbeat.json"
    heartbeat.parent.mkdir(parents=True)
    heartbeat.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "pid": 321,
                "host": socket.gethostname(),
                "process_start_identity": "linux:boot-id:99",
                "heartbeat_at_unix": 100.0,
            }
        ),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 2,
        "launch_id": "launch-abc",
        "run_id": "run-1",
        "status": "running",
        "stage": "internal_cv",
        "pid": 321,
        "host": socket.gethostname(),
        "process_start_identity": "linux:boot-id:99",
        "heartbeat_path": heartbeat.relative_to(project).as_posix(),
        "updated_at_unix": 100.0,
        "updated_at": "2026-08-02T10:00:00+00:00",
    }
    (launch_root / "launch-abc.json").write_text(json.dumps(receipt), encoding="utf-8")
    liveness = ProcessLiveness.DEAD if failure_mode == "dead" else ProcessLiveness.ALIVE
    monkeypatch.setattr(reliability, "process_identity_liveness", lambda identity: liveness)

    status = reliability.read_run_status(
        project_root=project,
        status_id="launch-abc",
        stale_after_seconds=60.0,
        now=200.0,
    )

    assert status["status"] == "failed"
    assert status["stored_status"] == "running"
    assert status["monitor_status"] == failure_mode
    assert "detached" in str(status["error"]).lower()


def test_read_run_status_rejects_heartbeat_from_another_process_identity(
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_reliability as reliability

    project, _ = _project(tmp_path)
    launch_root = project / "reports" / "hst" / "launches"
    launch_root.mkdir(parents=True)
    heartbeat = project / "data" / "outputs" / "hst" / "run-1" / "runtime" / "heartbeat.json"
    heartbeat.parent.mkdir(parents=True)
    heartbeat.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "pid": 999,
                "host": socket.gethostname(),
                "process_start_identity": "other-start",
                "heartbeat_at_unix": 100.0,
            }
        ),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 2,
        "launch_id": "launch-abc",
        "run_id": "run-1",
        "status": "running",
        "stage": "internal_cv",
        "pid": 321,
        "command": ["python"],
        "log_path": "logs/hst/launch-abc.log",
        "host": socket.gethostname(),
        "process_start_identity": "expected-start",
        "heartbeat_path": heartbeat.relative_to(project).as_posix(),
        "launched_at": "2026-08-02T10:00:00+00:00",
        "launched_at_unix": 100.0,
        "updated_at_unix": 100.0,
        "updated_at": "2026-08-02T10:00:00+00:00",
    }
    (launch_root / "launch-abc.json").write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="process identity"):
        reliability.read_run_status(
            project_root=project,
            status_id="launch-abc",
            stale_after_seconds=60.0,
            now=120.0,
        )


def test_wait_for_detached_run_has_a_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import covid_audio_btp.hst_reliability as reliability

    monotonic_now = [0.0]
    calls = []

    def fake_read_run_status(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"launch_id": "launch-abc", "status": "running", "stage": "internal_cv"}

    def fake_sleep(seconds: float) -> None:
        monotonic_now[0] += seconds

    monkeypatch.setattr(reliability, "read_run_status", fake_read_run_status)

    with pytest.raises(TimeoutError, match="launch-abc"):
        reliability.wait_for_detached_run(
            project_root=tmp_path,
            status_id="launch-abc",
            poll_interval_seconds=2.0,
            stale_after_seconds=60.0,
            timeout_seconds=5.0,
            _monotonic=lambda: monotonic_now[0],
            _sleep=fake_sleep,
        )

    assert monotonic_now[0] == 5.0
    assert len(calls) == 4


def test_read_hst_run_progress_reports_only_durable_checkpointed_work(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_reliability import read_hst_run_progress
    from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256

    project, _ = _project(tmp_path)
    run_id = "hst-progress-run"
    run_root = project / "data" / "outputs" / "hst" / run_id
    stage_root = run_root / "runtime" / "stages"
    stage_root.mkdir(parents=True)

    stage_outputs: dict[str, Path] = {}
    for stage in ("preflight", "data_contracts"):
        stage_output = run_root / "runtime" / "progress-fixtures" / f"{stage}.txt"
        stage_output.parent.mkdir(parents=True, exist_ok=True)
        stage_output.write_text(f"{stage}-complete", encoding="utf-8")
        relative_output = stage_output.relative_to(run_root).as_posix()
        stage_outputs[stage] = stage_output
        receipt = {
            "schema_version": 1,
            "receipt_type": "hst_stage",
            "status": "success",
            "run_id": run_id,
            "stage": stage,
            "output_paths": [relative_output],
            "output_checksums": {
                relative_output: stable_file_sha256(stage_output),
            },
        }
        receipt["record_hash"] = canonical_json_sha256(receipt)
        (stage_root / f"{stage}.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )

    scientific = run_root / "scientific" / "internal_cv"
    jobs_root = scientific / "jobs"
    completed_root = jobs_root / "internal-completed"
    active_root = jobs_root / "internal-active"
    completed_root.mkdir(parents=True)
    completed_output = completed_root / "summary.json"
    completed_output.write_text('{"status":"complete"}', encoding="utf-8")
    large_output = completed_root / "model.pt"
    large_output.write_bytes(b"a" * (16 * 1024 * 1024 + 1))
    (active_root / "training").mkdir(parents=True)
    plan_frame = pd.DataFrame(
        [
            {
                "job_id": "internal-completed",
                "job_spec_sha256": "1" * 64,
                "fold": 1,
                "modality": "cough",
                "protocol": "track_a",
            },
            {
                "job_id": "internal-active",
                "job_spec_sha256": "2" * 64,
                "fold": 2,
                "modality": "speech",
                "protocol": "track_a",
            },
            *[
                {
                    "job_id": f"internal-pending-{index}",
                    "job_spec_sha256": f"{index:064x}",
                    "fold": ((index - 1) % 10) + 1,
                    "modality": ("cough", "speech", "breath")[index % 3],
                    "protocol": "track_a",
                }
                for index in range(3, 41)
            ],
        ]
    )
    plan_frame.to_csv(scientific / "job_plan.csv", index=False)

    for job_root, status, job_id, job_hash, fold, modality in (
        (completed_root, "success", "internal-completed", "1" * 64, 1, "cough"),
        (active_root, "running", "internal-active", "2" * 64, 2, "speech"),
    ):
        receipt = {
            "schema_version": 1,
            "receipt_type": "hst_scientific_job",
            "status": status,
            "run_id": run_id,
            "job_id": job_id,
            "job_spec_sha256": job_hash,
            "attempt": 1,
        }
        if status == "running":
            receipt["job"] = {
                "stage": "internal_cv",
                "fold": fold,
                "seed": 52,
                "modality": modality,
                "protocol": "track_a",
            }
        if status == "success":
            receipt["outputs"] = [
                {
                    "path": completed_output.relative_to(scientific).as_posix(),
                    "size_bytes": completed_output.stat().st_size,
                    "sha256": stable_file_sha256(completed_output),
                },
                {
                    "path": large_output.relative_to(scientific).as_posix(),
                    "size_bytes": large_output.stat().st_size,
                    "sha256": stable_file_sha256(large_output),
                },
            ]
        receipt["record_hash"] = canonical_json_sha256(receipt)
        (job_root / "job_receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )

    training_root = active_root / "training"
    generation_root = training_root / ".last.pt.generations" / "generation-1"
    generation_root.mkdir(parents=True)
    checkpoint = generation_root / "checkpoint.pt"
    checkpoint.write_bytes(b"durable-checkpoint")
    checkpoint_record = {
        "generation": "generation-1",
        "checkpoint_path": checkpoint.relative_to(training_root).as_posix(),
        "sidecar_path": ".last.pt.generations/generation-1/checkpoint.pt.sha256.json",
        "sha256": stable_file_sha256(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
    }
    sidecar = generation_root / "checkpoint.pt.sha256.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "writer": "covid_audio_btp.hst_training._atomic_torch_save",
                "filename": "checkpoint.pt",
                "size_bytes": checkpoint.stat().st_size,
                "sha256": stable_file_sha256(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    pointer = {
        "schema_version": 2,
        "writer": "covid_audio_btp.hst_training._atomic_torch_save",
        "logical_name": "last.pt",
        "current": checkpoint_record,
        "previous": None,
    }
    pointer_path = training_root / "last.pt.current.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    progress = {
        "schema_version": 1,
        "receipt_type": "hst_training_progress",
        "status": "checkpointed",
        "run_id": run_id,
        "stage": "internal_cv",
        "job_id": "internal-active",
        "job_spec_sha256": "2" * 64,
        "fold": 2,
        "seed": 52,
        "modality": "speech",
        "protocol": "track_a",
        "completed_epoch": 24,
        "resume_epoch": 25,
        "next_consumed_batch_index": 50,
        "epoch_batch_count": 100,
        "completed_optimizer_boundaries": 25,
        "epoch_optimizer_boundary_count": 50,
        "max_epochs": 100,
        "checkpoint_reason": "wall_clock_interval",
        "checkpoint_resume_safe": True,
        "checkpoint_pointer_path": pointer_path.relative_to(training_root).as_posix(),
        "checkpoint_pointer_sha256": stable_file_sha256(pointer_path),
        "checkpoint": checkpoint_record,
        "updated_at_unix": 100.0,
    }
    progress["record_hash"] = canonical_json_sha256(progress)
    (training_root / "training_progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )

    observed = read_hst_run_progress(
        project_root=project,
        run_id=run_id,
        through="evidence_pack",
    )

    assert observed["pipeline_stages"] == {
        "completed": 2,
        "total": 17,
        "percent": pytest.approx(11.7647058824),
    }
    assert observed["confirmatory_training"]["completed_jobs"] == 1
    assert observed["confirmatory_training"]["total_jobs"] == 50
    assert observed["confirmatory_training"]["durable_job_equivalents"] == pytest.approx(
        1.245
    )
    assert observed["confirmatory_training"]["percent"] == pytest.approx(2.49)
    assert observed["current_job"]["job_id"] == "internal-active"
    assert observed["current_job"]["fold"] == 2
    assert observed["current_job"]["modality"] == "speech"
    assert observed["current_job"]["epoch_percent"] == pytest.approx(24.5)
    assert observed["current_job"]["checkpoint_resume_safe"] is True
    assert observed["current_job"]["checkpointed"] is True
    assert observed["current_job"]["checkpoint_generation"] == "generation-1"
    assert observed["current_job"]["checkpoint_path"] == checkpoint_record[
        "checkpoint_path"
    ]
    assert observed["current_job"]["checkpoint_sha256"] == stable_file_sha256(
        checkpoint
    )

    mismatched_progress = dict(progress)
    mismatched_progress["modality"] = "cough"
    mismatched_progress.pop("record_hash")
    mismatched_progress["record_hash"] = canonical_json_sha256(mismatched_progress)
    (training_root / "training_progress.json").write_text(
        json.dumps(mismatched_progress), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="progress identity.*frozen plan"):
        read_hst_run_progress(
            project_root=project,
            run_id=run_id,
            through="evidence_pack",
        )
    (training_root / "training_progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )

    progress_path = training_root / "training_progress.json"
    progress_contents = progress_path.read_text(encoding="utf-8")
    progress_path.unlink()
    before_first_checkpoint = read_hst_run_progress(
        project_root=project,
        run_id=run_id,
        through="evidence_pack",
    )
    assert before_first_checkpoint["current_job"]["job_id"] == "internal-active"
    assert before_first_checkpoint["current_job"]["fold"] == 2
    assert before_first_checkpoint["current_job"]["modality"] == "speech"
    assert before_first_checkpoint["current_job"]["checkpointed"] is False
    assert before_first_checkpoint["current_job"]["epoch_percent"] == 0.0
    progress_path.write_text(progress_contents, encoding="utf-8")

    completed_output.unlink()
    with pytest.raises((FileNotFoundError, ValueError), match="[Oo]utput.*missing"):
        read_hst_run_progress(
            project_root=project,
            run_id=run_id,
            through="evidence_pack",
        )
    completed_output.write_text('{"status":"complete"}', encoding="utf-8")
    completed_output.write_text('{"status":"tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="[Oo]utput checksum changed"):
        read_hst_run_progress(
            project_root=project,
            run_id=run_id,
            through="evidence_pack",
        )
    completed_output.write_text('{"status":"complete"}', encoding="utf-8")

    original_large_mtime = large_output.stat().st_mtime_ns
    large_output.write_bytes(b"b" + b"a" * (16 * 1024 * 1024))
    os.utime(
        large_output,
        ns=(original_large_mtime + 1_000_000, original_large_mtime + 1_000_000),
    )
    with pytest.raises(ValueError, match="[Oo]utput checksum changed"):
        read_hst_run_progress(
            project_root=project,
            run_id=run_id,
            through="evidence_pack",
        )
    large_output.write_bytes(b"a" * (16 * 1024 * 1024 + 1))

    stage_outputs["preflight"].unlink()
    with pytest.raises((FileNotFoundError, ValueError), match="[Oo]utput.*missing"):
        read_hst_run_progress(
            project_root=project,
            run_id=run_id,
            through="evidence_pack",
        )
    stage_outputs["preflight"].write_text("preflight-complete", encoding="utf-8")

    checkpoint.write_bytes(b"corrupt-checkpoint")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        read_hst_run_progress(
            project_root=project,
            run_id=run_id,
            through="evidence_pack",
        )
    checkpoint.write_bytes(b"durable-checkpoint")

    plan_frame.iloc[:-1].to_csv(scientific / "job_plan.csv", index=False)
    with pytest.raises(ValueError, match="exact frozen internal_cv budget"):
        read_hst_run_progress(
            project_root=project,
            run_id=run_id,
            through="evidence_pack",
        )
    plan_frame.to_csv(scientific / "job_plan.csv", index=False)

    second_running_root = jobs_root / "internal-pending-3"
    second_running_root.mkdir()
    second_running_receipt = {
        "schema_version": 1,
        "receipt_type": "hst_scientific_job",
        "status": "running",
        "run_id": run_id,
        "job_id": "internal-pending-3",
        "job_spec_sha256": f"{3:064x}",
        "attempt": 1,
        "job": {
            "stage": "internal_cv",
            "fold": 3,
            "seed": 52,
            "modality": "cough",
            "protocol": "track_a",
        },
    }
    second_running_receipt["record_hash"] = canonical_json_sha256(
        second_running_receipt
    )
    second_running_receipt_path = second_running_root / "job_receipt.json"
    second_running_receipt_path.write_text(
        json.dumps(second_running_receipt), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Multiple confirmatory training jobs"):
        read_hst_run_progress(
            project_root=project,
            run_id=run_id,
            through="evidence_pack",
        )
    second_running_receipt_path.unlink()
    second_running_root.rmdir()

    newer_root = training_root / ".last.pt.generations" / "generation-2"
    newer_root.mkdir(parents=True)
    newer_checkpoint = newer_root / "checkpoint.pt"
    newer_checkpoint.write_bytes(b"newer-durable-checkpoint")
    newer_sidecar = newer_root / "checkpoint.pt.sha256.json"
    newer_sidecar.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "writer": "covid_audio_btp.hst_training._atomic_torch_save",
                "filename": "checkpoint.pt",
                "size_bytes": newer_checkpoint.stat().st_size,
                "sha256": stable_file_sha256(newer_checkpoint),
            }
        ),
        encoding="utf-8",
    )
    newer_record = {
        "generation": "generation-2",
        "checkpoint_path": newer_checkpoint.relative_to(training_root).as_posix(),
        "sidecar_path": newer_sidecar.relative_to(training_root).as_posix(),
        "sha256": stable_file_sha256(newer_checkpoint),
        "size_bytes": newer_checkpoint.stat().st_size,
    }
    pointer["previous"] = checkpoint_record
    pointer["current"] = newer_record
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    recovered = read_hst_run_progress(
        project_root=project,
        run_id=run_id,
        through="evidence_pack",
    )

    assert recovered["current_job"]["checkpoint_sha256"] == stable_file_sha256(
        checkpoint
    )
    assert recovered["current_job"]["epoch_percent"] == pytest.approx(24.5)

    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint generation"):
        read_hst_run_progress(
            project_root=project,
            run_id=run_id,
            through="evidence_pack",
        )


def test_find_resumable_detached_run_matches_exact_frozen_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_reliability as reliability
    from covid_audio_btp.hst_runtime import ProcessLiveness

    project, _ = _project(tmp_path)
    launch_root = project / "reports" / "hst" / "launches"
    launch_root.mkdir(parents=True)
    receipt = {
        "schema_version": 2,
        "launch_id": "launch-match",
        "run_id": "run-1",
        "status": "running",
        "stage": "internal_cv",
        "pid": 321,
        "command": [
            "python",
            "scripts/72_run_hst_reliability.py",
            "--mode", "full",
            "--device", "cuda",
            "--through", "evidence_pack",
        "--expected-run-id", "run-1",
        "--launch-id", "launch-match",
        "--no-resume",
        "--force-stage", "spectrogram_cache",
        "--force-stage", "internal_cv",
        ],
        "log_path": "logs/hst/launch-match.log",
        "host": socket.gethostname(),
        "process_start_identity": "linux:boot-id:99",
        "heartbeat_path": None,
        "launched_at": "2026-08-02T10:00:00+00:00",
        "launched_at_unix": 100.0,
        "updated_at": "2026-08-02T10:00:00+00:00",
        "updated_at_unix": 100.0,
    }
    (launch_root / "launch-match.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        reliability,
        "process_identity_liveness",
        lambda _identity: ProcessLiveness.ALIVE,
    )

    found = reliability.find_resumable_detached_run(
        project_root=project,
        run_id="run-1",
        mode="full",
        device="cuda",
        through="evidence_pack",
        expected_run_id="run-1",
        resume=False,
        force_stage=("internal_cv", "spectrogram_cache"),
        now=110.0,
    )

    assert found is not None
    assert found["launch_id"] == "launch-match"
    assert reliability.find_resumable_detached_run(
        project_root=project,
        run_id="run-1",
        mode="full",
        device="cuda",
        through="evidence_pack",
        expected_run_id="run-1",
        resume=True,
        force_stage=("internal_cv", "spectrogram_cache"),
        now=110.0,
    ) is None
    assert reliability.find_resumable_detached_run(
        project_root=project,
        run_id="run-1",
        mode="full",
        device="cuda",
        through="aligned_comparator",
        expected_run_id="run-1",
        resume=False,
        force_stage=("internal_cv", "spectrogram_cache"),
        now=110.0,
    ) is None


def test_find_resumable_detached_run_rejects_ambiguous_active_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_reliability as reliability

    project, _ = _project(tmp_path)
    launch_root = project / "reports" / "hst" / "launches"
    launch_root.mkdir(parents=True)
    command = [
        "python", "scripts/72_run_hst_reliability.py",
        "--mode", "pilot",
        "--device", "cuda",
        "--through", "base_resource_pilot",
        "--expected-run-id", "auto",
    ]
    for launch_id in ("launch-one", "launch-two"):
        (launch_root / f"{launch_id}.json").write_text(
            json.dumps(
                {
                    "launch_id": launch_id,
                    "run_id": "run-pilot",
                    "status": "running",
                    "command": [*command, "--launch-id", launch_id],
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        reliability,
        "read_run_status",
        lambda **kwargs: json.loads(
            (launch_root / f"{kwargs['status_id']}.json").read_text(encoding="utf-8")
        ),
    )

    with pytest.raises(RuntimeError, match="Multiple active detached HST launches"):
        reliability.find_resumable_detached_run(
            project_root=project,
            run_id="run-pilot",
            mode="pilot",
            device="cuda",
            through="base_resource_pilot",
            expected_run_id="auto",
        )


def test_heartbeat_failure_prevents_success_and_latest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "72_run_hst_reliability.py"
    spec = importlib.util.spec_from_file_location("hst_controller_under_test", script_path)
    assert spec is not None and spec.loader is not None
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)

    events: list[str] = []
    statuses: list[str] = []

    class FakeHeartbeat:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> None:
            events.append("heartbeat.start")

        def stop(self) -> None:
            events.append("heartbeat.stop")

        def raise_if_failed(self) -> None:
            events.append("heartbeat.raise_if_failed")
            raise RuntimeError("heartbeat writer failed")

    class FakePipeline:
        STAGES = ("evidence_pack",)

        def __init__(self, config: object, *, stage_handlers: object) -> None:
            self.runtime_root = tmp_path / "runtime"
            self.run_root = tmp_path / "run"
            self.run_id = "run-1"
            self.configuration_hash = "a" * 64
            self.stage_hook = None

        def run(self, **kwargs: object) -> pd.DataFrame:
            events.append("pipeline.run")
            return pd.DataFrame([{"stage": "evidence_pack"}])

    args = SimpleNamespace(
        config=Path("config.json"),
        project_root=tmp_path,
        accepted_freezes=Path("accepted.json"),
        mode="full",
        device="cpu",
        through="evidence_pack",
        expected_run_id="auto",
        force_stage=[],
        resume=True,
        detach=False,
        status_id=None,
        launch_id="launch-abc",
    )
    config = SimpleNamespace(resume=True, device="cpu")
    monkeypatch.setattr(controller, "parse_args", lambda: args)
    monkeypatch.setattr(
        controller,
        "capture_process_identity",
        lambda: SimpleNamespace(
            pid=321,
            host=socket.gethostname(),
            start_identity="linux:boot-id:99",
        ),
    )
    monkeypatch.setattr(
        controller,
        "wait_for_parent_launch_initialization",
        lambda **kwargs: events.append("launch.handshake"),
    )
    monkeypatch.setattr(controller, "run_preflight", lambda **kwargs: {"status": "ready"})
    monkeypatch.setattr(controller, "load_controller_config", lambda **kwargs: config)
    monkeypatch.setattr(controller, "HSTPipeline", FakePipeline)
    monkeypatch.setattr(controller, "build_scientific_stage_handlers", lambda config: {})
    monkeypatch.setattr(controller, "HeartbeatEmitter", FakeHeartbeat)
    monkeypatch.setattr(controller, "acquire_run_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        controller,
        "publish_hst_latest",
        lambda **kwargs: events.append("publish.latest"),
    )
    monkeypatch.setattr(
        controller,
        "_write_status",
        lambda path, **kwargs: statuses.append(str(kwargs["status"])),
    )

    with pytest.raises(RuntimeError, match="heartbeat writer failed"):
        controller.main()

    assert statuses == ["running", "failed"]
    assert "success" not in statuses
    assert "publish.latest" not in events
    assert events.index("launch.handshake") < events.index("pipeline.run")
    assert events.index("heartbeat.stop") < events.index("heartbeat.raise_if_failed")


def test_cuda_lease_selector_tracks_the_first_visible_physical_device() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "72_run_hst_reliability.py"
    spec = importlib.util.spec_from_file_location("hst_cuda_selector_under_test", script_path)
    assert spec is not None and spec.loader is not None
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)

    assert controller._cuda_device_selector({}) == "0"
    assert controller._cuda_device_selector({"CUDA_VISIBLE_DEVICES": "2,3"}) == "2"
    assert controller._cuda_device_selector(
        {"CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}
    ).startswith("GPU-")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        controller._cuda_device_selector({"CUDA_VISIBLE_DEVICES": "-1"})


def test_controller_source_allowlist_rejects_unlisted_local_import(tmp_path: Path) -> None:
    from covid_audio_btp.hst_reliability import _controller_source_paths

    project, _ = _project(tmp_path)
    package = project / "src" / "covid_audio_btp"
    (package / "hst_demo.py").write_text(
        "from .unlisted_runtime_helper import VALUE\n",
        encoding="utf-8",
    )
    (package / "unlisted_runtime_helper.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unlisted local executable import"):
        _controller_source_paths(project)


def test_controller_source_allowlist_covers_runtime_closure_and_manual_gates() -> None:
    from covid_audio_btp.hst_reliability import _controller_source_paths

    project = Path(__file__).resolve().parents[1]
    relative = {
        path.relative_to(project).as_posix()
        for path in _controller_source_paths(project)
    }

    assert {
        "src/covid_audio_btp/config.py",
        "src/covid_audio_btp/features.py",
        "src/covid_audio_btp/fusion.py",
        "src/covid_audio_btp/metadata_baseline.py",
        "src/covid_audio_btp/metadata_confounding.py",
        "src/covid_audio_btp/preprocess.py",
        "src/covid_audio_btp/temporal_holdout.py",
        "scripts/76_prepare_hst_comparator_approval.py",
        "scripts/77_prepare_hst_comparator_generation_acceptance.py",
        "scripts/78_prepare_hst_coughvid_metadata.py",
    }.issubset(relative)


def test_controller_publishes_latest_only_after_evidence_pack() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "72_run_hst_reliability.py"
    ).read_text(encoding="utf-8")

    pipeline_call = script.index("summary = pipeline.run")
    publication_call = script.index("publish_hst_latest(")
    success_status = script.index('status="success"')
    assert pipeline_call < publication_call < success_status
    assert 'if args.through == "evidence_pack"' in script


@pytest.mark.parametrize(
    ("mode", "through"),
    [("pilot", "base_resource_pilot"), ("full", "evidence_pack")],
)
def test_real_scientific_registry_drives_resumable_controller_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    through: str,
) -> None:
    import covid_audio_btp.hst_reliability as reliability
    from covid_audio_btp.hst_reliability import HSTPipeline, HSTPipelineConfig
    import covid_audio_btp.hst_stages as stages

    if mode == "full":
        config = HSTPipelineConfig.full(
            tmp_path,
            accepted_hashes={
                "data_contracts_freeze": "a" * 64,
                "pilot_freeze": "b" * 64,
                "environment_lock": "c" * 64,
            },
            device="cpu",
        )
    else:
        config = HSTPipelineConfig.smoke(tmp_path, device="cpu")
        config.mode = "pilot"
        config.scientific_config = {
            **config.scientific_config,
            "mode": "pilot",
            "model": "hst_base",
        }

    registered = stages.build_scientific_stage_handlers(config)
    assert tuple(registered) == HSTPipeline.STAGES
    assert all(
        registered[name] is stages._IMPLEMENTED_HANDLERS[name]
        for name in HSTPipeline.STAGES
    )

    def stub(stage: str):
        def execute(pipeline: HSTPipeline, received_stage: str) -> dict[str, object]:
            assert received_stage == stage
            suffix = {
                "spectrogram_cache": "spectrogram_cache_index.csv",
                "aligned_comparator": "generation_identity.json",
            }.get(stage, f"{stage}.txt")
            output = pipeline.run_root / "integration" / suffix
            output.parent.mkdir(parents=True, exist_ok=True)
            if stage == "spectrogram_cache":
                pd.DataFrame(
                    columns=["eligible", "cache_path", "tensor_sha256"]
                ).to_csv(output, index=False)
            elif stage == "aligned_comparator":
                from covid_audio_btp.hst_runtime import stable_file_sha256

                trusted_root = pipeline.config.workspace_root.parent.resolve()
                approval = (
                    pipeline.config.workspace_root
                    / "configs"
                    / "hst_compare_is10_approval.approved.json"
                )
                accepted = (
                    pipeline.config.workspace_root
                    / "configs"
                    / "hst_comparator_accepted_freezes.approved.json"
                )
                approval.parent.mkdir(parents=True, exist_ok=True)
                approval.write_text('{"approval":"test"}\n', encoding="ascii")
                accepted.write_text('{"accepted":"test"}\n', encoding="ascii")
                output.write_text(
                    json.dumps(
                        {
                            "approval_path": approval.relative_to(trusted_root).as_posix(),
                            "approval_byte_sha256": stable_file_sha256(approval),
                            "accepted_freezes_path": accepted.relative_to(
                                trusted_root
                            ).as_posix(),
                            "accepted_freezes_byte_sha256": stable_file_sha256(accepted),
                        }
                    ),
                    encoding="ascii",
                )
            else:
                output.write_text(f"{pipeline.run_id}:{stage}\n", encoding="ascii")
            return {"output_paths": [output], "row_counts": {"records": 1}}

        return execute

    traversable = {name: stub(name) for name in registered}
    fingerprint_calls = 0
    real_stage_fingerprint = reliability.stage_fingerprint

    def counted_stage_fingerprint(*args: object, **kwargs: object) -> str:
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return real_stage_fingerprint(*args, **kwargs)

    monkeypatch.setattr(reliability, "stage_fingerprint", counted_stage_fingerprint)
    pipeline = HSTPipeline(config, stage_handlers=traversable)
    first = pipeline.run(through=through)
    second = pipeline.run(through=through)
    expected = HSTPipeline.STAGES[: HSTPipeline.STAGES.index(through) + 1]

    assert tuple(first["stage"]) == expected
    assert first["status"].eq("success").all()
    assert not first["reused"].astype(bool).any()
    assert second["reused"].astype(bool).all()
    assert fingerprint_calls <= 2 * len(expected)
    for stage in expected:
        receipt = json.loads(
            pipeline.stage_receipt_path(stage).read_text(encoding="utf-8")
        )
        supplied_hash = receipt.pop("record_hash")
        from covid_audio_btp.hst_runtime import canonical_json_sha256

        assert receipt["receipt_type"] == "hst_stage"
        assert supplied_hash == canonical_json_sha256(receipt)


def test_spectrogram_stage_receipt_is_not_reusable_after_shared_tensor_corruption(
    tmp_path: Path,
) -> None:
    import numpy as np
    from covid_audio_btp.hst_reliability import HSTPipeline, HSTPipelineConfig
    from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256
    from covid_audio_btp.hst_spectrograms import _tensor_sha256

    config = HSTPipelineConfig.smoke(tmp_path, device="cpu")
    pipeline = HSTPipeline(config)
    cache_root = tmp_path / "data" / "processed" / "hst_spectrogram_cache"
    tensor_path = cache_root / "config" / "tensors" / "one.npy"
    tensor_path.parent.mkdir(parents=True)
    image = np.ones((224, 224), dtype=np.float32)
    np.save(tensor_path, image, allow_pickle=False)
    index_path = pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
    index_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "recording_key": ["coswara::r1"],
            "eligible": [True],
            "cache_path": [tensor_path.as_posix()],
            "tensor_sha256": [_tensor_sha256(image)],
        }
    ).to_csv(index_path, index=False)
    relative = index_path.relative_to(pipeline.run_root).as_posix()
    receipt: dict[str, object] = {
        "status": "success",
        "receipt_type": "hst_stage",
        "run_id": pipeline.run_id,
        "stage": "spectrogram_cache",
        "fingerprint": "f" * 64,
        "output_paths": [relative],
        "output_checksums": {relative: stable_file_sha256(index_path)},
    }
    receipt["record_hash"] = canonical_json_sha256(receipt)
    assert pipeline._receipt_is_reusable(
        receipt,
        fingerprint="f" * 64,
        stage="spectrogram_cache",
    )

    tensor_path.write_bytes(b"corrupt")
    resumed_pipeline = HSTPipeline(config)
    assert not resumed_pipeline._receipt_is_reusable(
        receipt,
        fingerprint="f" * 64,
        stage="spectrogram_cache",
    )


def test_aligned_comparator_receipt_is_revoked_when_canonical_approval_changes(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_reliability import HSTPipeline, HSTPipelineConfig
    from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256

    config = HSTPipelineConfig.smoke(tmp_path / "covid_audio_btp", device="cpu")
    pipeline = HSTPipeline(config)
    approval = config.workspace_root / "configs" / "hst_compare_is10_approval.approved.json"
    accepted = (
        config.workspace_root
        / "configs"
        / "hst_comparator_accepted_freezes.approved.json"
    )
    approval.parent.mkdir(parents=True, exist_ok=True)
    approval.write_text('{"approval":"v1"}\n', encoding="ascii")
    accepted.write_text('{"accepted":"v1"}\n', encoding="ascii")
    identity_path = (
        pipeline.run_root
        / "scientific"
        / "aligned_comparator"
        / "generation_identity.json"
    )
    identity_path.parent.mkdir(parents=True)
    identity_path.write_text(
        json.dumps(
            {
                "approval_path": approval.relative_to(tmp_path).as_posix(),
                "approval_byte_sha256": stable_file_sha256(approval),
                "accepted_freezes_path": accepted.relative_to(tmp_path).as_posix(),
                "accepted_freezes_byte_sha256": stable_file_sha256(accepted),
            }
        ),
        encoding="ascii",
    )
    relative = identity_path.relative_to(pipeline.run_root).as_posix()
    receipt: dict[str, object] = {
        "status": "success",
        "receipt_type": "hst_stage",
        "run_id": pipeline.run_id,
        "stage": "aligned_comparator",
        "fingerprint": "a" * 64,
        "output_paths": [relative],
        "output_checksums": {relative: stable_file_sha256(identity_path)},
    }
    receipt["record_hash"] = canonical_json_sha256(receipt)
    assert pipeline._receipt_is_reusable(
        receipt,
        fingerprint="a" * 64,
        stage="aligned_comparator",
    )

    accepted.write_text('{"accepted":"revoked"}\n', encoding="ascii")
    assert not pipeline._receipt_is_reusable(
        receipt,
        fingerprint="a" * 64,
        stage="aligned_comparator",
    )
