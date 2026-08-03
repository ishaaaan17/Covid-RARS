from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _row(
    batch: int,
    precision: str,
    *,
    valid: bool = True,
    seconds: float = 10.0,
    free_vram: int = 2 * 1024**3,
    probability_difference: float = 0.0,
    relative_loss_difference: float = 0.0,
    skipped: int = 0,
) -> dict[str, object]:
    return {
        "physical_batch_size": batch,
        "precision": precision,
        "valid": valid,
        "optimizer_updates": 100,
        "skipped_optimizer_updates": skipped,
        "seconds": seconds,
        "free_vram_bytes": free_vram,
        "max_abs_probability_difference_from_fp32": probability_difference,
        "relative_loss_difference_from_fp32": relative_loss_difference,
        "finite_loss": True,
        "finite_gradients": True,
        "finite_parameters": True,
        "finite_predictions": True,
    }


def test_resource_pilot_selects_first_safe_batch_and_faster_valid_precision() -> None:
    from covid_audio_btp.hst_resource_pilot import select_base_resource_pilot

    rows = pd.DataFrame(
        [
            _row(8, "fp32", valid=False),
            _row(8, "amp", valid=False),
            _row(4, "fp32", seconds=15.0),
            _row(4, "amp", seconds=9.0, probability_difference=0.005),
            _row(2, "fp32", seconds=20.0),
        ]
    )
    selected = select_base_resource_pilot(rows, total_vram_bytes=8 * 1024**3)
    assert selected["physical_batch_size"] == 4
    assert selected["gradient_accumulation"] == 2
    assert selected["precision"] == "amp"
    assert selected["effective_batch_size"] == 8


def test_amp_fails_closed_on_numerical_or_skipped_update_drift() -> None:
    from covid_audio_btp.hst_resource_pilot import select_base_resource_pilot

    rows = pd.DataFrame(
        [
            _row(8, "fp32", seconds=14.0),
            _row(8, "amp", seconds=5.0, probability_difference=0.02),
            _row(4, "amp", seconds=4.0),
        ]
    )
    selected = select_base_resource_pilot(rows, total_vram_bytes=8 * 1024**3)
    assert selected["physical_batch_size"] == 8
    assert selected["precision"] == "fp32"

    rows.loc[1, "max_abs_probability_difference_from_fp32"] = 0.0
    rows.loc[1, "skipped_optimizer_updates"] = 1
    selected = select_base_resource_pilot(rows, total_vram_bytes=8 * 1024**3)
    assert selected["precision"] == "fp32"


def test_resource_pilot_requires_headroom_and_100_updates() -> None:
    from covid_audio_btp.hst_resource_pilot import select_base_resource_pilot

    too_little_memory = _row(8, "fp32", free_vram=512 * 1024**2)
    too_few_updates = _row(4, "fp32")
    too_few_updates["optimizer_updates"] = 99
    with pytest.raises(RuntimeError, match="No HST-Base resource configuration"):
        select_base_resource_pilot(
            pd.DataFrame([too_little_memory, too_few_updates]),
            total_vram_bytes=8 * 1024**3,
        )


def test_resource_selection_rejects_model_metrics_and_has_content_hash() -> None:
    from covid_audio_btp.hst_resource_pilot import select_base_resource_pilot

    rows = pd.DataFrame([_row(8, "fp32")])
    selected = select_base_resource_pilot(rows, total_vram_bytes=8 * 1024**3)
    assert len(str(selected["pilot_freeze_hash"])) == 64
    with pytest.raises(ValueError, match="model metric"):
        select_base_resource_pilot(rows.assign(auroc=0.9), total_vram_bytes=8 * 1024**3)


def test_resource_freeze_hash_ignores_authenticated_runtime_diagnostics() -> None:
    from covid_audio_btp.hst_resource_pilot import select_base_resource_pilot

    context = {
        "hst_commit": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "data_contracts_freeze_hash": "c" * 64,
        "dependency_lock_sha256": "d" * 64,
    }
    rows = pd.DataFrame(
        [
            _row(8, "fp32", valid=False),
            _row(8, "amp", valid=False),
            _row(4, "fp32", seconds=15.0),
            _row(4, "amp", seconds=9.0, probability_difference=0.005),
            _row(2, "fp32", seconds=20.0),
        ]
    )
    first = select_base_resource_pilot(
        rows,
        total_vram_bytes=8 * 1024**3,
        freeze_context=context,
    )
    changed_diagnostics = rows.copy()
    changed_diagnostics["seconds"] = changed_diagnostics["seconds"] + 100.0
    changed_diagnostics["free_vram_bytes"] = 3 * 1024**3
    second = select_base_resource_pilot(
        changed_diagnostics,
        total_vram_bytes=8 * 1024**3,
        freeze_context=context,
    )

    assert first["pilot_freeze_hash"] == second["pilot_freeze_hash"]
    assert first["benchmark_sha256"] != second["benchmark_sha256"]
    assert first["measured_free_vram_bytes"] != second["measured_free_vram_bytes"]


def test_resource_freeze_hash_binds_static_context_and_selected_decision() -> None:
    from covid_audio_btp.hst_resource_pilot import select_base_resource_pilot

    rows = pd.DataFrame([_row(8, "fp32"), _row(8, "amp")])
    first = select_base_resource_pilot(
        rows,
        total_vram_bytes=8 * 1024**3,
        freeze_context={"checkpoint_sha256": "a" * 64},
    )
    changed_context = select_base_resource_pilot(
        rows,
        total_vram_bytes=8 * 1024**3,
        freeze_context={"checkpoint_sha256": "b" * 64},
    )
    fp32_only = rows.copy()
    fp32_only.loc[fp32_only["precision"].eq("amp"), "valid"] = False
    changed_decision = select_base_resource_pilot(
        fp32_only,
        total_vram_bytes=8 * 1024**3,
        freeze_context={"checkpoint_sha256": "a" * 64},
    )

    assert first["pilot_freeze_hash"] != changed_context["pilot_freeze_hash"]
    assert first["pilot_freeze_hash"] != changed_decision["pilot_freeze_hash"]
    assert first["precision"] == "amp"
    assert changed_decision["precision"] == "fp32"


def test_runtime_projection_uses_measured_optimizer_throughput_and_ceiling() -> None:
    from covid_audio_btp.hst_resource_pilot import project_full_training_runtime

    projection = project_full_training_runtime(
        selected_trial_seconds=20.0,
        selected_trial_optimizer_updates=100,
        optimizer_updates_per_epoch_by_modality={
            "cough": 25,
            "speech": 30,
            "breath": 20,
        },
        planned_training_jobs_by_modality={
            "cough": 25,
            "speech": 15,
            "breath": 10,
        },
        confirmatory_epochs=100,
        end_to_end_overhead_multiplier=1.5,
        maximum_approved_runtime_hours=12.0,
    )

    assert projection["estimated_optimizer_updates"] == 127_500
    assert projection["training_only_serial_gpu_hours"] == pytest.approx(25_500 / 3600)
    assert projection["estimated_serial_gpu_hours"] == pytest.approx(38_250 / 3600)
    assert projection["within_approved_runtime_ceiling"] is True
    assert projection["estimate_is_completion_guarantee"] is False
    assert len(str(projection["runtime_projection_policy_sha256"])) == 64


def test_runtime_projection_policy_hash_excludes_timing_but_binds_ceiling() -> None:
    from covid_audio_btp.hst_resource_pilot import project_full_training_runtime

    arguments = {
        "selected_trial_optimizer_updates": 100,
        "optimizer_updates_per_epoch_by_modality": {"cough": 25},
        "planned_training_jobs_by_modality": {"cough": 50},
        "confirmatory_epochs": 100,
        "end_to_end_overhead_multiplier": 1.5,
        "maximum_approved_runtime_hours": 168.0,
    }
    first = project_full_training_runtime(selected_trial_seconds=20.0, **arguments)
    slower = project_full_training_runtime(selected_trial_seconds=40.0, **arguments)
    changed_ceiling = project_full_training_runtime(
        selected_trial_seconds=20.0,
        **{**arguments, "maximum_approved_runtime_hours": 200.0},
    )

    assert (
        first["runtime_projection_policy_sha256"]
        == slower["runtime_projection_policy_sha256"]
    )
    assert (
        first["runtime_projection_policy_sha256"]
        != changed_ceiling["runtime_projection_policy_sha256"]
    )


def test_runtime_projection_fails_closed_on_invalid_or_excessive_budget() -> None:
    from covid_audio_btp.hst_resource_pilot import project_full_training_runtime

    with pytest.raises(ValueError, match="positive"):
        project_full_training_runtime(
            selected_trial_seconds=0.0,
            selected_trial_optimizer_updates=100,
            optimizer_updates_per_epoch_by_modality={"cough": 25},
            planned_training_jobs_by_modality={"cough": 25},
            confirmatory_epochs=100,
            end_to_end_overhead_multiplier=1.5,
            maximum_approved_runtime_hours=8.0,
        )
    projection = project_full_training_runtime(
        selected_trial_seconds=40.0,
        selected_trial_optimizer_updates=100,
        optimizer_updates_per_epoch_by_modality={"cough": 25},
        planned_training_jobs_by_modality={"cough": 50},
        confirmatory_epochs=100,
        end_to_end_overhead_multiplier=1.5,
        maximum_approved_runtime_hours=8.0,
    )
    assert projection["within_approved_runtime_ceiling"] is False

    for invalid_updates in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            project_full_training_runtime(
                selected_trial_seconds=20.0,
                selected_trial_optimizer_updates=invalid_updates,
                optimizer_updates_per_epoch_by_modality={"cough": 25},
                planned_training_jobs_by_modality={"cough": 25},
                confirmatory_epochs=100,
                end_to_end_overhead_multiplier=1.5,
                maximum_approved_runtime_hours=8.0,
            )


def test_timeout_failure_preserves_captured_worker_streams() -> None:
    import subprocess

    from covid_audio_btp.hst_resource_pilot import _worker_exception_result

    error = subprocess.TimeoutExpired(
        cmd=["python", "worker.py"],
        timeout=5,
        output="partial stdout",
        stderr="partial stderr",
    )
    result = _worker_exception_result(
        error,
        job={"physical_batch_size": 8, "precision": "fp32"},
    )

    assert result["failure_type"] == "TimeoutExpired"
    assert result["worker_stdout_tail"] == "partial stdout"
    assert result["worker_stderr_tail"] == "partial stderr"
    assert len(str(result["worker_stdout_sha256"])) == 64
    assert len(str(result["worker_stderr_sha256"])) == 64


def test_failed_oom_trials_remain_hashable_and_do_not_block_safe_batch() -> None:
    from covid_audio_btp.hst_resource_pilot import select_base_resource_pilot

    failed = _row(8, "fp32", valid=False, seconds=math.inf, free_vram=0)
    failed["max_abs_probability_difference_from_fp32"] = math.inf
    failed["relative_loss_difference_from_fp32"] = math.nan
    selected = select_base_resource_pilot(
        pd.DataFrame(
            [
                failed,
                _row(8, "amp", valid=False, seconds=math.inf, free_vram=0),
                _row(4, "fp32"),
                _row(4, "amp", probability_difference=0.005),
            ]
        ),
        total_vram_bytes=8 * 1024**3,
    )

    assert selected["physical_batch_size"] == 4
    assert len(str(selected["benchmark_sha256"])) == 64


def test_conservative_workload_counts_participants_not_recordings() -> None:
    from covid_audio_btp.hst_resource_pilot import (
        conservative_balanced_optimizer_updates_per_epoch,
    )

    metadata = pd.DataFrame(
        {
            "participant_key": ["n1", "n1", "n2", "p1", "s-n", "s-p"],
            "label_binary": [
                "negative",
                "negative",
                "negative",
                "positive",
                "negative",
                "positive",
            ],
            "modality": ["cough", "cough", "cough", "cough", "speech", "speech"],
        }
    )

    updates = conservative_balanced_optimizer_updates_per_epoch(
        metadata,
        modalities=("cough", "speech"),
        effective_batch_size=2,
    )

    assert updates == {"cough": 2, "speech": 1}


def test_nonzero_worker_exit_preserves_structured_failure_evidence(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_resource_pilot import _read_worker_trial_result

    result_path = tmp_path / "failed.json"
    result_path.write_text(
        json.dumps(
            {
                "valid": False,
                "error": "OutOfMemoryError: CUDA out of memory",
                "traceback": "trace line 1\ntrace line 2",
            }
        ),
        encoding="utf-8",
    )
    completed = SimpleNamespace(returncode=1, stderr="worker stderr")

    result = _read_worker_trial_result(
        result_path=result_path,
        completed=completed,
        job={"physical_batch_size": 8, "precision": "fp32"},
    )

    assert result["valid"] is False
    assert result["failure_type"] == "OutOfMemoryError"
    assert result["worker_returncode"] == 1
    assert len(str(result["traceback_sha256"])) == 64
    assert len(str(result["worker_stderr_sha256"])) == 64
    assert "CUDA out of memory" in str(result["error"])


def test_trial_runner_retains_nonzero_worker_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_audio_btp.hst_resource_pilot as resource_pilot

    supplied = {
        name: tmp_path / name
        for name in ("cache.csv", "manifest.csv", "checkpoint.pth", "worker.py")
    }
    for path in supplied.values():
        path.write_text("fixture", encoding="utf-8")
    hst_repo = tmp_path / "hst"
    hst_repo.mkdir()

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        job_path = Path(command[command.index("--job-json") + 1])
        result_path = Path(command[command.index("--result-json") + 1])
        job = json.loads(job_path.read_text(encoding="utf-8"))
        batch = int(job["physical_batch_size"])
        precision = str(job["precision"])
        if batch == 8:
            result_path.write_text(
                json.dumps(
                    {
                        **job,
                        "valid": False,
                        "error": "OutOfMemoryError: CUDA out of memory",
                        "traceback": "worker traceback",
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=1, stderr="cuda allocation failed")
        result_path.write_text(
            json.dumps(
                {
                    **job,
                    "valid": True,
                    "finite_loss": True,
                    "finite_gradients": True,
                    "finite_parameters": True,
                    "finite_predictions": True,
                    "skipped_optimizer_updates": 0,
                    "seconds": 10.0,
                    "free_vram_bytes": 2 * 1024**3,
                    "total_vram_bytes": 8 * 1024**3,
                    "peak_allocated_vram_bytes": 1024**3,
                    "peak_reserved_vram_bytes": 1536 * 1024**2,
                    "completed_optimizer_updates": 100,
                    "optimizer_updates_per_epoch": 12,
                    "evaluation_loss": 0.5,
                    "evaluation_probabilities": [0.25, 0.75],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr=f"safe {precision}")

    monkeypatch.setattr(resource_pilot.subprocess, "run", fake_run)
    benchmark, selection = resource_pilot.run_base_resource_pilot_trials(
        cache_index_path=supplied["cache.csv"],
        manifest_path=supplied["manifest.csv"],
        checkpoint_path=supplied["checkpoint.pth"],
        hst_repo=hst_repo,
        worker_script=supplied["worker.py"],
        fold=1,
        modality="cough",
        seed=52,
    )

    failed = benchmark.loc[benchmark["physical_batch_size"].eq(8)]
    assert set(failed["failure_type"]) == {"OutOfMemoryError"}
    assert set(failed["worker_returncode"]) == {1}
    assert failed["traceback_sha256"].astype(str).str.len().eq(64).all()
    assert selection["physical_batch_size"] == 4
