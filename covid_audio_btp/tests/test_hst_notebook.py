from __future__ import annotations

import json
from pathlib import Path


def test_hst_notebook_is_a_thin_restart_safe_controller() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    sources = ["".join(cell["source"]) for cell in code]
    joined = "\n".join(sources)

    assert 8 <= len(code) <= 14
    assert "configs/hst_reliability.json" in joined
    assert "run_preflight" in joined
    assert "launch_detached_run" in joined
    assert "read_run_status" in joined
    assert "read_hst_run_progress" in joined
    assert "wait_for_detached_run" in joined
    assert "0.897" in joined
    assert "guarantee" in joined.lower()
    assert "--detach" not in joined
    assert not any(line.lstrip().startswith(("!", "%run")) for line in joined.splitlines())
    assert all(cell.get("execution_count") is None for cell in code)
    assert all(cell.get("outputs") == [] for cell in code)


def test_notebook_has_single_launch_cell_and_visible_failure_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code_sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    launch_cells = [source for source in code_sources if "launch_detached_run(" in source]
    assert len(launch_cells) == 1
    assert "raise" in "\n".join(code_sources)


def test_notebook_project_discovery_supports_override_and_ancestor_search() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb").read_text(
            encoding="utf-8"
        )
    )
    joined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "COVID_RARS_PROJECT_ROOT" in joined
    assert "for candidate in (start, *start.parents)" in joined
    assert '(candidate / "configs/hst_reliability.json").is_file()' in joined
    assert '(candidate / "src/covid_audio_btp").is_dir()' in joined


def test_notebook_detached_polling_is_bounded_and_stale_aware() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb").read_text(
            encoding="utf-8"
        )
    )
    joined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "DETACHED_STALE_AFTER_SECONDS" in joined
    assert "DETACHED_MAX_WAIT_SECONDS" in joined
    assert "wait_for_detached_run(" in joined
    assert 'while status["status"]' not in joined


def test_notebook_displays_integrity_checked_stage_job_epoch_and_checkpoint_progress() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb").read_text(
            encoding="utf-8"
        )
    )
    joined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "pipeline_stages" in joined
    assert "confirmatory_training" in joined
    assert "durable_job_equivalents" in joined
    assert "current_job" in joined
    assert "job_id" in joined
    assert "epoch_percent" in joined
    assert "checkpointed" in joined
    assert "checkpoint_resume_safe" in joined
    assert "checkpoint_generation" in joined
    assert "checkpoint_path" in joined
    assert "checkpoint_sha256" in joined
    assert "durable checkpoint" in joined.lower()
    assert "awaiting first optimizer-safe checkpoint" in joined
    assert "checkpoint_sha256'][:12]" not in joined


def test_notebook_reattaches_to_the_exact_active_frozen_run() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb").read_text(
            encoding="utf-8"
        )
    )
    joined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "find_resumable_detached_run" in joined
    assert "existing_launch = find_resumable_detached_run(" in joined
    assert "launch = existing_launch or launch_detached_run(" in joined
    assert joined.index("existing_launch = find_resumable_detached_run(") < joined.index(
        "launch = existing_launch or launch_detached_run("
    )


def test_notebook_defaults_to_capped_pilot_before_full_execution() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb").read_text(
            encoding="utf-8"
        )
    )
    joined = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'MODE = "pilot"' in joined
    assert 'THROUGH = "base_resource_pilot"' in joined


def test_notebook_exposes_every_manual_acceptance_gate_without_self_approval() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb").read_text(
            encoding="utf-8"
        )
    )
    joined = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "75_prepare_hst_acceptance.py" in joined
    assert "76_prepare_hst_comparator_approval.py" in joined
    assert "77_prepare_hst_comparator_generation_acceptance.py" in joined
    assert 'THROUGH_MANIFESTS = "manifests"' in joined
    assert 'THROUGH_COMPARATOR = "aligned_comparator"' in joined
    assert 'THROUGH_FINAL = "evidence_pack"' in joined
    assert "MANUAL_REVIEW_REQUIRED" in joined
    assert "--manifest-name" not in joined
    assert "write_text(" not in joined
    assert "shutil.copy" not in joined


def test_notebook_uses_hardened_two_pass_comparator_approval_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb").read_text(
            encoding="utf-8"
        )
    )
    joined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    approval_index = joined.index(
        '"scripts/76_prepare_hst_comparator_approval.py", "approval-record"'
    )
    initial_freeze_index = joined.index(
        '"scripts/76_prepare_hst_comparator_approval.py", "accepted-freezes"'
    )
    first_pass_index = joined.index(
        'expected_manual_gate="manual comparator generation acceptance required"'
    )
    generation_acceptance_index = joined.index(
        "scripts/77_prepare_hst_comparator_generation_acceptance.py"
    )
    final_resume_index = joined.index("FINAL_LAUNCH, FINAL_STATUS")

    assert approval_index < initial_freeze_index < first_pass_index
    assert first_pass_index < generation_acceptance_index < final_resume_index
    assert "manifest_name" not in joined


def test_notebook_enforces_single_gpu_and_evaluation_only_held_out_data() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "hst_reliability.json").read_text(encoding="ascii")
    )
    notebook = json.loads(
        (root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb").read_text(
            encoding="utf-8"
        )
    )
    joined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert 'MAX_CONCURRENT_GPU_JOBS = 1' in joined
    assert 'END_TO_END_OVERHEAD_MULTIPLIER = 1.5' in joined
    assert '"max_concurrent_gpu_jobs"' in joined
    assert '"end_to_end_overhead_multiplier"' in joined
    assert "does not permit concurrent GPU jobs" in joined
    evaluation_policy = config["experiment"]["test_evaluation_policy"]
    assert evaluation_policy in joined
    assert "one_evaluation_after_validation_freeze" not in joined
    assert "test and external labels are evaluation-only" in joined
    assert "Never rerun or change settings after inspecting held-out outcomes" in joined


def test_notebook_final_launch_is_resume_safe_and_requires_canonical_approvals() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "09_HST_RELIABILITY_E2E.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code_sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    joined = "\n".join(code_sources)

    assert "hst_compare_is10_approval.approved.json" in joined
    assert "hst_comparator_accepted_freezes.approved.json" in joined
    assert "reports/hst/accepted_freezes.json" in joined
    assert "expected_run_id=" in joined
    assert "resume" in joined.lower()
    assert "reports/hst/latest.json" in joined
    assert "sensitivity_execution_registry.csv" in joined
    assert "raw_status_sensitivity_metrics.csv" in joined


def test_ubuntu_runbook_documents_gates_resume_and_non_guaranteed_targets() -> None:
    root = Path(__file__).resolve().parents[1]
    runbook = root / "docs" / "HST_UBUNTU_RUNBOOK.md"
    text = runbook.read_text(encoding="ascii")
    normalized = " ".join(text.split())

    assert "HST-Small smoke" in text
    assert "HST-Base resource pilot" in text
    assert "Manual gate 1" in text
    assert "Manual gate 2" in text
    assert "Manual gate 3" in text
    assert "aligned_comparator" in text
    assert "evidence_pack" in text
    assert "same run ID" in text
    assert "not guaranteed" in text
    assert "test and external labels" in text
    assert "T1000 8GB" in text
    assert "max_concurrent_gpu_jobs" in text
    assert "Do not launch a second GPU job" in normalized
    assert "all contract-eligible Coswara participants" in normalized
    assert "25 cough, 15 speech, and 10 breath jobs" in normalized
    assert "1.5 end-to-end overhead multiplier" in normalized
    assert "--manifest-name" not in text
    assert r"C:\Users\nhnis\Desktop\Covid-RARS\covid_audio_btp" in text
    assert "/home/covid/Desktop/Covid-19-BTP/covid_audio_btp" in text
    assert (
        "cd /home/covid/Desktop/Covid-19-BTP\n"
        "git submodule update --init --recursive HST\n"
        "cd /home/covid/Desktop/Covid-19-BTP/covid_audio_btp"
    ) in text
    assert "Windows is for code review and Git synchronization" in normalized
    assert "Ubuntu is the scientific execution environment" in normalized
    assert "same full run ID" in normalized
    assert "dedicated `covid` Unix account" in normalized
    assert "cross-account exclusion" in normalized
    assert "host-wide GPU lease" not in text
    assert "test or external results" in normalized
    assert "relabeling the same frozen external probabilities" in normalized
    assert "event/SNR sensitivity" in normalized
    assert "remain explicitly deferred" in normalized
    assert "durable training-equivalent jobs out of 50" in normalized
    assert "training_progress.json" in text
    assert "optimizer-safe checkpoint" in normalized
    assert "generation named by the last durable progress record" in normalized
    assert "full SHA-256 agrees with the receipt" in normalized
    assert "modification time, and change time" in normalized
    assert "not a cryptographic authentication claim" in normalized
