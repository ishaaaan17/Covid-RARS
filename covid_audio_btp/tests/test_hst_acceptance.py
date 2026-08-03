from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _pilot_run(tmp_path: Path) -> Path:
    from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256
    from covid_audio_btp.hst_resource_pilot import (
        resource_pilot_freeze_payload,
        runtime_projection_policy_payload,
    )

    run_root = tmp_path / "hst-run"
    outputs = {
        "preflight": (
            "audits/environment.json",
            {"pip_freeze": ["example==1.0"]},
        ),
        "data_contracts": (
            "contracts/data_contracts_freeze.json",
            {
                "contract_metadata": {
                    "dataset_release_id": "coswara-release+coughvid-release",
                    "label_column": "coswara:label_binary;coughvid:status_SSL",
                },
            },
        ),
        "base_resource_pilot": (
            "audits/base_resource_pilot_freeze.json",
            {
                "schema_version": 2,
                "physical_batch_size": 4,
                "gradient_accumulation": 2,
                "effective_batch_size": 8,
                "precision": "fp32",
                "amp": False,
                "minimum_optimizer_updates": 100,
                "headroom_required_bytes": 1024**3,
                "probability_tolerance": 0.01,
                "relative_loss_tolerance": 0.01,
                "selection_basis": "first_safe_batch_then_amp_if_numerically_valid_else_fp32",
                "model_metrics_used": False,
                "total_vram_bytes": 8 * 1024**3,
                "freeze_context": {"checkpoint_sha256": "a" * 64},
                "runtime_projection": {
                    "schema_version": 1,
                    "estimate_basis": (
                        "selected_cough_fold_optimizer_throughput_times_modality_"
                        "specific_all_participant_upper_bound_and_frozen_job_epoch_plan"
                    ),
                    "selected_trial_seconds": 20.0,
                    "selected_trial_optimizer_updates": 100,
                    "optimizer_updates_per_epoch_by_modality": {
                        "breath": 20,
                        "cough": 25,
                        "speech": 30,
                    },
                    "planned_training_jobs_by_modality": {
                        "breath": 10,
                        "cough": 25,
                        "speech": 15,
                    },
                    "planned_training_jobs": 50,
                    "confirmatory_epochs": 100,
                    "estimated_optimizer_updates": 127500,
                    "training_only_serial_gpu_hours": 7.083333333333333,
                    "end_to_end_overhead_multiplier": 1.5,
                    "estimated_serial_gpu_hours": 10.625,
                    "maximum_approved_runtime_hours": 168.0,
                    "within_approved_runtime_ceiling": True,
                    "estimate_is_completion_guarantee": False,
                },
            },
        ),
    }
    for stage, (relative, payload) in outputs.items():
        if stage == "preflight":
            payload["pip_freeze_sha256"] = canonical_json_sha256(payload["pip_freeze"])
        if stage == "data_contracts":
            payload["manifest_sha256"] = canonical_json_sha256(payload)
        if stage == "base_resource_pilot":
            runtime = payload["runtime_projection"]
            policy = runtime_projection_policy_payload(
                optimizer_updates_per_epoch_by_modality=runtime[
                    "optimizer_updates_per_epoch_by_modality"
                ],
                planned_training_jobs_by_modality=runtime[
                    "planned_training_jobs_by_modality"
                ],
                confirmatory_epochs=runtime["confirmatory_epochs"],
                end_to_end_overhead_multiplier=runtime[
                    "end_to_end_overhead_multiplier"
                ],
                maximum_approved_runtime_hours=runtime[
                    "maximum_approved_runtime_hours"
                ],
            )
            policy_hash = canonical_json_sha256(policy)
            runtime["runtime_projection_policy"] = policy
            runtime["runtime_projection_policy_sha256"] = policy_hash
            payload["freeze_context"]["runtime_projection_policy_sha256"] = policy_hash
            payload["pilot_freeze_hash"] = canonical_json_sha256(
                resource_pilot_freeze_payload(payload)
            )
        output = run_root / relative
        _write_json(output, payload)
        receipt = {
            "schema_version": 1,
            "receipt_type": "hst_stage",
            "run_id": "hst-test",
            "stage": stage,
            "status": "success",
            "output_paths": [relative],
            "output_checksums": {relative: stable_file_sha256(output)},
        }
        receipt["record_hash"] = canonical_json_sha256(receipt)
        _write_json(run_root / "runtime" / "stages" / f"{stage}.json", receipt)
    return run_root


def test_build_acceptance_candidate_verifies_and_extracts_three_freezes(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_acceptance import build_pilot_acceptance_candidate

    run_root = _pilot_run(tmp_path)
    output = tmp_path / "candidate.json"
    payload = build_pilot_acceptance_candidate(run_root=run_root, output_path=output)
    contract = json.loads(
        (run_root / "contracts" / "data_contracts_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    pilot = json.loads(
        (run_root / "audits" / "base_resource_pilot_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    environment = json.loads(
        (run_root / "audits" / "environment.json").read_text(encoding="utf-8")
    )

    assert payload["candidate_status"] == "requires_manual_review"
    assert payload["accepted_hashes"] == {
        "data_contracts_freeze": contract["manifest_sha256"],
        "environment_lock": environment["pip_freeze_sha256"],
        "pilot_freeze": pilot["pilot_freeze_hash"],
    }
    assert payload["pip_freeze_hash"] == environment["pip_freeze_sha256"]
    assert payload["runtime_review"]["estimated_serial_gpu_hours"] == 10.625
    assert payload["review_gate"]["projected_runtime_within_frozen_ceiling"] is True
    assert "approval_status" not in payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_acceptance_candidate_rejects_tampered_stage_output(tmp_path: Path) -> None:
    from covid_audio_btp.hst_acceptance import build_pilot_acceptance_candidate

    run_root = _pilot_run(tmp_path)
    (run_root / "audits" / "environment.json").write_text(
        '{"pip_freeze_sha256":"' + "f" * 64 + '"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum"):
        build_pilot_acceptance_candidate(
            run_root=run_root,
            output_path=tmp_path / "candidate.json",
        )


def test_acceptance_candidate_rejects_metric_driven_resource_pilot(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_acceptance import build_pilot_acceptance_candidate
    from covid_audio_btp.hst_resource_pilot import resource_pilot_freeze_payload
    from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256

    run_root = _pilot_run(tmp_path)
    pilot = run_root / "audits" / "base_resource_pilot_freeze.json"
    payload = json.loads(pilot.read_text(encoding="utf-8"))
    payload["model_metrics_used"] = True
    payload["pilot_freeze_hash"] = canonical_json_sha256(
        resource_pilot_freeze_payload(payload)
    )
    _write_json(pilot, payload)
    receipt_path = run_root / "runtime" / "stages" / "base_resource_pilot.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_checksums"]["audits/base_resource_pilot_freeze.json"] = (
        stable_file_sha256(pilot)
    )
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="model metrics"):
        build_pilot_acceptance_candidate(
            run_root=run_root,
            output_path=tmp_path / "candidate.json",
        )


def test_acceptance_candidate_rejects_coswara_only_data_contract(tmp_path: Path) -> None:
    from covid_audio_btp.hst_acceptance import build_pilot_acceptance_candidate
    from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256

    run_root = _pilot_run(tmp_path)
    contract = run_root / "contracts" / "data_contracts_freeze.json"
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["contract_metadata"] = {
        "dataset_release_id": "coswara-release",
        "label_column": "coswara:label_binary",
    }
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    _write_json(contract, payload)
    receipt_path = run_root / "runtime" / "stages" / "data_contracts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_checksums"]["contracts/data_contracts_freeze.json"] = (
        stable_file_sha256(contract)
    )
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="Coswara and COUGHVID"):
        build_pilot_acceptance_candidate(
            run_root=run_root,
            output_path=tmp_path / "candidate.json",
        )


def test_acceptance_candidate_rejects_runtime_projection_over_ceiling(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_acceptance import build_pilot_acceptance_candidate
    from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256

    run_root = _pilot_run(tmp_path)
    pilot = run_root / "audits" / "base_resource_pilot_freeze.json"
    payload = json.loads(pilot.read_text(encoding="utf-8"))
    payload["runtime_projection"]["within_approved_runtime_ceiling"] = False
    _write_json(pilot, payload)
    receipt_path = run_root / "runtime" / "stages" / "base_resource_pilot.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_checksums"]["audits/base_resource_pilot_freeze.json"] = (
        stable_file_sha256(pilot)
    )
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="runtime.*ceiling"):
        build_pilot_acceptance_candidate(
            run_root=run_root,
            output_path=tmp_path / "candidate.json",
        )


def test_acceptance_candidate_rejects_legacy_single_modality_runtime_projection(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_acceptance import build_pilot_acceptance_candidate
    from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256

    run_root = _pilot_run(tmp_path)
    pilot = run_root / "audits" / "base_resource_pilot_freeze.json"
    payload = json.loads(pilot.read_text(encoding="utf-8"))
    del payload["runtime_projection"]["optimizer_updates_per_epoch_by_modality"]
    _write_json(pilot, payload)
    receipt_path = run_root / "runtime" / "stages" / "base_resource_pilot.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_checksums"]["audits/base_resource_pilot_freeze.json"] = (
        stable_file_sha256(pilot)
    )
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="modality-specific"):
        build_pilot_acceptance_candidate(
            run_root=run_root,
            output_path=tmp_path / "candidate.json",
        )


def test_runtime_projection_rejects_fractional_integer_fields(tmp_path: Path) -> None:
    from covid_audio_btp.hst_acceptance import (
        _validate_conservative_runtime_projection,
    )

    run_root = _pilot_run(tmp_path)
    payload = json.loads(
        (run_root / "audits" / "base_resource_pilot_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    projection = payload["runtime_projection"]
    projection["selected_trial_optimizer_updates"] = 100.5

    with pytest.raises(ValueError, match="positive integers"):
        _validate_conservative_runtime_projection(projection)


def test_acceptance_rejects_runtime_policy_not_bound_by_pilot_hash(
    tmp_path: Path,
) -> None:
    from covid_audio_btp.hst_acceptance import build_pilot_acceptance_candidate
    from covid_audio_btp.hst_runtime import canonical_json_sha256, stable_file_sha256

    run_root = _pilot_run(tmp_path)
    pilot = run_root / "audits" / "base_resource_pilot_freeze.json"
    payload = json.loads(pilot.read_text(encoding="utf-8"))
    payload["runtime_projection"]["maximum_approved_runtime_hours"] = 200.0
    _write_json(pilot, payload)
    receipt_path = run_root / "runtime" / "stages" / "base_resource_pilot.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_checksums"]["audits/base_resource_pilot_freeze.json"] = (
        stable_file_sha256(pilot)
    )
    receipt.pop("record_hash")
    receipt["record_hash"] = canonical_json_sha256(receipt)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="runtime projection policy"):
        build_pilot_acceptance_candidate(
            run_root=run_root,
            output_path=tmp_path / "candidate.json",
        )
