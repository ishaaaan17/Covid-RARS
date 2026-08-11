from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping

from .hst_runtime import atomic_write_json, canonical_json_sha256, stable_file_sha256
from .hst_resource_pilot import (
    resource_pilot_freeze_payload,
    runtime_projection_policy_payload,
)
from .hst_workloads import get_hst_workload_profile


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_STAGES = ("preflight", "data_contracts", "base_resource_pilot")
_RUNTIME_ESTIMATE_BASIS = (
    "selected_cough_fold_optimizer_throughput_times_modality_specific_"
    "all_participant_upper_bound_and_frozen_job_epoch_plan"
)


def _validate_conservative_runtime_projection(
    projection: Mapping[str, object],
) -> tuple[float, float]:
    if projection.get("schema_version") != 1 or projection.get(
        "estimate_basis"
    ) != _RUNTIME_ESTIMATE_BASIS:
        raise ValueError(
            "Resource-pilot runtime projection is not the frozen "
            "modality-specific estimate"
        )
    updates = projection.get("optimizer_updates_per_epoch_by_modality")
    jobs = projection.get("planned_training_jobs_by_modality")
    if not isinstance(updates, Mapping) or not isinstance(jobs, Mapping):
        raise ValueError("Resource-pilot runtime projection lacks modality-specific workload")
    integer_fields = {
        "selected_trial_optimizer_updates": projection.get(
            "selected_trial_optimizer_updates"
        ),
        "planned_training_jobs": projection.get("planned_training_jobs"),
        "confirmatory_epochs": projection.get("confirmatory_epochs"),
        "estimated_optimizer_updates": projection.get("estimated_optimizer_updates"),
        **{f"job:{key}": value for key, value in jobs.items()},
        **{f"updates:{key}": value for key, value in updates.items()},
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in integer_fields.values()
    ):
        raise ValueError("Resource-pilot runtime count fields must be positive integers")
    profile = get_hst_workload_profile(projection.get("workload_profile"))
    expected_jobs = dict(profile.training_jobs_by_modality)
    if set(updates) != set(expected_jobs) or dict(jobs) != expected_jobs:
        raise ValueError("Resource-pilot runtime projection changed the modality-specific job plan")
    if projection.get("planned_training_jobs") != profile.total_training_jobs or projection.get(
        "confirmatory_epochs"
    ) != 100:
        raise ValueError("Resource-pilot runtime projection changed the frozen job or epoch plan")
    expected_updates = 100 * sum(
        int(updates[modality]) * count
        for modality, count in expected_jobs.items()
    )
    if projection.get("estimated_optimizer_updates") != expected_updates:
        raise ValueError("Resource-pilot runtime projection has inconsistent optimizer updates")
    try:
        trial_seconds = float(projection["selected_trial_seconds"])
        trial_updates = int(projection["selected_trial_optimizer_updates"])
        training_hours = float(projection["training_only_serial_gpu_hours"])
        overhead = float(projection["end_to_end_overhead_multiplier"])
        estimated_hours = float(projection["estimated_serial_gpu_hours"])
        maximum_hours = float(projection["maximum_approved_runtime_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Resource-pilot runtime projection is incomplete") from exc
    numeric = (trial_seconds, training_hours, overhead, estimated_hours, maximum_hours)
    if trial_updates <= 0 or any(not math.isfinite(value) or value <= 0 for value in numeric):
        raise ValueError("Resource-pilot runtime projection contains invalid numeric values")
    expected_training_hours = expected_updates * trial_seconds / trial_updates / 3600.0
    if not math.isclose(training_hours, expected_training_hours, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Resource-pilot training-only runtime projection is inconsistent")
    if not math.isclose(overhead, 1.5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Resource-pilot runtime projection changed the frozen overhead multiplier")
    if not math.isclose(
        estimated_hours,
        training_hours * overhead,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Resource-pilot end-to-end runtime projection is inconsistent")
    expected_policy = runtime_projection_policy_payload(
        workload_profile=profile.name,
        optimizer_updates_per_epoch_by_modality={
            str(key): int(value) for key, value in updates.items()
        },
        planned_training_jobs_by_modality={
            str(key): int(value) for key, value in jobs.items()
        },
        confirmatory_epochs=int(projection["confirmatory_epochs"]),
        end_to_end_overhead_multiplier=overhead,
        maximum_approved_runtime_hours=maximum_hours,
    )
    supplied_policy = projection.get("runtime_projection_policy")
    supplied_policy_hash = projection.get("runtime_projection_policy_sha256")
    if supplied_policy != expected_policy or supplied_policy_hash != canonical_json_sha256(
        expected_policy
    ):
        raise ValueError("Resource-pilot runtime projection policy is inconsistent")
    return estimated_hours, maximum_hours


def _read_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _verified_stage_receipt(run_root: Path, stage: str) -> dict[str, object]:
    receipt_path = run_root / "runtime" / "stages" / f"{stage}.json"
    receipt = _read_object(receipt_path)
    claimed_record_hash = receipt.get("record_hash")
    unsigned = {key: value for key, value in receipt.items() if key != "record_hash"}
    if claimed_record_hash != canonical_json_sha256(unsigned):
        raise ValueError(f"Stage receipt record checksum mismatch: {stage}")
    if (
        receipt.get("receipt_type") != "hst_stage"
        or receipt.get("stage") != stage
        or receipt.get("status") != "success"
    ):
        raise ValueError(f"Stage receipt is not a successful HST stage: {stage}")
    paths = receipt.get("output_paths")
    checksums = receipt.get("output_checksums")
    if not isinstance(paths, list) or not paths or not isinstance(checksums, Mapping):
        raise ValueError(f"Stage receipt has no auditable outputs: {stage}")
    for supplied in paths:
        relative = Path(str(supplied))
        candidate = (run_root / relative).resolve()
        try:
            candidate.relative_to(run_root)
        except ValueError as exc:
            raise ValueError(f"Stage output escapes run root: {supplied}") from exc
        if not candidate.is_file():
            raise ValueError(f"Stage output is missing: {supplied}")
        if checksums.get(relative.as_posix()) != stable_file_sha256(candidate):
            raise ValueError(f"Stage output checksum mismatch: {supplied}")
    return receipt


def _receipt_output(run_root: Path, receipt: Mapping[str, object], suffix: str) -> Path:
    matches = [
        (run_root / str(value)).resolve()
        for value in receipt.get("output_paths", [])  # type: ignore[arg-type]
        if Path(str(value)).as_posix().endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one verified stage output ending in {suffix!r}")
    return matches[0]


def _sha256_field(payload: Mapping[str, object], field: str) -> str:
    value = str(payload.get(field, "")).strip().lower()
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"Acceptance source field {field!r} is not a SHA-256")
    return value


def build_pilot_acceptance_candidate(
    *,
    run_root: Path,
    output_path: Path,
) -> dict[str, object]:
    """Build a review-only candidate from a checksum-valid completed pilot.

    This function deliberately cannot create an approved freeze document.
    """
    run_root = Path(run_root).resolve(strict=True)
    receipts = {
        stage: _verified_stage_receipt(run_root, stage) for stage in _REQUIRED_STAGES
    }
    run_ids = {str(receipt.get("run_id", "")) for receipt in receipts.values()}
    if len(run_ids) != 1 or not next(iter(run_ids)):
        raise ValueError("Pilot stage receipts do not belong to one run")

    environment = _read_object(
        _receipt_output(run_root, receipts["preflight"], "audits/environment.json")
    )
    data_contract = _read_object(
        _receipt_output(
            run_root,
            receipts["data_contracts"],
            "contracts/data_contracts_freeze.json",
        )
    )
    pilot = _read_object(
        _receipt_output(
            run_root,
            receipts["base_resource_pilot"],
            "audits/base_resource_pilot_freeze.json",
        )
    )
    frozen_packages = environment.get("pip_freeze")
    if not isinstance(frozen_packages, list) or not frozen_packages:
        raise ValueError("Environment audit does not contain a nonempty pip freeze")
    if _sha256_field(environment, "pip_freeze_sha256") != canonical_json_sha256(
        frozen_packages
    ):
        raise ValueError("Environment audit pip-freeze checksum is inconsistent")

    claimed_contract_hash = _sha256_field(data_contract, "manifest_sha256")
    unsigned_contract = {
        key: value for key, value in data_contract.items() if key != "manifest_sha256"
    }
    if claimed_contract_hash != canonical_json_sha256(unsigned_contract):
        raise ValueError("Data-contract freeze self-checksum is inconsistent")
    contract_metadata = data_contract.get("contract_metadata")
    if not isinstance(contract_metadata, Mapping):
        raise ValueError("Data-contract freeze has no contract metadata")
    release_policy = str(contract_metadata.get("dataset_release_id", "")).casefold()
    label_policy = str(contract_metadata.get("label_column", "")).casefold()
    if not all(
        dataset in release_policy and dataset in label_policy
        for dataset in ("coswara", "coughvid")
    ):
        raise ValueError("Pilot data contract must freeze both Coswara and COUGHVID")

    claimed_pilot_hash = _sha256_field(pilot, "pilot_freeze_hash")
    if claimed_pilot_hash != canonical_json_sha256(
        resource_pilot_freeze_payload(pilot)
    ):
        raise ValueError("Resource-pilot freeze self-checksum is inconsistent")
    if pilot.get("model_metrics_used") is not False:
        raise ValueError("Resource-pilot acceptance must not use model metrics")
    if int(pilot.get("physical_batch_size", 0)) not in {2, 4, 8}:
        raise ValueError("Resource pilot selected an unsupported physical batch size")
    if int(pilot.get("gradient_accumulation", 0)) * int(
        pilot.get("physical_batch_size", 0)
    ) != 8:
        raise ValueError("Resource pilot does not preserve effective batch size 8")
    runtime_projection = pilot.get("runtime_projection")
    if not isinstance(runtime_projection, Mapping):
        raise ValueError("Resource pilot has no auditable full-runtime projection")
    if runtime_projection.get("estimate_is_completion_guarantee") is not False:
        raise ValueError("Resource-pilot runtime estimate must not be represented as a guarantee")
    if runtime_projection.get("within_approved_runtime_ceiling") is not True:
        raise ValueError("Projected full runtime exceeds the frozen operator ceiling")
    estimated_runtime_hours, maximum_runtime_hours = (
        _validate_conservative_runtime_projection(runtime_projection)
    )
    runtime_policy_hash = str(
        runtime_projection.get("runtime_projection_policy_sha256", "")
    )
    freeze_context = pilot.get("freeze_context")
    if (
        not isinstance(freeze_context, Mapping)
        or freeze_context.get("runtime_projection_policy_sha256")
        != runtime_policy_hash
    ):
        raise ValueError("Accepted pilot hash does not bind the runtime projection policy")
    if not (0 < estimated_runtime_hours <= maximum_runtime_hours):
        raise ValueError("Resource-pilot runtime projection is outside its approved ceiling")

    environment_hash = _sha256_field(environment, "pip_freeze_sha256")
    payload: dict[str, object] = {
        "schema_version": 1,
        "candidate_status": "requires_manual_review",
        "source_run_id": next(iter(run_ids)),
        "accepted_hashes": {
            "data_contracts_freeze": claimed_contract_hash,
            "environment_lock": environment_hash,
            "pilot_freeze": claimed_pilot_hash,
        },
        "pip_freeze_hash": environment_hash,
        "manifest_hashes": {},
        "review_gate": {
            "data_contract_includes_coswara_and_coughvid": True,
            "pilot_selected_by_resource_validity_and_throughput_only": True,
            "model_metrics_used": False,
            "projected_runtime_within_frozen_ceiling": True,
            "manual_promotion_required": True,
        },
        "runtime_review": dict(runtime_projection),
        "pilot_selection": {
            "physical_batch_size": int(pilot["physical_batch_size"]),
            "gradient_accumulation": int(pilot["gradient_accumulation"]),
            "amp": bool(pilot.get("amp", False)),
        },
        "source_receipt_sha256": {
            stage: stable_file_sha256(
                run_root / "runtime" / "stages" / f"{stage}.json"
            )
            for stage in _REQUIRED_STAGES
        },
    }
    payload["candidate_record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json(Path(output_path), payload)
    return payload
