from __future__ import annotations

from collections.abc import Mapping
import csv
from datetime import datetime
from hashlib import sha256
import json
from io import BytesIO, StringIO
import os
from pathlib import Path
import re
import stat
import subprocess

import pandas as pd

from .hst_comparators import (
    ENSEMBLE_MODEL_NAME,
    FROZEN_MODEL_NAMES,
    SELECTED_CANDIDATE_MODEL_NAME,
    _NON_FEATURE_COLUMNS,
    _REQUIRED_GENERATION_TABLES,
    _align_features,
    _approval_protocol_binding_sha256,
    _build_compare_is10_executable_recipe,
    _canonical_hash,
    _evidence_domain_sha256,
    _load_authenticated_accepted_freezes,
    _load_trusted_compare_is10_approval_document,
    _normalized_git_remote,
    _path_is_read_only,
    _trusted_project_git_root,
    _validate_manifest,
    _verify_generation_metrics,
    build_compare_is10_feature_contract,
    compare_is10_feature_artifact_sha256,
)
from .hst_runtime import atomic_write_json, canonical_json_sha256, stable_file_sha256


CANONICAL_APPROVAL_RELATIVE_PATH = Path(
    "configs/hst_compare_is10_approval.approved.json"
)
CANONICAL_ACCEPTED_FREEZES_RELATIVE_PATH = Path(
    "configs/hst_comparator_accepted_freezes.approved.json"
)
EXPECTED_COMPLETE_FEATURE_COLUMNS = 10_147
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_RUN_ID_RE = re.compile(r"^hst-[0-9a-f]{20}$")
_REQUIRED_PILOT_FREEZES = {
    "data_contracts_freeze",
    "environment_lock",
    "pilot_freeze",
}


def _read_ascii_object(path: Path, name: str) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"{name} must not use symlink indirection")
    if not path.is_file():
        raise ValueError(f"{name} must be a regular file")
    try:
        payload = json.loads(path.read_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not ASCII JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _stable_bytes(path: Path, name: str) -> tuple[bytes, str]:
    """Read twice and reject replacement or mutation across the trust-boundary read."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a non-symlink regular file")
    before = path.stat()
    first = path.read_bytes()
    middle = path.stat()
    second = path.read_bytes()
    after = path.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(middle) or identity(middle) != identity(after):
        raise ValueError(f"{name} changed or was replaced while being authenticated")
    first_hash = sha256(first).hexdigest()
    if first_hash != sha256(second).hexdigest() or first != second:
        raise ValueError(f"{name} changed or was replaced while being authenticated")
    return first, first_hash


def _stable_ascii_object(path: Path, name: str) -> tuple[dict[str, object], str]:
    payload, digest = _stable_bytes(path, name)
    try:
        decoded = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not canonical ASCII JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object")
    return decoded, digest


def _require_sha256(value: object, name: str) -> str:
    normalized = str(value).strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return normalized


def _inside(root: Path, candidate: Path, name: str) -> Path:
    root = root.resolve(strict=True)
    supplied = Path(candidate)
    lexical = Path(os.path.abspath(supplied))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the trusted root") from exc
    current = root
    for part in relative.parts:
        current /= part
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.exists() and (current.is_symlink() or is_junction()):
            raise ValueError(f"{name} contains symlink indirection")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the trusted root") from exc
    return resolved


def _candidate_output(project_root: Path, output_path: Path) -> Path:
    root = project_root.resolve(strict=True)
    supplied = Path(output_path)
    absolute = supplied if supplied.is_absolute() else root / supplied
    lexical = Path(os.path.abspath(absolute))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError("candidate output escapes the trusted project") from exc
    current = root
    for part in relative.parts:
        current /= part
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.exists() and (current.is_symlink() or is_junction()):
            raise ValueError("candidate output parent contains symlink indirection")
    resolved_parent = lexical.parent.resolve(strict=False)
    resolved = resolved_parent / absolute.name
    forbidden = {
        root / CANONICAL_APPROVAL_RELATIVE_PATH,
        root / CANONICAL_ACCEPTED_FREEZES_RELATIVE_PATH,
    }
    if resolved in forbidden:
        raise ValueError("candidate builder refuses a canonical approved path")
    if resolved.exists() and (resolved.is_symlink() or not resolved.is_file()):
        raise ValueError("candidate output is not a safe regular file")
    return resolved


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"Git-state verification failed: {detail}")
    return result.stdout.strip()


def _clean_git_artifact(root: Path, path: Path, name: str) -> tuple[str, str]:
    path = _inside(root, path, name)
    relative = path.relative_to(root).as_posix()
    try:
        _git(root, "cat-file", "-e", f"HEAD:{relative}")
        if _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative,
        ):
            raise ValueError(f"{name} must be a clean Git-tracked artifact")
        head_blob = _git(root, "rev-parse", f"HEAD:{relative}")
        working_blob = _git(root, "hash-object", "--", relative)
        if head_blob != working_blob:
            raise ValueError(f"{name} differs from the current Git blob")
        commit = _git(root, "log", "-1", "--format=%H", "--", relative).lower()
        if not commit:
            raise ValueError(f"{name} has no committed Git provenance")
        committed_blob = _git(root, "rev-parse", f"{commit}:{relative}")
        if committed_blob != working_blob:
            raise ValueError(f"{name} bytes differ from its provenance commit")
    except ValueError as exc:
        if "clean Git-tracked" in str(exc):
            raise
        raise ValueError(f"{name} must be a clean Git-tracked artifact") from exc
    return commit, working_blob


def _verified_pilot_acceptance(root: Path, path: Path) -> dict[str, object]:
    path = _inside(root, path, "pilot accepted-freezes file")
    _clean_git_artifact(root, path, "pilot accepted-freezes file")
    payload = _read_ascii_object(path, "pilot accepted-freezes file")
    if payload.get("approval_status") != "manually_approved":
        raise ValueError("pilot accepted-freezes file is not manually approved")
    if not str(payload.get("approved_by", "")).strip():
        raise ValueError("pilot accepted-freezes file has no human reviewer identity")
    timestamp_text = str(payload.get("approved_at_utc", "")).strip()
    try:
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("pilot manual approval timestamp is invalid") from exc
    if timestamp.tzinfo is None:
        raise ValueError("pilot manual approval timestamp must include a timezone")
    hashes = payload.get("accepted_hashes")
    if not isinstance(hashes, Mapping) or not _REQUIRED_PILOT_FREEZES <= set(hashes):
        raise ValueError("pilot accepted-freezes file lacks required freeze hashes")
    for key in _REQUIRED_PILOT_FREEZES:
        _require_sha256(hashes[key], f"pilot {key}")
    if _require_sha256(payload.get("pip_freeze_hash"), "pilot pip-freeze hash") != str(
        hashes["environment_lock"]
    ):
        raise ValueError("pilot environment and pip-freeze hashes disagree")
    return payload


def _verified_manifests_stage(
    *,
    project_root: Path,
    run_root: Path,
    receipt_path: Path,
    manifest_name: str,
) -> tuple[Path, pd.DataFrame, str, dict[str, object], str]:
    if manifest_name != "aligned_comparator":
        raise ValueError("comparator approval is restricted to manifest_name='aligned_comparator'")
    project_root = project_root.resolve(strict=True)
    supplied_run_root = Path(run_root)
    if supplied_run_root.is_symlink():
        raise ValueError("run root must not use symlink indirection")
    run_root = supplied_run_root.resolve(strict=True)
    canonical_parent = project_root / "data" / "outputs" / "hst"
    if run_root.parent != canonical_parent or _CONTENT_RUN_ID_RE.fullmatch(run_root.name) is None:
        raise ValueError("run root must be the canonical content-addressed HST run directory")
    receipt_path = _inside(run_root, receipt_path, "manifests stage receipt")
    expected_receipt = run_root / "runtime" / "stages" / "manifests.json"
    if receipt_path != expected_receipt:
        raise ValueError("manifests stage receipt must use its canonical run path")
    receipt, receipt_file_hash = _stable_ascii_object(
        receipt_path, "manifests stage receipt"
    )
    claimed_record_hash = _require_sha256(
        receipt.get("record_hash"), "manifests stage receipt record hash"
    )
    unsigned = {key: value for key, value in receipt.items() if key != "record_hash"}
    if claimed_record_hash != canonical_json_sha256(unsigned):
        raise ValueError("manifests stage receipt checksum mismatch")
    if (
        receipt.get("receipt_type") != "hst_stage"
        or receipt.get("stage") != "manifests"
        or receipt.get("status") != "success"
        or receipt.get("run_id") != run_root.name
    ):
        raise ValueError("manifests stage receipt is not a successful manifests stage")
    for field in (
        "configuration_hash",
        "source_hash",
        "dependency_lock_hash",
        "fingerprint",
    ):
        _require_sha256(receipt.get(field), f"manifests stage receipt {field}")
    accepted_hashes = receipt.get("accepted_hashes")
    if not isinstance(accepted_hashes, Mapping):
        raise ValueError("manifests stage receipt lacks accepted hash identity")
    paths = receipt.get("output_paths")
    checksums = receipt.get("output_checksums")
    if not isinstance(paths, list) or not paths or not isinstance(checksums, Mapping):
        raise ValueError("manifests stage receipt has no checksum-validated outputs")
    verified: dict[str, Path] = {}
    for supplied in paths:
        relative = Path(str(supplied))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifests stage output escapes the run root")
        relative_text = relative.as_posix()
        if relative_text in verified:
            raise ValueError("manifests stage receipt repeats an output")
        output = _inside(run_root, run_root / relative, "manifests stage output")
        expected = _require_sha256(
            checksums.get(relative_text), f"manifests stage output {relative_text}"
        )
        _payload, actual = _stable_bytes(output, f"manifests stage output {relative_text}")
        if actual != expected:
            raise ValueError(f"manifests stage output checksum mismatch: {relative_text}")
        verified[relative_text] = output

    index_matches = [
        path
        for relative, path in verified.items()
        if relative.endswith("manifests/manifest_index.json")
    ]
    if len(index_matches) != 1:
        raise ValueError("manifests stage must authenticate exactly one manifest index")
    index, _index_hash = _stable_ascii_object(index_matches[0], "manifest index")
    if index.get("schema_version") != 1 or not isinstance(index.get("manifests"), Mapping):
        raise ValueError("manifest index schema is invalid")
    descriptor = index["manifests"].get(str(manifest_name))  # type: ignore[index]
    if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256", "rows"}:
        raise ValueError(f"manifest index has no exact descriptor for {manifest_name!r}")
    relative = Path(str(descriptor["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("indexed manifest path escapes the run root")
    relative_text = relative.as_posix()
    if relative_text not in verified:
        raise ValueError("indexed manifest is absent from the stage receipt")
    path = verified[relative_text]
    manifest_bytes, file_hash = _stable_bytes(path, "aligned comparator manifest")
    if _require_sha256(descriptor["sha256"], "indexed manifest hash") != file_hash:
        raise ValueError("manifest index checksum differs from the stage receipt")
    from io import BytesIO

    manifest = _validate_manifest(pd.read_csv(BytesIO(manifest_bytes), low_memory=False))
    if int(descriptor["rows"]) != len(manifest):
        raise ValueError("manifest index row count differs from the manifest")
    if stable_file_sha256(receipt_path) != receipt_file_hash:
        raise ValueError("manifests stage receipt changed after output authentication")
    return path, manifest, file_hash, receipt, receipt_file_hash


def _verified_feature_table(
    path: Path,
    manifest: pd.DataFrame,
    *,
    expected_columns: int,
) -> tuple[dict[str, object], str, int, int]:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("feature table must not use symlink indirection")
    path = supplied.resolve(strict=True)
    if not path.is_file():
        raise ValueError("feature table must be a regular file")
    with path.open("rb") as handle:
        raw_header = handle.readline()
    with path.open("r", encoding="utf-8", newline="") as handle:
        try:
            raw_columns = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError("feature table is empty") from exc
    if len(raw_columns) != len(set(raw_columns)):
        raise ValueError("feature table contains duplicate raw column names")
    if len(raw_columns) != expected_columns:
        raise ValueError(f"complete feature table must contain exactly {expected_columns} columns")
    header = pd.read_csv(path, nrows=0)
    if len(header.columns) != expected_columns:
        raise ValueError(f"complete feature table must contain exactly {expected_columns} columns")
    features = pd.read_csv(path, low_memory=False)
    if len(features.columns) != expected_columns:
        raise ValueError(f"complete feature table must contain exactly {expected_columns} columns")
    feature_columns = tuple(
        str(column) for column in features.columns if column not in _NON_FEATURE_COLUMNS
    )
    contract = build_compare_is10_feature_contract(
        features, ordered_feature_columns=feature_columns
    )
    _align_features(features, manifest, contract)
    artifact_hash = compare_is10_feature_artifact_sha256(features)
    contract_payload = {
        "ordered_feature_columns": list(contract.ordered_feature_columns),
        "feature_dtypes": list(contract.feature_dtypes),
        "schema_sha256": contract.schema_sha256,
        "missing_policy": contract.missing_policy,
    }
    contract_payload["header_sha256"] = sha256(raw_header).hexdigest()
    return contract_payload, artifact_hash, len(features), len(feature_columns)


def _approval_record_payload(
    *,
    feature_schema_sha256: str,
    feature_artifact_sha256: str,
    manifest: pd.DataFrame,
    executable_recipe: Mapping[str, object],
    reviewed_input_bindings: Mapping[str, object],
) -> dict[str, object]:
    scientific = manifest["scientific_configuration_fingerprint"].astype(str).unique()
    eligibility = manifest["eligibility_alignment_fingerprint"].astype(str).unique()
    if len(scientific) != 1 or len(eligibility) != 1:
        raise ValueError("manifest scientific/eligibility bindings are not singular")
    payload: dict[str, object] = {
        "approval_record_version": 3,
        "approval_status": "MANUAL_REVIEW_REQUIRED",
        "approval_id": "SET_DURING_MANUAL_REVIEW",
        "approved_at_utc": "SET_DURING_MANUAL_REVIEW",
        "feature_schema_sha256": feature_schema_sha256,
        "feature_artifact_sha256": feature_artifact_sha256,
        "manifest_sha256": str(manifest["manifest_sha256"].iloc[0]),
        "scientific_configuration_fingerprint": str(scientific[0]),
        "eligibility_alignment_fingerprint": str(eligibility[0]),
        "protocol_binding_sha256": _approval_protocol_binding_sha256(manifest),
        "comparator_configuration": _frozen_comparator_configuration(),
        "executable_recipe": dict(executable_recipe),
        "reviewed_input_bindings": dict(reviewed_input_bindings),
    }
    payload["approval_record_sha256"] = _canonical_hash(payload)
    return payload


def _frozen_comparator_configuration() -> dict[str, object]:
    return {
        "selected_feature_k": 800,
        "ranker": "lightgbm",
        "selection_scope": "per_modality_mean",
        "model_names": list(FROZEN_MODEL_NAMES),
        "ensemble_policy": "uniform_probability_mean",
    }


def build_comparator_approval_candidate(
    *,
    project_root: Path,
    run_root: Path,
    manifests_receipt_path: Path,
    manifest_name: str,
    feature_table_path: Path,
    pilot_accepted_freezes_path: Path,
    environment_lock_path: Path,
    runtime_random_state: int,
    output_path: Path,
) -> dict[str, object]:
    """Prepare, but never approve, an exact comparator approval record."""
    root = _trusted_project_git_root(project_root)
    output = _candidate_output(root, output_path)
    pilot_path = _inside(root, pilot_accepted_freezes_path, "pilot accepted-freezes file")
    pilot = _verified_pilot_acceptance(root, pilot_path)
    environment_path = _inside(root, environment_lock_path, "environment-lock file")
    environment_commit, environment_blob = _clean_git_artifact(
        root, environment_path, "environment-lock file"
    )
    environment_hash = stable_file_sha256(environment_path)
    manifest_path, manifest, manifest_file_hash, receipt, receipt_file_hash = _verified_manifests_stage(
        project_root=root,
        run_root=run_root,
        receipt_path=manifests_receipt_path,
        manifest_name=manifest_name,
    )
    feature_path = _inside(root, feature_table_path, "feature table")
    feature_file_hash_before = stable_file_sha256(feature_path)
    contract, feature_artifact_hash, feature_rows, feature_count = _verified_feature_table(
        feature_path,
        manifest,
        expected_columns=EXPECTED_COMPLETE_FEATURE_COLUMNS,
    )
    feature_file_hash_after = stable_file_sha256(feature_path)
    if feature_file_hash_before != feature_file_hash_after:
        raise ValueError("feature table changed while it was being validated")
    recipe = _build_compare_is10_executable_recipe(
        root,
        random_state=int(runtime_random_state),
        accepted_environment_lock_sha256=environment_hash,
    )
    pilot_hash = stable_file_sha256(pilot_path)
    reviewed_bindings = {
        "manifest_name": "aligned_comparator",
        "manifest_path": manifest_path.relative_to(Path(run_root).resolve()).as_posix(),
        "manifest_file_sha256": manifest_file_hash,
        "manifests_stage_receipt_sha256": receipt_file_hash,
        "manifests_stage_record_hash": receipt["record_hash"],
        "run_identity": {
            key: receipt[key]
            for key in (
                "run_id",
                "configuration_hash",
                "source_hash",
                "dependency_lock_hash",
                "fingerprint",
                "accepted_hashes",
            )
        },
        "feature_table_path": feature_path.relative_to(root).as_posix(),
        "feature_table_file_sha256": feature_file_hash_after,
        "feature_table_header_sha256": contract["header_sha256"],
        "feature_artifact_sha256": feature_artifact_hash,
        "feature_schema_sha256": contract["schema_sha256"],
        "feature_table_columns": EXPECTED_COMPLETE_FEATURE_COLUMNS,
        "feature_columns": feature_count,
        "feature_rows": feature_rows,
        "pilot_accepted_freezes_path": pilot_path.relative_to(root).as_posix(),
        "pilot_accepted_freezes_sha256": pilot_hash,
        "pilot_accepted_hashes": dict(pilot["accepted_hashes"]),
        "environment_lock_path": environment_path.relative_to(root).as_posix(),
        "environment_lock_sha256": environment_hash,
        "environment_lock_git_commit": environment_commit,
        "environment_lock_git_blob": environment_blob,
        "project_git_head": _git(root, "rev-parse", "HEAD").lower(),
        "project_git_remote": _git(root, "remote", "get-url", "origin"),
        "executable_recipe_sha256": recipe["recipe_sha256"],
        "executable_source_sha256": recipe.get("executable_source_sha256", {}),
        "runtime_random_state": int(runtime_random_state),
    }
    proposed = _approval_record_payload(
        feature_schema_sha256=str(contract["schema_sha256"]),
        feature_artifact_sha256=feature_artifact_hash,
        manifest=manifest,
        executable_recipe=recipe,
        reviewed_input_bindings=reviewed_bindings,
    )
    payload = {
            "candidate_schema_version": 1,
            "candidate_type": "hst_compare_is10_approval_record",
            "candidate_status": "requires_manual_review",
            "canonical_target": CANONICAL_APPROVAL_RELATIVE_PATH.as_posix(),
            "manual_promotion": {
                "required": True,
                "steps": [
                    "review every source binding and proposed field",
                    "set approval_status to approved",
                    "set approval_id and approved_at_utc explicitly",
                    "recompute approval_record_sha256 over all other approval fields",
                    "write only the proposed approval record to the canonical target",
                    "commit the canonical record and make it read-only",
                ],
                "manual_fields": ["approval_status", "approval_id", "approved_at_utc"],
            },
            "source_bindings": reviewed_bindings,
            "proposed_approval_record": proposed,
        }
    atomic_write_json(output, payload)
    return payload


def _validate_approved_record_structure(
    payload: Mapping[str, object],
    *,
    root: Path,
    environment_hash: str,
    runtime_random_state: int,
) -> None:
    required = {
        "approval_record_version",
        "approval_status",
        "approval_id",
        "approved_at_utc",
        "feature_schema_sha256",
        "feature_artifact_sha256",
        "manifest_sha256",
        "scientific_configuration_fingerprint",
        "eligibility_alignment_fingerprint",
        "protocol_binding_sha256",
        "comparator_configuration",
        "executable_recipe",
        "reviewed_input_bindings",
        "approval_record_sha256",
    }
    if set(payload) != required:
        raise ValueError("canonical comparator approval schema is invalid")
    if payload.get("approval_record_version") != 3 or payload.get("approval_status") != "approved":
        raise ValueError("canonical comparator approval is not an approved version-3 record")
    if not str(payload.get("approval_id", "")).strip():
        raise ValueError("canonical comparator approval has no approval identity")
    timestamp_text = str(payload.get("approved_at_utc", ""))
    try:
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("canonical comparator approval timestamp is invalid") from exc
    if timestamp.tzinfo is None or not timestamp_text.endswith("Z"):
        raise ValueError("canonical comparator approval timestamp must be UTC with a Z suffix")
    for field in (
        "feature_schema_sha256",
        "feature_artifact_sha256",
        "manifest_sha256",
        "scientific_configuration_fingerprint",
        "eligibility_alignment_fingerprint",
        "protocol_binding_sha256",
    ):
        _require_sha256(payload.get(field), f"canonical approval {field}")
    if payload.get("comparator_configuration") != _frozen_comparator_configuration():
        raise ValueError("canonical approval differs from the frozen comparator configuration")
    supplied_hash = _require_sha256(
        payload.get("approval_record_sha256"), "approval record hash"
    )
    unsigned = {key: value for key, value in payload.items() if key != "approval_record_sha256"}
    if supplied_hash != _canonical_hash(unsigned):
        raise ValueError("canonical comparator approval self hash is invalid")
    live_recipe = _build_compare_is10_executable_recipe(
        root,
        random_state=int(runtime_random_state),
        accepted_environment_lock_sha256=environment_hash,
    )
    if payload.get("executable_recipe") != live_recipe:
        raise ValueError("canonical comparator approval recipe differs from the live recipe")
    bindings = payload.get("reviewed_input_bindings")
    if not isinstance(bindings, Mapping) or bindings.get("manifest_name") != "aligned_comparator":
        raise ValueError("canonical comparator approval lacks aligned reviewed input bindings")
    for field in (
        "manifest_file_sha256",
        "manifests_stage_receipt_sha256",
        "manifests_stage_record_hash",
        "feature_table_file_sha256",
        "feature_table_header_sha256",
        "feature_artifact_sha256",
        "feature_schema_sha256",
        "pilot_accepted_freezes_sha256",
        "environment_lock_sha256",
        "executable_recipe_sha256",
    ):
        _require_sha256(bindings.get(field), f"reviewed input binding {field}")


def build_comparator_accepted_freezes_candidate(
    *,
    project_root: Path,
    approval_record_path: Path,
    pilot_accepted_freezes_path: Path,
    environment_lock_path: Path,
    project_id: str,
    expected_remote_url: str,
    runtime_random_state: int,
    output_path: Path,
) -> dict[str, object]:
    """Prepare the canonical accepted-freezes payload after approval was committed."""
    root = _trusted_project_git_root(project_root)
    output = _candidate_output(root, output_path)
    if not str(project_id).strip():
        raise ValueError("project_id must be supplied explicitly")
    actual_remote = _normalized_git_remote(_git(root, "remote", "get-url", "origin"))
    expected_remote = _normalized_git_remote(expected_remote_url)
    if not expected_remote or actual_remote != expected_remote:
        raise ValueError("trusted Git remote does not match the explicitly expected remote")
    pilot_path = _inside(root, pilot_accepted_freezes_path, "pilot accepted-freezes file")
    _verified_pilot_acceptance(root, pilot_path)
    environment_path = _inside(root, environment_lock_path, "environment-lock file")
    environment_commit, environment_blob = _clean_git_artifact(
        root, environment_path, "environment-lock file"
    )
    environment_hash = stable_file_sha256(environment_path)
    approval_path = _inside(root, approval_record_path, "canonical comparator approval")
    if approval_path != root / CANONICAL_APPROVAL_RELATIVE_PATH:
        raise ValueError("comparator approval must use its canonical approved path")
    if not _path_is_read_only(approval_path):
        raise ValueError("canonical comparator approval must be read-only")
    approval_commit, approval_blob = _clean_git_artifact(
        root, approval_path, "canonical comparator approval"
    )
    approval = _read_ascii_object(approval_path, "canonical comparator approval")
    _validate_approved_record_structure(
        approval,
        root=root,
        environment_hash=environment_hash,
        runtime_random_state=runtime_random_state,
    )
    proposed = {
        "accepted_freezes_version": 1,
        "project_identity": {
            "project_id": str(project_id).strip(),
            "expected_remote_url": str(expected_remote_url).strip(),
            "accepted_ancestor_commit": approval_commit,
        },
        "compare_is10_approval": {
            "relative_path": CANONICAL_APPROVAL_RELATIVE_PATH.as_posix(),
            "commit_sha": approval_commit,
        },
        "environment_lock": {
            "relative_path": environment_path.relative_to(root).as_posix(),
            "sha256": environment_hash,
        },
        "accepted_generation_manifests": {},
    }
    payload = {
            "candidate_schema_version": 1,
            "candidate_type": "hst_comparator_accepted_freezes",
            "candidate_status": "requires_manual_review",
            "canonical_target": CANONICAL_ACCEPTED_FREEZES_RELATIVE_PATH.as_posix(),
            "manual_promotion": {
                "required": True,
                "steps": [
                    "review the committed approval, project identity, and environment lock",
                    "copy only proposed_accepted_freezes to the canonical target",
                    "commit the canonical target and make it read-only",
                    "pass its exact file SHA-256 to confirmatory execution",
                ],
            },
            "source_bindings": {
                "approval_record_sha256": stable_file_sha256(approval_path),
                "approval_record_content_sha256": approval["approval_record_sha256"],
                "approval_git_commit": approval_commit,
                "approval_git_blob": approval_blob,
                "pilot_accepted_freezes_sha256": stable_file_sha256(pilot_path),
                "environment_lock_sha256": environment_hash,
                "environment_lock_git_commit": environment_commit,
                "environment_lock_git_blob": environment_blob,
            },
            "proposed_accepted_freezes": proposed,
        }
    atomic_write_json(output, payload)
    return payload


def _authenticate_generation_without_acceptance(
    *,
    generation_manifest_path: Path,
    current_receipt_path: Path,
    approval: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    manifest_supplied = Path(generation_manifest_path)
    receipt_supplied = Path(current_receipt_path)
    if manifest_supplied.is_symlink() or receipt_supplied.is_symlink():
        raise ValueError("generation manifest/current receipt must not use symlinks")
    manifest_path = manifest_supplied.resolve(strict=True)
    receipt_path = receipt_supplied.resolve(strict=True)
    if manifest_path.name != "manifest.json" or manifest_path.parent.parent.name != "generations":
        raise ValueError("generation manifest path does not follow the atomic generation layout")
    expected_receipt = manifest_path.parents[2] / "current.json"
    if receipt_path != expected_receipt:
        raise ValueError("current.json does not belong to the supplied generation")
    if not _path_is_read_only(manifest_path) or not _path_is_read_only(receipt_path):
        raise ValueError("generation manifest/current receipt must be read-only before acceptance")
    generation, manifest_hash = _stable_ascii_object(manifest_path, "generation manifest")
    receipt, initial_receipt_hash = _stable_ascii_object(receipt_path, "current.json")
    if set(receipt) != {
        "generation_id",
        "generation_manifest_sha256",
        "receipt_sha256",
    }:
        raise ValueError("current.json schema is invalid")
    receipt_hash = _require_sha256(receipt["receipt_sha256"], "current receipt hash")
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if receipt_hash != _canonical_hash(unsigned_receipt):
        raise ValueError("current.json receipt checksum is invalid")
    generation_id = str(generation.get("generation_id", ""))
    if (
        not generation_id
        or generation_id != manifest_path.parent.name
        or generation_id != str(receipt["generation_id"])
    ):
        raise ValueError("generation identity differs across path, manifest, and receipt")
    if _require_sha256(
        receipt["generation_manifest_sha256"], "current generation manifest hash"
    ) != manifest_hash:
        raise ValueError("current.json does not authenticate the generation manifest")
    required_generation_fields = {
        "generation_manifest_version",
        "generation_id",
        "files",
        "model_names",
        "ensemble_model",
        "selected_candidate_model",
        "execution_class",
        "confirmatory_eligible",
        "test_mode",
        "reporting_guard",
        "evidence_domain_sha256",
        "executable_recipe_sha256",
        "approval_id",
        "approval_record_sha256",
        "approval_git_commit",
        "approval_git_blob",
    }
    if set(generation) != required_generation_fields:
        raise ValueError("generation manifest schema differs from the frozen version-2 schema")
    if (
        generation.get("generation_manifest_version") != 2
        or generation.get("execution_class") != "confirmatory"
        or generation.get("confirmatory_eligible") is not True
        or generation.get("test_mode") is not False
    ):
        raise ValueError("generation is not confirmatory version-2 evidence")
    expected_bindings = {
        "approval_id": approval["approval_id"],
        "approval_record_sha256": approval["approval_record_sha256"],
        "approval_git_commit": approval["approval_git_commit"],
        "approval_git_blob": approval["approval_git_blob"],
        "executable_recipe_sha256": approval["executable_recipe_sha256"],
        "evidence_domain_sha256": _evidence_domain_sha256(
            "confirmatory", str(approval["approval_record_sha256"])
        ),
    }
    for key, expected in expected_bindings.items():
        if generation.get(key) != expected:
            raise ValueError(f"generation {key} differs from the committed approval")
    if generation.get("model_names") != list(FROZEN_MODEL_NAMES):
        raise ValueError("generation model bank differs from the frozen comparator")
    if generation.get("ensemble_model") != ENSEMBLE_MODEL_NAME:
        raise ValueError("generation ensemble differs from the frozen comparator")
    if generation.get("selected_candidate_model") != SELECTED_CANDIDATE_MODEL_NAME:
        raise ValueError("generation selected candidate differs from the frozen comparator")
    files = generation.get("files")
    if not isinstance(files, Mapping) or not _REQUIRED_GENERATION_TABLES <= set(files):
        raise ValueError("generation manifest lacks the required evidence tables")
    generation_root = manifest_path.parent
    normalized_paths: set[str] = set()
    snapshots: dict[str, str] = {}
    for supplied, descriptor in files.items():
        relative = Path(str(supplied))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("generation artifact path escapes the generation directory")
        relative_text = relative.as_posix()
        if relative_text in normalized_paths:
            raise ValueError("generation manifest repeats an artifact path")
        normalized_paths.add(relative_text)
        try:
            resolved = _inside(
                generation_root,
                generation_root / relative,
                "generation artifact path",
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"generation artifact is missing or unsafe: {relative_text}") from exc
        if not resolved.is_file():
            raise ValueError(f"generation artifact is missing or unsafe: {relative_text}")
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"sha256", "size_bytes"}:
            raise ValueError(f"generation artifact descriptor is invalid: {relative_text}")
        if not _path_is_read_only(resolved):
            raise ValueError(f"generation artifact must be read-only before acceptance: {relative_text}")
        payload, payload_hash = _stable_bytes(resolved, f"generation artifact {relative_text}")
        if (
            _require_sha256(descriptor["sha256"], f"generation artifact {relative_text}")
            != payload_hash
            or int(descriptor["size_bytes"]) != len(payload)
        ):
            raise ValueError(f"generation artifact checksum/size mismatch: {relative_text}")
        snapshots[relative_text] = payload_hash

    actual_files = {
        path.relative_to(generation_root).as_posix()
        for path in generation_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != normalized_paths:
        raise ValueError("generation directory contains undeclared or missing files")

    tables: dict[str, pd.DataFrame] = {}
    for relative_text in sorted(_REQUIRED_GENERATION_TABLES):
        path = generation_root / relative_text
        payload, digest = _stable_bytes(path, f"generation table {relative_text}")
        try:
            header = next(csv.reader(StringIO(payload.decode("utf-8-sig"))))
        except (StopIteration, UnicodeDecodeError) as exc:
            raise ValueError(f"generation table is not a nonempty UTF-8 CSV: {relative_text}") from exc
        if not header or len(header) != len(set(header)):
            raise ValueError(f"generation table has duplicate or empty CSV headers: {relative_text}")
        try:
            frame = pd.read_csv(BytesIO(payload), low_memory=False)
        except Exception as exc:
            raise ValueError(f"generation table cannot be parsed: {relative_text}") from exc
        if frame.empty:
            raise ValueError(f"generation table contains no production rows: {relative_text}")
        tables[relative_text] = frame
        if stable_file_sha256(path) != digest:
            raise ValueError(f"generation table changed after parsing: {relative_text}")

    required_columns = {
        "comparator_predictions.csv": {
            "run_id", "protocol", "fold", "dataset", "participant_key",
            "recording_key", "split", "modality", "model", "label_binary",
            "probability", "checkpoint_hash", "manifest_sha256",
            "approval_record_sha256",
        },
        "comparator_participant_predictions.csv": {
            "run_id", "protocol", "fold", "dataset", "participant_key", "split",
            "modality", "model", "label_binary", "probability", "checkpoint_hash",
        },
        "comparator_metrics.csv": {
            "run_id", "protocol", "fold", "dataset", "split", "modality", "model",
            "auroc", "auprc", "balanced_accuracy", "f1", "checkpoint_hash",
        },
        "comparator_alignment_audit.csv": {
            "protocol", "fold", "dataset", "split", "modality", "aligned",
        },
        "comparator_feature_selection.csv": {
            "protocol", "fold", "feature", "selected",
            "feature_schema_sha256", "feature_artifact_sha256",
        },
        "comparator_model_audit.csv": {
            "run_id", "protocol", "fold", "modality", "model", "checkpoint_hash",
            "model_artifact", "manifest_sha256", "feature_schema_sha256",
            "feature_artifact_sha256", "approval_record_sha256",
        },
        "comparator_candidate_selection.csv": {
            "run_id", "protocol", "fold", "modality", "candidate_model",
            "selected", "selected_candidate_model", "selected_candidate_source_model",
            "selected_candidate_checkpoint_hash", "selected_candidate_model_artifact",
        },
    }
    for name, columns in required_columns.items():
        missing = sorted(columns - set(tables[name].columns))
        if missing:
            raise ValueError(f"generation table schema is incomplete for {name}: {missing}")

    model_audit = tables["comparator_model_audit.csv"]
    manifested_models = {name for name in normalized_paths if name.startswith("models/")}
    audited_models = set(model_audit["model_artifact"].astype(str))
    if not audited_models or manifested_models != audited_models:
        raise ValueError("generation model bundle set differs from the model audit")
    required_models = {*FROZEN_MODEL_NAMES, ENSEMBLE_MODEL_NAME, SELECTED_CANDIDATE_MODEL_NAME}
    for _, group in model_audit.groupby(["protocol", "fold", "modality"], dropna=False):
        if set(group["model"].astype(str)) != required_models:
            raise ValueError("generation model audit lacks a complete model bundle bank")
    for row in model_audit.itertuples(index=False):
        relative = str(row.model_artifact)
        descriptor = files[relative]
        if str(row.checkpoint_hash) != str(descriptor["sha256"]):
            raise ValueError("model audit checksum differs from declared model bundle")
        if snapshots.get(relative) != str(descriptor["sha256"]):
            raise ValueError("model bundle snapshot differs from its declared checksum")

    # Model bundles are intentionally treated as opaque bytes before manual
    # acceptance. Deserializing an unaccepted pickle would execute attacker-
    # controlled code. Semantic bundle loading occurs only after the exact
    # generation manifest hash is present in the canonical accepted-freezes file.

    reviewed = approval.get("reviewed_input_bindings", {})
    approved_manifest = str(approval["manifest_sha256"])
    if isinstance(reviewed, Mapping) and str(reviewed.get("manifest_name")) != "aligned_comparator":
        raise ValueError("generation approval is not bound to aligned_comparator")
    for name in ("comparator_predictions.csv", "comparator_model_audit.csv"):
        if set(tables[name]["manifest_sha256"].astype(str)) != {approved_manifest}:
            raise ValueError("generation manifest binding differs from the approved source manifest")
        if set(tables[name]["approval_record_sha256"].astype(str)) != {
            str(approval["approval_record_sha256"])
        }:
            raise ValueError("generation table approval binding differs from canonical approval")

    _verify_generation_metrics(generation_root)
    for relative_text, expected_hash in snapshots.items():
        if stable_file_sha256(generation_root / relative_text) != expected_hash:
            raise ValueError("generation changed after semantic validation")
    if stable_file_sha256(manifest_path) != manifest_hash:
        raise ValueError("generation manifest changed during acceptance validation")
    _receipt, final_receipt_hash = _stable_ascii_object(receipt_path, "current.json")
    if final_receipt_hash != initial_receipt_hash:
        raise ValueError("current.json changed during acceptance validation")
    return generation, manifest_hash


def build_comparator_generation_acceptance_candidate(
    *,
    project_root: Path,
    approval_record_path: Path,
    accepted_freezes_path: Path,
    expected_accepted_freezes_sha256: str,
    generation_manifest_path: Path,
    current_receipt_path: Path,
    runtime_random_state: int,
    output_path: Path,
) -> dict[str, object]:
    """Prepare a review-only accepted-freezes update for one exact generation."""
    root = _trusted_project_git_root(project_root)
    output = _candidate_output(root, output_path)
    generation_path = _inside(root, generation_manifest_path, "generation manifest")
    receipt_path = _inside(root, current_receipt_path, "current generation receipt")
    accepted = _load_authenticated_accepted_freezes(
        accepted_freezes_path,
        expected_accepted_freezes_sha256=expected_accepted_freezes_sha256,
        trusted_project_repository_root=root,
    )
    approval = _load_trusted_compare_is10_approval_document(
        approval_record_path,
        trusted_project_repository_root=root,
        accepted_freezes_path=accepted_freezes_path,
        expected_accepted_freezes_sha256=expected_accepted_freezes_sha256,
        runtime_random_state=int(runtime_random_state),
    )
    generation, generation_hash = _authenticate_generation_without_acceptance(
        generation_manifest_path=generation_path,
        current_receipt_path=receipt_path,
        approval=approval,
    )
    generation_id = str(generation["generation_id"])
    prior_generations = dict(accepted["accepted_generation_manifests"])
    existing = prior_generations.get(generation_id)
    if existing is not None and str(existing) != generation_hash:
        raise ValueError("generation id is already accepted with a different manifest hash")
    prior_generations[generation_id] = generation_hash
    proposed = {
        key: accepted[key]
        for key in (
            "accepted_freezes_version",
            "project_identity",
            "compare_is10_approval",
            "environment_lock",
        )
    }
    proposed["accepted_generation_manifests"] = dict(sorted(prior_generations.items()))
    payload = {
            "candidate_schema_version": 1,
            "candidate_type": "hst_comparator_generation_acceptance",
            "candidate_status": "requires_manual_review",
            "canonical_target": CANONICAL_ACCEPTED_FREEZES_RELATIVE_PATH.as_posix(),
            "manual_promotion": {
                "required": True,
                "steps": [
                    "review current.json, the generation manifest, and every file checksum",
                    "copy only proposed_accepted_freezes to the canonical target",
                    "commit the updated canonical target and make it read-only",
                    "pass the updated file SHA-256 to fusion/reporting",
                ],
            },
            "source_bindings": {
                "prior_accepted_freezes_sha256": str(
                    accepted["accepted_freezes_sha256"]
                ),
                "approval_record_sha256": approval["approval_record_sha256"],
                "current_receipt_sha256": stable_file_sha256(
                    receipt_path
                ),
            },
            "authenticated_generation": {
                "generation_id": generation_id,
                "generation_manifest_sha256": generation_hash,
            },
            "proposed_accepted_freezes": proposed,
        }
    atomic_write_json(output, payload)
    return payload
