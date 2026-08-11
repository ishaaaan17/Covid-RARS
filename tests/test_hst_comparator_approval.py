from __future__ import annotations

from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from covid_rars.hst_runtime import canonical_json_sha256, stable_file_sha256


CANONICAL_APPROVAL = Path(
    "configs/hst_compare_is10_approval.approved.json"
)
CANONICAL_ACCEPTED = Path(
    "configs/hst_comparator_accepted_freezes.approved.json"
)


class _DeterministicRanker:
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "_DeterministicRanker":
        self.feature_importances_ = np.arange(len(x.columns), 0, -1, dtype=float)
        return self


class _DeterministicEstimator:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "_DeterministicEstimator":
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(x.iloc[:, 0], errors="coerce").fillna(0.0).to_numpy(float)
        probability = 1.0 / (1.0 + np.exp(-np.clip(values, -6.0, 6.0)))
        return np.column_stack([1.0 - probability, probability])


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_ascii_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
            "ascii"
        )
    )


def _project_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    import covid_rars.hst_comparators as comparators

    root = tmp_path / "project"
    package_root = root
    source_root = package_root / "src" / "covid_rars"
    source_root.mkdir(parents=True)
    live_source_root = Path(comparators.__file__).resolve().parent
    for name in (
        "hst_comparators.py",
        "hst_data_contracts.py",
        "hst_protocols.py",
        "metrics.py",
        "strong_baseline.py",
    ):
        shutil.copy2(live_source_root / name, source_root / name)
    for name in ("requirements-hst.txt", "requirements-gpu.txt"):
        shutil.copy2(live_source_root.parents[1] / name, package_root / name)

    pilot = package_root / "reports" / "hst" / "accepted_freezes.json"
    _write_ascii_json(
        pilot,
        {
            "schema_version": 1,
            "approval_status": "manually_approved",
            "approved_by": "independent-reviewer",
            "approved_at_utc": "2026-08-02T10:00:00Z",
            "accepted_hashes": {
                "data_contracts_freeze": "a" * 64,
                "environment_lock": "b" * 64,
                "pilot_freeze": "c" * 64,
            },
            "pip_freeze_hash": "b" * 64,
            "manifest_hashes": {},
        },
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "approval-test@example.invalid")
    _git(root, "config", "user.name", "Approval Test")
    _git(root, "remote", "add", "origin", "https://example.invalid/covid-rars.git")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "Initial trusted project")
    monkeypatch.setattr(comparators, "__file__", str(source_root / "hst_comparators.py"))
    return root, pilot, package_root / "requirements-hst.txt"


def _manifest() -> pd.DataFrame:
    from covid_rars.hst_protocols import _finalize_manifest

    rows: list[dict[str, object]] = []
    split_people = {
        "train": (("tr0", "negative"), ("tr1", "positive")),
        "validation": (("va0", "negative"), ("va1", "positive")),
        "test": (("te0", "negative"), ("te1", "positive")),
    }
    for split, people in split_people.items():
        for participant, label in people:
            recording = f"{participant}_cough"
            rows.append(
                {
                    "run_id": "approval-source-run",
                    "protocol": "track_a",
                    "fold": 1,
                    "cohort": "all_eligible",
                    "split": split,
                    "dataset": "coswara",
                    "participant_key": f"coswara::{participant}",
                    "recording_key": f"coswara::{recording}",
                    "modality": "cough",
                    "label_binary": label,
                    "representation_id": "paper_logmel_224",
                    "source_audio_sha256": sha256(recording.encode("ascii")).hexdigest(),
                    "scientific_configuration_fingerprint": "d" * 64,
                    "eligibility_alignment_fingerprint": "e" * 64,
                    "analysis_scope": "internal_performance",
                    "analysis_role": "primary",
                    "estimand_id": "track_a_participant_auroc",
                    "multiplicity_family": "primary_internal",
                    "analysis_mode": "confirmatory",
                    "confirmatory_protocol": True,
                }
            )
    return _finalize_manifest(pd.DataFrame(rows))


def _feature_table(manifest: pd.DataFrame, *, n_columns: int = 10_147) -> pd.DataFrame:
    identity_columns = [
        "dataset",
        "participant_key",
        "recording_key",
        "modality",
        "label_binary",
        "source_audio_sha256",
    ]
    identity = manifest[identity_columns].drop_duplicates().reset_index(drop=True)
    feature_count = n_columns - len(identity_columns)
    values = np.arange(len(identity), dtype=np.float32)
    feature_frame = pd.DataFrame(
        {f"compare_is10__f{index:05d}": values + index / 10_000.0 for index in range(feature_count)}
    )
    return pd.concat([identity, feature_frame], axis=1)


def _stage_run(root: Path, manifest: pd.DataFrame) -> tuple[Path, Path]:
    run_id = "hst-" + "1" * 20
    run_root = root / "data" / "outputs" / "hst" / run_id
    manifest_path = run_root / "manifests" / "aligned_comparator.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest.to_csv(manifest_path, index=False)
    audit_path = run_root / "manifests" / "internal_audit.csv"
    pd.DataFrame({"ok": [True]}).to_csv(audit_path, index=False)
    index_path = run_root / "manifests" / "manifest_index.json"
    _write_ascii_json(
        index_path,
        {
            "schema_version": 1,
            "scientific_configuration_fingerprint": "d" * 64,
            "eligibility_alignment_fingerprint": "e" * 64,
            "manifests": {
                "aligned_comparator": {
                    "path": "manifests/aligned_comparator.csv",
                    "sha256": stable_file_sha256(manifest_path),
                    "rows": len(manifest),
                }
            },
        },
    )
    outputs = [manifest_path, audit_path, index_path]
    receipt = {
        "receipt_type": "hst_stage",
        "run_id": run_id,
        "stage": "manifests",
        "status": "success",
        "configuration_hash": "1" * 64,
        "source_hash": "2" * 64,
        "dependency_lock_hash": "3" * 64,
        "fingerprint": "4" * 64,
        "accepted_hashes": {
            "data_contracts_freeze": "a" * 64,
            "environment_lock": "b" * 64,
            "pilot_freeze": "c" * 64,
        },
        "output_paths": [path.relative_to(run_root).as_posix() for path in outputs],
        "output_checksums": {
            path.relative_to(run_root).as_posix(): stable_file_sha256(path)
            for path in outputs
        },
    }
    receipt["record_hash"] = canonical_json_sha256(receipt)
    receipt_path = run_root / "runtime" / "stages" / "manifests.json"
    _write_ascii_json(receipt_path, receipt)
    return run_root, receipt_path


def _fake_recipe(
    _root: str | Path,
    *,
    random_state: int,
    accepted_environment_lock_sha256: str,
) -> dict[str, object]:
    recipe: dict[str, object] = {
        "recipe_version": 2,
        "random_state": random_state,
        "environment_lock_sha256": accepted_environment_lock_sha256,
        "model_names": [
            "lightgbm_smote_f80",
            "svc_rbf_f60",
            "catboost_smote_f80",
            "xgboost_smote_f80",
        ],
        "selected_candidate_model": "validation_selected_candidate",
        "executable_source_sha256": {"hst_comparators.py": "5" * 64},
        "dependency_lock_sha256": "6" * 64,
    }
    recipe["recipe_sha256"] = canonical_json_sha256(recipe)
    return recipe


def _candidate_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path | int]:
    import covid_rars.hst_comparator_approval as approval
    import covid_rars.hst_comparators as comparators

    root, pilot, environment_lock = _project_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(approval, "_build_compare_is10_executable_recipe", _fake_recipe)
    monkeypatch.setattr(comparators, "_build_compare_is10_executable_recipe", _fake_recipe)
    manifest = _manifest()
    run_root, receipt = _stage_run(root, manifest)
    features_path = root / "data" / "processed" / "features.csv"
    features_path.parent.mkdir(parents=True)
    _feature_table(manifest).to_csv(features_path, index=False)
    return {
        "project_root": root,
        "run_root": run_root,
        "manifests_receipt_path": receipt,
        "feature_table_path": features_path,
        "pilot_accepted_freezes_path": pilot,
        "environment_lock_path": environment_lock,
        "runtime_random_state": 42,
    }


def _promote_approval(root: Path, candidate: dict[str, object]) -> tuple[Path, str]:
    payload = dict(candidate["proposed_approval_record"])
    payload["approval_status"] = "approved"
    payload["approval_id"] = "manual-comparator-freeze"
    payload["approved_at_utc"] = "2026-08-02T12:00:00Z"
    payload.pop("approval_record_sha256", None)
    payload["approval_record_sha256"] = canonical_json_sha256(payload)
    path = root / CANONICAL_APPROVAL
    _write_ascii_json(path, payload)
    _git(root, "add", path.relative_to(root).as_posix())
    _git(root, "commit", "-q", "-m", "Manually approve comparator")
    os.chmod(path, stat.S_IREAD)
    return path, _git(root, "rev-parse", "HEAD")


def _promote_accepted(root: Path, candidate: dict[str, object]) -> tuple[Path, str]:
    path = root / CANONICAL_ACCEPTED
    if path.exists():
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    _write_ascii_json(path, dict(candidate["proposed_accepted_freezes"]))
    _git(root, "add", path.relative_to(root).as_posix())
    _git(root, "commit", "-q", "-m", "Manually accept comparator freezes")
    os.chmod(path, stat.S_IREAD)
    return path, stable_file_sha256(path)


def _build_approval_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object], dict[str, Path | int]]:
    from covid_rars.hst_comparator_approval import (
        build_comparator_approval_candidate,
    )

    inputs = _candidate_inputs(tmp_path, monkeypatch)
    output = Path(inputs["project_root"]) / "reports/hst/candidates/approval.json"
    candidate = build_comparator_approval_candidate(
        **inputs,
        manifest_name="aligned_comparator",
        output_path=output,
    )
    return output, candidate, inputs


def _build_accepted_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object], dict[str, Path | int]]:
    from covid_rars.hst_comparator_approval import (
        build_comparator_accepted_freezes_candidate,
    )

    _, approval_candidate, inputs = _build_approval_candidate(tmp_path, monkeypatch)
    root = Path(inputs["project_root"])
    approval_path, _ = _promote_approval(root, approval_candidate)
    output = root / "reports/hst/candidates/accepted.json"
    accepted_candidate = build_comparator_accepted_freezes_candidate(
        project_root=root,
        approval_record_path=approval_path,
        pilot_accepted_freezes_path=Path(inputs["pilot_accepted_freezes_path"]),
        environment_lock_path=Path(inputs["environment_lock_path"]),
        project_id="covid-rars",
        expected_remote_url="https://example.invalid/covid-rars.git",
        runtime_random_state=42,
        output_path=output,
    )
    return approval_path, output, accepted_candidate, inputs


def _write_generation(
    root: Path,
    approval_path: Path,
    accepted_path: Path,
    accepted_hash: str,
    audit_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    import covid_rars.hst_comparators as comparators

    monkeypatch.setattr(
        comparators, "_lightgbm_ranker", lambda random_state: _DeterministicRanker()
    )
    monkeypatch.setattr(
        comparators,
        "_default_estimator_factory",
        lambda model_name, random_state: _DeterministicEstimator(model_name),
    )
    manifest = _manifest()
    features = pd.read_csv(
        root / "data/processed/features.csv", low_memory=False
    )
    feature_columns = tuple(
        column for column in features.columns if column not in comparators._NON_FEATURE_COLUMNS
    )
    contract = comparators.build_compare_is10_feature_contract(
        features, ordered_feature_columns=feature_columns
    )
    comparators.run_aligned_compare_is10(
        features,
        manifest,
        feature_contract=contract,
        approval_record_path=approval_path,
        trusted_project_repository_root=root,
        accepted_freezes_path=accepted_path,
        expected_accepted_freezes_sha256=accepted_hash,
        selected_feature_k=800,
        ranker="lightgbm",
        selection_scope="per_modality_mean",
        random_state=42,
        optuna_trials=0,
        ensemble_top_k=5,
        selection_metric="auroc",
        run_id="hst-" + "1" * 20,
        test_mode=False,
        allow_sklearn_fallback=False,
        audit_dir=audit_root,
    )
    current_path = audit_root / "current.json"
    current = json.loads(current_path.read_text(encoding="ascii"))
    manifest_path = audit_root / "generations" / current["generation_id"] / "manifest.json"
    for path in [*manifest_path.parent.rglob("*"), current_path]:
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
    return manifest_path, current_path


def test_approval_candidate_binds_verified_stage_feature_recipe_and_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, candidate, inputs = _build_approval_candidate(tmp_path, monkeypatch)

    assert output.is_file()
    assert candidate["candidate_status"] == "requires_manual_review"
    assert candidate["canonical_target"] == CANONICAL_APPROVAL.as_posix()
    assert "candidate_record_sha256" not in candidate
    proposed = candidate["proposed_approval_record"]
    assert proposed["approval_status"] == "MANUAL_REVIEW_REQUIRED"
    assert proposed["approval_id"] == "SET_DURING_MANUAL_REVIEW"
    assert proposed["approved_at_utc"] == "SET_DURING_MANUAL_REVIEW"
    assert proposed["executable_recipe"]["random_state"] == 42
    assert proposed["feature_artifact_sha256"]
    assert proposed["manifest_sha256"] == _manifest()["manifest_sha256"].iloc[0]
    reviewed = proposed["reviewed_input_bindings"]
    assert reviewed["manifest_name"] == "aligned_comparator"
    assert reviewed["manifests_stage_receipt_sha256"] == stable_file_sha256(
        Path(inputs["manifests_receipt_path"])
    )
    assert reviewed["run_identity"]["run_id"] == Path(inputs["run_root"]).name
    assert reviewed["run_identity"]["configuration_hash"] == "1" * 64
    assert reviewed["run_identity"]["source_hash"] == "2" * 64
    assert reviewed["feature_table_header_sha256"]
    assert candidate["source_bindings"]["pilot_accepted_freezes_sha256"] == stable_file_sha256(
        Path(inputs["pilot_accepted_freezes_path"])
    )
    from covid_rars.hst_comparator_approval import build_comparator_approval_candidate

    assert "expected_feature_table_columns" not in inspect.signature(
        build_comparator_approval_candidate
    ).parameters


def test_approval_candidate_rejects_tampered_stage_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_rars.hst_comparator_approval import build_comparator_approval_candidate

    inputs = _candidate_inputs(tmp_path, monkeypatch)
    receipt_path = Path(inputs["manifests_receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    receipt["status"] = "failed"
    _write_ascii_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="receipt checksum"):
        build_comparator_approval_candidate(
            **inputs,
            manifest_name="aligned_comparator",
            output_path=Path(inputs["project_root"]) / "candidate.json",
        )


def test_approval_candidate_rejects_every_manifest_except_aligned_comparator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_rars.hst_comparator_approval import build_comparator_approval_candidate

    inputs = _candidate_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="restricted to manifest_name='aligned_comparator'"):
        build_comparator_approval_candidate(
            **inputs,
            manifest_name="internal",
            output_path=Path(inputs["project_root"]) / "candidate.json",
        )


def test_approval_candidate_rejects_feature_file_changed_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import covid_rars.hst_comparator_approval as approval

    inputs = _candidate_inputs(tmp_path, monkeypatch)
    feature_path = Path(inputs["feature_table_path"]).resolve()
    real_hash = stable_file_sha256
    feature_calls = 0

    def unstable_hash(path: Path) -> str:
        nonlocal feature_calls
        if Path(path).resolve() == feature_path:
            feature_calls += 1
            return ("1" if feature_calls == 1 else "2") * 64
        return real_hash(Path(path))

    monkeypatch.setattr(approval, "stable_file_sha256", unstable_hash)
    with pytest.raises(ValueError, match="changed while it was being validated"):
        approval.build_comparator_approval_candidate(
            **inputs,
            manifest_name="aligned_comparator",
            output_path=Path(inputs["project_root"]) / "candidate.json",
        )


def test_approval_candidate_rejects_incomplete_or_symlinked_feature_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_rars.hst_comparator_approval import build_comparator_approval_candidate

    inputs = _candidate_inputs(tmp_path, monkeypatch)
    features = pd.read_csv(Path(inputs["feature_table_path"]), low_memory=False)
    features.iloc[:, :-1].to_csv(Path(inputs["feature_table_path"]), index=False)
    with pytest.raises(ValueError, match="exactly 10147 columns"):
        build_comparator_approval_candidate(
            **inputs,
            manifest_name="aligned_comparator",
            output_path=Path(inputs["project_root"]) / "candidate.json",
        )

    real = Path(inputs["feature_table_path"])
    _feature_table(_manifest()).to_csv(real, index=False)
    first_line, separator, remainder = real.read_text(encoding="utf-8").partition("\n")
    headers = first_line.split(",")
    headers[-1] = headers[-2]
    real.write_text(",".join(headers) + separator + remainder, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate raw column names"):
        build_comparator_approval_candidate(
            **inputs,
            manifest_name="aligned_comparator",
            output_path=Path(inputs["project_root"]) / "candidate.json",
        )

    _feature_table(_manifest()).to_csv(real, index=False)
    outside = tmp_path / "outside-features.csv"
    shutil.copy2(real, outside)
    inputs["feature_table_path"] = outside
    with pytest.raises(ValueError, match="escapes the trusted root"):
        build_comparator_approval_candidate(
            **inputs,
            manifest_name="aligned_comparator",
            output_path=Path(inputs["project_root"]) / "candidate.json",
        )

    link = real.with_name("feature-link.csv")
    try:
        link.symlink_to(real)
    except OSError:
        return
    inputs["feature_table_path"] = link
    with pytest.raises(ValueError, match="symlink"):
        build_comparator_approval_candidate(
            **inputs,
            manifest_name="aligned_comparator",
            output_path=Path(inputs["project_root"]) / "candidate.json",
        )


def test_accepted_freezes_candidate_requires_committed_approval_and_exact_remote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_rars.hst_comparator_approval import (
        build_comparator_accepted_freezes_candidate,
    )

    approval_path, output, candidate, inputs = _build_accepted_candidate(
        tmp_path, monkeypatch
    )
    root = Path(inputs["project_root"])
    proposed = candidate["proposed_accepted_freezes"]
    assert output.is_file()
    assert candidate["candidate_status"] == "requires_manual_review"
    assert proposed["compare_is10_approval"]["relative_path"] == CANONICAL_APPROVAL.as_posix()
    assert proposed["project_identity"]["expected_remote_url"] == (
        "https://example.invalid/covid-rars.git"
    )
    assert proposed["accepted_generation_manifests"] == {}
    assert not (root / CANONICAL_ACCEPTED).exists()

    with pytest.raises(ValueError, match="trusted Git remote"):
        build_comparator_accepted_freezes_candidate(
            project_root=root,
            approval_record_path=approval_path,
            pilot_accepted_freezes_path=Path(inputs["pilot_accepted_freezes_path"]),
            environment_lock_path=Path(inputs["environment_lock_path"]),
            project_id="covid-rars",
            expected_remote_url="https://example.invalid/different.git",
            runtime_random_state=42,
            output_path=output,
        )

    os.chmod(approval_path, stat.S_IWRITE | stat.S_IREAD)
    approval_payload = json.loads(approval_path.read_text(encoding="ascii"))
    approval_payload["approval_id"] = "dirty"
    _write_ascii_json(approval_path, approval_payload)
    os.chmod(approval_path, stat.S_IREAD)
    with pytest.raises(ValueError, match="clean Git-tracked"):
        build_comparator_accepted_freezes_candidate(
            project_root=root,
            approval_record_path=approval_path,
            pilot_accepted_freezes_path=Path(inputs["pilot_accepted_freezes_path"]),
            environment_lock_path=Path(inputs["environment_lock_path"]),
            project_id="covid-rars",
            expected_remote_url="https://example.invalid/covid-rars.git",
            runtime_random_state=42,
            output_path=output,
        )

    os.chmod(approval_path, stat.S_IWRITE | stat.S_IREAD)
    _git(root, "checkout", "--", approval_path.relative_to(root).as_posix())
    approval_payload = json.loads(approval_path.read_text(encoding="ascii"))
    approval_payload["comparator_configuration"]["selected_feature_k"] = 799
    approval_payload.pop("approval_record_sha256")
    approval_payload["approval_record_sha256"] = canonical_json_sha256(approval_payload)
    _write_ascii_json(approval_path, approval_payload)
    _git(root, "add", approval_path.relative_to(root).as_posix())
    _git(root, "commit", "-q", "-m", "Invalid comparator configuration")
    os.chmod(approval_path, stat.S_IREAD)
    with pytest.raises(ValueError, match="frozen comparator configuration"):
        build_comparator_accepted_freezes_candidate(
            project_root=root,
            approval_record_path=approval_path,
            pilot_accepted_freezes_path=Path(inputs["pilot_accepted_freezes_path"]),
            environment_lock_path=Path(inputs["environment_lock_path"]),
            project_id="covid-rars",
            expected_remote_url="https://example.invalid/covid-rars.git",
            runtime_random_state=42,
            output_path=output,
        )


def test_candidate_builders_refuse_canonical_approved_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_rars.hst_comparator_approval import build_comparator_approval_candidate

    inputs = _candidate_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="canonical approved path"):
        build_comparator_approval_candidate(
            **inputs,
            manifest_name="aligned_comparator",
            output_path=Path(inputs["project_root"]) / CANONICAL_APPROVAL,
        )


def test_generation_candidate_authenticates_receipt_files_and_approval_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_rars.hst_comparator_approval import (
        build_comparator_generation_acceptance_candidate,
    )

    approval_path, _, accepted_candidate, inputs = _build_accepted_candidate(
        tmp_path, monkeypatch
    )
    root = Path(inputs["project_root"])
    approval_commit = str(
        accepted_candidate["proposed_accepted_freezes"]["compare_is10_approval"][
            "commit_sha"
        ]
    )
    accepted_path, accepted_hash = _promote_accepted(root, accepted_candidate)
    manifest_path, current_path = _write_generation(
        root,
        approval_path,
        accepted_path,
        accepted_hash,
        root / "reports" / "hst" / "comparator_audit",
        monkeypatch,
    )
    output = root / "reports/hst/candidates/generation.json"
    candidate = build_comparator_generation_acceptance_candidate(
        project_root=root,
        approval_record_path=approval_path,
        accepted_freezes_path=accepted_path,
        expected_accepted_freezes_sha256=accepted_hash,
        generation_manifest_path=manifest_path,
        current_receipt_path=current_path,
        runtime_random_state=42,
        output_path=output,
    )

    generation_hash = stable_file_sha256(manifest_path)
    generation_id = manifest_path.parent.name
    assert candidate["candidate_status"] == "requires_manual_review"
    assert candidate["authenticated_generation"] == {
        "generation_id": generation_id,
        "generation_manifest_sha256": generation_hash,
    }
    assert candidate["proposed_accepted_freezes"]["accepted_generation_manifests"] == {
        generation_id: generation_hash
    }
    assert stable_file_sha256(accepted_path) == accepted_hash
    selected = json.loads(manifest_path.read_text(encoding="ascii"))["selected_candidate_model"]
    assert selected == "validation_selected_candidate"

    updated_path, updated_hash = _promote_accepted(root, candidate)
    manifest_mtime = manifest_path.stat().st_mtime_ns
    second = build_comparator_generation_acceptance_candidate(
        project_root=root,
        approval_record_path=approval_path,
        accepted_freezes_path=updated_path,
        expected_accepted_freezes_sha256=updated_hash,
        generation_manifest_path=manifest_path,
        current_receipt_path=current_path,
        runtime_random_state=42,
        output_path=root / "reports/hst/candidates/generation-second.json",
    )
    assert second["authenticated_generation"] == candidate["authenticated_generation"]
    assert manifest_path.stat().st_mtime_ns == manifest_mtime


def test_generation_acceptance_review_never_deserializes_unaccepted_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pickle
    from covid_rars.hst_comparator_approval import (
        build_comparator_generation_acceptance_candidate,
    )

    approval_path, _, accepted_candidate, inputs = _build_accepted_candidate(
        tmp_path, monkeypatch
    )
    root = Path(inputs["project_root"])
    accepted_path, accepted_hash = _promote_accepted(root, accepted_candidate)
    manifest_path, current_path = _write_generation(
        root,
        approval_path,
        accepted_path,
        accepted_hash,
        root / "reports" / "hst" / "comparator_audit",
        monkeypatch,
    )

    def forbidden_loads(_payload: bytes) -> object:
        raise AssertionError("unaccepted pickle was executed")

    monkeypatch.setattr(pickle, "loads", forbidden_loads)
    candidate = build_comparator_generation_acceptance_candidate(
        project_root=root,
        approval_record_path=approval_path,
        accepted_freezes_path=accepted_path,
        expected_accepted_freezes_sha256=accepted_hash,
        generation_manifest_path=manifest_path,
        current_receipt_path=current_path,
        runtime_random_state=42,
        output_path=root / "reports/hst/candidates/no-pickle.json",
    )

    assert candidate["candidate_status"] == "requires_manual_review"


def test_generation_candidate_fails_closed_on_artifact_tamper_and_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from covid_rars.hst_comparator_approval import (
        build_comparator_generation_acceptance_candidate,
    )

    approval_path, _, accepted_candidate, inputs = _build_accepted_candidate(
        tmp_path, monkeypatch
    )
    root = Path(inputs["project_root"])
    approval_commit = str(
        accepted_candidate["proposed_accepted_freezes"]["compare_is10_approval"][
            "commit_sha"
        ]
    )
    accepted_path, accepted_hash = _promote_accepted(root, accepted_candidate)
    manifest_path, current_path = _write_generation(
        root,
        approval_path,
        accepted_path,
        accepted_hash,
        root / "reports" / "hst" / "comparator_audit",
        monkeypatch,
    )
    original_generation = json.loads(manifest_path.read_text(encoding="ascii"))
    generation_with_extra = dict(original_generation)
    generation_with_extra["unexpected"] = "not-permitted"
    os.chmod(manifest_path, stat.S_IWRITE | stat.S_IREAD)
    _write_ascii_json(manifest_path, generation_with_extra)
    os.chmod(current_path, stat.S_IWRITE | stat.S_IREAD)
    receipt = json.loads(current_path.read_text(encoding="ascii"))
    receipt["generation_manifest_sha256"] = stable_file_sha256(manifest_path)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _write_ascii_json(current_path, receipt)
    os.chmod(manifest_path, stat.S_IREAD)
    os.chmod(current_path, stat.S_IREAD)
    with pytest.raises(ValueError, match="generation manifest schema"):
        build_comparator_generation_acceptance_candidate(
            project_root=root,
            approval_record_path=approval_path,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=accepted_hash,
            generation_manifest_path=manifest_path,
            current_receipt_path=current_path,
            runtime_random_state=42,
            output_path=root / "candidate.json",
        )

    os.chmod(manifest_path, stat.S_IWRITE | stat.S_IREAD)
    _write_ascii_json(manifest_path, original_generation)
    os.chmod(current_path, stat.S_IWRITE | stat.S_IREAD)
    receipt["generation_manifest_sha256"] = stable_file_sha256(manifest_path)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _write_ascii_json(current_path, receipt)
    artifact = manifest_path.parent / "comparator_metrics.csv"
    original_artifact = artifact.read_bytes()
    os.chmod(manifest_path, stat.S_IREAD)
    os.chmod(current_path, stat.S_IREAD)
    os.chmod(artifact, stat.S_IWRITE | stat.S_IREAD)
    artifact.write_bytes(b"tampered\n")
    with pytest.raises(
        ValueError,
        match="read-only before acceptance|checksum/size mismatch",
    ):
        build_comparator_generation_acceptance_candidate(
            project_root=root,
            approval_record_path=approval_path,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=accepted_hash,
            generation_manifest_path=manifest_path,
            current_receipt_path=current_path,
            runtime_random_state=42,
            output_path=root / "candidate.json",
        )

    os.chmod(artifact, stat.S_IWRITE | stat.S_IREAD)
    artifact.write_bytes(original_artifact)
    os.chmod(artifact, stat.S_IREAD)
    generation = json.loads(manifest_path.read_text(encoding="ascii"))
    generation["files"]["../escape.csv"] = {
        "sha256": "0" * 64,
        "size_bytes": 0,
    }
    os.chmod(manifest_path, stat.S_IWRITE | stat.S_IREAD)
    _write_ascii_json(manifest_path, generation)
    os.chmod(current_path, stat.S_IWRITE | stat.S_IREAD)
    receipt = json.loads(current_path.read_text(encoding="ascii"))
    receipt["generation_manifest_sha256"] = stable_file_sha256(manifest_path)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    _write_ascii_json(current_path, receipt)
    os.chmod(manifest_path, stat.S_IREAD)
    os.chmod(current_path, stat.S_IREAD)
    with pytest.raises(ValueError, match="escapes"):
        build_comparator_generation_acceptance_candidate(
            project_root=root,
            approval_record_path=approval_path,
            accepted_freezes_path=accepted_path,
            expected_accepted_freezes_sha256=accepted_hash,
            generation_manifest_path=manifest_path,
            current_receipt_path=current_path,
            runtime_random_state=42,
            output_path=root / "candidate.json",
        )


@pytest.mark.parametrize(
    "script_name",
    (
        "76_prepare_hst_comparator_approval.py",
        "77_prepare_hst_comparator_generation_acceptance.py",
    ),
)
def test_candidate_cli_help_requires_manual_review_and_explicit_paths(
    script_name: str,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / script_name
    help_result = subprocess.run(
        (sys.executable, str(script), "--help"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "manual review" in help_result.stdout.lower()
    assert "--output" in help_result.stdout

    missing_result = subprocess.run(
        (sys.executable, str(script)),
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_result.returncode == 2
    assert "required" in missing_result.stderr.lower()
