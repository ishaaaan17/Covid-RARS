from __future__ import annotations

import json
import gc
import inspect
import math
import os
import platform
import sys
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .hst_checkpoint import (
    hst_model_source_sha256,
    load_verified_hst_model,
    verify_file,
    verify_hst_source,
)
from . import hst_fusion as _fusion_contract
from . import hst_protocols as _protocol_contract
from .hst_comparators import (
    ENSEMBLE_MODEL_NAME,
    SELECTED_CANDIDATE_MODEL_NAME,
    _NON_FEATURE_COLUMNS,
    assert_confirmatory_comparator_table,
    build_compare_is10_feature_contract,
    compare_is10_feature_artifact_sha256,
    load_frozen_compare_is10_approval,
    load_verified_compare_is10_bundle,
    run_aligned_compare_is10,
)
from .hst_data_contracts import (
    audit_coughvid_labels,
    build_audited_coughvid_index,
    freeze_data_contracts,
    qualify_identifiers,
)
from .hst_parallel import benchmark_preprocess_workers, parallel_build_spectrograms
from .hst_evidence import build_hst_evidence_manifest
from .hst_fusion import AuthenticatedFusionBinding, run_hst_fusion_bank
from .hst_protocols import (
    audit_hst_manifest,
    build_common_late_test_manifests,
    build_external_hst_manifest,
    build_hst_task2_like_cough_manifest,
    build_protocol_matched_hst_manifest,
    build_reverse_temporal_hst_manifest,
    build_split_policy_contrast_manifests,
    intersect_representation_eligibility,
    scientific_configuration_fingerprint,
)
from .hst_reliability import (
    HSTPipeline,
    HSTPipelineConfig,
    StageHandler,
    audio_input_manifest_records,
    capture_live_pip_freeze,
)
from .hst_resource_pilot import (
    conservative_balanced_optimizer_updates_per_epoch,
    project_full_training_runtime,
    run_base_resource_pilot_trials,
    runtime_projection_policy_payload,
)
from .hst_workloads import (
    CAPACITY_INTERNAL_FUSION_PROFILE,
    FULL_RELIABILITY_PROFILE,
    workload_profile_from_scientific_config,
)
from .hst_runtime import atomic_write_json, canonical_json_sha256, stable_file_sha256
from .hst_spectrograms import HSTSpectrogramConfig
from .hst_training import (
    HSTTrainingConfig,
    _load_verified_checkpoint_with_path,
    _model_architecture_sha256,
    _model_state_sha256,
    aggregate_recording_predictions,
    load_verified_cached_image,
    make_hst_dataloaders,
    predict_hst_split,
    train_hst_fold,
    verify_executable_allowlist,
    verify_initial_model_load_audit,
)
from .metrics import binary_metric_bundle, labels_to_binary


_TRACK_A_SEEDS = (1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002)
_ALIGNED_COMPARATOR_COMPONENTS = (
    "internal",
    "task2_like_cough",
    "calendar_mixed",
    "early_to_late",
    "common_late_mixed",
    "common_late_chronological",
    "reverse_temporal",
    "external",
)
_GRADCAM_OUTCOMES = ("TP", "TN", "FP", "FN")
_ENGINEERING_OBJECTIVE_REFERENCES = (
    ("cough", 0.868),
    ("breath", 0.842),
    ("speech", 0.891),
    ("cough_speech_fusion", 0.897),
)


class ManualComparatorGenerationAcceptanceRequired(RuntimeError):
    """A durable comparator generation exists but has not been manually accepted."""


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _section(config: HSTPipelineConfig, name: str) -> Mapping[str, object]:
    value = config.scientific_config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Scientific configuration section {name!r} must be an object")
    return value


def _project_path(config: HSTPipelineConfig, supplied: object) -> Path:
    path = (config.workspace_root / str(supplied)).resolve()
    try:
        path.relative_to(config.workspace_root)
    except ValueError as exc:
        raise ValueError(f"Configured project path escapes project root: {supplied}") from exc
    return path


def _source_path(config: HSTPipelineConfig) -> Path:
    supplied = _section(config, "source").get("path", "HST")
    path = (config.workspace_root / str(supplied)).resolve()
    repository_root = config.workspace_root.resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"Configured HST source escapes the repository: {supplied}") from exc
    return path


def _checkpoint_path(config: HSTPipelineConfig, name: str) -> Path:
    paths = _section(config, "paths")
    root = _project_path(
        config,
        paths.get("checkpoint_directory", ".cache/hst/checkpoints"),
    )
    specification = _section(config, "checkpoints").get(name)
    if not isinstance(specification, Mapping):
        raise ValueError(f"Checkpoint specification {name!r} is missing")
    return (root / str(specification.get("filename", ""))).resolve()


def _scientific_handler(stage_name: str) -> Callable[[StageHandler], StageHandler]:
    def decorate(handler: StageHandler) -> StageHandler:
        setattr(handler, "scientific_stage_handler", True)
        setattr(handler, "scientific_stage_name", stage_name)
        return handler

    return decorate


def _normalize_binary_label(values: pd.Series) -> pd.Series:
    aliases = {
        0: "negative",
        1: "positive",
        0.0: "negative",
        1.0: "positive",
        "0": "negative",
        "1": "positive",
        "negative": "negative",
        "positive": "positive",
    }
    return values.map(aliases)


def _sensitivity_execution_registry(config: HSTPipelineConfig) -> pd.DataFrame:
    datasets = _section(config, "datasets")
    coughvid = datasets.get("coughvid", {})
    if not isinstance(coughvid, Mapping):
        raise ValueError("COUGHVID dataset configuration must be an object")
    preprocessing = _section(config, "preprocessing")
    specifications = (
        (
            "coughvid_raw_status_label",
            coughvid.get("raw_status_sensitivity"),
            "relabel_frozen_external_predictions",
        ),
        (
            "coughvid_event_quality",
            coughvid.get("event_quality_sensitivity"),
            "deferred_missing_checksum_pinned_algorithm",
        ),
        (
            "released_code_representation",
            preprocessing.get("released_code_sensitivity"),
            "deferred_optional_extension",
        ),
    )
    rows: list[dict[str, object]] = []
    for sensitivity_id, supplied, expected_execution in specifications:
        if supplied is None:
            rows.append(
                {
                    "sensitivity_id": sensitivity_id,
                    "execution": "not_declared",
                    "implemented_in_primary_e2e": False,
                    "primary_blocking": False,
                    "selection_use": False,
                    "reason": "not declared in the scientific configuration",
                }
            )
            continue
        if not isinstance(supplied, Mapping):
            raise ValueError(
                f"Sensitivity {sensitivity_id!r} must be a structured execution contract"
            )
        execution = str(supplied.get("execution", "")).strip()
        if execution != expected_execution:
            raise ValueError(
                f"Sensitivity {sensitivity_id!r} execution must be {expected_execution!r}"
            )
        selection_use = supplied.get("selection_use", False)
        primary_blocking = supplied.get("primary_blocking", False)
        if not isinstance(selection_use, bool) or not isinstance(primary_blocking, bool):
            raise ValueError("Sensitivity selection/blocking flags must be boolean")
        rows.append(
            {
                "sensitivity_id": sensitivity_id,
                "execution": execution,
                "implemented_in_primary_e2e": execution
                == "relabel_frozen_external_predictions",
                "primary_blocking": primary_blocking,
                "selection_use": selection_use,
                "reason": str(supplied.get("reason", "")),
                "label_column": str(supplied.get("label_column", "")),
                "label_provenance": str(supplied.get("label_provenance", "")),
                "representation": str(supplied.get("representation", "")),
            }
        )
    registry = pd.DataFrame(rows)
    if registry["selection_use"].astype(bool).any():
        raise ValueError("Sensitivity analyses cannot influence model selection")
    if registry["primary_blocking"].astype(bool).any():
        raise ValueError("Deferred sensitivities cannot block the primary confirmatory run")
    return registry


def _clinical_utility_scope(
    sources: Mapping[str, object],
    estimands: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, str], pd.DataFrame]:
    if set(sources) != set(estimands):
        raise ValueError("Clinical-utility sources and estimands must match exactly")
    scoped_sources: dict[str, object] = {}
    scoped_estimands: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for series in sorted(sources):
        external_pseudo_label = series.startswith("external_hst")
        included = not external_pseudo_label
        if included:
            scoped_sources[series] = sources[series]
            scoped_estimands[series] = estimands[series]
        rows.append(
            {
                "series": series,
                "estimand_id": estimands[series],
                "included_in_clinical_utility_outputs": included,
                "reason": (
                    "semi_supervised_external_pseudo_label"
                    if external_pseudo_label
                    else "source_label_endpoint"
                ),
            }
        )
    return scoped_sources, scoped_estimands, pd.DataFrame(rows)


def _assert_reporting_config_bound(reporting: Mapping[str, object]) -> None:
    from .hst_publication import PRIMARY_ESTIMAND_ID
    from .hst_reporting import REPORTING_CONTRACT

    expected = {
        **REPORTING_CONTRACT,
        "primary_estimand_id": PRIMARY_ESTIMAND_ID,
    }
    observed = {key: reporting.get(key) for key in expected}
    if canonical_json_sha256(observed) != canonical_json_sha256(expected):
        raise ValueError(
            "Fingerprinted reporting configuration disagrees with executed statistics"
        )


def _coswara_contract(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"participant_id", "recording_id", "modality", "audio_path", "label_binary"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Coswara metadata is missing contract columns: {missing}")
    frame = frame.copy()
    frame["dataset"] = "coswara"
    frame["dataset_release_id"] = frame.get(
        "dataset_release_id", "coswara_project_contract"
    )
    frame["label_source"] = frame.get("label_source", "label_binary")
    frame["label_provenance"] = frame.get(
        "label_provenance", "project_supervised_contract"
    )
    frame["source_manifest_sha256"] = stable_file_sha256(path)
    normalized = _normalize_binary_label(frame["label_binary"])
    frame["label_binary"] = normalized
    def resolve_audio(value: object) -> str:
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        return candidate.resolve().as_posix()

    frame["audio_path"] = frame["audio_path"].map(resolve_audio)
    exists = frame["audio_path"].map(lambda value: Path(value).is_file() or len(str(value or "").strip()) > 0)
    frame["contract_eligible"] = normalized.isin(["negative", "positive"]) & exists
    frame["contract_exclusion_reason"] = np.select(
        [~normalized.isin(["negative", "positive"]), ~exists],
        ["unknown_label", "missing_audio"],
        default="",
    )
    qualified_input = frame.drop(columns=["participant_key", "recording_key"], errors="ignore")
    frame = qualify_identifiers(qualified_input)
    audit = (
        frame.groupby(
            ["contract_eligible", "contract_exclusion_reason", "label_binary"],
            dropna=False,
        )
        .size()
        .rename("row_count")
        .reset_index()
    )
    if not frame.loc[frame["contract_eligible"], "label_binary"].isin(
        ["negative", "positive"]
    ).all():
        raise ValueError("Coswara supervised eligibility contains a non-binary label")
    return frame, audit


def _single_contract_value(frame: pd.DataFrame, column: str) -> object:
    if column not in frame:
        raise ValueError(f"COUGHVID contract is missing provenance column {column!r}")
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"COUGHVID provenance column {column!r} is not invariant")
    return values.iloc[0]


def _coughvid_source_provenance(
    *,
    coughvid: pd.DataFrame,
    coughvid_config: Mapping[str, object],
    source_manifest_sha256: str,
) -> pd.DataFrame:
    required_config = {
        "release_id",
        "source_release_reference",
        "metadata_input_level",
        "raw_release_membership_reconstructed",
        "identity_source",
        "subject_linkage_available",
        "primary_label_column",
        "primary_label_provenance",
    }
    missing = sorted(required_config - set(coughvid_config))
    if missing:
        raise ValueError(f"COUGHVID source provenance is missing configuration: {missing}")
    metadata_input_level = str(coughvid_config["metadata_input_level"])
    analysis_unit_type = str(_single_contract_value(coughvid, "analysis_unit_type"))
    indexed_metadata_level = str(
        _single_contract_value(coughvid, "metadata_source_level")
    )
    indexed_subject_linkage = bool(
        _single_contract_value(coughvid, "subject_linkage_available")
    )
    declared_subject_linkage = coughvid_config["subject_linkage_available"]
    reconstructed = coughvid_config["raw_release_membership_reconstructed"]
    if not isinstance(declared_subject_linkage, bool) or not isinstance(
        reconstructed, bool
    ):
        raise ValueError("COUGHVID provenance flags must be boolean")
    if metadata_input_level != indexed_metadata_level:
        raise ValueError("COUGHVID metadata source level disagrees with the audited index")
    if declared_subject_linkage != indexed_subject_linkage:
        raise ValueError("COUGHVID subject-linkage declaration disagrees with the audited index")
    if str(coughvid_config["identity_source"]) != analysis_unit_type:
        raise ValueError("COUGHVID identity-source declaration disagrees with the analysis unit")
    provenance = {
        "dataset": "coughvid",
        "cohort_release_id": str(coughvid_config["release_id"]),
        "source_release_reference": str(
            coughvid_config["source_release_reference"]
        ),
        "metadata_input_level": metadata_input_level,
        "raw_release_membership_reconstructed": reconstructed,
        "identity_source": str(coughvid_config["identity_source"]),
        "analysis_unit_type": analysis_unit_type,
        "subject_linkage_available": declared_subject_linkage,
        "primary_label_column": str(coughvid_config["primary_label_column"]),
        "primary_label_provenance": str(
            coughvid_config["primary_label_provenance"]
        ),
        "source_manifest_sha256": source_manifest_sha256,
    }
    for column in ("cohort_source_sha256", "label_metadata_source_sha256"):
        if column in coughvid:
            provenance[column] = str(_single_contract_value(coughvid, column))
    return pd.DataFrame([provenance])


def _write_environment_audit(pipeline: HSTPipeline, path: Path) -> str:
    frozen, freeze_hash = capture_live_pip_freeze()
    torch_audit: dict[str, object] = {"available": False}
    try:
        import torch

        torch_audit = {
            "available": True,
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": str(torch.version.cuda),
            "device_name": (
                str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
            ),
        }
    except ImportError:
        if pipeline.config.device == "cuda":
            raise RuntimeError("CUDA mode requires PyTorch")
    if pipeline.config.device == "cuda" and not bool(torch_audit.get("cuda_available")):
        raise RuntimeError("CUDA mode requires an available CUDA device")
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "run_id": pipeline.run_id,
            "python": sys.version,
            "platform": platform.platform(),
            "dependency_lock_sha256": stable_file_sha256(
                pipeline.config.dependency_lock_path
            ),
            "pip_freeze_sha256": freeze_hash,
            "pip_freeze": frozen,
            "torch": torch_audit,
        },
    )
    return freeze_hash


@_scientific_handler("preflight")
def _preflight(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    config = pipeline.config
    source = _section(config, "source")
    hst_root = _source_path(config)
    commit = verify_hst_source(hst_root, str(source.get("commit", "")))
    expected_source_hash = str(source.get("model_source_sha256", ""))
    if hst_model_source_sha256(hst_root, commit) != expected_source_hash:
        raise ValueError("Official HST model source checksum does not match the freeze")
    required_inputs: list[Path] = []
    paths = _section(config, "paths")
    required_inputs.append(_project_path(config, paths.get("coswara_metadata", "")))
    required_inputs.append(_project_path(config, paths.get("coughvid_metadata", "")))
    if config.mode == "full" and "aligned_comparator" in pipeline.STAGES:
        required_inputs.append(_project_path(config, paths.get("compare_is10_features", "")))
    for path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_rows: list[dict[str, object]] = []
    for name, specification in _section(config, "checkpoints").items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"Checkpoint {name!r} specification must be an object")
        path = _checkpoint_path(config, str(name))
        verify_file(
            path,
            expected_size=int(specification.get("size_bytes", -1)),
            expected_sha256=str(specification.get("sha256", "")),
        )
        checkpoint_rows.append(
            {
                "checkpoint": name,
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": stable_file_sha256(path),
            }
        )
    audit_root = pipeline.run_root / "audits"
    environment_path = audit_root / "environment.json"
    source_path = audit_root / "preflight.json"
    environment_hash = _write_environment_audit(pipeline, environment_path)
    if config.mode == "full" and (
        config.accepted_hashes.get("environment_lock") != environment_hash
    ):
        raise ValueError(
            "Current Python environment does not match the manually accepted lock hash"
        )
    atomic_write_json(
        source_path,
        {
            "schema_version": 1,
            "run_id": pipeline.run_id,
            "hst_commit": commit,
            "hst_model_source_sha256": expected_source_hash,
            "checkpoints": checkpoint_rows,
            "inputs": [
                {
                    "path": path.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": stable_file_sha256(path),
                }
                for path in required_inputs
            ],
        },
    )
    return {
        "output_paths": [source_path, environment_path],
        "row_counts": {"checkpoints": len(checkpoint_rows), "inputs": len(required_inputs)},
        "metadata": {"hst_commit": commit},
    }


@_scientific_handler("data_contracts")
def _data_contracts(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    config = pipeline.config
    paths = _section(config, "paths")
    coswara_path = _project_path(config, paths.get("coswara_metadata", ""))
    coswara, coswara_audit = _coswara_contract(coswara_path)
    contract_root = pipeline.run_root / "contracts"
    coswara_output = contract_root / "coswara_index.csv"
    coswara_audit_output = contract_root / "coswara_label_audio_audit.csv"
    for frame, path in (
        (coswara, coswara_output),
        (coswara_audit, coswara_audit_output),
    ):
        _atomic_csv(frame, path)
    source_paths: list[Path] = [coswara_path]
    label_audits: list[Path] = [coswara_audit_output]
    source_hashes = {"coswara": stable_file_sha256(coswara_path)}
    outputs = [coswara_output, coswara_audit_output]
    row_counts = {
        "coswara": len(coswara),
        "coswara_eligible": int(coswara["contract_eligible"].sum()),
    }
    release_id = "coswara_project_contract"
    label_columns = "coswara:label_binary"
    coughvid_path = _project_path(config, paths.get("coughvid_metadata", ""))
    cohort_setting = paths.get("coughvid_cohort_metadata")
    raw_metadata_setting = paths.get("coughvid_raw_metadata")
    if (cohort_setting is None) != (raw_metadata_setting is None):
        raise ValueError("Both COUGHVID upstream metadata paths must be configured together")
    coughvid_upstream_paths: list[Path] = []
    if cohort_setting is not None and raw_metadata_setting is not None:
        coughvid_upstream_paths = [
            _project_path(config, cohort_setting),
            _project_path(config, raw_metadata_setting),
        ]
    coughvid_config = _section(config, "datasets").get("coughvid")
    if not isinstance(coughvid_config, Mapping):
        raise ValueError("COUGHVID dataset configuration is missing")
    coughvid_source_sha256 = stable_file_sha256(coughvid_path)
    coughvid = build_audited_coughvid_index(
        coughvid_path,
        label_column=str(coughvid_config.get("primary_label_column", "")),
        dataset_release_id=str(coughvid_config.get("release_id", "")),
        source_manifest_sha256=coughvid_source_sha256,
        require_audio=True,
        metadata_source_level=str(coughvid_config.get("metadata_input_level", "")),
    )
    expected_upstream_hashes: dict[str, str] = {}
    valid_upstream_paths = [p for p in coughvid_upstream_paths if p.is_file()]
    if len(valid_upstream_paths) == len(coughvid_upstream_paths) and coughvid_upstream_paths:
        expected_upstream_hashes = {
            "cohort_source_sha256": stable_file_sha256(
                coughvid_upstream_paths[0]
            ),
            "label_metadata_source_sha256": stable_file_sha256(
                coughvid_upstream_paths[1]
            ),
        }
    for column, expected_hash in expected_upstream_hashes.items():
        if column in coughvid.columns and str(_single_contract_value(coughvid, column)) != expected_hash:
            raise ValueError(
                f"COUGHVID bound metadata does not match upstream source {column!r}"
            )
    coughvid, coughvid_audit = audit_coughvid_labels(coughvid)
    coughvid["label_provenance"] = str(
        coughvid_config.get("primary_label_provenance", "")
    )
    coughvid["contract_eligible"] = coughvid["label_binary"].isin(
        ["negative", "positive"]
    )
    coughvid["contract_exclusion_reason"] = np.where(
        coughvid["contract_eligible"], "", "unknown_label"
    )
    coughvid_output = contract_root / "coughvid_index.csv"
    coughvid_audit_output = contract_root / "coughvid_label_audit.csv"
    coughvid_provenance_output = contract_root / "coughvid_source_provenance.csv"
    coughvid_provenance = _coughvid_source_provenance(
        coughvid=coughvid,
        coughvid_config=coughvid_config,
        source_manifest_sha256=coughvid_source_sha256,
    )
    _atomic_csv(coughvid, coughvid_output)
    _atomic_csv(coughvid_audit, coughvid_audit_output)
    _atomic_csv(coughvid_provenance, coughvid_provenance_output)
    source_paths.extend([coughvid_path, *valid_upstream_paths])
    label_audits.append(coughvid_audit_output)
    source_hashes["coughvid"] = coughvid_source_sha256
    if expected_upstream_hashes:
        source_hashes["coughvid_cohort"] = expected_upstream_hashes[
            "cohort_source_sha256"
        ]
        source_hashes["coughvid_raw_metadata"] = expected_upstream_hashes[
            "label_metadata_source_sha256"
        ]
    outputs.extend(
        [coughvid_output, coughvid_audit_output, coughvid_provenance_output]
    )
    row_counts.update(
        {
            "coughvid": len(coughvid),
            "coughvid_eligible": int(coughvid["contract_eligible"].sum()),
        }
    )
    release_id += "+" + str(coughvid_config.get("release_id", ""))
    label_columns += ";coughvid:" + str(
        coughvid_config.get("primary_label_column", "")
    )

    raw_status_config = coughvid_config.get("raw_status_sensitivity")
    if raw_status_config is not None:
        if not isinstance(raw_status_config, Mapping):
            raise ValueError("COUGHVID raw-status sensitivity must be an object")
        if str(raw_status_config.get("execution", "")) != (
            "relabel_frozen_external_predictions"
        ):
            raise ValueError("COUGHVID raw-status sensitivity execution is not frozen")
        if raw_status_config.get("selection_use", False) is not False:
            raise ValueError("Raw COUGHVID status cannot be used for model selection")
        raw_status = build_audited_coughvid_index(
            coughvid_path,
            label_column=str(raw_status_config.get("label_column", "")),
            dataset_release_id=str(coughvid_config.get("release_id", "")),
            source_manifest_sha256=stable_file_sha256(coughvid_path),
            require_audio=True,
            metadata_source_level=str(
                coughvid_config.get("metadata_input_level", "")
            ),
        )
        raw_status, raw_status_audit = audit_coughvid_labels(
            raw_status,
            prior=coughvid,
        )
        if set(raw_status["recording_key"].astype(str)) != set(
            coughvid["recording_key"].astype(str)
        ):
            raise ValueError(
                "Primary and raw-status COUGHVID contracts must cover identical audio"
            )
        raw_status["label_provenance"] = str(
            raw_status_config.get("label_provenance", "")
        )
        raw_status["contract_eligible"] = raw_status["label_binary"].isin(
            ["negative", "positive"]
        )
        raw_status["contract_exclusion_reason"] = np.where(
            raw_status["contract_eligible"], "", "unknown_label"
        )
        raw_status_output = contract_root / "coughvid_raw_status_sensitivity.csv"
        raw_status_audit_output = contract_root / "coughvid_raw_status_label_audit.csv"
        _atomic_csv(raw_status, raw_status_output)
        _atomic_csv(raw_status_audit, raw_status_audit_output)
        label_audits.append(raw_status_audit_output)
        outputs.extend([raw_status_output, raw_status_audit_output])
        row_counts.update(
            {
                "coughvid_raw_status": len(raw_status),
                "coughvid_raw_status_eligible": int(
                    raw_status["contract_eligible"].sum()
                ),
            }
        )
        label_columns += ";coughvid:status(raw-label-sensitivity)"

    sensitivity_registry = _sensitivity_execution_registry(config)
    sensitivity_registry_path = contract_root / "sensitivity_execution_registry.csv"
    _atomic_csv(sensitivity_registry, sensitivity_registry_path)
    outputs.append(sensitivity_registry_path)
    row_counts["sensitivity_contracts"] = len(sensitivity_registry)
    combined_source_hash = canonical_json_sha256(source_hashes)
    freeze_path = contract_root / "data_contracts_freeze.json"
    freeze_hash = freeze_data_contracts(
        source_root=config.workspace_root,
        audit_root=contract_root,
        source_paths=tuple(source_paths),
        label_audits=tuple(label_audits),
        contract_metadata={
            "dataset_release_id": release_id,
            "label_column": label_columns,
            "label_normalization_version": "hst-contract-v1",
            "source_manifest_sha256": combined_source_hash,
            "eligibility_policy_version": "audio-exists_binary-label-v1",
            "coughvid_metadata_input_level": str(
                coughvid_config["metadata_input_level"]
            ),
            "coughvid_raw_release_membership_reconstructed": coughvid_config[
                "raw_release_membership_reconstructed"
            ],
            "coughvid_analysis_unit_type": str(
                coughvid_config["identity_source"]
            ),
            "coughvid_subject_linkage_available": coughvid_config[
                "subject_linkage_available"
            ],
        },
        output_path=freeze_path,
    )
    if config.mode == "full" and (
        config.accepted_hashes.get("data_contracts_freeze") != freeze_hash
    ):
        raise ValueError(
            "Generated data-contract freeze does not match the manually accepted hash"
        )
    outputs.append(freeze_path)
    return {
        "output_paths": outputs,
        "row_counts": row_counts,
        "metadata": {"data_contracts_freeze_hash": freeze_hash},
    }


@_scientific_handler("checkpoint")
def _checkpoint(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    config = pipeline.config
    hst_root = _source_path(config)
    rows: list[dict[str, object]] = []
    for model_name, checkpoint_name in (
        ("hst_small", "hst_small_imagenet"),
        ("hst_base", "hst_base_imagenet"),
    ):
        model, audit = load_verified_hst_model(
            model_name=model_name,
            checkpoint_path=_checkpoint_path(config, checkpoint_name),
            hst_repo=hst_root,
            seed=52,
        )
        del model
        rows.append({"model_name": model_name, **audit})
    output = pipeline.run_root / "audits" / "checkpoint_load.csv"
    _atomic_csv(pd.DataFrame(rows), output)
    return {
        "output_paths": [output],
        "row_counts": {"models": len(rows)},
        "metadata": {"strict_load_verified": True, "head_classes": 2},
    }


def _primary_contract_metadata(pipeline: HSTPipeline) -> pd.DataFrame:
    contract_root = pipeline.run_root / "contracts"
    frames = []
    for path in sorted(contract_root.glob("*_index.csv")):
        frame = pd.read_csv(path, low_memory=False)
        eligible = frame["contract_eligible"].map(
            {True: True, False: False, "True": True, "False": False}
        )
        if eligible.isna().any():
            raise ValueError(f"Contract eligibility is not boolean: {path}")
        frames.append(frame.loc[eligible.astype(bool)].copy())
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined["recording_key"].duplicated().any():
        raise ValueError("The combined HST data contract has duplicate recording keys")
    return combined


def _metadata_for_spectrogram_stage(
    metadata: pd.DataFrame,
    *,
    mode: str,
    workload_profile: str = FULL_RELIABILITY_PROFILE,
) -> pd.DataFrame:
    if (
        mode == "full"
        and workload_profile == CAPACITY_INTERNAL_FUSION_PROFILE
    ):
        required = {"dataset", "modality"}
        missing = sorted(required - set(metadata.columns))
        if missing:
            raise ValueError(f"Capacity preprocessing scope is missing columns: {missing}")
        selected = metadata.loc[
            metadata["dataset"].astype(str).eq("coswara")
            & metadata["modality"].astype(str).isin(["cough", "speech"])
        ].copy()
        observed = set(selected["modality"].astype(str))
        if observed != {"cough", "speech"}:
            raise ValueError(
                "Capacity preprocessing requires eligible Coswara cough and speech recordings"
            )
        return selected
    if mode not in {"smoke", "pilot"}:
        return metadata.copy()
    required = {"dataset", "modality"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Bounded preprocessing scope is missing columns: {missing}")
    selected = metadata.loc[
        metadata["dataset"].astype(str).eq("coswara")
        & metadata["modality"].astype(str).eq("cough")
    ].copy()
    if selected.empty:
        raise ValueError("Bounded preprocessing requires eligible Coswara cough recordings")
    return selected


def _bind_frozen_audio_sources(
    pipeline: HSTPipeline,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Bind preprocessing rows to the source bytes hashed into the run identity."""
    if pipeline.config.mode == "smoke":
        return metadata.copy()
    required = {"dataset", "modality", "audio_path"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Frozen audio binding is missing columns: {missing}")
    paths = _section(pipeline.config, "paths")
    project_root = pipeline.config.workspace_root.resolve()
    input_hashes = pipeline.config.input_hashes
    inventories: dict[tuple[str, str, str, str], tuple[str, int]] = {}
    selected_datasets = set(metadata["dataset"].astype(str))
    specifications = (
        (
            "coswara",
            _project_path(pipeline.config, paths.get("coswara_metadata", "")),
            "cough" if pipeline.config.mode == "pilot" else None,
        ),
        (
            "coughvid",
            _project_path(pipeline.config, paths.get("coughvid_metadata", "")),
            "cough",
        ),
    )
    for dataset, metadata_path, modality_filter in specifications:
        if dataset not in selected_datasets:
            continue
        records = audio_input_manifest_records(
            metadata_path,
            project_root=project_root,
            modality=modality_filter,
        )
        actual_manifest_hash = canonical_json_sha256(records)
        expected_manifest_hash = input_hashes.get(f"{dataset}_audio_content")
        if actual_manifest_hash != expected_manifest_hash:
            raise ValueError(
                f"{dataset} audio bytes changed after the frozen run identity was created"
            )
        for record in records:
            key = (
                dataset,
                str(record["modality"]),
                str(record["source_locator"]),
                str(record.get("archive_member") or ""),
            )
            value = (str(record["sha256"]), int(record["size_bytes"]))
            prior = inventories.get(key)
            if prior is not None and prior != value:
                raise ValueError(f"Conflicting frozen audio identity: {key}")
            inventories[key] = value

    def locator(value: object) -> tuple[str, str]:
        supplied = str(value)
        member = ""
        if "::" in supplied:
            source_text, member = supplied.split("::", 1)
        else:
            source_text = supplied
        source_path = Path(source_text)
        if not source_path.is_absolute():
            source_path = pipeline.config.workspace_root / source_path
        try:
            resolved = source_path.resolve()
            relative = resolved.relative_to(project_root).as_posix()
        except (ValueError, Exception):
            relative = source_path.as_posix()
        return relative, member

    bound = metadata.copy()
    expected_hashes: list[str] = []
    expected_sizes: list[int] = []
    for row in bound.to_dict(orient="records"):
        relative, member = locator(row["audio_path"])
        key = (
            str(row["dataset"]),
            str(row["modality"]),
            relative,
            member,
        )
        frozen = inventories.get(key)
        if frozen is None:
            raise ValueError(f"No frozen source-audio identity exists for {key}")
        expected_hashes.append(frozen[0])
        expected_sizes.append(frozen[1])
    bound["expected_source_sha256"] = expected_hashes
    bound["expected_source_size_bytes"] = expected_sizes
    return bound


def _external_source_rows_sha256(frame: pd.DataFrame) -> str:
    columns = (
        "participant_key",
        "recording_key",
        "label_binary",
        "modality",
        "split",
        "fold",
        "training_seed",
        "tensor_sha256",
        "representation_id",
    )
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Track-A source identity is missing columns: {missing}")
    records = (
        frame.loc[:, columns]
        .astype(str)
        .sort_values(list(columns), kind="mergesort")
        .to_dict(orient="records")
    )
    return canonical_json_sha256(records)


def _validate_external_source_binding(
    external: pd.DataFrame,
    internal: pd.DataFrame,
    *,
    internal_manifest_sha256: str,
) -> None:
    if len(str(internal_manifest_sha256)) != 64:
        raise ValueError("Frozen internal Track-A manifest hash is invalid")
    expected = internal.loc[
        internal["dataset"].astype(str).eq("coswara")
        & internal["modality"].astype(str).eq("cough")
    ].copy()
    observed = external.loc[
        external["dataset"].astype(str).eq("coswara")
        & external["modality"].astype(str).eq("cough")
    ].copy()
    if expected.empty or observed.empty:
        raise ValueError("External transfer requires frozen internal Track-A cough rows")
    expected_identity = _external_source_rows_sha256(expected)
    if _external_source_rows_sha256(observed) != expected_identity:
        raise ValueError(
            "External source is not the exact frozen internal Track-A cough rows"
        )
    required = {
        "source_track_a_manifest_sha256",
        "source_track_a_cough_rows_sha256",
    }
    missing = sorted(required - set(external.columns))
    if missing:
        raise ValueError(f"External source binding is missing columns: {missing}")
    if not external["source_track_a_manifest_sha256"].astype(str).eq(
        internal_manifest_sha256
    ).all():
        raise ValueError("External source does not bind the frozen internal Track-A manifest")
    if not external["source_track_a_cough_rows_sha256"].astype(str).eq(
        expected_identity
    ).all():
        raise ValueError("External source Track-A cough row identity changed")


def _bind_external_manifest_to_internal(
    external: pd.DataFrame,
    internal: pd.DataFrame,
    *,
    internal_manifest_sha256: str,
) -> pd.DataFrame:
    bound = external.copy()
    source = internal.loc[
        internal["dataset"].astype(str).eq("coswara")
        & internal["modality"].astype(str).eq("cough")
    ]
    bound["source_track_a_manifest_sha256"] = internal_manifest_sha256
    bound["source_track_a_cough_rows_sha256"] = _external_source_rows_sha256(source)
    bound = _protocol_contract._finalize_manifest(bound)
    _validate_external_source_binding(
        bound,
        internal,
        internal_manifest_sha256=internal_manifest_sha256,
    )
    return bound


def _build_aligned_comparator_manifest(
    components: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    names = tuple(components)
    expected = tuple(
        name for name in _ALIGNED_COMPARATOR_COMPONENTS if name in components
    )
    if names != expected or not names:
        raise ValueError(
            "Aligned comparator components must follow the declared canonical order"
        )
    frames: list[pd.DataFrame] = []
    for name in names:
        frame = components[name]
        if frame.empty or "manifest_sha256" not in frame:
            raise ValueError(f"Aligned comparator component {name!r} is not frozen")
        source_hashes = frame["manifest_sha256"].astype(str).unique().tolist()
        if len(source_hashes) != 1 or len(source_hashes[0]) != 64:
            raise ValueError(
                f"Aligned comparator component {name!r} has ambiguous manifest identity"
            )
        current = frame.drop(
            columns=[
                column
                for column in ("row_content_sha256", "manifest_sha256")
                if column in frame
            ]
        ).copy()
        if "source_manifest_sha256" in current:
            current = current.rename(
                columns={
                    "source_manifest_sha256": "data_contract_source_manifest_sha256"
                }
            )
        current["source_manifest_sha256"] = source_hashes[0]
        current["manifest_component"] = name
        if "source_protocol" in current:
            current = current.rename(columns={"source_protocol": "dataset_source_protocol"})
        current["source_protocol"] = current["protocol"].astype(str)
        if name == "task2_like_cough":
            current["protocol"] = "hst_task2_like_cough_exploratory"
        frames.append(current)
    union = pd.concat(frames, ignore_index=True, sort=False)
    identity = ["protocol", "fold", "recording_key", "modality"]
    missing = sorted(set(identity) - set(union.columns))
    if missing:
        raise ValueError(f"Aligned comparator union is missing identity columns: {missing}")
    if union.duplicated(identity, keep=False).any():
        raise ValueError(
            "Aligned comparator manifest has a protocol/fold/recording/modality context collision"
        )
    union = _protocol_contract._sort_manifest(union)
    union["row_content_sha256"] = _protocol_contract._row_hashes(union)
    if union["row_content_sha256"].duplicated().any():
        raise ValueError("Aligned comparator manifest contains duplicate content rows")
    union["manifest_sha256"] = _protocol_contract._manifest_digest(union)
    if union["manifest_sha256"].nunique() != 1:
        raise AssertionError("Aligned comparator union has ambiguous manifest identity")
    return union


@_scientific_handler("preprocess_worker_pilot")
def _preprocess_worker_pilot(
    pipeline: HSTPipeline,
    _stage: str,
) -> Mapping[str, object]:
    metadata = _metadata_for_spectrogram_stage(
        _primary_contract_metadata(pipeline),
        mode=pipeline.config.mode,
        workload_profile=workload_profile_from_scientific_config(
            pipeline.config.scientific_config
        ).name,
    )
    metadata = _bind_frozen_audio_sources(pipeline, metadata)
    spectrogram_config = HSTSpectrogramConfig.paper_default()
    benchmark = benchmark_preprocess_workers(
        metadata,
        candidates=(1, 2, 4),
        sample_size=min(12, len(metadata)),
        config=spectrogram_config,
    )
    valid = benchmark.loc[
        benchmark["valid"].astype(bool)
        & benchmark["completed"].eq(benchmark["sample_size"])
        & benchmark["swap_delta_bytes"].le(0)
    ].copy()
    if valid.empty:
        raise RuntimeError("No preprocessing worker count completed the bounded pilot safely")
    selected = valid.sort_values(
        ["recordings_per_second", "workers"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    output_root = pipeline.run_root / "audits"
    benchmark_path = output_root / "preprocess_worker_benchmark.csv"
    selection_path = output_root / "preprocess_worker_selection.json"
    _atomic_csv(benchmark, benchmark_path)
    atomic_write_json(
        selection_path,
        {
            "schema_version": 1,
            "workers": int(selected["workers"]),
            "recordings_per_second": float(selected["recordings_per_second"]),
            "sample_size": int(selected["sample_size"]),
            "preprocessing_configuration": asdict(spectrogram_config),
        },
    )
    return {
        "output_paths": [benchmark_path, selection_path],
        "row_counts": {"benchmark_rows": len(benchmark)},
        "metadata": {"selected_workers": int(selected["workers"])},
    }


@_scientific_handler("spectrogram_cache")
def _spectrogram_cache(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    metadata = _metadata_for_spectrogram_stage(
        _primary_contract_metadata(pipeline),
        mode=pipeline.config.mode,
        workload_profile=workload_profile_from_scientific_config(
            pipeline.config.scientific_config
        ).name,
    )
    metadata = _bind_frozen_audio_sources(pipeline, metadata)
    selection_path = pipeline.run_root / "audits" / "preprocess_worker_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    workers = int(selection["workers"])
    spectrogram_config = HSTSpectrogramConfig.paper_default()
    candidate_cache = (
        pipeline.config.workspace_root
        / "data"
        / "processed"
        / "hst_spectrogram_cache"
    ).resolve()
    alt_cache = (
        pipeline.config.workspace_root
        / ".cache"
        / "hst"
        / "spectrograms"
    ).resolve()
    cache_root = alt_cache if (not candidate_cache.exists() and alt_cache.exists()) else candidate_cache
    cache = parallel_build_spectrograms(
        metadata,
        workers=workers,
        config=spectrogram_config,
        output_dir=cache_root,
    )
    cache["source_audio_sha256"] = cache["source_sha256"].astype(str)
    cache["preprocessing_variant"] = spectrogram_config.representation_id
    if cache.empty or not cache["eligible"].astype(bool).any():
        raise RuntimeError("HST spectrogram cache produced no eligible recordings")
    if cache.loc[cache["eligible"].astype(bool), "tensor_sha256"].astype(str).str.len().ne(64).any():
        raise ValueError("Eligible HST cache rows are missing tensor checksums")
    if pipeline.config.mode != "smoke":
        if "expected_source_sha256" not in cache.columns:
            raise ValueError("Frozen source hash verification is missing expected_source_sha256")
        if not cache["source_sha256"].astype(str).eq(
            cache["expected_source_sha256"].astype(str)
        ).all():
            raise ValueError("HST cache source hashes disagree with the frozen run identity")
    eligible_cache = cache.loc[cache["eligible"].astype(bool)]
    for row in eligible_cache.itertuples(index=False):
        tensor_path = Path(str(row.cache_path)).resolve()
        try:
            tensor_path.relative_to(cache_root)
        except ValueError as exc:
            raise ValueError("HST tensor path escaped the shared cache root") from exc
        load_verified_cached_image(tensor_path, str(row.tensor_sha256))
    output = pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
    _atomic_csv(cache, output)
    return {
        "output_paths": [output],
        "row_counts": {
            "recordings": len(cache),
            "eligible": int(cache["eligible"].astype(bool).sum()),
            "excluded": int((~cache["eligible"].astype(bool)).sum()),
        },
        "metadata": {
            "representation": spectrogram_config.representation_id,
            "preprocessing_configuration": asdict(spectrogram_config),
            "workers": workers,
        },
    }


@_scientific_handler("manifests")
def _manifests(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    cache_path = pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
    cache = pd.read_csv(cache_path, low_memory=False)
    cache["eligible"] = cache["eligible"].map(
        {True: True, False: False, "True": True, "False": False}
    )
    if cache["eligible"].isna().any():
        raise ValueError("Spectrogram cache contains non-boolean eligibility values")
    cache["eligible"] = cache["eligible"].astype(bool)
    scientific_config = pipeline.config.scientific_config
    workload_profile = workload_profile_from_scientific_config(scientific_config)
    fingerprint = scientific_configuration_fingerprint(scientific_config)
    eligibility = intersect_representation_eligibility(
        cache,
        scientific_config=scientific_config,
        scientific_fingerprint=fingerprint,
    )
    alignment_values = eligibility["eligibility_alignment_fingerprint"].unique()
    if len(alignment_values) != 1:
        raise ValueError("HST eligibility mapping has ambiguous provenance")
    eligibility_hash = str(alignment_values[0])
    source_cache = cache.loc[cache["dataset"].astype(str).eq("coswara")].copy()
    if pipeline.config.mode == "smoke":
        source_cache = source_cache.loc[
            source_cache["modality"].astype(str).eq("cough")
        ].copy()
    if source_cache.empty:
        raise ValueError("HST source manifests require eligible Coswara recordings")
    source_eligibility = intersect_representation_eligibility(
        source_cache,
        scientific_config=scientific_config,
        scientific_fingerprint=fingerprint,
    )
    source_alignment_values = (
        source_eligibility["eligibility_alignment_fingerprint"]
        .astype(str)
        .unique()
    )
    if len(source_alignment_values) != 1:
        raise ValueError("Coswara HST eligibility mapping has ambiguous provenance")
    source_eligibility_hash = str(source_alignment_values[0])
    experiment = _section(pipeline.config, "experiment")
    # Use prespecified seeds for all modes, but smoke mode only uses first seed for validation
    seeds = tuple(int(value) for value in experiment.get("project_seeds", ()))
    if pipeline.config.mode != "smoke" and seeds != _TRACK_A_SEEDS:
        raise ValueError("Confirmatory HST seeds differ from the source-derived freeze")
    # For smoke mode, we still pass all seeds to validation but will only use the first fold
    internal = build_protocol_matched_hst_manifest(
        source_cache,
        seeds=seeds,
        test_fraction=0.20,
        validation_fraction_of_remaining=0.125,
        scientific_config=scientific_config,
        scientific_fingerprint=fingerprint,
        eligibility_mapping=source_eligibility,
        eligibility_fingerprint=source_eligibility_hash,
    )
    manifests = {"internal": internal}
    output_root = pipeline.run_root / "manifests"
    output_root.mkdir(parents=True, exist_ok=True)
    internal_path = output_root / "internal.csv"
    _atomic_csv(internal, internal_path)
    internal_manifest_sha256 = stable_file_sha256(internal_path)
    comparator_components: tuple[str, ...] = ()
    if pipeline.config.mode == "full" and workload_profile.name == FULL_RELIABILITY_PROFILE:
        task2_like = build_hst_task2_like_cough_manifest(
            source_cache,
            seeds=seeds,
            scientific_config=scientific_config,
            scientific_fingerprint=fingerprint,
            eligibility_mapping=source_eligibility,
            eligibility_fingerprint=source_eligibility_hash,
        )
        mixed, chronological = build_split_policy_contrast_manifests(
            source_cache,
            train_fraction=0.60,
            validation_fraction=0.20,
            candidate_count=1000,
            random_state=42,
            training_seed=42,
            analysis_mode="confirmatory",
            scientific_config=scientific_config,
            scientific_fingerprint=fingerprint,
            eligibility_mapping=source_eligibility,
            eligibility_fingerprint=source_eligibility_hash,
        )
        common_mixed, common_chronological = build_common_late_test_manifests(
            source_cache,
            candidate_count=1000,
            random_state=42,
            training_seed=42,
            analysis_mode="confirmatory",
            scientific_config=scientific_config,
            scientific_fingerprint=fingerprint,
            eligibility_mapping=source_eligibility,
            eligibility_fingerprint=source_eligibility_hash,
        )
        reverse = build_reverse_temporal_hst_manifest(
            source_cache,
            scientific_config=scientific_config,
            scientific_fingerprint=fingerprint,
            eligibility_mapping=source_eligibility,
            eligibility_fingerprint=source_eligibility_hash,
        )
        external_source = internal.loc[
            internal["dataset"].astype(str).eq("coswara")
            & internal["modality"].astype(str).eq("cough")
        ].copy()
        if external_source.empty:
            raise ValueError("External transfer requires frozen Track-A cough rows")
        external = build_external_hst_manifest(
            cache,
            external_source,
            scientific_config=scientific_config,
            scientific_fingerprint=fingerprint,
            eligibility_mapping=eligibility,
            eligibility_fingerprint=eligibility_hash,
        )
        external = _bind_external_manifest_to_internal(
            external,
            internal,
            internal_manifest_sha256=internal_manifest_sha256,
        )
        manifests.update(
            {
                "task2_like_cough": task2_like,
                "calendar_mixed": mixed,
                "early_to_late": chronological,
                "common_late_mixed": common_mixed,
                "common_late_chronological": common_chronological,
                "reverse_temporal": reverse,
                "external": external,
            }
        )
        comparator_components = _ALIGNED_COMPARATOR_COMPONENTS
    if comparator_components:
        manifests["aligned_comparator"] = _build_aligned_comparator_manifest(
            {name: manifests[name] for name in comparator_components}
        )
    outputs: list[Path] = []
    row_counts: dict[str, int] = {}
    eligibility_path = output_root / "representation_eligibility.csv"
    _atomic_csv(eligibility, eligibility_path)
    outputs.append(eligibility_path)
    row_counts["representation_eligibility"] = len(eligibility)
    for name, manifest in manifests.items():
        manifest_path = output_root / f"{name}.csv"
        audit_path = output_root / f"{name}_audit.csv"
        _atomic_csv(manifest, manifest_path)
        if name == "internal" and stable_file_sha256(manifest_path) != internal_manifest_sha256:
            raise RuntimeError("Frozen internal Track-A manifest bytes changed during generation")
        _atomic_csv(audit_hst_manifest(manifest), audit_path)
        outputs.extend([manifest_path, audit_path])
        row_counts[name] = len(manifest)
    manifest_index = output_root / "manifest_index.json"
    atomic_write_json(
        manifest_index,
        {
            "schema_version": 1,
            "scientific_configuration_fingerprint": fingerprint,
            "eligibility_alignment_fingerprint": eligibility_hash,
            "manifests": {
                name: {
                    "path": (output_root / f"{name}.csv").relative_to(
                        pipeline.run_root
                    ).as_posix(),
                    "sha256": stable_file_sha256(output_root / f"{name}.csv"),
                    "rows": len(manifest),
                    **(
                        {
                            "source_track_a_manifest_sha256": internal_manifest_sha256,
                            "source_track_a_cough_rows_sha256": _external_source_rows_sha256(
                                internal.loc[
                                    internal["dataset"].astype(str).eq("coswara")
                                    & internal["modality"].astype(str).eq("cough")
                                ]
                            ),
                        }
                        if name == "external"
                        else {}
                    ),
                    **(
                        {
                            "component_manifest_sha256": {
                                component: str(
                                    manifest.loc[
                                        manifest["manifest_component"].eq(component),
                                        "source_manifest_sha256",
                                    ].iloc[0]
                                )
                                for component in comparator_components
                            }
                        }
                        if name == "aligned_comparator"
                        else {}
                    ),
                }
                for name, manifest in manifests.items()
            },
        },
    )
    outputs.append(manifest_index)
    return {
        "output_paths": outputs,
        "row_counts": row_counts,
        "metadata": {
            "scientific_configuration_fingerprint": fingerprint,
            "eligibility_alignment_fingerprint": eligibility_hash,
        },
    }


def _executable_allowlist(
    config: HSTPipelineConfig,
) -> tuple[list[Path], dict[str, str], str]:
    paths = sorted(config.source_paths, key=lambda value: value.as_posix())
    allowlist = {
        path.relative_to(config.source_root).as_posix(): stable_file_sha256(path)
        for path in paths
    }
    digest = verify_executable_allowlist(
        executable_root=config.source_root,
        executable_paths=paths,
        frozen_allowlist=allowlist,
    )
    return paths, allowlist, digest


def _data_contract_freeze_hash(pipeline: HSTPipeline) -> str:
    path = pipeline.run_root / "contracts" / "data_contracts_freeze.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = str(payload.get("manifest_sha256", ""))
    if len(value) != 64:
        raise ValueError("Data-contract freeze manifest is missing its content hash")
    return value


def _training_context(
    *,
    pipeline: HSTPipeline,
    model_name: str,
    model: object,
    source_checkpoint: Path,
    executable_sha256: str,
    protocol: str,
    representation: str,
) -> dict[str, str]:
    return {
        "run_id": pipeline.run_id,
        "protocol": protocol,
        "model": model_name,
        "checkpoint_hash": stable_file_sha256(source_checkpoint),
        "representation": representation,
        "architecture_sha256": _model_architecture_sha256(model),
        "executable_sha256": executable_sha256,
    }


def _evaluation_registry_root(pipeline: HSTPipeline) -> Path:
    expected = (
        pipeline.config.workspace_root
        / "data"
        / "outputs"
        / "hst"
        / "_evaluation_registry"
    ).resolve()
    try:
        expected.relative_to(pipeline.config.workspace_root.resolve())
    except ValueError as exc:
        raise ValueError("Evaluation registry escaped the project root") from exc
    if expected == pipeline.run_root.resolve() or expected.is_relative_to(
        pipeline.run_root.resolve()
    ):
        raise ValueError("Test-once evaluation registry cannot be scoped to one run")
    if expected.exists() and expected.is_symlink():
        raise ValueError("Test-once evaluation registry cannot be a symbolic link")
    expected.mkdir(parents=True, exist_ok=True)
    return expected


def _call_train_hst_fold(
    pipeline: HSTPipeline,
    *args: object,
    confirmatory: bool,
    **kwargs: object,
) -> object:
    call_kwargs = dict(kwargs)
    if confirmatory:
        signature = inspect.signature(train_hst_fold)
        accepts_registry = "evaluation_registry_root" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_registry:
            raise RuntimeError(
                "Confirmatory train_hst_fold API lacks required evaluation_registry_root"
            )
        call_kwargs["evaluation_registry_root"] = _evaluation_registry_root(pipeline)
    return train_hst_fold(*args, **call_kwargs)  # type: ignore[arg-type]


@_scientific_handler("small_smoke")
def _small_smoke(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    import torch

    internal_manifest_path = pipeline.run_root / "manifests" / "internal.csv"
    cache_path = pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
    manifest = _derive_small_smoke_manifest(
        pd.read_csv(internal_manifest_path, low_memory=False)
    )
    manifest_path = pipeline.run_root / "manifests" / "small_smoke.csv"
    _atomic_csv(manifest, manifest_path)
    cache = pd.read_csv(cache_path, low_memory=False)
    cache["eligible"] = cache["eligible"].map(
        {True: True, False: False, "True": True, "False": False}
    ).astype(bool)
    folds = sorted(pd.to_numeric(manifest["fold"], errors="raise").astype(int).unique())
    if len(folds) != 1:
        raise ValueError("Frozen smoke experiment requires exactly one fold")
    if not manifest["modality"].astype(str).eq("cough").all():
        raise ValueError("Frozen smoke experiment requires a cough-only manifest")
    if "external_test" in set(manifest["split"].astype(str)):
        raise ValueError("External target rows cannot enter the smoke experiment")
    fold = int(folds[0])
    modality = "cough"
    seed = 52
    source_checkpoint = _checkpoint_path(pipeline.config, "hst_small_imagenet")
    model, initial_audit = load_verified_hst_model(
        model_name="hst_small",
        checkpoint_path=source_checkpoint,
        hst_repo=_source_path(pipeline.config),
        seed=seed,
    )
    model = model.to(pipeline.config.device)  # type: ignore[attr-defined]
    executable_paths, allowlist, executable_hash = _executable_allowlist(
        pipeline.config
    )
    initial_binding = verify_initial_model_load_audit(
        model,
        source_checkpoint_path=source_checkpoint,
        initial_model_audit=initial_audit,
        model_seed=seed,
    )
    representation = "paper_logmel_224"
    loaders = make_hst_dataloaders(
        cache,
        manifest,
        fold=fold,
        modality=modality,
        physical_batch_size=2,
        num_workers=0,
        seed=seed,
        representation_id=representation,
    )
    training_config = HSTTrainingConfig(
        pilot_freeze_hash=None,
        data_contracts_freeze_hash=_data_contract_freeze_hash(pipeline),
        dependency_lock_hash=stable_file_sha256(
            pipeline.config.dependency_lock_path
        ),
        accepted_environment_lock_hash=None,
        physical_batch_size=2,
        gradient_accumulation=4,
        amp=False,
        max_epochs=2,
        random_seed=seed,
        confirmatory=False,
    )
    run_dir = pipeline.run_root / "models" / "small_smoke" / f"fold-{fold}" / modality
    result = _call_train_hst_fold(
        pipeline,
        model,
        loaders,
        training_config,
        run_dir,
        confirmatory=False,
        prediction_context=_training_context(
            pipeline=pipeline,
            model_name="hst_small",
            model=model,
            source_checkpoint=source_checkpoint,
            executable_sha256=executable_hash,
            protocol=str(manifest["protocol"].iloc[0]),
            representation=representation,
        ),
        manifest_path=manifest_path,
        cache_index_path=cache_path,
        source_checkpoint_path=source_checkpoint,
        executable_root=pipeline.config.source_root,
        executable_paths=executable_paths,
        frozen_executable_allowlist=allowlist,
        manifest_sha256=stable_file_sha256(manifest_path),
        cache_index_sha256=stable_file_sha256(cache_path),
        source_checkpoint_sha256=stable_file_sha256(source_checkpoint),
        initial_model_state_sha256=_model_state_sha256(model),
        initial_model_audit=initial_audit,
        expected_initial_model_binding_sha256=initial_binding,
        resume=pipeline.config.resume,
        stop_after_epoch=2,
        evaluate_test=False,
        evaluate_external=False,
    )
    if (
        int(result.last_epoch) != 2
        or not bool(result.training_complete)
        or result.history.empty
        or result.validation_predictions.empty
    ):
        raise RuntimeError("Frozen HST-Small smoke experiment did not complete two real epochs")
    history_path = run_dir / "history.csv"
    validation_path = run_dir / "validation_predictions.csv"
    summary_path = run_dir / "summary.json"
    _atomic_csv(result.history, history_path)
    _atomic_csv(result.validation_predictions, validation_path)
    atomic_write_json(
        summary_path,
        {
            "schema_version": 1,
            "model": "hst_small",
            "fold": fold,
            "modality": modality,
            "epochs": result.last_epoch,
            "best_epoch": result.best_epoch,
            "validation_threshold": result.validation_threshold,
            "training_complete": result.training_complete,
            "test_evaluated": result.test_evaluated,
            "best_checkpoint_sha256": result.best_checkpoint_sha256,
            "training_contract_fingerprint": result.training_contract_fingerprint,
        },
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    outputs = [manifest_path, *sorted(path for path in run_dir.rglob("*") if path.is_file())]
    return {
        "output_paths": outputs,
        "row_counts": {
            "history": len(result.history),
            "validation_predictions": len(result.validation_predictions),
        },
        "metadata": {
            "model": "hst_small",
            "epochs": 2,
            "test_labels_opened": False,
        },
    }


def _derive_small_smoke_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select the same seed-52 Coswara cough fold in every pipeline mode."""
    required = {"dataset", "modality", "split", "fold", "training_seed"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"HST-Small smoke manifest is missing columns: {missing}")
    selected = manifest.loc[
        manifest["dataset"].astype(str).eq("coswara")
        & manifest["modality"].astype(str).eq("cough")
        & ~manifest["split"].astype(str).eq("external_test")
        & pd.to_numeric(manifest["training_seed"], errors="raise").astype(int).eq(52)
    ].copy()
    if selected.empty:
        raise ValueError("HST-Small smoke experiment requires the frozen seed-52 cough fold")
    folds = pd.to_numeric(selected["fold"], errors="raise").astype(int).unique()
    if len(folds) != 1:
        raise ValueError("HST-Small smoke experiment requires exactly one seed-52 fold")
    selected = _protocol_contract._sort_manifest(selected)
    selected["row_content_sha256"] = _protocol_contract._row_hashes(selected)
    if selected["row_content_sha256"].duplicated().any():
        raise ValueError("HST-Small smoke manifest contains duplicate content rows")
    selected["manifest_sha256"] = _protocol_contract._manifest_digest(selected)
    return selected.reset_index(drop=True)


def _frozen_runtime_projection_workload(
    metadata: pd.DataFrame,
    *,
    workload_profile: str = FULL_RELIABILITY_PROFILE,
    project_seeds: tuple[int, ...],
    primary_modalities: tuple[str, ...],
    secondary_modalities: tuple[str, ...],
    effective_batch_size: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return a source-only upper bound for an allowlisted HST workload."""
    if project_seeds != _TRACK_A_SEEDS:
        raise ValueError("Runtime projection requires the exact frozen HST seed hierarchy")
    profile = workload_profile_from_scientific_config(
        {
            "experiment": {
                "workload_profile": workload_profile,
                "primary_modalities": list(primary_modalities),
                "secondary_modalities": list(secondary_modalities),
            }
        }
    )
    if "dataset" not in metadata:
        raise ValueError("Runtime projection metadata is missing dataset")
    source = metadata.loc[metadata["dataset"].astype(str).eq("coswara")].copy()
    if source.empty:
        raise ValueError("Runtime projection requires contract-eligible Coswara rows")

    modalities = (*primary_modalities, *secondary_modalities)
    updates = conservative_balanced_optimizer_updates_per_epoch(
        source,
        modalities=modalities,
        effective_batch_size=effective_batch_size,
    )
    jobs = {str(key): int(value) for key, value in profile.training_jobs_by_modality.items()}
    if set(jobs) != set(modalities):
        raise AssertionError("Frozen HST workload modalities do not match its job plan")
    return updates, jobs


def _manifest_training_seed(
    manifest: pd.DataFrame,
    *,
    fold: int,
    modality: str,
) -> int:
    required = {"fold", "modality", "training_seed"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Resource-pilot manifest is missing columns: {missing}")
    selected = manifest.loc[
        pd.to_numeric(manifest["fold"], errors="raise").astype(int).eq(fold)
        & manifest["modality"].astype(str).eq(modality)
    ]
    seeds = pd.to_numeric(selected["training_seed"], errors="raise").astype(int).unique()
    if len(seeds) != 1:
        raise ValueError("Resource-pilot fold/modality must bind exactly one training seed")
    return int(seeds[0])


@_scientific_handler("base_resource_pilot")
def _base_resource_pilot(
    pipeline: HSTPipeline,
    _stage: str,
) -> Mapping[str, object]:
    if pipeline.config.device != "cuda":
        raise RuntimeError("The HST-Base resource pilot requires CUDA")
    manifest_path = pipeline.run_root / "manifests" / "internal.csv"
    cache_path = pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
    manifest = pd.read_csv(manifest_path, low_memory=False)
    fold = int(sorted(pd.to_numeric(manifest["fold"], errors="raise").unique())[0])
    modality = "cough"
    seed = _manifest_training_seed(manifest, fold=fold, modality=modality)
    runtime_config = _section(pipeline.config, "runtime")
    maximum_runtime_hours = float(
        runtime_config.get("maximum_projected_serial_gpu_hours", 0.0)
    )
    overhead_multiplier = float(
        runtime_config.get("end_to_end_overhead_multiplier", 0.0)
    )
    experiment = _section(pipeline.config, "experiment")
    profile = workload_profile_from_scientific_config(
        pipeline.config.scientific_config
    )
    project_seeds = tuple(int(value) for value in experiment.get("project_seeds", ()))
    primary = tuple(str(value) for value in experiment.get("primary_modalities", ()))
    secondary = tuple(str(value) for value in experiment.get("secondary_modalities", ()))
    updates_by_modality, jobs_by_modality = _frozen_runtime_projection_workload(
        _primary_contract_metadata(pipeline),
        workload_profile=profile.name,
        project_seeds=project_seeds,
        primary_modalities=primary,
        secondary_modalities=secondary,
        effective_batch_size=8,
    )
    runtime_policy = runtime_projection_policy_payload(
        workload_profile=profile.name,
        optimizer_updates_per_epoch_by_modality=updates_by_modality,
        planned_training_jobs_by_modality=jobs_by_modality,
        confirmatory_epochs=100,
        end_to_end_overhead_multiplier=overhead_multiplier,
        maximum_approved_runtime_hours=maximum_runtime_hours,
    )
    runtime_policy_hash = canonical_json_sha256(runtime_policy)
    checkpoint_path = _checkpoint_path(pipeline.config, "hst_base_imagenet")
    benchmark, selection = run_base_resource_pilot_trials(
        cache_index_path=cache_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        hst_repo=_source_path(pipeline.config),
        worker_script=pipeline.config.workspace_root
        / "scripts"
        / "hst_resource_pilot_worker.py",
        fold=fold,
        modality=modality,
        seed=seed,
        freeze_context={
            "model_name": "hst_base",
            "hst_commit": pipeline.config.hst_commit,
            "model_source_sha256": hst_model_source_sha256(
                _source_path(pipeline.config), pipeline.config.hst_commit
            ),
            "checkpoint_sha256": stable_file_sha256(checkpoint_path),
            "data_contracts_freeze_hash": _data_contract_freeze_hash(pipeline),
            "dependency_lock_sha256": stable_file_sha256(
                pipeline.config.dependency_lock_path
            ),
            "pilot_modality": modality,
            "pilot_fold": fold,
            "pilot_seed": seed,
            "pilot_optimizer_updates": 100,
            "runtime_projection_policy_sha256": runtime_policy_hash,
        },
    )
    selected_trial = benchmark.loc[
        pd.to_numeric(benchmark["physical_batch_size"], errors="raise")
        .astype(int)
        .eq(int(selection["physical_batch_size"]))
        & benchmark["precision"].astype(str).eq(str(selection["precision"]))
    ]
    if len(selected_trial) != 1:
        raise ValueError("Resource pilot selected trial is missing or duplicated")
    selected_trial_row = selected_trial.iloc[0]
    if int(selection["effective_batch_size"]) != 8:
        raise ValueError("Resource pilot changed the frozen effective batch size")
    projection = project_full_training_runtime(
        workload_profile=profile.name,
        selected_trial_seconds=float(selected_trial_row["seconds"]),
        selected_trial_optimizer_updates=int(selected_trial_row["optimizer_updates"]),
        optimizer_updates_per_epoch_by_modality=updates_by_modality,
        planned_training_jobs_by_modality=jobs_by_modality,
        confirmatory_epochs=100,
        end_to_end_overhead_multiplier=overhead_multiplier,
        maximum_approved_runtime_hours=maximum_runtime_hours,
    )
    if projection["runtime_projection_policy_sha256"] != runtime_policy_hash:
        raise ValueError("Resource pilot runtime policy changed during measurement")
    child_peak_allocated_mb = float(
        selected_trial_row["peak_allocated_vram_bytes"]
    ) / 1024**2
    child_peak_reserved_mb = float(
        selected_trial_row["peak_reserved_vram_bytes"]
    ) / 1024**2
    if (
        not math.isfinite(child_peak_allocated_mb)
        or not math.isfinite(child_peak_reserved_mb)
        or child_peak_allocated_mb < 0
        or child_peak_reserved_mb < child_peak_allocated_mb
    ):
        raise ValueError("Resource pilot child-process GPU peaks are inconsistent")
    selection["runtime_projection"] = projection
    output_root = pipeline.run_root / "audits"
    benchmark_path = output_root / "base_resource_pilot_trials.csv"
    selection_path = output_root / "base_resource_pilot_freeze.json"
    _atomic_csv(benchmark, benchmark_path)
    atomic_write_json(selection_path, selection)
    pilot_hash = str(selection.get("pilot_freeze_hash", ""))
    if len(pilot_hash) != 64:
        raise ValueError("Resource pilot did not produce a valid freeze hash")
    if pipeline.config.mode == "full" and (
        pipeline.config.accepted_hashes.get("pilot_freeze") != pilot_hash
    ):
        raise ValueError("Resource pilot does not match the manually accepted freeze")
    if pipeline.config.mode == "full" and not bool(
        projection["within_approved_runtime_ceiling"]
    ):
        raise ValueError("Current full-run projection exceeds the frozen runtime ceiling")
    return {
        "output_paths": [benchmark_path, selection_path],
        "row_counts": {"resource_trials": len(benchmark)},
        "metadata": {
            "pilot_freeze_hash": pilot_hash,
            "physical_batch_size": int(selection["physical_batch_size"]),
            "gradient_accumulation": int(selection["gradient_accumulation"]),
            "amp": bool(selection["amp"]),
            "model_metrics_used": bool(selection["model_metrics_used"]),
            "estimated_serial_gpu_hours": float(
                projection["estimated_serial_gpu_hours"]
            ),
            "within_approved_runtime_ceiling": bool(
                projection["within_approved_runtime_ceiling"]
            ),
            "child_gpu_memory_measured": True,
            "child_peak_gpu_memory_allocated_mb": child_peak_allocated_mb,
            "child_peak_gpu_memory_reserved_mb": child_peak_reserved_mb,
            "gpu_memory_measurement_scope": "selected_resource_pilot_child_process",
        },
    }


def _output_records(paths: list[Path], *, root: Path) -> list[dict[str, object]]:
    base = Path(root).resolve()
    records: list[dict[str, object]] = []
    for supplied in sorted({Path(path).resolve() for path in paths}, key=lambda value: value.as_posix()):
        try:
            relative = supplied.relative_to(base).as_posix()
        except ValueError as exc:
            raise ValueError(f"Job output escaped its stage root: {supplied}") from exc
        if not supplied.is_file():
            raise FileNotFoundError(f"Job output is missing: {supplied}")
        records.append(
            {
                "path": relative,
                "size_bytes": supplied.stat().st_size,
                "sha256": stable_file_sha256(supplied),
            }
        )
    if not records:
        raise ValueError("A successful scientific job requires durable outputs")
    return records


def _write_job_receipt(path: Path, payload: Mapping[str, object]) -> None:
    receipt = {**dict(payload), "receipt_type": "hst_scientific_job"}
    receipt.pop("record_hash", None)
    receipt["record_hash"] = canonical_json_sha256(receipt)
    atomic_write_json(Path(path), receipt)


def _validated_reusable_job(
    receipt_path: Path,
    *,
    job_spec_sha256: str,
    run_id: str,
    job_id: str,
    root: Path,
) -> bool:
    path = Path(receipt_path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Scientific job receipt is corrupt: {path}") from exc
    if payload.get("status") != "success":
        return False
    if (
        payload.get("receipt_type") != "hst_scientific_job"
        or payload.get("run_id") != run_id
        or payload.get("job_id") != job_id
    ):
        raise ValueError("Successful job receipt identity is invalid")
    record_hash = payload.get("record_hash")
    unsigned = dict(payload)
    unsigned.pop("record_hash", None)
    if (
        not isinstance(record_hash, str)
        or len(record_hash) != 64
        or canonical_json_sha256(unsigned) != record_hash
    ):
        raise ValueError("Successful job receipt self-hash is invalid")
    if payload.get("job_spec_sha256") != job_spec_sha256:
        raise ValueError("Successful job receipt belongs to a different immutable job specification")
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Successful job receipt has no checksummed outputs")
    base = Path(root).resolve()
    for record in outputs:
        if not isinstance(record, Mapping):
            raise ValueError("Scientific job output receipt is malformed")
        candidate = (base / str(record.get("path", ""))).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ValueError("Scientific job output receipt escapes its stage root") from exc
        if not candidate.is_file():
            raise ValueError(f"Scientific job output is missing: {candidate}")
        if candidate.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"Scientific job output size changed: {candidate}")
        if stable_file_sha256(candidate) != str(record.get("sha256", "")):
            raise ValueError(f"Scientific job output checksum changed: {candidate}")
    return True


def _receipt_output_paths(receipt_path: Path, *, root: Path) -> list[Path]:
    payload = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if payload.get("status") != "success" or not isinstance(payload.get("outputs"), list):
        raise ValueError("Scientific job receipt is not a complete success receipt")
    base = Path(root).resolve()
    paths: list[Path] = []
    for record in payload["outputs"]:
        if not isinstance(record, Mapping):
            raise ValueError("Scientific job receipt contains a malformed output")
        path = (base / str(record.get("path", ""))).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ValueError("Scientific job receipt output escaped its stage root") from exc
        paths.append(path)
    return paths


def _manifest_index(pipeline: HSTPipeline) -> dict[str, object]:
    path = pipeline.run_root / "manifests" / "manifest_index.json"
    if not path.is_file():
        raise FileNotFoundError("Frozen HST manifest index is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Frozen HST manifest index is corrupt") from exc
    manifests = payload.get("manifests")
    if not isinstance(manifests, Mapping) or not manifests:
        raise ValueError("Frozen HST manifest index contains no manifests")
    return payload


def _load_indexed_manifest(
    pipeline: HSTPipeline,
    name: str,
) -> tuple[Path, pd.DataFrame, str]:
    index = _manifest_index(pipeline)
    entry = index["manifests"].get(name)  # type: ignore[index]
    if not isinstance(entry, Mapping):
        raise ValueError(f"Frozen HST manifest {name!r} is absent from the manifest index")
    path = (pipeline.run_root / str(entry.get("path", ""))).resolve()
    try:
        path.relative_to(pipeline.run_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Frozen HST manifest {name!r} escapes the run root") from exc
    expected_path = (pipeline.run_root / "manifests" / f"{name}.csv").resolve()
    if path != expected_path or not path.is_file():
        raise ValueError(f"Frozen HST manifest {name!r} path is not canonical")
    digest = stable_file_sha256(path)
    if digest != str(entry.get("sha256", "")):
        raise ValueError(f"Frozen HST manifest {name!r} checksum changed")
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != int(entry.get("rows", -1)):
        raise ValueError(f"Frozen HST manifest {name!r} row count changed")
    return path, frame, digest


def _job_manifest_rows_sha256(frame: pd.DataFrame) -> str:
    columns = (
        "participant_key",
        "recording_key",
        "label_binary",
        "modality",
        "split",
        "tensor_sha256",
        "representation_id",
    )
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Training manifest is missing job-identity columns: {missing}")
    records = (
        frame.loc[:, columns]
        .astype(str)
        .sort_values(list(columns), kind="mergesort")
        .to_dict(orient="records")
    )
    return canonical_json_sha256(records)


def _single_manifest_value(frame: pd.DataFrame, column: str, *, name: str) -> object:
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"Frozen HST manifest {name!r} has conflicting {column} values")
    return values[0]


def _explicit_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Frozen manifest {field_name} must be an explicit boolean")


def _validate_manifest_analysis_hierarchy(
    frame: pd.DataFrame,
    *,
    manifest_name: str,
) -> None:
    expected = {
        "internal": ("internal_performance", "primary", True),
        "task2_like_cough": ("symptom_matched_cough", "exploratory", False),
        "calendar_mixed": ("reliability_evaluation", "secondary", True),
        "early_to_late": ("reliability_evaluation", "secondary", True),
        "common_late_mixed": ("reliability_evaluation", "secondary", True),
        "common_late_chronological": ("reliability_evaluation", "secondary", True),
        "reverse_temporal": ("sensitivity_analysis", "sensitivity", False),
    }
    if manifest_name not in expected:
        raise ValueError(f"Scientific training manifest {manifest_name!r} is not prespecified")
    observed_scope = str(
        _single_manifest_value(frame, "analysis_scope", name=manifest_name)
    )
    observed_role = str(
        _single_manifest_value(frame, "analysis_role", name=manifest_name)
    )
    observed_confirmatory = _explicit_bool(
        _single_manifest_value(frame, "confirmatory_protocol", name=manifest_name),
        field_name="confirmatory_protocol",
    )
    expected_scope, expected_role, expected_confirmatory = expected[manifest_name]
    if (observed_scope, observed_role, observed_confirmatory) != expected[manifest_name]:
        label = "Task-2-like" if manifest_name == "task2_like_cough" else manifest_name
        raise ValueError(
            f"{label} manifest must remain {expected_role} with "
            f"analysis_scope={expected_scope!r} and confirmatory_protocol="
            f"{expected_confirmatory}"
        )


def _enumerate_training_jobs(
    pipeline: HSTPipeline,
    *,
    stage: str,
    manifest_names: tuple[str, ...],
    modalities: tuple[str, ...],
    primary_modalities: tuple[str, ...],
) -> list[dict[str, object]]:
    if pipeline.config.mode != "full":
        raise RuntimeError(f"Scientific training stage {stage!r} requires full mode")
    cache_path = pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
    if not cache_path.is_file():
        raise FileNotFoundError("HST spectrogram cache index is missing")
    cache_sha256 = stable_file_sha256(cache_path)
    jobs: list[dict[str, object]] = []
    for manifest_name in manifest_names:
        manifest_path, manifest, manifest_sha256 = _load_indexed_manifest(
            pipeline, manifest_name
        )
        required = {
            "fold",
            "training_seed",
            "protocol",
            "modality",
            "split",
            "analysis_role",
            "analysis_scope",
            "estimand_id",
            "multiplicity_family",
            "confirmatory_protocol",
        }
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise ValueError(f"Frozen HST manifest {manifest_name!r} is missing columns: {missing}")
        _validate_manifest_analysis_hierarchy(
            manifest,
            manifest_name=manifest_name,
        )
        for modality in modalities:
            modality_frame = manifest.loc[manifest["modality"].astype(str).eq(modality)]
            if modality_frame.empty:
                raise ValueError(
                    f"Frozen HST manifest {manifest_name!r} has no {modality!r} rows"
                )
            folds = sorted(pd.to_numeric(modality_frame["fold"], errors="raise").astype(int).unique())
            for fold in folds:
                selected = modality_frame.loc[
                    pd.to_numeric(modality_frame["fold"], errors="raise").astype(int).eq(fold)
                ].copy()
                split_set = set(selected["split"].astype(str))
                if not {"train", "validation", "test"}.issubset(split_set):
                    raise ValueError(
                        f"Scientific training job {manifest_name}/{modality}/fold-{fold} "
                        "requires train, validation, and test rows"
                    )
                if "external_test" in split_set:
                    raise ValueError("External target rows cannot enter a scientific training job")
                protocols = selected["protocol"].astype(str).unique().tolist()
                seeds = pd.to_numeric(selected["training_seed"], errors="raise").astype(int).unique().tolist()
                if len(protocols) != 1 or len(seeds) != 1:
                    raise ValueError("A scientific HST job has ambiguous protocol or training seed")
                seed = int(seeds[0])
                if stage == "internal_cv":
                    if fold < 1 or fold > len(_TRACK_A_SEEDS):
                        raise ValueError("Track-A fold is outside the frozen ten repetitions")
                    if seed != _TRACK_A_SEEDS[fold - 1]:
                        raise ValueError("Track-A fold/seed mapping differs from the frozen plan")
                elif fold != 1 or seed != 42:
                    raise ValueError("Temporal scientific jobs must use fold 1 and model seed 42")
                train_validation = selected.loc[
                    selected["split"].astype(str).isin(["train", "validation"])
                ]
                job = {
                    "schema_version": 1,
                    "stage": stage,
                    "manifest_name": manifest_name,
                    "manifest_path": manifest_path.as_posix(),
                    "manifest_sha256": manifest_sha256,
                    "manifest_rows": len(manifest),
                    "fold_rows_sha256": _external_source_rows_sha256(selected),
                    "training_rows_sha256": _job_manifest_rows_sha256(train_validation),
                    "cache_index_path": cache_path.as_posix(),
                    "cache_index_sha256": cache_sha256,
                    "protocol": protocols[0],
                    "fold": int(fold),
                    "seed": seed,
                    "modality": modality,
                    "model_name": "hst_base",
                    "representation_id": "paper_logmel_224",
                    "analysis_queue": (
                        "primary"
                        if modality in primary_modalities
                        else (
                            "exploratory"
                            if str(selected["analysis_role"].iloc[0]) == "exploratory"
                            else "secondary"
                        )
                    ),
                    "analysis_role": str(
                        _single_manifest_value(selected, "analysis_role", name=manifest_name)
                    ),
                    "analysis_scope": str(
                        _single_manifest_value(selected, "analysis_scope", name=manifest_name)
                    ),
                    "estimand_id": str(
                        _single_manifest_value(selected, "estimand_id", name=manifest_name)
                    ),
                    "multiplicity_family": str(
                        _single_manifest_value(
                            selected, "multiplicity_family", name=manifest_name
                        )
                    ),
                    "confirmatory_protocol": _explicit_bool(
                        _single_manifest_value(
                            selected, "confirmatory_protocol", name=manifest_name
                        ),
                        field_name="confirmatory_protocol",
                    ),
                    "configuration_sha256": canonical_json_sha256(
                        pipeline.config.scientific_config
                    ),
                }
                job_spec_sha256 = canonical_json_sha256(job)
                jobs.append(
                    {
                        **job,
                        "job_spec_sha256": job_spec_sha256,
                        "job_id": f"{stage}-{job_spec_sha256[:20]}",
                    }
                )
    jobs.sort(
        key=lambda row: (
            str(row["manifest_name"]),
            int(row["fold"]),
            str(row["modality"]),
        )
    )
    if stage == "internal_cv":
        expected = {(fold, modality) for fold in range(1, 11) for modality in modalities}
        observed = {(int(job["fold"]), str(job["modality"])) for job in jobs}
        if observed != expected:
            raise ValueError("Track-A job plan is not the exact ten-fold modality bank")
    return jobs


def _load_confirmatory_bindings(pipeline: HSTPipeline) -> dict[str, object]:
    if pipeline.config.mode != "full":
        raise RuntimeError("Confirmatory HST-Base execution requires full mode")
    accepted = pipeline.config.accepted_hashes
    required = ("data_contracts_freeze", "pilot_freeze", "environment_lock")
    if any(len(str(accepted.get(name, ""))) != 64 for name in required):
        raise ValueError("Confirmatory HST execution is missing accepted freeze hashes")
    data_path = pipeline.run_root / "contracts" / "data_contracts_freeze.json"
    environment_path = pipeline.run_root / "audits" / "environment.json"
    pilot_path = pipeline.run_root / "audits" / "base_resource_pilot_freeze.json"
    for path in (data_path, environment_path, pilot_path):
        if not path.is_file():
            raise FileNotFoundError(f"Confirmatory prerequisite is missing: {path}")
    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Confirmatory HST prerequisite JSON is corrupt") from exc
    actual_freezes = {
        "data_contracts_freeze": str(data.get("manifest_sha256", "")),
        "pilot_freeze": str(pilot.get("pilot_freeze_hash", "")),
        "environment_lock": str(environment.get("pip_freeze_sha256", "")),
    }
    for name in required:
        if actual_freezes[name] != str(accepted[name]):
            raise ValueError(f"Confirmatory {name} differs from the manually accepted freeze")
    source_hash_method = getattr(pipeline, "_source_hash", None)
    if callable(source_hash_method) and source_hash_method() != pipeline.initial_source_hash:
        raise ValueError("Executable HST source changed after run identity was derived")
    executable_paths, allowlist, executable_sha256 = _executable_allowlist(pipeline.config)
    checkpoint_path = _checkpoint_path(pipeline.config, "hst_base_imagenet")
    checkpoint_spec = _section(pipeline.config, "checkpoints").get("hst_base_imagenet")
    if not isinstance(checkpoint_spec, Mapping):
        raise ValueError("Frozen HST-Base checkpoint specification is missing")
    verify_file(
        checkpoint_path,
        expected_size=int(checkpoint_spec.get("size_bytes", -1)),
        expected_sha256=str(checkpoint_spec.get("sha256", "")),
    )
    try:
        physical_batch_size = int(pilot["physical_batch_size"])
        gradient_accumulation = int(pilot["gradient_accumulation"])
        amp = bool(pilot["amp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Accepted resource pilot is missing its execution settings") from exc
    if physical_batch_size * gradient_accumulation != 8:
        raise ValueError("Accepted resource pilot does not preserve effective batch size 8")
    return {
        **actual_freezes,
        "resource_pilot_receipt_path": pilot_path,
        "resource_pilot_receipt_sha256": stable_file_sha256(pilot_path),
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation": gradient_accumulation,
        "amp": amp,
        "approved_resource_pairs": ((physical_batch_size, gradient_accumulation),),
        "source_checkpoint_path": checkpoint_path,
        "source_checkpoint_sha256": stable_file_sha256(checkpoint_path),
        "executable_paths": executable_paths,
        "executable_allowlist": allowlist,
        "executable_sha256": executable_sha256,
        "dependency_lock_file_sha256": stable_file_sha256(
            pipeline.config.dependency_lock_path
        ),
        "verified": True,
    }


def _bind_job(job: Mapping[str, object], bindings: Mapping[str, object]) -> dict[str, object]:
    payload = {
        key: value
        for key, value in job.items()
        if key not in {"job_id", "job_spec_sha256"}
    }
    for name in (
        "data_contracts_freeze",
        "pilot_freeze",
        "environment_lock",
        "resource_pilot_receipt_sha256",
        "source_checkpoint_sha256",
        "executable_sha256",
        "dependency_lock_file_sha256",
        "physical_batch_size",
        "gradient_accumulation",
        "amp",
    ):
        if name in bindings:
            if name == "source_checkpoint_sha256" and name in payload:
                payload["architecture_initial_checkpoint_sha256"] = bindings[name]
            else:
                payload[name] = bindings[name]
    digest = canonical_json_sha256(payload)
    return {**payload, "job_spec_sha256": digest, "job_id": f"{payload['stage']}-{digest[:20]}"}


def _participant_metric_rows(
    predictions: pd.DataFrame,
    *,
    job: Mapping[str, object],
    threshold: float,
    threshold_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, group in predictions.groupby("split", sort=False):
        if group.empty:
            continue
        metrics = binary_metric_bundle(
            labels_to_binary(group["label_binary"]),
            group["probability"].astype(float).to_numpy(),
            threshold=threshold,
        )
        rows.append(
            {
                **metrics,
                "stage": job["stage"],
                "job_id": job["job_id"],
                "protocol": job["protocol"],
                "fold": int(job["fold"]),
                "seed": int(job["seed"]),
                "modality": job["modality"],
                "model": "hst_base",
                "model_name": "hst_base",
                "metric_split": str(split),
                "threshold_source": threshold_source,
                "n_participants": int(len(group)),
                "analysis_queue": job.get("analysis_queue", "secondary"),
                "analysis_role": job.get("analysis_role", "secondary"),
                "analysis_scope": job.get("analysis_scope", "reliability_evaluation"),
                "estimand_id": job.get("estimand_id", ""),
                "multiplicity_family": job.get("multiplicity_family", ""),
            }
        )
    if not rows:
        raise ValueError("Scientific HST job produced no participant-level metric rows")
    return pd.DataFrame(rows)


def _cleanup_job_device(model: object | None, loaders: object | None) -> None:
    if isinstance(loaders, Mapping):
        for loader in loaders.values():
            iterator = getattr(loader, "_iterator", None)
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
    if model is not None:
        to_method = getattr(model, "to", None)
        if callable(to_method):
            try:
                to_method("cpu")
            except Exception:
                pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _load_reusable_job_outputs(job_root: Path) -> dict[str, object]:
    predictions_path = job_root / "participant_predictions.csv"
    metrics_path = job_root / "metrics.csv"
    summary_path = job_root / "summary.json"
    for path in (predictions_path, metrics_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Reusable scientific job output is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(str(summary["best_checkpoint_path"])).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Reusable HST source checkpoint is missing")
    if stable_file_sha256(checkpoint_path) != str(summary["best_checkpoint_sha256"]):
        raise ValueError("Reusable HST source checkpoint checksum changed")
    return {
        "participant_predictions": pd.read_csv(predictions_path, low_memory=False),
        "metrics": pd.read_csv(metrics_path, low_memory=False),
        "receipt_path": job_root / "job_receipt.json",
        "best_checkpoint_path": checkpoint_path,
        "best_checkpoint_sha256": str(summary["best_checkpoint_sha256"]),
        "validation_threshold": float(summary["validation_threshold"]),
        "training_contract_fingerprint": str(
            summary["training_contract_fingerprint"]
        ),
    }


def _execute_training_job(
    pipeline: HSTPipeline,
    job: dict[str, object],
    bindings: Mapping[str, object],
) -> dict[str, object]:
    stage_root = pipeline.run_root / "scientific" / str(job["stage"])
    job_root = stage_root / "jobs" / str(job["job_id"])
    receipt_path = job_root / "job_receipt.json"
    if _validated_reusable_job(
        receipt_path,
        job_spec_sha256=str(job["job_spec_sha256"]),
        run_id=pipeline.run_id,
        job_id=str(job["job_id"]),
        root=stage_root,
    ):
        return _load_reusable_job_outputs(job_root)
    previous_attempt = 0
    if receipt_path.is_file():
        try:
            previous_attempt = int(
                json.loads(receipt_path.read_text(encoding="utf-8")).get("attempt", 0)
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            raise ValueError(f"Prior scientific job receipt is corrupt: {receipt_path}")
    attempt = previous_attempt + 1
    _write_job_receipt(
        receipt_path,
        {
            "schema_version": 1,
            "status": "running",
            "run_id": pipeline.run_id,
            "job_id": job["job_id"],
            "job_spec_sha256": job["job_spec_sha256"],
            "attempt": attempt,
            "job": job,
        },
    )
    model: object | None = None
    loaders: object | None = None
    try:
        if job.get("model_name") != "hst_base":
            raise ValueError("Confirmatory scientific stages permit HST-Base only")
        source_checkpoint = Path(str(bindings["source_checkpoint_path"])).resolve()
        model, initial_audit = load_verified_hst_model(
            model_name="hst_base",
            checkpoint_path=source_checkpoint,
            hst_repo=_source_path(pipeline.config),
            seed=int(job["seed"]),
        )
        model = model.to(pipeline.config.device)  # type: ignore[attr-defined]
        initial_state_sha256 = _model_state_sha256(model)
        initial_binding_sha256 = verify_initial_model_load_audit(
            model,
            source_checkpoint_path=source_checkpoint,
            initial_model_audit=initial_audit,
            model_seed=int(job["seed"]),
        )
        manifest_path = Path(str(job["manifest_path"])).resolve()
        cache_path = Path(str(job["cache_index_path"])).resolve()
        manifest = pd.read_csv(manifest_path, low_memory=False)
        cache = pd.read_csv(cache_path, low_memory=False)
        loaders = make_hst_dataloaders(
            cache,
            manifest,
            fold=int(job["fold"]),
            modality=str(job["modality"]),
            physical_batch_size=int(bindings["physical_batch_size"]),
            num_workers=0,
            seed=int(job["seed"]),
            representation_id=str(job["representation_id"]),
        )
        training_config = HSTTrainingConfig(
            pilot_freeze_hash=str(bindings["pilot_freeze"]),
            data_contracts_freeze_hash=str(bindings["data_contracts_freeze"]),
            dependency_lock_hash=str(bindings["environment_lock"]),
            accepted_environment_lock_hash=str(bindings["environment_lock"]),
            resource_pilot_receipt_sha256=str(
                bindings["resource_pilot_receipt_sha256"]
            ),
            approved_resource_pairs=tuple(bindings["approved_resource_pairs"]),  # type: ignore[arg-type]
            physical_batch_size=int(bindings["physical_batch_size"]),
            gradient_accumulation=int(bindings["gradient_accumulation"]),
            amp=bool(bindings["amp"]),
            max_epochs=100,
            random_seed=int(job["seed"]),
            confirmatory=True,
        )
        prediction_context = _training_context(
            pipeline=pipeline,
            model_name="hst_base",
            model=model,
            source_checkpoint=source_checkpoint,
            executable_sha256=str(bindings["executable_sha256"]),
            protocol=str(job["protocol"]),
            representation=str(job["representation_id"]),
        )
        model_root = job_root / "training"
        result = _call_train_hst_fold(
            pipeline,
            model,
            loaders,  # type: ignore[arg-type]
            training_config,
            model_root,
            confirmatory=True,
            prediction_context=prediction_context,
            manifest_path=manifest_path,
            cache_index_path=cache_path,
            source_checkpoint_path=source_checkpoint,
            executable_root=pipeline.config.source_root,
            executable_paths=bindings["executable_paths"],  # type: ignore[arg-type]
            frozen_executable_allowlist=bindings["executable_allowlist"],  # type: ignore[arg-type]
            manifest_sha256=str(job["manifest_sha256"]),
            cache_index_sha256=str(job["cache_index_sha256"]),
            source_checkpoint_sha256=str(bindings["source_checkpoint_sha256"]),
            initial_model_state_sha256=initial_state_sha256,
            initial_model_audit=initial_audit,
            expected_initial_model_binding_sha256=initial_binding_sha256,
            progress_context={
                "run_id": pipeline.run_id,
                "stage": str(job["stage"]),
                "job_id": str(job["job_id"]),
                "job_spec_sha256": str(job["job_spec_sha256"]),
                "fold": int(job["fold"]),
                "seed": int(job["seed"]),
                "modality": str(job["modality"]),
                "protocol": str(job["protocol"]),
            },
            resource_pilot_receipt_path=Path(
                str(bindings["resource_pilot_receipt_path"])
            ),
            resume=pipeline.config.resume,
            evaluate_test=True,
            evaluate_external=False,
        )
        if result.interrupted or not result.training_complete or not result.test_evaluated:
            _write_job_receipt(
                receipt_path,
                {
                    "schema_version": 1,
                    "status": "stopped",
                    "run_id": pipeline.run_id,
                    "job_id": job["job_id"],
                    "job_spec_sha256": job["job_spec_sha256"],
                    "attempt": attempt,
                    "resumable": True,
                },
            )
            raise RuntimeError("Confirmatory HST job stopped before held-out evaluation")
        recording_predictions = pd.concat(
            [result.validation_predictions, result.test_predictions],
            ignore_index=True,
        )
        participant_predictions = aggregate_recording_predictions(recording_predictions)
        participant_predictions["job_id"] = job["job_id"]
        participant_predictions["seed"] = int(job["seed"])
        participant_predictions["analysis_queue"] = job["analysis_queue"]
        metrics = _participant_metric_rows(
            participant_predictions,
            job=job,
            threshold=float(result.validation_threshold),
            threshold_source="validation_balanced_accuracy",
        )
        recording_path = job_root / "recording_predictions.csv"
        participant_path = job_root / "participant_predictions.csv"
        metrics_path = job_root / "metrics.csv"
        history_path = job_root / "history.csv"
        for frame, path in (
            (recording_predictions, recording_path),
            (participant_predictions, participant_path),
            (metrics, metrics_path),
            (result.history, history_path),
        ):
            _atomic_csv(frame, path)
        checkpoint_payload, checkpoint_path = _load_verified_checkpoint_with_path(
            model_root / "best.pt"
        )
        if checkpoint_payload.get("checkpoint_role") != "best":
            raise ValueError("Scientific source checkpoint does not have immutable best role")
        checkpoint_sha256 = stable_file_sha256(checkpoint_path)
        if checkpoint_sha256 != result.best_checkpoint_sha256:
            raise ValueError("Scientific source checkpoint hash differs from training result")
        summary_path = job_root / "summary.json"
        atomic_write_json(
            summary_path,
            {
                "schema_version": 1,
                "run_id": pipeline.run_id,
                "job_id": job["job_id"],
                "job_spec_sha256": job["job_spec_sha256"],
                "model": "hst_base",
                "fold": int(job["fold"]),
                "seed": int(job["seed"]),
                "modality": job["modality"],
                "protocol": job["protocol"],
                "best_epoch": int(result.best_epoch),
                "validation_threshold": float(result.validation_threshold),
                "best_checkpoint_path": checkpoint_path.as_posix(),
                "best_checkpoint_sha256": checkpoint_sha256,
                "training_contract_fingerprint": result.training_contract_fingerprint,
                "held_out_test_opened_after_training_complete": True,
            },
        )
        durable = sorted(
            {
                recording_path,
                participant_path,
                metrics_path,
                history_path,
                summary_path,
                *[path for path in model_root.rglob("*") if path.is_file()],
            },
            key=lambda value: value.as_posix(),
        )
        _write_job_receipt(
            receipt_path,
            {
                "schema_version": 1,
                "status": "success",
                "run_id": pipeline.run_id,
                "job_id": job["job_id"],
                "job_spec_sha256": job["job_spec_sha256"],
                "attempt": attempt,
                "training_complete": True,
                "held_out_test_evaluated_once": True,
                "outputs": _output_records(durable, root=stage_root),
            },
        )
        return {
            "participant_predictions": participant_predictions,
            "metrics": metrics,
            "receipt_path": receipt_path,
            "best_checkpoint_path": checkpoint_path,
            "best_checkpoint_sha256": checkpoint_sha256,
            "validation_threshold": float(result.validation_threshold),
            "training_contract_fingerprint": result.training_contract_fingerprint,
        }
    except Exception as exc:
        current_status = None
        if receipt_path.is_file():
            try:
                current_status = json.loads(receipt_path.read_text(encoding="utf-8")).get(
                    "status"
                )
            except (OSError, json.JSONDecodeError):
                current_status = None
        if current_status != "stopped":
            _write_job_receipt(
                receipt_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "run_id": pipeline.run_id,
                    "job_id": job["job_id"],
                    "job_spec_sha256": job["job_spec_sha256"],
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    finally:
        _cleanup_job_device(model, loaders)


def _run_training_stage(
    pipeline: HSTPipeline,
    *,
    stage: str,
    manifest_names: tuple[str, ...],
    modalities: tuple[str, ...],
    primary_modalities: tuple[str, ...],
    additional_requests: tuple[
        tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...
    ] = (),
) -> Mapping[str, object]:
    bindings = _load_confirmatory_bindings(pipeline)
    enumerated = _enumerate_training_jobs(
        pipeline,
        stage=stage,
        manifest_names=manifest_names,
        modalities=modalities,
        primary_modalities=primary_modalities,
    )
    for extra_manifests, extra_modalities, extra_primary in additional_requests:
        enumerated.extend(
            _enumerate_training_jobs(
                pipeline,
                stage=stage,
                manifest_names=extra_manifests,
                modalities=extra_modalities,
                primary_modalities=extra_primary,
            )
        )
    jobs = [
        _bind_job(job, bindings)
        for job in enumerated
    ]
    stage_root = pipeline.run_root / "scientific" / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    plan_path = stage_root / "job_plan.csv"
    _atomic_csv(pd.DataFrame(jobs), plan_path)
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    output_paths: list[Path] = [plan_path]
    for job in jobs:
        result = _execute_training_job(pipeline, job, bindings)
        predictions = result.get("participant_predictions")
        metrics = result.get("metrics")
        receipt_path = Path(str(result.get("receipt_path", ""))).resolve()
        checkpoint_path = Path(str(result.get("best_checkpoint_path", ""))).resolve()
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            raise ValueError(f"Scientific HST job {job['job_id']} returned no participant predictions")
        if not isinstance(metrics, pd.DataFrame) or metrics.empty:
            raise ValueError(f"Scientific HST job {job['job_id']} returned no metric rows")
        for path in (receipt_path, checkpoint_path):
            if not path.is_file():
                raise FileNotFoundError(f"Scientific HST job output is missing: {path}")
        checkpoint_sha256 = stable_file_sha256(checkpoint_path)
        if checkpoint_sha256 != str(result.get("best_checkpoint_sha256", "")):
            raise ValueError("Scientific HST source checkpoint checksum changed before indexing")
        predictions = predictions.copy()
        predictions["manifest_name"] = str(job["manifest_name"])
        predictions["source_protocol"] = str(job["protocol"])
        predictions["source_manifest_sha256"] = str(job["manifest_sha256"])
        prediction_frames.append(predictions)
        metric_frames.append(metrics)
        output_paths.extend(
            [receipt_path, *_receipt_output_paths(receipt_path, root=stage_root)]
        )
        source_rows.append(
            {
                "training_job_id": job["job_id"],
                "training_job_spec_sha256": job["job_spec_sha256"],
                "fold": int(job["fold"]),
                "seed": int(job["seed"]),
                "modality": job["modality"],
                "protocol": job["protocol"],
                "manifest_name": job["manifest_name"],
                "manifest_sha256": job["manifest_sha256"],
                "source_fold_rows_sha256": job["fold_rows_sha256"],
                "training_rows_sha256": job["training_rows_sha256"],
                "training_contract_fingerprint": result[
                    "training_contract_fingerprint"
                ],
                "best_checkpoint_path": checkpoint_path.as_posix(),
                "best_checkpoint_sha256": checkpoint_sha256,
                "source_job_receipt_path": receipt_path.as_posix(),
                "source_job_receipt_sha256": stable_file_sha256(receipt_path),
                "validation_threshold": float(result["validation_threshold"]),
                "target_data_used_for_training": False,
            }
        )
    combined_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    combined_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    source_checkpoints = pd.DataFrame(source_rows)
    predictions_path = stage_root / "participant_predictions.csv"
    metrics_path = stage_root / "metrics.csv"
    source_path = stage_root / "source_checkpoints.csv"
    _atomic_csv(combined_predictions, predictions_path)
    _atomic_csv(combined_metrics, metrics_path)
    _atomic_csv(source_checkpoints, source_path)
    output_paths.extend([predictions_path, metrics_path, source_path])
    publication_rows: dict[str, int] = {}
    evaluation_splits = {"validation", "test", "temporal_test"}
    for (manifest_name, split), group in combined_predictions.loc[
        combined_predictions["split"].astype(str).isin(evaluation_splits)
    ].groupby(["manifest_name", "split"], sort=True):
        token = str(manifest_name)
        split_token = str(split)
        if not token.replace("_", "").isalnum() or not split_token.replace("_", "").isalnum():
            raise ValueError("Publication prediction partition has an unsafe identity")
        partition_path = (
            stage_root / f"publication_{token}_{split_token}_predictions.csv"
        )
        _atomic_csv(group.reset_index(drop=True), partition_path)
        output_paths.append(partition_path)
        publication_rows[f"{token}:{split_token}"] = len(group)
    return {
        "output_paths": output_paths,
        "row_counts": {
            "jobs": len(jobs),
            "participant_predictions": len(combined_predictions),
            "metrics": len(combined_metrics),
            "source_checkpoints": len(source_checkpoints),
            **publication_rows,
        },
        "metadata": {
            "model": "hst_base",
            "validation_only_model_selection": True,
            "held_out_test_opened_after_training_complete": True,
            "job_plan_sha256": stable_file_sha256(plan_path),
        },
    }


@_scientific_handler("internal_cv")
def _internal_cv(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    experiment = _section(pipeline.config, "experiment")
    profile = workload_profile_from_scientific_config(
        pipeline.config.scientific_config
    )
    primary = tuple(str(value) for value in experiment.get("primary_modalities", ()))
    secondary = tuple(str(value) for value in experiment.get("secondary_modalities", ()))
    if primary != profile.primary_modalities or secondary != profile.secondary_modalities:
        raise ValueError("Frozen Track-A modalities differ from the workload profile")
    additional_requests = (
        ((('task2_like_cough',), ('cough',), ()),)
        if profile.name == FULL_RELIABILITY_PROFILE
        else ()
    )
    return _run_training_stage(
        pipeline,
        stage="internal_cv",
        manifest_names=("internal",),
        modalities=primary + secondary,
        primary_modalities=primary,
        additional_requests=additional_requests,
    )


@_scientific_handler("split_policy_contrast")
def _split_policy_contrast(
    pipeline: HSTPipeline,
    _stage: str,
) -> Mapping[str, object]:
    return _run_training_stage(
        pipeline,
        stage="split_policy_contrast",
        manifest_names=(
            "calendar_mixed",
            "early_to_late",
            "common_late_mixed",
            "common_late_chronological",
        ),
        modalities=("cough", "speech"),
        primary_modalities=("cough", "speech"),
    )


@_scientific_handler("reverse_temporal")
def _reverse_temporal(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    return _run_training_stage(
        pipeline,
        stage="reverse_temporal",
        manifest_names=("reverse_temporal",),
        modalities=("cough", "speech"),
        primary_modalities=(),
    )


def _validate_external_manifest_scope(manifest: pd.DataFrame) -> None:
    required = {
        "dataset",
        "split",
        "modality",
        "analysis_scope",
        "analysis_role",
        "confirmatory_protocol",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"External HST manifest is missing scope columns: {missing}")
    coughvid = manifest.loc[manifest["dataset"].astype(str).eq("coughvid")]
    if coughvid.empty or not coughvid["split"].astype(str).eq("external_test").all():
        raise ValueError("Every COUGHVID row must remain external_test only")
    if not coughvid["modality"].astype(str).eq("cough").all():
        raise ValueError("COUGHVID external transfer is cough-only")
    source = manifest.loc[~manifest["dataset"].astype(str).eq("coughvid")]
    if source.empty or not source["dataset"].astype(str).eq("coswara").all():
        raise ValueError("External source rows must belong only to Coswara")
    if not source["split"].astype(str).isin(["train", "validation", "test"]).all():
        raise ValueError("External source rows have an unsupported split")
    for column, expected in (
        ("analysis_scope", "reliability_evaluation"),
        ("analysis_role", "secondary"),
    ):
        if not manifest[column].astype(str).eq(expected).all():
            raise ValueError(f"External HST manifest must keep {column}={expected!r}")
    confirmatory = manifest["confirmatory_protocol"].map(
        lambda value: _explicit_bool(value, field_name="confirmatory_protocol")
    )
    if not confirmatory.all():
        raise ValueError("External HST manifest must retain its prespecified confirmatory flag")


def _external_jobs(pipeline: HSTPipeline) -> list[dict[str, object]]:
    manifest_path, manifest, manifest_sha256 = _load_indexed_manifest(
        pipeline, "external"
    )
    source_manifest_path, source_manifest, source_manifest_sha256 = (
        _load_indexed_manifest(pipeline, "internal")
    )
    _validate_external_source_binding(
        manifest,
        source_manifest,
        internal_manifest_sha256=source_manifest_sha256,
    )
    _validate_external_manifest_scope(manifest)
    target = manifest.loc[
        manifest["dataset"].astype(str).eq("coughvid")
        & manifest["split"].astype(str).eq("external_test")
        & manifest["modality"].astype(str).eq("cough")
    ]
    if target.empty:
        raise ValueError("External manifest contains no COUGHVID cough target rows")
    source_index_path = (
        pipeline.run_root / "scientific" / "internal_cv" / "source_checkpoints.csv"
    )
    if not source_index_path.is_file():
        raise FileNotFoundError("Internal HST source-checkpoint index is missing")
    source_index = pd.read_csv(source_index_path, low_memory=False)
    required_source_columns = {
        "training_job_id",
        "training_job_spec_sha256",
        "fold",
        "seed",
        "modality",
        "protocol",
        "manifest_name",
        "manifest_sha256",
        "source_fold_rows_sha256",
        "training_rows_sha256",
        "training_contract_fingerprint",
        "best_checkpoint_path",
        "best_checkpoint_sha256",
        "source_job_receipt_path",
        "source_job_receipt_sha256",
        "validation_threshold",
    }
    missing_source_columns = sorted(required_source_columns - set(source_index.columns))
    if missing_source_columns:
        raise ValueError(
            "Internal HST source-checkpoint index is missing columns: "
            f"{missing_source_columns}"
        )
    source_index = source_index.loc[
        source_index["modality"].astype(str).eq("cough")
        & source_index["manifest_name"].astype(str).eq("internal")
        & source_index["protocol"].astype(str).eq(
            "hst_literature_aligned_repeated_holdout"
        )
    ].copy()
    if len(source_index) != 10 or set(pd.to_numeric(source_index["fold"], errors="raise").astype(int)) != set(
        range(1, 11)
    ):
        raise ValueError("External transfer requires exactly ten internal cough checkpoints")
    jobs: list[dict[str, object]] = []
    for source in source_index.sort_values("fold", kind="mergesort").to_dict(orient="records"):
        fold = int(source["fold"])
        seed = int(source["seed"])
        expected_seed = _TRACK_A_SEEDS[fold - 1]
        if seed != expected_seed:
            raise ValueError("External source checkpoint fold/seed mapping is invalid")
        checkpoint_path = Path(str(source["best_checkpoint_path"])).resolve()
        receipt_path = Path(str(source["source_job_receipt_path"])).resolve()
        for candidate, root in (
            (checkpoint_path, pipeline.run_root),
            (receipt_path, pipeline.run_root / "scientific" / "internal_cv"),
        ):
            try:
                candidate.relative_to(Path(root).resolve())
            except ValueError as exc:
                raise ValueError("External source artifact escaped the trusted internal run") from exc
            if not candidate.is_file():
                raise FileNotFoundError(f"External source artifact is missing: {candidate}")
        if stable_file_sha256(checkpoint_path) != str(source["best_checkpoint_sha256"]):
            raise ValueError("External source checkpoint checksum changed")
        if stable_file_sha256(receipt_path) != str(source["source_job_receipt_sha256"]):
            raise ValueError("External source job receipt checksum changed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "success":
            raise ValueError("External source job did not complete successfully")
        if (
            str(receipt.get("job_id", "")) != str(source["training_job_id"])
            or str(receipt.get("job_spec_sha256", ""))
            != str(source["training_job_spec_sha256"])
        ):
            raise ValueError("External source job receipt does not match its indexed training job")
        if str(source["manifest_sha256"]) != source_manifest_sha256:
            raise ValueError("External source checkpoint was not trained from frozen internal Track-A")
        source_fold = source_manifest.loc[
            pd.to_numeric(source_manifest["fold"], errors="raise").astype(int).eq(fold)
            & source_manifest["modality"].astype(str).eq("cough")
        ]
        source_training = source_fold.loc[
            source_fold["split"].astype(str).isin(["train", "validation"])
        ]
        if (
            _external_source_rows_sha256(source_fold)
            != str(source["source_fold_rows_sha256"])
            or _job_manifest_rows_sha256(source_training)
            != str(source["training_rows_sha256"])
        ):
            raise ValueError("External source checkpoint row identity changed after internal training")
        training_contract_fingerprint = str(source["training_contract_fingerprint"])
        if len(training_contract_fingerprint) != 64:
            raise ValueError("External source checkpoint lacks a training contract fingerprint")
        fold_target = target.loc[
            pd.to_numeric(target["fold"], errors="raise").astype(int).eq(fold)
        ]
        if fold_target.empty:
            raise ValueError(f"External manifest has no target rows for fold {fold}")
        payload = {
            "schema_version": 1,
            "stage": "external_transfer",
            "manifest_name": "external",
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": manifest_sha256,
            "cache_index_path": (
                pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
            ).as_posix(),
            "cache_index_sha256": stable_file_sha256(
                pipeline.run_root / "manifests" / "spectrogram_cache_index.csv"
            ),
            "protocol": str(fold_target["protocol"].iloc[0]),
            "source_protocol": str(source["protocol"]),
            "fold": fold,
            "seed": seed,
            "modality": "cough",
            "model_name": "hst_base",
            "representation_id": "paper_logmel_224",
            "source_training_job_id": source["training_job_id"],
            "source_training_job_spec_sha256": source.get(
                "training_job_spec_sha256", ""
            ),
            "source_manifest_path": source_manifest_path.as_posix(),
            "source_manifest_sha256": source_manifest_sha256,
            "source_fold_rows_sha256": source["source_fold_rows_sha256"],
            "source_training_rows_sha256": source["training_rows_sha256"],
            "source_training_contract_fingerprint": training_contract_fingerprint,
            "source_checkpoint_path": checkpoint_path.as_posix(),
            "source_checkpoint_sha256": source["best_checkpoint_sha256"],
            "source_job_receipt_path": receipt_path.as_posix(),
            "source_job_receipt_sha256": source["source_job_receipt_sha256"],
            "source_validation_threshold": float(source["validation_threshold"]),
            "target_rows_sha256": _job_manifest_rows_sha256(fold_target),
            "target_fit": False,
            "target_selection": False,
            "analysis_queue": "secondary",
            "analysis_role": "secondary",
            "analysis_scope": "reliability_evaluation",
            "estimand_id": str(fold_target["estimand_id"].iloc[0]),
            "multiplicity_family": str(fold_target["multiplicity_family"].iloc[0]),
        }
        digest = canonical_json_sha256(payload)
        jobs.append(
            {
                **payload,
                "job_spec_sha256": digest,
                "job_id": f"external-transfer-{digest[:20]}",
            }
        )
    return jobs


def _validate_external_checkpoint_binding(
    payload: Mapping[str, object],
    job: Mapping[str, object],
) -> None:
    execution_identity = payload.get("execution_identity")
    if not isinstance(execution_identity, Mapping):
        raise ValueError("External source checkpoint lacks training execution identity")
    if (
        int(execution_identity.get("fold", -1)) != int(job["fold"])
        or str(execution_identity.get("modality", "")) != "cough"
        or int(execution_identity.get("model_seed", -1)) != int(job["seed"])
    ):
        raise ValueError("External source checkpoint identity does not match its internal fold")
    if str(payload.get("training_contract_fingerprint", "")) != str(
        job.get("source_training_contract_fingerprint", "")
    ):
        raise ValueError(
            "External source checkpoint training contract does not match its internal job"
        )


def _build_raw_status_external_sensitivity(
    primary_predictions: pd.DataFrame,
    raw_status_contract: pd.DataFrame,
    *,
    jobs: list[Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-score frozen external probabilities against the predeclared raw label."""
    required_predictions = {
        "fold",
        "dataset",
        "participant_key",
        "split",
        "modality",
        "label_binary",
        "probability",
    }
    missing_predictions = sorted(required_predictions - set(primary_predictions.columns))
    if missing_predictions:
        raise ValueError(
            f"Raw-status sensitivity predictions are missing columns: {missing_predictions}"
        )
    if (
        primary_predictions.empty
        or not primary_predictions["dataset"].astype(str).eq("coughvid").all()
        or not primary_predictions["split"].astype(str).eq("external_test").all()
        or not primary_predictions["modality"].astype(str).eq("cough").all()
    ):
        raise ValueError("Raw-status sensitivity requires COUGHVID external cough predictions")
    if primary_predictions.duplicated(["fold", "participant_key"]).any():
        raise ValueError("Raw-status sensitivity predictions duplicate a fold participant")
    if primary_predictions.groupby("participant_key")["label_binary"].nunique().ne(1).any():
        raise ValueError("Primary external labels vary across repeated folds")

    job_by_fold: dict[int, Mapping[str, object]] = {}
    for job in jobs:
        fold = int(job["fold"])
        if fold in job_by_fold:
            raise ValueError("Raw-status sensitivity received duplicate source jobs")
        job_by_fold[fold] = job
    if not job_by_fold:
        raise ValueError("Raw-status sensitivity received no source jobs")
    primary_folds = set(
        pd.to_numeric(primary_predictions["fold"], errors="raise").astype(int)
    )
    if primary_folds != set(job_by_fold):
        raise ValueError("Raw-status sensitivity folds do not match source checkpoints")

    required_contract = {
        "participant_key",
        "label_binary",
        "contract_eligible",
        "label_source",
        "label_provenance",
    }
    missing_contract = sorted(required_contract - set(raw_status_contract.columns))
    if missing_contract:
        raise ValueError(
            f"Raw-status sensitivity contract is missing columns: {missing_contract}"
        )
    eligible = raw_status_contract["contract_eligible"].map(
        {True: True, False: False, "True": True, "False": False}
    )
    if eligible.isna().any():
        raise ValueError("Raw-status contract eligibility is not boolean")
    raw = raw_status_contract.loc[eligible.astype(bool)].copy()
    if not raw.empty and not raw["label_binary"].isin(["negative", "positive"]).all():
        raise ValueError("Raw-status sensitivity contains a non-binary supervised label")
    if not raw.empty and raw.groupby("participant_key")["label_binary"].nunique().ne(1).any():
        raise ValueError("Raw-status contract has conflicting participant labels")
    raw_participants = raw[
        ["participant_key", "label_binary", "label_source", "label_provenance"]
    ].drop_duplicates()
    if raw_participants["participant_key"].duplicated().any():
        raise ValueError("Raw-status participant provenance is not invariant")
    if not raw_participants.empty and set(
        raw_participants["label_source"].astype(str)
    ) != {"status"}:
        raise ValueError("Raw-status sensitivity must use only the status column")

    relabeled = primary_predictions.rename(
        columns={"label_binary": "primary_label_binary"}
    ).merge(
        raw_participants,
        on="participant_key",
        how="inner",
        validate="many_to_one",
    )
    relabeled["protocol"] = (
        "coswara_to_coughvid_hst_external_raw_status_sensitivity"
    )
    relabeled["analysis_scope"] = "sensitivity_analysis"
    relabeled["analysis_role"] = "sensitivity"
    relabeled["confirmatory_protocol"] = False
    relabeled["estimand_id"] = "coswara_to_coughvid_external_raw_status_sensitivity"
    relabeled["multiplicity_family"] = "exploratory_label_sensitivity"
    relabeled["target_fit"] = False
    relabeled["target_selection"] = False

    observed_folds = (
        set(pd.to_numeric(relabeled["fold"], errors="raise").astype(int))
        if not relabeled.empty
        else set()
    )
    if observed_folds and observed_folds != set(job_by_fold):
        raise ValueError("Raw-status sensitivity folds do not match source checkpoints")
    metric_frames: list[pd.DataFrame] = []
    unavailable_reason = ""
    if raw.empty:
        unavailable_reason = "raw_status_has_no_supervised_participants"
    elif relabeled.empty:
        unavailable_reason = "raw_status_has_no_aligned_frozen_predictions"
    for fold in sorted(job_by_fold):
        source_job = job_by_fold[fold]
        group = relabeled.loc[
            pd.to_numeric(relabeled["fold"], errors="raise").astype(int).eq(fold)
        ].copy()
        sensitivity_job = {
            "stage": "external_transfer",
            "job_id": f"raw-status-sensitivity-fold-{fold}",
            "protocol": "coswara_to_coughvid_hst_external_raw_status_sensitivity",
            "fold": fold,
            "seed": int(source_job["seed"]),
            "modality": "cough",
            "analysis_queue": "sensitivity",
            "analysis_role": "sensitivity",
            "analysis_scope": "sensitivity_analysis",
            "estimand_id": "coswara_to_coughvid_external_raw_status_sensitivity",
            "multiplicity_family": "exploratory_label_sensitivity",
        }
        skip_reason = unavailable_reason
        if not skip_reason and group["label_binary"].nunique() != 2:
            skip_reason = "raw_status_aligned_labels_do_not_contain_both_classes"
        if skip_reason:
            metric_frames.append(
                pd.DataFrame(
                    [
                        {
                            **{
                                name: float("nan")
                                for name in (
                                    "auroc",
                                    "auprc",
                                    "balanced_accuracy",
                                    "f1",
                                    "sensitivity",
                                    "specificity",
                                    "brier",
                                    "ece",
                                    "nll",
                                )
                            },
                            "threshold": float(source_job["source_validation_threshold"]),
                            "n_samples": float(len(group)),
                            "stage": sensitivity_job["stage"],
                            "job_id": sensitivity_job["job_id"],
                            "protocol": sensitivity_job["protocol"],
                            "fold": fold,
                            "seed": int(source_job["seed"]),
                            "modality": "cough",
                            "model": "hst_base",
                            "model_name": "hst_base",
                            "metric_split": "external_test",
                            "threshold_source": "source_validation_balanced_accuracy",
                            "n_participants": int(len(group)),
                            "analysis_queue": "sensitivity",
                            "analysis_role": "sensitivity",
                            "analysis_scope": "sensitivity_analysis",
                            "estimand_id": sensitivity_job["estimand_id"],
                            "multiplicity_family": sensitivity_job["multiplicity_family"],
                            "label_source": "status",
                            "label_provenance": "raw_self_report",
                            "target_fit": False,
                            "target_selection": False,
                            "confirmatory_protocol": False,
                            "skipped": True,
                            "skip_reason": skip_reason,
                        }
                    ]
                )
            )
            continue
        metrics = _participant_metric_rows(
            group,
            job=sensitivity_job,
            threshold=float(source_job["source_validation_threshold"]),
            threshold_source="source_validation_balanced_accuracy",
        )
        metrics["label_source"] = "status"
        metrics["label_provenance"] = "raw_self_report"
        metrics["target_fit"] = False
        metrics["target_selection"] = False
        metrics["confirmatory_protocol"] = False
        metrics["skipped"] = False
        metrics["skip_reason"] = ""
        metric_frames.append(metrics)

    primary_labels = primary_predictions[
        ["participant_key", "label_binary"]
    ].drop_duplicates()
    overlap = primary_labels.merge(
        raw_participants[["participant_key", "label_binary"]],
        on="participant_key",
        how="inner",
        suffixes=("_primary", "_raw_status"),
        validate="one_to_one",
    )
    combined_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    skip_reasons = sorted(
        {
            str(value)
            for value in combined_metrics.loc[
                combined_metrics["skipped"].astype(bool), "skip_reason"
            ]
            if str(value)
        }
    )
    audit = pd.DataFrame(
        [
            {
                "sensitivity_id": "coughvid_raw_status_label",
                "primary_external_participants": int(
                    primary_predictions["participant_key"].nunique()
                ),
                "raw_status_eligible_participants": int(
                    raw_participants["participant_key"].nunique()
                ),
                "aligned_sensitivity_participants": int(
                    relabeled["participant_key"].nunique()
                ),
                "label_disagreement_participants": int(
                    (
                        overlap["label_binary_primary"]
                        != overlap["label_binary_raw_status"]
                    ).sum()
                ),
                "external_labels_used_for_model_decisions": False,
                "threshold_reused_from_source_validation": True,
                "primary_blocking": False,
                "skipped": bool(combined_metrics["skipped"].astype(bool).all()),
                "partially_skipped": bool(combined_metrics["skipped"].astype(bool).any()),
                "skip_reason": ";".join(skip_reasons),
            }
        ]
    )
    return (
        relabeled.reset_index(drop=True),
        combined_metrics,
        audit,
    )


def _execute_external_job(
    pipeline: HSTPipeline,
    job: dict[str, object],
    bindings: Mapping[str, object],
) -> dict[str, object]:
    stage_root = pipeline.run_root / "scientific" / "external_transfer"
    job_root = stage_root / "jobs" / str(job["job_id"])
    receipt_path = job_root / "job_receipt.json"
    if _validated_reusable_job(
        receipt_path,
        job_spec_sha256=str(job["job_spec_sha256"]),
        run_id=pipeline.run_id,
        job_id=str(job["job_id"]),
        root=stage_root,
    ):
        predictions_path = job_root / "participant_predictions.csv"
        metrics_path = job_root / "metrics.csv"
        return {
            "participant_predictions": pd.read_csv(predictions_path, low_memory=False),
            "metrics": pd.read_csv(metrics_path, low_memory=False),
            "receipt_path": receipt_path,
        }
    previous_attempt = 0
    if receipt_path.is_file():
        try:
            previous_attempt = int(
                json.loads(receipt_path.read_text(encoding="utf-8")).get("attempt", 0)
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Prior external job receipt is corrupt: {receipt_path}") from exc
    attempt = previous_attempt + 1
    _write_job_receipt(
        receipt_path,
        {
            "schema_version": 1,
            "status": "running",
            "run_id": pipeline.run_id,
            "job_id": job["job_id"],
            "job_spec_sha256": job["job_spec_sha256"],
            "attempt": attempt,
            "target_fit": False,
            "target_selection": False,
        },
    )
    model: object | None = None
    loaders: object | None = None
    try:
        if job.get("target_fit") is not False or job.get("target_selection") is not False:
            raise ValueError("External HST evaluation cannot fit or select on target data")
        for field_name in ("manifest", "cache_index"):
            artifact_path = Path(str(job[f"{field_name}_path"])).resolve()
            if not artifact_path.is_file():
                raise FileNotFoundError(f"External {field_name} artifact is missing")
            if stable_file_sha256(artifact_path) != str(job[f"{field_name}_sha256"]):
                raise ValueError(f"External {field_name} checksum changed after job planning")
        source_manifest_path = Path(str(job["source_manifest_path"])).resolve()
        source_receipt_path = Path(str(job["source_job_receipt_path"])).resolve()
        for artifact_path, expected_hash, trusted_root in (
            (
                source_manifest_path,
                str(job["source_manifest_sha256"]),
                pipeline.run_root / "manifests",
            ),
            (
                source_receipt_path,
                str(job["source_job_receipt_sha256"]),
                pipeline.run_root / "scientific" / "internal_cv",
            ),
        ):
            try:
                artifact_path.relative_to(Path(trusted_root).resolve())
            except ValueError as exc:
                raise ValueError("External source binding escaped the trusted internal run") from exc
            if not artifact_path.is_file() or stable_file_sha256(artifact_path) != expected_hash:
                raise ValueError("External source binding changed after job planning")
        source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
        if (
            source_receipt.get("status") != "success"
            or str(source_receipt.get("job_id", ""))
            != str(job["source_training_job_id"])
            or str(source_receipt.get("job_spec_sha256", ""))
            != str(job["source_training_job_spec_sha256"])
        ):
            raise ValueError("External source receipt no longer matches the internal training job")
        source_manifest = pd.read_csv(source_manifest_path, low_memory=False)
        source_fold = source_manifest.loc[
            pd.to_numeric(source_manifest["fold"], errors="raise")
            .astype(int)
            .eq(int(job["fold"]))
            & source_manifest["modality"].astype(str).eq("cough")
        ]
        source_training = source_fold.loc[
            source_fold["split"].astype(str).isin(["train", "validation"])
        ]
        if (
            _external_source_rows_sha256(source_fold)
            != str(job["source_fold_rows_sha256"])
            or _job_manifest_rows_sha256(source_training)
            != str(job["source_training_rows_sha256"])
        ):
            raise ValueError("External source manifest rows changed after checkpoint selection")
        executable_sha256 = verify_executable_allowlist(
            executable_root=pipeline.config.source_root,
            executable_paths=bindings["executable_paths"],  # type: ignore[arg-type]
            frozen_allowlist=bindings["executable_allowlist"],  # type: ignore[arg-type]
        )
        if executable_sha256 != str(bindings["executable_sha256"]):
            raise ValueError("External executable allow-list changed after job planning")
        source_checkpoint = Path(str(job["source_checkpoint_path"])).resolve()
        if stable_file_sha256(source_checkpoint) != str(job["source_checkpoint_sha256"]):
            raise ValueError("External inference source checkpoint checksum changed")
        model, _initial_audit = load_verified_hst_model(
            model_name="hst_base",
            checkpoint_path=Path(str(bindings["source_checkpoint_path"])),
            hst_repo=_source_path(pipeline.config),
            seed=int(job["seed"]),
        )
        payload, verified_checkpoint_path = _load_verified_checkpoint_with_path(
            source_checkpoint
        )
        if verified_checkpoint_path != source_checkpoint:
            raise ValueError("External source checkpoint resolved to unexpected bytes")
        if payload.get("checkpoint_role") != "best":
            raise ValueError("External inference requires the validation-selected best checkpoint")
        source_prediction_context = payload.get("prediction_context")
        if not isinstance(source_prediction_context, Mapping):
            raise ValueError("External source checkpoint lacks frozen prediction provenance")
        if (
            str(source_prediction_context.get("checkpoint_hash", ""))
            != str(bindings["source_checkpoint_sha256"])
        ):
            raise ValueError("External source model was not initialized from the frozen HST-Base checkpoint")
        if (
            str(source_prediction_context.get("executable_sha256", ""))
            != str(bindings["executable_sha256"])
            or str(payload.get("executable_sha256", ""))
            != str(bindings["executable_sha256"])
        ):
            raise ValueError("External source checkpoint executable provenance changed")
        _validate_external_checkpoint_binding(payload, job)
        model.load_state_dict(payload["model_state_dict"])  # type: ignore[attr-defined]
        model = model.to(pipeline.config.device)  # type: ignore[attr-defined]
        if str(payload.get("architecture_sha256", "")) != _model_architecture_sha256(model):
            raise ValueError("External source checkpoint architecture provenance changed")
        manifest_path = Path(str(job["manifest_path"])).resolve()
        cache_path = Path(str(job["cache_index_path"])).resolve()
        manifest = pd.read_csv(manifest_path, low_memory=False)
        cache = pd.read_csv(cache_path, low_memory=False)
        loaders = make_hst_dataloaders(
            cache,
            manifest,
            fold=int(job["fold"]),
            modality="cough",
            physical_batch_size=int(bindings["physical_batch_size"]),
            num_workers=0,
            seed=int(job["seed"]),
            representation_id="paper_logmel_224",
        )
        context = {
            "run_id": pipeline.run_id,
            "protocol": str(job["protocol"]),
            "model": "hst_base",
            "checkpoint_hash": str(job["source_checkpoint_sha256"]),
            "representation": "paper_logmel_224",
            "architecture_sha256": _model_architecture_sha256(model),
            "executable_sha256": str(bindings["executable_sha256"]),
        }
        recording_predictions = predict_hst_split(
            model,
            loaders["external_test"],  # type: ignore[index]
            split="external_test",
            fold=int(job["fold"]),
            modality="cough",
            prediction_context=context,
        )
        participant_predictions = aggregate_recording_predictions(recording_predictions)
        if participant_predictions.empty:
            raise ValueError("External HST job produced no participant predictions")
        if not participant_predictions["dataset"].astype(str).eq("coughvid").all():
            raise ValueError("External HST predictions include non-COUGHVID participants")
        if not participant_predictions["split"].astype(str).eq("external_test").all():
            raise ValueError("External HST predictions include a non-external split")
        participant_predictions["job_id"] = job["job_id"]
        participant_predictions["seed"] = int(job["seed"])
        participant_predictions["source_training_job_id"] = job[
            "source_training_job_id"
        ]
        participant_predictions["source_protocol"] = str(job["source_protocol"])
        participant_predictions["source_manifest_sha256"] = str(
            job["source_manifest_sha256"]
        )
        threshold = float(job["source_validation_threshold"])
        metrics = _participant_metric_rows(
            participant_predictions,
            job=job,
            threshold=threshold,
            threshold_source="source_validation_balanced_accuracy",
        )
        recording_path = job_root / "recording_predictions.csv"
        participant_path = job_root / "participant_predictions.csv"
        metrics_path = job_root / "metrics.csv"
        for frame, path in (
            (recording_predictions, recording_path),
            (participant_predictions, participant_path),
            (metrics, metrics_path),
        ):
            _atomic_csv(frame, path)
        summary_path = job_root / "summary.json"
        atomic_write_json(
            summary_path,
            {
                "schema_version": 1,
                "run_id": pipeline.run_id,
                "job_id": job["job_id"],
                "job_spec_sha256": job["job_spec_sha256"],
                "source_training_job_id": job["source_training_job_id"],
                "source_checkpoint_path": source_checkpoint.as_posix(),
                "source_checkpoint_sha256": job["source_checkpoint_sha256"],
                "source_manifest_sha256": job["source_manifest_sha256"],
                "source_fold_rows_sha256": job["source_fold_rows_sha256"],
                "source_training_rows_sha256": job["source_training_rows_sha256"],
                "source_training_contract_fingerprint": job[
                    "source_training_contract_fingerprint"
                ],
                "source_validation_threshold": threshold,
                "target_fit": False,
                "target_selection": False,
                "target_labels_used_after_prediction_only": True,
            },
        )
        durable = [recording_path, participant_path, metrics_path, summary_path]
        _write_job_receipt(
            receipt_path,
            {
                "schema_version": 1,
                "status": "success",
                "run_id": pipeline.run_id,
                "job_id": job["job_id"],
                "job_spec_sha256": job["job_spec_sha256"],
                "attempt": attempt,
                "source_checkpoint_sha256": job["source_checkpoint_sha256"],
                "source_manifest_sha256": job["source_manifest_sha256"],
                "source_training_contract_fingerprint": job[
                    "source_training_contract_fingerprint"
                ],
                "target_fit": False,
                "target_selection": False,
                "outputs": _output_records(durable, root=stage_root),
            },
        )
        return {
            "participant_predictions": participant_predictions,
            "metrics": metrics,
            "receipt_path": receipt_path,
        }
    except Exception as exc:
        _write_job_receipt(
            receipt_path,
            {
                "schema_version": 1,
                "status": "failed",
                "run_id": pipeline.run_id,
                "job_id": job["job_id"],
                "job_spec_sha256": job["job_spec_sha256"],
                "attempt": attempt,
                "target_fit": False,
                "target_selection": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        _cleanup_job_device(model, loaders)


@_scientific_handler("external_transfer")
def _external_transfer(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    bindings = _load_confirmatory_bindings(pipeline)
    jobs = [_bind_job(job, bindings) for job in _external_jobs(pipeline)]
    stage_root = pipeline.run_root / "scientific" / "external_transfer"
    stage_root.mkdir(parents=True, exist_ok=True)
    plan_path = stage_root / "job_plan.csv"
    _atomic_csv(pd.DataFrame(jobs), plan_path)
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    output_paths: list[Path] = [plan_path]
    for job in jobs:
        result = _execute_external_job(pipeline, job, bindings)
        predictions = result.get("participant_predictions")
        metrics = result.get("metrics")
        receipt_path = Path(str(result.get("receipt_path", ""))).resolve()
        if not isinstance(predictions, pd.DataFrame) or predictions.empty:
            raise ValueError(f"External HST job {job['job_id']} returned no participant predictions")
        if not isinstance(metrics, pd.DataFrame) or metrics.empty:
            raise ValueError(f"External HST job {job['job_id']} returned no metric rows")
        if not receipt_path.is_file():
            raise FileNotFoundError(f"External HST job receipt is missing: {receipt_path}")
        prediction_frames.append(predictions)
        metric_frames.append(metrics)
        output_paths.extend(
            [receipt_path, *_receipt_output_paths(receipt_path, root=stage_root)]
        )
    combined_predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    combined_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    if not combined_predictions["dataset"].astype(str).eq("coughvid").all():
        raise ValueError("External stage output contains source-dataset predictions")
    predictions_path = stage_root / "participant_predictions.csv"
    metrics_path = stage_root / "metrics.csv"
    _atomic_csv(combined_predictions, predictions_path)
    _atomic_csv(combined_metrics, metrics_path)
    output_paths.extend([predictions_path, metrics_path])
    sensitivity_rows: dict[str, int] = {}
    coughvid_config = _section(pipeline.config, "datasets").get("coughvid", {})
    if not isinstance(coughvid_config, Mapping):
        raise ValueError("COUGHVID dataset configuration must be an object")
    raw_status_config = coughvid_config.get("raw_status_sensitivity")
    if raw_status_config is not None:
        if not isinstance(raw_status_config, Mapping) or str(
            raw_status_config.get("execution", "")
        ) != "relabel_frozen_external_predictions":
            raise ValueError("Raw-status sensitivity execution contract is invalid")
        raw_status_path = (
            pipeline.run_root / "contracts" / "coughvid_raw_status_sensitivity.csv"
        )
        if not raw_status_path.is_file():
            raise FileNotFoundError("Audited raw-status sensitivity contract is missing")
        raw_predictions, raw_metrics, raw_audit = (
            _build_raw_status_external_sensitivity(
                combined_predictions,
                pd.read_csv(raw_status_path, low_memory=False),
                jobs=jobs,
            )
        )
        for name, frame in (
            ("raw_status_sensitivity_predictions.csv", raw_predictions),
            ("raw_status_sensitivity_metrics.csv", raw_metrics),
            ("raw_status_sensitivity_audit.csv", raw_audit),
        ):
            path = stage_root / name
            _atomic_csv(frame, path)
            output_paths.append(path)
            sensitivity_rows[name.removesuffix(".csv")] = len(frame)
    _internal_receipt_path, internal_receipt, _internal_receipt_sha = (
        _verified_stage_receipt(pipeline, "internal_cv")
    )
    source_partition_rows: dict[str, int] = {}
    for split in ("validation", "test"):
        source_path = _receipt_output_file(
            pipeline,
            internal_receipt,
            f"/publication_internal_{split}_predictions.csv",
        )
        source = pd.read_csv(source_path, low_memory=False)
        source = source.loc[source["modality"].astype(str).eq("cough")].copy()
        if source.empty or set(pd.to_numeric(source["fold"], errors="raise").astype(int)) != set(
            range(1, 11)
        ):
            raise ValueError(
                f"External publication source {split} lacks all ten frozen cough folds"
            )
        export_path = stage_root / f"source_{split}_predictions.csv"
        _atomic_csv(source.reset_index(drop=True), export_path)
        output_paths.append(export_path)
        source_partition_rows[f"source_{split}_predictions"] = len(source)
    return {
        "output_paths": output_paths,
        "row_counts": {
            "inference_jobs": len(jobs),
            "participant_predictions": len(combined_predictions),
            "metrics": len(combined_metrics),
            **sensitivity_rows,
            **source_partition_rows,
        },
        "metadata": {
            "model": "hst_base",
            "source_checkpoints_reused": len(jobs),
            "target_fit": False,
            "target_selection": False,
            "external_labels_used_for_model_decisions": False,
            "raw_status_sensitivity_reuses_frozen_probabilities": bool(
                raw_status_config is not None
            ),
            "job_plan_sha256": stable_file_sha256(plan_path),
        },
    }


def _verified_stage_receipt(
    pipeline: HSTPipeline,
    stage: str,
) -> tuple[Path, dict[str, object], str]:
    """Load one exact upstream receipt and revalidate every receipted byte."""
    path = pipeline.run_root / "runtime" / "stages" / f"{stage}.json"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Authenticated upstream stage receipt is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Authenticated upstream stage receipt is corrupt: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Authenticated upstream stage receipt is not an object: {path}")
    if (
        payload.get("receipt_type") != "hst_stage"
        or payload.get("status") != "success"
        or payload.get("stage") != stage
        or payload.get("run_id") != pipeline.run_id
    ):
        raise ValueError(f"Authenticated upstream stage receipt identity is invalid: {stage}")
    claimed = str(payload.get("record_hash", ""))
    unsigned = {key: value for key, value in payload.items() if key != "record_hash"}
    if claimed != canonical_json_sha256(unsigned):
        raise ValueError(f"Authenticated upstream stage receipt hash changed: {stage}")
    outputs = payload.get("output_paths")
    checksums = payload.get("output_checksums")
    if not isinstance(outputs, list) or not outputs or not isinstance(checksums, Mapping):
        raise ValueError(f"Authenticated upstream stage has no output contract: {stage}")
    run_root = pipeline.run_root.resolve()
    for relative_value in outputs:
        relative = Path(str(relative_value)).as_posix()
        candidate = (run_root / relative).resolve()
        try:
            candidate.relative_to(run_root)
        except ValueError as exc:
            raise ValueError(f"Authenticated upstream output escapes run root: {relative}") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(f"Authenticated upstream output is missing: {candidate}")
        if stable_file_sha256(candidate) != str(checksums.get(relative, "")):
            raise ValueError(f"Authenticated upstream output checksum changed: {relative}")
    return path, payload, stable_file_sha256(path)


def _receipt_output_file(
    pipeline: HSTPipeline,
    receipt: Mapping[str, object],
    relative_suffix: str,
) -> Path:
    matches = [
        (pipeline.run_root / str(relative)).resolve()
        for relative in receipt.get("output_paths", [])  # type: ignore[arg-type]
        if Path(str(relative)).as_posix().endswith(relative_suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one authenticated output ending {relative_suffix!r}; found {len(matches)}"
        )
    return matches[0]


def _receipt_output_files(
    pipeline: HSTPipeline,
    receipt: Mapping[str, object],
    relative_suffix: str,
) -> list[Path]:
    paths = sorted(
        {
            (pipeline.run_root / str(relative)).resolve()
            for relative in receipt.get("output_paths", [])  # type: ignore[arg-type]
            if Path(str(relative)).as_posix().endswith(relative_suffix)
        },
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise ValueError(f"No authenticated output ends with {relative_suffix!r}")
    return paths


def _internal_track_a_recording_prediction_paths(
    pipeline: HSTPipeline,
    internal_stage_receipt: Mapping[str, object],
) -> list[Path]:
    """Resolve only receipted Track-A recording predictions, excluding Task-2 jobs."""
    source_index_path = _receipt_output_file(
        pipeline, internal_stage_receipt, "/source_checkpoints.csv"
    )
    source_index = pd.read_csv(source_index_path, low_memory=False)
    required = {
        "training_job_id",
        "training_job_spec_sha256",
        "manifest_name",
        "source_job_receipt_path",
        "source_job_receipt_sha256",
    }
    missing = sorted(required - set(source_index.columns))
    if missing:
        raise ValueError(f"Internal source-checkpoint index misses columns: {missing}")
    selected = source_index.loc[
        source_index["manifest_name"].astype(str).eq("internal")
    ].copy()
    if selected.empty or selected["training_job_id"].astype(str).duplicated().any():
        raise ValueError("Fusion requires unique authenticated internal Track-A jobs")
    allowed_outputs = {
        (pipeline.run_root / str(relative)).resolve()
        for relative in internal_stage_receipt.get("output_paths", [])  # type: ignore[arg-type]
    }
    stage_root = (pipeline.run_root / "scientific" / "internal_cv").resolve()
    recording_paths: list[Path] = []
    for row in selected.sort_values("training_job_id", kind="mergesort").to_dict(
        orient="records"
    ):
        receipt_path = Path(str(row["source_job_receipt_path"])).resolve()
        try:
            receipt_path.relative_to(stage_root)
        except ValueError as exc:
            raise ValueError("Internal Track-A job receipt escaped its stage root") from exc
        if receipt_path not in allowed_outputs:
            raise ValueError("Internal Track-A job receipt is not a receipted stage output")
        if (
            not receipt_path.is_file()
            or receipt_path.is_symlink()
            or stable_file_sha256(receipt_path)
            != str(row["source_job_receipt_sha256"])
        ):
            raise ValueError("Internal Track-A job receipt checksum changed")
        if not _validated_reusable_job(
            receipt_path,
            job_spec_sha256=str(row["training_job_spec_sha256"]),
            run_id=pipeline.run_id,
            job_id=str(row["training_job_id"]),
            root=stage_root,
        ):
            raise ValueError("Internal Track-A job is not a reusable successful job")
        candidates = [
            path
            for path in _receipt_output_paths(receipt_path, root=stage_root)
            if path.name == "recording_predictions.csv"
        ]
        if len(candidates) != 1 or candidates[0] not in allowed_outputs:
            raise ValueError(
                "Internal Track-A job lacks one stage-authenticated recording prediction table"
            )
        recording_paths.append(candidates[0])
    return sorted(set(recording_paths), key=lambda path: path.as_posix())


def _source_only_comparator_manifest(
    manifest: pd.DataFrame,
    *,
    require_union: bool = True,
) -> None:
    required = {"dataset", "split", "modality", "protocol", "fold", "manifest_sha256"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Aligned comparator manifest is missing columns: {missing}")
    if manifest.empty:
        raise ValueError("Aligned comparator manifest is empty")
    datasets = manifest["dataset"].astype(str).str.casefold()
    splits = manifest["split"].astype(str).str.casefold()
    target = ~datasets.eq("coswara")
    if (target & ~splits.eq("external_test")).any():
        raise ValueError(
            "COUGHVID target rows are forbidden from comparator fitting or selection"
        )
    if (datasets.eq("coswara") & splits.eq("external_test")).any():
        raise ValueError("Coswara source rows cannot be labeled as external_test")
    if not datasets.eq("coswara").any():
        raise ValueError("Aligned comparator has no Coswara fitting rows")
    if require_union and (
        "manifest_component" not in manifest
        or set(manifest["manifest_component"].astype(str))
        != set(_ALIGNED_COMPARATOR_COMPONENTS)
    ):
        raise ValueError("Aligned comparator manifest omits a declared analysis context")
    if manifest.duplicated(
        ["protocol", "fold", "recording_key", "modality"]
    ).any():
        raise ValueError("Aligned comparator manifest contains a context collision")


def _verify_comparator_generation(
    audit_root: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    current_path = audit_root / "current.json"
    if not current_path.is_file() or current_path.is_symlink():
        raise FileNotFoundError("Aligned comparator current generation receipt is missing")
    try:
        current = json.loads(current_path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Aligned comparator current generation receipt is corrupt") from exc
    if not isinstance(current, dict) or set(current) != {
        "generation_id",
        "generation_manifest_sha256",
        "receipt_sha256",
    }:
        raise ValueError("Aligned comparator current generation receipt schema is invalid")
    unsigned = {key: value for key, value in current.items() if key != "receipt_sha256"}
    if str(current["receipt_sha256"]) != canonical_json_sha256(unsigned):
        raise ValueError("Aligned comparator current generation receipt hash changed")
    generation_id = str(current["generation_id"])
    generation_root = (audit_root / "generations" / generation_id).resolve()
    try:
        generation_root.relative_to(audit_root.resolve())
    except ValueError as exc:
        raise ValueError("Aligned comparator generation escaped its audit root") from exc
    manifest_path = generation_root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("Aligned comparator generation manifest is missing")
    if stable_file_sha256(manifest_path) != str(current["generation_manifest_sha256"]):
        raise ValueError("Aligned comparator generation manifest checksum changed")
    try:
        generation = json.loads(manifest_path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Aligned comparator generation manifest is corrupt") from exc
    files = generation.get("files") if isinstance(generation, Mapping) else None
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Aligned comparator generation manifest contains no files")
    for relative, descriptor in files.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError("Aligned comparator generation file descriptor is invalid")
        candidate = (generation_root / str(relative)).resolve()
        try:
            candidate.relative_to(generation_root)
        except ValueError as exc:
            raise ValueError("Aligned comparator generation file escaped its generation") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(f"Aligned comparator generation file is missing: {candidate}")
        if candidate.stat().st_size != int(descriptor.get("size_bytes", -1)):
            raise ValueError("Aligned comparator generation file size changed")
        if stable_file_sha256(candidate) != str(descriptor.get("sha256", "")):
            raise ValueError("Aligned comparator generation file checksum changed")
    return current_path, manifest_path, dict(current), dict(generation)


def _comparator_trust_inputs(pipeline: HSTPipeline) -> dict[str, object]:
    """Resolve canonical trust documents without feeding them into run identity."""
    identity_freezes = {
        "data_contracts_freeze",
        "pilot_freeze",
        "environment_lock",
    }
    if set(pipeline.config.accepted_hashes) != identity_freezes:
        raise ValueError(
            "HST run identity must contain only the three frozen pilot/data/environment hashes"
        )
    trusted_root = pipeline.config.workspace_root.resolve()
    expected_workspace = pipeline.config.workspace_root.resolve()
    if not (
        (expected_workspace / "pyproject.toml").is_file()
        and (expected_workspace / "src" / "covid_rars").is_dir()
    ):
        raise ValueError("HST workspace is not the canonical trusted project root")
    approval_path = (
        expected_workspace / "configs" / "hst_compare_is10_approval.approved.json"
    ).resolve()
    accepted_freezes_path = (
        expected_workspace
        / "configs"
        / "hst_comparator_accepted_freezes.approved.json"
    ).resolve()
    for name, path in (
        ("approval", approval_path),
        ("accepted-freezes", accepted_freezes_path),
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Canonical comparator {name} file is missing: {path}")
        try:
            path.relative_to(expected_workspace)
        except ValueError as exc:
            raise ValueError(f"Canonical comparator {name} escaped the project root") from exc
    return {
        "trusted_root": trusted_root,
        "approval_path": approval_path,
        "approval_sha256": stable_file_sha256(approval_path),
        "accepted_freezes_path": accepted_freezes_path,
        "accepted_freezes_sha256": stable_file_sha256(accepted_freezes_path),
    }


def _complete_comparator_features(
    pipeline: HSTPipeline,
) -> tuple[pd.DataFrame, object, Path]:
    feature_path = _project_path(
        pipeline.config,
        _section(pipeline.config, "paths").get(
            "compare_is10_features", "data/processed/features_compare_is10_merged.csv"
        ),
    )
    if not feature_path.is_file() or feature_path.is_symlink():
        raise FileNotFoundError(f"Complete ComParE+IS10 feature table is missing: {feature_path}")
    features = pd.read_csv(feature_path, low_memory=False)
    feature_columns = tuple(
        column for column in features.columns if column not in _NON_FEATURE_COLUMNS
    )
    if len(feature_columns) <= 800:
        raise ValueError(
            "Aligned comparator requires the complete feature table before top-800 selection"
        )
    contract = build_compare_is10_feature_contract(
        features,
        ordered_feature_columns=feature_columns,
    )
    return features, contract, feature_path


_COMPARATOR_GENERATION_TABLES = {
    "recording_predictions": "comparator_predictions.csv",
    "participant_predictions": "comparator_participant_predictions.csv",
    "metrics": "comparator_metrics.csv",
    "alignment_audit": "comparator_alignment_audit.csv",
    "feature_selection": "comparator_feature_selection.csv",
    "model_audit": "comparator_model_audit.csv",
    "candidate_selection": "comparator_candidate_selection.csv",
}


def _authenticate_existing_comparator_generation(
    *,
    generation_manifest_path: Path,
    current_receipt_path: Path,
    trust: Mapping[str, object],
) -> tuple[dict[str, pd.DataFrame], int]:
    generation_root = generation_manifest_path.parent
    common = {
        "generation_manifest_path": generation_manifest_path,
        "current_receipt_path": current_receipt_path,
        "approval_record_path": Path(str(trust["approval_path"])),
        "trusted_project_repository_root": Path(str(trust["trusted_root"])),
        "accepted_freezes_path": Path(str(trust["accepted_freezes_path"])),
        "expected_accepted_freezes_sha256": str(trust["accepted_freezes_sha256"]),
        "runtime_random_state": 42,
    }
    tables = {
        name: assert_confirmatory_comparator_table(generation_root / filename, **common)
        for name, filename in _COMPARATOR_GENERATION_TABLES.items()
    }
    model_audit = tables["model_audit"]
    if "model_artifact" not in model_audit:
        raise ValueError("Authenticated comparator model audit lacks model artifacts")
    artifacts = sorted(model_audit["model_artifact"].astype(str).unique().tolist())
    if not artifacts:
        raise ValueError("Authenticated comparator generation contains no model bundles")
    for relative in artifacts:
        load_verified_compare_is10_bundle(generation_root / relative, **common)
    return tables, len(artifacts)


@_scientific_handler("aligned_comparator")
def _aligned_comparator(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    if pipeline.config.mode != "full":
        raise RuntimeError("The confirmatory aligned comparator requires full mode")
    _manifest_path, manifest, manifest_sha256 = _load_indexed_manifest(
        pipeline, "aligned_comparator"
    )
    _source_only_comparator_manifest(manifest)
    trust = _comparator_trust_inputs(pipeline)
    features, feature_contract, feature_path = _complete_comparator_features(pipeline)
    load_frozen_compare_is10_approval(
        trust["approval_path"],
        trusted_project_repository_root=trust["trusted_root"],
        accepted_freezes_path=trust["accepted_freezes_path"],
        expected_accepted_freezes_sha256=str(trust["accepted_freezes_sha256"]),
        runtime_random_state=42,
        feature_contract=feature_contract,
        feature_artifact_sha256=compare_is10_feature_artifact_sha256(features),
        manifest=manifest,
    )
    stage_root = pipeline.run_root / "scientific" / "aligned_comparator"
    audit_root = stage_root / "audit"
    current_candidate = audit_root / "current.json"
    if current_candidate.exists():
        current_path, generation_manifest_path, current, generation = (
            _verify_comparator_generation(audit_root)
        )
        authenticated, model_bundle_count = _authenticate_existing_comparator_generation(
            generation_manifest_path=generation_manifest_path,
            current_receipt_path=current_path,
            trust=trust,
        )
    else:
        run_aligned_compare_is10(
            features,
            manifest,
            feature_contract=feature_contract,
            approval_record_path=trust["approval_path"],
            trusted_project_repository_root=trust["trusted_root"],
            accepted_freezes_path=trust["accepted_freezes_path"],
            expected_accepted_freezes_sha256=str(trust["accepted_freezes_sha256"]),
            selected_feature_k=800,
            ranker="lightgbm",
            selection_scope="per_modality_mean",
            random_state=42,
            optuna_trials=0,
            ensemble_top_k=5,
            selection_metric="auroc",
            run_id=pipeline.run_id,
            test_mode=False,
            allow_sklearn_fallback=False,
            audit_dir=audit_root,
        )
        current_path, generation_manifest_path, current, _generation = (
            _verify_comparator_generation(audit_root)
        )
        raise ManualComparatorGenerationAcceptanceRequired(
            "Comparator generation was written durably but is not evidence. Run script 77, "
            "manually review/promote/commit the exact generation manifest, then rerun the "
            "same content-addressed run; manual comparator generation acceptance is required. "
            f"generation_id={current['generation_id']} "
            f"manifest={generation_manifest_path.as_posix()}"
        )

    tables = {
        f"{name}.csv": authenticated[name]
        for name in _COMPARATOR_GENERATION_TABLES
    }
    generation_root = generation_manifest_path.parent.resolve()
    generation_files = [
        (generation_root / str(relative)).resolve()
        for relative in generation["files"]
    ]
    output_paths: list[Path] = [
        current_path,
        generation_manifest_path,
        *generation_files,
    ]
    for filename, frame in tables.items():
        exported = frame.copy()
        if filename == "metrics.csv" and "model_name" not in exported:
            exported["model_name"] = exported.get("model", ENSEMBLE_MODEL_NAME)
        path = stage_root / filename
        _atomic_csv(exported, path)
        output_paths.append(path)
    identity_path = stage_root / "generation_identity.json"
    atomic_write_json(
        identity_path,
        {
            "schema_version": 1,
            "run_id": pipeline.run_id,
            "manifest_name": "aligned_comparator",
            "manifest_sha256": manifest_sha256,
            "feature_table_path": feature_path.relative_to(
                pipeline.config.workspace_root
            ).as_posix(),
            "feature_table_byte_sha256": stable_file_sha256(feature_path),
            "feature_schema_sha256": feature_contract.schema_sha256,
            "approval_path": Path(str(trust["approval_path"])).relative_to(
                Path(str(trust["trusted_root"]))
            ).as_posix(),
            "approval_byte_sha256": trust["approval_sha256"],
            "accepted_freezes_path": Path(str(trust["accepted_freezes_path"])).relative_to(
                Path(str(trust["trusted_root"]))
            ).as_posix(),
            "accepted_freezes_byte_sha256": trust["accepted_freezes_sha256"],
            "current_receipt_path": current_path.relative_to(
                pipeline.run_root
            ).as_posix(),
            "current_receipt_byte_sha256": stable_file_sha256(current_path),
            "generation_id": current["generation_id"],
            "generation_manifest_path": generation_manifest_path.relative_to(
                pipeline.run_root
            ).as_posix(),
            "generation_manifest_byte_sha256": stable_file_sha256(
                generation_manifest_path
            ),
            "generation_file_count": len(generation["files"]),
            "authenticated_model_bundles": model_bundle_count,
            "generation_reused": True,
            "self_approved": False,
        },
    )
    output_paths.append(identity_path)
    return {
        "output_paths": output_paths,
        "row_counts": {
            "recording_predictions": len(authenticated["recording_predictions"]),
            "participant_predictions": len(authenticated["participant_predictions"]),
            "metrics": len(authenticated["metrics"]),
            "selected_features": len(authenticated["feature_selection"]),
        },
        "metadata": {
            "execution_class": "confirmatory",
            "manifest_sha256": manifest_sha256,
            "generation_id": current["generation_id"],
            "generation_receipt_sha256": stable_file_sha256(current_path),
            "generation_reused": True,
            "authenticated_model_bundles": model_bundle_count,
            "self_approved": False,
            "target_fit": False,
            "target_selection": False,
        },
    }


def _build_authenticated_fusion_binding(
    pipeline: HSTPipeline,
    hst_predictions: pd.DataFrame,
    comparator_predictions: pd.DataFrame,
    *,
    manifest_name: str,
    comparator_current_path: Path,
) -> tuple[AuthenticatedFusionBinding, dict[str, object]]:
    manifest_receipt_path, manifest_receipt, manifest_receipt_sha = (
        _verified_stage_receipt(pipeline, "manifests")
    )
    internal_receipt_path, _internal_receipt, internal_receipt_sha = (
        _verified_stage_receipt(pipeline, "internal_cv")
    )
    comparator_receipt_path, comparator_receipt, comparator_receipt_sha = (
        _verified_stage_receipt(pipeline, "aligned_comparator")
    )
    current = Path(comparator_current_path).resolve()
    allowed_current = {
        (pipeline.run_root / str(relative)).resolve()
        for relative in comparator_receipt["output_paths"]  # type: ignore[index]
    }
    if current not in allowed_current:
        raise ValueError("Comparator current receipt is not an authenticated stage output")
    try:
        current_payload = json.loads(current.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Comparator current receipt is corrupt") from exc
    if not isinstance(current_payload, Mapping):
        raise ValueError("Comparator current receipt is not an object")
    generation_id = str(current_payload.get("generation_id", "")).strip()
    if not generation_id:
        raise ValueError("Comparator current receipt has no generation identity")

    _manifest_path, frozen_manifest, _manifest_file_sha256 = _load_indexed_manifest(
        pipeline, manifest_name
    )
    manifest_identities = frozen_manifest["manifest_sha256"].astype(str).unique().tolist()
    if len(manifest_identities) != 1:
        raise ValueError("Frozen fusion manifest has no single canonical manifest identity")
    manifest_sha256 = manifest_identities[0]
    hst = _fusion_contract._validate_predictions(
        hst_predictions, name="authenticated HST fusion input"
    )
    comparator = _fusion_contract._validate_predictions(
        comparator_predictions, name="authenticated comparator fusion input"
    )
    contexts = sorted(
        {
            tuple(row)
            for row in hst[list(_fusion_contract.CONTEXT_COLUMNS)]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
    )
    comparator_contexts = {
        tuple(row)
        for row in comparator[list(_fusion_contract.CONTEXT_COLUMNS)]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if set(contexts) != comparator_contexts:
        raise ValueError("Authenticated HST/comparator fusion contexts differ")
    entries: list[dict[str, object]] = []
    for context_values in contexts:
        context = dict(zip(_fusion_contract.CONTEXT_COLUMNS, context_values))
        hst_mask = np.logical_and.reduce(
            [hst[column].eq(value) for column, value in context.items()]
        )
        comparator_mask = np.logical_and.reduce(
            [comparator[column].eq(value) for column, value in context.items()]
        )
        hst_context = hst.loc[hst_mask].copy()
        comparator_context = comparator.loc[comparator_mask].copy()
        hst_manifest = _fusion_contract._single_sha256(
            hst_context, "manifest_sha256", name="authenticated HST context"
        )
        comparator_manifest = _fusion_contract._single_sha256(
            comparator_context,
            "manifest_sha256",
            name="authenticated comparator context",
        )
        if hst_manifest != manifest_sha256 or comparator_manifest != manifest_sha256:
            raise ValueError("Fusion context manifest differs from the frozen indexed manifest")
        intersection = _fusion_contract._recording_intersection_hash(hst_context)
        if (
            _fusion_contract._single_sha256(
                hst_context,
                "recording_intersection_sha256",
                name="authenticated HST context",
            )
            != intersection
            or _fusion_contract._single_sha256(
                comparator_context,
                "recording_intersection_sha256",
                name="authenticated comparator context",
            )
            != intersection
        ):
            raise ValueError("Fusion context recording intersection differs across sources")
        entries.append(
            {
                **context,
                "manifest_receipt": {
                    "receipt_id": str(manifest_receipt["record_hash"]),
                    "receipt_sha256": manifest_receipt_sha,
                    "manifest_sha256": manifest_sha256,
                    "recording_intersection_sha256": intersection,
                },
                "hst": {
                    "prediction_artifact_sha256": _fusion_contract._prediction_artifact_hash(
                        hst_context
                    ),
                    "branches": {
                        modality: _fusion_contract._branch_identity_values(
                            hst_context, modality
                        )
                        for modality in _fusion_contract.PRIMARY_MODALITIES
                    },
                },
                "comparator": {
                    "prediction_artifact_sha256": _fusion_contract._prediction_artifact_hash(
                        comparator_context
                    ),
                    "generation_id": generation_id,
                    "generation_receipt_sha256": stable_file_sha256(current),
                    "branches": {
                        modality: _fusion_contract._branch_identity_values(
                            comparator_context, modality
                        )
                        for modality in _fusion_contract.PRIMARY_MODALITIES
                    },
                },
            }
        )
    receipt_id = canonical_json_sha256(
        {
            "manifests": manifest_receipt_sha,
            "internal_cv": internal_receipt_sha,
            "aligned_comparator": comparator_receipt_sha,
        }
    )
    registry_receipt: dict[str, object] = {
        "schema_version": 1,
        "receipt_type": "hst_fusion_authenticated_registry",
        "registry_authority": "covid_rars.HSTPipeline.receipt_chain",
        "receipt_id": receipt_id,
        "contexts": entries,
    }
    receipt_sha256 = canonical_json_sha256(registry_receipt)
    binding = AuthenticatedFusionBinding.from_registry_receipt(
        registry_receipt,
        trusted_receipt_sha256=receipt_sha256,
    )
    del manifest_receipt_path, internal_receipt_path, comparator_receipt_path
    return binding, registry_receipt


def _manifest_audio_hash_column(manifest: pd.DataFrame) -> str:
    for column in (
        "source_audio_sha256",
        "audio_content_sha256",
        "source_sha256",
        "content_sha256",
        "audio_sha256",
    ):
        if column in manifest:
            return column
    raise ValueError("Frozen fusion manifest has no source-audio content hash")


def _prepare_fusion_source(
    raw: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    pipeline: HSTPipeline,
    source_family: str,
) -> pd.DataFrame:
    if source_family not in {"hst", "comparator"}:
        raise ValueError("Fusion source family must be hst or comparator")
    primary = set(_fusion_contract.PRIMARY_MODALITIES)
    selected = raw.loc[
        raw["modality"].astype(str).isin(primary)
        & raw["split"].astype(str).isin({"validation", "test"})
    ].copy()
    if source_family == "hst":
        selected = selected.loc[selected["model"].astype(str).eq("hst_base")].copy()
    else:
        comparator_config = _section(pipeline.config, "comparator")
        primary_endpoint = str(comparator_config.get("primary_endpoint", ""))
        if primary_endpoint != SELECTED_CANDIDATE_MODEL_NAME:
            raise ValueError(
                "Frozen comparator primary endpoint must be "
                f"{SELECTED_CANDIDATE_MODEL_NAME!r}"
            )
        selected = selected.loc[
            selected["model"].astype(str).eq(primary_endpoint)
        ].copy()
    if selected.empty:
        raise ValueError(f"Authenticated {source_family} fusion source is empty")

    expected = manifest.loc[
        manifest["modality"].astype(str).isin(primary)
        & manifest["split"].astype(str).isin({"validation", "test"})
    ].copy()
    expected_protocols = expected["protocol"].astype(str).drop_duplicates().tolist()
    if len(expected_protocols) != 1:
        raise ValueError("Frozen fusion manifest must contain exactly one protocol")
    selected = selected.loc[
        selected["protocol"].astype(str).eq(expected_protocols[0])
    ].copy()
    key_columns = [
        "protocol",
        "fold",
        "dataset",
        "participant_key",
        "recording_key",
        "split",
        "modality",
        "label_binary",
    ]
    missing = sorted(set(key_columns) - set(selected.columns))
    if missing:
        raise ValueError(f"Authenticated {source_family} predictions miss keys: {missing}")
    expected_keys = expected[key_columns].sort_values(key_columns, kind="mergesort").reset_index(
        drop=True
    )
    selected_keys = selected[key_columns].sort_values(key_columns, kind="mergesort").reset_index(
        drop=True
    )
    if not selected_keys.equals(expected_keys):
        raise ValueError(
            f"Authenticated {source_family} predictions do not cover the exact eligible manifest cohort"
        )
    if selected.duplicated(key_columns[:-1]).any():
        raise ValueError(f"Authenticated {source_family} predictions contain duplicate recordings")

    audio_hash_column = _manifest_audio_hash_column(expected)
    carry_columns = ["cohort", "manifest_sha256", audio_hash_column]
    for optional in ("tensor_sha256", "preprocessing_hash", "eligible"):
        if optional in expected:
            carry_columns.append(optional)
    missing_carry = sorted(set(carry_columns) - set(expected.columns))
    if missing_carry:
        raise ValueError(f"Frozen fusion manifest lacks provenance columns: {missing_carry}")
    manifest_provenance = expected[
        [*key_columns[:-1], "label_binary", *carry_columns]
    ].copy()
    selected = selected.drop(
        columns=[column for column in carry_columns if column in selected],
        errors="ignore",
    ).merge(
        manifest_provenance,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    selected["run_id"] = pipeline.run_id
    selected["source_family"] = source_family
    selected["audio_content_sha256"] = selected[audio_hash_column].astype(str)
    selected["eligible"] = True
    if source_family == "hst":
        if "tensor_sha256" not in selected or "preprocessing_hash" not in selected:
            raise ValueError("HST fusion source lacks tensor/preprocessing provenance")
        selected["feature_approval_id"] = (
            "hst-data-contract:" + str(pipeline.config.accepted_hashes["data_contracts_freeze"])
        )
        selected["preprocessing_sha256"] = selected["preprocessing_hash"].astype(str)
        selected["feature_artifact_sha256"] = ""
        for (*_context, modality), group in selected.groupby(
            [*_fusion_contract.CONTEXT_COLUMNS, "modality"],
            sort=True,
        ):
            digest = canonical_json_sha256(
                group[["recording_key", "tensor_sha256"]]
                .sort_values("recording_key", kind="mergesort")
                .to_dict(orient="records")
            )
            selected.loc[group.index, "feature_artifact_sha256"] = digest
    else:
        required = {
            "feature_artifact_sha256",
            "feature_contract_hash",
            "approval_id",
            "checkpoint_hash",
            "representation",
        }
        missing = sorted(required - set(selected.columns))
        if missing:
            raise ValueError(f"Comparator fusion source lacks provenance columns: {missing}")
        selected["feature_approval_id"] = selected["approval_id"].astype(str)
        selected["preprocessing_sha256"] = selected["feature_contract_hash"].astype(str)

    selected["recording_intersection_sha256"] = "0" * 64
    for _context, group in selected.groupby(
        list(_fusion_contract.CONTEXT_COLUMNS), sort=True
    ):
        selected.loc[group.index, "recording_intersection_sha256"] = (
            _fusion_contract._recording_intersection_hash(group)
        )
    return _fusion_contract._validate_predictions(
        selected,
        name=f"prepared {source_family} fusion source",
    )


def _verify_fusion_generation(
    output_root: Path,
    receipt: Mapping[str, object],
) -> tuple[Path, Path, list[Path]]:
    current = output_root / "current.json"
    if not current.is_file() or current.is_symlink():
        raise FileNotFoundError("Fusion current receipt is missing")
    current_payload = json.loads(current.read_text(encoding="utf-8"))
    if current_payload != dict(receipt):
        raise ValueError("Fusion current receipt differs from the returned generation receipt")
    claimed = str(current_payload.get("record_hash", ""))
    unsigned = {key: value for key, value in current_payload.items() if key != "record_hash"}
    if claimed != canonical_json_sha256(unsigned):
        raise ValueError("Fusion generation receipt record hash changed")
    generation = (output_root / str(current_payload.get("generation_path", ""))).resolve()
    try:
        generation.relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError("Fusion generation escaped its output root") from exc
    checksums = generation / "checksums.json"
    if not checksums.is_file() or stable_file_sha256(checksums) != stable_file_sha256(current):
        raise ValueError("Fusion generation checksums receipt differs from current receipt")
    artifacts = current_payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("Fusion generation contains no artifact contract")
    files = [current, checksums]
    for descriptor in artifacts.values():
        if not isinstance(descriptor, Mapping):
            raise ValueError("Fusion artifact descriptor is invalid")
        path = (generation / str(descriptor.get("relative_path", ""))).resolve()
        try:
            path.relative_to(generation)
        except ValueError as exc:
            raise ValueError("Fusion artifact escaped its generation") from exc
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Fusion artifact is missing: {path}")
        if stable_file_sha256(path) != str(descriptor.get("sha256", "")):
            raise ValueError("Fusion artifact checksum changed")
        files.append(path)
    return current, checksums, files


def _frozen_publication_analysis_plan() -> pd.DataFrame:
    from .hst_publication import ANALYSIS_SCOPE_REGISTRY, PRIMARY_ESTIMAND_ID, freeze_analysis_plan

    rows: list[dict[str, object]] = []
    for estimand_id, scope in ANALYSIS_SCOPE_REGISTRY.items():
        row: dict[str, object] = {
            "estimand_id": estimand_id,
            "analysis_role": scope.role,
            "analysis_scope": scope.scope,
            "multiplicity_family": scope.family,
            "metric": scope.metric,
            "comparison_design": scope.design,
            "candidate_family": "",
            "reference_family": "",
            "split": "",
            "fusion_method": "",
            "modality_combination": "",
            "complete_case": False,
        }
        if estimand_id == PRIMARY_ESTIMAND_ID:
            row.update(
                {
                    "candidate_family": "hst",
                    "reference_family": "comparator",
                    "split": "test",
                    "fusion_method": "uniform_mean",
                    "modality_combination": "cough+speech",
                    "complete_case": True,
                }
            )
        rows.append(row)
    return freeze_analysis_plan(pd.DataFrame(rows))


def _publication_fusion_partition(
    predictions: pd.DataFrame,
    *,
    source_family: str,
    split: str,
) -> pd.DataFrame:
    required = {
        "source_family",
        "split",
        "fusion_method",
        "modality_combination",
        "complete_case",
        "protocol",
        "manifest_sha256",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Fusion publication predictions miss columns: {missing}")
    complete_case = predictions["complete_case"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    )
    frame = predictions.loc[
        predictions["source_family"].astype(str).eq(source_family)
        & predictions["split"].astype(str).eq(split)
        & predictions["fusion_method"].astype(str).eq("uniform_mean")
        & predictions["modality_combination"].astype(str).eq("cough+speech")
        & complete_case
    ].copy()
    if frame.empty:
        raise ValueError(
            f"Fusion has no complete-case {source_family} cough+speech {split} predictions"
        )
    frame["source_protocol"] = frame["protocol"].astype(str)
    frame["source_manifest_sha256"] = frame["manifest_sha256"].astype(str)
    return frame.sort_values(
        ["fold", "participant_key"], kind="mergesort"
    ).reset_index(drop=True)


def _publication_fusion_method_table(
    fusion_predictions: object,
    reference: object,
    *,
    source_family: str,
    fusion_method: str,
    split: str,
    analysis_plan: object,
):
    from .hst_publication import derive_authenticated_table

    raw = fusion_predictions.frame.copy()
    complete_case = raw["complete_case"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    )
    frame = raw.loc[
        raw["source_family"].astype(str).eq(source_family)
        & raw["fusion_method"].astype(str).eq(fusion_method)
        & raw["split"].astype(str).eq(split)
        & complete_case
    ].copy()
    if frame.empty:
        raise ValueError(
            f"Fusion has no {source_family}/{fusion_method}/{split} complete-case rows"
        )
    keys = ["fold", "participant_key"]
    reference_frame = reference.frame.sort_values(keys, kind="mergesort").reset_index(
        drop=True
    )
    frame = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if frame.duplicated(keys).any() or reference_frame.duplicated(keys).any():
        raise ValueError("Publication fusion method table contains duplicate participants")
    if not frame[keys].equals(reference_frame[keys]) or not frame[
        "label_binary"
    ].astype("string").equals(reference_frame["label_binary"].astype("string")):
        raise ValueError("Publication fusion method does not use the exact primary cohort")
    return derive_authenticated_table(
        frame,
        source_name=f"fusion:{source_family}:{fusion_method}:{split}",
        sources=[fusion_predictions, reference],
        analysis_plan=analysis_plan,
        test_mode=bool(fusion_predictions.test_mode),
    )


def _publication_constituent_table(
    branch_predictions: object,
    fusion_reference: object,
    *,
    modality: str,
    split: str,
    analysis_plan: object,
):
    from .hst_publication import derive_authenticated_table

    reference = fusion_reference.frame.copy()
    protocols = reference["protocol"].astype(str).drop_duplicates().tolist()
    if len(protocols) != 1:
        raise ValueError("Fusion constituent reference must contain one protocol")
    frame = branch_predictions.frame.copy()
    frame = frame.loc[
        frame["protocol"].astype(str).eq(protocols[0])
        & frame["split"].astype(str).eq(split)
        & frame["modality"].astype(str).eq(modality)
        & frame["model"].astype(str).eq("hst_base")
    ].copy()
    if frame.empty:
        raise ValueError(f"HST {modality} constituent {split} predictions are empty")
    keys = ["fold", "participant_key"]
    frame = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    reference = reference.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if frame.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError("Fusion constituent comparison contains duplicate participants")
    if not frame[keys].equals(reference[keys]) or not frame["label_binary"].astype(
        "string"
    ).equals(reference["label_binary"].astype("string")):
        raise ValueError(
            f"HST {modality} constituent does not cover the exact fusion {split} cohort"
        )
    return derive_authenticated_table(
        frame,
        source_name=f"hst_constituent:{modality}:{split}",
        sources=[branch_predictions, fusion_reference],
        analysis_plan=analysis_plan,
        test_mode=bool(branch_predictions.test_mode),
    )


def _publication_uniform_cough_speech_table(
    branch_predictions: object,
    *,
    source_name: str,
    analysis_plan: object,
):
    from .hst_publication import derive_authenticated_table

    frame = branch_predictions.frame.copy()
    frame = frame.loc[
        frame["modality"].astype(str).isin(["cough", "speech"])
        & frame["model"].astype(str).eq("hst_base")
    ].copy()
    if frame.empty:
        raise ValueError(f"{source_name} contains no HST cough/speech predictions")
    keys = ["fold", "participant_key"] if "fold" in frame else ["participant_key"]
    if frame.duplicated([*keys, "modality"]).any():
        raise ValueError(f"{source_name} contains duplicate participant-modality rows")
    context_columns = [
        column
        for column in (
            "dataset",
            "split",
            "protocol",
            "cohort",
            "manifest_sha256",
            "source_protocol",
            "source_manifest_sha256",
        )
        if column in frame
    ]
    for column in ["label_binary", *context_columns]:
        if frame.groupby(keys, sort=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(
                f"{source_name} disagrees within participant on {column!r}"
            )
    probability = frame.pivot(
        index=keys,
        columns="modality",
        values="probability",
    )
    complete = probability.dropna(subset=["cough", "speech"]).copy()
    if complete.empty:
        raise ValueError(f"{source_name} has no complete cough+speech participants")
    base = (
        frame.sort_values([*keys, "modality"], kind="mergesort")
        .groupby(keys, sort=False, as_index=False)
        .first()
        .set_index(keys)
        .loc[complete.index]
        .reset_index()
    )
    base = base.drop(
        columns=[
            column
            for column in (
                "modality",
                "probability",
                "submodality",
                "representation_id",
            )
            if column in base
        ]
    )
    base["probability"] = complete[["cough", "speech"]].mean(axis=1).to_numpy()
    base["modality_combination"] = "cough+speech"
    base["fusion_method"] = "uniform_mean"
    base["source_family"] = "hst"
    base["model"] = "uniform_mean"
    base["complete_case"] = True
    base["available_modalities"] = "cough,speech"
    base["n_modalities"] = 2
    return derive_authenticated_table(
        base.sort_values(keys, kind="mergesort").reset_index(drop=True),
        source_name=source_name,
        sources=[branch_predictions],
        analysis_plan=analysis_plan,
        test_mode=bool(branch_predictions.test_mode),
    )


def _capacity_hst_only_fusion(pipeline: HSTPipeline) -> Mapping[str, object]:
    """Run the bounded HST model bank without the full-study comparator gate."""
    _manifest_path, manifest, _manifest_file_sha256 = _load_indexed_manifest(
        pipeline, "internal"
    )
    _source_only_comparator_manifest(manifest, require_union=False)
    _hst_receipt_path, hst_receipt, _hst_receipt_sha = _verified_stage_receipt(
        pipeline, "internal_cv"
    )
    hst_paths = _internal_track_a_recording_prediction_paths(
        pipeline, hst_receipt
    )
    hst_raw = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in hst_paths],
        ignore_index=True,
        sort=False,
    )
    protocol = str(manifest["protocol"].iloc[0])
    hst_raw = hst_raw.loc[hst_raw["protocol"].astype(str).eq(protocol)].copy()
    hst = _prepare_fusion_source(
        hst_raw,
        manifest,
        pipeline=pipeline,
        source_family="hst",
    )
    result = run_hst_fusion_bank(
        hst,
        analysis_mode="exploratory",
    )
    stage_root = pipeline.run_root / "scientific" / "fusion"
    stage_root.mkdir(parents=True, exist_ok=True)
    hst_path = stage_root / "hst_recording_predictions.csv"
    _atomic_csv(hst, hst_path)
    generation_root = stage_root / "generation"
    generation_receipt = result.save_generation(generation_root)
    current, checksums, _generation_files = _verify_fusion_generation(
        generation_root, generation_receipt
    )
    output_paths: list[Path] = [hst_path, current, checksums]
    for name, _filename in _fusion_contract.FUSION_TABLE_FILENAMES:
        frame = getattr(result, name).copy()
        if name == "metrics" and "model_name" not in frame:
            frame["model_name"] = frame.get(
                "model", frame.get("source_family", "fusion")
            )
        path = stage_root / f"fusion_{name}.csv"
        _atomic_csv(frame, path)
        output_paths.append(path)
    publication_partitions = {
        f"primary_hst_{split}_predictions": _publication_fusion_partition(
            result.predictions,
            source_family="hst",
            split=split,
        )
        for split in ("validation", "test")
    }
    for name, frame in publication_partitions.items():
        path = stage_root / f"{name}.csv"
        _atomic_csv(frame, path)
        output_paths.append(path)
    return {
        "output_paths": output_paths,
        "row_counts": {
            name: len(getattr(result, name))
            for name, _filename in _fusion_contract.FUSION_TABLE_FILENAMES
        }
        | {name: len(frame) for name, frame in publication_partitions.items()},
        "metadata": {
            "analysis_mode": "exploratory",
            "analysis_scope": "internal_hst_cough_speech_model_bank_extension",
            "primary_method": "complete_case_uniform_cough_speech",
            "secondary_methods": (
                "validation_weighted_auprc,stacked_logistic_validation"
            ),
            "generation_id": generation_receipt["generation_id"],
            "comparator_required": False,
            "target_fit": False,
            "target_selection": False,
        },
    }


@_scientific_handler("fusion")
def _fusion(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    if pipeline.config.mode != "full":
        raise RuntimeError("Confirmatory fusion requires full mode")
    profile = workload_profile_from_scientific_config(
        pipeline.config.scientific_config
    )
    if profile.name == CAPACITY_INTERNAL_FUSION_PROFILE:
        return _capacity_hst_only_fusion(pipeline)
    _manifest_path, manifest, _manifest_file_sha256 = _load_indexed_manifest(
        pipeline, "internal"
    )
    _source_only_comparator_manifest(manifest, require_union=False)
    _hst_receipt_path, hst_receipt, _hst_receipt_sha = _verified_stage_receipt(
        pipeline, "internal_cv"
    )
    _comparator_receipt_path, comparator_receipt, _comparator_receipt_sha = (
        _verified_stage_receipt(pipeline, "aligned_comparator")
    )
    hst_paths = _internal_track_a_recording_prediction_paths(
        pipeline, hst_receipt
    )
    hst_raw = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in hst_paths],
        ignore_index=True,
        sort=False,
    )
    protocol = str(manifest["protocol"].iloc[0])
    hst_raw = hst_raw.loc[hst_raw["protocol"].astype(str).eq(protocol)].copy()
    comparator_path = _receipt_output_file(
        pipeline, comparator_receipt, "/recording_predictions.csv"
    )
    comparator_raw = pd.read_csv(comparator_path, low_memory=False)
    hst = _prepare_fusion_source(
        hst_raw,
        manifest,
        pipeline=pipeline,
        source_family="hst",
    )
    comparator = _prepare_fusion_source(
        comparator_raw,
        manifest,
        pipeline=pipeline,
        source_family="comparator",
    )
    identity = [
        *_fusion_contract.CONTEXT_COLUMNS,
        "participant_key",
        "recording_key",
        "split",
        "modality",
        "label_binary",
        "audio_content_sha256",
    ]
    if not hst[identity].sort_values(identity, kind="mergesort").reset_index(
        drop=True
    ).equals(
        comparator[identity].sort_values(identity, kind="mergesort").reset_index(
            drop=True
        )
    ):
        raise ValueError("HST and comparator do not have exact identical eligible cohorts")
    current_path = _receipt_output_file(
        pipeline, comparator_receipt, "/current.json"
    )
    authenticated_binding, registry = _build_authenticated_fusion_binding(
        pipeline,
        hst,
        comparator,
        manifest_name="internal",
        comparator_current_path=current_path,
    )
    result = run_hst_fusion_bank(
        hst,
        comparator,
        analysis_mode="confirmatory",
        authenticated_binding=authenticated_binding,
    )
    stage_root = pipeline.run_root / "scientific" / "fusion"
    stage_root.mkdir(parents=True, exist_ok=True)
    registry_path = stage_root / "authenticated_registry_receipt.json"
    atomic_write_json(registry_path, registry)
    hst_path = stage_root / "hst_recording_predictions.csv"
    comparator_input_path = stage_root / "comparator_recording_predictions.csv"
    _atomic_csv(hst, hst_path)
    _atomic_csv(comparator, comparator_input_path)
    generation_root = stage_root / "generation"
    generation_receipt = result.save_generation(generation_root)
    current, checksums, _generation_files = _verify_fusion_generation(
        generation_root, generation_receipt
    )
    output_paths: list[Path] = [
        registry_path,
        hst_path,
        comparator_input_path,
        current,
        checksums,
    ]
    for name, _filename in _fusion_contract.FUSION_TABLE_FILENAMES:
        frame = getattr(result, name).copy()
        if name == "metrics" and "model_name" not in frame:
            frame["model_name"] = frame.get("model", frame.get("source_family", "fusion"))
        path = stage_root / f"fusion_{name}.csv"
        _atomic_csv(frame, path)
        output_paths.append(path)
    analysis_plan = _frozen_publication_analysis_plan()
    analysis_plan_path = stage_root / "analysis_plan.csv"
    _atomic_csv(analysis_plan, analysis_plan_path)
    output_paths.append(analysis_plan_path)
    publication_partitions: dict[str, pd.DataFrame] = {}
    for source_family in ("hst", "comparator"):
        for split in ("validation", "test"):
            key = f"primary_{source_family}_{split}_predictions"
            publication_partitions[key] = _publication_fusion_partition(
                result.predictions,
                source_family=source_family,
                split=split,
            )
    for name, frame in publication_partitions.items():
        path = stage_root / f"{name}.csv"
        _atomic_csv(frame, path)
        output_paths.append(path)
    return {
        "output_paths": output_paths,
        "row_counts": {
            name: len(getattr(result, name))
            for name, _filename in _fusion_contract.FUSION_TABLE_FILENAMES
        }
        | {name: len(frame) for name, frame in publication_partitions.items()}
        | {"analysis_plan": len(analysis_plan)},
        "metadata": {
            "analysis_mode": "confirmatory",
            "primary_method": "complete_case_uniform_cough_speech",
            "authenticated_registry_receipt_sha256": authenticated_binding.receipt_sha256,
            "generation_id": generation_receipt["generation_id"],
            "identical_eligible_cohorts": True,
            "target_fit": False,
            "target_selection": False,
        },
    }


def _load_publication_table(
    pipeline: HSTPipeline,
    *,
    stage: str,
    suffix: str,
):
    from .hst_publication import load_receipted_table

    _receipt_path, receipt, receipt_sha256 = _verified_stage_receipt(pipeline, stage)
    path = _receipt_output_file(pipeline, receipt, suffix)
    relative = path.relative_to(pipeline.run_root.resolve()).as_posix()
    return load_receipted_table(
        run_root=pipeline.run_root,
        stage=stage,
        relative_path=relative,
        expected_receipt_sha256=receipt_sha256,
    )


def _load_publication_manifest_partition(
    pipeline: HSTPipeline,
    *,
    stage: str,
    manifest_name: str,
):
    from .hst_publication import load_receipted_table

    _receipt_path, receipt, receipt_sha256 = _verified_stage_receipt(pipeline, stage)
    marker = f"/publication_{manifest_name}_"
    candidates = [
        (pipeline.run_root / str(relative)).resolve()
        for relative in receipt["output_paths"]  # type: ignore[index]
        if marker in Path(str(relative)).as_posix()
        and Path(str(relative)).name.endswith("_predictions.csv")
        and "_validation_" not in Path(str(relative)).name
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one evaluation partition for manifest {manifest_name!r}; "
            f"found {len(candidates)}"
        )
    relative = candidates[0].relative_to(pipeline.run_root.resolve()).as_posix()
    return load_receipted_table(
        run_root=pipeline.run_root,
        stage=stage,
        relative_path=relative,
        expected_receipt_sha256=receipt_sha256,
    )


def _mean_fold_auroc(frame: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score

    values: list[float] = []
    groups = frame.groupby("fold", sort=True) if "fold" in frame else [(None, frame)]
    for _fold, group in groups:
        labels = labels_to_binary(group["label_binary"])
        if len(np.unique(labels)) != 2:
            raise ValueError("AUROC figure source contains a single-class evaluation fold")
        values.append(float(roc_auc_score(labels, group["probability"].astype(float))))
    if not values:
        raise ValueError("AUROC figure source is empty")
    return float(np.mean(values))


def _build_engineering_objective_audit(
    branch_test_predictions: pd.DataFrame,
    fusion_test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare frozen test results to engineering references without selecting models."""
    common = {"fold", "participant_key", "label_binary", "probability", "split", "protocol"}
    branch_missing = sorted(
        (common | {"modality"}) - set(branch_test_predictions.columns)
    )
    fusion_missing = sorted(
        (common | {"fusion_method", "modality_combination", "complete_case"})
        - set(fusion_test_predictions.columns)
    )
    if branch_missing:
        raise ValueError(f"Engineering branch audit misses columns: {branch_missing}")
    if fusion_missing:
        raise ValueError(f"Engineering fusion audit misses columns: {fusion_missing}")

    protocol = "hst_literature_aligned_repeated_holdout"
    branch = branch_test_predictions.loc[
        branch_test_predictions["split"].astype(str).eq("test")
        & branch_test_predictions["protocol"].astype(str).eq(protocol)
    ].copy()
    complete_case = fusion_test_predictions["complete_case"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    )
    fusion = fusion_test_predictions.loc[
        fusion_test_predictions["split"].astype(str).eq("test")
        & fusion_test_predictions["protocol"].astype(str).eq(protocol)
        & fusion_test_predictions["fusion_method"].astype(str).eq("uniform_mean")
        & fusion_test_predictions["modality_combination"].astype(str).eq("cough+speech")
        & complete_case
    ].copy()

    rows: list[dict[str, object]] = []
    references = dict(_ENGINEERING_OBJECTIVE_REFERENCES)
    for branch_name in ("cough", "breath", "speech"):
        frame = branch.loc[branch["modality"].astype(str).eq(branch_name)].copy()
        if frame.empty:
            raise ValueError(f"Engineering audit has no HST {branch_name} test predictions")
        rows.append({"branch": branch_name, "frame": frame})
    if fusion.empty:
        raise ValueError("Engineering audit has no complete-case cough+speech uniform fusion")
    rows.append({"branch": "cough_speech_fusion", "frame": fusion})

    audits: list[dict[str, object]] = []
    expected_folds = set(range(1, 11))
    for item in rows:
        branch_name = str(item["branch"])
        frame = item["frame"]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Engineering objective frame is invalid")
        folds = set(pd.to_numeric(frame["fold"], errors="raise").astype(int))
        if folds != expected_folds:
            raise ValueError(
                f"Engineering objective {branch_name} requires folds 1-10; found {sorted(folds)}"
            )
        if frame.duplicated(["fold", "participant_key"]).any():
            raise ValueError(
                f"Engineering objective {branch_name} has duplicate participant-fold rows"
            )
        observed = _mean_fold_auroc(frame)
        reference = float(references[branch_name])
        audits.append(
            {
                "branch": branch_name,
                "metric": "participant_auroc_mean_across_repeated_holdouts",
                "protocol": protocol,
                "observed_auroc": observed,
                "reference_auroc": reference,
                "delta_observed_minus_reference": observed - reference,
                "achieved": bool(observed > reference),
                "fold_count": 10,
                "targets_not_selection_rules": True,
                "generated_after_model_selection": True,
                "test_set_is_not_a_stopping_rule": True,
                "analysis_scope": "descriptive_engineering_audit",
            }
        )
    return pd.DataFrame(audits)


def _validate_engineering_objective_freeze(pipeline: HSTPipeline) -> None:
    configured = _section(pipeline.config, "performance_objectives")
    expected = {name: value for name, value in _ENGINEERING_OBJECTIVE_REFERENCES}
    references = configured.get("references")
    if not isinstance(references, Mapping):
        raise ValueError("Frozen engineering objective references are missing")
    observed = {str(key): float(value) for key, value in references.items()}
    if observed != expected:
        raise ValueError("Engineering objective references differ from the frozen plan")
    if configured.get("metric") != "participant_auroc":
        raise ValueError("Engineering objective metric differs from the frozen plan")
    if configured.get("engineering_targets_not_guarantees") is not True:
        raise ValueError("Engineering objectives must be marked as targets, not guarantees")
    if configured.get("test_set_is_not_a_stopping_rule") is not True:
        raise ValueError("Held-out test results cannot be an engineering stopping rule")


def _derived_figure_table(
    frame: pd.DataFrame,
    *,
    source_name: str,
    sources: list[object],
    analysis_plan: object,
):
    from .hst_publication import derive_authenticated_table

    return derive_authenticated_table(
        frame,
        source_name=f"derived:{source_name}",
        sources=sources,
        analysis_plan=analysis_plan,
        test_mode=bool(getattr(analysis_plan, "test_mode", False)),
    )


def _runtime_figure_frame(pipeline: HSTPipeline) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in HSTPipeline.STAGES:
        if stage == "statistics":
            break
        path = pipeline.run_root / "runtime" / "stages" / f"{stage}.json"
        if not path.is_file():
            continue
        receipt_path, payload, receipt_sha256 = _verified_stage_receipt(
            pipeline, stage
        )
        started = pd.Timestamp(str(payload["started_at"]))
        completed = pd.Timestamp(str(payload["completed_at"]))
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Runtime stage metadata is not a mapping: {stage}")
        measured = metadata.get("gpu_memory_measured", False)
        if not isinstance(measured, bool):
            raise ValueError(f"Runtime GPU-memory flag is not boolean: {stage}")
        child_measured = metadata.get("child_gpu_memory_measured", False)
        if not isinstance(child_measured, bool):
            raise ValueError(f"Runtime child GPU-memory flag is not boolean: {stage}")
        if child_measured:
            if (
                stage != "base_resource_pilot"
                or metadata.get("gpu_memory_measurement_scope")
                != "selected_resource_pilot_child_process"
                or measured
            ):
                raise ValueError(
                    f"Runtime child GPU-memory measurement has invalid scope: {stage}"
                )
            measured = True
        allocated = np.nan
        reserved = np.nan
        if measured:
            try:
                if child_measured:
                    allocated = float(
                        metadata["child_peak_gpu_memory_allocated_mb"]
                    )
                    reserved = float(
                        metadata["child_peak_gpu_memory_reserved_mb"]
                    )
                else:
                    allocated = float(metadata["peak_gpu_memory_allocated_mb"])
                    reserved = float(metadata["peak_gpu_memory_reserved_mb"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Measured runtime GPU-memory fields are invalid: {stage}"
                ) from exc
            if (
                not math.isfinite(allocated)
                or not math.isfinite(reserved)
                or allocated < 0
                or reserved < allocated
            ):
                raise ValueError(
                    f"Measured runtime GPU-memory fields are inconsistent: {stage}"
                )
        rows.append(
            {
                "stage": stage,
                "elapsed_seconds": max(0.0, float((completed - started).total_seconds())),
                "gpu_memory_measured": measured,
                "peak_gpu_memory_allocated_mb": allocated,
                "peak_gpu_memory_reserved_mb": reserved,
                "peak_gpu_memory_mb": allocated,
                "stage_receipt_path": receipt_path.relative_to(
                    pipeline.run_root.resolve()
                ).as_posix(),
                "stage_receipt_sha256": receipt_sha256,
                "stage_receipt_record_hash": str(payload["record_hash"]),
            }
        )
    if not rows:
        raise ValueError("Runtime figure has no successful upstream stage receipts")
    return pd.DataFrame(rows)


def _build_estimand_execution_audit(
    analysis_plan: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    if "estimand_id" not in analysis_plan or "estimand_id" not in evidence:
        raise ValueError("Estimand execution audit requires estimand_id columns")
    rows: list[dict[str, object]] = []
    for plan_row in analysis_plan.to_dict(orient="records"):
        estimand_id = str(plan_row["estimand_id"])
        observed = evidence.loc[evidence["estimand_id"].astype(str).eq(estimand_id)]
        executed = False
        explicitly_skipped = False
        skip_reason = ""
        if not observed.empty:
            skipped = observed.get(
                "skipped", pd.Series(False, index=observed.index)
            ).fillna(False).astype(bool)
            executed = bool((~skipped).any())
            reasons = observed.get(
                "skip_reason", pd.Series("", index=observed.index)
            ).fillna("").astype(str).str.strip()
            explicitly_skipped = bool(skipped.all() and reasons.ne("").all())
            if explicitly_skipped:
                skip_reason = " | ".join(sorted(set(reasons.tolist())))
        rows.append(
            {
                "estimand_id": estimand_id,
                "analysis_role": str(plan_row["analysis_role"]),
                "evidence_row_count": int(len(observed)),
                "executed": executed,
                "explicitly_skipped": explicitly_skipped,
                "executed_or_explicitly_skipped": bool(
                    executed or explicitly_skipped
                ),
                "skip_reason": skip_reason,
            }
        )
    audit = pd.DataFrame(rows)
    incomplete = audit.loc[~audit["executed_or_explicitly_skipped"]]
    if not incomplete.empty:
        missing = ", ".join(incomplete["estimand_id"].astype(str))
        raise ValueError(f"Frozen estimands were not executed or explicitly skipped: {missing}")
    return audit


@_scientific_handler("statistics")
def _statistics(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    if pipeline.config.mode != "full":
        raise RuntimeError("Confirmatory publication statistics require full mode")
    from . import hst_publication as publication

    _validate_engineering_objective_freeze(pipeline)
    _assert_reporting_config_bound(_section(pipeline.config, "reporting"))

    plan_table = _load_publication_table(
        pipeline, stage="fusion", suffix="/analysis_plan.csv"
    )
    analysis_plan = publication.bind_analysis_plan(plan_table)
    primary_hst_validation = _load_publication_table(
        pipeline, stage="fusion", suffix="/primary_hst_validation_predictions.csv"
    )
    primary_hst_test = _load_publication_table(
        pipeline, stage="fusion", suffix="/primary_hst_test_predictions.csv"
    )
    primary_comparator_validation = _load_publication_table(
        pipeline,
        stage="fusion",
        suffix="/primary_comparator_validation_predictions.csv",
    )
    primary_comparator_test = _load_publication_table(
        pipeline, stage="fusion", suffix="/primary_comparator_test_predictions.csv"
    )
    internal_branch_test = _load_publication_table(
        pipeline,
        stage="internal_cv",
        suffix="/publication_internal_test_predictions.csv",
    )
    internal_branch_validation = _load_publication_table(
        pipeline,
        stage="internal_cv",
        suffix="/publication_internal_validation_predictions.csv",
    )
    fusion_predictions = _load_publication_table(
        pipeline,
        stage="fusion",
        suffix="/fusion_predictions.csv",
    )
    hybrid_test = _publication_fusion_method_table(
        fusion_predictions,
        primary_hst_test,
        source_family="hybrid",
        fusion_method="hybrid_uniform_four_branch",
        split="test",
        analysis_plan=analysis_plan,
    )
    constituent_validation = {
        modality: _publication_constituent_table(
            internal_branch_validation,
            primary_hst_validation,
            modality=modality,
            split="validation",
            analysis_plan=analysis_plan,
        )
        for modality in ("cough", "speech")
    }
    constituent_test = {
        modality: _publication_constituent_table(
            internal_branch_test,
            primary_hst_test,
            modality=modality,
            split="test",
            analysis_plan=analysis_plan,
        )
        for modality in ("cough", "speech")
    }
    calendar = _load_publication_manifest_partition(
        pipeline, stage="split_policy_contrast", manifest_name="calendar_mixed"
    )
    chronological = _load_publication_manifest_partition(
        pipeline, stage="split_policy_contrast", manifest_name="early_to_late"
    )
    common_mixed = _load_publication_manifest_partition(
        pipeline, stage="split_policy_contrast", manifest_name="common_late_mixed"
    )
    common_chronological = _load_publication_manifest_partition(
        pipeline,
        stage="split_policy_contrast",
        manifest_name="common_late_chronological",
    )
    chronological_validation = _load_publication_table(
        pipeline,
        stage="split_policy_contrast",
        suffix="/publication_early_to_late_validation_predictions.csv",
    )
    calendar_fusion = _publication_uniform_cough_speech_table(
        calendar,
        source_name="split_policy:calendar_mixed:test:uniform_cough_speech",
        analysis_plan=analysis_plan,
    )
    chronological_fusion = _publication_uniform_cough_speech_table(
        chronological,
        source_name="split_policy:early_to_late:test:uniform_cough_speech",
        analysis_plan=analysis_plan,
    )
    chronological_validation_fusion = _publication_uniform_cough_speech_table(
        chronological_validation,
        source_name="split_policy:early_to_late:validation:uniform_cough_speech",
        analysis_plan=analysis_plan,
    )
    common_mixed_fusion = _publication_uniform_cough_speech_table(
        common_mixed,
        source_name="split_policy:common_late_mixed:test:uniform_cough_speech",
        analysis_plan=analysis_plan,
    )
    common_chronological_fusion = _publication_uniform_cough_speech_table(
        common_chronological,
        source_name="split_policy:common_late_chronological:test:uniform_cough_speech",
        analysis_plan=analysis_plan,
    )
    external_source_validation = _load_publication_table(
        pipeline, stage="external_transfer", suffix="/source_validation_predictions.csv"
    )
    external_source_test = _load_publication_table(
        pipeline, stage="external_transfer", suffix="/source_test_predictions.csv"
    )
    external_target = _load_publication_table(
        pipeline, stage="external_transfer", suffix="/participant_predictions.csv"
    )

    calibration_pairs = {
        "internal_hst": (primary_hst_validation, primary_hst_test),
        "aligned_comparator": (
            primary_comparator_validation,
            primary_comparator_test,
        ),
        "temporal_hst": (
            chronological_validation_fusion,
            chronological_fusion,
        ),
        "external_hst": (external_source_validation, external_target),
    }
    calibrated_validations: dict[str, publication.AuthenticatedTable] = {}
    calibrated_evaluations: dict[str, publication.AuthenticatedTable] = {}
    calibration_audits: list[pd.DataFrame] = []
    for series, (validation_table, evaluation_table) in calibration_pairs.items():
        calibrated_validation, calibrated_evaluation, calibration_audit = (
            publication.derive_source_platt_calibrated_pair(
                validation_table,
                evaluation_table,
                source_name=series,
                analysis_plan=analysis_plan,
            )
        )
        calibrated_validations[series] = calibrated_validation
        calibrated_evaluations[series] = calibrated_evaluation
        calibration_audits.append(calibration_audit)
    source_platt_calibration_audit = pd.concat(
        calibration_audits,
        ignore_index=True,
        sort=False,
    )

    comparisons = [
        publication.PublicationComparison(
            publication.PRIMARY_ESTIMAND_ID,
            primary_hst_test,
            primary_comparator_test,
        ),
        publication.PublicationComparison(
            "secondary_hst_vs_comparator_uniform_cough_speech_auprc",
            primary_hst_test,
            primary_comparator_test,
        ),
        publication.PublicationComparison(
            "secondary_hybrid_vs_hst_auroc",
            hybrid_test,
            primary_hst_test,
        ),
        publication.PublicationComparison(
            "secondary_hybrid_vs_comparator_auroc",
            hybrid_test,
            primary_comparator_test,
        ),
        publication.PublicationComparison(
            "split_policy_temporal_contrast",
            calendar_fusion,
            chronological_fusion,
            common_test=False,
        ),
        publication.PublicationComparison(
            "common_late_temporal_contrast",
            common_mixed_fusion,
            common_chronological_fusion,
            common_test=True,
        ),
        publication.PublicationComparison(
            "coswara_to_coughvid_external_transfer",
            external_source_test,
            external_target,
            ensemble_right=calibrated_evaluations["external_hst"],
        ),
    ]
    bootstrap = publication.build_bootstrap_evidence(
        comparisons,
        analysis_plan=analysis_plan,
    )
    fusion_vs_constituent, constituent_selection = (
        publication.build_fusion_vs_best_constituent_evidence(
            primary_hst_test,
            constituent_validation,
            constituent_test,
            analysis_plan=analysis_plan,
        )
    )
    bootstrap = pd.concat(
        [bootstrap, fusion_vs_constituent],
        ignore_index=True,
        sort=False,
    )
    bootstrap = publication.adjust_secondary_holm(
        bootstrap,
        analysis_plan=analysis_plan,
    )
    delong = publication.build_paired_delong_evidence(
        comparisons[0],
        analysis_plan=analysis_plan,
    )
    base_calibration_estimands = {
        "internal_hst": publication.PRIMARY_ESTIMAND_ID,
        "aligned_comparator": publication.PRIMARY_ESTIMAND_ID,
        "temporal_hst": "split_policy_temporal_contrast",
        "external_hst": "coswara_to_coughvid_external_transfer",
    }
    raw_calibration_sources = {
        "internal_hst": primary_hst_test,
        "aligned_comparator": primary_comparator_test,
        "temporal_hst": chronological_fusion,
        "external_hst": external_target,
    }
    calibration_sources: dict[str, publication.AuthenticatedTable] = {}
    calibration_estimands: dict[str, str] = {}
    for series, estimand_id in base_calibration_estimands.items():
        calibration_sources[f"{series}_platt"] = calibrated_evaluations[series]
        calibration_estimands[f"{series}_platt"] = estimand_id
        calibration_sources[f"{series}_raw"] = raw_calibration_sources[series]
        calibration_estimands[f"{series}_raw"] = estimand_id
    calibration_bins, calibration_summary = publication.build_calibration_evidence(
        calibration_sources,
        analysis_plan=analysis_plan,
        evidence_estimand_ids=calibration_estimands,
    )
    clinical_utility_sources, clinical_utility_estimands, clinical_utility_audit = (
        _clinical_utility_scope(calibration_sources, calibration_estimands)
    )
    decision_curve = publication.build_decision_curve_evidence(
        clinical_utility_sources,
        analysis_plan=analysis_plan,
        evidence_estimand_ids=clinical_utility_estimands,
    )
    operating_frames = []
    for series, estimand_id in base_calibration_estimands.items():
        if series == "external_hst":
            continue
        platt_series = f"{series}_platt"
        operating_frames.append(
            publication.build_fixed_sensitivity_evidence(
                calibrated_validations[series],
                {platt_series: calibrated_evaluations[series]},
                analysis_plan=analysis_plan,
                evidence_estimand_ids={platt_series: estimand_id},
            )
        )
    operating_points = pd.concat(operating_frames, ignore_index=True, sort=False)
    engineering_objective_audit = _build_engineering_objective_audit(
        internal_branch_test.frame,
        primary_hst_test.frame,
    )
    estimand_execution_audit = _build_estimand_execution_audit(
        analysis_plan.frame,
        bootstrap,
    )

    stage_root = pipeline.run_root / "scientific" / "statistics"
    table_root = stage_root / "tables"
    tables = {
        "bootstrap_evidence.csv": bootstrap,
        "paired_delong.csv": delong,
        "calibration_bins.csv": calibration_bins,
        "calibration_summary.csv": calibration_summary,
        "source_platt_calibration_audit.csv": source_platt_calibration_audit,
        "clinical_utility_scope_audit.csv": clinical_utility_audit,
        "fixed_sensitivity_operating_points.csv": operating_points,
        "decision_curve.csv": decision_curve,
        "engineering_objective_audit.csv": engineering_objective_audit,
        "fusion_vs_best_constituent_evidence.csv": fusion_vs_constituent,
        "fusion_constituent_validation_selection.csv": constituent_selection,
        "estimand_execution_audit.csv": estimand_execution_audit,
    }
    output_paths: list[Path] = []
    for filename, frame in tables.items():
        path = table_root / filename
        _atomic_csv(frame, path)
        output_paths.append(path)

    hst_auroc = _mean_fold_auroc(primary_hst_test.frame)
    comparator_auroc = _mean_fold_auroc(primary_comparator_test.frame)
    internal_cough = internal_branch_test.frame.loc[
        internal_branch_test.frame["protocol"].astype(str).eq(
            "hst_literature_aligned_repeated_holdout"
        )
        & internal_branch_test.frame["split"].astype(str).eq("test")
        & internal_branch_test.frame["modality"].astype(str).eq("cough")
        & internal_branch_test.frame["model"].astype(str).eq("hst_base")
    ].copy()
    chronological_cough = chronological.frame.loc[
        chronological.frame["modality"].astype(str).eq("cough")
    ].copy()
    external_cough = external_target.frame.loc[
        external_target.frame["modality"].astype(str).eq("cough")
    ].copy()
    if internal_cough.empty or chronological_cough.empty or external_cough.empty:
        raise ValueError("Cough-matched validation ladder has an empty endpoint")
    figure_frames = {
        "branch_fusion_performance": pd.DataFrame(
            {
                "label": ["HST cough+speech", "Aligned comparator cough+speech"],
                "auroc": [hst_auroc, comparator_auroc],
                "kind": ["fusion", "fusion"],
            }
        ),
        "paired_comparison": pd.DataFrame(
            {
                "label": ["Complete-case test"],
                "hst_auroc": [hst_auroc],
                "comparator_auroc": [comparator_auroc],
            }
        ),
        "validation_ladder": pd.DataFrame(
            {
                "stage": [
                    "Internal cough",
                    "Early-to-late cough",
                    "External cough",
                ],
                "auroc": [
                    _mean_fold_auroc(internal_cough),
                    _mean_fold_auroc(chronological_cough),
                    _mean_fold_auroc(external_cough),
                ],
            }
        ),
        "calibration": calibration_bins,
        "decision_curve": decision_curve,
        "runtime_gpu": _runtime_figure_frame(pipeline),
    }
    figure_sources = {
        "branch_fusion_performance": [primary_hst_test, primary_comparator_test],
        "paired_comparison": [primary_hst_test, primary_comparator_test],
        "validation_ladder": [
            internal_branch_test,
            chronological,
            external_target,
        ],
        "calibration": [
            primary_hst_test,
            primary_comparator_test,
            chronological_fusion,
            external_target,
        ],
        "decision_curve": [
            primary_hst_test,
            primary_comparator_test,
            chronological_fusion,
            external_target,
        ],
        "runtime_gpu": [plan_table],
    }
    authenticated_figures = {
        name: _derived_figure_table(
            frame,
            source_name=name,
            sources=figure_sources[name],
            analysis_plan=analysis_plan,
        )
        for name, frame in figure_frames.items()
    }
    figures_root = stage_root / "figures"
    figure_manifest = publication.build_publication_figures(
        figures_root,
        authenticated_figures,
        analysis_plan=analysis_plan,
    )
    figure_manifest_path = table_root / "figure_manifest.csv"
    _atomic_csv(figure_manifest, figure_manifest_path)
    output_paths.append(figure_manifest_path)
    figure_paths = [Path(value).resolve() for value in figure_manifest["path"]]
    if len(figure_paths) != 12 or {path.suffix for path in figure_paths} != {".svg", ".png"}:
        raise ValueError("Publication figure contract requires six SVG/PNG figure pairs")
    output_paths.extend(figure_paths)
    return {
        "output_paths": output_paths,
        "row_counts": {name.removesuffix(".csv"): len(frame) for name, frame in tables.items()}
        | {"figures": len(figure_manifest)},
        "metadata": {
            "analysis_plan_sha256": analysis_plan.plan_sha256,
            "primary_bootstrap": "paired_participant_cluster",
            "external_bootstrap": "independent_source_target_participant_cluster",
            "holm_scope": "declared_secondary_families_only",
            "figure_formats": ["svg", "png"],
            "engineering_targets_not_selection_rules": True,
            "engineering_audit_generated_after_model_selection": True,
            "test_set_is_not_a_stopping_rule": True,
        },
    }


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _frozen_gradcam_context(
    predictions: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    from .hst_gradcam import select_gradcam_examples

    context = predictions.loc[
        pd.to_numeric(predictions["fold"], errors="raise").astype(int).eq(1)
        & predictions["modality"].astype(str).eq("cough")
        & predictions["split"].astype(str).eq("test")
        & predictions["protocol"].astype(str).eq(
            "hst_literature_aligned_repeated_holdout"
        )
    ].copy()
    if context.empty:
        raise ValueError("Frozen Grad-CAM fold-1 cough held-out context is absent")
    context["threshold"] = float(threshold)
    context["threshold_source"] = "validation_balanced_accuracy"
    annotated = select_gradcam_examples(
        context,
        threshold=float(threshold),
        per_cell=len(context),
    )
    if len(annotated) != len(context):
        raise ValueError("Frozen Grad-CAM context annotation dropped eligible rows")
    return annotated.drop(columns="selection_rule").reset_index(drop=True)


def _select_frozen_gradcam_examples(
    predictions: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    context = _frozen_gradcam_context(predictions, threshold=threshold)
    selected = (
        context.sort_values(
            ["outcome", "selection_confidence", "participant_key", "recording_key"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .groupby("outcome", sort=False, as_index=False)
        .head(1)
        .copy()
    )
    selected["selection_rule"] = (
        "fixed_fold_context_one_highest_confidence_example_per_available_cell"
    )
    return selected.sort_values("outcome", kind="mergesort").reset_index(drop=True)


def _gradcam_showcase_cell_audit(
    context: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if context.empty:
        raise ValueError("Grad-CAM showcase audit requires a frozen context")
    selected_outcomes = set(selected.get("outcome", pd.Series(dtype=str)).astype(str))
    if not selected_outcomes.issubset(set(_GRADCAM_OUTCOMES)):
        raise ValueError("Grad-CAM showcase selection has an invalid outcome")
    rows: list[dict[str, object]] = []
    for outcome in _GRADCAM_OUTCOMES:
        eligible = context.loc[context["outcome"].astype(str).eq(outcome)]
        chosen = selected.loc[selected["outcome"].astype(str).eq(outcome)]
        if len(chosen) > 1:
            raise ValueError("Grad-CAM selected more than one showcase per outcome")
        rows.append(
            {
                "outcome": outcome,
                "available": bool(not eligible.empty),
                "eligible_row_count": int(len(eligible)),
                "selected_row_count": int(len(chosen)),
                "selected_participant_key": "" if chosen.empty else str(chosen.iloc[0]["participant_key"]),
                "selected_recording_key": "" if chosen.empty else str(chosen.iloc[0]["recording_key"]),
                "frozen_protocol": str(context["protocol"].iloc[0]),
                "frozen_fold": int(context["fold"].iloc[0]),
                "frozen_split": str(context["split"].iloc[0]),
                "frozen_modality": str(context["modality"].iloc[0]),
                "missing_cell_is_not_replaced_from_another_fold": bool(eligible.empty),
            }
        )
    return pd.DataFrame(rows)


def _frozen_gradcam_group_rows(
    predictions: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    context = _frozen_gradcam_context(predictions, threshold=threshold)
    return context.loc[context["outcome"].isin(["TP", "TN"])].sort_values(
        ["participant_key", "recording_key"], kind="mergesort"
    ).reset_index(drop=True)


def _gradcam_source_job(
    pipeline: HSTPipeline,
) -> tuple[pd.Series, Path, Path, dict[str, object], str]:
    _stage_receipt_path, receipt, receipt_sha256 = _verified_stage_receipt(
        pipeline, "internal_cv"
    )
    source_path = _receipt_output_file(pipeline, receipt, "/source_checkpoints.csv")
    source = pd.read_csv(source_path, low_memory=False)
    rows = source.loc[
        source["manifest_name"].astype(str).eq("internal")
        & source["modality"].astype(str).eq("cough")
        & pd.to_numeric(source["fold"], errors="raise").astype(int).eq(1)
        & pd.to_numeric(source["seed"], errors="raise").astype(int).eq(_TRACK_A_SEEDS[0])
    ]
    if len(rows) != 1:
        raise ValueError("Grad-CAM requires exactly one frozen fold-1 cough checkpoint")
    row = rows.iloc[0]
    stage_root = (pipeline.run_root / "scientific" / "internal_cv").resolve()
    job_receipt_path = Path(str(row["source_job_receipt_path"])).resolve()
    checkpoint_path = Path(str(row["best_checkpoint_path"])).resolve()
    for path, expected in (
        (job_receipt_path, str(row["source_job_receipt_sha256"])),
        (checkpoint_path, str(row["best_checkpoint_sha256"])),
    ):
        try:
            path.relative_to(pipeline.run_root.resolve())
        except ValueError as exc:
            raise ValueError("Grad-CAM source artifact escaped its run root") from exc
        if not path.is_file() or path.is_symlink() or stable_file_sha256(path) != expected:
            raise ValueError("Grad-CAM source artifact checksum changed")
    if not _validated_reusable_job(
        job_receipt_path,
        job_spec_sha256=str(row["training_job_spec_sha256"]),
        run_id=pipeline.run_id,
        job_id=str(row["training_job_id"]),
        root=stage_root,
    ):
        raise ValueError("Grad-CAM source training job is not a successful frozen job")
    job_outputs = _receipt_output_paths(job_receipt_path, root=stage_root)
    predictions = [path for path in job_outputs if path.name == "recording_predictions.csv"]
    if len(predictions) != 1:
        raise ValueError("Grad-CAM source job lacks one recording-prediction table")
    allowed = {
        (pipeline.run_root / str(relative)).resolve()
        for relative in receipt["output_paths"]  # type: ignore[index]
    }
    if predictions[0] not in allowed or job_receipt_path not in allowed:
        raise ValueError("Grad-CAM source job outputs are absent from the stage receipt")
    return row, checkpoint_path, predictions[0], receipt, receipt_sha256


@_scientific_handler("gradcam")
def _gradcam(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    if pipeline.config.mode != "full":
        raise RuntimeError("Grad-CAM evidence requires the frozen full run")
    from PIL import Image
    from .hst_gradcam import (
        build_gradcam_evidence,
        build_participant_gradcam_summary,
        build_stage_embedding_figure,
        extract_stage_participant_embeddings,
    )
    from .hst_reporting import REPORTING_CONTRACT
    from .hst_spectrograms import image_to_model_tensor

    source, checkpoint_path, predictions_path, internal_receipt, internal_receipt_sha = (
        _gradcam_source_job(pipeline)
    )
    predictions = pd.read_csv(predictions_path, low_memory=False)
    frozen_context = _frozen_gradcam_context(
        predictions,
        threshold=float(source["validation_threshold"]),
    )
    selected = _select_frozen_gradcam_examples(
        predictions,
        threshold=float(source["validation_threshold"]),
    )
    showcase_audit = _gradcam_showcase_cell_audit(frozen_context, selected)
    group_rows = frozen_context.loc[
        frozen_context["outcome"].isin(["TP", "TN"])
    ].sort_values(
        ["participant_key", "recording_key"], kind="mergesort"
    ).reset_index(drop=True)
    _manifest_receipt_path, _manifest_receipt, manifest_receipt_sha = (
        _verified_stage_receipt(pipeline, "manifests")
    )
    manifest_path, manifest, manifest_sha256 = _load_indexed_manifest(
        pipeline, "internal"
    )
    _cache_receipt_path, cache_receipt, cache_receipt_sha = _verified_stage_receipt(
        pipeline, "spectrogram_cache"
    )
    cache_path = _receipt_output_file(
        pipeline, cache_receipt, "/spectrogram_cache_index.csv"
    )
    cache = pd.read_csv(cache_path, low_memory=False)
    cache_keys = ["dataset", "participant_key", "recording_key", "modality"]
    cache_columns = [*cache_keys, "cache_path", "tensor_sha256", "representation_id"]
    missing_cache = sorted(set(cache_columns) - set(cache.columns))
    if missing_cache:
        raise ValueError(f"Grad-CAM cache index misses columns: {missing_cache}")
    bindings = _load_confirmatory_bindings(pipeline)
    model, _initial_audit = load_verified_hst_model(
        model_name="hst_base",
        checkpoint_path=Path(str(bindings["source_checkpoint_path"])),
        hst_repo=_source_path(pipeline.config),
        seed=int(source["seed"]),
    )
    checkpoint, verified_checkpoint_path = _load_verified_checkpoint_with_path(
        checkpoint_path
    )
    if verified_checkpoint_path != checkpoint_path or checkpoint.get("checkpoint_role") != "best":
        raise ValueError("Grad-CAM requires the validation-selected immutable best checkpoint")
    if str(checkpoint.get("training_contract_fingerprint", "")) != str(
        source["training_contract_fingerprint"]
    ):
        raise ValueError("Grad-CAM checkpoint training identity differs from source index")
    model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[attr-defined]
    model = model.to(pipeline.config.device)  # type: ignore[attr-defined]
    if _model_architecture_sha256(model) != str(checkpoint["architecture_sha256"]):
        raise ValueError("Grad-CAM checkpoint architecture identity changed")

    spectrogram_config = HSTSpectrogramConfig.paper_default()

    def attach_verified_inputs(frame: pd.DataFrame) -> pd.DataFrame:
        attached = frame.merge(
            cache[cache_columns],
            on=cache_keys,
            how="left",
            validate="one_to_one",
        )
        if attached[["cache_path", "tensor_sha256"]].isna().any().any():
            raise ValueError("Grad-CAM rows are absent from the verified tensor cache")
        model_inputs: list[object] = []
        images: list[Image.Image] = []
        for row in attached.itertuples(index=False):
            cached_image = load_verified_cached_image(row.cache_path, row.tensor_sha256)
            tensor = image_to_model_tensor(cached_image, spectrogram_config).unsqueeze(0)
            tensor = tensor.to(pipeline.config.device)
            normalized = cached_image - float(cached_image.min())
            denominator = float(normalized.max())
            if denominator > 0:
                normalized = normalized / denominator
            rendered = Image.fromarray(
                np.uint8(np.clip(normalized, 0, 1) * 255), mode="L"
            )
            model_inputs.append(tensor)
            images.append(rendered.convert("RGB"))
        attached["model_input"] = model_inputs
        attached["image"] = images
        return attached

    selected = attach_verified_inputs(selected)
    prepared_group = (
        attach_verified_inputs(group_rows) if not group_rows.empty else group_rows.copy()
    )

    stage_root = pipeline.run_root / "scientific" / "gradcam"
    evidence_root = stage_root / "evidence"
    showcase_root = evidence_root / "showcase"
    showcase_evidence = build_gradcam_evidence(
        model,
        selected,
        output_dir=showcase_root,
    )
    output_paths: list[Path] = []
    selected_path = stage_root / "selected_examples.csv"
    _atomic_csv(selected.drop(columns=["model_input", "image"]), selected_path)
    output_paths.append(selected_path)
    showcase_audit_path = stage_root / "showcase_cell_audit.csv"
    _atomic_csv(showcase_audit, showcase_audit_path)
    output_paths.append(showcase_audit_path)

    group_rows_path = stage_root / "group_summary" / "eligible_correct_rows.csv"
    _atomic_csv(
        prepared_group.drop(columns=["model_input", "image"], errors="ignore"),
        group_rows_path,
    )
    output_paths.append(group_rows_path)

    participant_audit = pd.DataFrame()
    group_evidence = pd.DataFrame()
    group_summary_available = False
    group_summary_reason = "no_correctly_classified_rows"
    if not prepared_group.empty:
        group_evidence_root = evidence_root / "group_summary"
        group_evidence = build_gradcam_evidence(
            model,
            prepared_group,
            output_dir=group_evidence_root,
        )
        group_heatmaps = group_evidence.copy()
        group_heatmaps["heatmap"] = [
            np.load(group_evidence_root / path, allow_pickle=False)
            for path in group_heatmaps["heatmap_path"].astype(str)
        ]
        correct_classes = set(
            labels_to_binary(group_heatmaps["label_binary"]).astype(int).tolist()
        )
        if correct_classes == {0, 1}:
            summary = build_participant_gradcam_summary(
                group_heatmaps,
                bootstrap_replicates=int(REPORTING_CONTRACT["bootstrap_replicates"]),
                seed=int(REPORTING_CONTRACT["bootstrap_seed"]),
            )
            summary_arrays = {
                "negative_mean.npy": summary.negative_mean,
                "positive_mean.npy": summary.positive_mean,
                "mean_difference.npy": summary.mean_difference,
                "ci_low.npy": summary.ci_low,
                "ci_high.npy": summary.ci_high,
            }
            for filename, array in summary_arrays.items():
                path = stage_root / "group_summary" / filename
                _atomic_npy(array, path)
                output_paths.append(path)
            participant_audit = summary.participant_heatmaps.drop(
                columns="heatmap"
            ).copy()
            participant_audit_path = (
                stage_root / "group_summary" / "participant_heatmaps.csv"
            )
            _atomic_csv(participant_audit, participant_audit_path)
            output_paths.append(participant_audit_path)
            group_summary_available = True
            group_summary_reason = ""
        else:
            group_summary_reason = "correct_predictions_do_not_cover_both_labels"

    group_summary_audit = pd.DataFrame(
        [
            {
                "available": group_summary_available,
                "reason": group_summary_reason,
                "eligible_correct_recordings": int(len(prepared_group)),
                "eligible_correct_participants": int(
                    prepared_group["participant_key"].nunique()
                    if not prepared_group.empty
                    else 0
                ),
                "participant_clustered_before_class_summary": True,
                "showcase_rows_used_for_group_summary": bool(
                    selected["outcome"].astype(str).isin(["TP", "TN"]).any()
                ),
                "showcase_error_rows_excluded_from_group_summary": bool(
                    selected["outcome"].astype(str).isin(["FP", "FN"]).any()
                ),
                "frozen_fold": 1,
                "frozen_split": "test",
                "frozen_modality": "cough",
            }
        ]
    )
    group_summary_audit_path = stage_root / "group_summary" / "summary_audit.csv"
    _atomic_csv(group_summary_audit, group_summary_audit_path)
    output_paths.append(group_summary_audit_path)
    output_paths.extend(
        sorted(
            [path for path in evidence_root.rglob("*") if path.is_file()],
            key=lambda path: path.as_posix(),
        )
    )

    loaders = make_hst_dataloaders(
        cache,
        manifest,
        fold=1,
        modality="cough",
        physical_batch_size=1,
        num_workers=0,
        seed=_TRACK_A_SEEDS[0],
        representation_id="paper_logmel_224",
    )
    embeddings = extract_stage_participant_embeddings(model, loaders["test"])
    stage_numbers = embeddings["stage"].astype(str).str.extract(r"(\d+)$")[0]
    if stage_numbers.isna().any():
        raise ValueError("HST stage embeddings have an invalid stage identity")
    final_stage = f"stage_{int(stage_numbers.astype(int).max())}"
    final_embeddings = embeddings.loc[embeddings["stage"].eq(final_stage)].copy()
    embeddings_path = stage_root / "stage_embeddings.csv"
    _atomic_csv(embeddings, embeddings_path)
    output_paths.append(embeddings_path)
    coordinates: pd.DataFrame | None = None
    for suffix in ("svg", "png"):
        figure = build_stage_embedding_figure(
            final_embeddings,
            output_path=stage_root / f"held_out_{final_stage}_embedding.{suffix}",
            method="pca",
            seed=int(REPORTING_CONTRACT["bootstrap_seed"]),
        )
        output_paths.append(figure.output_path)
        if coordinates is None:
            coordinates = figure.coordinates
        else:
            pd.testing.assert_frame_equal(coordinates, figure.coordinates)
    if coordinates is None:
        raise RuntimeError("Stage embedding projection did not return coordinates")
    coordinates_path = stage_root / "stage_embedding_coordinates.csv"
    _atomic_csv(coordinates, coordinates_path)
    output_paths.append(coordinates_path)

    context_path = stage_root / "gradcam_context_manifest.json"
    atomic_write_json(
        context_path,
        {
            "schema_version": 1,
            "run_id": pipeline.run_id,
            "protocol": "hst_literature_aligned_repeated_holdout",
            "fold": 1,
            "seed": _TRACK_A_SEEDS[0],
            "modality": "cough",
            "split": "test",
            "checkpoint_path": checkpoint_path.relative_to(
                pipeline.run_root.resolve()
            ).as_posix(),
            "checkpoint_sha256": stable_file_sha256(checkpoint_path),
            "training_contract_fingerprint": source["training_contract_fingerprint"],
            "validation_threshold": float(source["validation_threshold"]),
            "selection_rule": (
                "fixed_context_one_deterministic_example_per_available_TP_TN_FP_FN_cell"
            ),
            "missing_cells": showcase_audit.loc[
                ~showcase_audit["available"], "outcome"
            ].astype(str).tolist(),
            "missing_cells_replaced_from_other_folds": False,
            "group_summary_population": "all_correctly_classified_rows_in_frozen_context",
            "group_summary_participant_clustered": True,
            "group_summary_available": group_summary_available,
            "group_summary_reason": group_summary_reason,
            "internal_cv_receipt_sha256": internal_receipt_sha,
            "manifest_receipt_sha256": manifest_receipt_sha,
            "manifest_sha256": manifest_sha256,
            "spectrogram_cache_receipt_sha256": cache_receipt_sha,
            "target_layer": "layers[-1].HSTblocks[-1].attn2:input",
            "embedding_stage": final_stage,
            "label_blind_embedding_fit": True,
        },
    )
    output_paths.append(context_path)
    _cleanup_job_device(model, loaders)
    return {
        "output_paths": output_paths,
        "row_counts": {
            "selected_examples": len(selected),
            "showcase_gradcam_evidence": len(showcase_evidence),
            "group_gradcam_evidence": len(group_evidence),
            "group_eligible_correct_recordings": len(prepared_group),
            "participant_heatmaps": len(participant_audit),
            "stage_embeddings": len(embeddings),
            "embedding_participants": int(coordinates["participant_key"].nunique()),
        },
        "metadata": {
            "frozen_context": "internal_track_a_fold_1_cough_test",
            "selection_outcomes": list(_GRADCAM_OUTCOMES),
            "missing_outcomes": showcase_audit.loc[
                ~showcase_audit["available"], "outcome"
            ].astype(str).tolist(),
            "missing_outcomes_replaced": False,
            "group_summary_uses_all_correct_rows": True,
            "group_summary_participant_clustered": True,
            "validation_selected_checkpoint": True,
            "target_layer": "layers[-1].HSTblocks[-1].attn2:input",
            "label_blind_embedding_projection": True,
        },
    }


@_scientific_handler("evidence_pack")
def _evidence_pack(pipeline: HSTPipeline, _stage: str) -> Mapping[str, object]:
    profile = workload_profile_from_scientific_config(
        pipeline.config.scientific_config
    )
    pipeline_stages = tuple(getattr(pipeline, "STAGES", HSTPipeline.STAGES))
    required = tuple(stage for stage in pipeline_stages if stage != "evidence_pack")
    stage_root = pipeline.run_root / "runtime" / "stages"
    stale = stage_root / "evidence_pack.json"
    quarantine = stage_root / ".evidence_pack.previous"
    if quarantine.exists():
        if stale.exists():
            raise FileExistsError(
                "Both active and quarantined evidence-pack receipts exist; "
                "manual integrity review is required"
            )
        if not quarantine.is_file() or quarantine.is_symlink():
            raise ValueError("The evidence-pack receipt quarantine is not a regular file")
        try:
            recovered = json.loads(quarantine.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("The evidence-pack receipt quarantine is corrupt") from exc
        if not isinstance(recovered, dict):
            raise ValueError("The evidence-pack receipt quarantine is not a JSON object")
        unsigned = {
            key: value for key, value in recovered.items() if key != "record_hash"
        }
        if (
            recovered.get("receipt_type") != "hst_stage"
            or recovered.get("status") != "success"
            or recovered.get("run_id") != pipeline.run_id
            or recovered.get("stage") != "evidence_pack"
            or recovered.get("record_hash") != canonical_json_sha256(unsigned)
        ):
            raise ValueError("The evidence-pack receipt quarantine identity is invalid")
        outputs = recovered.get("output_paths")
        checksums = recovered.get("output_checksums")
        if not isinstance(outputs, list) or not outputs or not isinstance(checksums, Mapping):
            raise ValueError("The evidence-pack receipt quarantine has no output contract")
        run_root = pipeline.run_root.resolve()
        for relative_value in outputs:
            relative = Path(str(relative_value)).as_posix()
            candidate = (run_root / relative).resolve()
            try:
                candidate.relative_to(run_root)
            except ValueError as exc:
                raise ValueError(
                    f"Quarantined evidence-pack output escapes run root: {relative}"
                ) from exc
            if not candidate.is_file() or candidate.is_symlink():
                raise FileNotFoundError(
                    f"Quarantined evidence-pack output is missing: {candidate}"
                )
            if stable_file_sha256(candidate) != str(checksums.get(relative, "")):
                raise ValueError(
                    f"Quarantined evidence-pack output checksum changed: {relative}"
                )
        os.replace(quarantine, stale)
    if stale.exists():
        os.replace(stale, quarantine)
    output_path = pipeline.run_root / "evidence" / "hst_evidence_manifest.json"
    try:
        manifest = build_hst_evidence_manifest(
            run_root=pipeline.run_root,
            output_path=output_path,
            required_stages=required,
        )
        if "evidence_pack" in manifest.get("stages", []):
            raise ValueError("Evidence manifest included its own future stage receipt")
        if profile.name == FULL_RELIABILITY_PROFILE:
            engineering_relative = (
                "scientific/statistics/tables/engineering_objective_audit.csv"
            )
            engineering_artifacts = [
                artifact
                for artifact in manifest.get("artifacts", [])
                if isinstance(artifact, Mapping)
                and artifact.get("path") == engineering_relative
                and artifact.get("producer_stages") == ["statistics"]
            ]
            if len(engineering_artifacts) != 1:
                raise ValueError(
                    "Evidence manifest lacks the receipted descriptive engineering-objective audit"
                )
            engineering_path = pipeline.run_root / engineering_relative
            engineering = pd.read_csv(engineering_path)
            required_flags = {
                "targets_not_selection_rules",
                "generated_after_model_selection",
                "test_set_is_not_a_stopping_rule",
            }
            missing_flags = sorted(required_flags - set(engineering.columns))
            if missing_flags:
                raise ValueError(
                    f"Engineering-objective audit misses anti-selection flags: {missing_flags}"
                )
            for column in required_flags:
                normalized = engineering[column].astype(str).str.casefold()
                if engineering.empty or not normalized.eq("true").all():
                    raise ValueError(
                        f"Engineering-objective audit violates descriptive-only flag {column}"
                    )
        else:
            required_capacity_outputs = {
                "scientific/internal_cv/metrics.csv": "internal_cv",
                "scientific/fusion/fusion_metrics.csv": "fusion",
                "scientific/fusion/primary_hst_test_predictions.csv": "fusion",
            }
            artifacts = manifest.get("artifacts", [])
            for relative, producer in required_capacity_outputs.items():
                matches = [
                    artifact
                    for artifact in artifacts
                    if isinstance(artifact, Mapping)
                    and artifact.get("path") == relative
                    and artifact.get("producer_stages") == [producer]
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Capacity evidence manifest lacks required output: {relative}"
                    )
    except Exception:
        if quarantine.exists() and not stale.exists():
            os.replace(quarantine, stale)
        raise
    quarantine.unlink(missing_ok=True)
    metadata: dict[str, object] = {
        "manifest_sha256": manifest["manifest_sha256"],
        "future_receipt_excluded": True,
        "workload_profile": profile.name,
    }
    if profile.name == FULL_RELIABILITY_PROFILE:
        metadata.update(
            {
                "engineering_targets_not_selection_rules": True,
                "engineering_objective_audit_sha256": stable_file_sha256(
                    pipeline.run_root
                    / "scientific"
                    / "statistics"
                    / "tables"
                    / "engineering_objective_audit.csv"
                ),
            }
        )
    else:
        metadata.update(
            {
                "analysis_scope": "internal_cough_speech_model_bank_extension",
                "temporal_or_external_hst_claims_authorized": False,
            }
        )
    return {
        "output_paths": [output_path],
        "row_counts": {
            "artifacts": int(manifest["artifact_count"]),
            "stages": int(manifest["stage_count"]),
        },
        "metadata": metadata,
    }


def _unwired_scientific_stage(_pipeline: HSTPipeline, stage: str) -> Mapping[str, object]:
    raise RuntimeError(
        f"Scientific stage {stage!r} has not yet been connected to its verified implementation"
    )


_IMPLEMENTED_HANDLERS: dict[str, StageHandler] = {
    "preflight": _preflight,
    "data_contracts": _data_contracts,
    "checkpoint": _checkpoint,
    "preprocess_worker_pilot": _preprocess_worker_pilot,
    "spectrogram_cache": _spectrogram_cache,
    "manifests": _manifests,
    "small_smoke": _small_smoke,
    "base_resource_pilot": _base_resource_pilot,
    "internal_cv": _internal_cv,
    "split_policy_contrast": _split_policy_contrast,
    "reverse_temporal": _reverse_temporal,
    "external_transfer": _external_transfer,
    "aligned_comparator": _aligned_comparator,
    "fusion": _fusion,
    "statistics": _statistics,
    "gradcam": _gradcam,
    "evidence_pack": _evidence_pack,
}


def build_scientific_stage_handlers(
    config: HSTPipelineConfig,
) -> dict[str, StageHandler]:
    if config.mode == "smoke":
        smoke_stages = HSTPipeline.STAGES[
            : HSTPipeline.STAGES.index("small_smoke") + 1
        ]
        return {
            stage: _IMPLEMENTED_HANDLERS[stage]
            for stage in smoke_stages
        }
    handlers: dict[str, StageHandler] = {}
    for stage in HSTPipeline.STAGES:
        handler = _IMPLEMENTED_HANDLERS.get(stage, _unwired_scientific_stage)
        if handler is _unwired_scientific_stage:
            def bound(
                pipeline: HSTPipeline,
                _received_stage: str,
                *,
                expected_stage: str = stage,
            ) -> Mapping[str, object]:
                return _unwired_scientific_stage(pipeline, expected_stage)

            _scientific_handler(stage)(bound)
            handler = bound
        handlers[stage] = handler
    return handlers
