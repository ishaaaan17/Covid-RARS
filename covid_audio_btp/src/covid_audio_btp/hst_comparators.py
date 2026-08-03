from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from importlib import metadata as importlib_metadata
import inspect
import json
import marshal
import os
from pathlib import Path
import pickle
import platform
import shutil
import stat
import struct
import subprocess
import sys
from typing import Callable, Iterable
import uuid
import warnings

import numpy as np
import pandas as pd

from covid_audio_btp.hst_data_contracts import (
    aggregate_to_participant,
    assert_prediction_key_contract,
)
from covid_audio_btp.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    labels_to_binary,
)
from covid_audio_btp.hst_protocols import audit_hst_manifest


FROZEN_MODEL_NAMES = (
    "lightgbm_smote_f80",
    "svc_rbf_f60",
    "catboost_smote_f80",
    "xgboost_smote_f80",
)
ENSEMBLE_MODEL_NAME = "top_4_validation_ensemble"
SELECTED_CANDIDATE_MODEL_NAME = "validation_selected_candidate"
_CANONICAL_APPROVAL_RELATIVE_PATH = Path(
    "covid_audio_btp/configs/hst_compare_is10_approval.approved.json"
)
_CANONICAL_ACCEPTED_FREEZES_RELATIVE_PATH = Path(
    "covid_audio_btp/configs/hst_comparator_accepted_freezes.approved.json"
)
_EXECUTABLE_SOURCE_NAMES = (
    "hst_comparators.py",
    "hst_data_contracts.py",
    "hst_protocols.py",
    "metrics.py",
    "strong_baseline.py",
)
_DEPENDENCY_LOCK_RELATIVE_PATHS = (
    Path("covid_audio_btp/requirements-hst.txt"),
    Path("covid_audio_btp/requirements-gpu.txt"),
)
_RECIPE_PACKAGE_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "imbalanced-learn",
    "joblib",
    "threadpoolctl",
    "lightgbm",
    "xgboost",
    "catboost",
)
_THREAD_ENVIRONMENT_VARIABLES = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_FROZEN_RANKER_HYPERPARAMETERS = {
    "class": "lightgbm.LGBMClassifier",
    "n_estimators": 700,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.75,
    "reg_lambda": 2.0,
    "objective": "binary",
    "class_weight": "balanced",
    "random_state": "$context_seed_plus_modality_offset",
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
_FROZEN_MODEL_HYPERPARAMETERS = {
    "lightgbm_smote_f80": {
        "preprocessing": {
            "variance_threshold": 0.0,
            "select_percentile": 80,
            "score_function": "strong_baseline._safe_f_classif",
        },
        "smote": {"k_neighbors": 3, "random_state": "$model_seed"},
        "model": {
            "class": "lightgbm.LGBMClassifier",
            "n_estimators": 900,
            "learning_rate": 0.025,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "objective": "binary",
            "n_jobs": -1,
            "random_state": "$model_seed",
            "verbosity": -1,
        },
    },
    "svc_rbf_f60": {
        "preprocessing": {
            "variance_threshold": 0.0,
            "standard_scaler": {"with_mean": True, "with_std": True},
            "select_percentile": 60,
            "score_function": "strong_baseline._safe_f_classif",
        },
        "calibration": {"class": "CalibratedClassifierCV", "method": "sigmoid", "cv": 3},
        "model": {
            "class": "sklearn.svm.SVC",
            "C": 2.0,
            "gamma": "scale",
            "kernel": "rbf",
            "class_weight": "balanced",
            "random_state": "$model_seed",
        },
    },
    "catboost_smote_f80": {
        "preprocessing": {
            "variance_threshold": 0.0,
            "select_percentile": 80,
            "score_function": "strong_baseline._safe_f_classif",
        },
        "smote": {"k_neighbors": 3, "random_state": "$model_seed"},
        "model": {
            "class": "catboost.CatBoostClassifier",
            "iterations": 900,
            "depth": 5,
            "learning_rate": 0.025,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": "$model_seed",
            "verbose": False,
            "allow_writing_files": False,
        },
    },
    "xgboost_smote_f80": {
        "preprocessing": {
            "variance_threshold": 0.0,
            "select_percentile": 80,
            "score_function": "strong_baseline._safe_f_classif",
        },
        "smote": {"k_neighbors": 3, "random_state": "$model_seed"},
        "model": {
            "class": "xgboost.XGBClassifier",
            "n_estimators": 800,
            "max_depth": 3,
            "learning_rate": 0.025,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2.0,
            "reg_lambda": 2.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": "$model_seed",
        },
    },
}

_KEY_COLUMNS = ("dataset", "participant_key", "recording_key", "modality")
_CONTEXT_COLUMNS = ("protocol", "fold")
_VALID_SPLITS = {"train", "validation", "test", "external_test"}
_NON_FEATURE_COLUMNS = {
    "run_id",
    "protocol",
    "fold",
    "cohort",
    "split",
    "dataset",
    "participant_id",
    "participant_key",
    "recording_id",
    "recording_key",
    "modality",
    "submodality",
    "label_binary",
    "label_source",
    "label_provenance",
    "quality_flag",
    "manual_quality_label",
    "representation",
    "representation_id",
    "eligible",
    "cache_path",
    "audio_path",
    "is_augmented",
    "augmentation_seed",
    "source_recording_id",
    "source_sha256",
    "source_audio_sha256",
    "manifest_sha256",
    "row_content_sha256",
    "content_sha256",
    "audio_sha256",
    "tensor_sha256",
    "preprocessing_hash",
}

RankerFactory = Callable[[int], object]
EstimatorFactory = Callable[[str, int], object]


@dataclass(frozen=True)
class CompareIS10FeatureContract:
    ordered_feature_columns: tuple[str, ...]
    feature_dtypes: tuple[str, ...]
    schema_sha256: str
    missing_policy: str = "train_median"


@dataclass(frozen=True)
class HSTComparatorResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    participant_predictions: pd.DataFrame
    feature_selection: pd.DataFrame
    model_audit: pd.DataFrame
    alignment_audit: pd.DataFrame
    candidate_selection: pd.DataFrame

    @property
    def recording_predictions(self) -> pd.DataFrame:
        return self.predictions

    @property
    def selection(self) -> pd.DataFrame:
        return self.feature_selection


def _classify_evidence_table(frame: pd.DataFrame, *, test_mode: bool) -> pd.DataFrame:
    out = frame.copy()
    out["execution_class"] = "exploratory_test_only" if test_mode else "confirmatory"
    out["confirmatory_eligible"] = not test_mode
    out["test_mode"] = test_mode
    out["reporting_guard"] = (
        "EXPLORATORY_TEST_MODE_DO_NOT_USE_AS_CONFIRMATORY"
        if test_mode
        else "CONFIRMATORY_FROZEN_APPROVAL_REQUIRED"
    )
    return out


def _evidence_domain_sha256(execution_class: str, approval_record_sha256: str) -> str:
    payload = (
        "covid-audio-hst-comparator-evidence-v1\0"
        + str(execution_class)
        + "\0"
        + str(approval_record_sha256)
    )
    return sha256(payload.encode("ascii")).hexdigest()


_REQUIRED_GENERATION_TABLES = {
    "comparator_predictions.csv",
    "comparator_participant_predictions.csv",
    "comparator_metrics.csv",
    "comparator_alignment_audit.csv",
    "comparator_feature_selection.csv",
    "comparator_model_audit.csv",
    "comparator_candidate_selection.csv",
}

_REQUIRED_METRIC_COLUMNS = {
    "auroc",
    "auprc",
    "balanced_accuracy",
    "f1",
    "sensitivity",
    "specificity",
    "brier",
    "ece",
    "nll",
    "threshold",
    "n_samples",
    "run_id",
    "protocol",
    "fold",
    "dataset",
    "split",
    "modality",
    "model",
    "checkpoint_hash",
    "representation",
    "analysis_unit",
    "threshold_source",
    "n_participants",
}


def _read_ascii_json_object(path: Path, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not canonical ASCII JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _verify_generation_metrics(generation_dir: Path) -> None:
    predictions = pd.read_csv(generation_dir / "comparator_predictions.csv")
    stored_participants = pd.read_csv(
        generation_dir / "comparator_participant_predictions.csv"
    )
    stored_metrics = pd.read_csv(generation_dir / "comparator_metrics.csv")
    missing_metrics = sorted(_REQUIRED_METRIC_COLUMNS - set(stored_metrics.columns))
    if missing_metrics:
        raise ValueError(
            f"stored comparator metrics lack required metric schema columns: {missing_metrics}"
        )
    context_columns = ["run_id", "protocol", "fold", "modality", "model"]
    _require_columns(predictions, context_columns, "recording predictions")
    participant_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for _, group in predictions.groupby(context_columns, dropna=False, sort=False):
        _, thresholded, metrics = _threshold_and_metrics(group.copy())
        participant_frames.append(aggregate_comparator_participants(thresholded))
        metric_frames.append(metrics)
    recomputed_participants = pd.concat(participant_frames, ignore_index=True, sort=False)
    recomputed_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)

    participant_columns = [
        column
        for column in (
            "run_id",
            "protocol",
            "fold",
            "dataset",
            "participant_key",
            "split",
            "modality",
            "model",
            "label_binary",
            "probability",
            "n_recordings",
            "threshold",
            "threshold_source",
        )
        if column in recomputed_participants.columns
    ]
    recomputed_missing = sorted(_REQUIRED_METRIC_COLUMNS - set(recomputed_metrics.columns))
    if recomputed_missing:
        raise ValueError(
            f"recomputed comparator metrics lack required metric schema columns: {recomputed_missing}"
        )
    metric_columns = sorted(_REQUIRED_METRIC_COLUMNS)
    try:
        pd.testing.assert_frame_equal(
            stored_participants[participant_columns]
            .sort_values(participant_columns[:8], kind="mergesort")
            .reset_index(drop=True),
            recomputed_participants[participant_columns]
            .sort_values(participant_columns[:8], kind="mergesort")
            .reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        metric_sort = [
            column
            for column in ("run_id", "protocol", "fold", "dataset", "split", "modality", "model")
            if column in metric_columns
        ]
        pd.testing.assert_frame_equal(
            stored_metrics[metric_columns]
            .sort_values(metric_sort, kind="mergesort")
            .reset_index(drop=True),
            recomputed_metrics[metric_columns]
            .sort_values(metric_sort, kind="mergesort")
            .reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(
            "stored participant predictions or metrics differ from recomputation"
        ) from exc

    _verify_generation_candidate_selection(generation_dir, predictions)


def _verify_generation_candidate_selection(
    generation_dir: Path,
    predictions: pd.DataFrame,
) -> None:
    selection = pd.read_csv(generation_dir / "comparator_candidate_selection.csv")
    model_audit = pd.read_csv(generation_dir / "comparator_model_audit.csv")
    context_columns = ["run_id", "protocol", "fold", "modality"]
    score_columns = [
        "candidate_model",
        "validation_auroc",
        "validation_auprc",
        "n_validation_participants",
        "selection_rank",
        "selected",
        "selection_split",
        "selection_primary_metric",
        "selection_tiebreak_metric",
        "selection_final_tiebreak",
    ]
    selection_columns = [
        *context_columns,
        *score_columns,
        "selected_candidate_model",
        "selected_candidate_source_model",
        "selected_candidate_checkpoint_hash",
        "selected_candidate_model_artifact",
    ]
    _require_columns(selection, selection_columns, "candidate selection")
    _require_columns(
        model_audit,
        [
            *context_columns,
            "model",
            "checkpoint_hash",
            "model_artifact",
            "selected_candidate_source_model",
            "candidate_selection_validation_auroc",
            "candidate_selection_validation_auprc",
        ],
        "model audit",
    )
    expected_contexts = predictions[context_columns].drop_duplicates()
    observed_contexts = selection[context_columns].drop_duplicates()
    try:
        pd.testing.assert_frame_equal(
            expected_contexts.sort_values(context_columns, kind="mergesort").reset_index(drop=True),
            observed_contexts.sort_values(context_columns, kind="mergesort").reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError("candidate selection contexts differ from prediction contexts") from exc

    candidate_models = {*FROZEN_MODEL_NAMES, ENSEMBLE_MODEL_NAME}
    prediction_keys = [
        "dataset",
        "participant_key",
        "recording_key",
        "split",
        "label_binary",
    ]
    for context in expected_contexts.to_dict(orient="records"):
        prediction_mask = pd.Series(True, index=predictions.index)
        selection_mask = pd.Series(True, index=selection.index)
        audit_mask = pd.Series(True, index=model_audit.index)
        for column, value in context.items():
            prediction_mask &= predictions[column].astype(str).eq(str(value))
            selection_mask &= selection[column].astype(str).eq(str(value))
            audit_mask &= model_audit[column].astype(str).eq(str(value))
        context_predictions = predictions.loc[prediction_mask].copy()
        candidate_predictions = context_predictions.loc[
            context_predictions["model"].astype(str).isin(candidate_models)
        ].copy()
        selected_source, recomputed_scores = _select_validation_candidate(
            candidate_predictions
        )
        stored_scores = selection.loc[selection_mask, score_columns].copy()
        try:
            pd.testing.assert_frame_equal(
                stored_scores.sort_values("selection_rank", kind="mergesort").reset_index(drop=True),
                recomputed_scores[score_columns]
                .sort_values("selection_rank", kind="mergesort")
                .reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ValueError(
                "candidate selection differs from recomputation on validation predictions"
            ) from exc

        context_selection = selection.loc[selection_mask]
        expected_singletons = {
            "selected_candidate_model": SELECTED_CANDIDATE_MODEL_NAME,
            "selected_candidate_source_model": selected_source,
        }
        for column, expected in expected_singletons.items():
            observed = set(context_selection[column].astype(str))
            if observed != {expected}:
                raise ValueError(
                    "candidate selection differs from recomputation on validation predictions"
                )

        selected_audit = model_audit.loc[
            audit_mask
            & model_audit["model"].astype(str).eq(SELECTED_CANDIDATE_MODEL_NAME)
        ]
        if len(selected_audit) != 1:
            raise ValueError("candidate selection lacks one selected model-audit row")
        selected_audit_row = selected_audit.iloc[0]
        selected_score = recomputed_scores.loc[
            recomputed_scores["candidate_model"].eq(selected_source)
        ].iloc[0]
        if (
            str(selected_audit_row["selected_candidate_source_model"])
            != selected_source
            or not np.isclose(
                float(selected_audit_row["candidate_selection_validation_auroc"]),
                float(selected_score["validation_auroc"]),
                rtol=1e-12,
                atol=1e-12,
            )
            or not np.isclose(
                float(selected_audit_row["candidate_selection_validation_auprc"]),
                float(selected_score["validation_auprc"]),
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "candidate selection model audit differs from validation predictions"
            )
        if set(context_selection["selected_candidate_checkpoint_hash"].astype(str)) != {
            str(selected_audit_row["checkpoint_hash"])
        } or set(context_selection["selected_candidate_model_artifact"].astype(str)) != {
            str(selected_audit_row["model_artifact"])
        }:
            raise ValueError("candidate selection artifact binding differs from model audit")

        selected_predictions = context_predictions.loc[
            context_predictions["model"].astype(str).eq(SELECTED_CANDIDATE_MODEL_NAME),
            [*prediction_keys, "probability"],
        ]
        source_predictions = context_predictions.loc[
            context_predictions["model"].astype(str).eq(selected_source),
            [*prediction_keys, "probability"],
        ]
        try:
            pd.testing.assert_frame_equal(
                selected_predictions.sort_values(prediction_keys, kind="mergesort")
                .reset_index(drop=True),
                source_predictions.sort_values(prediction_keys, kind="mergesort")
                .reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ValueError(
                "selected-candidate probabilities differ from the validation-selected source"
            ) from exc


def _authenticate_confirmatory_generation(
    generation_manifest_path: str | Path,
    current_receipt_path: str | Path,
    *,
    approval_record_path: str | Path,
    trusted_project_repository_root: str | Path,
    accepted_freezes_path: str | Path,
    expected_accepted_freezes_sha256: str,
    runtime_random_state: int,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    manifest_candidate = Path(generation_manifest_path)
    receipt_candidate = Path(current_receipt_path)
    if manifest_candidate.is_symlink() or receipt_candidate.is_symlink():
        raise ValueError("confirmatory generation manifest/receipt must not be symlinks")
    manifest_path = manifest_candidate.resolve(strict=True)
    receipt_path = receipt_candidate.resolve(strict=True)
    if manifest_path.name != "manifest.json" or not manifest_path.is_file():
        raise ValueError("generation manifest must be a regular manifest.json")
    expected_receipt_path = manifest_path.parents[2] / "current.json"
    if receipt_path != expected_receipt_path or not receipt_path.is_file():
        raise ValueError("current.json receipt does not belong to the supplied generation")
    if not _path_is_read_only(manifest_path) or not _path_is_read_only(receipt_path):
        raise ValueError(
            "confirmatory generation manifest and current receipt must be immutable/read-only"
        )
    generation = _read_ascii_json_object(manifest_path, "generation manifest")
    receipt = _read_ascii_json_object(receipt_path, "current.json receipt")
    if set(receipt) != {
        "generation_id",
        "generation_manifest_sha256",
        "receipt_sha256",
    }:
        raise ValueError("current.json receipt schema is invalid")
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if str(receipt["receipt_sha256"]) != _canonical_hash(unsigned_receipt):
        raise ValueError("current.json receipt checksum is invalid")
    generation_id = str(generation.get("generation_id", ""))
    if manifest_path.parent.name != generation_id or str(receipt["generation_id"]) != generation_id:
        raise ValueError("generation identity differs across path, manifest, and receipt")
    manifest_sha256 = sha256(manifest_path.read_bytes()).hexdigest()
    if str(receipt["generation_manifest_sha256"]) != manifest_sha256:
        raise ValueError("current.json receipt does not authenticate the generation manifest")
    if (
        generation.get("execution_class") != "confirmatory"
        or generation.get("confirmatory_eligible") is not True
        or generation.get("test_mode") is not False
    ):
        raise ValueError("exploratory/test-mode generation is not confirmatory evidence")

    approval = _load_trusted_compare_is10_approval_document(
        approval_record_path,
        trusted_project_repository_root=trusted_project_repository_root,
        accepted_freezes_path=accepted_freezes_path,
        expected_accepted_freezes_sha256=expected_accepted_freezes_sha256,
        runtime_random_state=runtime_random_state,
    )
    accepted_generations = approval["accepted_generation_manifests"]
    if not isinstance(accepted_generations, Mapping) or str(
        accepted_generations.get(generation_id, "")
    ) != manifest_sha256:
        raise ValueError(
            "generation manifest hash is absent from the authenticated accepted-freezes file"
        )
    if generation.get("model_names") != list(FROZEN_MODEL_NAMES) or generation.get(
        "ensemble_model"
    ) != ENSEMBLE_MODEL_NAME:
        raise ValueError("generation model bank differs from the frozen models")
    if generation.get("selected_candidate_model") != SELECTED_CANDIDATE_MODEL_NAME:
        raise ValueError("generation lacks the frozen validation-selected candidate endpoint")

    files = generation.get("files")
    if not isinstance(files, Mapping) or not _REQUIRED_GENERATION_TABLES.issubset(files):
        raise ValueError("generation manifest lacks the complete required evidence table set")
    generation_dir = manifest_path.parent
    for relative, descriptor in files.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("generation artifact path escapes the generation directory")
        artifact = generation_dir / relative_path
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"generation artifact is missing or unsafe: {relative}")
        if not _path_is_read_only(artifact):
            raise ValueError(
                f"generation artifact must be immutable/read-only: {relative}"
            )
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"generation checksum descriptor is invalid: {relative}")
        payload = artifact.read_bytes()
        if descriptor["sha256"] != sha256(payload).hexdigest() or descriptor[
            "size_bytes"
        ] != len(payload):
            raise ValueError(f"generation artifact checksum/size mismatch: {relative}")

    model_audit = pd.read_csv(generation_dir / "comparator_model_audit.csv")
    _require_columns(
        model_audit, ("model_artifact", "checkpoint_hash"), "model audit"
    )
    expected_models = set(model_audit["model_artifact"].astype(str))
    manifested_models = {str(path) for path in files if str(path).startswith("models/")}
    if not expected_models or expected_models != manifested_models:
        raise ValueError("generation model artifacts are incomplete or unreferenced")
    required_models = set(FROZEN_MODEL_NAMES) | {
        ENSEMBLE_MODEL_NAME,
        SELECTED_CANDIDATE_MODEL_NAME,
    }
    for _, group in model_audit.groupby(
        ["protocol", "fold", "modality"], dropna=False, sort=False
    ):
        if set(group["model"].astype(str)) != required_models:
            raise ValueError("model audit does not contain the complete frozen model bank")
    for row in model_audit[["model_artifact", "checkpoint_hash"]].drop_duplicates().itertuples(
        index=False
    ):
        if files[str(row.model_artifact)]["sha256"] != str(row.checkpoint_hash):
            raise ValueError("model audit checksum differs from generation model artifact")

    _verify_generation_metrics(generation_dir)
    return generation_dir, generation, approval


def assert_confirmatory_comparator_table(
    table_path: str | Path | pd.DataFrame,
    *,
    generation_manifest_path: str | Path | None = None,
    current_receipt_path: str | Path | None = None,
    approval_record_path: str | Path | None = None,
    trusted_project_repository_root: str | Path | None = None,
    accepted_freezes_path: str | Path | None = None,
    expected_accepted_freezes_sha256: str | None = None,
    runtime_random_state: int | None = None,
) -> pd.DataFrame:
    """Load confirmatory evidence only through its checksummed atomic generation."""
    if isinstance(table_path, pd.DataFrame):
        raise TypeError(
            "confirmatory comparator ingestion requires a file path; a naked DataFrame "
            "cannot cross the reporting trust boundary"
        )
    if generation_manifest_path is None or current_receipt_path is None:
        raise TypeError(
            "generation_manifest_path and current_receipt_path are required for confirmatory ingestion"
        )
    table_candidate = Path(table_path)
    if table_candidate.is_symlink():
        raise ValueError("confirmatory generation files must not use symlink indirection")
    path = table_candidate.resolve(strict=True)

    if any(
        value is None
        for value in (
            approval_record_path,
            trusted_project_repository_root,
            accepted_freezes_path,
            expected_accepted_freezes_sha256,
            runtime_random_state,
        )
    ):
        raise TypeError("confirmatory ingestion requires all trusted approval inputs")
    generation_dir, generation, approval = _authenticate_confirmatory_generation(
        generation_manifest_path,
        current_receipt_path,
        approval_record_path=approval_record_path,
        trusted_project_repository_root=trusted_project_repository_root,
        accepted_freezes_path=accepted_freezes_path,
        expected_accepted_freezes_sha256=str(expected_accepted_freezes_sha256),
        runtime_random_state=int(runtime_random_state),
    )
    if path.parent != generation_dir or path.name not in _REQUIRED_GENERATION_TABLES:
        raise ValueError("table must be a required direct member of the supplied generation")
    expected_bindings = {
        "approval_id": str(approval["approval_id"]),
        "approval_record_sha256": str(approval["approval_record_sha256"]),
        "approval_git_commit": str(approval["approval_git_commit"]),
        "approval_git_blob": str(approval["approval_git_blob"]),
        "executable_recipe_sha256": str(approval["executable_recipe_sha256"]),
    }
    for key, expected in expected_bindings.items():
        if str(generation.get(key, "")) != expected:
            raise ValueError(f"generation manifest {key} differs from the trusted approval")
    expected_domain = _evidence_domain_sha256(
        "confirmatory", expected_bindings["approval_record_sha256"]
    )
    if generation.get("evidence_domain_sha256") != expected_domain:
        raise ValueError("generation manifest is outside the confirmatory cryptographic domain")

    frame = pd.read_csv(path)
    required = {
        "execution_class",
        "confirmatory_eligible",
        "test_mode",
        "reporting_guard",
        "approval_id",
        "approval_record_sha256",
        "approval_git_commit",
        "approval_git_blob",
        "executable_recipe_sha256",
        "evidence_domain_sha256",
    }
    _require_columns(frame, required, "comparator evidence table")
    eligible = frame["confirmatory_eligible"].eq(True)  # noqa: E712
    confirmatory = frame["execution_class"].astype(str).eq("confirmatory")
    production = frame["test_mode"].eq(False)  # noqa: E712
    frozen = frame["reporting_guard"].astype(str).eq(
        "CONFIRMATORY_FROZEN_APPROVAL_REQUIRED"
    )
    approval = frame["approval_record_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}")
    approval_id = frame["approval_id"].astype(str).str.strip().ne("")
    approval_commit = frame["approval_git_commit"].astype(str).str.fullmatch(
        r"[0-9a-f]{40,64}"
    )
    approval_blob = frame["approval_git_blob"].astype(str).str.fullmatch(r"[0-9a-f]{40,64}")
    recipe = frame["executable_recipe_sha256"].astype(str).eq(
        expected_bindings["executable_recipe_sha256"]
    )
    domain = frame["evidence_domain_sha256"].astype(str).eq(expected_domain)
    bindings = pd.Series(True, index=frame.index)
    for key, expected in expected_bindings.items():
        bindings &= frame[key].astype(str).eq(expected)
    if frame.empty or not (
        eligible
        & confirmatory
        & production
        & frozen
        & approval
        & approval_id
        & approval_commit
        & approval_blob
        & recipe
        & domain
        & bindings
    ).all():
        raise ValueError("comparator table is not eligible for confirmatory reporting")
    return frame


def load_verified_compare_is10_bundle(
    bundle_path: str | Path,
    *,
    generation_manifest_path: str | Path,
    current_receipt_path: str | Path,
    approval_record_path: str | Path,
    trusted_project_repository_root: str | Path,
    accepted_freezes_path: str | Path,
    expected_accepted_freezes_sha256: str,
    runtime_random_state: int,
) -> dict[str, object]:
    """Authenticate a complete generation before loading any model pickle."""
    generation_dir, generation, approval = _authenticate_confirmatory_generation(
        generation_manifest_path,
        current_receipt_path,
        approval_record_path=approval_record_path,
        trusted_project_repository_root=trusted_project_repository_root,
        accepted_freezes_path=accepted_freezes_path,
        expected_accepted_freezes_sha256=expected_accepted_freezes_sha256,
        runtime_random_state=runtime_random_state,
    )
    candidate = Path(bundle_path)
    if candidate.is_symlink():
        raise ValueError("model bundle must not use symlink indirection")
    path = candidate.resolve(strict=True)
    if generation_dir not in path.parents:
        raise ValueError("model bundle is outside the authenticated generation")
    relative = path.relative_to(generation_dir).as_posix()
    files = generation["files"]
    if not isinstance(files, Mapping) or relative not in files or not relative.startswith(
        "models/"
    ):
        raise ValueError("model bundle is absent from the authenticated generation")

    model_audit = pd.read_csv(generation_dir / "comparator_model_audit.csv")
    rows = model_audit[model_audit["model_artifact"].astype(str).eq(relative)]
    if len(rows) != 1:
        raise ValueError("model bundle does not resolve to one model-audit row")
    row = rows.iloc[0]
    if str(row["checkpoint_hash"]) != str(files[relative]["sha256"]):
        raise ValueError("model bundle hash differs from the model audit")
    bundle = pickle.loads(path.read_bytes())
    if not isinstance(bundle, dict) or bundle.get("bundle_version") != 3:
        raise ValueError("model bundle schema/version is invalid")
    required = {
        "threshold",
        "threshold_source",
        "model_identity",
        "protocol",
        "fold",
        "modality",
        "cohort",
        "model_seed",
        "manifest_sha256",
        "datasets",
        "splits",
        "label_mapping",
        "approval_record_sha256",
        "executable_recipe_sha256",
        "member_artifacts",
    }
    if not required.issubset(bundle):
        raise ValueError("model bundle lacks complete protocol/provenance fields")
    if bundle["label_mapping"] != {"negative": 0, "positive": 1}:
        raise ValueError("model bundle label mapping differs from the frozen contract")
    expected_fields = {
        "protocol": row["protocol"],
        "fold": row["fold"],
        "modality": row["modality"],
        "cohort": row["cohort"],
        "model_seed": row["random_state"],
        "manifest_sha256": row["manifest_sha256"],
        "approval_record_sha256": approval["approval_record_sha256"],
        "executable_recipe_sha256": approval["executable_recipe_sha256"],
    }
    for key, expected in expected_fields.items():
        if str(bundle[key]) != str(expected):
            raise ValueError(f"model bundle {key} differs from authenticated model audit")
    identity = bundle["model_identity"]
    if not isinstance(identity, Mapping) or str(identity.get("name", "")) != str(row["model"]):
        raise ValueError("model bundle identity differs from authenticated model audit")

    loaded = dict(bundle)
    loaded["verified_generation_manifest_sha256"] = sha256(
        Path(generation_manifest_path).read_bytes()
    ).hexdigest()
    members = bundle["member_artifacts"]
    if str(row["model"]) == ENSEMBLE_MODEL_NAME:
        if not isinstance(members, Mapping) or set(members) != set(FROZEN_MODEL_NAMES):
            raise ValueError("ensemble bundle lacks its complete frozen member map")
        verified_members: dict[str, dict[str, object]] = {}
        for model_name, descriptor in members.items():
            if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
                raise ValueError("ensemble member descriptor schema is invalid")
            member_relative = str(descriptor["path"])
            if member_relative not in files or files[member_relative]["sha256"] != str(
                descriptor["sha256"]
            ):
                raise ValueError("ensemble member is not authenticated by the generation")
            member_rows = model_audit[
                model_audit["model_artifact"].astype(str).eq(member_relative)
                & model_audit["model"].astype(str).eq(str(model_name))
            ]
            if len(member_rows) != 1:
                raise ValueError("ensemble member does not resolve in the model audit")
            member_path = generation_dir / member_relative
            if member_path.is_symlink() or not member_path.is_file():
                raise ValueError("ensemble member bundle is missing or unsafe")
            member_bundle = pickle.loads(member_path.read_bytes())
            if not isinstance(member_bundle, dict) or member_bundle.get("bundle_version") != 3:
                raise ValueError("ensemble member bundle schema/version is invalid")
            member_identity = member_bundle.get("model_identity")
            if not isinstance(member_identity, Mapping) or str(
                member_identity.get("name", "")
            ) != str(model_name):
                raise ValueError("ensemble member bundle identity is invalid")
            for key in (
                "protocol",
                "fold",
                "modality",
                "cohort",
                "manifest_sha256",
                "approval_record_sha256",
                "executable_recipe_sha256",
                "datasets",
                "splits",
                "label_mapping",
            ):
                if member_bundle.get(key) != bundle.get(key):
                    raise ValueError(f"ensemble member {key} differs from ensemble bundle")
            verified_members[str(model_name)] = member_bundle
        loaded["verified_member_bundles"] = verified_members
    elif str(row["model"]) == SELECTED_CANDIDATE_MODEL_NAME:
        if not isinstance(members, Mapping) or len(members) != 1:
            raise ValueError("selected-candidate bundle must authenticate exactly one source endpoint")
        selected_name, descriptor = next(iter(members.items()))
        if selected_name not in {*FROZEN_MODEL_NAMES, ENSEMBLE_MODEL_NAME}:
            raise ValueError("selected-candidate bundle references an invalid source endpoint")
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
            raise ValueError("selected-candidate source descriptor schema is invalid")
        selected_relative = str(descriptor["path"])
        if selected_relative not in files or files[selected_relative]["sha256"] != str(
            descriptor["sha256"]
        ):
            raise ValueError("selected-candidate source is not authenticated by the generation")
        selected_rows = model_audit[
            model_audit["model_artifact"].astype(str).eq(selected_relative)
            & model_audit["model"].astype(str).eq(str(selected_name))
        ]
        if len(selected_rows) != 1:
            raise ValueError("selected-candidate source does not resolve in the model audit")
        selected_bundle = load_verified_compare_is10_bundle(
            generation_dir / selected_relative,
            generation_manifest_path=generation_manifest_path,
            current_receipt_path=current_receipt_path,
            approval_record_path=approval_record_path,
            trusted_project_repository_root=trusted_project_repository_root,
            accepted_freezes_path=accepted_freezes_path,
            expected_accepted_freezes_sha256=expected_accepted_freezes_sha256,
            runtime_random_state=runtime_random_state,
        )
        loaded["verified_member_bundles"] = {str(selected_name): selected_bundle}
    elif members:
        raise ValueError("non-ensemble bundle unexpectedly declares member artifacts")
    else:
        loaded["verified_member_bundles"] = {}
    return loaded


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("ascii")).hexdigest()


def _stable_config_value(value: object) -> object:
    if isinstance(value, np.generic):
        return _stable_config_value(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _stable_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_config_value(item) for item in value]
    if callable(value):
        return {
            "callable_module": str(getattr(value, "__module__", "")),
            "callable_qualname": str(
                getattr(value, "__qualname__", getattr(value, "__name__", ""))
            ),
        }
    return repr(value)


def _object_metadata(value: object) -> dict[str, str]:
    if hasattr(value, "get_params"):
        config = value.get_params(deep=True)
    else:
        config = {
            key: item
            for key, item in vars(value).items()
            if not key.endswith("_") and not key.startswith("_")
        }
    return {
        "class": type(value).__name__,
        "module": type(value).__module__,
        "config": json.dumps(
            _stable_config_value(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    }


def _callable_code_identity(value: Callable[..., object]) -> dict[str, str]:
    code = getattr(value, "__code__", None)
    if code is None:
        raise ValueError("frozen comparator factory must expose Python executable code")
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise ValueError("frozen comparator factory source is not inspectable") from exc
    return {
        "module": str(getattr(value, "__module__", "")),
        "qualname": str(getattr(value, "__qualname__", getattr(value, "__name__", ""))),
        "source_sha256": sha256(source).hexdigest(),
        "code_sha256": sha256(marshal.dumps(code)).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"frozen recipe file must be a non-symlink regular file: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _build_compare_is10_executable_recipe(
    trusted_project_repository_root: str | Path,
    *,
    random_state: int,
    accepted_environment_lock_sha256: str,
) -> dict[str, object]:
    """Build the complete runtime recipe that an accepted approval must freeze."""
    root = Path(trusted_project_repository_root).resolve(strict=True)
    module_source_root = root / "covid_audio_btp" / "src" / "covid_audio_btp"
    executable_source_sha256 = {
        name: _sha256_file(module_source_root / name) for name in _EXECUTABLE_SOURCE_NAMES
    }
    dependency_lock_files_sha256 = {
        path.name: _sha256_file(root / path) for path in _DEPENDENCY_LOCK_RELATIVE_PATHS
    }
    dependency_lock_sha256 = _canonical_hash(dependency_lock_files_sha256)
    environment_lock_sha256 = str(accepted_environment_lock_sha256).strip().lower()
    if len(environment_lock_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in environment_lock_sha256
    ):
        raise ValueError("accepted environment-lock hash must be a lowercase SHA-256")

    package_versions: dict[str, str] = {}
    for distribution in _RECIPE_PACKAGE_DISTRIBUTIONS:
        try:
            package_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise ValueError(
                f"confirmatory comparator dependency is not installed: {distribution}"
            ) from exc

    try:
        blas_runtime_raw = np.__config__.show(mode="dicts")
    except TypeError:
        blas_runtime_raw = getattr(np.__config__, "CONFIG", {})
    blas_runtime = _stable_config_value(blas_runtime_raw)
    python_runtime = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:5]),
        "cache_tag": str(getattr(sys.implementation, "cache_tag", "")),
        "abi_flags": str(getattr(sys, "abiflags", "")),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }
    thread_environment = {
        name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_VARIABLES
    }

    recipe: dict[str, object] = {
        "recipe_version": 2,
        "random_state": int(random_state),
        "context_seed_policy": "sha256(base_seed,protocol,fold)+base_seed_mod_2^31-1",
        "model_seed_policy": "context_seed+model_bank_offset_mod_2^31-1",
        "model_names": list(FROZEN_MODEL_NAMES),
        "model_hyperparameters": _stable_config_value(_FROZEN_MODEL_HYPERPARAMETERS),
        "ranker_name": "lightgbm",
        "ranker_hyperparameters": _stable_config_value(_FROZEN_RANKER_HYPERPARAMETERS),
        "ranker_factory_identity": _callable_code_identity(_lightgbm_ranker),
        "estimator_factory_identity": _callable_code_identity(_default_estimator_factory),
        "selected_feature_k": 800,
        "selection_scope": "per_modality_mean",
        "selection_split": "train",
        "selection_metric": "auroc",
        "optuna_trials": 0,
        "requested_ensemble_cap": 5,
        "ensemble_members": list(FROZEN_MODEL_NAMES),
        "ensemble_policy": "uniform_probability_mean",
        "ensemble_weights": {name: 0.25 for name in FROZEN_MODEL_NAMES},
        "selected_candidate_model": SELECTED_CANDIDATE_MODEL_NAME,
        "selected_candidate_pool": [*FROZEN_MODEL_NAMES, ENSEMBLE_MODEL_NAME],
        "selected_candidate_policy": {
            "selection_split": "validation",
            "primary_metric": "auroc",
            "tiebreak_metric": "auprc",
            "final_tiebreak": "model_name_ascending",
        },
        "threshold_policy": "validation_participant_balanced_accuracy",
        "missing_policy": "train_median",
        "executable_source_sha256": executable_source_sha256,
        "dependency_lock_files_sha256": dependency_lock_files_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "package_versions": package_versions,
        "package_versions_sha256": _canonical_hash(package_versions),
        "python_runtime": python_runtime,
        "python_runtime_sha256": _canonical_hash(python_runtime),
        "thread_environment": thread_environment,
        "thread_environment_sha256": _canonical_hash(thread_environment),
        "blas_runtime": blas_runtime,
        "blas_runtime_sha256": _canonical_hash(blas_runtime),
    }
    recipe["recipe_sha256"] = _canonical_hash(recipe)
    return recipe


def _feature_schema_hash(
    columns: tuple[str, ...],
    dtypes: tuple[str, ...],
    missing_policy: str,
) -> str:
    return _canonical_hash(
        {
            "ordered_feature_columns": list(columns),
            "feature_dtypes": list(dtypes),
            "missing_policy": missing_policy,
        }
    )


def build_compare_is10_feature_contract(
    features: pd.DataFrame,
    *,
    ordered_feature_columns: Iterable[str],
    declared_feature_dtypes: Mapping[str, str] | None = None,
) -> CompareIS10FeatureContract:
    """Freeze the semantic numeric schema before fold-local model execution."""
    columns = tuple(str(column) for column in ordered_feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("ordered feature columns must be non-empty and unique")
    _require_columns(features, columns, "features")
    if declared_feature_dtypes is None:
        dtypes = tuple(str(features[column].dtype) for column in columns)
    else:
        if set(declared_feature_dtypes) != set(columns):
            raise ValueError("declared feature dtypes must match the ordered feature columns")
        dtypes = tuple(str(declared_feature_dtypes[column]) for column in columns)
    for column, dtype in zip(columns, dtypes):
        try:
            numeric = np.dtype(dtype).kind in "biuf"
        except TypeError:
            numeric = False
        if not numeric:
            raise ValueError(f"feature contract requires a numeric semantic dtype for {column!r}")
    return CompareIS10FeatureContract(
        ordered_feature_columns=columns,
        feature_dtypes=dtypes,
        schema_sha256=_feature_schema_hash(columns, dtypes, "train_median"),
        missing_policy="train_median",
    )


def _normalize_feature_contract(
    value: CompareIS10FeatureContract | Mapping[str, object],
) -> CompareIS10FeatureContract:
    if isinstance(value, CompareIS10FeatureContract):
        contract = value
    elif isinstance(value, Mapping):
        contract = CompareIS10FeatureContract(
            ordered_feature_columns=tuple(str(item) for item in value["ordered_feature_columns"]),
            feature_dtypes=tuple(str(item) for item in value["feature_dtypes"]),
            schema_sha256=str(value["schema_sha256"]),
            missing_policy=str(value.get("missing_policy", "train_median")),
        )
    else:
        raise TypeError("feature_contract must be a CompareIS10FeatureContract or mapping")
    if len(contract.ordered_feature_columns) != len(contract.feature_dtypes):
        raise ValueError("feature contract columns and dtypes have different lengths")
    if contract.missing_policy != "train_median":
        raise ValueError("feature contract missing_policy must be 'train_median'")
    calculated = _feature_schema_hash(
        contract.ordered_feature_columns,
        contract.feature_dtypes,
        contract.missing_policy,
    )
    if contract.schema_sha256 != calculated:
        raise ValueError("feature contract schema_sha256 does not match its canonical schema")
    if len(set(contract.ordered_feature_columns)) != len(contract.ordered_feature_columns):
        raise ValueError("feature contract contains duplicate ordered feature columns")
    for column, dtype in zip(contract.ordered_feature_columns, contract.feature_dtypes):
        try:
            numeric = np.dtype(dtype).kind in "biuf"
        except TypeError:
            numeric = False
        if not numeric:
            raise ValueError(f"feature contract has non-numeric dtype {dtype!r} for {column!r}")
    return contract


def _typed_raw_value(value: object, declared_dtype: object) -> bytes:
    try:
        numpy_dtype = np.dtype(declared_dtype)
    except TypeError:
        numpy_dtype = None
    if numpy_dtype is not None and numpy_dtype.kind in "biufc":
        canonical_dtype = numpy_dtype.newbyteorder("<")
        raw = np.asarray(value, dtype=canonical_dtype).reshape(()).tobytes()
        return b"numpy:" + canonical_dtype.str.encode("ascii") + b":" + raw
    if value is None:
        return b"missing:none"
    if value is pd.NA:
        return b"missing:pd.NA"
    if value is pd.NaT:
        return b"missing:pd.NaT"
    if isinstance(value, np.generic):
        dtype = value.dtype.newbyteorder("<")
        raw = np.asarray(value, dtype=dtype).reshape(()).tobytes()
        return b"numpy-scalar:" + dtype.str.encode("ascii") + b":" + raw
    if isinstance(value, bool):
        return b"python:bool:" + (b"1" if value else b"0")
    if isinstance(value, int):
        return b"python:int:" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"python:float64:" + struct.pack("<d", value)
    if isinstance(value, Decimal):
        return b"python:decimal:" + str(value.as_tuple()).encode("ascii")
    if isinstance(value, str):
        return b"python:str:" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"python:bytes:" + value
    if isinstance(value, pd.Timestamp):
        timezone = "" if value.tz is None else str(value.tz)
        return f"pandas:Timestamp:{value.value}:{timezone}".encode("utf-8")
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    return b"python:other:" + type_name.encode("utf-8") + b":" + repr(value).encode(
        "utf-8"
    )


def _update_length_prefixed(digest: object, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def compare_is10_feature_artifact_sha256(features: pd.DataFrame) -> str:
    """Stream a type- and bit-exact SHA-256 in canonical recording-key order."""
    _require_columns(features, _KEY_COLUMNS, "features")
    ordered = features.sort_values(list(_KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    descriptor = json.dumps(
        {
            "columns": [str(column) for column in ordered.columns],
            "dtypes": [str(dtype) for dtype in ordered.dtypes],
            "n_rows": len(ordered),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest = sha256()
    _update_length_prefixed(digest, b"covid-audio-typed-feature-artifact-v1")
    _update_length_prefixed(digest, descriptor)
    for column in ordered.columns:
        _update_length_prefixed(digest, str(column).encode("utf-8"))
        _update_length_prefixed(digest, str(ordered[column].dtype).encode("ascii"))
    for row_index in range(len(ordered)):
        _update_length_prefixed(digest, b"row")
        for column in ordered.columns:
            value = ordered[column].iloc[row_index]
            _update_length_prefixed(
                digest, _typed_raw_value(value, ordered[column].dtype)
            )
    return digest.hexdigest()


def _approval_protocol_binding_sha256(manifest: pd.DataFrame) -> str:
    columns = [
        "protocol",
        "fold",
        "cohort",
        "split",
        "dataset",
        "participant_key",
        "recording_key",
        "modality",
        "label_binary",
        "source_audio_sha256",
        "analysis_scope",
        "analysis_role",
        "estimand_id",
        "multiplicity_family",
        "analysis_mode",
        "confirmatory_protocol",
    ]
    _require_columns(manifest, columns, "manifest approval binding")
    ordered = manifest[columns].astype(str).sort_values(columns, kind="mergesort")
    return _canonical_hash(ordered.to_dict(orient="records"))


def _path_is_read_only(path: Path) -> bool:
    metadata = path.stat()
    mode_read_only = not bool(
        stat.S_IMODE(metadata.st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    )
    attributes = getattr(metadata, "st_file_attributes", 0)
    windows_read_only = bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_READONLY", 0)
    )
    return mode_read_only or windows_read_only


def _git_text(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(
            "approval record must be a clean, tracked, version-controlled Git artifact"
            + (f": {detail}" if detail else "")
        )
    return result.stdout.strip()


def _trusted_project_git_root(trusted_project_repository_root: str | Path) -> Path:
    candidate = Path(trusted_project_repository_root)
    if candidate.is_symlink():
        raise ValueError("trusted project repository root must not be a symlink")
    root = candidate.resolve(strict=True)
    supplied_git_root = Path(_git_text("rev-parse", "--show-toplevel", cwd=root)).resolve(
        strict=True
    )
    module_path = Path(__file__)
    if module_path.is_symlink():
        raise ValueError("comparator executable source must not be a symlink")
    module_root = Path(
        _git_text("rev-parse", "--show-toplevel", cwd=module_path.resolve(strict=True).parent)
    ).resolve(strict=True)
    if root != supplied_git_root or root != module_root:
        raise ValueError(
            "trusted project repository root must be the Git repository containing "
            "the executing comparator source"
        )
    return root


def _git_frozen_artifact_identity(
    path: Path,
    *,
    trusted_project_repository_root: str | Path,
    accepted_approval_commit_sha: str,
) -> tuple[str, str]:
    root = _trusted_project_git_root(trusted_project_repository_root)
    canonical_path = root / _CANONICAL_APPROVAL_RELATIVE_PATH
    if path != canonical_path:
        raise ValueError(
            "comparator approval record must use the canonical approval path under configs"
        )
    relative = _CANONICAL_APPROVAL_RELATIVE_PATH.as_posix()

    accepted_commit = str(accepted_approval_commit_sha).strip().lower()
    if not accepted_commit or not all(character in "0123456789abcdef" for character in accepted_commit):
        raise ValueError("accepted approval commit SHA is invalid")
    try:
        accepted_commit = _git_text(
            "rev-parse", "--verify", f"{accepted_commit}^{{commit}}", cwd=root
        )
    except ValueError as exc:
        raise ValueError("accepted approval commit SHA is not present in the trusted repository") from exc
    ancestry = subprocess.run(
        ("git", "merge-base", "--is-ancestor", accepted_commit, "HEAD"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("current HEAD must descend from the accepted approval commit")

    _git_text("cat-file", "-e", f"{accepted_commit}:{relative}", cwd=root)
    if _git_text("status", "--porcelain=v1", "--untracked-files=all", "--", relative, cwd=root):
        raise ValueError("approval record must be clean and tracked at the accepted Git commit")
    accepted_blob = _git_text("rev-parse", f"{accepted_commit}:{relative}", cwd=root)
    working_blob = _git_text("hash-object", "--", relative, cwd=root)
    if accepted_blob != working_blob:
        raise ValueError("approval record bytes differ from the accepted Git blob")

    blob_result = subprocess.run(
        ("git", "show", f"{accepted_commit}:{relative}"),
        cwd=root,
        capture_output=True,
        check=False,
    )
    if blob_result.returncode != 0 or blob_result.stdout != path.read_bytes():
        raise ValueError("approval record bytes differ from the accepted Git blob")
    return accepted_commit, accepted_blob


def _require_sha256(value: object, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _normalized_git_remote(value: str) -> str:
    normalized = str(value).strip().replace("\\", "/").rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def _git_clean_current_artifact(path: Path, root: Path, relative: Path, name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a non-symlink regular file")
    if path != root / relative:
        raise ValueError(f"{name} must use its canonical project path")
    relative_text = relative.as_posix()
    _git_text("cat-file", "-e", f"HEAD:{relative_text}", cwd=root)
    if _git_text(
        "status", "--porcelain=v1", "--untracked-files=all", "--", relative_text, cwd=root
    ):
        raise ValueError(f"{name} must be a clean Git-tracked immutable artifact")
    head_blob = _git_text("rev-parse", f"HEAD:{relative_text}", cwd=root)
    working_blob = _git_text("hash-object", "--", relative_text, cwd=root)
    if head_blob != working_blob:
        raise ValueError(f"{name} bytes differ from the current Git blob")
    blob_result = subprocess.run(
        ("git", "show", f"HEAD:{relative_text}"),
        cwd=root,
        capture_output=True,
        check=False,
    )
    if blob_result.returncode != 0 or blob_result.stdout != path.read_bytes():
        raise ValueError(f"{name} bytes differ from the current Git blob")
    return head_blob


def _load_authenticated_accepted_freezes(
    accepted_freezes_path: str | Path,
    *,
    expected_accepted_freezes_sha256: str,
    trusted_project_repository_root: str | Path,
) -> dict[str, object]:
    root = _trusted_project_git_root(trusted_project_repository_root)
    candidate = Path(accepted_freezes_path)
    if candidate.is_symlink():
        raise ValueError("accepted-freezes file must not use symlink indirection")
    path = candidate.resolve(strict=True)
    canonical = root / _CANONICAL_ACCEPTED_FREEZES_RELATIVE_PATH
    if path != canonical:
        raise ValueError("accepted-freezes file must use the canonical project path")
    if not _path_is_read_only(path):
        raise ValueError("accepted-freezes file must be immutable/read-only")
    expected_hash = _require_sha256(
        expected_accepted_freezes_sha256, "expected accepted-freezes hash"
    )
    actual_hash = sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("accepted-freezes file hash differs from the controller identity")
    _git_clean_current_artifact(
        path,
        root,
        _CANONICAL_ACCEPTED_FREEZES_RELATIVE_PATH,
        "accepted-freezes file",
    )
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("accepted-freezes file is not canonical ASCII JSON") from exc
    required = {
        "accepted_freezes_version",
        "project_identity",
        "compare_is10_approval",
        "environment_lock",
        "accepted_generation_manifests",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("accepted-freezes file fields differ from the canonical schema")
    if payload["accepted_freezes_version"] != 1:
        raise ValueError("accepted-freezes file version is unsupported")

    project = payload["project_identity"]
    approval = payload["compare_is10_approval"]
    environment = payload["environment_lock"]
    generations = payload["accepted_generation_manifests"]
    if not all(isinstance(value, Mapping) for value in (project, approval, environment, generations)):
        raise ValueError("accepted-freezes nested contracts must be mappings")
    if set(project) != {"project_id", "expected_remote_url", "accepted_ancestor_commit"}:
        raise ValueError("accepted-freezes project identity schema is invalid")
    if not str(project["project_id"]).strip():
        raise ValueError("accepted-freezes project identity is empty")
    ancestor = str(project["accepted_ancestor_commit"]).strip().lower()
    try:
        ancestor = _git_text("rev-parse", "--verify", f"{ancestor}^{{commit}}", cwd=root)
    except ValueError as exc:
        raise ValueError("accepted project ancestor is absent from the trusted repository") from exc
    ancestry = subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, "HEAD"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("current HEAD does not descend from the accepted project ancestor")
    expected_remote = _normalized_git_remote(str(project["expected_remote_url"]))
    if expected_remote:
        try:
            actual_remote = _normalized_git_remote(
                _git_text("remote", "get-url", "origin", cwd=root)
            )
        except ValueError as exc:
            raise ValueError("trusted project remote identity is unavailable") from exc
        if actual_remote != expected_remote:
            raise ValueError("trusted project remote does not match accepted project identity")

    if set(approval) != {"relative_path", "commit_sha"}:
        raise ValueError("accepted comparator approval binding schema is invalid")
    if str(approval["relative_path"]) != _CANONICAL_APPROVAL_RELATIVE_PATH.as_posix():
        raise ValueError("accepted comparator approval path is not canonical")
    approval_commit = str(approval["commit_sha"]).strip().lower()
    if approval_commit != ancestor:
        raise ValueError("accepted approval commit differs from the accepted project ancestor")

    if set(environment) != {"relative_path", "sha256"}:
        raise ValueError("accepted environment-lock binding schema is invalid")
    environment_relative = Path(str(environment["relative_path"]))
    if environment_relative.is_absolute() or ".." in environment_relative.parts:
        raise ValueError("environment-lock path must remain within the trusted project")
    environment_path = (root / environment_relative).resolve(strict=True)
    if root not in environment_path.parents or environment_path.is_symlink():
        raise ValueError("environment-lock path escapes or uses symlink indirection")
    expected_environment_hash = _require_sha256(
        environment["sha256"], "accepted environment-lock hash"
    )
    if sha256(environment_path.read_bytes()).hexdigest() != expected_environment_hash:
        raise ValueError("recomputed environment-lock content differs from accepted hash")
    _git_clean_current_artifact(
        environment_path, root, environment_relative, "environment-lock file"
    )
    payload["accepted_freezes_sha256"] = actual_hash
    payload["accepted_approval_commit"] = approval_commit
    payload["environment_lock_sha256"] = expected_environment_hash
    return payload


def _load_trusted_compare_is10_approval_document(
    approval_record_path: str | Path,
    *,
    trusted_project_repository_root: str | Path,
    accepted_freezes_path: str | Path,
    expected_accepted_freezes_sha256: str,
    runtime_random_state: int,
) -> dict[str, object]:
    accepted = _load_authenticated_accepted_freezes(
        accepted_freezes_path,
        expected_accepted_freezes_sha256=expected_accepted_freezes_sha256,
        trusted_project_repository_root=trusted_project_repository_root,
    )
    frozen_commit = str(accepted["accepted_approval_commit"])
    environment_lock_sha256 = str(accepted["environment_lock_sha256"])
    candidate = Path(approval_record_path)
    if candidate.is_symlink():
        raise ValueError("comparator approval record must not use symlink indirection")
    path = candidate.resolve(strict=True)
    if not path.is_file():
        raise ValueError("comparator approval record must be an immutable regular file")
    if not _path_is_read_only(path):
        raise ValueError("comparator approval record must be immutable/read-only before execution")
    approval_git_commit, approval_git_blob = _git_frozen_artifact_identity(
        path,
        trusted_project_repository_root=trusted_project_repository_root,
        accepted_approval_commit_sha=frozen_commit,
    )
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("comparator approval record is not canonical ASCII JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("comparator approval record must be a JSON object")

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
        "approval_record_sha256",
    }
    approval_version = payload.get("approval_record_version")
    if approval_version == 3:
        required.add("reviewed_input_bindings")
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(
            f"comparator approval record fields differ from the frozen schema; "
            f"missing={missing}, extra={extra}"
        )
    supplied_record_sha256 = str(payload["approval_record_sha256"])
    unsigned = {key: value for key, value in payload.items() if key != "approval_record_sha256"}
    if supplied_record_sha256 != _canonical_hash(unsigned):
        raise ValueError("approval_record_sha256 does not match approval record content")
    if approval_version not in {2, 3} or payload["approval_status"] != "approved":
        raise ValueError("comparator approval record is not an approved supported freeze")
    if approval_version == 3:
        reviewed = payload["reviewed_input_bindings"]
        required_reviewed_hashes = {
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
        }
        if (
            not isinstance(reviewed, Mapping)
            or reviewed.get("manifest_name") != "aligned_comparator"
            or not required_reviewed_hashes.issubset(reviewed)
        ):
            raise ValueError("version-3 approval lacks complete aligned reviewed input bindings")
        for key in required_reviewed_hashes:
            _require_sha256(reviewed[key], f"reviewed approval {key}")
    if not str(payload["approval_id"]).strip() or not str(payload["approved_at_utc"]).endswith("Z"):
        raise ValueError("comparator approval identity/timestamp is invalid")

    frozen_configuration = {
        "selected_feature_k": 800,
        "ranker": "lightgbm",
        "selection_scope": "per_modality_mean",
        "model_names": list(FROZEN_MODEL_NAMES),
        "ensemble_policy": "uniform_probability_mean",
    }
    if payload["comparator_configuration"] != frozen_configuration:
        raise ValueError("approval record comparator_configuration is not the frozen recipe")
    live_recipe = _build_compare_is10_executable_recipe(
        trusted_project_repository_root,
        random_state=int(runtime_random_state),
        accepted_environment_lock_sha256=environment_lock_sha256,
    )
    if payload["executable_recipe"] != live_recipe:
        raise ValueError(
            "approval record executable recipe differs from the runtime seed, factories, "
            "source, dependencies, environment, hyperparameters, or policies"
        )
    payload["executable_recipe_sha256"] = str(live_recipe["recipe_sha256"])
    payload["approval_git_commit"] = approval_git_commit
    payload["approval_git_blob"] = approval_git_blob
    payload["accepted_freezes_sha256"] = str(accepted["accepted_freezes_sha256"])
    payload["accepted_generation_manifests"] = dict(
        accepted["accepted_generation_manifests"]
    )
    return payload


def load_frozen_compare_is10_approval(
    approval_record_path: str | Path,
    *,
    trusted_project_repository_root: str | Path,
    accepted_freezes_path: str | Path,
    expected_accepted_freezes_sha256: str,
    runtime_random_state: int,
    feature_contract: CompareIS10FeatureContract | Mapping[str, object],
    feature_artifact_sha256: str,
    manifest: pd.DataFrame,
) -> dict[str, object]:
    """Load a separately frozen approval record and verify every scientific binding."""
    payload = _load_trusted_compare_is10_approval_document(
        approval_record_path,
        trusted_project_repository_root=trusted_project_repository_root,
        accepted_freezes_path=accepted_freezes_path,
        expected_accepted_freezes_sha256=expected_accepted_freezes_sha256,
        runtime_random_state=runtime_random_state,
    )

    schema = _normalize_feature_contract(feature_contract)
    frozen_manifest = _validate_manifest(manifest)
    manifest_sha256 = str(frozen_manifest["manifest_sha256"].iloc[0])
    scientific = frozen_manifest["scientific_configuration_fingerprint"].astype(str).unique()
    eligibility = frozen_manifest["eligibility_alignment_fingerprint"].astype(str).unique()
    if len(scientific) != 1 or len(eligibility) != 1:
        raise ValueError("manifest approval bindings must be canonical single fingerprints")
    expected = {
        "feature_schema_sha256": schema.schema_sha256,
        "feature_artifact_sha256": str(feature_artifact_sha256),
        "manifest_sha256": manifest_sha256,
        "scientific_configuration_fingerprint": str(scientific[0]),
        "eligibility_alignment_fingerprint": str(eligibility[0]),
        "protocol_binding_sha256": _approval_protocol_binding_sha256(frozen_manifest),
    }
    for key, value in expected.items():
        if str(payload[key]) != value:
            raise ValueError(f"approval record {key} does not match the live frozen input")
    return payload


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _qualified_key_expected(dataset: pd.Series, key: pd.Series) -> pd.Series:
    suffix = key.astype(str).str.split("::", n=1).str[-1]
    return dataset.astype(str) + "::" + suffix


def _validate_qualified_keys(frame: pd.DataFrame, name: str) -> None:
    dataset = frame["dataset"].astype("string").str.strip()
    for column in ("participant_key", "recording_key"):
        values = frame[column].astype("string").str.strip()
        valid = values.str.count("::").eq(1) & values.str.split("::", n=1).str[0].eq(dataset)
        if values.isna().any() or not valid.all():
            raise ValueError(f"{name} requires dataset-qualified {column} values")
        if not values.astype(str).equals(_qualified_key_expected(dataset, values)):
            raise ValueError(f"{name} has conflicting dataset and {column} values")


def _manifest_context_columns(manifest: pd.DataFrame) -> list[str]:
    columns = [*_CONTEXT_COLUMNS]
    if "cohort" in manifest:
        columns.append("cohort")
    return columns


def _validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        *_KEY_COLUMNS,
        *_CONTEXT_COLUMNS,
        "cohort",
        "split",
        "label_binary",
        "manifest_sha256",
    }
    _require_columns(manifest, required, "manifest")
    if manifest.empty:
        raise ValueError("manifest is empty")
    out = manifest.copy()
    _validate_qualified_keys(out, "manifest")
    if not out["label_binary"].isin(["negative", "positive"]).all():
        raise ValueError("manifest contains labels outside the frozen class map")
    if not out["split"].astype(str).isin(_VALID_SPLITS).all():
        raise ValueError("manifest contains an unsupported split")
    manifest_hashes = out["manifest_sha256"].astype("string")
    if (
        manifest_hashes.isna().any()
        or manifest_hashes.nunique(dropna=False) != 1
        or not manifest_hashes.str.fullmatch(r"[0-9a-f]{64}").all()
    ):
        raise ValueError("one canonical manifest_sha256 is required per comparator invocation")
    manifest_audit = audit_hst_manifest(out)
    if manifest_audit.empty or not manifest_audit["manifest_hash_valid"].all():
        raise ValueError("manifest_sha256 does not match canonical manifest recomputation")
    if "invalid_row_hash_count" in manifest_audit and manifest_audit["invalid_row_hash_count"].gt(0).any():
        raise ValueError("manifest row_content_sha256 does not match canonical row recomputation")

    content_columns = [
        column
        for column in (
            "content_sha256",
            "source_sha256",
            "source_audio_sha256",
            "audio_sha256",
            "tensor_sha256",
        )
        if column in out
    ]
    if not content_columns:
        raise ValueError("manifest requires a content/source hash")
    has_content_hash = pd.Series(False, index=out.index)
    for column in content_columns:
        values = out[column].astype("string")
        has_content_hash |= values.notna() & values.str.fullmatch(r"[0-9a-f]{64}")
    if not has_content_hash.all():
        raise ValueError("every manifest row requires a canonical content/source hash")

    context = list(_CONTEXT_COLUMNS)
    if out[list(_KEY_COLUMNS) + context].isna().any(axis=None):
        raise ValueError("manifest contains empty alignment keys")
    duplicate_identity = [*context, "split", "recording_key", "modality"]
    if out.duplicated(duplicate_identity).any():
        raise ValueError("manifest contains duplicate recording identities")

    for keys, unit in out.groupby(context, dropna=False, sort=False):
        if "cohort" in unit and unit["cohort"].nunique(dropna=False) != 1:
            raise ValueError(f"manifest mixes cohorts inside protocol/fold context {keys}")
        split_counts = unit.groupby("participant_key", dropna=False)["split"].nunique(dropna=False)
        if (split_counts > 1).any():
            raise ValueError(f"manifest participant leakage inside protocol/fold context {keys}")
        recording_splits = unit.groupby(["recording_key", "modality"], dropna=False)["split"].nunique()
        if (recording_splits > 1).any():
            raise ValueError(f"manifest recording leakage inside protocol/fold context {keys}")
        for entity in ("participant_key", "recording_key"):
            label_counts = unit.groupby(entity, dropna=False)["label_binary"].nunique(dropna=False)
            if (label_counts > 1).any():
                raise ValueError(f"manifest has conflicting labels for {entity}")
        splits = set(unit["split"].astype(str))
        if not {"train", "validation"} <= splits or not splits.intersection({"test", "external_test"}):
            raise ValueError(f"manifest context {keys} requires train, validation, and test rows")

        content_values: list[pd.DataFrame] = []
        for column in content_columns:
            values = unit[[column, "split"]].dropna().rename(columns={column: "content_hash"})
            values = values[values["content_hash"].astype(str).str.strip().ne("")]
            content_values.append(values)
        all_content = pd.concat(content_values, ignore_index=True)
        leaked = all_content.groupby("content_hash", dropna=False)["split"].nunique(dropna=False)
        if (leaked > 1).any():
            raise ValueError("manifest content leakage detected across content/source hashes")
    return out


def _align_features(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    contract: CompareIS10FeatureContract,
) -> tuple[pd.DataFrame, list[str], int]:
    if "source_audio_sha256" not in features:
        raise ValueError("features require a source audio hash column")
    _require_columns(
        features,
        {*_KEY_COLUMNS, "label_binary", "source_audio_sha256"},
        "features",
    )
    if features.empty:
        raise ValueError("features is empty")
    _validate_qualified_keys(features, "features")
    if not features["label_binary"].isin(["negative", "positive"]).all():
        raise ValueError("features contain labels outside the frozen class map")

    feature_columns = list(contract.ordered_feature_columns)
    _require_columns(features, feature_columns, "features")
    observed_order = tuple(column for column in features.columns if column in set(feature_columns))
    if observed_order != contract.ordered_feature_columns:
        raise ValueError("global feature table does not match the contract's ordered feature columns")
    undeclared = [
        str(column)
        for column in features.columns
        if column not in _NON_FEATURE_COLUMNS and column not in feature_columns
    ]
    if undeclared:
        raise ValueError(f"global table contains undeclared feature columns: {undeclared}")

    join_columns = list(_KEY_COLUMNS)
    if features.duplicated(join_columns).any():
        raise ValueError(f"features contain duplicate alignment rows for {join_columns}")
    recording_mapping = features.groupby(
        ["dataset", "recording_key", "modality"], dropna=False
    ).agg(participants=("participant_key", "nunique"), labels=("label_binary", "nunique"))
    if recording_mapping[["participants", "labels"]].gt(1).any(axis=None):
        raise ValueError("global feature table has conflicting recording identity or labels")

    manifest_keys = manifest[join_columns].drop_duplicates()
    feature_keys = features[join_columns].drop_duplicates()
    missing = manifest_keys.merge(feature_keys, on=join_columns, how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")]
    if not missing.empty:
        raise ValueError("features are missing manifest recordings or have conflicting recording keys")

    labels = manifest[[*join_columns, "label_binary"]].merge(
        features[[*join_columns, "label_binary"]],
        on=join_columns,
        how="left",
        validate="many_to_one",
        suffixes=("_manifest", "_feature"),
    )
    if not labels["label_binary_manifest"].astype(str).equals(labels["label_binary_feature"].astype(str)):
        raise ValueError("manifest and feature labels conflict")

    feature_source_hashes = features["source_audio_sha256"].astype("string")
    if (
        feature_source_hashes.isna().any()
        or not feature_source_hashes.str.fullmatch(r"[0-9a-f]{64}").all()
    ):
        raise ValueError("every feature row requires a canonical source audio hash")

    common_source_hashes = [
        column
        for column in (
            "source_audio_sha256",
            "content_sha256",
            "source_sha256",
            "audio_sha256",
            "tensor_sha256",
        )
        if column in manifest and column in features
    ]
    provenance = manifest[[*join_columns, *common_source_hashes]].merge(
        features[[*join_columns, *common_source_hashes]],
        on=join_columns,
        how="left",
        validate="many_to_one",
        suffixes=("_manifest", "_feature"),
    )
    for column in common_source_hashes:
        manifest_values = provenance[f"{column}_manifest"].astype("string").fillna("")
        feature_values = provenance[f"{column}_feature"].astype("string").fillna("")
        if not manifest_values.equals(feature_values):
            raise ValueError(f"manifest and feature source hash {column!r} conflict")

    payload = features[[*join_columns, *feature_columns]].copy()
    aligned = manifest.merge(payload, on=join_columns, how="left", validate="many_to_one")
    if len(aligned) != len(manifest):
        raise ValueError("manifest/feature mismatch changed the aligned row count")
    included_keys = set(map(tuple, manifest_keys.itertuples(index=False, name=None)))
    excluded_global_row_count = int(
        sum(
            tuple(row) not in included_keys
            for row in feature_keys.itertuples(index=False, name=None)
        )
    )
    return aligned, feature_columns, excluded_global_row_count


def _coerce_declared_numeric(
    values: pd.Series,
    declared_dtype: str,
) -> tuple[pd.Series, pd.Series]:
    dtype = np.dtype(declared_dtype)
    numeric = pd.Series(np.nan, index=values.index, dtype=float)
    invalid = pd.Series(False, index=values.index, dtype=bool)

    for index, value in values.items():
        genuine_missing = (
            value is None
            or value is pd.NA
            or value is pd.NaT
            or (
                isinstance(value, (float, np.floating))
                and bool(np.isnan(value))
            )
        )
        if genuine_missing:
            continue
        try:
            if dtype.kind in "iu":
                text = str(value).strip()
                if not text:
                    raise ValueError("empty numeric text")
                exact = Decimal(text)
                if not exact.is_finite() or exact != exact.to_integral_value():
                    raise ValueError("non-integral integer value")
                integer = int(exact)
                limits = np.iinfo(dtype)
                if integer < int(limits.min) or integer > int(limits.max):
                    raise ValueError("integer outside declared dtype")
                converted = dtype.type(integer)
                as_float = float(converted)
                if not np.isfinite(as_float) or int(as_float) != integer:
                    raise ValueError("integer cannot be represented losslessly by model matrix")
                numeric.at[index] = as_float
            elif dtype.kind == "b":
                if value not in (False, True, 0, 1):
                    raise ValueError("invalid boolean value")
                numeric.at[index] = float(bool(value))
            elif dtype.kind == "f":
                text = str(value).strip()
                if not text:
                    raise ValueError("empty numeric text")
                exact_decimal = Decimal(text)
                if not exact_decimal.is_finite():
                    raise ValueError("non-finite numeric content")
                converted = dtype.type(value)
                as_float = float(converted)
                if not np.isfinite(as_float):
                    raise ValueError("non-finite numeric content")
                source = float(value)
                if not np.isfinite(source) or as_float != source:
                    raise ValueError("floating value changes under declared dtype")
                if isinstance(value, (int, np.integer)) and np.isfinite(as_float):
                    if int(as_float) != int(value):
                        raise ValueError("integer cannot be represented losslessly by model matrix")
                if isinstance(value, (str, Decimal)):
                    binary_decimal = Decimal.from_float(as_float)
                    if binary_decimal != exact_decimal:
                        raise ValueError(
                            "decimal text cannot be represented exactly by declared floating dtype"
                        )
                numeric.at[index] = as_float
            else:
                raise ValueError("declared dtype is not supported")
        except (InvalidOperation, TypeError, ValueError, OverflowError):
            invalid.at[index] = True

    return numeric, invalid


def _training_matrix(
    unit: pd.DataFrame,
    feature_columns: list[str],
    declared_dtypes: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    train = unit[unit["split"].astype(str).eq("train")]
    medians: dict[str, float] = {}
    removed: dict[str, str] = {}
    values: dict[str, pd.Series] = {}
    for column in feature_columns:
        numeric, invalid = _coerce_declared_numeric(train[column], declared_dtypes[column])
        if invalid.any():
            raise ValueError(
                f"malformed or lossy training values in feature {column!r}: "
                f"{int(invalid.sum())} rows"
            )
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        if not finite.any():
            removed[column] = "no_finite_training_values"
            continue
        median = float(np.median(numeric.to_numpy(dtype=float)[finite]))
        clean = numeric.where(finite, median)
        if clean.nunique(dropna=False) <= 1:
            removed[column] = "constant_in_train"
            continue
        medians[column] = median
        values[column] = clean
    return pd.DataFrame(values, index=train.index), medians, removed


def _validate_all_evaluation_feature_values(
    aligned: pd.DataFrame,
    feature_columns: Iterable[str],
    declared_dtypes: Mapping[str, str],
) -> None:
    evaluation = aligned.loc[~aligned["split"].astype(str).eq("train")]
    for column in feature_columns:
        _, invalid = _coerce_declared_numeric(evaluation[column], declared_dtypes[column])
        if invalid.any():
            raise ValueError(
                f"malformed or lossy evaluation values in feature {column!r}: "
                f"{int(invalid.sum())} rows"
            )


def _lightgbm_ranker(random_state: int) -> object:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        objective="binary",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def _sklearn_ranker(random_state: int) -> object:
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,
        random_state=random_state,
    )


def _resolve_ranker(
    ranker: str,
    random_state: int,
    ranker_factory: RankerFactory | None,
    allow_sklearn_fallback: bool,
) -> tuple[object, str]:
    if ranker_factory is not None:
        fitted_ranker = ranker_factory(random_state)
        return fitted_ranker, f"injected:{type(fitted_ranker).__name__}"
    if ranker == "lightgbm":
        try:
            return _lightgbm_ranker(random_state), "lightgbm"
        except (ImportError, ModuleNotFoundError) as exc:
            if not allow_sklearn_fallback:
                raise RuntimeError(
                    "LightGBM ranking is unavailable; enable the explicit sklearn fallback"
                ) from exc
            return _sklearn_ranker(random_state), "sklearn_extra_trees_fallback"
    if ranker == "sklearn_extra_trees":
        if not allow_sklearn_fallback:
            raise ValueError("sklearn feature-ranking fallback is disabled")
        return _sklearn_ranker(random_state), "sklearn_extra_trees"
    raise ValueError(f"Unsupported comparator feature ranker: {ranker}")


def _ranker_importance(ranker: object, n_features: int) -> np.ndarray:
    booster = getattr(ranker, "booster_", None)
    if booster is not None:
        importance = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    elif hasattr(ranker, "feature_importances_"):
        importance = np.asarray(getattr(ranker, "feature_importances_"), dtype=float)
    else:
        raise ValueError("feature ranker does not expose deterministic feature importance")
    if importance.shape != (n_features,):
        raise ValueError("feature ranker returned an invalid importance vector")
    return np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)


def _rank_fold_features(
    unit: pd.DataFrame,
    feature_columns: list[str],
    declared_dtypes: Mapping[str, str],
    *,
    selected_feature_k: int,
    ranker: str,
    selection_scope: str,
    random_state: int,
    ranker_factory: RankerFactory | None,
    allow_sklearn_fallback: bool,
) -> tuple[list[str], dict[str, float], pd.DataFrame, str]:
    if selection_scope != "per_modality_mean":
        raise ValueError("aligned comparator requires selection_scope='per_modality_mean'")
    x_train, medians, removed = _training_matrix(unit, feature_columns, declared_dtypes)
    if len(x_train.columns) < selected_feature_k:
        raise ValueError(
            f"insufficient usable training features: need exactly {selected_feature_k}, "
            f"found {len(x_train.columns)}"
        )

    train = unit.loc[x_train.index]
    scopes = []
    backends: list[str] = []
    ranker_metadata: list[dict[str, str]] = []
    for offset, (modality, indices) in enumerate(train.groupby("modality", sort=True).groups.items()):
        scope_y = labels_to_binary(train.loc[indices, "label_binary"])
        if len(np.unique(scope_y)) != 2:
            raise ValueError(f"training modality {modality!r} does not contain both classes")
        scope_ranker, backend = _resolve_ranker(
            ranker,
            random_state + offset,
            ranker_factory,
            allow_sklearn_fallback,
        )
        scope_x = x_train.loc[indices]
        scope_ranker.fit(scope_x, scope_y)
        importance = _ranker_importance(scope_ranker, len(scope_x.columns))
        scale = float(np.max(importance)) if np.any(importance > 0.0) else 1.0
        scopes.append(importance / scale)
        backends.append(backend)
        metadata = _object_metadata(scope_ranker)
        ranker_metadata.append({"modality": str(modality), **metadata})
    if len(set(backends)) != 1:
        raise ValueError("feature ranking used inconsistent backends across modalities")
    backend = backends[0]

    aggregate = np.mean(np.vstack(scopes), axis=0)
    usable = pd.DataFrame({"feature": x_train.columns, "importance": aggregate})
    usable = usable.sort_values(["importance", "feature"], ascending=[False, True], kind="mergesort")
    usable = usable.reset_index(drop=True)
    usable["rank"] = np.arange(1, len(usable) + 1, dtype=int)
    selected = usable.head(selected_feature_k)["feature"].astype(str).tolist()

    removed_rows = pd.DataFrame(
        {
            "feature": list(removed),
            "importance": np.nan,
            "rank": pd.NA,
            "removal_reason": list(removed.values()),
        }
    )
    usable["removal_reason"] = ""
    provenance = pd.concat([usable, removed_rows], ignore_index=True, sort=False)
    provenance["selected"] = provenance["feature"].isin(selected)
    provenance["selection_split"] = "train"
    provenance["selection_scope"] = selection_scope
    provenance["ranker"] = ranker
    provenance["ranker_backend"] = backend
    provenance["ranker_class"] = "|".join(
        sorted({metadata["class"] for metadata in ranker_metadata})
    )
    provenance["ranker_module"] = "|".join(
        sorted({metadata["module"] for metadata in ranker_metadata})
    )
    provenance["ranker_config"] = json.dumps(
        ranker_metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    provenance["n_train_rows"] = len(train)
    provenance["n_train_positive"] = int(train["label_binary"].eq("positive").sum())
    provenance["n_train_negative"] = int(train["label_binary"].eq("negative").sum())
    return selected, medians, provenance, backend


def _clean_selected_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    medians: dict[str, float],
    declared_dtypes: Mapping[str, str],
) -> pd.DataFrame:
    cleaned_columns: dict[str, pd.Series] = {}
    for column in columns:
        numeric, invalid = _coerce_declared_numeric(frame[column], declared_dtypes[column])
        if invalid.any():
            raise ValueError(
                f"malformed or lossy evaluation values in feature {column!r}: "
                f"{int(invalid.sum())} rows"
            )
        cleaned_columns[column] = numeric.where(np.isfinite(numeric), medians[column])
    cleaned = pd.DataFrame(cleaned_columns, index=frame.index)
    cleaned.index = frame["recording_key"].astype(str)
    return cleaned


def _default_estimator_factory(model_name: str, random_state: int) -> object:
    from covid_audio_btp.strong_baseline import _make_model

    return _make_model(model_name, random_state=random_state)


def _predict_probability(model: object, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("model predict_proba returned an invalid shape")
        result = probabilities[:, 1]
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        result = 1.0 / (1.0 + np.exp(-scores))
    else:
        raise ValueError("model does not expose predict_proba or decision_function")
    if result.shape != (len(x),) or not np.isfinite(result).all():
        raise ValueError("model returned invalid probabilities")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("model probabilities must be in [0, 1]")
    return result


def _prediction_rows(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    run_id: str,
    model_name: str,
    checkpoint_hash: str,
    feature_contract_hash: str,
    feature_schema_sha256: str,
    feature_artifact_sha256: str,
    representation: str,
) -> pd.DataFrame:
    columns = [
        "protocol",
        "fold",
        "cohort",
        "manifest_sha256",
        "dataset",
        "participant_key",
        "recording_key",
        "split",
        "modality",
        "label_binary",
    ]
    out = source[columns].copy()
    out.insert(0, "run_id", run_id)
    out["model"] = model_name
    out["checkpoint_hash"] = checkpoint_hash
    out["feature_contract_hash"] = feature_contract_hash
    out["feature_schema_sha256"] = feature_schema_sha256
    out["feature_artifact_sha256"] = feature_artifact_sha256
    out["representation"] = representation
    out["probability"] = probabilities
    if "participant_id" in source:
        out["participant_id"] = source["participant_id"].to_numpy()
    assert_prediction_key_contract(out, repeated=False)
    return out


def aggregate_comparator_participants(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average recording probabilities once per participant and prediction unit."""
    participant = aggregate_to_participant(predictions)
    group_columns = [
        "run_id",
        "protocol",
        "fold",
        "dataset",
        "participant_key",
        "split",
        "modality",
        "model",
        "checkpoint_hash",
        "representation",
    ]
    extras = [
        column
        for column in (
            "cohort",
            "manifest_sha256",
            "feature_contract_hash",
            "feature_schema_sha256",
            "feature_artifact_sha256",
            "threshold",
            "threshold_source",
        )
        if column in predictions
    ]
    if extras:
        grouped = predictions.groupby(group_columns, dropna=False, sort=False)
        for column in extras:
            counts = grouped[column].nunique(dropna=False)
            if (counts != 1).any():
                raise ValueError(f"recordings have conflicting participant-level {column}")
        metadata = grouped[extras].first().reset_index()
        participant = participant.merge(metadata, on=group_columns, how="left", validate="one_to_one")
    return participant


def _threshold_and_metrics(
    predictions: pd.DataFrame,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    participants = aggregate_comparator_participants(predictions)
    validation = participants[participants["split"].astype(str).eq("validation")]
    threshold = best_threshold_by_balanced_accuracy(
        labels_to_binary(validation["label_binary"]),
        validation["probability"].to_numpy(dtype=float),
    )
    threshold_source = "validation_participant_balanced_accuracy"
    predictions = predictions.copy()
    participants = participants.copy()
    predictions["threshold"] = threshold
    predictions["threshold_source"] = threshold_source
    participants["threshold"] = threshold
    participants["threshold_source"] = threshold_source

    metric_rows: list[dict[str, object]] = []
    for (dataset, split), group in participants.groupby(["dataset", "split"], sort=False):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A single label was found in 'y_true' and 'y_pred'.*",
                category=UserWarning,
            )
            metrics = binary_metric_bundle(
                labels_to_binary(group["label_binary"]),
                group["probability"].to_numpy(dtype=float),
                threshold=threshold,
            )
        exemplar = group.iloc[0]
        metrics.update(
            {
                "run_id": exemplar["run_id"],
                "protocol": exemplar["protocol"],
                "fold": exemplar["fold"],
                "dataset": dataset,
                "split": split,
                "modality": exemplar["modality"],
                "model": exemplar["model"],
                "checkpoint_hash": exemplar["checkpoint_hash"],
                "feature_contract_hash": exemplar.get("feature_contract_hash", ""),
                "feature_schema_sha256": exemplar.get("feature_schema_sha256", ""),
                "feature_artifact_sha256": exemplar.get("feature_artifact_sha256", ""),
                "representation": exemplar["representation"],
                "analysis_unit": "participant",
                "threshold_source": threshold_source,
                "n_participants": len(group),
            }
        )
        metric_rows.append(metrics)
    return threshold, predictions, pd.DataFrame(metric_rows)


def _ensemble_predictions(
    member_predictions: pd.DataFrame,
    feature_contract_hash: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    identity = [
        "run_id",
        "protocol",
        "fold",
        "cohort",
        "manifest_sha256",
        "dataset",
        "participant_key",
        "recording_key",
        "split",
        "modality",
        "label_binary",
        "representation",
        "feature_schema_sha256",
        "feature_artifact_sha256",
    ]
    counts = member_predictions.groupby(identity, dropna=False)["model"].nunique()
    if not counts.eq(len(FROZEN_MODEL_NAMES)).all():
        raise ValueError("top-4 ensemble does not have exactly four predictions per recording")
    members = set(member_predictions["model"].astype(str))
    if members != set(FROZEN_MODEL_NAMES):
        raise ValueError("top-4 ensemble membership differs from the frozen four-model bank")
    ensemble = member_predictions.groupby(identity, dropna=False, sort=False)["probability"].mean().reset_index()
    ensemble["model"] = ENSEMBLE_MODEL_NAME
    ensemble["feature_contract_hash"] = feature_contract_hash
    member_hashes = {
        model: str(
            member_predictions.loc[
                member_predictions["model"].astype(str).eq(model), "checkpoint_hash"
            ].iloc[0]
        )
        for model in FROZEN_MODEL_NAMES
    }
    return ensemble, member_hashes


def _select_validation_candidate(
    candidate_recordings: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    expected = {*FROZEN_MODEL_NAMES, ENSEMBLE_MODEL_NAME}
    observed = set(candidate_recordings["model"].astype(str))
    if observed != expected:
        raise ValueError("validation candidate bank differs from the prespecified endpoints")
    rows: list[dict[str, object]] = []
    for model_name, group in candidate_recordings.groupby("model", sort=True):
        participants = aggregate_comparator_participants(group)
        validation = participants.loc[
            participants["split"].astype(str).eq("validation")
        ].copy()
        if validation.empty or validation["label_binary"].nunique() != 2:
            raise ValueError(
                f"validation selection candidate {model_name!r} lacks both validation classes"
            )
        metrics = binary_metric_bundle(
            labels_to_binary(validation["label_binary"]),
            validation["probability"].to_numpy(dtype=float),
            threshold=0.5,
        )
        auroc = float(metrics["auroc"])
        auprc = float(metrics["auprc"])
        if not np.isfinite(auroc) or not np.isfinite(auprc):
            raise ValueError("validation candidate discrimination metrics must be finite")
        rows.append(
            {
                "candidate_model": str(model_name),
                "validation_auroc": auroc,
                "validation_auprc": auprc,
                "n_validation_participants": len(validation),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["validation_auroc", "validation_auprc", "candidate_model"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scores["selection_rank"] = np.arange(1, len(scores) + 1)
    scores["selected"] = scores["selection_rank"].eq(1)
    scores["selection_split"] = "validation"
    scores["selection_primary_metric"] = "auroc"
    scores["selection_tiebreak_metric"] = "auprc"
    scores["selection_final_tiebreak"] = "model_name_ascending"
    return str(scores.loc[0, "candidate_model"]), scores


def _model_artifact_path(
    protocol: object,
    fold: object,
    modality: object,
    model_name: str,
    suffix: str,
    *,
    cohort: object | None = None,
    manifest_sha256: str | None = None,
    feature_contract_hash: str | None = None,
    feature_schema_sha256: str | None = None,
    feature_artifact_sha256: str | None = None,
    approval_record_sha256: str | None = None,
) -> str:
    context_sha256 = _canonical_hash(
        _stable_config_value({
            "protocol": protocol,
            "fold": fold,
            "cohort": cohort,
            "modality": modality,
            "model": model_name,
            "manifest_sha256": manifest_sha256,
            "feature_contract_hash": feature_contract_hash,
            "feature_schema_sha256": feature_schema_sha256,
            "feature_artifact_sha256": feature_artifact_sha256,
            "approval_record_sha256": approval_record_sha256,
            "artifact_suffix": suffix,
        })
    )
    return f"models/{context_sha256}.{suffix}"


def _register_artifact(
    artifact_blobs: dict[str, bytes],
    artifact_path: str,
    payload: bytes,
) -> None:
    if artifact_path in artifact_blobs:
        raise ValueError(f"duplicate model artifact path: {artifact_path}")
    artifact_blobs[artifact_path] = payload


def _serialize_model_bundle(
    estimator: object,
    *,
    threshold: float,
    threshold_source: str,
    model_identity: Mapping[str, object],
    protocol: object,
    fold: object,
    modality: object,
    cohort: object,
    model_seed: int,
    selected: list[str],
    medians: Mapping[str, float],
    declared_dtypes: Mapping[str, str],
    feature_contract_hash: str,
    feature_schema_sha256: str,
    feature_artifact_sha256: str,
    manifest_sha256: str,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    label_mapping: Mapping[str, int],
    approval_id: str,
    approval_record_sha256: str,
    approval_git_commit: str,
    approval_git_blob: str,
    executable_recipe_sha256: str,
    executable_source_sha256: Mapping[str, str],
    dependency_lock_sha256: str,
    environment_lock_sha256: str,
    member_artifacts: Mapping[str, Mapping[str, str]] | None = None,
) -> bytes:
    bundle = {
        "bundle_version": 3,
        "estimator": estimator,
        "threshold": float(threshold),
        "threshold_source": str(threshold_source),
        "model_identity": _stable_config_value(model_identity),
        "protocol": str(protocol),
        "fold": fold,
        "modality": str(modality),
        "cohort": str(cohort),
        "model_seed": int(model_seed),
        "selected_feature_columns": tuple(selected),
        "training_medians": {column: float(medians[column]) for column in selected},
        "declared_feature_dtypes": {column: declared_dtypes[column] for column in selected},
        "missing_policy": "train_median",
        "feature_contract_hash": feature_contract_hash,
        "feature_schema_sha256": feature_schema_sha256,
        "feature_artifact_sha256": feature_artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "datasets": tuple(datasets),
        "splits": tuple(splits),
        "label_mapping": {str(label): int(value) for label, value in label_mapping.items()},
        "approval_id": approval_id,
        "approval_record_sha256": approval_record_sha256,
        "approval_git_commit": approval_git_commit,
        "approval_git_blob": approval_git_blob,
        "executable_recipe_sha256": executable_recipe_sha256,
        "executable_source_sha256": dict(executable_source_sha256),
        "dependency_lock_sha256": dependency_lock_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "member_artifacts": {
            str(name): {"path": str(value["path"]), "sha256": str(value["sha256"])}
            for name, value in (member_artifacts or {}).items()
        },
    }
    return pickle.dumps(bundle, protocol=5)


def _context_seed(random_state: int, protocol: object, fold: object) -> int:
    digest = _canonical_hash([int(random_state), str(protocol), str(fold)])
    return (int(digest[:8], 16) + int(random_state)) % (2**31 - 1)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _make_read_only(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)
    if not _path_is_read_only(path):
        raise OSError(f"failed to make durable generation artifact read-only: {path}")


def _make_writable(path: Path) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def _remove_generation_tree(path: Path) -> None:
    if not path.exists():
        return
    for artifact in path.rglob("*"):
        if artifact.is_file() and not artifact.is_symlink():
            try:
                _make_writable(artifact)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    restore_existing_read_only = False
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
        if path.name == "current.json":
            _make_read_only(temporary)
            if path.exists() and _path_is_read_only(path):
                _make_writable(path)
                restore_existing_read_only = True
        os.replace(temporary, path)
        restore_existing_read_only = False
        if path.name == "current.json" and not _path_is_read_only(path):
            raise OSError("atomic current receipt was not published read-only")
    except Exception:
        if restore_existing_read_only and path.exists():
            _make_read_only(path)
        raise
    finally:
        if temporary.exists():
            if _path_is_read_only(temporary):
                _make_writable(temporary)
            temporary.unlink()


def _atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _export_generation(
    *,
    predictions: pd.DataFrame,
    participant_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    alignment_audit: pd.DataFrame,
    feature_selection: pd.DataFrame,
    model_audit: pd.DataFrame,
    candidate_selection: pd.DataFrame,
    artifact_blobs: Mapping[str, bytes],
    audit_dir: Path,
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    generations = audit_dir / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation_id = uuid.uuid4().hex
    staging = audit_dir / f".{generation_id}.tmp"
    final = generations / generation_id
    staging.mkdir()
    try:
        evidence_tables = {
            "comparator_predictions.csv": predictions,
            "comparator_participant_predictions.csv": participant_predictions,
            "comparator_metrics.csv": metrics,
            "comparator_alignment_audit.csv": alignment_audit,
            "comparator_feature_selection.csv": feature_selection,
            "comparator_model_audit.csv": model_audit,
            "comparator_candidate_selection.csv": candidate_selection,
        }
        for relative_path, frame in evidence_tables.items():
            _atomic_csv(frame, staging / relative_path)
        for relative_path, payload in sorted(artifact_blobs.items()):
            _atomic_bytes(payload, staging / relative_path)

        files: dict[str, dict[str, object]] = {}
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            relative = path.relative_to(staging).as_posix()
            payload = path.read_bytes()
            files[relative] = {
                "sha256": sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        manifest_payload = {
            "generation_manifest_version": 2,
            "generation_id": generation_id,
            "files": files,
            "model_names": list(FROZEN_MODEL_NAMES),
            "ensemble_model": ENSEMBLE_MODEL_NAME,
            "selected_candidate_model": SELECTED_CANDIDATE_MODEL_NAME,
            "execution_class": str(predictions["execution_class"].iloc[0]),
            "confirmatory_eligible": bool(predictions["confirmatory_eligible"].iloc[0]),
            "test_mode": bool(predictions["test_mode"].iloc[0]),
            "reporting_guard": str(predictions["reporting_guard"].iloc[0]),
            "evidence_domain_sha256": str(
                predictions["evidence_domain_sha256"].iloc[0]
            ),
            "executable_recipe_sha256": str(
                predictions["executable_recipe_sha256"].iloc[0]
            ),
            "approval_id": str(predictions["approval_id"].iloc[0]),
            "approval_record_sha256": str(
                predictions["approval_record_sha256"].iloc[0]
            ),
            "approval_git_commit": str(predictions["approval_git_commit"].iloc[0]),
            "approval_git_blob": str(predictions["approval_git_blob"].iloc[0]),
        }
        _atomic_json(manifest_payload, staging / "manifest.json")
        manifest_sha256 = sha256((staging / "manifest.json").read_bytes()).hexdigest()
        for artifact in sorted(item for item in staging.rglob("*") if item.is_file()):
            _make_read_only(artifact)
        os.replace(staging, final)
        receipt = {
            "generation_id": generation_id,
            "generation_manifest_sha256": manifest_sha256,
        }
        receipt["receipt_sha256"] = _canonical_hash(receipt)
        _atomic_json(receipt, audit_dir / "current.json")
    except Exception:
        _remove_generation_tree(staging)
        _remove_generation_tree(final)
        raise


def run_aligned_compare_is10(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    feature_contract: CompareIS10FeatureContract | Mapping[str, object],
    approval_record_path: str | Path | None = None,
    trusted_project_repository_root: str | Path | None = None,
    accepted_freezes_path: str | Path | None = None,
    expected_accepted_freezes_sha256: str | None = None,
    model_names: tuple[str, ...] = FROZEN_MODEL_NAMES,
    selected_feature_k: int = 800,
    ranker: str = "lightgbm",
    selection_scope: str = "per_modality_mean",
    random_state: int = 42,
    optuna_trials: int = 0,
    ensemble_top_k: int = 5,
    selection_metric: str = "auroc",
    run_id: str | None = None,
    test_mode: bool = False,
    allow_sklearn_fallback: bool = False,
    estimator_factory: EstimatorFactory | None = None,
    ranker_factory: RankerFactory | None = None,
    audit_dir: str | Path | None = None,
) -> HSTComparatorResult:
    """Fit the frozen ComParE+IS10 comparator on exact HST manifest folds."""
    if tuple(model_names) != FROZEN_MODEL_NAMES:
        raise ValueError(f"model_names must equal the frozen comparator bank: {FROZEN_MODEL_NAMES}")
    if selected_feature_k <= 0:
        raise ValueError("selected_feature_k must be positive")
    if not test_mode and selected_feature_k != 800:
        raise ValueError("confirmatory comparator execution requires selected_feature_k==800")
    if not test_mode and (estimator_factory is not None or ranker_factory is not None):
        raise ValueError("injected estimator/ranker factories require explicit test_mode=True")
    if not test_mode and audit_dir is None:
        raise ValueError("confirmatory comparator execution requires an audit/model artifact directory")
    if optuna_trials != 0:
        raise ValueError("aligned comparator does not permit test- or validation-tuned Optuna trials")
    if not test_mode and ensemble_top_k != 5:
        raise ValueError("confirmatory ensemble cap is frozen to exactly 5")
    if test_mode and ensemble_top_k < len(FROZEN_MODEL_NAMES):
        raise ValueError("ensemble cap cannot exclude a frozen comparator model")
    if selection_metric != "auroc":
        raise ValueError("selection_metric is frozen to auroc for provenance compatibility")

    schema = _normalize_feature_contract(feature_contract)
    feature_artifact_sha256 = compare_is10_feature_artifact_sha256(features)
    if not test_mode and approval_record_path is None:
        raise ValueError("confirmatory execution requires an independent frozen approval record")
    if test_mode and approval_record_path is not None:
        raise ValueError("test_mode must not consume a confirmatory approval record")
    if not test_mode and ranker != "lightgbm":
        raise ValueError("confirmatory ranker must be exactly lightgbm")
    if not test_mode and allow_sklearn_fallback:
        raise ValueError("confirmatory LightGBM ranking forbids sklearn fallback")
    if not test_mode and len(schema.ordered_feature_columns) <= selected_feature_k:
        raise ValueError("confirmatory execution requires a full feature schema, not a preselected top-800 table")
    if not test_mode and any(
        value is None
        for value in (
            trusted_project_repository_root,
            accepted_freezes_path,
            expected_accepted_freezes_sha256,
        )
    ):
        raise ValueError(
            "confirmatory execution requires trusted project root and accepted-freezes bindings"
        )
    declared_dtypes = dict(zip(schema.ordered_feature_columns, schema.feature_dtypes))
    frozen_manifest = _validate_manifest(manifest)
    approval: dict[str, object] | None = None
    if not test_mode:
        approval_path = Path(str(approval_record_path)).resolve(strict=True)
        audit_path = Path(audit_dir).resolve() if audit_dir is not None else None
        if audit_path is not None and (approval_path == audit_path or audit_path in approval_path.parents):
            raise ValueError("approval record must be frozen independently outside the audit output")
        approval = load_frozen_compare_is10_approval(
            approval_path,
            trusted_project_repository_root=trusted_project_repository_root,
            accepted_freezes_path=accepted_freezes_path,
            expected_accepted_freezes_sha256=str(expected_accepted_freezes_sha256),
            runtime_random_state=random_state,
            feature_contract=schema,
            feature_artifact_sha256=feature_artifact_sha256,
            manifest=frozen_manifest,
        )
    approval_id = str(approval["approval_id"]) if approval is not None else ""
    approval_record_sha256 = (
        str(approval["approval_record_sha256"]) if approval is not None else ""
    )
    approval_git_commit = str(approval["approval_git_commit"]) if approval is not None else ""
    approval_git_blob = str(approval["approval_git_blob"]) if approval is not None else ""
    if approval is not None:
        approved_recipe = approval["executable_recipe"]
        if not isinstance(approved_recipe, Mapping):
            raise ValueError("approved executable recipe must be a mapping")
        executable_recipe_sha256 = str(approval["executable_recipe_sha256"])
        executable_source_sha256 = dict(approved_recipe["executable_source_sha256"])
        dependency_lock_sha256 = str(approved_recipe["dependency_lock_sha256"])
        environment_lock_sha256 = str(approved_recipe["environment_lock_sha256"])
    else:
        module_path = Path(__file__).resolve(strict=True)
        executable_source_sha256 = {module_path.name: sha256(module_path.read_bytes()).hexdigest()}
        package_root = module_path.parents[2]
        available_locks = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in (
                package_root / "requirements-hst.txt",
                package_root / "requirements-gpu.txt",
            )
            if path.is_file() and not path.is_symlink()
        }
        dependency_lock_sha256 = _canonical_hash(available_locks)
        environment_lock_sha256 = _canonical_hash(
            {"execution_class": "exploratory_test_only", "dependency_lock": available_locks}
        )
        executable_recipe_sha256 = _canonical_hash(
            {
                "execution_class": "exploratory_test_only",
                "random_state": int(random_state),
                "ranker": ranker,
                "model_names": list(FROZEN_MODEL_NAMES),
                "executable_source_sha256": executable_source_sha256,
            }
        )
    aligned, feature_columns, excluded_global_row_count = _align_features(
        features, frozen_manifest, schema
    )
    _validate_all_evaluation_feature_values(aligned, feature_columns, declared_dtypes)
    manifest_digest = str(frozen_manifest["manifest_sha256"].iloc[0])
    effective_run_id = run_id or f"compare-is10-{manifest_digest[:12]}"
    make_estimator = estimator_factory or _default_estimator_factory

    recording_frames: list[pd.DataFrame] = []
    participant_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    model_audit_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    candidate_selection_frames: list[pd.DataFrame] = []
    artifact_blobs: dict[str, bytes] = {}

    for (protocol, fold), unit in aligned.groupby(list(_CONTEXT_COLUMNS), dropna=False, sort=True):
        seed = _context_seed(random_state, protocol, fold)
        selected, medians, provenance, backend = _rank_fold_features(
            unit,
            feature_columns,
            declared_dtypes,
            selected_feature_k=selected_feature_k,
            ranker=ranker,
            selection_scope=selection_scope,
            random_state=seed,
            ranker_factory=ranker_factory,
            allow_sklearn_fallback=allow_sklearn_fallback,
        )
        feature_contract_hash = _canonical_hash(
            {
                "protocol": str(protocol),
                "fold": str(fold),
                "features": selected,
                "training_medians": {column: medians[column] for column in selected},
                "declared_dtypes": {column: declared_dtypes[column] for column in selected},
                "feature_schema_sha256": schema.schema_sha256,
                "feature_artifact_sha256": feature_artifact_sha256,
                "missing_policy": schema.missing_policy,
                "ranker_backend": backend,
                "selection_scope": selection_scope,
            }
        )
        provenance["protocol"] = protocol
        provenance["fold"] = fold
        provenance["selected_feature_k"] = selected_feature_k
        provenance["feature_contract_hash"] = feature_contract_hash
        provenance["manifest_sha256"] = manifest_digest
        provenance["feature_schema_sha256"] = schema.schema_sha256
        provenance["feature_artifact_sha256"] = feature_artifact_sha256
        provenance["declared_dtype"] = provenance["feature"].map(
            dict(zip(schema.ordered_feature_columns, schema.feature_dtypes))
        )
        provenance["missing_policy"] = schema.missing_policy
        provenance["excluded_global_row_count"] = excluded_global_row_count
        feature_frames.append(provenance)

        representation = f"compare_is10_full_merged_top{selected_feature_k}"
        for modality, modality_unit in unit.groupby("modality", dropna=False, sort=True):
            train = modality_unit[modality_unit["split"].astype(str).eq("train")].copy()
            evaluation = modality_unit[~modality_unit["split"].astype(str).eq("train")].copy()
            if train.empty or evaluation.empty:
                raise ValueError(f"modality {modality!r} lacks train or evaluation rows")
            y_train = labels_to_binary(train["label_binary"])
            if len(np.unique(y_train)) != 2:
                raise ValueError(f"training modality {modality!r} does not contain both classes")
            x_train = _clean_selected_matrix(train, selected, medians, declared_dtypes)
            x_evaluation = _clean_selected_matrix(
                evaluation, selected, medians, declared_dtypes
            )

            member_frames: list[pd.DataFrame] = []
            member_artifact_map: dict[str, dict[str, str]] = {}
            for model_offset, model_name in enumerate(FROZEN_MODEL_NAMES):
                model_seed = (seed + model_offset) % (2**31 - 1)
                estimator = make_estimator(model_name, model_seed)
                estimator_metadata = _object_metadata(estimator)
                estimator.fit(x_train, y_train)
                probability = _predict_probability(estimator, x_evaluation)
                artifact_path = _model_artifact_path(
                    protocol,
                    fold,
                    modality,
                    model_name,
                    "pkl",
                    cohort=modality_unit["cohort"].iloc[0],
                    manifest_sha256=manifest_digest,
                    feature_contract_hash=feature_contract_hash,
                    feature_schema_sha256=schema.schema_sha256,
                    feature_artifact_sha256=feature_artifact_sha256,
                    approval_record_sha256=approval_record_sha256,
                )
                predictions = _prediction_rows(
                    evaluation,
                    probability,
                    run_id=effective_run_id,
                    model_name=model_name,
                    checkpoint_hash="0" * 64,
                    feature_contract_hash=feature_contract_hash,
                    feature_schema_sha256=schema.schema_sha256,
                    feature_artifact_sha256=feature_artifact_sha256,
                    representation=representation,
                )
                threshold, predictions, metrics = _threshold_and_metrics(predictions)
                threshold_source = "validation_participant_balanced_accuracy"
                model_identity = {
                    "name": model_name,
                    "backend": f"injected:{type(estimator).__name__}"
                    if estimator_factory is not None
                    else "repository_strong_baseline",
                    "class": estimator_metadata["class"],
                    "module": estimator_metadata["module"],
                    "config": estimator_metadata["config"],
                }
                artifact = _serialize_model_bundle(
                    estimator,
                    threshold=threshold,
                    threshold_source=threshold_source,
                    model_identity=model_identity,
                    protocol=protocol,
                    fold=fold,
                    modality=modality,
                    cohort=modality_unit["cohort"].iloc[0],
                    model_seed=model_seed,
                    selected=selected,
                    medians=medians,
                    declared_dtypes=declared_dtypes,
                    feature_contract_hash=feature_contract_hash,
                    feature_schema_sha256=schema.schema_sha256,
                    feature_artifact_sha256=feature_artifact_sha256,
                    manifest_sha256=manifest_digest,
                    datasets=tuple(
                        sorted(modality_unit["dataset"].astype(str).unique())
                    ),
                    splits=tuple(sorted(modality_unit["split"].astype(str).unique())),
                    label_mapping={"negative": 0, "positive": 1},
                    approval_id=approval_id,
                    approval_record_sha256=approval_record_sha256,
                    approval_git_commit=approval_git_commit,
                    approval_git_blob=approval_git_blob,
                    executable_recipe_sha256=executable_recipe_sha256,
                    executable_source_sha256=executable_source_sha256,
                    dependency_lock_sha256=dependency_lock_sha256,
                    environment_lock_sha256=environment_lock_sha256,
                )
                checkpoint_hash = sha256(artifact).hexdigest()
                _register_artifact(artifact_blobs, artifact_path, artifact)
                member_artifact_map[model_name] = {
                    "path": artifact_path,
                    "sha256": checkpoint_hash,
                }
                predictions["checkpoint_hash"] = checkpoint_hash
                metrics["checkpoint_hash"] = checkpoint_hash
                participants = aggregate_comparator_participants(predictions)
                member_frames.append(predictions)
                recording_frames.append(predictions)
                participant_frames.append(participants)
                metric_frames.append(metrics)
                model_audit_rows.append(
                    {
                        "run_id": effective_run_id,
                        "protocol": protocol,
                        "fold": fold,
                        "modality": modality,
                        "model": model_name,
                        "model_backend": f"injected:{type(estimator).__name__}"
                        if estimator_factory is not None
                        else "repository_strong_baseline",
                        "estimator_class": estimator_metadata["class"],
                        "estimator_module": estimator_metadata["module"],
                        "estimator_config": estimator_metadata["config"],
                        "random_state": model_seed,
                        "ranker_backend": backend,
                        "ranker_class": provenance["ranker_class"].iloc[0],
                        "ranker_module": provenance["ranker_module"].iloc[0],
                        "ranker_config": provenance["ranker_config"].iloc[0],
                        "selection_split": "train",
                        "selected_feature_k": selected_feature_k,
                        "feature_schema_sha256": schema.schema_sha256,
                        "feature_artifact_sha256": feature_artifact_sha256,
                        "feature_contract_hash": feature_contract_hash,
                        "checkpoint_hash": checkpoint_hash,
                        "model_artifact": artifact_path,
                        "manifest_sha256": manifest_digest,
                        "cohort": modality_unit["cohort"].iloc[0],
                        "test_mode": test_mode,
                        "threshold": threshold,
                        "threshold_source": threshold_source,
                        "ensemble_members": "",
                        "requested_ensemble_cap": ensemble_top_k,
                        "effective_ensemble_size": 1,
                        "n_train_recordings": len(train),
                    }
                )

            members = pd.concat(member_frames, ignore_index=True, sort=False)
            ensemble, member_hashes = _ensemble_predictions(members, feature_contract_hash)
            ensemble_estimator = {
                "policy": "uniform_probability_mean",
                "members": member_hashes,
                "weights": {model: 0.25 for model in FROZEN_MODEL_NAMES},
            }
            ensemble_artifact_path = _model_artifact_path(
                protocol,
                fold,
                modality,
                ENSEMBLE_MODEL_NAME,
                "pkl",
                cohort=modality_unit["cohort"].iloc[0],
                manifest_sha256=manifest_digest,
                feature_contract_hash=feature_contract_hash,
                feature_schema_sha256=schema.schema_sha256,
                feature_artifact_sha256=feature_artifact_sha256,
                approval_record_sha256=approval_record_sha256,
            )
            ensemble["checkpoint_hash"] = "0" * 64
            threshold, ensemble, metrics = _threshold_and_metrics(ensemble)
            threshold_source = "validation_participant_balanced_accuracy"
            ensemble_artifact = _serialize_model_bundle(
                ensemble_estimator,
                threshold=threshold,
                threshold_source=threshold_source,
                model_identity={
                    "name": ENSEMBLE_MODEL_NAME,
                    "backend": "uniform_probability_mean",
                    "class": "UniformProbabilityEnsemble",
                    "module": __name__,
                    "config": {
                        "members": list(FROZEN_MODEL_NAMES),
                        "weights": {model: 0.25 for model in FROZEN_MODEL_NAMES},
                    },
                },
                protocol=protocol,
                fold=fold,
                modality=modality,
                cohort=modality_unit["cohort"].iloc[0],
                model_seed=seed,
                selected=selected,
                medians=medians,
                declared_dtypes=declared_dtypes,
                feature_contract_hash=feature_contract_hash,
                feature_schema_sha256=schema.schema_sha256,
                feature_artifact_sha256=feature_artifact_sha256,
                manifest_sha256=manifest_digest,
                datasets=tuple(sorted(modality_unit["dataset"].astype(str).unique())),
                splits=tuple(sorted(modality_unit["split"].astype(str).unique())),
                label_mapping={"negative": 0, "positive": 1},
                approval_id=approval_id,
                approval_record_sha256=approval_record_sha256,
                approval_git_commit=approval_git_commit,
                approval_git_blob=approval_git_blob,
                executable_recipe_sha256=executable_recipe_sha256,
                executable_source_sha256=executable_source_sha256,
                dependency_lock_sha256=dependency_lock_sha256,
                environment_lock_sha256=environment_lock_sha256,
                member_artifacts=member_artifact_map,
            )
            ensemble_checkpoint_hash = sha256(ensemble_artifact).hexdigest()
            _register_artifact(artifact_blobs, ensemble_artifact_path, ensemble_artifact)
            ensemble["checkpoint_hash"] = ensemble_checkpoint_hash
            metrics["checkpoint_hash"] = ensemble_checkpoint_hash
            participants = aggregate_comparator_participants(ensemble)
            recording_frames.append(ensemble)
            participant_frames.append(participants)
            metric_frames.append(metrics)
            model_audit_rows.append(
                {
                    "run_id": effective_run_id,
                    "protocol": protocol,
                    "fold": fold,
                    "modality": modality,
                    "model": ENSEMBLE_MODEL_NAME,
                    "model_backend": "uniform_probability_mean",
                    "estimator_class": "UniformProbabilityEnsemble",
                    "estimator_module": __name__,
                    "estimator_config": json.dumps(
                        {"members": list(FROZEN_MODEL_NAMES), "weights": [0.25] * 4},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "random_state": seed,
                    "ranker_backend": backend,
                    "ranker_class": provenance["ranker_class"].iloc[0],
                    "ranker_module": provenance["ranker_module"].iloc[0],
                    "ranker_config": provenance["ranker_config"].iloc[0],
                    "selection_split": "train",
                    "selected_feature_k": selected_feature_k,
                    "feature_schema_sha256": schema.schema_sha256,
                    "feature_artifact_sha256": feature_artifact_sha256,
                    "feature_contract_hash": feature_contract_hash,
                    "checkpoint_hash": ensemble_checkpoint_hash,
                    "model_artifact": ensemble_artifact_path,
                    "manifest_sha256": manifest_digest,
                    "cohort": modality_unit["cohort"].iloc[0],
                    "test_mode": test_mode,
                    "threshold": threshold,
                    "threshold_source": threshold_source,
                    "ensemble_members": "|".join(FROZEN_MODEL_NAMES),
                    "requested_ensemble_cap": ensemble_top_k,
                    "effective_ensemble_size": len(FROZEN_MODEL_NAMES),
                    "n_train_recordings": len(train),
                }
            )

            candidate_bank = pd.concat([members, ensemble], ignore_index=True, sort=False)
            selected_source_model, selection_scores = _select_validation_candidate(
                candidate_bank
            )
            candidate_artifacts = {
                **member_artifact_map,
                ENSEMBLE_MODEL_NAME: {
                    "path": ensemble_artifact_path,
                    "sha256": ensemble_checkpoint_hash,
                },
            }
            selected_source = candidate_bank.loc[
                candidate_bank["model"].astype(str).eq(selected_source_model)
            ].copy()
            selected_source["model"] = SELECTED_CANDIDATE_MODEL_NAME
            selected_source["selected_candidate_source_model"] = selected_source_model
            selected_source["checkpoint_hash"] = "0" * 64
            selected_threshold, selected_predictions, selected_metrics = (
                _threshold_and_metrics(selected_source)
            )
            selected_row = selection_scores.loc[
                selection_scores["candidate_model"].eq(selected_source_model)
            ].iloc[0]
            selection_contract = {
                "selected_source_model": selected_source_model,
                "validation_auroc": float(selected_row["validation_auroc"]),
                "validation_auprc": float(selected_row["validation_auprc"]),
                "selection_split": "validation",
                "primary_metric": "auroc",
                "tiebreak_metric": "auprc",
                "final_tiebreak": "model_name_ascending",
                "candidate_scores": selection_scores.to_dict(orient="records"),
            }
            selected_artifact_path = _model_artifact_path(
                protocol,
                fold,
                modality,
                SELECTED_CANDIDATE_MODEL_NAME,
                "pkl",
                cohort=modality_unit["cohort"].iloc[0],
                manifest_sha256=manifest_digest,
                feature_contract_hash=feature_contract_hash,
                feature_schema_sha256=schema.schema_sha256,
                feature_artifact_sha256=feature_artifact_sha256,
                approval_record_sha256=approval_record_sha256,
            )
            selected_artifact = _serialize_model_bundle(
                {"policy": "validation_selected_endpoint", **selection_contract},
                threshold=selected_threshold,
                threshold_source="validation_participant_balanced_accuracy",
                model_identity={
                    "name": SELECTED_CANDIDATE_MODEL_NAME,
                    "backend": "validation_selected_endpoint",
                    "class": "ValidationSelectedComparatorEndpoint",
                    "module": __name__,
                    "config": selection_contract,
                },
                protocol=protocol,
                fold=fold,
                modality=modality,
                cohort=modality_unit["cohort"].iloc[0],
                model_seed=seed,
                selected=selected,
                medians=medians,
                declared_dtypes=declared_dtypes,
                feature_contract_hash=feature_contract_hash,
                feature_schema_sha256=schema.schema_sha256,
                feature_artifact_sha256=feature_artifact_sha256,
                manifest_sha256=manifest_digest,
                datasets=tuple(sorted(modality_unit["dataset"].astype(str).unique())),
                splits=tuple(sorted(modality_unit["split"].astype(str).unique())),
                label_mapping={"negative": 0, "positive": 1},
                approval_id=approval_id,
                approval_record_sha256=approval_record_sha256,
                approval_git_commit=approval_git_commit,
                approval_git_blob=approval_git_blob,
                executable_recipe_sha256=executable_recipe_sha256,
                executable_source_sha256=executable_source_sha256,
                dependency_lock_sha256=dependency_lock_sha256,
                environment_lock_sha256=environment_lock_sha256,
                member_artifacts={selected_source_model: candidate_artifacts[selected_source_model]},
            )
            selected_checkpoint_hash = sha256(selected_artifact).hexdigest()
            _register_artifact(
                artifact_blobs,
                selected_artifact_path,
                selected_artifact,
            )
            selected_predictions["checkpoint_hash"] = selected_checkpoint_hash
            selected_metrics["checkpoint_hash"] = selected_checkpoint_hash
            selected_participants = aggregate_comparator_participants(selected_predictions)
            for frame in (selected_predictions, selected_participants, selected_metrics):
                frame["selected_candidate_source_model"] = selected_source_model
                frame["candidate_selection_validation_auroc"] = float(
                    selected_row["validation_auroc"]
                )
                frame["candidate_selection_validation_auprc"] = float(
                    selected_row["validation_auprc"]
                )
                frame["candidate_selection_split"] = "validation"
            recording_frames.append(selected_predictions)
            participant_frames.append(selected_participants)
            metric_frames.append(selected_metrics)

            selection_scores["run_id"] = effective_run_id
            selection_scores["protocol"] = protocol
            selection_scores["fold"] = fold
            selection_scores["modality"] = modality
            selection_scores["selected_candidate_model"] = SELECTED_CANDIDATE_MODEL_NAME
            selection_scores["selected_candidate_source_model"] = selected_source_model
            selection_scores["selected_candidate_checkpoint_hash"] = (
                selected_checkpoint_hash
            )
            selection_scores["selected_candidate_model_artifact"] = (
                selected_artifact_path
            )
            selection_scores["manifest_sha256"] = manifest_digest
            selection_scores["feature_contract_hash"] = feature_contract_hash
            candidate_selection_frames.append(selection_scores)
            model_audit_rows.append(
                {
                    "run_id": effective_run_id,
                    "protocol": protocol,
                    "fold": fold,
                    "modality": modality,
                    "model": SELECTED_CANDIDATE_MODEL_NAME,
                    "model_backend": "validation_selected_endpoint",
                    "estimator_class": "ValidationSelectedComparatorEndpoint",
                    "estimator_module": __name__,
                    "estimator_config": json.dumps(
                        selection_contract,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "random_state": seed,
                    "ranker_backend": backend,
                    "ranker_class": provenance["ranker_class"].iloc[0],
                    "ranker_module": provenance["ranker_module"].iloc[0],
                    "ranker_config": provenance["ranker_config"].iloc[0],
                    "selection_split": "train",
                    "selected_feature_k": selected_feature_k,
                    "feature_schema_sha256": schema.schema_sha256,
                    "feature_artifact_sha256": feature_artifact_sha256,
                    "feature_contract_hash": feature_contract_hash,
                    "checkpoint_hash": selected_checkpoint_hash,
                    "model_artifact": selected_artifact_path,
                    "manifest_sha256": manifest_digest,
                    "cohort": modality_unit["cohort"].iloc[0],
                    "test_mode": test_mode,
                    "threshold": selected_threshold,
                    "threshold_source": "validation_participant_balanced_accuracy",
                    "ensemble_members": "",
                    "requested_ensemble_cap": ensemble_top_k,
                    "effective_ensemble_size": 1,
                    "n_train_recordings": len(train),
                    "selected_candidate_source_model": selected_source_model,
                    "candidate_selection_split": "validation",
                    "candidate_selection_primary_metric": "auroc",
                    "candidate_selection_tiebreak_metric": "auprc",
                    "candidate_selection_final_tiebreak": "model_name_ascending",
                    "candidate_selection_validation_auroc": float(
                        selected_row["validation_auroc"]
                    ),
                    "candidate_selection_validation_auprc": float(
                        selected_row["validation_auprc"]
                    ),
                }
            )

            for split, split_unit in evaluation.groupby("split", sort=False):
                alignment_rows.append(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "dataset": "|".join(sorted(split_unit["dataset"].astype(str).unique())),
                        "split": split,
                        "modality": modality,
                        "n_recordings": split_unit["recording_key"].nunique(),
                        "n_participants": split_unit["participant_key"].nunique(),
                        "excluded_global_row_count": excluded_global_row_count,
                        "aligned": True,
                    }
                )

    predictions = pd.concat(recording_frames, ignore_index=True, sort=False)
    participant_predictions = pd.concat(participant_frames, ignore_index=True, sort=False)
    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    feature_selection = pd.concat(feature_frames, ignore_index=True, sort=False)
    model_audit = pd.DataFrame(model_audit_rows)
    alignment_audit = pd.DataFrame(alignment_rows)
    candidate_selection = pd.concat(
        candidate_selection_frames,
        ignore_index=True,
        sort=False,
    )
    for table in (predictions, participant_predictions, metrics, model_audit):
        table["comparator_endpoint_role"] = np.where(
            table["model"].astype(str).eq(SELECTED_CANDIDATE_MODEL_NAME),
            "primary_validation_selected_endpoint",
            "secondary_prespecified_model_bank",
        )
        table["test_selection_use"] = False
        table["held_out_evaluation_policy"] = (
            "single_nonadaptive_pass_after_validation_freeze"
        )
    candidate_selection["test_selection_use"] = False
    candidate_selection["held_out_evaluation_policy"] = (
        "single_nonadaptive_pass_after_validation_freeze"
    )
    predictions = _classify_evidence_table(predictions, test_mode=test_mode)
    participant_predictions = _classify_evidence_table(
        participant_predictions, test_mode=test_mode
    )
    metrics = _classify_evidence_table(metrics, test_mode=test_mode)
    feature_selection = _classify_evidence_table(feature_selection, test_mode=test_mode)
    model_audit = _classify_evidence_table(model_audit, test_mode=test_mode)
    alignment_audit = _classify_evidence_table(alignment_audit, test_mode=test_mode)
    candidate_selection = _classify_evidence_table(
        candidate_selection, test_mode=test_mode
    )
    for table in (
        predictions,
        participant_predictions,
        metrics,
        feature_selection,
        model_audit,
        alignment_audit,
        candidate_selection,
    ):
        table["approval_id"] = approval_id
        table["approval_record_sha256"] = approval_record_sha256
        table["approval_git_commit"] = approval_git_commit
        table["approval_git_blob"] = approval_git_blob
        table["executable_recipe_sha256"] = executable_recipe_sha256
        table["evidence_domain_sha256"] = _evidence_domain_sha256(
            "exploratory_test_only" if test_mode else "confirmatory",
            approval_record_sha256,
        )
    assert_prediction_key_contract(predictions, repeated=predictions["fold"].nunique(dropna=False) > 1)
    assert_prediction_key_contract(
        participant_predictions,
        repeated=participant_predictions["fold"].nunique(dropna=False) > 1,
    )
    if audit_dir is not None:
        _export_generation(
            predictions=predictions,
            participant_predictions=participant_predictions,
            metrics=metrics,
            alignment_audit=alignment_audit,
            feature_selection=feature_selection,
            model_audit=model_audit,
            candidate_selection=candidate_selection,
            artifact_blobs=artifact_blobs,
            audit_dir=Path(audit_dir),
        )
    return HSTComparatorResult(
        metrics=metrics,
        predictions=predictions,
        participant_predictions=participant_predictions,
        feature_selection=feature_selection,
        model_audit=model_audit,
        alignment_audit=alignment_audit,
        candidate_selection=candidate_selection,
    )


def audit_comparator_alignment(
    hst_predictions: pd.DataFrame,
    comparator_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Assert exact HST/comparator prediction contexts and eligible cohorts."""
    for name, frame in (
        ("HST", hst_predictions),
        ("comparator", comparator_predictions),
    ):
        _require_columns(
            frame,
            {"recording_key", "cohort", "manifest_sha256"},
            f"{name} predictions",
        )
        assert_prediction_key_contract(frame, repeated=frame["fold"].nunique(dropna=False) > 1)
        label_unit = [
            "protocol",
            "fold",
            "cohort",
            "manifest_sha256",
            "dataset",
            "participant_key",
            "recording_key",
            "split",
            "modality",
        ]
        labels = frame.groupby(label_unit, dropna=False)["label_binary"].nunique(dropna=False)
        if (labels != 1).any():
            raise ValueError(f"{name} predictions contain conflicting cohort labels")

    contexts = ["protocol", "fold", "cohort", "manifest_sha256", "split", "modality"]
    left_context = set(map(tuple, hst_predictions[contexts].drop_duplicates().itertuples(index=False, name=None)))
    right_context = set(
        map(tuple, comparator_predictions[contexts].drop_duplicates().itertuples(index=False, name=None))
    )
    if left_context != right_context:
        raise ValueError("HST and comparator context sets differ by protocol/fold/split/modality")

    identity = [
        "protocol",
        "fold",
        "cohort",
        "manifest_sha256",
        "dataset",
        "participant_key",
        "recording_key",
        "split",
        "modality",
    ]
    left = hst_predictions[[*identity, "label_binary"]].drop_duplicates()
    right = comparator_predictions[[*identity, "label_binary"]].drop_duplicates()
    left_keys = set(map(tuple, left.itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right.itertuples(index=False, name=None)))
    if left_keys != right_keys:
        raise ValueError("HST and comparator representation-eligibility cohorts or labels differ")

    rows: list[dict[str, object]] = []
    for key, group in left.groupby(contexts, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(contexts, key))
        row.update(
            {
                "n_recordings": group["recording_key"].nunique() if "recording_key" in group else len(group),
                "n_participants": group["participant_key"].nunique(),
                "aligned": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
