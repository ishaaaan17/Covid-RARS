from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score

from covid_rars.calibration import PlattCalibrator
from covid_rars.metrics import binary_metric_bundle, expected_calibration_error, labels_to_binary


REPORTING_CONTRACT = {
    "bootstrap_replicates": 1000,
    "bootstrap_seed": 42,
    "confidence_level": 0.95,
    "ece_bins": 10,
    "fixed_sensitivity": 0.90,
    "decision_thresholds": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "probability_clip_epsilon": 1e-6,
}


@dataclass(frozen=True)
class ScreeningOperatingPoint:
    threshold: float
    target_sensitivity: float
    validation_sensitivity: float
    validation_specificity: float
    threshold_source: str = "source_validation_fixed_sensitivity"


def _validate_participant_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_key", "label_binary", "probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Participant prediction table missing columns: {missing}")
    if frame.empty:
        raise ValueError("Participant prediction table is empty")
    if not frame["label_binary"].isin(["negative", "positive"]).all():
        raise ValueError("Participant predictions contain unknown labels")
    probability = pd.to_numeric(frame["probability"], errors="coerce")
    if probability.isna().any() or (~probability.between(0.0, 1.0)).any():
        raise ValueError("Participant probabilities must be finite and in [0, 1]")
    identity = ["participant_key"] + (["fold"] if "fold" in frame else [])
    if frame.duplicated(identity).any():
        raise ValueError(f"Participant prediction rows are not unique by {identity}")
    participant_labels = frame.groupby("participant_key", sort=False)[
        "label_binary"
    ].nunique(dropna=False)
    if participant_labels.gt(1).any():
        raise ValueError("A participant has conflicting labels across prediction rows")
    return frame.assign(probability=probability.astype(float)).copy()


def _metric(frame: pd.DataFrame, metric: str) -> float:
    y = labels_to_binary(frame["label_binary"])
    probability = frame["probability"].to_numpy(dtype=float)
    if metric == "auroc":
        if np.unique(y).size < 2:
            return math.nan
        return float(roc_auc_score(y, probability))
    if metric == "auprc":
        if np.unique(y).size < 2:
            return math.nan
        return float(average_precision_score(y, probability))
    bundle = binary_metric_bundle(y, probability, threshold=0.5)
    if metric not in bundle:
        raise ValueError(f"Unsupported bootstrap metric: {metric}")
    return float(bundle[metric])


def _analysis_unit_metadata(frame: pd.DataFrame) -> tuple[str, bool | None]:
    external = (
        "split" in frame
        and frame["split"].astype(str).eq("external_test").any()
    ) or (
        "dataset" in frame
        and frame["dataset"].astype(str).eq("coughvid").any()
    )
    if external and not {
        "analysis_unit_type",
        "subject_linkage_available",
    }.issubset(frame.columns):
        raise ValueError("External predictions lack analysis-unit provenance")
    unit = "participant"
    if "analysis_unit_type" in frame:
        units = frame["analysis_unit_type"].astype(str).unique().tolist()
        if len(units) != 1 or not units[0]:
            raise ValueError("Prediction table mixes multiple analysis-unit types")
        unit = units[0]
    linkage: bool | None = None
    if "subject_linkage_available" in frame:
        values = frame["subject_linkage_available"].unique().tolist()
        if len(values) != 1 or not isinstance(values[0], (bool, np.bool_)):
            raise ValueError("Prediction table mixes subject-linkage availability")
        linkage = bool(values[0])
    if external and (unit != "recording_uuid" or linkage is not False):
        raise ValueError("External predictions have invalid analysis-unit provenance")
    return unit, linkage


def _stratified_sample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sampled: list[pd.DataFrame] = []
    for label, group in frame.groupby("label_binary", sort=True):
        participants = group["participant_key"].drop_duplicates().to_numpy()
        selected = rng.choice(participants, size=len(participants), replace=True)
        for replicate_index, participant in enumerate(selected):
            rows = group.loc[group["participant_key"].eq(participant)].copy()
            rows["participant_key"] = rows["participant_key"].astype(str) + f"::bootstrap::{label}::{replicate_index}"
            sampled.append(rows)
    return pd.concat(sampled, ignore_index=True)


def _percentile_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(values, alpha, method="linear")),
        float(np.quantile(values, 1.0 - alpha, method="linear")),
    )


def _valid_bootstrap(
    sampler: Callable[[np.random.Generator], float],
    *,
    n_bootstrap: int,
    seed: int,
    max_attempt_multiplier: int = 10,
) -> tuple[list[float], int]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    attempts = 0
    max_attempts = n_bootstrap * max_attempt_multiplier
    while len(values) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        value = float(sampler(rng))
        if math.isfinite(value):
            values.append(value)
    if len(values) != n_bootstrap:
        raise RuntimeError(
            f"Bootstrap produced {len(values)}/{n_bootstrap} valid replicates in {attempts} attempts"
        )
    return values, attempts


def external_transfer_delta(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    metric: str = "auroc",
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    source = _validate_participant_predictions(source)
    target = _validate_participant_predictions(target)
    source_point = _metric(source, metric)
    target_point = _metric(target, metric)
    source_unit, source_linkage = _analysis_unit_metadata(source)
    target_unit, target_linkage = _analysis_unit_metadata(target)

    def sample(rng: np.random.Generator) -> float:
        return _metric(_stratified_sample(source, rng), metric) - _metric(_stratified_sample(target, rng), metric)

    replicates, attempts = _valid_bootstrap(sample, n_bootstrap=n_bootstrap, seed=seed)
    low, high = _percentile_interval(replicates)
    return {
        "metric": metric,
        "source_point": source_point,
        "target_point": target_point,
        "delta": source_point - target_point,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_design": "independent_label_stratified_participants",
        "paired": False,
        "resampling_unit": "participant_key",
        "source_analysis_unit_type": source_unit,
        "target_analysis_unit_type": target_unit,
        "source_subject_linkage_available": source_linkage,
        "target_subject_linkage_available": target_linkage,
        "valid_replicates": len(replicates),
        "attempts": attempts,
        "seed": seed,
        "p_value": math.nan,
        "hypothesis_test": "not_tested_bootstrap_ci_only",
    }


def repeated_holdout_cluster_ci(
    predictions: pd.DataFrame,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    predictions = _validate_participant_predictions(predictions)
    if "fold" not in predictions:
        raise ValueError("Repeated-holdout inference requires fold")

    def mean_fold_metric(frame: pd.DataFrame) -> float:
        values = [_metric(group, metric) for _, group in frame.groupby("fold", sort=True)]
        return float(np.mean(values)) if values and all(math.isfinite(value) for value in values) else math.nan

    point = mean_fold_metric(predictions)

    def sample(rng: np.random.Generator) -> float:
        return mean_fold_metric(_stratified_sample(predictions, rng))

    replicates, attempts = _valid_bootstrap(sample, n_bootstrap=n_bootstrap, seed=seed)
    low, high = _percentile_interval(replicates)
    return {
        "metric": metric,
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "estimand": "mean_repetition_metric",
        "conditional_on_fitted_models": True,
        "independent_row_pooling": False,
        "resampling_unit": "participant_key_cluster_across_folds",
        "valid_replicates": len(replicates),
        "attempts": attempts,
        "p_value": math.nan,
        "hypothesis_test": "not_tested_absolute_metric",
    }


def _align_paired(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    allow_model_input_context_difference: bool = False,
    allow_protocol_context_difference: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = _validate_participant_predictions(left)
    right = _validate_participant_predictions(right)
    keys = ["participant_key"] + (["fold"] if "fold" in left or "fold" in right else [])
    if ("fold" in left) != ("fold" in right):
        raise ValueError("Paired predictions must either both contain fold or both omit it")
    left_sorted = left.sort_values(keys).reset_index(drop=True)
    right_sorted = right.sort_values(keys).reset_index(drop=True)
    if not left_sorted[keys].equals(right_sorted[keys]):
        raise ValueError("Paired model predictions do not contain identical participant keys")
    if not left_sorted["label_binary"].equals(right_sorted["label_binary"]):
        raise ValueError("Paired model predictions disagree on labels")
    context_columns = [
        "dataset",
        "split",
        "protocol",
        "evaluation_protocol",
        "modality",
        "submodality",
        "cohort",
        "task",
        "label_source",
        "representation_id",
        "eligibility_fingerprint",
        "manifest_sha256",
    ]
    if allow_model_input_context_difference:
        context_columns = [
            column for column in context_columns if column not in {"modality", "submodality"}
        ]
    if allow_protocol_context_difference:
        context_columns = [
            column
            for column in context_columns
            if column not in {"protocol", "evaluation_protocol"}
        ]
    for column in context_columns:
        in_left = column in left_sorted
        in_right = column in right_sorted
        if in_left != in_right:
            raise ValueError(
                f"Paired predictions disagree on scientific context column {column!r}"
            )
        if in_left and not left_sorted[column].astype("string").equals(
            right_sorted[column].astype("string")
        ):
            raise ValueError(
                f"Paired predictions disagree on scientific context column {column!r}"
            )
    return left_sorted, right_sorted


def paired_model_cluster_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
    allow_model_input_context_difference: bool = False,
    allow_protocol_context_difference: bool = False,
) -> dict[str, object]:
    left, right = _align_paired(
        left,
        right,
        allow_model_input_context_difference=allow_model_input_context_difference,
        allow_protocol_context_difference=allow_protocol_context_difference,
    )
    keys = ["participant_key"] + (["fold"] if "fold" in left else [])
    combined = left[keys + ["label_binary", "probability"]].rename(columns={"probability": "left_probability"})
    combined["right_probability"] = right["probability"].to_numpy()

    def delta(frame: pd.DataFrame) -> float:
        left_values = frame.rename(columns={"left_probability": "probability"})
        right_values = frame.rename(columns={"right_probability": "probability"})
        if "fold" in frame:
            left_metric = np.mean([_metric(group, metric) for _, group in left_values.groupby("fold")])
            right_metric = np.mean([_metric(group, metric) for _, group in right_values.groupby("fold")])
            return float(left_metric - right_metric)
        return _metric(left_values, metric) - _metric(right_values, metric)

    point = delta(combined)

    def sample(rng: np.random.Generator) -> float:
        return delta(_stratified_sample(combined, rng))

    replicates, attempts = _valid_bootstrap(sample, n_bootstrap=n_bootstrap, seed=seed)
    low, high = _percentile_interval(replicates)
    return {
        "metric": metric,
        "delta": point,
        "ci_low": low,
        "ci_high": high,
        "paired": True,
        "bootstrap_design": "paired_participant_cluster",
        "valid_replicates": len(replicates),
        "attempts": attempts,
        "seed": seed,
        "p_value": math.nan,
        "hypothesis_test": "not_tested_bootstrap_ci_only",
    }


def split_policy_delta(
    predictions: tuple[pd.DataFrame, pd.DataFrame],
    *,
    common_test: bool,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    left, right = predictions
    if common_test:
        result = paired_model_cluster_delta(
            left,
            right,
            metric=metric,
            n_bootstrap=n_bootstrap,
            seed=seed,
            allow_protocol_context_difference=True,
        )
        result["bootstrap_design"] = "paired_participant"
        return result
    left = _validate_participant_predictions(left)
    right = _validate_participant_predictions(right)
    left_keys = set(left["participant_key"].astype(str))
    right_keys = set(right["participant_key"].astype(str))
    overlap_keys = left_keys & right_keys
    if not overlap_keys:
        result = external_transfer_delta(
            left,
            right,
            metric=metric,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        result["comparison"] = "independent_split_policy_test_cohorts"
        result["overlap_participants"] = 0
        result["left_only_participants"] = len(left_keys)
        result["right_only_participants"] = len(right_keys)
        return result

    overlap_left = left.loc[left["participant_key"].astype(str).isin(overlap_keys)].copy()
    overlap_right = right.loc[right["participant_key"].astype(str).isin(overlap_keys)].copy()
    left_overlap_labels = overlap_left.set_index("participant_key")["label_binary"].sort_index()
    right_overlap_labels = overlap_right.set_index("participant_key")["label_binary"].sort_index()
    if not left_overlap_labels.equals(right_overlap_labels):
        raise ValueError("Overlapping split-policy participants disagree on labels")
    overlap = overlap_left[["participant_key", "label_binary", "probability"]].rename(
        columns={"probability": "left_probability"}
    )
    overlap = overlap.merge(
        overlap_right[["participant_key", "probability"]].rename(
            columns={"probability": "right_probability"}
        ),
        on="participant_key",
        how="inner",
        validate="one_to_one",
    )
    left_only = left.loc[~left["participant_key"].astype(str).isin(overlap_keys)].copy()
    right_only = right.loc[~right["participant_key"].astype(str).isin(overlap_keys)].copy()

    def sample_optional(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        return _stratified_sample(frame, rng) if not frame.empty else frame.copy()

    def sample(rng: np.random.Generator) -> float:
        sampled_overlap = _stratified_sample(overlap, rng)
        sampled_left_overlap = sampled_overlap[
            ["participant_key", "label_binary", "left_probability"]
        ].rename(columns={"left_probability": "probability"})
        sampled_right_overlap = sampled_overlap[
            ["participant_key", "label_binary", "right_probability"]
        ].rename(columns={"right_probability": "probability"})
        sampled_left = pd.concat(
            [sampled_left_overlap, sample_optional(left_only, rng)],
            ignore_index=True,
            sort=False,
        )
        sampled_right = pd.concat(
            [sampled_right_overlap, sample_optional(right_only, rng)],
            ignore_index=True,
            sort=False,
        )
        return _metric(sampled_left, metric) - _metric(sampled_right, metric)

    replicates, attempts = _valid_bootstrap(
        sample,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    low, high = _percentile_interval(replicates)
    return {
        "metric": metric,
        "left_point": _metric(left, metric),
        "right_point": _metric(right, metric),
        "delta": _metric(left, metric) - _metric(right, metric),
        "ci_low": low,
        "ci_high": high,
        "paired": False,
        "bootstrap_design": "partially_paired_label_stratified_participants",
        "comparison": "partially_overlapping_split_policy_test_cohorts",
        "overlap_participants": len(overlap_keys),
        "left_only_participants": len(left_keys - overlap_keys),
        "right_only_participants": len(right_keys - overlap_keys),
        "valid_replicates": len(replicates),
        "attempts": attempts,
        "seed": seed,
        "p_value": math.nan,
        "hypothesis_test": "not_tested_bootstrap_ci_only",
    }


def external_fold_cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    predictions = _validate_participant_predictions(predictions)
    if "fold" not in predictions:
        raise ValueError("External fold bootstrap requires fold")
    fold_keys = [
        set(group["participant_key"].astype(str))
        for _, group in predictions.groupby("fold", sort=True)
    ]
    if len(fold_keys) < 2:
        raise ValueError("External fold bootstrap requires at least two source folds")
    if any(keys != fold_keys[0] for keys in fold_keys[1:]):
        raise ValueError(
            "Every source fold must predict the same target cohort before bootstrapping"
        )

    def fold_mean(frame: pd.DataFrame) -> float:
        values = [_metric(group, metric) for _, group in frame.groupby("fold")]
        return float(np.mean(values)) if values and all(math.isfinite(value) for value in values) else math.nan

    def sample(rng: np.random.Generator) -> float:
        return fold_mean(_stratified_sample(predictions, rng))

    replicates, attempts = _valid_bootstrap(sample, n_bootstrap=n_bootstrap, seed=seed)
    low, high = _percentile_interval(replicates)
    return {
        "metric": metric,
        "point": fold_mean(predictions),
        "ci_low": low,
        "ci_high": high,
        "target_resampling_unit": "participant_key",
        "same_target_resample_across_folds": True,
        "valid_replicates": len(replicates),
        "attempts": attempts,
    }


def _mean_fold_metric(frame: pd.DataFrame, metric: str) -> float:
    values = [_metric(group, metric) for _, group in frame.groupby("fold", sort=True)]
    if not values or not all(math.isfinite(value) for value in values):
        return math.nan
    return float(np.mean(values))


def _validate_equal_fold_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[object]]:
    validated = _validate_participant_predictions(frame)
    if "fold" not in validated:
        raise ValueError("Equal-fold external analysis requires fold")
    folds = sorted(validated["fold"].drop_duplicates().tolist())
    if len(folds) < 2:
        raise ValueError("Equal-fold external analysis requires at least two source folds")
    cohort_keys = [
        set(group["participant_key"].astype(str))
        for _, group in validated.groupby("fold", sort=True)
    ]
    if any(keys != cohort_keys[0] for keys in cohort_keys[1:]):
        raise ValueError("Every external fold must predict the same target cohort")
    return validated, folds


def external_repeated_fold_delta(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    metric: str = "auroc",
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    source = _validate_participant_predictions(source)
    target, target_folds = _validate_equal_fold_target(target)
    if "fold" not in source:
        raise ValueError("Repeated-fold external comparison requires source folds")
    source_folds = sorted(source["fold"].drop_duplicates().tolist())
    if source_folds != target_folds:
        raise ValueError("Source and external fold identities must match exactly")

    source_point = _mean_fold_metric(source, metric)
    target_point = _mean_fold_metric(target, metric)
    source_unit, source_linkage = _analysis_unit_metadata(source)
    target_unit, target_linkage = _analysis_unit_metadata(target)

    def sample(rng: np.random.Generator) -> float:
        sampled_source = _stratified_sample(source, rng)
        sampled_target = _stratified_sample(target, rng)
        return _mean_fold_metric(sampled_source, metric) - _mean_fold_metric(
            sampled_target, metric
        )

    replicates, attempts = _valid_bootstrap(
        sample,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    low, high = _percentile_interval(replicates)
    return {
        "metric": metric,
        "source_point": source_point,
        "target_point": target_point,
        "delta": source_point - target_point,
        "ci_low": low,
        "ci_high": high,
        "p_value": math.nan,
        "hypothesis_test": "not_tested_bootstrap_ci_only",
        "bootstrap_design": (
            "independent_source_target_label_stratified_"
            "participant_clusters_across_folds"
        ),
        "source_fold_count": len(source_folds),
        "external_fold_count": len(target_folds),
        "source_target_resampling": "independent",
        "same_source_resample_across_repeated_folds": True,
        "same_target_resample_across_external_folds": True,
        "independent_row_pooling": False,
        "resampling_unit": (
            "source_participant_cluster_across_folds_and_"
            "target_participant_cluster_across_folds"
        ),
        "source_analysis_unit_type": source_unit,
        "target_analysis_unit_type": target_unit,
        "source_subject_linkage_available": source_linkage,
        "target_subject_linkage_available": target_linkage,
        "valid_replicates": len(replicates),
        "attempts": attempts,
        "seed": seed,
    }


def equal_fold_probability_ensemble_ci(
    predictions: pd.DataFrame,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    predictions, folds = _validate_equal_fold_target(predictions)
    ensemble = (
        predictions.groupby(["participant_key", "label_binary"], sort=True, as_index=False)[
            "probability"
        ]
        .mean()
        .reset_index(drop=True)
    )
    point = _metric(ensemble, metric)
    analysis_unit, subject_linkage = _analysis_unit_metadata(predictions)

    def sample(rng: np.random.Generator) -> float:
        return _metric(_stratified_sample(ensemble, rng), metric)

    replicates, attempts = _valid_bootstrap(
        sample,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    low, high = _percentile_interval(replicates)
    return {
        "metric": metric,
        "point": point,
        "ci_low": low,
        "ci_high": high,
        "endpoint": "equal_source_fold_probability_ensemble",
        "source_fold_count": len(folds),
        "n_participants": int(ensemble["participant_key"].nunique()),
        "independent_row_pooling": False,
        "resampling_unit": "external_participant_key_after_equal_fold_probability_mean",
        "analysis_unit_type": analysis_unit,
        "subject_linkage_available": subject_linkage,
        "valid_replicates": len(replicates),
        "attempts": attempts,
        "seed": seed,
    }


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    result = np.empty(len(values), dtype=float)
    result[order] = ranks
    return result


def _fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    classifier_count, total = predictions.shape
    negative_count = total - positive_count
    if positive_count < 2 or negative_count < 2:
        raise ValueError("Paired DeLong requires at least two participants in each class")
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    tx = np.empty((classifier_count, positive_count), dtype=float)
    ty = np.empty((classifier_count, negative_count), dtype=float)
    tz = np.empty((classifier_count, total), dtype=float)
    for index in range(classifier_count):
        tx[index] = _compute_midrank(positive[index])
        ty[index] = _compute_midrank(negative[index])
        tz[index] = _compute_midrank(predictions[index])
    aucs = tz[:, :positive_count].sum(axis=1) / positive_count / negative_count
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    covariance = sx / positive_count + sy / negative_count
    return aucs, covariance


def paired_delong_auc_test(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> dict[str, object]:
    if "fold" in left or "fold" in right:
        raise ValueError("Paired DeLong requires one single exact test set, not repeated folds")
    left_aligned, right_aligned = _align_paired(left, right)
    splits = set(left_aligned["split"].astype(str)) if "split" in left_aligned else set()
    if not splits or len(splits) != 1:
        raise ValueError("Paired DeLong requires one single exact test split")
    split = next(iter(splits))
    if "external" in split.casefold():
        raise ValueError("Paired DeLong is not valid for external source-target comparisons")
    if split != "test":
        raise ValueError("Paired DeLong is restricted to an exact paired test set")

    y = labels_to_binary(left_aligned["label_binary"])
    order = np.argsort(-y, kind="mergesort")
    positive_count = int(np.sum(y == 1))
    predictions = np.vstack(
        [
            left_aligned["probability"].to_numpy(dtype=float)[order],
            right_aligned["probability"].to_numpy(dtype=float)[order],
        ]
    )
    aucs, covariance = _fast_delong(predictions, positive_count)
    contrast = np.asarray([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    if variance < -1e-12:
        raise RuntimeError("DeLong covariance produced a negative contrast variance")
    standard_error = math.sqrt(max(0.0, variance))
    delta = float(aucs[0] - aucs[1])
    alpha = 1.0 - confidence
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    if standard_error == 0.0:
        p_value = 1.0 if delta == 0.0 else 0.0
        low = high = delta
    else:
        p_value = float(2.0 * norm.sf(abs(delta / standard_error)))
        low = delta - critical * standard_error
        high = delta + critical * standard_error
    return {
        "method": "paired_delong",
        "metric": "auroc",
        "left_auroc": float(aucs[0]),
        "right_auroc": float(aucs[1]),
        "delta": delta,
        "standard_error": standard_error,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence_level": confidence,
        "p_value": p_value,
        "n": len(left_aligned),
        "paired": True,
        "repeated_folds": False,
        "external": False,
    }


def fit_source_platt_calibrator(validation_predictions: pd.DataFrame) -> tuple[PlattCalibrator | None, dict[str, object]]:
    validation = _validate_participant_predictions(validation_predictions)
    if validation["label_binary"].nunique() < 2:
        return None, {"skipped": True, "skip_reason": "one_class_source_validation"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        calibrator = PlattCalibrator().fit(
            validation["probability"].to_numpy(dtype=float),
            labels_to_binary(validation["label_binary"]),
        )
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        return None, {"skipped": True, "skip_reason": "platt_non_convergence"}
    return calibrator, {
        "skipped": False,
        "skip_reason": "",
        "fit_split": "source_validation",
        "n_fit_participants": len(validation),
        "coefficient": float(calibrator.model.coef_[0, 0]),
        "intercept": float(calibrator.model.intercept_[0]),
    }


def apply_source_platt_calibration(
    validation_predictions: pd.DataFrame,
    evaluation_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit Platt scaling per source-validation fold and apply it without refitting."""

    validation = _validate_participant_predictions(validation_predictions)
    evaluation = _validate_participant_predictions(evaluation_predictions)
    if "split" not in validation or not validation["split"].astype(str).eq(
        "validation"
    ).all():
        raise ValueError("Platt calibration must be fit only on source validation")
    if ("fold" in validation) != ("fold" in evaluation):
        raise ValueError("Validation and evaluation calibration folds must match")
    if "fold" in validation and (
        validation["fold"].isna().any() or evaluation["fold"].isna().any()
    ):
        raise ValueError("Validation and evaluation calibration require non-null folds")
    if "_calibration_order" in validation or "_calibration_order" in evaluation:
        raise ValueError("Reserved calibration order column is already present")

    validation = validation.assign(_calibration_order=np.arange(len(validation)))
    evaluation = evaluation.assign(_calibration_order=np.arange(len(evaluation)))
    if "fold" in validation:
        validation_groups = {
            fold: group.copy()
            for fold, group in validation.groupby("fold", sort=True, dropna=False)
        }
        evaluation_groups = {
            fold: group.copy()
            for fold, group in evaluation.groupby("fold", sort=True, dropna=False)
        }
    else:
        validation_groups = {None: validation.copy()}
        evaluation_groups = {None: evaluation.copy()}
    if set(validation_groups) != set(evaluation_groups):
        raise ValueError("Validation and evaluation calibration folds must match exactly")

    calibrated_validation: list[pd.DataFrame] = []
    calibrated_evaluation: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for fold in sorted(
        validation_groups,
        key=lambda value: "" if value is None else str(value),
    ):
        validation_group = validation_groups[fold]
        evaluation_group = evaluation_groups[fold]
        calibrator, audit = fit_source_platt_calibrator(validation_group)
        for group in (validation_group, evaluation_group):
            raw = group["probability"].to_numpy(dtype=float)
            group["raw_probability"] = raw
            if calibrator is None:
                group["probability"] = raw
                group["probability_scale"] = "raw_fallback"
            else:
                group["probability"] = calibrator.transform(raw)
                group["probability_scale"] = "source_validation_platt"
            group["calibration_fit_split"] = "source_validation"
            group["calibration_fit_fold"] = fold
        calibrated_validation.append(validation_group)
        calibrated_evaluation.append(evaluation_group)
        audits.append({"fold": fold, **audit})

    def restore(groups: list[pd.DataFrame]) -> pd.DataFrame:
        return (
            pd.concat(groups, ignore_index=True, sort=False)
            .sort_values("_calibration_order", kind="mergesort")
            .drop(columns="_calibration_order")
            .reset_index(drop=True)
        )

    return restore(calibrated_validation), restore(calibrated_evaluation), pd.DataFrame(audits)


def build_calibration_report(predictions: pd.DataFrame, *, n_bins: int = 10) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = _validate_participant_predictions(predictions)
    y = labels_to_binary(frame["label_binary"])
    probability = frame["probability"].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, object]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probability >= lower) & (probability <= upper if index == n_bins - 1 else probability < upper)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_index": index,
                "lower": lower,
                "upper": upper,
                "n": int(mask.sum()),
                "mean_probability": float(probability[mask].mean()),
                "observed_prevalence": float(y[mask].mean()),
            }
        )
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    summary = {
        "brier": float(np.mean((probability - y) ** 2)),
        "ece": float(expected_calibration_error(y, probability, n_bins=n_bins)),
        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
    }
    return pd.DataFrame(rows), summary


def fit_screening_operating_point(
    validation_predictions: pd.DataFrame,
    *,
    target_sensitivity: float = 0.90,
) -> ScreeningOperatingPoint:
    validation = _validate_participant_predictions(validation_predictions)
    y = labels_to_binary(validation["label_binary"])
    if np.unique(y).size < 2:
        raise ValueError("Screening operating point requires both source-validation classes")
    probability = validation["probability"].to_numpy(dtype=float)
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probability)))
    candidates: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        predicted = probability >= threshold
        sensitivity = float(np.sum(predicted & (y == 1)) / np.sum(y == 1))
        specificity = float(np.sum(~predicted & (y == 0)) / np.sum(y == 0))
        if sensitivity + 1e-12 >= target_sensitivity:
            candidates.append((specificity, sensitivity, float(threshold)))
    if not candidates:
        raise RuntimeError("No source-validation threshold reaches the requested sensitivity")
    specificity, sensitivity, threshold = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return ScreeningOperatingPoint(threshold, target_sensitivity, sensitivity, specificity)


def apply_screening_operating_point(
    test_predictions: pd.DataFrame,
    operating_point: ScreeningOperatingPoint,
) -> dict[str, float | str]:
    test = _validate_participant_predictions(test_predictions)
    metrics = binary_metric_bundle(
        labels_to_binary(test["label_binary"]),
        test["probability"].to_numpy(dtype=float),
        threshold=operating_point.threshold,
    )
    return {**metrics, "threshold_source": operating_point.threshold_source}


def build_decision_curve(predictions: pd.DataFrame, *, thresholds: list[float]) -> pd.DataFrame:
    frame = _validate_participant_predictions(predictions)
    y = labels_to_binary(frame["label_binary"])
    probability = frame["probability"].to_numpy(dtype=float)
    prevalence = float(y.mean())
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Decision thresholds must be strictly between zero and one")
        predicted = probability >= threshold
        tp = float(np.sum(predicted & (y == 1)))
        fp = float(np.sum(predicted & (y == 0)))
        penalty = threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "model_net_benefit": tp / len(y) - fp / len(y) * penalty,
                "treat_all_net_benefit": prevalence - (1.0 - prevalence) * penalty,
                "treat_none_net_benefit": 0.0,
            }
        )
    return pd.DataFrame(rows)
