from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from covid_rars.metrics import best_threshold_by_balanced_accuracy, binary_metric_bundle, labels_to_binary
from covid_rars.rescue_experiments import (
    _apply_calibrator,
    _fit_platt_or_identity,
    _predict_probability,
    _prediction_frame,
    make_rescue_model,
    select_feature_columns_for_strategy,
)


@dataclass(frozen=True)
class DomainAdaptationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    mmd: pd.DataFrame


@dataclass(frozen=True)
class CoralTransform:
    source_mean: np.ndarray
    target_mean: np.ndarray
    matrix: np.ndarray


def _as_float_matrix(values: pd.DataFrame | np.ndarray) -> np.ndarray:
    matrix = values.to_numpy(dtype=float) if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _matrix_power_symmetric(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-12, None)
    return vectors @ np.diag(values**power) @ vectors.T


def _covariance(matrix: np.ndarray, regularization: float) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.eye(matrix.shape[1]) * regularization
    cov = np.cov(matrix, rowvar=False)
    cov = np.atleast_2d(cov)
    return cov + np.eye(matrix.shape[1]) * regularization


def fit_coral_transform(
    source: pd.DataFrame | np.ndarray,
    target: pd.DataFrame | np.ndarray,
    regularization: float = 1e-5,
) -> CoralTransform:
    """Fit a CORAL transform using source-train and unlabeled target features."""
    source_matrix = _as_float_matrix(source)
    target_matrix = _as_float_matrix(target)
    if source_matrix.shape[1] != target_matrix.shape[1]:
        raise ValueError("Source and target matrices must have the same number of columns")

    source_mean = source_matrix.mean(axis=0, keepdims=True)
    target_mean = target_matrix.mean(axis=0, keepdims=True)
    source_centered = source_matrix - source_mean
    target_centered = target_matrix - target_mean
    source_cov = _covariance(source_centered, regularization)
    target_cov = _covariance(target_centered, regularization)
    matrix = _matrix_power_symmetric(source_cov, -0.5) @ _matrix_power_symmetric(target_cov, 0.5)
    return CoralTransform(source_mean=source_mean, target_mean=target_mean, matrix=matrix)


def apply_coral_transform(source: pd.DataFrame | np.ndarray, transform: CoralTransform) -> np.ndarray:
    """Apply a train-fitted CORAL transform to source-domain features."""
    source_matrix = _as_float_matrix(source)
    if source_matrix.shape[1] != transform.matrix.shape[0]:
        raise ValueError("Source matrix and CORAL transform must have compatible column counts")
    return (source_matrix - transform.source_mean) @ transform.matrix + transform.target_mean


def coral_align_source_to_target(
    source: pd.DataFrame | np.ndarray,
    target: pd.DataFrame | np.ndarray,
    regularization: float = 1e-5,
) -> np.ndarray:
    """Apply linear CORAL whitening/recoloring from source into target feature space."""
    return apply_coral_transform(source, fit_coral_transform(source, target, regularization=regularization))


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    return np.maximum(x_norm + y_norm - 2.0 * x @ y.T, 0.0)


def _median_gamma(x: np.ndarray, y: np.ndarray) -> float:
    combined = np.vstack([x, y])
    distances = _squared_distances(combined, combined)
    positive = distances[distances > 0]
    if positive.size == 0:
        return 1.0
    median = float(np.median(positive))
    return 1.0 / max(median, 1e-12)


def _sample_matrix(matrix: np.ndarray, max_samples: int | None, random_state: int) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or len(matrix) <= max_samples:
        return matrix
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(matrix), size=max_samples, replace=False)
    return matrix[np.sort(indices)]


def rbf_mmd_squared(
    source: pd.DataFrame | np.ndarray,
    target: pd.DataFrame | np.ndarray,
    *,
    gamma: float | None = None,
    max_samples: int | None = None,
    random_state: int = 42,
) -> float:
    """Biased squared RBF-kernel MMD; stable and non-negative for audit reporting."""
    source_matrix = _sample_matrix(_as_float_matrix(source), max_samples, random_state)
    target_matrix = _sample_matrix(_as_float_matrix(target), max_samples, random_state + 1)
    if source_matrix.shape[1] != target_matrix.shape[1]:
        raise ValueError("Source and target matrices must have the same number of columns")
    effective_gamma = _median_gamma(source_matrix, target_matrix) if gamma is None else float(gamma)
    k_xx = np.exp(-effective_gamma * _squared_distances(source_matrix, source_matrix))
    k_yy = np.exp(-effective_gamma * _squared_distances(target_matrix, target_matrix))
    k_xy = np.exp(-effective_gamma * _squared_distances(source_matrix, target_matrix))
    return float(max(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean(), 0.0))


def _filtered_source_external(
    source_features: pd.DataFrame,
    external_features: pd.DataFrame,
    *,
    modality: str,
    source_train_split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = source_features[
        (source_features["label_binary"].isin(["positive", "negative"]))
        & (source_features["modality"] == modality)
        & (source_features["split"].isin([source_train_split, "validation"]))
    ].copy()
    external = external_features[
        (external_features["label_binary"].isin(["positive", "negative"]))
        & (external_features["modality"] == modality)
    ].copy()
    if source.empty or external.empty:
        raise ValueError("Need non-empty source train rows and external labeled rows")
    train = source[source["split"] == source_train_split].copy()
    validation = source[source["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Need non-empty source train and validation rows")
    return train, validation, external


def _prepared_feature_matrices(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    external: pd.DataFrame,
    *,
    feature_strategy: str,
    shift_report: pd.DataFrame | None,
    smd_threshold: float,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = select_feature_columns_for_strategy(
        train,
        external,
        strategy=feature_strategy,
        shift_report=shift_report,
        smd_threshold=smd_threshold,
    )
    x_train = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols = [col for col in cols if x_train[col].nunique(dropna=False) > 1]
    if not cols:
        raise ValueError(f"Feature strategy {feature_strategy} selected no train-varying columns")
    return (
        cols,
        x_train[cols],
        validation[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0),
        external[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0),
    )


def _as_frame(matrix: np.ndarray, columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, columns=list(columns))


def _evaluate_prepared_method(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    external: pd.DataFrame,
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    x_external: pd.DataFrame,
    model_name: str,
    feature_strategy: str,
    adaptation_method: str,
    random_state: int,
    mmd_before: float,
    mmd_after: float,
    representation: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    model = make_rescue_model(model_name, train["label_binary"], random_state=random_state)
    model.fit(x_train, labels_to_binary(train["label_binary"]))

    validation_raw = _predict_probability(model, x_validation)
    calibrator, calibration_method = _fit_platt_or_identity(validation_raw, validation["label_binary"])
    validation_prob = _apply_calibrator(validation_raw, calibrator)
    threshold = best_threshold_by_balanced_accuracy(labels_to_binary(validation["label_binary"]), validation_prob)

    raw_external = _predict_probability(model, x_external)
    probabilities = _apply_calibrator(raw_external, calibrator)
    predictions = _prediction_frame(external, probabilities, model_name, feature_strategy, split="external")
    predictions["raw_probability"] = raw_external
    predictions["calibration_method"] = calibration_method
    predictions["adaptation_method"] = adaptation_method
    predictions["representation"] = representation

    metrics = binary_metric_bundle(labels_to_binary(external["label_binary"]), probabilities, threshold=threshold)
    metrics.update(
        {
            "model_name": model_name,
            "modality": str(external["modality"].iloc[0]) if "modality" in external.columns else "cough",
            "feature_strategy": feature_strategy,
            "adaptation_method": adaptation_method,
            "representation": representation,
            "calibration_method": calibration_method,
            "source_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "external_rows": int(len(external)),
            "n_samples": int(len(external)),
            "n_features": int(x_train.shape[1]),
            "mmd_before": float(mmd_before),
            "mmd_after": float(mmd_after),
            "mmd_reduction": float(mmd_before - mmd_after),
        }
    )
    return predictions, metrics


def run_domain_adaptation_baseline(
    source_features: pd.DataFrame,
    external_features: pd.DataFrame,
    *,
    model_name: str = "logistic_regression",
    feature_strategy: str = "all",
    shift_report: pd.DataFrame | None = None,
    modality: str = "cough",
    source_train_split: str = "train",
    random_state: int = 42,
    smd_threshold: float = 0.5,
    representation: str = "mfcc",
    regularization: float = 1e-5,
    n_mmd_samples: int | None = 1000,
) -> DomainAdaptationResult:
    train, validation, external = _filtered_source_external(
        source_features,
        external_features,
        modality=modality,
        source_train_split=source_train_split,
    )
    cols, x_train, x_validation, x_external = _prepared_feature_matrices(
        train,
        validation,
        external,
        feature_strategy=feature_strategy,
        shift_report=shift_report,
        smd_threshold=smd_threshold,
    )

    mmd_before = rbf_mmd_squared(x_train, x_external, max_samples=n_mmd_samples, random_state=random_state)
    coral_transform = fit_coral_transform(x_train, x_external, regularization=regularization)
    aligned_train = _as_frame(apply_coral_transform(x_train, coral_transform), cols)
    aligned_validation = _as_frame(apply_coral_transform(x_validation, coral_transform), cols)
    mmd_after = rbf_mmd_squared(aligned_train, x_external, max_samples=n_mmd_samples, random_state=random_state)

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for method_name, train_matrix, validation_matrix, method_mmd_after in [
        ("source_only", x_train, x_validation, mmd_before),
        ("coral", aligned_train, aligned_validation, mmd_after),
    ]:
        predictions, metrics = _evaluate_prepared_method(
            train=train,
            validation=validation,
            external=external,
            x_train=train_matrix,
            x_validation=validation_matrix,
            x_external=x_external,
            model_name=model_name,
            feature_strategy=feature_strategy,
            adaptation_method=method_name,
            random_state=random_state,
            mmd_before=mmd_before,
            mmd_after=method_mmd_after,
            representation=representation,
        )
        prediction_frames.append(predictions)
        metric_rows.append(metrics)

    mmd = pd.DataFrame(
        [
            {
                "representation": representation,
                "model_name": model_name,
                "feature_strategy": feature_strategy,
                "n_features": int(len(cols)),
                "source_rows": int(len(train)),
                "external_rows": int(len(external)),
                "mmd_before": float(mmd_before),
                "mmd_after_coral": float(mmd_after),
                "mmd_reduction_coral": float(mmd_before - mmd_after),
            }
        ]
    )
    return DomainAdaptationResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        mmd=mmd,
    )
