from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from covid_rars.labels import label_to_int


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lower) & (y_prob <= upper)
        if lower == 0.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return float(ece)


def best_threshold_by_balanced_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_prob.size == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    finite = np.isfinite(y_prob)
    if not np.any(finite):
        return 0.5
    values = np.unique(np.clip(y_prob[finite], 0.0, 1.0))
    if values.size > 1:
        mids = (values[:-1] + values[1:]) / 2.0
        thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], values, mids)))
    else:
        thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], values)))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        score = balanced_accuracy_score(y_true, (y_prob >= threshold).astype(int))
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold

def binary_metric_bundle(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {}
    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["sensitivity"] = float(tp / max(1, tp + fn))
    metrics["specificity"] = float(tn / max(1, tn + fp))
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    metrics["ece"] = expected_calibration_error(y_true, y_prob)
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    metrics["nll"] = float(log_loss(y_true, clipped, labels=[0, 1]))
    metrics["threshold"] = float(threshold)
    metrics["n_samples"] = float(len(y_true))
    return metrics


def labels_to_binary(labels: pd.Series) -> np.ndarray:
    return labels.map(label_to_int).to_numpy(dtype=int)


def evaluate_predictions(
    predictions: pd.DataFrame,
    probability_column: str = "probability",
    label_column: str = "label_binary",
    group_columns: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    group_columns = group_columns or []
    rows = []
    if group_columns:
        iterator = predictions.groupby(group_columns, dropna=False)
    else:
        iterator = [((), predictions)]
    for group_key, group in iterator:
        if group.empty or not group[label_column].isin(["positive", "negative"]).all():
            continue
        y_true = labels_to_binary(group[label_column])
        y_prob = group[probability_column].astype(float).to_numpy()
        row = binary_metric_bundle(y_true, y_prob, threshold=threshold)
        if group_columns:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row.update(dict(zip(group_columns, group_key)))
        rows.append(row)
    return pd.DataFrame(rows)
