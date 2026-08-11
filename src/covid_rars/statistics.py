from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from covid_rars.metrics import expected_calibration_error, labels_to_binary


def _metric_value(y_true: np.ndarray, y_prob: np.ndarray, metric: str) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if metric in {"auroc", "roc_auc"}:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    if metric in {"auprc", "average_precision"}:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_prob))
    if metric == "brier":
        return float(brier_score_loss(y_true, y_prob))
    if metric == "ece":
        return float(expected_calibration_error(y_true, y_prob))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float | str | int]:
    """Estimate a nonparametric bootstrap confidence interval for a binary metric."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have the same length")
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        value = _metric_value(y_true[idx], y_prob[idx], metric)
        if np.isfinite(value):
            values.append(value)
    arr = np.asarray(values, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    point = _metric_value(y_true, y_prob, metric)
    if arr.size == 0:
        low = high = mean = float("nan")
    else:
        low, high = np.quantile(arr, [alpha, 1.0 - alpha])
        mean = float(np.mean(arr))
    return {
        "metric": metric,
        "point": float(point),
        "mean": mean,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
        "n_bootstraps": int(n_bootstraps),
        "n_samples": int(n),
    }


def bootstrap_prediction_table(
    predictions: pd.DataFrame,
    metrics: list[str] | None = None,
    probability_column: str = "probability",
    label_column: str = "label_binary",
    group_columns: list[str] | None = None,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    metrics = metrics or ["auroc", "auprc", "brier", "ece"]
    group_columns = group_columns or []
    rows: list[dict[str, object]] = []
    iterator = predictions.groupby(group_columns, dropna=False) if group_columns else [((), predictions)]
    for group_key, group in iterator:
        group = group[group[label_column].isin(["positive", "negative"])]
        if group.empty:
            continue
        y_true = labels_to_binary(group[label_column])
        y_prob = group[probability_column].astype(float).to_numpy()
        for metric in metrics:
            row = bootstrap_metric_ci(
                y_true,
                y_prob,
                metric=metric,
                n_bootstraps=n_bootstraps,
                random_state=random_state,
            )
            if group_columns:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                row.update(dict(zip(group_columns, group_key)))
            rows.append(row)
    return pd.DataFrame(rows)
