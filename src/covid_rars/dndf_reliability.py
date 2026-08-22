from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from covid_rars.metrics import (
    best_threshold_by_balanced_accuracy,
    binary_metric_bundle,
    expected_calibration_error,
    labels_to_binary,
)

logger = logging.getLogger(__name__)


def compute_dndf_calibration_summary(
    predictions: pd.DataFrame,
    prob_col: str = "probability",
    label_col: str = "label_binary",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration metrics (ECE, Brier, NLL, mean probability vs observed positive rate)."""
    if predictions.empty:
        return pd.DataFrame()

    results: list[dict[str, Any]] = []
    group_cols = [c for c in ["protocol", "dataset", "modality", "model_name", "split", "fusion_method"] if c in predictions.columns]

    for key, group in (predictions.groupby(group_cols) if group_cols else [("all", predictions)]):
        y_true = labels_to_binary(group[label_col])
        y_prob = group[prob_col].to_numpy(dtype=float)

        if len(np.unique(y_true)) < 2:
            continue

        bundle = binary_metric_bundle(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        observed_prev = float(np.mean(y_true))
        mean_pred = float(np.mean(y_prob))

        rec = {
            "n_samples": len(group),
            "observed_prevalence": observed_prev,
            "mean_predicted_probability": mean_pred,
            "calibration_gap": abs(mean_pred - observed_prev),
            "ece": ece,
            "brier_score": bundle["brier"],
            "nll": bundle["nll"],
            "auroc": bundle["auroc"],
            "auprc": bundle["auprc"],
        }
        if isinstance(key, tuple):
            for col_name, col_val in zip(group_cols, key):
                rec[col_name] = col_val
        elif group_cols:
            rec[group_cols[0]] = key

        results.append(rec)

    return pd.DataFrame(results)


def compute_dndf_fixed_sensitivity_operating_points(
    predictions: pd.DataFrame,
    min_sensitivity: float = 0.90,
    prob_col: str = "probability",
    label_col: str = "label_binary",
) -> pd.DataFrame:
    """Evaluate screening operating points at fixed high sensitivity (e.g. >= 90%)."""
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = [c for c in ["protocol", "dataset", "modality", "model_name", "split", "fusion_method"] if c in predictions.columns]

    for key, group in (predictions.groupby(group_cols) if group_cols else [("all", predictions)]):
        y_true = labels_to_binary(group[label_col])
        y_prob = group[prob_col].to_numpy(dtype=float)

        if len(np.unique(y_true)) < 2:
            continue

        thresholds = np.linspace(0.0, 1.0, 1001)
        best_t = 0.0
        best_spec = -1.0
        best_bundle = None

        for t in thresholds:
            b = binary_metric_bundle(y_true, y_prob, threshold=t)
            if b["sensitivity"] >= min_sensitivity:
                if b["specificity"] > best_spec:
                    best_spec = b["specificity"]
                    best_t = t
                    best_bundle = b

        if best_bundle is None:
            best_t = 0.0
            best_bundle = binary_metric_bundle(y_true, y_prob, threshold=0.0)

        rec = {
            "target_sensitivity": min_sensitivity,
            "selected_threshold": best_t,
            "achieved_sensitivity": best_bundle["sensitivity"],
            "achieved_specificity": best_bundle["specificity"],
            "precision": best_bundle["precision"],
            "f1": best_bundle["f1"],
            "balanced_accuracy": best_bundle["balanced_accuracy"],
            "n_samples": len(group),
        }
        if isinstance(key, tuple):
            for col_name, col_val in zip(group_cols, key):
                rec[col_name] = col_val
        elif group_cols:
            rec[group_cols[0]] = key

        rows.append(rec)

    return pd.DataFrame(rows)


def compute_dndf_decision_curve_analysis(
    predictions: pd.DataFrame,
    threshold_grid: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    prob_col: str = "probability",
    label_col: str = "label_binary",
) -> pd.DataFrame:
    """Compute Decision Curve Analysis (DCA) Net Benefit across threshold probability grid."""
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = [c for c in ["protocol", "dataset", "modality", "model_name", "split", "fusion_method"] if c in predictions.columns]

    for key, group in (predictions.groupby(group_cols) if group_cols else [("all", predictions)]):
        y_true = labels_to_binary(group[label_col])
        y_prob = group[prob_col].to_numpy(dtype=float)
        n = len(y_true)
        if n == 0 or len(np.unique(y_true)) < 2:
            continue

        prevalence = float(np.mean(y_true))

        for pt in threshold_grid:
            weight = pt / (1.0 - pt)
            y_pred = (y_prob >= pt).astype(int)
            tp = float(np.sum((y_pred == 1) & (y_true == 1)))
            fp = float(np.sum((y_pred == 1) & (y_true == 0)))

            net_benefit_model = (tp / n) - (fp / n) * weight
            net_benefit_all = prevalence - (1.0 - prevalence) * weight
            net_benefit_none = 0.0

            rec = {
                "threshold_probability": pt,
                "net_benefit_model": net_benefit_model,
                "net_benefit_all": net_benefit_all,
                "net_benefit_none": net_benefit_none,
                "prevalence": prevalence,
                "n_samples": n,
            }
            if isinstance(key, tuple):
                for col_name, col_val in zip(group_cols, key):
                    rec[col_name] = col_val
            elif group_cols:
                rec[group_cols[0]] = key

            rows.append(rec)

    return pd.DataFrame(rows)


def run_dndf_bootstrap_uncertainty(
    predictions: pd.DataFrame,
    n_bootstraps: int = 1000,
    prob_col: str = "probability",
    label_col: str = "label_binary",
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute participant-level bootstrap confidence intervals for DNDT/DNDF predictions."""
    if predictions.empty:
        return pd.DataFrame()

    rng = np.random.RandomState(random_state)
    rows: list[dict[str, Any]] = []
    group_cols = [c for c in ["protocol", "dataset", "modality", "model_name", "split", "fusion_method"] if c in predictions.columns]

    for key, group in (predictions.groupby(group_cols) if group_cols else [("all", predictions)]):
        y_true = labels_to_binary(group[label_col])
        y_prob = group[prob_col].to_numpy(dtype=float)
        n = len(y_true)

        if n < 5 or len(np.unique(y_true)) < 2:
            continue

        point_bundle = binary_metric_bundle(y_true, y_prob)

        aurocs, auprcs, baccs = [], [], []
        for _ in range(n_bootstraps):
            indices = rng.randint(0, n, size=n)
            b_true = y_true[indices]
            b_prob = y_prob[indices]
            if len(np.unique(b_true)) < 2:
                continue
            b_bundle = binary_metric_bundle(b_true, b_prob)
            aurocs.append(b_bundle["auroc"])
            auprcs.append(b_bundle["auprc"])
            baccs.append(b_bundle["balanced_accuracy"])

        rec = {
            "n_samples": n,
            "auroc_point": point_bundle["auroc"],
            "auroc_ci_low": float(np.percentile(aurocs, 2.5)) if aurocs else point_bundle["auroc"],
            "auroc_ci_high": float(np.percentile(aurocs, 97.5)) if aurocs else point_bundle["auroc"],
            "auprc_point": point_bundle["auprc"],
            "auprc_ci_low": float(np.percentile(auprcs, 2.5)) if auprcs else point_bundle["auprc"],
            "auprc_ci_high": float(np.percentile(auprcs, 97.5)) if auprcs else point_bundle["auprc"],
            "bacc_point": point_bundle["balanced_accuracy"],
            "bacc_ci_low": float(np.percentile(baccs, 2.5)) if baccs else point_bundle["balanced_accuracy"],
            "bacc_ci_high": float(np.percentile(baccs, 97.5)) if baccs else point_bundle["balanced_accuracy"],
        }
        if isinstance(key, tuple):
            for col_name, col_val in zip(group_cols, key):
                rec[col_name] = col_val
        elif group_cols:
            rec[group_cols[0]] = key

        rows.append(rec)

    return pd.DataFrame(rows)
