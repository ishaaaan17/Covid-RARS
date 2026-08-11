from __future__ import annotations

import numpy as np
import pandas as pd

from covid_rars.metrics import labels_to_binary
from covid_rars.statistics import _metric_value


def _aligned_pair(
    predictions: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    model_column: str,
    id_column: str,
    probability_column: str,
    label_column: str,
) -> pd.DataFrame:
    needed = {id_column, model_column, probability_column, label_column}
    missing = needed - set(predictions.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    df = predictions[predictions[label_column].isin(["positive", "negative"])].copy()
    base = df[df[model_column].astype(str) == str(baseline_name)][[id_column, label_column, probability_column]].rename(
        columns={probability_column: "baseline_probability"}
    )
    cand = df[df[model_column].astype(str) == str(candidate_name)][[id_column, label_column, probability_column]].rename(
        columns={probability_column: "candidate_probability", label_column: "candidate_label"}
    )
    merged = base.merge(cand, on=id_column, how="inner")
    merged = merged[merged[label_column].astype(str) == merged["candidate_label"].astype(str)].copy()
    if merged.empty:
        raise ValueError("No matched predictions found for paired comparison")
    return merged.drop(columns=["candidate_label"])


def paired_bootstrap_difference(
    predictions: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    model_column: str = "model_name",
    id_column: str = "recording_id",
    probability_column: str = "probability",
    label_column: str = "label_binary",
    metric: str = "auprc",
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, object]:
    """Compare two models on matched rows using paired bootstrap resampling.

    The reported difference is candidate metric minus baseline metric. For AUROC/AUPRC
    positive is better; for Brier/ECE/NLL negative is better.
    """
    paired = _aligned_pair(
        predictions,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        model_column=model_column,
        id_column=id_column,
        probability_column=probability_column,
        label_column=label_column,
    )
    y_true = labels_to_binary(paired[label_column])
    base_prob = paired["baseline_probability"].astype(float).to_numpy()
    cand_prob = paired["candidate_probability"].astype(float).to_numpy()
    base_value = _metric_value(y_true, base_prob, metric)
    cand_value = _metric_value(y_true, cand_prob, metric)
    point_diff = cand_value - base_value

    rng = np.random.default_rng(random_state)
    values: list[float] = []
    n = len(paired)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        boot_base = _metric_value(y_true[idx], base_prob[idx], metric)
        boot_cand = _metric_value(y_true[idx], cand_prob[idx], metric)
        diff = boot_cand - boot_base
        if np.isfinite(diff):
            values.append(float(diff))
    arr = np.asarray(values, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    if arr.size:
        ci_low, ci_high = np.quantile(arr, [alpha, 1.0 - alpha])
        p_two_sided = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
        p_two_sided = min(max(p_two_sided, 0.0), 1.0)
    else:
        ci_low = ci_high = p_two_sided = float("nan")
    return {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "model_column": model_column,
        "metric": metric,
        "baseline_value": float(base_value),
        "candidate_value": float(cand_value),
        "difference": float(point_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_two_sided_bootstrap": float(p_two_sided),
        "confidence": float(confidence),
        "n_bootstraps": int(n_bootstraps),
        "n_matched": int(n),
    }


def paired_comparison_table(
    predictions: pd.DataFrame,
    baseline_name: str,
    candidate_names: list[str] | None = None,
    model_column: str = "model_name",
    metrics: list[str] | None = None,
    group_columns: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    metrics = metrics or ["auroc", "auprc", "brier", "ece"]
    group_columns = group_columns or []
    if candidate_names is None:
        names = sorted(str(v) for v in predictions[model_column].dropna().unique())
        candidate_names = [name for name in names if name != str(baseline_name)]
    rows: list[dict[str, object]] = []
    iterator = predictions.groupby(group_columns, dropna=False) if group_columns else [((), predictions)]
    for group_key, group in iterator:
        for candidate in candidate_names:
            for metric in metrics:
                try:
                    row = paired_bootstrap_difference(
                        group,
                        baseline_name=baseline_name,
                        candidate_name=candidate,
                        model_column=model_column,
                        metric=metric,
                        **kwargs,
                    )
                except Exception as exc:
                    row = {
                        "baseline_name": baseline_name,
                        "candidate_name": candidate,
                        "model_column": model_column,
                        "metric": metric,
                        "error": str(exc),
                    }
                if group_columns:
                    if not isinstance(group_key, tuple):
                        group_key = (group_key,)
                    row.update(dict(zip(group_columns, group_key)))
                rows.append(row)
    return pd.DataFrame(rows)
