from __future__ import annotations

import pandas as pd

from covid_rars.model_comparison import paired_bootstrap_difference


def _comparison_name(frame: pd.DataFrame) -> pd.Series:
    if "feature_strategy" in frame.columns:
        return frame["model_name"].astype(str) + "/" + frame["feature_strategy"].astype(str)
    return frame["model_name"].astype(str)


def _best_candidate_name(metrics: pd.DataFrame, baseline_name: str, metric: str = "auroc") -> str | None:
    if metrics.empty or metric not in metrics.columns or "model_name" not in metrics.columns:
        return None
    frame = metrics.copy()
    frame["comparison_name"] = _comparison_name(frame)
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[frame["comparison_name"].astype(str) != str(baseline_name)]
    frame = frame.dropna(subset=[metric])
    if frame.empty:
        return None
    return str(frame.loc[frame[metric].idxmax(), "comparison_name"])


def build_best_vs_baseline_paired_comparisons(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    prediction_source: str,
    baseline_model: str = "logistic_regression",
    baseline_strategy: str = "all",
    metrics_to_compare: list[str] | None = None,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    required = {"model_name", "label_binary", "probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Predictions missing required columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["comparison_name"] = _comparison_name(frame)
    baseline_name = f"{baseline_model}/{baseline_strategy}" if "feature_strategy" in frame.columns else baseline_model
    candidate_name = _best_candidate_name(metrics, baseline_name=baseline_name)
    if candidate_name is None:
        return pd.DataFrame()
    id_column = "recording_id" if "recording_id" in frame.columns else "participant_id"
    metrics_to_compare = metrics_to_compare or ["auroc", "auprc", "brier", "ece"]
    rows: list[dict[str, object]] = []
    for metric in metrics_to_compare:
        row = paired_bootstrap_difference(
            frame,
            baseline_name=baseline_name,
            candidate_name=candidate_name,
            model_column="comparison_name",
            id_column=id_column,
            metric=metric,
            n_bootstraps=n_bootstraps,
            confidence=confidence,
            random_state=random_state,
        )
        row["prediction_source"] = prediction_source
        row["baseline_model"] = baseline_model
        row["baseline_strategy"] = baseline_strategy
        row["comparison_type"] = "best_auroc_vs_logistic_all"
        rows.append(row)
    return pd.DataFrame(rows)
