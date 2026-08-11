from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from covid_rars.metrics import labels_to_binary
from covid_rars.statistics import _metric_value


SELECTOR_COLUMNS = [
    "prediction_source",
    "evaluation_protocol",
    "analysis_family",
    "model_name",
    "modality",
    "submodality",
    "modality_combination",
    "fusion_method",
    "feature_strategy",
    "selected_feature_k",
    "metric_split",
    "split",
]


@dataclass(frozen=True)
class DeltaComparison:
    comparison_id: str
    left_name: str
    right_name: str
    left_selector: dict[str, Any]
    right_selector: dict[str, Any]
    paired_on: tuple[str, ...] = ("participant_id",)
    left_cluster_on: tuple[str, ...] = ()
    right_cluster_on: tuple[str, ...] = ()


def _clean_selector(selector: dict[str, Any]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in selector.items():
        if key not in SELECTOR_COLUMNS:
            continue
        if value is None or pd.isna(value):
            continue
        clean[key] = str(value)
    return clean


def _selector_from_row(row: pd.Series) -> dict[str, str]:
    return _clean_selector({col: row[col] for col in SELECTOR_COLUMNS if col in row.index})


def _filter_predictions(predictions: pd.DataFrame, selector: dict[str, Any]) -> pd.DataFrame:
    frame = predictions.copy()
    for col, value in _clean_selector(selector).items():
        if col not in frame.columns:
            continue
        frame = frame[frame[col].astype(str).eq(str(value))]
    frame = frame[frame["label_binary"].isin(["positive", "negative"])].copy()
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    frame = frame[np.isfinite(frame["probability"])]
    return frame.reset_index(drop=True)


def _aggregate_cluster_means(
    frame: pd.DataFrame,
    cluster_on: tuple[str, ...],
    side_name: str,
) -> tuple[pd.DataFrame, str]:
    """Return one equally weighted analysis row per requested cluster."""
    if not cluster_on:
        return frame.reset_index(drop=True), "row"

    missing = [column for column in cluster_on if column not in frame.columns]
    if missing:
        raise KeyError(f"{side_name} predictions missing cluster columns: {missing}")

    label_counts = frame.groupby(list(cluster_on), dropna=False)["label_binary"].nunique(dropna=False)
    inconsistent = label_counts[label_counts != 1]
    if not inconsistent.empty:
        raise ValueError(
            f"{side_name} predictions contain clusters with inconsistent labels: "
            f"{inconsistent.index.tolist()[:5]}"
        )

    aggregated = (
        frame.groupby(list(cluster_on), dropna=False, as_index=False)
        .agg(label_binary=("label_binary", "first"), probability=("probability", "mean"))
        .reset_index(drop=True)
    )
    return aggregated, "+".join(cluster_on)


def _bootstrap_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    n_bootstraps: int,
    random_state: int,
    paired_on: tuple[str, ...],
    left_cluster_on: tuple[str, ...],
    right_cluster_on: tuple[str, ...],
) -> dict[str, object]:
    pair_cols = [col for col in paired_on if col in left.columns and col in right.columns]
    paired = False
    point_estimate_level = "row_level"
    paired_left = paired_right = pd.DataFrame()
    if pair_cols:
        left_pair = left.groupby(pair_cols, dropna=False).agg(label_binary=("label_binary", "first"), probability=("probability", "mean")).reset_index()
        right_pair = right.groupby(pair_cols, dropna=False).agg(label_binary=("label_binary", "first"), probability=("probability", "mean")).reset_index()
        merged = left_pair.merge(right_pair, on=pair_cols, suffixes=("_left", "_right"), how="inner")
        merged = merged[merged["label_binary_left"].astype(str).eq(merged["label_binary_right"].astype(str))]
        if len(merged) >= 2 and merged["label_binary_left"].nunique() >= 2:
            paired = True
            paired_left = merged[["label_binary_left", "probability_left"]].rename(
                columns={"label_binary_left": "label_binary", "probability_left": "probability"}
            )
            paired_right = merged[["label_binary_right", "probability_right"]].rename(
                columns={"label_binary_right": "label_binary", "probability_right": "probability"}
            )
            point_estimate_level = "paired_participant"

    if paired:
        analysis_left = paired_left
        analysis_right = paired_right
        left_resampling_level = right_resampling_level = "+".join(pair_cols)
    else:
        analysis_left, left_resampling_level = _aggregate_cluster_means(
            left,
            cluster_on=left_cluster_on,
            side_name="left",
        )
        analysis_right, right_resampling_level = _aggregate_cluster_means(
            right,
            cluster_on=right_cluster_on,
            side_name="right",
        )
        if left_cluster_on or right_cluster_on:
            point_estimate_level = (
                f"independent_left_{left_resampling_level}_right_{right_resampling_level}"
            )

    left_y = labels_to_binary(analysis_left["label_binary"])
    right_y = labels_to_binary(analysis_right["label_binary"])
    left_prob = analysis_left["probability"].astype(float).to_numpy()
    right_prob = analysis_right["probability"].astype(float).to_numpy()
    left_point = _metric_value(left_y, left_prob, metric)
    right_point = _metric_value(right_y, right_prob, metric)

    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(int(n_bootstraps)):
        if paired:
            n = len(paired_left)
            idx = rng.integers(0, n, size=n)
            l_y = labels_to_binary(paired_left.iloc[idx]["label_binary"])
            r_y = labels_to_binary(paired_right.iloc[idx]["label_binary"])
            l_prob = paired_left.iloc[idx]["probability"].astype(float).to_numpy()
            r_prob = paired_right.iloc[idx]["probability"].astype(float).to_numpy()
        else:
            l_idx = rng.integers(0, len(analysis_left), size=len(analysis_left))
            r_idx = rng.integers(0, len(analysis_right), size=len(analysis_right))
            l_y = left_y[l_idx]
            r_y = right_y[r_idx]
            l_prob = left_prob[l_idx]
            r_prob = right_prob[r_idx]
        l_value = _metric_value(l_y, l_prob, metric)
        r_value = _metric_value(r_y, r_prob, metric)
        if np.isfinite(l_value) and np.isfinite(r_value):
            values.append(float(l_value - r_value))
    arr = np.asarray(values, dtype=float)
    if arr.size:
        ci_low, ci_high = np.quantile(arr, [0.025, 0.975])
        boot_mean = float(arr.mean())
    else:
        ci_low = ci_high = boot_mean = float("nan")
    return {
        "metric": metric,
        "left_point": float(left_point),
        "right_point": float(right_point),
        "delta": float(left_point - right_point) if np.isfinite(left_point) and np.isfinite(right_point) else float("nan"),
        "bootstrap_mean": boot_mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "paired": bool(paired),
        "paired_n": int(len(paired_left)) if paired else 0,
        "point_estimate_level": point_estimate_level,
        "resampling_scheme": "paired" if paired else "independent_two_sample",
        "left_resampling_level": left_resampling_level,
        "right_resampling_level": right_resampling_level,
        "left_analysis_n": int(len(analysis_left)),
        "right_analysis_n": int(len(analysis_right)),
    }


def build_delta_bootstrap_table(
    predictions: pd.DataFrame,
    comparisons: list[DeltaComparison],
    metrics: list[str] | None = None,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    metrics = metrics or ["auroc", "auprc", "brier", "ece"]
    rows: list[dict[str, object]] = []
    for comp_idx, comparison in enumerate(comparisons):
        left = _filter_predictions(predictions, comparison.left_selector)
        right = _filter_predictions(predictions, comparison.right_selector)
        if left.empty or right.empty:
            rows.append(
                {
                    "comparison_id": comparison.comparison_id,
                    "left_name": comparison.left_name,
                    "right_name": comparison.right_name,
                    "metric": "skipped",
                    "skip_reason": "missing left or right prediction rows",
                    "left_n": int(len(left)),
                    "right_n": int(len(right)),
                }
            )
            continue
        for metric_idx, metric in enumerate(metrics):
            row = _bootstrap_delta(
                left,
                right,
                metric=metric,
                n_bootstraps=n_bootstraps,
                random_state=random_state + comp_idx * 1009 + metric_idx * 917,
                paired_on=comparison.paired_on,
                left_cluster_on=comparison.left_cluster_on,
                right_cluster_on=comparison.right_cluster_on,
            )
            row.update(
                {
                    "comparison_id": comparison.comparison_id,
                    "left_name": comparison.left_name,
                    "right_name": comparison.right_name,
                    "left_n": int(len(left)),
                    "right_n": int(len(right)),
                    "left_selector": str(_clean_selector(comparison.left_selector)),
                    "right_selector": str(_clean_selector(comparison.right_selector)),
                    "n_bootstraps": int(n_bootstraps),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _best_row(frame: pd.DataFrame, **filters: object) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    work = frame.copy()
    for col, value in filters.items():
        if col not in work.columns:
            return None
        work = work[work[col].astype(str).eq(str(value))]
    if "skipped" in work.columns:
        work = work[~work["skipped"].fillna(False).astype(bool)]
    if work.empty:
        return None
    work["auroc"] = pd.to_numeric(work.get("auroc"), errors="coerce")
    work["auprc"] = pd.to_numeric(work.get("auprc"), errors="coerce")
    return work.sort_values(["auroc", "auprc"], ascending=False).iloc[0]


def build_auto_reviewer_comparisons(
    final_summary: pd.DataFrame,
    final_metrics: pd.DataFrame,
    external_metrics: pd.DataFrame,
) -> list[DeltaComparison]:
    comparisons: list[DeltaComparison] = []
    existing = _best_row(final_summary, evaluation_protocol="compare_is10_existing_participant_split")
    stratified = _best_row(final_summary, evaluation_protocol="compare_is10_time_stratified_participant_split")
    temporal = _best_row(final_summary, evaluation_protocol="compare_is10_temporal_early_to_late")
    external = _best_row(external_metrics, evaluation_protocol="coswara_to_coughvid_compare_is10_external")

    if existing is not None and stratified is not None:
        comparisons.append(
            DeltaComparison(
                "existing_participant_split_minus_time_stratified",
                "existing_participant_split",
                "time_stratified_participant_split",
                _selector_from_row(existing),
                _selector_from_row(stratified),
            )
        )
    if stratified is not None and temporal is not None:
        comparisons.append(
            DeltaComparison(
                "time_stratified_minus_temporal_early_to_late",
                "time_stratified_participant_split",
                "temporal_early_to_late",
                _selector_from_row(stratified),
                _selector_from_row(temporal),
            )
        )
    if existing is not None and temporal is not None:
        comparisons.append(
            DeltaComparison(
                "existing_participant_split_minus_temporal_early_to_late",
                "existing_participant_split",
                "temporal_early_to_late",
                _selector_from_row(existing),
                _selector_from_row(temporal),
            )
        )
    if existing is not None and external is not None:
        comparisons.append(
            DeltaComparison(
                "existing_participant_split_minus_coughvid_external",
                "existing_participant_split",
                "coughvid_external_transfer",
                _selector_from_row(existing),
                _selector_from_row(external),
            )
        )

    external_work = external_metrics.copy() if external_metrics is not None else pd.DataFrame()
    if not external_work.empty and "skipped" in external_work.columns:
        external_work = external_work[~external_work["skipped"].fillna(False).astype(bool)]
    for _, ext_row in external_work.iterrows():
        if str(ext_row.get("modality", "")) != "cough":
            continue
        model_name = str(ext_row.get("model_name", ""))
        left = _best_row(
            final_metrics,
            evaluation_protocol="compare_is10_existing_participant_split",
            model_name=model_name,
            modality="cough",
        )
        if left is None:
            continue
        comparisons.append(
            DeltaComparison(
                f"existing_cough_{model_name}_minus_coughvid_external",
                f"existing_cough_{model_name}",
                f"coughvid_external_{model_name}",
                _selector_from_row(left),
                _selector_from_row(ext_row),
                paired_on=(),
                left_cluster_on=("participant_id",),
            )
        )
    return comparisons
