from __future__ import annotations

import numpy as np
import pandas as pd

from covid_rars.temporal_holdout import (
    _participant_dates,
    _split_counts,
    _summarize_split_participants,
)


REVERSE_TEMPORAL_PROTOCOL = "compare_is10_temporal_late_to_early"


def build_reverse_temporal_split_assignments(
    metadata: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    date_column: str = "recording_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign earliest participants to test and latest participants to train."""
    participants = _participant_dates(metadata, date_column=date_column)
    train_n, validation_n, test_n = _split_counts(len(participants), train_fraction, validation_fraction)
    split_values = np.array(["train"] * len(participants), dtype=object)
    split_values[:test_n] = "test"
    split_values[test_n : test_n + validation_n] = "validation"
    split_values[test_n + validation_n :] = "train"
    assignments = participants.copy()
    assignments["reverse_temporal_split"] = split_values
    assignments["reverse_temporal_order"] = np.arange(len(assignments), dtype=int)
    assignments["temporal_train_fraction"] = float(train_fraction)
    assignments["temporal_validation_fraction"] = float(validation_fraction)
    summary = _summarize_split_participants(assignments, "reverse_temporal_split", REVERSE_TEMPORAL_PROTOCOL)
    return assignments, summary


def summarize_multiseed_metrics(
    metrics: pd.DataFrame,
    group_columns: list[str] | None = None,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    if "random_state" not in metrics.columns:
        raise KeyError("metrics must contain random_state for multi-seed summary")
    default_groups = [
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
    ]
    groups = [col for col in (group_columns or default_groups) if col in metrics.columns]
    metric_names = [col for col in (metric_columns or ["auroc", "auprc", "balanced_accuracy", "f1", "brier", "ece"]) if col in metrics.columns]
    work = metrics.copy()
    for metric in metric_names:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    agg_spec: dict[str, tuple[str, str]] = {"n_seeds": ("random_state", "nunique")}
    for metric in metric_names:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
        agg_spec[f"{metric}_min"] = (metric, "min")
        agg_spec[f"{metric}_max"] = (metric, "max")
    summary = work.groupby(groups, dropna=False).agg(**agg_spec).reset_index() if groups else work.agg(**agg_spec).to_frame().T
    if "auroc_mean" in summary.columns:
        summary = summary.sort_values(["auroc_mean", "auprc_mean" if "auprc_mean" in summary.columns else "n_seeds"], ascending=False)
    return summary.reset_index(drop=True)
