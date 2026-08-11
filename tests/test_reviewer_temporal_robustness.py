from __future__ import annotations

import pandas as pd

from covid_rars.reviewer_temporal_robustness import (
    build_reverse_temporal_split_assignments,
    summarize_multiseed_metrics,
)


def _metadata_frame(n: int = 30) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        rows.append(
            {
                "participant_id": f"p{idx:03d}",
                "label_binary": "positive" if idx % 2 == 0 else "negative",
                "recording_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=idx),
                "split": "train",
            }
        )
    return pd.DataFrame(rows)


def test_reverse_temporal_split_trains_late_and_tests_early() -> None:
    assignments, summary = build_reverse_temporal_split_assignments(
        _metadata_frame(),
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    by_split = assignments.groupby("reverse_temporal_split")["recording_date"]
    assert by_split.min()["test"] < by_split.min()["validation"] < by_split.min()["train"]
    assert by_split.max()["test"] < by_split.max()["validation"] < by_split.max()["train"]
    assert set(summary["temporal_split"]) == {"train", "validation", "test"}
    assert summary["evaluation_protocol"].eq("compare_is10_temporal_late_to_early").all()


def test_summarize_multiseed_metrics_aggregates_mean_and_std() -> None:
    metrics = pd.DataFrame(
        {
            "random_state": [1, 2, 3],
            "evaluation_protocol": ["p"] * 3,
            "analysis_family": ["a"] * 3,
            "model_name": ["m"] * 3,
            "modality": ["cough"] * 3,
            "metric_split": ["test"] * 3,
            "auroc": [0.70, 0.75, 0.80],
            "auprc": [0.60, 0.65, 0.70],
            "balanced_accuracy": [0.55, 0.60, 0.65],
            "brier": [0.20, 0.18, 0.16],
        }
    )

    summary = summarize_multiseed_metrics(metrics)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_seeds"] == 3
    assert abs(row["auroc_mean"] - 0.75) < 1e-9
    assert row["auroc_std"] > 0
    assert abs(row["brier_mean"] - 0.18) < 1e-9
