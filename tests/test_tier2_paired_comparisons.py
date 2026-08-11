from __future__ import annotations

import pandas as pd


def test_best_vs_baseline_paired_comparison_uses_matched_predictions() -> None:
    from covid_rars.tier2_comparisons import build_best_vs_baseline_paired_comparisons

    ids = [f"r{i}" for i in range(10)]
    labels = ["negative"] * 5 + ["positive"] * 5
    predictions = pd.DataFrame(
        {
            "recording_id": ids + ids,
            "label_binary": labels + labels,
            "model_name": ["logistic_regression"] * 10 + ["lightgbm"] * 10,
            "feature_strategy": ["all"] * 20,
            "probability": [0.30, 0.55, 0.60, 0.70, 0.80] + [0.20, 0.35, 0.40, 0.45, 0.50] + [0.10] * 5 + [0.90] * 5,
        }
    )
    metrics = pd.DataFrame(
        {
            "model_name": ["logistic_regression", "lightgbm"],
            "feature_strategy": ["all", "all"],
            "auroc": [0.50, 1.00],
            "auprc": [0.50, 1.00],
        }
    )

    table = build_best_vs_baseline_paired_comparisons(
        predictions,
        metrics,
        prediction_source="toy_external_predictions",
        baseline_model="logistic_regression",
        baseline_strategy="all",
        metrics_to_compare=["auroc", "brier"],
        n_bootstraps=100,
        random_state=3,
    )

    assert set(table["metric"]) == {"auroc", "brier"}
    assert set(table["candidate_name"]) == {"lightgbm/all"}
    auroc = table[table["metric"].eq("auroc")].iloc[0]
    brier = table[table["metric"].eq("brier")].iloc[0]
    assert auroc["difference"] > 0
    assert brier["difference"] < 0
    assert auroc["n_matched"] == 10
