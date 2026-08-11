from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def test_paired_comparisons_cli_writes_table(tmp_path, monkeypatch) -> None:
    ids = [f"r{i}" for i in range(8)]
    labels = ["negative"] * 4 + ["positive"] * 4
    predictions = pd.DataFrame(
        {
            "recording_id": ids + ids,
            "label_binary": labels + labels,
            "model_name": ["logistic_regression"] * 8 + ["random_forest"] * 8,
            "feature_strategy": ["all"] * 16,
            "probability": [0.45] * 4 + [0.55] * 4 + [0.20] * 4 + [0.80] * 4,
        }
    )
    metrics = pd.DataFrame(
        {
            "model_name": ["logistic_regression", "random_forest"],
            "feature_strategy": ["all", "all"],
            "auroc": [0.5, 1.0],
        }
    )
    predictions_path = tmp_path / "external_model_grid_predictions.csv"
    metrics_path = tmp_path / "external_model_grid_metrics.csv"
    output_path = tmp_path / "paired.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    script_path = Path(__file__).parents[1] / "scripts" / "40_paired_bootstrap_comparisons.py"
    spec = importlib.util.spec_from_file_location("paired_comparisons_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "40_paired_bootstrap_comparisons.py",
            "--prediction-metrics-pairs",
            f"{predictions_path}:{metrics_path}:toy_predictions",
            "--output",
            str(output_path),
            "--n-bootstraps",
            "100",
        ],
    )

    module.main()

    out = pd.read_csv(output_path)
    assert not out.empty
    assert set(out["prediction_source"]) == {"toy_predictions"}
