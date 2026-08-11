from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def test_prevalence_recalibration_cli_writes_summary_and_predictions(tmp_path, monkeypatch) -> None:
    predictions = pd.DataFrame(
        {
            "recording_id": [f"r{i}" for i in range(12)],
            "label_binary": ["positive"] * 3 + ["negative"] * 9,
            "probability": [0.8, 0.7, 0.6] + [0.4] * 9,
            "model_name": "logistic_regression",
            "feature_strategy": "all",
        }
    )
    predictions_path = tmp_path / "external_model_grid_predictions.csv"
    summary_path = tmp_path / "summary.csv"
    recalibrated_path = tmp_path / "recalibrated.csv"
    predictions.to_csv(predictions_path, index=False)

    script_path = Path(__file__).parents[1] / "scripts" / "39_external_prevalence_recalibration.py"
    spec = importlib.util.spec_from_file_location("prevalence_recalibration_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "39_external_prevalence_recalibration.py",
            "--predictions",
            str(predictions_path),
            "--summary-output",
            str(summary_path),
            "--predictions-output",
            str(recalibrated_path),
        ],
    )

    module.main()

    summary = pd.read_csv(summary_path)
    assert set(summary["recalibration_method"]) == {"source_calibrated", "target_prevalence_intercept"}
    assert "prevalence_recalibrated_probability" in pd.read_csv(recalibrated_path).columns
