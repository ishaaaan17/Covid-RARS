from __future__ import annotations

import numpy as np
import pandas as pd


def test_prevalence_recalibration_reduces_global_calibration_gap_without_changing_rank() -> None:
    from covid_rars.prevalence_recalibration import build_prevalence_recalibration_report

    predictions = pd.DataFrame(
        {
            "recording_id": [f"r{i}" for i in range(20)],
            "label_binary": ["positive"] * 2 + ["negative"] * 18,
            "probability": [0.80, 0.70] + [0.35] * 18,
            "prediction_source": "external_model_grid_beats_predictions",
            "model_name": "logistic_regression",
            "feature_strategy": "drop_high_shift",
        }
    )

    summary, recalibrated = build_prevalence_recalibration_report(
        predictions,
        group_columns=["prediction_source", "model_name", "feature_strategy"],
    )

    original = summary[summary["recalibration_method"].eq("source_calibrated")].iloc[0]
    corrected = summary[summary["recalibration_method"].eq("target_prevalence_intercept")].iloc[0]

    assert corrected["abs_calibration_gap"] < original["abs_calibration_gap"]
    assert np.isclose(corrected["auroc"], original["auroc"])
    assert "prevalence_recalibrated_probability" in recalibrated.columns
    assert recalibrated["prevalence_recalibrated_probability"].between(0, 1).all()
