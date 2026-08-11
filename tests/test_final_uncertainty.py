from __future__ import annotations

from pathlib import Path

import pandas as pd


def _prediction_file(path: Path, source_name: str, split: str) -> Path:
    rows: list[dict[str, object]] = []
    for idx in range(24):
        label = "positive" if idx % 2 else "negative"
        probability = 0.72 if label == "positive" else 0.28
        if split == "external_test":
            probability = 0.58 if label == "positive" else 0.48
        rows.append(
            {
                "recording_id": f"{source_name}_{split}_{idx}",
                "participant_id": f"{source_name}_{split}_p{idx}",
                "label_binary": label,
                "probability": probability,
                "split": split,
                "evaluation_protocol": "unit_protocol",
                "analysis_family": "unit_family",
                "model_name": source_name,
                "modality": "cough",
                "feature_strategy": "unit_features",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_final_uncertainty_builds_ci_and_calibration_tables(tmp_path: Path) -> None:
    from covid_rars.final_uncertainty import (
        build_final_uncertainty_and_calibration,
        save_calibration_curve_figure,
    )

    internal = _prediction_file(tmp_path / "internal_predictions.csv", "internal_model", "test")
    external = _prediction_file(tmp_path / "external_predictions.csv", "external_model", "external_test")

    ci, calibration_summary, calibration_bins = build_final_uncertainty_and_calibration(
        [internal, external],
        group_columns=["prediction_source", "model_name", "split"],
        bootstrap_metrics=["auroc", "auprc", "brier"],
        n_bootstraps=20,
        n_bins=5,
        random_state=0,
    )

    assert not ci.empty
    assert set(ci["metric"]) == {"auroc", "auprc", "brier"}
    assert {"ci_low", "ci_high", "point", "prediction_source", "split"}.issubset(ci.columns)
    assert set(calibration_summary["split"]) == {"test", "external_test"}
    assert not calibration_bins.empty

    figure = tmp_path / "calibration.svg"
    save_calibration_curve_figure(
        calibration_bins,
        figure,
        group_columns=["prediction_source", "model_name", "split"],
        max_series=4,
    )
    assert figure.exists()
    assert "<svg" in figure.read_text(encoding="utf-8", errors="ignore")
