from __future__ import annotations

import numpy as np
import pandas as pd

from covid_rars.reviewer_evidence import (
    build_audio_metadata_residual_correlation,
    build_fixed_sensitivity_table,
    build_partial_target_recalibration,
    build_shuffle_label_sanity,
)


def _prediction_frame(n: int = 80) -> pd.DataFrame:
    y = np.array(["negative"] * (n // 2) + ["positive"] * (n // 2), dtype=object)
    prob = np.concatenate([np.linspace(0.02, 0.35, n // 2), np.linspace(0.65, 0.98, n // 2)])
    return pd.DataFrame(
        {
            "participant_id": [f"p{i:03d}" for i in range(n)],
            "label_binary": y,
            "probability": prob,
            "evaluation_protocol": "unit_protocol",
            "model_name": "unit_model",
            "split": "test",
        }
    )


def test_shuffle_label_sanity_reports_observed_and_permuted_auc() -> None:
    predictions = _prediction_frame()

    table = build_shuffle_label_sanity(
        predictions,
        group_columns=["evaluation_protocol", "model_name", "split"],
        n_permutations=80,
        random_state=7,
        metrics=["auroc", "auprc"],
    )

    auc = table[table["metric"].eq("auroc")].iloc[0]
    assert auc["observed"] > 0.99
    assert 0.35 < auc["permuted_mean"] < 0.65
    assert auc["n_permutations"] == 80


def test_fixed_sensitivity_table_uses_existing_operating_point_logic() -> None:
    predictions = _prediction_frame()

    table = build_fixed_sensitivity_table(
        predictions,
        group_columns=["evaluation_protocol", "model_name", "split"],
        target_sensitivities=[0.9],
    )

    assert len(table) == 1
    row = table.iloc[0]
    assert row["operating_constraint"] == "sensitivity>=0.900"
    assert row["sensitivity"] >= 0.9
    assert row["specificity"] >= 0.9
    assert row["n_samples"] == 80


def test_audio_metadata_residual_correlation_aligns_participants() -> None:
    audio = _prediction_frame()
    metadata = audio.copy()
    metadata["probability"] = np.clip(audio["probability"] * 0.95 + 0.02, 0.0, 1.0)
    audio["prediction_source"] = "audio"
    metadata["prediction_source"] = "metadata"
    audio["model_name"] = "audio_model"
    metadata["model_name"] = "metadata_model"

    table = build_audio_metadata_residual_correlation(
        audio,
        metadata,
        group_columns=["split"],
    )

    assert len(table) == 1
    row = table.iloc[0]
    assert row["n_aligned"] == 80
    assert row["probability_pearson"] > 0.99
    assert row["audio_abs_error_vs_metadata_probability_pearson"] < 0


def test_partial_target_recalibration_emits_original_and_calibrated_rows() -> None:
    predictions = _prediction_frame(100)
    predictions["probability"] = np.where(predictions["label_binary"].eq("positive"), 0.58, 0.42)

    metrics, recalibrated = build_partial_target_recalibration(
        predictions,
        group_columns=["evaluation_protocol", "model_name", "split"],
        calibration_fraction=0.3,
        random_state=13,
        methods=["platt", "isotonic"],
    )

    assert set(metrics["recalibration_method"]) == {"original", "platt", "isotonic"}
    assert set(recalibrated["recalibration_method"]) == {"original", "platt", "isotonic"}
    assert metrics["n_calibration"].nunique() == 1
    assert metrics["n_evaluation"].nunique() == 1
    assert int(metrics["n_evaluation"].iloc[0]) == 70
