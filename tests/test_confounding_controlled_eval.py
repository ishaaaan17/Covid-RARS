from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _toy_metadata_and_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    labels = ["positive"] * 10 + ["negative"] * 10
    countries = ["India"] * 8 + ["Canada"] * 2 + ["India"] * 2 + ["Canada"] * 8
    for i, (label, country) in enumerate(zip(labels, countries)):
        pid = f"p{i}"
        rows.append(
            {
                "participant_id": pid,
                "recording_id": f"r{i}_cough",
                "label_binary": label,
                "split": "test",
                "age": 52 if label == "positive" else 31,
                "gender": "male" if i % 2 else "female",
                "country": country,
                "recording_date": f"2020-05-{(i % 28) + 1:02d}",
                "duration_sec": 8.0 if label == "positive" else 5.0,
                "sample_rate_original": 48000 if i % 2 else 44100,
                "quality_flag": "ok",
            }
        )
        preds.append(
            {
                "participant_id": pid,
                "label_binary": label,
                "split": "test",
                "probability": 0.70 if label == "positive" else 0.30,
                "fusion_method": "quality_weighted_auprc",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(preds)


def test_participant_predictions_merge_to_confounders_without_duplication():
    from covid_rars.confounding_controlled_eval import merge_predictions_with_confounders

    metadata, predictions = _toy_metadata_and_predictions()

    merged = merge_predictions_with_confounders(predictions, metadata)

    assert len(merged) == len(predictions)
    assert {"age", "country", "recording_date", "duration_sec", "quality_flag"}.issubset(merged.columns)
    assert merged["participant_id"].is_unique


def test_ipw_controlled_metrics_reduce_country_imbalance_and_preserve_groups():
    from covid_rars.confounding_controlled_eval import evaluate_confounding_controlled_predictions

    metadata, predictions = _toy_metadata_and_predictions()

    result = evaluate_confounding_controlled_predictions(
        predictions,
        metadata,
        covariates=["country", "age", "gender"],
        group_columns=["fusion_method"],
    )

    assert set(result.metrics["control_method"]) == {"unweighted", "ipw_label_propensity"}
    assert set(result.metrics["fusion_method"]) == {"quality_weighted_auprc"}
    weighted = result.metrics[result.metrics["control_method"] == "ipw_label_propensity"].iloc[0]
    assert 0 < weighted["effective_sample_size"] <= weighted["n_samples"]

    country_rows = result.balance[result.balance["feature"].str.startswith("country_")]
    assert not country_rows.empty
    assert country_rows["after_abs_smd"].max() < country_rows["before_abs_smd"].max()


def test_confounding_controlled_cli_writes_metrics_and_balance(tmp_path, monkeypatch):
    metadata, predictions = _toy_metadata_and_predictions()
    metadata_path = tmp_path / "metadata.csv"
    predictions_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.csv"
    balance_path = tmp_path / "balance.csv"
    weights_path = tmp_path / "weights.csv"
    metadata.to_csv(metadata_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    script_path = Path(__file__).parents[1] / "scripts" / "30_confounding_controlled_audio_eval.py"
    spec = importlib.util.spec_from_file_location("confounding_controlled_audio_eval_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "30_confounding_controlled_audio_eval.py",
            "--predictions",
            str(predictions_path),
            "--metadata",
            str(metadata_path),
            "--metrics-output",
            str(metrics_path),
            "--balance-output",
            str(balance_path),
            "--weights-output",
            str(weights_path),
            "--covariates",
            "country",
            "age",
            "gender",
            "--group-columns",
            "fusion_method",
        ],
    )

    module.main()

    assert set(pd.read_csv(metrics_path)["control_method"]) == {"unweighted", "ipw_label_propensity"}
    assert not pd.read_csv(balance_path).empty
    assert "ipw_weight" in pd.read_csv(weights_path).columns
