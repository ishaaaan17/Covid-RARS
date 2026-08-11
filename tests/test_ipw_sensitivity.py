from __future__ import annotations

import pandas as pd


def _toy_metadata_and_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_rows = []
    prediction_rows = []
    labels = ["positive"] * 12 + ["negative"] * 12
    countries = ["India"] * 10 + ["Canada"] * 2 + ["India"] * 2 + ["Canada"] * 10
    for idx, (label, country) in enumerate(zip(labels, countries)):
        pid = f"p{idx}"
        metadata_rows.append(
            {
                "participant_id": pid,
                "recording_id": f"r{idx}",
                "label_binary": label,
                "split": "test",
                "age": 55 if label == "positive" else 30,
                "gender": "female" if idx % 2 else "male",
                "country": country,
                "recording_date": f"2020-06-{(idx % 28) + 1:02d}",
                "duration_sec": 8.0 if label == "positive" else 4.0,
                "sample_rate_original": 48000,
                "quality_flag": "ok",
            }
        )
        prediction_rows.append(
            {
                "participant_id": pid,
                "label_binary": label,
                "split": "test",
                "probability": 0.75 if label == "positive" else 0.25,
                "fusion_method": "quality_weighted_auprc",
            }
        )
    return pd.DataFrame(metadata_rows), pd.DataFrame(prediction_rows)


def test_ipw_sensitivity_reports_caps_ess_and_balance() -> None:
    from covid_rars.ipw_sensitivity import run_ipw_sensitivity_analysis

    metadata, predictions = _toy_metadata_and_predictions()

    result = run_ipw_sensitivity_analysis(
        predictions,
        metadata,
        covariates=["country", "age", "gender"],
        group_columns=["fusion_method"],
        weight_caps=[2.0, 5.0],
        clip_quantiles=[1.0],
        threshold=0.5,
    )

    assert {"unweighted", "ipw_label_propensity"}.issubset(set(result.metrics["control_method"]))
    weighted = result.metrics[result.metrics["control_method"].eq("ipw_label_propensity")]
    assert set(weighted["weight_cap"]) == {2.0, 5.0}
    assert weighted["effective_sample_size"].between(0, len(predictions)).all()
    assert {"mean_abs_smd_before", "mean_abs_smd_after", "max_abs_smd_after"}.issubset(result.metrics.columns)
    assert not result.balance.empty
    assert not result.weights.empty
