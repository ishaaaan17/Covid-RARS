from __future__ import annotations

import pandas as pd


def _metadata_rows() -> pd.DataFrame:
    rows = []
    for split, n in [("train", 12), ("validation", 6), ("test", 6)]:
        for idx in range(n):
            positive = idx % 2 == 0
            rows.append(
                {
                    "participant_id": f"{split}_{idx}",
                    "recording_id": f"rec_{split}_{idx}",
                    "dataset": "coswara",
                    "split": split,
                    "label_binary": "positive" if positive else "negative",
                    "age": 45 if positive else 25,
                    "gender": "male" if positive else "female",
                    "country": "India" if positive else "Canada",
                    "recording_date": "2021-05-01" if positive else "2020-04-01",
                    "duration_sec": 8.0 if positive else 4.0,
                    "sample_rate_original": 48000 if positive else 44100,
                    "quality_flag": "ok" if positive else "short",
                    "symptoms_json": "{}",
                    "comorbidities_json": "{}",
                }
            )
    rows.append(
        {
            "participant_id": "unk_1",
            "recording_id": "rec_unk_1",
            "dataset": "coswara",
            "split": "unused",
            "label_binary": "unknown",
            "age": 60,
            "gender": "male",
            "country": "India",
            "recording_date": "2022-01-01",
            "duration_sec": 3.0,
            "sample_rate_original": 16000,
            "quality_flag": "mostly_silence",
            "symptoms_json": "{}",
            "comorbidities_json": "{}",
        }
    )
    return pd.DataFrame(rows)


def test_linear_metadata_shap_ranks_demographic_protocol_features() -> None:
    from covid_rars.manuscript_support import linear_metadata_shap_table

    table = linear_metadata_shap_table(_metadata_rows(), top_n=5, random_state=0)

    assert not table.empty
    assert table.iloc[0]["audit_model"] == "demographic_protocol_only"
    assert table["mean_abs_shap"].is_monotonic_decreasing
    assert {"feature", "feature_group", "mean_abs_shap", "coefficient"}.issubset(table.columns)
    assert table["feature"].head(5).notna().all()


def test_ipw_residual_smd_table_filters_cap_and_sorts() -> None:
    from covid_rars.manuscript_support import ipw_residual_smd_table

    balance = pd.DataFrame(
        {
            "feature": ["age", "country_India", "recording_year"],
            "after_abs_smd": [0.2, 0.724, 0.4],
            "before_abs_smd": [0.5, 0.8, 0.7],
            "weight_config": ["ipw_cap_2_q_0.95", "ipw_cap_2_q_0.95", "ipw_cap_3_q_0.95"],
            "control_method": ["ipw_label_propensity"] * 3,
        }
    )

    table = ipw_residual_smd_table(balance, weight_config="ipw_cap_2_q_0.95")

    assert list(table["feature"]) == ["country_India", "age"]
    assert table.iloc[0]["balance_severity"] == "severe_residual_imbalance"


def test_auprc_lift_over_prevalence_table_marks_near_random() -> None:
    from covid_rars.manuscript_support import auprc_lift_over_prevalence_table

    metrics = {
        "mfcc": pd.DataFrame({"auroc": [0.53], "auprc": [0.042], "model_name": ["lr"], "feature_strategy": ["all"]}),
        "panns": pd.DataFrame({"auroc": [0.50], "auprc": [0.035], "model_name": ["lr"], "feature_strategy": ["all"]}),
    }

    table = auprc_lift_over_prevalence_table(metrics, target_prevalence=0.034)

    assert table.loc[table["representation"].eq("mfcc"), "absolute_auprc_lift"].iloc[0] == 0.008
    assert table.loc[table["representation"].eq("panns"), "pr_lift_interpretation"].iloc[0] == "near_prevalence"


def test_unknown_label_audit_compares_known_and_unknown_rows() -> None:
    from covid_rars.manuscript_support import unknown_label_audit_tables

    summary, balance = unknown_label_audit_tables(_metadata_rows())

    assert set(summary["label_availability"]) == {"known", "unknown"}
    assert {"feature", "comparison_type", "abs_difference"}.issubset(balance.columns)
    assert "country_India" in set(balance["feature"])
