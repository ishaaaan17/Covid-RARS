from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from covid_rars.reviewer_extension_checks import (
    build_context_control_exposure_table,
    build_decision_curve_table,
    build_duration_shortcut_table,
    build_feature_selection_stability,
    build_label_construction_audit_table,
    build_nested_metadata_audio_comparison,
    build_performance_equity_table,
    build_quality_label_month_table,
    build_specification_curve,
    build_support_overlap_diagnostic,
    paired_delong_auc_comparison,
)


def _load_reviewer_extension_cli():
    script_path = Path(__file__).parents[1] / "scripts" / "67_run_reviewer_extension_checks.py"
    spec = importlib.util.spec_from_file_location("reviewer_extension_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predictions(n: int = 80, split: str = "test") -> pd.DataFrame:
    y = np.array(["negative"] * (n // 2) + ["positive"] * (n // 2), dtype=object)
    prob = np.concatenate([np.linspace(0.05, 0.35, n // 2), np.linspace(0.65, 0.95, n // 2)])
    return pd.DataFrame(
        {
            "participant_id": [f"p{i:03d}" for i in range(n)],
            "recording_id": [f"r{i:03d}" for i in range(n)],
            "label_binary": y,
            "probability": prob,
            "model_name": "unit_model",
            "evaluation_protocol": "unit_protocol",
            "split": split,
            "threshold": 0.5,
        }
    )


def test_specification_curve_ranks_all_non_skipped_test_rows() -> None:
    metrics = pd.DataFrame(
        {
            "evaluation_protocol": ["a", "a", "b"],
            "model_name": ["m1", "m2", "m3"],
            "metric_split": ["test", "test", "validation"],
            "auroc": [0.7, 0.9, 0.99],
            "auprc": [0.6, 0.8, 0.95],
            "skipped": [False, False, False],
        }
    )

    curve = build_specification_curve(metrics)

    assert list(curve["model_name"]) == ["m2", "m1"]
    assert list(curve["specification_rank"]) == [1, 2]
    assert curve.iloc[0]["n_specifications_in_protocol"] == 2


def test_quality_label_month_table_reports_counts_and_prevalence() -> None:
    metadata = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p3", "p4"],
            "label_binary": ["positive", "negative", "positive", "negative"],
            "quality_flag": ["ok", "ok", "short", "short"],
            "recording_date": ["2020-04-01", "2020-04-02", "2020-05-01", "2020-05-02"],
            "modality": ["cough"] * 4,
            "split": ["test"] * 4,
        }
    )

    table = build_quality_label_month_table(metadata)

    assert {"quality_flag", "recording_year_month", "positive_prevalence"}.issubset(table.columns)
    ok_row = table[table["quality_flag"].eq("ok")].iloc[0]
    assert ok_row["n_rows"] == 2
    assert ok_row["positive_prevalence"] == 0.5


def test_decision_curve_table_computes_net_benefit_against_reference_policies() -> None:
    predictions = _predictions()

    table = build_decision_curve_table(
        predictions,
        group_columns=["evaluation_protocol", "model_name", "split"],
        thresholds=[0.2, 0.5],
    )

    assert len(table) == 2
    assert (table["model_net_benefit"] > table["treat_none_net_benefit"]).all()
    assert "net_benefit_minus_treat_all" in table.columns


def test_duration_shortcut_table_aligns_metadata_and_predictions() -> None:
    predictions = _predictions()
    metadata = predictions[["participant_id", "recording_id"]].copy()
    metadata["duration_sec"] = predictions["probability"] * 10 + 1

    table = build_duration_shortcut_table(
        predictions,
        metadata,
        group_columns=["evaluation_protocol", "model_name", "split"],
    )

    assert len(table) == 1
    assert table.iloc[0]["probability_duration_pearson"] > 0.99
    assert table.iloc[0]["n_aligned"] == len(predictions)


def test_performance_equity_table_reports_subgroup_metrics() -> None:
    predictions = _predictions()
    metadata = predictions[["participant_id", "recording_id"]].copy()
    metadata["gender"] = ["female", "male"] * (len(metadata) // 2)

    table = build_performance_equity_table(
        predictions,
        metadata,
        subgroup_columns=["gender"],
        group_columns=["evaluation_protocol", "model_name", "split"],
        min_subgroup_size=4,
    )

    assert set(table["subgroup_column"]) == {"gender"}
    assert set(table["subgroup_value"]) == {"female", "male"}
    assert {"sensitivity", "specificity", "positive_prevalence"}.issubset(table.columns)


def test_nested_metadata_audio_comparison_fits_validation_combiner() -> None:
    validation = _predictions(split="validation")
    test = _predictions(split="test")
    audio = pd.concat([validation, test], ignore_index=True)
    metadata = audio.copy()
    metadata["probability"] = np.clip(audio["probability"] * 0.85 + 0.07, 0.0, 1.0)
    metadata["model_name"] = "metadata_model"

    metrics, predictions = build_nested_metadata_audio_comparison(audio, metadata)

    assert {"audio_only", "metadata_only", "metadata_plus_audio"} == set(metrics["nested_model"])
    assert not predictions.empty
    combined = metrics[metrics["nested_model"].eq("metadata_plus_audio")].iloc[0]
    assert combined["incremental_auroc_over_metadata"] >= -1e-9
    assert combined["comparison_level"] == "participant_prediction_level"
    assert "paired_delong_p_value_vs_metadata" in metrics.columns


def test_support_overlap_diagnostic_flags_shifted_external_distribution() -> None:
    source = pd.DataFrame(
        {
            "split": ["train"] * 80,
            "modality": ["cough"] * 80,
            "f1": np.linspace(-1, 1, 80),
            "f2": np.linspace(-1, 1, 80),
        }
    )
    external = pd.DataFrame({"modality": ["cough"] * 80, "f1": np.linspace(4, 6, 80), "f2": np.linspace(4, 6, 80)})

    table = build_support_overlap_diagnostic(source, external)

    assert len(table) == 1
    assert table.iloc[0]["source_split_scope"] == "source_train"
    assert table.iloc[0]["domain_classifier_auroc"] > 0.95
    assert table.iloc[0]["external_probably_outside_source_support_fraction"] > 0.5


def test_feature_selection_stability_reports_jaccard_overlap() -> None:
    rng = np.random.default_rng(4)
    n = 80
    labels = np.array(["negative", "positive"] * 40)
    label_int = (labels == "positive").astype(float)
    features = pd.DataFrame(
        {
            "recording_id": [f"r{i}" for i in range(n)],
            "participant_id": [f"p{i}" for i in range(n)],
            "label_binary": labels,
            "split": ["train"] * n,
            "recording_date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
            "stable_signal": label_int + rng.normal(0, 0.1, n),
            "noise": rng.normal(size=n),
        }
    )

    table = build_feature_selection_stability(features, top_k=1, ranker="univariate")

    assert table.iloc[0]["top_k"] == 1
    assert "jaccard_overlap" in table.columns
    assert table.iloc[0]["n_early_rows"] > 0
    assert table.iloc[0]["n_late_rows"] > 0


def test_paired_delong_auc_comparison_returns_delta_ci_and_p_value() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    stronger = np.array([0.05, 0.2, 0.8, 0.9, 0.1, 0.7])
    weaker = np.array([0.3, 0.4, 0.6, 0.7, 0.2, 0.55])

    result = paired_delong_auc_comparison(y, stronger, weaker)

    assert result["left_auc"] >= result["right_auc"]
    assert "delta_ci_low" in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_context_control_exposure_table_uses_available_context_columns() -> None:
    metadata = pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(24)],
            "label_binary": ["negative", "positive"] * 12,
            "split": ["train"] * 12 + ["test"] * 12,
            "sample_rate_original": [16000, 48000] * 12,
        }
    )

    table = build_context_control_exposure_table(metadata, candidate_columns=["sample_rate_original"])

    assert len(table) == 1
    assert table.iloc[0]["context_column"] == "sample_rate_original"
    assert table.iloc[0]["skipped"] is False or table.iloc[0]["skipped"] == False


def test_label_construction_audit_table_reports_processed_label_counts() -> None:
    metadata = pd.DataFrame(
        {
            "participant_id": ["p1", "p2", "p3"],
            "dataset": ["coswara", "coswara", "coswara"],
            "label_binary": ["positive", "negative", "unknown"],
            "label_raw": ["healthy", "covid", "missing"],
        }
    )

    table = build_label_construction_audit_table(metadata)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["dataset"] == "coswara"
    assert row["n_positive"] == 1
    assert row["n_negative"] == 1
    assert row["n_unknown_or_other"] == 1
    assert "label_raw" in row["available_label_columns"]


def test_cli_helpers_combine_external_metadata_and_restrict_modalities() -> None:
    module = _load_reviewer_extension_cli()
    source_meta = pd.DataFrame({"recording_id": ["c1"], "dataset": ["coswara"]})
    external_meta = pd.DataFrame({"recording_id": ["e1"], "dataset": ["coughvid"]})
    combined = module._combined_context_metadata(source_meta, external_meta)

    assert set(combined["dataset"]) == {"coswara", "coughvid"}

    source_features = pd.DataFrame({"modality": ["cough", "breath"], "f": [1.0, 2.0]})
    external_features = pd.DataFrame({"modality": ["cough"], "f": [3.0]})
    source_filtered, external_filtered = module._restrict_to_common_modalities(source_features, external_features)

    assert set(source_filtered["modality"]) == {"cough"}
    assert set(external_filtered["modality"]) == {"cough"}
