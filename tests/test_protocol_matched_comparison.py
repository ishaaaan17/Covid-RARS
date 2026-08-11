from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_id": "eswa_dndf_internal_coswara",
                "paper_name": "ESWA DNDF",
                "modality": "cough",
                "dataset": "Coswara",
                "split_unit": "unclear_in_paper_text",
                "split_style": "10-fold stratified cross-validation",
                "n_splits": 10,
                "reported_metric": "AUROC",
                "reported_value": 0.92,
                "comparison_notes": "internal",
            },
            {
                "paper_id": "eswa_dndf_cross_coswara_to_coughvid",
                "paper_name": "ESWA DNDF",
                "modality": "cough",
                "dataset": "Coswara to COUGHVID",
                "split_unit": "dataset_level",
                "split_style": "cross-dataset train-source test-target",
                "n_splits": 1,
                "reported_metric": "AUROC",
                "reported_value": 0.53,
                "comparison_notes": "external",
            },
        ]
    )


def _protocol_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evaluation_protocol": "protocol_matched_participant_10fold_cv",
                "analysis_family": "protocol_matched_cv",
                "model_name": "svc_rbf_f60",
                "modality": "cough",
                "feature_strategy": "compare_is10_top800_lightgbm",
                "selected_feature_k": 800.0,
                "fold_unit": "participant",
                "threshold_source": "inner_validation_balanced_accuracy",
                "metric_split": "test_aggregate",
                "fold": "aggregate",
                "auroc": 0.89,
                "auprc": 0.80,
                "n_samples": 100.0,
                "skipped": False,
            }
        ]
    )


def test_protocol_gap_summary_separates_internal_and_cross_dataset_targets() -> None:
    from covid_rars.protocol_matched_comparison import build_protocol_matched_gap_summary

    external = pd.DataFrame(
        [
            {
                "family_model": "compare_is10_lightgbm_smote_f80",
                "model_family": "compare_is10_handcrafted",
                "external_auroc": 0.543,
                "external_auprc": 0.04,
                "external_n_samples": 8331,
            }
        ]
    )
    summary = build_protocol_matched_gap_summary(
        _targets(),
        _protocol_metrics(),
        external_transfer_summary=external,
    )

    internal = summary[summary["paper_id"].eq("eswa_dndf_internal_coswara")].iloc[0]
    cross = summary[summary["paper_id"].eq("eswa_dndf_cross_coswara_to_coughvid")].iloc[0]
    assert internal["comparison_type"] == "paper_style_internal"
    assert internal["our_protocol_matched_auroc"] == 0.89
    assert round(float(internal["paper_minus_our_protocol_matched_auroc"]), 3) == 0.03
    assert cross["comparison_type"] == "cross_dataset_transfer"
    assert cross["our_external_auroc"] == 0.543
    assert bool(cross["skipped"]) is False


def test_protocol_gap_cli_writes_summary(tmp_path: Path, monkeypatch) -> None:
    targets_path = tmp_path / "targets.csv"
    metrics_path = tmp_path / "metrics.csv"
    external_path = tmp_path / "external.csv"
    output_path = tmp_path / "summary.csv"
    _targets().to_csv(targets_path, index=False)
    _protocol_metrics().to_csv(metrics_path, index=False)
    pd.DataFrame(
        [
            {
                "family_model": "compare_is10_lightgbm_smote_f80",
                "model_family": "compare_is10_handcrafted",
                "external_auroc": 0.543,
                "external_auprc": 0.04,
                "external_n_samples": 8331,
            }
        ]
    ).to_csv(external_path, index=False)

    script = Path(__file__).parents[1] / "scripts" / "70_make_protocol_matched_gap_table.py"
    spec = importlib.util.spec_from_file_location("protocol_gap_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = [
        "70_make_protocol_matched_gap_table.py",
        "--targets",
        str(targets_path),
        "--protocol-metrics",
        str(metrics_path),
        "--external-transfer-summary",
        str(external_path),
        "--output",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    summary = pd.read_csv(output_path)
    assert set(summary["paper_id"]) == {
        "eswa_dndf_internal_coswara",
        "eswa_dndf_cross_coswara_to_coughvid",
    }
