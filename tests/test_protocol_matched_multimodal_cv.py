from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _multimodal_feature_frame(n_participants: int = 48, modalities: tuple[str, ...] = ("cough", "speech")) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant_idx in range(n_participants):
        label = "positive" if participant_idx % 4 in {1, 2} else "negative"
        y = 1.0 if label == "positive" else -1.0
        for modality in modalities:
            for recording_idx in range(2):
                idx = len(rows)
                modality_offset = 0.2 if modality == "speech" else 0.0
                rows.append(
                    {
                        "recording_id": f"{modality}_rec_{idx:03d}",
                        "participant_id": f"p_{participant_idx:03d}",
                        "dataset": "coswara",
                        "modality": modality,
                        "submodality": modality,
                        "label_binary": label,
                        "quality_flag": "ok",
                        "signal": y + modality_offset + recording_idx * 0.01,
                        "weak_signal": y * 0.2 + (participant_idx % 5) * 0.01,
                        "noise": (idx % 7) * 0.03,
                    }
                )
    return pd.DataFrame(rows)


def test_protocol_matched_multimodal_cv_keeps_participants_disjoint_and_reports_fusion() -> None:
    from covid_rars.protocol_matched_multimodal_cv import run_protocol_matched_multimodal_cv

    result = run_protocol_matched_multimodal_cv(
        _multimodal_feature_frame(),
        modalities=["cough", "speech"],
        n_splits=3,
        test_fraction=0.25,
        validation_fraction=0.25,
        top_k_values=[2],
        ranker="univariate",
        selection_scope="per_modality_mean",
        model_names=["logistic_l2_f80"],
        random_state=0,
        ensemble_top_k=1,
    )

    assert not result.metrics.empty
    assert not result.predictions.empty
    assert set(result.split_audit["overlap_count"]) == {0}
    assert set(result.metrics["fold_unit"].dropna()) == {"participant"}
    assert set(result.feature_selection["selection_scope"]) == {"per_modality_mean"}

    fusion = result.metrics[
        result.metrics["analysis_family"].eq("strong_multimodal_fusion")
        & result.metrics["metric_split"].eq("test_aggregate")
    ]
    assert not fusion.empty
    assert "cough+speech" in set(fusion["modality_combination"])
    assert "stacked_logistic_validation" in set(fusion["fusion_method"])
    summary = result.summary
    assert not summary.empty
    assert summary.iloc[0]["metric_split"] == "test_aggregate"


def test_protocol_matched_multimodal_cv_aggregate_keeps_fusion_methods_separate() -> None:
    from covid_rars.protocol_matched_multimodal_cv import aggregate_protocol_matched_metrics

    metrics = pd.DataFrame(
        [
            {
                "evaluation_protocol": "protocol_matched_multimodal_participant_10fold_cv",
                "analysis_family": "strong_multimodal_fusion",
                "model_name": "strong_baseline_selected_fusion",
                "modality": "multimodal",
                "modality_combination": "cough+speech",
                "fusion_method": "uniform_mean",
                "feature_strategy": "compare_is10_top2_univariate",
                "selected_feature_k": 2.0,
                "fold": 1,
                "fold_unit": "participant",
                "metric_split": "test",
                "threshold_source": "validation_balanced_accuracy",
                "auroc": 0.7,
                "auprc": 0.6,
                "balanced_accuracy": 0.65,
                "n_samples": 10.0,
            },
            {
                "evaluation_protocol": "protocol_matched_multimodal_participant_10fold_cv",
                "analysis_family": "strong_multimodal_fusion",
                "model_name": "strong_baseline_selected_fusion",
                "modality": "multimodal",
                "modality_combination": "cough+speech",
                "fusion_method": "stacked_logistic_validation",
                "feature_strategy": "compare_is10_top2_univariate",
                "selected_feature_k": 2.0,
                "fold": 1,
                "fold_unit": "participant",
                "metric_split": "test",
                "threshold_source": "validation_balanced_accuracy",
                "auroc": 0.8,
                "auprc": 0.7,
                "balanced_accuracy": 0.75,
                "n_samples": 10.0,
            },
        ]
    )

    aggregate = aggregate_protocol_matched_metrics(metrics)

    assert len(aggregate) == 2
    assert set(aggregate["fusion_method"]) == {"uniform_mean", "stacked_logistic_validation"}


def test_protocol_matched_multimodal_cv_allows_explicit_feature_strategy_label() -> None:
    from covid_rars.protocol_matched_multimodal_cv import run_protocol_matched_multimodal_cv

    result = run_protocol_matched_multimodal_cv(
        _multimodal_feature_frame(),
        modalities=["cough", "speech"],
        n_splits=2,
        test_fraction=0.25,
        validation_fraction=0.25,
        top_k_values=[2],
        ranker="univariate",
        selection_scope="per_modality_mean",
        feature_strategy_label="fixed_top800_feature_bank_sensitivity",
        model_names=["logistic_l2_f80"],
        random_state=0,
        ensemble_top_k=1,
    )

    assert set(result.metrics["feature_strategy"].dropna()) == {"fixed_top800_feature_bank_sensitivity"}
    assert set(result.predictions["feature_strategy"].dropna()) == {"fixed_top800_feature_bank_sensitivity"}
    assert set(result.feature_selection["feature_strategy"].dropna()) == {"fixed_top800_feature_bank_sensitivity"}
    assert set(result.summary["feature_strategy"].dropna()) == {"fixed_top800_feature_bank_sensitivity"}


def test_protocol_matched_multimodal_cli_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    features_path = tmp_path / "features.csv"
    _multimodal_feature_frame().to_csv(features_path, index=False)

    script = Path(__file__).parents[1] / "scripts" / "71_run_protocol_matched_multimodal_cv.py"
    spec = importlib.util.spec_from_file_location("protocol_multimodal_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metrics_output = tmp_path / "metrics.csv"
    predictions_output = tmp_path / "predictions.csv"
    selection_output = tmp_path / "selection.csv"
    split_audit_output = tmp_path / "split_audit.csv"
    branch_selection_output = tmp_path / "branch_selection.csv"
    summary_output = tmp_path / "summary.csv"
    argv = [
        "71_run_protocol_matched_multimodal_cv.py",
        "--features",
        str(features_path),
        "--modalities",
        "cough",
        "speech",
        "--n-splits",
        "3",
        "--test-fraction",
        "0.25",
        "--validation-fraction",
        "0.25",
        "--top-k-values",
        "2",
        "--ranker",
        "univariate",
        "--selection-scope",
        "per_modality_mean",
        "--feature-strategy-label",
        "fixed_top800_feature_bank_sensitivity",
        "--model-names",
        "logistic_l2_f80",
        "--ensemble-top-k",
        "1",
        "--metrics-output",
        str(metrics_output),
        "--predictions-output",
        str(predictions_output),
        "--feature-selection-output",
        str(selection_output),
        "--split-audit-output",
        str(split_audit_output),
        "--branch-selection-output",
        str(branch_selection_output),
        "--summary-output",
        str(summary_output),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    metrics = pd.read_csv(metrics_output)
    split_audit = pd.read_csv(split_audit_output)
    summary = pd.read_csv(summary_output)
    assert "test_aggregate" in set(metrics["metric_split"])
    assert set(metrics["feature_strategy"].dropna()) == {"fixed_top800_feature_bank_sensitivity"}
    assert set(split_audit["overlap_count"]) == {0}
    assert not summary.empty
