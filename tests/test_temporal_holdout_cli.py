from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from test_temporal_holdout import _feature_frame, _metadata_frame


def test_temporal_holdout_cli_writes_expected_artifacts(tmp_path: Path) -> None:
    metadata = _metadata_frame()
    features = _feature_frame(metadata)
    metadata_path = tmp_path / "metadata.csv"
    features_path = tmp_path / "features.csv"
    metrics_path = tmp_path / "metrics.csv"
    predictions_path = tmp_path / "predictions.csv"
    split_summary_path = tmp_path / "split_summary.csv"
    coverage_path = tmp_path / "coverage.csv"
    importance_path = tmp_path / "importance.csv"
    group_summary_path = tmp_path / "group_summary.csv"
    metadata_ablation_path = tmp_path / "metadata_ablation.csv"
    stability_summary_path = tmp_path / "stability_summary.csv"
    bootstrap_ci_path = tmp_path / "bootstrap_ci.csv"
    external_unification_path = tmp_path / "external_unification.csv"
    metadata.to_csv(metadata_path, index=False)
    features.to_csv(features_path, index=False)

    script = Path(__file__).parents[1] / "scripts" / "44_temporal_holdout_audit.py"
    spec = importlib.util.spec_from_file_location("temporal_holdout_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = [
        "44_temporal_holdout_audit.py",
        "--metadata",
        str(metadata_path),
        "--features",
        str(features_path),
        "--model-names",
        "logistic_regression",
        "--metrics-output",
        str(metrics_path),
        "--predictions-output",
        str(predictions_path),
        "--split-summary-output",
        str(split_summary_path),
        "--modality-coverage-output",
        str(coverage_path),
        "--metadata-importance-output",
        str(importance_path),
        "--metadata-group-summary-output",
        str(group_summary_path),
        "--metadata-ablation-output",
        str(metadata_ablation_path),
        "--stability-summary-output",
        str(stability_summary_path),
        "--bootstrap-ci-output",
        str(bootstrap_ci_path),
        "--external-unification-output",
        str(external_unification_path),
        "--bootstrap-samples",
        "25",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old_argv

    metrics = pd.read_csv(metrics_path)
    assert {"audio_modality", "multimodal_fusion", "metadata_confounding"}.issubset(
        set(metrics["analysis_family"])
    )
    fusion = metrics[metrics["analysis_family"].eq("multimodal_fusion")]
    assert {"breath+cough", "cough+speech", "breath+speech", "breath+cough+speech"}.issubset(
        set(fusion["modality_combination"])
    )
    assert "auprc_lift_over_prevalence" in metrics.columns
    assert predictions_path.exists()
    assert split_summary_path.exists()
    assert coverage_path.exists()
    assert importance_path.exists()
    assert group_summary_path.exists()
    assert metadata_ablation_path.exists()
    assert stability_summary_path.exists()
    assert bootstrap_ci_path.exists()
    assert external_unification_path.exists()
    ablation = pd.read_csv(metadata_ablation_path)
    assert "demographic_protocol_no_recording_year_month" in set(ablation["ablation_name"])
    stability = pd.read_csv(stability_summary_path)
    assert "delta_auroc_temporal_minus_existing" in stability.columns
    bootstrap = pd.read_csv(bootstrap_ci_path)
    assert "ci_low" in bootstrap.columns
