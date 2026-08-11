from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "20_make_paper_tables.py"
    spec = importlib.util.spec_from_file_location("make_paper_tables_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_metric_paths_discovers_representation_outputs(tmp_path) -> None:
    module = _load_script_module()
    metrics_dir = tmp_path / "data" / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "external_model_grid_opensmile_egemaps_metrics.csv").write_text("auroc\\n0.5\\n")
    (metrics_dir / "coughvid_internal_beats_bootstrap_ci.csv").write_text("metric\\nauroc\\n")
    (metrics_dir / "coughvid_internal_beats_metrics.csv").write_text("auroc\\n0.7\\n")
    (metrics_dir / "sota_ssl_metrics_hf_ssl_microsoft_wavlm-base-plus_cough.csv").write_text("auroc\\n0.9\\n")
    (metrics_dir / "sota_fusion_metrics.csv").write_text("auroc\\n0.93\\n")
    (metrics_dir / "sota_swarm_feature_metrics.csv").write_text("auroc\\n0.91\\n")
    (metrics_dir / "sota_gated_stack_metrics.csv").write_text("auroc\\n0.92\\n")

    metric_paths = module.default_metric_paths(metrics_dir=metrics_dir)
    ci_paths = module.default_ci_paths(metrics_dir=metrics_dir)

    assert Path("reports/tables/calibration_under_shift_summary.csv") in metric_paths
    assert Path("reports/tables/final_validation_fixed_sensitivity_operating_points.csv") in metric_paths
    assert Path("reports/tables/coughvid_partial_recalibration_metrics.csv") in metric_paths
    assert Path("reports/tables/reviewer_nested_metadata_audio_comparison.csv") in metric_paths
    assert Path("reports/tables/reviewer_context_control_exposure.csv") in metric_paths
    assert Path("data/outputs/metrics/metadata_permutation_importance_metrics.csv") in metric_paths
    assert Path("data/outputs/metrics/domain_shift_audit_metrics.csv") in metric_paths
    assert Path("data/outputs/metrics/domain_adaptation_baseline_metrics.csv") in metric_paths
    assert Path("data/outputs/metrics/ipw_sensitivity_metrics.csv") in metric_paths
    assert Path("reports/tables/external_prevalence_recalibration.csv") in metric_paths
    assert Path("data/outputs/metrics/temporal_holdout_metrics.csv") in metric_paths
    assert Path("reports/tables/temporal_metadata_ablation.csv") in metric_paths
    assert Path("reports/tables/temporal_matched_cohort_metrics.csv") in metric_paths
    assert metrics_dir / "external_model_grid_opensmile_egemaps_metrics.csv" in metric_paths
    assert metrics_dir / "coughvid_internal_beats_metrics.csv" in metric_paths
    assert metrics_dir / "sota_ssl_metrics_hf_ssl_microsoft_wavlm-base-plus_cough.csv" in metric_paths
    assert metrics_dir / "sota_fusion_metrics.csv" in metric_paths
    assert metrics_dir / "sota_swarm_feature_metrics.csv" in metric_paths
    assert metrics_dir / "sota_gated_stack_metrics.csv" in metric_paths
    (metrics_dir / "compare_is10_shuffle_retrain_metrics.csv").write_text("auroc\n0.51\n")
    metric_paths = module.default_metric_paths(metrics_dir=metrics_dir)
    assert metrics_dir / "compare_is10_shuffle_retrain_metrics.csv" in metric_paths
    assert Path("data/outputs/metrics/temporal_holdout_bootstrap_ci.csv") in ci_paths
    assert metrics_dir / "coughvid_internal_beats_bootstrap_ci.csv" in ci_paths


def test_read_existing_csvs_skips_headerless_empty_files(tmp_path) -> None:
    import pandas as pd

    from covid_rars.reporting import read_existing_csvs

    empty_path = tmp_path / "empty_metrics.csv"
    good_path = tmp_path / "good_metrics.csv"
    empty_path.write_text("\n")
    good_path.write_text("auroc\n0.75\n")

    combined = read_existing_csvs([empty_path, good_path])

    assert len(combined) == 1
    assert pd.to_numeric(combined["auroc"]).iloc[0] == 0.75
    assert combined["table_source"].iloc[0] == "good_metrics"


def test_group_columns_preserve_temporal_ablation_identity() -> None:
    import pandas as pd

    module = _load_script_module()
    metrics = pd.DataFrame(
        columns=[
            "table_source",
            "evaluation_protocol",
            "analysis_family",
            "model_name",
            "modality",
            "modality_combination",
            "base_feature_set",
            "ablation_name",
            "removed_features",
            "feature_strategy",
            "nested_model",
            "auroc",
        ]
    )

    group_columns = module._group_columns(metrics)

    assert "evaluation_protocol" in group_columns
    assert "analysis_family" in group_columns
    assert "modality_combination" in group_columns
    assert "base_feature_set" in group_columns
    assert "ablation_name" in group_columns
    assert "removed_features" in group_columns
    assert "feature_strategy" in group_columns
    assert "nested_model" in group_columns
