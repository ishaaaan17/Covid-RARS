from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).parents[1] / "scripts" / "24_make_experiment_manifest.py"
    spec = importlib.util.spec_from_file_location("make_manifest_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_artifact_paths_discovers_representation_outputs(tmp_path) -> None:
    module = _load_script_module()
    metrics_dir = tmp_path / "data" / "outputs" / "metrics"
    tables_dir = tmp_path / "reports" / "tables"
    metrics_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    (metrics_dir / "external_model_grid_beats_metrics.csv").write_text("auroc\\n0.5\\n")
    (metrics_dir / "coughvid_internal_panns_cnn14_bootstrap_ci.csv").write_text("metric\\nauroc\\n")
    (metrics_dir / "sota_ssl_metrics_hf_ssl_microsoft_wavlm-base-plus_cough.csv").write_text("auroc\\n0.9\\n")
    (metrics_dir / "sota_ssl_predictions_hf_ssl_microsoft_wavlm-base-plus_cough.csv").write_text("probability\\n0.8\\n")
    (metrics_dir / "sota_fusion_metrics.csv").write_text("auroc\\n0.93\\n")
    (metrics_dir / "sota_swarm_feature_metrics.csv").write_text("auroc\\n0.91\\n")
    (metrics_dir / "sota_swarm_feature_predictions.csv").write_text("probability\\n0.8\\n")
    (metrics_dir / "sota_gated_stack_metrics.csv").write_text("auroc\\n0.92\\n")
    (metrics_dir / "sota_gated_stack_predictions.csv").write_text("probability\\n0.8\\n")
    (tables_dir / "sota_swarm_feature_selection.csv").write_text("selected_features\\nsignal\\n")
    (tables_dir / "sota_gated_stack_candidates.csv").write_text("selected\\ntrue\\n")
    (tables_dir / "feature_shift_beats_cough.csv").write_text("feature\\nsome_dim\\n")

    artifacts = module.default_artifact_paths(project_root=tmp_path)

    assert tmp_path / "reports" / "tables" / "calibration_under_shift_bins.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "calibration_under_shift_summary.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "clinical_operating_points.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "final_validation_shuffle_label_sanity.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "metadata_confounding_shuffle_retrain_sanity.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "coughvid_partial_recalibration_metrics.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "reviewer_nested_metadata_audio_comparison.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "reviewer_support_overlap_positivity.csv" in artifacts
    assert tmp_path / "reports" / "figures" / "reviewer_decision_curve_analysis.svg" in artifacts
    assert tmp_path / "reports" / "final" / "LABEL_CONSTRUCTION_AUDIT.md" in artifacts
    assert tmp_path / "reports" / "final" / "MULTIPLICITY_AND_ANALYSIS_SCOPE.md" in artifacts
    assert tmp_path / "data" / "outputs" / "metrics" / "metadata_permutation_importance_metrics.csv" in artifacts
    assert tmp_path / "data" / "outputs" / "metrics" / "domain_shift_audit_metrics.csv" in artifacts
    assert tmp_path / "data" / "outputs" / "metrics" / "domain_adaptation_baseline_metrics.csv" in artifacts
    assert tmp_path / "data" / "outputs" / "metrics" / "domain_adaptation_baseline_predictions.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "domain_adaptation_mmd.csv" in artifacts
    assert tmp_path / "data" / "outputs" / "metrics" / "domain_shift_audit_predictions.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "domain_shift_feature_importance.csv" in artifacts
    assert tmp_path / "data" / "outputs" / "metrics" / "ipw_sensitivity_metrics.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "ipw_sensitivity_balance.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "external_prevalence_recalibration.csv" in artifacts
    assert tmp_path / "data" / "outputs" / "metrics" / "external_prevalence_recalibrated_predictions.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "paired_bootstrap_comparisons.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "publication_evidence_matrix.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "publication_evidence_matrix.md" in artifacts
    assert tmp_path / "reports" / "tables" / "related_paper_comparison.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "related_paper_comparison.md" in artifacts
    assert (
        tmp_path / "reports" / "final" / "COVID_RARS_PUBLICATION_RESULTS_REPORT.md"
        in artifacts
    )
    assert (
        tmp_path / "reports" / "final" / "COVID_RARS_PUBLICATION_RESULTS_SUMMARY.md"
        in artifacts
    )
    assert tmp_path / "reports" / "tables" / "manuscript_demographic_protocol_linear_shap.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "manuscript_ipw_residual_smd.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "manuscript_external_auprc_lift.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "manuscript_unknown_label_summary.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "manuscript_unknown_label_balance.csv" in artifacts
    assert tmp_path / "reports" / "final" / "MANUSCRIPT_SUPPORT_ANALYSES.md" in artifacts
    assert (
        tmp_path / "reports" / "final" / "COVID_RARS_PHASED_RESULTS_BRIEF_2026-06-15.md"
        in artifacts
    )
    assert tmp_path / "data" / "outputs" / "metrics" / "temporal_holdout_metrics.csv" in artifacts
    assert tmp_path / "data" / "outputs" / "metrics" / "temporal_holdout_predictions.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_holdout_split_summary.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_holdout_modality_coverage.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_holdout_metadata_feature_importance.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_holdout_metadata_group_summary.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_metadata_ablation.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_stability_summary.csv" in artifacts
    assert tmp_path / "data" / "outputs" / "metrics" / "temporal_holdout_bootstrap_ci.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_external_unification.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_stress_test_summary.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_metadata_feature_attribution_comparison.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_stress_test_significance.csv" in artifacts
    assert tmp_path / "reports" / "final" / "TEMPORAL_ROBUSTNESS_CAUSAL_CHAIN.md" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_month_year_ablation_paper_table.csv" in artifacts
    assert tmp_path / "reports" / "figures" / "temporal_stress_test_figure.svg" in artifacts
    assert tmp_path / "reports" / "final" / "TEMPORAL_RESULTS_SECTION_DRAFT.md" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_month_label_shift.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_month_covariate_shift.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_matched_cohort_metrics.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_failure_modes_by_shift.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_uncertainty_under_shift.csv" in artifacts
    assert tmp_path / "reports" / "tables" / "temporal_month_ablation_effect_sizes.csv" in artifacts
    assert tmp_path / "reports" / "final" / "TEMPORAL_MONTH_CAUSAL_DAG.md" in artifacts
    assert tmp_path / "reports" / "final" / "TEMPORAL_SHORTCUT_THEORY.md" in artifacts
    assert metrics_dir / "external_model_grid_beats_metrics.csv" in artifacts
    assert metrics_dir / "coughvid_internal_panns_cnn14_bootstrap_ci.csv" in artifacts
    assert metrics_dir / "sota_ssl_metrics_hf_ssl_microsoft_wavlm-base-plus_cough.csv" in artifacts
    assert metrics_dir / "sota_ssl_predictions_hf_ssl_microsoft_wavlm-base-plus_cough.csv" in artifacts
    assert metrics_dir / "sota_fusion_metrics.csv" in artifacts
    assert metrics_dir / "sota_swarm_feature_metrics.csv" in artifacts
    assert metrics_dir / "sota_swarm_feature_predictions.csv" in artifacts
    assert metrics_dir / "sota_gated_stack_metrics.csv" in artifacts
    assert metrics_dir / "sota_gated_stack_predictions.csv" in artifacts
    assert tables_dir / "sota_swarm_feature_selection.csv" in artifacts
    assert tables_dir / "sota_gated_stack_candidates.csv" in artifacts
    assert tables_dir / "feature_shift_beats_cough.csv" in artifacts

    (metrics_dir / "compare_is10_reverse_temporal_metrics.csv").write_text("auroc\n0.7\n")
    (tables_dir / "metadata_confounding_permutation_importance.csv").write_text("feature\nage\n")
    artifacts = module.default_artifact_paths(project_root=tmp_path)
    assert metrics_dir / "compare_is10_reverse_temporal_metrics.csv" in artifacts
    assert tables_dir / "metadata_confounding_permutation_importance.csv" in artifacts
