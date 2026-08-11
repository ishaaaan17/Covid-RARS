#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from covid_rars.manifest import build_experiment_manifest


DEFAULT_ARTIFACTS = [
    Path("data/interim/coswara_index.csv"),
    Path("data/processed/metadata_clean.csv"),
    Path("data/processed/audio_quality.csv"),
    Path("data/processed/features_mfcc.csv"),
    Path("data/processed/features_strong_acoustic.csv"),
    Path("data/outputs/metrics/ml_baseline_metrics.csv"),
    Path("data/outputs/metrics/calibration_metrics.csv"),
    Path("data/outputs/metrics/fusion_metrics.csv"),
    Path("data/outputs/metrics/quality_weighted_fusion_metrics.csv"),
    Path("data/outputs/metrics/cross_dataset_metrics.csv"),
    Path("data/outputs/metrics/external_model_grid_metrics.csv"),
    Path("data/outputs/metrics/external_model_grid_bootstrap_ci.csv"),
    Path("data/outputs/metrics/coughvid_internal_metrics.csv"),
    Path("data/outputs/metrics/coughvid_internal_bootstrap_ci.csv"),
    Path("data/outputs/metrics/metadata_confounding_metrics.csv"),
    Path("data/outputs/metrics/metadata_confounding_predictions.csv"),
    Path("reports/tables/metadata_confounding_feature_importance.csv"),
    Path("reports/tables/metadata_confounding_group_summary.csv"),
    Path("reports/tables/metadata_confounding_subgroup_availability.csv"),
    Path("reports/tables/metadata_confounding_subgroup_breakdown.csv"),
    Path("reports/tables/metadata_confounding_subgroup_metrics.csv"),
    Path("reports/tables/metadata_confounding_shuffle_label_sanity.csv"),
    Path("reports/tables/metadata_confounding_shuffle_retrain_sanity.csv"),
    Path("data/outputs/metrics/metadata_permutation_importance_metrics.csv"),
    Path("data/outputs/metrics/metadata_permutation_importance_predictions.csv"),
    Path("reports/tables/metadata_confounding_permutation_importance.csv"),
    Path("reports/tables/metadata_confounding_permutation_group_summary.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_metrics.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_weights.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_bootstrap_ci.csv"),
    Path("reports/tables/confounding_controlled_balance.csv"),
    Path("reports/tables/calibration_under_shift_bins.csv"),
    Path("reports/tables/calibration_under_shift_summary.csv"),
    Path("reports/tables/final_validation_bootstrap_ci.csv"),
    Path("reports/tables/final_validation_delta_bootstrap_ci.csv"),
    Path("reports/tables/final_validation_calibration_summary.csv"),
    Path("reports/tables/final_validation_calibration_bins.csv"),
    Path("reports/tables/final_validation_shuffle_label_sanity.csv"),
    Path("reports/tables/final_validation_fixed_sensitivity_operating_points.csv"),
    Path("reports/tables/audio_metadata_residual_correlation.csv"),
    Path("reports/tables/coughvid_partial_recalibration_metrics.csv"),
    Path("data/outputs/metrics/coughvid_partial_recalibration_predictions.csv"),
    Path("reports/tables/reviewer_specification_curve.csv"),
    Path("reports/tables/reviewer_quality_label_month_correlation.csv"),
    Path("reports/tables/reviewer_decision_curve_analysis.csv"),
    Path("reports/tables/reviewer_duration_shortcut_correlation.csv"),
    Path("reports/tables/reviewer_performance_equity_subgroups.csv"),
    Path("reports/tables/reviewer_nested_metadata_audio_comparison.csv"),
    Path("data/outputs/metrics/reviewer_nested_metadata_audio_predictions.csv"),
    Path("data/outputs/metrics/reviewer_incremental_audio_metadata_metrics.csv"),
    Path("data/outputs/metrics/reviewer_incremental_audio_metadata_predictions.csv"),
    Path("reports/tables/reviewer_incremental_audio_metadata_candidates.csv"),
    Path("reports/tables/reviewer_external_model_family_transfer_summary.csv"),
    Path("reports/tables/reviewer_support_overlap_positivity.csv"),
    Path("reports/tables/reviewer_feature_selection_stability.csv"),
    Path("reports/tables/reviewer_matched_cohort_quality.csv"),
    Path("reports/tables/reviewer_context_control_exposure.csv"),
    Path("reports/tables/reviewer_paired_delong_auc_comparisons.csv"),
    Path("reports/tables/reviewer_label_construction_audit.csv"),
    Path("reports/figures/reviewer_specification_curve.svg"),
    Path("reports/figures/reviewer_decision_curve_analysis.svg"),
    Path("reports/final/LABEL_CONSTRUCTION_AUDIT.md"),
    Path("reports/final/MULTIPLICITY_AND_ANALYSIS_SCOPE.md"),
    Path("reports/figures/final_validation_calibration_curves.svg"),
    Path("reports/tables/clinical_operating_points.csv"),
    Path("data/outputs/metrics/domain_shift_audit_metrics.csv"),
    Path("data/outputs/metrics/domain_shift_audit_predictions.csv"),
    Path("reports/tables/domain_shift_feature_importance.csv"),
    Path("data/outputs/metrics/domain_adaptation_baseline_metrics.csv"),
    Path("data/outputs/metrics/domain_adaptation_baseline_predictions.csv"),
    Path("reports/tables/domain_adaptation_mmd.csv"),
    Path("data/outputs/metrics/ipw_sensitivity_metrics.csv"),
    Path("data/outputs/metrics/ipw_sensitivity_weights.csv"),
    Path("reports/tables/ipw_sensitivity_balance.csv"),
    Path("reports/tables/external_prevalence_recalibration.csv"),
    Path("data/outputs/metrics/external_prevalence_recalibrated_predictions.csv"),
    Path("reports/tables/paired_bootstrap_comparisons.csv"),
    Path("reports/tables/publication_evidence_matrix.csv"),
    Path("reports/tables/publication_evidence_matrix.md"),
    Path("reports/tables/related_paper_comparison.csv"),
    Path("reports/tables/related_paper_comparison.md"),
    Path("reports/final/COVID_RARS_PUBLICATION_RESULTS_REPORT.md"),
    Path("reports/final/COVID_RARS_PUBLICATION_RESULTS_SUMMARY.md"),
    Path("reports/tables/manuscript_demographic_protocol_linear_shap.csv"),
    Path("reports/tables/manuscript_ipw_residual_smd.csv"),
    Path("reports/tables/manuscript_external_auprc_lift.csv"),
    Path("reports/tables/manuscript_unknown_label_summary.csv"),
    Path("reports/tables/manuscript_unknown_label_balance.csv"),
    Path("reports/final/MANUSCRIPT_SUPPORT_ANALYSES.md"),
    Path("reports/final/COVID_RARS_PHASED_RESULTS_BRIEF_2026-06-15.md"),
    Path("data/outputs/metrics/temporal_holdout_metrics.csv"),
    Path("data/outputs/metrics/temporal_holdout_predictions.csv"),
    Path("reports/tables/temporal_holdout_split_summary.csv"),
    Path("reports/tables/temporal_holdout_modality_coverage.csv"),
    Path("reports/tables/temporal_holdout_metadata_feature_importance.csv"),
    Path("reports/tables/temporal_holdout_metadata_group_summary.csv"),
    Path("reports/tables/temporal_metadata_ablation.csv"),
    Path("reports/tables/temporal_stability_summary.csv"),
    Path("data/outputs/metrics/temporal_holdout_bootstrap_ci.csv"),
    Path("reports/tables/temporal_external_unification.csv"),
    Path("reports/tables/temporal_stress_test_summary.csv"),
    Path("reports/tables/temporal_metadata_feature_attribution_comparison.csv"),
    Path("reports/tables/temporal_stress_test_significance.csv"),
    Path("reports/final/TEMPORAL_ROBUSTNESS_CAUSAL_CHAIN.md"),
    Path("reports/tables/temporal_month_year_ablation_paper_table.csv"),
    Path("reports/figures/temporal_stress_test_figure.svg"),
    Path("reports/final/TEMPORAL_RESULTS_SECTION_DRAFT.md"),
    Path("reports/tables/temporal_month_label_shift.csv"),
    Path("reports/tables/temporal_month_covariate_shift.csv"),
    Path("reports/tables/temporal_matched_cohort_metrics.csv"),
    Path("reports/tables/temporal_failure_modes_by_shift.csv"),
    Path("reports/tables/temporal_uncertainty_under_shift.csv"),
    Path("reports/tables/temporal_month_ablation_effect_sizes.csv"),
    Path("reports/final/TEMPORAL_MONTH_CAUSAL_DAG.md"),
    Path("reports/final/TEMPORAL_SHORTCUT_THEORY.md"),
    Path("data/outputs/metrics/strong_baseline_metrics.csv"),
    Path("data/outputs/metrics/strong_baseline_predictions.csv"),
    Path("reports/tables/strong_baseline_model_selection.csv"),
    Path("reports/tables/strong_baseline_protocol_audit.csv"),
    Path("reports/tables/strong_baseline_participant_audit.csv"),
    Path("data/processed/sota_segment_index.csv"),
    Path("reports/tables/sota_segment_index_audit.csv"),
    Path("data/outputs/metrics/sota_fusion_metrics.csv"),
    Path("data/outputs/metrics/sota_fusion_predictions.csv"),
    Path("data/outputs/metrics/sota_swarm_feature_metrics.csv"),
    Path("data/outputs/metrics/sota_swarm_feature_predictions.csv"),
    Path("reports/tables/sota_swarm_feature_selection.csv"),
    Path("data/outputs/metrics/sota_gated_stack_metrics.csv"),
    Path("data/outputs/metrics/sota_gated_stack_predictions.csv"),
    Path("reports/tables/sota_gated_stack_candidates.csv"),
    Path("data/outputs/metrics/compare_is10_final_validation_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_final_validation_predictions.csv"),
    Path("data/outputs/metrics/compare_is10_external_transfer_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_external_transfer_predictions.csv"),
    Path("data/outputs/metrics/compare_is10_shuffle_retrain_metrics.csv"),
    Path("reports/tables/compare_is10_shuffle_retrain_summary.csv"),
    Path("data/outputs/metrics/compare_is10_reverse_temporal_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_reverse_temporal_predictions.csv"),
    Path("reports/tables/compare_is10_reverse_temporal_summary.csv"),
    Path("reports/tables/compare_is10_reverse_temporal_split_summary.csv"),
    Path("reports/tables/compare_is10_reverse_temporal_modality_coverage.csv"),
    Path("data/outputs/metrics/compare_is10_multiseed_metrics.csv"),
    Path("reports/tables/compare_is10_multiseed_final_summary.csv"),
    Path("reports/tables/compare_is10_multiseed_stability_summary.csv"),
    Path("data/outputs/metrics/deep_external_wavlm_cough_metrics.csv"),
    Path("data/outputs/metrics/deep_external_wavlm_cough_predictions.csv"),
    Path("data/outputs/metrics/deep_external_wavlm_cough_history.csv"),
    Path("data/outputs/metrics/deep_external_cnn_bigru_cough_metrics.csv"),
    Path("data/outputs/metrics/deep_external_cnn_bigru_cough_validation_predictions.csv"),
    Path("data/outputs/metrics/deep_external_cnn_bigru_cough_test_predictions.csv"),
    Path("data/outputs/metrics/deep_external_cnn_bigru_cough_external_predictions.csv"),
    Path("data/outputs/metrics/deep_external_cnn_bigru_cough_history.csv"),
    Path("reports/tables/compare_is10_final_validation_summary.csv"),
    Path("reports/tables/compare_is10_final_validation_split_summary.csv"),
    Path("reports/tables/compare_is10_final_validation_modality_coverage.csv"),
    Path("reports/figures/compare_is10_temporal_degradation.svg"),
    Path("reports/figures/compare_is10_final_validation_summary.svg"),
    Path("reports/tables/feature_shift_report.csv"),
    Path("reports/tables/feature_shift_summary.csv"),
    Path("reports/tables/paper_metric_table.csv"),
]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def default_artifact_paths(project_root: Path | None = None) -> list[Path]:
    root = Path(".") if project_root is None else Path(project_root)
    base = [root / path for path in DEFAULT_ARTIFACTS]
    metrics_dir = root / "data" / "outputs" / "metrics"
    tables_dir = root / "reports" / "tables"
    figures_dir = root / "reports" / "figures"
    patterns = [
        (metrics_dir, "external_model_grid_*_metrics.csv"),
        (metrics_dir, "external_model_grid_*_bootstrap_ci.csv"),
        (metrics_dir, "coughvid_internal_*_metrics.csv"),
        (metrics_dir, "coughvid_internal_*_bootstrap_ci.csv"),
        (metrics_dir, "cnn_metrics_*.csv"),
        (metrics_dir, "cnn_logits_validation_*.csv"),
        (metrics_dir, "cnn_logits_test_*.csv"),
        (metrics_dir, "cnn_logits_external_test*.csv"),
        (metrics_dir, "cnn_training_history_*.csv"),
        (metrics_dir, "cnn_metrics_external_test*.csv"),
        (metrics_dir, "sota_ssl_metrics_*.csv"),
        (metrics_dir, "sota_ssl_predictions_*.csv"),
        (metrics_dir, "sota_ssl_history_*.csv"),
        (metrics_dir, "sota_spectrogram_metrics_*.csv"),
        (metrics_dir, "sota_spectrogram_predictions_*.csv"),
        (metrics_dir, "sota_foundation_metrics_*.csv"),
        (metrics_dir, "sota_foundation_predictions_*.csv"),
        (metrics_dir, "sota_swarm_feature_metrics*.csv"),
        (metrics_dir, "sota_swarm_feature_predictions*.csv"),
        (metrics_dir, "sota_compare_is10*_metrics.csv"),
        (metrics_dir, "sota_compare_is10*_predictions.csv"),
        (metrics_dir, "paper_comparable_cv*_metrics.csv"),
        (metrics_dir, "paper_comparable_cv*_predictions.csv"),
        (metrics_dir, "compare_is10_final_validation*_metrics.csv"),
        (metrics_dir, "compare_is10_final_validation*_predictions.csv"),
        (metrics_dir, "compare_is10_external_transfer*_metrics.csv"),
        (metrics_dir, "compare_is10_external_transfer*_predictions.csv"),
        (metrics_dir, "compare_is10_shuffle_retrain*_metrics.csv"),
        (metrics_dir, "compare_is10_shuffle_retrain*_predictions.csv"),
        (metrics_dir, "compare_is10_reverse_temporal*_metrics.csv"),
        (metrics_dir, "compare_is10_reverse_temporal*_predictions.csv"),
        (metrics_dir, "compare_is10_multiseed*_metrics.csv"),
        (metrics_dir, "metadata_permutation_importance*_metrics.csv"),
        (metrics_dir, "metadata_permutation_importance*_predictions.csv"),
        (metrics_dir, "coughvid_partial_recalibration*_predictions.csv"),
        (metrics_dir, "reviewer_*_predictions.csv"),
        (metrics_dir, "deep_external_*_metrics.csv"),
        (metrics_dir, "deep_external_*_predictions.csv"),
        (metrics_dir, "deep_external_*_history.csv"),
        (metrics_dir, "sota_compare_is10*_metrics.csv"),
        (metrics_dir, "sota_compare_is10*_predictions.csv"),
        (metrics_dir, "paper_comparable_cv*_metrics.csv"),
        (metrics_dir, "paper_comparable_cv*_predictions.csv"),
        (metrics_dir, "sota_gated_stack_metrics*.csv"),
        (metrics_dir, "sota_gated_stack_predictions*.csv"),
        (metrics_dir, "sota_fusion_metrics*.csv"),
        (metrics_dir, "sota_fusion_predictions*.csv"),
        (tables_dir, "sota_swarm_feature_selection*.csv"),
        (tables_dir, "sota_gated_stack_candidates*.csv"),
        (tables_dir, "compare_is10_final_validation*.csv"),
        (tables_dir, "compare_is10_reverse_temporal*.csv"),
        (tables_dir, "compare_is10_multiseed*.csv"),
        (tables_dir, "compare_is10_shuffle_retrain*.csv"),
        (tables_dir, "feature_shift_*_cough.csv"),
        (tables_dir, "feature_shift_*_summary.csv"),
        (tables_dir, "final_validation_*.csv"),
        (tables_dir, "metadata_confounding_subgroup*.csv"),
        (tables_dir, "metadata_confounding_shuffle*.csv"),
        (tables_dir, "metadata_confounding_permutation*.csv"),
        (tables_dir, "audio_metadata_residual_correlation.csv"),
        (tables_dir, "coughvid_partial_recalibration_metrics.csv"),
        (tables_dir, "reviewer_*.csv"),
        (figures_dir, "compare_is10_*.svg"),
        (figures_dir, "final_validation_*.svg"),
        (figures_dir, "reviewer_*.svg"),
    ]
    discovered: list[Path] = []
    for directory, pattern in patterns:
        discovered.extend(sorted(directory.glob(pattern)))
    return _dedupe_paths([*base, *discovered])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write reproducibility manifest with config, package versions, and artifact hashes.")
    parser.add_argument("--output", type=Path, default=Path("reports/experiment_manifest.json"))
    parser.add_argument("--artifacts", nargs="*", type=Path, default=None)
    parser.add_argument("--run-name", default="covid_audio_publication_run")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts if args.artifacts is not None else default_artifact_paths()
    manifest = build_experiment_manifest(
        config={"run_name": args.run_name, "seed": args.seed},
        artifact_paths=artifacts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote experiment manifest: {args.output}")


if __name__ == "__main__":
    main()
