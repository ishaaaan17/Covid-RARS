#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.reporting import build_paper_metric_table, read_existing_csvs


DEFAULT_METRIC_PATHS = [
    Path("data/outputs/metrics/ml_baseline_metrics.csv"),
    Path("data/outputs/metrics/cnn_metrics.csv"),
    Path("data/outputs/metrics/calibration_metrics.csv"),
    Path("data/outputs/metrics/fusion_metrics.csv"),
    Path("data/outputs/metrics/quality_weighted_fusion_metrics.csv"),
    Path("data/outputs/metrics/metadata_baseline_metrics.csv"),
    Path("data/outputs/metrics/cross_dataset_metrics.csv"),
    Path("data/outputs/metrics/external_model_grid_metrics.csv"),
    Path("data/outputs/metrics/coughvid_internal_metrics.csv"),
    Path("data/outputs/metrics/metadata_confounding_metrics.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_metrics.csv"),
    Path("reports/tables/calibration_under_shift_summary.csv"),
    Path("reports/tables/final_validation_calibration_summary.csv"),
    Path("reports/tables/final_validation_fixed_sensitivity_operating_points.csv"),
    Path("reports/tables/coughvid_partial_recalibration_metrics.csv"),
    Path("reports/tables/reviewer_nested_metadata_audio_comparison.csv"),
    Path("reports/tables/reviewer_context_control_exposure.csv"),
    Path("data/outputs/metrics/reviewer_incremental_audio_metadata_metrics.csv"),
    Path("data/outputs/metrics/metadata_permutation_importance_metrics.csv"),
    Path("data/outputs/metrics/domain_shift_audit_metrics.csv"),
    Path("data/outputs/metrics/domain_adaptation_baseline_metrics.csv"),
    Path("data/outputs/metrics/ipw_sensitivity_metrics.csv"),
    Path("reports/tables/external_prevalence_recalibration.csv"),
    Path("data/outputs/metrics/temporal_holdout_metrics.csv"),
    Path("reports/tables/temporal_metadata_ablation.csv"),
    Path("reports/tables/temporal_matched_cohort_metrics.csv"),
    Path("data/outputs/metrics/strong_baseline_metrics.csv"),
    Path("data/outputs/metrics/sota_fusion_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_reverse_temporal_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_multiseed_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_shuffle_retrain_metrics.csv"),
    Path("data/outputs/metrics/deep_external_wavlm_cough_metrics.csv"),
    Path("data/outputs/metrics/deep_external_cnn_bigru_cough_metrics.csv"),
]

DEFAULT_CI_PATHS = [
    Path("data/outputs/metrics/quality_weighted_fusion_bootstrap_ci.csv"),
    Path("data/outputs/metrics/cross_dataset_bootstrap_ci.csv"),
    Path("data/outputs/metrics/external_model_grid_bootstrap_ci.csv"),
    Path("data/outputs/metrics/coughvid_internal_bootstrap_ci.csv"),
    Path("data/outputs/metrics/confounding_controlled_audio_bootstrap_ci.csv"),
    Path("data/outputs/metrics/temporal_holdout_bootstrap_ci.csv"),
    Path("reports/tables/final_validation_bootstrap_ci.csv"),
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


def default_metric_paths(metrics_dir: Path = Path("data/outputs/metrics")) -> list[Path]:
    representation_patterns = [
        "external_model_grid_*_metrics.csv",
        "coughvid_internal_*_metrics.csv",
        "cnn_metrics_*.csv",
        "sota_ssl_metrics_*.csv",
        "sota_spectrogram_metrics_*.csv",
        "sota_foundation_metrics_*.csv",
        "sota_swarm_feature_metrics*.csv",
        "sota_gated_stack_metrics*.csv",
        "sota_fusion_metrics*.csv",
        "sota_compare_is10*_metrics.csv",
        "paper_comparable_cv*_metrics.csv",
        "compare_is10_final_validation*_metrics.csv",
        "compare_is10_external_transfer*_metrics.csv",
        "compare_is10_reverse_temporal*_metrics.csv",
        "compare_is10_multiseed*_metrics.csv",
        "compare_is10_shuffle_retrain*_metrics.csv",
        "metadata_permutation_importance*_metrics.csv",
        "reviewer_*_metrics.csv",
        "deep_external_*_metrics.csv",
        "sota_compare_is10*_metrics.csv",
        "paper_comparable_cv*_metrics.csv",
        "compare_is10_final_validation*_metrics.csv",
        "compare_is10_external_transfer*_metrics.csv",
    ]
    discovered: list[Path] = []
    for pattern in representation_patterns:
        discovered.extend(sorted(metrics_dir.glob(pattern)))
    return _dedupe_paths([*DEFAULT_METRIC_PATHS, *discovered])


def default_ci_paths(metrics_dir: Path = Path("data/outputs/metrics")) -> list[Path]:
    representation_patterns = [
        "external_model_grid_*_bootstrap_ci.csv",
        "coughvid_internal_*_bootstrap_ci.csv",
        "sota_*_bootstrap_ci.csv",
    ]
    discovered: list[Path] = []
    for pattern in representation_patterns:
        discovered.extend(sorted(metrics_dir.glob(pattern)))
    return _dedupe_paths([*DEFAULT_CI_PATHS, *discovered])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready metric tables from experiment CSVs.")
    parser.add_argument("--metrics", nargs="*", type=Path, default=None)
    parser.add_argument("--ci", nargs="*", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/tables/paper_metric_table.csv"))
    parser.add_argument("--raw-output", type=Path, default=Path("reports/tables/paper_metric_table_raw.csv"))
    return parser.parse_args()


def _group_columns(metrics: pd.DataFrame) -> list[str]:
    candidates = [
        "table_source",
        "evaluation_protocol",
        "analysis_family",
        "model_name",
        "model",
        "modality",
        "metric_split",
        "modality_combination",
        "feature_strategy",
        "base_feature_set",
        "ablation_name",
        "removed_features",
        "fusion_method",
        "dataset",
        "split",
        "audit_model",
        "control_method",
        "prediction_source",
        "calibration_method",
        "analysis",
        "nested_model",
        "context_column",
        "subgroup_column",
        "subgroup_value",
        "feature_strategy",
    ]
    return [col for col in candidates if col in metrics.columns]


def main() -> None:
    args = parse_args()
    metric_paths = args.metrics if args.metrics is not None else default_metric_paths()
    ci_paths = args.ci if args.ci is not None else default_ci_paths()
    metrics = read_existing_csvs(metric_paths)
    ci_table = read_existing_csvs(ci_paths)
    if metrics.empty:
        checked = ", ".join(str(path) for path in metric_paths)
        raise FileNotFoundError(f"No metric CSVs found. Checked: {checked}")
    group_columns = _group_columns(metrics)
    paper = build_paper_metric_table(metrics, ci_table=ci_table, group_columns=group_columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.raw_output, index=False)
    paper.to_csv(args.output, index=False)
    print(f"Wrote paper metric table: {args.output} ({len(paper)} rows)")
    print(f"Wrote raw combined metric table: {args.raw_output} ({len(metrics)} rows)")


if __name__ == "__main__":
    main()
