#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.compare_is10_final_validation import (
    _build_final_summary,
    _modality_coverage,
    _run_strong_protocol,
    run_compare_is10_final_validation,
)
from covid_rars.reviewer_temporal_robustness import (
    REVERSE_TEMPORAL_PROTOCOL,
    build_reverse_temporal_split_assignments,
    summarize_multiseed_metrics,
)
from covid_rars.strong_baseline import DEFAULT_MODALITIES
from covid_rars.temporal_holdout import _apply_split_to_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reviewer robustness additions: late-to-early temporal validation and multi-seed final validation."
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_compare_is10_top800.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--feature-strategy", default="compare_is10_top800_lightgbm")
    parser.add_argument("--selected-feature-k", type=int, default=800)
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["lightgbm_smote_f80", "svc_rbf_f60", "catboost_smote_f80", "xgboost_smote_f80"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--random-states", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--ensemble-top-k", type=int, default=5)
    parser.add_argument("--enable-feature-level-fusion", action="store_true")
    parser.add_argument("--global-stack-top-k", type=int, default=0)
    parser.add_argument("--skip-reverse-temporal", action="store_true")
    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument(
        "--reverse-metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_reverse_temporal_metrics.csv"),
    )
    parser.add_argument(
        "--reverse-predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_reverse_temporal_predictions.csv"),
    )
    parser.add_argument(
        "--reverse-summary-output",
        type=Path,
        default=Path("reports/tables/compare_is10_reverse_temporal_summary.csv"),
    )
    parser.add_argument(
        "--reverse-split-summary-output",
        type=Path,
        default=Path("reports/tables/compare_is10_reverse_temporal_split_summary.csv"),
    )
    parser.add_argument(
        "--reverse-modality-coverage-output",
        type=Path,
        default=Path("reports/tables/compare_is10_reverse_temporal_modality_coverage.csv"),
    )
    parser.add_argument(
        "--multiseed-metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_multiseed_metrics.csv"),
    )
    parser.add_argument(
        "--multiseed-final-summary-output",
        type=Path,
        default=Path("reports/tables/compare_is10_multiseed_final_summary.csv"),
    )
    parser.add_argument(
        "--multiseed-stability-output",
        type=Path,
        default=Path("reports/tables/compare_is10_multiseed_stability_summary.csv"),
    )
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {label}: {path} ({len(frame)} rows)")


def _run_reverse_temporal(args: argparse.Namespace, features: pd.DataFrame, metadata: pd.DataFrame) -> None:
    assignments, split_summary = build_reverse_temporal_split_assignments(metadata)
    reverse_features = _apply_split_to_features(features, assignments, "reverse_temporal_split")
    metrics, predictions = _run_strong_protocol(
        reverse_features,
        protocol=REVERSE_TEMPORAL_PROTOCOL,
        feature_strategy=args.feature_strategy,
        selected_feature_k=args.selected_feature_k,
        modalities=args.modalities,
        model_names=args.model_names,
        random_state=args.random_state,
        optuna_trials=args.optuna_trials,
        ensemble_top_k=args.ensemble_top_k,
        enable_feature_level_fusion=args.enable_feature_level_fusion,
        global_stack_top_k=args.global_stack_top_k,
    )
    summary = _build_final_summary(metrics, feature_strategy=args.feature_strategy, selected_feature_k=args.selected_feature_k)
    coverage = _modality_coverage(reverse_features, REVERSE_TEMPORAL_PROTOCOL)
    _write_csv(metrics, args.reverse_metrics_output, "reverse temporal metrics")
    _write_csv(predictions, args.reverse_predictions_output, "reverse temporal predictions")
    _write_csv(summary, args.reverse_summary_output, "reverse temporal summary")
    _write_csv(split_summary, args.reverse_split_summary_output, "reverse temporal split summary")
    _write_csv(coverage, args.reverse_modality_coverage_output, "reverse temporal modality coverage")


def _run_multiseed(args: argparse.Namespace, features: pd.DataFrame, metadata: pd.DataFrame) -> None:
    metric_frames: list[pd.DataFrame] = []
    final_summary_frames: list[pd.DataFrame] = []
    for seed in args.random_states:
        result = run_compare_is10_final_validation(
            features=features,
            metadata=metadata,
            feature_strategy=args.feature_strategy,
            selected_feature_k=args.selected_feature_k,
            modalities=args.modalities,
            model_names=args.model_names,
            random_state=int(seed),
            optuna_trials=args.optuna_trials,
            ensemble_top_k=args.ensemble_top_k,
            enable_feature_level_fusion=args.enable_feature_level_fusion,
            global_stack_top_k=args.global_stack_top_k,
        )
        metrics = result.metrics.copy()
        metrics["random_state"] = int(seed)
        final_summary = result.final_summary.copy()
        final_summary["random_state"] = int(seed)
        metric_frames.append(metrics)
        final_summary_frames.append(final_summary)
    all_metrics = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    all_final = pd.concat(final_summary_frames, ignore_index=True, sort=False) if final_summary_frames else pd.DataFrame()
    stability = summarize_multiseed_metrics(all_final)
    _write_csv(all_metrics, args.multiseed_metrics_output, "multi-seed metrics")
    _write_csv(all_final, args.multiseed_final_summary_output, "multi-seed final summary")
    _write_csv(stability, args.multiseed_stability_output, "multi-seed stability summary")


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    metadata = pd.read_csv(args.metadata)
    if not args.skip_reverse_temporal:
        _run_reverse_temporal(args, features, metadata)
    if not args.skip_multiseed:
        _run_multiseed(args, features, metadata)


if __name__ == "__main__":
    main()
