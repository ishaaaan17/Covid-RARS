#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.compare_is10_final_validation import (
    COMPARE_EXTERNAL_PROTOCOL,
    run_compare_is10_external_transfer,
    run_compare_is10_final_validation,
    write_final_validation_summary_figure,
    write_temporal_degradation_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final ComParE+IS10 validation: internal, time-stratified, temporal, and optional external transfer."
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_compare_is10_top800.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--external-features", type=Path, default=None)
    parser.add_argument("--feature-strategy", default="compare_is10_top800_lightgbm")
    parser.add_argument("--selected-feature-k", type=int, default=800)
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"])
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["lightgbm_smote_f80", "svc_rbf_f60", "catboost_smote_f80", "xgboost_smote_f80"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--ensemble-top-k", type=int, default=5)
    parser.add_argument("--enable-feature-level-fusion", action="store_true")
    parser.add_argument("--global-stack-top-k", type=int, default=0)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_final_validation_metrics.csv"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_final_validation_predictions.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/tables/compare_is10_final_validation_summary.csv"),
    )
    parser.add_argument(
        "--split-summary-output",
        type=Path,
        default=Path("reports/tables/compare_is10_final_validation_split_summary.csv"),
    )
    parser.add_argument(
        "--modality-coverage-output",
        type=Path,
        default=Path("reports/tables/compare_is10_final_validation_modality_coverage.csv"),
    )
    parser.add_argument(
        "--external-metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_external_transfer_metrics.csv"),
    )
    parser.add_argument(
        "--external-predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_external_transfer_predictions.csv"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/compare_is10_temporal_degradation.svg"),
    )
    parser.add_argument(
        "--summary-figure-output",
        type=Path,
        default=Path("reports/figures/compare_is10_final_validation_summary.svg"),
    )
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {path} ({len(frame)} rows)")


def _external_skip_frame(feature_strategy: str, selected_feature_k: int, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evaluation_protocol": COMPARE_EXTERNAL_PROTOCOL,
                "analysis_family": "compare_is10_external_transfer",
                "model_name": "not_run",
                "modality": "cough",
                "metric_split": "external_test",
                "feature_strategy": feature_strategy,
                "selected_feature_k": float(selected_feature_k),
                "skipped": True,
                "skip_reason": reason,
            }
        ]
    )


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    metadata = pd.read_csv(args.metadata)

    result = run_compare_is10_final_validation(
        features=features,
        metadata=metadata,
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

    if args.external_features is not None:
        target_features = pd.read_csv(args.external_features)
        external_metrics, external_predictions = run_compare_is10_external_transfer(
            source_features=features,
            target_features=target_features,
            feature_strategy=args.feature_strategy,
            selected_feature_k=args.selected_feature_k,
            model_names=args.model_names,
            modality="cough",
            random_state=args.random_state,
        )
    else:
        external_metrics = _external_skip_frame(
            args.feature_strategy,
            args.selected_feature_k,
            "external feature table not supplied",
        )
        external_predictions = pd.DataFrame()

    _write_csv(result.metrics, args.metrics_output)
    _write_csv(result.predictions, args.predictions_output)
    _write_csv(result.final_summary, args.summary_output)
    _write_csv(result.split_summary, args.split_summary_output)
    _write_csv(result.modality_coverage, args.modality_coverage_output)
    _write_csv(external_metrics, args.external_metrics_output)
    _write_csv(external_predictions, args.external_predictions_output)

    write_temporal_degradation_figure(result.final_summary, args.figure_output)
    print(f"Wrote {args.figure_output}")
    write_final_validation_summary_figure(result.final_summary, external_metrics, args.summary_figure_output)
    print(f"Wrote {args.summary_figure_output}")


if __name__ == "__main__":
    main()
