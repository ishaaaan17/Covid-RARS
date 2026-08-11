#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.rescue_experiments import (
    DEFAULT_FEATURE_STRATEGIES,
    DEFAULT_RESCUE_MODELS,
    evaluate_source_to_external,
)
from covid_rars.statistics import bootstrap_prediction_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Coswara-to-COUGHVID external transfer models with boosted trees and stable-feature strategies."
    )
    parser.add_argument("--source-features", type=Path, default=Path("data/processed/features_mfcc.csv"))
    parser.add_argument("--external-features", type=Path, default=Path("data/processed/coughvid_features_mfcc.csv"))
    parser.add_argument("--feature-shift-report", type=Path, default=Path("reports/tables/feature_shift_report.csv"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_RESCUE_MODELS)
    parser.add_argument("--feature-strategies", nargs="+", default=DEFAULT_FEATURE_STRATEGIES)
    parser.add_argument("--modality", default="cough")
    parser.add_argument("--source-train-split", default="train")
    parser.add_argument("--smd-threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--predictions-output", type=Path, default=Path("data/outputs/metrics/external_model_grid_predictions.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/external_model_grid_metrics.csv"))
    parser.add_argument("--bootstrap-output", type=Path, default=Path("data/outputs/metrics/external_model_grid_bootstrap_ci.csv"))
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--strict-models", action="store_true", help="Fail instead of skipping missing optional model packages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.source_features)
    external = pd.read_csv(args.external_features)
    shift_report = pd.read_csv(args.feature_shift_report) if args.feature_shift_report.exists() else pd.DataFrame()

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for model_name in args.models:
        for strategy in args.feature_strategies:
            try:
                result = evaluate_source_to_external(
                    source,
                    external,
                    model_name=model_name,
                    feature_strategy=strategy,
                    shift_report=shift_report,
                    modality=args.modality,
                    source_train_split=args.source_train_split,
                    random_state=args.random_state,
                    smd_threshold=args.smd_threshold,
                )
            except (RuntimeError, ValueError) as exc:
                if args.strict_models:
                    raise
                print(f"SKIP {model_name}/{strategy}: {exc}")
                continue
            prediction_frames.append(result.predictions)
            metric_rows.append(result.metrics)
            print(
                f"Ran {model_name}/{strategy}: "
                f"AUROC={result.metrics['auroc']:.4f}, AUPRC={result.metrics['auprc']:.4f}, "
                f"features={result.feature_count}"
            )

    if not prediction_frames:
        raise RuntimeError("No external model-grid predictions were produced. Check installed optional model packages.")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    bootstrap = bootstrap_prediction_table(
        predictions,
        group_columns=["model_name", "feature_strategy"],
        n_bootstraps=args.n_bootstraps,
        random_state=args.random_state,
    )

    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.bootstrap_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)
    metrics.to_csv(args.metrics_output, index=False)
    bootstrap.to_csv(args.bootstrap_output, index=False)
    print(f"Wrote external grid predictions: {args.predictions_output} ({len(predictions)} rows)")
    print(f"Wrote external grid metrics: {args.metrics_output} ({len(metrics)} rows)")
    print(f"Wrote external grid bootstrap CIs: {args.bootstrap_output} ({len(bootstrap)} rows)")


if __name__ == "__main__":
    main()
