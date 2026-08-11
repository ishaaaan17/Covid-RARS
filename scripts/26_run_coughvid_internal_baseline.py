#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.rescue_experiments import DEFAULT_RESCUE_MODELS, evaluate_internal_splits, make_stratified_external_splits
from covid_rars.statistics import bootstrap_prediction_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run COUGHVID-only internal baselines to diagnose whether COUGHVID labels are learnable."
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/coughvid_features_mfcc.csv"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_RESCUE_MODELS)
    parser.add_argument("--modality", default="cough")
    parser.add_argument("--train-size", type=float, default=0.60)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-features-output", type=Path, default=Path("data/processed/coughvid_features_mfcc_internal_split.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("data/outputs/metrics/coughvid_internal_predictions.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/coughvid_internal_metrics.csv"))
    parser.add_argument("--bootstrap-output", type=Path, default=Path("data/outputs/metrics/coughvid_internal_bootstrap_ci.csv"))
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--strict-models", action="store_true", help="Fail instead of skipping missing optional model packages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    features = features[(features["modality"] == args.modality) & features["label_binary"].isin(["positive", "negative"])].copy()
    split_features = make_stratified_external_splits(
        features,
        train_size=args.train_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for model_name in args.models:
        try:
            result = evaluate_internal_splits(
                split_features,
                model_name=model_name,
                modality=args.modality,
                random_state=args.random_state,
            )
        except RuntimeError as exc:
            if args.strict_models:
                raise
            print(f"SKIP {model_name}: {exc}")
            continue
        prediction_frames.append(result.predictions)
        metric_rows.append(result.metrics)
        print(
            f"Ran COUGHVID internal {model_name}: "
            f"AUROC={result.metrics['auroc']:.4f}, AUPRC={result.metrics['auprc']:.4f}, "
            f"threshold={result.metrics['threshold']:.4f}"
        )

    if not prediction_frames:
        raise RuntimeError("No COUGHVID internal predictions were produced. Check installed optional model packages.")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    bootstrap = bootstrap_prediction_table(
        predictions,
        group_columns=["model_name"],
        n_bootstraps=args.n_bootstraps,
        random_state=args.random_state,
    )

    args.split_features_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.bootstrap_output.parent.mkdir(parents=True, exist_ok=True)
    split_features.to_csv(args.split_features_output, index=False)
    predictions.to_csv(args.predictions_output, index=False)
    metrics.to_csv(args.metrics_output, index=False)
    bootstrap.to_csv(args.bootstrap_output, index=False)
    print(f"Wrote COUGHVID split features: {args.split_features_output} ({len(split_features)} rows)")
    print(f"Wrote COUGHVID internal predictions: {args.predictions_output} ({len(predictions)} rows)")
    print(f"Wrote COUGHVID internal metrics: {args.metrics_output} ({len(metrics)} rows)")
    print(f"Wrote COUGHVID internal bootstrap CIs: {args.bootstrap_output} ({len(bootstrap)} rows)")


if __name__ == "__main__":
    main()
