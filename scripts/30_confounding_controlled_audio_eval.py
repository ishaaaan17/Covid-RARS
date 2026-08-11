#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.confounding_controlled_eval import (
    DEFAULT_CONFOUNDERS,
    evaluate_confounding_controlled_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate audio predictions before and after inverse-probability confounder weighting."
    )
    parser.add_argument("--predictions", type=Path, default=Path("data/outputs/metrics/quality_weighted_fusion_predictions.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_clean.csv"))
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/confounding_controlled_audio_metrics.csv"),
    )
    parser.add_argument(
        "--balance-output",
        type=Path,
        default=Path("reports/tables/confounding_controlled_balance.csv"),
    )
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=Path("data/outputs/metrics/confounding_controlled_audio_weights.csv"),
    )
    parser.add_argument("--covariates", nargs="+", default=DEFAULT_CONFOUNDERS)
    parser.add_argument("--group-columns", nargs="*", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _default_group_columns(predictions: pd.DataFrame) -> list[str]:
    candidates = ["fusion_method", "model_name", "modality", "feature_strategy", "dataset"]
    return [col for col in candidates if col in predictions.columns]


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    metadata = pd.read_csv(args.metadata)
    group_columns = args.group_columns if args.group_columns is not None else _default_group_columns(predictions)
    result = evaluate_confounding_controlled_predictions(
        predictions,
        metadata,
        covariates=args.covariates,
        group_columns=group_columns,
        split=args.split,
        threshold=args.threshold,
        random_state=args.random_state,
    )
    for path in [args.metrics_output, args.balance_output, args.weights_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(args.metrics_output, index=False)
    result.balance.to_csv(args.balance_output, index=False)
    result.weights.to_csv(args.weights_output, index=False)
    print(f"Wrote confounding-controlled audio metrics: {args.metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote confounder balance diagnostics: {args.balance_output} ({len(result.balance)} rows)")
    print(f"Wrote IPW weights: {args.weights_output} ({len(result.weights)} rows)")
    if not result.metrics.empty:
        display_cols = [
            *group_columns,
            "control_method",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "f1",
            "effective_sample_size",
            "n_samples",
        ]
        print(result.metrics[[col for col in display_cols if col in result.metrics.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
