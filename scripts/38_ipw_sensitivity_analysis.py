#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.confounding_controlled_eval import DEFAULT_CONFOUNDERS
from covid_rars.ipw_sensitivity import run_ipw_sensitivity_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IPW cap/clip sensitivity analysis for controlled audio metrics.")
    parser.add_argument("--predictions", type=Path, default=Path("data/outputs/metrics/quality_weighted_fusion_predictions.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_clean.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/ipw_sensitivity_metrics.csv"))
    parser.add_argument("--balance-output", type=Path, default=Path("reports/tables/ipw_sensitivity_balance.csv"))
    parser.add_argument("--weights-output", type=Path, default=Path("data/outputs/metrics/ipw_sensitivity_weights.csv"))
    parser.add_argument("--covariates", nargs="+", default=DEFAULT_CONFOUNDERS)
    parser.add_argument("--group-columns", nargs="*", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--weight-caps", nargs="+", type=float, default=[2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--clip-quantiles", nargs="+", type=float, default=[0.95, 0.99, 1.0])
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
    result = run_ipw_sensitivity_analysis(
        predictions,
        metadata,
        covariates=args.covariates,
        group_columns=group_columns,
        split=args.split,
        threshold=args.threshold,
        weight_caps=args.weight_caps,
        clip_quantiles=args.clip_quantiles,
        random_state=args.random_state,
    )
    for path in [args.metrics_output, args.balance_output, args.weights_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(args.metrics_output, index=False)
    result.balance.to_csv(args.balance_output, index=False)
    result.weights.to_csv(args.weights_output, index=False)
    print(f"Wrote IPW sensitivity metrics: {args.metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote IPW sensitivity balance: {args.balance_output} ({len(result.balance)} rows)")
    print(f"Wrote IPW sensitivity weights: {args.weights_output} ({len(result.weights)} rows)")
    if not result.metrics.empty:
        display_cols = [
            *group_columns,
            "control_method",
            "weight_config",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "f1",
            "effective_sample_size",
            "max_weight",
            "mean_abs_smd_after",
        ]
        print(result.metrics[[col for col in display_cols if col in result.metrics.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
