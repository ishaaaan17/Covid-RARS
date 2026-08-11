#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.confounding import balance_table, coarsened_exact_match
from covid_rars.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create matched metadata subset and balance diagnostics.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--covariates", nargs="+", default=["age_bucket", "gender"])
    parser.add_argument("--matched-output", type=Path, default=Path("data/processed/metadata_matched.csv"))
    parser.add_argument("--balance-output", type=Path, default=Path("reports/tables/confounding_balance.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/matched_subset_metrics.csv"))
    return parser.parse_args()


def filter_predictions_to_matched_subset(predictions: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    if "recording_id" in predictions.columns and "recording_id" in matched.columns:
        matched_ids = matched["recording_id"].dropna().astype(str).unique()
        return predictions[predictions["recording_id"].astype(str).isin(matched_ids)].copy()
    if "participant_id" in predictions.columns and "participant_id" in matched.columns:
        matched_ids = matched["participant_id"].dropna().astype(str).unique()
        return predictions[predictions["participant_id"].astype(str).isin(matched_ids)].copy()
    raise KeyError(
        "Predictions must contain recording_id or participant_id matching the matched metadata table"
    )


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    matched = coarsened_exact_match(metadata, covariates=args.covariates)
    balance = balance_table(matched, covariates=args.covariates)
    args.matched_output.parent.mkdir(parents=True, exist_ok=True)
    args.balance_output.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.matched_output, index=False)
    balance.to_csv(args.balance_output, index=False)
    print(f"Wrote matched metadata: {args.matched_output} ({len(matched)} rows)")
    print(f"Wrote balance table: {args.balance_output}")
    if args.predictions and args.predictions.exists() and not matched.empty:
        predictions = pd.read_csv(args.predictions)
        subset = filter_predictions_to_matched_subset(predictions, matched)
        metrics = evaluate_predictions(subset)
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.metrics_output, index=False)
        print(f"Wrote matched-subset metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
