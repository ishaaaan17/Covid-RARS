#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.delta_bootstrap import build_auto_reviewer_comparisons, build_delta_bootstrap_table
from covid_rars.final_uncertainty import load_final_prediction_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap metric drops for final validation comparisons. Matched Coswara-to-COUGHVID "
            "cough comparisons use independent two-sample resampling with participant-level Coswara units."
        )
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=Path,
        default=[
            Path("data/outputs/metrics/compare_is10_final_validation_predictions.csv"),
            Path("data/outputs/metrics/compare_is10_external_transfer_predictions.csv"),
        ],
    )
    parser.add_argument(
        "--final-summary",
        type=Path,
        default=Path("reports/tables/compare_is10_final_validation_summary.csv"),
    )
    parser.add_argument(
        "--final-metrics",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_final_validation_metrics.csv"),
    )
    parser.add_argument(
        "--external-metrics",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_external_transfer_metrics.csv"),
    )
    parser.add_argument("--metrics", nargs="+", default=["auroc", "auprc", "brier", "ece"])
    parser.add_argument("--n-bootstraps", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/final_validation_delta_bootstrap_ci.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_final_prediction_files(args.predictions)
    final_summary = pd.read_csv(args.final_summary)
    final_metrics = pd.read_csv(args.final_metrics)
    external_metrics = pd.read_csv(args.external_metrics)
    comparisons = build_auto_reviewer_comparisons(final_summary, final_metrics, external_metrics)
    table = build_delta_bootstrap_table(
        predictions,
        comparisons=comparisons,
        metrics=args.metrics,
        n_bootstraps=args.n_bootstraps,
        random_state=args.random_state,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote final validation delta bootstrap CIs: {args.output} ({len(table)} rows)")
    if not table.empty:
        display_columns = [
            "comparison_id",
            "metric",
            "left_point",
            "right_point",
            "delta",
            "ci_low",
            "ci_high",
            "resampling_scheme",
            "left_resampling_level",
            "right_resampling_level",
            "left_analysis_n",
            "right_analysis_n",
        ]
        print(table[[column for column in display_columns if column in table.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
