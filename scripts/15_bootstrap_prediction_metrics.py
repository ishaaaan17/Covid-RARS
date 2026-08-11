#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.statistics import bootstrap_prediction_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bootstrap confidence intervals for prediction metrics.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", nargs="+", default=["auroc", "auprc", "brier", "ece"])
    parser.add_argument("--group-columns", nargs="*", default=[])
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    table = bootstrap_prediction_table(
        predictions,
        metrics=args.metrics,
        group_columns=args.group_columns,
        n_bootstraps=args.n_bootstraps,
        random_state=args.random_state,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote bootstrap metric intervals: {args.output} ({len(table)} rows)")


if __name__ == "__main__":
    main()
