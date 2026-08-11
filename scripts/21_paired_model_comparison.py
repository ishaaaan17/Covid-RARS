#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.model_comparison import paired_comparison_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap model comparison on matched predictions.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--candidate-names", nargs="*", default=None)
    parser.add_argument("--model-column", default="model_name")
    parser.add_argument("--metrics", nargs="+", default=["auroc", "auprc", "brier", "ece"])
    parser.add_argument("--group-columns", nargs="*", default=[])
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/metrics/paired_model_comparison.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    table = paired_comparison_table(
        predictions,
        baseline_name=args.baseline_name,
        candidate_names=args.candidate_names,
        model_column=args.model_column,
        metrics=args.metrics,
        group_columns=args.group_columns,
        n_bootstraps=args.n_bootstraps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote paired comparison table: {args.output} ({len(table)} rows)")


if __name__ == "__main__":
    main()
