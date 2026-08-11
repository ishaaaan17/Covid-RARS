#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.shift import feature_shift_report, shift_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute source-vs-external feature distribution shift diagnostics.")
    parser.add_argument("--source-features", required=True, type=Path)
    parser.add_argument("--external-features", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/tables/feature_shift_report.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/tables/feature_shift_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.source_features)
    external = pd.read_csv(args.external_features)
    report = feature_shift_report(source, external)
    summary = pd.DataFrame([shift_summary(report)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)
    print(f"Wrote feature shift report: {args.output} ({len(report)} rows)")
    print(f"Wrote feature shift summary: {args.summary_output}")


if __name__ == "__main__":
    main()
