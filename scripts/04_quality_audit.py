#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.quality import attach_quality_flags, quality_summary, run_quality_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run audio quality and active-region audit.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path, default=Path("reports/tables/quality_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    quality = run_quality_audit(metadata)
    metadata_with_quality = attach_quality_flags(metadata, quality)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    quality.to_csv(args.output, index=False)
    metadata_with_quality.to_csv(args.metadata_output, index=False)
    quality_summary(quality).to_csv(args.summary_output, index=False)

    print(f"Wrote quality audit: {args.output} ({len(quality)} rows)")
    print(f"Updated metadata: {args.metadata_output}")
    print(f"Wrote quality summary: {args.summary_output}")
    print(quality["quality_flag"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
