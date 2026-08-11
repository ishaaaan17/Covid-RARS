#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from covid_rars.external_features import write_external_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MFCC/acoustic features from a COUGHVID index.")
    parser.add_argument("--index", type=Path, default=Path("data/interim/coughvid_index.csv"))
    parser.add_argument("--features-output", type=Path, default=Path("data/processed/coughvid_features_mfcc.csv"))
    parser.add_argument("--split-name", default="external")
    parser.add_argument("--include-unknown-labels", action="store_true")
    parser.add_argument("--quality-ok-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test limit before full extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = write_external_feature_table(
        args.index,
        args.features_output,
        split_name=args.split_name,
        labeled_only=not args.include_unknown_labels,
        quality_ok_only=args.quality_ok_only,
        max_rows=args.max_rows,
    )
    print(f"Wrote external features: {args.features_output} ({len(features)} rows, {features.shape[1]} columns)")


if __name__ == "__main__":
    main()
