#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.opensmile_features import extract_opensmile_feature_table
from covid_rars.representation_features import validate_feature_table, write_feature_table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OpenSMILE functionals into a validated feature table.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/features_opensmile_egemaps.csv"))
    parser.add_argument(
        "--feature-set",
        default="egemaps",
        choices=["egemaps", "egemapsv02", "compare", "compare2016", "is10", "is10_paraling"],
    )
    parser.add_argument("--quality-mode", default="all_samples", choices=["all_samples", "quality_ok_only"])
    parser.add_argument("--modality", default=None, help="Optional modality filter such as cough, breath, or speech.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test limit before full extraction.")
    parser.add_argument("--split-name", default=None, help="Force all input rows to this split, e.g. external.")
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--strict", action="store_true", help="Fail immediately on the first unreadable audio file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = pd.read_csv(args.metadata)
    if args.split_name is not None:
        metadata = metadata.copy()
        metadata["split"] = args.split_name
    features = extract_opensmile_feature_table(
        metadata,
        feature_set=args.feature_set,
        quality_mode=args.quality_mode,
        modality=args.modality,
        max_rows=args.max_rows,
        strict=args.strict,
        progress_interval=args.progress_interval,
    )
    validate_feature_table(features)
    write_feature_table(features, args.output)
    print(f"Wrote OpenSMILE features: {args.output} ({len(features)} rows, {len(features.columns)} columns)")


if __name__ == "__main__":
    main()
