#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.strong_features import build_strong_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract extended paper-style acoustic features.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/features_strong_acoustic.csv"))
    parser.add_argument("--quality-mode", choices=["all_samples", "quality_ok_only"], default="all_samples")
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument(
        "--augment-train-copies",
        type=int,
        default=0,
        help="Create this many deterministic augmented copies for training rows only.",
    )
    parser.add_argument("--augmentation-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    features = build_strong_feature_table(
        metadata,
        quality_mode=args.quality_mode,
        progress_interval=args.progress_interval,
        augment_train_copies=args.augment_train_copies,
        augmentation_seed=args.augmentation_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote strong acoustic features: {args.output} ({len(features)} rows, {features.shape[1]} columns)")


if __name__ == "__main__":
    main()
