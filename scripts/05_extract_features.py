#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.features import extract_feature_table
from covid_rars.spectrograms import build_spectrogram_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MFCC/acoustic features and spectrogram files.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--features-output", required=True, type=Path)
    parser.add_argument("--spectrogram-dir", type=Path, default=Path("data/processed/spectrograms"))
    parser.add_argument("--spectrogram-index-output", type=Path, default=Path("data/processed/spectrogram_index.csv"))
    parser.add_argument("--quality-mode", choices=["all_samples", "quality_ok_only"], default="all_samples")
    parser.add_argument("--skip-spectrograms", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    features = extract_feature_table(metadata, quality_mode=args.quality_mode)
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features_output, index=False)
    print(f"Wrote features: {args.features_output} ({len(features)} rows, {features.shape[1]} columns)")

    if not args.skip_spectrograms:
        spec_index = build_spectrogram_index(metadata, output_dir=args.spectrogram_dir)
        args.spectrogram_index_output.parent.mkdir(parents=True, exist_ok=True)
        spec_index.to_csv(args.spectrogram_index_output, index=False)
        print(f"Wrote spectrogram index: {args.spectrogram_index_output} ({len(spec_index)} rows)")


if __name__ == "__main__":
    main()
