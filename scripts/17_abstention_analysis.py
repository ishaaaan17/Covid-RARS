#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.abstention import apply_abstention, coverage_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate uncertainty/quality abstention decisions and coverage curve.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/metrics/abstention_decisions.csv"))
    parser.add_argument("--curve-output", type=Path, default=Path("data/outputs/metrics/abstention_coverage_curve.csv"))
    parser.add_argument("--uncertainty-low", type=float, default=0.4)
    parser.add_argument("--uncertainty-high", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    if args.metadata and args.metadata.exists() and "quality_flag" not in predictions.columns:
        metadata = pd.read_csv(args.metadata)
        if "recording_id" in predictions.columns and "recording_id" in metadata.columns:
            cols = [c for c in ["recording_id", "quality_flag"] if c in metadata.columns]
            predictions = predictions.merge(metadata[cols].drop_duplicates("recording_id"), on="recording_id", how="left")
        elif "participant_id" in predictions.columns and "participant_id" in metadata.columns and "quality_flag" in metadata.columns:
            participant_quality = metadata.groupby("participant_id")["quality_flag"].agg(lambda s: s.value_counts(dropna=False).index[0]).reset_index()
            predictions = predictions.merge(participant_quality, on="participant_id", how="left")
    decisions = apply_abstention(predictions, uncertainty_low=args.uncertainty_low, uncertainty_high=args.uncertainty_high)
    curve = coverage_curve(decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(args.output, index=False)
    curve.to_csv(args.curve_output, index=False)
    print(f"Wrote abstention decisions: {args.output}")
    print(f"Wrote abstention coverage curve: {args.curve_output}")


if __name__ == "__main__":
    main()
