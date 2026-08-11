#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.metadata_confounding_subgroups import build_metadata_confounding_subgroup_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subgroup evidence tables for metadata-confounding audit results.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/outputs/metrics/metadata_confounding_predictions.csv"),
    )
    parser.add_argument(
        "--subgroups",
        nargs="+",
        default=["country", "device", "recording_year", "recording_month"],
    )
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument(
        "--availability-output",
        type=Path,
        default=Path("reports/tables/metadata_confounding_subgroup_availability.csv"),
    )
    parser.add_argument(
        "--breakdown-output",
        type=Path,
        default=Path("reports/tables/metadata_confounding_subgroup_breakdown.csv"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/tables/metadata_confounding_subgroup_metrics.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    predictions = pd.read_csv(args.predictions) if args.predictions.exists() else pd.DataFrame()
    availability, breakdown, metrics = build_metadata_confounding_subgroup_tables(
        metadata=metadata,
        predictions=predictions,
        subgroup_columns=args.subgroups,
        min_samples=args.min_samples,
    )
    for path, frame, label in (
        (args.availability_output, availability, "subgroup availability"),
        (args.breakdown_output, breakdown, "subgroup label breakdown"),
        (args.metrics_output, metrics, "subgroup metadata-confounding metrics"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"Wrote {label}: {path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
