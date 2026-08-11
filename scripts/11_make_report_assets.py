#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.reporting import save_class_distribution, save_reliability_diagram, write_report_outline, write_slides_outline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate basic report assets.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_clean.csv"))
    parser.add_argument("--predictions", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metadata.exists():
        metadata = pd.read_csv(args.metadata)
        save_class_distribution(metadata, Path("reports/figures/class_distribution.png"))
    if args.predictions and args.predictions.exists():
        predictions = pd.read_csv(args.predictions)
        save_reliability_diagram(predictions, Path("reports/figures/reliability_diagram.png"))
    write_report_outline(Path("reports/report_outline.md"))
    write_slides_outline(Path("reports/slides_outline.md"))
    print("Report assets generated")


if __name__ == "__main__":
    main()
