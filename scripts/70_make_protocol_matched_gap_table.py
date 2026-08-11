#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.protocol_matched_comparison import (
    build_protocol_matched_gap_summary,
    read_existing_csvs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a paper-target versus protocol-matched result gap table."
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("reports/tables/protocol_matched_paper_targets.csv"),
    )
    parser.add_argument(
        "--protocol-metrics",
        action="append",
        type=Path,
        default=[],
        help="Protocol-matched metrics CSV. Can be repeated.",
    )
    parser.add_argument(
        "--final-validation-summary",
        type=Path,
        default=Path("reports/tables/compare_is10_final_validation_summary.csv"),
    )
    parser.add_argument(
        "--external-transfer-summary",
        type=Path,
        default=Path("reports/tables/reviewer_external_model_family_transfer_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/protocol_matched_gap_summary.csv"),
    )
    return parser.parse_args()


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.targets)
    protocol_metrics = read_existing_csvs(args.protocol_metrics)
    final_validation = _read_optional(args.final_validation_summary)
    external_transfer = _read_optional(args.external_transfer_summary)
    summary = build_protocol_matched_gap_summary(
        targets,
        protocol_metrics,
        final_validation_summary=final_validation,
        external_transfer_summary=external_transfer,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Wrote protocol-matched gap summary: {args.output} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
