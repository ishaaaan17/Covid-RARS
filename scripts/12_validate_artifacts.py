#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.validation import (
    ValidationIssue,
    issues_to_frame,
    raise_on_errors,
    validate_index,
    validate_metadata,
    validate_quality,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated artifacts before modeling/reporting.")
    parser.add_argument("--index", type=Path, default=Path("data/interim/coswara_index.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_clean.csv"))
    parser.add_argument("--quality", type=Path, default=Path("data/processed/audio_quality.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/tables/validation_issues.csv"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issues = []
    if args.index.exists():
        issues.extend(validate_index(pd.read_csv(args.index)))
    else:
        issues.append(ValidationIssue("error", "index_exists", f"Missing {args.index}"))
    if args.metadata.exists():
        issues.extend(validate_metadata(pd.read_csv(args.metadata)))
    else:
        issues.append(ValidationIssue("error", "metadata_exists", f"Missing {args.metadata}"))
    if args.quality.exists():
        issues.extend(validate_quality(pd.read_csv(args.quality)))
    else:
        issues.append(ValidationIssue("warning", "quality_exists", f"Missing {args.quality}"))

    frame = issues_to_frame(issues)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    if frame.empty:
        print("No validation issues found")
    else:
        print(frame.to_string(index=False))
    if args.strict and not frame.empty:
        raise_on_errors(issues)


if __name__ == "__main__":
    main()
