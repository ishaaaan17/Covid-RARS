#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.metadata import clean_metadata, metadata_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean indexed metadata and build modality availability.")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--availability-output", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path, default=Path("reports/tables/dataset_audit.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = pd.read_csv(args.index)
    metadata, availability = clean_metadata(index_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.availability_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(args.output, index=False)
    availability.to_csv(args.availability_output, index=False)
    metadata_audit(metadata).to_csv(args.audit_output, index=False)

    print(f"Wrote metadata: {args.output} ({len(metadata)} rows)")
    print(f"Wrote availability: {args.availability_output} ({len(availability)} participants)")
    print(f"Wrote audit: {args.audit_output}")
    print(metadata["label_binary"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
