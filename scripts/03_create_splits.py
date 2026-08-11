#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.split import create_participant_splits, split_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-safe participant-level splits.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path, default=Path("reports/tables/split_audit.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    manifest, metadata_with_split = create_participant_splits(metadata)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(args.output, index=False)
    metadata_with_split.to_csv(args.metadata_output, index=False)
    split_audit(metadata_with_split).to_csv(args.audit_output, index=False)

    print(f"Wrote split manifest: {args.output} ({len(manifest)} participants)")
    print(f"Updated metadata with split column: {args.metadata_output}")
    print(f"Wrote split audit: {args.audit_output}")
    print(manifest["split"].value_counts())


if __name__ == "__main__":
    main()
