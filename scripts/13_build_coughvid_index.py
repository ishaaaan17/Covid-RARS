#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from covid_rars.external_datasets import build_coughvid_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a project-compatible COUGHVID cough index.")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-audio", action="store_true")
    parser.add_argument("--min-cough-detected", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_coughvid_index(
        args.raw_dir,
        metadata_path=args.metadata,
        require_audio=args.require_audio,
        min_cough_detected=args.min_cough_detected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote COUGHVID index: {args.output} ({len(df)} rows)")
    if not df.empty:
        print(df["label_binary"].value_counts(dropna=False))
        print(df["manual_quality_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
