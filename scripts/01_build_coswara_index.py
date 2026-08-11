#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from covid_rars.data_index import build_audio_index, discover_audio_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Coswara audio index.")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        files = discover_audio_files(args.raw_dir)
        print(f"Discovered {len(files)} audio files under {args.raw_dir}")
        return
    df = build_audio_index(args.raw_dir, dataset="coswara")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    if not df.empty:
        print(df.groupby(["modality", "submodality"]).size().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()
