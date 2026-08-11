#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.data_index import AUDIO_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw dataset layout before indexing.")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/tables/coswara_layout_audit.csv"))
    parser.add_argument("--max-depth", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    rows = []
    audio_count = 0
    metadata_count = 0
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(raw_dir)
        depth = len(rel.parts)
        if depth > args.max_depth:
            continue
        suffix = path.suffix.lower()
        is_audio = suffix in AUDIO_EXTENSIONS
        is_metadata = suffix in {".csv", ".json", ".xlsx", ".tsv"}
        audio_count += int(is_audio)
        metadata_count += int(is_metadata)
        if is_audio or is_metadata:
            rows.append(
                {
                    "relative_path": rel.as_posix(),
                    "suffix": suffix,
                    "depth": depth,
                    "parent": path.parent.name,
                    "is_audio": is_audio,
                    "is_metadata": is_metadata,
                    "size_bytes": path.stat().st_size,
                }
            )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote layout audit: {args.output}")
    print(f"Audio files sampled/listed: {audio_count}")
    print(f"Metadata-like files sampled/listed: {metadata_count}")
    if not df.empty:
        print(df.groupby(["suffix", "is_audio", "is_metadata"]).size().reset_index(name="n"))
        print("Example files:")
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
