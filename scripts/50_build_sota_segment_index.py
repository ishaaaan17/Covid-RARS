#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.sota_segments import build_sota_segment_index, validate_segment_index_no_leakage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe active segment index for SOTA audio branches.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/sota_segment_index.csv"))
    parser.add_argument("--audit-output", type=Path, default=Path("reports/tables/sota_segment_index_audit.csv"))
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"])
    parser.add_argument("--quality-mode", choices=["all_samples", "quality_ok_only"], default="quality_ok_only")
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max-segments-per-recording", type=int, default=8)
    parser.add_argument("--augment-train-copies", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    index = build_sota_segment_index(
        metadata,
        modalities=args.modalities,
        quality_mode=args.quality_mode,
        window_sec=args.window_sec,
        overlap=args.overlap,
        max_segments_per_recording=args.max_segments_per_recording,
        augment_train_copies=args.augment_train_copies,
        random_state=args.random_state,
    )
    audit = validate_segment_index_no_leakage(index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output, index=False)
    audit.to_csv(args.audit_output, index=False)
    print(f"Wrote SOTA segment index: {args.output} ({len(index)} rows)")
    print(f"Wrote SOTA segment audit: {args.audit_output}")
    errors = audit[audit["severity"].astype(str).eq("error")]
    if not errors.empty:
        messages = "\n".join(f"- {row.check}: {row.message}" for row in errors.itertuples(index=False))
        raise ValueError(f"SOTA segment index failed leakage validation:\n{messages}")


if __name__ == "__main__":
    main()
