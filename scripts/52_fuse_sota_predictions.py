#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.sota_predictions import fuse_sota_prediction_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse SOTA branch prediction CSVs with validation-only stackers.")
    parser.add_argument("--metrics", nargs="+", type=Path, required=True)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--fusion-name", default="sota_validation_stack")
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/sota_fusion_metrics.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("data/outputs/metrics/sota_fusion_predictions.csv"))
    return parser.parse_args()


def _read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def main() -> None:
    args = parse_args()
    metrics = _read_csvs(args.metrics)
    predictions = _read_csvs(args.predictions)
    fused_metrics, fused_predictions = fuse_sota_prediction_sources(
        metrics,
        predictions,
        top_k=args.top_k,
        fusion_name=args.fusion_name,
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    fused_metrics.to_csv(args.metrics_output, index=False)
    fused_predictions.to_csv(args.predictions_output, index=False)
    print(f"Wrote SOTA fusion metrics: {args.metrics_output} ({len(fused_metrics)} rows)")
    print(f"Wrote SOTA fusion predictions: {args.predictions_output} ({len(fused_predictions)} rows)")


if __name__ == "__main__":
    main()
