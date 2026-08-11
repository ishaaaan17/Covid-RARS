#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.sota_prediction_stack import run_gated_prediction_stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation-gated prediction stacking over model output CSVs.")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Named sources in the form name=metrics.csv:predictions.csv",
    )
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-validation-drop", type=float, default=0.03)
    parser.add_argument("--min-validation-auroc", type=float, default=0.0)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/sota_gated_stack_metrics.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("data/outputs/metrics/sota_gated_stack_predictions.csv"))
    parser.add_argument("--candidates-output", type=Path, default=Path("reports/tables/sota_gated_stack_candidates.csv"))
    return parser.parse_args()


def _read_source(spec: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "=" not in spec or ":" not in spec:
        raise ValueError(f"Invalid source spec {spec!r}; expected name=metrics.csv:predictions.csv")
    name, rest = spec.split("=", 1)
    metrics_path, predictions_path = _split_existing_path_pair(rest)
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    metrics["source_run"] = name
    predictions["source_run"] = name
    return metrics, predictions


def _split_existing_path_pair(spec: str) -> tuple[Path, Path]:
    """Split metrics:predictions while allowing Windows drive-letter colons."""
    for idx, char in enumerate(spec):
        if char != ":":
            continue
        left = Path(spec[:idx])
        right = Path(spec[idx + 1 :])
        if left.exists() and right.exists():
            return left, right
    left_text, right_text = spec.split(":", 1)
    return Path(left_text), Path(right_text)


def main() -> None:
    args = parse_args()
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for spec in args.sources:
        metrics, predictions = _read_source(spec)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
    result = run_gated_prediction_stack(
        pd.concat(metric_frames, ignore_index=True, sort=False),
        pd.concat(prediction_frames, ignore_index=True, sort=False),
        top_k=args.top_k,
        max_validation_drop=args.max_validation_drop,
        min_validation_auroc=args.min_validation_auroc,
        min_sources=args.min_sources,
    )
    for path, frame in (
        (args.metrics_output, result.metrics),
        (args.predictions_output, result.predictions),
        (args.candidates_output, result.candidates),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print(f"Wrote gated stack metrics: {args.metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote gated stack predictions: {args.predictions_output} ({len(result.predictions)} rows)")
    print(f"Wrote gated stack candidates: {args.candidates_output} ({len(result.candidates)} rows)")
    if not result.candidates.empty:
        if "has_predictions" in result.candidates.columns:
            print("Candidate prediction availability:")
            print(result.candidates["has_predictions"].value_counts(dropna=False).to_string())
        if "reject_reason" in result.candidates.columns:
            print("Candidate rejection reasons:")
            print(result.candidates["reject_reason"].replace("", "selected_or_available").value_counts(dropna=False).to_string())
    if result.metrics.empty:
        print("No gated stack metrics were produced; inspect the candidate table for missing_predictions or label/split coverage.")


if __name__ == "__main__":
    main()
