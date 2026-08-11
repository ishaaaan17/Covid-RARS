#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.tier2_comparisons import build_best_vs_baseline_paired_comparisons


DEFAULT_PAIRS = [
    (
        Path("data/outputs/metrics/external_model_grid_predictions.csv"),
        Path("data/outputs/metrics/external_model_grid_metrics.csv"),
        "external_model_grid_predictions",
    ),
    (
        Path("data/outputs/metrics/external_model_grid_opensmile_egemaps_predictions.csv"),
        Path("data/outputs/metrics/external_model_grid_opensmile_egemaps_metrics.csv"),
        "external_model_grid_opensmile_egemaps_predictions",
    ),
    (
        Path("data/outputs/metrics/external_model_grid_beats_predictions.csv"),
        Path("data/outputs/metrics/external_model_grid_beats_metrics.csv"),
        "external_model_grid_beats_predictions",
    ),
    (
        Path("data/outputs/metrics/external_model_grid_panns_predictions.csv"),
        Path("data/outputs/metrics/external_model_grid_panns_metrics.csv"),
        "external_model_grid_panns_predictions",
    ),
]


def _parse_pair(spec: str) -> tuple[Path, Path, str]:
    for first_idx, char in enumerate(spec):
        if char != ":":
            continue
        predictions_path = Path(spec[:first_idx])
        if not predictions_path.exists():
            continue
        for second_idx in range(first_idx + 1, len(spec)):
            if spec[second_idx] != ":":
                continue
            metrics_path = Path(spec[first_idx + 1 : second_idx])
            source = spec[second_idx + 1 :]
            if metrics_path.exists() and source:
                return predictions_path, metrics_path, source
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError("Prediction/metric pair specs must be predictions_csv:metrics_csv:prediction_source")
    return Path(parts[0]), Path(parts[1]), parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired bootstrap best-vs-baseline comparison tables.")
    parser.add_argument("--prediction-metrics-pairs", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/tables/paired_bootstrap_comparisons.csv"))
    parser.add_argument("--baseline-model", default="logistic_regression")
    parser.add_argument("--baseline-strategy", default="all")
    parser.add_argument("--metrics", nargs="+", default=["auroc", "auprc", "brier", "ece"])
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _pairs_from_args(args: argparse.Namespace) -> list[tuple[Path, Path, str]]:
    if args.prediction_metrics_pairs:
        return [_parse_pair(spec) for spec in args.prediction_metrics_pairs]
    return [(pred, metrics, source) for pred, metrics, source in DEFAULT_PAIRS if pred.exists() and metrics.exists()]


def main() -> None:
    args = parse_args()
    rows: list[pd.DataFrame] = []
    pairs = _pairs_from_args(args)
    if not pairs:
        raise FileNotFoundError("No usable prediction/metric pairs found for paired bootstrap comparisons")
    for predictions_path, metrics_path, prediction_source in pairs:
        predictions = pd.read_csv(predictions_path)
        metrics = pd.read_csv(metrics_path)
        table = build_best_vs_baseline_paired_comparisons(
            predictions,
            metrics,
            prediction_source=prediction_source,
            baseline_model=args.baseline_model,
            baseline_strategy=args.baseline_strategy,
            metrics_to_compare=args.metrics,
            n_bootstraps=args.n_bootstraps,
            confidence=args.confidence,
            random_state=args.random_state,
        )
        if not table.empty:
            rows.append(table)
    if not rows:
        raise RuntimeError("No paired comparison rows were produced")
    out = pd.concat(rows, ignore_index=True, sort=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote paired bootstrap comparisons: {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()
