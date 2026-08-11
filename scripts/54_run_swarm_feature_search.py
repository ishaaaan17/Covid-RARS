#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.swarm_feature_search import run_swarm_feature_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run binary-PSO feature selection over strong acoustic features.")
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_strong_acoustic.csv"))
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"])
    parser.add_argument("--classifier", choices=["logistic", "lightgbm", "extra_trees"], default="lightgbm")
    parser.add_argument("--particles", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--max-candidate-features", type=int, default=256)
    parser.add_argument("--min-selected-features", type=int, default=8)
    parser.add_argument("--max-selected-features", type=int, default=64)
    parser.add_argument("--sparsity-penalty", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/sota_swarm_feature_metrics.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("data/outputs/metrics/sota_swarm_feature_predictions.csv"))
    parser.add_argument("--selection-output", type=Path, default=Path("reports/tables/sota_swarm_feature_selection.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    result = run_swarm_feature_search(
        features,
        modalities=args.modalities,
        classifier=args.classifier,
        particles=args.particles,
        iterations=args.iterations,
        max_candidate_features=args.max_candidate_features,
        min_selected_features=args.min_selected_features,
        max_selected_features=args.max_selected_features,
        sparsity_penalty=args.sparsity_penalty,
        random_state=args.random_state,
        verbose=args.verbose,
    )
    for path, frame in (
        (args.metrics_output, result.metrics),
        (args.predictions_output, result.predictions),
        (args.selection_output, result.selection),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print(f"Wrote swarm feature metrics: {args.metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote swarm feature predictions: {args.predictions_output} ({len(result.predictions)} rows)")
    print(f"Wrote swarm feature selection: {args.selection_output} ({len(result.selection)} rows)")


if __name__ == "__main__":
    main()
