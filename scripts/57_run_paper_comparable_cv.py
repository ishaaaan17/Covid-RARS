#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.paper_comparable_cv import run_paper_comparable_cv
from covid_rars.strong_baseline import DEFAULT_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-comparable 10-fold intra-dataset CV for headline SOTA comparison."
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_compare_is10_merged.csv"))
    parser.add_argument("--modality", default="cough")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[500, 800, 1200])
    parser.add_argument("--ranker", default="lightgbm", choices=["lightgbm", "extra_trees", "univariate", "auto"])
    parser.add_argument("--model-names", nargs="+", default=list(DEFAULT_MODEL_NAMES))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--optuna-trials", type=int, default=25)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/paper_comparable_cv_metrics.csv"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/paper_comparable_cv_predictions.csv"),
    )
    parser.add_argument(
        "--feature-selection-output",
        type=Path,
        default=Path("reports/tables/paper_comparable_cv_feature_selection.csv"),
    )
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    result = run_paper_comparable_cv(
        features,
        modality=args.modality,
        n_splits=args.n_splits,
        validation_fraction=args.validation_fraction,
        top_k_values=args.top_k_values,
        ranker=args.ranker,
        model_names=args.model_names,
        random_state=args.random_state,
        optuna_trials=args.optuna_trials,
    )
    _write_csv(result.metrics, args.metrics_output)
    _write_csv(result.predictions, args.predictions_output)
    _write_csv(result.feature_selection, args.feature_selection_output)
    print(f"Wrote paper-comparable CV metrics: {args.metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote paper-comparable CV predictions: {args.predictions_output} ({len(result.predictions)} rows)")
    print(
        f"Wrote paper-comparable CV feature selection: "
        f"{args.feature_selection_output} ({len(result.feature_selection)} rows)"
    )


if __name__ == "__main__":
    main()
