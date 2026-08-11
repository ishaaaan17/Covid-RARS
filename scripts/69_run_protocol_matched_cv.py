#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.protocol_matched_cv import run_protocol_matched_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run participant-disjoint paper-style CV for protocol-matched comparison."
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_compare_is10_merged.csv"))
    parser.add_argument("--modality", default="cough")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of participants held out as test in each repeated split; 0.2 matches the HST paper's stated 20%% test set.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.125,
        help="Fraction of non-test participants used for validation; 0.125 with --test-fraction 0.2 gives about 70/10/20 train/validation/test.",
    )
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[800])
    parser.add_argument("--ranker", default="lightgbm", choices=["lightgbm", "extra_trees", "univariate", "auto"])
    parser.add_argument("--model-names", nargs="+", default=["svc_rbf_f60"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/protocol_matched_hst_style_cough_metrics.csv"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/protocol_matched_hst_style_cough_predictions.csv"),
    )
    parser.add_argument(
        "--feature-selection-output",
        type=Path,
        default=Path("reports/tables/protocol_matched_hst_style_cough_feature_selection.csv"),
    )
    parser.add_argument(
        "--split-audit-output",
        type=Path,
        default=Path("reports/tables/protocol_matched_hst_style_cough_split_audit.csv"),
    )
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    result = run_protocol_matched_cv(
        features,
        modality=args.modality,
        n_splits=args.n_splits,
        test_fraction=args.test_fraction,
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
    _write_csv(result.split_audit, args.split_audit_output)
    print(f"Wrote protocol-matched CV metrics: {args.metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote protocol-matched CV predictions: {args.predictions_output} ({len(result.predictions)} rows)")
    print(
        f"Wrote protocol-matched CV feature selection: "
        f"{args.feature_selection_output} ({len(result.feature_selection)} rows)"
    )
    print(f"Wrote protocol-matched CV split audit: {args.split_audit_output} ({len(result.split_audit)} rows)")


if __name__ == "__main__":
    main()
