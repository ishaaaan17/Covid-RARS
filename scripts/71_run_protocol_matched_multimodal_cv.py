#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.protocol_matched_multimodal_cv import run_protocol_matched_multimodal_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run participant-disjoint paper-style CV for the final multimodal fusion pipeline."
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_compare_is10_merged.csv"))
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"])
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of participants held out as test in each repeated split.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.125,
        help="Fraction of non-test participants used for validation; 0.125 with test_fraction=0.2 gives about 70/10/20.",
    )
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[800])
    parser.add_argument("--ranker", default="lightgbm", choices=["lightgbm", "extra_trees", "univariate", "auto"])
    parser.add_argument(
        "--selection-scope",
        default="per_modality_mean",
        choices=["per_modality_mean", "global"],
        help="Feature-ranking scope. per_modality_mean matches the final ComParE+IS10 top-k selection policy.",
    )
    parser.add_argument(
        "--feature-strategy-label",
        default=None,
        help="Optional explicit feature-strategy label for sensitivity runs that reuse a fixed feature bank.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["lightgbm_smote_f80", "svc_rbf_f60", "catboost_smote_f80", "xgboost_smote_f80"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--ensemble-top-k", type=int, default=5)
    parser.add_argument("--global-stack-top-k", type=int, default=0)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/protocol_matched_multimodal_hst_style_metrics.csv"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/protocol_matched_multimodal_hst_style_predictions.csv"),
    )
    parser.add_argument(
        "--feature-selection-output",
        type=Path,
        default=Path("reports/tables/protocol_matched_multimodal_hst_style_feature_selection.csv"),
    )
    parser.add_argument(
        "--branch-selection-output",
        type=Path,
        default=Path("reports/tables/protocol_matched_multimodal_hst_style_branch_selection.csv"),
    )
    parser.add_argument(
        "--split-audit-output",
        type=Path,
        default=Path("reports/tables/protocol_matched_multimodal_hst_style_split_audit.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/tables/protocol_matched_multimodal_hst_style_summary.csv"),
    )
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    result = run_protocol_matched_multimodal_cv(
        features,
        modalities=args.modalities,
        n_splits=args.n_splits,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        top_k_values=args.top_k_values,
        ranker=args.ranker,
        selection_scope=args.selection_scope,
        feature_strategy_label=args.feature_strategy_label,
        model_names=args.model_names,
        random_state=args.random_state,
        optuna_trials=args.optuna_trials,
        ensemble_top_k=args.ensemble_top_k,
        global_stack_top_k=args.global_stack_top_k,
    )
    _write_csv(result.metrics, args.metrics_output)
    _write_csv(result.predictions, args.predictions_output)
    _write_csv(result.feature_selection, args.feature_selection_output)
    _write_csv(result.branch_selection, args.branch_selection_output)
    _write_csv(result.split_audit, args.split_audit_output)
    _write_csv(result.summary, args.summary_output)
    print(f"Wrote protocol-matched multimodal CV metrics: {args.metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote protocol-matched multimodal CV predictions: {args.predictions_output} ({len(result.predictions)} rows)")
    print(
        "Wrote protocol-matched multimodal CV feature selection: "
        f"{args.feature_selection_output} ({len(result.feature_selection)} rows)"
    )
    print(
        "Wrote protocol-matched multimodal CV branch selection: "
        f"{args.branch_selection_output} ({len(result.branch_selection)} rows)"
    )
    print(f"Wrote protocol-matched multimodal CV split audit: {args.split_audit_output} ({len(result.split_audit)} rows)")
    print(f"Wrote protocol-matched multimodal CV summary: {args.summary_output} ({len(result.summary)} rows)")


if __name__ == "__main__":
    main()
