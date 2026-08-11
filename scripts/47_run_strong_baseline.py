#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.strong_baseline import DEFAULT_MODEL_NAMES, run_strong_baseline, save_strong_baseline_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-comparable strong internal Coswara baseline."
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_strong_acoustic.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"])
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=list(DEFAULT_MODEL_NAMES),
    )
    parser.add_argument("--require-quality-ok", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--optuna-trials", type=int, default=25)
    parser.add_argument(
        "--ensemble-top-k",
        type=int,
        default=3,
        help="Average the top-k validation models per modality as an additional model candidate.",
    )
    parser.add_argument(
        "--feature-level-fusion",
        action="store_true",
        help="Train multimodal classifiers on concatenated participant-level modality features before prediction fusion.",
    )
    parser.add_argument(
        "--global-stack-top-k",
        type=int,
        default=0,
        help="Fit validation-trained global stackers over the top-k validation prediction sources.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/strong_baseline_metrics.csv"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/strong_baseline_predictions.csv"),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path("reports/tables/strong_baseline_model_selection.csv"),
    )
    parser.add_argument(
        "--protocol-audit-output",
        type=Path,
        default=Path("reports/tables/strong_baseline_protocol_audit.csv"),
    )
    parser.add_argument(
        "--participant-audit-output",
        type=Path,
        default=Path("reports/tables/strong_baseline_participant_audit.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.features.exists():
        raise FileNotFoundError(
            f"Feature table not found: {args.features}. Run the E2E wrapper with --raw-dir, "
            "or run scripts/05_extract_features.py first."
        )
    features = pd.read_csv(args.features)
    metadata = pd.read_csv(args.metadata) if args.metadata.exists() else None
    result = run_strong_baseline(
        features=features,
        metadata=metadata,
        modalities=args.modalities,
        model_names=args.model_names,
        require_quality_ok=args.require_quality_ok,
        random_state=args.random_state,
        optuna_trials=args.optuna_trials,
        ensemble_top_k=args.ensemble_top_k,
        enable_feature_level_fusion=args.feature_level_fusion,
        global_stack_top_k=args.global_stack_top_k,
    )
    save_strong_baseline_result(
        result,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
        selection_output=args.selection_output,
        protocol_audit_output=args.protocol_audit_output,
        participant_audit_output=args.participant_audit_output,
    )
    print(f"Wrote strong baseline metrics: {args.metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote strong baseline predictions: {args.predictions_output} ({len(result.predictions)} rows)")
    print(f"Wrote strong baseline model selection: {args.selection_output} ({len(result.selection)} rows)")
    print(f"Wrote strong baseline protocol audit: {args.protocol_audit_output}")


if __name__ == "__main__":
    main()
