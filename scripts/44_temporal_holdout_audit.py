#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.temporal_holdout import build_temporal_external_unification, run_temporal_holdout_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run participant-level temporal holdout audit for Coswara audio and metadata."
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_clean.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_mfcc.csv"))
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"])
    parser.add_argument("--model-names", nargs="+", default=["logistic_regression"])
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--no-existing-split-reference", action="store_true")
    parser.add_argument("--no-time-stratified-reference", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--evidence-matrix", type=Path, default=Path("reports/tables/publication_evidence_matrix.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/temporal_holdout_metrics.csv"))
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/temporal_holdout_predictions.csv"),
    )
    parser.add_argument(
        "--split-summary-output",
        type=Path,
        default=Path("reports/tables/temporal_holdout_split_summary.csv"),
    )
    parser.add_argument(
        "--modality-coverage-output",
        type=Path,
        default=Path("reports/tables/temporal_holdout_modality_coverage.csv"),
    )
    parser.add_argument(
        "--metadata-importance-output",
        type=Path,
        default=Path("reports/tables/temporal_holdout_metadata_feature_importance.csv"),
    )
    parser.add_argument(
        "--metadata-group-summary-output",
        type=Path,
        default=Path("reports/tables/temporal_holdout_metadata_group_summary.csv"),
    )
    parser.add_argument(
        "--metadata-ablation-output",
        type=Path,
        default=Path("reports/tables/temporal_metadata_ablation.csv"),
    )
    parser.add_argument(
        "--stability-summary-output",
        type=Path,
        default=Path("reports/tables/temporal_stability_summary.csv"),
    )
    parser.add_argument(
        "--bootstrap-ci-output",
        type=Path,
        default=Path("data/outputs/metrics/temporal_holdout_bootstrap_ci.csv"),
    )
    parser.add_argument(
        "--external-unification-output",
        type=Path,
        default=Path("reports/tables/temporal_external_unification.csv"),
    )
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    features = pd.read_csv(args.features)
    result = run_temporal_holdout_audit(
        metadata=metadata,
        features=features,
        modalities=args.modalities,
        model_names=args.model_names,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        include_existing_split_reference=not args.no_existing_split_reference,
        include_time_stratified_reference=not args.no_time_stratified_reference,
        bootstrap_samples=args.bootstrap_samples,
    )
    _write_csv(result.metrics, args.metrics_output)
    _write_csv(result.predictions, args.predictions_output)
    _write_csv(result.split_summary, args.split_summary_output)
    _write_csv(result.modality_coverage, args.modality_coverage_output)
    _write_csv(result.metadata_feature_importance, args.metadata_importance_output)
    _write_csv(result.metadata_group_summary, args.metadata_group_summary_output)
    _write_csv(result.metadata_ablation, args.metadata_ablation_output)
    _write_csv(result.stability_summary, args.stability_summary_output)
    _write_csv(result.bootstrap_ci, args.bootstrap_ci_output)
    external_reference = pd.DataFrame()
    if args.evidence_matrix.exists():
        external_reference = pd.read_csv(args.evidence_matrix)
    external_unification = build_temporal_external_unification(result.metrics, external_reference=external_reference)
    _write_csv(external_unification, args.external_unification_output)
    print(f"Wrote temporal holdout metrics: {args.metrics_output} ({len(result.metrics)} rows)")
    print(f"Wrote temporal holdout predictions: {args.predictions_output} ({len(result.predictions)} rows)")
    print(f"Wrote temporal holdout split summary: {args.split_summary_output} ({len(result.split_summary)} rows)")
    print(f"Wrote temporal holdout modality coverage: {args.modality_coverage_output} ({len(result.modality_coverage)} rows)")
    print(f"Wrote temporal metadata ablation: {args.metadata_ablation_output} ({len(result.metadata_ablation)} rows)")
    print(f"Wrote temporal stability summary: {args.stability_summary_output} ({len(result.stability_summary)} rows)")
    print(f"Wrote temporal bootstrap CIs: {args.bootstrap_ci_output} ({len(result.bootstrap_ci)} rows)")
    print(f"Wrote temporal/external unification table: {args.external_unification_output} ({len(external_unification)} rows)")


if __name__ == "__main__":
    main()
