#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.domain_shift_audit import run_domain_shift_audit


DEFAULT_FEATURE_PAIRS = [
    ("mfcc", Path("data/processed/features_mfcc.csv"), Path("data/processed/coughvid_features_mfcc.csv")),
    (
        "opensmile_egemaps",
        Path("data/processed/features_opensmile_egemaps_coswara_cough.csv"),
        Path("data/processed/features_opensmile_egemaps_coughvid_cough.csv"),
    ),
    ("beats", Path("data/processed/features_beats_coswara_cough.csv"), Path("data/processed/features_beats_coughvid_cough.csv")),
    ("panns", Path("data/processed/features_panns_coswara_cough.csv"), Path("data/processed/features_panns_coughvid_cough.csv")),
]


def _parse_pair(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError("Feature pair specs must be representation:source_csv:external_csv")
    return parts[0], Path(parts[1]), Path(parts[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether audio features reveal dataset/source domain.")
    parser.add_argument("--source-features", type=Path, default=None)
    parser.add_argument("--external-features", type=Path, default=None)
    parser.add_argument("--representation", default="custom")
    parser.add_argument("--feature-pairs", nargs="*", default=None, help="Specs: representation:source_csv:external_csv")
    parser.add_argument("--modality", default="cough")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/domain_shift_audit_metrics.csv"))
    parser.add_argument("--predictions-output", type=Path, default=Path("data/outputs/metrics/domain_shift_audit_predictions.csv"))
    parser.add_argument("--importance-output", type=Path, default=Path("reports/tables/domain_shift_feature_importance.csv"))
    return parser.parse_args()


def _feature_pairs_from_args(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.feature_pairs:
        return [_parse_pair(spec) for spec in args.feature_pairs]
    if args.source_features is not None and args.external_features is not None:
        return [(args.representation, args.source_features, args.external_features)]
    return [(name, source, external) for name, source, external in DEFAULT_FEATURE_PAIRS if source.exists() and external.exists()]


def main() -> None:
    args = parse_args()
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    pairs = _feature_pairs_from_args(args)
    if not pairs:
        raise FileNotFoundError("No domain-shift feature pairs found")
    for representation, source_path, external_path in pairs:
        source = pd.read_csv(source_path)
        external = pd.read_csv(external_path)
        result = run_domain_shift_audit(
            source,
            external,
            representation=representation,
            modality=args.modality,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        metric_frames.append(result.metrics)
        prediction_frames.append(result.predictions)
        importance_frames.append(result.feature_importance)
        row = result.metrics.iloc[0]
        print(f"Ran domain-shift audit {representation}: AUROC={row['domain_auroc']:.4f}, features={int(row['n_features'])}")

    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    importance = pd.concat(importance_frames, ignore_index=True, sort=False)
    for path in [args.metrics_output, args.predictions_output, args.importance_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False)
    predictions.to_csv(args.predictions_output, index=False)
    importance.to_csv(args.importance_output, index=False)
    print(f"Wrote domain-shift metrics: {args.metrics_output} ({len(metrics)} rows)")
    print(f"Wrote domain-shift predictions: {args.predictions_output} ({len(predictions)} rows)")
    print(f"Wrote domain-shift feature importance: {args.importance_output} ({len(importance)} rows)")


if __name__ == "__main__":
    main()
