#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from covid_rars.metadata_baseline import train_metadata_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train metadata/symptom-only baseline for confounding analysis.")
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--models-dir", type=Path, default=Path("data/outputs/models"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/metadata_baseline_metrics.csv"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/outputs/metrics/metadata_predictions_validation.csv"))
    parser.add_argument("--test-output", type=Path, default=Path("data/outputs/metrics/metadata_predictions_test.csv"))
    parser.add_argument("--feature-columns", nargs="+", default=["age", "gender", "country", "symptoms_json", "comorbidities_json", "quality_flag"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    result = train_metadata_baseline(metadata, feature_columns=args.feature_columns)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, args.models_dir / "metadata_logistic_regression.joblib")
    pd.DataFrame([result.metrics]).to_csv(args.metrics_output, index=False)
    result.validation_predictions.to_csv(args.validation_output, index=False)
    result.test_predictions.to_csv(args.test_output, index=False)
    print(f"Wrote metadata baseline metrics: {args.metrics_output}")
    print(result.metrics)


if __name__ == "__main__":
    main()
