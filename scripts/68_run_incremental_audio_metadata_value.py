#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.incremental_value import (
    build_external_model_family_transfer_summary,
    build_incremental_audio_metadata_value,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run participant-aligned metadata-only vs audio-only vs metadata+audio "
            "incremental value checks and summarize external transfer by model family."
        )
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument(
        "--audio-predictions",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_final_validation_predictions.csv"),
    )
    parser.add_argument("--metadata-feature-sets", nargs="*", default=["symptoms_only", "full_safe_metadata"])
    parser.add_argument("--top-k-audio-sources", type=int, default=5)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/reviewer_incremental_audio_metadata_metrics.csv"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/reviewer_incremental_audio_metadata_predictions.csv"),
    )
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=Path("reports/tables/reviewer_incremental_audio_metadata_candidates.csv"),
    )
    parser.add_argument(
        "--family-summary-output",
        type=Path,
        default=Path("reports/tables/reviewer_external_model_family_transfer_summary.csv"),
    )
    parser.add_argument(
        "--compare-internal-metrics",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_final_validation_metrics.csv"),
    )
    parser.add_argument(
        "--compare-external-metrics",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_external_transfer_metrics.csv"),
    )
    parser.add_argument(
        "--wavlm-metrics",
        type=Path,
        default=Path("data/outputs/metrics/deep_external_wavlm_cough_metrics.csv"),
    )
    parser.add_argument(
        "--cnn-metrics",
        type=Path,
        default=Path("data/outputs/metrics/deep_external_cnn_bigru_cough_metrics.csv"),
    )
    return parser.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {label}: {path} ({len(frame)} rows)")


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    audio_predictions = pd.read_csv(args.audio_predictions)
    metrics, predictions, candidates = build_incremental_audio_metadata_value(
        metadata=metadata,
        audio_predictions=audio_predictions,
        metadata_feature_sets=args.metadata_feature_sets,
        top_k_audio_sources=args.top_k_audio_sources,
        n_bootstraps=args.n_bootstraps,
        random_state=args.random_state,
    )
    _write_csv(metrics, args.metrics_output, "incremental metadata+audio metrics")
    _write_csv(predictions, args.predictions_output, "incremental metadata+audio predictions")
    _write_csv(candidates, args.candidates_output, "incremental metadata+audio candidates")

    family_summary = build_external_model_family_transfer_summary(
        compare_internal_metrics=_read_csv_if_exists(args.compare_internal_metrics),
        compare_external_metrics=_read_csv_if_exists(args.compare_external_metrics),
        wavlm_metrics=_read_csv_if_exists(args.wavlm_metrics),
        cnn_metrics=_read_csv_if_exists(args.cnn_metrics),
    )
    _write_csv(family_summary, args.family_summary_output, "external model-family transfer summary")


if __name__ == "__main__":
    main()
