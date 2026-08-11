#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.final_uncertainty import DEFAULT_FINAL_GROUP_COLUMNS, load_final_prediction_files
from covid_rars.reviewer_evidence import (
    build_audio_metadata_residual_correlation,
    build_fixed_sensitivity_table,
    build_partial_target_recalibration,
    build_shuffle_label_sanity,
)
from covid_rars.shuffle_retrain_sanity import run_metadata_shuffle_retrain_sanity


FINAL_PREDICTION_DEFAULTS = [
    Path("data/outputs/metrics/compare_is10_final_validation_predictions.csv"),
    Path("data/outputs/metrics/compare_is10_external_transfer_predictions.csv"),
]
METADATA_PREDICTION_DEFAULTS = [Path("data/outputs/metrics/metadata_confounding_predictions.csv")]
SELECTOR_COLUMNS = [
    "evaluation_protocol",
    "analysis_family",
    "model_name",
    "modality",
    "submodality",
    "modality_combination",
    "fusion_method",
    "feature_strategy",
    "selected_feature_k",
    "metric_split",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reviewer-facing sanity, operating-point, residual-correlation, and target-recalibration tables."
    )
    parser.add_argument("--final-predictions", nargs="+", type=Path, default=FINAL_PREDICTION_DEFAULTS)
    parser.add_argument("--metadata-predictions", nargs="+", type=Path, default=METADATA_PREDICTION_DEFAULTS)
    parser.add_argument("--final-summary", type=Path, default=Path("reports/tables/compare_is10_final_validation_summary.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--metadata-metrics", type=Path, default=Path("data/outputs/metrics/metadata_confounding_metrics.csv"))
    parser.add_argument("--metadata-feature-sets", nargs="+", default=["full_safe_metadata", "symptoms_only", "demographic_protocol_only"])
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--metadata-retrain-permutations", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-sensitivities", nargs="+", type=float, default=[0.90])
    parser.add_argument("--recalibration-fraction", type=float, default=0.25)
    parser.add_argument(
        "--final-shuffle-output",
        type=Path,
        default=Path("reports/tables/final_validation_shuffle_label_sanity.csv"),
    )
    parser.add_argument(
        "--metadata-shuffle-output",
        type=Path,
        default=Path("reports/tables/metadata_confounding_shuffle_label_sanity.csv"),
    )
    parser.add_argument(
        "--metadata-shuffle-retrain-output",
        type=Path,
        default=Path("reports/tables/metadata_confounding_shuffle_retrain_sanity.csv"),
    )
    parser.add_argument(
        "--fixed-sensitivity-output",
        type=Path,
        default=Path("reports/tables/final_validation_fixed_sensitivity_operating_points.csv"),
    )
    parser.add_argument(
        "--residual-correlation-output",
        type=Path,
        default=Path("reports/tables/audio_metadata_residual_correlation.csv"),
    )
    parser.add_argument(
        "--recalibration-metrics-output",
        type=Path,
        default=Path("reports/tables/coughvid_partial_recalibration_metrics.csv"),
    )
    parser.add_argument(
        "--recalibration-predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/coughvid_partial_recalibration_predictions.csv"),
    )
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {label}: {path} ({len(frame)} rows)")


def _read_optional_prediction_files(paths: list[Path]) -> pd.DataFrame:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        return pd.DataFrame()
    return load_final_prediction_files(existing)


def _has_skipped_column(frame: pd.DataFrame) -> pd.Series:
    if "skipped" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["skipped"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _best_existing_summary_row(summary_path: Path) -> pd.Series | None:
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path)
    if summary.empty or "evaluation_protocol" not in summary.columns:
        return None
    work = summary[summary["evaluation_protocol"].astype(str).eq("compare_is10_existing_participant_split")].copy()
    if work.empty:
        return None
    work = work[~_has_skipped_column(work)].copy()
    work["auroc"] = pd.to_numeric(work.get("auroc"), errors="coerce")
    work["auprc"] = pd.to_numeric(work.get("auprc"), errors="coerce")
    return work.sort_values(["auroc", "auprc"], ascending=False).iloc[0] if not work.empty else None


def _best_metadata_predictions(predictions: pd.DataFrame, metrics_path: Path) -> pd.DataFrame:
    if predictions.empty or not metrics_path.exists():
        return predictions
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return predictions
    metrics = metrics[~_has_skipped_column(metrics)].copy()
    metrics["auroc"] = pd.to_numeric(metrics.get("auroc"), errors="coerce")
    metrics["auprc"] = pd.to_numeric(metrics.get("auprc"), errors="coerce")
    if metrics.empty:
        return predictions
    row = metrics.sort_values(["auroc", "auprc"], ascending=False).iloc[0]
    out = predictions.copy()
    for col in ["audit_model", "feature_strategy", "model_name", "split"]:
        if col in out.columns and col in row.index and pd.notna(row[col]):
            out = out[out[col].astype(str).eq(str(row[col]))].copy()
    return out if not out.empty else predictions


def _filter_predictions_by_row(predictions: pd.DataFrame, row: pd.Series | None) -> pd.DataFrame:
    if row is None or predictions.empty:
        return pd.DataFrame()
    out = predictions.copy()
    for col in SELECTOR_COLUMNS:
        if col in out.columns and col in row.index and pd.notna(row[col]):
            out = out[out[col].astype(str).eq(str(row[col]))].copy()
    return out


def main() -> None:
    args = parse_args()
    final_predictions = _read_optional_prediction_files(args.final_predictions)
    metadata_predictions = _read_optional_prediction_files(args.metadata_predictions)

    if final_predictions.empty:
        checked = ", ".join(str(path) for path in args.final_predictions)
        raise FileNotFoundError(f"No usable final prediction files found. Checked: {checked}")

    final_shuffle = build_shuffle_label_sanity(
        final_predictions,
        group_columns=[col for col in DEFAULT_FINAL_GROUP_COLUMNS if col in final_predictions.columns],
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )
    _write_csv(final_shuffle, args.final_shuffle_output, "final prediction shuffle-label sanity")

    fixed_sensitivity = build_fixed_sensitivity_table(
        final_predictions,
        group_columns=[col for col in DEFAULT_FINAL_GROUP_COLUMNS if col in final_predictions.columns],
        target_sensitivities=args.target_sensitivities,
    )
    _write_csv(fixed_sensitivity, args.fixed_sensitivity_output, "fixed-sensitivity operating points")

    external = final_predictions[final_predictions.get("split", pd.Series(index=final_predictions.index, dtype=object)).astype(str).eq("external_test")].copy()
    if external.empty:
        external = final_predictions[
            final_predictions.get("metric_split", pd.Series(index=final_predictions.index, dtype=object)).astype(str).eq("external_test")
        ].copy()
    if external.empty:
        recalibration_metrics = pd.DataFrame()
        recalibration_predictions = pd.DataFrame()
    else:
        recalibration_metrics, recalibration_predictions = build_partial_target_recalibration(
            external,
            group_columns=[
                col
                for col in [
                    "prediction_source",
                    "evaluation_protocol",
                    "analysis_family",
                    "model_name",
                    "modality",
                    "feature_strategy",
                    "selected_feature_k",
                    "split",
                ]
                if col in external.columns
            ],
            calibration_fraction=args.recalibration_fraction,
            random_state=args.random_state,
        )
    _write_csv(recalibration_metrics, args.recalibration_metrics_output, "COUGHVID partial recalibration metrics")
    _write_csv(recalibration_predictions, args.recalibration_predictions_output, "COUGHVID partial recalibration predictions")

    if metadata_predictions.empty:
        metadata_shuffle = pd.DataFrame()
        residual = pd.DataFrame()
    else:
        metadata_shuffle = build_shuffle_label_sanity(
            metadata_predictions,
            group_columns=[col for col in ["prediction_source", "audit_model", "feature_strategy", "model_name", "split"] if col in metadata_predictions.columns],
            n_permutations=args.n_permutations,
            random_state=args.random_state,
        )
        audio_selected = _filter_predictions_by_row(final_predictions, _best_existing_summary_row(args.final_summary))
        metadata_selected = _best_metadata_predictions(metadata_predictions, args.metadata_metrics)
        residual = build_audio_metadata_residual_correlation(
            audio_selected,
            metadata_selected,
            group_columns=[],
        )
        if not residual.empty:
            residual["audio_selection"] = "best_existing_final_validation_row"
            residual["metadata_selection"] = "best_metadata_confounding_row"
    _write_csv(metadata_shuffle, args.metadata_shuffle_output, "metadata-only shuffle-label sanity")
    _write_csv(residual, args.residual_correlation_output, "audio-metadata residual correlation")

    if args.metadata.exists():
        metadata = pd.read_csv(args.metadata)
        metadata_shuffle_retrain = run_metadata_shuffle_retrain_sanity(
            metadata,
            feature_sets=args.metadata_feature_sets,
            n_permutations=args.metadata_retrain_permutations,
            random_state=args.random_state,
        )
    else:
        metadata_shuffle_retrain = pd.DataFrame()
    _write_csv(metadata_shuffle_retrain, args.metadata_shuffle_retrain_output, "metadata retrain shuffle-label sanity")


if __name__ == "__main__":
    main()
