#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.delta_bootstrap import build_auto_reviewer_comparisons
from covid_rars.final_uncertainty import load_final_prediction_files
from covid_rars.compare_is10_rescue import merge_feature_tables
from covid_rars.reporting import read_existing_csvs
from covid_rars.reviewer_extension_checks import (
    DEFAULT_SELECTOR_COLUMNS,
    build_context_control_exposure_table,
    build_decision_curve_table,
    build_duration_shortcut_table,
    build_feature_selection_stability,
    build_label_construction_audit_table,
    build_matched_cohort_quality_table,
    build_metadata_predictions_for_nested_comparison,
    build_nested_metadata_audio_comparison,
    build_paired_delong_table,
    build_performance_equity_table,
    build_quality_label_month_table,
    build_specification_curve,
    build_support_overlap_diagnostic,
    write_label_construction_note,
    write_multiplicity_scope_note,
)


DEFAULT_METRIC_PATHS = [
    Path("data/outputs/metrics/strong_baseline_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_final_validation_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_external_transfer_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_reverse_temporal_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_multiseed_metrics.csv"),
    Path("data/outputs/metrics/compare_is10_shuffle_retrain_metrics.csv"),
    Path("data/outputs/metrics/metadata_confounding_metrics.csv"),
    Path("data/outputs/metrics/metadata_permutation_importance_metrics.csv"),
    Path("data/outputs/metrics/sota_ssl_metrics_practical_wavlm_cough.csv"),
    Path("data/outputs/metrics/deep_external_wavlm_cough_metrics.csv"),
    Path("data/outputs/metrics/deep_external_cnn_bigru_cough_metrics.csv"),
]
FINAL_PREDICTION_DEFAULTS = [
    Path("data/outputs/metrics/compare_is10_final_validation_predictions.csv"),
    Path("data/outputs/metrics/compare_is10_external_transfer_predictions.csv"),
]
METADATA_PREDICTION_DEFAULTS = [
    Path("data/outputs/metrics/metadata_confounding_predictions.csv"),
    Path("data/outputs/metrics/metadata_permutation_importance_predictions.csv"),
]
EXTERNAL_FEATURE_CANDIDATES = [
    Path("data/processed/features_compare_is10_coughvid_top800.csv"),
    Path("data/processed/features_compare_is10_coughvid_cough_top800.csv"),
    Path("data/processed/features_compare_is10_coughvid_merged.csv"),
    Path("data/processed/features_compare_is10_external_top800.csv"),
]
EXTERNAL_FEATURE_COMPONENTS = {
    "strong": Path("data/processed/features_strong_acoustic_coughvid_cough.csv"),
    "compare2016": Path("data/processed/features_opensmile_compare2016_coughvid_cough.csv"),
    "is10": Path("data/processed/features_opensmile_is10_coughvid_cough.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reviewer-requested extension checks: nested metadata+audio, support overlap, DCA, equity, quality, and methods notes."
    )
    parser.add_argument("--metrics", nargs="*", type=Path, default=None)
    parser.add_argument("--final-predictions", nargs="*", type=Path, default=None)
    parser.add_argument("--metadata-predictions", nargs="*", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--external-metadata", type=Path, default=Path("data/processed/coughvid_metadata_compare_is10_external.csv"))
    parser.add_argument("--final-summary", type=Path, default=Path("reports/tables/compare_is10_final_validation_summary.csv"))
    parser.add_argument("--final-metrics", type=Path, default=Path("data/outputs/metrics/compare_is10_final_validation_metrics.csv"))
    parser.add_argument("--external-metrics", type=Path, default=Path("data/outputs/metrics/compare_is10_external_transfer_metrics.csv"))
    parser.add_argument("--source-features", type=Path, default=Path("data/processed/features_compare_is10_merged.csv"))
    parser.add_argument("--external-features", type=Path, default=None)
    parser.add_argument("--feature-stability-top-k", type=int, default=800)
    parser.add_argument("--feature-stability-ranker", default="lightgbm", choices=["univariate", "extra_trees", "lightgbm", "auto"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-subgroup-size", type=int, default=20)
    parser.add_argument("--decision-thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--final-dir", type=Path, default=Path("reports/final"))
    parser.add_argument(
        "--nested-predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/reviewer_nested_metadata_audio_predictions.csv"),
    )
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {label}: {path} ({len(frame)} rows)")


def _read_existing(paths: list[Path]) -> pd.DataFrame:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return pd.DataFrame()
    return read_existing_csvs(existing)


def _read_predictions(paths: list[Path]) -> pd.DataFrame:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return pd.DataFrame()
    return load_final_prediction_files(existing)


def _is_skipped(frame: pd.DataFrame) -> pd.Series:
    if "skipped" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["skipped"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _best_existing_summary_row(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    work = frame.copy()
    if "evaluation_protocol" in work.columns:
        work = work[work["evaluation_protocol"].astype(str).eq("compare_is10_existing_participant_split")].copy()
    work = work[~_is_skipped(work)].copy()
    if work.empty or "auroc" not in work.columns:
        return None
    work["auroc"] = pd.to_numeric(work["auroc"], errors="coerce")
    work["auprc"] = pd.to_numeric(work.get("auprc"), errors="coerce")
    return work.sort_values(["auroc", "auprc"], ascending=False).iloc[0] if not work.empty else None


def _filter_predictions_by_row(predictions: pd.DataFrame, row: pd.Series | None) -> pd.DataFrame:
    if predictions.empty or row is None:
        return pd.DataFrame()
    out = predictions.copy()
    for col in DEFAULT_SELECTOR_COLUMNS:
        if col in out.columns and col in row.index and pd.notna(row[col]):
            out = out[out[col].astype(str).eq(str(row[col]))].copy()
    return out


def _features_with_dates(features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if features.empty or "recording_date" in features.columns or metadata.empty:
        return features
    date_cols = [col for col in ["recording_id", "participant_id", "recording_date"] if col in metadata.columns]
    if "recording_date" not in date_cols:
        return features
    right = metadata[date_cols].drop_duplicates(date_cols[0], keep="first") if "recording_id" in date_cols else metadata[date_cols].drop_duplicates("participant_id")
    if "recording_id" in features.columns and "recording_id" in right.columns:
        return features.merge(right, on="recording_id", how="left")
    if "participant_id" in features.columns and "participant_id" in right.columns:
        return features.merge(right, on="participant_id", how="left")
    return features


def _selected_external_feature_path(path: Path | None) -> Path | None:
    if path is not None:
        return path if path.exists() else None
    for candidate in EXTERNAL_FEATURE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _combined_context_metadata(metadata: pd.DataFrame, external_metadata: pd.DataFrame) -> pd.DataFrame:
    frames = [frame.copy() for frame in [metadata, external_metadata] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _read_external_features(path: Path | None) -> pd.DataFrame:
    selected = _selected_external_feature_path(path)
    if selected is not None:
        return pd.read_csv(selected)
    available = {
        name: pd.read_csv(component_path)
        for name, component_path in EXTERNAL_FEATURE_COMPONENTS.items()
        if component_path.exists()
    }
    if len(available) >= 2:
        return merge_feature_tables(available)
    return pd.DataFrame()


def _restrict_to_common_modalities(source_features: pd.DataFrame, external_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "modality" not in source_features.columns or "modality" not in external_features.columns:
        return source_features, external_features
    source_modalities = set(source_features["modality"].dropna().astype(str))
    external_modalities = set(external_features["modality"].dropna().astype(str))
    common = source_modalities & external_modalities
    if not common:
        return source_features, external_features
    return (
        source_features[source_features["modality"].astype(str).isin(common)].copy(),
        external_features[external_features["modality"].astype(str).isin(common)].copy(),
    )


def _save_specification_curve_figure(curve: pd.DataFrame, output: Path) -> None:
    if curve.empty or "auroc" not in curve.columns:
        return
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    work = curve.copy()
    work["auroc"] = pd.to_numeric(work["auroc"], errors="coerce")
    work = work[np.isfinite(work["auroc"])].sort_values("auroc", ascending=True).tail(160)
    plt.figure(figsize=(9.0, 5.2))
    plt.scatter(range(len(work)), work["auroc"], s=14, alpha=0.75, color="#275a8a")
    plt.ylabel("AUROC")
    plt.xlabel("Model specification rank")
    plt.title("Specification curve across evaluated model rows")
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(output, format="svg")
    plt.close()


def _save_decision_curve_figure(curve: pd.DataFrame, output: Path) -> None:
    if curve.empty or "threshold_probability" not in curve.columns:
        return
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    groups = [col for col in ["evaluation_protocol", "model_name", "modality", "fusion_method", "split"] if col in curve.columns]
    plt.figure(figsize=(9.0, 5.2))
    plotted = 0
    for key, group in curve.groupby(groups, dropna=False) if groups else [(("model",), curve)]:
        if plotted >= 8:
            break
        label = " | ".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))[:110]
        group = group.sort_values("threshold_probability")
        plt.plot(group["threshold_probability"], group["model_net_benefit"], linewidth=1.4, label=label)
        plotted += 1
    ref = curve.sort_values("threshold_probability").drop_duplicates("threshold_probability")
    plt.plot(ref["threshold_probability"], ref["treat_all_net_benefit"], linestyle="--", color="#777777", label="Treat all")
    plt.axhline(0.0, linestyle=":", color="#222222", label="Treat none")
    plt.ylabel("Net benefit")
    plt.xlabel("Threshold probability")
    plt.title("Decision curve analysis")
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output, format="svg")
    plt.close()


def main() -> None:
    args = parse_args()
    metric_paths = args.metrics if args.metrics is not None else DEFAULT_METRIC_PATHS
    final_paths = args.final_predictions if args.final_predictions is not None else FINAL_PREDICTION_DEFAULTS
    metadata_prediction_paths = args.metadata_predictions if args.metadata_predictions is not None else METADATA_PREDICTION_DEFAULTS
    metadata = pd.read_csv(args.metadata) if args.metadata.exists() else pd.DataFrame()
    external_metadata = pd.read_csv(args.external_metadata) if args.external_metadata.exists() else pd.DataFrame()
    context_metadata = _combined_context_metadata(metadata, external_metadata)
    metrics = _read_existing(metric_paths)
    final_predictions = _read_predictions(final_paths)
    metadata_predictions = _read_predictions(metadata_prediction_paths)

    specification = build_specification_curve(metrics) if not metrics.empty else pd.DataFrame()
    _write_csv(specification, args.output_dir / "reviewer_specification_curve.csv", "reviewer specification curve")
    _save_specification_curve_figure(specification, args.figure_dir / "reviewer_specification_curve.svg")

    quality = build_quality_label_month_table(context_metadata) if not context_metadata.empty else pd.DataFrame()
    _write_csv(quality, args.output_dir / "reviewer_quality_label_month_correlation.csv", "quality-label/month table")

    if final_predictions.empty:
        final_predictions = pd.DataFrame()
    decision = build_decision_curve_table(
        final_predictions,
        group_columns=[col for col in DEFAULT_SELECTOR_COLUMNS if col in final_predictions.columns],
        thresholds=args.decision_thresholds,
    ) if not final_predictions.empty else pd.DataFrame()
    _write_csv(decision, args.output_dir / "reviewer_decision_curve_analysis.csv", "decision curve analysis")
    _save_decision_curve_figure(decision, args.figure_dir / "reviewer_decision_curve_analysis.svg")

    duration = build_duration_shortcut_table(
        final_predictions,
        context_metadata,
        group_columns=[col for col in DEFAULT_SELECTOR_COLUMNS if col in final_predictions.columns],
    ) if not final_predictions.empty and not context_metadata.empty else pd.DataFrame()
    _write_csv(duration, args.output_dir / "reviewer_duration_shortcut_correlation.csv", "duration shortcut correlation")

    equity = build_performance_equity_table(
        final_predictions,
        context_metadata,
        subgroup_columns=["age_band", "gender", "country", "recording_year_month"],
        group_columns=[col for col in DEFAULT_SELECTOR_COLUMNS if col in final_predictions.columns],
        min_subgroup_size=args.min_subgroup_size,
    ) if not final_predictions.empty and not context_metadata.empty else pd.DataFrame()
    _write_csv(equity, args.output_dir / "reviewer_performance_equity_subgroups.csv", "performance-equity subgroup table")

    selected_audio = _filter_predictions_by_row(final_predictions, _best_existing_summary_row(args.final_summary))
    nested_metadata = metadata_predictions
    if nested_metadata.empty or "validation" not in set(nested_metadata.get("split", pd.Series(dtype=object)).astype(str)):
        nested_metadata = build_metadata_predictions_for_nested_comparison(metadata, random_state=args.random_state) if not metadata.empty else pd.DataFrame()
    nested_metrics, nested_predictions = build_nested_metadata_audio_comparison(
        selected_audio,
        nested_metadata,
        random_state=args.random_state,
    )
    _write_csv(nested_metrics, args.output_dir / "reviewer_nested_metadata_audio_comparison.csv", "nested metadata+audio comparison")
    _write_csv(nested_predictions, args.nested_predictions_output, "nested metadata+audio predictions")

    if args.source_features.exists():
        source_features = pd.read_csv(args.source_features)
        source_features = _features_with_dates(source_features, metadata)
    else:
        source_features = pd.DataFrame()
    external_features = _read_external_features(args.external_features)
    if not source_features.empty and not external_features.empty:
        source_features_support, external_features_support = _restrict_to_common_modalities(source_features, external_features)
    else:
        source_features_support, external_features_support = source_features, external_features
    support = build_support_overlap_diagnostic(source_features_support, external_features_support) if not source_features_support.empty and not external_features_support.empty else pd.DataFrame(
        [{"analysis": "support_overlap_positivity", "skipped": True, "skip_reason": "source or external feature table not available"}]
    )
    _write_csv(support, args.output_dir / "reviewer_support_overlap_positivity.csv", "support-overlap positivity diagnostic")

    stability = build_feature_selection_stability(
        source_features,
        top_k=args.feature_stability_top_k,
        ranker=args.feature_stability_ranker,
        random_state=args.random_state,
    ) if not source_features.empty else pd.DataFrame(
        [{"analysis": "feature_selection_stability", "skipped": True, "skip_reason": "source feature table not available"}]
    )
    _write_csv(stability, args.output_dir / "reviewer_feature_selection_stability.csv", "feature-selection stability")

    matched_quality = build_matched_cohort_quality_table(metadata) if not metadata.empty else pd.DataFrame()
    _write_csv(matched_quality, args.output_dir / "reviewer_matched_cohort_quality.csv", "matched-cohort quality table")

    context_control = build_context_control_exposure_table(metadata, random_state=args.random_state) if not metadata.empty else pd.DataFrame()
    _write_csv(context_control, args.output_dir / "reviewer_context_control_exposure.csv", "context-control exposure table")

    delong_rows: list[pd.DataFrame] = []
    if args.final_summary.exists() and args.final_metrics.exists() and args.external_metrics.exists() and not final_predictions.empty:
        final_summary = pd.read_csv(args.final_summary)
        final_metrics = pd.read_csv(args.final_metrics)
        external_metrics = pd.read_csv(args.external_metrics)
        comparisons = build_auto_reviewer_comparisons(final_summary, final_metrics, external_metrics)
        for comparison in comparisons:
            delong_rows.append(
                build_paired_delong_table(
                    final_predictions,
                    comparison.left_selector,
                    comparison.right_selector,
                    comparison_id=comparison.comparison_id,
                    paired_on="participant_id",
                )
            )
    delong = pd.concat(delong_rows, ignore_index=True, sort=False) if delong_rows else pd.DataFrame()
    _write_csv(delong, args.output_dir / "reviewer_paired_delong_auc_comparisons.csv", "paired DeLong AUROC comparisons")

    label_audit = build_label_construction_audit_table(metadata, external_metadata)
    _write_csv(label_audit, args.output_dir / "reviewer_label_construction_audit.csv", "label construction audit table")
    write_label_construction_note(args.final_dir / "LABEL_CONSTRUCTION_AUDIT.md", dataset_notes=label_audit)
    print(f"Wrote label construction audit: {args.final_dir / 'LABEL_CONSTRUCTION_AUDIT.md'}")
    write_multiplicity_scope_note(args.final_dir / "MULTIPLICITY_AND_ANALYSIS_SCOPE.md")
    print(f"Wrote multiplicity/scope note: {args.final_dir / 'MULTIPLICITY_AND_ANALYSIS_SCOPE.md'}")


if __name__ == "__main__":
    main()
