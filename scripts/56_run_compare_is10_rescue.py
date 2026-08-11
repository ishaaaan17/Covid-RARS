#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.compare_is10_rescue import (
    merge_feature_tables,
    read_feature_table,
    select_top_k_feature_tables,
)
from covid_rars.opensmile_features import extract_opensmile_feature_csv
from covid_rars.representation_features import validate_feature_table
from covid_rars.strong_baseline import DEFAULT_MODEL_NAMES, run_strong_baseline


METRIC_COLUMNS = [
    "evaluation_protocol",
    "analysis_family",
    "model_name",
    "modality",
    "modality_combination",
    "fusion_method",
    "metric_split",
    "feature_strategy",
    "selected_feature_k",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "f1",
    "threshold",
    "n_participants",
]

PREDICTION_COLUMNS = [
    "recording_id",
    "participant_id",
    "dataset",
    "modality",
    "submodality",
    "label_binary",
    "split",
    "model_name",
    "analysis_family",
    "evaluation_protocol",
    "modality_combination",
    "fusion_method",
    "feature_strategy",
    "selected_feature_k",
    "probability",
]

MODEL_SELECTION_COLUMNS = [
    "evaluation_protocol",
    "modality",
    "selected_model_name",
    "selection_metric",
    "validation_auroc",
    "validation_auprc",
    "validation_balanced_accuracy",
    "threshold",
    "feature_strategy",
    "selected_feature_k",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the bounded ComParE+IS10 rescue branch: extract missing OpenSMILE descriptors, "
            "merge with strong acoustic features, select train-only top-k features, and run the strong baseline."
        )
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--strong-features", type=Path, default=Path("data/processed/features_strong_acoustic.csv"))
    parser.add_argument("--compare-features", type=Path, default=Path("data/processed/features_opensmile_compare2016.csv"))
    parser.add_argument("--is10-features", type=Path, default=Path("data/processed/features_opensmile_is10.csv"))
    parser.add_argument("--no-extract-missing", action="store_true")
    parser.add_argument("--quality-mode", default="quality_ok_only", choices=["all_samples", "quality_ok_only"])
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--extract-chunk-size", type=int, default=64)
    parser.add_argument("--modalities", nargs="+", default=["cough", "breath", "speech"])
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[500, 800, 1200])
    parser.add_argument("--ranker", default="lightgbm", choices=["lightgbm", "extra_trees", "univariate", "auto"])
    parser.add_argument("--selection-scope", default="per_modality_mean", choices=["global", "per_modality_mean"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-names", nargs="+", default=list(DEFAULT_MODEL_NAMES))
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--ensemble-top-k", type=int, default=5)
    parser.add_argument("--skip-feature-level-fusion", action="store_true")
    parser.add_argument("--global-stack-top-k", type=int, default=10)
    quality = parser.add_mutually_exclusive_group()
    quality.add_argument("--require-quality-ok", dest="require_quality_ok", action="store_true", default=True)
    quality.add_argument("--include-non-ok-quality", dest="require_quality_ok", action="store_false")
    parser.add_argument("--merged-output", type=Path, default=Path("data/processed/features_compare_is10_merged.csv"))
    parser.add_argument(
        "--selected-feature-output-template",
        default="data/processed/features_compare_is10_top{k}.csv",
    )
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/sota_compare_is10_metrics.csv"))
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/sota_compare_is10_predictions.csv"),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path("reports/tables/sota_compare_is10_model_selection.csv"),
    )
    parser.add_argument(
        "--importance-output",
        type=Path,
        default=Path("reports/tables/sota_compare_is10_feature_importance.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/tables/sota_compare_is10_feature_selection_summary.csv"),
    )
    return parser.parse_args()


def _extract_if_missing(
    path: Path,
    metadata: pd.DataFrame,
    *,
    feature_set: str,
    quality_mode: str,
    progress_interval: int,
    chunk_size: int,
) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    n_written = extract_opensmile_feature_csv(
        metadata,
        path,
        feature_set=feature_set,
        quality_mode=quality_mode,
        strict=False,
        progress_interval=progress_interval,
        chunk_size=chunk_size,
    )
    features = pd.read_csv(path)
    validate_feature_table(features)
    print(f"Wrote OpenSMILE {feature_set} features: {path} ({n_written} rows, {len(features.columns)} columns)")
    return features


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty and len(frame.columns) == 0:
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.Series(dtype=object)
    return out


def _feature_strategy(k: int, ranker: str) -> str:
    return f"compare_is10_top{k}_{ranker}"


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    strong = read_feature_table(args.strong_features)

    if args.no_extract_missing:
        compare = read_feature_table(args.compare_features)
        is10 = read_feature_table(args.is10_features)
    else:
        compare = _extract_if_missing(
            args.compare_features,
            metadata,
            feature_set="compare2016",
            quality_mode=args.quality_mode,
            progress_interval=args.progress_interval,
            chunk_size=args.extract_chunk_size,
        )
        is10 = _extract_if_missing(
            args.is10_features,
            metadata,
            feature_set="is10",
            quality_mode=args.quality_mode,
            progress_interval=args.progress_interval,
            chunk_size=args.extract_chunk_size,
        )

    merged = merge_feature_tables(
        {
            "strong": strong,
            "compare2016": compare,
            "is10": is10,
        }
    )
    _write_csv(merged, args.merged_output)
    print(f"Wrote merged ComParE+IS10 feature table: {args.merged_output} ({len(merged)} rows, {len(merged.columns)} columns)")

    selected = select_top_k_feature_tables(
        merged,
        k_values=args.top_k_values,
        ranker=args.ranker,
        selection_scope=args.selection_scope,
        random_state=args.random_state,
    )
    _write_csv(selected.importance, args.importance_output)
    _write_csv(selected.summary, args.summary_output)

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    for k, table in selected.tables.items():
        feature_path = Path(args.selected_feature_output_template.format(k=k))
        _write_csv(table, feature_path)
        print(f"Wrote selected feature table k={k}: {feature_path} ({len(table)} rows, {len(table.columns)} columns)")

        result = run_strong_baseline(
            features=table,
            metadata=metadata,
            modalities=args.modalities,
            model_names=args.model_names,
            require_quality_ok=args.require_quality_ok,
            random_state=args.random_state,
            optuna_trials=args.optuna_trials,
            ensemble_top_k=args.ensemble_top_k,
            enable_feature_level_fusion=not args.skip_feature_level_fusion,
            global_stack_top_k=args.global_stack_top_k,
        )
        strategy = _feature_strategy(k, args.ranker)
        if not result.metrics.empty:
            metrics = result.metrics.copy()
            metrics["feature_strategy"] = strategy
            metrics["selected_feature_k"] = float(k)
            metric_frames.append(metrics)
        if not result.predictions.empty:
            predictions = result.predictions.copy()
            predictions["feature_strategy"] = strategy
            predictions["selected_feature_k"] = float(k)
            prediction_frames.append(predictions)
        if not result.selection.empty:
            model_selection = result.selection.copy()
            model_selection["feature_strategy"] = strategy
            model_selection["selected_feature_k"] = float(k)
            selection_frames.append(model_selection)

    metrics_out = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    predictions_out = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    selection_out = pd.concat(selection_frames, ignore_index=True, sort=False) if selection_frames else pd.DataFrame()
    metrics_out = _ensure_columns(metrics_out, METRIC_COLUMNS)
    predictions_out = _ensure_columns(predictions_out, PREDICTION_COLUMNS)
    selection_out = _ensure_columns(selection_out, MODEL_SELECTION_COLUMNS)

    _write_csv(metrics_out, args.metrics_output)
    _write_csv(predictions_out, args.predictions_output)
    _write_csv(selection_out, args.selection_output)
    print(f"Wrote ComParE+IS10 metrics: {args.metrics_output} ({len(metrics_out)} rows)")
    print(f"Wrote ComParE+IS10 predictions: {args.predictions_output} ({len(predictions_out)} rows)")
    print(f"Wrote ComParE+IS10 model selection: {args.selection_output} ({len(selection_out)} rows)")


if __name__ == "__main__":
    main()
