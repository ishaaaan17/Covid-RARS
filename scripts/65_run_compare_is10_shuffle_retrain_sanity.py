#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.compare_is10_final_validation import run_compare_is10_final_validation
from covid_rars.shuffle_retrain_sanity import shuffle_labels_by_participant
from covid_rars.strong_baseline import DEFAULT_MODALITIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain the ComParE+IS10 final-validation ladder after participant-level label shuffling."
    )
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_compare_is10_top800.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--feature-strategy", default="compare_is10_top800_lightgbm")
    parser.add_argument("--selected-feature-k", type=int, default=800)
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["lightgbm_smote_f80", "svc_rbf_f60", "catboost_smote_f80", "xgboost_smote_f80"],
    )
    parser.add_argument("--n-permutations", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--ensemble-top-k", type=int, default=5)
    parser.add_argument("--enable-feature-level-fusion", action="store_true")
    parser.add_argument("--global-stack-top-k", type=int, default=0)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/compare_is10_shuffle_retrain_metrics.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/tables/compare_is10_shuffle_retrain_summary.csv"),
    )
    parser.add_argument("--predictions-output", type=Path, default=None)
    return parser.parse_args()


def _participant_label_map(metadata: pd.DataFrame) -> dict[str, str]:
    labeled = metadata[metadata["label_binary"].isin(["positive", "negative"])].copy()
    labels = (
        labeled.groupby("participant_id", dropna=False)["label_binary"]
        .agg(lambda values: str(values.value_counts().index[0]))
        .reset_index()
    )
    return dict(zip(labels["participant_id"].astype(str), labels["label_binary"].astype(str)))


def _apply_participant_labels(features: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    out = features.copy()
    mask = out["participant_id"].astype(str).isin(label_map)
    out.loc[mask, "label_binary"] = out.loc[mask, "participant_id"].astype(str).map(label_map)
    return out


def _write_csv(frame: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {label}: {path} ({len(frame)} rows)")


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    metadata = pd.read_csv(args.metadata)
    rng = np.random.default_rng(args.random_state)
    metric_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for iteration in range(max(1, int(args.n_permutations))):
        shuffled_metadata = shuffle_labels_by_participant(metadata, rng)
        shuffled_features = _apply_participant_labels(features, _participant_label_map(shuffled_metadata))
        result = run_compare_is10_final_validation(
            features=shuffled_features,
            metadata=shuffled_metadata,
            feature_strategy=args.feature_strategy,
            selected_feature_k=args.selected_feature_k,
            modalities=args.modalities,
            model_names=args.model_names,
            random_state=args.random_state + iteration + 1,
            optuna_trials=args.optuna_trials,
            ensemble_top_k=args.ensemble_top_k,
            enable_feature_level_fusion=args.enable_feature_level_fusion,
            global_stack_top_k=args.global_stack_top_k,
        )
        for frame in [result.metrics, result.final_summary, result.predictions]:
            if not frame.empty:
                frame["shuffle_iteration"] = int(iteration)
                frame["sanity_check"] = "compare_is10_retrain_with_shuffled_labels"
        metric_frames.append(result.metrics)
        summary_frames.append(result.final_summary)
        if args.predictions_output is not None:
            prediction_frames.append(result.predictions)

    metrics = pd.concat(metric_frames, ignore_index=True, sort=False) if metric_frames else pd.DataFrame()
    summary = pd.concat(summary_frames, ignore_index=True, sort=False) if summary_frames else pd.DataFrame()
    _write_csv(metrics, args.metrics_output, "ComParE+IS10 shuffle-retrain metrics")
    _write_csv(summary, args.summary_output, "ComParE+IS10 shuffle-retrain summary")
    if args.predictions_output is not None:
        predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
        _write_csv(predictions, args.predictions_output, "ComParE+IS10 shuffle-retrain predictions")


if __name__ == "__main__":
    main()
