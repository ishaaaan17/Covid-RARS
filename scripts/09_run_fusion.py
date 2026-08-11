#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.fusion import uniform_fusion, validation_weighted_fusion
from covid_rars.metrics import best_threshold_by_balanced_accuracy, evaluate_predictions, labels_to_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibrated multimodal fusion.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--validation-metrics", required=True, type=Path)
    parser.add_argument("--validation-predictions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/metrics/fusion_predictions.csv"))
    parser.add_argument("--metrics-output", type=Path, default=Path("data/outputs/metrics/fusion_metrics.csv"))
    parser.add_argument("--thresholds-output", type=Path, default=Path("data/outputs/metrics/fusion_thresholds.csv"))
    return parser.parse_args()



def _read_validation_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    if path.name == "ml_baseline_metrics.csv" and "metric_split" not in metrics.columns:
        raise ValueError(
            "Refusing to use ml_baseline_metrics.csv as validation metrics because it is test-set output. "
            "Use data/outputs/metrics/ml_validation_metrics.csv for fusion weights."
        )
    if "metric_split" in metrics.columns:
        splits = set(metrics["metric_split"].dropna().astype(str))
        if splits and splits != {"validation"}:
            raise ValueError(f"Validation metrics file contains non-validation metric_split values: {sorted(splits)}")
    return metrics

def _run_fusion(predictions: pd.DataFrame, validation_metrics: pd.DataFrame) -> pd.DataFrame:
    frames = [
        uniform_fusion(predictions),
        validation_weighted_fusion(predictions, validation_metrics, metric_column="auprc"),
    ]
    return pd.concat(frames, ignore_index=True)


def _resolve_validation_predictions(args: argparse.Namespace) -> Path | None:
    if args.validation_predictions is not None:
        return args.validation_predictions if args.validation_predictions.exists() else None
    candidate = args.predictions.with_name("calibrated_branch_predictions_validation.csv")
    return candidate if candidate.exists() else None


def _thresholds_from_validation(validation_fused: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in validation_fused.groupby("fusion_method", dropna=False):
        if group.empty or not group["label_binary"].isin(["positive", "negative"]).all():
            threshold = 0.5
        else:
            threshold = best_threshold_by_balanced_accuracy(
                labels_to_binary(group["label_binary"]),
                group["probability"].astype(float).to_numpy(),
            )
        rows.append({"fusion_method": method, "threshold": threshold, "threshold_source": "validation"})
    return pd.DataFrame(rows)


def _evaluate_fused(fused: pd.DataFrame, thresholds: pd.DataFrame | None = None) -> pd.DataFrame:
    metric_frames = []
    threshold_map = {}
    if thresholds is not None and not thresholds.empty:
        threshold_map = thresholds.set_index("fusion_method")["threshold"].to_dict()
    for method, group in fused.groupby("fusion_method", dropna=False):
        threshold = float(threshold_map.get(method, 0.5))
        metrics = evaluate_predictions(group, threshold=threshold)
        metrics["fusion_method"] = method
        metric_frames.append(metrics)
    return pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    validation_metrics = _read_validation_metrics(args.validation_metrics)
    fused = _run_fusion(predictions, validation_metrics)

    validation_predictions_path = _resolve_validation_predictions(args)
    thresholds = None
    if validation_predictions_path is not None:
        validation_predictions = pd.read_csv(validation_predictions_path)
        validation_fused = _run_fusion(validation_predictions, validation_metrics)
        thresholds = _thresholds_from_validation(validation_fused)

    metrics = _evaluate_fused(fused, thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(args.output, index=False)
    metrics.to_csv(args.metrics_output, index=False)
    if thresholds is not None:
        thresholds.to_csv(args.thresholds_output, index=False)
        print(f"Wrote fusion thresholds: {args.thresholds_output}")
    print(f"Wrote fusion predictions: {args.output}")
    print(f"Wrote fusion metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
