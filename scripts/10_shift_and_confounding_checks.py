#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.metrics import evaluate_predictions


QUALITY_SEVERITY = {
    "ok": 0,
    "good": 0,
    "not_audited": 0,
    "unknown": 1,
    "": 1,
    "uncertain": 2,
    "low_quality": 2,
    "short": 3,
    "mostly_silence": 3,
    "clipped": 3,
    "bad": 3,
    "corrupt": 4,
    "missing": 4,
    "unreadable": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quality/subgroup/confounding metric tables.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/metrics/subgroup_metrics.csv"))
    return parser.parse_args()


def _age_bucket(value: object) -> str:
    try:
        age = float(value)
    except Exception:
        return "unknown"
    if age < 30:
        return "<30"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def _first_non_empty(values: pd.Series) -> object:
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[~cleaned.str.lower().isin({"", "nan", "none", "unknown"})]
    if cleaned.empty:
        return "unknown"
    modes = cleaned.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else cleaned.iloc[0]


def _worst_quality_flag(values: pd.Series) -> str:
    cleaned = values.dropna().astype(str).str.strip().str.lower()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return "unknown"
    return max(cleaned, key=lambda flag: QUALITY_SEVERITY.get(flag, 2))


def merge_predictions_with_metadata(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    cols = ["recording_id", "participant_id", "quality_flag", "age", "gender", "country"]
    available_cols = [c for c in cols if c in metadata.columns]
    if "recording_id" in predictions.columns and "recording_id" in metadata.columns:
        right = metadata[available_cols].drop_duplicates("recording_id", keep="first")
        return predictions.merge(right, on="recording_id", how="left")

    if "participant_id" not in predictions.columns or "participant_id" not in metadata.columns:
        return predictions.copy()

    aggregations = {}
    if "quality_flag" in metadata.columns:
        aggregations["quality_flag"] = _worst_quality_flag
    for column in ["age", "gender", "country"]:
        if column in metadata.columns:
            aggregations[column] = _first_non_empty

    if aggregations:
        right = metadata.groupby("participant_id", as_index=False).agg(aggregations)
    else:
        right = metadata[["participant_id"]].drop_duplicates("participant_id")
    return predictions.merge(right, on="participant_id", how="left")


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    metadata = pd.read_csv(args.metadata)
    merged = merge_predictions_with_metadata(predictions, metadata)
    if "age" in merged.columns:
        merged["age_bucket"] = merged["age"].map(_age_bucket)
    group_columns = [c for c in ["quality_flag", "gender", "age_bucket"] if c in merged.columns]
    frames = []
    for group_col in group_columns:
        metrics = evaluate_predictions(merged.dropna(subset=[group_col]), group_columns=[group_col])
        metrics["analysis_group"] = group_col
        frames.append(metrics)
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote subgroup metrics: {args.output}")


if __name__ == "__main__":
    main()
