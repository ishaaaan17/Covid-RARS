#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.prevalence_recalibration import build_prevalence_recalibration_report


DEFAULT_GROUP_COLUMNS = ["prediction_source", "model_name", "modality", "feature_strategy", "dataset", "calibration_method", "split"]


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def default_prediction_paths(metrics_dir: Path = Path("data/outputs/metrics")) -> list[Path]:
    base = [
        metrics_dir / "cross_dataset_predictions.csv",
        metrics_dir / "external_model_grid_predictions.csv",
    ]
    discovered = sorted(metrics_dir.glob("external_model_grid_*_predictions.csv"))
    return _dedupe_paths([*base, *discovered])


def _read_prediction_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    required = {"label_binary", "probability"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    frame["prediction_source"] = path.stem
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply target-prevalence intercept recalibration to external prediction files.")
    parser.add_argument("--predictions", nargs="*", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=Path("reports/tables/external_prevalence_recalibration.csv"))
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/external_prevalence_recalibrated_predictions.csv"),
    )
    parser.add_argument("--group-columns", nargs="*", default=DEFAULT_GROUP_COLUMNS)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-prevalence", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_paths = args.predictions if args.predictions is not None else default_prediction_paths()
    frames = [_read_prediction_csv(Path(path)) for path in prediction_paths if Path(path).exists()]
    predictions = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False) if frames else pd.DataFrame()
    if predictions.empty:
        checked = ", ".join(str(path) for path in prediction_paths)
        raise FileNotFoundError(f"No usable external prediction CSVs found. Checked: {checked}")
    summary, recalibrated = build_prevalence_recalibration_report(
        predictions,
        group_columns=args.group_columns,
        n_bins=args.n_bins,
        threshold=args.threshold,
        target_prevalence=args.target_prevalence,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    recalibrated.to_csv(args.predictions_output, index=False)
    print(f"Wrote external prevalence recalibration summary: {args.summary_output} ({len(summary)} rows)")
    print(f"Wrote external prevalence recalibrated predictions: {args.predictions_output} ({len(recalibrated)} rows)")


if __name__ == "__main__":
    main()
