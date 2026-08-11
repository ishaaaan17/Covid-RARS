#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.final_uncertainty import (
    DEFAULT_FINAL_GROUP_COLUMNS,
    build_final_uncertainty_and_calibration,
    save_calibration_curve_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bootstrap confidence intervals and calibration tables for final validation predictions."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=Path,
        default=[
            Path("data/outputs/metrics/compare_is10_final_validation_predictions.csv"),
            Path("data/outputs/metrics/compare_is10_external_transfer_predictions.csv"),
        ],
    )
    parser.add_argument("--ci-output", type=Path, default=Path("reports/tables/final_validation_bootstrap_ci.csv"))
    parser.add_argument(
        "--calibration-summary-output",
        type=Path,
        default=Path("reports/tables/final_validation_calibration_summary.csv"),
    )
    parser.add_argument(
        "--calibration-bins-output",
        type=Path,
        default=Path("reports/tables/final_validation_calibration_bins.csv"),
    )
    parser.add_argument(
        "--calibration-figure-output",
        type=Path,
        default=Path("reports/figures/final_validation_calibration_curves.svg"),
    )
    parser.add_argument("--calibration-figure-max-series", type=int, default=12)
    parser.add_argument("--group-columns", nargs="*", default=DEFAULT_FINAL_GROUP_COLUMNS)
    parser.add_argument("--metrics", nargs="+", default=["auroc", "auprc", "brier", "ece"])
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ci, calibration_summary, calibration_bins = build_final_uncertainty_and_calibration(
        prediction_paths=args.predictions,
        group_columns=args.group_columns,
        bootstrap_metrics=args.metrics,
        n_bootstraps=args.n_bootstraps,
        n_bins=args.n_bins,
        random_state=args.random_state,
    )
    for path, frame, label in (
        (args.ci_output, ci, "bootstrap confidence intervals"),
        (args.calibration_summary_output, calibration_summary, "calibration summary"),
        (args.calibration_bins_output, calibration_bins, "calibration bins"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"Wrote {label}: {path} ({len(frame)} rows)")
    save_calibration_curve_figure(
        calibration_bins,
        args.calibration_figure_output,
        group_columns=args.group_columns,
        max_series=args.calibration_figure_max_series,
    )
    print(f"Wrote calibration curves: {args.calibration_figure_output}")


if __name__ == "__main__":
    main()
