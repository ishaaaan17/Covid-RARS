#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covid_rars.metadata_permutation_importance import run_metadata_permutation_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run permutation-importance analysis for metadata-only shortcut/confounding models."
    )
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_with_quality.csv"))
    parser.add_argument("--feature-sets", nargs="+", default=["full_safe_metadata", "symptoms_only", "demographic_protocol_only"])
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/outputs/metrics/metadata_permutation_importance_metrics.csv"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("data/outputs/metrics/metadata_permutation_importance_predictions.csv"),
    )
    parser.add_argument(
        "--importance-output",
        type=Path,
        default=Path("reports/tables/metadata_confounding_permutation_importance.csv"),
    )
    parser.add_argument(
        "--group-summary-output",
        type=Path,
        default=Path("reports/tables/metadata_confounding_permutation_group_summary.csv"),
    )
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {label}: {path} ({len(frame)} rows)")


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    result = run_metadata_permutation_importance(
        metadata,
        feature_sets=args.feature_sets,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    _write_csv(result.metrics, args.metrics_output, "metadata permutation metrics")
    _write_csv(result.predictions, args.predictions_output, "metadata permutation predictions")
    _write_csv(result.importance, args.importance_output, "metadata permutation importance")
    _write_csv(result.group_summary, args.group_summary_output, "metadata permutation group summary")


if __name__ == "__main__":
    main()
