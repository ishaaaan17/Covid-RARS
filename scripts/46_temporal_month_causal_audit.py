#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covid_rars.temporal_month_causal import run_temporal_month_causal_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit why recording month harms temporal generalization.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/metadata_clean.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_mfcc.csv"))
    parser.add_argument("--temporal-predictions", type=Path, default=Path("data/outputs/metrics/temporal_holdout_predictions.csv"))
    parser.add_argument("--month-ablation", type=Path, default=Path("reports/tables/temporal_month_year_ablation_paper_table.csv"))
    parser.add_argument("--month-label-shift-output", type=Path, default=Path("reports/tables/temporal_month_label_shift.csv"))
    parser.add_argument("--month-covariate-shift-output", type=Path, default=Path("reports/tables/temporal_month_covariate_shift.csv"))
    parser.add_argument("--matched-cohort-output", type=Path, default=Path("reports/tables/temporal_matched_cohort_metrics.csv"))
    parser.add_argument("--failure-modes-output", type=Path, default=Path("reports/tables/temporal_failure_modes_by_shift.csv"))
    parser.add_argument("--uncertainty-output", type=Path, default=Path("reports/tables/temporal_uncertainty_under_shift.csv"))
    parser.add_argument("--effect-sizes-output", type=Path, default=Path("reports/tables/temporal_month_ablation_effect_sizes.csv"))
    parser.add_argument("--dag-output", type=Path, default=Path("reports/final/TEMPORAL_MONTH_CAUSAL_DAG.md"))
    parser.add_argument("--theory-output", type=Path, default=Path("reports/final/TEMPORAL_SHORTCUT_THEORY.md"))
    return parser.parse_args()


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    features = _read_optional_csv(args.features)
    predictions = _read_optional_csv(args.temporal_predictions)
    month_ablation = _read_optional_csv(args.month_ablation)

    result = run_temporal_month_causal_audit(
        metadata=metadata,
        features=features,
        predictions=predictions,
        month_ablation=month_ablation,
    )
    _write_csv(result.month_label_shift, args.month_label_shift_output)
    _write_csv(result.month_covariate_shift, args.month_covariate_shift_output)
    _write_csv(result.matched_cohort_metrics, args.matched_cohort_output)
    _write_csv(result.failure_modes, args.failure_modes_output)
    _write_csv(result.uncertainty_summary, args.uncertainty_output)
    _write_csv(result.month_ablation_effect_sizes, args.effect_sizes_output)
    _write_text(result.causal_dag_markdown, args.dag_output)
    _write_text(result.theory_markdown, args.theory_output)

    print(f"Wrote temporal month label shift: {args.month_label_shift_output} ({len(result.month_label_shift)} rows)")
    print(f"Wrote temporal month covariate shift: {args.month_covariate_shift_output} ({len(result.month_covariate_shift)} rows)")
    print(f"Wrote matched temporal cohort metrics: {args.matched_cohort_output} ({len(result.matched_cohort_metrics)} rows)")
    print(f"Wrote temporal failure modes: {args.failure_modes_output} ({len(result.failure_modes)} rows)")
    print(f"Wrote temporal uncertainty summary: {args.uncertainty_output} ({len(result.uncertainty_summary)} rows)")
    print(f"Wrote temporal month ablation effect sizes: {args.effect_sizes_output} ({len(result.month_ablation_effect_sizes)} rows)")
    print(f"Wrote temporal causal DAG: {args.dag_output}")
    print(f"Wrote temporal shortcut theory: {args.theory_output}")


if __name__ == "__main__":
    main()
