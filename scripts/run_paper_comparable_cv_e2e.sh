#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/57_run_paper_comparable_cv.py "$@"
python scripts/20_make_paper_tables.py \
  --metrics data/outputs/metrics/paper_comparable_cv_metrics.csv \
  --output reports/tables/paper_comparable_cv_metric_table.csv \
  --raw-output reports/tables/paper_comparable_cv_metric_table_raw.csv
python scripts/24_make_experiment_manifest.py \
  --output reports/experiment_manifest_paper_comparable_cv.json \
  --artifacts \
    data/processed/features_compare_is10_merged.csv \
    data/outputs/metrics/paper_comparable_cv_metrics.csv \
    data/outputs/metrics/paper_comparable_cv_predictions.csv \
    reports/tables/paper_comparable_cv_feature_selection.csv \
    reports/tables/paper_comparable_cv_metric_table.csv \
    reports/tables/paper_comparable_cv_metric_table_raw.csv
