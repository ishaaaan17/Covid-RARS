#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/56_run_compare_is10_rescue.py "$@"
python scripts/20_make_paper_tables.py
python scripts/24_make_experiment_manifest.py
