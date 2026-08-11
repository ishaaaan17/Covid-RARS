# Protocol-Matched Paper Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a defensible paper-style comparison workflow that can run the current best cough-only model under prior-paper split styles, especially the HST/JBHI participant-disjoint 10 repeated splits with approximately 70% train, 20% test, and 10% validation participants.

**Architecture:** Add a separate protocol-matched CV path instead of editing the existing paper-comparable CV script. Add a target table and gap-summary script that keeps prior-paper internal, temporal, and cross-dataset comparisons separate.

**Tech Stack:** Python, pandas, scikit-learn, existing feature/model helpers, pytest.

---

## Scope Lock

Do not rerun the full model zoo.

Use one focused model setup:

```text
modality = cough
feature_strategy = compare_is10_top800_lightgbm
model_name = svc_rbf_f60
reason = best existing cough-only paper-comparable aggregate AUROC in the current results table
```

This is the cleanest "same split as papers" check for cough-only papers. The multimodal final ladder remains separate because HST and ESWA are cough-only comparisons.

## Files

- Create: `src/covid_rars/protocol_matched_cv.py`
- Create: `scripts/69_run_protocol_matched_cv.py`
- Create: `tests/test_protocol_matched_cv.py`
- Create: `src/covid_rars/protocol_matched_comparison.py`
- Create: `scripts/70_make_protocol_matched_gap_table.py`
- Create: `tests/test_protocol_matched_comparison.py`
- Create: `reports/tables/protocol_matched_paper_targets.csv`
- Create: `docs/professor/COVID_AUDIO_PAPER_PROTOCOL_AUDIT.md`

## Tasks

- [x] Create standalone participant-disjoint CV tests.
- [x] Create standalone participant-disjoint repeated split implementation matching the HST paper's stated 70/20/10 participant proportions.
- [x] Create standalone CLI for the focused protocol-matched run.
- [x] Create standalone target/gap summary tests.
- [x] Create standalone target/gap summary implementation.
- [x] Create confirmed paper target CSV.
- [x] Create protocol audit note.
- [ ] Run targeted tests in an environment with project dependencies.
- [ ] Run the focused protocol-matched experiment on Ubuntu.
- [ ] Generate `protocol_matched_gap_summary.csv`.

## Ubuntu Commands

```bash
export COVID_RARS_PROJECT_ROOT="${COVID_RARS_PROJECT_ROOT:-$HOME/Desktop/Covid-RARS}"
cd "$COVID_RARS_PROJECT_ROOT"
source .venv/bin/activate

python scripts/69_run_protocol_matched_cv.py \
  --features data/processed/features_compare_is10_merged.csv \
  --modality cough \
  --n-splits 10 \
  --test-fraction 0.2 \
  --validation-fraction 0.125 \
  --top-k-values 800 \
  --ranker lightgbm \
  --model-names svc_rbf_f60 \
  --metrics-output data/outputs/metrics/protocol_matched_hst_style_cough_metrics.csv \
  --predictions-output data/outputs/metrics/protocol_matched_hst_style_cough_predictions.csv \
  --feature-selection-output reports/tables/protocol_matched_hst_style_cough_feature_selection.csv \
  --split-audit-output reports/tables/protocol_matched_hst_style_cough_split_audit.csv

python scripts/70_make_protocol_matched_gap_table.py \
  --targets reports/tables/protocol_matched_paper_targets.csv \
  --protocol-metrics data/outputs/metrics/protocol_matched_hst_style_cough_metrics.csv \
  --external-transfer-summary reports/tables/reviewer_external_model_family_transfer_summary.csv \
  --final-validation-summary reports/tables/compare_is10_final_validation_summary.csv \
  --output reports/tables/protocol_matched_gap_summary.csv
```

Expected runtime: about 20-60 minutes if features already exist; slower if LightGBM feature ranking is CPU-heavy.
