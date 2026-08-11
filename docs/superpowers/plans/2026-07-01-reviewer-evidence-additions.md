# Reviewer Evidence Additions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reviewer-facing evidence tables for leakage sanity checks, temporal directionality, seed stability, metadata shortcut mechanisms, clinical operating points, residual shortcut correlation, and COUGHVID recalibration without adding unsupported claims.

**Architecture:** Reuse existing metric/calibration/temporal/final-validation modules and add small focused modules where the current codebase lacks a reusable API. Heavy analyses are exposed as scripts so they can run on the Ubuntu machine with the full datasets; local tests use synthetic data only.

**Tech Stack:** Python, pandas, numpy, scikit-learn, pytest, existing `covid_rars` package.

---

### Task 1: Reviewer evidence utilities

**Files:**
- Create: `src/covid_rars/reviewer_evidence.py`
- Test: `tests/test_reviewer_evidence.py`

- [ ] Write failing tests for prediction-level shuffle-label sanity, fixed-sensitivity operating points, audio-vs-metadata residual correlation, and partial target-domain recalibration.
- [ ] Run the focused test and verify it fails because `covid_rars.reviewer_evidence` is missing.
- [ ] Implement the minimal reusable functions.
- [ ] Run the focused test and verify it passes.

### Task 2: Metadata permutation importance

**Files:**
- Create: `src/covid_rars/metadata_permutation_importance.py`
- Test: `tests/test_metadata_permutation_importance.py`

- [ ] Write a failing test proving permutation importance ranks an intentionally predictive metadata field.
- [ ] Run the focused test and verify it fails because the module is missing.
- [ ] Implement metadata logistic fitting with grouped permutation importance using the existing metadata feature-frame builder.
- [ ] Run the focused test and verify it passes.

### Task 3: Reverse temporal and seed-stability runners

**Files:**
- Create: `src/covid_rars/reviewer_temporal_robustness.py`
- Create: `scripts/66_run_review_temporal_seed_robustness.py`
- Test: `tests/test_reviewer_temporal_robustness.py`

- [ ] Write failing tests for reversing temporal assignments and aggregating metrics across seeds.
- [ ] Run the focused test and verify it fails because the module is missing.
- [ ] Implement split reversal and seed-summary helpers.
- [ ] Add a script that runs late-to-early temporal validation and multi-seed final-validation summaries on Ubuntu.
- [ ] Run focused tests and compile the script.

### Task 4: CLI scripts and artifact discovery

**Files:**
- Create: `scripts/63_run_reviewer_evidence_checks.py`
- Create: `scripts/64_run_metadata_permutation_importance.py`
- Update: `scripts/20_make_paper_tables.py`
- Update: `scripts/24_make_experiment_manifest.py`

- [ ] Add CLIs for reviewer evidence checks and metadata permutation importance.
- [ ] Wire new table/metric outputs into paper table and manifest discovery.
- [ ] Run focused tests for new modules plus paper-table and manifest tests.

### Task 5: Verification

**Files:**
- All touched source, script, and test files.

- [ ] Run focused pytest for all new behavior.
- [ ] Run existing paper-table/manifest tests.
- [ ] Run `py_compile` for new scripts and modules.
- [ ] Report exactly which analyses are implemented locally and which still require full Ubuntu execution.
