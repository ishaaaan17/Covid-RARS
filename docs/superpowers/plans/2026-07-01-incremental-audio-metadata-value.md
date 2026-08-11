# Incremental Audio Metadata Value Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reviewer-facing fixed-split experiment that tests whether audio predictions add incremental discrimination beyond metadata predictions, plus a compact model-family external-transfer summary.

**Architecture:** Use existing Coswara metadata and existing participant-level prediction CSVs. Train metadata base models on train participants, fit a small logistic combiner on validation participants for metadata-only/audio-only/metadata+audio, and evaluate all three on the same test participants. Keep this separate from the existing skipped final-fusion nested row so the new experiment only uses audio sources with real validation/test overlap.

**Tech Stack:** pandas, numpy, scikit-learn logistic regression, existing `covid_rars.metrics`, existing metadata feature builder.

---

### Task 1: Incremental-Value Core

**Files:**
- Create: `src/covid_rars/incremental_value.py`
- Test: `tests/test_incremental_value.py`

- [ ] **Step 1: Write failing tests** for source selection, same-participant alignment, and metadata+audio delta reporting.
- [ ] **Step 2: Run tests and verify they fail** because `covid_rars.incremental_value` does not exist.
- [ ] **Step 3: Implement the core functions**:
  - `build_metadata_probability_table`
  - `build_audio_source_candidates`
  - `build_incremental_audio_metadata_value`
  - `build_external_model_family_transfer_summary`
- [ ] **Step 4: Run focused tests and verify they pass.**

### Task 2: CLI Runner

**Files:**
- Create: `scripts/68_run_incremental_audio_metadata_value.py`
- Modify: `scripts/20_make_paper_tables.py`
- Modify: `scripts/24_make_experiment_manifest.py`

- [ ] **Step 1: Add a CLI that reads metadata and prediction CSVs and writes metrics/predictions/summary tables.**
- [ ] **Step 2: Register new metric and artifact paths in paper-table and manifest scripts.**
- [ ] **Step 3: Run focused tests plus paper/manifest tests.**

### Task 3: Ubuntu Run

**Files:**
- No code changes.

- [ ] **Step 1: Push code from Windows.**
- [ ] **Step 2: Pull on Ubuntu.**
- [ ] **Step 3: Run `python scripts/68_run_incremental_audio_metadata_value.py`.**
- [ ] **Step 4: Regenerate paper tables and manifest.**
