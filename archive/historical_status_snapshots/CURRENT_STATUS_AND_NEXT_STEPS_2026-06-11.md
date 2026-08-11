# Covid-RARS - Current Status And Next Steps

Date: 2026-06-11

This document captures the current working state after the Coswara baseline and Phase 3 reliability extras completed on the local Ubuntu lab machine.

## Current Status

The Coswara-only pipeline is complete through Phase 3.

Completed stages:

- Dataset layout audit
- Coswara index creation
- Metadata cleaning
- Participant-level split creation
- Audio quality audit
- MFCC feature extraction
- Classical ML baselines
- Calibration
- Validation-thresholded fusion
- Subgroup/shift checks
- Report asset generation
- Metadata-only baseline
- Quality-weighted fusion
- Abstention analysis
- Bootstrap confidence intervals
- Paired model comparison
- Confounding matched subset
- Paper metric table
- Experiment manifest

Validation status:

- `python scripts/12_validate_artifacts.py --strict` passed with only the expected warning:
  - `unknown_labels`: 5688 rows have unknown labels
- `python -m pytest` passed:
  - `50 passed`
  - 7 sklearn warnings from tiny single-class test subgroups; not a pipeline failure

Created result bundle:

```text
~/Desktop/Covid-19-BTP/Phase3_Coswara_Results.zip
```

## Important Artifacts

Core artifacts:

```text
data/interim/coswara_index.csv
data/processed/metadata_clean.csv
data/interim/modality_availability.csv
data/interim/split_manifest.csv
data/processed/audio_quality.csv
data/processed/features_mfcc.csv
data/processed/spectrogram_index.csv
data/outputs/metrics/ml_baseline_metrics.csv
data/outputs/metrics/calibration_metrics.csv
data/outputs/metrics/calibrated_branch_predictions.csv
data/outputs/metrics/calibrated_branch_predictions_validation.csv
data/outputs/metrics/fusion_predictions.csv
data/outputs/metrics/fusion_metrics.csv
data/outputs/metrics/fusion_thresholds.csv
```

Phase 3 artifacts:

```text
data/outputs/metrics/metadata_baseline_metrics.csv
data/outputs/metrics/quality_weighted_fusion_predictions.csv
data/outputs/metrics/quality_weighted_fusion_metrics.csv
data/outputs/metrics/quality_weighted_fusion_thresholds.csv
data/outputs/metrics/abstention_decisions.csv
data/outputs/metrics/abstention_coverage_curve.csv
data/outputs/metrics/quality_weighted_fusion_bootstrap_ci.csv
data/outputs/metrics/paired_model_comparison.csv
data/outputs/metrics/matched_subset_metrics.csv
data/processed/metadata_matched.csv
reports/tables/confounding_balance.csv
reports/tables/paper_metric_table.csv
reports/tables/paper_metric_table_raw.csv
reports/experiment_manifest.json
```

Expected missing artifact for now:

```text
reports/tables/feature_shift_report.csv
```

This is expected because `RUN_FEATURE_SHIFT_REPORT = False` and COUGHVID is still disabled.

## Current Main Results

Best current Coswara model:

```text
validation_weighted_auprc fusion
AUROC: 0.877670
AUPRC: 0.842766
Balanced accuracy: 0.812305
F1: 0.753927
Sensitivity: 0.699029
Specificity: 0.925581
Threshold: 0.357049
N test participants: 318
```

Uniform fusion:

```text
AUROC: 0.878709
AUPRC: 0.840627
Balanced accuracy: 0.810996
F1: 0.743961
Sensitivity: 0.747573
Specificity: 0.874419
Threshold: 0.346350
```

Quality-weighted fusion:

```text
AUROC: 0.878709
AUPRC: 0.838230
Balanced accuracy: 0.791488
F1: 0.708861
Sensitivity: 0.815534
Specificity: 0.767442
Threshold: 0.333910
```

Interpretation:

- Main result for BTP reporting should be validation-weighted fusion.
- Quality-weighted fusion should be presented as a reliability/robustness ablation, not as the best model.
- The validation-tuned thresholds fixed the earlier invalid all-negative threshold behavior.

## Fixes Applied During This Stage

Fixes already applied and verified:

- Ignored macOS AppleDouble sidecar files such as `._*.wav` and `__MACOSX`.
- Added calibrated validation predictions.
- Added validation-tuned fusion thresholds.
- Fixed participant-level subgroup merge.
- Fixed participant-level matched-subset confounding metrics.
- Added validation-tuned thresholding for quality-weighted fusion.
- Included quality-weighted fusion metrics in paper tables.

Verification after fixes:

```text
50 passed
```

## What Not To Do Now

Do not rerun Phase 3 unless changing code or config.

Do not delete:

```text
data/interim/
data/processed/
data/outputs/
reports/
```

Do not enable:

```python
RUN_CNN = True
RUN_COUGHVID_FEATURES = True
RUN_CROSS_DATASET = True
RUN_FEATURE_SHIFT_REPORT = True
```

until the next step is deliberately selected.

Do not report quality-weighted fusion as the best model. It is an ablation.

## Immediate Next Step

Send or inspect:

```text
~/Desktop/Covid-19-BTP/Phase3_Coswara_Results.zip
```

The next action should be interpretation of the Phase 3 CSVs:

- Decide final BTP result table
- Decide which plots to include
- Decide final report/slides narrative
- Decide whether to proceed to COUGHVID

## Recommended Next Technical Step

Recommended next technical branch:

```text
COUGHVID cough-only external validation
```

Reason:

- It strengthens publication claims.
- It directly addresses cross-dataset generalization.
- COUGHVID is cough-only, so it should validate only cough-compatible pipelines, not full cough+breath+speech fusion.

Do this only after the current Phase 3 result bundle is reviewed.

## COUGHVID Guardrails

COUGHVID must not be treated as a multimodal external dataset.

Allowed:

- COUGHVID cough-only index
- COUGHVID cough feature extraction smoke test
- Coswara cough-only model to COUGHVID evaluation
- COUGHVID feature shift report

Not allowed:

- Claiming COUGHVID validates breath or speech
- Claiming COUGHVID validates full multimodal fusion
- Enabling full COUGHVID feature extraction before a 25-row smoke test passes

## Current Decision

For BTP:

```text
Current Coswara Phase 3 result is sufficient for a strong BTP milestone.
```

For publication:

```text
Proceed to COUGHVID cough-only external validation next, then revisit claims.
```

