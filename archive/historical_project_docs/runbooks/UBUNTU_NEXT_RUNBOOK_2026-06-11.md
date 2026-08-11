# COVID Audio BTP Ubuntu Continuation Runbook

Date: 2026-06-11

Purpose: exact next-session handoff for resuming on the Ubuntu lab machine. This is the file to use next time, not the older generic handoff.

## Current Priority

Regenerate corrected no-leakage Coswara metrics on Ubuntu before starting COUGHVID.

Do not run COUGHVID yet. Do not run multi-disease/ICBHI work yet. Do not cite old validation-weighted fusion results as final.

Actual stop point today:

```text
Coswara first-run results were generated.
CNN cough ablation was generated.
Leakage hotfix/tests passed on Windows.
Corrected no-leakage Coswara metrics have NOT yet been regenerated on Ubuntu.
COUGHVID has NOT yet been extracted.
No COUGHVID index/features/cross-dataset metrics exist yet.
```

## Ubuntu Lab System

Use:

```text
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp
```

Known lab specs:

```text
CPU: Intel Core i7-14700, 24 cores/threads exposed
RAM: about 19 GiB
GPU: NVIDIA T1000 8GB
Driver: 595.71.05
nvidia-smi CUDA: 13.2
PyTorch checked earlier: torch 2.11.0+cu128, CUDA available, GPU NVIDIA T1000 8GB
```

The immediate no-leakage Coswara regeneration is mostly CPU/scikit-learn/pandas. Do not rerun CNN now.

## What Was Fixed

Old bug:

```text
ml_baseline_metrics.csv was test-set metrics.
The fusion scripts/notebooks used it as --validation-metrics.
```

That contaminates validation-weighted fusion weights with test information.

Fixed behavior:

```text
scripts/06_train_ml_baselines.py now writes:
  data/outputs/metrics/ml_validation_metrics.csv

Fusion scripts must use:
  --validation-metrics data/outputs/metrics/ml_validation_metrics.csv
```

The fusion scripts now reject:

```text
data/outputs/metrics/ml_baseline_metrics.csv
```

when it is passed as validation metrics.

## Relevant Files Edited

The leakage hotfix touches these files:

```text
src/covid_audio_btp/models_ml.py
scripts/06_train_ml_baselines.py
scripts/09_run_fusion.py
scripts/16_run_quality_weighted_fusion.py
notebooks/00_RUN_EVERYTHING_PUBLICATION.ipynb
notebooks/00_MASTER_RUN_ALL.ipynb
tests/test_publication_strengthening.py
```

Expected roles:

```text
models_ml.py:
  TrainResult carries validation metrics as well as test metrics.

06_train_ml_baselines.py:
  adds --validation-metrics-output
  default: data/outputs/metrics/ml_validation_metrics.csv

09_run_fusion.py:
  refuses test metrics as validation weights

16_run_quality_weighted_fusion.py:
  same refusal guard for quality-weighted fusion

master notebooks:
  fusion cells point to ml_validation_metrics.csv

test_publication_strengthening.py:
  regression tests cover validation metrics and fusion guard
```

## Verified Status Today

On Windows, after:

```powershell
python -m pip install -e .
```

tests passed:

```text
targeted leakage test 1: passed
targeted leakage test 2: passed
full pytest: 52 passed, 7 warnings
```

The Windows copy does not have processed Coswara data:

```powershell
Test-Path data/processed/features_mfcc.csv
```

returned false. Therefore metrics must be regenerated on Ubuntu.

## Step 1: Start Ubuntu Session

Run:

```bash
cd /home/covid/Desktop/Covid-19-BTP/covid_audio_btp
source .venv/bin/activate
which python
python --version
git status --short
```

Expected Python path:

```text
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp/.venv/bin/python
```

If `which python` is not inside `.venv`, stop and reactivate the venv.

## Step 2: Confirm Hotfix Is Present

Run:

```bash
git pull
ls tests/test_publication_strengthening.py
rg -n "ml_validation_metrics|Refusing to use ml_baseline_metrics|validation-metrics-output" scripts tests notebooks
```

Expected:

```text
test_publication_strengthening.py exists
rg output mentions scripts/06_train_ml_baselines.py
rg output mentions scripts/09_run_fusion.py
rg output mentions scripts/16_run_quality_weighted_fusion.py
```

If the grep finds nothing, the Ubuntu machine does not have the hotfix. Stop and ask Codex to re-host or re-apply the patch. Do not rely on an old Cloudflare URL unless a tunnel is active in the current session.

## Step 3: Run Tests On Ubuntu

Run:

```bash
python -m pip install -e .
python -m pytest tests/test_publication_strengthening.py::test_ml_baseline_script_writes_validation_metrics_for_fusion_weights
python -m pytest tests/test_publication_strengthening.py::test_fusion_scripts_reject_test_metrics_as_validation_weights
python -m pytest
```

Expected:

```text
targeted leakage tests pass
full pytest passes
```

If tests fail, stop and send the exact output.

## Step 4: Confirm Processed Data Exists

Run:

```bash
ls -lh data/processed/features_mfcc.csv
ls -lh data/processed/audio_quality.csv
ls -lh data/processed/metadata_clean.csv
ls -lh data/interim/split_manifest.csv
```

Expected:

```text
features_mfcc.csv exists, previously around 90 MB
audio_quality.csv exists, previously around 8.6 MB
metadata_clean.csv exists, previously around 15 MB
split_manifest.csv exists
```

If any file is missing, stop. You are in the wrong folder or the processed run is not available.

## Step 5: Regenerate Corrected Coswara Metrics

Run from:

```text
/home/covid/Desktop/Covid-19-BTP/covid_audio_btp
```

with `.venv` active:

```bash
python scripts/06_train_ml_baselines.py --features data/processed/features_mfcc.csv
python scripts/08_calibrate_branches.py --validation-predictions data/outputs/metrics/ml_predictions_validation.csv --test-predictions data/outputs/metrics/ml_predictions_test.csv --method platt
python scripts/09_run_fusion.py --predictions data/outputs/metrics/calibrated_branch_predictions.csv --validation-metrics data/outputs/metrics/ml_validation_metrics.csv
python scripts/16_run_quality_weighted_fusion.py --predictions data/outputs/metrics/calibrated_branch_predictions.csv --quality data/processed/audio_quality.csv --validation-metrics data/outputs/metrics/ml_validation_metrics.csv
python scripts/17_abstention_analysis.py --predictions data/outputs/metrics/quality_weighted_fusion_predictions.csv --metadata data/processed/metadata_clean.csv
python scripts/15_bootstrap_prediction_metrics.py --predictions data/outputs/metrics/quality_weighted_fusion_predictions.csv --output data/outputs/metrics/quality_weighted_fusion_bootstrap_ci.csv --group-columns fusion_method
python scripts/21_paired_model_comparison.py --predictions data/outputs/metrics/calibrated_branch_predictions.csv --baseline-name logistic_regression --model-column model_name --group-columns modality --output data/outputs/metrics/paired_model_comparison.csv
python scripts/22_confounding_matching.py --metadata data/processed/metadata_clean.csv --predictions data/outputs/metrics/quality_weighted_fusion_predictions.csv --covariates age_bucket gender
python scripts/20_make_paper_tables.py
python scripts/24_make_experiment_manifest.py
python scripts/12_validate_artifacts.py --strict
```

Expected time:

```text
roughly 10-30 minutes total
```

Expected validation note:

```text
warning unknown_labels 5688 rows have unknown labels
```

This warning is expected and not a blocker by itself.

## Step 6: Confirm Correct Files Were Produced

Run:

```bash
ls -lh data/outputs/metrics/ml_validation_metrics.csv
ls -lh data/outputs/metrics/ml_baseline_metrics.csv
ls -lh data/outputs/metrics/fusion_metrics.csv
ls -lh data/outputs/metrics/quality_weighted_fusion_metrics.csv
ls -lh reports/tables/paper_metric_table.csv
ls -lh reports/experiment_manifest.json
```

Then run:

```bash
python - <<'PY'
import pandas as pd

path = "data/outputs/metrics/ml_validation_metrics.csv"
df = pd.read_csv(path)
print(df.head().to_string(index=False))
print("\ncolumns:", list(df.columns))
if "metric_split" in df.columns:
    print("\nmetric_split values:", sorted(df["metric_split"].dropna().astype(str).unique()))
PY
```

Expected:

```text
ml_validation_metrics.csv exists
if metric_split exists, it should contain validation only
```

## Step 7: Print Corrected Metrics

Run:

```bash
python - <<'PY'
import pandas as pd

print("\nValidation metrics used for fusion weights:")
print(pd.read_csv("data/outputs/metrics/ml_validation_metrics.csv").to_string(index=False))

print("\nCorrected fusion metrics:")
print(pd.read_csv("data/outputs/metrics/fusion_metrics.csv").to_string(index=False))

print("\nCorrected quality-weighted fusion metrics:")
print(pd.read_csv("data/outputs/metrics/quality_weighted_fusion_metrics.csv").to_string(index=False))

print("\nPaper metric table tail:")
print(pd.read_csv("reports/tables/paper_metric_table.csv").tail(20).to_string(index=False))
PY
```

Send this output to Codex before starting COUGHVID.

## Step 8: Bundle Corrected Coswara Results

Only after metrics print successfully:

```bash
cd /home/covid/Desktop/Covid-19-BTP
rm -rf results/frozen/Corrected_Coswara_NoLeakage_Results
mkdir -p results/frozen/Corrected_Coswara_NoLeakage_Results artifacts/bundles
cp covid_audio_btp/data/outputs/metrics/*.csv results/frozen/Corrected_Coswara_NoLeakage_Results/
cp covid_audio_btp/reports/tables/*.csv results/frozen/Corrected_Coswara_NoLeakage_Results/
cp covid_audio_btp/reports/experiment_manifest.json results/frozen/Corrected_Coswara_NoLeakage_Results/
cp covid_audio_btp/reports/report_outline.md results/frozen/Corrected_Coswara_NoLeakage_Results/ 2>/dev/null || true
cp covid_audio_btp/reports/slides_outline.md results/frozen/Corrected_Coswara_NoLeakage_Results/ 2>/dev/null || true
zip -r artifacts/bundles/Corrected_Coswara_NoLeakage_Results.zip results/frozen/Corrected_Coswara_NoLeakage_Results
ls -lh artifacts/bundles/Corrected_Coswara_NoLeakage_Results.zip
```

Do not include raw datasets in the bundle.

## COUGHVID Stage: Not Before Corrected Coswara Review

COUGHVID is cough-only. It cannot validate full cough + breath + speech fusion.

Current COUGHVID state:

```text
The COUGHVID zip has been downloaded by the user.
The COUGHVID zip has not been extracted yet.
No COUGHVID scripts should be run until corrected Coswara metrics are regenerated and reviewed.
```

After corrected Coswara metrics are reviewed, expected local zip path:

```text
data/raw/coughvid/public_dataset.zip
```

Expected extraction folder:

```text
data/raw/coughvid/v3_extracted
```

Commands for later:

```bash
cd /home/covid/Desktop/Covid-19-BTP/covid_audio_btp
source .venv/bin/activate
mkdir -p data/raw/coughvid/v3_extracted
unzip -q data/raw/coughvid/public_dataset.zip -d data/raw/coughvid/v3_extracted
find data/raw/coughvid/v3_extracted -name "._*" -type f -delete

# Verify extraction before indexing.
find data/raw/coughvid/v3_extracted -maxdepth 2 -type f | head
find data/raw/coughvid/v3_extracted -type f | wc -l

python scripts/13_build_coughvid_index.py --raw-dir data/raw/coughvid/v3_extracted --output data/interim/coughvid_index.csv --require-audio --min-cough-detected 0.80
python scripts/19_extract_coughvid_features.py --index data/interim/coughvid_index.csv --features-output data/processed/coughvid_features_mfcc_smoke.csv --quality-ok-only --max-rows 25
```

If `unzip` says the file does not exist, locate the downloaded zip first:

```bash
find /home/covid -name "public_dataset*.zip" -o -name "*coughvid*.zip"
```

Then copy or move the zip to:

```text
data/raw/coughvid/public_dataset.zip
```

Only if the smoke test succeeds:

```bash
python scripts/19_extract_coughvid_features.py --index data/interim/coughvid_index.csv --features-output data/processed/coughvid_features_mfcc.csv --quality-ok-only
python scripts/18_cross_dataset_feature_eval.py --source-features data/processed/features_mfcc.csv --external-features data/processed/coughvid_features_mfcc.csv --modality cough --model-name logistic_regression --predictions-output data/outputs/metrics/cross_dataset_predictions_logistic_regression.csv --metrics-output data/outputs/metrics/cross_dataset_metrics_logistic_regression.csv
python scripts/18_cross_dataset_feature_eval.py --source-features data/processed/features_mfcc.csv --external-features data/processed/coughvid_features_mfcc.csv --modality cough --model-name random_forest --predictions-output data/outputs/metrics/cross_dataset_predictions_random_forest.csv --metrics-output data/outputs/metrics/cross_dataset_metrics_random_forest.csv
python scripts/23_feature_shift_report.py --source-features data/processed/features_mfcc.csv --external-features data/processed/coughvid_features_mfcc.csv --output reports/tables/feature_shift_report.csv --summary-output reports/tables/feature_shift_summary.csv
python scripts/20_make_paper_tables.py
python scripts/24_make_experiment_manifest.py
python scripts/12_validate_artifacts.py --strict
```

## Later Work After COUGHVID Baseline

Do later, not tomorrow morning before corrected Coswara:

```text
XGBoost or LightGBM champion
multivariate feature selection: SelectFromModel or RFECV
calibration
Coswara internal evaluation
COUGHVID cough-only external evaluation
7-modality cost-benefit table
```

## Publication Framing

Use:

```text
COVID-19 is used as the initial benchmark for a general respiratory-audio screening framework. The main contribution is not COVID detection alone, but a leakage-controlled, calibrated, multimodal pipeline with confounding analysis and cross-dataset shift evaluation. Multi-disease respiratory screening is a natural extension, but requires clinically reliable labels and modality-compatible datasets.
```

Do not claim:

```text
SOTA COVID detector
clinical diagnostic system
COVID-specific acoustic biomarker discovery
COUGHVID validates multimodal fusion
ICBHI validates the smartphone cough+breath+speech pipeline
IEEE Transactions-level novelty
```

Realistic targets after all work:

```text
IEEE J-BHI
Elsevier Biomedical Signal Processing and Control
IEEE BHI
IEEE EMBC
CHIL Applications and Practice
```

## What To Send Codex Next

If successful:

```text
I regenerated corrected Coswara no-leakage metrics. Here is ml_validation_metrics, fusion_metrics, quality_weighted_fusion_metrics, and paper_metric_table tail:

[paste output]
```

If failed:

```text
This command failed:
[paste command]

Error:
[paste full traceback/output]
```

