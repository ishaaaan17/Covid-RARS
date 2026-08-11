# Notebook Workflow

The implementation is now notebook-first and module-backed.

## Primary Entry Point

```text
notebooks/00_MASTER_RUN_ALL.ipynb
```

This notebook runs the project in the intended order:

```text
raw Coswara layout audit
  -> audio index
  -> cleaned metadata
  -> participant-level split
  -> quality audit
  -> validation gate
  -> MFCC/acoustic features + log-mel spectrograms
  -> classical ML baselines
  -> optional compact CNN on GPU
  -> branch calibration
  -> multimodal fusion
  -> subgroup/confounding checks
  -> report assets
```

## Why Multiple Notebooks

A single giant notebook is fragile because skipped cells create hidden state bugs. These notebooks use saved artifacts as contracts, so you can restart Jupyter and continue safely.

The key artifacts are:

```text
data/interim/coswara_index.csv
data/processed/metadata_clean.csv
data/interim/modality_availability.csv
data/interim/split_manifest.csv
data/processed/audio_quality.csv
reports/tables/validation_issues.csv
data/processed/features_mfcc.csv
data/processed/spectrogram_index.csv
data/outputs/metrics/*.csv
reports/tables/nb*.csv
reports/figures/nb*.png
```

## Hard Gates

Do not train models until these pass in `00_MASTER_RUN_ALL.ipynb` and `01_dataset_audit.ipynb`:

- both positive and negative labels are present;
- participants appear in only one split;
- unknown label rate is low;
- quality audit is present;
- corrupt audio rate is low;
- enough recordings have `quality_flag == ok`.

## Review Notebooks

Use the stage notebooks after the corresponding master stage creates artifacts. They are not empty wrappers; they load artifacts, run checks, save tables/figures, and raise clear errors when a previous stage is missing or invalid.

```text
01_dataset_audit.ipynb
02_quality_review.ipynb
03_feature_review.ipynb
04_ml_baseline_review.ipynb
05_cnn_review.ipynb
06_calibration_fusion_review.ipynb
07_shift_confounding_review.ipynb
```

## GPU Use

`RUN_CNN` is false by default. Turn it on only after quality and ML baselines are sane, and only on the machine where you want to spend GPU time.
