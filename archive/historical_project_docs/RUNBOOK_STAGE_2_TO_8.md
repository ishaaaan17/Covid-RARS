# Stage 2-8 Runbook

Run only after Stage 1 artifacts are correct.

## Stage 2: Features

```bash
python scripts/05_extract_features.py \
  --metadata data/processed/metadata_clean.csv \
  --features-output data/processed/features_mfcc.csv \
  --spectrogram-dir data/processed/spectrograms \
  --spectrogram-index-output data/processed/spectrogram_index.csv
```

## Stage 3: Classical ML

```bash
python scripts/06_train_ml_baselines.py \
  --features data/processed/features_mfcc.csv
```

## Stage 4: CNN

Train cough first:

```bash
python scripts/07_train_cnn.py \
  --spectrogram-index data/processed/spectrogram_index.csv \
  --modality cough \
  --epochs 50 \
  --batch-size 32
```

Train breath/speech only after cough works:

```bash
python scripts/07_train_cnn.py --spectrogram-index data/processed/spectrogram_index.csv --modality breath
python scripts/07_train_cnn.py --spectrogram-index data/processed/spectrogram_index.csv --modality speech
```

## Stage 5: Calibration

Classical ML:

```bash
python scripts/08_calibrate_branches.py \
  --validation-predictions data/outputs/metrics/ml_predictions_validation.csv \
  --test-predictions data/outputs/metrics/ml_predictions_test.csv \
  --method platt
```

CNN:

```bash
python scripts/08_calibrate_branches.py \
  --validation-predictions data/outputs/metrics/cnn_logits_validation.csv \
  --test-predictions data/outputs/metrics/cnn_logits_test.csv \
  --method temperature \
  --output data/outputs/metrics/cnn_calibrated_predictions.csv \
  --metrics-output data/outputs/metrics/cnn_calibration_metrics.csv
```

## Stage 6: Fusion

Use calibrated predictions and validation metrics.

```bash
python scripts/09_run_fusion.py \
  --predictions data/outputs/metrics/calibrated_branch_predictions.csv \
  --validation-metrics data/outputs/metrics/ml_baseline_metrics.csv
```

## Stage 7: Quality/Subgroup Checks

```bash
python scripts/10_shift_and_confounding_checks.py \
  --predictions data/outputs/metrics/calibrated_branch_predictions.csv \
  --metadata data/processed/metadata_clean.csv
```

## Stage 8: Report Assets

```bash
python scripts/11_make_report_assets.py \
  --metadata data/processed/metadata_clean.csv \
  --predictions data/outputs/metrics/calibrated_branch_predictions.csv
```

## Demo Skeleton

```bash
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

The demo initially shows preprocessing/quality/spectrogram. Model inference should be enabled only after final calibrated models are selected.

