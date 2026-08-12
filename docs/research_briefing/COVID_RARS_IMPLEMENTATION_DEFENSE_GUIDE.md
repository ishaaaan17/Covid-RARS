# Covid-RARS Implementation Defense Guide

This guide is for answering implementation questions in a viva, research meeting, or code demonstration. It is deliberately high-level but specific enough to memorize.

For non-technical explanations of the difficult terms, use `COVID_RARS_PLAIN_LANGUAGE_EXPLANATION_GUIDE.md` before the detailed sections below.

## One-Line Implementation Summary

We implemented a complete COVID respiratory-audio evaluation pipeline: metadata/quality processing, strong acoustic features, OpenSMILE ComParE 2016 and IS10 features, train-only feature selection, multiple model families, multimodal fusion, WavLM transformer and CNN-BiGRU deep branches, external COUGHVID transfer, and reviewer-grade checks for confidence intervals, calibration, confounding, shuffle sanity, feature stability, decision curves, and incremental audio-over-metadata value.

## Repository Walkthrough Order

If asked to show code, use this order:

1. `scripts/56_run_compare_is10_rescue.py`
   - Shows the ComParE+IS10 feature pipeline and top-k feature selection.

2. `src/covid_rars/compare_is10_rescue.py`
   - Shows feature merging, prefixing, train-only ranking, and top-k table creation.

3. `src/covid_rars/strong_baseline.py`
   - Shows the main model bank, modality handling, fusion, metric computation, and selection logic.

4. `scripts/58_run_compare_is10_final_validation.py`
   - Shows final validation ladder execution.

5. `src/covid_rars/compare_is10_final_validation.py`
   - Shows existing split, time-stratified split, temporal split, and external transfer logic.

6. `scripts/62_run_deep_coughvid_external_transfer.py`
   - Shows WavLM transformer and CNN-BiGRU external-transfer execution.

7. `scripts/59_run_final_uncertainty_calibration.py` through `scripts/68_run_incremental_audio_metadata_value.py`
   - Shows reviewer evidence additions.

8. `scripts/20_make_paper_tables.py` and `scripts/24_make_experiment_manifest.py`
   - Shows how outputs are collected into paper tables and manifest.

## Data Inputs

Main source:

- Coswara processed metadata: `data/processed/metadata_with_quality.csv`
- Coswara audio index: `data/interim/coswara_index.csv`
- Audio quality table: `data/processed/audio_quality.csv`

External source:

- COUGHVID external metadata: `data/processed/coughvid_metadata_compare_is10_external.csv`
- COUGHVID strong features and OpenSMILE features were generated separately before external transfer.

Important point:

```text
Coswara provides cough, breath, and speech. COUGHVID is cough-only. Therefore, multimodal fusion is internally validated on Coswara, while external validation is cough-only.
```

## Preprocessing

Audio loading:

- Audio is loaded through `src/covid_rars/audio_io.py`.
- If PySoundFile fails, librosa/audioread fallback is used.
- This is why logs contain warnings such as "PySoundFile failed. Trying audioread instead." Those warnings are not fatal if features are successfully written.

Feature preprocessing:

- Audio is converted into a stable waveform representation for feature extraction.
- Quality filtering can use:
  - `all_samples`
  - `quality_ok_only`
- Final deep transfer used `quality_ok_only`.

Augmentation:

- Conservative training-only waveform augmentation exists for some feature/deep branches.
- Augmentations include small time stretch, pitch shift, and low-amplitude noise.
- Validation/test/external rows are not augmented for training leakage.

## Feature Extraction

### Strong acoustic feature bank

Implemented in `src/covid_rars/strong_features.py`.

Feature families:

- duration and waveform amplitude statistics
- MFCCs
- delta MFCCs
- delta-delta MFCCs
- mel-band summaries
- chroma
- spectral contrast
- tonnetz
- RMS energy
- zero-crossing rate
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral flatness
- tempogram summary

Why chosen:

```text
These are standard acoustic descriptors that summarize spectral shape, energy, rhythm, voice quality, and timbre. They are useful as a strong classical baseline before deep models.
```

### OpenSMILE ComParE 2016

Used because:

- It is an established paralinguistic feature set.
- COVID cough and speech papers often rely on OpenSMILE-style acoustic descriptors.
- It expands beyond the custom strong acoustic bank.

### OpenSMILE IS10

Used because:

- It is a standard INTERSPEECH paralinguistic set.
- It complements ComParE with another established descriptor family.

### Merged feature table

The merged engineered table contains `10,147` columns in total. Seven identifier and label fields are excluded, leaving `10,140` numeric feature candidates for ranking.

Why feature selection is mandatory:

```text
Using more than 10,000 features directly on a limited number of participants would overfit. Therefore, features are selected on the training split only.
```

## Feature Selection Config

Main final strategy:

```text
compare_is10_top800_lightgbm
```

Settings:

- Top-k values explored: `500`, `800`, `1200`.
- Final top-k: `800`.
- Ranker: LightGBM.
- Selection split: train only.
- Selection scope: per-modality mean.
- Random state: `42`.

LightGBM ranker settings in code:

- `n_estimators=700`
- `learning_rate=0.03`
- `num_leaves=31`
- `subsample=0.9`
- `colsample_bytree=0.75`
- `reg_lambda=2.0`
- `class_weight="balanced"`
- `n_jobs=-1`

Defense:

```text
Feature selection was not done on the test set. The top-800 list is learned from training data and then applied to validation/test/external rows.
```

## Model Families

### Classical tabular models

| Model | Why included |
|---|---|
| LightGBM | Strong gradient-boosted tree baseline for high-dimensional tabular acoustic features |
| CatBoost | Independent boosted-tree family; robust to non-linear feature interactions |
| XGBoost | Standard high-performance gradient boosting baseline |
| RBF SVC | Non-tree non-linear model; useful to show findings are not tree-specific |
| Logistic/ExtraTrees/RandomForest in earlier searches | Used for screening and robustness during broad experiments |

SMOTE-style naming:

- Model names such as `lightgbm_smote_f80` refer to class-imbalance handling and selected feature fraction/strategy in the strong baseline model bank.
- The key point is that multiple model families were compared under the same validation protocols.

### Fusion models

Fusion methods:

- uniform mean
- validation-weighted mean
- validation-weighted AUPRC/AUROC variants
- stacked logistic validation fusion
- global stacking in earlier searches

Why fusion:

```text
Respiratory modalities carry partially different information. Fusion combines participant-level probabilities from cough, breath, and speech when available.
```

### Deep models

WavLM:

- Model: `microsoft/wavlm-base-plus`
- Type: self-supervised transformer.
- Window length: `3.0` seconds.
- Overlap: `0.5`.
- Max segments per recording: `4`.
- Epochs: `8`.
- Batch size: `8`.
- gradient accumulation: `4`.
- Learning rate: `2e-5`.
- Head learning rate: `1e-4`.
- Unfrozen top layers: `4`.
- Patience: `3`.

CNN-BiGRU:

- Architecture: `cnn_bigru`.
- Epochs: `8`.
- Batch size: `16`.
- Learning rate: `1e-3`.
- Input: spectrogram index.

Defense:

```text
We did use a transformer. WavLM is a transformer-based self-supervised speech model. It performed reasonably internally but failed on COUGHVID transfer, so the problem is not solved by using a transformer.
```

## Main Validation Protocols

### Existing participant split

Purpose:

- Shows performance under the existing internal split.

Main result:

- Primary equal-weight fusion: `0.895` AUROC, `0.862` AUPRC.
- Exploratory logistic stack: `0.897` AUROC, `0.863` AUPRC.

### Time-stratified participant split

Purpose:

- Separates participants while respecting time structure more strictly.

Main result:

- `0.849` AUROC, `0.783` AUPRC.

### Early-to-late temporal split

Purpose:

- Trains on earlier collection period and tests on later period.
- Tests temporal drift.

Main result:

- Around `0.698` AUROC in the final summary.
- Multi-seed temporal mean about `0.691 +/- 0.006`.

### Reverse temporal split

Purpose:

- Trains late, tests early.
- Checks whether drift is symmetric or tied to base-rate/calibration changes.

Main result:

- AUROC `0.920`, but AUPRC `0.029`, F1 `0.011`, ECE `0.471`.

Interpretation:

```text
High AUROC alone can be misleading under base-rate and calibration shift.
```

### COUGHVID external transfer

Purpose:

- Tests whether cough representations trained on Coswara transfer to an independent cough dataset.

Main result:

- measured audio-summary models `0.523-0.543` AUROC.
- WavLM `0.484` AUROC.
- CNN-BiGRU `0.548` AUROC.

## Reviewer Evidence Additions

### Bootstrap confidence intervals

Implemented to quantify uncertainty in validation metrics and drops.

Main result:

- Internal-to-COUGHVID AUROC drop `0.354`, CI `[0.300, 0.404]`.

### DeLong / paired comparisons

Implemented for paired AUROC comparisons where aligned prediction rows exist.

Important limitation:

- Some nested metadata+audio comparisons have small aligned sample sizes.
- Do not overclaim significance where p-values are non-significant.

### Calibration

Implemented:

- Calibration summary.
- Calibration bins.
- Calibration curves.
- Brier score and ECE.

Why:

```text
In clinical screening, probability quality matters. AUROC alone does not tell whether predicted risk is calibrated.
```

### Decision curve analysis

Implemented to assess net benefit across thresholds.

Why:

```text
Decision curves translate model predictions into clinical utility. They show whether using the model is better than simple treat-all or treat-none strategies.
```

### Fixed-sensitivity operating points

At screening-like sensitivity `>=0.90`, external precision is about `0.035-0.037`, close to base prevalence.

Defense:

```text
Even if sensitivity is forced high, the external model gives almost no useful positive predictive value.
```

### Recalibration-only check

Purpose:

- Tests whether COUGHVID collapse is only a threshold/calibration issue.

Interpretation:

- Recalibration does not rescue AUROC, so ranking/discrimination is weak.

### Shuffle-label sanity

Implemented for:

- Metadata-only prediction.
- Final prediction sanity.
- ComParE+IS10 shuffle-retrain sanity.

Defense:

```text
When labels are shuffled, performance collapses toward chance. Therefore, the high observed performance is not a trivial script leakage artifact.
```

### Metadata shortcut analysis

Metadata-only models:

- Full safe metadata AUROC `0.964`.
- Symptoms-only AUROC `0.932`.
- Demographic/protocol-only AUROC `0.914`.

Permutation importance:

- Recording/protocol group dominates.
- `recording_year` is a key driver.

### Feature-selection stability

Top-800 early vs late feature overlap:

- Shared features: `110`.
- Union: `1490`.
- Jaccard: `0.074`.

Interpretation:

```text
The acoustic features that look important are not stable across time.
```

### Support overlap

Domain classifier:

- AUROC `0.750`.
- External outside source-support band: `25.2%`.

Interpretation:

```text
COUGHVID partly lies outside the Coswara feature distribution, so the model is forced to extrapolate.
```

### Incremental audio + metadata

Purpose:

- Answers whether audio adds value beyond symptoms/metadata.

Best symptoms-only example:

- metadata-only AUROC `0.888`
- audio-only AUROC `0.818`
- metadata+audio AUROC `0.951`
- delta `+0.063`
- CI `[-0.005, 0.149]`
- p `0.104`

Interpretation:

```text
There may be incremental signal over symptoms-only in small aligned subsets, but it is not statistically secure. Over fuller metadata/context, audio adds little.
```

## Important Terms to Memorize

| Term | Meaning |
|---|---|
| AUROC | Ranking metric: probability a random positive is ranked above a random negative |
| AUPRC | Precision-recall area; more sensitive to class imbalance |
| Balanced accuracy | Average of sensitivity and specificity |
| Sensitivity | True positive rate |
| Specificity | True negative rate |
| Brier score | Proper scoring rule for probability error; lower is better |
| ECE | Expected calibration error; lower means probabilities match observed frequencies better |
| DeLong test | Statistical test for comparing correlated/paired AUROCs |
| Bootstrap CI | Resampling-based confidence interval |
| IPW | Inverse probability weighting; adjusts for covariate imbalance by weighting samples |
| DCA | Decision curve analysis; estimates clinical net benefit across thresholds |
| Support overlap | Whether target-domain samples lie inside source-domain feature space |
| Shuffle-label sanity | Checks that model performance disappears when labels are permuted |
| Feature stability | Whether selected features remain similar across splits/time |

## Commands That Were Central

ComParE+IS10 rescue:

```bash
python scripts/56_run_compare_is10_rescue.py \
  --top-k-values 500 800 1200 \
  --ranker lightgbm \
  --extract-chunk-size 32 \
  --progress-interval 250
```

Paper-comparable CV:

```bash
python scripts/57_run_paper_comparable_cv.py \
  --features data/processed/features_compare_is10_merged.csv \
  --modality cough \
  --n-splits 10 \
  --top-k-values 800 \
  --ranker lightgbm \
  --model-names lightgbm_smote_f80 catboost_smote_f80 xgboost_smote_f80 svc_rbf_f60 \
  --optuna-trials 0
```

Final validation:

```bash
python scripts/58_run_compare_is10_final_validation.py \
  --features data/processed/features_compare_is10_top800.csv \
  --external-features data/processed/features_compare_is10_coughvid_cough_top800.csv
```

Deep external transfer:

```bash
python scripts/62_run_deep_coughvid_external_transfer.py \
  --quality-mode quality_ok_only \
  --window-sec 3.0 \
  --overlap 0.5 \
  --max-segments-per-recording 4 \
  --augment-train-copies 1 \
  --max-epochs 8 \
  --batch-size 8 \
  --gradient-accumulation 4 \
  --learning-rate 2e-5 \
  --head-learning-rate 1e-4 \
  --unfreeze-top-layers 4 \
  --patience 3 \
  --cnn-architecture cnn_bigru \
  --cnn-epochs 8 \
  --cnn-batch-size 16 \
  --cnn-learning-rate 1e-3 \
  --device cuda
```

Reviewer evidence pack:

```bash
python scripts/59_run_final_uncertainty_calibration.py
python scripts/60_run_final_delta_bootstrap.py
python scripts/61_metadata_confounding_subgroups.py
python scripts/63_run_reviewer_evidence_checks.py
python scripts/64_run_metadata_permutation_importance.py
python scripts/65_run_compare_is10_shuffle_retrain_sanity.py
python scripts/66_run_review_temporal_seed_robustness.py
python scripts/67_run_reviewer_extension_checks.py
python scripts/68_run_incremental_audio_metadata_value.py
python scripts/20_make_paper_tables.py
python scripts/24_make_experiment_manifest.py
```

## Weaknesses to Admit Honestly

| Weakness | How to state it safely |
|---|---|
| COUGHVID is cough-only | "External transfer validates cough portability, not full multimodal fusion." |
| External metrics are low | "That is the reliability finding; the model should not be deployed based on internal metrics." |
| Incremental audio+metadata sample is small | "Exploratory aligned-subset analysis; suggestive but not conclusive." |
| No Grad-CAM | "Final model is not a spectrogram transformer; validation-level interpretability was prioritized." |
| No full prospective clinical dataset | "Public benchmark audit; prospective clinical validation remains future work." |
| Not universal SOTA | "Strong internal benchmark, but contribution is rigorous validation and shortcut evidence." |

## Final Code Defense Sentence

```text
The implementation is not one model; it is a full evaluation system. It extracts multiple acoustic feature families, selects features only on training data, trains several model families and fusion routes, evaluates them under internal, temporal, and external protocols, and adds reviewer-grade checks for uncertainty, calibration, confounding, shuffle sanity, support overlap, decision curves, and incremental value over metadata.
```

