# COVID Audio BTP E2E Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strong BTech research project and working prototype for reliability-aware COVID/respiratory audio screening from cough, breath, and speech, using safe IT/ML framing and avoiding clinical-diagnosis claims.

**Architecture:** The project is a reproducible ML pipeline with strict participant-level splitting, modular audio preprocessing, classical and CNN baselines, late multimodal fusion, calibration, uncertainty rejection, quality analysis, limited dataset-shift analysis, and a demo. The research contribution is reliability, calibration, quality, and generalization, not "COVID diagnosis from cough."

**Tech Stack:** Python 3.10+, pandas, numpy, scipy, librosa, soundfile, scikit-learn, xgboost or lightgbm, PyTorch, matplotlib, seaborn, Streamlit or Gradio, pytest.

---

## 0. Planning Boundaries

All planning and implementation files must stay inside:

```text
.audio_btp/
```

Do not create project files in the visible workspace root.

Do not mix this project with mutation/genomic/variant-prediction work. That is a separate topic and is intentionally excluded.

This project is safe because it stays within:

- audio signal processing,
- machine learning classification,
- reliability evaluation,
- dataset documentation,
- non-diagnostic research prototype design.

The demo and report must always include:

```text
Research prototype only. Not a clinical diagnostic tool.
```

## 1. Final Topic Decision

Final title:

```text
Confidence-Calibrated and Drift-Aware Multimodal COVID/Respiratory Screening from Cough, Breath, and Speech Sounds
```

Short professor title:

```text
Reliable COVID/Respiratory Screening from Cough, Breath, and Speech Using Calibrated AI
```

Recommended implementation title for repo/report:

```text
Confidence-Calibrated Multimodal Respiratory Audio Screening Using Coswara
```

Reason for the implementation title: it is safer, narrower, and easier to finish. It still allows the report to discuss COVID/respiratory screening while avoiding overclaiming.

## 2. Research Thesis

Weak thesis to avoid:

```text
CNN detects COVID from cough with high accuracy.
```

Strong thesis to use:

```text
Crowdsourced respiratory-audio models are fragile under participant variation, noisy recordings, class imbalance, calibration error, and dataset shift. A BTech-scale system can make this problem stronger by comparing cough, breath, and speech modalities, calibrating confidence, rejecting uncertain predictions, analyzing quality, and reporting realistic limitations.
```

Core contribution:

1. A reproducible Coswara-based multimodal audio pipeline.
2. Participant-level split to avoid leakage.
3. Classical ML and CNN baselines.
4. Cough, breath, speech, and late-fusion comparison.
5. Confidence calibration and uncertainty rejection.
6. Audio-quality analysis.
7. External cough validation with COUGHVID if time and labels permit.
8. Non-diagnostic demo and professor-ready report.

## 3. Verified Source Backbone

Use these as the citation backbone. Keep exact DOI/page details in the literature table during report writing.

### Base Paper

1. JMIR 2025 drift-adaptive cough-audio framework
   - Link: https://www.jmir.org/2025/1/e66919
   - Use: main base paper.
   - Key support: cough-audio models degrade over time; drift detection and adaptation improve robustness.
   - Project gap: cough-only, limited multimodal breath/speech integration, limited BTech-friendly calibration/quality/fusion pipeline.

### Dataset Papers

2. Coswara Scientific Data / dataset paper
   - GitHub: https://github.com/iiscleap/Coswara-Data
   - Paper/arXiv: https://arxiv.org/abs/2305.12741
   - Use: primary dataset and main reproducibility anchor.
   - Key support: same-participant cough, breath, vowel/speech-style recordings plus metadata.

3. COVID-19 Sounds NeurIPS Datasets and Benchmarks 2021
   - Link: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/e2c0be24560d78c5e599c2a9c9d0bbd2-Abstract-round2.html
   - Use: large external reference and motivation for digital respiratory screening.
   - Project status: do not depend on raw access for the BTech deliverable.

4. COUGHVID Scientific Data 2021
   - Paper: https://www.nature.com/articles/s41597-021-00937-4
   - Zenodo: https://zenodo.org/records/4048312
   - Use: optional external cough validation.
   - Caveat: labels are noisy; use as robustness evidence, not final ground truth.

### Reliability And Caution Papers

5. Nature Machine Intelligence 2024 caution paper
   - Link: https://www.nature.com/articles/s42256-023-00773-8
   - Use: required caution citation.
   - Key support: audio-only screening may not meaningfully beat symptom checkers under realistic controls.
   - Project response: do not overclaim; measure calibration, uncertainty, quality, and confounding risks.

6. Confidence-calibrated respiratory screening, IEEE-BHI 2024
   - Link: https://openreview.net/forum?id=chVymJKep2
   - Use: calibration justification.
   - Project response: include ECE, Brier score, reliability diagrams, and rejection curves.

### Modality And Architecture Papers

7. Hierarchical Spectrogram Transformer respiratory-sound paper
   - arXiv: https://arxiv.org/abs/2207.09529
   - IEEE: https://ieeexplore.ieee.org/abstract/document/10342847
   - Use: high-end deep architecture reference.
   - Project response: cite it, but implement a small CNN first.

8. SympCoughNet 2025
   - Link: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1551298/full
   - Use: symptom-assisted recent paper.
   - Project response: symptoms are metadata/confounding analysis, not the central model claim.

9. Speech-based respiratory diagnostics, PLOS ONE 2025
   - Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC12637952/
   - Use: supports speech/vowel modality.
   - Project response: include speech/vowel experiments but do not rely only on speech.

10. Cross-dataset cough robustness paper, Expert Systems with Applications 2026
   - Link: https://researchoutput.csu.edu.au/en/publications/robust-covid-19-detection-from-cough-sounds-using-deep-neural-dec/
   - Use: cross-dataset motivation.
   - Project response: include a simple train-on-Coswara, test-on-COUGHVID experiment if feasible.

## 4. Non-Negotiable Method Rules

These rules prevent weak research, leakage, and last-minute redesign.

1. Split by participant, not by recording.
2. Never augment before splitting.
3. Keep train, validation, and test participant IDs fixed in files.
4. Report class distribution for every split.
5. Report AUROC and AUPRC, not accuracy alone.
6. Report calibration metrics, not just classification metrics.
7. Keep low-quality audio with a flag first; compare filtered vs unfiltered later.
8. Never claim clinical diagnosis.
9. For cross-dataset testing, expect performance drop and present it as reliability evidence.
10. Use a classical ML baseline before deep learning.

## 5. Scope Locks

### Must-Have Scope

This is the minimum strong BTech.

1. Coswara data setup.
2. Metadata cleaning.
3. Participant-level train/validation/test split.
4. Audio preprocessing.
5. MFCC and simple acoustic features.
6. Classical ML baseline.
7. Log-mel CNN baseline.
8. Cough, breath, speech, and late-fusion comparison.
9. Calibration and reliability diagrams.
10. Quality-aware analysis.
11. Streamlit or Gradio demo.
12. Final report and slides.

### Strong Scope

Add these after must-have scope passes.

1. COUGHVID external cough validation.
2. Simple chronological or month-wise drift analysis if recording dates are usable.
3. Feature importance or SHAP for classical models.
4. Grad-CAM or saliency on spectrogram CNN.

### Out Of Scope

These are intentionally excluded unless the professor specifically demands them.

1. Claiming diagnostic replacement.
2. Building a mobile app.
3. Training large transformers from scratch.
4. Complex domain adaptation as the main implementation.
5. Cambridge/UK restricted raw datasets as required dependencies.
6. Biological protocol, genomics, mutation, or wet-lab content.

## 6. Final Hidden Repository Layout

Create the implementation repo inside:

```text
.audio_btp/covid_audio_btp/
```

Final layout:

```text
.audio_btp/
  COVID_AUDIO_BTP_CONTEXT.md
  plans/
    2026-05-24-covid-audio-btp-e2e-master-plan.md
  references/
    literature_matrix.md
  covid_audio_btp/
    README.md
    pyproject.toml
    requirements.txt
    .gitignore
    data/
      raw/
        coswara/
        coughvid/
      interim/
        coswara_index.csv
        split_manifest.csv
      processed/
        metadata_clean.csv
        audio_quality.csv
        features_mfcc.csv
        features_acoustic.csv
        spectrogram_index.csv
      outputs/
        models/
        metrics/
        figures/
    notebooks/
      01_dataset_audit.ipynb
      02_feature_extraction_check.ipynb
      03_baseline_review.ipynb
      04_calibration_review.ipynb
    src/
      covid_audio_btp/
        __init__.py
        config.py
        paths.py
        schemas.py
        data_index.py
        labels.py
        split.py
        audio_io.py
        quality.py
        preprocess.py
        features.py
        spectrograms.py
        datasets.py
        models_ml.py
        models_cnn.py
        train_ml.py
        train_cnn.py
        calibration.py
        fusion.py
        metrics.py
        evaluation.py
        explain.py
        reporting.py
    tests/
      test_labels.py
      test_split.py
      test_quality.py
      test_features.py
      test_metrics.py
      test_calibration.py
      fixtures/
        tiny_audio/
    scripts/
      00_check_environment.py
      01_build_coswara_index.py
      02_clean_metadata.py
      03_create_splits.py
      04_extract_features.py
      05_train_ml_baselines.py
      06_train_cnn.py
      07_run_fusion.py
      08_calibrate_models.py
      09_quality_analysis.py
      10_external_coughvid_eval.py
      11_make_report_assets.py
    app/
      app.py
      demo_assets/
    reports/
      report_outline.md
      final_report.md
      slides_outline.md
      tables/
      figures/
```

## 7. Data Schema

### `metadata_clean.csv`

Required columns:

```text
participant_id
recording_id
dataset
modality
submodality
audio_path
label_raw
label_binary
label_group
recording_date
age
gender
country
symptoms_json
comorbidities_json
duration_sec
sample_rate_original
quality_flag
split
```

Allowed `dataset`:

```text
coswara
coughvid
```

Allowed `modality`:

```text
cough
breath
speech
```

Allowed `submodality` examples:

```text
heavy_cough
shallow_cough
deep_breath
shallow_breath
vowel_a
vowel_e
vowel_o
counting_normal
counting_fast
```

Allowed `label_binary`:

```text
positive
negative
unknown
```

Exclude `unknown` from supervised training. Keep it in audit tables.

### `split_manifest.csv`

Required columns:

```text
participant_id
dataset
split
label_binary
n_recordings
modalities_available
split_seed
```

Allowed `split`:

```text
train
validation
test
external
unused
```

### `audio_quality.csv`

Required columns:

```text
recording_id
duration_sec
rms_mean
rms_std
zero_crossing_rate_mean
silence_ratio
clipping_ratio
spectral_centroid_mean
spectral_flatness_mean
snr_proxy
quality_flag
quality_reasons
```

Allowed `quality_flag`:

```text
ok
short
mostly_silence
clipped
corrupt
unsupported_format
```

## 8. Audio Processing Decisions

Use these defaults unless experiments show a clear reason to change.

1. Load with `soundfile` first, fallback to `librosa.load`.
2. Convert stereo to mono by mean.
3. Resample to 16000 Hz.
4. Trim silence using librosa energy-based trimming.
5. Peak normalize after trimming.
6. Store derived features, not full processed waveforms, unless storage is cheap.
7. Fixed spectrogram length:
   - cough: 4 seconds
   - breath: 5 seconds
   - speech/vowel/counting: 5 seconds
8. Pad shorter clips with zeros.
9. Center-crop or random-crop longer clips during training.
10. Use deterministic center crop for validation/test.

Quality thresholds:

```text
cough short threshold: duration_sec < 0.5
breath/speech short threshold: duration_sec < 1.0
mostly_silence threshold: silence_ratio > 0.70
clipped threshold: clipping_ratio > 0.01
```

These thresholds must be reported as design choices, not medical rules.

## 9. Feature Plan

### Classical Features

For each clip:

```text
mfcc_1_mean ... mfcc_40_mean
mfcc_1_std ... mfcc_40_std
delta_mfcc_1_mean ... delta_mfcc_40_mean
delta_mfcc_1_std ... delta_mfcc_40_std
delta2_mfcc_1_mean ... delta2_mfcc_40_mean
delta2_mfcc_1_std ... delta2_mfcc_40_std
rms_mean
rms_std
zcr_mean
zcr_std
spectral_centroid_mean
spectral_bandwidth_mean
spectral_rolloff_mean
spectral_flatness_mean
duration_sec
```

Expected feature size:

```text
40 * 2 * 3 + 9 = 249 features
```

### Spectrogram Features

For CNN:

```text
sample_rate = 16000
n_fft = 400
hop_length = 160
win_length = 400
n_mels = 64
fmin = 125
fmax = 7500
power = 2.0
log_scale = decibel
```

Expected shape:

```text
1 x 64 x T
```

Where `T` depends on the fixed clip length.

## 10. Model Plan

### Baseline 0: Majority And Dummy Baselines

Purpose: sanity check.

Models:

```text
DummyClassifier(strategy="most_frequent")
DummyClassifier(strategy="stratified")
```

Must be beaten by all serious models.

### Baseline 1: Classical ML

Models:

```text
LogisticRegression(class_weight="balanced")
LinearSVC with calibrated probabilities
RandomForestClassifier(class_weight="balanced")
XGBoost or LightGBM if installation is stable
```

Primary classical model:

```text
XGBoost if available, RandomForest otherwise
```

### Baseline 2: CNN

Architecture:

```text
Conv2d(1, 16, kernel_size=3, padding=1)
BatchNorm2d(16)
ReLU
MaxPool2d(2)
Conv2d(16, 32, kernel_size=3, padding=1)
BatchNorm2d(32)
ReLU
MaxPool2d(2)
Conv2d(32, 64, kernel_size=3, padding=1)
BatchNorm2d(64)
ReLU
AdaptiveAvgPool2d((1, 1))
Dropout(0.3)
Linear(64, 1)
```

Loss:

```text
BCEWithLogitsLoss with pos_weight from train split
```

Optimizer:

```text
AdamW(lr=1e-3, weight_decay=1e-4)
```

Early stopping:

```text
monitor validation AUPRC
patience 8 epochs
max_epochs 50
```

### Fusion

Use late fusion because it is explainable and robust.

Fusion methods:

```text
mean_probability
validation_auroc_weighted_probability
logistic_stacking_on_validation_predictions
```

Use stacking only if validation set size is sufficient. Otherwise, use mean probability.

## 11. Calibration Plan

Calibration methods:

```text
Platt scaling
Isotonic regression
Temperature scaling for CNN logits
```

Calibration split:

```text
Use validation split only.
Never fit calibration on test split.
```

Metrics:

```text
Expected Calibration Error
Brier score
negative log likelihood
reliability diagram
confidence histogram
```

Demo uncertainty rule:

```text
if 0.40 <= calibrated_probability <= 0.60:
    prediction = "uncertain"
elif calibrated_probability > 0.60:
    prediction = "screening signal: higher"
else:
    prediction = "screening signal: lower"
```

Do not display "COVID positive" or "COVID negative" in the demo as a medical result. Use screening-support language.

## 12. Experiment Matrix

### Experiment A: Dataset Audit

Question:

```text
What data do we have, and is it suitable for participant-level ML?
```

Outputs:

```text
reports/tables/dataset_counts.csv
reports/tables/split_counts.csv
reports/figures/modality_distribution.png
reports/figures/class_distribution.png
reports/figures/example_spectrograms.png
```

Acceptance:

```text
Every supervised sample has participant_id, modality, path, label_binary, and split.
No participant appears in more than one split.
```

### Experiment B: Classical ML Baseline

Question:

```text
How strong are simple acoustic features?
```

Models:

```text
Logistic Regression
SVM or LinearSVC with calibration
Random Forest
XGBoost/LightGBM if stable
```

Outputs:

```text
reports/tables/ml_baseline_metrics.csv
reports/figures/ml_roc_curves.png
reports/figures/ml_pr_curves.png
reports/figures/ml_confusion_matrices.png
```

Acceptance:

```text
At least one non-dummy model trains and beats dummy baseline on validation AUROC and AUPRC.
```

### Experiment C: CNN Spectrogram Baseline

Question:

```text
Does a compact CNN improve over classical features?
```

Outputs:

```text
data/outputs/models/cnn_<modality>.pt
reports/tables/cnn_metrics.csv
reports/figures/cnn_training_curves.png
```

Acceptance:

```text
Training loss decreases.
Validation metrics are non-random.
No train/test participant leakage.
```

### Experiment D: Modality Comparison

Question:

```text
Which modality is most reliable: cough, breath, speech, or fusion?
```

Runs:

```text
cough
breath
speech
cough+breath
cough+speech
breath+speech
cough+breath+speech
```

Outputs:

```text
reports/tables/modality_comparison.csv
reports/figures/modality_auprc_bar.png
reports/figures/modality_ece_bar.png
```

Acceptance:

```text
Report both best-performing modality and most calibrated modality.
If fusion does not improve performance, explain it honestly.
```

### Experiment E: Calibration

Question:

```text
Can calibrated probabilities make the prototype more reliable?
```

Outputs:

```text
reports/tables/calibration_metrics.csv
reports/figures/reliability_diagrams.png
reports/figures/confidence_histograms.png
reports/figures/rejection_curve.png
```

Acceptance:

```text
At least one calibration method improves ECE or Brier score on test.
If it does not, report that calibration did not help and keep raw model for comparison.
```

### Experiment F: Quality Analysis

Question:

```text
Do low-quality recordings cause more errors or worse calibration?
```

Runs:

```text
all_samples
quality_flag_ok_only
quality_features_added
```

Outputs:

```text
reports/tables/quality_effect_metrics.csv
reports/figures/error_by_quality_flag.png
reports/figures/calibration_by_quality_flag.png
```

Acceptance:

```text
Report whether quality filtering improves or hurts metrics.
Do not silently discard samples.
```

### Experiment G: External COUGHVID Validation

Question:

```text
How much does cough-only performance drop across datasets?
```

Runs:

```text
train Coswara cough -> test COUGHVID cough
train COUGHVID labeled subset -> test Coswara cough
```

Outputs:

```text
reports/tables/external_validation_metrics.csv
reports/figures/external_shift_roc_pr.png
```

Acceptance:

```text
If labels are too noisy or mapping is unreliable, document the reason and keep COUGHVID as dataset discussion only.
```

### Experiment H: Drift Or Time-Split Analysis

Question:

```text
Do metrics change across recording time?
```

Runs:

```text
train early participants -> test later participants
month-wise feature distribution comparison
```

Outputs:

```text
reports/tables/time_split_metrics.csv
reports/figures/monthly_feature_shift.png
```

Acceptance:

```text
Run only if recording_date is usable for enough participants.
If dates are missing/noisy, document date coverage and skip time-split claims.
```

## 13. Report Argument Structure

Final report chapters:

1. Introduction
   - problem context,
   - low-cost audio screening motivation,
   - non-diagnostic boundary.
2. Literature Review
   - dataset papers,
   - model papers,
   - calibration/reliability papers,
   - caution/confounding papers.
3. Research Gap
   - model drift,
   - cross-dataset shift,
   - overconfidence,
   - quality issues,
   - multimodal underuse.
4. Methodology
   - data,
   - preprocessing,
   - participant split,
   - features,
   - baselines,
   - CNN,
   - fusion,
   - calibration,
   - quality analysis.
5. Experiments And Results
   - dataset audit,
   - classical ML,
   - CNN,
   - modality comparison,
   - fusion,
   - calibration,
   - quality,
   - external validation/drift if completed.
6. Discussion
   - what worked,
   - what failed,
   - why high accuracy is not enough,
   - relation to caution paper.
7. Demo
   - system workflow,
   - screenshots,
   - limitations.
8. Conclusion
   - reliability-aware prototype,
   - future work.

## 14. Professor-Facing Pitch

Use this pitch:

```text
We propose a reliability-aware respiratory audio screening prototype using cough, breath, and speech recordings. Existing COVID-audio papers often report promising accuracy, but recent literature shows that these systems can be fragile under noisy crowdsourced recordings, participant differences, temporal drift, cross-dataset shift, and overconfident predictions. Our project focuses on a practical and defensible BTech contribution: multimodal comparison, confidence calibration, uncertainty rejection, audio-quality analysis, and realistic evaluation using participant-level splits.
```

Base paper:

```text
A Comprehensive Drift-Adaptive Framework for Sustaining Model Performance in COVID-19 Detection From Dynamic Cough Audio Data, JMIR 2025.
```

Gap:

```text
The base paper demonstrates temporal drift and adaptation for cough audio, but the BTech-scale opportunity is to build a simpler reliability-aware multimodal system that adds breath/speech, calibration, quality filtering, and cross-dataset analysis.
```

## 15. Grading Strategy

What makes this strong for BTech:

1. It has recent base paper support.
2. It has public data available immediately.
3. It has a working demo.
4. It has multiple measurable experiments.
5. It avoids weak claims and handles limitations maturely.
6. It can be completed even without external restricted datasets.
7. It has clear fallbacks if deep learning or external validation becomes hard.

What to show in review meetings:

1. Week 1: dataset audit table and spectrograms.
2. Week 2: first baseline metrics.
3. Week 3: CNN training curve.
4. Week 4: modality comparison table.
5. Week 5: reliability diagram.
6. Week 6: quality/error analysis.
7. Week 7: demo.
8. Week 8: final report and slides.

## 16. Implementation Task Plan

### Task 1: Project Scaffold

**Files:**
- Create: `.audio_btp/covid_audio_btp/README.md`
- Create: `.audio_btp/covid_audio_btp/pyproject.toml`
- Create: `.audio_btp/covid_audio_btp/requirements.txt`
- Create: `.audio_btp/covid_audio_btp/.gitignore`
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/__init__.py`

- [ ] **Step 1: Create hidden project directories**

Run:

```bash
mkdir -p .audio_btp/covid_audio_btp/{data/raw/coswara,data/raw/coughvid,data/interim,data/processed,data/outputs/models,data/outputs/metrics,data/outputs/figures,notebooks,reports/tables,reports/figures,app/demo_assets,scripts,tests/fixtures/tiny_audio,src/covid_audio_btp}
```

Expected:

```text
All directories are created under .audio_btp/covid_audio_btp.
```

- [ ] **Step 2: Create dependency files**

`requirements.txt` content:

```text
numpy
pandas
scipy
librosa
soundfile
scikit-learn
matplotlib
seaborn
torch
torchaudio
xgboost
joblib
tqdm
pytest
streamlit
```

- [ ] **Step 3: Add project README**

`README.md` must include:

```text
# COVID/Respiratory Audio BTP

Research prototype for reliability-aware respiratory audio screening from cough, breath, and speech.

This is not a clinical diagnostic tool.
```

- [ ] **Step 4: Verify imports**

Run:

```bash
python scripts/00_check_environment.py
```

Expected:

```text
Environment check passed
```

### Task 2: Schemas And Label Rules

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/schemas.py`
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/labels.py`
- Test: `.audio_btp/covid_audio_btp/tests/test_labels.py`

- [ ] **Step 1: Write label tests**

Test cases:

```python
from covid_audio_btp.labels import normalize_label

def test_normalize_positive_labels():
    assert normalize_label("positive") == "positive"
    assert normalize_label("COVID positive") == "positive"

def test_normalize_negative_labels():
    assert normalize_label("negative") == "negative"
    assert normalize_label("healthy") == "negative"

def test_unknown_label_is_preserved():
    assert normalize_label("") == "unknown"
    assert normalize_label(None) == "unknown"
```

- [ ] **Step 2: Implement label normalization**

Core behavior:

```python
def normalize_label(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in {"", "unknown", "na", "nan", "none"}:
        return "unknown"
    if "positive" in text or text in {"covid", "covid-19"}:
        return "positive"
    if "negative" in text or text in {"healthy", "normal"}:
        return "negative"
    return "unknown"
```

- [ ] **Step 3: Run tests**

Run:

```bash
pytest tests/test_labels.py -v
```

Expected:

```text
3 passed
```

### Task 3: Data Indexing

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/data_index.py`
- Create: `.audio_btp/covid_audio_btp/scripts/01_build_coswara_index.py`
- Test: `.audio_btp/covid_audio_btp/tests/test_split.py`

- [ ] **Step 1: Define index output contract**

Every indexed row must contain:

```text
participant_id, recording_id, dataset, modality, submodality, audio_path, label_raw, label_binary
```

- [ ] **Step 2: Implement file discovery**

Use `pathlib.Path.rglob` to find:

```text
*.wav
*.flac
*.mp3
*.ogg
*.webm
```

- [ ] **Step 3: Add a dry-run mode**

Command:

```bash
python scripts/01_build_coswara_index.py --raw-dir data/raw/coswara --output data/interim/coswara_index.csv --dry-run
```

Expected:

```text
Print count of discovered audio files without writing CSV.
```

### Task 4: Participant-Level Split

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/split.py`
- Create: `.audio_btp/covid_audio_btp/scripts/03_create_splits.py`
- Test: `.audio_btp/covid_audio_btp/tests/test_split.py`

- [ ] **Step 1: Write no-leakage test**

Test:

```python
import pandas as pd
from covid_audio_btp.split import assert_no_participant_leakage

def test_no_participant_leakage_passes():
    df = pd.DataFrame({
        "participant_id": ["p1", "p2", "p3"],
        "split": ["train", "validation", "test"],
    })
    assert_no_participant_leakage(df)

def test_no_participant_leakage_fails():
    df = pd.DataFrame({
        "participant_id": ["p1", "p1"],
        "split": ["train", "test"],
    })
    try:
        assert_no_participant_leakage(df)
    except ValueError as exc:
        assert "participant leakage" in str(exc).lower()
    else:
        raise AssertionError("Expected leakage error")
```

- [ ] **Step 2: Implement split check**

Core behavior:

```python
def assert_no_participant_leakage(df):
    counts = df.groupby("participant_id")["split"].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        raise ValueError(f"participant leakage detected: {list(leaked.index[:10])}")
```

- [ ] **Step 3: Create stratified participant split**

Use a fixed seed:

```text
split_seed = 42
```

Default ratios:

```text
train = 0.70
validation = 0.15
test = 0.15
```

### Task 5: Audio Quality

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/audio_io.py`
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/quality.py`
- Test: `.audio_btp/covid_audio_btp/tests/test_quality.py`

- [ ] **Step 1: Write quality threshold tests**

Test:

```python
from covid_audio_btp.quality import assign_quality_flag

def test_short_cough_flag():
    flag, reasons = assign_quality_flag("cough", duration_sec=0.2, silence_ratio=0.1, clipping_ratio=0.0)
    assert flag == "short"
    assert "duration" in reasons

def test_mostly_silence_flag():
    flag, reasons = assign_quality_flag("speech", duration_sec=2.0, silence_ratio=0.9, clipping_ratio=0.0)
    assert flag == "mostly_silence"
    assert "silence" in reasons

def test_ok_flag():
    flag, reasons = assign_quality_flag("breath", duration_sec=2.0, silence_ratio=0.2, clipping_ratio=0.0)
    assert flag == "ok"
    assert reasons == []
```

- [ ] **Step 2: Implement quality flags**

Thresholds:

```python
MIN_DURATION = {"cough": 0.5, "breath": 1.0, "speech": 1.0}
MAX_SILENCE_RATIO = 0.70
MAX_CLIPPING_RATIO = 0.01
```

- [ ] **Step 3: Generate quality CSV**

Run:

```bash
python scripts/09_quality_analysis.py --metadata data/processed/metadata_clean.csv --output data/processed/audio_quality.csv
```

Expected:

```text
audio_quality.csv written with one row per recording
```

### Task 6: Feature Extraction

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/features.py`
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/spectrograms.py`
- Create: `.audio_btp/covid_audio_btp/scripts/04_extract_features.py`
- Test: `.audio_btp/covid_audio_btp/tests/test_features.py`

- [ ] **Step 1: Write feature shape test**

Test:

```python
import numpy as np
from covid_audio_btp.features import extract_mfcc_features

def test_mfcc_feature_count():
    y = np.zeros(16000, dtype=np.float32)
    features = extract_mfcc_features(y, sample_rate=16000, n_mfcc=40)
    assert len(features) == 249
```

- [ ] **Step 2: Implement MFCC feature extraction**

Required output:

```text
249 numeric features per recording
```

- [ ] **Step 3: Implement log-mel spectrogram**

Required output:

```text
float32 numpy array with shape (1, 64, T)
```

### Task 7: Classical ML Training

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/models_ml.py`
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/train_ml.py`
- Create: `.audio_btp/covid_audio_btp/scripts/05_train_ml_baselines.py`

- [ ] **Step 1: Train dummy baselines**

Run:

```bash
python scripts/05_train_ml_baselines.py --features data/processed/features_mfcc.csv --metadata data/processed/metadata_clean.csv --model-set dummy
```

Expected:

```text
dummy_metrics.csv written
```

- [ ] **Step 2: Train classical models**

Run:

```bash
python scripts/05_train_ml_baselines.py --features data/processed/features_mfcc.csv --metadata data/processed/metadata_clean.csv --model-set classical
```

Expected:

```text
ml_baseline_metrics.csv written
```

### Task 8: CNN Training

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/datasets.py`
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/models_cnn.py`
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/train_cnn.py`
- Create: `.audio_btp/covid_audio_btp/scripts/06_train_cnn.py`

- [ ] **Step 1: Build spectrogram dataset loader**

Expected:

```text
Returns spectrogram tensor, binary label, participant_id, recording_id.
```

- [ ] **Step 2: Train one modality**

Run:

```bash
python scripts/06_train_cnn.py --metadata data/processed/metadata_clean.csv --modality cough --epochs 50
```

Expected:

```text
cnn_cough.pt and cnn_cough_metrics.csv written
```

### Task 9: Evaluation Metrics

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/metrics.py`
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/evaluation.py`
- Test: `.audio_btp/covid_audio_btp/tests/test_metrics.py`

- [ ] **Step 1: Implement metric bundle**

Metrics:

```text
AUROC
AUPRC
balanced_accuracy
sensitivity
specificity
F1
Brier score
ECE
```

- [ ] **Step 2: Write ECE test**

Test:

```python
import numpy as np
from covid_audio_btp.metrics import expected_calibration_error

def test_ece_perfect_predictions_is_low():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.01, 0.99, 0.02, 0.98])
    assert expected_calibration_error(y_true, y_prob, n_bins=2) < 0.05
```

### Task 10: Calibration

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/calibration.py`
- Create: `.audio_btp/covid_audio_btp/scripts/08_calibrate_models.py`
- Test: `.audio_btp/covid_audio_btp/tests/test_calibration.py`

- [ ] **Step 1: Fit calibration only on validation predictions**

Inputs:

```text
validation_predictions.csv
test_predictions.csv
```

Outputs:

```text
calibrated_test_predictions.csv
calibration_metrics.csv
```

- [ ] **Step 2: Verify no test fitting**

Test behavior:

```text
The calibrator fit function accepts validation y/prob only.
The transform function accepts test probabilities only after fit.
```

### Task 11: Fusion

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/fusion.py`
- Create: `.audio_btp/covid_audio_btp/scripts/07_run_fusion.py`

- [ ] **Step 1: Implement mean probability fusion**

Rule:

```text
Group predictions by participant_id, then average available modality probabilities.
```

- [ ] **Step 2: Implement validation-weighted fusion**

Rule:

```text
weight each modality by max(validation_auroc - 0.5, 0.0)
normalize weights to sum to 1
fallback to equal weights if all weights are zero
```

### Task 12: Reporting Assets

**Files:**
- Create: `.audio_btp/covid_audio_btp/src/covid_audio_btp/reporting.py`
- Create: `.audio_btp/covid_audio_btp/scripts/11_make_report_assets.py`
- Create: `.audio_btp/covid_audio_btp/reports/report_outline.md`

- [ ] **Step 1: Generate tables**

Required tables:

```text
dataset_counts.csv
split_counts.csv
ml_baseline_metrics.csv
cnn_metrics.csv
modality_comparison.csv
calibration_metrics.csv
quality_effect_metrics.csv
external_validation_metrics.csv if available
```

- [ ] **Step 2: Generate figures**

Required figures:

```text
class_distribution.png
modality_distribution.png
example_spectrograms.png
roc_curves.png
pr_curves.png
reliability_diagrams.png
quality_effects.png
```

### Task 13: Demo App

**Files:**
- Create: `.audio_btp/covid_audio_btp/app/app.py`

- [ ] **Step 1: Build upload UI**

Inputs:

```text
audio file upload
modality selector: cough, breath, speech
```

Outputs:

```text
waveform
log-mel spectrogram
quality flag
screening signal
calibrated confidence
uncertainty warning
research disclaimer
```

- [ ] **Step 2: Run app locally**

Run:

```bash
streamlit run app/app.py
```

Expected:

```text
App opens and accepts one audio file.
```

## 17. Two-Month Execution Schedule

Week 1:

- Scaffold repo.
- Download or mount Coswara.
- Build metadata index.
- Create participant-level splits.
- Produce dataset audit.

Week 2:

- Implement audio loading and quality features.
- Extract MFCC/acoustic features.
- Train dummy and classical ML baselines.

Week 3:

- Build spectrogram extraction.
- Train CNN for cough.
- Extend CNN to breath and speech if cough pipeline is stable.

Week 4:

- Run modality comparison.
- Implement late fusion.
- Create first result tables.

Week 5:

- Add Platt/isotonic calibration.
- Build reliability diagrams.
- Add uncertainty rejection curve.

Week 6:

- Run quality-aware analysis.
- Try COUGHVID external cough validation.
- Decide whether drift/time-split is feasible.

Week 7:

- Add explainability.
- Build demo.
- Draft report methods and results.

Week 8:

- Final report.
- Slides.
- Demo screenshots.
- Reproducibility run.

## 18. Decision Gates

Gate 1 after Week 1:

```text
If Coswara indexing fails, switch to Hugging Face mirror or manually prepared subset.
```

Gate 2 after Week 2:

```text
If XGBoost install fails, use Random Forest and Logistic Regression only.
```

Gate 3 after Week 3:

```text
If CNN overfits badly, keep CNN as secondary and make calibrated classical ML the main result.
```

Gate 4 after Week 4:

```text
If fusion does not improve metrics, report fusion as not beneficial under the dataset and emphasize calibration/quality.
```

Gate 5 after Week 6:

```text
If COUGHVID label mapping is unreliable, remove external validation from required scope and keep it as limitations/future work.
```

## 19. Final Deliverables

Required deliverables:

```text
code repository under .audio_btp/covid_audio_btp
final_report.md
slides_outline.md
dataset audit table
metrics tables
figures
trained baseline model
demo app
```

Professor/demo deliverables:

```text
1-page pitch
10-12 slide presentation
demo screenshot
short result table
limitations slide
```

## 20. Self-Review Checklist

Before implementation starts, verify:

- [ ] No project files are in the visible root except existing user files.
- [ ] Plan excludes mutation/genomic/variant-prediction work.
- [ ] Main dataset does not require access approval.
- [ ] External datasets are not blocking.
- [ ] Participant-level split is mandatory.
- [ ] Calibration is mandatory.
- [ ] Quality analysis is mandatory.
- [ ] Demo language is non-diagnostic.
- [ ] Report includes the caution paper.
- [ ] There is a fallback if CNN, fusion, or external validation underperforms.

