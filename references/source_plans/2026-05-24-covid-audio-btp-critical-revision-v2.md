# COVID Audio BTP Critical Revision V2

> **For agentic workers:** This file supersedes conflicting sequencing/method details in the E2E master plan. Use it together with `2026-05-24-covid-audio-btp-e2e-master-plan.md`.

**Goal:** Fix the structural risks found during critical review so implementation does not require major rework later.

**Architecture:** The revised pipeline is quality-first, split-first, calibration-before-fusion, confounding-aware, and shift-aware. It treats multimodal fusion as an experiment that must beat cough-only, not as an assumed improvement.

**Tech Stack:** Same as the master plan: Python, librosa/soundfile, scikit-learn, PyTorch, xgboost/lightgbm, Streamlit or Gradio.

---

## 1. Verdict On Gemini Critique

The critique is useful but overstated in places.

Valid and must change:

1. Quality filtering cannot wait until Week 6.
2. Fusion should not average uncalibrated probabilities as the final method.
3. CNN logits must be saved before temperature scaling.
4. Demo integration should start early, not in Week 7.
5. Missing modalities require explicit cohort design before training.
6. Confounding cannot be handled by just mentioning symptoms.
7. Feature selection is needed for high-dimensional classical features.
8. Cross-dataset validation needs normalization and careful interpretation.
9. Fixed crop/pad should happen after event-focused trimming/segmentation.
10. Uniform fusion should be a baseline only, not the main fusion method.

Overstated but still useful:

1. Temporal drift is only possible if recording dates are reliable. If dates are not usable, the mandatory pillar should become "distribution shift" rather than strictly "temporal drift."
2. Propensity score matching is not always required for a BTech project, but subgroup and matched sensitivity analysis should be attempted when metadata supports it.
3. Domain-adversarial alignment is too heavy for the core BTech scope. CMVN/source-trained standardization and honest direct-transfer reporting are enough.
4. fmin/fmax clipping concerns are valid, but the solution is an ablation/default change, not overclaiming respiratory biomarker coverage.

## 2. Revised Project Claim

Replace:

```text
drift-aware multimodal COVID/respiratory screening
```

With:

```text
reliability-aware and shift-aware multimodal respiratory audio screening
```

Use "temporal drift" only if recording dates are sufficiently complete.

Final defensible novelty:

```text
A reliability-first respiratory audio screening pipeline that performs leakage-safe participant splitting, early audio-quality/event filtering, feature-selection-controlled baselines, branch-level calibration before fusion, uncertainty rejection, subgroup/confounding sensitivity checks, and at least one distribution-shift evaluation.
```

## 3. Revised Pipeline Order

Old risky order:

```text
metadata -> features -> ML -> CNN -> fusion -> calibration -> quality
```

New order:

```text
metadata audit
label audit
modality availability audit
participant split
audio quality and event segmentation
feature extraction
feature selection
dummy and classical baselines
CNN baselines with saved logits
branch-level calibration
calibrated fusion
uncertainty/rejection
quality and subgroup sensitivity
distribution-shift evaluation
demo and report
```

## 4. Quality Must Move To Stage 1

Quality analysis is no longer Week 6. It is mandatory before any model training.

Stage 1 outputs:

```text
data/processed/metadata_clean.csv
data/processed/audio_quality.csv
data/interim/split_manifest.csv
data/interim/modality_availability.csv
reports/tables/quality_summary.csv
reports/figures/quality_by_modality.png
```

Training must support three modes from the beginning:

```text
all_samples
quality_ok_only
quality_flag_as_features
```

This prevents re-extracting and retraining everything after discovering corrupted or silent audio.

## 5. Event-Focused Preprocessing Replaces Naive Crop/Pad

Naive global crop/pad is demoted.

Revised preprocessing:

1. Load audio.
2. Convert to mono.
3. Resample.
4. Trim leading/trailing silence.
5. Compute RMS envelope.
6. For cough: isolate high-energy cough-event windows.
7. For breath/speech: use silence trimming and center crop after active-region detection.
8. Only then crop/pad to fixed spectrogram length.

Store:

```text
event_start_sec
event_end_sec
event_duration_sec
active_audio_ratio
segmentation_method
```

Minimum implementation:

```text
energy_trim_only
```

Strong implementation:

```text
rms_event_window for cough
energy_trim plus active-region center crop for breath/speech
```

## 6. Frequency Range Revision

Old default:

```text
fmin = 125
fmax = 7500
```

New default at 16 kHz:

```text
fmin = 20
fmax = 8000
```

Reason:

- Preserve lower cough/breath energy.
- Avoid unnecessarily cutting the top of the available 16 kHz signal.

Optional ablation:

```text
20-8000 Hz vs 125-7500 Hz
```

Do not claim clinical biomarker coverage from this choice. Present it as a conservative signal-preservation default.

## 7. Feature Selection Is Mandatory For Classical ML

Feature extraction can still generate the 249-feature MFCC/acoustic vector, but classical models must not train on the full raw set without a feature-selection step.

Feature-selection ladder:

1. Remove constant and near-constant features.
2. Remove highly correlated features with absolute correlation above 0.95.
3. Standardize features using train split only.
4. Try SelectKBest with mutual information.
5. Try PCA retaining 95 percent variance.
6. Try RFECV only if runtime is manageable.

Report:

```text
number_of_features_before_selection
number_of_features_after_selection
selection_method
validation_AUROC
validation_AUPRC
test_AUROC
test_AUPRC
```

## 8. Calibration Must Happen Before Fusion

Old sequence:

```text
fusion -> calibration
```

New sequence:

```text
train modality model
save validation predictions
save test predictions
calibrate each modality on validation split
fuse calibrated modality probabilities
optionally calibrate final fused probability
```

For CNN:

```text
save logits_validation.csv
save logits_test.csv
fit temperature on validation logits
transform test logits
```

For tree/classical models:

```text
save raw validation probabilities
save raw test probabilities
fit Platt/isotonic on validation probabilities
transform test probabilities
```

Fusion methods:

1. Best single modality.
2. Uniform mean of calibrated probabilities.
3. Validation-AUPRC weighted calibrated probabilities.
4. Missing-modality aware weighted fusion.
5. Optional validation stacking if validation cohort is large enough.

Uniform averaging is only a baseline.

## 9. Missing-Modality Cohorts Must Be Explicit

Create `modality_availability.csv` before modeling:

```text
participant_id
has_cough
has_breath
has_speech
n_cough
n_breath
n_speech
complete_case
available_modalities
```

Training cohorts:

```text
cough_cohort: participants with cough
breath_cohort: participants with breath
speech_cohort: participants with speech
complete_fusion_cohort: participants with cough+breath+speech
partial_fusion_cohort: participants with at least two modalities
```

Evaluation rules:

1. Compare cough-only on cough cohort.
2. Compare complete-case fusion against cough-only restricted to the same complete-case participants.
3. Compare missing-aware fusion on partial-fusion cohort.
4. Never compare metrics from different participant sets without saying so.

This prevents the train/test leak and missing-array problem.

## 10. Confounding Analysis Is Now Required

Metadata permitting, report subgroup metrics by:

```text
age bucket
gender
symptom presence
recording country/region if available
quality flag
```

Minimum confounding check:

```text
train audio-only model
train metadata-only model using age/gender/symptom fields
compare audio-only vs metadata-only vs audio+metadata on same split
```

If the metadata-only model performs strongly, the report must state that confounding risk is high.

Matched sensitivity analysis:

```text
Create a reduced matched subset where positive and negative groups are balanced by age bucket, gender, and symptom presence when feasible.
Evaluate the best audio model on this matched subset.
```

If matching leaves too few samples, report that matching was not statistically reliable and provide subgroup metrics instead.

## 11. Cross-Dataset Evaluation Needs Normalization

Direct Coswara-to-COUGHVID testing is still useful, but it must be framed correctly.

Run order:

1. Train source model on Coswara cough.
2. Evaluate on Coswara test.
3. Evaluate direct transfer on COUGHVID.
4. Evaluate with train-fitted feature standardization.
5. Evaluate with CMVN-style cepstral normalization if implemented.

Rules:

- Fit all scalers on source training data only.
- Never fit normalization using target test labels.
- Report direct transfer and normalized transfer separately.
- If performance collapses, treat it as evidence of dataset shift, not project failure.

## 12. Demo Integration Starts Early

Do not wait until Week 7.

Demo skeleton starts after Stage 1:

```text
audio upload
waveform plot
quality flag
active event region plot
spectrogram plot
placeholder prediction card
research disclaimer
```

Model integration happens later:

```text
load trained model
run same preprocessing function as training
show calibrated probability
show uncertainty warning
```

Owner rule:

Person 3 can build UI, but Person 1 and Person 2 must own backend integration points:

```text
preprocess_for_inference(audio_file, modality)
predict_with_model(processed_audio, modality)
calibrate_probability(raw_output, modality)
```

## 13. Revised 8-Week Timeline

### Week 1: Data, Quality, Splits, Demo Skeleton

- Build Coswara index.
- Clean labels.
- Build modality availability table.
- Create participant-level splits.
- Run audio quality analysis.
- Add energy/event segmentation.
- Create demo skeleton with upload, waveform, quality flag, spectrogram.

Gate:

```text
No model training starts until split_manifest.csv and audio_quality.csv exist.
```

### Week 2: Features And Feature Selection

- Extract MFCC/acoustic features from event-focused audio.
- Extract log-mel spectrogram references.
- Implement feature-selection ladder.
- Train dummy baselines.
- Train first classical baselines.

Gate:

```text
At least one classical model beats dummy baseline on validation AUROC/AUPRC.
```

### Week 3: CNN And Prediction Artifacts

- Train compact CNN for cough.
- Save raw logits.
- Extend to breath and speech if stable.
- Produce validation/test prediction CSVs.

Gate:

```text
Every model must write reusable prediction artifacts before calibration/fusion.
```

### Week 4: Branch Calibration

- Calibrate each modality model separately.
- Produce reliability diagrams per branch.
- Compare raw vs calibrated metrics.
- Add uncertainty-rejection curves.

Gate:

```text
Fusion cannot start until branch-level calibrated probabilities exist.
```

### Week 5: Fusion And Complete-Case Evaluation

- Run best-single-modality baseline.
- Run uniform calibrated fusion.
- Run validation-weighted calibrated fusion.
- Run missing-modality-aware fusion.
- Compare fusion against cough-only on identical participants.

Gate:

```text
Fusion is accepted only if it improves or provides reliability insight. Otherwise cough-only plus calibration remains the main model.
```

### Week 6: Confounding, Quality Sensitivity, Shift

- Run all vs quality-ok-only comparison.
- Run metadata-only baseline if metadata supports it.
- Run subgroup metrics.
- Run matched sensitivity if sample size permits.
- Run temporal/cross-dataset/quality shift analysis.

Gate:

```text
The final report must include at least one shift analysis and one confounding-risk analysis.
```

### Week 7: External Validation, Explainability, Demo Integration

- Try COUGHVID external cough validation.
- Add feature importance/SHAP or simpler model explanations.
- Integrate best calibrated model into demo.
- Finalize result tables.

Gate:

```text
Demo must run with the same preprocessing code used during training.
```

### Week 8: Report, Slides, Reproducibility

- Final report.
- Slides.
- Reproducibility run.
- Demo screenshots.
- Limitations slide.

## 14. Revised Acceptance Criteria

The project is strong if it produces:

1. Dataset audit.
2. Quality audit before training.
3. Participant-level split verification.
4. Modality availability table.
5. Event-focused preprocessing.
6. Feature selection.
7. Classical ML baseline.
8. CNN baseline.
9. Raw and calibrated branch metrics.
10. Fusion compared fairly against cough-only.
11. Uncertainty rejection.
12. Quality sensitivity.
13. Confounding-risk analysis.
14. One distribution-shift analysis.
15. Demo using the real preprocessing and model path.

## 15. Report Language Changes

Do not say:

```text
Our fusion model improves COVID detection.
```

Say:

```text
We evaluate whether multimodal fusion improves respiratory-audio screening reliability under participant-level splits, calibration, quality filtering, and subgroup sensitivity checks.
```

Do not say:

```text
Our model detects disease-specific biomarkers.
```

Say:

```text
The model learns acoustic patterns associated with dataset labels, but confounding and dataset-shift risks limit clinical interpretation.
```

