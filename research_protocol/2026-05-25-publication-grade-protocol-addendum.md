---

# Publication-Grade Addendum: Transactions-Level Research Protocol

Last updated: 2026-05-25

This addendum appends the original Drive plan with the stronger publication-oriented direction discussed after reviewing the risks raised by recent literature. The goal is no longer only a strong BTech demo. The goal is to produce a manuscript-quality experimental study that could be submitted to a strong biomedical signal processing / health informatics venue if results are rigorous.

## A. Revised Paper Positioning

Weak framing to avoid:

> COVID-19 detection from cough using machine learning.

Stronger framing:

> A quality-calibrated, externally validated, multimodal respiratory-audio framework for evaluating the reliability and failure modes of COVID-style audio screening under crowdsourced recording conditions.

Working manuscript title:

**Q-CalFuse: Quality-Calibrated Multimodal Fusion for Cross-Dataset Reliability Analysis of COVID-19 Respiratory Audio Screening**

Alternative conservative title:

**Reliability Limits of Multimodal Respiratory Audio Screening Under Quality Variation, Calibration Error, and Cross-Dataset Shift**

## B. Why This Is Worth Publication Attempt

High-end venues are unlikely to care about another plain cough-CNN accuracy paper. They may care if the work answers these harder questions:

1. Does respiratory audio still work under external validation?
2. Does audio add information beyond symptoms and metadata?
3. Does multimodal fusion help, or can weaker modalities damage the best branch?
4. Do quality controls and calibration improve trustworthy decision support?
5. Does the model fail differently across datasets, demographics, symptoms, and audio-quality strata?
6. Can we quantify uncertainty and abstain on low-quality or ambiguous recordings?

This makes the paper a reliability study, not a clinical diagnostic claim.

## C. Core Literature Drawbacks And Our Response

| Literature drawback | Threat to publication | Required response in our work |
|---|---|---|
| Audio models may learn confounding signals instead of COVID-specific audio biomarkers | Reviewers may reject high accuracy as spurious | Add metadata-only and symptom-only baselines; subgroup analysis by age, gender, symptoms, quality; optional matched sensitivity analysis if metadata supports it |
| Audio classifiers may not outperform symptom checkers | Audio-only claim becomes weak | Compare audio-only, metadata/symptom-only, and audio+metadata. Report when audio adds no value |
| Cross-dataset performance often drops | Internal accuracy is not convincing | Mandatory train-on-one-dataset, test-on-another evaluation for cough; report direct transfer and normalized transfer |
| Crowdsourced audio has variable quality | Model may learn microphone/noise artifacts | Early quality audit, quality-filtered ablation, quality-weighted fusion, low-quality rejection |
| Labels are noisy or self-reported in public datasets | Ground truth may be unreliable | Dataset-specific label audit; PCR-referenced subsets when available; noisy-label sensitivity analysis; avoid clinical claims |
| Multimodal data have missing channels | Fusion can silently change cohorts | Explicit availability table; complete-case and available-case results; missing-aware fusion |
| Uncalibrated probabilities are unsafe for screening | Confidence cannot be trusted | Per-branch Platt/isotonic/temperature calibration; report ECE, Brier, NLL, reliability diagrams |
| Many studies overclaim accuracy | Paper may look like prior weak work | Use AUROC, AUPRC, sensitivity/specificity, calibration, bootstrap confidence intervals, external validation, and limitations |

## D. Final Research Questions

RQ1. Under participant-level splitting, how do cough, breath, and speech modalities compare for COVID-style respiratory-audio screening?

RQ2. Does calibrated multimodal fusion improve over the best single modality, or does it suffer from modality drag?

RQ3. How much does audio quality affect discrimination, calibration, and uncertainty?

RQ4. How well do models trained on one dataset transfer to another public respiratory-audio dataset?

RQ5. Does audio add predictive value beyond symptoms/demographics, or are apparent gains explained by confounding?

RQ6. Can uncertainty/quality-aware abstention improve reliability at reduced coverage?

## E. Hypotheses

H1. Cough will usually be the strongest single modality, while breath and speech will be more variable.

H2. Naive uniform fusion will not consistently beat the best single modality; calibrated and validation/quality-weighted fusion should be more stable.

H3. Quality filtering or quality-aware weighting will improve calibration and may improve AUPRC, especially on noisy external data.

H4. External validation will show a large performance drop compared with internal validation.

H5. Metadata/symptom-only baselines will be competitive in some settings; therefore audio contribution must be measured, not assumed.

H6. Abstention on low-confidence or low-quality samples will increase AUROC/AUPRC/calibration among retained samples while reducing coverage.

## F. Datasets And Roles

Minimum publication attempt:

1. **Coswara**
   - Role: primary multimodal dataset.
   - Modalities: cough, breath, vowel/speech/counting.
   - Use: internal multimodal experiments, quality analysis, calibration/fusion.

2. **COUGHVID**
   - Role: public external cough-only validation.
   - Use: train Coswara cough -> test COUGHVID cough; train COUGHVID subset -> test Coswara cough.
   - Caveat: label noise and expert-agreement limitations must be discussed.

Strong version if access arrives:

3. **COVID-19 Sounds / Cambridge**
   - Role: external multimodal validation and possible drift analysis.
   - Use: cross-dataset multimodal validation; date-based shift if timestamps exist.
   - Risk: access agreement may delay work.

Optional if access/ethics permit:

4. **UK COVID-19 Vocal Audio Dataset**
   - Role: strong confounding/symptom baseline analysis.
   - Use: reproduce or compare against symptom-checker concerns.

Do not make Cambridge or UK Vocal required for the basic deliverable. They strengthen the paper but cannot be the only path.

## G. Method: Q-CalFuse

Q-CalFuse is the named framework. It is not just a model; it is a reliability-aware evaluation and fusion pipeline.

### G1. Quality And Event Module

Inputs:

- raw waveform,
- modality label,
- participant metadata.

Outputs:

- duration,
- sample rate,
- silence ratio,
- clipping ratio,
- RMS statistics,
- zero crossing rate,
- spectral centroid/flatness,
- SNR proxy,
- active event start/end,
- active audio ratio,
- quality flag.

Rules:

- Quality audit must happen before features/models.
- Do not delete bad samples first. Keep `quality_flag` and compare all vs filtered.
- Event-aware trimming must happen before fixed-length crop/pad.

### G2. Feature Families

Try all publication-relevant feature families in controlled ablation, not randomly:

1. MFCC + delta + delta-delta statistical features.
2. Handcrafted spectral features.
3. OpenSMILE/eGeMAPS or ComParE if installation is feasible.
4. Log-mel spectrograms for CNN.
5. Pretrained audio embeddings if feasible: YAMNet, VGGish, PANNs, AST, wav2vec-style audio embeddings.

The report must say which feature families were successfully run and which were not due to runtime/access constraints.

### G3. Branch Models

For each modality:

- dummy most-frequent and dummy stratified,
- logistic regression,
- SVM if runtime permits,
- random forest,
- XGBoost/LightGBM,
- compact log-mel CNN,
- pretrained embedding + shallow classifier,
- optional transformer only if time remains.

Do not train a large transformer from scratch.

### G4. Calibration

Calibration happens before fusion:

- Platt scaling for classical probabilities,
- isotonic regression when validation size supports it,
- temperature scaling for neural logits,
- reliability diagram,
- ECE,
- Brier score,
- NLL.

Save raw validation/test predictions before calibration.

### G5. Fusion Methods

Compare in this exact order:

1. best single modality,
2. uniform mean of calibrated branch probabilities,
3. validation-AUPRC weighted calibrated fusion,
4. quality-weighted calibrated fusion,
5. learned stacking/meta-fusion if validation cohort is large enough,
6. missing-modality aware version of each fusion method.

Uniform fusion is a baseline only. The paper should explicitly report if fusion is worse than cough-only.

### G6. Abstention

Abstention rules:

- reject low-quality recordings,
- reject ambiguous calibrated probability interval, e.g. 0.4 to 0.6,
- reject high uncertainty or high branch disagreement,
- report performance-vs-coverage curves.

This is important for screening reliability.

## H. Mandatory Experiments

### Experiment 1: Dataset Audit

Outputs:

- participants per dataset,
- recordings per modality,
- positive/negative/unknown counts,
- missing modality counts,
- metadata coverage,
- quality distribution.

### Experiment 2: Internal Baselines

For Coswara:

- cough-only,
- breath-only,
- speech/vowel-only,
- each branch with classical ML,
- CNN branch where feasible.

### Experiment 3: Feature Family Ablation

Compare:

- MFCC/acoustic,
- OpenSMILE/eGeMAPS if available,
- log-mel CNN,
- pretrained embeddings.

### Experiment 4: Quality Ablation

Compare:

- all samples,
- quality-ok-only,
- quality flags as features,
- quality-weighted fusion.

### Experiment 5: Calibration Ablation

Compare:

- raw probabilities,
- Platt,
- isotonic,
- temperature for CNN.

Report calibration metrics, not only AUROC/AUPRC.

### Experiment 6: Multimodal Fusion Ablation

Compare:

- best single modality,
- uniform fusion,
- validation-weighted fusion,
- quality-calibrated fusion,
- complete-case vs available-case.

### Experiment 7: Cross-Dataset Evaluation

Minimum:

- train Coswara cough -> test COUGHVID cough,
- train COUGHVID cough subset -> test Coswara cough.

If Cambridge access arrives:

- train Coswara -> test Cambridge,
- train Cambridge -> test Coswara,
- multimodal cross-dataset if modality mapping is feasible.

### Experiment 8: Confounding Analysis

Compare:

- audio-only,
- metadata/symptom-only,
- audio + metadata,
- quality-only baseline.

If metadata supports it:

- stratified metrics by age/gender/symptom status,
- matched sensitivity analysis,
- subgroup calibration.

### Experiment 9: Drift / Shift Analysis

If dates exist:

- train early, test later,
- month/period-wise performance,
- MMD/Wasserstein feature shift,
- quality distribution over time.

If dates are sparse:

- do not claim temporal drift;
- report distribution shift across datasets/modalities instead.

### Experiment 10: Explainability

Use cautiously:

- feature importance / SHAP for classical models,
- spectrogram saliency/Grad-CAM for CNN,
- modality contribution and branch disagreement plots.

Do not claim heatmaps prove biological causality.

## I. Metrics And Statistical Reporting

Main metrics:

- AUROC,
- AUPRC,
- balanced accuracy,
- F1,
- sensitivity,
- specificity,
- sensitivity at fixed specificity,
- Brier score,
- ECE,
- NLL,
- coverage for abstention.

Statistical reporting:

- bootstrap 95% confidence intervals for main metrics,
- paired bootstrap or DeLong-style comparison if feasible,
- report sample size for every subgroup,
- mark subgroup results with n < 20 as descriptive only.

Accuracy must never be the primary metric.

## J. Required Tables And Figures For Paper

Tables:

1. Literature comparison table.
2. Dataset statistics and metadata coverage.
3. Audio quality distribution by dataset/modality.
4. Internal modality performance.
5. Feature family ablation.
6. Calibration ablation.
7. Fusion ablation.
8. Cross-dataset validation.
9. Confounding and subgroup analysis.
10. Abstention coverage-vs-performance.

Figures:

1. End-to-end Q-CalFuse architecture.
2. Dataset/modality availability diagram.
3. Example waveform and event-window spectrogram.
4. Reliability diagrams before/after calibration.
5. Fusion comparison plot.
6. Cross-dataset shift visualization: UMAP/t-SNE/MMD heatmap.
7. Quality-vs-error plot.
8. Coverage-vs-performance curve.

## K. Target Venue Strategy

Ambitious targets:

- IEEE Journal of Biomedical and Health Informatics.
- IEEE Transactions on Biomedical Engineering.
- Journal of Biomedical Informatics.
- Computers in Biology and Medicine.

More realistic strong targets:

- Biomedical Signal Processing and Control.
- Computer Methods and Programs in Biomedicine.
- Scientific Reports.
- IEEE Access.

Submission strength depends on actual external validation and whether Q-CalFuse improves reliability over baselines.

## L. Acceptance-Critical Success Criteria

For a high-end paper, at least one of these must be true:

1. Q-CalFuse improves calibrated cross-dataset reliability over naive baselines.
2. Quality-aware calibration/fusion significantly reduces ECE/Brier while preserving useful AUROC/AUPRC.
3. The study produces a strong, carefully controlled negative result showing public COVID audio benchmarks fail under external/confounding tests.
4. Multimodal fusion is shown to help only under specific quality/calibration conditions, giving a useful methodological finding.

If none of these happens, the work is still a strong BTech project but likely not a high-end Transactions paper.

## M. Claims To Make And Claims To Avoid

Allowed claims:

- reliability-aware respiratory audio screening prototype,
- cross-dataset robustness evaluation,
- calibration and uncertainty analysis,
- quality-aware fusion framework,
- failure-mode analysis of public crowdsourced datasets.

Avoid:

- clinical diagnosis,
- COVID biomarker discovery,
- hospital deployment readiness,
- biological causality from spectrograms,
- claiming fusion is better without ablation evidence.

## N. Immediate Implementation Changes Needed

The current hidden implementation already supports the notebook-first Coswara workflow. To move toward publication-grade work, add next:

1. Dataset adapters for COUGHVID and Cambridge-style metadata.
2. Cross-dataset split/evaluation scripts.
3. Bootstrap confidence interval utilities.
4. Metadata/symptom-only baseline model.
5. Quality-weighted fusion method.
6. Abstention curve utility.
7. Optional pretrained embedding extraction module.
8. Paper table generator that exports every result table reproducibly.

## O. Professor-Facing One-Minute Pitch

> We are not building another cough-CNN accuracy project. We are studying whether COVID-style respiratory audio screening is reliable under real public-dataset conditions. Our framework combines quality-aware preprocessing, participant-safe splits, calibrated branch models, multimodal fusion, external validation, and confounding checks. This directly addresses the main criticisms in recent literature: confounding, poor calibration, quality variation, and cross-dataset drift. If performance improves, we have a method contribution. If it fails under external validation, we still have a valuable reliability/failure-mode study.

## P. Final Decision

Proceed with the current implementation, but upgrade the research target:

- BTech deliverable: notebook-first working Coswara pipeline and demo.
- Publication deliverable: Q-CalFuse reliability study using Coswara plus at least COUGHVID, with Cambridge/UK datasets added if access arrives.

The publication manuscript should be written around reliability, calibration, external validity, and failure modes, not around raw accuracy.
