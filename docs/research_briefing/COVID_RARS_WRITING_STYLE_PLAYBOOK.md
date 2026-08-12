# Covid-RARS Writing Style and Venue Playbook

This playbook explains how to write and present Covid-RARS as a serious biomedical AI study rather than an engineering log or a failed SOTA attempt.

## Core Writing Rule

Do not start with failure. Start with strength, then reveal why strength is not enough.

Weak opening:

```text
Our model does not beat SOTA and fails on COUGHVID.
```

Strong opening:

```text
We developed a multimodal respiratory-audio pipeline whose validation-selected equal-weight fusion reaches 0.895 AUROC under internal participant-level evaluation. An exploratory logistic stack reaches 0.897. We then show that favorable source performance is not sufficient evidence for deployment because temporal validation, metadata-confounding audits, calibration, and COUGHVID transfer reveal major instability.
```

## Recommended Title Direction

The title must sound strong, not apologetic.

Best current title:

```text
Temporal and Cross-Dataset Shifts Reveal Reliability Limits in Multimodal COVID-19 Respiratory-Audio Modeling
```

Why this works:

- "Temporal and Cross-Dataset Shifts" names the two principal reliability stresses.
- "Reveal Reliability Limits" states the observed finding without claiming clinical validation or a causal mechanism.
- "Multimodal COVID-19 Respiratory-Audio Modeling" captures the source workflow while leaving the cough-only external endpoint to the abstract and methods.

Alternative for a medical/digital-health journal:

```text
When High Internal Accuracy Fails: A Temporal and External Validation Audit of COVID-19 Respiratory-Audio Screening Models
```

Alternative for signal-processing venue:

```text
Validation Fragility in COVID-19 Respiratory-Audio Classification: Evidence from Multimodal Features, Transformers, and External Transfer
```

Avoid titles like:

- "A SOTA COVID Audio Classifier"
- "COVID Detection Using Machine Learning"
- "Failure of COVID Audio Models"
- "A Novel Framework for COVID Audio Detection" unless the framework is clearly defined as a validation/audit framework.

## Abstract Pattern

Use this structure:

1. **Problem**: COVID respiratory-audio papers report high internal metrics, but deployment validity is uncertain.
2. **Gap**: Many evaluations do not jointly test temporal drift, metadata shortcuts, calibration, and external transfer.
3. **Method**: We build a multimodal Coswara pipeline with strong acoustic, OpenSMILE ComParE+IS10, WavLM transformer, CNN-BiGRU, model fusion, and strict validation.
4. **Results**: Primary internal `0.895` AUROC (`0.897` exploratory stack); time-stratified `0.849`; temporal `0.698`; COUGHVID external `0.523-0.543`; WavLM `0.484`; CNN-BiGRU `0.548`; metadata-only `0.964`; feature stability `0.074`.
5. **Conclusion**: High internal audio metrics should not be interpreted as deployable screening performance without temporal/external validation.

Do not use code file names in the abstract.

## Contribution Bullets

Use these contribution bullets:

1. We construct a multimodal COVID respiratory-audio benchmark pipeline combining strong acoustic descriptors, OpenSMILE ComParE 2016, IS10, model selection, and multimodal probability fusion.
2. We evaluate the pipeline across a validation ladder: existing participant split, time-stratified participant split, early-to-late temporal split, and COUGHVID external cough transfer.
3. We show that strong internal performance (primary `0.895` AUROC; exploratory stack `0.897`) degrades under temporal and external validation, with COUGHVID cough transfer collapsing near chance across measured audio-summary, CNN-BiGRU, and WavLM transformer branches.
4. We quantify shortcut-learning mechanisms using metadata-only models, permutation importance, shuffle-label sanity checks, feature-selection stability, support-overlap diagnostics, calibration, and decision-curve analysis.
5. We test whether audio adds incremental value beyond metadata/symptoms and show that any gain is sample-limited and not yet statistically secure.

## Manuscript Structure

### Introduction

Write the introduction around a clinical AI reliability problem:

- Respiratory audio is attractive because it is low-cost, remote, and scalable.
- High internal metrics have been reported in COVID-audio literature.
- But public health datasets are affected by collection wave, symptoms, device, label construction, and recruitment protocol.
- Therefore, internal validation alone can create over-optimistic estimates.
- Our paper builds a strong pipeline and then audits its validity.

Do not write:

```text
We propose a novel ML classifier that detects COVID-19.
```

Write:

```text
We evaluate whether high-performing respiratory-audio COVID-19 classifiers remain reliable under temporal, confounding, calibration, and external-transfer stress tests.
```

### Methods

Recommended subsections:

1. Datasets and label construction
2. Quality audit and preprocessing
3. Acoustic feature extraction
4. Feature selection and model bank
5. Multimodal fusion
6. Validation ladder
7. Metadata confounding and shortcut checks
8. External transfer and recalibration
9. Statistical uncertainty and decision-curve analysis

Important wording:

- Feature selection must be described as train-only.
- COUGHVID must be described as cough-only external transfer.
- WavLM must be described as a transformer branch.
- Incremental metadata+audio must be described as exploratory because aligned sample size is small.

### Results

Recommended order:

1. Strong internal performance
2. Validation ladder degradation
3. Cough-only external transfer and deep/transformer transfer
4. Metadata shortcut evidence
5. Calibration, operating points, decision curves
6. Feature non-stationarity and support overlap
7. Incremental audio+metadata analysis

Do not lead with COUGHVID collapse alone. Lead with internal strength first.

### Discussion

Discussion should say:

- The pipeline can fit Coswara well.
- The deployment problem is not solved by stronger internal models.
- Metadata and collection context explain why internal respiratory-audio results can look strong.
- External transfer collapse is consistent across model families.
- The work supports stricter reporting standards for respiratory-audio AI.

Do not say:

- "Audio is useless."
- "All prior work is invalid."
- "Our model is clinically deployable."

## Figure Plan

Use figures that look like biomedical AI evidence, not decorative AI diagrams.

| Figure | Content | Why it matters |
|---|---|---|
| Figure 1 | Dataset and validation ladder pipeline | Shows this is a reliability study |
| Figure 2 | Internal -> time-stratified -> temporal -> external performance ladder | Main result visually |
| Figure 3 | External transfer by model family: measured audio-summary, WavLM, CNN-BiGRU | Shows collapse is architecture-independent |
| Figure 4 | Metadata shortcut/permutation importance | Shows mechanism |
| Figure 5 | Calibration/decision-curve evidence | Shows clinical screening weakness |
| Supplement | Feature-selection stability, subgroup/equity, support overlap | Reviewer defense |

Avoid:

- Decorative blobs, generic AI icons, or ungrounded architecture art.
- Overcomplicated diagrams with code names.
- Figures that only repeat one table without interpretation.

## Table Plan

| Table | Purpose |
|---|---|
| Dataset table | Coswara/COUGHVID modality, label source, sample counts, limitations |
| Feature/model table | Strong acoustic, ComParE+IS10, WavLM, CNN-BiGRU, fusion |
| Validation ladder table | Main internal/time/temporal/external results |
| Cough-only external family table | Fair COUGHVID transfer comparison |
| Metadata shortcut table | Metadata-only, symptoms-only, shuffle sanity |
| Reviewer evidence table | CI, calibration, DCA, support overlap, feature stability |
| Literature comparison table | Internal paper numbers separated from strict validation numbers |

## Venue Strategy

Use a two-tier strategy so the positioning shows ambition and realism.

### Reach target

**npj Digital Medicine**

Why it can fit:

- The journal covers digital/mobile health, AI, informatics, and implementation-oriented medical technology.
- Our study fits if framed as clinical AI reliability and benchmark-validity audit.

Risk:

- Very selective.
- Will desk-reject if written as a routine classifier or if the novelty sounds like "we trained LightGBM."

How to pitch:

```text
This is a digital medicine reliability study showing why internal respiratory-audio screening benchmarks can mislead deployment decisions.
```

### Strong realistic targets

**PLOS Digital Health**

- Good fit for open, reproducible digital-health evaluation.
- More receptive to methodological audits than pure leaderboard papers.

**IEEE Journal of Biomedical and Health Informatics**

- Good fit if the paper emphasizes rigorous biomedical AI validation, calibration, DeLong/bootstrap, decision curves, and model-family analysis.
- Needs strong technical writing and clean figures.

**Computers in Biology and Medicine**

- Good fit for computational biomedical validation and signal-processing/AI analysis.
- More realistic than npj if the manuscript is technically solid.

### Backup but still respectable

**Biomedical Signal Processing and Control**

- Strong signal-processing fit.
- Good for respiratory-audio methods, feature extraction, and validation.

**JMIR AI / JMIR Formative Research**

- Good digital-health fit.
- May be less persuasive to readers who only recognize very famous venues, but methodologically appropriate.

## How to Talk About "Fame"

Say this:

```text
Ma'am, we can aim high first, but the manuscript has to match the venue. For npj Digital Medicine, we should not sell it as just a classifier. We should sell it as a digital-health reliability audit with strong internal performance and strict external validation. For IEEE JBHI or Computers in Biology and Medicine, we can emphasize the technical pipeline, uncertainty, calibration, and model-family robustness.
```

## Meeting Q&A

| Question | Answer |
|---|---|
| Why not SOTA? | We have strong internal performance, but universal SOTA is not the honest claim. The stronger publishable claim is that internal SOTA-style numbers are not enough under strict validation. |
| Why should this be accepted if external performance is low? | Because the paper is about reliability and safety. Low external performance is the finding, not a failure, if it is proven rigorously. |
| Did you use transformer? | Yes. WavLM base-plus is a self-supervised transformer. It reached `0.812` internal cough AUROC and `0.484` external COUGHVID AUROC. |
| Why not use HST/AST? | We tested WavLM as a transformer branch. A full HST/AST architecture can be future work, but another architecture is unlikely to remove the need for temporal/external validation. |
| Why not Grad-CAM? | Our final strongest model is feature/fusion based, not a final spectrogram transformer. We used feature importance, metadata shortcut, calibration, DCA, and external validation instead. |
| Why COUGHVID? | It is a public independent cough dataset. It cannot test full multimodal fusion, but it fairly tests cough-to-cough transfer. |
| Does audio add beyond symptoms? | In small aligned subsets, metadata+audio can improve over symptoms-only, but confidence intervals are wide. Over full safe metadata, audio adds little. This is exploratory, not a strong clinical claim. |

## Manuscript Tone

Use:

- "strict validation"
- "deployment-relevant evaluation"
- "shortcut learning"
- "temporal drift"
- "external transfer"
- "calibration and clinical net benefit"
- "benchmark validity"

Avoid:

- "our model failed"
- "fabricated SOTA"
- "just not enough time"
- "we tried everything"
- "AI suggested"
- code file names in main prose

## Final 30-Second Defense

```text
The validation-selected equal-weight source fusion reaches 0.895 AUROC, with an exploratory stack at 0.897. We tested conventional, deep, and transformer branches, then showed that temporal instability, metadata structure, feature non-stationarity, and external dataset shift limit the interpretation of internal COVID-audio performance.
```


