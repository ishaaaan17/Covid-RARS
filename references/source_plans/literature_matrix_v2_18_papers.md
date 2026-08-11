# COVID/Respiratory Audio BTP Literature Matrix V2

Last updated: 2026-05-24

Purpose: strengthen the project with roughly 15 highly relevant papers/datasets and convert the literature into concrete methodology decisions. This file supersedes the shorter literature matrix when writing the final report.

Boundary: respiratory audio, signal processing, ML evaluation, calibration, dataset reliability, and non-diagnostic screening only.

## 18-Paper Evidence Table

| ID | Source | Role In Project | Methodology Impact |
|---|---|---|---|
| P1 | JMIR 2025 drift-adaptive cough-audio framework | Main base paper | Make distribution-shift analysis mandatory |
| P2 | Coswara dataset paper | Primary dataset | Main multimodal dataset; subgroup and quality audit |
| P3 | COVID-19 Sounds NeurIPS 2021 | Large dataset reference | Background only; do not depend on raw access |
| P4 | COUGHVID Scientific Data 2021 | External cough dataset | Optional cross-dataset cough validation |
| P5 | Nature Machine Intelligence 2024 caution paper | Confounding critique | Add subgroup and matched sensitivity analysis |
| P6 | IEEE-BHI/OpenReview 2024 calibration paper | Calibration base | Calibrate each branch before final fusion |
| P7 | HST respiratory-sound paper | Advanced architecture reference | Cite transformer work; implement compact CNN first |
| P8 | SympCoughNet 2025 | Symptom-assisted recent paper | Treat symptoms as confounding/audit metadata |
| P9 | PLOS ONE 2025 speech/vowel paper | Speech/vowel support | Include speech/vowel experiments |
| P10 | ESWA 2026 cross-dataset cough robustness | Cross-dataset robustness | Add RFECV/feature selection and external validation |
| P11 | Audio-Based Screening of Respiratory Diseases, MDPI 2026 | Segmentation framework | Add cough-event segmentation before crop/pad |
| P12 | Sensors 2022 systematic review | Methodology breadth | Use multi-metric reporting and document preprocessing |
| P13 | Deep transfer learning and bottleneck features, CIBM 2022 | Multimodal baseline | Cough may dominate; fusion must be weighted/calibrated |
| P14 | Pay Attention to the Speech, AEJ 2022 | Speech/fusion precedent | Test speech contribution but compare against cough-only |
| P15 | Audio texture analysis, BSPC/Frontiers 2022 | Spectrogram/texture baseline | Optional texture features/future work |
| P16 | Audio feature ranking for sound-based COVID patient detection | Feature ranking | Add feature selection before classical ML |
| P17 | Interspeech 2021 feature selection and explainability | XAI/feature selection | Add feature importance and explainability |
| P18 | COVID-19 cough classification using global smartphone recordings | Real-world recording variability | Add device/noise/domain-shift limitations |

## Key Sources And How We Use Them

### P1. JMIR 2025 Drift-Adaptive Cough-Audio Framework

Link: https://www.jmir.org/2025/1/e66919

Use: main base paper.

Important for our novelty:

- Static cough-audio models degrade over time.
- Distribution shift is not a side note; it is a central reliability issue.
- We should not make "drift-aware" optional if it appears in the title.

Plan consequence:

- Rename the implementation focus to "shift-aware" unless temporal dates are usable.
- Make at least one shift analysis mandatory:
  - temporal shift if recording dates are reliable,
  - cross-dataset shift if COUGHVID label mapping works,
  - quality/subgroup shift if neither external data nor dates are usable.

### P2. Coswara Dataset Paper

GitHub: https://github.com/iiscleap/Coswara-Data

Paper/arXiv: https://arxiv.org/abs/2305.12741

Use: primary dataset.

Important for our novelty:

- Same participants can have cough, breath, and speech-style recordings.
- Metadata enables subgroup and confounding analysis.
- Quality annotations/quality variability matter.

Plan consequence:

- Build modality-availability manifests before training.
- Do not assume every participant has every modality.
- Create separate cohorts:
  - cough cohort,
  - breath cohort,
  - speech cohort,
  - complete-case fusion cohort,
  - partial-case fusion cohort.

### P3. COVID-19 Sounds NeurIPS 2021

Link: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/e2c0be24560d78c5e599c2a9c9d0bbd2-Abstract-round2.html

Use: large dataset reference.

Plan consequence:

- Cite as a major field dataset.
- Do not make raw access required.

### P4. COUGHVID Scientific Data 2021

Paper: https://www.nature.com/articles/s41597-021-00937-4

Zenodo: https://zenodo.org/records/4048312

Use: optional external cough validation.

Plan consequence:

- Treat labels as noisy.
- Normalize source and target consistently.
- Report direct transfer and normalized transfer separately.

### P5. Nature Machine Intelligence 2024 Caution Paper

Link: https://www.nature.com/articles/s42256-023-00773-8

Use: required critique paper.

Important for our novelty:

- Audio models can capture confounders rather than disease-specific acoustic signatures.
- Symptom, age, gender, and sampling differences can inflate metrics.

Plan consequence:

- Add subgroup metrics by age/gender/symptom availability where metadata supports it.
- Add a matched sensitivity analysis if enough positive/negative participants remain after matching.
- Never present unadjusted metrics as clinical proof.

### P6. Confidence-Calibrated Respiratory Screening, IEEE-BHI 2024

Link: https://openreview.net/forum?id=chVymJKep2

Use: calibration base.

Plan consequence:

- Save raw logits for CNN.
- Save raw probabilities for classical models.
- Calibrate each modality branch on validation predictions before fusion.
- Then run fusion on calibrated probabilities.
- Optionally calibrate the final fused score as a second stage, but do not rely only on post-fusion calibration.

### P7. Hierarchical Spectrogram Transformer Respiratory-Sound Paper

arXiv: https://arxiv.org/abs/2207.09529

IEEE: https://ieeexplore.ieee.org/abstract/document/10342847

Use: advanced architecture reference.

Plan consequence:

- Cite as high-end architecture.
- Keep transformer as future work.

### P8. SympCoughNet 2025

Link: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1551298/full

Use: symptom-assisted recent paper.

Plan consequence:

- Do not blindly add symptoms as model inputs.
- Use symptoms for confounding audit and optional comparison only.
- If symptoms are used, report audio-only vs symptoms-only vs audio+symptoms.

### P9. Speech-Based Respiratory Diagnostics, PLOS ONE 2025

Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC12637952/

Use: speech/vowel support.

Plan consequence:

- Include vowel/speech experiments.
- Do not expect speech to dominate cough.

### P10. ESWA 2026 Cross-Dataset Cough Robustness

Link: https://researchoutput.csu.edu.au/en/publications/robust-covid-19-detection-from-cough-sounds-using-deep-neural-dec/

Use: cross-dataset and feature-selection motivation.

Plan consequence:

- Add RFECV or simpler feature-selection alternatives.
- Cross-dataset result should be framed as robustness, not grading-critical success.

### P11. Audio-Based Screening of Respiratory Diseases, MDPI 2026

Link: https://www.mdpi.com/2504-4990/8/3/80

Use: segmentation framework.

Plan consequence:

- Replace naive global crop/pad with:
  1. load audio,
  2. trim leading/trailing silence,
  3. detect cough/event regions where applicable,
  4. build features/spectrograms from event-focused audio,
  5. crop/pad only after event isolation.

### P12. Sensors 2022 Systematic Review

Link: https://www.mdpi.com/1901888

Use: broad methodology review.

Plan consequence:

- Document preprocessing choices.
- Use multiple metrics.
- Avoid comparing our numbers directly to weakly controlled papers.

### P13. Deep Transfer Learning And Bottleneck Features, CIBM 2022

ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0010482521009471

PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8679499/

Use: multimodal baseline.

Plan consequence:

- Cough-only remains the primary baseline.
- Fusion must prove it helps.
- Report negative fusion results honestly.

### P14. Pay Attention To The Speech, AEJ 2022

Link: https://www.sciencedirect.com/science/article/pii/S1110016821005858

PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8397542/

Use: speech/fusion precedent.

Plan consequence:

- Include speech as a meaningful experiment.
- Compare uniform averaging, weighted averaging, and best-single-modality.

### P15. Audio Texture Analysis, BSPC/Frontiers 2022

PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9013601/

Frontiers: https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2022.986293/full

Use: spectrogram/texture baseline.

Plan consequence:

- Optional texture feature extension if time permits.

### P16. Audio Feature Ranking For Sound-Based COVID-19 Patient Detection

Link: https://arxiv.org/abs/2104.07128

Use: feature-selection evidence.

Plan consequence:

- Classical ML must include feature selection:
  - variance threshold,
  - correlation pruning,
  - SelectKBest/mutual information,
  - PCA or RFECV as stronger options.

### P17. Investigating Feature Selection And Explainability For COVID-19 Diagnostics From Cough Sounds

Link: https://www.isca-archive.org/interspeech_2021/avila21_interspeech.html

Use: feature selection and explainability support.

Plan consequence:

- Add feature importance or SHAP after baseline.
- Treat explainability as inspection, not proof of causality.

### P18. COVID-19 Cough Classification Using Global Smartphone Recordings

Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC8213969/

Use: real-world recording variation.

Plan consequence:

- Include device/noise/crowdsourcing limitations.
- Do not expect cross-dataset transfer to be high.

## Final Novelty Statement

The defensible novelty is:

```text
A BTech-scale, reliability-first respiratory audio screening pipeline that combines multimodal Coswara experiments with early quality filtering, leakage-safe participant splitting, branch-level calibration before fusion, feature-selection-controlled classical baselines, event-focused preprocessing, subgroup/confounding sensitivity checks, and at least one distribution-shift analysis.
```

This is stronger than:

```text
CNN detects COVID from cough.
```

## Worthiness Assessment

The topic is worthwhile if the project emphasizes:

- reliability over headline accuracy,
- confounding-aware evaluation,
- calibration before final decision,
- participant-level leakage prevention,
- quality/event segmentation before feature extraction,
- honest cross-dataset or shift limitations.

The topic is weak if it becomes:

- one CNN on cough spectrograms,
- accuracy-only reporting,
- no participant-level split,
- no calibration,
- no quality analysis,
- no confounding discussion.

