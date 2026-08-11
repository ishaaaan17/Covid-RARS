# BTP E2E Detailed Plan: COVID/Respiratory Audio Screening

Last updated: 2026-05-22

> Historical protocol record. The repository layout proposed below predates the
> Covid-RARS restructure and is retained as planning provenance. For current
> executable paths, use `docs/repository/REPOSITORY_MAP.md`.

Working title:

**Confidence-Calibrated and Drift-Aware Multimodal COVID/Respiratory Screening from Cough, Breath, and Speech Sounds**

Short professor-facing title:

**Reliable COVID/Respiratory Screening from Cough, Breath, and Speech Using Calibrated AI**

## 1. Direct Decision

For a 2-month BTech project where the main goal is high grades and publication is secondary, this topic is practical if we avoid the weak framing of "COVID detection from cough with high accuracy."

The stronger and safer framing is:

> Build a reliability-aware respiratory-audio screening pipeline using cough, breath, and speech sounds, then evaluate confidence calibration, uncertainty, audio quality, temporal drift, and cross-dataset generalization.

This gives a clear implementation, a working demo, and strong same-topic fallbacks.

## 2. Main Research Gap

The best recent base paper is the 2025 JMIR drift-adaptive framework. It shows that cough-audio COVID models degrade over time and need adaptation. Its own limitations/future work create a good BTech gap:

1. It detects drift but does not explicitly interpret why drift occurs.
2. It focuses on temporal drift, not cross-dataset drift.
3. It does not fully study interdemographic variability.
4. It is cough-only, while breathing and voice/speech integration is stated as a straightforward future extension.

Our gap:

> A BTech-scale system that combines cough, breath, and speech from Coswara, adds audio-quality filtering and confidence calibration, and evaluates whether the model remains reliable under noisy recordings and cross-dataset shift.

This is better than chasing a high accuracy number.

---

# 3. Base Papers and Relevant Papers

## B1. A Comprehensive Drift-Adaptive Framework for Sustaining Model Performance in COVID-19 Detection From Dynamic Cough Audio Data

- Source/year: Journal of Medical Internet Research, 2025.
- Link: https://www.jmir.org/2025/1/e66919/
- DOI: https://doi.org/10.2196/66919
- Role: **main base paper**.
- Datasets:
  - COVID-19 Sounds.
  - Coswara.
  - COVID-19 Sounds subset used in paper: 1461 English-language cough samples.
  - Coswara filtered subset used in paper: 1996 samples from their accessible Coswara data.
- Method:
  - Cough recordings only.
  - Audio normalization.
  - Leading/trailing silence removal.
  - Mel spectrograms.
  - VGGish-based feature extraction.
  - Chronological development/postdevelopment split.
  - Drift detection using maximum mean discrepancy.
  - CUSUM alerts for drift.
  - Unsupervised domain adaptation.
  - Active learning.
- Reported results:
  - Development baseline AUC-ROC: 69.1% on COVID-19 Sounds.
  - Development baseline AUC-ROC: 66.8% on Coswara.
  - Postdevelopment baseline dropped to AUC-ROC 60.7% and 59.7%, showing drift.
  - Unsupervised domain adaptation improved balanced accuracy by roughly 10%-20% in several COVID-19 Sounds adaptation phases.
  - The paper reports UDA and active learning can maintain model performance closer to development-period benchmarks.
- Limitations/future work:
  - Does not explicitly interpret causes of drift.
  - Does not fully explore cross-dataset drift.
  - Does not fully explore interdemographic variability.
  - Uses cough only; breathing and voice integration is proposed as future work.
- Why it is ideal for our project:
  - 2025 base paper.
  - Clear gap.
  - We can extend at BTech scale using Coswara multimodal audio and calibration.

## B2. Robust COVID-19 detection from cough sounds using deep neural decision tree and forest: A comprehensive cross-datasets evaluation

- Source/year: Expert Systems with Applications, 2026.
- Link: https://researchoutput.csu.edu.au/en/publications/robust-covid-19-detection-from-cough-sounds-using-deep-neural-dec/
- PDF found: https://researchoutput.csu.edu.au/ws/portalfiles/portal/620568799/1-s2.0-S0957417426001491-main.pdf
- DOI: https://doi.org/10.1016/j.eswa.2026.131235
- Role: latest cross-dataset robustness reference.
- Datasets:
  - Cambridge COVID-19 Sounds.
  - Coswara.
  - COUGHVID.
  - Virufy.
  - NoCoCoDa.
- Method:
  - Deep Neural Decision Tree.
  - Deep Neural Decision Forest.
  - Recursive feature elimination with cross-validation.
  - Bayesian hyperparameter optimization.
  - SMOTE oversampling.
  - threshold moving.
- Reported results:
  - AUC values across individual datasets/settings reported around 0.92 to 0.99.
  - Combined dataset DNDF setting reports accuracy around 0.97 and AUC around 0.97.
- Why useful:
  - It supports the importance of cross-dataset evaluation.
- Caveat:
  - It is cough-only and method-heavy.
  - We should not attempt to reproduce everything in 2 months.
- Use in our project:
  - Use as a strong relevant paper and comparison target.
  - Our implementation can be simpler: XGBoost/CNN + calibration + cross-dataset test.

## B3. Confidence-Calibrated Clinical Decision Support System for Reliable Respiratory Disease Screening

- Source/year: IEEE-BHI, 2024.
- Link: https://openreview.net/forum?id=chVymJKep2
- PDF: https://openreview.net/pdf?id=chVymJKep2
- DOI: https://doi.org/10.1109/BHI62660.2024.10913797
- Role: calibration base paper.
- Datasets:
  - Coswara.
  - Cambridge COVID-19 Sounds.
- Method:
  - MFCC features.
  - Deep neural network.
  - Ensemble-based confidence calibration.
  - LIME-style interpretability.
- Reported results:
  - ENCL-DNN AUROC: 0.834 on Coswara.
  - ENCL-DNN AUROC: 0.854 on Cambridge.
  - Expected Calibration Error reduced by 50.0% on Coswara.
  - Expected Calibration Error reduced by 28.74% on Cambridge.
  - Uncertainty accuracy improved from 0.670 to 0.751 on Coswara and from 0.713 to 0.825 on Cambridge.
- Why useful:
  - Strong support for using calibration as a central contribution.
- Gap:
  - Mostly cough-focused and not a full multimodal pipeline.
- Use in our project:
  - Implement ECE, Brier score, reliability diagrams, and rejection based on uncertainty.

## B4. COVID-19 Detection From Respiratory Sounds With Hierarchical Spectrogram Transformers

- Source/year: IEEE Journal of Biomedical and Health Informatics, 2023/2024.
- IEEE link: https://ieeexplore.ieee.org/abstract/document/10342847
- arXiv: https://arxiv.org/abs/2207.09529
- DOI: https://doi.org/10.1109/JBHI.2023.3339700
- Role: deep transformer respiratory-sound reference.
- Datasets:
  - Crowdsourced respiratory sound datasets with cough and breathing sounds.
- Method:
  - Spectrogram representation.
  - Hierarchical Spectrogram Transformer.
  - Local-to-global attention over respiratory sound spectrograms.
- Reported results:
  - Reports over 83% AUC for COVID-19 detection from respiratory sounds.
- Why useful:
  - Strong architecture paper.
- BTech caveat:
  - Transformer should be optional. Start with CNN and XGBoost first.

## B5. SympCoughNet: symptom assisted audio-based COVID-19 detection

- Source/year: Frontiers in Digital Health, 2025.
- Link: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1551298/full
- DOI: https://doi.org/10.3389/fdgth.2025.1551298
- Role: symptom-assisted recent paper.
- Dataset:
  - UK COVID-19 Vocal Audio Dataset.
  - Paper states 72,999 participants and 25,766 PCR-positive cases.
  - Uses cough audio and symptom information.
- Method:
  - Log-mel spectrograms.
  - CNN backbone.
  - Symptom-encoded channel attention.
  - Data augmentation.
  - Voice activity/noise preprocessing.
- Reported results:
  - Accuracy: 89.30%.
  - AUROC: 94.74%.
  - PR: 91.62%.
- Why useful:
  - Recent 2025 paper.
  - Shows symptoms can improve performance.
- Caution:
  - Symptom-assisted models can become confounded if symptoms dominate audio.
  - For our BTP, use symptoms only as analysis metadata or optional comparison, not as the central claim.

## B6. Speech-based respiratory diagnostics: A study on COVID-19 detection with machine learning

- Source/year: PLOS ONE, 2025.
- Link: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0332146
- Role: speech/vowel modality reference.
- Dataset:
  - Coswara.
  - Uses vowel sounds /a/, /e/, and /o/.
- Method:
  - Active speech level normalization using ITU-T P.56.
  - OpenSMILE feature extraction.
  - 1582-dimensional feature vector per recording.
  - Random Forest, SVM, Decision Tree, ANN.
  - Feature selection using ANOVA, chi-square, Information Gain, ReliefF, and Gini index.
- Reported results:
  - Random Forest with ANOVA-selected features performed best.
  - Accuracy around 76.47% for vowel /a/.
  - Accuracy around 75.54% for combined vowels /a/ and /o/.
- Why useful:
  - Supports speech/vowel modality.
- Gap:
  - Speech-only is moderate, so our project should combine speech with cough/breath rather than depend only on speech.

## B7. Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers

- Source/year: Nature Machine Intelligence, 2024.
- Link: https://www.nature.com/articles/s42256-023-00773-8
- Role: critical caution paper.
- Dataset:
  - UK COVID-19 Vocal Audio Dataset.
  - PCR-referenced data.
  - Audio modalities include speech, exhalation, cough.
- Main point:
  - Respiratory-audio classifiers may learn confounding signals or symptoms rather than a causal COVID acoustic signature.
  - Audio models may not outperform simple symptom checkers under realistic evaluation.
- Why useful:
  - This prevents overclaiming.
  - It justifies our reliability-focused framing.
- How we use it:
  - Add symptom/confounding analysis.
  - Avoid claiming clinical diagnosis.
  - Use participant-level and cross-dataset evaluation.

## B8. Coswara: A respiratory sounds and symptoms dataset for remote screening of SARS-CoV-2 infection

- Source/year: Scientific Data, 2023.
- Link: https://www.nature.com/articles/s41597-023-02266-0
- DOI: https://doi.org/10.1038/s41597-023-02266-0
- Role: primary dataset paper.
- Dataset:
  - 2635 individuals.
  - 1819 SARS-CoV-2 negative.
  - 674 positive.
  - 142 recovered.
  - 23,700 recordings.
  - About 65 hours of audio.
  - 9 sound categories:
    - shallow cough,
    - heavy cough,
    - shallow breath,
    - deep breath,
    - vowel /a/,
    - vowel /e/,
    - vowel /o/,
    - normal counting,
    - fast counting.
- Why useful:
  - Best day-one dataset for our BTP.
  - Supports multimodal cough + breath + speech.

## B9. COVID-19 Sounds: A Large-Scale Audio Dataset for Digital Respiratory Screening

- Source/year: NeurIPS Datasets and Benchmarks, 2021.
- Link: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/e2c0be24560d78c5e599c2a9c9d0bbd2-Abstract-round2.html
- PDF: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/e2c0be24560d78c5e599c2a9c9d0bbd2-Paper-round2.pdf
- Role: large multimodal dataset paper.
- Dataset:
  - 53,449 audio samples.
  - Over 552 hours.
  - 36,116 participants.
  - 2106 COVID-positive samples.
  - Modalities: breathing, cough, voice.
- Access:
  - Requires request / data transfer agreement.
- Why useful:
  - Strong external validation if access is granted.
- Risk:
  - Cannot rely on it for day-one coding.

## B10. The COUGHVID crowdsourcing dataset

- Source/year: Scientific Data, 2021.
- Link: https://www.nature.com/articles/s41597-021-00937-4
- Dataset page: https://www.epfl.ch/labs/esl/research/cough-characterization/coughvid/
- Role: external cough validation dataset.
- Dataset:
  - More than 25,000 crowdsourced cough recordings.
  - More than 2800 expert-labeled recordings.
  - Metadata includes COVID status, location, age, gender, and respiratory condition.
- Important validation detail:
  - Expert agreement was low for some labels.
  - Authors provide cough detection and quality tools.
- Why useful:
  - Good for cross-dataset cough validation.
- Risk:
  - Labels are noisy. Treat as external robustness test, not perfect ground truth.

## B11. COVID-19 detection in cough, breath and speech using deep transfer learning and bottleneck features

- Source/year: Computers in Biology and Medicine, 2022.
- DOI: https://doi.org/10.1016/j.compbiomed.2021.105153
- Role: older but important multimodal baseline.
- Datasets:
  - Coswara.
  - ComParE.
  - Sarcos and related respiratory audio data.
- Reported results:
  - Reported AUC about 0.98 for cough.
  - Reported AUC about 0.94 for breath.
  - Reported AUC about 0.92 for speech.
- Use:
  - Baseline/relevant paper, not main base.

## B12. Audio texture analysis of COVID-19 cough, breath, and speech sounds

- Source/year: Biomedical Signal Processing and Control, 2022.
- DOI: https://doi.org/10.1016/j.bspc.2022.103703
- Role: interpretable feature-engineering baseline.
- Dataset:
  - Cambridge COVID-19 Sounds subset.
  - 1141 cough samples.
  - 392 breath samples.
  - 893 speech samples.
- Reported results:
  - Cough 5-class accuracy around 71.7%.
  - Breath 5-class accuracy around 72.2%.
  - Speech binary accuracy around 79.7%.
  - Binary COVID/non-COVID cough result reported very high in paper summaries.
- Use:
  - Feature baseline and hand-crafted feature comparison.

## B13. Machine learning for detecting COVID-19 from cough sounds: An ensemble-based MCDM method

- Source/year: Computers in Biology and Medicine, 2022.
- DOI: https://doi.org/10.1016/j.compbiomed.2022.105405
- Role: multi-dataset classical ML/ensemble reference.
- Datasets:
  - Cambridge.
  - Coswara.
  - Virufy.
  - NoCoCoDa.
- Use:
  - Supports multi-dataset cough evaluation.

## B14. COVID-19 cough classification using machine learning and global smartphone recordings

- Source/year: Computers in Biology and Medicine, 2021.
- Link: https://www.sciencedirect.com/science/article/pii/S0010482521003668
- DOI: https://doi.org/10.1016/j.compbiomed.2021.104572
- Dataset:
  - Coswara.
  - Sarcos/South Africa smartphone cough recordings.
- Reported result:
  - Highest AUC around 0.98 from residual architecture in paper highlights.
- Use:
  - Older baseline, useful for historical literature but not primary.

## B15. Fused Audio Instance and Representation for Respiratory Disease Detection

- Source/year: 2024, open-access article.
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC11479208/
- Role:
  - Broader respiratory disease audio paper.
- Why useful:
  - Supports going beyond COVID-only toward respiratory screening.
  - Uses waveform/spectrogram representations and multi-instance modeling.

## B16. Deep learning-based cough classification using application-recorded sounds

- Source/year: BMC Medical Informatics and Decision Making, 2025.
- Link: https://link.springer.com/article/10.1186/s12911-025-03065-w
- Role:
  - General cough disease classification reference.
- Dataset:
  - App-recorded cough sounds for asthma, COPD, pneumonia, and normal classes.
- Reported result:
  - Cough detection accuracy 0.9883.
  - Disease classification metrics around 0.84-0.87 depending task.
- Use:
  - Not COVID-specific, but supports broader respiratory audio workflow and privacy/annotation concerns.

---

# 4. Papers Needing Full PDF / Extra Access

The plan is strong enough to start, but exact table-level extraction may require PDFs for:

1. Some ScienceDirect papers if institutional full text is blocked.
2. Cambridge COVID-19 Sounds raw data, which needs academic request and data transfer agreement.
3. UK COVID-19 Vocal Audio Dataset raw data, if we want to reproduce SympCoughNet or Nature Machine Intelligence experiments.

If you can download/provide these PDFs/datasets, we can extract exact tables and add them to the report:

- Expert Systems with Applications 2026 full paper, if the institutional PDF link stops working.
- Computers in Biology and Medicine 2022 papers.
- Biomedical Signal Processing and Control 2022 audio texture paper.

---

# 5. Datasets

## Dataset 1: Coswara

- Main use: primary dataset.
- GitHub: https://github.com/iiscleap/Coswara-Data
- HuggingFace mirror: https://huggingface.co/datasets/szzs1693/coswara-data
- Dataset paper: https://www.nature.com/articles/s41597-023-02266-0
- Access: public.
- License: Coswara GitHub indicates Creative Commons Attribution 4.0 International.
- Contents:
  - 2635 individuals.
  - 23,700 recordings.
  - 65 hours of audio.
  - 9 audio categories:
    - shallow cough,
    - heavy cough,
    - shallow breath,
    - deep breath,
    - vowel /a/,
    - vowel /e/,
    - vowel /o/,
    - normal counting,
    - fast counting.
- Metadata:
  - COVID test status.
  - symptoms.
  - age.
  - gender.
  - country/region.
  - comorbidities.
  - recording date.
  - quality score in some packaged versions.
- Why it is best:
  - Public and startable immediately.
  - Multimodal recordings from same participants.
  - Perfect for cough vs breath vs speech fusion.

## Dataset 2: COUGHVID

- Main use: external cough validation.
- EPFL page: https://www.epfl.ch/labs/esl/research/cough-characterization/coughvid/
- Scientific Data paper: https://www.nature.com/articles/s41597-021-00937-4
- Zenodo: https://zenodo.org/record/4048312
- Access: public/open.
- Contents:
  - More than 25,000 cough recordings.
  - More than 2800 expert-labeled recordings.
  - Metadata: COVID status, age, gender, location, respiratory condition.
- Why useful:
  - External validation.
  - Quality/noise analysis.
  - Cough/non-cough filtering.
- Risk:
  - Labels are crowdsourced/noisy.
  - Expert agreement was limited.

## Dataset 3: Cambridge COVID-19 Sounds

- Dataset website: https://www.covid-19-sounds.org/
- NeurIPS dataset paper: https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/e2c0be24560d78c5e599c2a9c9d0bbd2-Abstract-round2.html
- Access: request/license required.
- Contact from JMIR paper: covid-19-sounds@cl.cam.ac.uk
- Contents:
  - 53,449 audio samples.
  - 552+ hours.
  - 36,116 participants.
  - 2106 COVID-positive samples.
  - breathing, cough, voice.
- Use:
  - External validation if access arrives.
  - Drift analysis if timestamp data is available.
- Risk:
  - Not instant-download.

## Dataset 4: UK COVID-19 Vocal Audio Dataset

- Related open paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11211414/
- Nature Machine Intelligence caution paper: https://www.nature.com/articles/s42256-023-00773-8
- Access:
  - Open-access version exists, but full raw access and usage need checking.
- Contents:
  - Large PCR-referenced vocal audio dataset.
  - cough, exhalation, speech.
  - symptom metadata.
- Use:
  - Useful if we want symptom-confounding analysis.
- Risk:
  - More complex governance/access than Coswara.

## Dataset 5: ICBHI 2017 Respiratory Sound Database

- Link: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- Access: public challenge dataset.
- Contents:
  - 920 recordings.
  - 126 subjects.
  - lung auscultation respiratory sounds.
  - diagnoses include COPD, pneumonia, bronchiectasis, bronchiolitis, URTI, LRTI, healthy.
- Use:
  - Same-topic fallback for general respiratory sound screening.
- Risk:
  - Stethoscope/lung sounds, not smartphone cough/speech.

## Dataset 6: DiCOVA

- Link: https://dicova2021.github.io/
- Access: challenge registration/request.
- Use:
  - Additional COVID acoustic evaluation if available.
- Risk:
  - Challenge-specific and smaller.

---

# 6. Final Proposed Contribution

## Core contribution

Develop a COVID/respiratory audio screening pipeline that is:

1. multimodal,
2. confidence-calibrated,
3. quality-aware,
4. participant-split evaluated,
5. tested for cross-dataset drop where possible,
6. demo-ready.

## What is novel enough for BTech

Not novel:

- "CNN detects COVID from cough."

Novel enough for BTech:

- "We compare cough, breath, and speech modalities from the same participants, fuse them, calibrate confidence, reject uncertain/low-quality inputs, and test cross-dataset generalization."

## What can become publishable

The publishable angle is:

> Reliability-aware respiratory audio screening under real-world noisy crowdsourced data.

The paper should emphasize:

- calibration,
- uncertainty,
- quality filtering,
- cross-dataset drop,
- multimodal fusion,
- limitations and confounding.

---

# 7. Same-Topic Fallback Plan

## Plan A: Full strong version

Title:

**Confidence-Calibrated and Drift-Aware Multimodal COVID/Respiratory Screening from Cough, Breath, and Speech Sounds**

Data:

- Coswara cough + breath + speech.
- COUGHVID external cough validation.
- Cambridge optional if access arrives.

Models:

- MFCC/OpenSMILE + RF/SVM/XGBoost.
- Mel spectrogram CNN.
- Late fusion.
- Calibration.
- Quality filtering.
- Cross-dataset test.
- Optional drift analysis.

Deliverable:

- Strong complete BTP plus demo.

## Plan B: If external datasets are hard

Title:

**Confidence-Calibrated Multimodal COVID Screening Using Coswara Cough, Breath, and Speech**

Data:

- Coswara only.

Experiments:

- cough-only,
- breath-only,
- speech-only,
- cough + breath,
- cough + speech,
- cough + breath + speech.

Deliverable:

- Strong modality comparison and fusion study.

## Plan C: If fusion is hard

Title:

**Reliable COVID Screening From Cough Sounds With Calibration and Quality Filtering**

Data:

- Coswara cough.
- COUGHVID cough if possible.

Experiments:

- MFCC + XGBoost.
- Mel spectrogram CNN.
- calibration.
- quality filtering.

Deliverable:

- Good cough-only BTP.

## Plan D: If deep learning is hard

Title:

**Classical Machine Learning for COVID/Respiratory Audio Screening With Calibration and Explainability**

Data:

- Coswara.

Models:

- Logistic Regression.
- SVM.
- Random Forest.
- XGBoost.

Deliverable:

- No-GPU project with good report, calibration, and demo.

## Plan E: Emergency deliverable

Title:

**Reproducible Analysis of COVID/Respiratory Audio Screening Using Coswara**

Data:

- Coswara only.

Deliverable:

- dataset cleaning,
- feature extraction,
- baseline model,
- metrics,
- reliability diagram,
- demo.

This still stays in the same topic.

---

# 8. Exact Implementation Details

## Recommended tech stack

- Python 3.10+.
- pandas.
- numpy.
- librosa.
- soundfile.
- scipy.
- scikit-learn.
- xgboost or lightgbm.
- PyTorch.
- matplotlib.
- seaborn.
- SHAP optional.
- Streamlit or Gradio for demo.

## Historical proposed repository structure (superseded)

```text
covid_audio_btp/
  README.md
  requirements.txt
  data/
    raw/
      coswara/
      coughvid/
      cambridge_optional/
    processed/
      metadata_clean.csv
      features_mfcc.csv
      spectrogram_index.csv
  notebooks/
    01_dataset_audit.ipynb
    02_feature_extraction.ipynb
    03_baseline_ml.ipynb
    04_cnn_spectrogram.ipynb
    05_multimodal_fusion.ipynb
    06_calibration_uncertainty.ipynb
    07_quality_cross_dataset.ipynb
  src/
    config.py
    data_index.py
    audio_preprocess.py
    feature_extract.py
    train_ml.py
    train_cnn.py
    fusion.py
    calibrate.py
    evaluate.py
    explain.py
  reports/
    figures/
    tables/
    final_report.md
  app/
    app.py
```

## Audio preprocessing

For every file:

1. Load audio.
2. Convert to mono.
3. Resample to 16 kHz.
4. Trim leading/trailing silence.
5. Normalize amplitude.
6. Reject or flag extremely short recordings.
7. Calculate audio-quality features.
8. Save processed path and metadata.

Quality features:

- duration,
- RMS energy,
- zero-crossing rate,
- silence ratio,
- clipping ratio,
- spectral centroid,
- spectral flatness,
- estimated SNR proxy,
- missing/corrupt flag.

Recommended flags:

- duration less than 0.5 s for cough: low quality.
- duration less than 1.0 s for breath/speech: low quality.
- clipping ratio more than 1%-2%: low quality.
- silence ratio more than 70%: low quality.

Do not delete low-quality samples permanently. Keep a `quality_flag` column and compare:

```text
all samples vs quality-filtered samples
```

## Feature extraction

### MFCC features

Use:

- 13 or 40 MFCC coefficients.
- delta.
- delta-delta.
- mean and standard deviation over time.

Additional acoustic features:

- RMS.
- zero-crossing rate.
- spectral centroid.
- spectral bandwidth.
- spectral rolloff.
- spectral flatness.
- duration.

### Mel spectrogram

Recommended:

- sample rate: 16 kHz.
- n_mels: 64.
- window length: 25 ms.
- hop length: 10 ms.
- fmin: 125 Hz.
- fmax: 7500 Hz.
- log-mel conversion.
- fixed time length by crop/pad.

### OpenSMILE

If install is easy:

- use eGeMAPS or ComParE features.
- PLOS ONE 2025 used OpenSMILE and 1582-dimensional feature vectors.

OpenSMILE is optional. MFCC + mel spectrogram is enough.

---

# 9. Experiments

## Experiment 1: Dataset audit

Questions:

- How many participants?
- How many COVID-positive, negative, recovered?
- How many recordings per modality?
- How many bad/missing/corrupt files?
- What is class imbalance?

Outputs:

- dataset statistics table.
- modality distribution plot.
- class distribution plot.
- example spectrograms.

## Experiment 2: Classical ML baseline

Models:

- Logistic Regression.
- SVM.
- Random Forest.
- XGBoost/LightGBM.

Input:

- MFCC/acoustic features.

Metrics:

- AUROC.
- AUPRC.
- balanced accuracy.
- sensitivity.
- specificity.
- F1.
- confusion matrix.

Why:

- Fastest usable baseline.
- Good no-GPU fallback.

## Experiment 3: CNN on spectrograms

Models:

- simple custom CNN.
- ResNet18 on spectrogram image.
- EfficientNet-B0 optional.

Augmentation:

- time masking.
- frequency masking.
- random gain.
- background noise injection.
- time shift.

Avoid:

- augmenting before split.
- same participant in train/test.

## Experiment 4: Modality comparison

Train:

- cough-only.
- breath-only.
- vowel /a/ only.
- vowels /a/e/o.
- cough + breath.
- cough + speech.
- breath + speech.
- cough + breath + speech.

Fusion:

- late fusion first.
- average probabilities.
- weighted by validation AUROC if needed.

Why late fusion:

- easiest,
- explainable,
- robust if one modality fails.

## Experiment 5: Calibration

Methods:

- Platt scaling.
- isotonic regression.
- temperature scaling for neural model.

Metrics:

- Expected Calibration Error.
- Brier score.
- negative log likelihood.
- reliability diagram.

Reject option:

```text
if calibrated probability is between 0.4 and 0.6:
    output = uncertain
else:
    output = predicted class
```

This is important for healthcare safety.

## Experiment 6: Quality-aware analysis

Compare:

- all data,
- quality-filtered data,
- quality features added as model inputs.

Questions:

- Do low-quality clips cause more errors?
- Does filtering improve calibration?
- Does filtering improve AUPRC/balanced accuracy?

## Experiment 7: Cross-dataset validation

If COUGHVID is used:

- train on Coswara cough.
- test on COUGHVID cough.
- train on COUGHVID expert-labeled subset.
- test on Coswara cough.

Expected result:

- performance likely drops.

This is not failure. It supports the paper gap.

## Experiment 8: Drift analysis

If recording dates are usable:

- split by time.
- train early, test later.
- calculate performance drop.
- compute simple feature distribution drift.

Drift metrics:

- MMD.
- Wasserstein distance.
- cosine distance between monthly feature means.

## Experiment 9: Explainability

For classical ML:

- feature importance.
- SHAP if time permits.

For CNN:

- Grad-CAM/saliency over spectrograms.

Caution:

- Do not claim spectrogram heatmaps prove a medical causal signal.
- Use them as inspection/explainability, not diagnosis proof.

## Experiment 10: Demo

Use Streamlit or Gradio.

Input:

- upload cough/breath/speech file.

Output:

- waveform.
- spectrogram.
- quality score.
- model prediction.
- calibrated confidence.
- uncertainty warning.
- explanation/heatmap if available.

Safety text:

> Research prototype only. Not a clinical diagnostic tool.

---

# 10. Evaluation Protocol

## Split rule

Use participant-level split.

Do not use recording-level split if the same participant can appear in train and test.

Suggested split:

- train: 70%.
- validation: 15%.
- test: 15%.

For drift:

- chronological split.

For cross-dataset:

- train on one dataset.
- test on another dataset.

## Metrics

Main:

- AUROC.
- AUPRC.
- balanced accuracy.
- sensitivity/recall.
- specificity.
- F1.
- ECE.
- Brier score.

Why:

- Accuracy alone is misleading with imbalanced COVID audio datasets.

## Result tables to produce

1. Dataset statistics.
2. ML baseline comparison.
3. CNN comparison.
4. Modality comparison.
5. Fusion comparison.
6. Calibration before/after.
7. Quality-filtering effect.
8. Cross-dataset effect if available.

---

# 11. Two-Month Timeline

## Week 1: Dataset setup and audit

Tasks:

- Download Coswara.
- Build metadata table.
- Clean COVID labels.
- Identify cough, breath, vowel files.
- Generate first spectrograms.
- Run audio-loading sanity checks.

Deliverables:

- `metadata_clean.csv`.
- dataset audit table.
- sample plots.

Go/no-go:

- If Coswara download/preprocessing fails, use HuggingFace mirror.

## Week 2: Classical ML baseline

Tasks:

- Extract MFCC/acoustic features.
- Train Logistic Regression, SVM, Random Forest, XGBoost.
- Use participant-level split.

Deliverables:

- baseline metrics.
- ROC/PR curves.
- confusion matrix.

Minimum success:

- cough-only XGBoost/Random Forest runs end-to-end.

## Week 3: CNN spectrogram model

Tasks:

- Build log-mel spectrogram dataset.
- Train small CNN.
- Compare CNN vs ML baseline.

Deliverables:

- CNN training curves.
- CNN test metrics.

Minimum success:

- CNN trains without leakage and gives non-random output.

## Week 4: Multimodal fusion

Tasks:

- Train modality-specific models.
- Implement late fusion.
- Compare cough/breath/speech.

Deliverables:

- modality table.
- fusion table.

Minimum success:

- at least cough + breath fusion result exists.

## Week 5: Calibration and uncertainty

Tasks:

- Platt scaling.
- isotonic regression.
- temperature scaling if CNN.
- reliability diagrams.
- reject option.

Deliverables:

- ECE/Brier table.
- reliability plot.
- uncertainty/rejection curve.

Minimum success:

- calibrated model reduces ECE or Brier score.

## Week 6: Quality and cross-dataset

Tasks:

- Add quality flags.
- Compare all vs quality-filtered samples.
- Download/process COUGHVID if feasible.
- Run external cough validation.

Deliverables:

- quality table.
- cross-dataset table if available.

Minimum success:

- quality-aware internal analysis complete.

## Week 7: Explainability and demo

Tasks:

- feature importance/SHAP.
- Grad-CAM/saliency if CNN.
- build Streamlit/Gradio demo.

Deliverables:

- explanation figures.
- working demo.

Minimum success:

- demo uploads one file and returns prediction + confidence.

## Week 8: Report and presentation

Tasks:

- final report.
- slides.
- code cleanup.
- reproduce key results.
- record demo video/screenshots.

Deliverables:

- final BTP report.
- final presentation.
- code repository.

---

# 12. Division of Work for 3 People

## Person 1: Data/preprocessing

- Coswara download.
- metadata cleaning.
- participant split.
- audio preprocessing.
- feature extraction.
- quality analysis.

## Person 2: Models/evaluation

- ML baselines.
- CNN model.
- fusion.
- calibration.
- evaluation tables.

## Person 3: Report/demo/visuals

- literature table.
- figures.
- demo app.
- slides.
- final report writing.

Since two members may contribute less, the minimum project should be possible with Person 1 and Person 2 only.

---

# 13. Minimum Viable BTP

If time becomes tight:

1. Coswara cough-only.
2. MFCC + XGBoost baseline.
3. Mel spectrogram CNN.
4. Calibration and reliability diagram.
5. Quality filtering.
6. Demo.

This is enough for a defensible BTech project.

## Strong BTP version

1. Coswara cough + breath + speech.
2. ML + CNN.
3. multimodal late fusion.
4. calibration.
5. quality filtering.
6. COUGHVID external test.
7. explainability.
8. demo.

## Publication/workshop version

1. multimodal Coswara,
2. external validation,
3. calibration,
4. uncertainty rejection,
5. drift/quality analysis,
6. careful discussion of confounding.

---

# 14. Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| COVID labels are noisy | High | Medium | Use calibration, uncertainty, and cautious claims |
| Cross-dataset performance drops | High | Low/medium | Present as reliability evidence, not failure |
| Cambridge data delayed | Medium | Low | Use Coswara + COUGHVID |
| CNN overfits | Medium | Medium | participant split, dropout, augmentation, simpler baselines |
| Fusion does not improve | Medium | Low | report modality comparison honestly |
| Two teammates do little | High | Medium | keep minimum viable project small |
| Accuracy lower than old papers | High | Low | focus on calibration and reliability |
| Professor asks about novelty | Medium | Low | point to 2025 JMIR limitations and our multimodal/calibration extension |

---

# 15. What To Tell Professor

Short pitch:

> We propose a reliable COVID/respiratory audio screening system using cough, breath, and speech recordings. Recent papers show that audio models can detect COVID-related signals, but their major limitation is reliability under noisy crowdsourced recordings, temporal drift, cross-dataset shift, demographic imbalance, and overconfident predictions. Our project will train models and also evaluate calibration, uncertainty, audio quality, and modality fusion.

Base paper:

> A Comprehensive Drift-Adaptive Framework for Sustaining Model Performance in COVID-19 Detection From Dynamic Cough Audio Data, JMIR 2025.

Gap:

> The base paper handles temporal drift but leaves multimodal breath/voice integration, cross-dataset drift, and drift-cause interpretation as future directions. We address these at BTech scale using Coswara and COUGHVID.

Expected output:

> A working prototype that takes cough/breath/speech audio and returns prediction, calibrated confidence, uncertainty warning, quality score, and spectrogram explanation.

Safety statement:

> This is a screening-support research prototype, not a clinical diagnostic replacement.

---

# 16. Files To Create During Implementation

```text
01_dataset_audit_report.md
02_literature_review_table.md
03_baseline_results.md
04_cnn_results.md
05_multimodal_fusion_results.md
06_calibration_results.md
07_quality_filtering_results.md
08_cross_dataset_results.md
09_final_btp_report.md
10_presentation_slides.pdf
```

---

# 17. Final Decision

Choose this topic if professor accepts COVID/respiratory audio.

Final project title:

**Confidence-Calibrated and Drift-Aware Multimodal COVID/Respiratory Screening from Cough, Breath, and Speech Sounds**

Main base paper:

**A Comprehensive Drift-Adaptive Framework for Sustaining Model Performance in COVID-19 Detection From Dynamic Cough Audio Data, JMIR 2025.**

Strongest BTech novelty:

**Combine multimodal Coswara audio, quality-aware filtering, calibrated confidence, and limited cross-dataset validation.**

Do not claim:

> We diagnose COVID from cough.

Claim:

> We build and evaluate a reliability-aware respiratory audio screening prototype and quantify where it succeeds or fails.

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
