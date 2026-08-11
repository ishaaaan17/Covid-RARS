# BTP E2E Detailed Plan: COVID/Respiratory Audio Screening

Last updated: 2026-05-22

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

## Suggested repository structure

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

