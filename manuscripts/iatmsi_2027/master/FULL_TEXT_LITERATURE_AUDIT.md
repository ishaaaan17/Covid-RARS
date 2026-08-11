# Full-Text Literature and Protocol Audit

## Purpose

This file is the source-of-truth audit for literature claims in the master
manuscript. It is not a ranking of classifiers. A reported value can enter a
numeric comparison only after the dataset, endpoint, modality, analysis unit,
split, target-data exposure, metric, and uncertainty have been checked in the
full paper. Page numbers below refer to the rendered published PDF unless noted.

The audit covered extracted text and every rendered page, including figures,
tables, captions, appendices, and stated limitations. Publisher abstracts alone
are listed separately and are not treated as full-paper evidence.

## Comparison Classes

- **A-close:** same disease endpoint and a substantially similar dataset,
  modality, and evaluation direction. Residual differences must still be stated.
- **B-partial:** relevant to the same scientific question, but at least one
  material element differs, such as cohort, modality, label construction,
  analysis unit, target size, or target exposure.
- **C-context only:** answers a different estimand and must not appear in a
  direct metric leaderboard.

## Audited Evidence Matrix

| Study | Dataset, endpoint, and unit | Model and modality | Evaluation protocol | Result used in discussion | Target-label exposure | Class and exact boundary | Source anchor |
|---|---|---|---|---:|---|---|---|
| Bhattacharya et al., *Coswara* (Scientific Data, 2023), DOI 10.1038/s41597-023-02266-0 | Coswara; COVID status; participant | Nine BLSTM audio branches plus symptoms; cough, breath, vowels, counting | Participant-stratified 70/15/15 internal split | AUROC 0.915, 95% CI 0.885-0.941 | Internal validation and test only | B-partial: same resource family, but a different snapshot, nine tasks, symptoms, and model | Table 4, p. 9; limitations pp. 10-11 |
| Orlandic et al., *COUGHVID* (Scientific Data, 2021), DOI 10.1038/s41597-021-00937-4 | COUGHVID; cough-event detection benchmark; recording | XGBoost on 68 descriptors; cough only | Nested 10-fold internal CV | AUC 0.964 +/- 0.033 | Internal | C-context only: this is cough-versus-noncough detection, **not COVID classification** | Table 3, p. 4; limitations pp. 8-9 |
| Xia et al., *COVID-19 Sounds* (NeurIPS Datasets and Benchmarks, 2021) | COVID-19 Sounds; self-reported test status; participant | OpenSMILE-SVM and VGGish; cough, breath, voice | Speaker-independent 7:1:2 internal split | Multimodal VGGish AUROC 0.71, 95% CI 0.65-0.76 | Internal | B-partial: participant independent and multimodal, but a different cohort and endpoint process | Table 3, p. 7; limitations pp. 8-9 |
| Chetupalli et al., *Multi-Modal Point-of-Care Diagnostics* (IEEE JTEHM, 2023), DOI 10.1109/JTEHM.2023.3250700 | Coswara-derived cohort; participant | LR/SVM/LSTM audio branches and symptom tree; deep breath, heavy cough, counting | Subject-disjoint 80/20 test; five-fold tuning in development | Audio-only fusion 0.88; audio+symptoms 0.963 | Internal; later observation cohorts are not independent datasets | A-close for 0.88; C-context for symptom-assisted 0.963 | Participant flow and fusion, pp. 3-7; result Fig. 9/Table V, pp. 7-8 |
| Truong et al., *FAIR* (Sensors, 2024), DOI 10.3390/s24196176 | Coswara, 1,359 participants and 9,513 recordings | Frozen DeiT-S/16 and wav2vec encoders with attention over seven sound instances | Fixed 226-participant test; five rotating train/validation folds, repeated with two seeds | AUROC 0.8658 +/- 0.0115 | Internal | A-close: internal multimodal Coswara comparator, but different cohort, seven-instance input, frozen encoders, and test protocol | Table 6, p. 11; Appendix Table A4, p. 16; limitations p. 14 |
| Han et al., *Sounds of COVID-19: Exploring Realistic Performance* (npj Digital Medicine, 2022), DOI 10.1038/s41746-021-00553-x | 2,478 English-speaking participants; self-reported labels | VGGish multimodal fusion; cough, breath, voice | Participant-independent development/test plus deliberately biased variants | Multimodal AUROC 0.71 [0.65, 0.77]; biased designs up to 0.90 | Internal | B-partial: direct evidence that cohort construction changes measured performance; not a direct numeric comparator | Figs. 1-4, pp. 3-5; limitations p. 6 |
| Coppock et al., *Audio-Based AI Classifiers Show No Evidence...* (Nature Machine Intelligence, 2024), DOI 10.1038/s42256-023-00773-8 | 67,842 PCR-validated participants | ComParE-SVM, ResNet-50 BNN, SSAST, and symptom/demographic RF | Participant-disjoint random, matched, longitudinal, and longitudinal-matched protocols | Random SSAST 0.846 [0.838, 0.854]; matched 0.619 [0.594, 0.644] | Internal but prespecified and covariate-controlled | B-partial: strongest methodological comparator for confounding and incremental value; larger PCR cohort than ours | Figs. 1-3 and Table 1, pp. 230-236; limitations p. 237 |
| Ganitidis et al., *Drift-Adaptive Framework* (JMIR, 2025), DOI 10.2196/66919 | Coswara and COVID-19 Sounds; cough; recording/sample | VGGish, MMD, CUSUM, adaptation, active learning | Chronological 70/30; participant overlap prevented | Coswara AUROC 0.668 to 0.597; COVID-19 Sounds 0.691 to 0.607 | Later labels used for evaluation/adaptation experiments | B-partial: close temporal question, but cough only and includes adaptation rather than our diagnostic audit | Table 1 and Figs. 1-13, pp. 4-15; limitations p. 16 |
| Kim and Lee, *Limitations of the Cough Sound-Based...* (JMIR, 2024), PMID 38319694 | Cambridge source; Virufy and Coswara period targets; cough | Xception plus five acoustic statistics | Cambridge 80/20 and 3-fold CV; later dataset/period tests; participant grouping not explicit | Cambridge 0.9346; Coswara Alpha 0.8250, Delta 0.7724, Omicron 0.5509 | No target retraining described | B-partial: dataset, acquisition, calendar period, and presumed variant change together | Fig. 1, p. 3; methods pp. 2-3 |
| Grant et al., *Considerations and Challenges for Real-World Deployment* (Sensors, 2022), DOI 10.3390/s22239530 | DiCOVA/Coswara source; Cambridge, web, Android, and repeatability targets | MFCC/RASTA-PLP MLP; cough, breath, speech | Five-fold source CV plus several external stress tests | Web cough 0.42 versus Android cough 0.75; mean AUROC 0.77 | Reports both frozen source threshold and target-optimized thresholds | B-partial: direct deployment-condition evidence; different source-target pairs | Figs. 2, 4-6, pp. 3-11; limitations pp. 17-18 |
| Atmaja et al., *Cross-Dataset Transfer With Data Augmentation* (International Journal of Information Technology, 2025), DOI 10.1007/s41870-025-02433-z | Coswara and COUGHVID development; ComParE test; cough segments | PANN CNN14 with cough segmentation, SpecAugment, and mixup | Pooled multi-source development; fixed 154-segment ComParE test | Unweighted accuracy 0.8819 | The same test is reported while split ratio, cough threshold, segmentation, augmentation, and mixup are chosen | C-context numerically: result is not AUROC and the test was used during configuration; B-partial for augmentation evidence | Tables 2 and 5-6, pp. 5 and 11-12; limitations p. 12 |
| Brown et al., *Exploring Automatic Diagnosis...* (KDD Health Day, 2020) | Early COVID-19 Sounds cohort; participant | 477 measured descriptors plus VGGish/PCA; cough and breath | Ten disjoint-user 80/20 outer splits with inner five-fold tuning | Main AUROC about 0.80-0.82; selected augmented tasks 0.87-0.88 | Internal | B-partial: historical participant-separated baseline with small positive subgroups | Fig. 4 and Table 1, pp. 5-8; supplementary Table 2, p. 11 |
| Aytekin et al., *Hierarchical Spectrogram Transformers* (IEEE JBHI, 2024), DOI 10.1109/JBHI.2023.3339700 | Cambridge tasks and a separate balanced COUGHVID task; participant/recording as defined by dataset | ImageNet-pretrained five-stage HST; cough and breath modeled separately | Ten repeated approximately 70/20/10 participant-disjoint internal partitions in each dataset | COUGHVID HST AUROC **0.90 +/- 0.01**, precision 0.80, recall 0.84, F1 0.82 | COUGHVID is used for its own training and validation | C-context for our frozen transfer; valid internal architecture comparator only | Table VII, journal p. 1280; limitations pp. 1282-1283 |
| Islam et al., *DNDT/DNDF Cross-Datasets Evaluation* (Expert Systems with Applications, 2026), DOI 10.1016/j.eswa.2026.131235 | Five cough datasets; selected balanced cohorts | 193 descriptors, RFECV, Bayesian optimization, SMOTE, DNDT/DNDF | Stratified 10-fold internal CV; explicit train-one/test-another transfer | Internal Coswara 0.92; internal COUGHVID 0.93; Coswara to COUGHVID **0.53** | No target fitting in transfer; target cohort selected/balanced by label | A-close for direction and modality, not exact because target is 680/680 and participant grouping is unreported | Tables VII and IX; Section 5.4, pp. 15-18 |
| Pahar et al., *Deep Transfer Learning and Bottleneck Features* (Computers in Biology and Medicine, 2022), DOI 10.1016/j.compbiomed.2021.105153 | Coswara; Sarcos external target with 44 subjects | CNN/LSTM/ResNet50; cough, breath, speech evaluated separately | Nested leave-p-out with subject aggregation; Coswara-trained external Sarcos test | Coswara cough 0.982; Coswara to Sarcos 0.954 | No external target fitting described | B-partial: genuine transfer, but a different and very small target (18 positive, 26 negative) | Tables II and IV-VIII, pp. 4-11 |
| Avila et al., *Feature Selection and Explainability* (Interspeech, 2021) | DiCOVA/Coswara cough; blind 233-recording test | 6,373 ComParE RBF-SVM, log-mel CNN, average fusion | Official five folds plus blind test | Blind fusion 0.808; nonnested feature CV 0.96 +/- 0.02 fell to 0.768 when nested | Blind test excluded from selection | B-partial and key leakage precedent | Tables I-II and Figs. 1-4, pp. 952-954 |
| Laguarta et al., *COVID-19 AI Diagnosis Using Only Cough Recordings* (IEEE OJEMB, 2020), DOI 10.1109/OJEMB.2020.3026928 | Private 5,320-subject cough cohort; mixed label sources | Three pretrained ResNet50 biomarker branches | Random 80/20 subject split | Official-test subset AUROC 0.97 | Internal | C-context: private internally sampled cohort and heterogeneous label ascertainment | Fig. 1, Table II, Fig. 2, pp. 3-6 |
| Chowdhury et al., *Ensemble-Based MCDM* (Computers in Biology and Medicine, 2022), DOI 10.1016/j.compbiomed.2022.105405 | Multiple cough datasets analyzed separately or merged | 193 descriptors, ten classifiers, RFECV, SMOTE, threshold moving, TOPSIS ranking | Stratified 10-fold CV; optional five-fold inner optimization; participant grouping unreported | Coswara approximately 0.64-0.66; other datasets up to 0.97-0.98 | Internal/merged; no frozen source-to-target test | C-context for external claims; B-partial for paper-style internal context | Tables I, V, IX-X, pp. 4-13 |
| Celik, *CovidCoughNet* (Computers in Biology and Medicine, 2023), DOI 10.1016/j.compbiomed.2023.107153 | COUGHVID and Coswara recordings | 26 averaged measurements, InceptionFireNet, DeepConvNet; modalities separate | 80/20 recording split; participant and augmentation-parent grouping not reported | COUGHVID binary 0.9505; Coswara cough 0.9848 | Internal or merged internal | C-context: record-level internal estimate, not external transfer or multimodal fusion | Tables V-IX and Figs. 13-17, pp. 10-15 |
| Hussain et al., *Cough2COVID-19* (Scientific Reports, 2024), DOI 10.1038/s41598-024-76639-9 | Coswara, Virufy, ComParE development; balanced selected COUGHVID target (651/660) | MFCC MLP, spectrogram CNN, chromagram MLP, concatenated 768-D embedding | Reported leave-one-dataset-out experiments | Multi-source to COUGHVID AUROC 0.981 | Labeled results from all four datasets informed feature-family ranking; table captions and prose conflict for some held-out datasets | B-partial: important high external result, but target-aware representation choice, multi-source training, selected balanced target, and no uncertainty | Tables III and V-X, pp. 5 and 9-16 |
| de Brito et al., *Fine-Tuning Pre-Trained Audio Models...* (arXiv technical report, 2025) | Coswara and COUGHVID; cough | Audio-MAE and PANN CNN6/10/14 | Stratified internal 80/20 and bidirectional cross-dataset transfer after age-gender undersampling | Coswara to COUGHVID: 0.51, 0.58, 0.68, 0.60 across models | No target training in transfer; labeled demographic undersampling defines cohorts | A-close scientifically, but non-peer-reviewed and not exact | Tables IV, VII, X, pp. 3-10 |
| Lin et al., *SympCoughNet* (Frontiers in Digital Health, 2025), DOI 10.3389/fdgth.2025.1551298 | UK COVID-19 Vocal Audio Dataset; cough plus symptoms; sample/participant grouping not stated | Symptom-fused attention CNN | Stratified 70/15/15 split; training-only augmentation; participant separation not reported | Symptom+audio AUROC 0.9474; audio-only ablation 0.8127 | Internal | C-context: symptom-assisted UK cohort; headline cannot be compared with audio-only Coswara | Tables 2-3, pp. 7-8; discussion/conclusion pp. 10-11 |

## Comparisons That Are Defensible

### Internal multimodal context

The nearest internal Coswara comparisons are Chetupalli's audio-only fusion
(0.88) and FAIR (0.8658 +/- 0.0115). The present selected internal multimodal
estimate is 0.897. These values are contextual rather than formal superiority
tests because cohort snapshots, sound instances, model selection, and split
construction differ. Chetupalli's 0.963 and SympCoughNet's 0.9474 include
symptoms and must not be presented as audio-only comparators.

### Frozen Coswara-to-COUGHVID transfer

The closest peer-reviewed result is DNDF's Coswara-to-COUGHVID AUROC 0.53. The
present conventional cough range, 0.523-0.543, agrees closely in direction and
magnitude. DNDF uses a label-balanced 680/680 target and does not report
participant grouping, while the present target retains 285 positive and 8,046
negative recordings. The Audio-MAE/PANN technical report provides relevant but
non-peer-reviewed deep-transfer values of 0.51-0.68. The present WavLM and
CNN-BiGRU values, 0.484 and 0.548, fall within the same broad failure pattern.

### Important non-equivalent high external result

Cough2COVID-19 reports 0.981 for a multi-source-to-COUGHVID experiment. It is
not interchangeable with Coswara-only transfer: its development set combines
three sources, its COUGHVID target is a selected balanced subset, and labeled
results from all four datasets informed feature-family ranking. The paper must
be cited as a counterexample, with these protocol differences stated explicitly.

## Literature Limitations Covered by the Present Work

| Limitation identified in prior work | Present coverage | Boundary |
|---|---|---|
| Participant overlap or unreported grouping | Participant-level source partitions and leakage audits | COUGHVID participant linkage is unavailable, so external inference remains recording-level |
| Optimism from nonnested feature selection | Training-only ranking; frozen feature names for test and target | Chronological top-800 selection was not refitted inside the early block |
| No chronological test | Early-to-late and reverse temporal stress tests | Calendar, prevalence, and case mix change together; variant causation is not identified |
| No calibration analysis | Brier, ECE, reliability curves, and disjoint target recalibration sensitivity | Target recalibration is not external validation because it uses target labels |
| No independent target | Frozen Coswara-to-COUGHVID cough transfer | Full multimodal transfer is not evaluated because COUGHVID is cough-only |
| Reliance on one representation | Conventional, WavLM, and CNN-BiGRU transfer | This does not exhaust modern architectures; HST is integrated only after its run completes |
| Confounding and collection-context sensitivity | Metadata-only models, label shuffles, grouped permutation importance, matching, and subgroup checks | Association does not prove which metadata shortcut, if any, is encoded in audio |
| Threshold optimized on target | Source-validation thresholds plus a separately labeled recalibration sensitivity | Recalibration results cannot be called untouched external performance |
| No base-rate-aware operating analysis | AUPRC, fixed-sensitivity specificity/precision, calibration, and decision curves | Utility depends on the target prevalence and assumed clinical action |

## Full Text Still Required

The following records were verified from primary publisher or PubMed metadata,
but full PDFs were not locally available during this audit. They must not supply
page-level claims or detailed protocol comparisons until the PDFs are added.

1. **A comprehensive validation study on the influencing factors of
   cough-based COVID-19 detection through multi-center data with abundant
   metadata**. Jiakun Shen et al., *Journal of Biomedical Informatics*, 2025,
   164:104798. DOI: https://doi.org/10.1016/j.jbi.2025.104798
2. **Cough Sound Based Deep Learning Models for Diagnosis of COVID-19 Using
   Statistical Features and Time-Frequency Spectrum**. Jina Kim and Jinseok
   Lee, IEEE EMBC 2024, pp. 1-4. DOI:
   https://doi.org/10.1109/EMBC53108.2024.10781593

## Writing Rules Derived From the Audit

1. Never compare an internally target-trained value with frozen transfer as if
   both estimate portability.
2. Never compare symptom-assisted performance with an audio-only result without
   labeling the additional inputs.
3. Distinguish participant, recording, and segment as analysis units.
4. State whether target labels affected cohort selection, feature choice,
   threshold choice, calibration, or model fitting.
5. Report uncertainty only at the correct independent unit.
6. Describe high internal results as answers to their published protocol, not as
   fabricated numbers.
7. State limitations from the source paper and then state exactly which one the
   present analysis addresses; do not claim complete coverage.
