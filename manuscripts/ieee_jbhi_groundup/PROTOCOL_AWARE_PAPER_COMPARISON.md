# Protocol-Aware Paper Comparison and Research Defense

## Purpose

This note compares the COVID-RARS evidence with closely related respiratory-audio papers. It is designed for manuscript revision, supervisor discussion, and reviewer response. It does not rank incomparable AUROC values in one leaderboard.

Every numerical comparison must align, or explicitly distinguish, the following:

1. Dataset and collection period.
2. COVID-19 label construction.
3. Audio modalities and non-audio inputs.
4. Participant, recording, or segment as the analysis unit.
5. Internal validation versus frozen source-to-target transfer.
6. Whether target labels influenced training, feature selection, thresholding, or calibration.
7. Uncertainty and class prevalence.

## Provenance Checkpoint

The exact protocol-matched values 0.8489 and 0.8641 are preserved in the dated reviewer-defense record from the completed Ubuntu runs. The corresponding raw protocol-matched metrics, predictions, and split-audit CSV files are not present in this Windows checkout. They must be synchronized from the Ubuntu system before the values are treated as submission-final or added to a public reproducibility package. This file does not elevate the defense record into a substitute for the raw outputs.

## Executive Comparison Verdict

| Comparison class | Published estimate | Present estimate | Numerical position | Defensible reading |
|---|---:|---:|---:|---|
| Closest internal multimodal context: FAIR | 0.8658 | 0.8489 clean repeated CV | -0.0169 | Slightly lower under the cleaner fold-local selection analysis. |
| Closest internal multimodal context: FAIR | 0.8658 | 0.8641 fixed-feature sensitivity | -0.0017 | Numerically almost identical, but the fixed feature bank makes this a sensitivity result rather than the primary clean estimate. |
| Chetupalli audio-only fusion | 0.880 | 0.8489 clean repeated CV | -0.0311 | Lower, although dataset snapshot and modeling differ. |
| Chetupalli audio-only fusion | 0.880 | 0.895 validation-selected equal-weight fusion | +0.015 | Numerically close on one source split, which is not sufficient for a superiority claim. |
| Closest directional frozen transfer: Islam et al. | 0.530 | 0.523-0.543 | -0.007 to +0.013 | Essentially the same external magnitude under the closest published transfer direction. |
| HST internal COUGHVID | 0.90 +/- 0.01 | 0.523-0.543 external | Not computable as a fair gap | HST learned from COUGHVID. Our models did not. This is target learnability versus source-to-target portability. |
| Cough2COVID-19 multi-source transfer | 0.981 | 0.523-0.543 | Not computable as a fair gap | Three source datasets and target-aware feature-family analysis make this an important counterexample, not the same estimand. |

The numerical verdict is mixed. The final pipeline is competitive with close internal Coswara multimodal baselines, especially in the fixed-feature sensitivity analysis, but it is not uniformly superior. The strongest direct external comparison supports rather than contradicts the near-chance transfer result. The strongest published COUGHVID numbers use materially different target exposure and cannot be used to calculate a fair performance deficit.

## Current Study in Numbers

| Evidence rung | AUROC | What it establishes |
|---|---:|---|
| Validation-selected participant-disjoint Coswara cough-speech fusion | 0.895 | Strong source discrimination for the prespecified equal-weight rule. The exploratory stack reached 0.897. |
| Repeated participant-disjoint CV with fold-local feature selection | 0.849 +/- 0.020 | Cleaner paper-style internal estimate with selection repeated inside each fold. |
| Repeated CV with the fixed final top-800 feature bank | 0.864 +/- 0.020 | Sensitivity analysis for the frozen selected representation, not the cleanest nested estimate. |
| Conventional cough source tests | 0.849-0.868 | Matched source denominators for the four controlled transfer models. |
| Frozen Coswara-to-COUGHVID cough transfer | 0.523-0.543 | Near-chance discrimination after collection and endpoint change. |
| WavLM source to target | 0.812 to 0.484 | A self-supervised transformer representation did not remove the transfer failure. |
| CNN-BiGRU source to target | 0.737 to 0.548 | A spectrogram neural representation produced the same qualitative failure pattern. |
| Early-to-late temporal stress test | 0.691 +/- 0.007 | Source associations were unstable under calendar and prevalence change. |
| Full symptoms and collection-context model | 0.964 observed, 0.503 after shuffled-label retraining | Source labels contain strong non-audio structure, while the shuffle control argues against simple implementation leakage. |
| Early versus late top-800 feature overlap | Jaccard 0.074 | Selected acoustic associations were highly unstable across the chosen temporal boundary. |

## Numerical Comparisons

### Close Internal Coswara Context

| Paper | Published result | Present result that may be discussed beside it | Boundary |
|---|---:|---:|---|
| Chetupalli et al., *Multi-Modal Point-of-Care Diagnostics for COVID-19 Based on Acoustics and Symptoms* | Audio-only fusion 0.88. Audio plus symptoms 0.963. | 0.895 primary source split, 0.849 clean repeated CV, 0.864 fixed-feature sensitivity. | The audio-only value is close context. The 0.963 value adds symptoms and is not an audio-only comparison. |
| Truong et al., *Fused Audio Instance and Representation for Respiratory Disease Detection* | 0.8658 +/- 0.0115 for cough, breath, and speech on a fixed 226-person internal test set. | 0.849 +/- 0.020 clean repeated CV and 0.864 +/- 0.020 fixed-feature sensitivity. | This is the closest multimodal internal numerical comparison, but the cohort snapshot, seven inputs, learned representations, and fixed-test design differ. |
| Bhattacharya et al., *Coswara* | 0.915 for its published multimodal workflow. | 0.895 primary source split. | Same resource family but a different snapshot, task bank, model, and symptom handling. Context only for magnitude. |
| Aytekin et al., *Hierarchical Spectrogram Transformers* | COUGHVID AUROC 0.90 +/- 0.01. | COUGHVID external AUROC 0.523-0.543. | Not a direct comparison. HST was trained and validated inside a balanced COUGHVID task, while our models were frozen after Coswara-only development. |

The defensible conclusion is that the current source pipeline is reasonably strong and numerically near relevant Coswara multimodal systems. It is not justified to claim universal state of the art.

### Closest Frozen Transfer Context

| Paper | Transfer design and result | Comparison with the present study |
|---|---|---|
| Islam et al., *Robust COVID-19 Detection from Cough Sounds Using Deep Neural Decision Tree and Forest* | Explicit Coswara-to-COUGHVID transfer AUROC 0.53. Internal Coswara and COUGHVID values were 0.92 and 0.93. | This is the closest peer-reviewed directional comparison. Our four conventional models give 0.523-0.543, closely reproducing the external magnitude. Their COUGHVID target was label-balanced and smaller, while ours retained 285 positive and 8,046 negative recording UUIDs. |
| Hussain et al., *Cough2COVID-19* | Multi-source Coswara, Virufy, and ComParE development followed by selected balanced COUGHVID testing, AUROC 0.981. | This is an important counterexample, not an equivalent experiment. It uses three source datasets rather than Coswara alone. The paper also reports feature-family ranking results over all four datasets, so representation choice was target-aware even though final model fitting excluded COUGHVID. |
| Pahar et al., *COVID-19 Detection in Cough, Breath and Speech Using Deep Transfer Learning and Bottleneck Features* | High internal estimates and reported Coswara-to-Sarcos cough transfer around 0.954. | Genuine transfer evidence, but the target contained only 44 subjects and was not COUGHVID. It demonstrates that transport can succeed for some source-target constructions and prevents a universal claim that respiratory-audio transfer is impossible. |

The defensible conclusion is not that every external model fails. It is that frozen Coswara-only transfer to the evaluated COUGHVID endpoint fails consistently and agrees with the closest matched published direction.

### Rigorous Evaluation Papers

| Paper | Central finding | What the present study adds |
|---|---|---|
| Han et al., *Sounds of COVID-19: Exploring Realistic Performance of Audio-Based Digital Testing* | AUROC 0.71 under participant-independent, demographically controlled evaluation and up to 0.90 under biased scenarios. | An independent cough target, temporal stress test, feature-stability analysis, calibration analysis, and matched transfer across model families. |
| Coppock et al., *Audio-Based AI Classifiers Show No Evidence of Improved COVID-19 Screening over Simple Symptoms Checkers* | AUROC 0.846 before matching and 0.619 after matching in a much larger PCR-referenced cohort, with no clear practical gain over symptoms. | Frozen public source-to-target transfer, repeated model-family transfer, source-target separation, and disjoint target recalibration. Coppock et al. remain stronger for clinical labels, cohort size, causal enrollment reasoning, and symptom comparison. |
| Ganitidis et al., *A Comprehensive Drift-Adaptive Framework* | Chronological degradation and adaptation in dynamic cough streams. | A separate frozen external target and feature-ranking instability. Their work is stronger on online adaptation, which we did not implement. |

## Qualitative Contribution Matrix

`Yes` means the paper directly evaluates the item. `Partial` means a related analysis is present but the estimand or design differs. `NR` means not reported in the audited full text.

| Capability | Present | HST | DNDT/DNDF | Han | Coppock | Ganitidis | Cough2COVID-19 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multimodal respiratory-audio source model | Yes | Separate cough/breath tasks | No | Yes | Yes | No | No |
| Participant-disjoint source evaluation | Yes | Yes | NR | Yes | Yes | No, overlap acknowledged | NR |
| Frozen Coswara-to-COUGHVID cough transfer | Yes | No | Yes | No | No | No | Partial, multi-source |
| External delta confidence interval | Yes | No | No | Not applicable | Not applicable | Not applicable | No |
| Temporal evaluation | Yes | No | No | No | Yes | Yes | No |
| Repeated-seed temporal stability | Yes | No | No | No | Partial | NR | No |
| Calibration and held-out recalibration | Yes | No | No | No | Partial | No | No |
| Metadata-only and symptom-only controls | Yes | No | No | Partial | Yes | No | No |
| Shuffled-label retraining control | Yes | No | No | No | No | No | No |
| Feature-selection stability over time | Yes | No | No | No | No | No | No |
| Incremental audio value beyond symptoms | Exploratory, underpowered | No | No | Partial | Yes | No | No |
| Learned representation transfer check | WavLM and CNN-BiGRU | HST internally | DNDT/DNDF | VGGish internally | SSAST and ResNet internally | VGGish | CNN ensemble |

The novelty is the coordinated reliability design, not ownership of every individual method. Bootstrap intervals, calibration, metadata controls, temporal evaluation, and external validation each exist elsewhere. Their use around one frozen source workflow and one explicit transfer estimand is the contribution.

## Reviewer Defenses

### "Your external AUROC is low because the pipeline is weak"

The criticism cannot be dismissed completely. The system is not state of the art under every internal protocol. The strongest response is:

> The clean internal multimodal result is 0.849 AUROC and the fixed-feature sensitivity result is 0.864, close to FAIR's 0.866 and Chetupalli's 0.88 audio-only fusion. The validation-selected equal-weight source split reaches 0.895, while an exploratory stack reaches 0.897. More importantly, the frozen Coswara-to-COUGHVID result is not unique to one classifier. Four conventional models, WavLM, and CNN-BiGRU all degrade, and Islam et al. independently report 0.53 for the same transfer direction. This supports a dataset-transport problem in the evaluated setting, although it does not prove that every possible architecture must fail.

### "HST reaches 0.90 on COUGHVID, so why are your values near 0.53?"

> HST's 0.90 is an internal COUGHVID estimate after COUGHVID contributes training and validation data. Our 0.523-0.543 estimates freeze the model after Coswara-only development and then evaluate it on COUGHVID. The numbers answer different questions: learnability inside COUGHVID versus portability from Coswara to COUGHVID.

### "Cough2COVID-19 reports 0.981 on COUGHVID"

> That system uses Coswara, Virufy, and ComParE jointly for development, while ours uses Coswara alone. Its feature-ranking analysis reports labeled performance across all four datasets before selecting MFCC, spectrogram, and chromagram families. It is therefore a multi-source, target-aware design rather than a frozen Coswara-only transfer. It is a valid high-performance result under its protocol, but it does not contradict our estimand.

### "Why call the work multimodal when external validation is cough only?"

> Multimodal refers to the source workflow, where cough, breath, and speech improve internal discrimination. COUGHVID contains cough only, so external transfer deliberately evaluates the cough branch. The manuscript does not claim that full multimodal fusion was externally validated.

### "Did you prove shortcut learning in the audio models?"

> No. Metadata AUROC, recording-year importance, temporal degradation, and feature instability establish strong label-context structure and shortcut risk. They do not prove that a particular audio model directly encoded every metadata variable. The manuscript uses association and risk language rather than causal attribution.

### "What is the actual contribution if there is no new architecture?"

> The contribution is an analysis design that keeps the source workflow frozen and separates four questions that are often conflated: internal source discrimination, temporal stability, cough-matched dataset transport, and operational-label calibration. It then connects the observed failure to feature instability, context predictability, source-target separation, and model-family consistency.

## Claims That Are Supported

1. Internal multimodal discrimination is strong but not universally state of the art.
2. The clean internal result is numerically close to relevant Coswara multimodal comparators.
3. Frozen Coswara-to-COUGHVID cough transfer is near chance across the evaluated conventional and learned representations.
4. The closest published transfer direction reports essentially the same AUROC.
5. Target-supervised recalibration improves calibration error but not ranking discrimination.
6. Source labels exhibit strong associations with symptoms and collection context.
7. Temporal feature rankings and performance are unstable under the tested retrospective split.

## Claims That Are Not Supported

1. The model is universally state of the art.
2. Prior papers fabricated their reported metrics.
3. Full multimodal fusion failed on COUGHVID.
4. No COVID-19 acoustic information exists.
5. Every transformer or future architecture will fail externally.
6. Metadata controls prove the exact shortcut used by each audio model.
7. External operating points against `status_SSL` establish clinical diagnostic utility.

## Concise Contribution Statement

> We provide a protocol-aware reliability evaluation of multimodal COVID-19 respiratory-audio models. The study combines participant-disjoint source evaluation, repeated internal CV, retrospective temporal stress testing, modality-matched frozen transfer from Coswara to COUGHVID, analysis-unit-level bootstrap uncertainty, calibration and recalibration checks, metadata and shuffle controls, and temporal feature-stability analysis. The central finding is not a new classifier record. It is that source discrimination near established Coswara baselines does not imply transport to an independently collected operational endpoint.

## Primary Sources

- Chetupalli et al., DOI: https://doi.org/10.1109/JTEHM.2023.3250700
- Truong et al., DOI: https://doi.org/10.3390/s24196176
- Aytekin et al., DOI: https://doi.org/10.1109/JBHI.2023.3339700
- Han et al., DOI: https://doi.org/10.1038/s41746-021-00553-x
- Coppock et al., DOI: https://doi.org/10.1038/s42256-023-00773-8
- Ganitidis et al., DOI: https://doi.org/10.2196/66919
- Islam et al., DOI: https://doi.org/10.1016/j.eswa.2026.131235
- Hussain et al., DOI: https://doi.org/10.1038/s41598-024-76639-9
- Pahar et al., DOI: https://doi.org/10.1016/j.compbiomed.2021.105153
