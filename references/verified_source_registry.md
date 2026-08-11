# Verified Source And Code Registry

Last updated: 2026-05-26

This file is the source-backed guardrail for the Covid-RARS implementation. It keeps the project focused on audio ML and prevents unsupported claims from entering the code or report.

## Scope Lock

In scope:

- Cough, breath, vowel, counting, and speech-derived audio features.
- Leakage-safe participant-level splits.
- Audio quality, active-event/voice/cough filtering, and missing-modality handling.
- Classical ML baselines, compact CNN baseline, calibration, fusion, uncertainty, and subgroup/shift checks.
- Non-diagnostic research prototype wording only.

Out of scope:

- Mutation prediction.
- Genomic, sequencing, variant, wet-lab, or molecular biology workflows.
- Clinical diagnosis claims.
- Copying paper code blindly without matching dataset, license, dependency, and reproducibility constraints.

Required disclaimer:

```text
Research prototype only. Not a clinical diagnostic tool.
```

## Main Project Goal

Build a reliability-aware and shift-aware multimodal respiratory-audio screening research prototype using Coswara as the core dataset. The scientifically defensible contribution is not just accuracy; it is the complete pipeline: dataset audit, quality filtering before training, participant-level splits, modality availability handling, feature selection, calibrated branch predictions, cautious fusion, and confounding/shift analysis.

## Source Registry

| Source | Official / author code or data checked | What it tells us | Implementation decision |
|---|---|---|---|
| Coswara dataset, IISc LEAP | https://github.com/iiscleap/Coswara-Data | Raw Coswara is the required core dataset. It includes participant folders, `combined_data.csv`, `csv_labels_legend.json`, nine audio categories, metadata, and manual quality labels in `annotations`. | Keep Coswara as mandatory. Our layout audit must verify file structure before indexing. Prefer using official metadata and quality labels if present. |
| Coswara dataset paper / Zenodo | https://zenodo.org/records/7188627 and https://arxiv.org/abs/2305.12741 | Dataset paper reports respiratory sounds plus metadata for remote screening and subgroup/bias analysis. | Report should cite dataset paper, not only GitHub. Subgroup analysis remains part of the project. |
| COUGHVID dataset | inspected Zenodo record/API: https://zenodo.org/api/records/4048312 ; later/current dataset reference: https://zenodo.org/records/7024894 ; Scientific Data article: https://www.nature.com/articles/s41597-021-00937-4 | Large cough-only external dataset; inspected archive layout contains `public_dataset/<uuid>.json` plus `.webm`/`.ogg` audio; later metadata distributions may use CSV fields such as `status_SSL`. | Optional external validation only after Coswara works. Adapter supports sidecar JSON, direct zip inspection, and CSV metadata. Never mix COUGHVID into Coswara train/test splits. |
| COUGHVID detection/segmentation code | https://github.com/bagustris/detect-segment-cough and original reference https://c4science.ch/diffusion/10770/ | Provides cough detection, cough segmentation, SNR estimation, feature extraction ideas, and conversion from webm/ogg to wav. It uses older model/dependency assumptions. | Reuse ideas, not raw pickled models. Our code should implement clean event/quality logic natively and avoid unsafe old pickle dependencies. |
| COVID-19 Sounds NeurIPS supplementary code | https://github.com/cam-mobsys/covid19-sounds-neurips | Provides OpenSMILE+SVM, VGGish, and YAMNet-based quality checks for respiratory symptoms and COVID status tasks. Also separates noisy/silent samples. | Confirms quality filtering must happen early. OpenSMILE/VGGish can be cited as baselines, while the implementation should add complexity only when it answers a defined research question. |
| Nature Machine Intelligence confounding critique | https://www.nature.com/articles/s42256-023-00773-8 | Matching age, gender, symptoms, and other covariates can collapse inflated audio-only claims; simple symptom models are strong comparators. | Must include subgroup/confounding checks and symptom/demographic baseline where metadata allows. Do not claim diagnosis. |
| Turing/RSS Biomedical Acoustic Markers code | https://github.com/alan-turing-institute/Turing-RSS-Health-Data-Lab-Biomedical-Acoustic-Markers | Repro code includes OpenSMILE-SVM, BNN/ResNet, SSAST, split checks, matching notebooks, and Docker. It is compute-heavy. | Use as methodological reference for split validation, matching, and robust reporting. Do not port whole stack into BTech project. |
| Drift-adaptive cough-audio framework, JMIR 2025 | https://www.jmir.org/2025/1/e66919 | Framework separates baseline training, drift detection, and adaptation; mentions CUSUM/MMD style monitoring and post-development data. | Keep drift/shift analysis structural, but implement BTech-scale checks first: time/date splits if available, dataset-source checks, quality-stratified metrics. |
| HST official implementation | https://github.com/icon-lab/HST | Official HST code uses spectrogram conversion, pretrained HST weights, CUDA-oriented training, and Cambridge respiratory sound tasks. | Optional advanced reference only. Our compact CNN remains primary deep baseline; HST can be a future extension, not a required dependency. |
| Calibration paper, IEEE BHI 2024 / OpenReview | https://openreview.net/forum?id=chVymJKep2 | ENCL-DNN emphasizes confidence calibration and ECE reduction for respiratory cough screening. | Keep branch-level calibration before fusion. Report ECE/Brier/NLL where possible. |
| SympCoughNet 2025 | https://www.frontiersin.org/articles/10.3389/fdgth.2025.1551298/full | Uses VAD/data augmentation, cough duration standardization, and symptom vectors. | Use as support for quality/event filtering and symptom-aware evaluation. Avoid complex gated architecture unless basic baselines are stable. |
| Pay Attention to the Speech | https://pmc.ncbi.nlm.nih.gov/articles/PMC8397542/ | Uses Coswara, multiple sound types, handcrafted audio features, class imbalance handling, and model combination. | Supports multimodal experiments, but fusion must be calibrated/validated because naive fusion can degrade results. |
| Audio feature ranking paper | https://arxiv.org/abs/2104.07128 | Compares and ranks audio features on Cambridge and Coswara; reports feature selection can improve baseline accuracy. | Keep feature selection inside ML pipelines and avoid training high-dimensional tabular models blindly. |
| Feature selection and explainability, Interspeech 2021 | https://www.isca-archive.org/interspeech_2021/avila21_interspeech.html | Combines high-dimensional acoustic features with log-mel CNN and probability-score fusion; reports DiCOVA challenge improvement. | Support dual-track tabular + spectrogram approach. Treat probability fusion as a baseline, not final proof. |
| Wadhwani AI Cough Against COVID code | https://github.com/WadhwaniAI/cough-against-covid | Public archived code includes cough ResNet-18, context TabNet, public split files, pretrained model download, Docker workflow, and training/eval configs. | Useful reference for codebase organization, split assets, cough/context separation, and demo notebooks. Do not depend on archived stack. |
| Virufy data repository | https://github.com/virufy/virufy-data | Provides small clinical/crowdsourced cough dataset with PCR labels and segmented cough folders. | Optional citation or future external validation only; too small/different for core project. |



## Dataset Layout Inspection Trace

A temporary inspection on 2026-05-26 verified the real Coswara and COUGHVID schemas used by the implementation. The durable trace is:

```text
research_protocol/2026-05-26-dataset-schema-inspection.md
```

Consequences added after this inspection:

- Coswara official short metadata columns are supported (`a`, `g`, `l_c`, `record_date`, `testType`, `test_status`).
- COUGHVID official sidecar JSON layout and direct zip indexing are supported.
- COUGHVID CSV metadata with `status_SSL` is supported.
- `archive.zip::member` audio paths can be loaded through temporary materialization.

## Rules For Using Paper Code

1. Prefer ideas and reproducible design patterns over direct copying.
2. Do not import old pickled models from external repos into our core pipeline.
3. Do not add heavy transformer/OpenSMILE stacks unless the basic artifact pipeline is already validated.
4. Any borrowed algorithmic idea must be mapped to our dataset schema, split manifest, and artifact outputs.
5. Any future code copied or adapted must include license attribution and a comment naming the source.
6. Public paper results are not accepted as our results; our metrics must come from our own split manifest and reports.

## Implementation Consequences Already Reflected In Our Code

- `scripts/00_inspect_dataset_layout.py` exists because Coswara layout must be verified before indexing.
- `scripts/04_quality_audit.py` happens before feature extraction or modeling.
- `scripts/12_validate_artifacts.py --strict` is now a master-notebook gate.
- Participant leakage checks are mandatory before supervised modeling.
- Feature selection is part of classical ML training.
- Calibration is performed before fusion.
- Fusion is compared against best single modality instead of assumed superior.
- Subgroup/confounding checks are part of the planned report path.

## Do Not Claim

- Do not claim clinical diagnostic performance.
- Do not claim COVID-specific acoustic biomarkers without confounding analysis.
- Do not claim mutation or variant prediction capability.
- Do not claim cross-dataset generalization unless COUGHVID or another external dataset is actually evaluated.
- Do not claim transformer-level novelty if we only run compact CNN and classical ML baselines.
