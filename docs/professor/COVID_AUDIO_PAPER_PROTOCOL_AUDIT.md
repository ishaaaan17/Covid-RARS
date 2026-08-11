# COVID Audio Paper Protocol Audit

Date: 2026-07-14

This note records the split protocols and key numbers checked from the actual PDF page images and text extracts. It is intended for protocol-matched comparison planning, not as manuscript prose.

## Core Point

Prior papers report two different kinds of numbers:

1. Internal paper-style validation numbers, usually random or cross-validation within the same dataset.
2. Strict reliability numbers, such as temporal or cross-dataset transfer.

These should not be mixed in one comparison row. The most important example is the ESWA DNDT/DNDF paper: it reports Coswara AUC 0.92 and COUGHVID AUC 0.93 under within-dataset evaluation, but its own Coswara-to-COUGHVID cross-dataset AUC is 0.53.

## Checked Papers

| Paper | Checked pages | Protocol | Reported values | Use in our work |
|---|---:|---|---|---|
| Robust COVID-19 detection from cough sounds using deep neural decision tree and forest, ESWA 2026 | 10, 15, 16, 20, 21, 23 | Within-dataset results use 10-fold stratified cross-validation. Cross-dataset table trains on one dataset and tests on another. Uses RFECV, Bayesian optimization, SMOTE, and threshold moving. | Coswara internal DNDF AUC 0.92; COUGHVID internal DNDF AUC 0.93; Coswara to COUGHVID AUC 0.53; COUGHVID to Coswara AUC 0.57. | Main paper-style target and strongest evidence that cross-dataset collapse is not unique to our pipeline. |
| COVID-19 Detection From Respiratory Sounds With Hierarchical Spectrogram Transformers, IEEE JBHI 2024 | 6-10 | 10-fold CV, about 70% train, 20% test, 10% validation, no participant-level overlap. Operationally, this is closer to 10 repeated participant-disjoint stratified splits than a classic 90/10 K-fold split. | COUGHVID HST AUC 0.90; Cambridge task 2 cough HST AUC 0.98. | Requires participant-disjoint repeated splits with a 20% test fraction if we claim a split-style match. |
| A Comprehensive Drift-Adaptive Framework for Sustaining Model Performance in COVID-19 Detection From Dynamic Cough Audio Data, JMIR 2025 | 4, 5, 9-12, 14, 15 | Cough-only chronological split: 70:30 development/postdevelopment; development further split 60:20:20; no participant overlap. | Coswara development AUC 0.668; Coswara postdevelopment AUC 0.597. | Supports temporal drift framing, not a SOTA leaderboard target. |
| Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers, Nature Machine Intelligence 2024 | 3-10, 15-18 | Random, standard, matched, longitudinal, matched-longitudinal splits. Matched sets balance recruitment channel, 10-year age bins, gender, and symptom covariates. | Random sentence SSAST AUC 0.846; matched sentence SSAST AUC 0.619; random cough SSAST AUC 0.790; matched cough SSAST AUC 0.561. | Methodological base for confounding and incremental value framing. |

## Implementation Decision

For the professor-requested same-split comparison, do not rerun all models.

Use:

```text
modality: cough
features: ComParE+IS10 top-800 selected by LightGBM ranking
model: svc_rbf_f60
split: participant-disjoint 10 repeated stratified splits
test set: about 20% of participants per split
validation set: about 10% of participants per split, produced by validation_fraction=0.125 after the 20% test participants are removed
```

Reason: the current best cough-only paper-comparable aggregate row is SVC RBF with AUROC about 0.819. The new run fixes the earlier weakness that the old paper-comparable CV used recording-level folds, while HST explicitly used participant-level separation.

## Wording To Use

Use:

> The high numbers in prior papers are internal protocol-dependent results. When evaluated under cross-dataset transfer, even the ESWA DNDF/DNDT paper reports Coswara-to-COUGHVID AUC 0.53, which is the same failure regime as our external-transfer results.

Do not use:

> Prior papers fabricated results.
