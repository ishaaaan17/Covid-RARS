# Covid-RARS Results Comparison

This document explains how to compare our results with published work without making unfair or unsafe claims.

## Executive Summary

Our work should be compared on two axes:

1. **Apples-to-apples internal comparison**: When we use internal or 10-fold style evaluation, our numbers are good but not leaderboard SOTA.
2. **Apples-to-oranges strict validation comparison**: When we use temporal and external validation, our numbers are lower because the test is harder and more deployment-relevant.

A reviewer may focus on the highest reported AUROC in other papers. The correct response is not to dismiss those papers. The correct response is:

```text
Those numbers are usually reported under internal validation protocols. Our primary internal number is also high at 0.895 AUROC, with an exploratory stack at 0.897. The difference is that we additionally tested temporal drift, metadata shortcuts, calibration, decision-curve behavior, and COUGHVID transfer. Under those stricter tests, the model becomes unreliable.
```

## Our Main Result Inventory

| Evaluation type | Our result | How to use it |
|---|---:|---|
| Existing participant split | `0.895` AUROC | Primary validation-selected equal-weight fusion; exploratory stack `0.897` |
| Time-stratified participant split | `0.849` AUROC | More conservative internal evidence |
| Paper-comparable cough 10-fold CV | `0.819` AUROC | Fairer comparison to cough-only CV papers |
| Temporal early-to-late | `0.698` AUROC | Main drift evidence |
| COUGHVID measured audio-summary cough external | `0.523-0.543` AUROC | Main external collapse evidence |
| WavLM transformer COUGHVID external | `0.484` AUROC | Shows transformer does not solve transfer |
| CNN-BiGRU COUGHVID external | `0.548` AUROC | Shows neural spectrogram model also weak |
| Metadata-only full safe metadata | `0.964` AUROC | Shortcut/confounding evidence |
| Symptoms-only metadata | `0.932` AUROC | Important comparison to symptom-checker literature |
| DNDF Differentiable Forest (Breath) | `0.693` AUROC | Peak differentiable tabular tree ensemble |
| DNDF Multimodal Stacked Fusion (Cough+Breath) | `0.688` AUROC | Peak balanced accuracy ($65.15\%$) for differentiable tree fusion |
| DNDF Temporal Early-to-Late Drift | `0.596` AUROC | Confirms temporal degradation on differentiable tree models |

## Fair Comparison Types

### Type A: Internal benchmark comparison

Use this when the other paper reports internal train/test, cross-validation, or participant-level internal testing.

Our comparable rows:

- Existing participant split: `0.895` AUROC for the primary equal-weight fusion (`0.897` for the exploratory stack).
- Time-stratified split: `0.849` AUROC.
- Paper-comparable cough 10-fold CV: `0.819` AUROC.

Safe wording:

```text
Under internal validation, our model reaches strong but not universal SOTA performance. The paper's main value is not the highest internal score; it is the additional reliability audit.
```

### Type B: Strict validation comparison

Use this when discussing deployment, real-world screening, or clinical usefulness.

Our strict rows:

- Temporal early-to-late: `0.698` AUROC.
- COUGHVID external measured audio-summary: `0.523-0.543` AUROC.
- COUGHVID external WavLM: `0.484` AUROC.
- COUGHVID external CNN-BiGRU: `0.548` AUROC.

Safe wording:

```text
These are stricter than most internal benchmark numbers. They are lower because they test temporal and dataset shift rather than only fitting the source benchmark.
```

## Important Related Work Anchors

### 1. Audio-based AI classifiers show no evidence of improved COVID-19 screening over simple symptoms checkers

This paper is extremely relevant because it asks whether audio adds value over symptom checkers.

Key public result to remember:

- It reported strong unadjusted audio performance but much weaker matched/confounder-adjusted performance.
- Reported public figures discussed in our notes: unadjusted audio AUROC about `0.846`; matched/confounder-adjusted audio AUROC about `0.619`.
- Their conclusion is aligned with our project: respiratory audio can look strong before proper adjustment, but simple symptoms/context can dominate.

How our work complements it:

| Coppock-style question | Our answer |
|---|---|
| Does audio outperform symptoms? | In our aligned subsets, symptoms-only metadata is already strong. Metadata+audio sometimes improves over symptoms-only but the confidence intervals are wide. |
| Is unadjusted audio inflated? | Yes, our metadata-only `0.964` AUROC and temporal/external collapse support this concern. |
| Is this only one dataset? | We audit Coswara internally and test cough transfer to COUGHVID. |
| Does a transformer solve it? | No. WavLM external AUROC is `0.484`. |

Use this paper as a supportive anchor, not as a threat.

### 2. Robust COVID-19 detection from cough sounds using deep neural decision tree and forest: A comprehensive cross-datasets evaluation

This ESWA paper is important because it reports strong cough results and uses cross-dataset language.

Safe comparison:

- Their reported internal/cross-dataset setup is not automatically identical to our strict Coswara-to-COUGHVID transfer setup.
- If they report high internal AUC values on Coswara or COUGHVID, compare those to our internal or 10-fold rows, not directly to our external-transfer row.
- Our stricter point is that when a model trained on Coswara cough is applied to COUGHVID cough with our frozen pipeline, AUROC falls to `0.523-0.543` for measured audio-summary models and `0.484-0.548` for deep branches.

What to say:

```text
Their model may be strong under their protocol. Our contribution is not to copy their classifier but to show how much evaluation protocol changes the conclusion. We also include confidence intervals, calibration, decision curves, metadata shortcuts, feature stability, and transformer/deep transfer checks.
```

Do not say:

```text
Their result is fabricated.
```

Say instead:

```text
Their result is not directly comparable unless the split, label construction, participant separation, and external transfer protocol are identical.
```

### 3. COVID-19 Detection From Respiratory Sounds With Hierarchical Spectrogram Transformers

This paper is relevant because a reviewer may ask why we did not use a transformer or spectrogram saliency.

Safe comparison:

- It is a transformer/spectrogram-focused paper.
- Our work did test a transformer branch: WavLM base-plus.
- Our WavLM branch reached strong internal cough performance (`0.812` AUROC pooled cough) but failed externally (`0.484` AUROC on COUGHVID).
- We did not implement their exact HST architecture or Grad-CAM-style saliency.

What to say:

```text
We did not claim that our architecture is stronger than HST. We used WavLM as a transformer branch to test whether modern pretrained representations solve the reliability problem. They did not, which strengthens our validation-focused claim.
```

### 4. JMIR / temporal-validity respiratory-audio papers

Use these papers to justify the temporal validation design.

Safe comparison:

- They support the idea that pandemic time, collection wave, device, symptoms, and recruitment process matter.
- Our contribution is to quantify temporal degradation and feature non-stationarity inside the Coswara/COUGHVID benchmark pipeline.
- Our feature-selection stability result (`0.074` Jaccard) is a strong mechanistic addition.

## Literature Comparison Strategy

Use this table in meetings.

| Question | Best response |
|---|---|
| "This paper has 0.92 AUC. Why are you lower?" | "That is an internal or different-protocol number. Our primary internal result is 0.895, with an exploratory stack at 0.897. Our temporal and external rows test different, stricter transport questions." |
| "Why not just do their model?" | "We tested multiple model families including boosted trees, SVC, CNN-BiGRU, and WavLM transformer. The failure persists across families, so the issue is structural validation, not one missing classifier." |
| "Can we claim SOTA?" | "We should not claim universal SOTA. We can claim a strong reliability-audit contribution with strong internal performance and unusually comprehensive validation." |
| "Will reviewers reject low external numbers?" | "Not if framed correctly. Biomedical AI reviewers value external validation and will see honest external failure as evidence against unsafe deployment claims." |
| "Why should this go to a good journal?" | "Because it addresses shortcut learning, temporal drift, calibration, decision curves, and external validation in a clinically relevant audio-AI domain." |

## Apples-to-Apples and Apples-to-Oranges

### Apples-to-apples

Use when comparing to papers with internal Coswara/cough CV:

- Our cough-only paper-comparable CV: `0.819` AUROC.
- Our primary internal participant-split multimodal result: `0.895` AUROC (`0.897` exploratory stack).
- Our time-stratified multimodal: `0.849` AUROC.

Interpretation:

- Good, not universal SOTA.
- Strong enough that the pipeline is not weak.
- Honest enough to avoid inflated claims.

### Apples-to-oranges

Use when comparing our strict external result to papers that only report internal validation:

- Our external COUGHVID result is not meant to beat internal Coswara scores.
- It is a deployment stress test.
- The correct claim is "internal numbers are not sufficient," not "our classifier is numerically higher."

## Reputable Venue Positioning

Current venue fit, using official/public scope pages checked during doc creation:

| Venue | Fit | Reality |
|---|---|---|
| npj Digital Medicine | Best prestige reach if framed as AI reliability, shortcut learning, and clinical screening safety | Very selective; must not read like a routine classifier paper |
| PLOS Digital Health | Strong realistic target for digital health reliability, open science, and benchmark audit | Good fit if the story is methodological and clinically cautious |
| IEEE Journal of Biomedical and Health Informatics | Good if technical/statistical validation is emphasized | Needs strong engineering/statistical rigor and clean figures |
| Computers in Biology and Medicine | Realistic Elsevier biomedical-computing target | Better if written as computational validation and biomedical signal-processing reliability |
| Biomedical Signal Processing and Control | Realistic signal-processing fallback | Good if the focus is respiratory audio features and validation |
| JMIR AI / JMIR Formative Research route | Useful if the paper is framed around digital-health AI evaluation | Less famous to some readers, but credible in digital health |

How to present:

```text
Ma'am, npj Digital Medicine is the high-prestige reach target only if we write this as a clinical AI reliability audit, not as a classifier paper. The most realistic strong targets are PLOS Digital Health, IEEE JBHI, and Computers in Biology and Medicine. The submission order can start ambitious, then move to the more realistic venue without changing the core manuscript.
```

## Final Positioning Statement

Use this exact positioning in a presentation:

> Our internal model performance is strong, but the paper's novelty is the reliability evidence: temporal drift, metadata shortcuts, calibration failure, decision-curve weakness, feature non-stationarity, and external collapse across measured audio-summary, CNN, and transformer branches. This is why our study is stronger as a biomedical AI validation paper than as a SOTA classifier paper.


