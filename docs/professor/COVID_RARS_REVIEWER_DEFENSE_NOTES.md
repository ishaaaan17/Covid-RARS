# COVID-RARS Reviewer Defense Notes

Date: 2026-07-19

Purpose: preserve the current honest interpretation of the research after the protocol-matched multimodal CV runs. This is not manuscript prose. It is a defense and reviewer-risk note so that the argument does not get distorted later.

## Current Bottom Line

The newest protocol-matched runs do not prove that our final pipeline is SOTA-competitive with the highest reported COVID-audio papers. They prove a narrower but still useful point:

1. The final multimodal pipeline is functional and reasonably strong under random participant-disjoint internal CV.
2. It does not close the gap to papers reporting AUC/AUROC around 0.90 or higher.
3. The reliability-audit evidence remains valid, but the "our model matches SOTA under their easy split, then collapses under strict validation" support argument is not supported by the latest numbers.

Therefore the safe claim is:

> Our evaluated multimodal respiratory-audio pipelines reach strong but not SOTA internal performance, while temporal, external, calibration, shortcut, and subgroup audits show that internal performance is not a reliable proxy for deployment validity.

The unsafe claim is:

> Our final pipeline matches or beats SOTA under paper-style settings.

## Latest Protocol-Matched Results

These runs used a paper-style participant-disjoint 10-fold repeated split with approximately 70 percent train, 10 percent validation, and 20 percent test participants per fold. Every fold had zero participant overlap.

### Clean Fold-Local Feature Selection

This is the cleaner primary protocol-matched run. Top-800 ComParE+IS10 acoustic features are selected inside each fold using the per-modality mean selection policy.

| Model setting | AUROC | AUPRC | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|
| Cough + breath + speech, validation-AUPRC weighted fusion | 0.8489 +/- 0.0195 | 0.7969 +/- 0.0218 | 0.7660 | 0.6817 |
| Cough + breath + speech, stacked logistic fusion | 0.8486 +/- 0.0195 | 0.7966 +/- 0.0224 | 0.7626 | 0.6772 |
| Cough + breath + speech, uniform mean fusion | 0.8481 +/- 0.0195 | 0.7956 +/- 0.0214 | 0.7597 | 0.6721 |

Interpretation: this is the most defensible internal paper-style result because feature selection is nested inside each split. It is also lower than desired for a SOTA comparison.

### Fixed Top-800 Feature-Bank Sensitivity

This run reused the final fixed top-800 feature bank and is labeled as a sensitivity result. It is faster and closer to the final fixed feature-bank pipeline, but it should not be presented as the cleanest fold-local selection result.

| Model setting | AUROC | AUPRC | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|
| Cough + breath + speech, uniform mean fusion | 0.8641 +/- 0.0195 | 0.8159 +/- 0.0240 | 0.7768 | 0.6962 |
| Cough + breath + speech, validation-AUPRC weighted fusion | 0.8641 +/- 0.0196 | 0.8159 +/- 0.0237 | 0.7761 | 0.6966 |
| Cough + breath + speech, stacked logistic fusion | 0.8637 +/- 0.0193 | 0.8151 +/- 0.0228 | 0.7759 | 0.6948 |

Interpretation: this is the best protocol-matched internal result from the final multimodal pipeline, but it still remains below the highest reported 0.90+ papers.

### Comparison Against Our Other Validation Rungs

| Validation setting | Approximate AUROC | Interpretation |
|---|---:|---|
| Original internal participant split | 0.897 | Highest internal result from the final pipeline. |
| Paper-style clean multimodal CV | 0.849 | Clean participant-disjoint repeated CV with fold-local feature selection. |
| Paper-style fixed top-800 sensitivity | 0.864 | Stronger internal sensitivity result, but fixed feature-bank caveat applies. |
| Strict temporal early-to-late | about 0.67 to 0.70 | Deployment-like time shift causes large degradation. |
| COUGHVID external transfer | about 0.48 to 0.55 depending on model family | Cross-dataset portability largely collapses. |

Important: the paper-style run is lower than the original internal 0.897, but still far above strict temporal and external transfer. The validation ladder remains real, but the new internal paper-style rung is not high enough to silence every "weak pipeline" criticism.

## What This Run Strengthened

The run strengthened these points:

1. The multimodal pipeline is materially stronger than cough-only. Earlier cough-only protocol-matched SVC was about 0.799 AUROC, while multimodal rises to 0.849 clean and 0.864 fixed-bank sensitivity.
2. The final system benefits from multimodal information. Cough + breath + speech is better than cough-only in this protocol.
3. The split audit is clean: participant-disjoint, 10 folds, 1461 train participants, 209 validation participants, 418 test participants, and zero overlap in every fold.
4. Fusion does not depend on arbitrary manually hardcoded weights. The fusion methods are explicit:
   - uniform mean: equal averaging of selected branch probabilities;
   - validation-AUPRC weighted mean: branch weights are derived from validation AUPRC using `max(validation_AUPRC - 0.5, 0.01)` and normalized;
   - stacked logistic fusion: logistic regression trained on validation branch probabilities.

## What This Run Weakened

The run weakened this intended support argument:

> Under paper-style settings, our final pipeline becomes essentially comparable to the high-number papers, so later drops can be attributed mainly to strict validation.

That argument is not supported strongly enough. The best fixed-bank sensitivity result is about 0.864 AUROC, and the clean fold-local result is about 0.849 AUROC. If a prior paper reports about 0.90 or higher, the gap is not within 1 percent.

The safe interpretation is:

> The pipeline is reasonably strong but not SOTA. The reliability audit remains valuable, but the work should not depend on claiming that our final classifier matches the highest internal-leaderboard papers.

## Strict Reviewer Objection

Likely reviewer criticism:

> The temporal and external failures may be due to a weaker pipeline, not due to structural shortcut learning or benchmark unreliability.

This is a real risk. It should not be dismissed.

### Strongest Defense

The defense is not that our pipeline is SOTA. The defense is that the failure pattern is not explained by one weak model alone.

Evidence:

1. The final multimodal pipeline reaches up to 0.864 AUROC under internal participant-disjoint paper-style CV. This is not a trivial or broken classifier.
2. External degradation appears across multiple model families: LightGBM, SVC, CatBoost, XGBoost, WavLM, and CNN-BiGRU.
3. WavLM is a transformer/self-supervised representation branch, yet it also failed externally.
4. CNN-BiGRU is a spectrogram deep-learning branch, yet it also failed externally.
5. Metadata-only models achieve very high internal prediction, around 0.964 AUROC, indicating strong dataset/context shortcut structure independent of audio model quality.
6. Shuffle-label sanity checks collapse toward chance, which argues against a simple software leakage explanation.
7. Recording year/protocol variables dominate metadata importance, supporting temporal and administrative shortcut mechanisms.
8. Feature-selection stability over time is very low, showing that the selected acoustic features are non-stationary across time.
9. Calibration and clinical utility degrade under shift, so the issue is not only threshold choice.

### Honest Limitation

The limitation that must be stated:

> We include handcrafted acoustic feature fusion, WavLM transformer representations, and CNN-BiGRU spectrogram modeling, but we do not reproduce every published high-performing HST/AST transformer architecture. Therefore, the conclusion is about the evaluated benchmark pipelines and validation protocols, not proof that no possible architecture can generalize.

This limitation is better stated proactively than discovered by a reviewer.

## How To Answer The "Weak Pipeline" Question

If asked:

> Are the low strict/external results just because your pipeline is weak?

Answer:

> That is a valid concern, so we ran a protocol-matched internal participant-CV experiment. The final multimodal pipeline reaches 0.849 AUROC with fold-local feature selection and 0.864 AUROC when using the fixed final top-800 feature bank. So the pipeline is not broken, and it clearly improves over cough-only. However, it is not SOTA. To avoid overclaiming, our conclusion is not that every possible COVID-audio model fails, but that internal validation performance alone is insufficient and that evaluated respiratory-audio pipelines are highly sensitive to temporal, metadata, and external-domain shift.

If asked:

> Why are high papers above 0.90 and your protocol-matched result is lower?

Answer:

> Our run matched the broad participant-CV style but did not reproduce their exact architecture, preprocessing, augmentation, inclusion rules, or hyperparameter choices. It is therefore a protocol-style comparison of our final pipeline, not an exact reproduction of the base paper. A direct architecture comparison would require implementing one base paper, such as HST/AST, under the same folds and then applying the same temporal/external audit.

If asked:

> Does this invalidate the shortcut-learning claim?

Answer:

> No. It weakens the claim that our classifier is leaderboard-competitive, but the shortcut-learning evidence is separate: metadata-only prediction is extremely high, recording year/protocol variables dominate, shuffle-label controls collapse to chance, time-based validation degrades strongly, and external transfer fails across multiple model families.

## What Not To Say

Do not say:

> Prior papers fabricated results.

Say instead:

> Prior papers often report internal protocol-dependent performance. Our audit shows that internal performance does not necessarily imply temporal or external deployment reliability.

Do not say:

> Our method beats SOTA.

Say instead:

> Our method is a strong evaluated baseline with multimodal fusion, but the main contribution is the reliability audit.

Do not say:

> COUGHVID proves full multimodal transfer failed.

Say instead:

> COUGHVID is cough-only, so it tests the cough/external portability component, not full cough + breath + speech multimodal transfer.

Do not say:

> The base papers are wrong.

Say instead:

> The base papers answer a different question: internal classification under their chosen protocol. Our work asks whether such performance survives stricter temporal, confounding, calibration, and external-transfer checks.

## Strong Manuscript Position

The strongest manuscript position is:

> We evaluate multimodal respiratory-audio COVID-19 classification using a broad model bank and then audit the same system under progressively stricter validation conditions. The results show that internal participant-disjoint CV can produce strong performance, but temporal shift, metadata confounding, calibration failure, feature instability, and external transfer substantially weaken deployment claims.

This is stronger and safer than:

> We propose a new SOTA COVID-19 audio classifier.

## Reviewer-Risk Table

| Reviewer concern | Risk level | Evidence we have | Remaining weakness |
|---|---:|---|---|
| Pipeline is weak | High | 0.849 clean CV; 0.864 fixed-bank sensitivity; multimodal better than cough-only | Not at 0.90+ SOTA level; no exact HST reproduction |
| External failure is modality mismatch | Medium | Cough-only matched comparisons and multiple cough model families fail externally | COUGHVID cannot validate breath/speech transfer |
| Shortcut claim is speculative | Medium-low | Metadata-only AUROC, permutation importance, subgroup checks, shuffle-label sanity | Need careful wording: shortcut risk, not proof of causal disease absence |
| Temporal degradation is random seed artifact | Low | Multi-seed temporal robustness and reverse temporal checks | Exact numbers should be reported with uncertainty |
| Threshold/calibration caused external failure | Low | Recalibration-only check and calibration summaries | Make clear discrimination AUROC remains poor |
| Too many exploratory tests | Medium | Multiplicity/scope note exists | Keep primary ladder separate from exploratory evidence |

## If One More Experiment Is Forced

The only high-value experiment that directly addresses the weak-pipeline objection is:

> Reproduce one strong published transformer-style architecture, such as HST/AST, on the same internal participant-CV split and then run the same temporal/external audit.

Do not run another broad model search. It will consume time and probably create more unstructured results. A single focused reproduction is more defensible.

Expected value:

1. If HST/AST reaches around 0.90 internally and still fails temporally/externally, the paper becomes much stronger.
2. If HST/AST also fails to reach 0.90 on our cleaned pipeline, that suggests implementation/data/protocol differences drive much of the prior gap.
3. If HST/AST generalizes externally, the current manuscript thesis must be revised.

This should be presented as an extension, not silently mixed into the current results.

## Final One-Minute Defense

> We added the protocol-matched internal CV comparison because the main criticism was that strict validation might only be exposing a weak pipeline. The result is mixed. The final multimodal pipeline reaches 0.849 AUROC with clean fold-local feature selection and 0.864 AUROC with the fixed final top-800 feature bank, so it is not broken and it improves clearly over cough-only. But it is not SOTA and does not match the highest 0.90+ papers. Therefore, we should not frame the work as a leaderboard classifier paper. The defensible contribution is a rigorous validation audit: internal respiratory-audio performance is much higher than temporal and external performance, and this is supported by metadata shortcut evidence, feature instability, calibration checks, shuffle-label controls, and multi-model external-transfer failure. The remaining limitation is that an exact HST/AST reproduction would be needed to fully eliminate the weak-architecture objection.

