# Draft Results Section: Temporal Robustness

This draft is intentionally limited to results language. It does not decide the final contribution or paper framing.

## RQ1: Can multimodal audio detect COVID internally?

Under the existing participant-level split, full multimodal fusion across breath, cough, and speech achieved AUROC 0.873 and AUPRC 0.807. This establishes that the pipeline can learn a strong source-domain signal when train and test participants are randomly separated but drawn from the same temporal collection distribution.

## RQ2: Does performance survive temporal stress?

Performance weakened under calendar control and collapsed under strict chronological evaluation. The calendar-balanced participant split achieved AUROC 0.787, while the early-to-late temporal holdout achieved AUROC 0.566. The participant-to-temporal AUROC difference was -0.308 with 95% bootstrap CI -0.391 to -0.221 and two-sided bootstrap p=<0.0002. This indicates that the original participant-split performance is not stable under temporal stress.

## RQ3: Does performance survive external transfer?

The best independent Coswara-to-COUGHVID external transfer result achieved AUROC 0.553 and AUPRC lift 0.005 over the COUGHVID prevalence baseline. The external result is nearly identical to the temporal holdout result, suggesting that chronological stress inside Coswara and independent dataset transfer expose the same fragility.

## RQ4: What drives the failure?

The temporal metadata ablation isolates recording month as a major driver of poor chronological generalization in the full safe metadata model.

| Metadata configuration | Temporal AUROC | Temporal AUPRC |
| --- | ---: | ---: |
| Full metadata | 0.531 | 0.837 |
| Remove year | 0.531 | 0.837 |
| Remove month | 0.779 | 0.935 |
| Remove year + month | 0.779 | 0.935 |

Removing recording month increased temporal full-safe-metadata AUROC from 0.531 to 0.779, while removing recording year alone did not change the temporal result. This means recording month is not merely predictive; in the early-to-late setting it encodes collection-period structure that harms chronological generalization. Together with the participant-split, time-stratified, temporal-holdout, and external-transfer results, this supports the conclusion that temporal/protocol confounding inflates internal COVID-audio performance and weakens out-of-period transfer.
