# Covid-RARS Results Summary

- Best external transfer: auroc=0.553; this remains weak and cautionary.
- IPW-controlled audio: auroc=0.780; audio signal persists but is reduced after measured confounder control.
- Strongest metadata confounding result: auroc=0.964; non-audio variables strongly predict labels.
- Worst external calibration shift: ece=0.286; external probabilities are not reliable calibrated risks.
- Dataset-domain separability: domain_auroc=0.966; this tests whether representations encode source artifacts.
- CORAL domain adaptation: auroc=0.551; this tests whether simple covariance alignment closes the external gap.
- Prevalence recalibration: ece_reduction=0.285; this separates calibration repair from discrimination failure.

Conclusion: the project is defensible as a robustness and external-validation study, not as a clinically deployable diagnostic model.
