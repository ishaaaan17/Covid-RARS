# Covid-RARS Phase-Wise Results Brief

This brief organizes the completed research into three presentation phases. It is written for result communication and paper planning, not as a final manuscript. The old HTML slide deck is not used.

## Phase 1: Leakage-Controlled Internal Audio Baseline

### What Was Established

The first phase established a clean internal audio baseline on Coswara using participant-level separation, quality checks, reproducible preprocessing, and calibrated model evaluation. The goal was not to claim clinical deployability, but to determine whether the cleaned pipeline could produce a coherent internal audio signal after avoiding obvious leakage.

### Main Evidence

| Result | Value | Interpretation |
| --- | ---: | --- |
| Best internal quality-weighted fusion AUROC | 0.879 | Strong internal discrimination under the current leakage-controlled split |
| Internal fusion AUPRC | 0.832 | Good internal precision-recall behavior in the same source-domain setting |
| Internal balanced accuracy | 0.804 | Internal thresholded performance is coherent, but still source-domain only |
| Internal Brier score | 0.190 | Calibration is imperfect but usable for internal reliability comparison |
| Internal ECE | 0.150 | Probabilities are not perfectly calibrated even before external shift |

### Decision Taken

The internal result is useful as the starting point, but it cannot be used alone as the project conclusion. It shows that the pipeline can learn a source-domain signal, while later phases test whether that signal is robust, clinically reliable, and externally transferable.

## Phase 2: Representation Comparison And External Transfer

### What Was Added

The second phase tested whether stronger representations improve generalization. The pipeline compared traditional handcrafted features and learned frozen embeddings:

| Representation | Role In Study |
| --- | --- |
| MFCC | Traditional handcrafted baseline and continuity with the original pipeline |
| OpenSMILE eGeMAPSv02 | stronger clinical-acoustic handcrafted baseline |
| BEATs | modern transformer-style general-audio frozen representation |
| PANNs CNN14 | CNN-based audio embedding comparison |

### External Transfer Result

Coswara-trained models transferred poorly to COUGHVID across all representation families.

| Representation | Best external AUROC | Best external AUPRC | Interpretation |
| --- | ---: | ---: | --- |
| MFCC | 0.535 | 0.042 | Weak external transfer despite being the best AUPRC row |
| OpenSMILE eGeMAPSv02 | 0.552 | 0.039 | Slight AUROC improvement, still weak |
| BEATs | 0.553 | 0.039 | Highest AUROC, but not meaningfully deployable |
| PANNs CNN14 | 0.502 | 0.035 | Essentially prevalence-level behavior |

### Internal COUGHVID Context

COUGHVID internal baselines showed that representations can carry signal inside the target dataset itself:

| COUGHVID internal baseline | AUROC |
| --- | ---: |
| MFCC | 0.781 |
| OpenSMILE eGeMAPSv02 | 0.763 |
| BEATs | 0.756 |
| PANNs CNN14 | 0.652 |

This separates representation capacity from cross-dataset transfer. The representations are not useless; the failure appears when a Coswara-trained decision rule is transferred to COUGHVID.

### Related-Paper Position

The related-paper comparison shows that many COVID audio studies report high internal or cross-dataset accuracy, but often without the full reliability audit used here. The important distinction is that this project does not compete as a new architecture paper. It evaluates whether apparent COVID audio performance survives external validation, confounding analysis, calibration checks, operating-point analysis, domain-shift diagnostics, and statistical comparison.

## Phase 3: Robustness, Confounding, Calibration, And Research Closure

### Why Phase 3 Was Needed

After Phase 2, the key question became: why does strong internal performance fail externally? Phase 3 closes that question through metadata confounding, inverse-propensity weighting, calibration-under-shift, clinical operating points, domain separability, domain adaptation, prevalence recalibration, paired bootstrap comparison, and manuscript-support analyses.

In compact form, Phase 3 establishes: metadata AUROC 0.964, demographic/protocol AUROC 0.914, residual recording_year SMD = 0.724 after cap=2 IPW sensitivity, BEATs domain AUROC 0.966, BEATs CORAL MMD 0.055 to 0.004 without useful AUROC rescue, external ECE 0.286 before prevalence correction, ECE 0.001 after prevalence recalibration, and paired bootstrap p = 0.304 for best external model versus baseline.

### Metadata Confounding

| Audit model | AUROC | AUPRC | Interpretation |
| --- | ---: | ---: | --- |
| Full safe metadata | 0.964 | 0.928 | Non-audio metadata strongly predicts labels |
| Symptoms only | 0.932 | 0.898 | Symptom fields are strong label proxies |
| Demographic/protocol only | 0.914 | 0.737 | Non-symptom structural variables alone are highly predictive |

The demographic/protocol result is central. It means the label structure is strongly predictable without audio and without symptoms.

### Linear Attribution Of The Demographic/Protocol Model

The demographic/protocol-only model was analyzed using exact linear-logit attribution for the standardized logistic regression model. The top drivers were:

| Rank | Feature | Feature group | Interpretation |
| ---: | --- | --- | --- |
| 1 | recording_year | recording_protocol | strongest driver of demographic/protocol label prediction |
| 2 | recording_month | recording_protocol | additional collection-period signal |
| 3 | country_India | demographic | geographic composition contributes to label predictability |
| 4 | age | demographic | demographic contribution, smaller than time variables |
| 5 | duration_sec | recording_protocol | recording-length/protocol contribution |

This supports a specific mechanism: internal labels are strongly associated with collection time and protocol structure. The correct wording is temporal/protocol confounding, not simply generic dataset shift.

### Confounding-Controlled Audio Evaluation

| Evaluation | AUROC | AUPRC | Balanced accuracy | ESS | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Unweighted quality-weighted fusion | 0.879 | 0.832 | 0.804 | 318.0 | Strong internal result before confounder weighting |
| IPW label-propensity controlled | 0.780 | 0.537 | 0.721 | 130.4 | Audio signal remains visible but is substantially reduced |

The IPW-controlled estimate should be treated as qualified evidence, not fully deconfounded performance. ESS = 130.4 indicates that effective sample size drops sharply after weighting.

### IPW Sensitivity And Residual Imbalance

The cap=2 IPW sensitivity analysis identified the worst residual balance issue:

| Feature | Before abs SMD | After abs SMD | Weight config | Interpretation |
| --- | ---: | ---: | --- | --- |
| recording_year | 1.384 | 0.724 | ipw_cap_2_q_0.95 | severe residual temporal/protocol imbalance |
| country_India | 0.525 | 0.438 | ipw_cap_2_q_0.95 | moderate residual geography imbalance |
| country_United States | 0.296 | 0.250 | ipw_cap_2_q_0.95 | moderate residual geography imbalance |

The same variable that drives the demographic/protocol model, recording_year, remains the hardest confounder to balance. The IPW result is therefore best framed as partial control over measured confounding with unresolved temporal imbalance.

### External AUPRC Lift Over Prevalence

COUGHVID external positive prevalence is approximately 0.034. External AUPRC barely exceeds this prevalence baseline.

| Representation | External AUROC | External AUPRC | AUPRC lift over prevalence |
| --- | ---: | ---: | ---: |
| MFCC | 0.535 | 0.042 | +0.008 |
| BEATs | 0.553 | 0.039 | +0.005 |
| OpenSMILE eGeMAPSv02 | 0.552 | 0.039 | +0.005 |
| PANNs CNN14 | 0.502 | 0.035 | +0.001 |

The best external precision-recall lift is only +0.008. PANNs is effectively prevalence-level. This makes the external failure clinically interpretable, not just statistically visible.

### Clinical Operating Points

The quality-weighted fusion operating points show the internal tradeoff clearly:

| Constraint | Threshold | Sensitivity | Specificity | Precision | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| specificity >= 0.800 | 0.337 | 0.806 | 0.814 | 0.675 | reasonable internal screening-like point, source-domain only |
| specificity >= 0.900 | 0.356 | 0.699 | 0.907 | 0.783 | higher specificity loses sensitivity |
| specificity >= 0.950 | 0.374 | 0.612 | 0.958 | 0.875 | too much sensitivity loss for screening claims |
| sensitivity >= 0.900 | 0.313 | 0.903 | 0.591 | 0.514 | high sensitivity creates many false positives |

These operating points support the conclusion that the model should not be presented as a clinical diagnostic or screening system.

### Calibration Under External Shift

External probabilities are badly miscalibrated before correction.

| Calibration finding | Value | Interpretation |
| --- | ---: | --- |
| Worst external ECE | 0.286 | severe external probability inflation |
| External observed prevalence | 0.034 | low target prevalence in COUGHVID |
| External mean predicted probability | 0.321 | source-calibrated probabilities overestimate target risk |

This shows that raw external probabilities cannot be used as calibrated risks.

### Prevalence Recalibration

Prevalence recalibration separates calibration repair from discrimination failure.

| Finding | Value | Interpretation |
| --- | ---: | --- |
| ECE before recalibration | 0.286 | severe calibration shift |
| ECE after prevalence recalibration | 0.001 | prevalence mismatch explains most calibration error |
| AUROC after recalibration | 0.484 | discrimination is not rescued |

The model can be recalibrated to the target base rate, but it still cannot rank individual positives correctly. Calibration failure and discrimination failure are separate problems.

### Domain-Shift Audit

The domain classifier showed that learned representations encode dataset identity.

| Domain-shift finding | Value | Interpretation |
| --- | ---: | --- |
| BEATs domain AUROC | 0.966 | BEATs strongly separates Coswara from COUGHVID |

This means the representation contains dataset-specific structure. It helps explain why a decision boundary trained in Coswara does not transfer reliably to COUGHVID.

### CORAL Domain Adaptation Closure

CORAL tested whether second-order feature alignment could close the external gap.

| Representation | MMD before | MMD after CORAL | External AUROC effect | Interpretation |
| --- | ---: | ---: | --- | --- |
| BEATs | 0.055 | 0.004 | 0.542 to 0.549 | large distribution-distance reduction without useful discrimination gain |
| PANNs | 0.033 | 0.002 | about chance-level after correction | distribution alignment does not rescue label discrimination |
| OpenSMILE eGeMAPSv02 | 0.005 | 0.025 | no rescue | CORAL can worsen already-low second-order distance |

The key finding is that reducing measured feature-space mismatch is not sufficient. The remaining failure is consistent with label-generating mismatch and temporal/protocol confounding, not just covariate shift.

### Paired Bootstrap Comparison

The paired bootstrap comparison prevents overclaiming small representation differences.

| Comparison | AUROC difference | 95% CI | p-value | Interpretation |
| --- | ---: | --- | ---: | --- |
| Best external model vs baseline | 0.025 | [-0.024, 0.072] | 0.304 | no statistically reliable external representation advantage |

This supports the claim that external failure is systematic across representations.

### Unknown-Label Audit

Coswara contains 5,688 unknown-label rows excluded from supervised training and evaluation. The known and unknown subsets were broadly similar on coarse observed metadata.

| Group | Rows | Participants | Mean age | Median age | Mean duration | Top country | Top quality flag | Top gender |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Known labels | 19,024 | 2,114 | 35.24 | 31 | 10.06 sec | India | ok | male |
| Unknown labels | 5,688 | 632 | 34.87 | 30 | 9.17 sec | India | ok | male |

The unknown-label subset does not appear grossly different on these observed variables, but non-random missingness on unobserved factors cannot be excluded.

### Phase 3 Bottom Line

The strongest internal audio result is real as a source-domain modeling result, but the broader research conclusion is cautionary. Metadata predicts labels more strongly than audio, recording time is the dominant non-symptom structural driver, IPW leaves severe temporal imbalance, external AUPRC barely exceeds prevalence, BEATs encodes dataset identity, CORAL reduces feature-space MMD without rescuing AUROC, prevalence recalibration fixes ECE but not discrimination, and paired bootstrap shows no reliable external representation advantage. The final result is a robustness and external-validity study, not a clinical diagnostic model and not a clinical screening deployment claim.

## Final Non-Writing Closure State

The research implementation is complete. The remaining work is manuscript writing, venue formatting, cover letter preparation, and submission packaging. No additional model tuning should be added unless requested after review of these results.
