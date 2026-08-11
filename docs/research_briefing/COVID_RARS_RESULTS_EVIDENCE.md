# Covid-RARS Results Evidence Ledger

This file is the evidence ledger for research discussion and manuscript writing. It records the main result, the source artifact, and the safe interpretation.

## Source Snapshot

The current evidence is generated from the pushed branch:

```text
publication-upgrade-confounding-da
result artifacts synced through: a276a20 Add incremental audio metadata reviewer evidence
briefing documentation commits added after that evidence snapshot
```

Core generated artifacts:

| Evidence area | Source artifact |
|---|---|
| Final validation ladder | `reports/tables/compare_is10_final_validation_summary.csv` |
| COUGHVID external family comparison | `reports/tables/reviewer_external_model_family_transfer_summary.csv` |
| Confidence intervals for drops | `reports/tables/final_validation_delta_bootstrap_ci.csv` |
| Calibration | `reports/tables/final_validation_calibration_summary.csv`, `reports/figures/final_validation_calibration_curves.svg` |
| Operating points | `reports/tables/final_validation_fixed_sensitivity_operating_points.csv` |
| Decision curve analysis | `reports/tables/reviewer_decision_curve_analysis.csv`, `reports/figures/reviewer_decision_curve_analysis.svg` |
| Metadata confounding | `reports/tables/metadata_confounding_*.csv` |
| Shuffle-label sanity | `reports/tables/final_validation_shuffle_label_sanity.csv`, `reports/tables/metadata_confounding_shuffle_label_sanity.csv`, `reports/tables/compare_is10_shuffle_retrain_summary.csv` |
| Recalibration-only check | `reports/tables/coughvid_partial_recalibration_metrics.csv` |
| Support overlap | `reports/tables/reviewer_support_overlap_positivity.csv` |
| Feature-selection stability | `reports/tables/reviewer_feature_selection_stability.csv` |
| Incremental metadata+audio | `data/outputs/metrics/reviewer_incremental_audio_metadata_metrics.csv` |
| Label construction note | `reports/final/LABEL_CONSTRUCTION_AUDIT.md` |
| Multiplicity note | `reports/final/MULTIPLICITY_AND_ANALYSIS_SCOPE.md` |

## Result 1: Final Validation Ladder

| Validation setting | Best selected row | AUROC | AUPRC | Balanced accuracy | F1 | n |
|---|---|---:|---:|---:|---:|---:|
| Existing participant split | Multimodal cough+speech stacked logistic fusion | `0.897` | `0.863` | `0.825` | `0.752` | `314` |
| Time-stratified participant split | Multimodal cough+breath+speech uniform mean | `0.849` | `0.783` | `0.783` | `0.705` | `431` |
| Early-to-late temporal split | Breath top-4 validation ensemble | `0.698` | `0.896` | `0.656` | `0.751` | `411` |

Safe interpretation:

- The pipeline is strong internally.
- Performance decreases when validation becomes more deployment-like.
- Temporal validation shows the model is not learning a fully stable acoustic biomarker.
- The high temporal AUPRC must be interpreted carefully because prevalence/base-rate changes can inflate or distort threshold-independent ranking metrics.

What to say:

```text
Our internal score is high, but the validation ladder shows that the same system becomes less reliable when calendar structure and external deployment are stressed.
```

## Result 2: COUGHVID External Transfer

COUGHVID is cough-only. Therefore, the fair external test is cough-to-cough transfer, not full multimodal fusion transfer.

| Model family | Internal AUROC | COUGHVID AUROC | AUROC drop | Internal AUPRC | COUGHVID AUPRC |
|---|---:|---:|---:|---:|---:|
| ComParE+IS10 CatBoost | `0.849` | `0.531` | `0.318` | `0.788` | `0.040` |
| ComParE+IS10 LightGBM | `0.853` | `0.543` | `0.310` | `0.797` | `0.040` |
| ComParE+IS10 SVC | `0.868` | `0.523` | `0.345` | `0.812` | `0.037` |
| ComParE+IS10 XGBoost | `0.850` | `0.532` | `0.318` | `0.780` | `0.039` |
| WavLM transformer | `0.812` | `0.484` | `0.328` | `0.734` | `0.032` |
| CNN-BiGRU spectrogram | `0.737` | `0.548` | `0.189` | `0.574` | `0.044` |

Safe interpretation:

- The external collapse is consistent across classical, kernel, transformer, and neural spectrogram families.
- WavLM is important defensively: it proves we did try a transformer and that transformer pretraining did not solve the transfer problem.
- CNN-BiGRU has the best external AUROC among the tested deep branches, but `0.548` is still weak and not clinically useful.

What to say:

```text
The external failure is not only a measured-audio-summary issue. WavLM and CNN-BiGRU were also tested, and neither produced reliable external discrimination.
```

## Result 3: Bootstrap Drop Confidence Intervals

Key delta result:

| Comparison | Metric | Drop | Confidence interval |
|---|---|---:|---:|
| Internal participant split minus COUGHVID external | AUROC | `0.354` | `[0.300, 0.404]` |
| Internal participant split minus COUGHVID external | AUPRC | `0.822` | `[0.764, 0.871]` |

Cough-only matched family drops:

| Model | AUROC drop pattern |
|---|---|
| LightGBM | roughly `0.30` AUROC drop |
| CatBoost | roughly `0.30` AUROC drop |
| XGBoost | roughly `0.30` AUROC drop |
| SVC | roughly `0.32-0.35` AUROC drop |

Safe interpretation:

- The external collapse is not a small fluctuation.
- The family-level consistency makes the result stronger.
- The interval should be described as a bootstrap uncertainty estimate for the observed validation-drop comparison, not as a universal law.

## Result 4: Paper-Comparable 10-Fold CV

Paper-comparable cough CV was run because many papers report 10-fold internal metrics.

| Model | Aggregate cough 10-fold AUROC | AUROC std | AUPRC |
|---|---:|---:|---:|
| SVC RBF | `0.819` | `0.029` | `0.728` |
| LightGBM | `0.819` | `0.027` | `0.732` |
| CatBoost | `0.806` | `0.027` | `0.717` |
| XGBoost | `0.803` | `0.026` | `0.715` |

Safe interpretation:

- On paper-comparable cough-only CV, the method is decent but not SOTA.
- This result should be used to compare protocol strictness, not to claim leaderboard dominance.
- The final paper should not hide this. It shows honest benchmarking.

## Result 5: Metadata Confounding

Main metadata result:

| Metadata branch | Observed AUROC | Shuffle-label AUROC |
|---|---:|---:|
| Full safe metadata | `0.964` | about `0.499` |
| Symptoms-only | `0.932` | about `0.500` |
| Demographic/protocol-only | `0.914` | about `0.501` |

Safe interpretation:

- Metadata alone can predict labels very strongly.
- The shuffle-label sanity check drops to chance, so the result is not a leakage bug in the scoring script.
- Recording protocol/time variables are not harmless metadata; they can encode collection and pandemic-wave effects.

What to say:

```text
This is the central shortcut-learning evidence. If metadata can predict labels at 0.964 AUROC, then audio models trained on the same collection process must be checked carefully for indirect shortcut learning.
```

## Result 6: Metadata Permutation Importance

Permutation audits show recording/protocol variables dominate metadata prediction. The group importance share for administrative recording-protocol variables ranges approximately from `71.7%` to `93.4%`, driven heavily by `recording_year`.

Safe interpretation:

- The shortcut is not vague.
- The mechanism is connected to collection time/protocol, not just random demographic noise.
- This supports the temporal-drift story.

## Result 7: Temporal Robustness

Multi-seed robustness summary:

| Setting | Mean AUROC | Std |
|---|---:|---:|
| Existing participant-split stacked fusion | `0.895` | `0.003` |
| Strict early-to-late temporal validation | about `0.691` | `0.006` |

Reverse temporal result:

| Setting | AUROC | AUPRC | F1 | ECE |
|---|---:|---:|---:|---:|
| Late-to-early temporal validation | `0.920` | `0.029` | `0.011` | `0.471` |

Safe interpretation:

- The internal result is stable across seeds.
- Temporal degradation is also stable across seeds.
- Reverse temporal AUROC can look high while AUPRC, F1, and calibration are poor. This is a good example of why AUROC alone is unsafe for clinical screening claims.

## Result 8: Feature Non-Stationarity

Feature-selection stability:

| Analysis | Value |
|---|---:|
| Top-k | `800` |
| Early rows | `6512` |
| Late rows | `6173` |
| Shared selected features | `110` |
| Union selected features | `1490` |
| Jaccard overlap | `0.074` |

Safe interpretation:

- The acoustic features selected as important are not stable over time.
- This converts "temporal AUROC dropped" into a mechanism: the discriminative feature set itself changes.

## Result 9: Support Overlap / Positivity

Support-overlap diagnostic:

| Quantity | Value |
|---|---:|
| Domain classifier AUROC | `0.750` |
| Source rows | `2861` |
| External rows | `8331` |
| Common features used | `500` |
| External within source probability band | `0.748` |
| External probably outside source support | `0.252` |

Safe interpretation:

- Coswara and COUGHVID are geometrically different in feature space.
- About one quarter of external examples fall outside the source-domain probability band.
- This explains why simple covariate correction or recalibration cannot fully rescue external performance.

## Result 10: Recalibration-Only Check

COUGHVID partial recalibration was run to test whether failure is merely threshold/calibration mismatch.

Safe interpretation:

- Recalibration may change threshold metrics, but AUROC remains weak.
- Therefore, the external failure is a discrimination/ranking problem, not only a threshold-setting problem.

Use this wording:

```text
Target-domain recalibration does not convert the external model into a useful discriminator, so the collapse is not simply a calibration artifact.
```

## Result 11: Clinical Operating Points and Decision Curves

At fixed high sensitivity (`>= 0.90`) on COUGHVID:

- Specificity collapses to roughly `0.11-0.16`.
- Precision remains roughly `0.035-0.037`.
- COUGHVID positive prevalence is about `0.034`.

Safe interpretation:

- Precision is barely above prevalence.
- At a screening-like sensitivity, the model would generate many false positives without useful triage benefit.
- Decision curve analysis supports the same conclusion: weak or negative clinical net benefit under external transfer.

## Result 12: Incremental Audio + Metadata Value

This was added to answer a strict reviewer question: does audio add value over metadata/symptoms?

Best symptoms-only aligned candidate:

| Branch | AUROC | AUPRC | n |
|---|---:|---:|---:|
| Symptoms-only metadata | `0.888` | `0.825` | `61` |
| Audio-only | `0.818` | `0.756` | `61` |
| Metadata + audio | `0.951` | `0.879` | `61` |
| Delta over metadata | `+0.063` AUROC | CI `[-0.005, 0.149]`, DeLong p `0.104` | |

Full-safe-metadata aligned candidate pattern:

- Metadata-only is already very high.
- Adding audio gives small, statistically insecure gain.
- In representative rows, delta over metadata is around `+0.009` AUROC with p around `0.54`.

Safe interpretation:

- Audio may add some signal over symptoms-only in small aligned subsets.
- The evidence is not strong enough to claim clinically decisive incremental value.
- Audio adds little once full safe metadata/context is available.
- This result protects the paper from the criticism that we never tested incremental value.

## Result 13: Shuffle Sanity Checks

Sanity checks performed:

- Metadata prediction label-shuffle sanity.
- Audio prediction shuffle-label sanity.
- ComParE+IS10 shuffle-retrain sanity.

Safe interpretation:

- Pipelines collapse toward chance after label permutation.
- This reduces concern that the high internal results are caused by simple software leakage.
- It does not remove the broader shortcut-learning problem; in fact, it helps show the shortcut is structural.

## Result 14: Label Construction

The label construction audit explicitly says:

- Coswara and COUGHVID labels are not assumed clinically identical.
- External transfer should be interpreted as dataset-transfer stress testing.
- External failure is not proof that the same clinical label has identical annotation semantics across corpora.

This is important because it prevents overclaiming.

## Safe Overall Conclusion

The strongest defensible conclusion is:

> The project demonstrates that strong internal respiratory-audio COVID detection performance can be achieved, but strict temporal validation, metadata-confounding analysis, calibration, and external transfer reveal that these models are not robust enough for deployment claims without stronger validation.

## Unsupported or Risky Claims

Do not write these:

- "Our model beats all SOTA COVID-audio papers."
- "Prior papers fabricated results."
- "The external result proves no COVID acoustic marker exists."
- "COUGHVID proves full multimodal fusion failure."
- "WavLM was not tried."
- "The audio+metadata result proves audio is clinically useful beyond symptoms."


