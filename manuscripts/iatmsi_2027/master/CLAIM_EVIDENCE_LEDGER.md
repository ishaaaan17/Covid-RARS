# Claim-Evidence Ledger

This ledger prevents the manuscript from combining incompatible cohorts,
modalities, or estimands. It names internal artifacts for author verification;
artifact paths do not belong in the paper prose.

| Claim | Point estimate | Evidence boundary | Repository evidence |
|---|---:|---|---|
| Selected participant-disjoint multimodal result | AUROC 0.897; AUPRC 0.863; n=314 | Cough+speech validation-fitted stack on Coswara; internal only | `compare_is10_final_validation_summary.csv` |
| Selected time-stratified result | AUROC 0.849; AUPRC 0.783; n=431 | Cough+breath+speech uniform mean; not paired with the row above | `compare_is10_final_validation_summary.csv` |
| Selected early-to-late result | AUROC 0.698; ECE 0.711; n=411 | Breath top-four validation ensemble; not the same modality/model as internal fusion | `compare_is10_final_validation_summary.csv` |
| Internal multi-seed stack | AUROC 0.895 +/- 0.003 | Four available seeds, cough+speech | `compare_is10_multiseed_stability_summary.csv` |
| Early-to-late multi-seed ensemble | AUROC 0.691 +/- 0.007 | Three available seeds, breath ensemble | `compare_is10_multiseed_stability_summary.csv` |
| Reverse temporal stress test | AUROC 0.920; AUPRC 0.029; F1 0.011; ECE 0.471 | Late-to-early exploratory row; prevalence and case mix differ | `compare_is10_reverse_temporal_summary.csv` |
| Matched conventional cough transfer | Internal AUROC 0.849--0.868; external 0.523--0.543; differences 0.310--0.345 with 95% CI lower bounds 0.247--0.291 | Cough-only; source predictions are averaged to 316 participants, and source participants and 8,331 target recordings are resampled independently | `final_validation_delta_bootstrap_ci.csv` |
| WavLM cough transfer | 0.812 internal to 0.484 external | Pooled Coswara cough submodalities; COUGHVID cough target | `reviewer_external_model_family_transfer_summary.csv` |
| CNN--BiGRU cough transfer | 0.737 internal to 0.548 external | Spectrogram cough model; COUGHVID cough target | `reviewer_external_model_family_transfer_summary.csv` |
| External prevalence | 285/8,331 = 3.42% | Processed known-label COUGHVID cohort | `reviewer_label_construction_audit.csv` |
| Full metadata association | AUROC 0.964 | Internal Coswara association; not a causal effect | `metadata_confounding_shuffle_retrain_sanity.csv` |
| Metadata shuffle retraining | mean AUROC 0.503 | 20 full retraining permutations | `metadata_confounding_shuffle_retrain_sanity.csv` |
| Incremental audio over full metadata | delta AUROC 0.012; 95% CI [-0.033, 0.053]; p=0.572; n=61 | Highest validation-ranked aligned cough+speech candidate; paired test | `reviewer_incremental_audio_metadata_metrics.csv` |
| Incremental audio over symptoms | delta AUROC 0.063; 95% CI [-0.005, 0.149]; p=0.104; n=61 | Same candidate; paired test | `reviewer_incremental_audio_metadata_metrics.csv` |
| Temporal feature stability | Jaccard 0.074; 110 shared of 1,490 union | Early vs late top-800 LightGBM ranking | `reviewer_feature_selection_stability.csv` |
| Source-target overlap diagnostic | domain AUROC 0.750; 25.2% outside source probability band | Diagnostic using 500 common features; not formal positivity proof | `reviewer_support_overlap_positivity.csv` |
| Audio label shuffle | AUROC 0.537--0.551 | Full retraining under shuffled labels; finite-sample sanity check | `compare_is10_shuffle_retrain_summary.csv` |

## Comparisons that must remain separate

- `0.897` multimodal internal performance versus `0.543` external cough
  performance is an orientation contrast, not a modality-matched causal delta.
- HST Cambridge/COUGHVID within-dataset 10-fold values are not external-transfer
  estimates and cannot be compared directly with Coswara-to-COUGHVID transfer.
- ESWA Coswara `0.92` and COUGHVID `0.93` are within-dataset estimates. Its
  explicit Coswara-to-COUGHVID cross-dataset result is approximately `0.53`.
- The reverse temporal AUROC must always be accompanied by AUPRC, F1, and ECE.
- Metadata prediction supports shortcut susceptibility; it does not prove that
  the audio branch encoded the same variables.
