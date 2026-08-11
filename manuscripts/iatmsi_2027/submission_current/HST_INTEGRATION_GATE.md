# HST Integration Gate

HST is not part of the current manuscript. It may be added only after every
item below is satisfied.

## Required Evidence

- A completed, immutable training receipt and checkpoint hash.
- Participant-disjoint train, validation, and test manifests.
- Participant-level probabilities for every reported split.
- A validation-only checkpoint and model-selection record.
- The same label protocol and analysis unit used by the conference baseline.
- A successful leakage audit showing no participant or augmentation-parent
  overlap.
- AUROC, AUPRC, balanced accuracy, F1, calibration, and uncertainty calculated
  from the frozen selected checkpoint.
- A same-participant comparison against the aligned ComParE+IS10 comparator.

## Decision Rules

1. **HST is better on the aligned internal endpoint.** Make HST the primary
   architecture, retain the current fusion system as the controlled baseline,
   and rewrite the title, abstract, contribution statements, figure, and result
   table. Do not simply append one row.
2. **HST is comparable or weaker.** Report it as a modern architecture
   comparator if space permits. The conference claim remains about
   validation-guided multimodal development, not HST superiority.
3. **HST is incomplete or protocol-incompatible.** Exclude it. A partial run,
   pilot metric, or unmatched split cannot appear in the manuscript.

## Required Re-Audits

- Validation-only selection check.
- Exact cohort and modality alignment check.
- Bootstrap unit check.
- Five-seed or explicitly bounded single-seed wording.
- Six-page layout and final-size visual inspection.
- Conference/journal overlap review.

