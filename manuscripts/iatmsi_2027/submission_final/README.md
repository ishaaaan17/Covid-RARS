# Final IATMSI Conference Manuscript

The IATMSI-2027 submission limit is a strict six IEEE double-column pages,
including figures, tables, and references. The compiled manuscript uses all six
pages.

## Paper

**Title:** Participant-Disjoint Evaluation of Multisound Fusion for COVID-19
Audio Classification

The paper answers one bounded question: under fixed participant-disjoint
Coswara train, validation, and test roles, which audio-only sound combination
is selected from validation evidence, and what is its held-out participant-level
test performance?

The primary result is the validation-selected uniform cough--speech probability
mean: AUROC 0.895 (95% participant-bootstrap CI 0.852--0.933), AUPRC 0.862
(95% CI 0.802--0.909), and 314 complete-case test participants. On those same participants, fusion
improved AUROC over the selected cough branch by 0.033 (paired DeLong 95% CI
0.012--0.053; Holm-adjusted p=0.004). Its 0.007 AUROC difference from the
selected speech branch was inconclusive (95% CI -0.011--0.024; p=0.460), so the
paper does not claim superiority over speech.

Two supporting ablations strengthen the architectural claim. The uniform
score-level fusion exceeded validation-selected XGBoost feature concatenation
by 0.017 AUROC and 0.033 AUPRC on the held-out test cohort. Repeating model fitting
and validation-guided branch selection across seeds 42--46 gave AUROC
0.8945 +/- 0.0068 (range 0.8862--0.9049) on the same 314 participants; this measures
workflow variability, not independent population uncertainty.

## Rebuild

Run from the repository root:

```powershell
& '.venv\Scripts\python.exe' `
  'manuscripts\iatmsi_2027\submission_final\scripts\build_assets.py'

Push-Location 'manuscripts\iatmsi_2027\submission_final'
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
& '..\..\..\.venv\Scripts\python.exe' `
  'scripts\audit_submission.py'
Pop-Location
```

The asset builder reads the repository evidence tables, regenerates value
macros, tables, and vector figures, and fails if the selected rows are missing
or ambiguous. The audit checks citations, generated macros, prohibited
journal-only claims, page count, LaTeX errors, and expected evidence values in
the compiled PDF.

## Evidence Boundary

- The conference estimand is internal, participant-disjoint Coswara evaluation.
- Feature ranking uses training rows only; the retained budget is 800 of 10,140
  numeric candidates.
- Validation selects modality branches, the sound combination, and threshold.
- The test partition supplies final participant-level estimates, paired
  comparisons, and uncertainty intervals only after selection is frozen.
- The bootstrap interval quantifies participant sampling for a fixed system and
  cohort; it does not include retraining or selection variability.
- Temporal, metadata-confounding, calibration, external-transfer, deep-transfer,
  IPW, subgroup, and incremental-value analyses belong to the separate journal
  study described in `SCOPE_AND_OVERLAP.md`.
- Incomplete HST experiments are not reported.

## Submission Checks

Before upload, replace or confirm author metadata, add the conference copyright
notice required by the final author kit, rerun the build and audit commands, and
inspect the generated PDF after IEEE PDF eXpress processing.
