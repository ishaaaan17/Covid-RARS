# IATMSI 2027 Current-Evidence Submission

This folder is the submission-format conference manuscript. It is separate
from `../master`, which is an evidence-rich working draft and is not suitable
for submission.

## Scope

The paper answers one source-development question:

> Under a participant-disjoint Coswara split, which modality combination is
> selected for a fixed uniform-mean fusion rule, and how does its held-out
> estimate compare descriptively with the selected single modalities?

The current paper uses completed Coswara evidence only. It deliberately omits
temporal, metadata, calibration, COUGHVID, and incremental-value analyses,
which belong to the full reliability journal paper. HST is also omitted until
its run has a complete checkpoint, participant-level predictions, selection
record, and audited metrics.

## Official Format

- Venue: IEEE IATMSI 2027
- Class: `IEEEtran` with the `conference` option
- Limit: six double-column pages including figures, tables, and references
- Current compiled length: four pages; verify again with `pdfinfo main.pdf`
- Current submission deadline: 20 December 2026 (recheck the venue website
  before submission)
- Official instructions: <https://iatmsi.iiitm.ac.in/paper-submission/>

## Rebuild

From the repository root:

```powershell
.venv\Scripts\python.exe `
  manuscripts\iatmsi_2027\submission_current\scripts\build_conference_assets.py
```

Then from this folder:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The asset builder fails if the evidence table no longer contains exactly one
audited row for each stated result. It also selects the fusion configuration
on validation AUROC/AUPRC and only then retrieves the matching test row.

## Evidence Inputs

- `reports/tables/paper_metric_table_raw.csv`
- `reports/tables/final_validation_bootstrap_ci.csv`
- `reports/tables/strong_baseline_participant_audit.csv`
- `reports/tables/strong_baseline_protocol_audit.csv`

Generated manuscript-specific tables are written under `tables/`. The vector
figure is written under `figures/` as PDF and SVG. The builder also checks the
fixed partition counts and the 1,000-resample AUROC interval before writing
`cohort_partition_record.csv` and `selected_auroc_bootstrap_record.csv`.

## Before Submission

1. Confirm author order, affiliation spelling, email addresses, and ORCID IDs.
2. Confirm the original Coswara ethics/consent wording against the source paper.
3. Run IEEE LaTeX validation and PDF eXpress when the conference supplies its
   event identifier.
4. Re-run the asset builder after any result-table change.
5. Do not cite the existing multi-seed summary until the raw runs have been
   regenerated with validation-first selection.
6. Complete the HST gate before adding any HST statement or number.
7. Compare the conference and journal manuscripts using the overlap audit.
