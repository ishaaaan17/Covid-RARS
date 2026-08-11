# IATMSI 2027 Evidence-Audited Submission

This directory contains the focused conference manuscript. It is deliberately
narrower than the planned journal article: it evaluates validation-guided,
participant-disjoint cough--speech probability fusion within Coswara. It does
not report unfinished HST experiments or reuse the journal paper's temporal,
confounding, calibration, or external-transfer contribution.

## Submission constraints

- Venue: IEEE IATMSI 2027.
- Format: `IEEEtran` conference, two columns.
- Limit: six pages including figures, tables, and references.
- Official instructions: <https://iatmsi.iiitm.ac.in/paper-submission/>.
- IEEE templates: <https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/>.

## Evidence build

Run from the repository root:

```powershell
.venv\Scripts\python.exe `
  manuscripts\iatmsi_2027\submission_evidence_audited\scripts\build_assets.py
```

The script reads the archived result tables, enforces the fixed protocol and
validation-selection keys, and regenerates:

- all numerical LaTeX macros used for primary claims;
- separate vector participant-flow and validation/results figures;
- cohort, branch-selection, fusion-sensitivity, and bootstrap evidence records;
- a claim-to-source ledger with explicit inferential boundaries.

## Compile and audit

```powershell
Set-Location manuscripts\iatmsi_2027\submission_evidence_audited
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Set-Location ..\..\..
.venv\Scripts\python.exe `
  manuscripts\iatmsi_2027\submission_evidence_audited\scripts\audit_submission.py
```

The audit rejects missing or uncited references, unsupported superiority/SOTA
language, leakage of journal-only results, unresolved LaTeX references,
overfull boxes, absent evidence values, and a PDF longer than six pages.

## Statistical boundary

The reported 95% interval is a participant bootstrap for the fixed primary
fusion on the held-out cohort. It does not include model-refit variability.
The fusion-minus-speech difference is descriptive because the archived local
evidence does not contain aligned participant predictions for a paired delta
interval. The manuscript must not claim statistical superiority.
