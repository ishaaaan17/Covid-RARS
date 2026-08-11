# Unrestricted Master Manuscript

This directory is the complete scientific manuscript from which a focused
IATMSI-2027 conference paper may later be derived. The master is intentionally
not constrained to six pages. It preserves the study design, model families,
validation protocols, uncertainty analyses, and claim boundaries needed for a
journal-scale account.

The IATMSI submission must not be produced by shrinking this document until it
fits. It will be a separate six-page extraction with one research question and
one coherent set of comparisons. The official venue audit is maintained in
`../IATMSI_2027_REQUIREMENTS_AND_SCOPE_AUDIT.md`.

## Source rules

- Numerical statements must trace to repository artifacts listed in
  `CLAIM_EVIDENCE_LEDGER.md`.
- Literature statements must trace to a primary paper in `references.bib`.
- Protocol-sensitive comparisons must also match the full-text and page-image
  checks in `FULL_TEXT_LITERATURE_AUDIT.md`; abstract-only records cannot supply
  detailed split, leakage, or limitation claims.
- Results from the running HST experiment remain excluded until its evidence
  pack is complete and passes the checks in `PENDING_HST_INTEGRATION.md`.
- COUGHVID is a cough-only external target. It does not test transfer of the
  breath, speech, or multimodal fusion branches.
- The manuscript does not claim clinical diagnosis, absence of an acoustic
  COVID-19 signal, or state-of-the-art performance.

## Build

From this directory:

```powershell
python scripts/build_publication_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The figure command rebuilds the manuscript graphics from the tracked result
tables; no numerical values are entered manually in the artwork.

The working draft uses one-column IEEE draft mode for review. The eventual
conference extraction will use the official `IEEEtran` conference mode and the
strict IATMSI page limit.
