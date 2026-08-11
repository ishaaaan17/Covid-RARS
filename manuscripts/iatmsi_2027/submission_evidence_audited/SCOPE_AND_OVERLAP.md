# Publication Scope and Overlap Control

## IATMSI conference paper

**Question:** Under a fixed participant-disjoint Coswara protocol, which
audio-only probability-fusion configuration is selected using validation
evidence, and what is its held-out performance relative to the selected
single-sound branches?

**Included evidence:**

- Coswara only;
- cough, breathing, and speech audio only;
- quality screening and fixed participant-level train/validation/test roles;
- training-only acoustic feature ranking;
- validation selection of modality branches, modality combination, and threshold;
- uniform cough--speech fusion as the primary rule;
- weighted mean and logistic stacking only as sensitivity analyses;
- fixed-test-cohort participant-bootstrap uncertainty.

## Journal paper

The journal submission is a separate reliability study. Its central estimands
and evidence are not claims of the conference paper:

- chronological and reverse-temporal validation;
- multi-seed stability and feature non-stationarity;
- metadata confounding, permutation, and shuffle-label analyses;
- calibration, fixed-sensitivity operating points, and decision curves;
- COUGHVID transfer and source-support diagnostics;
- WavLM and CNN-BiGRU external transfer;
- IPW, matching, subgroup, and equity analyses;
- incremental value of audio beyond metadata.

## HST boundary

The Hierarchical Spectrogram Transformer may be cited as related work. No HST
performance is reported because the full protocol run was not completed
reliably. It must not be represented as implemented evidence in either abstract,
results table, figure, or conclusion.

## Reuse rule

The conference and journal papers may share dataset background and standard
method definitions where necessary, but not the same primary question,
headline result set, principal figure, or principal table. Any later journal
submission must cite the conference paper if it has been accepted or published
and must disclose the relationship in its cover letter.
