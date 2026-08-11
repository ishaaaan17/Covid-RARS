# Claim, Evidence, and Overlap Audit

## Conference Claims

| Claim | Evidence | Boundary |
|---|---|---|
| The data partitions are participant-disjoint | Quality-passing participant and protocol audits | Applies to Coswara only |
| Training-only ranking reduced 10,140 numeric candidates to a fixed 800-feature configuration | Feature-selection code and summary | 500 and 1,200 were also explored; no globally preregistered rule selected 800 |
| Cough+speech was selected among uniform-mean modality combinations | Validation AUROC 0.842, AUPRC 0.800 | Fitted stacking and heuristic weighting are exploratory |
| Primary test AUROC is 0.895 | Matching frozen uniform-mean test row, n=314 | Internal Coswara estimate only |
| AUROC 95% CI is 0.852-0.933 | 1,000-resample participant bootstrap | Sampling uncertainty for the fixed test cohort; not a CI for the fusion-speech difference |
| Fusion is numerically above speech in AUROC and AUPRC | 0.895 versus 0.888 AUROC; 0.862 versus 0.833 AUPRC | Descriptive only; balanced accuracy is unchanged and F1 is lower |

The existing multi-seed summary is excluded because its generator selected
the best test row within each seed. It may be reinstated only after rerunning
the raw per-seed experiment with validation-first candidate selection.

The primary uniform-mean ranking is a retrospective validation-first
reconstruction. Test rows already existed from earlier development, but the
audited selection code does not read their metrics when choosing the candidate.
The manuscript must not describe this analysis as preregistered or historically
blinded to test results.

The 0.897 logistic-stack row is exploratory because its coefficients and
validation score use the same validation participants. It must not replace the
uniform-mean primary endpoint unless cross-fitted meta-model selection is run.

## Journal-Only Evidence

The following evidence must not be presented as a new conference contribution:

- chronological and reverse-temporal analyses;
- COUGHVID transfer;
- metadata-only, shuffle-label, permutation, matching, and IPW analyses;
- calibration, target recalibration, fixed-sensitivity, and decision curves;
- incremental metadata-plus-audio value;
- subgroup, equity, context-control, duration, and support-overlap analyses;
- the full literature limitations matrix.

The journal paper may cite the conference system as a source-development
baseline. It must not reuse the conference text, figure, or tables. Its primary
hypothesis, endpoints, visuals, and conclusions must concern reliability and
transportability rather than internal fusion.

## Comparability Rules

- The internal literature table is contextual. Cohort snapshots, inputs, and
  partitions differ, so the table is not a statistical ranking.
- Audio-plus-symptom values are labeled separately from audio-only values.
- Participant-, recording-, and segment-level estimates are never merged.
- Internal target-trained results are not compared with frozen transfer as if
  they estimate the same quantity.
- No architecture result enters the paper before its final selected checkpoint
  and complete participant-level predictions exist.

## Originality Rules

- Write conference and journal prose independently.
- Do not reproduce a source paper's figure geometry, caption, or paragraph.
- Cite ideas, datasets, descriptor sets, and algorithms at first use.
- Disclose the conference paper to a later journal editor and state the exact
  new analyses in the cover letter and manuscript.
- Do not submit materially overlapping versions concurrently.
