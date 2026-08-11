# COVID-RARS Publication Split and Originality Plan

**Status:** Binding drafting plan until the authors approve a different venue
strategy.  
**Prepared:** 4 August 2026

## 1. Why two papers are defensible

IEEE recognizes an evolutionary publication path in which early ideas appear
in a workshop, a developed method appears in a conference, and a substantially
expanded contribution appears in a journal. This is acceptable only when the
earlier work is disclosed and cited and the later paper clearly states what is
new. It is not permission to divide one result into minimally different papers
or to submit materially overlapping manuscripts concurrently.

The COVID-RARS project contains two separable research questions:

1. **Model-development question:** Can cough, breathing, and speech be combined
   under participant-disjoint source evaluation using validation-selected
   representations and fusion?
2. **Reliability question:** Does apparent source performance survive temporal,
   metadata, calibration, and independent cross-dataset stress tests?

These questions require different primary comparisons, figures, tables, and
conclusions. Keeping them separate is therefore defensible. Publishing the
same cross-dataset result as the headline of both papers would not be.

## 2. IATMSI-2027 conference paper

### Primary question

Which fixed probability-fusion configuration is selected using validation data
under participant-disjoint Coswara evaluation, and how does its held-out
performance compare descriptively with its constituent modality models?

### Included evidence

- Coswara only.
- Cough, breathing, and speech source modalities.
- Audio quality control and participant-level partitioning.
- The 10,140-candidate ComParE+IS10+project acoustic representation and
  training-only reduction to the top 800 candidates.
- Four selected conventional classifier families and their imbalance handling.
- Uniform, validation-weighted, and validation-fitted logistic fusion.
- Participant-disjoint validation and test performance.
- A compact time-stratified source sensitivity only if the same model,
  modalities, and selection rule can be compared without changing the
  estimand.
- HST only if the completed run has an auditable checkpoint, participant-level
  predictions, a source-validation selection record, and final metrics. No
  placeholder or partial result may enter the paper.

### Excluded evidence

- COUGHVID external transfer.
- The temporal early-to-late and reverse-temporal audits.
- Metadata-only prediction, permutation importance, matching, and IPW.
- Calibration, recalibration, decision curves, and fixed-sensitivity target
  analyses.
- Incremental metadata-plus-audio value.
- Subgroup/equity, context-control, duration, and support-overlap analyses.
- The complete literature audit and the full model-search history.

### Conference contribution

The conference paper is a focused empirical fusion-selection study. It must not claim
external validity, clinical deployment readiness, or comprehensive shortcut
analysis. Its figures will show the multimodal architecture and the controlled
internal comparison that supports the fusion claim.

## 3. Full journal paper

### Primary question

How reliable and transportable are COVID-19 respiratory-audio models when
evaluation explicitly stresses calendar time, collection context, calibration,
and independent dataset origin?

### Included evidence

- The conference model as a cited source-development baseline, not as a new
  journal contribution.
- Participant-disjoint, calendar-aware, and carefully bounded chronological
  analyses.
- Multi-seed stability and cluster-aware uncertainty.
- Matched cough-only Coswara-to-COUGHVID transfer.
- Conventional, WavLM, CNN-BiGRU, and completed HST reliability evidence where
  protocols are legitimate.
- Metadata/confounding, shuffle-label, feature-stability, matching/IPW, and
  subgroup analyses.
- Calibration, recalibration sensitivity, fixed-sensitivity operating points,
  and decision analysis.
- Metadata-plus-audio incremental-value analysis with its sample-size limits.
- Label-construction, analysis-unit, and modality-availability limitations.

### Journal contribution

The journal paper is a reliability and transportability study. It must cite the
conference paper, identify the source model as prior work, and state in the
cover letter and manuscript exactly which analyses, data uses, conclusions,
and figures are new.

## 4. Non-overlap rules

1. Do not submit materially overlapping conference and journal manuscripts
   concurrently without explicit disclosure to both venues.
2. Cite the conference paper in the journal paper once it is citable; include a
   copy and a difference statement with the journal submission.
3. Do not copy conference paragraphs into the journal manuscript. Describe
   necessary common methods concisely and cite the prior paper.
4. Do not reuse a conference figure or table in the journal manuscript. Build
   journal visuals around the new reliability analyses. If exact reuse ever
   becomes necessary, obtain permission and identify the source in the caption.
5. Shared dataset names, standard model names, equations, and unavoidable
   technical terminology are not the contribution. The hypotheses, primary
   endpoints, result tables, figures, and conclusions must be distinct.
6. Do not advertise the same `0.897` internal result as a novel finding twice.
   In the journal it is background for the validation ladder.
7. Do not publish the matched COUGHVID transfer as the conference headline if
   it is intended to remain the journal's primary transportability result.
8. Keep an overlap table during drafting that maps every conference figure,
   table, and claim to either `conference only`, `cited baseline in journal`, or
   `journal only`.

## 5. HST decision gate

| Completed HST evidence | Conference action | Journal action |
|---|---|---|
| Better internal result with complete participant-level audit | Use HST as the main internal branch and compare with the established fusion baseline | Cite the conference method; evaluate reliability only where the same HST artifact is legitimately available |
| Comparable or weaker internal result but stable and auditable | Report as a modern architecture comparator, not as a superiority claim | Use as evidence that reliability findings are not restricted to conventional features |
| Incomplete, failed, or protocol-incompatible result | Exclude from the conference paper | Retain only in internal records unless a complete interpretable result is later produced |

## 6. Drafting order

1. Freeze the current conventional-system evidence. Add HST only after its run
   passes the separate integration gate and improves the conference question.
2. Write the six-page IATMSI manuscript from a clean IEEE conference template.
3. Use no text, diagrams, or tables from the current 25-page working draft.
4. Rebuild the full journal manuscript separately for the selected journal's
   article structure and reporting requirements.
5. Before either submission, perform a claim audit, citation audit, visual
   inspection at final size, and an explicit cross-paper overlap review.

## 7. Policy sources

- IEEE conference submission policy: https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/
- IATMSI-2027 paper submission requirements: https://iatmsi.iiitm.ac.in/paper-submission/
- Nature Portfolio conference-proceedings policy: https://www.nature.com/npjdigisurg/for-authors-and-referees/about/editorial-policies/preprints-conference-proceedings
- npj Digital Medicine editorial policies: https://www.nature.com/npjdigitalmed/editorial-policies
- COPE guidance on duplicate and concurrent submissions: https://doi.org/10.24318/y9lyqPiR

The npj Digital Medicine policy expressly identifies conference abstracts as
the overlap exception; it does not guarantee consideration of a substantially
extended full proceedings paper. A complete IATMSI paper therefore creates a
material overlap risk for a later npj submission even with disclosure. If npj
remains the journal target, obtain an editorial ruling before conference
publication or keep the two papers scientifically distinct enough that the
journal does not depend on republication of the conference method and result.
