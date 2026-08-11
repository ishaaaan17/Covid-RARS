# Ground-Up IEEE JBHI Manuscript Design

## Objective

Build a new IEEE Journal of Biomedical and Health Informatics manuscript from completed COVID-RARS experiments. The article is a retrospective biomedical model-reliability and validation study. It does not propose a new classifier, claim state-of-the-art performance, or treat source-domain AUROC as evidence of clinical readiness.

## Source Boundary

Permitted evidence sources are:

- experiment implementations under `src/covid_rars/` and `scripts/`;
- experiment manifests and audit files under `reports/`;
- metric and prediction outputs under `data/outputs/metrics/`;
- dataset documentation and original dataset publications;
- primary methodological and clinical literature.

Excluded sources are all previously written COVID-RARS manuscripts, their prose, their figures, and their table layouts. They must not be consulted or paraphrased while drafting the new article.

No new feature extraction, model training, tuning, resampling experiment, or validation run will be introduced. Existing analyses may be aggregated and plotted without changing their results.

## Scientific Question

The main question is whether favorable source-domain performance from COVID-19 respiratory-audio models remains credible when the evaluation target changes across participant separation, calendar structure, independent dataset transfer, metadata controls, calibration, and clinically relevant operating points.

The article will distinguish four questions that cannot be merged into one leaderboard:

1. How well can multimodal respiratory audio discriminate labels within Coswara under participant-disjoint source-domain evaluation?
2. Does cough-only discrimination transport from Coswara to an independently collected COUGHVID cohort when modality and model family are held fixed?
3. Are source-domain conclusions stable under retrospective chronological stress and repeated random seeds?
4. How much of the observed discrimination is also available from symptoms, demographics, acquisition context, and audio-quality metadata, and does audio add measurable discrimination on aligned participants?

## Evidence Hierarchy

### Primary confirmatory evidence

- Model- and modality-matched Coswara cough to COUGHVID transfer for LightGBM, SVC, CatBoost, and XGBoost using the frozen combined engineered feature representation.
- Independent two-sample participant/recording-level bootstrap intervals for internal-to-external AUROC differences, with the resampling unit stated explicitly.
- COUGHVID fixed-sensitivity operating points and prevalence-relative precision.

These analyses provide the cleanest controlled result because modality and model family remain fixed while the dataset changes.

### Primary supporting evidence

- Source-domain multimodal fusion under participant-disjoint evaluation, reported as source-domain discrimination only.
- WavLM and CNN-BiGRU cough transfer as representation-family stress tests.
- External recalibration analysis separating calibration improvement from unchanged discrimination.

### Secondary robustness evidence

- Time-stratified participant evaluation.
- Retrospective early-to-late temporal stress test, explicitly not described as leakage-free prospective validation because the frozen top-800 representation was selected before the chronological split analysis.
- Multi-seed stability.
- Early-versus-late feature-ranking overlap.
- Source/target support-overlap diagnostic.

### Exploratory evidence

- Metadata-only prediction and retrain-with-shuffled-label controls.
- Participant-aligned symptoms-only, audio-only, and symptoms-plus-audio incremental-value comparisons.
- Full metadata-plus-audio comparisons.
- Inverse-propensity weighting sensitivity analysis.
- Subgroup, equity, duration, context-control, decision-curve, and specification-curve analyses.

Exploratory results will be labelled as such. Isolated p-values will not be promoted to confirmatory evidence.

## Claim Rules

- Do not connect heterogeneous multimodal, unimodal, temporal, and external endpoints with a line or imply that their differences identify one causal degradation effect.
- Do not describe COUGHVID as multimodal external validation.
- Do not claim that respiratory audio contains no portable signal. State that the evaluated source-trained models showed near-chance discrimination on the evaluated COUGHVID endpoint.
- Do not claim that an audio model learned a specific metadata shortcut without direct model-level evidence. Use "consistent with shortcut risk" or "collection-context dependence."
- Do not call the temporal stress test prospective, strict, or leakage-free.
- Do not compare AUPRC values without reporting endpoint prevalence.
- Do not describe the combined engineered representation as only ComParE+IS10. It also contains the strong feature bank, including timing and duration descriptors.
- Do not call timing, quality, sample-rate, or duration fields demographic metadata.
- Do not claim clinical benefit from AUROC. Clinical interpretation must use sensitivity, specificity, precision, calibration, prevalence, and decision-curve evidence.
- Distinguish same-participant paired comparisons from independent source/target comparisons.

## Manuscript Structure

1. **Introduction**: clinical screening context, validation failure modes, precise evidence gap, and four contributions.
2. **Related Work**: respiratory-audio COVID-19 modeling; realistic evaluation and confounding; temporal and cross-dataset validation; calibration and clinical utility.
3. **Materials and Cohorts**: dataset provenance, label construction, modalities, inclusion/exclusion, participant and recording counts, dates, prevalence, quality control, and evaluation units.
4. **Methods**: preprocessing; engineered and learned representations; candidate models; fusion; split definitions; source-only external transfer; metadata controls; uncertainty, calibration, incremental-value, and utility analyses.
5. **Results**: cohort flow; source-domain multimodal discrimination; controlled cough-only external transfer; representation-family stress tests; chronological and feature stability; metadata and incremental value; calibration and operating points.
6. **Discussion**: interpretation by validation target, comparison with prior studies, implications for benchmark design, and clinical meaning.
7. **Limitations**: retrospective crowdsourced labels, unmatched external modalities, temporal feature-selection boundary, low external prevalence, small complete-case incremental-value cohort, measured confounding, and absence of prospective clinical validation.
8. **Conclusion**: one bounded conclusion about evidence requirements for respiratory-audio screening.

## Main Figures

1. **Study design and information flow**: separate source development, source robustness protocols, and independent external transfer. Target observations must never appear upstream of the frozen source workflow.
2. **Controlled external transport**: source-to-target estimates for the four matched engineered-feature model families plus WavLM and CNN--BiGRU, matched-model AUROC-decline intervals, and external AUPRC relative to target prevalence.
3. **External endpoint reliability**: zero-based fixed-sensitivity specificity and precision against COUGHVID prevalence, followed by held-out recalibration metrics on a common linear scale.
4. **Context and stability diagnostics**: metadata shuffle controls, grouped permutation importance, early-versus-late feature-set overlap, and temporal-cohort prevalence.

Every figure must be generated directly from CSV artifacts, use a color-blind-safe palette, remain legible in grayscale, and survive single-column or double-column IEEE rendering. No text may overlap, clip, or fall below practical print size.

## Main Tables

1. Cohort definitions and dataset roles, including evaluation unit and label source.
2. Source-domain multimodal and time-aware results with participant counts, positive/negative counts, prevalence, AUROC, AUPRC, calibration, and uncertainty where available.
3. Controlled cough-only internal-to-external transfer across model and representation families.
4. Metadata controls and aligned incremental-value comparisons, including sample size, effect size, interval, and paired DeLong result.

Detailed model banks, fold-level rows, subgroup tables, calibration bins, IPW balance diagnostics, reverse-temporal results, and specification curves belong in supplementary material.

## Writing Standard

- Use direct technical prose and define each term before using it.
- State the evaluation unit and denominator for every reported result.
- Present methods before results and avoid naming implementation files in manuscript prose.
- Report effect sizes and intervals before p-values.
- Separate observation from interpretation and interpretation from causal speculation.
- Use literature comparisons only when dataset, modality, label and validation protocol are explicitly aligned.
- Use approximately 40-60 necessary references, emphasizing primary studies, original methods, reporting guidelines, and directly relevant JBHI work.
- Avoid slogans, rhetorical questions, exaggerated adjectives, fabricated clinical implications, and formulaic contribution language.

## Deliverables

- `main.tex`: complete new JBHI manuscript.
- `references.bib`: primary, verified bibliography.
- `figures/build_figures.m`: deterministic MATLAB artifact-to-figure generator with fixed panel geometry.
- `figures/*.pdf` and `figures/*.png`: publication and inspection versions.
- `supplement.tex`: detailed analyses excluded from the main article.
- `evidence_map.csv`: every manuscript number mapped to its source artifact and row selector.
- `README.md`: build and verification commands.

## Acceptance Gate

The draft is not ready for submission until:

- every numerical statement maps to an artifact row;
- the temporal analysis is labelled according to its actual feature-selection boundary;
- heterogeneous endpoints are not represented as a controlled progression;
- all citations resolve to the claimed source;
- LaTeX compiles without unresolved references or overfull boxes;
- every PDF page and figure is visually inspected at print scale;
- ethics, data-use, funding, conflicts, author contributions, and data/code availability statements are supplied by the authors.
