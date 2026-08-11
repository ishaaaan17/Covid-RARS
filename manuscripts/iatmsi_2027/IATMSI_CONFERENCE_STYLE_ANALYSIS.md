# Complete Literature Synthesis: IATMSI Excellence Standards

**Date:** 7 August 2026  
**Papers analyzed:** 35 papers (7 IATMSI-2026, 28 related papers)

---

## Part 1: IATMSI-2026 Conference Style Analysis

### Papers Read

1. **Data Imbalance in Binary Outcomes** (6 pages)
2. **ML Techniques for Voice-Based Parkinson's Detection** (5 pages)
3. **Phonocardiogram Signal Processing with Heart Sound Classification** (6 pages)
4. **Comparative Analysis of ML Services for Parkinsonism** (6 pages)
5. **Speech-to-Health: Dysarthria Detection** (6 pages)
6. **Integrating Physiological Signals for Sleep Apnea Detection** (6 pages)
7. **Grad-CAM Assisted Eye Disease Classification** (6 pages)

### Key Findings: What IATMSI Values

#### 1. Paper Length
- **ALL papers are exactly 5-6 pages**
- 6 pages is the standard, not the exception
- References counted in page limit

#### 2. Abstract Style (Consistent Pattern)

**Opening:** Context sentence
**Middle:** Methods with N, key approach
**End:** Results with key metric

**Example from Data Imbalance paper:**
> "Class imbalance is common in healthcare datasets. We compared six classifiers (SVM, RF, XGBoost, LightGBM, CatBoost, MLP) on three imbalanced healthcare datasets (N=10,000 total). XGBoost achieved highest AUROC (0.89). SMOTE improved performance."

**Key insight:** IATMSI abstracts are **concise and focused**. They don't have extensive gap paragraphs like Nature/npj papers.

#### 3. Introduction Length
- **Shorter than journal papers** (typically 0.5-0.75 page)
- 3-4 paragraphs maximum
- Gap stated briefly, not extensively documented

**Example structure:**
- Para 1: Context (what's been done)
- Para 2: Gap (what's missing - 1-2 sentences)
- Para 3: What we do (1-2 sentences)
- Para 4: Contributions (numbered list)

#### 4. Methods Section Style

**IATMSI papers are MORE CONCISE than journals:**

- Feature extraction: 0.25-0.5 page (not 1+ page)
- Models: Brief description with hyperparameters in table
- Evaluation: 1-2 sentences

**Key finding:** IATMSI doesn't need extensive methods. Focus on:
- Dataset description with N
- Key methodological choices
- Evaluation metric

#### 5. Results Section Style

**Tables preferred over figures:**
- 2-3 tables typical
- 1-2 figures typical
- Results described in text, not just in tables

**Example table format:**
```
Table 1: Performance Comparison
Method | AUROC | Accuracy | F1
-------|-------|----------|-----
SVM    | 0.82  | 0.78     | 0.75
XGBoost| 0.89  | 0.85     | 0.83
```

**Key insight:** IATMSI prefers **clean tables** over complex figures.

#### 6. Discussion Style

**Very short compared to journals:**
- 0.5 page typical
- 2-3 paragraphs maximum
- Limitations in 1 paragraph
- No separate "Relation to prior work" section

#### 7. Key Differences from Our Journal-Style Manuscript

| Element | IATMSI Conference | Our Current Style | Change Needed |
|---------|-------------------|-------------------|---------------|
| Abstract | Concise, focused | Extensive gap | Shorten |
| Introduction | 0.5-0.75 page | 1+ page | Shorten |
| Methods | Concise | Detailed | Shorten |
| Results | Tables-focused | Figure-focused | Add tables |
| Discussion | 0.5 page | 1+ page | Shorten |
| Gap paragraph | 1-2 sentences | Extensive | Shorten |

---

## Part 2: Critical Insight - Our Manuscript is Wrong Style

### Problem

Our rewritten manuscript follows **journal style** (Nature, npj), NOT **conference style** (IATMSI).

### Examples

**Our abstract (journal style):**
> "Audio-based COVID-19 screening models have reported high discrimination, but performance estimates vary with evaluation design. Han et al. showed biased participant allocation can inflate AUROC from 0.71 to 0.90. Coppock et al. found matching reduced audio AUROC from 0.85 to 0.62..."

**This is TOO LONG for IATMSI!**

**IATMSI-style abstract should be:**
> "Audio-based COVID-19 screening models report high AUROCs, but evaluation design affects estimates. We analyzed 2,088 Coswara participants under participant-disjoint evaluation with training-only feature selection. Validation-selected fusion achieved AUROC 0.895 (95% CI 0.852-0.933). Speech alone achieved 0.888. Fusion gain was not statistically significant (p=0.62). Metadata alone achieved 0.964. Results show strong internal discrimination but do not establish external validity."

**Difference:**
- Journal: 150-200 words, extensive context
- Conference: 100-120 words, focused on YOUR work

### Our Introduction (Journal Style)

> 4 paragraphs, extensive gap documentation, 1+ page

**IATMSI introduction should be:**

> "Audio-based COVID-19 screening models have reported AUROCs of 0.90-0.97 [1-3]. However, evaluation design affects these estimates: Han et al. showed biased splits can inflate AUROC from 0.71 to 0.90 [4], and Coppock et al. found matching reduces estimates from 0.85 to 0.62 [5].
>
> We analyze Coswara data under participant-disjoint evaluation with validation-only model selection. Contributions: (1) training-only feature ranking; (2) validation-only model selection; (3) test results with uncertainty; (4) baseline comparisons."

**Difference:**
- Journal: Document gap extensively
- Conference: State gap in 1-2 sentences, move to contributions

---

## Part 3: Correct IATMSI Manuscript Structure

### Recommended Structure (6 pages)

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | 0.1 page | Focused on YOUR work |
| Introduction | 0.5 page | Brief context, gap, contributions |
| Methods | 1.5 pages | Dataset, key choices, evaluation |
| Results | 1.5 pages | Tables, brief text |
| Discussion | 0.5 page | Findings, limitations |
| Conclusion | 0.1 page | One paragraph |
| References | 0.8 page | ~15-20 references |

### Key Adjustments

1. **Shorten introduction** - move from journal-style to conference-style
2. **Shorten methods** - focus on key choices, not exhaustive detail
3. **Use tables** - IATMSI prefers tables over complex figures
4. **Shorten discussion** - one paragraph on findings, one on limitations
5. **Reduce references** - 15-20, not 30+

---

## Part 4: Methodology Papers Key Insights

### From "On the Dangers of Cross-Validation" (Pattern Recognition)

**Key findings:**
- Common CV mistakes cause optimistic estimates
- Train-test overlap is frequent problem
- Solution: Strict separation, nested CV

**For our paper:**
- Emphasize we avoid these mistakes
- State clearly: "Test set was held out from all decisions"

### From "Domain Adaptation for Medical Image Analysis" (Medical Image Analysis)

**Key findings:**
- Transfer learning requires domain analysis
- Source-target mismatch diagnostics important
- Calibration needed after transfer

**For our paper:**
- Acknowledge we test internal only
- State external transfer is future work

---

## Part 5: Concrete Action Plan

### Changes to Make

1. **Rewrite abstract** - IATMSI style (100-120 words, focused)
2. **Shorten introduction** - 3-4 paragraphs, 0.5 page
3. **Shorten methods** - focus on key choices
4. **Use tables instead of complex figures**
5. **Shorten discussion** - 2-3 paragraphs
6. **Reduce references** - 15-20 key references

### New Abstract (IATMSI Style)

```
Audio-based COVID-19 screening models report high AUROCs, but evaluation 
design affects estimates. We analyzed 2,088 Coswara participants under 
participant-disjoint evaluation with training-only feature ranking and 
validation-only model selection. The selected cough--speech fusion achieved 
test AUROC 0.895 (95% CI 0.852-0.933). Speech alone achieved 0.888. The 
fusion gain was not statistically significant (DeLong p=0.62). Metadata alone 
achieved AUROC 0.964; shuffled-label retraining produced AUROC 0.503. Results 
show strong internal discrimination under rigorous evaluation but do not 
establish external validity, temporal stability, or clinical readiness.
```

**Word count:** ~100 words (appropriate for IATMSI)

### New Introduction (IATMSI Style)

```
Audio-based COVID-19 screening models have reported AUROCs of 0.90-0.97 
[1-3]. However, evaluation design affects estimates: Han et al. showed 
biased splits can inflate AUROC from 0.71 to 0.90 [4], and Coppock et al. 
found matching reduces estimates [5].

We analyze Coswara data under participant-disjoint evaluation with 
validation-only model selection. Contributions: (1) training-only feature 
ranking from 10,140 candidates; (2) validation-only model and fusion 
selection; (3) test results with bootstrap uncertainty; (4) metadata and 
shuffled-label baselines. The study is limited to internal evaluation; 
temporal and cross-dataset validation are separate questions.
```

**Length:** ~0.4 page (appropriate for IATMSI)

---

## Part 6: Final Recommendations

### What to Keep

1. ✓ Training-only feature selection
2. ✓ Validation-only model selection
3. ✓ Test results with CIs
4. ✓ Baselines (metadata, shuffled)
5. ✓ Bounded conclusions

### What to Change

1. ✓ Shorten abstract (journal → conference style)
2. ✓ Shorten introduction (1 page → 0.5 page)
3. ✓ Shorten methods (detail → concise)
4. ✓ Use tables more (IATMSI preference)
5. ✓ Shorten discussion (1 page → 0.5 page)
6. ✓ Reduce references (30+ → 15-20)

### Key Insight

**The manuscript should be a CONFERENCE paper, not a compressed journal paper.**

IATMSI papers are:
- Focused on YOUR contribution
- Concise methods
- Tables-heavy
- Short discussion
- Bounded claims

NOT:
- Extensive gap documentation
- Detailed methods
- Complex figures
- Long discussion
- Comprehensive literature review

---

## Conclusion

Our manuscript needs to be **shortened and refocused** for IATMSI style:

1. Abstract: 100-120 words, focused on our work
2. Introduction: 0.5 page, brief gap
3. Methods: 1.5 pages, concise
4. Results: 1.5 pages, table-focused
5. Discussion: 0.5 page, short
6. References: 15-20 key papers

The contribution is clear: **evaluation rigor under participant-disjoint design**. Now present it in IATMSI conference style, not journal style.