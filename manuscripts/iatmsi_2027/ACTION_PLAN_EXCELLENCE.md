# Concrete Action Plan for IATMSI 2027 Paper Excellence

**Created:** 7 August 2026  
**Based on:** Literature synthesis of 10+ high-quality papers

---

## Executive Summary

The current manuscript is **good** but not yet **exceptional**. Based on analysis of Coppock, Han, Islam, and other papers, we need to improve:

1. **Abstract** - add N, improve flow
2. **Introduction** - strengthen gap paragraph
3. **Methods** - add participant flow, more transparency
4. **Results** - ensure all numbers have CIs
5. **Discussion** - more specific limitations
6. **Comparisons** - add symptoms baseline

---

## Phase 1: Abstract Improvements (Day 1)

### Current Abstract Issues

1. Missing exact cohort N
2. Missing comparator context
3. Results flow could be better

### Target Pattern (from Coppock)

```
Line 1: Context/claim
Line 2: Gap/question
Line 3: Methods with N
Line 4-6: Key results with CIs
Line 7: Bounded conclusion
```

### Concrete Changes Needed

**Change 1: Add cohort size**

Current:
> "We developed an audio-only Coswara system..."

Change to:
> "We analyzed 2,088 quality-passing Coswara participants with fixed train/validation/test partitions (N=1,460/312/316)."

**Change 2: Add comparison context**

Add after primary result:
> "The closest internal multimodal context is Chetupalli et al.'s audio-only fusion at AUROC 0.88."

**Change 3: Bound the claim**

Current:
> "The result does not establish statistical superiority, external validity, or clinical validity."

Keep this, but add:
> "This estimate applies only to the Coswara test cohort and does not generalize to other datasets or time periods."

---

## Phase 2: Introduction Improvements (Day 1-2)

### Current Introduction Strengths

- Good structure overall
- Cites relevant work
- Contributions listed

### Current Introduction Weaknesses

1. Gap paragraph not specific enough
2. Could cite Coppock and Han more prominently
3. Stakes not clear enough

### Concrete Changes

**Para 3 (Gap paragraph) - Needs Strengthening**

Current approach:
> "Evaluation design is equally important. Han et al. showed..."

Improve to:
> "Evaluation design critically affects reported performance. Han et al. demonstrated that biased participant allocation can inflate AUROC from 0.71 to 0.90 using the same model architecture [ref]. Coppock et al. found that matching on confounders reduced audio AUROC from 0.85 to 0.62 and showed that symptoms alone outperformed audio [ref]. These results raise the question: what would the performance of a strong audio model be under controlled participant-disjoint evaluation, and does that estimate predict cross-dataset transfer?"

**Add after contributions:**

> "The study deliberately does not claim external validity for multimodal fusion, because COUGHVID provides only cough recordings. It also does not claim clinical deployment readiness, which would require prospective validation in a target population."

---

## Phase 3: Methods Improvements (Day 2-3)

### Add Participant Flow Diagram

**Create Figure 1: Participant Flow**

Based on TRIPOD+AI requirements, must show:
```
Indexed Coswara release (N=2,746)
    ↓
Label resolution (N=2,114 with positive/negative)
    ↓
Quality screening (N=2,088 passed)
    ↓ Partition
Train (N=1,460, 476 positive)
Validation (N=312, 101 positive)
Test (N=316, 103 positive)
```

**Show exclusions:**
- N=632 unresolved labels → excluded from supervised analysis
- N=26 quality failures → excluded

### Methods Transparency Additions

**Add subsection: "Evaluation Design and Data Separation"**

> "All recordings from a participant remained in one partition. Feature ranking was computed only on training data (N=1,460). Modality model selection used validation data only (N=312). The primary fusion rule (uniform mean) was selected from validation AUROC. The balanced-accuracy threshold was selected on validation data. Test data (N=316) was excluded from all ranking, selection, and threshold decisions."

**Add missingness statement:**

> "Participants with missing modalities were retained and contributed to the modalities they had available. For the cough--speech fusion analysis, 314 test participants had both cough and speech predictions available."

---

## Phase 4: Results Improvements (Day 3-4)

### Ensure All Numbers Have Uncertainty

**Audit checklist:**
- [ ] Every AUROC has 95% CI
- [ ] Primary comparison has statistical test
- [ ] Descriptive differences labeled as such

### Add Table: Literature Context

**Create Table III: Nearest Internal Context**

| Study | Inputs | Protocol | AUROC |
|-------|--------|----------|-------|
| Coswara paper [ref] | 9 sounds + symptoms | Internal 70/15/15 | 0.915 |
| Chetupalli [ref] | Audio fusion | Subject-disjoint | 0.880 |
| FAIR [ref] | 7 sounds | Fixed test | 0.866±0.012 |
| This work | Cough + speech | Participant-disjoint | 0.895 (0.852-0.933) |

**Add footnote:**
> "These are not head-to-head comparisons: cohort snapshots, label definitions, and modalities differ. They establish that the present estimate is within the range of prior internal Coswara results."

### Clarify External Transfer

**Add explicit statement:**

> "COUGHVID contains only cough recordings. External evaluation therefore tests only the cough branch. It does not test transfer of the multimodal fusion system, which requires breath and speech inputs unavailable in COUGHVID."

---

## Phase 5: Discussion Improvements (Day 4-5)

### Strengthen Relation to Prior Work

**Add paragraph:**

> "Our results are consistent with the methodological findings of Han et al. and Coppock et al. The external AUROC range of 0.52-0.55 is comparable to Islam et al.'s reported 0.53 for Coswara-to-COUGHVID transfer. The temporal feature-set instability (Jaccard 0.074) complements Ganitidis et al.'s drift analysis. Together, these studies suggest that evaluation protocol is a first-order determinant of reported performance, and that strong internal results do not establish external portability."

### Make Limitations More Specific

**Current limitation:**
> "This study has five main limitations."

**Improve each limitation:**

**Limitation 1 (Dataset):**
> "First, Coswara is a crowdsourced dataset with self-reported COVID-19 status. The status labels were not independently adjudicated by the present authors and may contain errors. The analysis estimates performance on this specific label definition, not on PCR-confirmed status."

**Limitation 2 (Validation roles):**
> "Second, validation data served multiple roles: modality model selection, fusion selection, and threshold selection. The 800-feature budget was one of three explored during development. A fully nested outer evaluation would estimate the complete procedure more conservatively, but would require additional held-out data that was not available."

**Limitation 3 (Missing modalities):**
> "Third, only 314 test participants had complete cough and speech predictions. The fusion estimate therefore applies to this subset with both modalities available, not to all participants."

**Limitation 4 (External limitation):**
> "Fourth, COUGHVID provides only cough recordings and lacks participant linkage. External evaluation tests cough-only transfer to recordings, not multimodal transfer to participants. A different external dataset with matched modalities would be needed to test multimodal portability."

**Limitation 5 (Scope):**
> "Fifth, this is a retrospective analysis of public data. It does not evaluate prospective clinical use, fairness in deployment populations, or comparison to approved diagnostic tests. Decision curves describe the present sample, not future deployment outcomes."

---

## Phase 6: Missing Analyses (Day 5-7)

### Priority 1: Symptoms Baseline (CRITICAL)

**Rationale:** Coppock showed symptoms outperform audio. We should test this.

**Required:**
1. Extract symptom variables from Coswara metadata
2. Build symptoms-only model
3. Compare AUROC: symptoms vs. audio vs. combined

**If not possible, acknowledge:**
> "A limitation of this analysis is that symptom-only models were not directly compared. Coppock et al. found symptoms outperformed audio, suggesting this comparison is important for future work."

### Priority 2: Metadata-Only Model (We have this)

**Already in manuscript:**
> "The full safe metadata model reached AUROC 0.964"

**Keep and emphasize:**
> "This establishes that non-audio variables are strongly predictive. It does not prove that the audio model encodes the same variables, but it shows that label-predictive collection context exists."

### Priority 3: Negative Controls (We have this)

**Already in manuscript:**
> "After full model retraining with permuted labels, mean AUROC values were 0.503"

**Keep:** This establishes credibility.

---

## Phase 7: Figure Improvements (Day 7-8)

### Figure 1: Study Design

**Must show:**
- Data sources
- Partitioning
- Exclusions
- N at each stage

**Pattern:** Follow TRIPOD+AI flow diagram template

### Figure 2: Results

**Improve current figure:**
- Add N labels to each point
- Add CI error bars
- Distinguish internal vs. external
- Add legend explaining protocol differences

### Figure 3: Mechanism

**Keep but ensure:**
- Clear caption explaining what it shows
- Each panel labeled (A, B, C, D)
- Key finding stated in caption

---

## Phase 8: Final Polish (Day 8-10)

### Abstract Checklist
- [ ] Exact N present
- [ ] CIs with all results
- [ ] Clear interpretation
- [ ] Bounded conclusion

### Introduction Checklist
- [ ] Gap paragraph cites Han and Coppock
- [ ] Contributions specific
- [ ] Stakes clear

### Methods Checklist
- [ ] Participant flow diagram
- [ ] Every decision's data source stated
- [ ] Missingness handled
- [ ] Evaluation unit clear

### Results Checklist
- [ ] All numbers have CIs
- [ ] Literature context table
- [ ] External limitation explicit
- [ ] Descriptive differences labeled

### Discussion Checklist
- [ ] Relation to Han and Coppock explicit
- [ ] Limitations specific (not generic)
- [ ] Each limitation: scope + mechanism + consequence
- [ ] Conclusion bounded

### References Checklist
- [ ] All claims cited
- [ ] DOI links verified
- [ ] No fabricated references

---

## Timeline

| Day | Phase | Deliverable |
|-----|-------|-------------|
| 1 | Abstract + Introduction | Revised opening sections |
| 2 | Methods | Participant flow, transparency |
| 3-4 | Results | CIs audit, literature table |
| 5 | Discussion | Limitations revision |
| 6-7 | Missing analyses | Symptoms baseline (if data) |
| 8 | Figures | Flow diagram, result improvements |
| 9-10 | Final polish | Complete review |

---

## Success Criteria

The paper will be **exceptional** when:

1. ✓ Abstract has N, CIs, bounded conclusion
2. ✓ Introduction gap paragraph cites Han and Coppock
3. ✓ Methods has participant flow and decision transparency
4. ✓ All results have uncertainty
5. ✓ Limitations are specific
6. ✓ External transfer scope is explicit
7. ✓ Literature comparison is protocol-aware
8. ✓ No claims exceed evidence

---

## Key Papers to Re-Read During Revision

1. **Coppock et al.** - for symptoms comparison, negative controls
2. **Han et al.** - for evaluation design emphasis
3. **Islam et al.** - for cross-dataset table structure
4. **TRIPOD+AI** - for checklist compliance
5. **Chetupalli et al.** - for multimodal fusion reporting