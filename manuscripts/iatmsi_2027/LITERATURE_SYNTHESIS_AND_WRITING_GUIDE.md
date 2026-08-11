# Literature Synthesis and Writing Excellence Guide for IATMSI 2027

**Created:** 7 August 2026  
**Purpose:** Extract best practices from 15+ key papers to write an exceptional IATMSI 2027 submission

---

## Part 1: Paper Structure Analysis

### 1.1 Abstract Patterns (Based on High-Impact Papers)

**Coppock et al. (Nature Machine Intelligence 2024) - GOLD STANDARD:**
```
Structure:
1. Context sentence (what's claimed)
2. Challenge sentence (why it matters)
3. Methods sentence (what we did - cohort size)
4. Methods sentence (comparisons made)
5. Results sentence (primary finding with numbers)
6. Results sentence (secondary finding)
7. Results sentence (negative control)
8. Conclusion sentence (implication)
9. Limitation hint
```

**Key insight:** They report **exact cohort sizes** in abstract: "67,842 UK participants with PCR tests"

**Han et al. (npj Digital Medicine 2022):**
```
Structure:
1. What exists (audio testing proposed)
2. Gap (performance under realistic evaluation unknown)
3. What we did (same models, different protocols)
4. Results with confidence intervals
5. Implication (bias causes optimism)
```

**Best Practice for IATMSI:**
- First sentence: Context and claim
- Second sentence: Gap/question
- Third sentence: Methods and cohort size
- Fourth-sixth sentences: Key results with CIs
- Final sentence: Bounded conclusion

---

### 1.2 Introduction Flow Analysis

**Common pattern across all high-quality papers:**

| Paragraph | Content | Example from Coppock |
|-----------|---------|----------------------|
| 1 | What's been proposed | Audio COVID-19 screening proposed |
| 2 | What's been claimed | High AUCs reported |
| 3 | What's missing/wrong | Lack of proper controls, symptoms not compared |
| 4 | Why it matters | Clinical deployment consequences |
| 5 | What we do | Large-scale controlled comparison |
| 6 | Contributions | (1) Size (2) Controls (3) Symptoms baseline |

**Key insight:** The gap paragraph (3) is CRITICAL. It must be:
- Specific (not generic "limited evaluation")
- Evidence-based (cite specific papers showing the problem)
- Stakes-clear (explain why this matters)

**Avoid:** "There is limited research on..." (too vague)
**Better:** "Han et al. showed biased splits can inflate AUROC from 0.71 to 0.90 [cite]. Coppock et al. found matching reduced AUROC from 0.85 to 0.62 [cite]. However, these analyses did not..."

---

### 1.3 Methods Section Best Practices

**From TRIPOD+AI requirements:**

**MUST HAVE in Methods:**
1. **Participant flow diagram** - explicit inclusion/exclusion
2. **Missingness handling** - how many excluded, why
3. **Feature selection location** - training-only or not
4. **Hyperparameter selection** - validation-only or nested
5. **Threshold selection** - where, how
6. **Evaluation unit** - participant, recording, segment
7. **Pairing structure** - what's paired, what's independent

**Example from Han et al.:**
- "We used the same model architecture across three evaluation designs"
- "Feature selection was performed only on training folds"
- "Test set was held out from all model development"

**Common mistake:** Not stating WHERE decisions were made
**Best practice:** Every decision point states its data source

---

### 1.4 Results Section Excellence

**Coppock et al. pattern:**

1. **Primary result first** with CI: "AUROC 0.62 (95% CI 0.58-0.66)"
2. **Comparator result** with same metric: "Symptoms alone: AUROC 0.68"
3. **Statistical test**: "Difference: -0.06 (95% CI -0.11 to -0.01, p=0.02)"
4. **Negative controls**: "Shuffled labels: AUROC 0.50"
5. **Subgroup results**: In supplementary, not main text

**Key tables pattern:**
- Table 1: Cohort characteristics (ALWAYS)
- Table 2: Primary results with CIs
- Table 3: Literature comparison (or supplementary)

**Figures pattern:**
- Figure 1: Study design/participant flow
- Figure 2: Primary results (often with CIs)
- Figure 3: Mechanism/sensitivity analysis

---

### 1.5 Discussion Structure

**Consistent pattern across all papers:**

| Section | Content | Length |
|---------|---------|--------|
| Principal findings | What did we find? | 1 paragraph |
| Relation to prior work | How does this compare? | 1-2 paragraphs |
| Implications | What does this mean? | 1 paragraph |
| Strengths | What did we do well? | 1 paragraph |
| Limitations | What could be wrong? | 1-2 paragraphs |
| Conclusion | So what? | 1 paragraph |

**Critical: Limitations must be SPECIFIC**

**Bad:** "This study has limitations."
**Good:** "First, COUGHVID lacks participant linkage, so external analysis uses recordings rather than participants. Second, the metadata variables are measured imperfectly and cannot exhaust all confounding paths. Third, only 61 participants had complete audio and metadata for the incremental value analysis."

---

## Part 2: Claim Structure Analysis

### 2.1 How High-Quality Papers Structure Claims

**Claim = Observation + Boundary + Uncertainty**

**Example from Coppock:**
```
Claim: "Audio classifiers showed no evidence of improved screening over symptoms checkers."
Boundary: "In a UK cohort of 67,842 participants with PCR tests."
Uncertainty: "95% CI for difference: -0.11 to -0.01."
```

**Example from Han:**
```
Claim: "Biased participant allocation can inflate AUROC by 0.19."
Boundary: "Using the same COVID-19 Sounds data and model architecture."
Uncertainty: "95% CI: 0.14 to 0.24."
```

**Key pattern:**
1. State the observation
2. State the scope (what population, what comparison)
3. State the uncertainty (CI, p-value, or acknowledgment of imprecision)

---

### 2.2 Negative Results Reporting

**Best practice from reviewed papers:**

When results are negative/null:
1. **Report exactly** - don't round up or down
2. **State what it means** - "does not establish X"
3. **State what it doesn't mean** - "does not prove absence of Y"

**Example:**
"The fusion AUROC was 0.895 vs. speech AUROC of 0.888. The difference was 0.007. Without paired uncertainty, this is not evidence of superiority. It is also not evidence of equivalence."

---

## Part 3: Figure Design Excellence

### 3.1 Common Figure Patterns

**Study Design Figure (Figure 1 pattern):**

All high-quality papers have a study design diagram showing:
- Data sources (boxes)
- Splitting/partitioning (arrows)
- What happens at each stage (labels)
- What's excluded (explicit "excluded" boxes)

**Best practices:**
- Use consistent shapes/colors
- Show N at each stage
- Show exclusions explicitly
- Time flow: left-to-right or top-to-bottom

**Results Figure (Figure 2 pattern):**

**Option A: Bar/point plot with CIs**
- X-axis: Methods/protocols
- Y-axis: Metric (AUROC)
- Error bars: 95% CI
- Dotted line: chance/baseline

**Option B: ROC curves (avoid if possible)**
- Only if comparing specific thresholds
- Add AUROC values in legend
- Add CI for AUROC

**Key insight:** Bar/point plots with CIs are more informative than ROC curves for comparison

### 3.2 Figure Caption Excellence

**Pattern:**
```
Figure X: [What it shows]. [Panel descriptions]. [Key finding]. [Boundary].

Example from Coppock:
"Figure 2: Model performance comparison. (A) Audio-only versus symptoms-based models. (B) Combined models versus symptoms. Error bars show 95% bootstrap CIs. AUROC differences are shown with 95% CIs and p-values."
```

**Key elements:**
1. Title/what it shows
2. Panel labels and descriptions
3. What error bars/uncertainty means
4. Key comparison or finding

---

## Part 4: Table Design Excellence

### 4.1 Cohort Table (Always First)

**Essential columns:**
- Characteristic
- Overall (N, %)
- Positive class (N, %)
- Negative class (N, %)
- P-value or standardized difference

**Example from reviewed papers:**
```
Table 1: Cohort Characteristics

| Characteristic | Overall (N=67,842) | Positive (N=4,571) | Negative (N=63,271) | P-value |
|----------------|---------------------|---------------------|----------------------|---------|
| Age, mean (SD) | 45.2 (15.3) | 42.1 (14.8) | 45.4 (15.3) | <0.001 |
| Female, n (%) | 38,291 (56.4%) | 2,314 (50.6%) | 35,977 (56.8%) | <0.001 |
```

### 4.2 Results Table Pattern

**Essential columns:**
- Model/method name
- N (participants or recordings)
- Primary metric (AUROC)
- Secondary metrics (AUPRC, accuracy)
- Uncertainty (95% CI)
- Notes/explanations

**Key practice:** 
- Report N for each row (may differ)
- Use consistent decimal places
- Put primary comparison in bold or separate section
- Add footnotes for exploratory rows

---

## Part 5: Writing Style Excellence

### 5.1 Sentence-Level Patterns

**Avoid hedge stacking:**
- Bad: "It may potentially suggest that there might be some indication of..."
- Good: "This suggests X. However, the CI includes Y."

**Be precise about sources:**
- Bad: "Previous studies have shown..."
- Good: "Han et al. showed AUROC inflation from 0.71 to 0.90 when participant allocation was biased [ref]."

**State limitations proactively:**
- Don't wait for Discussion
- In Methods: "A limitation of this design is..."
- In Results: "This comparison is descriptive because..."

### 5.2 Paragraph Structure

**Claim-Evidence-Boundary pattern:**

1. **Claim sentence:** What did we find?
2. **Evidence sentence:** With what numbers?
3. **Boundary sentence:** What are the limits?
4. **Comparison sentence:** How does this compare?

**Example:**
"The selected multimodal system achieved AUROC 0.897 (95% CI 0.854-0.935). This estimate applies only to the Coswara test cohort and does not generalize to other datasets or time periods. The closest comparable result is Chetupalli et al.'s audio-only fusion at AUROC 0.88, although their cohort snapshot and model differed."

---

## Part 6: Specific Recommendations for IATMSI 2027

### 6.1 Based on Literature Review

**Paper focus should be:**

Based on Coppock and Han papers showing:
1. **Evaluation protocol sensitivity** is the key contribution
2. **Symptoms baseline** is essential comparator
3. **Negative controls** establish credibility
4. **External transfer** shows transportability limits

**For 6-page IATMSI:**

1. **Title:** Should name the contribution, not just the topic
   - Bad: "COVID-19 Audio Classification with Deep Learning"
   - Good: "Protocol-Sensitive Evaluation of COVID-19 Respiratory Audio Models"

2. **Abstract:** Follow Coppock pattern with exact Ns and CIs

3. **Introduction:** 
   - Para 1: Audio screening proposed
   - Para 2: High performance claimed
   - Para 3: Evaluation gap (Han/Coppock showed protocol matters)
   - Para 4: What we do (controlled protocol comparison)
   - Para 5: Contributions

4. **Methods:**
   - Follow TRIPOD+AI checklist
   - Explicit participant flow
   - State every decision point's data source

5. **Results:**
   - Table 1: Cohort
   - Table 2: Protocol comparison with CIs
   - Figure 1: Study design
   - Figure 2: Protocol performance ladder with CIs
   - One mechanism figure if space

6. **Discussion:**
   - Principal findings
   - Relation to Han/Coppock
   - Limitations (specific)
   - Conclusion (bounded)

### 6.2 Common Pitfalls to Avoid

**From literature review:**

1. **Don't compare across datasets** without acknowledging differences
2. **Don't claim superiority** without statistical support
3. **Don't report internal performance** as if it generalizes
4. **Don't omit cohort sizes** - always report N
5. **Don't use vague limitations** - be specific about what could be wrong
6. **Don't extrapolate** from Coswara to "clinical deployment"
7. **Don't claim multimodal transfer** to cough-only COUGHVID

---

## Part 7: Writing Checklist

### Before Writing

- [ ] Read Coppock et al. (2024) fully - this is the gold standard
- [ ] Read Han et al. (2022) fully - bias evaluation benchmark
- [ ] Read TRIPOD+AI checklist - know what must be reported
- [ ] Identify the ONE question the paper answers
- [ ] List contributions as specific claims with boundaries

### While Writing

- [ ] Abstract: exact Ns and CIs
- [ ] Introduction: gap paragraph cites specific evidence
- [ ] Methods: every decision states its data source
- [ ] Results: every number has uncertainty
- [ ] Discussion: limitations are specific, not generic
- [ ] Conclusion: bounded, not extrapolated

### After Writing

- [ ] Every table has N column
- [ ] Every figure has caption explaining uncertainty
- [ ] Every claim has boundary
- [ ] Every comparison acknowledges differences
- [ ] Limitations are specific
- [ ] No "novel" without definition
- [ ] No "state-of-the-art" without protocol-matched comparison

---

## Part 8: Key Lessons Summary

### From Coppock et al. (Nature MI 2024):

1. **Large cohorts enable stronger claims** - N=67,842
2. **Symptoms are strong comparators** - must include
3. **Negative controls establish credibility** - shuffled labels
4. **Precise language** - "no evidence of improvement" not "no improvement"
5. **Limitations upfront** - in abstract, in methods, in discussion

### From Han et al. (npj DM 2022):

1. **Same model, different protocols** - isolates the evaluation effect
2. **Show the range** - from realistic (0.71) to biased (0.90)
3. **Clear implications** - "these biases could affect deployment"
4. **Exact numbers** - not just "significant difference"

### From TRIPOD+AI (BMJ 2024):

1. **Participant flow diagram** - mandatory
2. **Missingness handling** - explicit
3. **Hyperparameter selection location** - must state
4. **Evaluation unit** - participant, not recording
5. **Uncertainty reporting** - CIs, not just p-values

### From Aytekin et al. (IEEE JBHI 2024 - HST):

1. **Architecture paper structure:**
   - Method description in detail
   - Ablation studies
   - Multiple datasets
   - Comparison to baselines
2. **BUT:** Internal results only - acknowledge limitation
3. **Good practice:** Report exact training details, hyperparameters

### From Islam et al. (ESWA 2026 - DNDT/DNDF):

1. **Cross-dataset table** - shows transfer explicitly
2. **Multiple source-target directions** - comprehensive
3. **Cohort balancing** - acknowledges selection
4. **Key finding:** Internal AUROC 0.92-0.93, cross-dataset 0.53
5. **Important:** Shows the gap between internal and external

---

## Conclusion

The exceptional papers share these characteristics:

1. **Precise claims** with boundaries and uncertainty
2. **Evaluation rigor** that matches the claim
3. **Specific limitations** that acknowledge scope
4. **Literature comparison** that respects protocol differences
5. **Figures/tables** that show N and uncertainty
6. **Discussion** that doesn't overclaim

For IATMSI 2027, the paper should:
- Ask one focused question about protocol sensitivity
- Answer it with controlled comparisons
- Report uncertainty and boundaries explicitly
- Compare to Han and Coppock as methodological benchmarks
- Acknowledge what external transfer tests (cough) vs. doesn't test (multimodal)