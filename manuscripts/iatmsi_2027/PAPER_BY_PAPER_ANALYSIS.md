# Paper-by-Paper Analysis: What Makes Each Exceptional

**Created:** 7 August 2026  
**Purpose:** Detailed extraction of excellence patterns from each key paper

---

## 1. Coppock et al. (Nature Machine Intelligence 2024)

**Full Title:** "Audio-Based AI Classifiers Show No Evidence of Improved COVID-19 Screening Over Simple Symptoms Checkers"

### Why This Paper is Exceptional

**1. Cohort Scale and Quality**
- N = 67,842 UK participants
- All had PCR tests (gold standard labels)
- Prospective data collection (not crowdsourced)
- Clear inclusion/exclusion criteria

**2. Comparator Rigor**
- Compared audio to SYMPTOMS (not just chance)
- Symptoms alone: AUROC 0.68
- Audio alone: AUROC 0.62
- Combined: AUROC 0.69
- **Key insight:** Audio added nothing beyond symptoms

**3. Evaluation Rigor**
- Random split (AUROC 0.85)
- Matched split (AUROC 0.62) - dramatic difference
- Longitudinal analysis
- Prespecified analysis plan

**4. Writing Excellence**

**Abstract pattern:**
```
Line 1: "AI-based COVID-19 screening from respiratory audio has been proposed..."
Line 2: "However, rigorous evaluation has been limited."
Line 3: "We enrolled 67,842 UK participants with PCR tests..."
Line 4: "Audio achieved AUROC 0.62 (95% CI 0.58-0.66)."
Line 5: "Symptoms alone achieved AUROC 0.68 (95% CI 0.64-0.72)."
Line 6: "Combined model: AUROC 0.69 (95% CI 0.65-0.73)."
Line 7: "Audio classifiers showed no evidence of improvement."
```

**Key writing patterns:**
- Exact N in abstract: "67,842 participants"
- CIs with every result
- Negative finding stated clearly, not hedged
- "No evidence of improvement" (correct interpretation of null result)
- NOT "audio is useless" (too strong)

**5. Discussion Structure**
- Paragraph 1: Principal findings
- Paragraph 2: Relation to prior work (Han et al.)
- Paragraph 3: Mechanisms (why might this be?)
- Paragraph 4: Strengths (size, PCR, matching)
- Paragraph 5: Limitations (specific, not generic)
- Paragraph 6: Implications (for deployment)

**6. Limitations Excellence**

Each limitation is:
- Specific: "COUGHVID lacks PCR confirmation"
- Explained: "This means labels may contain noise"
- Scoped: "This affects the external estimate, not internal"

### What We Should Emulate

1. **Symptoms comparator** - we need this in our paper
2. **Exact N in abstract** - we should add "N = 2,088 participants"
3. **Negative controls** - we have shuffle-label results
4. **Precise language** - "no evidence of superiority" not "equivalent"
5. **Matching analysis** - we could add basic matching

---

## 2. Han et al. (npj Digital Medicine 2022)

**Full Title:** "Sounds of COVID-19: Exploring Realistic Performance of Audio-Based Digital Testing"

### Why This Paper is Exceptional

**1. Same Model, Different Protocols**

The brilliance: They used ONE model architecture and showed how AUROC changes with evaluation design:
- Realistic split: AUROC 0.71
- Biased split: AUROC 0.90

This isolates the **evaluation effect** from the model effect.

**2. Clear Demonstration of Bias Sources**

They showed specific biases:
- Participant overlap → inflation
- Demographic imbalance → inflation
- Label leakage → inflation

**3. Methods Transparency**

They explicitly state:
- "Feature selection was performed only on training folds"
- "Test set was held out from all model development"
- "We used the same architecture across all designs"

**4. Results Presentation**

Table showing:
| Design | AUROC | 95% CI |
|--------|-------|---------|
| Realistic | 0.71 | 0.65-0.77 |
| Moderate bias | 0.82 | 0.76-0.88 |
| High bias | 0.90 | 0.85-0.95 |

This is **cleaner** than ROC curves.

**5. Writing Pattern**

Introduction structure:
- Para 1: Audio screening proposed
- Para 2: High performance claimed (cite specific papers)
- Para 3: But evaluation varies (cite specific differences)
- Para 4: We test how much this matters
- Para 5: Contributions

### What We Should Emulate

1. **Same model across protocols** - we do this
2. **Explicit design comparison** - we should make this clearer
3. **Clean AUROC table** - better than ROC curves
4. **Methods transparency** - state every decision's data source

---

## 3. Aytekin et al. (IEEE JBHI 2024) - HST Paper

**Full Title:** "COVID-19 Detection From Respiratory Sounds With Hierarchical Spectrogram Transformers"

### Why This Paper is Good (But Not Perfect)

**Strengths:**

1. **Clear architecture description**
   - Exact dimensions: 224×224 spectrograms
   - Patch sizes, attention windows, merging stages
   - Reproducible details

2. **Multiple datasets**
   - Cambridge internal
   - COUGHVID internal (separate task)
   - Shows generalization within-dataset

3. **Ablation studies**
   - With/without attention
   - Different patch sizes
   - Shows what matters

**Limitations (Important to Note):**

1. **Internal only** - no frozen transfer to new dataset
2. **10-fold CV** - participant grouping unclear
3. **Dataset-specific tasks** - Cambridge ≠ COUGHVID ≠ Coswara
4. **No comparison to symptoms** - what's the baseline?

### What We Should Emulate

1. **Architecture details** - for reproducibility
2. **Ablation studies** - for understanding
3. **Multiple datasets** - for credibility

### What We Should AVOID

1. **Claiming transfer** without frozen external
2. **Mixing datasets** without acknowledging differences
3. **Omitting baselines** (symptoms, metadata)

---

## 4. Islam et al. (ESWA 2026) - DNDT/DNDF Paper

**Full Title:** "Robust COVID-19 Detection From Cough Sounds Using Deep Neural Decision Tree and Forest: A Comprehensive Cross-Datasets Evaluation"

### Why This Paper is Excellent for Cross-Dataset

**1. Honest Cross-Dataset Table**

| Source | Target | AUROC |
|--------|--------|-------|
| Coswara | Coswara (internal) | 0.92 |
| COUGHVID | COUGHVID (internal) | 0.93 |
| Coswara | COUGHVID (transfer) | 0.53 |

This shows the **transfer gap** clearly.

**2. Multiple Directions**

They test:
- Coswara → COUGHVID
- COUGHVID → Coswara
- Other datasets too

**3. Cohort Selection Disclosure**

They explicitly state:
- "Target cohort balanced by label (680/680)"
- "No target fitting in transfer"
- This is honest about selection

**4. Methods Clarity**

- "We froze the source model"
- "Applied unchanged to target"
- "No target-label retraining"

### What We Should Emulate

1. **Cross-dataset table** - clear source/target separation
2. **Honest reporting** - show the gap
3. **Transfer methodology** - frozen, no retraining

---

## 5. Chetupalli et al. (IEEE JTEHM 2023)

**Full Title:** "Multi-Modal Point-of-Care Diagnostics for COVID-19 Based on Acoustics and Symptoms"

### Why This Paper is Important

**1. Directly Relevant to Our Work**

- Uses Coswara
- Tests multimodal fusion
- Subject-disjoint protocol

**2. Shows Fusion Works Internally**

- Audio only: AUROC 0.88
- Audio + symptoms: AUROC 0.96

**3. Clear Methods**

- "Subject-disjoint 80/20 split"
- "We selected fusion from validation"
- Specific modalities tested

**4. Results Table Pattern**

| System | Modality | AUROC |
|--------|----------|-------|
| Single | Cough | 0.82 |
| Single | Breath | 0.75 |
| Single | Speech | 0.85 |
| Fusion | All | 0.88 |
| Fusion | + Symptoms | 0.96 |

### What We Should Emulate

1. **Subject-disjoint split** - we do this
2. **Validation-selected fusion** - we do this
3. **Clear comparison to symptoms** - we should add

---

## 6. Bhattacharya et al. (Scientific Data 2023) - Coswara Paper

**Full Title:** "Coswara: A Respiratory Sounds and Symptoms Dataset for Remote Screening of SARS-CoV-2 Infection"

### Why This Paper is the Foundation

**1. Dataset Documentation Excellence**

- Exact cohort construction
- Inclusion/exclusion criteria
- Quality control procedures
- Missingness handling
- Label definitions

**2. Ethics and Consent**

- Explicit consent statement
- IRB approval details
- Data use terms

**3. Baseline Results**

- Their own baseline: AUROC 0.915
- With symptoms: higher
- Clear this is internal only

**4. Limitations Honesty**

- "Self-reported status"
- "Not clinically validated"
- "Retrospective collection"

### What We Should Emulate

1. **Cite for cohort construction** - our baseline
2. **Use their consent wording** - for ethics section
3. **Acknowledge their baseline** - as comparison

---

## 7. TRIPOD+AI (BMJ 2024)

**Full Title:** "TRIPOD+AI Statement: Updated Guidance for Reporting Clinical Prediction Models That Use Regression or Machine Learning Methods"

### Why This is Essential

**1. Checklist of Required Items**

Every AI prediction paper MUST report:
- Participant flow (diagram)
- Missingness handling
- Feature selection location
- Hyperparameter tuning location
- Threshold selection location
- Evaluation unit
- Uncertainty quantification
- Comparison methods
- Limitations specific

**2. Structured Abstract Template**

- Background: one sentence
- Methods: cohort size, design, analysis
- Results: primary finding with CI
- Conclusions: bounded interpretation

**3. Transparent Reporting**

- Not just "we used ML"
- Exact algorithms, exact tuning, exact data splits
- Every decision's data source stated

### What We MUST Do

1. **Create participant flow diagram**
2. **State every decision's data source**
3. **Report missingness**
4. **Report uncertainty with CIs**
5. **State limitations specifically**

---

## 8. Laguarta et al. (IEEE OJEMB 2020)

**Full Title:** "COVID-19 Artificial Intelligence Diagnosis Using Only Cough Recordings"

### Why This Paper is Notable (But Limited)

**Strengths:**
- Private cohort (N=5,320)
- Forced cough (standardized)
- High performance claimed (AUROC 0.97)

**Limitations:**
- Private data (not reproducible)
- Internal only (no external)
- No comparison to symptoms
- Selection not transparent

**Key lesson:** Private datasets limit credibility

---

## 9. Pahar et al. (Computers in Biology and Medicine 2022)

**Full Title:** "COVID-19 Detection in Cough, Breath and Speech Using Deep Transfer Learning and Bottleneck Features"

### Why This Paper is Good

**1. Transfer Learning Focus**

- Tests transfer across modalities
- Shows what transfers and what doesn't

**2. Small External Target**

- Sarcos: only 44 subjects
- Shows genuine transfer
- But limited by size

**3. Honest Reporting**

- Internal: AUROC 0.982
- Transfer: AUROC 0.954
- Gap acknowledged

---

## 10. Grant et al. (Sensors 2022)

**Full Title:** "Considerations and Challenges for Real-World Deployment of an Acoustic-Based COVID-19 Screening System"

### Why This Paper is Important

**1. Deployment Perspective**

- Not just performance metrics
- Device mismatch analysis
- Real-world challenges

**2. Key Finding**

- Web cough: AUROC 0.42
- Android cough: AUROC 0.75
- **Device matters!**

**3. Lessons**

- Deployment is different from evaluation
- Device/platform affects performance
- Need deployment-specific testing

---

## Synthesis: Common Patterns of Excellence

### What ALL Excellent Papers Have

1. **Explicit cohort description**
   - N at each stage
   - Inclusion/exclusion
   - Missingness

2. **Clear evaluation protocol**
   - Split type
   - What's held out
   - What's paired

3. **Uncertainty with every number**
   - 95% CI
   - Interpretation of CI

4. **Specific limitations**
   - Not generic "we have limitations"
   - Each limitation: scope, mechanism, consequence

5. **Honest comparison**
   - Acknowledge differences
   - Don't claim superiority without stats

6. **Bounded conclusions**
   - What was tested
   - What generalizes
   - What doesn't

### What Mediocre Papers Do Wrong

1. Vague methods ("we used deep learning")
2. No uncertainty reporting
3. Generic limitations
4. Unbounded claims ("our model is superior")
5. Internal-only results presented as general
6. Missing cohort details
7. No comparison to baselines (symptoms, metadata)

---

## Specific Recommendations for IATMSI 2027

### Based on Coppock:
- Add symptoms baseline comparison
- Report exact N in abstract
- Use "no evidence of" for null results

### Based on Han:
- Show protocol comparison clearly
- Use same model across protocols
- State evaluation differences explicitly

### Based on Islam:
- Create cross-dataset table
- Show transfer gap honestly
- State frozen transfer clearly

### Based on Chetupalli:
- Keep subject-disjoint protocol
- Use validation-selected fusion
- Show modalities separately

### Based on TRIPOD+AI:
- Create participant flow diagram
- State every decision's data source
- Add CIs to all results
- Specify limitations

### Based on Aytekin:
- Report architecture clearly
- But acknowledge internal-only limitation

### Key Gaps in Current Manuscript

1. **Symptoms baseline missing** - need to add
2. **N not in abstract** - add "N = 2,088"
3. **Participant flow diagram** - need to create
4. **Uncertainty on all numbers** - some missing
5. **Limitations more specific** - make concrete