# CLAIMS DOCUMENT: IATMSI 2027 Manuscript

**Purpose:** Define EXACTLY what we can and cannot claim, with evidence

---

## ALLOWED CLAIMS (with evidence)

### Claim 1: Training-only feature ranking
**What we proved:** Feature ranking computed on training data only (N=1,460) and frozen for validation/test
**Evidence:** `hst_stages.py` lines 1157-1165, feature ranking code
**Metric:** 800 features selected from 10,140 candidates
**Specific:** "Feature ranking used only training partition (N=1,460)"

### Claim 2: Validation-only model selection
**What we proved:** Modality models and fusion selected using validation AUROC, test metrics excluded
**Evidence:** `compare_is10_final_validation.py`, selection logic
**Metric:** Validation AUROC 0.842 for cough+speech combination
**Specific:** "Validation AUROC selected cough+speech (0.842) over cough+breath+speech (0.841)"

### Claim 3: Primary test AUROC with CI
**What we proved:** Participant-disjoint test AUROC 0.895 with bootstrap CI
**Evidence:** `final_validation_delta_bootstrap_ci.csv`
**Metric:** AUROC 0.895 (95% CI: 0.852-0.933), N=314
**Specific:** "Cough+speech mean: AUROC 0.895, 95% CI [0.852, 0.933], N=314"

### Claim 4: Fusion vs speech comparison
**What we proved:** Fusion numerically higher than speech but NOT statistically significant
**Evidence:** DeLong test on paired predictions
**Metric:** Difference = 0.007 AUROC, p=0.62
**Specific:** "Fusion AUROC 0.895 vs speech 0.888, DeLong test: difference 0.007, p=0.62, NOT significant"

### Claim 5: Metadata baseline
**What we proved:** Non-audio variables highly predictive
**Evidence:** `metadata_confounding_metrics.csv`
**Metric:** Metadata AUROC 0.964 (95% CI: 0.938-0.984)
**Specific:** "Metadata-only model: AUROC 0.964, 95% CI [0.938, 0.984]"

### Claim 6: Negative control
**What we proved:** Shuffled labels produce chance-level performance (no leakage)
**Evidence:** `metadata_confounding_metrics.csv`
**Metric:** Shuffled AUROC 0.503
**Specific:** "Shuffled-label retraining: mean AUROC 0.503"

### Claim 7: Comparable to prior internal results
**What we proved:** Our estimate (0.895) is within range of prior Coswara internal estimates (0.88-0.92)
**Evidence:** Literature comparison
**Metric:** Chetupalli 0.880, Bhattacharya 0.915, FAIR 0.866
**Specific:** "Within range of prior internal estimates (0.88-0.92) but NOT superior"

---

## NEGATIVE FINDINGS (Report with same rigor)

### Negative Finding 1: Fusion not statistically superior to speech
**What failed:** Fusion vs single-modality comparison
**Metric:** Difference 0.007 AUROC, p=0.62, 95% CI [-0.021, 0.035]
**Evidence:** DeLong test, paired predictions
**Report as:** "Fusion gain (0.007 AUROC) not statistically significant (p=0.62)"

### Negative Finding 2: Fusion does not improve F1
**What failed:** F1 metric
**Metric:** Fusion F1 0.729 vs speech F1 0.740
**Evidence:** Test results table
**Report as:** "F1 decreased: fusion 0.729 vs speech 0.740"

### Negative Finding 3: Breathing excluded from primary model
**What failed:** Three-modality fusion
**Metric:** Validation AUROC 0.841 (3-modality) vs 0.842 (2-modality)
**Evidence:** Validation selection
**Report as:** "Breathing excluded: 2-modality (0.842) selected over 3-modality (0.841)"

### Negative Finding 4: External transfer NOT tested
**What's missing:** Cross-dataset validation
**Reason:** Outside conference scope
**Report as:** "External transfer to COUGHVID not tested in this paper (separate protocol required)"

### Negative Finding 5: Temporal validation NOT tested
**What's missing:** Time-stratified evaluation
**Reason:** Outside conference scope
**Report as:** "Temporal stability not evaluated (requires separate protocol)"

---

## FORBIDDEN CLAIMS (What NOT to claim)

### Forbidden Claim 1: "Fusion improves performance"
**Reason:** Not statistically significant (p=0.62)
**Correct phrasing:** "Fusion achieved 0.895 vs speech 0.888; difference not statistically significant"

### Forbidden Claim 2: "State-of-the-art results"
**Reason:** No protocol-matched comparison, different cohort snapshots
**Correct phrasing:** "Comparable to prior internal estimates (0.88-0.92)"

### Forbidden Claim 3: "Model generalizes to external datasets"
**Reason:** External transfer not tested
**Correct phrasing:** "External validity not established"

### Forbidden Claim 4: "Clinical deployment ready"
**Reason:** Retrospective analysis, no prospective validation
**Correct phrasing:** "Does not establish clinical readiness"

### Forbidden Claim 5: "Audio outperforms metadata"
**Reason:** Metadata AUROC 0.964 > Audio AUROC 0.895
**Correct phrasing:** "Metadata (0.964) outperformed audio (0.895)"

### Forbidden Claim 6: "Novel fusion methodology"
**Reason:** Used standard uniform averaging
**Correct phrasing:** "Evaluated fusion under rigorous selection protocol"

---

## METRICS SPECIFICATION

### Primary Metrics (MUST report)
- AUROC: 0.895 (95% CI: 0.852-0.933)
- AUPRC: 0.862
- Balanced Accuracy: 0.808
- F1: 0.729
- N: 314 participants

### Comparison Metrics (MUST report)
- Speech AUROC: 0.888 (95% CI: 0.842-0.930)
- Cough AUROC: 0.862 (95% CI: 0.812-0.908)
- Breathing AUROC: 0.828 (95% CI: 0.775-0.877)

### Baseline Metrics (MUST report)
- Metadata AUROC: 0.964 (95% CI: 0.938-0.984)
- Shuffled AUROC: 0.503

### Statistical Tests (MUST report)
- Fusion vs speech: DeLong test, difference 0.007, p=0.62
- 95% CI for difference: [-0.021, 0.035]

---

## EVIDENCE FILES

### Primary Results
- `reports/tables/final_validation_delta_bootstrap_ci.csv` - Test metrics with CIs
- `reports/tables/compare_is10_final_validation_metrics.csv` - Validation selection

### Baselines
- `manuscripts/source_artifacts/data/outputs/metrics/metadata_confounding_metrics.csv` - Metadata baseline

### Literature Context
- `manuscripts/source_artifacts/derived/manuscript_final_validation_summary.csv` - Summary for manuscript

---

## TERMINOLOGY (Maximum 5 essential terms)

### Essential Terms (USE THESE)
1. **Participant-disjoint** - All recordings from one participant in one partition
2. **Training-only** - Computed using training partition only
3. **Validation-only** - Selected using validation partition only
4. **Uniform fusion** - Mean of modality probabilities
5. **Bootstrap CI** - 95% confidence interval from 1000 resamples

### FORBIDDEN Terminology
- "Train-only scaler" (use "training-only")
- "Deploy-clean unit" (use "test partition")
- Internal jargon (archived, legacy, cached)
- Code variable names (multi_config, set_seed)
- Directory structure names

---

## STATISTICAL REPORTING STANDARDS

### For EVERY metric, include:
- Mean value
- 95% confidence interval
- Sample size (N)
- Statistical test (if comparison)

### Example format:
"AUROC 0.895 (95% CI 0.852-0.933, N=314)"
"Difference 0.007 (95% CI -0.021 to 0.035, p=0.62, DeLong test)"

---

## VERIFICATION CHECKLIST

Before writing manuscript:
- [x] All allowed claims have specific metric
- [x] All allowed claims have evidence file
- [x] All negative findings documented
- [x] All forbidden claims listed
- [x] Terminology limited to 5 terms
- [x] No vague claims ("good", "high", "significant")
- [x] Statistical tests specified for all comparisons