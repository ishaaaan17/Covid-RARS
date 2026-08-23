# DNDT / DNDF Google Colab & Kaggle 3-Track Runbook

This runbook guides execution of the **Deep Neural Decision Tree (DNDT)** and **Deep Neural Decision Forest (DNDF)** 3-Track Benchmark Suite on Google Colab or Kaggle.

---

## 🚀 1. One-Click Google Colab Execution

1. Open the interactive notebook directly:  
   👉 **[11_DNDT_DNDF_RELIABILITY_E2E.ipynb on Google Colab](https://colab.research.google.com/github/ishaaaan17/Covid-RARS/blob/main/notebooks/11_DNDT_DNDF_RELIABILITY_E2E.ipynb)**
2. Set Runtime to standard **T4 GPU** (Menu: `Runtime` -> `Change runtime type` -> `T4 GPU`).
3. Run **Section 1, 2, 3** to sync the clean repository and load the features table.
4. Run **Section 3.5 (Preflight Smoke Test)** to verify all tracks in $<15$ seconds:
   ```python
   !python scripts/79_run_dndf_reliability.py --smoke-test --device {device}
   ```
5. Execute individual research tracks:
   * **Cell 4:** **Track 1 (Authors' Exact Paper Reproduction)** -> Replicates Islam et al. (ESWA 2026) 10-fold CV on Cough.
   * **Cell 5:** **Track 2 (Corrected Leak-Free Reproduction)** -> Evaluates nested 10-fold CV with 0% data leakage.
   * **Cell 6:** **Track 3 (COVID-RARS Clinical Reliability Suite)** -> Evaluates participant-disjoint holdouts, true temporal drift, and multimodal fusion.

---

## ⚡ 2. Terminal / Command-Line Execution

You can also run individual tracks via the unified CLI script `scripts/79_run_dndf_reliability.py`:

```bash
# 1. Fast 15-Second Smoke Test
python scripts/79_run_dndf_reliability.py --smoke-test --device auto

# 2. Run Track 1 (Authors' Exact 10-Fold CV)
python scripts/79_run_dndf_reliability.py --track 1 --device auto

# 3. Run Track 2 (Zero-Leakage Corrected 10-Fold CV)
python scripts/79_run_dndf_reliability.py --track 2 --device auto

# 4. Run Track 3 (Full Clinical Reliability Suite + Multimodal Fusion)
python scripts/79_run_dndf_reliability.py --track 3 --device auto

# 5. Run All 3 Tracks Sequentially
python scripts/79_run_dndf_reliability.py --track all --device auto --output-dir reports/dndf
```

---

## 📊 3. Output Artifacts Generated

Results are exported to `reports/dndf/`:

| Artifact File | Description |
|---|---|
| `track1_author_paper_reproduction.csv` | Fold-by-fold accuracy, AUROC, sensitivity, specificity, and precision matching Islam et al. (ESWA 2026). |
| `track2_corrected_leak_free_reproduction.csv` | Methodologically audited fold metrics with zero data leakage. |
| `dndf_final_validation_summary.csv` | Participant-disjoint Track 3 metrics across unimodal and multimodal combinations. |
| `dndf_calibration_summary.csv` | Expected Calibration Error (ECE) and Brier calibration scores. |
