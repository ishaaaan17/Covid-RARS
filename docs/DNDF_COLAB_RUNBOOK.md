# DNDT / DNDF Google Colab Runbook

This runbook explains how to execute the Deep Neural Decision Tree (DNDT) and Deep Neural Decision Forest (DNDF) pipeline on **Google Colab** using GPU or CPU acceleration.

---

## 1. Quick Start (Colab One-Click)

1. Open **Google Colab** ([colab.research.google.com](https://colab.research.google.com)).
2. Upload or open [`notebooks/11_DNDT_DNDF_RELIABILITY_E2E.ipynb`](../notebooks/11_DNDT_DNDF_RELIABILITY_E2E.ipynb).
3. Set Runtime to **GPU** (Menu: `Runtime` -> `Change runtime type` -> `T4 GPU` or `V100`).
4. Run all cells (`Ctrl + F9` or Menu: `Runtime` -> `Run all`).

---

## 2. Command-Line Execution

If running via terminal or shell in Colab:

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Run DNDF reliability study across all tracks with optimized parameters
python scripts/79_run_dndf_reliability.py \
    --features data/processed/features_compare_is10_top800.csv \
    --external-features data/processed/features_compare_is10_coughvid_cough_top800.csv \
    --num-trees 50 \
    --depth 5 \
    --lr 0.005 \
    --max-epochs 60 \
    --device auto \
    --output-dir reports/dndf

# 3. Generate comparative publication matrix
python scripts/80_make_dndf_evidence_pack.py \
    --dndf-dir reports/dndf \
    --output-csv reports/tables/dndf_comparative_publication_matrix.csv
```

---

## 3. Configuration Customization

Modify hyperparameters in [`configs/dndf_reliability.json`](../configs/dndf_reliability.json):

```json
{
  "architecture": {
    "dndt": {
      "depth": 5,
      "used_features_rate": 1.0,
      "temperature": 1.0
    },
    "dndf": {
      "num_trees": 50,
      "depth": 5,
      "used_features_rate": 0.7,
      "temperature": 1.0
    }
  },
  "training": {
    "learning_rate": 0.005,
    "batch_size": 32,
    "max_epochs": 60,
    "patience": 15,
    "use_smote": true,
    "feature_selection": "f_classif",
    "n_selected_features": 80
  }
}
```

---

## 4. Key Output Artifacts

| Output File | Content |
|---|---|
| `reports/dndf/dndf_final_validation_summary.csv` | Summary metrics across Track A, Track B, Track C, and Multimodal Fusion. |
| `reports/dndf/dndf_calibration_summary.csv` | ECE, Brier score, and NLL before/after shift. |
| `reports/dndf/dndf_operating_points.csv` | Specificity and PPV at fixed $\ge 90\%$ screening sensitivity. |
| `reports/dndf/dndf_decision_curves.csv` | Net clinical benefit across threshold probabilities ($0.05 - 0.50$). |
| `reports/dndf/dndf_bootstrap_ci.csv` | 95% bootstrap confidence intervals for all evaluated configurations. |
