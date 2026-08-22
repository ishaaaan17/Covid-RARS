# Notebook Order

This project is now single-notebook-first. Open the hidden project folder in Jupyter and start here:

```text
notebooks/00_RUN_EVERYTHING_PUBLICATION.ipynb
```

This single notebook contains the full execution path with toggles for core Coswara work, CNN, COUGHVID, publication extras, and paper tables. It has 20 cells total: 10 code cells and 10 markdown cells.

The older master notebook is still available:

```text
notebooks/00_MASTER_RUN_ALL.ipynb
```

Use it only if you want the smaller core-only Coswara runner.

Optional review/debug notebooks:

1. `01_dataset_audit.ipynb` - labels, participants, splits, modality availability, leakage checks.
2. `02_quality_review.ipynb` - corrupt/short/silent/clipped audio, event duration, quality by modality and label.
3. `03_feature_review.ipynb` - MFCC/acoustic feature health, missing values, constants, correlation, spectrogram index.
4. `04_ml_baseline_review.ipynb` - dummy-vs-real baseline comparison, AUROC/AUPRC/ECE, prediction sanity checks.
5. `05_cnn_review.ipynb` - optional GPU CNN metrics, training curves, logits, CNN-vs-ML comparison.
6. `06_calibration_fusion_review.ipynb` - branch calibration, reliability diagram, weighted-vs-uniform fusion.
7. `07_shift_confounding_review.ipynb` - subgroup metrics, quality sensitivity, metadata coverage, limitation notes.
8. `08_publication_grade_experiments.ipynb` - publication extras as a separate notebook if you do not use the single full runner.
9. `09_HST_RELIABILITY_E2E.ipynb` - hierarchical spectrogram transformer reliability pipeline.
10. `10_HST_CAPACITY_INTERNAL_FUSION.ipynb` - HST capacity and internal fusion.
11. `11_DNDT_DNDF_RELIABILITY_E2E.ipynb` - Deep Neural Decision Tree/Forest (DNDT/DNDF) Colab-ready reliability study.

Rule: core logic lives in `src/covid_rars/`; notebooks orchestrate, inspect, visualize, and save report assets.
