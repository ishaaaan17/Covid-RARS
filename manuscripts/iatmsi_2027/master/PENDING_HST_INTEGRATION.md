# HST Integration Gate

The confirmatory HST run is intentionally absent from the manuscript results.
It may be integrated only after all conditions below are satisfied.

## Required evidence

1. All 20 predefined jobs finish: 10 cough folds and 10 speech folds.
2. Each job uses exactly 100 epochs under the accepted resource freeze.
3. The final evidence pack confirms participant-disjoint folds, no test-driven
   checkpoint selection, and finite optimization updates.
4. Branch metrics are participant-level and include AUROC, AUPRC, calibration,
   operating-point metrics, and uncertainty.
5. Fusion is applied only to aligned cough--speech participants.
6. Uniform, validation-AUPRC-weighted, and validation-only logistic fusion are
   all reported; test labels do not choose the reported method.
7. Fusion-versus-best-branch differences use paired participant predictions and
   confidence intervals.
8. Failed folds, excluded participants, runtime deviations, or checkpoint
   substitutions are disclosed.

## Current run identity

- Full run: `hst-2848635e44aa841b0da7`
- Launch: `launch-462fd56a03eb4019b530dc3aa7d3d730`
- Accepted physical batch: 8
- Gradient accumulation: 1
- Mixed precision: disabled
- Planned optimizer updates: 718,000
- Conservative runtime estimate: 139.3 GPU-hours
- Frozen ceiling: 168 GPU-hours

These values describe execution, not model performance.

