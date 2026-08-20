# DNDF-Tabular Reliability Execution Freeze

Status: implementation freeze for pilot review. This document does not record
an accepted run or any result. It covers the deployment produced by
`deploy_dndf_full_repo_suite.ps1`, referred to here as the **DNDF-Tabular**
pipeline to distinguish it from the separate in-repo DNDT/DNDF core module
covered by `DNDT_DNDF_CORE_EXECUTION_FREEZE.md`.

## Scientific objective

The DNDF-Tabular branch tests whether a from-scratch Deep Neural Decision
Forest, trained directly on pre-extracted `.npy` acoustic feature arrays, can
discriminate COVID-positive from COVID-negative respiratory recordings under
stratified k-fold cross-validation. No AUROC/AUPRC targets are pinned in the
source config; the numbers reported by the pipeline are descriptive fold and
out-of-fold estimates, not a pass/fail gate against a pre-registered target.

## Immutable source and initialization

- Implementation is in-repo, first-party code — not vendored, not a
  submodule.
- Files and role:

| File | Role | SHA-256 (as generated) |
|---|---|---|
| `models/dndf_model.py` | `DNDT` (per-tree soft router) and `DNDF` (ensemble + optional MLP feature extractor) `nn.Module` definitions | `b6251aab8d6eafb71c85f95098871e09fe8f35c496f450b97033f1e1d7c082c4` |
| `evaluation/metrics_engine.py` | `compute_classification_metrics`, `bootstrap_confidence_intervals` | `f5cd6962d46870ec4a2f5acf40ee30fd40907785494b74c0b18cfd41bef45e0e` |
| `pipelines/train_dndf_pipeline.py` | Data loading, stratified 5-fold training/eval driver, CLI entrypoint | `670e24a85513a9cee06adc5097535f5346f1bcfc76cac89f6178cde09d19b0b2` |
| `configs/dndf_benchmark.json` | Frozen hyperparameter and dataset-path manifest | `573bfbe6c10bd49b0510abbf1f9c07d2d029cbbb87c5744d79a3f1b7db74efd8` |

  Hashes above were computed directly against the file contents this
  documentation was generated from (`sha256sum`) and should be recomputed
  and compared against the working copy before any run is treated as
  matching this freeze; they are a documentation-time anchor, not a
  runtime-enforced gate — the pipeline itself performs no integrity check.

- Cold-start initialization only. `DNDF.__init__` builds an optional
  `Linear -> BatchNorm1d -> ReLU -> Dropout(0.2)` (x2) feature extractor
  (`hidden_dim=128` by default) feeding `num_trees` independent `DNDT`
  routers; each `DNDT` uses `kaiming_uniform_` for its routing weights and
  `N(0, 0.1)` for its leaf-distribution logits. No pretrained or warm-start
  checkpoint is loaded anywhere in this pipeline.

## Architecture

- **DNDT** (per tree): a single linear routing layer projects the input
  feature vector to `2**depth` leaf logits, softmax-normalized to a leaf
  distribution; each leaf holds its own softmax class distribution over
  `num_classes`; the tree's output is the leaf-probability-weighted mixture
  of per-leaf class distributions. `temperature` divides the routing logits
  before the softmax.
- **DNDF** (ensemble): `num_trees` independent `DNDT` instances (default 15,
  depth 5) share one optional MLP feature extractor and are combined by an
  unweighted mean of their class-probability outputs — no boosting, no
  learned tree weighting.
- Constraint: unlike a from-arXiv reference DNDT, this implementation places
  no explicit bound on `num_features` per tree, because routing is a full
  linear projection rather than per-feature binning; the practical limit is
  the size of `feature_weights` (`num_features x 2**depth`).

## Data and labels

- Input contract: each `dataset_dir` must contain `cough_X_features_np.npy`
  (float array, `[n_samples, n_features]`) and
  `cough_y_features_np.npy` (integer or one-hot labels). Missing either file
  is a fatal `FileNotFoundError` — there is no fallback discovery.
- Label handling: a `y` array with more than one column is reduced with
  `argmax` (assumed one-hot); otherwise it is raveled to a 1-D integer
  vector. The pipeline assumes exactly two classes end-to-end (`num_classes`
  is hardcoded to `2` at model construction).
- Configured dataset roots (`configs/dndf_benchmark.json`): `Extracted
  Features/Coswara`, `Extracted Features/COUGHVID`, `Extracted
  Features/Cambridge/Task 1`, `Extracted Features/Cambridge/Task 2`,
  `Extracted Features/NOCOCODA and Virufy`, `Extracted Features/Virufy`. The
  CLI (`--dataset_dir`) runs one dataset per invocation; the config list is a
  manifest of intended runs, not an automatic multi-dataset loop — the
  driver script does not iterate it.
- No participant-level split enforcement, no cross-dataset external-transfer
  design, and no duplicate-content or provenance checking is implemented in
  this pipeline. `StratifiedKFold(shuffle=True, random_state=42)` splits at
  the row (recording) level, so if a participant contributes more than one
  row to `X`/`y`, that participant can appear in both the train and
  validation fold of a given split.

## Preprocessing

- None performed in this pipeline. Feature extraction into the `.npy` files
  is an external, upstream step not covered by these three files. The only
  in-pipeline transform is per-fold `StandardScaler`, fit on the training
  split only and applied to both splits (no leakage from validation into
  the scaler).

## Training

- Optimizer: Adam, `lr=0.001` (config) / CLI default `0.001`, `weight_decay=1e-4`
  hardcoded in `run_cross_validation` (not exposed as a CLI flag; the
  config's `weight_decay: 0.0001` value is documentation only and is not
  read by the current driver).
- Loss: `NLLLoss` over `log(clamp(probs, 1e-7, 1-1e-7))`. Because `DNDF`
  already outputs class probabilities (softmax-mixed), this is
  cross-entropy computed via an explicit log rather than `nn.CrossEntropyLoss`
  on logits.
- Batch size: 32 (config and CLI default), `DataLoader(shuffle=True)` for
  train, `shuffle=False` for validation.
- Epochs: 50 (config and CLI default), no early stopping and no learning-rate
  schedule — the loop always runs the full epoch count.
- Per-fold model selection: the epoch whose validation AUROC is highest
  is kept (`best_auroc` / `best_probs`); ties keep the earlier epoch because
  the comparison is strict `>`. There is no checkpoint file written to disk
  — "best" exists only in memory for the duration of the fold.
- No resumability. The script has no checkpoint/resume mechanism; a killed
  process must restart the whole `run_cross_validation` call, including
  already-completed folds.
- No GPU/AMP negotiation logic. `device` is `"cuda"` if
  `torch.cuda.is_available()` else `"cpu"`, passed straight through; there is
  no mixed-precision path, no gradient clipping, and no batch/accumulation
  fallback ladder.

## Evaluation

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` at the row
  level (see Data and labels for the participant-leakage caveat).
- Out-of-fold (OOF) probabilities are assembled from each fold's
  best-validation-AUROC epoch and used for the headline metrics — this
  reuses the same data that selected the epoch, so the reported OOF numbers
  are a validation-selected estimate, not a held-out test estimate distinct
  from model selection.
- Metrics (`evaluation/metrics_engine.py`): AUROC, AUPRC, sensitivity,
  specificity, accuracy at a fixed 0.5 probability threshold, and Brier
  score. No ECE, no calibration curve, no DeLong test, no Grad-CAM
  (inapplicable — the model consumes tabular feature vectors, not images).
- Uncertainty: 1,000-replicate non-parametric percentile bootstrap, seed 42,
  95% CI, computed once over the pooled OOF predictions
  (`bootstrap_confidence_intervals`). A bootstrap draw with a single-class
  resample is silently dropped from the percentile calculation rather than
  raising or recording a skip reason.
- No external-dataset transfer evaluation is implemented — the `datasets`
  list in the config is a set of independent candidate inputs, not a
  source-to-target transfer design.

## What this pipeline does not (yet) provide, relative to the HST freeze style

Documented explicitly so this freeze cannot be mistaken for a stronger
guarantee than the code gives:

- No manifest/checksum-gated data contract, no participant-qualification
  ledger, no fail-closed cohort join.
- No manual promotion gates, no `accepted_freezes.json`-style evidence
  pack, no environment lock file.
- No resumable/checkpointed training; no runtime progress receipts.
- No calibration (ECE), ROC/DeLong significance testing, or decision-curve
  analysis.
- No cross-dataset external transportability endpoint, despite multiple
  dataset roots being configured.

## Pilot acceptance

There is no automated pilot/gate mechanism in this codebase. Before treating
any run as a reportable result, a reviewer should manually confirm: (1) the
four file hashes above still match the working copy, (2) the `--dataset_dir`
used matches the intended entry in `configs/dndf_benchmark.json`, and (3) the
reported OOF metrics are understood as validation-selected, not held-out-test,
estimates.
