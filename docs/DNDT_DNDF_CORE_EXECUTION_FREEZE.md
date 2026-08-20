# DNDT / DNDF Core — Immutable Source, Architecture & Execution Freeze

Status: implementation freeze for pilot review. This document does not record
an accepted run or any result. It covers the in-repo core module produced by
`generate_dndt_dndf_freeze_and_runbook.ps1` — distinct from the
`models/dndf_model.py` pipeline covered by `DNDF_TABULAR_EXECUTION_FREEZE.md`.
The two implementations are separate: different classes, different file
layout, different hyperparameters. Do not cite one freeze while running the
other's code.

## Provenance and scope

Status: in-repo original implementation. Not vendored, not a Git submodule.

| File | Role |
|---|---|
| `app/models/deep_trees.py` | Model definitions: `DeepNeuralDecisionTree`, `DecisionTree`, `DeepNeuralDecisionForest` |
| `scripts/run_dndt_dndf.py` | Training/eval driver (k-fold CV, bootstrap CI, calibration) |
| `configs/dndt_dndf_reliability.json` | Hyperparameter config for the benchmark run |

## Design lineage and mathematical specification

Per the module's own docstrings and the foundational publications it
implements:

- **DeepNeuralDecisionTree (DNDT)** — Yang, Morillo & Hospedales (2018),
  "Deep Neural Decision Trees" (arXiv:1806.06988).
  - Soft per-feature binning via cutpoints, joint routing via a tensor outer
    product across features, differentiable leaf layer.
  - Soft routing formulation:

    P(leaf_i | x) = Softmax( (W_i x + b_i) / tau )

  - Constraint: `DeepNeuralDecisionTree` asserts `in_features <= 12`, to
    prevent the `(c+1)^D` leaf-count explosion inherent to per-feature
    cutpoint binning.

- **DecisionTree / DeepNeuralDecisionForest (DNDF)** — Kontschieder et al.
  (2015), "Deep Neural Decision Forests" (ICCV 2015).
  - Oblique soft split via a linear decision layer; ensemble of independent
    trees averaged at the output:

    P(y=c | x) = (1/M) * sum_m sum_i P_m(leaf_i | x) * pi_{m,i,c}

  - `DecisionTree` leaf count is `2^depth`; internal node count is
    `2^depth - 1`.
  - Ensemble aggregation: unweighted averaging across `num_trees` — no
    boosting.

The two classes implement different routing mechanisms (per-feature cutpoint
binning for `DeepNeuralDecisionTree` vs. an oblique linear split for
`DecisionTree`) despite the shared "deep neural decision tree" lineage; they
are not interchangeable and are configured independently below.

## Pinned hyperparameters (`configs/dndt_dndf_reliability.json`)

| Component | Parameter | Frozen value |
|---|---|---|
| Feature reduction | `method` | `select_k_best` (ANOVA F-test) |
| Feature reduction | `k` (DNDT input) | `8` |
| Feature reduction | `k` (DNDF input) | `32` |
| DNDT | `num_cutpoints` | `1` |
| DNDT | `temperature` (tau) | `1.0` |
| DNDT | `lr` | `0.01` |
| DNDT | `epochs` | `60` |
| DNDF | `num_trees` (M) | `12` |
| DNDF | `depth` (D) | `4` |
| DNDF | `lr` | `0.003` |
| DNDF | `epochs` | `80` |
| Validation | `n_splits` | `5` |
| Validation | `bootstrap_rounds` | `1000` |
| Validation | `leakage_safe` | `true` |

Note the `k=8` feature-selection cap for DNDT is required by, and must stay
under, the module's `in_features <= 12` assertion; do not raise `k` for the
DNDT branch without also relaxing (and re-justifying) that assertion.

Modalities: `cough`, `speech`, `breath`.
Feature representations: `opensmile_compare_is10`, `beats`, `panns`.

## File-content integrity anchors (SHA-256)

Recorded by the original author via `Get-FileHash -Algorithm SHA256` against
the anchor working copy (2026-08-20). These are carried over as documented
and have not been independently recomputed here, since the source of
`app/models/deep_trees.py`, `scripts/run_dndt_dndf.py`, and
`configs/dndt_dndf_reliability.json` was not supplied alongside this freeze —
recompute and diff against the table below before treating any run as
matching this freeze:

| File | SHA-256 digest |
|---|---|
| `app/models/deep_trees.py` | `3488EE60178F844C6993FBA601EAE21F822781774F402E1C2865B4704227F462` |
| `scripts/run_dndt_dndf.py` | `0461E2123B4E311DFE5448A774807742F465AA23C0F9E5ADF50A3A7260CDE4E7` |
| `configs/dndt_dndf_reliability.json` | `3006885DB096F6A6C3B285F95B7729918970266FE2990DEEC5DE300167F927CA` |

Each digest above is longer than a standard 64-hex-character SHA-256
(65–66 characters) as transcribed from the original draft — flag this for
correction against `Get-FileHash`'s actual output before relying on it as an
integrity gate.

## Checkpoint and initialization policy

- Cold-start initialization only for both architectures: leaf weights via
  `nn.Parameter(torch.randn(...) * 0.01)`, cutpoints via a `linspace`-based
  scheme, and standard `nn.Linear` initialization elsewhere.
- Zero external checkpoints. No pretrained weights and no warm-start
  checkpoint are loaded by either class.

## What is not yet specified in this freeze

Unlike the HST freeze, this document (as drafted) does not specify: data
provenance and participant-safety rules for the Coswara/COUGHVID/etc. cohorts
feeding `opensmile_compare_is10`/`beats`/`panns` features, the train/
validation/test split policy referenced by `n_splits: 5` and `leakage_safe:
true` beyond those two flags, the calibration and operating-point metrics
`run_dndt_dndf.py` actually reports, and any pilot/manual-gate promotion
process analogous to the HST controller. Fill these in from
`scripts/run_dndt_dndf.py` and `app/models/deep_trees.py` directly before
treating a run under this freeze as reportable evidence.
