# DNDF-Tabular Ubuntu Execution Runbook

Covers the pipeline scaffolded by `deploy_dndf_full_repo_suite.ps1`:
`models/dndf_model.py`, `evaluation/metrics_engine.py`,
`pipelines/train_dndf_pipeline.py`, `configs/dndf_benchmark.json`. The
Windows-authored deploy script only writes source files; all training and
evaluation runs happen on the Ubuntu execution host, consistent with the
Windows-for-code / Ubuntu-for-execution split used elsewhere in this repo.

The pipeline has **no target AUROC gate, no manifest freeze, and no
resumable checkpointing**. Treat every invocation as a single, non-resumable
job that must run to completion.

## Workspace mapping

- Windows checkout: wherever `deploy_dndf_full_repo_suite.ps1` was run
  (creates `models/`, `evaluation/`, `pipelines/`, `configs/`, `docs/`
  under the PowerShell working directory at run time).
- Ubuntu execution root: set an environment variable pointing at the
  synchronized checkout, e.g. `DNDF_TABULAR_PROJECT_ROOT`, and `cd` there
  before running anything below. Synchronize source through Git, not by
  copying generated artifacts (feature `.npy` files, logs, metrics CSVs)
  from Windows to Ubuntu or back.

## 1. Install and verify

```bash
export DNDF_TABULAR_PROJECT_ROOT="${DNDF_TABULAR_PROJECT_ROOT:-$HOME/Desktop/Covid-RARS}"
cd "$DNDF_TABULAR_PROJECT_ROOT"
python3 -m venv .venv-dndf
source .venv-dndf/bin/activate
pip install --upgrade pip
pip install torch scikit-learn numpy
```

`torch` should be installed with the CUDA build matching the workstation's
driver if a GPU is present (`pip install torch --index-url
https://download.pytorch.org/whl/<cuXXX>`); the pipeline falls back to CPU
automatically via `torch.cuda.is_available()`.

Verify the four source files against the hashes recorded in
`DNDF_TABULAR_EXECUTION_FREEZE.md` before running anything you intend to
report:

```bash
sha256sum models/dndf_model.py evaluation/metrics_engine.py \
  pipelines/train_dndf_pipeline.py configs/dndf_benchmark.json
```

Confirm the feature arrays exist for the dataset you intend to run, e.g.:

```bash
ls -lh "Extracted Features/Coswara/cough_X_features_np.npy" \
       "Extracted Features/Coswara/cough_y_features_np.npy"
```

The pipeline raises `FileNotFoundError` immediately if either file is
missing — there is no partial-run or auto-download fallback.

## 2. Smoke test

Before a full 50-epoch, 5-fold run, do a fast correctness check with a
reduced epoch count on the smallest available dataset:

```bash
python3 pipelines/train_dndf_pipeline.py \
  --dataset_dir "Extracted Features/Coswara" \
  --epochs 2
```

Confirm fold-by-fold AUROC/AUPRC/sensitivity/specificity print without
errors and that the final "Overall Out-Of-Fold Evaluation" block completes.
This exercises the full code path (model, loss, both metrics functions,
bootstrap CI) without committing to a full run.

## 3. Full execution

```bash
python3 pipelines/train_dndf_pipeline.py \
  --dataset_dir "Extracted Features/Coswara" \
  --num_trees 15 \
  --depth 5 \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001 \
  2>&1 | tee "logs/dndf_tabular_coswara_$(date +%Y%m%dT%H%M%S).log"
```

Repeat once per dataset root listed in `configs/dndf_benchmark.json` —
the driver does not iterate the list itself; each dataset is a separate
invocation with its own `--dataset_dir`.

For a detached run that survives an SSH disconnect:

```bash
nohup python3 pipelines/train_dndf_pipeline.py \
  --dataset_dir "Extracted Features/Coswara" \
  > "logs/dndf_tabular_coswara_$(date +%Y%m%dT%H%M%S).log" 2>&1 &
disown
```

Reconnect and follow with `tail -f logs/dndf_tabular_coswara_<timestamp>.log`.

## 4. Monitoring and recovery

- There is no progress file, no checkpoint, and no resumable state. The log
  stream (stdout) is the only progress signal: per-fold "Best AUROC" lines,
  then the final OOF block.
- If the process is killed mid-run (SSH drop without `nohup`/`disown`, OOM
  kill, manual interrupt), the entire invocation must be re-run from
  fold 1 — no partial credit, no checkpoint to resume from. `nohup` +
  `disown` (or a persistent session tool such as `tmux`/`screen`) is
  strongly recommended for any run expected to exceed a single interactive
  session.
- GPU memory: `num_trees=15`, `depth=5` gives `2**5 = 32` leaves per tree;
  memory scales with `hidden_dim x num_trees x 2**depth` for the leaf
  parameters plus the feature-extractor MLP. On a memory-constrained GPU,
  reduce `--batch_size` first; there is no AMP path in this pipeline to
  fall back to.
- To confirm you are on the frozen source before re-running after any edit,
  recompute the four hashes and diff against `DNDF_TABULAR_EXECUTION_FREEZE.md`.

## 5. Output artifacts

This driver currently prints results to stdout only — it does not write a
metrics CSV, a bootstrap-CI file, or a run manifest to disk. Capture the
`tee`'d log as the run's evidence record, or extend
`pipelines/train_dndf_pipeline.py` to serialize `overall_metrics` and `ci`
(e.g. to JSON) before treating any run as reportable, since there is
currently no persisted machine-readable output to audit after the process
exits.
