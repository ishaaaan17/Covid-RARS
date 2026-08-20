# DNDT & DNDF Core — Ubuntu / Linux Execution Runbook

Target architecture: `DeepNeuralDecisionTree` / `DecisionTree` /
`DeepNeuralDecisionForest` in `app/models/deep_trees.py`.
Environment: Ubuntu 20.04 / 22.04 LTS (x86_64, CUDA >= 11.8).
Repository working directory: `Covid-RARS-main`.

This covers the same core module documented in
`DNDT_DNDF_CORE_EXECUTION_FREEZE.md`. If your checkout instead has
`models/dndf_model.py` / `pipelines/train_dndf_pipeline.py`, use
`DNDF_TABULAR_UBUNTU_RUNBOOK.md` — the two are not run the same way.

## 1. System requirements

- Python `>= 3.9` (recommended 3.10 or 3.11)
- PyTorch `>= 2.0.0` with CUDA support
- `scikit-learn >= 1.2.0`, `pandas >= 1.5.0`, `numpy >= 1.23.0`
- `scipy >= 1.10.0`, `tqdm >= 4.65.0`

## 2. Environment provisioning

Conda/mamba:

```bash
conda create -n covid-audio python=3.10 -y
conda activate covid-audio
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn pandas numpy scipy tqdm
```

Or plain virtualenv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn pandas numpy scipy tqdm
```

Health check:

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## 3. Pre-flight verification and smoke testing

Syntax/import check:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 -m py_compile app/models/deep_trees.py scripts/run_dndt_dndf.py tests/test_deep_trees.py
```

Unit smoke test (shapes and backprop gradients):

```bash
python3 tests/test_deep_trees.py
```

Expect: `[SUCCESS] All shape, forward, and backward gradient assertions passed!`
Do not proceed to a full run if this fails — the assertion in
`DeepNeuralDecisionTree` (`in_features <= 12`) and the shape contracts
between the feature-selection stage and the model are exactly what this test
is meant to catch early.

Synthetic end-to-end dry run:

```bash
python3 scripts/run_dndt_dndf.py --dry-run --output_dir data/outputs/metrics
```

Expect a `[DRY-RUN]` banner, per-model AUROC/AUPRC lines for `[DNDT]` and
`[DNDF]`, and a saved `dndt_dndf_dryrun_metrics.csv` under
`data/outputs/metrics/`. Confirm this file exists and is non-empty before
trusting a full run's output path.

Before running any of the above for a result you intend to report, verify
the three source files against the hashes in
`DNDT_DNDF_CORE_EXECUTION_FREEZE.md`:

```bash
sha256sum app/models/deep_trees.py scripts/run_dndt_dndf.py configs/dndt_dndf_reliability.json
```

(Note the freeze doc flags a formatting issue in the recorded hash strings —
resolve that before treating the comparison as authoritative.)

## 4. Execution: full acoustic feature benchmark

Standard stratified 5-fold evaluation across all configured feature
representations (`opensmile_compare_is10`, `beats`, `panns`):

```bash
python3 scripts/run_dndt_dndf.py \
    --config configs/dndt_dndf_reliability.json \
    --output_dir data/outputs/metrics \
    2>&1 | tee data/outputs/metrics/dndt_dndf_benchmark.log
```

Detached background execution (survives SSH disconnect):

```bash
nohup python3 scripts/run_dndt_dndf.py \
    --config configs/dndt_dndf_reliability.json \
    --output_dir data/outputs/metrics \
    > data/outputs/metrics/dndt_dndf_run.log 2>&1 &
disown

tail -f data/outputs/metrics/dndt_dndf_run.log
```

## 5. Post-execution artifact validation

Expected outputs under `data/outputs/metrics/`:

- `dndt_dndf_paper_comparable_cv_metrics.csv` — fold-level and aggregate
  AUROC / AUPRC for both DNDT and DNDF.
- `dndt_dndf_bootstrap_ci.csv` — 1,000-round non-parametric bootstrap 95%
  confidence intervals (per `bootstrap_rounds` in the config).
- `dndt_dndf_feature_selection_record.csv` — retained top-ranked acoustic
  feature indices per fold (per the `select_k_best` / ANOVA F-test setting;
  `k=8` for DNDT, `k=32` for DNDF).

```bash
ls -lh data/outputs/metrics/dndt_dndf*
```

Confirm row counts in the CV metrics CSV match `n_splits x len(modalities) x
len(feature_representations)` before treating a run as complete — a
short file usually means the run was interrupted partway through the
modality/representation sweep.

## 6. Downstream asset and manuscript rebuild

```bash
python3 manuscripts/iatmsi_2027/submission_final/scripts/build_assets.py
python3 manuscripts/iatmsi_2027/submission_final/scripts/audit_submission.py
```

Run the audit script after any rebuild and before citing generated figures
or tables in a submission draft; it is the last check between "the pipeline
ran" and "the manuscript assets reflect the frozen config."

## Monitoring and recovery

This runbook, as currently specified, does not document a checkpoint/resume
mechanism, a progress-receipt file, or a manual promotion gate for
`scripts/run_dndt_dndf.py` — unlike the HST controller's per-minute progress
coordinates and checksum-gated resume. Until that is added or confirmed
present in the script, treat a full benchmark run the same way as the
DNDF-Tabular pipeline: run it under `nohup`/`disown` or a persistent session
(`tmux`/`screen`), and re-run from the start if the process is killed
mid-sweep. If `run_dndt_dndf.py` does implement its own resume logic,
document the exact resume invocation here before relying on it operationally.
