# HST Ubuntu Execution Runbook

This runbook is for the Ubuntu workstation with an NVIDIA T1000 8GB GPU and
19GB host RAM. The notebook `notebooks/09_HST_RELIABILITY_E2E.ipynb` is the
preferred controller. It launches detached processes, records status every
minute, and resumes checksummed stages after SSH or notebook interruption.

The performance objectives are AUROC above 0.868 for cough, 0.842 for breath,
0.891 for speech, and 0.897 for complete-case cough+speech fusion. They are not guaranteed.
Validation data may select an epoch, threshold, or declared secondary fusion
rule; test and external labels must never select or stop a model.

## Workspace mapping

The development checkout on Windows is
`C:\Users\nhnis\Desktop\Covid-RARS\covid_audio_btp`. Windows is for code review and Git synchronization. The execution checkout is
`/home/covid/Desktop/Covid-19-BTP/covid_audio_btp`. Ubuntu is the scientific execution environment because it contains the datasets, approved artifacts, and NVIDIA GPU. Synchronize code through the reviewed Git branch; do not replace Ubuntu data or generated evidence with files from Windows.

After opening the notebook, confirm that `PROJECT_ROOT` resolves to the Ubuntu
execution checkout above. All notebook paths are derived from that package root;
the approval scripts receive its parent repository directory where required.

## 1. Install and verify

```bash
cd /home/covid/Desktop/Covid-19-BTP
git submodule update --init --recursive HST
cd /home/covid/Desktop/Covid-19-BTP/covid_audio_btp
source .venv/bin/activate
python -m pip install -r requirements-hst.txt
python -m pip install -e .
python - <<'PY'
from pathlib import Path

from covid_audio_btp.hst_reliability import prepare_hst_prerequisites

result = prepare_hst_prerequisites(
    config_path=Path("configs/hst_reliability.json"),
    project_root=Path.cwd(),
)
print(result)
PY
python -m pytest tests/test_hst_*.py -q -W error
```

`prepare_hst_prerequisites` initializes and verifies the pinned HST source and
downloads both official ImageNet initialization files to
`covid_audio_btp/.cache/hst/checkpoints`, the same directory used by production
preflight and the checkpoint tests. It also creates the HST COUGHVID metadata
input by joining the fixed processed cohort one-to-one to released v3 metadata
on UUID. The prerequisite fails if a UUID is missing, duplicated, has no
`status_SSL`, or disagrees with the cohort's legacy binary label, and records
the SHA-256 of both source tables. A fresh checkout must not place or copy
checkpoint files into `HST/model/imagenet_weights`; that submodule path is not a
tracked checkpoint store.

The shared spectrogram cache is versioned by preprocessing settings and the
preprocessing implementation. A code revision that changes preprocessing must
bump that implementation version rather than reuse an unversioned cache.

The frozen runtime setting is `max_concurrent_gpu_jobs: 1`. Do not launch a
second GPU job while the HST controller is active. CPU-only inspection is safe,
but comparator fitting is serialized by the pipeline and should not be started
manually in parallel.

The controller locks the first device exposed by `CUDA_VISIBLE_DEVICES`. On the
single-T1000 workstation, leave that variable unset unless device visibility is
being managed deliberately; never change it while a run is active.

Run every launch under the dedicated `covid` Unix account. The CUDA-UUID lease
and content-derived run lease exclude competing processes and repository clones
owned by that execution account. They do not claim cross-account exclusion;
mixed-account execution on the same host is outside this run contract and must
be prevented administratively.

The frozen projected-runtime ceiling is 168 serial GPU hours. The pilot
candidate reports its measured seconds per optimizer update and a conservative
workload computed from all contract-eligible Coswara participants. It counts 25
cough, 15 speech, and 10 breath jobs at 100 epochs and applies a frozen 1.5
end-to-end overhead multiplier for evaluation, checkpointing, and orchestration.
Treat the resulting serial-GPU projection as a capacity decision, not an ETA
guarantee. Do not promote an over-ceiling candidate.

Check that the two official ImageNet initialization files remain available in
`.cache/hst/checkpoints`. The controller verifies their byte sizes and SHA-256
values before use. Do not substitute a task-trained checkpoint.

## 2. Smoke and resource pilot

Run all notebook cells through the pilot launch. This executes a real
HST-Small smoke fit for two epochs, then a capped HST-Base resource pilot. The pilot checks
preprocessing throughput, physical batch size, gradient accumulation, AMP
agreement, skipped optimizer updates, host RAM, GPU memory, and estimated full
runtime. It does not open test or external outcomes.

### Manual gate 1

Generate the pilot candidate with `75_prepare_hst_acceptance.py`. Review the
data-contract, environment, and resource-pilot receipts. Manually promote only
the reviewed proposal to `reports/hst/accepted_freezes.json`, with reviewer and
UTC approval fields. Copy the exact reviewed Ubuntu environment audit to
`configs/hst_environment_lock.approved.json`, commit that small lock file, and
make it read-only. Do not approve a pilot with invalid AMP agreement, unstable
memory use, non-finite output, an effective batch size other than eight, or a
projected serial runtime above 168 hours.

## 3. Freeze manifests and comparator recipe

Continue the notebook through `manifests`. Full mode creates participant-safe
internal, calendar-mixed, early-to-late, late-to-early, external cough, and
`aligned_comparator` manifests. It also freezes the complete-case modality
cohorts and content hashes. The full content-derived run ID shown here is the
same run ID that must be used for every later resume.

The data-contract receipt must also contain
`contracts/coughvid_source_provenance.csv`. Review it before acceptance: the
input level must be `derived_processed_csv`, raw-release membership
reconstruction must be false, the analysis unit must be `recording_uuid`, and
subject linkage must be unavailable. The external `participant_key` is only a
schema-compatible recording proxy.

### Manual gate 2

Use `76_prepare_hst_comparator_approval.py approval-record` from the notebook.
The hardened command always resolves the canonical `aligned_comparator`
manifest from the authenticated manifests receipt; the operator cannot choose a
different manifest at the command line.
Review that the proposal binds:

- the authenticated `manifests` stage receipt;
- the exact `aligned_comparator` manifest;
- the complete 10,147-column ComParE+IS10 table, not a preselected table;
- top-800 feature selection fitted inside each training fold;
- LightGBM ranking and the frozen LightGBM, SVC, CatBoost, and XGBoost bank;
- validation-only candidate selection, source code, dependency lock, Git
  identity, random state, and environment lock.

Manually promote the approved proposal to
`configs/hst_compare_is10_approval.approved.json`, commit it, and make it
read-only. Then run the script's `accepted-freezes` command, review its proposal,
and manually promote it to
`configs/hst_comparator_accepted_freezes.approved.json`. The initial file has no
accepted generation and therefore cannot authenticate results that do not yet
exist.

## 4. Generate and accept comparator evidence

Continue through `aligned_comparator`. The first pass fits the approved
comparator recipe and writes a content-addressed generation. It intentionally
ends at the message `manual comparator generation acceptance required`; this is
a safety gate, not a failed scientific fit.

### Manual gate 3

Run `77_prepare_hst_comparator_generation_acceptance.py` from the notebook. It
verifies `current.json`, the generation manifest, every result table, and every
model bundle before producing a candidate. Manually replace the canonical
comparator accepted-freezes file with only the reviewed proposal, commit it, and
make it read-only.

Resume the same run ID. The stage must reuse and authenticate the exact existing
generation. A new generation ID, changed checksum, dirty approval file, or
different Git identity is fatal.

The accepted comparator generation contains Python-serialized estimator
bundles. Treat those files as trusted executable artifacts, not passive data.
They are loaded only after the manually accepted generation, repository,
source-code recipe, environment, file size, and SHA-256 bindings all verify.
Never point this workflow at an externally supplied or unreviewed bundle.

The required order is therefore: create and promote the recipe approval, create
and promote the initial accepted-freezes document, run the first comparator
pass, manually accept that generated evidence, then resume the same full run ID.
Candidate scripts never edit canonical approvals themselves.

## 5. Final scientific execution

Run the final notebook cell through `evidence_pack`. Checksummed completed stages
are reused. Remaining stages execute HST-Base internal folds, split-policy
contrasts, reverse temporal evaluation, COUGHVID cough-only external transfer,
aligned fusion, bootstrap confidence intervals, paired-only DeLong tests,
calibration, operating points, decision curves, Grad-CAM, and the final evidence
manifest.

Bootstrap comparisons report effect estimates and confidence intervals only;
ordinary bootstrap sign proportions are not reported as formal p-values. Holm
adjustment applies only to a declared family of valid formal secondary tests.

The aligned comparator opens each held-out partition once for a locked,
prespecified model bank. Its primary endpoint is selected using validation AUROC,
validation AUPRC, and model name only; the remaining frozen model-bank test rows
are secondary evidence and cannot alter the endpoint or threshold.

The final evidence also contains a sensitivity-execution registry. Raw
COUGHVID `status` labels are evaluated by relabeling the same frozen external
probabilities and reusing source-validation thresholds; no target label affects
training or selection. If that supervised overlap is empty or single-class, the
registry and metric table record a nonblocking skip. The event/SNR sensitivity
and released-code model run remain explicitly deferred and must not be reported
as completed. The deterministic released-code renderer is available only for a
future separately accepted extension.

Interpret the external endpoint narrowly. The configured `status_SSL` field is
a semi-supervised label rather than RT-PCR-confirmed ground truth, and the
processed COUGHVID cohort does not expose verified subject linkage. Report the
result as cough-recording transportability and label agreement, not full-release
reconstruction, participant-level transfer, or clinical diagnostic validation.
For the same reason, external `status_SSL` rows are omitted from clinical
fixed-sensitivity and decision-curve outputs. External discrimination and
calibration-agreement summaries remain reportable with that limitation.

Only a verified evidence pack may update `reports/hst/latest.json`. Record the
published run ID and evidence-manifest SHA-256 before interpreting results. Never
rerun or change settings because test or external results missed an objective.

## Monitoring and recovery

While a launch is active, the notebook prints three independent progress
coordinates every minute:

- integrity-checked pipeline stages completed out of the requested stage target;
- durable training-equivalent jobs out of 50, including only the checkpointed
  fraction of the current 100-epoch job;
- the exact active job ID, stage, fold, seed, modality, and protocol; before the
  first checkpoint it explicitly reports zero durable fractional progress;
- after a checkpoint, the completed epoch, batch boundary, checkpoint reason,
  generation, relative path, and full checkpoint SHA-256.

These values come from self-hashed stage/job receipts and
`training_progress.json`, which is emitted only after the transactional writer
commits an optimizer-safe checkpoint. It does not expose validation, test, or
external metrics. Progress can therefore be used to monitor execution, but not
to adapt the experiment. A hard kill between committing a new checkpoint and
updating the progress record falls back to the generation named by the last
durable progress record. That generation remains pinned across later checkpoint
writes until a newer progress record is published. The displayed value may be
conservative but cannot claim uncheckpointed work.

For runtime monitoring, a completed stage or job is counted only when every
declared output still exists and its full SHA-256 agrees with the receipt. Hash
results are memoized only while file size, modification time, and change time
remain unchanged, avoiding repeated reads of immutable model checkpoints.
Publication evidence independently performs full artifact verification.
Receipt self-hashes and these runtime checks detect
inconsistency or corruption; they are not a cryptographic authentication claim
against an attacker who can rewrite the local run directory.

The notebook continues independently of the browser connection. To inspect a
launch after reconnecting, first return to the Ubuntu checkout and activate the
same environment:

```bash
cd /home/covid/Desktop/Covid-19-BTP/covid_audio_btp
source .venv/bin/activate
ls -1t reports/hst/launches/launch-*.json | head
```

Use the launch ID from the status filename with:

```bash
python scripts/72_run_hst_reliability.py --project-root . --status-id LAUNCH_ID
tail -f logs/hst/LAUNCH_ID.log
```

If the SSH connection or browser closes while the status remains `running`, do
not launch another process; reconnect and keep monitoring the existing launch.
If a process has stopped, rerun only the relevant notebook launch cell with the
same full run ID shown in its status receipt. A notebook-kernel restart does not
invalidate the detached process or its receipts. Do not delete stage receipts,
evaluation-registry records, checkpoints, or the comparator generation. The
controller rejects incomplete or tampered artifacts and resumes only from
an optimizer-safe checkpoint boundary.
