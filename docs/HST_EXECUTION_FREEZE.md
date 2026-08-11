# HST Integrated Reliability Execution Freeze

Status: implementation freeze for pilot review. This document does not record
an accepted pilot or any result.

## Scientific objective

The HST branch tests whether a released hierarchical spectrogram transformer,
used inside the same participant-level reliability ladder as the existing
pipeline, can improve internal discrimination without hiding temporal or
cross-dataset failure. The engineering objectives are participant AUROC above
0.868 for cough, 0.842 for breath, 0.891 for speech, and 0.897 for the complete-
case cough+speech fusion. These are targets, not guaranteed outcomes, and the
held-out test set cannot be used as a stopping rule.

## Immutable source and initialization

- HST repository: `https://github.com/icon-lab/HST.git`
- Source commit: `7f94ad81e392da856c7aac6d364d036c28e26c32`
- Official model source SHA-256:
  `5f9503df584d3a427722e0de5e1d52d1bbb79933f337181b7eeb65fcf9d2cc8f`
  (computed from the pinned Git blob, independent of checkout line endings)
- HST-Small ImageNet checkpoint SHA-256:
  `e7086d1b87d598120296b9a1b5f094c7587cb06f50bf609a4ca13badc95e3112`
- HST-Base ImageNet checkpoint SHA-256:
  `f39f001d5f8cd90cb78d45612486202a4ea280e23df0b2c1d6ce35d96b30cce4`
- The released classification head is replaced with a two-class head after
  strict checkpoint loading. Missing or unexpected non-head weights are fatal.
- HST-Small is used only for implementation smoke testing. HST-Base is the
  confirmatory architecture.

## Data and labels

- Coswara supplies development, validation, internal testing, temporal testing,
  and reverse-temporal testing for cough, breath, and speech.
- COUGHVID supplies cough-only external testing. It is not represented as an
  external validation of breath, speech, or multimodal fusion.
- Coswara uses the contributor as the analysis unit. Every Coswara participant
  and recording key is dataset-qualified, and a participant cannot occur in
  more than one split in a protocol/fold.
- COUGHVID membership is fixed by the checksum-pinned processed external cohort.
  A fail-closed prerequisite joins that cohort one-to-one to the released
  COUGHVID-v3 `metadata_compiled.csv` by processed `participant_id` = released
  `uuid`, verifies every legacy binary label against `status_SSL`, and records
  both upstream SHA-256 values. It does not expand the cohort to other release
  rows. The UUID is a recording identifier; because no verified contributor
  linkage is available, the external analysis unit is the recording UUID and
  the compatibility `participant_key` is marked as a recording proxy.
- Duplicate audio content hashes cannot cross split boundaries.
- Spectrogram-cache identity includes both the frozen preprocessing settings and
  an explicit preprocessing-implementation version. Recordings shorter than the
  silence-trimming frame are excluded by the unchanged duration threshold; they
  are never padded to make trimming or model ingestion succeed.
- Each cached tensor records the SHA-256 of the complete `.npy` artifact used by
  the training loader and a separate SHA-256 of its canonical float32 payload.
- COUGHVID primary labels are the explicitly selected released `status_SSL`
  field. Raw `status` and available expert `diagnosis_*` fields are retained for
  label-source auditing; `status` is also evaluated by relabeling the same frozen
  external probabilities and reusing each source-validation threshold. They are
  never an implicit fallback or a source of model, checkpoint, or threshold
  choices.
- `status_SSL` is a semi-supervised COUGHVID label, not RT-PCR-confirmed clinical
  ground truth. The external endpoint is therefore a transportability and label-
  agreement analysis, not clinical validation against a diagnostic reference.
- Training, validation, and testing use the exact same frozen eligibility table
  for HST and the aligned ComParE+IS10 comparator.

Sensitivity execution is explicit and auditable. The raw-`status` label
sensitivity runs in the primary end-to-end controller but is nonblocking: an
empty or single-class supervised overlap produces a skipped row rather than
invalidating the primary `status_SSL` estimate. The COUGHVID cough-event/SNR
analysis is deferred because the pinned official HST source contains no
checksum-verifiable event/SNR implementation. No replacement is invented or
described as paper-exact.

## Audio representation

Primary representation: `paper_logmel_224`.

- Mono audio at 22,050 Hz, resampled with `soxr_hq`.
- Silence trimming: 60 dB threshold, 2,205-sample frame, 1,102-sample hop.
- Recordings whose trimmed duration is not greater than 2 seconds are excluded.
- STFT: 2,048-point periodic Hann window, 2,048-sample window, 1,920-sample
  hop, no centering, power spectrum.
- Mel projection: 224 bands, 0 to 11,025 Hz, Slaney scale/normalization.
- Convert to dB relative to each recording maximum, clip to an 80 dB range,
  and map to `[0, 1]`.
- Bilinear antialiased resize to 224 by 224, replicate to three channels, then
  normalize each channel with mean 0.5 and standard deviation 0.5.
- The deterministic released linear-spectrogram renderer is implemented and
  tested, but its model run is a deferred optional extension. It is not part of
  the frozen 50-job run and cannot replace the primary Mel representation after
  seeing outcomes.
- Cache hits are accepted only when audio content, preprocessing configuration,
  tensor checksum, shape, dtype, and finite-value checks all agree.
- Before preprocessing and again inside each immutable audio snapshot, source
  size and SHA-256 must match the audio-content inventory that created the run
  identity. A path whose bytes changed after preflight is rejected.

## Augmentation and sampling

- Training only: random rotation from -20 to +20 degrees and horizontal flip
  with probability 0.5. No validation/test augmentation and no test-time
  augmentation.
- Hierarchical sampling is class -> participant -> recording. Classes are
  sampled uniformly, participants are visited by seeded round-robin order, and
  one recording is sampled uniformly within the selected participant.
- Validation and test sets retain their natural class distribution.

## Training

- Optimizer: AdamW.
- Loss: unweighted two-class cross entropy. Class balance is handled by the
  training sampler, not duplicated through loss weights.
- Effective batch size: 8. The physical batch and gradient accumulation pair is
  selected only by the resource pilot from `(8,1)`, `(4,2)`, or `(2,4)`.
- Maximum and confirmatory epoch count: exactly 100; no early stopping.
- Learning rate: `1e-5`; weight decay: `1e-8`; gradient norm clip: `0.1`.
- Scheduler: OneCycleLR, cosine annealing, `pct_start=0.3`, `div_factor=25`,
  `final_div_factor=10000`, stepped only after a successful optimizer update.
- Validation checkpoint selection: participant AUROC, then participant AUPRC,
  then participant NLL, then earlier epoch. The selection threshold is 0.5.
- Deterministic seed inputs bind fold, epoch, participant, recording, and draw.
- AMP may be accepted only if its 100-update pilot differs from FP32 by no more
  than 0.01 absolute probability and 1% relative cross-entropy loss and has no
  skipped optimizer updates.
- Selection takes the first safe physical batch in the order 8, 4, 2, then uses
  AMP if and only if it passes those checks; otherwise it uses FP32. Wall-clock
  timing and current free VRAM remain audited diagnostics but are not part of
  the reproducible pilot-decision hash.
- The pilot projects the serial runtime of the frozen 50-job, 100-epoch plan
  from measured optimizer-update throughput. The workload upper bound uses all
  contract-eligible Coswara participants separately for 25 cough, 15 speech,
  and 10 breath jobs, then applies a frozen 1.5 end-to-end overhead multiplier
  for evaluation, checkpointing, and orchestration. The operator ceiling is 168
  hours. This bounded planning estimate is not a completion-time guarantee; an
  over-ceiling pilot cannot be promoted, and full mode repeats the check.
- Confirmatory execution must resume only from a checksum-verified optimizer-
  safe checkpoint with matching manifest, cache, checkpoint, architecture,
  executable-source, fold, modality, representation, and sampler identity.
- Runtime progress is derived only from self-hashed stage/job receipts and the
  transactional last-checkpoint pointer. The current job contributes fractional
  progress only through its latest optimizer-safe checkpoint; live but
  uncheckpointed batches are never counted. Progress records contain no
  validation, test, or external metrics. Before the first checkpoint, the exact
  running job identity is visible but contributes zero fractional progress. The
  generation named by the latest progress record remains pinned across later
  checkpoint writes. Runtime polling verifies the full SHA-256 of every declared
  output and memoizes a result only while size, modification time, and change
  time remain unchanged; final evidence generation independently performs full
  artifact verification. These are integrity checks, not an adversarial
  authentication guarantee for a writable local run directory.

## Evaluation hierarchy

Primary internal estimate:

- Ten repeated participant-level 70/10/20 train/validation/test holdouts.
- HST and comparator predictions are compared only on identical eligible
  participant-recording-modality rows.

Reliability estimates:

- Calendar-mixed 60/20/20 control selected from 1,000 prespecified candidate
  assignments using seed 42 and label/date balance only.
- Strict chronological early-to-late 60/20/20 evaluation.
- Reverse late-to-early 60/20/20 sensitivity analysis.
- Coswara-to-COUGHVID cough-only external transfer with no target-domain model
  or threshold selection.

The validation threshold is frozen before each test evaluation. Test and
external labels are opened once in a non-adaptive pass of the prespecified,
fully frozen endpoints. For the comparator, validation alone selects the
primary endpoint; individual model-bank test rows are prespecified secondary
evidence. Repeated-holdout uncertainty
resamples Coswara participant clusters jointly across all of their repeated-
fold appearances and then averages fold-level metrics. Paired model comparisons
reuse the same participant-cluster draws. External source-target deltas
independently resample Coswara participants and COUGHVID recording UUIDs; the
same target draw is reused across source folds.

## Comparator and fusion

- Comparator features: ComParE 2016 plus IS10, 10,147 inputs before selection.
- In every fold and modality, LightGBM ranks training-only features and retains
  800. Four frozen model families are evaluated: LightGBM, SVC-RBF, CatBoost,
  and XGBoost. The selected ensemble is based on validation metrics only.
- Primary multimodal fusion is a complete-case uniform mean of cough and speech
  participant probabilities: 0.5 cough + 0.5 speech.
- Validation-AUPRC weighting and validation-only logistic stacking are
  secondary analyses.
- The four-branch HST/comparator hybrid uses fixed 0.25 weights and is secondary,
  not the primary HST estimand.
- Fusion never imputes missing modalities in the primary analysis.

## Reporting

- Analysis units: Coswara contributor for internal and temporal endpoints;
  COUGHVID recording UUID for external endpoints. No COUGHVID subject-level
  linkage or participant-level external claim is made.
- Metrics: AUROC, AUPRC, balanced accuracy, F1, sensitivity, specificity, Brier
  score, ECE, and NLL.
- Uncertainty: 1,000 bootstrap replicates, seed 42, 95% confidence intervals.
- Bootstrap deltas are interval estimates, not formal hypothesis tests; their
  sign proportions are not labeled as p-values. DeLong is used only for an
  exactly paired single-cohort AUROC comparison, and multiplicity correction is
  limited to declared families of valid formal secondary tests.
- Clinical screen on test-confirmed Coswara endpoints: specificity and precision
  at sensitivity at least 0.90 plus decision-curve net benefit at the frozen
  threshold grid. COUGHVID `status_SSL` is a semi-supervised pseudo-label, so
  external rows are excluded from clinical operating-point and decision-curve
  claims while retaining discrimination and calibration-agreement results.
- Grad-CAM: deterministic TP/TN/FP/FN examples from one frozen held-out context,
  final `attn2` input, participant-cluster summaries, and zero-map audit.
- Stage embeddings: held-out recordings only, averaged within participant before
  PCA or t-SNE visualization.
- All metric rows declare confirmatory, secondary, sensitivity, or exploratory
  scope and their multiplicity family.

## Pilot acceptance and full-run gate

The one-click notebook defaults to pilot mode and stops after the HST-Base
resource pilot. Full mode remains blocked until all three exact hashes are
manually reviewed and accepted:

1. data-contract freeze,
2. HST-Base resource-pilot freeze,
3. Ubuntu dependency/environment lock.

Acceptance is based on integrity, memory safety, deterministic behavior, and
runtime feasibility, never on favorable AUROC. Only after those records are
committed may the notebook run the confirmatory jobs through the evidence pack.
