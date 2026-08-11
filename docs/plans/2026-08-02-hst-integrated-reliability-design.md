# HST-Integrated Reliability Study Design

## Purpose

This study adds the Hierarchical Spectrogram Transformer (HST) as a strong,
published transformer branch inside the existing COVID respiratory-audio
reliability pipeline. It is not a standalone rerun of the HST authors' script.
The same HST branch will be evaluated under participant-independent internal
validation, temporal validation, external COUGHVID transfer, multimodal fusion,
calibration, uncertainty, operating-point, and explainability analyses.

The design answers four questions:

1. How competitive is HST when trained on the same Coswara participants used by
   the current final pipeline?
2. How does HST behave under the paper's broad participant-disjoint proportions
   of approximately 70% training, 20% test, and 10% validation (written as
   70/10/20 when this project consistently orders train/validation/test), while
   keeping published Cambridge/COUGHVID values as non-identical-dataset context?
3. Does a stronger transformer remain stable under chronological and
   cross-dataset shift?
4. Does a prespecified cough+speech hybrid of HST and the existing
   ComParE+IS10 branches add reproducible value on the project's strongest
   modality definition?

No new dataset is introduced. Coswara remains the development dataset and
COUGHVID remains the cough-only external dataset.

## Performance Objectives And Anti-Overfitting Rule

The HST branch is intended to improve the internal model bank, but improvement
is an engineering objective rather than a promised result. The frozen
same-protocol reference AUROCs are `0.868` for cough, `0.842` for breath,
`0.891` for speech, and `0.897` for the historical cough+speech fusion. The
development objectives are to exceed each corresponding unimodal reference and
to exceed `0.897` with the prespecified complete-case cough+speech fusion.

These numbers are not tuning targets on held-out test participants. Architecture
choices, checkpoint selection, and any learned fusion weights use training and
validation participants only. Once selected, the recipe is frozen and each test
partition is opened once in a non-adaptive pass of all prespecified locked
endpoints. Test performance cannot trigger a new seed,
preprocessing choice, checkpoint, fusion rule, or hyperparameter search. Results
below an objective remain valid results and are reported; a value above `0.897`
is publishable evidence only when it survives this rule and the aligned
participant-level comparison.

## Source And Checkpoint Provenance

The official source is preserved in the workspace at `HST/`:

- Repository: `https://github.com/icon-lab/HST.git`
- Audited commit: `7f94ad81e392da856c7aac6d364d036c28e26c32`
- Paper: `COVID-19 Detection From Respiratory Sounds With Hierarchical
  Spectrogram Transformers`, IEEE JBHI, DOI `10.1109/JBHI.2023.3339700`
- COUGHVID dataset description: DOI `10.1038/s41597-021-00937-4`
- COUGHVID-v3 release note for `status_SSL`: Zenodo record `7024894`

Only author-released ImageNet initialization checkpoints are permitted.

| Role | Checkpoint | Bytes | SHA-256 | Tensors | Parameters |
|---|---|---:|---|---:|---:|
| Smoke/runtime test | HST-Small ImageNet | 111,266,629 | `E7086D1B87D598120296B9A1B5F094C7587CB06F50BF609A4CA13BADC95E3112` | 190 | 27,770,596 |
| Primary experiment | HST-Base ImageNet | 197,063,145 | `F39F001D5F8CD90CB78D45612486202A4EA280E23DF0B2C1D6CE35D96B30CCE4` | 364 | 49,174,936 |

HST-Large is excluded because it adds compute and memory cost without a clear
benefit in the paper's size ablation. The authors' Task-2 cough checkpoint is
also excluded: it is already fine-tuned on Cambridge labels and is not a clean
initialization for an independent Coswara experiment.

Both author checkpoints contain a two-output classification head. Its origin is
not sufficiently documented for reuse. The loader must therefore:

1. verify file size and SHA-256;
2. verify the official source commit;
3. load with `weights_only=True`;
4. remove `head.weight` and `head.bias`;
5. strictly load every backbone tensor;
6. reinitialize a new two-class head using the experiment seed;
7. record checkpoint provenance in every run directory.

This uses the released representation weights without importing an ambiguous
task head.

Runtime copies are stored under `.cache/hst/checkpoints/`, which
is ignored by Git. Model weights are never committed or placed in Git LFS; the
small tracked manifest contains their URLs, sizes, and hashes.

## Why The Released Trainer Is Not Used Directly

The official repository supplies architecture and checkpoint provenance, but
its execution path is replaced because:

- it performs a random 80/20 image split rather than the paper's explicit
  participant-independent train/validation/test split;
- it obtains precision, recall, F1, and AUROC from the first validation batch
  in `validation_epoch_end` instead of aggregating every participant;
- the demo applies a random horizontal flip during inference;
- exact participant fold manifests are not saved;
- temporal validation, external transfer, calibration, confidence intervals,
  and resumable multi-fold execution are absent.

The replacement preserves the HST architecture and reported optimizer settings
while using the project's tested splitting and evaluation contracts. The
experiment is called a **paper-and-code-anchored HST adaptation**, not an exact
reproduction: the article and released scripts disagree on the spectrogram and
gradient-clipping details, and the Cambridge cohort is not available here.

The primary official-source constructor is frozen as:

```text
HSTModel(img_size=224, h=4, img_channel=3, num_labels=2, d=96,
         num_blocks=[1, 1, 9, 1],
         num_attention_heads=[3, 6, 12, 24], win_size=7,
         mlp_ratio=4.0, use_bias=True, dropout_rate=0.0,
         attn_dropout_rate=0.0, drop_path_rate=0.1,
         use_checkpoint=False)
```

The class contract is also frozen: `negative = 0`, `positive = 1`, and
`softmax(logits, dim=1)[:, 1]` is always the COVID-positive probability.
Checkpoint, metric, confusion-matrix, prediction-export, and Grad-CAM tests all
assert this convention. `ImageFolder` alphabetical class discovery is never
used.

## External Label And Identity Gate

COUGHVID is admitted only after a label-source audit. The primary continuity
analysis freezes `status_SSL` from the checksum-pinned processed cohort CSV as
its configured label column. That field contains COUGHVID-v3 semi-supervised labels expanded
from user and physician information; it is **not** described as RT-PCR-confirmed
ground truth. A separate raw-self-report sensitivity uses `status`. All
available `status`, `status_SSL`, and physician/expert annotation fields are
preserved side by side, and their coverage and disagreements are reported.
There is no silent column-priority rule.

For each configured label source, normalization uses exact, ordered aliases
before any pattern logic: explicit negative forms such as
`healthy`, `negative`, `no covid`, and `covid negative` map to `negative`;
explicit positive forms such as `COVID-19`, `positive`, and `covid positive`
map to `positive`; symptomatic/unknown/unreviewed values map to `unknown`.
Unseen or ambiguous strings fail closed to `unknown`. In particular, a generic
`"covid" in value` rule is prohibited because it misclassifies negated forms.

The audit writes dataset version/checksum, selected label column and provenance,
every raw value, normalized value, count, and exclusion reason.
It also compares the corrected labels with labels embedded in prior COUGHVID
artifacts. If any supervised row changes, all affected external metrics are
invalid until regenerated; no historical number is silently carried forward.
The HST study and its aligned comparator ingest the existing processed COUGHVID
cohort CSV through this new audited adapter only; they do not call the legacy
external adapter. The source file and its derived provenance declaration are
checksum-pinned. The study does not reconstruct or claim row-for-row membership
of the raw COUGHVID-v3 release. Corrected run-specific metadata and outputs have
new names, leaving existing files untouched while preventing their labels from
entering the new analysis.

Because the Windows checkout does not contain the Ubuntu raw dataset, the first
Ubuntu `data_contracts` stage computes the archive or sorted extracted-file
manifest hash and writes an immutable `data_contracts_freeze_hash`. Its
canonical contract descriptor includes the dataset/release identifiers,
selected label columns, label-normalization version, source-manifest hash, and
eligibility policy in addition to the source-file and label-audit hashes. It
also freezes the complete merged ComParE+IS10 comparator table by file hash,
row count, ordered feature names/dtypes, and feature-generation provenance; a
changed comparator table therefore invalidates the aligned-comparator stage.
Full
mode requires that exact accepted hash, just as it requires the resource-pilot
hash; a changed dataset release, label policy, metadata file, or source audio
cannot be silently reused.

The primary external transfer is named `coughvid_v3_ssl_status_transfer`, not
`confirmed-status` or `paper-matched`. An HST-paper COUGHVID cohort
reconstruction is a separate sensitivity and is created only when a non-COVID
status, cough symptoms, physician/expert label provenance, SNR, and event
segmentation are independently available. Otherwise it is emitted as skipped
with the missing fields documented; symptomatic rows are never relabeled by
assumption.

Every row receives immutable qualified identifiers:

```text
participant_key = dataset + "::" + participant_id
recording_key   = dataset + "::" + recording_id
```

These keys, not unqualified IDs, are used for splits, joins, aggregation,
fusion, bootstrapping, caching, and leakage audits.

For Coswara, `participant_id` identifies a contributor and is the internal and
temporal analysis unit. In the processed COUGHVID cohort, the available UUID is
a recording identifier and no verified contributor-linkage field is available.
The adapter therefore records `analysis_unit_type=recording_uuid`,
`subject_linkage_available=false`, and
`participant_id_is_recording_proxy=true`. The schema-compatible
`participant_key` does not convert a COUGHVID recording into a known person;
external counts, resampling, and claims remain recording-UUID based.

## Preprocessing Contract

### Primary paper-text reconstruction

The confirmatory input is `paper_logmel_224`. It follows the article where the
article is explicit and freezes every otherwise missing choice before outcomes
are viewed:

- mono audio at 22,050 Hz;
- resampling with pinned `soxr_hq` and `float32` output;
- silence trimming with `top_db=60`, frame length 2,205, hop length 1,102;
- require post-trim duration strictly greater than 2.0 seconds;
- periodic Hann-window STFT with `n_fft=2048` and `win_length=2048`;
- 128-point overlap, hence `hop_length=1920`;
- no centered padding (`center=False`), matching complete windowed segments;
- power spectrogram (`power=2.0`) followed by Mel mapping from 0 Hz to the
  11,025 Hz Nyquist frequency;
- 224 Mel bands because the paper specifies a final 224 x 224 matrix but does
  not report a separate Mel-band count;
- Slaney Mel filters (`htk=False`, `norm="slaney"`), recorded as a project
  reconstruction choice;
- `power_to_db(ref=np.max, top_db=80)`, followed by deterministic mapping of
  [-80, 0] dB to [0, 1];
- frequency increases bottom-to-top (stored array row 0 is the highest Mel
  band), followed by bilinear antialiased resize to 224 x 224, matching the
  paper's stated linear downsampling;
- one grayscale matrix replicated into three channels;
- intensity normalization with mean 0.5 and standard deviation 0.5.

Rotation and horizontal flip operate on the `[0, 1]` image before channel-wise
mean/std normalization. The resampler, window periodicity, Mel convention,
dtype, orientation, antialiasing, and transform order are part of the
preprocessing hash and golden-array tests.

### COUGHVID event-quality sensitivity

The HST article reports excluding COUGHVID recordings with SNR below 0.8 and
segmenting recordings into individual cough events with spectral peak
detection. The pinned official repository does not provide a checksum-
verifiable implementation of that event/SNR procedure. The frozen run therefore
retains full-recording external transfer, records the event-quality sensitivity
as `deferred_missing_checksum_pinned_algorithm`, and makes no claim that this
part of the paper was reproduced. A future extension may run only after one
versioned algorithm is frozen and applied identically to source and target.

The Mel-band count, 80 dB dynamic range, and no-padding boundary rule are
explicit reconstruction choices because the paper does not report them. This
is why the representation is a reconstruction rather than a claim of bitwise
paper reproduction. A constant or non-finite spectrogram is ineligible and
recorded rather than divided by zero.

### Released-code sensitivity representation

`released_linear_specgram_224` approximates public `wave2spectogram.py`
behavior: Matplotlib `specgram`, FFT 2048, overlap 128, grayscale rendering,
and resize to 224 x 224. The deterministic renderer and its golden hash test are
implemented, but model training with this representation is a deferred optional
extension because the released plotting code conflicts with the paper's Mel-
spectrogram description and contains inconsistent `Fs` values and plotting
paths. It is excluded from the frozen 50-job run and cannot be activated or
selected using test performance.

### Augmentation

The released trainer's training-only image transforms are retained:

- random rotation sampled uniformly from `[-20, +20]` degrees;
- random horizontal flip with probability 0.5.

Rotation uses torchvision's released-code defaults: no expansion, nearest
interpolation, and zero fill. The horizontal axis is time, so no vertical flip
or frequency-axis inversion is permitted.

The paper additionally states that training classes were balanced with random
amplification in `[1.15, 2.0]`, a playback-speed/pitch factor in `[0.8, 0.99]`,
and white noise, but it does not report the noise magnitude. The primary
adaptation therefore uses a deterministic hierarchical sampler: choose class
uniformly, choose a participant uniformly within that class, then choose one of
that participant's eligible recordings uniformly. This equalizes classes
without giving participants with more recordings greater expected influence.
Every draw has a unique `draw_id` and receives fresh seeded image augmentation.
This preserves the
reported balancing purpose without inventing an unreported audio-noise level.
No SMOTE, class weighting, or balancing is applied to validation or test data.
The per-fold before/after class counts and ordered participant/recording/draw
keys are saved.

The number of draws is also frozen: each class contributes
`max(n_negative_participants, n_positive_participants)` draws per epoch. A
seeded round-robin participant permutation gives every participant in the larger
class one draw and repeats participants in the smaller class as evenly as
possible. This avoids an arbitrary epoch length and prevents recording-rich
participants from dominating training.

A one-fold exploratory `paper_waveform_augmentation_reconstruction` may apply
the reported amplification and playback-speed ranges plus white noise fixed at
30 dB SNR. It is explicitly labeled exploratory because 30 dB is a project
choice, and it cannot replace or select the primary branch based on test
performance.

Validation, internal test, temporal test, and external test transforms are
deterministic. No random test-time flip is allowed.

## Split And Unit-Of-Analysis Contract

The unit of splitting and evaluation is the participant, not the recording or
spectrogram.

### Three evaluation questions, one frozen modeling recipe

The architecture, initialization checkpoint, preprocessing, augmentation,
optimizer, epoch selection, participant aggregation, fusion, threshold,
calibration, and metric implementations are invariant. Each source split is
trained separately; "frozen" refers to the modeling recipe, not reuse of one
fitted checkpoint across incompatible training cohorts. Three distinct
questions are kept separate so a single delta is not asked to explain several
simultaneous changes.

**Track A - literature-aligned internal evaluation**

- Ten repeated calendar-mixed, stratified, participant-disjoint 70/10/20
  holdouts using the ten seeds present in the released HST baseline scripts.
- Measures same-collection internal performance under the broad split style
  described in the HST paper.
- It is named `ten repeated stratified participant holdouts`, not ordinary
  `10-fold CV`: a participant can be tested in several repetitions or none.

Track A has two clearly separated cohorts:

1. `project_target_all_eligible`, the primary target used throughout the
   existing reliability study; and
2. `hst_task2_like_cough`, a secondary symptom-matched sensitivity restricted
   to COVID-positive participants reporting cough and COVID-negative
   participants reporting cough, using cough audio only.

The second cohort is the closest feasible Coswara analogue of the HST paper's
COVID-versus-non-COVID-with-symptom Task 2 and is included so task definition is
not ignored when contextualizing reported HST values. The exact symptom fields,
missing-value exclusions, participant counts, and class counts are frozen before
training. It still is not an exact Cambridge comparison because case
ascertainment, recruitment, recordings, and symptom definitions differ.

Track A aligns split proportions and participant isolation, not the Cambridge
dataset, labels, recruitment process, class construction, or task composition.
Published Cambridge metrics are therefore literature context, not a
same-dataset superiority test. There is no scientifically justified requirement
that Coswara performance fall within one percentage point of Cambridge.

The primary project-facing Track-A bank trains cough and speech because the
project's strongest existing multimodal definition is cough+speech. Cough is
also the direct modality anchor to the HST article. Breathing is a prespecified
secondary paper-modality sensitivity and cannot delay the primary bank. This
ordering answers the project's incremental-model question without pretending
that speech was evaluated in the HST article.

**Track B - matched-cohort split-policy contrast**

- Begin with one shared set of date-eligible, post-preprocessing participants.
- A participant date is the earliest valid UTC-normalized recording timestamp,
  matching the existing temporal audit. A participant is date-eligible only when
  every included recording has a parseable timestamp; participants with no valid
  date or a mixture of valid and unparseable dates are excluded under distinct
  audited reasons. Chronological ties are ordered by
  timestamp then `participant_key`; boundary tie counts are reported.
- The chronological assignment uses the earliest 60% for training, next 20%
  for validation, and latest 20% for testing.
- A calendar-mixed reference uses the exact same participant pool and the exact
  positive/negative counts in each split. From 1,000 seeded candidate
  assignments (`seed = 42 + candidate_index`), select the metadata-only
  candidate minimizing, across splits, the maximum of (a) absolute standardized
  mean difference in participant month ordinal relative to the full pool and
  (b) the Kolmogorov-Smirnov distance from the full date distribution. The
  lowest candidate seed breaks exact ties. No audio features or model outcomes
  enter this choice; all candidate scores are retained.
- Use model seed 42 and the same scientific configuration for both assignments;
  candidate-assignment seeds do not become model seeds.

This contrast holds eligibility, overall cohort, split sizes, label counts,
model, and analysis code fixed. The intended change is calendar assignment,
but the endpoint test participants differ. It is therefore a split-policy
sensitivity, not a causal estimate attributable only to time or a fully paired
test. Month-balance diagnostics and all participant keys are reported.

A secondary common-test control fixes the latest 20% of participants as the
test cohort for both models. Within the remaining historical 80%, one source
manifest uses chronological 60/20 train/validation assignment and the other
uses a date-balanced train/validation assignment with exactly the same class
counts. The date-balanced assignment reuses the frozen 1,000-candidate
month-SMD/KS objective and tie rule, restricted to train/validation. Both models
are evaluated on the identical late test participants.
This isolates sensitivity to source train/validation assignment while holding
the deployment target fixed; it does not substitute for the broader split-
policy contrast.

**Track C - deployment-oriented robustness**

- late-to-early reverse temporal sensitivity on the same date-eligible universe,
  assigning the latest 60% to train, preceding 20% to validation, and earliest
  20% to test after the same timestamp/key ordering, with model seed 42;
- Coswara-to-COUGHVID cough-only external transfer;
- calibration, fixed-sensitivity, decision-curve, and uncertainty analyses.

The manuscript uses `literature-aligned internal evaluation`, `matched-cohort
split-policy contrast`, and `external transfer`, not `non-strict` versus `strict`.
Every internal split remains participant-disjoint. Recording-random splitting
is prohibited.

### Protocol-matched internal experiment

- Ten repeated stratified participant splits.
- Approximately 70% train, 10% validation, 20% test.
- HST and existing final comparators use exactly the same recordings and
  participants in the primary matched analysis.
- A participant can never occur in more than one split within a fold.
- The deterministic base spectrogram cache is built first. Final manifests are
  frozen only after decoding, duration, finite-value, and representation
  eligibility are known.
- The matched HST-versus-comparator analysis uses the exact shared
  `(recording_key, modality)` post-preprocessing intersection before participant
  aggregation. Participant-matched/recording-unmatched and representation-
  specific full cohorts are reported only as secondary results.

### Reliability ladder

After configuration is frozen, evaluate:

1. literature-aligned repeated 70/10/20 holdouts;
2. matched-count calendar-mixed reference;
3. chronological early-to-late validation;
4. chronological late-to-early sensitivity analysis;
5. Coswara-cough to COUGHVID cough-only external transfer.

COUGHVID validates only the cough branch, not the multimodal model. Each of the
ten Track-A source-fold cough checkpoints predicts every eligible COUGHVID
  participant. Fold-specific external results are retained. A secondary ensemble
  first applies each fold's source-validation Platt calibrator and then averages
  the ten calibrated probabilities with equal weights. Neither checkpoints,
  weights, calibration, thresholds, nor ensemble membership can use COUGHVID
  labels. Because no one source participant set is out of sample for all ten
  repeated-holdout models, this ensemble has no validation-selected threshold or
  fixed-sensitivity operating point. It reports AUROC, AUPRC, Brier, ECE, NLL,
  and a fixed-0.5 row; fold-specific external rows retain their own
  source-validation thresholds. Decision curves and clinical fixed-sensitivity
  operating points are not computed against the semi-supervised external
  `status_SSL` pseudo-label.

## Training Contract

The confirmatory HST-Base configuration is:

- full-backbone fine-tuning;
- cross-entropy loss;
- AdamW optimizer;
- OneCycleLR scheduler from the released trainer with maximum learning rate
  `1e-5`, cosine annealing, `pct_start=0.3`, `div_factor=25`, and
  `final_div_factor=10000`;
- L2 weight decay `1e-8`;
- physical batch size 8, reduced to 4 and then 2 only if the next larger size
  fails the frozen 8 GB GPU pilot;
- FP32 and CUDA automatic mixed precision are compared at the first viable
  physical batch size; AMP is accepted only when a fixed-batch numerical check
  against FP32, from identical initial weights in evaluation mode, has maximum
  absolute positive-probability difference at most `0.01` and relative
  cross-entropy-loss difference at most `0.01`, and the 100-update pilot has
  zero skipped optimizer updates;
- gradient norm clipping at 0.1;
- maximum 100 epochs;
- all 100 epochs completed for the confirmatory paper/code-anchored run, with the
  best checkpoint selected by participant-level validation AUROC, with
  participant-level AUPRC and NLL as tie-breaks after recording probabilities are averaged
  within participant;
- unweighted cross-entropy with the frozen training-only hierarchical sampler
  defined above; no class weighting or SMOTE;
- prespecified project seeds `[1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002]`,
  taken from released HST baseline scripts, one per repeated split;
- test/external labels never used for model, epoch, threshold, or fusion choice.

The article describes task-specific stopping within epochs 1-100 from continued
validation-loss increase or F1 decrease, but gives no numeric rule; the released
trainer executes all 100 epochs and then reports the epoch with maximum
validation AUROC. Allowing an invented patience 10 from epoch 1 could also stop
before OneCycleLR reaches its 30%-schedule maximum. The confirmatory paper-text
adaptation therefore trains all 100 epochs and selects the best saved epoch by
participant-level validation AUROC, with participant-level AUPRC, lower
participant-level NLL, and then earlier epoch as deterministic tie-breaks. A separately labeled
runtime sensitivity may use patience 10 only after at least 40 completed epochs
and cannot replace the confirmatory result. An operating threshold is selected
from validation only after the epoch is frozen. This prevents per-epoch
threshold optimization from influencing checkpoint selection. The article states gradient-norm clipping at
0.1 while the released trainer calls value clipping; norm clipping is the
confirmatory paper-text choice and the discrepancy is recorded. The one-fold
released-code sensitivity instead uses maximum validation AUROC and value
clipping, so these incompatible paper/code choices are never silently mixed.

Random, NumPy, PyTorch, CUDA, sampling, augmentation, and DataLoader-worker
seeds are derived from `(fold_seed, epoch, participant_key, recording_key,
draw_id)`. At each epoch the sampler first materializes an immutable ordered
draw plan containing class, participant key, selected recording key, draw ID,
and augmentation seed for every occurrence, including repeated minority-class
draws. The DataLoader consumes that plan in order without a second shuffle.
Augmentation is therefore stateless and reproducible even with persistent
workers or resume. CuDNN benchmarking is disabled and
deterministic algorithms are requested. If a required official HST operation
has no deterministic CUDA implementation, the pilot stops and documents the
exact operation; it does not silently continue nondeterministically.

When gradient accumulation is required, each mean micro-batch loss is
multiplied by its sample count and accumulated. At the update boundary, AMP
gradients are unscaled once and all gradients are divided by the actual number
of accumulated samples before the optimizer update. The final incomplete
accumulation group is flushed at epoch end rather than dropped, so a short final
micro-batch does not bias or disappear from the update. At every effective-batch
boundary, gradient finiteness is checked, norm clipping is applied, and the
optimizer/scaler update occurs. The OneCycleLR scheduler
advances only when the optimizer update succeeds; skipped AMP steps do not
advance it. Scheduler `total_steps` is the sum of `ceil(epoch_batches /
gradient_accumulation)` across epochs, not the number of micro-batches.

Every epoch writes atomic `last.pt` and `best.pt` checkpoints containing model,
optimizer, scheduler, AMP scaler, the immutable epoch-draw-plan hash,
`next_consumed_batch_index`, random-number states, epoch, and best-validation
state. The consumed-batch index is updated only after the corresponding
optimizer-safe work is complete; a prefetched DataLoader sampler cursor is
never treated as consumed data. A temporary checkpoint is flushed, reloaded,
and verified before `os.replace`; an invalid new file never replaces the prior
checkpoint. If a measured epoch exceeds 30 minutes, an additional checkpoint is
written at safe optimizer-update boundaries. A browser or SSH disconnection
must not discard completed work.

## Prediction Aggregation

HST predicts each eligible recording. Probabilities are reduced in two steps:

1. average recordings within participant and modality;
2. fuse participant-level modality probabilities.

Participants with more recordings therefore receive no additional metric
weight. Aggregation keys always include `run_id`, protocol, fold, dataset,
participant key, split, modality, model, checkpoint hash, and representation.
Repeated-holdout predictions missing `fold` are rejected. A uniqueness assertion
prevents predictions from separately trained folds from being averaged before
fold-level evaluation.

## Multimodal And Hybrid Fusion

The model bank contains:

- HST cough;
- HST speech as the project-facing extension required to match the current
  strongest cough+speech pipeline;
- HST breathing as a secondary HST-paper-modality sensitivity;
- aligned existing ComParE+IS10 branches.

The primary project comparison uses cough+speech because that is the modality
definition behind the project's strongest existing multimodal result. Its
aligned ComParE+IS10 comparator is retrained on the same frozen rows and folds;
the historical 0.897 row remains context unless its cohort is exactly reproduced.
The cough-only branch provides the direct modality-level anchor to the HST
article. Breathing and cough+breath fusion are secondary paper-modality
sensitivities. Speech is always labeled as a project extension absent from the
HST paper, so project relevance and paper fidelity are not conflated.

For HST-versus-ComParE+IS10 comparisons, both models are retrained on the exact
same frozen fold manifests. The primary paired table uses the intersection of
`(recording_key, modality)` eligible for both representations in training,
validation, and test, then aggregates those identical recordings to
participants. A participant-matched but recording-unmatched comparison and each
family's full cohort are secondary and cannot support paired claims.
Eligibility counts and class proportions are reported for every fold and split.

Pre-specified fusion rules are:

1. `uniform_mean`, the primary matched multimodal comparison;
2. `legacy_validation_weighted_auprc`, a secondary rule reproducing the
   existing final pipeline, where branch weight is
   `max(validation_AUPRC - 0.5, 0.01)` and weights are normalized; the fixed
   `0.5` baseline is retained for reproducibility and not described as a
   prevalence correction;
3. `stacked_logistic_validation`, a class-balanced L2 logistic stack with
   `C=1.0`, `max_iter=2000`, and `random_state=42`, fitted only to complete-case
   validation predictions. It is prespecified because it produced the
   historical 0.897 cough+speech row, but its new-fold result is reported
   regardless of direction.

The primary fusion cohort is one joint complete-case set: every participant has
every required HST and ComParE+IS10 branch in train, validation, and test. Exact
participant-key equality between the two fused model-family outputs is asserted
for every fold, split, and modality combination. The same uniform rule is
applied to both families. Available-modality fusion, which renormalizes weights
over present branches, and validation-weighted fusion are reported separately as
sensitivity analyses. All weights and sample counts are saved per fold. No
weight is manually chosen after viewing test or COUGHVID results. Every
pre-specified fusion is reported; the stacker's in-sample validation score is
never used to select or discard it.

The secondary hybrid system uses the same joint complete-case cohort and four
participant probabilities: HST cough, HST speech, selected-comparator cough,
and selected-comparator speech. Its method-neutral rule is their uniform mean,
so each branch has weight 0.25. The legacy validation-AUPRC rule and the frozen
logistic stack are also applied to these four columns as prespecified secondary
hybrids. Hybrid deltas versus HST-only and comparator-only fusion are paired on
the same participant/fold rows. A hybrid is never allowed to inherit a test-
selected branch or to hide a weak constituent by changing its inputs after
outcomes are known.

The aligned ComParE+IS10 comparator reproduces the existing final branch-bank
logic rather than assuming one universal SVC: within each fold and modality it
trains the four frozen model families (LightGBM, SVC-RBF, CatBoost, and XGBoost),
requests the historical ensemble cap of five, and therefore constructs an
effective four-member ensemble because only four candidates are supplied. This
must be recorded as `top_4_validation_ensemble`, exactly as in the existing
final-validation artifact, rather than mislabeled as top five. It then selects
one branch by validation AUROC with validation AUPRC and model name as
deterministic tie-breaks. HST test outcomes cannot alter this selection. The
four frozen members, their fixed uniform ensemble, and the validation-selected
endpoint are evaluated in one non-adaptive held-out pass. The selected endpoint
is primary; individual bank rows are prespecified secondary evidence rather than
additional adaptive model choices. The
larger library-level `DEFAULT_MODEL_NAMES` bank is not substituted into the
primary analysis because it was not the CLI configuration that generated the
reported final branch; running it would be a separately labeled sensitivity.
A fixed SVC/top-800 row is retained as a simpler paper-comparable sensitivity.
Feature ranking, model-specific learned feature filters, scaling, and SMOTE are
fitted inside the training partition only. SMOTE, when required by a frozen
comparator name, acts only on the selected training matrix; validation and test
participants are never resampled or used to fit a preprocessing transform.

## Evaluation And Statistical Contract

The analysis hierarchy is fixed before any HST outcome is available:

1. The single primary estimand is the participant-level AUROC delta between
   HST and the aligned ComParE+IS10 system for complete-case uniform
   cough+speech fusion in Track A, summarized as the mean across the ten
   repeated holdouts with a paired participant-cluster bootstrap interval.
2. Key secondary estimands are HST's calendar-mixed versus chronological AUROC
   contrast, Coswara-cough versus COUGHVID AUROC contrast, and cough+speech
   fusion versus the constituent HST modality selected independently within each
   fold by source-validation AUROC only. Their effect sizes and intervals are
   emphasized; any family of
   secondary p-values is Holm-adjusted.
3. AUPRC, balanced accuracy, calibration, fixed-sensitivity operation, DCA,
   reverse temporal results, label-source sensitivities, and fixed-SVC
   comparisons are secondary or sensitivity evidence as labeled.
4. Task-2-like cohort reconstruction, breathing, released-code spectrograms,
   waveform augmentation reconstruction, available-modality fusion, t-SNE,
   and individual Grad-CAM examples are exploratory.

All prespecified rows are retained regardless of direction. The evidence
manifest stores `analysis_scope`, `estimand_id`, and multiplicity family, so an
exploratory result cannot silently replace the primary comparison.

Every branch and fusion reports the following where its validation design makes
the quantity identifiable; structurally undefined rows, such as an ensemble
threshold without a common out-of-sample validation cohort, are emitted as
skipped with a reason rather than fabricated:

- AUROC and AUPRC;
- balanced accuracy and F1;
- sensitivity and specificity;
- Brier score, ECE, and NLL;
- threshold 0.5 metrics and validation-selected threshold metrics;
- specificity and precision at sensitivity at least 0.90;
- participant bootstrap confidence intervals;
- paired deltas for the same participants;
- independent two-sample bootstrap deltas for Coswara versus COUGHVID;
- decision-curve net benefit;
- runtime and peak GPU memory.

For repeated 70/10/20 holdouts, the literature-aligned summary is mean and
standard deviation across the ten repetition metrics. Test counts are never
summed and called unique participants. Confidence intervals are conditional on
the already fitted models and use a participant-cluster bootstrap: resampling a
participant retains all of that participant's
test appearances and recomputes the mean of repetition-level metrics. HST-versus-
comparator deltas are computed on the shared participant/fold predictions and
use the same clustered resamples. Any fold-level Wilcoxon result is secondary
and explicitly labeled as dependent repeated-holdout evidence.

The matched-cohort calendar-mixed versus chronological endpoints use different
test participants, so their descriptive delta uses independent, label-
stratified participant bootstraps within each endpoint and is not called paired
or causal. The secondary common-latest-test control predicts the exact same test
participants from the two training assignments and therefore uses a paired
participant bootstrap even though the two protocol labels differ; all other
pairing keys and the participant/label rows must match exactly. Fixed temporal
endpoint confidence intervals bootstrap
unique test participants.
Cross-dataset deltas are computed separately per source fold. In each bootstrap
replicate, target COUGHVID recording UUIDs are sampled once and that same sampled
target cohort is used for every fold, so ten predictions of one recording are
not treated as ten independent observations. Fold estimates are labeled
dependent. The calibrated ensemble first reduces to one probability per
external recording UUID and then uses independent stratified source-participant
and target-recording resampling. DeLong is never used across unpaired datasets.

Threshold-dependent metrics use both a fixed 0.5 threshold and a threshold
selected only on the corresponding source validation split. Screening
operating points are also selected on validation, never on test. Raw
probabilities are always reported; Platt calibration is fitted on source
validation predictions and then applied unchanged to internal, temporal, and
external tests.

The ordinary validation-selected operating threshold reuses the existing
`best_threshold_by_balanced_accuracy` function on raw participant
probabilities. Its candidates are `0`, `0.5`, `1`, every unique validation
probability, and every midpoint between adjacent unique probabilities. It
maximizes validation balanced accuracy; ties prefer the threshold closest to
0.5 and then the lower threshold. This frozen threshold is applied unchanged to
raw held-out probabilities. It is distinct from the calibrated 90%-sensitivity
screening operating point below.

Calibration reuses the existing project `PlattCalibrator`: scikit-learn
logistic regression with `solver="lbfgs"`, default `C=1.0`, an intercept, no
class weighting, `max_iter=100`, `tol=1e-4`, and participant probability as its
single input. A
single-class or non-converged source-validation calibrator is emitted as skipped
with raw probabilities retained; it is never repaired using test or target
labels.

The numerical reporting definitions are frozen to the existing project
conventions. ECE uses 10 equal-width bins on `[0, 1]`, omits empty bins, and is
the sample-count-weighted mean absolute difference between mean predicted
probability and observed positive rate. Brier score is the mean squared
probability error. NLL clips probabilities to `[1e-6, 1 - 1e-6]`. Reliability
plots and these scores are evaluated on held-out participant probabilities;
the source-validation rows used to fit Platt scaling are not presented as an
unbiased calibration endpoint.

The screening threshold is selected by enumerating `0`, `0.5`, `1`, and all
unique calibrated source-validation probabilities. Among thresholds with sensitivity at
least 0.90, choose the one with highest specificity, then sensitivity, then
threshold as deterministic tie-breaks, and apply it unchanged to the held-out
endpoint. Precision, NPV, confusion counts, and achieved sensitivity and
specificity are reported with the selected threshold.

Primary decision-curve analysis on test-confirmed Coswara endpoints uses
source-validation Platt-calibrated
probabilities because threshold probabilities have a risk interpretation. It
evaluates the fixed grid `0.05, 0.10, ..., 0.50` and reports model, treat-all,
and treat-none net benefit, where
`NB_model = TP/N - (FP/N) * p_t/(1-p_t)`,
`NB_all = prevalence - (1-prevalence) * p_t/(1-p_t)`, and
`NB_none = 0`. A raw-probability curve is retained as a labeled sensitivity;
neither curve is used for model selection. COUGHVID `status_SSL` is a
semi-supervised pseudo-label, so external rows are excluded from decision curves
and clinical fixed-sensitivity summaries rather than being interpreted as
clinical utility.

Production confidence intervals use 1,000 deterministic participant-level
bootstrap replicates with seed 42 and percentile 2.5%/97.5% bounds. The
200-replicate calls in unit tests are speed-only fixtures, not report settings.
Resampling is label-stratified for unpaired single-endpoint and source-target
comparisons. Paired model deltas resample one shared participant-key vector.
The repeated-holdout and external-fold dependence rules above take precedence
over ordinary row resampling. A replicate in which an AUROC/AUPRC endpoint
lacks one class is rejected and redrawn; the report requires 1,000 valid
replicates within 10,000 deterministic attempts or fails rather than silently
changing the denominator.

The primary paired comparison is HST versus an aligned instance of the current
ComParE+IS10 model family on identical Coswara recordings, participants,
modalities, and folds. The historical 0.897 cough+speech fusion remains context
unless its complete cohort, feature eligibility, modality, split, and aggregation
definition is exactly reproduced in the aligned bank. Published Cambridge and
COUGHVID values remain literature context, not same-dataset superiority tests.

## Explainability Contract

Grad-CAM targets the class-1 pre-softmax logit at the final released-code LWMSA
module, `model.layers[-1].HSTblocks[-1].attn2`, rather than the post-hierarchy
`norm_layer`. The primary paper-proximal map captures the final LWMSA input,
corresponding to the paper's wording "prior to the final attention layer"; the
module output is a separately labeled sensitivity. At the final stage there is
one 7 x 7 window, so the 49 tokens can be reshaped to the spatial grid and
upsampled to 224 x 224. This is a documented reconstruction, not a claim that
the paper released its visualization implementation. Deterministic held-out
examples are selected from true-positive, true-negative, false-positive, and
false-negative categories defined at fixed threshold 0.5. A
validation-selected-threshold panel is separately labeled as a sensitivity.
Raw heatmaps, source spectrograms, overlays, labels, probabilities, threshold
source, folds, participants, and recordings are saved.

Grad-CAM is model-attention evidence, not proof of a physiological biomarker.
Examples cannot be hand-picked for visual attractiveness.

In addition to individual audit examples, participant-balanced group maps are
formed from all held-out participants correctly classified at fixed 0.5.
Recording maps are first averaged within participant. The primary summaries are class-wise
participant-mean heatmaps and their mean difference with participant-bootstrap
uncertainty. A secondary pooled PCA basis is fitted once across both classes and
class projection distributions are compared in that common basis. To mirror the
paper descriptively, separate class-wise first-PC maps may also be shown after
orienting each sign against its class mean, but separately fitted PCs are never
subtracted or treated as inferential evidence. These maps remain explanatory,
not biological validation.

To parallel the HST paper's hierarchy analysis, held-out latent vectors are
also captured after each of the four HST stages, averaged first across
recordings within participant, and visualized with deterministic PCA. A fixed-
seed t-SNE panel uses `perplexity=30`, PCA initialization, automatic learning
rate, 1,000 iterations, and seed 42; it is skipped when the participant count
does not exceed 30. It is a labeled visual sensitivity, not quantitative
evidence of class separation. No embedding is chosen because
it looks more separated, and train participants never appear in these plots.

## Notebook Automation

`notebooks/09_HST_RELIABILITY_E2E.ipynb` is a controller, not the owner of a
multi-day training process. Run All:

1. verifies environment, GPU, source commit, and checkpoints;
2. audits data and participant leakage;
3. selects safe preprocessing-worker candidates from a raw-audio pilot;
4. builds/reuses deterministic spectrograms and freezes manifests;
5. runs HST-Small smoke and resume checks;
6. runs HST-Base DataLoader, precision, batch, and timing pilots;
7. runs the aligned ComParE+IS10 comparator on the frozen HST-eligible manifests
   in an isolated CPU stage;
8. runs protocol-matched internal HST training;
9. runs the split-policy, temporal, and external HST ladder;
10. builds multimodal and hybrid fusion from completed branch predictions;
11. computes uncertainty, calibration, operating points, DCA, Grad-CAM, and the
    final evidence dashboard.

The notebook launches `scripts/72_run_hst_reliability.py` as a detached process
with a launch receipt, PID file, heartbeat, log path, and exit-status file, then
displays its status. Closing the browser, notebook kernel, terminal, or SSH
connection does not terminate the worker. The default first execution is
`pilot`; a full run requires the exact successful pilot freeze hash in the
configuration.

Pilot launch avoids a circular run identity. Before the data-contract hash is
known, the detached worker owns
`data/outputs/hst/_bootstrap/<launch_id>/`, where `launch_id` hashes the mode,
configuration, source-root locator, code/dependencies, HST commit, and
checkpoint provenance. It runs preflight and data-contract hashing there while
checking each source file's size and mtime before and after hashing. Once the
immutable data-contract hash exists, it derives the final content-addressed
`run_id`, atomically promotes the bootstrap directory to
`data/outputs/hst/<run_id>/`, and writes an atomic
`data/outputs/hst/_launches/<launch_id>.json` receipt pointing to it. If an
identical validated final run already exists, the worker attaches to it instead
of overwriting it. Stage records contain run-root-relative output paths. Before
promotion the worker flushes and closes bootstrap runtime files; after rename it
rebuilds its path context and reopens log, heartbeat, PID, and exit files under
the final root before continuing. Full mode already supplies accepted
data/pilot/environment hashes and can derive the final run ID immediately. Re-running Run
All resolves the same launch receipt and cannot create a timestamp-named
duplicate.

Every mutable artifact lives under a run-specific directory. An exclusive
host/PID/heartbeat lock prevents two controllers from starting the same run.
Stale locks are recoverable only after the recorded process is absent and the
heartbeat age exceeds the configured limit. A stable `latest` evidence pointer
is published only after output validation succeeds.

A second execution-account-wide GPU lease is keyed by CUDA device UUID and held by the final
detached worker with a non-blocking exclusive OS file lock for its lifetime. Its
record contains host, final worker PID, process start identity, run ID, and
heartbeat. It prevents different run IDs and workspaces from starting
concurrent trainers on the single T1000. CPU-only reporting may run in parallel
only when its measured memory budget does not violate the training reserve.

Each stage hashes the scientific configuration, implementation source files,
Python dependency lock, HST commit, checkpoint, manifests, and upstream
artifact checksums. A stage is reused only if its record says `success`, all
outputs exist, and their checksums match. Changed settings, code, dependencies,
or corrupted outputs invalidate that stage and its dependents. Interrupted
training resumes from the last verified checkpoint.

Project source identity is a Merkle hash over an explicit allow-list of
executable HST modules and entry points plus every reused project module in the
runtime import path, including metrics, calibration, labels, audio I/O,
strong-baseline modeling, protocol construction, and final-validation logic.
Preflight fails if a newly imported project module is absent from that
allow-list. Test files have a separate QA hash and invalidate
verification/evidence publication, not completed model training. Scientific
JSON and the exact environment lock are hashed as separate inputs. Generated reports, acceptance
records, notebooks outputs, caches, and model artifacts are excluded from the
source hash. Git commit and dirty status are recorded for provenance, but a
commit identifier never replaces file-content hashes. Consequently, committing
the reviewed environment lock or pilot report does not pretend that executable
model code changed.

## Parallelism And Runtime Reduction

Parallelism is applied only where it preserves correctness and improves measured
throughput.

### Fixed execution host

The implementation is sized for the Ubuntu research machine already audited in
this repository:

| Resource | Recorded configuration | Execution consequence |
|---|---|---|
| CPU | Intel Core i7-14700, 24 logical CPUs exposed | CPU-only cache and statistical work may use bounded process pools |
| RAM | approximately 19 GiB | worker selection must preserve at least 4 GiB available memory |
| Swap | 8 GiB | swap is an emergency guard, not usable training capacity |
| GPU | NVIDIA T1000, 8 GiB VRAM, one device | exactly one HST trainer may occupy the GPU |
| Driver/runtime | driver 595.71.05; `nvidia-smi` CUDA 13.2 | preflight records live values and checks CUDA availability |
| PyTorch | 2.11.0+cu128 previously verified | Ubuntu preflight reconfirms the installed build before training |
| Disk | approximately 430 GB total | cache free space is checked before materialization |

These are expected defaults, not silently trusted constants. The notebook records
live CPU count, available RAM, free disk, driver, CUDA, PyTorch, and GPU memory.
A changed host triggers a new resource pilot before the full run.

### CPU-parallel work

- Spectrogram generation benchmarks candidate counts 1, 2, 4, 8, and 12 on a
  fixed stratified pilot containing long recordings and every
  encountered codec. A candidate is eligible only if live `MemAvailable`,
  cgroup limits, CPU affinity, parent-plus-child RSS, `/dev/shm`, and swap growth
  stay within bounds. Total installed RAM is never treated as available RAM.
- Selection maximizes valid recordings per second while preserving at least
  4 GiB actually available memory. Eight or twelve workers are used only after
  measurement; no test asserts that this host must sustain twelve.
- Each direct preprocessing subprocess starts a fresh interpreter. Every worker
  uses one OpenMP/MKL/OpenBLAS thread, enforced before numeric imports with
  environment variables and again with `threadpoolctl`, to prevent nested
  oversubscription. No daemon process-pool worker is asked to create a child.
- File hashes, eligibility audits, bootstrap resamples, and independent figure
  generation can use bounded CPU workers.
- The 10,147-column aligned ComParE+IS10 comparator runs as its own CPU stage
  after manifests are frozen and before scientific HST training. It is not run
  concurrently with the GPU trainer; it releases its feature table and worker
  processes before the first HST fold starts.
- Cache entries are transactional: the parent supervisor atomically claims a
  recording with `O_CREAT|O_EXCL`, recording PID/start identity and input/config
  hashes. It owns a bounded set of direct persistent subprocess workers and
  assigns one recording at a time. On a hard per-file timeout, the parent kills
  that worker's process group, records the attempt, starts a replacement, and
  permits one logged retry. Workers never create a nested process pool. The
  worker writes a unique temporary file in the destination directory, flushes
  and fsyncs it, verifies shape/dtype/finite values/checksum, uses `os.replace`,
  and fsyncs the destination directory. Each recording also receives an atomic
  result fragment so the parent can reconstruct the shared index after a crash;
  workers never write that shared index. Stale claims and temporary files are
  audited. Source audio hashes, decode timeout/retry state, and exclusion reasons
  are stored. Cache generation starts only with estimated cache size plus 20 GiB
  free disk.
- The primary full-recording cache stores one 224 x 224 grayscale `float32`
  array per recording and replicates the channel at load time. The separate
  event-segmentation sensitivity stores one array per qualified `event_key` and
  an immutable event-to-recording map before recording/participant aggregation.
  Both avoid threefold cache I/O and storage.

### GPU pipeline work

- Only one HST training job runs on the single T1000 GPU at a time.
- Folds and modalities are placed in a resumable queue and executed sequentially
  on the GPU.
- DataLoader workers prepare upcoming batches in parallel with GPU computation.
- `pin_memory=True`, `persistent_workers=True`, and `prefetch_factor=2` are used
  after a fixed 200-batch pilot selects a stable worker count from 0, 2, 4, and
  8 using throughput, valid tensors, and peak host memory only. Accuracy is not
  used for this systems decision.
- `persistent_workers` and `prefetch_factor` are disabled when `num_workers=0`,
  as required by the DataLoader contract.
- Fresh isolated pilot processes first benchmark FP32 and AMP at physical batch
  8 for at least 100 optimizer updates plus validation. If neither precision is
  valid, the same comparison is repeated at batch 4, then batch 2; a smaller
  batch is never selected merely because it is faster. A candidate must have
  finite losses, gradients, parameters, and predictions, zero skipped optimizer
  updates, fixed-batch AMP-versus-FP32 maximum absolute positive-probability
  difference at most `0.01`, relative cross-entropy-loss difference at most
  `0.01`, and free VRAM headroom of `max(1 GiB, 15% of total)`,
  measured with allocator peaks and `torch.cuda.mem_get_info`. The faster valid
  precision at the first viable batch size is frozen in JSON. Gradient
  accumulation preserves effective batch size 8. An out-of-memory fallback is
  allowed only during the pilot, never halfway through a scientific fold.
- Telemetry every 60 seconds records GPU utilization, temperature, clocks,
  free VRAM, host memory, and throttling indicators. The run refuses to start
  when an unrelated compute process occupies the GPU and pauses rather than
  silently continuing through sustained thermal throttling.

Running multiple HST-Base trainers concurrently on one 8 GB GPU is prohibited:
model parameters, gradients, optimizer states, activations, and input batches
would compete for memory and usually cause out-of-memory failure or lower total
throughput. True fold-level parallel training requires multiple GPUs; if that
hardware becomes available, the queue can assign one fold to each GPU without
changing the scientific configuration.
The evaluation tracks therefore share the deterministic base cache but enter
the single-GPU queue as separate, sequential jobs. The queue is materialized as
a durable job-plan table. A training key hashes every sorted train/validation
row (`participant_key`, `recording_key`, label, modality, split, and tensor
SHA-256) together with checkpoint, preprocessing, sampler, augmentation,
optimizer, model, and seed settings. External inference is a separate job keyed
by the verified source-checkpoint hash and the complete external-manifest hash.
Identical source jobs are reused; in particular, Track-A cough checkpoints are
reused for COUGHVID inference. Each job has an atomic `pending`, `running`,
`success`, `failed`, or `stopped` record with attempt, final worker identity,
heartbeat, checkpoint/output checksums, and error text. Resume skips only
checksum-valid successful jobs and recovers stale running jobs. Every job tears
down DataLoader workers and pinned buffers in `finally` and verifies process
count and RSS return within tolerance before the queue continues.

Heavy CPU preprocessing is completed before full GPU training. During training,
CPU concurrency is reserved for DataLoader workers so preprocessing/reporting
jobs do not starve the GPU input pipeline.

## Execution Gates

The full run starts only after:

1. COUGHVID release, `status_SSL`/`status`/expert provenance, disagreements, and
   ordered mappings pass the label-source audit;
2. the accepted `data_contracts_freeze_hash` matches the current source-file
   manifest, metadata, label audit, and dataset release identifier;
3. the accepted Ubuntu environment-lock hash matches the active `pip freeze`;
4. source commit and checkpoint hashes match;
5. backbone load has no unexplained missing/unexpected tensors;
6. a synthetic forward pass returns finite logits of shape `[B, 2]` and class
   index 1 is verified as COVID-positive;
7. cached spectrograms are finite, nonblank, checksum-valid, and final
   post-cache manifests show zero participant overlap;
8. matched comparisons use the exact shared `(recording_key, modality)`
   intersection before participant aggregation;
9. HST-Small trains, terminates safely, and resumes identically;
10. HST-Base passes the 100-update resource pilot with required headroom;
11. metrics aggregate every held-out participant at participant level and retain
   fold/protocol keys;
12. external labels never influence source checkpoints, calibration, fusion,
    thresholding, or model selection.
13. the execution-account-wide CUDA-UUID lease is exclusive across run IDs and all completed
    queue jobs have checksum-valid durable state.

## Interpretation Rules

- High internal HST performance followed by temporal/external collapse supports
  a reliability gap not limited to a weak classifier.
- High HST performance under every protocol weakens the current failure-mode
  thesis and requires manuscript revision.
- Low internal HST performance does not prove HST is weak; cohort, label,
  preprocessing, and implementation differences must be reported.
- A multimodal gain is claimed only when repeatable across folds with a
  meaningful paired interval.
- No SOTA claim is made without genuinely comparable dataset, cohort, labels,
  split, and metric definitions.

## Runtime Expectation

The paper's complexity table reports 5.5 minutes of HST training on NVIDIA
A4000 GPUs for its own cohort and software setup; that number is not a valid
end-to-end estimate for this multi-protocol run. The project's T1000 8 GB is
substantially less capable, and the one-fold pilot is the only reliable local
estimator.
Before full launch, the generated job plan and resource freeze bind exactly 50
serial HST-Base training jobs: 30 repeated internal jobs (ten each for cough,
speech, and breath), ten Task-2-like cough jobs, eight split-policy/common-late
jobs (four protocols for cough and speech), and two reverse-temporal jobs (cough
and speech). COUGHVID external inference reuses the ten internal cough
checkpoints and creates no target-trained job. The released-code and event/SNR
representations are deferred optional extensions; fusion adds no training job.
Planning ranges before pilot measurement are:

- preflight: under 30 minutes;
- spectrogram cache: 4-10 hours;
- HST-Small smoke: 1-3 hours;
- aligned ComParE+IS10 comparator stage: approximately 2-8 hours;
- one HST-Base fold/modality: 8-24 hours;
- ten-fold cough plus speech: approximately 7-20 days;
- the exact 50-job HST ladder: accepted only when the measured pilot projection,
  including the frozen 1.5 overhead multiplier, is at most 168 serial GPU hours;
- deferred representation sensitivities: not included in that estimate.

These are planning ranges, not promised completion times.
