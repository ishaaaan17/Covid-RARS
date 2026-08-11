# HST-Integrated Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checkpoint-verified HST-Base branch to the existing participant-level multimodal and reliability evaluation pipeline, controlled by one restart-safe Jupyter notebook.

**Implementation-status addendum (2026-08-02):** The frozen primary controller
runs the paper-described Mel representation and a raw-`status` label sensitivity
that reuses frozen external probabilities. The COUGHVID event/SNR branch is
explicitly deferred because the pinned official source has no checksum-
verifiable event/SNR implementation. The deterministic released-code renderer
is implemented and tested, but its model run is a deferred optional extension.
Neither deferred branch is part of, or may block, the frozen 50-job run.

**Architecture:** Keep the official HST repository pinned as an external source, load only verified author checkpoints, and place all project-specific preprocessing, training, fusion, evaluation, and reporting in `covid_rars` modules. The notebook calls one resumable pipeline API; scientific logic is tested outside notebook state.

**Tech Stack:** Python 3.12, PyTorch, torchvision, timm, librosa, NumPy, pandas, scikit-learn, matplotlib, JupyterLab, pytest.

---

## Command And Path Convention

Paths in **Files** lists are relative to the repository root. Unless a block
explicitly says otherwise, every shell command is run from the project root
`<repository>` with `.venv` activated; test, script, source, and report paths
are therefore relative to the repository root.
Only Task 1's submodule creation and its corresponding commit run from the
repository root because `HST/` and `.gitmodules` live there. Do not infer the
working directory from a previous shell block.

---

## File Map

**External source**

- Track as submodule: `HST/`
- Pin commit: `7f94ad81e392da856c7aac6d364d036c28e26c32`

**New configuration and dependencies**

- Create: `configs/hst_reliability.json`
- Create: `requirements-hst.txt`
- Create after the Ubuntu pilot: `requirements-hst-lock.txt`

**New modules**

- Create: `src/covid_rars/hst_data_contracts.py`
- Create: `src/covid_rars/hst_checkpoint.py`
- Create: `src/covid_rars/hst_comparators.py`
- Create: `src/covid_rars/hst_spectrograms.py`
- Create: `src/covid_rars/hst_protocols.py`
- Create: `src/covid_rars/hst_parallel.py`
- Create: `src/covid_rars/hst_training.py`
- Create: `src/covid_rars/hst_fusion.py`
- Create: `src/covid_rars/hst_reliability.py`
- Create: `src/covid_rars/hst_gradcam.py`
- Create: `src/covid_rars/hst_reporting.py`
- Create: `src/covid_rars/hst_runtime.py`

**New entry points**

- Create: `scripts/hst_preprocess_worker.py`
- Create: `scripts/72_run_hst_reliability.py`
- Create: `scripts/73_make_hst_evidence_pack.py`
- Create: `scripts/74_register_hst_evidence.py`
- Create: `notebooks/09_HST_RELIABILITY_E2E.ipynb`

**New tests**

- Create: `tests/hst_test_helpers.py`
- Create: `tests/test_hst_data_contracts.py`
- Create: `tests/test_hst_checkpoint.py`
- Create: `tests/test_hst_comparators.py`
- Create: `tests/test_hst_spectrograms.py`
- Create: `tests/test_hst_protocols.py`
- Create: `tests/test_hst_parallel.py`
- Create: `tests/test_hst_training.py`
- Create: `tests/test_hst_fusion.py`
- Create: `tests/test_hst_reliability.py`
- Create: `tests/test_hst_gradcam.py`
- Create: `tests/test_hst_notebook.py`

Existing model/result files are not overwritten. Every new artifact uses an
`hst_` prefix and a run identifier. Shared test builders live in
`tests/hst_test_helpers.py`; examples below must import those builders or define
self-contained fixtures rather than relying on undefined pytest fixtures.

**Frozen development objectives:** same-protocol participant AUROC above the
historical `0.868` cough, `0.842` breath, `0.891` speech, and `0.897`
cough+speech references. These are validation-guided engineering objectives,
not test-set stopping rules or guaranteed outcomes. Test partitions are opened
once after the configuration and fusion policy are frozen; missing an objective
must not cause post-test retuning.

**Generated artifact contracts**

- Cache: `.cache/hst/checkpoints/`
- Cache: `data/processed/hst_spectrogram_cache/<preprocessing_hash>/`
- Run root: `data/outputs/hst/<run_id>/`
- Bootstrap root: `data/outputs/hst/_bootstrap/<launch_id>/`
- Launch receipt: `data/outputs/hst/_launches/<launch_id>.json`
- Models: `<run_root>/models/<protocol>/<fold>/<modality>/`
- Predictions: `<run_root>/metrics/hst_recording_predictions.csv` and
  `<run_root>/metrics/hst_participant_predictions.csv`
- Metrics/history: `<run_root>/metrics/hst_reliability_metrics.csv` and
  `<run_root>/metrics/hst_training_history.csv`
- Audits/tables: `<run_root>/reports/tables/hst_*.csv`
- Figures: `<run_root>/reports/figures/hst_*.svg` and matching PNG files
- Runtime state: `<run_root>/runtime/{lock.json,pid.json,heartbeat.json,exit.json,run.log}`
- Immutable run manifest: `<run_root>/hst_evidence_manifest.json`
- Validated publication pointer: `reports/hst/latest.json`

No global mutable HST CSV is written. `latest.json` is atomically updated only
after schema, checksum, provenance, and completeness validation succeeds.

## Task 0: Lock External Labels, IDs, And Analysis Units

**Files:**
- Create: `src/covid_rars/hst_data_contracts.py`
- Create: `tests/hst_test_helpers.py`
- Create: `tests/test_hst_data_contracts.py`

- [ ] **Step 1: Write failing ordered-label tests**

```python
import pytest


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("COVID-19", "positive"),
        ("covid positive", "positive"),
        ("positive", "positive"),
        ("healthy", "negative"),
        ("negative", "negative"),
        ("no covid", "negative"),
        ("covid negative", "negative"),
        ("symptomatic", "unknown"),
        ("unreviewed", "unknown"),
        ("possibly covid", "unknown"),
    ],
)
def test_coughvid_labels_fail_closed(raw: str, expected: str) -> None:
    from covid_rars.hst_data_contracts import normalize_coughvid_status

    assert normalize_coughvid_status(raw) == expected


def test_positive_class_index_is_frozen() -> None:
    from covid_rars.hst_data_contracts import CLASS_TO_INDEX

    assert CLASS_TO_INDEX == {"negative": 0, "positive": 1}


def test_coughvid_label_source_must_be_explicit(tmp_path) -> None:
    from covid_rars.hst_data_contracts import build_audited_coughvid_index

    with pytest.raises(TypeError):
        build_audited_coughvid_index(tmp_path)


def test_data_contract_hash_changes_when_source_bytes_change(tmp_path) -> None:
    from covid_rars.hst_data_contracts import freeze_data_contracts

    source = tmp_path / "metadata.csv"
    audit = tmp_path / "label_audit.csv"
    source.write_text("recording_id,status_SSL\na,healthy\n")
    audit.write_text("raw,normalized\nhealthy,negative\n")
    first = freeze_data_contracts(
        source_root=tmp_path,
        audit_root=tmp_path,
        source_paths=(source,),
        label_audits=(audit,),
        contract_metadata={
            "dataset_release_id": "coughvid-v3-7024894",
            "label_column": "status_SSL",
            "label_normalization_version": 1,
            "source_manifest_sha256": "a" * 64,
            "eligibility_policy_version": 1,
        },
        output_path=tmp_path / "a.json",
    )
    source.write_text("recording_id,status_SSL\na,COVID-19\n")
    second = freeze_data_contracts(
        source_root=tmp_path,
        audit_root=tmp_path,
        source_paths=(source,),
        label_audits=(audit,),
        contract_metadata={
            "dataset_release_id": "coughvid-v3-7024894",
            "label_column": "status_SSL",
            "label_normalization_version": 1,
            "source_manifest_sha256": "a" * 64,
            "eligibility_policy_version": 1,
        },
        output_path=tmp_path / "b.json",
    )
    assert first != second


def test_data_contract_hash_changes_when_release_or_label_policy_changes(tmp_path) -> None:
    from covid_rars.hst_data_contracts import freeze_data_contracts

    source = tmp_path / "metadata.csv"
    audit = tmp_path / "label_audit.csv"
    source.write_text("recording_id,status_SSL\na,healthy\n")
    audit.write_text("raw,normalized\nhealthy,negative\n")
    common = {
        "label_column": "status_SSL",
        "label_normalization_version": 1,
        "source_manifest_sha256": "a" * 64,
        "eligibility_policy_version": 1,
    }
    first = freeze_data_contracts(
        source_root=tmp_path, audit_root=tmp_path,
        source_paths=(source,), label_audits=(audit,),
        contract_metadata={**common, "dataset_release_id": "release-a"},
        output_path=tmp_path / "a.json",
    )
    second = freeze_data_contracts(
        source_root=tmp_path, audit_root=tmp_path,
        source_paths=(source,), label_audits=(audit,),
        contract_metadata={**common, "dataset_release_id": "release-b"},
        output_path=tmp_path / "b.json",
    )
    third = freeze_data_contracts(
        source_root=tmp_path, audit_root=tmp_path,
        source_paths=(source,), label_audits=(audit,),
        contract_metadata={
            **common,
            "dataset_release_id": "release-a",
            "label_normalization_version": 2,
        },
        output_path=tmp_path / "c.json",
    )
    assert first != second
    assert first != third
```

- [ ] **Step 2: Implement safe data-contract APIs**

Required interfaces:

- `normalize_coughvid_status(value: object) -> str`
- `build_audited_coughvid_index(raw_source: Path, *, label_column: str, dataset_release_id: str, source_manifest_sha256: str, require_audio: bool = True) -> pd.DataFrame`
- `qualify_identifiers(frame: pd.DataFrame) -> pd.DataFrame`
- `audit_coughvid_labels(frame: pd.DataFrame, *, prior: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]`
- `freeze_data_contracts(*, source_root: Path, audit_root: Path, source_paths: tuple[Path, ...], label_audits: tuple[Path, ...], contract_metadata: Mapping[str, object], output_path: Path) -> str`
- `assert_prediction_key_contract(frame: pd.DataFrame, *, repeated: bool) -> None`
- `aggregate_to_participant(frame: pd.DataFrame) -> pd.DataFrame`

Use exact ordered aliases and no broad COVID substring fallback. Reject a label
column that is missing or not explicitly allow-listed by the frozen config. Add immutable
`participant_key=dataset::participant_id` and
`recording_key=dataset::recording_id`. The prediction key must include run ID,
protocol, fold, dataset, participant key, split, modality, model, checkpoint
hash, and representation. Repeated predictions without `fold` fail.

`freeze_data_contracts` hashes a canonical, sorted manifest containing every
source relative path, byte size, SHA-256, release identifier, selected label
column, normalized-label audit hash, audio existence count, and exclusion count.
Required sources include the Coswara and COUGHVID metadata/audio manifests and
the full merged ComParE+IS10 feature table used by the aligned comparator. For
that table, also store row count, ordered column names and dtypes, feature-bank
configuration, and generating-code provenance; hashing only a selected top-800
derivative is insufficient because ranking is refitted inside each fold.
The required `contract_metadata` mapping contains dataset/release identifiers,
selected label columns, label-normalization version, source-manifest SHA-256,
and eligibility-policy version; it is recursively key-sorted and must be JSON
serializable. Source and audit paths are stored only as normalized POSIX-style
paths relative to their explicit roots; a path escaping its declared root is
rejected. This prevents a checkout-location change from changing the scientific
contract while still detecting a renamed or substituted source. Missing
required keys fail closed.
It writes the manifest atomically and returns the manifest SHA-256. A changed
metadata byte, source path set, release identifier, label mapping, or audit row
must change the freeze hash.

The HST orchestrator and aligned comparator must call
`build_audited_coughvid_index` directly from raw metadata and must not import
the legacy `build_coughvid_index` or `normalize_coughvid_label`. Add a static
dependency test for that rule. All corrected external metadata/features/metrics
are run-local new artifacts; no existing result file is overwritten.

- [ ] **Step 3: Add a raw-to-normalized audit gate**

Read `status`, `status_SSL`, and every available physician/expert annotation
column before filtering. Save their provenance, every unique raw value,
normalized value, row count, supervised count, and exclusion reason, plus
pairwise disagreement/coverage tables. The primary continuity analysis uses the
configured `status_SSL` column from the checksum-pinned COUGHVID-v3 release and
labels it semi-supervised, not PCR-confirmed. A separate `status` sensitivity
uses raw self-report. An HST-paper cohort reconstruction is created only if
non-COVID status, cough symptoms, physician/expert provenance, SNR, and cough
events are independently supported; otherwise write a skipped audit row.
Join prior external metadata by `recording_key` and count changed labels. If any
supervised label changed, mark prior external metrics `invalidated=True`; the
new run must regenerate comparator and HST external evaluation using corrected
labels.

- [ ] **Step 4: Test participant-level aggregation and key isolation**

Assert that multiple recordings do not give a participant extra metric weight,
that equal raw IDs from Coswara and COUGHVID do not collide, and that fold 1 and
fold 2 predictions cannot be averaged together accidentally.

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest tests/test_hst_data_contracts.py -q
git add src/covid_rars/hst_data_contracts.py tests/hst_test_helpers.py tests/test_hst_data_contracts.py
git commit -m "Add safe HST data and label contracts"
```

## Task 1: Pin Source And Configuration

**Files:**
- Create: `.gitmodules`
- Track: `HST/`
- Create: `configs/hst_reliability.json`
- Create: `requirements-hst.txt`
- Test: `tests/test_hst_checkpoint.py`

- [ ] **Step 1: Convert the existing clone to a pinned submodule**

```bash
# Run from the repository root.
mkdir -p .cache/hst/checkpoints
cp HST/model/imagenet_weights/hst_small_imagenet.pth .cache/hst/checkpoints/ 2>/dev/null || true
cp HST/model/imagenet_weights/hst_base_imagenet.pth .cache/hst/checkpoints/ 2>/dev/null || true
git submodule add --force https://github.com/icon-lab/HST.git HST
git -C HST checkout 7f94ad81e392da856c7aac6d364d036c28e26c32
git config -f .gitmodules submodule.HST.ignore untracked
git submodule status HST
```

Expected: output begins with the pinned commit. Local author checkpoint files
remain untracked inside the external clone and are ignored by the parent
submodule status. Cache copies are accepted only after Task 2 verifies hashes;
an invalid cache copy is replaced by a verified download.

- [ ] **Step 2: Write the failing configuration test**

```python
import json
from pathlib import Path


def test_hst_config_pins_source_and_primary_checkpoint() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/hst_reliability.json").read_text())
    assert config["source"]["commit"] == "7f94ad81e392da856c7aac6d364d036c28e26c32"
    small = config["checkpoints"]["hst_small_imagenet"]
    assert small["sha256"] == "e7086d1b87d598120296b9a1b5f094c7587cb06f50bf609a4ca13badc95e3112"
    assert small["size_bytes"] == 111266629
    assert small["google_drive_file_id"] == "1MHSIBpM3-pa2xXKSrk5oEDTvlhIaC_M3"
    base = config["checkpoints"]["hst_base_imagenet"]
    assert base["sha256"] == "f39f001d5f8cd90cb78d45612486202a4ea280e23df0b2c1d6ce35d96b30cce4"
    assert base["size_bytes"] == 197063145
    assert base["google_drive_file_id"] == "1jol7869ixS77FyoAXzb_m3oJGTtKuOVO"
    assert config["experiment"]["primary_model"] == "hst_base"
```

- [ ] **Step 3: Confirm the test fails**

```bash
python -m pytest tests/test_hst_checkpoint.py::test_hst_config_pins_source_and_primary_checkpoint -q
```

Expected: FAIL because the configuration is absent.

- [ ] **Step 4: Create the frozen configuration**

The JSON must contain the exact source/checkpoint values from the design; class
mapping `{negative: 0, positive: 1}`; every constructor argument; all
spectrogram numeric choices; COUGHVID release identifier and metadata schema,
with the runtime source-manifest checksum supplied by the audited
`data_contracts` stage; primary
`label_column="status_SSL"`, semi-supervised label provenance, and raw-status
sensitivity; training-only hierarchical class/participant/recording sampling;
image transforms;
training values `effective_batch_size=8`, `max_epochs=100`,
`learning_rate=1e-5`, `weight_decay=1e-8`, `gradient_clip_norm=0.1`, fixed-0.5
participant-level validation AUROC selection with participant AUPRC and NLL tie-breaks,
`train_all_epochs=true`, and no confirmatory early
stopping; AMP acceptance tolerances `max_abs_probability=0.01`,
`relative_loss=0.01`, and `max_skipped_updates=0`; complete-case uniform primary fusion; and project seeds
`[1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002]` with provenance pointing to the
released HST baseline scripts that contain the list.

Freeze `res_type="soxr_hq"`, periodic Hann window, `htk=false`,
`mel_norm="slaney"`, `dtype="float32"`, high-frequency-first array rows with
display origin `upper`, bilinear antialiased resize, and image augmentation before mean/std
normalization. Also pin the source revision and checksum of any official
COUGHVID SNR/event-segmentation code. If it cannot be pinned, set the sensitivity
identifier to `cough_event_reconstruction` rather than `paper_exact`.

Freeze these architecture fields exactly:

```json
{
  "img_size": 224,
  "h": 4,
  "img_channel": 3,
  "num_labels": 2,
  "d": 96,
  "num_blocks": [1, 1, 9, 1],
  "num_attention_heads": [3, 6, 12, 24],
  "win_size": 7,
  "mlp_ratio": 4.0,
  "use_bias": true,
  "dropout_rate": 0.0,
  "attn_dropout_rate": 0.0,
  "drop_path_rate": 0.1,
  "use_checkpoint": false
}
```

OneCycleLR is frozen to cosine annealing, `pct_start=0.3`, `div_factor=25`,
`final_div_factor=10000`, and scheduler steps per successful optimizer update.
The reporting block freezes `bootstrap_replicates=1000`,
`bootstrap_seed=42`, `confidence_level=0.95`, `ece_bins=10`,
`fixed_sensitivity=0.90`, `decision_thresholds=[0.05, 0.10, ..., 0.50]`,
and probability clipping epsilon `1e-6`. These are scientific configuration
inputs and therefore contribute to the run and stage hashes.

- [ ] **Step 5: Create isolated dependencies**

Install the matched PyTorch family from the official CUDA 12.8 index before the
ordinary requirements file:

```bash
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-hst.txt
```

`requirements-hst.txt` contains:

```text
-r requirements-gpu.txt
timm==0.9.16
ml-collections==1.1.0
threadpoolctl==3.6.0
psutil==7.0.0
nbformat==5.10.4
gdown==5.2.0
numpy==2.4.6
pandas==3.0.3
scikit-learn==1.9.0
xgboost==3.2.0
lightgbm==4.6.0
catboost==1.2.10
imbalanced-learn==0.14.2
```

Preflight asserts the exact torch/vision/audio versions and CUDA build and saves
the installed versions of every package above plus librosa, soundfile,
matplotlib, Pillow, SciPy, and joblib. It saves `pip freeze`, platform data, and
library versions into the run manifest. The
driver's reported CUDA 13.2 is not confused with PyTorch's bundled CUDA 12.8
runtime.

- [ ] **Step 6: Run the configuration test**

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
# Run from the repository root.
git add .gitmodules HST configs/hst_reliability.json requirements-hst.txt tests/test_hst_checkpoint.py
git commit -m "Pin official HST source and experiment configuration"
```

## Task 2: Verified Checkpoint Loading

**Files:**
- Create: `src/covid_rars/hst_checkpoint.py`
- Modify: `tests/test_hst_checkpoint.py`

- [ ] **Step 1: Add failing checksum and head tests**

```python
import pytest


def test_verify_file_rejects_wrong_hash(tmp_path) -> None:
    from covid_rars.hst_checkpoint import verify_file
    path = tmp_path / "weights.pth"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_file(path, expected_size=16, expected_sha256="0" * 64)


def test_load_verified_model_reinitializes_only_head() -> None:
    torch = pytest.importorskip("torch")
    from tests.hst_test_helpers import (
        expected_backbone_parameter_count,
        expected_base_architecture,
        official_hst_paths,
    )
    from covid_rars.hst_checkpoint import load_verified_hst_model
    hst_base_checkpoint, hst_repo = official_hst_paths()
    model, audit = load_verified_hst_model(
        model_name="hst_base",
        checkpoint_path=hst_base_checkpoint,
        hst_repo=hst_repo,
        seed=42,
    )
    assert set(audit["missing_keys"]) == {"head.bias", "head.weight"}
    assert set(audit["unexpected_keys"]) == set()
    assert audit["head_reinitialized"] is True
    assert audit["architecture"] == expected_base_architecture()
    assert audit["backbone_parameter_count"] == expected_backbone_parameter_count()
    logits = model(torch.zeros(2, 3, 224, 224))
    assert tuple(logits.shape) == (2, 2)
    assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Confirm failure**

Expected: FAIL because `hst_checkpoint.py` is absent.

- [ ] **Step 3: Implement this public API**

Required interfaces, including exact signatures:

- `sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str`
- `verify_file(path: Path, *, expected_size: int, expected_sha256: str) -> None`
- `verify_hst_source(hst_repo: Path, expected_commit: str) -> str`
- `download_verified_checkpoint(*, google_drive_file_id: str, destination: Path, expected_size: int, expected_sha256: str) -> Path`
- `load_verified_hst_model(*, model_name: str, checkpoint_path: Path, hst_repo: Path, seed: int) -> tuple[object, dict[str, object]]`

`load_verified_hst_model` must use `weights_only=True`, remove exactly the two
head keys, require every backbone key, initialize the new head with truncated
normal standard deviation 0.02 and zero bias, and return an audit containing
commit, checkpoint hash, tensor count, parameter count, and key audit.
Compare missing/unexpected keys as sets, not serialization order. Instantiate
the exact frozen constructor and assert representative tensor shapes plus the
full backbone parameter count before accepting a checkpoint.

Use the official README file IDs `1MHSIBpM3-pa2xXKSrk5oEDTvlhIaC_M3` (Small)
and `1jol7869ixS77FyoAXzb_m3oJGTtKuOVO` (Base) through `gdown`; do not treat a
Google Drive HTML response as a checkpoint. Download to a unique temporary
file, verify size/hash, fsync, and atomically replace the cache destination.

- [ ] **Step 4: Run focused tests**

```bash
python -m pytest tests/test_hst_checkpoint.py -q
```

Expected: PASS on Ubuntu; PyTorch-dependent tests skip only where PyTorch is
not installed.

- [ ] **Step 5: Commit**

```bash
git add src/covid_rars/hst_checkpoint.py tests/test_hst_checkpoint.py
git commit -m "Add verified HST checkpoint loader"
```

## Task 3: Deterministic Spectrogram Cache

**Files:**
- Create: `src/covid_rars/hst_spectrograms.py`
- Create: `scripts/hst_preprocess_worker.py`
- Create: `tests/test_hst_spectrograms.py`

- [ ] **Step 1: Write failing preprocessing tests**

```python
import numpy as np


def test_paper_logmel_is_deterministic_and_single_channel_cache() -> None:
    from covid_rars.hst_spectrograms import HSTSpectrogramConfig, waveform_to_hst_image
    sr = 22050
    t = np.arange(sr * 3) / sr
    y = 0.2 * np.sin(2 * np.pi * 440 * t)
    config = HSTSpectrogramConfig.paper_default()
    first = waveform_to_hst_image(y, sr, config)
    second = waveform_to_hst_image(y, sr, config)
    assert first.shape == (224, 224)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert float(first.std()) > 0.01
    assert first.dtype == np.float32


def test_post_trim_audio_must_be_strictly_longer_than_two_seconds() -> None:
    from covid_rars.hst_spectrograms import HSTSpectrogramConfig, preprocess_recording
    config = HSTSpectrogramConfig.paper_default()
    for n_samples in (22050, 2 * 22050):
        result = preprocess_recording(np.ones(n_samples), 22050, config)
        assert result.eligible is False
        assert result.reason == "post_trim_duration_not_above_2_seconds"
```

- [ ] **Step 2: Confirm failure**

Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement immutable configuration and result types**

```python
@dataclass(frozen=True)
class HSTSpectrogramConfig:
    representation_id: str
    sample_rate: int
    resample_type: str
    dtype: str
    trim_top_db: float
    trim_frame_length: int
    trim_hop_length: int
    minimum_duration_seconds: float
    n_fft: int
    win_length: int
    window: str
    hann_periodic: bool
    noverlap: int
    hop_length: int
    n_mels: int
    mel_htk: bool
    mel_norm: str
    fmin: float
    fmax: float
    power: float
    center: bool
    db_ref: str
    top_db: float
    image_size: int
    resize_interpolation: str
    resize_antialias: bool
    array_row_zero_frequency: str
    display_origin: str
    augment_before_normalize: bool
    normalization_mean: float
    normalization_std: float

    @classmethod
    def paper_default(cls) -> "HSTSpectrogramConfig":
        return cls(
            representation_id="paper_logmel_224",
            sample_rate=22050,
            resample_type="soxr_hq",
            dtype="float32",
            trim_top_db=60.0,
            trim_frame_length=2205,
            trim_hop_length=1102,
            minimum_duration_seconds=2.0,
            n_fft=2048,
            win_length=2048,
            window="hann",
            hann_periodic=True,
            noverlap=128,
            hop_length=1920,
            n_mels=224,
            mel_htk=False,
            mel_norm="slaney",
            fmin=0.0,
            fmax=11025.0,
            power=2.0,
            center=False,
            db_ref="max",
            top_db=80.0,
            image_size=224,
            resize_interpolation="bilinear",
            resize_antialias=True,
            array_row_zero_frequency="high",
            display_origin="upper",
            augment_before_normalize=True,
            normalization_mean=0.5,
            normalization_std=0.5,
        )


@dataclass(frozen=True)
class PreprocessResult:
    eligible: bool
    reason: str
    image: np.ndarray | None
    original_duration_seconds: float
    trimmed_duration_seconds: float
```

- [ ] **Step 4: Implement cache functions**

Required interfaces:

- `waveform_to_hst_image(y: np.ndarray, sr: int, config: HSTSpectrogramConfig) -> np.ndarray`
- `preprocess_recording(y: np.ndarray, sr: int, config: HSTSpectrogramConfig) -> PreprocessResult`
- `preprocess_audio_path(path: Path, config: HSTSpectrogramConfig) -> PreprocessResult`
- `build_hst_spectrogram_cache(metadata: pd.DataFrame, *, output_dir: Path, config: HSTSpectrogramConfig, force: bool = False) -> pd.DataFrame`
- Deferred interface design only (not required by the primary run):
  `estimate_coughvid_snr`, `segment_cough_events`, and
  `build_hst_event_cache`. These may be implemented only after a versioned,
  checksum-pinned algorithm is approved.

The cache index records qualified IDs, dataset, modality, corrected label,
eligibility, reason, durations, source path/size/mtime/SHA-256, decode attempt,
tensor SHA-256, and preprocessing hash. The base cache is built before final
split manifests, so it contains no mutable `split` assumption. Store one
grayscale `float32` 224 x 224 array; replicate to three channels in the loader.
JPEG is never a training input.

Duration eligibility is a strict comparison: a recording is eligible only when
post-trim duration is greater than `minimum_duration_seconds`; equality at 2.0
seconds is ineligible with reason
`post_trim_duration_not_above_2_seconds`.

The event cache remains a deferred sensitivity contract. The pinned official
HST source does not expose a checksum-verifiable event/SNR algorithm, so the
primary implementation records `deferred_missing_checksum_pinned_algorithm`
and does not manufacture a reconstruction. A future accepted extension must
record source revision/checksum, SNR, event mapping, exclusions, and apply one
algorithm identically to source and target before aggregating event to recording
to participant.

Claims use atomic `O_CREAT|O_EXCL` and contain owner PID/start identity plus
source/config hashes. A parent supervisor launches a bounded set of direct,
persistent `hst_preprocess_worker.py` subprocesses, with thread-limit environment
variables set before numeric imports. It assigns one recording at a time,
enforces a hard deadline, and on timeout kills the worker process group, records
the attempt, launches a replacement, and permits one logged retry. Workers never
spawn child pools. A worker writes a unique temporary file in the
destination directory, flushes and fsyncs it, reload-validates shape/dtype/
finite values/checksum, calls `os.replace`, then fsyncs the directory. Workers
write one atomic per-record result fragment but never the shared index; the
parent can reconstruct and serialize the index after interruption. Reject
constant and non-finite arrays. A cache hit is accepted only after source and
tensor hashes match.

- [ ] **Step 5: Test the released-code sensitivity representation**

Require `released_linear_specgram_224` to be deterministic, finite, 224 x 224,
and to have a different configuration hash from `paper_logmel_224`. Pin
Matplotlib/Pillow versions, render dimensions, DPI, margins, axes, colormap, and
interpolation. Commit a small synthetic waveform plus expected array hash as a
golden fixture for both representations. The paper reconstruction fixture also
pins `soxr_hq`, periodic Hann, Slaney-Mel, `float32`, frequency orientation,
antialiasing, and augmentation-before-normalization order.

- [ ] **Step 6: Test dataset loading and augmentation separation**

Assert the loader replicates the grayscale array into exactly three identical
channels and normalizes each with mean/std 0.5. Assert image rotation/flip and
balanced sampling run only for training, are stateless under
`(seed, fold, epoch, recording_key, draw_id)`, give duplicate draws distinct
augmentations, and never alter validation/test tensors.
Implement augmentation with local seeded generators and torchvision functional
rotation/flip calls, not stateful `Compose` random transforms that depend on
worker scheduling.

- [ ] **Step 7: Run tests and commit**

```bash
python -m pytest tests/test_hst_spectrograms.py -q
git add src/covid_rars/hst_spectrograms.py scripts/hst_preprocess_worker.py tests/test_hst_spectrograms.py
git commit -m "Add deterministic HST spectrogram cache"
```

## Task 4: Freeze Participant Manifests

**Files:**
- Create: `src/covid_rars/hst_protocols.py`
- Create: `tests/test_hst_protocols.py`

- [ ] **Step 1: Write leakage tests**

```python
def test_hst_folds_are_participant_disjoint() -> None:
    from tests.hst_test_helpers import PRESPECIFIED_HST_REPO_SEEDS, example_multimodal_cache_index
    from covid_rars.hst_protocols import build_protocol_matched_hst_manifest, audit_hst_manifest
    manifest = build_protocol_matched_hst_manifest(
        example_multimodal_cache_index(), seeds=PRESPECIFIED_HST_REPO_SEEDS
    )
    audit = audit_hst_manifest(manifest)
    assert manifest["fold"].nunique() == 10
    assert audit["participant_overlap_count"].eq(0).all()
    assert audit["mixed_label_participant_count"].eq(0).all()


def test_split_policy_manifests_share_cohort_counts_and_labels() -> None:
    from tests.hst_test_helpers import example_dated_cache_index
    from covid_rars.hst_protocols import build_split_policy_contrast_manifests

    mixed, chronological = build_split_policy_contrast_manifests(example_dated_cache_index())
    mixed_people = mixed[["participant_key", "label_binary", "split"]].drop_duplicates()
    chronological_people = chronological[["participant_key", "label_binary", "split"]].drop_duplicates()
    assert mixed_people["participant_key"].is_unique
    assert chronological_people["participant_key"].is_unique
    assert set(mixed_people["participant_key"]) == set(chronological_people["participant_key"])
    for split in ("train", "validation", "test"):
        left = mixed_people[mixed_people["split"].eq(split)]["label_binary"].value_counts().to_dict()
        right = chronological_people[chronological_people["split"].eq(split)]["label_binary"].value_counts().to_dict()
        assert left == right


def test_hst_task2_like_cohort_requires_cough_symptom_in_both_classes() -> None:
    from tests.hst_test_helpers import PRESPECIFIED_HST_REPO_SEEDS, example_symptom_cache_index
    from covid_rars.hst_protocols import build_hst_task2_like_cough_manifest

    manifest = build_hst_task2_like_cough_manifest(
        example_symptom_cache_index(), seeds=PRESPECIFIED_HST_REPO_SEEDS
    )
    assert manifest["modality"].eq("cough").all()
    assert manifest["cough_symptom_present"].eq(True).all()


def test_external_rows_never_enter_source_training() -> None:
    from tests.hst_test_helpers import example_source_external_cache_index, source_fold_manifest
    from covid_rars.hst_protocols import build_external_hst_manifest
    manifest = build_external_hst_manifest(
        example_source_external_cache_index(), source_fold_manifest()
    )
    external = manifest[manifest["dataset"].eq("coughvid")]
    assert external["split"].eq("external_test").all()


def test_protocols_change_manifest_not_scientific_configuration() -> None:
    from tests.hst_test_helpers import two_protocol_configs
    from covid_rars.hst_protocols import scientific_configuration_fingerprint
    conventional_track_config, deployment_track_config = two_protocol_configs()
    conventional = scientific_configuration_fingerprint(conventional_track_config)
    deployment = scientific_configuration_fingerprint(deployment_track_config)
    assert conventional == deployment
    assert conventional_track_config.manifest_path != deployment_track_config.manifest_path
```

- [ ] **Step 2: Implement manifest APIs**

Required interfaces:

- `build_protocol_matched_hst_manifest(cache_index: pd.DataFrame, *, seeds: tuple[int, ...], test_fraction: float = 0.2, validation_fraction_of_remaining: float = 0.125) -> pd.DataFrame`
- `build_hst_task2_like_cough_manifest(cache_index: pd.DataFrame, *, seeds: tuple[int, ...]) -> pd.DataFrame`
- `build_split_policy_contrast_manifests(cache_index: pd.DataFrame, *, train_fraction: float = 0.6, validation_fraction: float = 0.2, candidate_count: int = 1000, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]`
- `build_common_late_test_manifests(cache_index: pd.DataFrame, *, candidate_count: int = 1000, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]`
- `build_reverse_temporal_hst_manifest(cache_index: pd.DataFrame) -> pd.DataFrame`
- `build_external_hst_manifest(cache_index: pd.DataFrame, source_manifest: pd.DataFrame) -> pd.DataFrame`
- `intersect_representation_eligibility(*indices: pd.DataFrame) -> pd.DataFrame`
- `audit_hst_manifest(manifest: pd.DataFrame) -> pd.DataFrame`
- `scientific_configuration_fingerprint(config: object) -> str`

Manifests are built from checksum-verified eligible cache rows, never raw
metadata. Save one row per fold and eligible recording, including its qualified
participant, recording key, modality, label, and split. Every recording of one
participant must receive the same split within a repetition. Reject mixed
labels or multiple splits for one participant in a repetition.
Track A uses the prespecified ten seeds found in the released HST baseline
scripts and is named repeated holdout, not ordinary 10-fold CV. The manifest
records that seed provenance; it does not imply that `train.py` published a
ten-seed list.
The split is performed in two participant-level steps: reserve 20% for test,
then reserve 12.5% of the remaining 80% for validation, yielding approximately
70/10/20 overall. Assertions check the realized participant and class counts.

Build the secondary Task-2-like cough cohort from an explicit, tested symptom
parser: positive and negative participants must both have a recorded cough
symptom, missing/ambiguous symptom values are excluded, and only cough modality
rows remain. Save the raw symptom fields and exclusion audit. Label it
`hst_task2_like_cough`; never call it Cambridge Task 2 or use it to replace the
project-target cohort after seeing results.

Participant date is the earliest valid UTC-normalized recording timestamp.
Exclude and audit unparseable dates; sort chronological ties by timestamp then
qualified participant key and report boundary ties/date ranges.

The reverse temporal manifest uses the same eligible participant universe and
ordering but assigns latest 60% to train, the preceding 20% to validation, and
earliest 20% to test. It reports realized class/prevalence counts and is labeled
a reverse-direction sensitivity rather than a deployment simulation. Both
split-policy models and the reverse model use training seed 42; the 1,000
candidate-assignment seeds are used only to construct manifests.

For the split-policy pair, the chronological manifest fixes split sizes and
participant-level class counts. Generate candidates using seeds
`random_state + candidate_index`. Score each candidate by the maximum, across
splits, of absolute standardized mean difference in participant month ordinal
and Kolmogorov-Smirnov distance from the full date distribution; smallest seed
breaks exact ties. Save every score. Assert identical eligible universe, split
sizes, and class counts after reducing manifests to one row per participant.

Also build a common-latest-test control: both manifests use the exact latest 20%
test participants; chronological versus date-balanced assignment is applied
only to the remaining 80% train/validation pool with matched class counts. The
date-balanced source assignment uses the same 1,000-candidate month-SMD/KS
objective and deterministic tie-breaking as the primary split-policy reference,
restricted to train and validation rows; no audio outcome enters the choice.

For external transfer, attach every eligible COUGHVID cough participant in the
primary full-recording `status_SSL` cohort as `external_test` to every Track-A
source fold. The raw-`status` sensitivity relabels those frozen probabilities on
the supervised overlap. The SNR/event cohort remains explicitly deferred. Store
`label_source`, `label_provenance`, `dataset_release_id`,
`source_manifest_sha256`, and `preprocessing_variant` in every row. Assert no external row appears in
train/validation/test and no source/target qualified ID collides.

The scientific fingerprint includes architecture, source checkpoint,
preprocessing, augmentation, optimizer, stopping rule, participant aggregation,
fusion, thresholding, and metric settings, but excludes the manifest path and
protocol label.

- [ ] **Step 3: Test parity with existing protocol folds**

For each official seed, Track-A participant sets must exactly match the
literature-aligned splitter. Also assert every final HST-versus-comparator
manifest is the exact shared post-preprocessing intersection and publish
representation-specific exclusion tables by dataset, protocol, split, label,
and modality.

- [ ] **Step 4: Run tests and commit**

```bash
python -m pytest tests/test_hst_protocols.py -q
git add src/covid_rars/hst_protocols.py tests/test_hst_protocols.py
git commit -m "Add frozen HST participant manifests"
```

## Task 5: Resumable Participant-Level Training

**Files:**
- Create: `src/covid_rars/hst_training.py`
- Create: `tests/test_hst_training.py`

- [ ] **Step 1: Write aggregation and resume tests**

```python
def test_participant_probabilities_do_not_weight_extra_recordings() -> None:
    from tests.hst_test_helpers import make_recording_predictions
    from covid_rars.hst_training import aggregate_recording_predictions
    recordings = make_recording_predictions(
        participant_ids=["p1", "p1", "p2"], probabilities=[0.2, 0.8, 0.7]
    )
    participants = aggregate_recording_predictions(recordings)
    assert participants.set_index("participant_key").loc["coswara::p1", "probability"] == 0.5


def test_resume_continues_after_last_completed_epoch(tmp_path) -> None:
    from tests.hst_test_helpers import make_tiny_hst_model, make_tiny_loaders
    from covid_rars.hst_training import HSTTrainingConfig, train_hst_fold
    config = HSTTrainingConfig(
        pilot_freeze_hash="test-freeze",
        data_contracts_freeze_hash="test-data-freeze",
        dependency_lock_hash="test-environment-freeze",
        accepted_environment_lock_hash="test-environment-freeze",
        physical_batch_size=2,
        gradient_accumulation=4,
        amp=False,
        max_epochs=2,
    )
    first = train_hst_fold(make_tiny_hst_model(), make_tiny_loaders(), config, tmp_path, stop_after_epoch=1)
    assert first.last_epoch == 1
    resumed = train_hst_fold(make_tiny_hst_model(), make_tiny_loaders(), config, tmp_path, resume=True)
    assert resumed.last_epoch == 2
    assert resumed.resumed_from_epoch == 1


def test_hierarchical_sampler_balances_classes_not_recording_counts() -> None:
    from tests.hst_test_helpers import imbalanced_multirecording_training_rows
    from covid_rars.hst_training import build_hierarchical_epoch_draw_plan

    plan = build_hierarchical_epoch_draw_plan(
        *imbalanced_multirecording_training_rows(), fold=1, modality="cough",
        epoch=1, seed=1
    )
    assert plan.groupby("label_binary").size().nunique() == 1
    assert plan["draw_id"].is_unique
    for _, class_draws in plan.groupby("label_binary"):
        per_person = class_draws.groupby("participant_key").size()
        assert per_person.max() - per_person.min() <= 1
```

- [ ] **Step 2: Implement training APIs**

```python
@dataclass(frozen=True)
class HSTTrainingConfig:
    pilot_freeze_hash: str | None
    data_contracts_freeze_hash: str
    dependency_lock_hash: str
    accepted_environment_lock_hash: str | None
    physical_batch_size: int
    gradient_accumulation: int
    amp: bool
    max_epochs: int = 100
    effective_batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-8
    gradient_clip_norm: float = 0.1
    train_all_epochs: bool = True
    early_stopping_min_epochs: int = 40
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.001
    scheduler_pct_start: float = 0.3
    scheduler_div_factor: float = 25.0
    scheduler_final_div_factor: float = 10000.0
    scheduler_anneal_strategy: str = "cos"
    epoch_selection_threshold: float = 0.5
    balance_training_classes: bool = True
    amp_probability_tolerance: float = 0.01
    amp_relative_loss_tolerance: float = 0.01
    amp_max_skipped_updates: int = 0


```

`pilot_freeze_hash=None` is valid only for smoke and resource-pilot jobs, which
are explicitly non-confirmatory. Full mode must inject the accepted pilot hash
and rejects `None`; this avoids circularly requiring a pilot result before the
pilot can run. The data-contract hash exists before all training and is required
in every mode. The actual dependency-lock hash is required in every mode;
`accepted_environment_lock_hash=None` is permitted only before pilot acceptance,
while full mode requires it to equal the actual dependency-lock hash.

Required interfaces:

- `build_hierarchical_epoch_draw_plan(cache_index: pd.DataFrame, manifest: pd.DataFrame, *, fold: int, modality: str, epoch: int, seed: int) -> pd.DataFrame`
- `make_hst_dataloaders(cache_index: pd.DataFrame, manifest: pd.DataFrame, *, fold: int, modality: str, physical_batch_size: int, num_workers: int, seed: int) -> dict[str, object]`
- `train_hst_fold(model: object, loaders: dict[str, object], config: HSTTrainingConfig, run_dir: Path, *, resume: bool = True, stop_after_epoch: int | None = None) -> HSTFoldResult`
- `predict_hst_split(model: object, loader: object, *, split: str, fold: int, modality: str) -> pd.DataFrame`
- `aggregate_recording_predictions(predictions: pd.DataFrame) -> pd.DataFrame`

Concatenate every validation batch, average recording probabilities within
participant, assert one participant label, and compute checkpoint-selection
metrics on those participant rows. Training cross-entropy remains draw-level.
Always export probability column 1 under the frozen
`{negative: 0, positive: 1}` mapping. For the confirmatory run, complete all 100
epochs, select by participant-level validation AUROC with participant-level
AUPRC, lower participant-level NLL, and then earlier epoch as deterministic tie-breaks, freeze the operating threshold after epoch
selection, evaluate test once after loading verified `best.pt`, and write atomic
checkpoints containing optimizer, scheduler, scaler, epoch-draw-plan hash,
`next_consumed_batch_index`, epoch, and RNG states.

The hierarchical sampler chooses class uniformly, participant uniformly within
class, and one eligible recording uniformly within participant. It records all
draw keys and is absent from validation/test. Before each epoch it
materializes an immutable ordered draw table containing class, participant key,
the already selected recording key, `draw_id`, and augmentation seed for every
occurrence, including duplicated minority recordings. The DataLoader consumes
that table with `shuffle=False`; there is no second sampling layer. Derive
augmentation randomness statelessly from fold/epoch/recording key/draw ID.
Checkpoints store
the draw-table hash and `next_consumed_batch_index`; they never store or trust a
prefetched sampler cursor. Uninterrupted and resumed runs must therefore produce
identical sample order, transforms, learning-rate history, weights, and
predictions.

Epoch length is fixed before training from participant counts, not tuned from
metrics: each class contributes `max(n_negative_participants,
n_positive_participants)` draws. A seeded round-robin permutation makes every
participant in the larger class appear once per epoch and repeats participants
in the smaller class as evenly as possible; the recording for each participant
is then sampled uniformly. This defines balancing without allowing recording-
rich participants or an arbitrary draw count to change optimization exposure.

The article gives only a qualitative stop rule within epochs 1-100, while the
released trainer completes 100 epochs and reports the maximum-validation-AUROC
epoch; neither defines a reproducible patience matching the article.
OneCycleLR also peaks at approximately epoch 30. The confirmatory reconstruction
therefore completes 100 epochs and selects the saved epoch by participant-level
validation AUROC with participant AUPRC, participant NLL, and then earlier epoch as
tie-breaks. A separately labeled runtime sensitivity may set patience 10 only
after at least 40 completed epochs; it cannot replace or select the confirmatory
result. Full confirmatory mode rejects `train_all_epochs=False`.
The released-code renderer remains available for a separately accepted optional
extension, but no released-code model job is included in the primary 50-job
run. If later activated before outcomes, it must use value clipping and maximum
validation AUROC as in `HST/train.py` and remain separate from the paper-text
confirmatory branch.

Validate that `physical_batch_size * gradient_accumulation ==
effective_batch_size` for ordinary groups. For each group, multiply each mean
micro-batch loss by its sample count and accumulate gradients. Flush a final
incomplete group at epoch end rather than dropping it. At each update boundary,
unscale AMP gradients once when AMP is active, then divide every gradient by the
group's actual accumulated sample count; check finiteness, clip norm, step
scaler/optimizer, then advance OneCycleLR only
after a successful optimizer update. Set scheduler steps from
`ceil(epoch_batches / gradient_accumulation)`. Add an equivalence test comparing
one effective update at physical batches 8, 4, and 2 with stochastic transforms
and dropout disabled, including a short final micro-batch.

- [ ] **Step 3: Test threshold provenance**

Require test operating-point metrics to use a validation-derived threshold;
AUROC/AUPRC always use raw probabilities. Add tests that swapping class-column
order fails, that validation/test labels never reach the sampler or loss, and
that a corrupt newest checkpoint falls back to the prior verified checkpoint.
Add an interruption-equivalence test with `num_workers > 0`, persistent workers,
prefetching, duplicated minority draws, and a stop at an optimizer-safe
mid-epoch boundary. Assert the resumed draw IDs, model checksum, learning-rate
history, and predictions exactly match the uninterrupted run.

- [ ] **Step 4: Run tests and commit**

```bash
python -m pytest tests/test_hst_training.py -q
git add src/covid_rars/hst_training.py tests/test_hst_training.py
git commit -m "Add resumable participant-level HST training"
```

## Task 5A: Resource-Aware Parallel Scheduler

**Files:**
- Create: `src/covid_rars/hst_parallel.py`
- Create: `tests/test_hst_parallel.py`
- Modify: `tests/hst_test_helpers.py`

- [ ] **Step 1: Write host-sizing and serialization tests**

```python
import pytest


def test_preprocess_workers_reject_candidates_that_break_live_reserve() -> None:
    from covid_rars.hst_parallel import ResourceSnapshot, choose_preprocess_workers

    workers = choose_preprocess_workers(
        snapshot=ResourceSnapshot(
            logical_cpus=24,
            cpu_affinity_count=24,
            mem_available_bytes=6 * 1024**3,
            cgroup_headroom_bytes=6 * 1024**3,
            parent_rss_bytes=1 * 1024**3,
            dev_shm_available_bytes=8 * 1024**3,
            swap_used_bytes=0,
        ),
        estimated_worker_bytes=700 * 1024**2,
        reserve_cpus=4,
        reserve_ram_bytes=4 * 1024**3,
        candidates=(1, 2, 4, 8, 12),
    )
    assert workers == 2


def test_single_gpu_queue_never_overlaps_training_jobs() -> None:
    from tests.hst_test_helpers import fake_gpu_jobs
    from covid_rars.hst_parallel import run_single_gpu_job_queue

    ledger = run_single_gpu_job_queue(fake_gpu_jobs(), device_count=1)
    assert ledger["concurrent_gpu_jobs"].max() == 1


def test_different_run_ids_cannot_share_one_gpu_uuid(tmp_path) -> None:
    from covid_rars.hst_parallel import acquire_gpu_execution_lease

    with acquire_gpu_execution_lease(tmp_path, gpu_uuid="GPU-test", run_id="run-a"):
        with pytest.raises(BlockingIOError):
            with acquire_gpu_execution_lease(tmp_path, gpu_uuid="GPU-test", run_id="run-b"):
                pass


def test_loader_choice_does_not_use_model_metrics() -> None:
    from tests.hst_test_helpers import loader_benchmark_rows
    from covid_rars.hst_parallel import select_dataloader_workers

    rows = loader_benchmark_rows()
    selected = select_dataloader_workers(rows)
    assert selected in {0, 2, 4, 8}
    assert "auroc" not in rows.columns
```

- [ ] **Step 2: Implement fixed-host resource APIs**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceSnapshot:
    logical_cpus: int
    cpu_affinity_count: int
    mem_available_bytes: int
    cgroup_headroom_bytes: int
    parent_rss_bytes: int
    dev_shm_available_bytes: int
    swap_used_bytes: int
```

Required interfaces:

- `capture_resource_snapshot() -> ResourceSnapshot`
- `choose_preprocess_workers(*, snapshot: ResourceSnapshot, estimated_worker_bytes: int, reserve_cpus: int, reserve_ram_bytes: int, candidates: tuple[int, ...]) -> int`
- `benchmark_preprocess_workers(metadata: pd.DataFrame, *, candidates: tuple[int, ...], sample_size: int, config: object) -> pd.DataFrame`
- `parallel_build_spectrograms(metadata: pd.DataFrame, *, workers: int, config: object, output_dir: Path) -> pd.DataFrame`
- `benchmark_dataloader_workers(cache_index: pd.DataFrame, *, candidates: tuple[int, ...], batches: int, batch_size: int) -> pd.DataFrame`
- `select_dataloader_workers(benchmark: pd.DataFrame) -> int`
- `run_single_gpu_job_queue(jobs: list[object], *, device_count: int = 1) -> pd.DataFrame`
- `build_deduplicated_job_plan(config: object, manifests: dict[str, pd.DataFrame]) -> pd.DataFrame`
- `acquire_gpu_execution_lease(lease_root: Path, *, gpu_uuid: str, run_id: str) -> ContextManager[GPULease]`

The known-host profile is i7-14700 with 24 exposed logical CPUs, approximately
19 GiB RAM, 8 GiB swap, and one NVIDIA T1000 with 8 GiB VRAM. Use live
`MemAvailable`, cgroup limits, CPU affinity, parent-plus-child RSS, `/dev/shm`,
and swap deltas; never substitute total installed RAM. Benchmark candidates
1/2/4/8/12 and return a resource error if none preserves all reserves.
Preprocessing uses the direct subprocess supervisor from Task 3, with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` supplied
in each subprocess environment before imports and
`threadpoolctl.threadpool_limits(1)` applied inside. PyTorch DataLoader workers
use `multiprocessing.get_context("spawn")`; they do not launch nested children.

Production GPU leases use
`$XDG_RUNTIME_DIR/covid_rars/hst_gpu/<gpu_uuid>.lock` (or the same-user
`/var/tmp` fallback), not a workspace-relative path, so separate clones cannot
overlap on the same device. The Ubuntu implementation holds `flock` on the open
descriptor for the entire CUDA job.

- [ ] **Step 3: Implement the HST-Base memory pilot**

At physical batch 8, launch FP32 and AMP in separate fresh processes for at
least 100 optimizer updates plus validation. Require finite loss, gradients,
parameters, and predictions; zero skipped optimizer updates; and free VRAM headroom
of `max(1 GiB, 15% total)` using allocator peaks and
`torch.cuda.mem_get_info`. On the same fixed evaluation batch, also require AMP
to remain within `0.01` maximum absolute probability difference and `1%`
relative loss difference from FP32. Select the faster valid precision at batch
8. Only if neither passes may the pilot repeat at batch 4, then batch 2. Freeze
the first viable batch level, selected precision, measured tolerances, and set
gradient accumulation to `8 // physical_batch_size`. If none passes, stop
instead of changing the model or image size.

For DataLoaders, transfer pinned tensors with `non_blocking=True`; enable
`persistent_workers=True` and `prefetch_factor=2` only when `num_workers > 0`.
Benchmark cold and warm cache behavior, discard the first 20 batches as warm-up,
and select on the remaining 180. Use timeouts and terminate failed workers.

Build the complete job plan before training. Its training key hashes sorted
train/validation rows containing participant key, recording key, label,
modality, tensor SHA-256, and split, plus architecture/checkpoint,
preprocessing, sampler, augmentation, optimizer, representation, and seed
configuration. External inference is a separate evaluation job keyed by the
verified source-checkpoint hash and full external-manifest hash. Reuse identical
source jobs; never train a COUGHVID-selected source model. Every job writes an
atomic state record containing status (`pending`, `running`, `success`,
`failed`, or `stopped`), attempt, final PID identity, heartbeat, checkpoint and
output checksums, and error text. Resume skips only checksum-valid successes and
recovers stale running jobs. Estimate full runtime from measured 20-, 50-, and
100-update pilots and write best/median/worst ETA.

The frozen plan contains exactly 50 training jobs: 30 repeated internal jobs
(ten each for cough, speech, and breath), ten Task-2-like cough jobs, eight
split-policy/common-late jobs (four protocols for cough and speech), and two
reverse-temporal jobs (cough and speech). COUGHVID inference reuses the ten
Track-A cough checkpoints and never creates a target-selected training job.

Each queue job owns its DataLoaders in `try/finally`. Before the next job, stop
persistent workers, release iterators and pinned batches, wait for worker PIDs,
and verify process count and RSS return within a configured tolerance. Abort if
workers or pinned-memory growth remain.

- [ ] **Step 4: Write the systems benchmark artifact**

Save candidate worker count, recordings or batches per second, parent/child RSS,
MemAvailable, swap and `/dev/shm` deltas, peak GPU allocation/reservation/free
memory, validity, rejection reason, host profile, and selected setting under the
run directory. Systems settings are selected only by validity and throughput,
never by AUROC or test performance. Record GPU temperature/utilization/clocks
every 60 seconds and refuse an unrelated compute process on the T1000.

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest tests/test_hst_parallel.py -q
git add src/covid_rars/hst_parallel.py tests/test_hst_parallel.py
git commit -m "Add resource-aware HST execution scheduler"
```

## Task 5B: Aligned ComParE+IS10 Comparator

**Files:**
- Create: `src/covid_rars/hst_comparators.py`
- Create: `tests/test_hst_comparators.py`

- [ ] **Step 1: Write same-manifest and participant-metric tests**

Construct a fixture where one participant has three recordings and another has
one. Assert training rows come only from the supplied manifest, validation/test
thresholds are based on participant-averaged validation probabilities, and each
test participant contributes exactly once to primary metrics. Assert comparator
prediction keys match HST keys through fold, protocol, dataset, participant,
split, modality, and representation-eligibility cohort.

- [ ] **Step 2: Implement the pre-specified comparator**

Required interfaces:

- `run_aligned_compare_is10(features: pd.DataFrame, manifest: pd.DataFrame, *, model_names: tuple[str, ...] = ("lightgbm_smote_f80", "svc_rbf_f60", "catboost_smote_f80", "xgboost_smote_f80"), selected_feature_k: int = 800, ranker: str = "lightgbm", selection_scope: str = "per_modality_mean", random_state: int = 42, optuna_trials: int = 0, ensemble_top_k: int = 5, selection_metric: str = "auroc") -> HSTComparatorResult`
- `aggregate_comparator_participants(predictions: pd.DataFrame) -> pd.DataFrame`
- `audit_comparator_alignment(hst_predictions: pd.DataFrame, comparator_predictions: pd.DataFrame) -> pd.DataFrame`

Feature ranking is fitted on training rows only in every repetition/protocol.
Use the full merged ComParE+IS10 table and call the existing
`compare_is10_rescue.rank_train_features` contract on each fold's training
subset with `selection_scope="per_modality_mean"`; do not reuse the globally materialized top-800 CSV in the primary
matched analysis.
The primary comparator reproduces the pre-HST final branch-bank logic: fit the
four frozen model families and retain the historical `ensemble_top_k=5`
configuration. Because that run supplies only four eligible candidates, the
effective ensemble must contain four members and be named
`top_4_validation_ensemble`; a test must fail if it is reported as top five or
contains a different member set. Select one candidate per modality by validation
AUROC, with validation AUPRC and model name as deterministic tie-breaks. Save the
selected model names, ranked candidate table, requested cap, and effective
ensemble size for every fold. The broader library-level `DEFAULT_MODEL_NAMES`
bank is not the primary comparator because the final-validation CLI that
generated the reported artifact explicitly overrode it with these four names.
If the broader bank is run, label it as a separate sensitivity and never replace
the primary result with it. HST outcomes cannot enter selection. A fixed
SVC/top-800 row is retained as a simpler sensitivity because it was the strongest
aggregate cough row in the prior paper-comparable CV. Primary model fitting uses
every training recording in the exact shared manifest and no recording outside
it. Primary threshold
selection and metrics occur after recording
probabilities are averaged within participant. Recording-level metrics are
secondary sensitivity rows and must not be labeled participant-level.

Use the exact shared `(recording_key, modality)` post-preprocessing intersection
for train, validation, and test in the primary paired table; assert identical
recording-key sets before participant aggregation. Participant-matched but
recording-unmatched and representation-specific full-cohort results are
secondary. External comparator evaluation uses corrected labels and the same
source folds; the primary matched model-family table also restricts HST and the
comparator to the exact eligible target `(recording_key, modality)` intersection.
Target labels never choose features, hyperparameters, threshold, or model family.
Every learned preprocessing operation remains inside the training partition:
the outer top-800 ranker, model-specific percentile filter, scaler, and any
SMOTE step are fitted on training rows only. A frozen `*_smote_*` comparator may
resample only its selected training matrix; validation and test rows are never
synthetically sampled and never fit those transforms.

- [ ] **Step 3: Add alignment failure tests**

Fail on missing fold keys, different recording or participant sets, changed
labels, duplicate participant predictions, target-domain fitting, or a threshold
derived from test/external rows.

- [ ] **Step 4: Run tests and commit**

```bash
python -m pytest tests/test_hst_comparators.py -q
git add src/covid_rars/hst_comparators.py tests/test_hst_comparators.py
git commit -m "Add participant-aligned HST comparator"
```

## Task 6: Validation-Only Multimodal And Hybrid Fusion

**Files:**
- Create: `src/covid_rars/hst_fusion.py`
- Create: `tests/test_hst_fusion.py`

- [ ] **Step 1: Write fusion-weight tests**

```python
def test_legacy_auprc_weights_are_validation_derived_and_normalized() -> None:
    from tests.hst_test_helpers import frame_with_modalities
    from covid_rars.hst_fusion import legacy_validation_auprc_weights
    metrics = frame_with_modalities(cough=0.80, breath=0.70, speech=0.50, prevalence=0.25)
    weights = legacy_validation_auprc_weights(metrics)
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["cough"] > weights["breath"] > weights["speech"]


def test_primary_uniform_fusion_is_complete_case() -> None:
    from tests.hst_test_helpers import participant_predictions_with_missing_breath
    from covid_rars.hst_fusion import fuse_uniform_complete_case
    fused = fuse_uniform_complete_case(participant_predictions_with_missing_breath())
    assert "coswara::missing_breath" not in set(fused["participant_key"])
```

- [ ] **Step 2: Implement fusion APIs**

Required interfaces:

- `legacy_validation_auprc_weights(validation_metrics: pd.DataFrame, *, floor: float = 0.01, reference: float = 0.5) -> dict[str, float]`
- `fuse_uniform_complete_case(predictions: pd.DataFrame) -> pd.DataFrame`
- `fuse_with_fixed_weights(predictions: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame`
- `fuse_available_modalities_sensitivity(predictions: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame`
- `fit_validation_logistic_stacker(validation_predictions: pd.DataFrame, *, random_state: int) -> object`
- `apply_validation_logistic_stacker(model: object, predictions: pd.DataFrame) -> pd.DataFrame`
- `build_four_branch_hybrid_inputs(hst_predictions: pd.DataFrame, compare_predictions: pd.DataFrame) -> pd.DataFrame`
- `run_hst_fusion_bank(hst_predictions: pd.DataFrame, compare_predictions: pd.DataFrame | None = None) -> HSTFusionResult`

Write fold-level validation metrics, raw/normalized weights, complete-case
counts, available modalities, and source families. Complete-case uniform mean
is the primary matched fusion and is applied identically to HST and comparator
branches. Construct one joint complete-case participant set across every
required HST and comparator branch, and assert exact participant-key equality
between fused outputs for each fold, split, and modality combination.
Legacy validation-AUPRC weighting and logistic stacking are prespecified
secondary analyses; available-modality renormalization is a sensitivity
analysis. None can replace the primary complete-case uniform result.

The primary project-domain bank uses cough and speech because that is the
project's strongest existing multimodal definition. The aligned comparator is
retrained on the same recording intersection and folds; the historical 0.897
row is used as a direct comparator only if its complete cohort definition is
exactly reproduced. Cough is the direct HST-paper modality anchor. Breathing and
cough+breath fusion are separately labeled secondary HST-paper-modality
sensitivities. Speech is always identified as a project extension absent from
the HST paper.

The legacy validation-weighted rule uses
`max(validation_AUPRC - 0.5, 0.01)` before normalization, exactly matching the
existing final code; it is not mislabeled as prevalence adjustment. The logistic
stack is class-balanced L2 logistic regression with `C=1.0`, `max_iter=2000`,
and `random_state=42`, fitted to complete-case validation rows. Uniform mean is
the primary method-neutral fusion; the two historical rules are prespecified
secondary comparisons and all are reported.

Define the secondary hybrid on exactly four joint complete-case participant
columns: HST cough, HST speech, selected-comparator cough, and selected-
comparator speech. `hybrid_uniform_four_branch` assigns each column weight 0.25.
Also run the same legacy AUPRC weighting and frozen logistic stack on those four
columns as prespecified secondary hybrids. Report paired hybrid-minus-HST-only
and hybrid-minus-comparator deltas on identical participant/fold keys. Tests
assert the uniform weights, exact four-column schema, and failure on any cohort
or fold mismatch.

- [ ] **Step 3: Add anti-leakage tests**

Provide deliberately superior test metrics and assert that weights remain
unchanged because only `split == "validation"` rows are read.

- [ ] **Step 4: Run tests and commit**

```bash
python -m pytest tests/test_hst_fusion.py -q
git add src/covid_rars/hst_fusion.py tests/test_hst_fusion.py
git commit -m "Add validation-only HST multimodal fusion"
```

## Task 7: Staged Reliability Orchestrator

**Files:**
- Create: `src/covid_rars/hst_reliability.py`
- Create: `src/covid_rars/hst_runtime.py`
- Create: `scripts/72_run_hst_reliability.py`
- Create: `tests/test_hst_reliability.py`
- Modify: `tests/hst_test_helpers.py`

- [ ] **Step 1: Write stage-resume test**

```python
from pathlib import Path

import pytest


def test_completed_stage_with_matching_hash_is_skipped(tmp_path) -> None:
    from covid_rars.hst_reliability import HSTPipeline, HSTPipelineConfig
    calls = []
    pipeline = HSTPipeline(HSTPipelineConfig.smoke(tmp_path), stage_hook=calls.append)
    pipeline.run_stage("preflight")
    pipeline.run_stage("preflight")
    assert calls == ["preflight"]


def test_corrupt_output_invalidates_completed_stage(tmp_path) -> None:
    from tests.hst_test_helpers import make_smoke_pipeline
    pipeline = make_smoke_pipeline(tmp_path)
    record = pipeline.run_stage("preflight")
    Path(record["output_paths"][0]).write_text("corrupt")
    assert pipeline.run_stage("preflight")["reused"] is False


@pytest.mark.parametrize(
    ("stage", "mutation"),
    (
        ("preflight", "source_bytes"),
        ("preflight", "dependency_lock"),
        ("aligned_comparator", "reused_comparator_source"),
        ("internal_cv", "pilot_freeze"),
        ("internal_cv", "data_contracts_freeze"),
        ("small_smoke", "upstream_checksum"),
        ("preflight", "missing_output"),
        ("preflight", "running_status"),
        ("preflight", "failed_status"),
        ("small_smoke", "forced_upstream_rerun"),
    ),
)
def test_stage_reuse_is_invalidated_by_every_provenance_change(
    tmp_path, stage, mutation
) -> None:
    from tests.hst_test_helpers import make_smoke_pipeline, mutate_pipeline_provenance

    pipeline = make_smoke_pipeline(tmp_path)
    pipeline.run(through=stage)
    mutate_pipeline_provenance(pipeline, stage=stage, mutation=mutation)
    assert pipeline.run_stage(stage)["reused"] is False
```

Implement `make_smoke_pipeline` and `mutate_pipeline_provenance` in
`tests/hst_test_helpers.py` in this step; each mutation changes exactly the
named fingerprint input or state without changing unrelated inputs.

- [ ] **Step 2: Implement stages**

`HSTPipeline.STAGES` must equal:

```python
(
    "preflight", "data_contracts", "checkpoint", "preprocess_worker_pilot",
    "spectrogram_cache", "manifests", "small_smoke", "base_resource_pilot",
    "aligned_comparator", "internal_cv", "split_policy_contrast", "reverse_temporal",
    "external_transfer", "fusion", "statistics", "gradcam",
    "evidence_pack",
)
```

Required methods:

- `run(*, through: str = "evidence_pack", force: set[str] | None = None) -> pd.DataFrame`
- `run_stage(stage: str, *, force: bool = False) -> dict[str, object]`

The preprocessing worker pilot runs before cache construction. DataLoader-worker
and `(FP32, AMP) x batch` benchmarks run only in `base_resource_pilot`, after a
verified cache, frozen manifests, and HST-Small smoke run exist.
`aligned_comparator` then runs the fold-local ComParE+IS10 feature selection and
four-model branch bank for every required manifest in an isolated CPU process.
It writes its own candidate, selection, prediction, metric, and alignment
artifacts, exits, and releases the high-dimensional feature table before
`internal_cv` may acquire the GPU lease. It is never concurrent with scientific
HST training on the fixed 19 GiB host.

Each stage record contains configuration, source-code, dependency-lock, HST
commit, checkpoint, manifest, upstream-artifact hashes, timestamps, status,
output paths/checksums, row counts, and error text. Skip only a matching
successful record whose outputs still exist and validate. Code, dependency,
configuration, input, or checksum changes invalidate dependent stages.
Every training-stage fingerprint includes the accepted pilot-freeze hash, the
accepted data-contracts-freeze hash, accepted environment-lock hash, and the
actual `pip freeze` content hash.

Compute project source identity from an explicit sorted allow-list of executable
HST modules and worker/CLI scripts plus every reused project module in their
runtime import closure. At minimum this includes `metrics.py`, `calibration.py`,
`labels.py`, `audio_io.py`, `strong_baseline.py`,
`strong_baseline_protocol.py`, `compare_is10_rescue.py`, and
`compare_is10_final_validation.py`.
Preflight compares the discovered project import closure with the allow-list and
fails on an unlisted executable module. Hash tests as a separate QA input that
invalidates verification/evidence publication but not completed model training.
Hash scientific JSON and the environment lock separately. Do not include generated reports, acceptance
records, notebook outputs, caches, checkpoints, or model outputs in the source
Merkle tree. Record Git commit and dirty status as provenance fields, but do not
use the commit ID as a replacement for content hashes. Add a test showing that
changing either a new HST module or a reused comparator/metric module
invalidates dependent stages while writing an accepted report or committing an
unchanged dependency lock does not change
the executable-source hash. A test-only change invalidates QA/evidence
publication but reuses checksum-valid completed training.

For smoke/pilot launches where the data-contract hash does not yet exist, use a
deterministic bootstrap directory under
`data/outputs/hst/_bootstrap/<launch_id>/`. The launch ID hashes mode,
scientific configuration, normalized source-root locator, project/HST source,
dependency/checkpoint provenance, and any accepted hashes already available.
The detached worker executes `preflight` and `data_contracts` there, checking
source size and mtime before and after each content hash. It then derives the
final content-addressed run ID including the new data-contract hash and
atomically renames the bootstrap directory to `data/outputs/hst/<run_id>/`.
Write an atomic `data/outputs/hst/_launches/<launch_id>.json` receipt after the
promotion. If a checksum-valid final run already exists, attach to it; never
merge or overwrite two run roots. A bootstrap lock prevents duplicate workers.
Full mode, which receives accepted data, pilot, and environment-lock hashes,
derives the final run ID immediately but still validates those artifacts before
reuse.

All stage-ledger output paths are relative to the current run root. Immediately
before promotion, flush/fsync and close bootstrap log/state handles. After the
atomic rename, construct a new run-path context and reopen log, PID, heartbeat,
lock, and exit paths beneath the final root before any later stage runs. Tests
must assert that no ledger or runtime path still points to `_bootstrap` after
promotion and that status lookup through the launch receipt survives process
restart.

Implement an exclusive run lock containing host, PID, process start identity,
config hash, and heartbeat. Refuse a live lock. Recover a stale lock only when
the recorded process is absent and heartbeat expired. All outputs remain under
`data/outputs/hst/<run_id>/`; publish `reports/hst/latest.json` only after the
evidence pack passes validation.

Also acquire an execution-account-wide lock path keyed by CUDA UUID before any CUDA stage,
using non-blocking `flock(LOCK_EX | LOCK_NB)`. The final detached worker keeps
the file descriptor open for the lifetime of the CUDA work. The lease record
contains device UUID, host, final worker PID/start identity, run ID, and atomic
heartbeat, so different run IDs and workspaces cannot train concurrently.
Release it in `finally`; stale recovery requires both the OS lock to be
acquirable and PID/start/heartbeat checks to pass.

- [ ] **Step 3: Implement CLI options**

```text
--config configs/hst_reliability.json
--run-id auto|EXPECTED_RUN_ID
--mode smoke|pilot|full
--through preflight|data_contracts|checkpoint|preprocess_worker_pilot|spectrogram_cache|manifests|small_smoke|base_resource_pilot|aligned_comparator|internal_cv|split_policy_contrast|reverse_temporal|external_transfer|fusion|statistics|gradcam|evidence_pack
--force-stage STAGE
--device cuda|cpu
--resume / --no-resume
--detach
--status-id LAUNCH_OR_RUN_ID
```

`--run-id` defaults to `auto`. A supplied value is an assertion and must equal
the content-derived final run ID once available; arbitrary human or timestamp
names are rejected. Detached smoke/pilot launch returns a launch ID immediately,
and `--status-id` resolves either that launch receipt/bootstrap state or a final
run ID.

- [ ] **Step 4: Add mode tests**

Require smoke mode to run one cough fold, HST-Small, two epochs. Pilot mode runs
the systems benchmarks plus one capped HST-Base resource probe of 100 optimizer
updates and validation; it is not a completed scientific fold. Full mode requires the exact
accepted pilot-freeze, data-contracts-freeze, and environment-lock hashes; it
uses HST-Base for every configured primary job and cannot change scientific
settings.

- [ ] **Step 5: Test detached execution and locking**

`launch_detached_run` is the only detachment implementation. The notebook calls
it directly; CLI `--detach` delegates to the same function and never creates a
second launcher layer. It uses a new process session, closed inherited streams,
an absolute working directory, and redirected log, and transactionally records
the final worker PID rather than an intermediate shell PID. Heartbeat and exit
files use temporary-file replacement. On first `SIGTERM`, the worker completes
the current optimizer update, writes and reload-verifies a checkpoint, marks
the job `stopped`, and exits; a second signal may terminate immediately. An
integration test launches a short worker, terminates its controller process,
and verifies the worker continues and writes atomic heartbeat/exit status. Add
tests for cross-run GPU-lock rejection, duplicate-live-run-lock rejection, stale
lock recovery, signal-safe stop, bootstrap-to-final promotion, status lookup by
launch ID after promotion, asserted run-ID mismatch, and atomic `latest.json`
publication.

- [ ] **Step 6: Run tests and commit**

```bash
python -m pytest tests/test_hst_reliability.py -q
git add src/covid_rars/hst_reliability.py src/covid_rars/hst_runtime.py scripts/72_run_hst_reliability.py tests/test_hst_reliability.py
git commit -m "Add staged HST reliability orchestrator"
```

## Task 8: Statistical Evidence Pack

**Files:**
- Create: `src/covid_rars/hst_reporting.py`
- Create: `scripts/73_make_hst_evidence_pack.py`
- Create: `scripts/74_register_hst_evidence.py`
- Modify: `tests/test_hst_reliability.py`

- [ ] **Step 1: Write the external-bootstrap test**

```python
def test_external_delta_uses_independent_two_sample_bootstrap() -> None:
    from tests.hst_test_helpers import external_predictions, source_predictions
    from covid_rars.hst_reporting import external_transfer_delta
    result = external_transfer_delta(
        source_predictions(), external_predictions(), n_bootstrap=200, seed=42
    )
    assert result["bootstrap_design"] == "independent_label_stratified_participants"
    assert result["paired"] is False
    assert result["resampling_unit"] == "participant_key"


def test_repeated_holdout_bootstrap_clusters_participant_fold_rows() -> None:
    from tests.hst_test_helpers import repeated_holdout_predictions
    from covid_rars.hst_reporting import repeated_holdout_cluster_ci
    result = repeated_holdout_cluster_ci(
        repeated_holdout_predictions(), metric="auroc", n_bootstrap=200, seed=42
    )
    assert result["estimand"] == "mean_repetition_metric"
    assert result["conditional_on_fitted_models"] is True
    assert result["independent_row_pooling"] is False


def test_external_fold_bootstrap_samples_each_target_participant_once() -> None:
    from tests.hst_test_helpers import repeated_external_predictions
    from covid_rars.hst_reporting import external_fold_cluster_bootstrap
    result = external_fold_cluster_bootstrap(
        repeated_external_predictions(), metric="auroc", n_bootstrap=200, seed=42
    )
    assert result["target_resampling_unit"] == "participant_key"
    assert result["same_target_resample_across_folds"] is True


def test_split_policy_delta_is_unpaired_but_common_test_control_is_paired() -> None:
    from tests.hst_test_helpers import split_policy_predictions
    from covid_rars.hst_reporting import split_policy_delta

    different_tests, common_test = split_policy_predictions()
    primary = split_policy_delta(
        different_tests, common_test=False, metric="auroc",
        n_bootstrap=200, seed=42,
    )
    control = split_policy_delta(
        common_test, common_test=True, metric="auroc",
        n_bootstrap=200, seed=42,
    )
    assert primary["paired"] is False
    assert primary["bootstrap_design"] == "independent_label_stratified_participants"
    assert control["paired"] is True
    assert control["bootstrap_design"] == "paired_participant"


def test_reporting_contract_freezes_calibration_operating_point_and_dca() -> None:
    from covid_rars.hst_reporting import REPORTING_CONTRACT

    assert REPORTING_CONTRACT["bootstrap_replicates"] == 1000
    assert REPORTING_CONTRACT["bootstrap_seed"] == 42
    assert REPORTING_CONTRACT["ece_bins"] == 10
    assert REPORTING_CONTRACT["fixed_sensitivity"] == 0.90
    assert REPORTING_CONTRACT["decision_thresholds"] == [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
    ]


def test_screening_threshold_is_fitted_on_validation_only() -> None:
    from tests.hst_test_helpers import validation_and_test_predictions
    from covid_rars.hst_reporting import (
        apply_screening_operating_point,
        fit_screening_operating_point,
    )

    validation, test = validation_and_test_predictions()
    operating_point = fit_screening_operating_point(
        validation, target_sensitivity=0.90
    )
    original = apply_screening_operating_point(test, operating_point)
    changed_test = test.copy()
    changed_test["label_binary"] = changed_test["label_binary"].iloc[::-1].to_numpy()
    changed = apply_screening_operating_point(changed_test, operating_point)
    assert original["threshold"] == operating_point.threshold
    assert changed["threshold"] == operating_point.threshold
```

- [ ] **Step 2: Implement evidence tables**

Build branch/fusion metrics, repetition mean and standard deviation,
participant-cluster bootstrap CIs for repeated holdouts, paired clustered HST-
comparator deltas, independent label-stratified split-policy deltas, paired
common-latest-test deltas, independent two-sample source-external deltas,
calibration bins, fixed-sensitivity operating
points, decision curves, runtime, GPU memory, and comparator alignment. Never
pool repeated-fold rows as independent observations or sum repeated test counts
as unique participants. DeLong is limited to one genuinely paired participant
set, such as a single fold or the common-latest-test control; repeated-fold
participant rows are never pooled for DeLong. It is prohibited for
Coswara-versus-COUGHVID deltas.

Create a frozen analysis-scope registry before metrics are read. Its single
primary estimand is the participant-level Track-A complete-case uniform
cough+speech AUROC delta between HST and the aligned ComParE+IS10 system. Key
secondary estimands are the calendar-mixed/chronological AUROC contrast, the
Coswara-cough/COUGHVID AUROC contrast, and cough+speech fusion versus the
constituent HST modality selected using source-validation AUROC only. Assign
every other row to secondary,
sensitivity, or exploratory scope exactly as specified in the design. Emit
`analysis_scope`, `estimand_id`, and `multiplicity_family` in the evidence
tables; Holm-adjust any reported family of secondary p-values. Do not choose
scope from effect size, confidence interval, or p-value.

Freeze production reporting to 1,000 bootstrap replicates, seed 42, and
percentile 2.5%/97.5% intervals. The 200-replicate values in tests above exist
only to keep unit tests fast. Use 10 equal-width ECE bins on `[0, 1]`, omit
empty bins, compute Brier score as mean squared probability error, and compute
NLL after clipping probabilities to `[1e-6, 1-1e-6]`. Fit Platt scaling on
source validation participant probabilities and apply it unchanged to test,
temporal, and external participants; never report the fitting rows as an
unbiased calibration endpoint.

Delegate calibration to the existing `covid_rars.calibration.PlattCalibrator`
and include that source file in the executable hash. Its frozen estimator is
logistic regression on one probability feature with `solver="lbfgs"`, default
`C=1.0`, intercept enabled, `max_iter=100`, `tol=1e-4`, and no class weighting.
If source validation has
one class or the fit does not converge, emit a skipped calibration audit and
retain raw probabilities; never use test or external labels to repair it.

For AUROC/AUPRC, reject and deterministically redraw any bootstrap replicate
whose evaluated endpoint lacks one class. Require 1,000 valid replicates within
at most 10,000 attempts and fail with an audit row if that gate is not met;
never report an interval from a silently reduced replicate count.

Select the fixed-sensitivity operating threshold only on source validation by
enumerating `0`, `0.5`, `1`, and each unique calibrated validation probability. Among
thresholds achieving sensitivity at least 0.90, maximize specificity, then
sensitivity, then threshold. Apply that frozen threshold to the held-out rows.
Decision curves use source-validation Platt-calibrated probabilities on the
fixed threshold grid `0.05, 0.10, ..., 0.50` and report
`TP/N - (FP/N) * p_t/(1-p_t)`, treat-all, and treat-none net benefit. Also emit
a clearly labeled raw-probability sensitivity curve. No operating-point,
calibration, or DCA result may select a model or checkpoint.

For the ordinary validation-selected threshold, call the existing
`best_threshold_by_balanced_accuracy` on raw source-validation participant
probabilities. It considers `0`, `0.5`, `1`, unique values, and adjacent
midpoints; maximize balanced accuracy, then prefer distance to 0.5, then the
lower threshold. Apply it unchanged to raw held-out probabilities. Keep its
rows distinct from the calibrated fixed-sensitivity screening rows.

Every source-fold cough checkpoint predicts the full external cohort. Report
fold-specific external metrics and their distribution. The secondary external
ensemble applies each fold's source-validation Platt calibrator and averages
calibrated probabilities equally; external labels cannot select members,
weights, calibration, or threshold. Since the repeated-holdout source folds do
not provide one cohort that is out of sample for all ten models, do not invent a
validation-selected ensemble threshold or fixed-sensitivity operating point.
For that ensemble report threshold-free metrics, calibration scores, DCA, and
the fixed-0.5 row only; retain fold-specific source-validation thresholds for
the fold-specific external rows.

Required statistical interfaces include:

- `repeated_holdout_cluster_ci(predictions, *, metric, n_bootstrap, seed)`;
- `paired_model_cluster_delta(left, right, *, metric, n_bootstrap, seed)`;
- `split_policy_delta(predictions, *, common_test, metric, n_bootstrap, seed)`;
- `external_fold_cluster_bootstrap(predictions, *, metric, n_bootstrap, seed)`;
- `external_transfer_delta(source, target, *, metric, n_bootstrap, seed)`;
- `fit_source_platt_calibrator(validation_predictions)`;
- `build_calibration_report(predictions, *, n_bins=10)`;
- `fit_screening_operating_point(validation_predictions, *, target_sensitivity=0.90)`;
- `apply_screening_operating_point(test_predictions, operating_point)`;
- `build_decision_curve(predictions, *, thresholds)`.

Preserve multiplicities within `(participant_key, fold)` when a participant is
resampled. Label repeated-holdout intervals `conditional_on_fitted_models`; they
do not include full retraining uncertainty. Source-external fold deltas use one
target-participant resample shared across all folds. Ensemble predictions reduce
to one row per target participant before ordinary target bootstrapping.

- [ ] **Step 3: Generate publication figures**

Generate SVG and PNG for branch/fusion performance, aligned HST versus
ComParE+IS10, the validation ladder, calibration, decision curves, and runtime.
Every figure records its source table in a figure manifest.

Create a dedicated `hst_evidence_manifest.json` that discovers every run-local
metric, manifest, label audit, cache/eligibility audit, checkpoint provenance,
prediction, figure, and statistics table. Validate required columns and hashes.
Do not rely on the existing paper-table scripts silently discovering new HST
filenames. `74_register_hst_evidence.py` reads the validated HST `latest.json`
plus the existing paper table/experiment manifest and writes new combined
`reports/hst/combined_paper_metric_table.csv` and
`reports/hst/combined_experiment_manifest.json`. Existing central artifacts are
not overwritten, but the combined outputs must contain all prior entries plus
the validated HST run, metric-table hash, and figure manifest. Tests assert no
prior row disappears and every final HST row is registered.

- [ ] **Step 4: Run tests and commit**

```bash
python -m pytest tests/test_hst_reliability.py -q
git add src/covid_rars/hst_reporting.py scripts/73_make_hst_evidence_pack.py scripts/74_register_hst_evidence.py tests/test_hst_reliability.py
git commit -m "Add HST statistical evidence pack"
```

## Task 9: Deterministic Grad-CAM

**Files:**
- Create: `src/covid_rars/hst_gradcam.py`
- Create: `tests/test_hst_gradcam.py`

- [ ] **Step 1: Write heatmap test**

```python
def test_gradcam_heatmap_is_finite_and_normalized() -> None:
    import torch
    from tests.hst_test_helpers import make_tiny_hst_model
    from covid_rars.hst_gradcam import hst_gradcam
    heatmaps = hst_gradcam(make_tiny_hst_model(), torch.randn(1, 3, 224, 224), target_class=1)
    assert heatmaps.shape == (1, 224, 224)
    assert torch.isfinite(heatmaps).all()
    assert float(heatmaps.min()) >= 0.0
    assert float(heatmaps.max()) <= 1.0


def test_gradcam_is_repeatable_batch_independent_and_cleans_hooks() -> None:
    import copy
    import torch
    from tests.hst_test_helpers import make_tiny_hst_model
    from covid_rars.hst_gradcam import hst_gradcam

    model = make_tiny_hst_model()
    model.train()
    was_training = model.training
    before = copy.deepcopy(model.state_dict())
    hook_counts = {
        id(module): (
            len(module._forward_pre_hooks),
            len(module._forward_hooks),
            len(module._backward_hooks),
        )
        for module in model.modules()
    }
    grad_state = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    x = torch.randn(2, 3, 224, 224)
    first = hst_gradcam(model, x[:1], target_class=1)[0]
    second = hst_gradcam(model, x, target_class=1)[0]
    assert torch.allclose(first, second)
    assert all(torch.equal(before[key], model.state_dict()[key]) for key in before)
    assert model.training is was_training
    assert all(
        hook_counts[id(module)] == (
            len(module._forward_pre_hooks),
            len(module._forward_hooks),
            len(module._backward_hooks),
        )
        for module in model.modules()
    )
    for name, parameter in model.named_parameters():
        expected = grad_state[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected)
```

- [ ] **Step 2: Implement HST hooks**

Resolve the final LWMSA as `model.layers[-1].HSTblocks[-1].attn2`. Capture its
input as the paper-proximal target ("prior to the final attention layer") and
its output only as a separately labeled sensitivity. Assert `[B, 49, 768]` for
HST-Base and reshape 49 tokens to the known 7 x 7 map; reject unexpected token
counts rather than guessing a grid.

Run in `model.eval()` with autocast disabled, target the class-1 pre-softmax
logit, and obtain only its activation gradient with `torch.autograd.grad` so
existing parameter gradients are not cleared or populated. Compute
gradient-weighted activations, apply ReLU, normalize,
and upsample. Install hooks inside `try/finally` and remove them even on failure.
For captured activations `A[b, t, c]` and target logit `y_b`, use the fixed
transformer-token adaptation
`alpha[b, c] = mean_t(d y_b / d A[b, t, c])` and
`L[b, t] = ReLU(sum_c(alpha[b, c] * A[b, t, c]))`, then reshape the 49 token
values to 7 x 7. A forward pre-hook retains the actual attention-input tensor;
gradients are captured from that tensor rather than guessed from parameter
gradients.
If the post-ReLU map maximum is zero, return an all-zero map plus an explicit
`zero_map=True` audit field rather than divide by zero. Tests cover repeatability,
batch independence, hook cleanup after exceptions, zero maps, unchanged model
parameters/buffers, and restoration of the caller's original train/eval mode.

- [ ] **Step 3: Implement deterministic example selection**

Required interfaces:

- `select_gradcam_examples(predictions: pd.DataFrame, *, per_category: int, seed: int) -> pd.DataFrame`
- `build_gradcam_evidence(model: object, examples: pd.DataFrame, *, output_dir: Path) -> pd.DataFrame`
- `build_participant_gradcam_summary(heatmaps: pd.DataFrame, *, bootstrap_replicates: int, seed: int) -> GradCAMGroupSummary`
- `extract_stage_participant_embeddings(model: object, loader: object) -> pd.DataFrame`
- `build_stage_embedding_figure(embeddings: pd.DataFrame, *, method: str, seed: int) -> Path`

Select primary TP, TN, FP, and FN examples at fixed threshold 0.5 with a fixed
seed. A validation-selected-threshold panel is an explicitly labeled
sensitivity. Save source, heatmap, overlay, probability, threshold source,
label, fold, participant, and recording metadata. If a category has fewer than
`per_category` participants, use every available participant and record the
shortfall; never duplicate examples or change the threshold to fill a panel.

Capture outputs after all four HST stages for held-out rows only, average
recordings within participant, and generate deterministic PCA panels. Add one
fixed t-SNE sensitivity with `perplexity=30`, `init="pca"`,
`learning_rate="auto"`, `max_iter=1000`, and `random_state=42`; skip it when
the participant count is at most 30. Do not search visual hyperparameters or
report embedding separation as classifier performance.

For primary group maps, retain held-out participants correctly classified at
fixed 0.5, average recording heatmaps within participant, then report class-wise
participant means and their mean difference with participant
bootstrap uncertainty. Fit one pooled PCA basis across both classes before
comparing class projections. Optional separate class-wise first PCs are oriented
against their class mean and shown descriptively, but are never subtracted.

- [ ] **Step 4: Run tests and commit**

```bash
python -m pytest tests/test_hst_gradcam.py -q
git add src/covid_rars/hst_gradcam.py tests/test_hst_gradcam.py
git commit -m "Add deterministic HST Grad-CAM evidence"
```

## Task 10: One-Click Jupyter Controller

**Files:**
- Create: `notebooks/09_HST_RELIABILITY_E2E.ipynb`
- Create: `tests/test_hst_notebook.py`

- [ ] **Step 1: Write notebook-structure test**

```python
import json
from pathlib import Path


def test_hst_notebook_contains_restart_safe_stages() -> None:
    from covid_rars.hst_reliability import HSTPipeline
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/09_HST_RELIABILITY_E2E.ipynb").read_text())
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for stage in HSTPipeline.STAGES:
        assert stage in text
    assert "launch_detached_run" in text
    assert "subprocess.run" not in text
```

- [ ] **Step 2: Create notebook cells in this order**

1. title, scientific scope, and long-runtime warning;
2. imports and root resolution;
3. one config cell with `RUN_MODE="pilot"`, `DEVICE="cuda"`,
   `FORCE_STAGES=set()`, `RUN_THROUGH="base_resource_pilot"`, and
   `ACCEPTED_PILOT_FREEZE_HASH=None` plus
   `ACCEPTED_DATA_CONTRACTS_FREEZE_HASH=None` and
   `ACCEPTED_ENVIRONMENT_LOCK_HASH=None`;
4. immutable run-ID/config preview and job-count/ETA preview;
5. one `launch_detached_run(...)` call;
6. environment/GPU/checkpoint, preprocessing-worker, and base-resource dashboard;
7. live PID, heartbeat, log tail, stage ledger, and exit status;
8. final tables and figures;
9. exact resume instructions.

The notebook contains no repeated shell-command cells and never owns the worker.
`launch_detached_run` starts the CLI in a new process session with absolute
paths and redirected streams. Jupyter Run All is the single action for a pilot
or an already-approved full configuration. Closing the notebook or SSH session
must not stop the worker.

The final run ID is deterministic from mode, scientific-configuration hash,
data-contract hash, HST source commit, project source-code hash, checkpoint
hash, dependency hash, and, in full mode, accepted pilot hash. Before a pilot's
data hash exists, the notebook receives a deterministic launch ID and follows
its atomic launch receipt to the promoted final run. Re-running the notebook
attaches to the matching live launch/run or reuses/resumes its durable stage
ledger; it cannot create a second timestamp-named duplicate job.
`FORCE_STAGES` is the only explicit rerun path.

Full mode is rejected until `ACCEPTED_PILOT_FREEZE_HASH`,
`ACCEPTED_DATA_CONTRACTS_FREEZE_HASH`, and
`ACCEPTED_ENVIRONMENT_LOCK_HASH` exactly match their successful audited
artifacts. This prevents a multi-day launch with untested resource settings,
changed packages, or silently changed source data and labels.

- [ ] **Step 3: Preserve failure visibility**

The controller may catch display/polling interruptions, but the worker writes
its own exception and traceback to `exit.json` and `run.log`. Never convert a
failed worker into a successful notebook cell without showing that failure.
Add an integration test that kills the notebook-side controller while the
worker continues to heartbeat and completes.

- [ ] **Step 4: Run test and commit**

```bash
python -m pytest tests/test_hst_notebook.py -q
git add notebooks/09_HST_RELIABILITY_E2E.ipynb tests/test_hst_notebook.py
git commit -m "Add one-click HST reliability notebook"
```

## Task 11: Unit And Smoke Verification

**Files:**
- No new files.

- [ ] **Step 1: Run all HST tests**

```bash
python -m pytest tests/test_hst_data_contracts.py tests/test_hst_checkpoint.py tests/test_hst_spectrograms.py tests/test_hst_protocols.py tests/test_hst_parallel.py tests/test_hst_training.py tests/test_hst_comparators.py tests/test_hst_fusion.py tests/test_hst_reliability.py tests/test_hst_gradcam.py tests/test_hst_notebook.py -q
```

Expected: PASS; CUDA tests skip only where CUDA is absent.

- [ ] **Step 2: Run reused-contract regressions**

```bash
python -m pytest tests/test_metrics.py tests/test_protocol_matched_cv.py tests/test_protocol_matched_multimodal_cv.py tests/test_final_uncertainty.py tests/test_reviewer_extension_checks.py -q
```

Expected: PASS.

- [ ] **Step 3: Run Ubuntu checkpoint preflight**

```bash
python scripts/72_run_hst_reliability.py --mode smoke --through checkpoint --device cuda
```

Expected: verified hashes/commit, exact backbone key audit, finite `[2, 2]`
logits, positive-class-column audit, CUDA identity, and peak-memory record.

- [ ] **Step 4: Run and resume smoke training**

```bash
python scripts/72_run_hst_reliability.py --mode smoke --through small_smoke --device cuda
```

Terminate the controller once while the detached worker continues, then stop
the worker after an epoch and relaunch with resume. Completed stages must skip,
the training stage must resume after the last verified checkpoint, and the
resumed result must match an uninterrupted smoke run.

- [ ] **Step 5: Commit only small audits**

Do not commit checkpoints, spectrogram tensors, model states, or raw prediction
files. Commit source, tests, configuration, and small provenance/audit tables.

## Task 12: Base Pilot And Full-Run Freeze

**Files:**
- Create: `reports/final/HST_EXECUTION_FREEZE.md`

- [ ] **Step 1: Run the capped HST-Base resource pilot in the notebook**

```python
RUN_MODE = "pilot"
RUN_THROUGH = "base_resource_pilot"
FORCE_STAGES = set()
```

- [ ] **Step 2: Record measured resources**

Record the live host profile, preprocessing and DataLoader worker benchmarks,
physical/effective batch size, gradient accumulation, AMP state, epoch time,
finite probe loss/predictions, peak host/GPU memory, cache throughput,
job-plan counts, estimated primary/secondary duration from 20/50/100-update
measurements, hashes, source commit, and any batch-size deviation from the
paper. The probe's validation metric is diagnostic only and cannot select a
scientific configuration. Write the exact reviewed Ubuntu `pip freeze` output
to `requirements-hst-lock.txt` and store its SHA-256 in the pilot audit; full
mode requires the active environment to match that lock exactly. Store the
resulting immutable `pilot_freeze_hash`. After manually
reviewing the data-contract and resource-pilot audit tables, write their exact
hashes to `reports/hst/accepted_data_contracts_freeze.sha256` and
`reports/hst/accepted_pilot_freeze.sha256`, and write the lock hash to
`reports/hst/accepted_environment_lock.sha256`. These small acceptance records
are inputs to full mode; they are never inferred from outcome metrics.

- [ ] **Step 3: Apply go/no-go gates**

Proceed only if label/leakage/checkpoint/cache audits pass, test labels were
never used for training or selection, probabilities are finite, free GPU
headroom is at least `max(1 GiB, 15%)`, detached execution and deterministic
resume work, and measured runtime is accepted. Internal AUROC is not a code-
validity gate and cannot justify silent hyperparameter changes.

- [ ] **Step 4: Run the full notebook**

```python
from pathlib import Path

RUN_MODE = "full"
RUN_THROUGH = "evidence_pack"
FORCE_STAGES = set()
ACCEPTED_PILOT_FREEZE_HASH = Path(
    "reports/hst/accepted_pilot_freeze.sha256"
).read_text().strip()
ACCEPTED_DATA_CONTRACTS_FREEZE_HASH = Path(
    "reports/hst/accepted_data_contracts_freeze.sha256"
).read_text().strip()
ACCEPTED_ENVIRONMENT_LOCK_HASH = Path(
    "reports/hst/accepted_environment_lock.sha256"
).read_text().strip()
```

Run All. Reopening and rerunning skips matching complete stages and resumes
incomplete training.

- [ ] **Step 5: Freeze artifacts**

Hash configuration, code, dependency lock, manifests, checkpoint/cache/label
audits, metrics, predictions, and report tables. Mark confirmatory/exploratory
rows. Run strict schema/checksum/artifact validation and all HST/regression tests
before atomically publishing `reports/hst/latest.json` or using results in a
manuscript.

- [ ] **Step 6: Commit**

```bash
git add reports/final/HST_EXECUTION_FREEZE.md requirements-hst-lock.txt
git commit -m "Freeze HST reliability execution protocol"
```

## Self-Review Checklist

- Checkpoint provenance, hashes, tensor counts, and head replacement are tested.
- Official HST source is pinned and not silently modified.
- Positive class is index 1 everywhere; COUGHVID negated labels fail closed.
- COUGHVID release, selected label column, semi-supervised/self-report/expert
  provenance, disagreements, SNR, and event construction are explicit.
- Full execution requires an explicitly accepted immutable data-contract hash.
- Full execution requires the accepted Ubuntu environment-lock hash.
- Dataset-qualified participant/recording keys are used everywhere.
- Paper-text reconstruction and released-code approximation are separately labeled.
- Final participant manifests are frozen after base-cache eligibility is known.
- Literature-aligned and matched-cohort split-policy questions are separate.
- Test/external labels cannot affect model, epoch, threshold, or fusion choice.
- Recording predictions reduce to participant probabilities without crossing folds.
- HST and comparator primary metrics use identical participants and analysis units.
- Primary multimodal fusion is complete-case uniform for both model families.
- Cough-only external transfer is not called multimodal external validation.
- Every source fold predicts COUGHVID; external labels select nothing.
- Full-recording external transfer is not conflated with the explicitly deferred
  HST-style SNR/event sensitivity.
- Repeated holdouts use participant-clustered inference, not pooled fold rows.
- Long stages are detached, locked, resumable, and full-provenance-hash aware.
- Cache writes and checkpoints are transactional and checksum verified.
- Grad-CAM selection is deterministic and outcome-category based.
- Existing result files remain untouched; HST outputs use new names.
- No SOTA claim is inferred from non-comparable data or protocols.
