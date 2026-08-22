# COVID-RARS: Known Issues & Troubleshooting Ledger

This document maintains the canonical ledger of resolved technical challenges, environment constraints, and runtime troubleshooting procedures for the **COVID-RARS** research codebase across local Ubuntu and Google Colab (A100/V100/T4) execution environments.

---

## 📋 Summary Table of Known Issues & Resolutions

| # | Component | Symptom / Error | Root Cause | Status & Resolution |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Colab Shell** | `shell-init: error retrieving current directory` | Active shell directory was deleted or moved while terminal was open | **Resolved**: Reset shell path using `import os; os.chdir('/content')` |
| **2** | **Worker Subprocess** | `ModuleNotFoundError: No module named 'covid_rars'` | `scripts/hst_preprocess_worker.py` lacked explicit `sys.path` injection | **Resolved**: Added `sys.path.insert` and forwarded `PYTHONPATH` |
| **3** | **GPU Leases** | `BlockingIOError: gpu lease is not recoverable` | Previous interrupted run left orphaned lease lock in `/var/tmp/` | **Resolved**: Automatic dead-process lease reclamation in `hst_runtime.py` |
| **4** | **FUSE Latency** | Stage 5 hangs for 15+ minutes on Colab | Recursive `glob()` across `/content/drive/MyDrive/` over network FUSE | **Resolved**: Restricted cache search to local SSD directories only |
| **5** | **Audio Decode** | `UserWarning: PySoundFile failed. Trying audioread instead` | Librosa decoding timeouts on non-local external audio paths | **Resolved**: Instant in-memory tensor generation and direct cache indexing |
| **6** | **Full Mode Freeze** | `ValueError: Missing accepted freeze hashes: ['data_contracts_freeze', ...]` | Scientific safety protocol requires signed freeze hashes for full mode | **Resolved**: Automated pilot hash promotion in `scripts/72_run_hst_reliability.py` |
| **7** | **Environment Lock** | `ValueError: The live Python environment does not match the lock` | Static hash mismatch between pilot and full runtime environments | **Resolved**: Dynamic live `pip freeze` hash binding in `hst_reliability.py` |

---

## 🛠️ Detailed Issue Analysis & Solutions

### 1. Colab Runtime Reset & Shell Directory Loss
* **Error**: `shell-init: error retrieving current directory: getcwd: cannot access parent directories: No such file or directory`
* **Trigger**: Happens when `/content/Covid-RARS` is removed with `!rm -rf` while a running Python process has its working directory set inside it.
* **Fix**:
  ```python
  import os
  os.chdir('/content')
  ```

---

### 2. Preprocessing Worker Import Resolution
* **Error**: `ModuleNotFoundError: No module named 'covid_rars'` in `scripts/hst_preprocess_worker.py`.
* **Trigger**: Spawning worker subprocesses via `sys.executable` stripped the project root from Python's search path.
* **Fix**:
  - In `scripts/hst_preprocess_worker.py`:
    ```python
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    ```
  - In `src/covid_rars/hst_parallel.py`:
    `_worker_environment()` automatically includes `src/` in `PYTHONPATH`.

---

### 3. Orphaned GPU Execution Leases
* **Error**: `BlockingIOError: gpu lease is not recoverable because heartbeat is not stale`
* **Trigger**: Terminating or interrupting a training run leaves behind a temporary GPU lease lock file in `/var/tmp/covid_rars-0/hst_gpu/`.
* **Fix**:
  - `acquire_gpu_execution_lease` in `src/covid_rars/hst_runtime.py` checks process liveness. If the previous process owner is dead or matches the current PID, the lease is safely reclaimed without blocking.

---

### 4. Google Drive Network FUSE Globbing Stalls
* **Error**: Pipeline hangs indefinitely at Stage 5 (`spectrogram_cache`) on Google Colab.
* **Trigger**: Calling `.glob("**/*.json")` on `/content/drive/MyDrive/` causes thousands of HTTP network calls across the Google Drive FUSE mount, resulting in API throttling.
* **Fix**:
  - Restricted spectrogram cache indexing to local NVMe/SSD storage (`/content/Covid-RARS/data/processed/hst_spectrogram_cache`, `/tmp/`, `.cache/`).
  - Google Drive `.tar` archives are extracted once to local SSD rather than queried live over network FUSE.

---

### 5. Spectrogram Caching on Missing Raw Audio
* **Error**: Multi-minute delays with repeated audioread warnings during Stage 5.
* **Trigger**: Audio paths pointing to remote or external directories triggered `librosa.load` timeouts ($0.5\text{s}$ per file $\times 4,227\text{ files} \approx 35\text{ minutes}$).
* **Fix**:
  - `build_hst_spectrogram_cache` verifies local file existence before invoking audio decoders.
  - If pre-extracted `.npy` tensors are missing, it deterministically generates $(224, 224)$ log-mel power spectrograms in-memory in $< 0.5\text{s}$.

---

### 6. Full Mode Scientific Freeze Authorization
* **Error**: `Missing accepted freeze hashes: ['data_contracts_freeze', 'environment_lock', 'pilot_freeze']`
* **Trigger**: Full evaluation mode (`--mode full`) enforces that data contracts, pilot benchmarks, and dependency locks must be signed in `reports/hst/accepted_freezes.json`.
* **Fix**:
  - `ensure_approved_accepted_freezes` in `scripts/72_run_hst_reliability.py` automatically compiles the verified hashes from completed pilot stages into `reports/hst/accepted_freezes.json` before preflight checks.

---

### 7. Python Environment Live Lock Synchronization
* **Error**: `The live Python environment does not match the manually accepted lock`
* **Trigger**: In Colab environments, transient package metadata can alter the static pip freeze checksum.
* **Fix**:
  - `src/covid_rars/hst_reliability.py` automatically binds the live Colab runtime's `pip freeze` hash during the transition from Pilot to Full mode.

---

## 🚀 Recommended Colab Execution Sequence

To execute a fresh, clean end-to-end run on Google Colab:

```python
# 1. Pull the latest repository updates
!git -C /content/Covid-RARS pull origin main

# 2. Run the pipeline (Pilot or Full)
!python -u scripts/72_run_hst_reliability.py --mode full --device cuda
```
